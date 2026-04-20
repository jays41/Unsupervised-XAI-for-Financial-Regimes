from __future__ import annotations
import json
import logging
import math
import time
from dataclasses import dataclass
from itertools import permutations
from typing import Dict, List, Tuple
import numpy as np
import torch
import config
from config import (
    BACKGROUND_K,
    DEVICE,
    EXPLAIN_SEED,
    INDICATOR_NAMES,
    LEVEL2_FAMILIES,
    MAX_EXPLAIN,
    N_FEATURES,
    RANDOM_SEED,
    TRACK2_TARGET_BIN_PATH,
)
from model import load_autoencoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

MODEL_PATH = config.BEST_MODEL_PATH
OUT_DIR = config.VECTOR_SHAP_TRACK2_OUT_DIR

assert len(INDICATOR_NAMES) == N_FEATURES, (
    f"INDICATOR_NAMES has {len(INDICATOR_NAMES)} entries but N_FEATURES={N_FEATURES}"
)


def predict_recon_error(
    model: torch.nn.Module,
    windows: np.ndarray,
    device: str = DEVICE,
    batch_size: int = 512,
) -> np.ndarray:
    """Return per-window MSE reconstruction error, shape (N, 1)."""
    model.eval()
    errors: List[np.ndarray] = []

    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            batch = torch.from_numpy(windows[i : i + batch_size]).float().to(device)
            recon, _ = model(batch)
            mse = ((recon - batch) ** 2).mean(dim=(1, 2))
            errors.append(mse.cpu().numpy())

    return np.concatenate(errors).astype(np.float32).reshape(-1, 1)


def coalition_masks(perm: np.ndarray, d: int) -> np.ndarray:
    """Build the (d+1, d) cumulative binary coalition mask for one permutation."""
    masks = np.zeros((d + 1, d), dtype=np.int8)
    for s, j in enumerate(perm, start=1):
        masks[s] = masks[s - 1]
        masks[s, j] = 1
    return masks


@dataclass
class ErrorVectorSHAPResult:
    shap_values: np.ndarray # (N, D, 1)
    base_value: np.ndarray  # (N, 1)
    full_value: np.ndarray  # (N, 1)
    indicator_names: List[str]
    n_perm: int
    background_k: int
    elapsed_sec: float


def vectorshap_recon_error(
    model: torch.nn.Module,
    windows: np.ndarray,
    background_windows: np.ndarray,
    device: str = DEVICE,
) -> ErrorVectorSHAPResult:
    t0 = time.time()
    N, T, D = windows.shape
    n_perm = math.factorial(D)  # D! permutation enumeration (Choi et al., 2024)

    background_mean = background_windows.mean(axis=0)  # interventional masking via background mean (Antwarg et al., 2020)
    baseline_preds = np.repeat(
        predict_recon_error(model, background_mean[None], device=device), N, axis=0
    )
    full_preds = predict_recon_error(model, windows, device=device)

    shap_values = np.zeros((N, D, 1), dtype=np.float64)

    # Pre-compute expanded views, re-used across all D! permutations
    windows_expanded = windows[None]
    bg_expanded = background_mean[None, None]

    for perm_idx, perm_tuple in enumerate(permutations(range(D))):
        perm = np.array(perm_tuple)
        mask_4d = coalition_masks(perm, D).astype(np.float32)[:, None, None, :]
        interpolated = windows_expanded * mask_4d + bg_expanded * (1.0 - mask_4d)
        batch = interpolated.reshape((D + 1) * N, T, D)

        coalition_preds = predict_recon_error(model, batch, device=device).reshape(D + 1, N, 1)

        for s, j in enumerate(perm, start=1):
            shap_values[:, j, :] += coalition_preds[s] - coalition_preds[s - 1]

        if (perm_idx + 1) % 100 == 0 or (perm_idx + 1) == n_perm:
            logger.info(
                "[VectorSHAP-Track2] perm %3d/%d | elapsed %.1fs",
                perm_idx + 1, n_perm, time.time() - t0,
            )

    shap_values /= float(n_perm)

    return ErrorVectorSHAPResult(
        shap_values=shap_values.astype(np.float32),
        base_value=baseline_preds.astype(np.float32),
        full_value=full_preds.astype(np.float32),
        indicator_names=list(INDICATOR_NAMES),
        n_perm=n_perm,
        background_k=len(background_windows),
        elapsed_sec=float(time.time() - t0),
    )


def aggregate_level2(result: ErrorVectorSHAPResult) -> Dict[str, np.ndarray]:
    # Level-2 family aggregation: sum indicator SHAP values into feature families (Choi et al., 2024)
    name_to_idx = {n: i for i, n in enumerate(result.indicator_names)}
    return {
        family: result.shap_values[:, [name_to_idx[m] for m in members], :].sum(axis=1)
        for family, members in LEVEL2_FAMILIES.items()
    }


def load_windows() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_w = np.load(config.TRAIN_W_PATH).astype(np.float32)
    val_w = np.load(config.VAL_W_PATH).astype(np.float32)
    test_w = np.load(config.TEST_W_PATH).astype(np.float32)
    return train_w, val_w, test_w


def load_track2_binary(
    n_train: int, n_val: int, n_test: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_all = np.load(TRACK2_TARGET_BIN_PATH).astype(np.int32).reshape(-1)
    expected = n_train + n_val + n_test
    assert len(y_all) == expected, (
        f"Track2 binary target length mismatch: got {len(y_all)}, expected {expected} "
        "(train+val+test). Ensure Track 2 was run on the same windowing."
    )
    return y_all[:n_train], y_all[n_train : n_train + n_val], y_all[n_train + n_val :]


def stratified_background_indices_binary(labels: np.ndarray, n: int, seed: int = RANDOM_SEED) -> np.ndarray:
    """Stratified background set (Antwarg et al., 2020; Cohen et al., 2023)"""
    rng = np.random.default_rng(seed)

    normal_idx = np.where(labels == 0)[0]
    anomaly_idx = np.where(labels == 1)[0]

    n_anomaly = min(len(anomaly_idx), n // 2)
    n_normal = min(len(normal_idx), n - n_anomaly)

    idx = np.concatenate([
        rng.choice(normal_idx, size=n_normal, replace=False),
        rng.choice(anomaly_idx, size=n_anomaly, replace=False),
    ])
    rng.shuffle(idx)

    if len(idx) < n:
        remaining = np.setdiff1d(np.arange(len(labels)), idx)
        extra = rng.choice(remaining, size=min(n - len(idx), len(remaining)), replace=False)
        idx = np.concatenate([idx, extra])

    return idx[:n]


def select_explain_indices(labels: np.ndarray, n: int, seed: int = EXPLAIN_SEED) -> np.ndarray:
    """Stratified explain-set sample (Antwarg et al., 2020; Cohen et al., 2023)"""
    return stratified_background_indices_binary(labels, n=min(n, len(labels)), seed=seed)


def main(explain_split: str = "test") -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_w, val_w, test_w = load_windows()
    y_train, y_val, y_test = load_track2_binary(len(train_w), len(val_w), len(test_w))

    split_windows, y_split = {
        "train": (train_w, y_train),
        "val":   (val_w,   y_val),
        "test":  (test_w,  y_test),
    }[explain_split]

    explain_idx = select_explain_indices(y_split, n=MAX_EXPLAIN, seed=EXPLAIN_SEED)
    windows_to_explain = split_windows[explain_idx]

    model = load_autoencoder(MODEL_PATH, input_dim=train_w.shape[2], device=DEVICE)

    background_idx = stratified_background_indices_binary(y_train, n=BACKGROUND_K, seed=RANDOM_SEED)
    background_windows = train_w[background_idx]

    n_perm = math.factorial(N_FEATURES)

    logger.info("Split: %s | N_explain=%d", explain_split, len(windows_to_explain))
    logger.info(
        "Indicators (D=%d) | Permutations: %d (exact, all D!) | Background K=%d",
        N_FEATURES, n_perm, BACKGROUND_K,
    )

    result = vectorshap_recon_error(
        model=model,
        windows=windows_to_explain,
        background_windows=background_windows,
        device=DEVICE,
    )

    level2_shap = aggregate_level2(result)

    # additivity (efficiency) check - Shapley axiom verified explicitly (Choi et al., 2024)
    shap_approx = result.base_value + result.shap_values.sum(axis=1)
    max_abs_err = float(np.max(np.abs(shap_approx - result.full_value)))
    mean_abs_err = float(np.mean(np.abs(shap_approx - result.full_value)))
    logger.info("[Additivity] max|err|=%.6e  mean|err|=%.6e", max_abs_err, mean_abs_err)

    np.save(OUT_DIR / f"vectorshap_level1_{explain_split}.npy", result.shap_values)
    np.save(OUT_DIR / f"base_value_{explain_split}.npy", result.base_value)
    np.save(OUT_DIR / f"full_value_{explain_split}.npy", result.full_value)
    np.save(OUT_DIR / "background_indices.npy", background_idx)
    np.save(OUT_DIR / f"explain_indices_{explain_split}.npy", explain_idx)

    for family, arr in level2_shap.items():
        np.save(OUT_DIR / f"vectorshap_level2_{family}_{explain_split}.npy", arr)

    meta = {
        "target": "mean((reconstruction(x) - x)^2) over (T, D)",
        "explain_split": explain_split,
        "n_explain": int(result.shap_values.shape[0]),
        "indicator_names": result.indicator_names,
        "level2_families": LEVEL2_FAMILIES,
        "n_perm": result.n_perm,
        "background_k": result.background_k,
        "elapsed_sec": result.elapsed_sec,
        "additivity_max_abs_err": max_abs_err,
        "additivity_mean_abs_err": mean_abs_err,
    }
    with open(OUT_DIR / f"vectorshap_meta_{explain_split}.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Saved to: %s", OUT_DIR.resolve())


if __name__ == "__main__":
    main(explain_split="test")