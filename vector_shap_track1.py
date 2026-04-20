"""Exact grouped Shapley (Vector SHAP) for Track 1: f(x) = P(cluster=k | encoder(x))."""

from __future__ import annotations
import json
import logging
import pickle
import time
from dataclasses import dataclass
from math import factorial
from typing import Dict, List
import numpy as np
import torch
import config
from config import RANDOM_SEED, DEVICE, BACKGROUND_K, MAX_EXPLAIN
from model import load_autoencoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


OUT_DIR = config.VECTOR_SHAP_TRACK1_OUT_DIR
MODEL_PATH = config.BEST_MODEL_PATH
KM_PATH = config.KM_MODEL_PATH
LR_PATH = config.LR_MODEL_PATH
CLUST_MU_PATH = config.CLUSTER_STANDARDISE_MU_PATH
CLUST_SD_PATH = config.CLUSTER_STANDARDISE_SD_PATH
TRAIN_CLUSTER_Z_PATH = config.TRAIN_CLUSTER_Z_PATH
TRAIN_W_PATH = config.TRAIN_W_PATH
VAL_W_PATH = config.VAL_W_PATH
TEST_W_PATH = config.TEST_W_PATH

INDICATOR_NAMES = ["Log_Returns", "Realised_Volatility", "RSI", "MACD", "Normalised_Volume", "Momentum"]

LEVEL2_FAMILIES: Dict[str, List[str]] = {
    "price_dynamics": ["Log_Returns", "Momentum"],
    "volatility_indicators": ["Realised_Volatility", "Normalised_Volume"],
    "technical_indicators": ["RSI", "MACD"],
}


def standardise_apply(x: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (x - mu) / sd


def encode_latent(model: torch.nn.Module, windows: np.ndarray, device: str = DEVICE, batch_size: int = 512) -> np.ndarray:
    model.eval()
    latents = []
    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            xb = torch.from_numpy(windows[i:i+batch_size]).float().to(device)
            _, z = model(xb)
            latents.append(z.detach().cpu().numpy())
    return np.concatenate(latents, axis=0)


def window_stats_from_windows(windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sd = windows.std(axis=1)
    sd = np.where(sd == 0, 1.0, sd)  # masked-out features are constant across T, giving sd=0
    return windows.mean(axis=1), sd


def build_cluster_features_from_windows(
    model: torch.nn.Module,
    windows: np.ndarray,
    win_mu: np.ndarray,
    win_sd: np.ndarray,
    device: str = DEVICE,
) -> np.ndarray:
    z = encode_latent(model, windows, device=device)
    assert len(win_mu) == len(z) and len(win_sd) == len(z), \
        "win_mu/win_sd not aligned with windows/latents (check shuffle=False and saved arrays)."
    return np.hstack([z, win_mu, win_sd])  # consistent with cluster feature construction in train.py


def predict_softmax_proba_from_windows(
    model: torch.nn.Module,
    logreg,
    windows: np.ndarray,
    clust_mu: np.ndarray,
    clust_sd: np.ndarray,
    win_mu: np.ndarray,
    win_sd: np.ndarray,
    device: str = DEVICE,
) -> np.ndarray:
    x = build_cluster_features_from_windows(model, windows, win_mu, win_sd, device=device)
    cluster_features_z = standardise_apply(x, clust_mu, clust_sd)
    return logreg.predict_proba(cluster_features_z)


def stratified_background_indices(labels: np.ndarray, n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    uniq, counts = np.unique(labels, return_counts=True)
    probs = counts / counts.sum()
    alloc = np.floor(probs * n).astype(int)

    diff = n - alloc.sum()
    if diff > 0:
        order = np.argsort(-probs)
        for i in order[:diff]:
            alloc[i] += 1

    idxs = []
    for label, n_allocated in zip(uniq, alloc):
        pool = np.where(labels == label)[0]
        idxs.append(rng.choice(pool, size=int(n_allocated), replace=False))
    return np.concatenate(idxs)


@dataclass
class ClusterVectorSHAPOutput:
    shap_values: np.ndarray # (N, D, K)
    base_value: np.ndarray  # (N, K)
    full_value: np.ndarray  # (N, K)
    indicator_names: List[str]
    n_coalitions: int
    background_k: int
    elapsed_sec: float


def vectorshap_softmax_proba(
    model: torch.nn.Module,
    logreg,
    windows: np.ndarray,    # (N,T,D)
    background_windows: np.ndarray, # (B,T,D)
    clust_mu: np.ndarray,
    clust_sd: np.ndarray,
    device: str = DEVICE,
) -> ClusterVectorSHAPOutput:
    t0 = time.time()

    N, T, D = windows.shape
    assert D == len(INDICATOR_NAMES), f"Window feature dim D={D} does not match INDICATOR_NAMES={len(INDICATOR_NAMES)}"

    n_subsets = 1 << D  # 2^D exact enumeration, feasible for small D (Choi et al., 2024)
    bg_mean_td = background_windows.mean(axis=0)    # (T,D); masking via background mean (Antwarg et al., 2020)

    all_masked = np.empty((n_subsets * N, T, D), dtype=windows.dtype)
    for mask_int in range(n_subsets):
        ind_mask = np.array([(mask_int >> d) & 1 for d in range(D)], dtype=bool)
        sl = slice(mask_int * N, (mask_int + 1) * N)
        all_masked[sl] = np.where(ind_mask[None, None, :], windows, bg_mean_td[None, :, :])

    # Recompute win_mu/win_sd from each masked window (avoids leakage)
    all_mu, all_sd = window_stats_from_windows(all_masked)

    logger.info("Evaluating f for all %d coalitions x %d windows (single pass)...", n_subsets, N)
    all_preds = predict_softmax_proba_from_windows(
        model, logreg, all_masked, clust_mu, clust_sd,
        win_mu=all_mu, win_sd=all_sd, device=device,
    )  # (n_subsets*N, K)
    K = all_preds.shape[1]
    all_preds = all_preds.reshape(n_subsets, N, K)  # (2^D, N, K)

    # Shapley value formula: weighted marginal contributions (Lundberg & Lee, 2017; Antwarg et al., 2020; Choi et al., 2024)
    # If |S| = D then D - s - 1 = -1, giving factorial(-1)
    # This case is skipped by the inner loop, so weight 0.0 is assigned
    def _shapley_weight(s: int) -> float:
        return factorial(s) * factorial(D - s - 1) / factorial(D)

    weights = np.array([
        _shapley_weight(bin(mask_int).count('1')) if bin(mask_int).count('1') < D else 0.0
        for mask_int in range(n_subsets)
    ])

    shap_vals = np.zeros((N, D, K), dtype=np.float64)
    for i in range(D):
        bit_i = 1 << i
        for mask_int in range(n_subsets):
            if (mask_int >> i) & 1:  # skip as i already in coalition
                continue
            shap_vals[:, i, :] += weights[mask_int] * (all_preds[mask_int | bit_i] - all_preds[mask_int])

    base_pred = all_preds[0]
    full_pred = all_preds[n_subsets - 1]

    elapsed = time.time() - t0
    logger.info("Exact SHAP done in %.1fs", elapsed)
    return ClusterVectorSHAPOutput(
        shap_values=shap_vals.astype(np.float32),
        base_value=base_pred.astype(np.float32),
        full_value=full_pred.astype(np.float32),
        indicator_names=list(INDICATOR_NAMES),
        n_coalitions=n_subsets,
        background_k=int(len(background_windows)),
        elapsed_sec=float(elapsed),
    )


def aggregate_level2(level1: ClusterVectorSHAPOutput) -> Dict[str, np.ndarray]:
    # Level-2 family aggregation: sum indicator SHAP values into feature families (Choi et al., 2024)
    name_to_idx = {n: i for i, n in enumerate(level1.indicator_names)}
    return {
        fam: level1.shap_values[:, [name_to_idx[m] for m in members], :].sum(axis=1)
        for fam, members in LEVEL2_FAMILIES.items()
    }


def main(explain_split: str = "test"):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_w = np.load(TRAIN_W_PATH)
    val_w = np.load(VAL_W_PATH)
    test_w = np.load(TEST_W_PATH)

    if explain_split == "train":
        explain_w = train_w
    elif explain_split == "val":
        explain_w = val_w
    else:
        explain_w = test_w

    if MAX_EXPLAIN is not None:
        explain_w = explain_w[:MAX_EXPLAIN]

    input_dim = train_w.shape[2]
    model = load_autoencoder(MODEL_PATH, input_dim=input_dim, device=DEVICE)

    with open(KM_PATH, "rb") as f:
        kmeans = pickle.load(f)
    with open(LR_PATH, "rb") as f:
        logreg = pickle.load(f)
    clust_mu = np.load(CLUST_MU_PATH)
    clust_sd = np.load(CLUST_SD_PATH)

    train_cluster_z = np.load(TRAIN_CLUSTER_Z_PATH)
    train_labels = kmeans.predict(train_cluster_z)

    background_idx = stratified_background_indices(train_labels, n=BACKGROUND_K, seed=RANDOM_SEED)
    background_w = train_w[background_idx]
    n_clusters = int(kmeans.n_clusters)

    logger.info(
        "Track 1 Vector SHAP | split=%s | N=%d | D=%d | coalitions=2^%d=%d | K_bg=%d | clusters=%d",
        explain_split, len(explain_w), len(INDICATOR_NAMES), len(INDICATOR_NAMES),
        1 << len(INDICATOR_NAMES), len(background_w), n_clusters,
    )

    shap_result = vectorshap_softmax_proba(
        model=model,
        logreg=logreg,
        windows=explain_w,
        background_windows=background_w,
        clust_mu=clust_mu,
        clust_sd=clust_sd,
        device=DEVICE,
    )

    level2_shap = aggregate_level2(shap_result)

    # additivity (efficiency) check - Shapley axiom verified explicitly (Choi et al., 2024)
    approx = shap_result.base_value + shap_result.shap_values.sum(axis=1)   # (N,K)
    max_err = float(np.max(np.abs(approx - shap_result.full_value)))
    mean_err = float(np.mean(np.abs(approx - shap_result.full_value)))
    logger.info("[Check] Additivity | max |err|=%.6e | mean |err|=%.6e", max_err, mean_err)
    logger.info("Elapsed: %.1fs", shap_result.elapsed_sec)

    np.save(OUT_DIR / f"vectorshap_level1_{explain_split}.npy", shap_result.shap_values)    # (N,D,K)
    np.save(OUT_DIR / f"base_value_{explain_split}.npy", shap_result.base_value)    # (N,K)
    np.save(OUT_DIR / f"full_value_{explain_split}.npy", shap_result.full_value)    # (N,K)
    np.save(OUT_DIR / "background_indices.npy", background_idx)

    for fam_name, arr in level2_shap.items():
        np.save(OUT_DIR / f"vectorshap_level2_{fam_name}_{explain_split}.npy", arr) # (N,K)

    meta = {
        "target": "P(cluster=k | encoder(x)) - KMeans + LogReg softmax",
        "explain_split": explain_split,
        "n_explain": int(shap_result.shap_values.shape[0]),
        "seq_len": int(explain_w.shape[1]),
        "n_indicators": int(explain_w.shape[2]),
        "indicator_names": shap_result.indicator_names,
        "level2_families": LEVEL2_FAMILIES,
        "n_coalitions": shap_result.n_coalitions,
        "background_k": shap_result.background_k,
        "n_clusters": int(n_clusters),
        "elapsed_sec": shap_result.elapsed_sec,
        "additivity_max_abs_err": max_err,
        "additivity_mean_abs_err": mean_err,
    }
    with open(OUT_DIR / f"vectorshap_meta_{explain_split}.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Saved to: %s", OUT_DIR.resolve())


if __name__ == "__main__":
    main(explain_split="test")