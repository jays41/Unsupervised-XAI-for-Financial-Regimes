from __future__ import annotations
import json
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from model import LSTMAutoencoder
from config import (
    RANDOM_SEED,
    TIMESHAP_PRUNE_ETA as PRUNE_ETA,
    TIMESHAP_MIN_TIMESTEPS_KEEP as MIN_TIMESTEPS_KEEP,
    TIMESHAP_MC_EVENT as MC_EVENT,
    TIMESHAP_N_EXPLAIN_TEST as N_EXPLAIN_TEST,
    TIMESHAP_N_BACKGROUND as N_BACKGROUND,
    TRAIN_W_PATH,
    VAL_W_PATH,
    TEST_W_PATH,
    TIMESHAP_TRACK2_OUT_DIR as OUT_DIR,
    TRACK2_TARGET_CONT_PATH as TARGET_CONT_PATH,
    TRACK2_TARGET_BIN_PATH as TARGET_BIN_PATH,
    BEST_MODEL_PATH as MODEL_PATH,
    FEATURE_NAMES_PATH,
    DEVICE as DEVICE_STR,
    HIDDEN_DIM,
    LATENT_DIM,
    NUM_LAYERS,
    DROPOUT,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEVICE = torch.device(DEVICE_STR)


def load_windows() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.load(TRAIN_W_PATH).astype(np.float32),
        np.load(VAL_W_PATH).astype(np.float32),
        np.load(TEST_W_PATH).astype(np.float32),
    )


def load_track2_targets(
    n_train: int, n_val: int, n_test: int
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
]:
    y_cont_all = np.load(TARGET_CONT_PATH).astype(np.float32).reshape(-1)
    y_bin_all = np.load(TARGET_BIN_PATH).astype(np.int32).reshape(-1)

    expected = n_train + n_val + n_test
    if len(y_cont_all) != expected or len(y_bin_all) != expected:
        raise ValueError(
            f"Target length mismatch. Got cont={len(y_cont_all)} bin={len(y_bin_all)} "
            f"but expected {expected} (train+val+test). Ensure Track 2 was run on the same windowing."
        )

    split1, split2 = n_train, n_train + n_val
    y_cont_train, y_cont_val, y_cont_test = np.split(y_cont_all, [split1, split2])
    y_bin_train,  y_bin_val,  y_bin_test  = np.split(y_bin_all,  [split1, split2])

    return (y_cont_train, y_cont_val, y_cont_test), (y_bin_train, y_bin_val, y_bin_test)


def sample_idx(n: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if k >= n:
        return np.arange(n)
    return rng.choice(n, size=k, replace=False)


def stratified_indices_binary(labels: np.ndarray, n: int, seed: int) -> np.ndarray:
    # Stratified explain set (Antwarg et al., 2020; Cohen et al., 2023)
    rng = np.random.default_rng(seed)
    idx0 = np.where(labels == 0)[0]
    idx1 = np.where(labels == 1)[0]

    if len(idx0) == 0 or len(idx1) == 0:
        return sample_idx(len(labels), n, seed)

    n1 = min(len(idx1), n // 2)
    n0 = min(len(idx0), n - n1)

    pick0 = rng.choice(idx0, size=n0, replace=False)
    pick1 = rng.choice(idx1, size=n1, replace=False)

    idx = np.concatenate([pick0, pick1])
    rng.shuffle(idx)
    return idx


@dataclass
class ExplainResult:
    event_shap: np.ndarray
    feature_shap: np.ndarray
    n_pruned: int
    y_full: np.ndarray
    y_base: np.ndarray


class GroupShapleyExplainer:
    # MC Shapley attributions with baseline masking (Bento et al., 2021)

    def __init__(self, baseline: np.ndarray, out_dim: int) -> None:
        self.baseline = baseline.astype(np.float32)  # baseline = mean of background windows (Bento et al., 2021)
        self.out_dim = out_dim

    @staticmethod
    def mask_by_timesteps(x: np.ndarray, baseline: np.ndarray, keep: np.ndarray) -> np.ndarray:
        out = baseline.copy()
        out[keep] = x[keep]
        return out

    @staticmethod
    def mask_by_features(x: np.ndarray, baseline: np.ndarray, keep: np.ndarray) -> np.ndarray:
        out = baseline.copy()
        out[:, keep] = x[:, keep]
        return out

    def mc_shapley(
        self,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        x: np.ndarray,
        group_type: str,
        m: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        T, F = x.shape
        if group_type == "event":
            G = T
        elif group_type == "feature":
            G = F
        else:
            raise ValueError("group_type must be 'event' or 'feature'")

        marginal_contributions = np.zeros((G, self.out_dim), dtype=np.float32)
        y_base = predict_fn(self.baseline[None, ...])[0]

        for _ in range(m):
            perm = rng.permutation(G)
            present = np.zeros(G, dtype=bool)
            prev = y_base

            for g in perm:
                present[g] = True
                if group_type == "event":
                    x_mask = self.mask_by_timesteps(x, self.baseline, present)
                else:
                    x_mask = self.mask_by_features(x, self.baseline, present)
                y_new = predict_fn(x_mask[None, ...])[0]
                marginal_contributions[g] += (y_new - prev).astype(np.float32)
                prev = y_new

        return marginal_contributions / float(m)

    @staticmethod
    def temporal_prune_prefix(event_shap: np.ndarray, eta: float, min_keep: int) -> int:
        """Retain the most-recent timesteps whose cumulative |SHAP| mass >= (1 - eta)
        Temporal pruning with eta=0.025, min_keep=5 (Bento et al., 2021)
        """
        mass = np.abs(event_shap).sum(axis=1)
        total = float(mass.sum())
        if total <= 0:
            return 0
        T = len(mass)
        cumulative = 0.0
        keep = 0
        for t in range(T - 1, -1, -1):
            cumulative += float(mass[t])
            keep += 1
            if (1.0 - cumulative / total) <= eta and keep >= min_keep:
                break
        return T - keep

    def explain(
        self,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        x: np.ndarray,
        mc_event: int,
        eta: float,
        min_keep: int,
        rng: np.random.Generator | None = None,
    ) -> ExplainResult:
        if rng is None:
            rng = np.random.default_rng(RANDOM_SEED)

        y_full = predict_fn(x[None, ...])[0]
        y_base = predict_fn(self.baseline[None, ...])[0]

        event_shap = self.mc_shapley(predict_fn, x, group_type="event", m=mc_event, rng=rng)
        n_pruned = self.temporal_prune_prefix(event_shap, eta=eta, min_keep=min_keep)
        feature_shap = exact_feature_shap_coalitions(predict_fn, x, self.baseline, self.out_dim)

        if n_pruned > 0:
            event_shap[:n_pruned, :] = 0.0

        return ExplainResult(
            event_shap=event_shap,
            feature_shap=feature_shap,
            n_pruned=int(n_pruned),
            y_full=y_full,
            y_base=y_base,
        )


def exact_feature_shap_coalitions(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    baseline: np.ndarray,
    out_dim: int,
) -> np.ndarray:
    """Exact feature-level Shapley via 2^F coalition enumeration (Bento et al., 2021; Choi et al., 2024)."""
    T, F = x.shape
    n_coalitions = 1 << F

    all_masks = np.empty((n_coalitions, T, F), dtype=np.float32)
    for mask_bits in range(n_coalitions):
        xm = baseline.copy()
        for j in range(F):
            if (mask_bits >> j) & 1:
                xm[:, j] = x[:, j]
        all_masks[mask_bits] = xm

    coalition_values = predict_fn(all_masks).astype(np.float32)

    factorials = [math.factorial(k) for k in range(F + 1)]
    denom = float(factorials[F])
    sizes = [bin(s).count("1") for s in range(n_coalitions)]

    shap = np.zeros((F, out_dim), dtype=np.float32)
    for i in range(F):
        i_bit = 1 << i
        for s_bits in range(n_coalitions):
            if s_bits & i_bit:
                continue
            s_size = sizes[s_bits]
            weight = factorials[s_size] * factorials[F - s_size - 1] / denom
            shap[i] += weight * (coalition_values[s_bits | i_bit] - coalition_values[s_bits])

    return shap


def plot_global_event(shap_array: np.ndarray, title: str, path: Path) -> None:
    mass = np.abs(shap_array).mean(axis=0).sum(axis=1)

    plt.figure(figsize=(12, 4))
    plt.plot(np.arange(len(mass)), mass, marker="o", markersize=2)
    plt.title(title)
    plt.xlabel("Timestep (0=oldest, T-1=most recent)")
    plt.ylabel("Mean |SHAP| (summed over output)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_global_feature(
    shap_array: np.ndarray, feature_names: list[str], title: str, path: Path
) -> None:
    vals = np.abs(shap_array).mean(axis=0).sum(axis=1)
    idx = np.argsort(vals)[::-1]

    plt.figure(figsize=(8, 5))
    y_pos = np.arange(len(vals))
    plt.barh(y_pos, vals[idx])
    plt.yticks(y_pos, [feature_names[i] for i in idx])
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel("Mean |SHAP| (summed over output)")
    plt.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def load_model(input_dim: int) -> LSTMAutoencoder:
    model = LSTMAutoencoder(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(DEVICE)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict)
    model.eval()
    return model


def recon_error_scalar(model: LSTMAutoencoder, x_batch: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        xt = torch.tensor(x_batch, dtype=torch.float32, device=DEVICE)
        rec, _ = model(xt)
        mse = ((rec - xt) ** 2).mean(dim=(1, 2)).cpu().numpy().astype(np.float32)
    return mse.reshape(-1, 1)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_w, val_w, test_w = load_windows()
    n_train, n_val, n_test = len(train_w), len(val_w), len(test_w)

    (_, _, y_cont_test), (_, _, y_bin_test) = load_track2_targets(n_train, n_val, n_test)

    F = test_w.shape[2]
    feature_names: list[str] = (
        np.load(FEATURE_NAMES_PATH, allow_pickle=True).tolist()
        if FEATURE_NAMES_PATH.exists()
        else [f"Feature_{i}" for i in range(F)]
    )

    model = load_model(input_dim=F)

    bg_idx = sample_idx(len(train_w), min(N_BACKGROUND, len(train_w)), seed=RANDOM_SEED)
    baseline = train_w[bg_idx].mean(axis=0).astype(np.float32)  # baseline as mean of background windows (Bento et al., 2021)

    explain_idx = stratified_indices_binary(y_bin_test, N_EXPLAIN_TEST, seed=RANDOM_SEED + 1)

    def predict_fn(x_batch: np.ndarray) -> np.ndarray:
        return recon_error_scalar(model, x_batch)

    explainer = GroupShapleyExplainer(baseline=baseline, out_dim=1)

    results = []
    for sample_num, win_idx in enumerate(explain_idx):
        x = test_w[win_idx]
        r = explainer.explain(
            predict_fn=predict_fn,
            x=x,
            mc_event=MC_EVENT,
            eta=PRUNE_ETA,
            min_keep=MIN_TIMESTEPS_KEEP,
            rng=np.random.default_rng([RANDOM_SEED, sample_num]),
        )
        results.append(r)

        logger.info(
            "[track2] %d/%d  idx=%d  label=%d  f_pre=%.6f  f_model=%.6f  pruned=%d",
            sample_num + 1, len(explain_idx), int(win_idx),
            int(y_bin_test[win_idx]),
            float(y_cont_test[win_idx]),
            float(r.y_full[0]),
            r.n_pruned,
        )

    event_shap = np.stack([r.event_shap for r in results], axis=0)
    feat_shap = np.stack([r.feature_shap for r in results], axis=0)
    pruned = np.array([r.n_pruned for r in results], dtype=int)
    y_full = np.stack([r.y_full for r in results], axis=0)

    np.save(OUT_DIR / "track2_event_shap.npy", event_shap)
    np.save(OUT_DIR / "track2_feature_shap.npy", feat_shap)
    np.save(OUT_DIR / "track2_pruned_timesteps.npy", pruned)
    np.save(OUT_DIR / "track2_outputs_model.npy", y_full)
    np.save(OUT_DIR / "explain_indices_test.npy", explain_idx.astype(int))
    np.save(OUT_DIR / "track2_outputs_precomputed.npy", y_cont_test[explain_idx].astype(np.float32))
    np.save(OUT_DIR / "track2_labels_precomputed.npy", y_bin_test[explain_idx].astype(np.int32))

    plot_global_event(event_shap, "Global event importance (Track 2 f(x))", OUT_DIR / "global_event.png")
    plot_global_feature(feat_shap, feature_names, "Global feature importance (Track 2 f(x))", OUT_DIR / "global_feature.png")

    sanity_mae = float(np.mean(np.abs(y_cont_test[explain_idx] - y_full.reshape(-1))))
    summary = {
        "track": "Track 2 (Reconstruction Error)",
        "n_explained_test": int(len(explain_idx)),
        "explain_indices_test": [int(i) for i in explain_idx],
        "background_train_n": int(len(bg_idx)),
        "mc_event": int(MC_EVENT),
        "feature_shap_method": "exact_coalitions_2^F",
        "prune_eta": float(PRUNE_ETA),
        "min_timesteps_keep": int(MIN_TIMESTEPS_KEEP),
        "avg_pruned": float(pruned.mean()),
        "sanity_check": {
            "mean_abs_diff_precomputed_vs_model_on_explained": sanity_mae,
        },
    }

    with open(OUT_DIR / "timeshap_track2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Saved Track 2 TimeSHAP outputs to: %s", OUT_DIR.resolve())


if __name__ == "__main__":
    main()