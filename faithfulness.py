"""
Faithfulness evaluation of Vector SHAP explanations via perturbation analysis.

For each explained window, features are ranked by aggregate |SHAP| importance.
The top-k features are masked (replaced with the TRAIN background mean) and the change in model output is measured.
SHAP-ranked removal should cause a larger output change than random-ranked removal if the explanations are faithful.
"""

from __future__ import annotations
import json
import logging
import pickle
from math import factorial
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
from scipy.stats import wilcoxon
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
import config
import torch
from model import LSTMAutoencoder

logger = logging.getLogger(__name__)

DEVICE = config.DEVICE
HIDDEN_DIM = config.HIDDEN_DIM
LATENT_DIM = config.LATENT_DIM
NUM_LAYERS = config.NUM_LAYERS
DROPOUT = config.DROPOUT

N_RANDOM_SEEDS = 50
N_STABILITY_RUNS = 10
N_PERM_STABILITY = 500

INDICATOR_NAMES = [
    "Log Returns", "Realised Volatility", "RSI",
    "MACD", "Normalised Volume", "Momentum",
]
D = len(INDICATOR_NAMES)

OUT_DIR = Path("outputs/faithfulness")
PREP_DIR = config.PREP_DIR
MODEL_PATH = config.BEST_MODEL_PATH
T1_DIR = config.VECTOR_SHAP_TRACK1_OUT_DIR
T2_DIR = config.VECTOR_SHAP_TRACK2_OUT_DIR
KM_PATH = config.KM_MODEL_PATH
LR_PATH = config.LR_MODEL_PATH
CLUST_MU_PATH = config.CLUSTER_STANDARDISE_MU_PATH
CLUST_SD_PATH = config.CLUSTER_STANDARDISE_SD_PATH

OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_autoencoder(input_dim: int) -> LSTMAutoencoder:
    model = LSTMAutoencoder(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(DEVICE)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def encode_latent(model: LSTMAutoencoder, windows: np.ndarray, batch: int = 512) -> np.ndarray:
    out = []
    with torch.no_grad():
        for i in range(0, len(windows), batch):
            xb = torch.from_numpy(windows[i:i+batch]).float().to(DEVICE)
            _, z = model(xb)
            out.append(z.cpu().numpy())
    return np.vstack(out)


def recon_mse(model: LSTMAutoencoder, windows: np.ndarray, batch: int = 512) -> np.ndarray:
    out = []
    with torch.no_grad():
        for i in range(0, len(windows), batch):
            xb = torch.from_numpy(windows[i:i+batch]).float().to(DEVICE)
            recon, _ = model(xb)
            out.append(((recon - xb) ** 2).mean(dim=(1, 2)).cpu().numpy())
    return np.concatenate(out)


def standardise_apply(x: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (x - mu) / np.where(sd == 0, 1.0, sd)


def build_cluster_features(
    model: LSTMAutoencoder,
    windows: np.ndarray,
    win_mu: np.ndarray,
    win_sd: np.ndarray,
) -> np.ndarray:
    z = encode_latent(model, windows)
    return np.hstack([z, win_mu, win_sd])


def predict_track1(
    model: LSTMAutoencoder, lr,
    windows: np.ndarray,
    clust_mu: np.ndarray, clust_sd: np.ndarray,
) -> np.ndarray:
    win_mu = windows.mean(axis=1)
    win_sd = windows.std(axis=1)
    feats = build_cluster_features(model, windows, win_mu, win_sd)
    return lr.predict_proba(standardise_apply(feats, clust_mu, clust_sd))


def predict_track2(model: LSTMAutoencoder, windows: np.ndarray) -> np.ndarray:
    return recon_mse(model, windows)


def compute_baseline(train_windows: np.ndarray, bg_indices: np.ndarray) -> np.ndarray:
    """Mean of background TRAIN windows -> (T, D)."""
    return train_windows[bg_indices].mean(axis=0)


def rank_randomly(n: int, d: int, seed: int) -> np.ndarray:
    """Independent random permutation per window -> (N, D)."""
    rng = np.random.default_rng(seed)
    return np.stack([rng.permutation(d) for _ in range(n)])


def _mask_top_k_batch(
    windows: np.ndarray,    # (N, T, D)
    sorted_idx: np.ndarray, # (N, D) - feature removal order per window
    k: int,
    baseline: np.ndarray,   # (T, D)
) -> np.ndarray:
    N, _, Dw = windows.shape
    feat_mask = np.zeros((N, Dw), dtype=bool)
    np.put_along_axis(feat_mask, sorted_idx[:, :k], True, axis=1)  # (N, D)
    return np.where(feat_mask[:, np.newaxis, :], baseline[np.newaxis, :, :], windows)


def evaluate_perturbations(
    windows: np.ndarray,
    shap_values: np.ndarray,
    f_original: np.ndarray,
    baseline: np.ndarray,
    predict_fn,
    track: int,
) -> Tuple[np.ndarray, np.ndarray]:
    
    n = len(windows)
    T = windows.shape[1]
    R = N_RANDOM_SEEDS

    importance = np.abs(shap_values).sum(axis=-1)       # (N, D)
    sorted_shap_idx = np.argsort(-importance, axis=1)   # (N, D) descending

    logger.info("  SHAP-ranked perturbations ...")
    delta_shap = np.zeros((n, D))
    for k in range(1, D + 1):
        masked = _mask_top_k_batch(windows, sorted_shap_idx, k, baseline)
        f_pert = predict_fn(masked)
        if track == 1:
            delta_shap[:, k - 1] = np.linalg.norm(f_original - f_pert, axis=1)
        else:
            delta_shap[:, k - 1] = np.abs(f_original - f_pert)

    logger.info("  Random-ranked perturbations (all %d seeds batched per k) ...", R)
    all_sorted_idx = np.stack([rank_randomly(n, D, seed=r) for r in range(R)])  # (R, N, D)
    delta_random = np.zeros((n, D, R))

    for k in range(1, D + 1):
        feat_mask = np.zeros((R, n, D), dtype=bool)
        np.put_along_axis(feat_mask, all_sorted_idx[:, :, :k], True, axis=2)
        masked_all = np.where(
            feat_mask[:, :, np.newaxis, :],
            baseline[np.newaxis, np.newaxis, :, :],
            windows[np.newaxis, :, :, :],
        ).reshape(R * n, T, D)
        f_pert_all = predict_fn(masked_all)
        if track == 1:
            f_orig_rep = np.tile(f_original, (R, 1))
            delta_random[:, k - 1, :] = np.linalg.norm(
                f_orig_rep - f_pert_all, axis=1
            ).reshape(R, n).T
        else:
            f_orig_rep = np.tile(f_original, R)
            delta_random[:, k - 1, :] = np.abs(
                f_orig_rep - f_pert_all
            ).reshape(R, n).T

    return delta_shap, delta_random


def compute_metrics(delta_shap: np.ndarray, delta_random: np.ndarray) -> Dict:
    mean_shap = delta_shap.mean(axis=0)
    mean_rand = delta_random.mean(axis=(0, 2))
    rand_per_w = delta_random.mean(axis=2)
    gap = mean_shap - mean_rand

    ks = np.arange(1, D + 1)
    _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    auc_shap = float(_trapz(mean_shap, ks))
    auc_rand = float(_trapz(mean_rand, ks))
    auc_ratio = auc_shap / auc_rand if auc_rand else float("inf")

    def _wilcoxon_p(k: int) -> float:
        diff = delta_shap[:, k] - rand_per_w[:, k]
        if np.all(diff == 0):
            return 1.0
        _, p = wilcoxon(diff, alternative="greater")
        return float(p)

    def stars(p: float) -> str:
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
        return "ns"

    per_k = [
        {
            "k":                 k + 1,
            "mean_delta_shap":   float(mean_shap[k]),
            "mean_delta_random": float(mean_rand[k]),
            "gap":               float(gap[k]),
            "wilcoxon_p":        _wilcoxon_p(k),
            "significance":      stars(_wilcoxon_p(k)),
        }
        for k in range(D)
    ]

    return {
        "auc_shap":           auc_shap,
        "auc_random":         auc_rand,
        "auc_ratio":          auc_ratio,
        "faithful":           bool(auc_ratio > 1.0),
        "mean_gap_across_k":  float(gap.mean()),
        "per_k":              per_k,
    }


def _permutation_shap(
    windows: np.ndarray,      # (N, T, D) - windows to explain
    predict_fn,               # callable: (N, T, D) -> (N, K)
    bg_mean: np.ndarray,      # (T, D) - baseline masking value
    n_perm: int,
    seed: int,
) -> np.ndarray:

    rng = np.random.default_rng(seed)
    N, T, Dw = windows.shape

    sample_pred = predict_fn(windows[:1])
    K = sample_pred.shape[1] if sample_pred.ndim == 2 else 1

    shap_vals = np.zeros((N, Dw, K), dtype=np.float64)

    for _ in range(n_perm):
        perm = rng.permutation(Dw)

        batch_parts = []
        present = np.zeros(Dw, dtype=bool)
        for step in range(Dw + 1):
            masked = np.where(
                present[np.newaxis, np.newaxis, :],
                windows,
                bg_mean[np.newaxis, :, :],
            )
            batch_parts.append(masked)
            if step < Dw:
                present[perm[step]] = True

        batch = np.concatenate(batch_parts, axis=0)
        preds = predict_fn(batch)
        if preds.ndim == 1:
            preds = preds[:, np.newaxis]
        preds = preds.reshape(Dw + 1, N, K)

        for step in range(1, Dw + 1):
            j = perm[step - 1]
            shap_vals[:, j, :] += preds[step] - preds[step - 1]

    return (shap_vals / n_perm).astype(np.float32)


def _exact_shap(
    windows: np.ndarray,   # (N, T, D)
    predict_fn,            # callable: (N, T, D) -> (N, K)
    bg_mean: np.ndarray,   # (T, D) - baseline masking value
) -> np.ndarray:
    
    N, T, Dw = windows.shape
    n_subsets = 1 << Dw

    sample_pred = predict_fn(windows[:1])
    K = sample_pred.shape[1] if sample_pred.ndim == 2 else 1

    all_masked = np.empty((n_subsets * N, T, Dw), dtype=windows.dtype)
    for mask_int in range(n_subsets):
        ind_mask = np.array([(mask_int >> d) & 1 for d in range(Dw)], dtype=bool)
        sl = slice(mask_int * N, (mask_int + 1) * N)
        all_masked[sl] = np.where(ind_mask[None, None, :], windows, bg_mean[None, :, :])

    all_preds = predict_fn(all_masked)
    if all_preds.ndim == 1:
        all_preds = all_preds[:, np.newaxis]
    all_preds = all_preds.reshape(n_subsets, N, K)

    def _weight(s: int) -> float:
        return factorial(s) * factorial(Dw - s - 1) / factorial(Dw) if s < Dw else 0.0

    weights = np.array([_weight(bin(m).count('1')) for m in range(n_subsets)])

    shap_vals = np.zeros((N, Dw, K), dtype=np.float64)
    for i in range(Dw):
        bit_i = 1 << i
        for mask_int in range(n_subsets):
            if (mask_int >> i) & 1:
                continue
            shap_vals[:, i, :] += weights[mask_int] * (all_preds[mask_int | bit_i] - all_preds[mask_int])

    return shap_vals.astype(np.float32)

def compute_stability_metrics(shap_runs: np.ndarray) -> Dict:
    R, N, Dw, K = shap_runs.shape
    importance = np.abs(shap_runs).sum(axis=-1)

    std_r = importance.std(axis=0)
    mean_r = importance.mean(axis=0)

    # avoid division by zero
    with np.errstate(invalid="ignore", divide="ignore"):
        rsd = np.where(mean_r > 0, std_r / mean_r, 0.0)

    per_feature_rsd = rsd.mean(axis=0)
    overall_rsd = float(per_feature_rsd.mean())
    target_met = bool(overall_rsd < 1.0)

    return {
        "n_runs": R,
        "n_perm_per_run": N_PERM_STABILITY,
        "overall_rsd": overall_rsd,
        "target_met":  target_met,
        "per_feature": [
            {"feature": INDICATOR_NAMES[d], "rsd": float(per_feature_rsd[d])}
            for d in range(Dw)
        ],
    }


def run_stability_track1() -> Dict:
    logger.info("TRACK 1 - SHAP stability (background bootstrap sensitivity)")
    logger.info("  %d bootstrap background resamples, exact 2^D Shapley each", N_STABILITY_RUNS)

    shap_ref = np.load(T1_DIR / "vectorshap_level1_test.npy")
    bg_indices = np.load(T1_DIR / "background_indices.npy")
    n = shap_ref.shape[0]

    train_w = np.load(PREP_DIR / "train_windows.npy").astype(np.float32)
    test_w = np.load(PREP_DIR / "test_windows.npy").astype(np.float32)
    windows = test_w[:n]

    model = load_autoencoder(input_dim=train_w.shape[-1])
    with open(LR_PATH, "rb") as f:
        lr = pickle.load(f)
    clust_mu = np.load(CLUST_MU_PATH)
    clust_sd = np.load(CLUST_SD_PATH)

    def predict_fn(w: np.ndarray) -> np.ndarray:
        return predict_track1(model, lr, w, clust_mu, clust_sd)

    # Resample background N_STABILITY_RUNS times and compute exact SHAP for each resample
    rng = np.random.default_rng(42)
    bg_size = len(bg_indices)
    shap_runs = []
    for r in range(N_STABILITY_RUNS):
        logger.info("  Bootstrap %d/%d ...", r + 1, N_STABILITY_RUNS)
        bg_idx_r = rng.choice(len(train_w), size=bg_size, replace=False)
        bg_mean_r = train_w[bg_idx_r].mean(axis=0)
        sv = _exact_shap(windows, predict_fn, bg_mean_r)
        shap_runs.append(sv)

    shap_runs = np.stack(shap_runs)
    np.save(OUT_DIR / "track1_stability_runs.npy", shap_runs)

    metrics = compute_stability_metrics(shap_runs)
    with open(OUT_DIR / "track1_stability_results.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("  Overall RSD: %.4f  (%s, target < 1.0)",
                metrics['overall_rsd'], 'STABLE' if metrics['target_met'] else 'UNSTABLE')

    return metrics


def run_stability_track2() -> Dict:
    logger.info("TRACK 2 - SHAP stability (permutation variance)")
    logger.info("  %d runs x %d permutations each", N_STABILITY_RUNS, N_PERM_STABILITY)

    shap_ref = np.load(T2_DIR / "vectorshap_level1_test.npy")  # (N, D, 1)
    bg_indices = np.load(T2_DIR / "background_indices.npy")
    explain_idx = np.load(T2_DIR / "explain_indices_test.npy")
    n = shap_ref.shape[0]

    train_w = np.load(PREP_DIR / "train_windows.npy").astype(np.float32)
    test_w = np.load(PREP_DIR / "test_windows.npy").astype(np.float32)
    windows = test_w[explain_idx]
    bg_mean = compute_baseline(train_w, bg_indices)
    model = load_autoencoder(input_dim=train_w.shape[-1])

    def predict_fn(w: np.ndarray) -> np.ndarray:
        return predict_track2(model, w).reshape(-1, 1)   # (N, 1)

    shap_runs = []
    for r in range(N_STABILITY_RUNS):
        logger.info("  Run %d/%d ...", r + 1, N_STABILITY_RUNS)
        sv = _permutation_shap(windows, predict_fn, bg_mean, N_PERM_STABILITY, seed=r)
        shap_runs.append(sv)

    shap_runs = np.stack(shap_runs)   # (R, N, D, 1)
    np.save(OUT_DIR / "track2_stability_runs.npy", shap_runs)

    metrics = compute_stability_metrics(shap_runs)
    with open(OUT_DIR / "track2_stability_results.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("  Overall RSD: %.4f  (%s, target < 1.0)",
                metrics['overall_rsd'], 'STABLE' if metrics['target_met'] else 'UNSTABLE')

    return metrics


def compute_continuity_ratios(
    windows: np.ndarray,        # (N, T, D)
    shap_values: np.ndarray,    # (N, D, K)
    min_input_dist: float = 1e-6,
) -> np.ndarray:
    """For every unique pair (i, j) of explained windows, compute the Lipschitz-continuity ratio"""
    N = windows.shape[0]
    importance = np.abs(shap_values).sum(axis=-1).astype(np.float64)  # (N, D)
    x_flat = windows.reshape(N, -1).astype(np.float64)                # (N, T*D)

    d_x = cdist(x_flat, x_flat, metric="euclidean")       # (N, N)
    d_r = cdist(importance, importance, metric="cityblock")  # (N, N) - L1

    i_idx, j_idx = np.triu_indices(N, k=1)
    valid = d_x[i_idx, j_idx] >= min_input_dist
    return (d_r[i_idx[valid], j_idx[valid]] / d_x[i_idx[valid], j_idx[valid]]).astype(np.float64)



def run_continuity_track1() -> Dict:
    logger.info("TRACK 1 - Continuity of explanations")

    shap_values = np.load(T1_DIR / "vectorshap_level1_test.npy")  # (N, D, K)
    n = shap_values.shape[0]
    test_w = np.load(PREP_DIR / "test_windows.npy").astype(np.float32)
    windows = test_w[:n]

    ratios = compute_continuity_ratios(windows, shap_values)
    n_pairs = len(ratios)
    metrics = {
        "n_explained": int(n),
        "n_pairs":     int(n_pairs),
        "mean":        float(np.mean(ratios)),
        "median":      float(np.median(ratios)),
        "max":         float(np.max(ratios)),
        "p95":         float(np.percentile(ratios, 95)),
    }
    with open(OUT_DIR / "track1_continuity_results.json", "w") as f:
        json.dump(metrics, f, indent=2)
    np.save(OUT_DIR / "track1_continuity_ratios.npy", ratios)
    logger.info("  Pairs: %d  |  mean ratio: %.4f  |  max: %.4f", n_pairs, metrics['mean'], metrics['max'])
    return metrics


def run_continuity_track2() -> Dict:
    logger.info("TRACK 2 - Continuity of explanations")

    shap_values = np.load(T2_DIR / "vectorshap_level1_test.npy")  # (N, D, 1)
    explain_idx = np.load(T2_DIR / "explain_indices_test.npy")
    n = shap_values.shape[0]
    test_w = np.load(PREP_DIR / "test_windows.npy").astype(np.float32)
    windows = test_w[explain_idx]

    ratios = compute_continuity_ratios(windows, shap_values)
    n_pairs = len(ratios)
    metrics = {
        "n_explained": int(n),
        "n_pairs":     int(n_pairs),
        "mean":        float(np.mean(ratios)),
        "median":      float(np.median(ratios)),
        "max":         float(np.max(ratios)),
        "p95":         float(np.percentile(ratios, 95)),
    }
    with open(OUT_DIR / "track2_continuity_results.json", "w") as f:
        json.dump(metrics, f, indent=2)
    np.save(OUT_DIR / "track2_continuity_ratios.npy", ratios)
    logger.info("  Pairs: %d  |  mean ratio: %.4f  |  max: %.4f", n_pairs, metrics['mean'], metrics['max'])
    return metrics


def _randomise_model_weights(input_dim: int, seed: int = 0) -> LSTMAutoencoder:
    torch.manual_seed(seed)
    model = LSTMAutoencoder(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(DEVICE)
    model.eval()
    return model


def _fit_shuffled_lr(train_feats_z: np.ndarray, labels: np.ndarray, seed: int = 0):
    """Refit LogisticRegression on randomly permuted cluster labels."""
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(labels)
    lr = LogisticRegression(max_iter=1000, random_state=seed)
    lr.fit(train_feats_z, shuffled)
    return lr


def _importance_from_shap(shap_vals: np.ndarray) -> np.ndarray:
    return np.abs(shap_vals).sum(axis=-1)


def compute_sanity_disruption(
    orig_imp: np.ndarray,     # (N, D)
    sanity_imp: np.ndarray,   # (N, D)
    eps: float = 1e-8,
) -> Dict:
    num = np.abs(orig_imp - sanity_imp).sum(axis=1)
    den = np.abs(orig_imp).sum(axis=1) + np.abs(sanity_imp).sum(axis=1) + eps
    d = num / den
    return {
        "mean_disruption":   float(d.mean()),
        "median_disruption": float(np.median(d)),
        "min_disruption":    float(d.min()),
    }


def run_sanity_track1() -> Dict:
    logger.info("TRACK 1 - Sanity checks")

    shap_ref = np.load(T1_DIR / "vectorshap_level1_test.npy")  # (N, D, K)
    bg_indices = np.load(T1_DIR / "background_indices.npy")
    n = shap_ref.shape[0]
    orig_imp = _importance_from_shap(shap_ref)                    # (N, D)

    train_w = np.load(PREP_DIR / "train_windows.npy").astype(np.float32)
    test_w = np.load(PREP_DIR / "test_windows.npy").astype(np.float32)
    windows = test_w[:n]
    bg_mean = compute_baseline(train_w, bg_indices)
    clust_mu = np.load(CLUST_MU_PATH)
    clust_sd = np.load(CLUST_SD_PATH)

    with open(LR_PATH, "rb") as f:
        lr_orig = pickle.load(f)

    logger.info("  [1/2] Random model weights")
    rand_model = _randomise_model_weights(input_dim=train_w.shape[-1])

    def _pred_rand_weights(w: np.ndarray) -> np.ndarray:
        return predict_track1(rand_model, lr_orig, w, clust_mu, clust_sd)

    imp_rand_w = _importance_from_shap(
        _exact_shap(windows, _pred_rand_weights, bg_mean)
    )
    d_rand_w = compute_sanity_disruption(orig_imp, imp_rand_w)
    logger.info("    Disruption (Bray-Curtis): %.4f  (1 = fully changed)", d_rand_w['mean_disruption'])

    logger.info("  [2/2] Shuffled cluster labels")
    train_latents = np.load(Path("outputs/train_latents.npy")).astype(np.float32)
    t_mu = np.load(PREP_DIR / "train_win_mu.npy")
    t_sd = np.load(PREP_DIR / "train_win_sd.npy")
    train_feats = np.hstack([train_latents, t_mu, t_sd])
    train_feats_z = standardise_apply(train_feats, clust_mu, clust_sd)
    train_labels = np.load(Path("outputs/clustering_analysis/labels_train_k3.npy"))
    lr_shuffled = _fit_shuffled_lr(train_feats_z, train_labels, seed=42)
    model_orig = load_autoencoder(input_dim=train_w.shape[-1])

    def _pred_shuffled_labels(w: np.ndarray) -> np.ndarray:
        return predict_track1(model_orig, lr_shuffled, w, clust_mu, clust_sd)

    imp_shuffled = _importance_from_shap(
        _exact_shap(windows, _pred_shuffled_labels, bg_mean)
    )
    d_shuffled = compute_sanity_disruption(orig_imp, imp_shuffled)
    logger.info("    Disruption (Bray-Curtis): %.4f  (1 = fully changed)", d_shuffled['mean_disruption'])

    metrics = {"random_weights": d_rand_w, "shuffled_labels": d_shuffled}
    with open(OUT_DIR / "track1_sanity_results.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def run_sanity_track2(n_perm: int = N_PERM_STABILITY) -> Dict:
    logger.info("TRACK 2 - Sanity checks")

    shap_ref = np.load(T2_DIR / "vectorshap_level1_test.npy")  # (N, D, 1)
    bg_indices = np.load(T2_DIR / "background_indices.npy")
    explain_idx = np.load(T2_DIR / "explain_indices_test.npy")
    n = shap_ref.shape[0]
    orig_imp = _importance_from_shap(shap_ref)                     # (N, D)

    train_w = np.load(PREP_DIR / "train_windows.npy").astype(np.float32)
    test_w = np.load(PREP_DIR / "test_windows.npy").astype(np.float32)
    windows = test_w[explain_idx]
    bg_mean = compute_baseline(train_w, bg_indices)

    logger.info("  [1/2] Random model weights")
    rand_model = _randomise_model_weights(input_dim=train_w.shape[-1])

    def _pred_rand_weights(w: np.ndarray) -> np.ndarray:
        return predict_track2(rand_model, w).reshape(-1, 1)

    imp_rand_w = _importance_from_shap(
        _permutation_shap(windows, _pred_rand_weights, bg_mean, n_perm, seed=0)
    )
    d_rand_w = compute_sanity_disruption(orig_imp, imp_rand_w)
    logger.info("    Disruption (Bray-Curtis): %.4f  (1 = fully changed)", d_rand_w['mean_disruption'])

    logger.info("  [2/2] Constant (mean-MSE) predictor")
    mean_mse = float(np.load(Path("outputs/train_errors.npy")).mean())

    def _pred_constant(w: np.ndarray) -> np.ndarray:
        return np.full((len(w), 1), mean_mse, dtype=np.float32)

    imp_constant = _importance_from_shap(
        _permutation_shap(windows, _pred_constant, bg_mean, n_perm, seed=0)
    )
    d_constant = compute_sanity_disruption(orig_imp, imp_constant)
    logger.info("    Disruption (Bray-Curtis): %.4f  (1 = fully changed)", d_constant['mean_disruption'])

    metrics = {"random_weights": d_rand_w, "constant_predictor": d_constant}
    with open(OUT_DIR / "track2_sanity_results.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics



def save_results(tag: str, delta_shap: np.ndarray, delta_random: np.ndarray, metrics: Dict):
    np.save(OUT_DIR / f"{tag}_delta_shap.npy",   delta_shap)
    np.save(OUT_DIR / f"{tag}_delta_random.npy", delta_random)
    with open(OUT_DIR / f"{tag}_faithfulness_results.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("  Results saved for %s.", tag)


def run_track1() -> Dict:
    logger.info("TRACK 1 - Cluster-probability SHAP faithfulness")

    shap_values = np.load(T1_DIR / "vectorshap_level1_test.npy")  # (N, D, K)
    bg_indices = np.load(T1_DIR / "background_indices.npy")
    full_value = np.load(T1_DIR / "full_value_test.npy")           # (N, K)
    n = shap_values.shape[0]
    logger.info("  N=%d, D=%d, K=%d", n, shap_values.shape[1], shap_values.shape[2])

    train_w = np.load(PREP_DIR / "train_windows.npy").astype(np.float32)
    test_w = np.load(PREP_DIR / "test_windows.npy").astype(np.float32)
    explained = test_w[:n]
    baseline = compute_baseline(train_w, bg_indices)
    model = load_autoencoder(input_dim=train_w.shape[-1])

    with open(LR_PATH, "rb") as f:
        lr = pickle.load(f)
    clust_mu = np.load(CLUST_MU_PATH)
    clust_sd = np.load(CLUST_SD_PATH)

    def predict_fn(w: np.ndarray) -> np.ndarray:
        return predict_track1(model, lr, w, clust_mu, clust_sd)

    delta_shap, delta_random = evaluate_perturbations(
        explained, shap_values, full_value, baseline, predict_fn, track=1
    )
    metrics = compute_metrics(delta_shap, delta_random)
    logger.info("  AUC ratio: %.4f  (%s)", metrics['auc_ratio'], 'FAITHFUL' if metrics['faithful'] else 'NOT faithful')

    save_results("track1", delta_shap, delta_random, metrics)
    return metrics


def run_track2() -> Dict:
    logger.info("TRACK 2 - Reconstruction-error SHAP faithfulness")

    shap_values = np.load(T2_DIR / "vectorshap_level1_test.npy")  # (N, D, 1)
    bg_indices = np.load(T2_DIR / "background_indices.npy")
    explain_idx = np.load(T2_DIR / "explain_indices_test.npy")
    full_value = np.load(T2_DIR / "full_value_test.npy")           # (N, 1)
    n = shap_values.shape[0]
    logger.info("  N=%d, D=%d", n, shap_values.shape[1])

    train_w = np.load(PREP_DIR / "train_windows.npy").astype(np.float32)
    test_w = np.load(PREP_DIR / "test_windows.npy").astype(np.float32)
    explained = test_w[explain_idx]
    baseline = compute_baseline(train_w, bg_indices)
    model = load_autoencoder(input_dim=train_w.shape[-1])

    delta_shap, delta_random = evaluate_perturbations(
        explained, shap_values, full_value.squeeze(-1),
        baseline, lambda w: predict_track2(model, w), track=2,
    )
    metrics = compute_metrics(delta_shap, delta_random)
    logger.info("  AUC ratio: %.4f  (%s)", metrics['auc_ratio'], 'FAITHFUL' if metrics['faithful'] else 'NOT faithful')

    save_results("track2", delta_shap, delta_random, metrics)
    return metrics


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.info("Faithfulness + Stability + Continuity + Sanity evaluation - D=%d features, R=%d random seeds", D, N_RANDOM_SEEDS)

    m1 = run_track1()
    m2 = run_track2()
    s1 = run_stability_track1()
    s2 = run_stability_track2()
    c1 = run_continuity_track1()
    c2 = run_continuity_track2()
    sc1 = run_sanity_track1()
    sc2 = run_sanity_track2()

    logger.info("SUMMARY")
    logger.info("  Faithfulness - Track 1  AUC ratio: %.4f  (%s)", m1['auc_ratio'], 'FAITHFUL' if m1['faithful'] else 'NOT faithful')
    logger.info("  Faithfulness - Track 2  AUC ratio: %.4f  (%s)", m2['auc_ratio'], 'FAITHFUL' if m2['faithful'] else 'NOT faithful')
    logger.info("  Stability    - Track 1  RSD: %.4f  (%s, target < 1.0)", s1['overall_rsd'], 'STABLE' if s1['target_met'] else 'UNSTABLE')
    logger.info("  Stability    - Track 2  RSD: %.4f  (%s, target < 1.0)", s2['overall_rsd'], 'STABLE' if s2['target_met'] else 'UNSTABLE')
    logger.info("  Continuity   - Track 1  mean ratio: %.4f  (median %.4f, max %.4f)", c1['mean'], c1['median'], c1['max'])
    logger.info("  Continuity   - Track 2  mean ratio: %.4f  (median %.4f, max %.4f)", c2['mean'], c2['median'], c2['max'])
    logger.info("  Sanity       - Track 1  rand-weights disruption: %.4f | shuffled-labels disruption: %.4f",
                sc1['random_weights']['mean_disruption'], sc1['shuffled_labels']['mean_disruption'])
    logger.info("  Sanity       - Track 2  rand-weights disruption: %.4f | const-predictor disruption: %.4f",
                sc2['random_weights']['mean_disruption'], sc2['constant_predictor']['mean_disruption'])
    logger.info("  Outputs: %s/", OUT_DIR)


if __name__ == "__main__":
    main()
