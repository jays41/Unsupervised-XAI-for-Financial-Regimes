from __future__ import annotations
import json
import logging
import math
import numpy as np
import torch
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from sklearn.base import ClassifierMixin
import matplotlib.pyplot as plt
import config
from model import LSTMAutoencoder

logger = logging.getLogger(__name__)


RANDOM_SEED = config.RANDOM_SEED
DEVICE = torch.device(config.DEVICE)
OUT_DIR = config.TIMESHAP_TRACK1_OUT_DIR
PRUNE_ETA = config.TIMESHAP_PRUNE_ETA
MIN_TIMESTEPS_KEEP = config.TIMESHAP_MIN_TIMESTEPS_KEEP
MC_EVENT = config.TIMESHAP_MC_EVENT
N_EXPLAIN_TEST = config.TIMESHAP_N_EXPLAIN_TEST
N_BACKGROUND = config.TIMESHAP_N_BACKGROUND


def load_splits() -> dict[str, np.ndarray]:
    return {
        "train_w":  np.load(config.TRAIN_W_PATH).astype(np.float32),
        "test_w":   np.load(config.TEST_W_PATH).astype(np.float32),
        "train_mu": np.load(config.TRAIN_WIN_MU_PATH).astype(np.float32),
        "test_mu":  np.load(config.TEST_WIN_MU_PATH).astype(np.float32),
        "train_sd": np.load(config.TRAIN_WIN_SD_PATH).astype(np.float32),
        "test_sd":  np.load(config.TEST_WIN_SD_PATH).astype(np.float32),
    }


def load_model(input_dim: int) -> LSTMAutoencoder:
    model = LSTMAutoencoder(
        input_dim=input_dim,
        hidden_dim=config.HIDDEN_DIM,
        latent_dim=config.LATENT_DIM,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
    ).to(DEVICE)
    ckpt = torch.load(config.BEST_MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def load_lr_model() -> ClassifierMixin:
    with open(config.LR_MODEL_PATH, "rb") as f:
        return pickle.load(f)


def subsample_indices(n: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if k >= n:
        return np.arange(n)
    return rng.choice(n, size=k, replace=False)


def best_k_from_summary() -> int:
    with open(config.CLUSTERING_OUT_DIR / "selection_summary.json") as f:
        summary = json.load(f)
    k_range = sorted(int(k) for k in summary)
    stable_ks = [k for k in k_range if summary[str(k)].get("stable", False)]
    candidates = stable_ks if stable_ks else k_range
    return min(candidates, key=lambda k: summary[str(k)]["entropy"]["val"])


def stratified_indices_from_labels(labels: np.ndarray, n: int, seed: int) -> np.ndarray:
    # Stratified explain set (Antwarg et al., 2020; Cohen et al., 2023)
    rng = np.random.default_rng(seed)
    uniq = np.unique(labels)
    per = n // len(uniq)
    remainder = n - per * len(uniq)

    idx: list[int] = []
    for c in uniq:
        cand = np.where(labels == c)[0]
        idx.extend(rng.choice(cand, size=min(per, len(cand)), replace=False).tolist())

    if remainder > 0:
        chosen = set(idx)
        pool = np.array([i for i in range(len(labels)) if i not in chosen], dtype=int)
        if len(pool) > 0:
            idx.extend(rng.choice(pool, size=min(remainder, len(pool)), replace=False).tolist())

    idx_arr = np.array(idx, dtype=int)
    rng.shuffle(idx_arr)
    return idx_arr[:min(n, len(idx_arr))]


def encode_latent(model: LSTMAutoencoder, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        xt = torch.tensor(x, dtype=torch.float32, device=DEVICE)
        _, z = model(xt)
        return z.cpu().numpy().astype(np.float32)


def build_cluster_features(
    model: LSTMAutoencoder,
    x: np.ndarray,
    win_mu: np.ndarray,
    win_sd: np.ndarray,
) -> np.ndarray:
    z = encode_latent(model, x)
    return np.hstack([z, win_mu, win_sd])


def standardise_cluster_features(
    x_raw: np.ndarray,
    mu: np.ndarray,
    sd: np.ndarray,
) -> np.ndarray:
    assert x_raw.shape[1] == mu.shape[1], (
        f"Cluster feature dim mismatch: data has {x_raw.shape[1]} dims, "
        f"standardiser expects {mu.shape[1]}. "
        "Check that win_mu/win_sd are included consistently with training."
    )
    return (x_raw - mu) / sd


def cluster_proba(
    model: LSTMAutoencoder,
    lr_model: ClassifierMixin,
    x: np.ndarray,
    win_mu: np.ndarray,
    win_sd: np.ndarray,
    std_mu: np.ndarray,
    std_sd: np.ndarray,
) -> np.ndarray:
    x_raw = build_cluster_features(model, x, win_mu, win_sd)
    return lr_model.predict_proba(standardise_cluster_features(x_raw, std_mu, std_sd)).astype(np.float32)


@dataclass
class ExplainResult:
    event_shap: np.ndarray  # (T, K)
    feature_shap: np.ndarray    # (F, K)
    n_pruned: int
    y_full: np.ndarray  # (K,)
    y_base: np.ndarray  # (K,)


class GroupShapleyExplainer:
    # MC event-level Shapley with baseline masking (Bento et al., 2021)
    def __init__(self, baseline: np.ndarray, out_dim: int) -> None:
        self.baseline = baseline.astype(np.float32)  # (T, F); baseline = mean of background windows (Bento et al., 2021)
        self.out_dim = out_dim

    @staticmethod
    def apply_timestep_mask(
        x: np.ndarray, baseline: np.ndarray, keep: np.ndarray
    ) -> np.ndarray:
        out = baseline.copy()
        out[keep] = x[keep]
        return out

    def mc_shapley(
        self,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        x: np.ndarray,
        n_permutations: int,
        y_base: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        T = x.shape[0]
        shap_accumulator = np.zeros((T, self.out_dim), dtype=np.float32)

        for perm in [rng.permutation(T) for _ in range(n_permutations)]:
            present = np.zeros(T, dtype=bool)
            active_timesteps: list[int] = []
            masked_windows: list[np.ndarray] = []

            for t in perm:
                present[t] = True
                active_timesteps.append(int(t))
                masked_windows.append(self.apply_timestep_mask(x, self.baseline, present.copy()))

            batch = np.stack(masked_windows, axis=0)
            y_all = predict_fn(batch).astype(np.float32)
            prev_ys = np.concatenate([y_base[None], y_all[:-1]], axis=0)
            deltas = y_all - prev_ys

            for step, t in enumerate(active_timesteps):
                shap_accumulator[t] += deltas[step]

        return shap_accumulator / float(n_permutations)

    @staticmethod
    def prune_prefix_length(event_shap: np.ndarray, eta: float, min_keep: int) -> int:
        # Temporal pruning: retain timesteps covering (1-eta) of total |SHAP| mass (Bento et al., 2021)
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
        y_base = predict_fn(self.baseline[None, ...])[0]
        y_full = predict_fn(x[None, ...])[0]

        event_shap = self.mc_shapley(
            predict_fn, x, n_permutations=mc_event, y_base=y_base, rng=rng
        )
        n_pruned = self.prune_prefix_length(event_shap, eta=eta, min_keep=min_keep)
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


def plot_global_event_shap(arr: np.ndarray, title: str, path: Path) -> None:
    mass = np.abs(arr).mean(axis=0).sum(axis=1)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(np.arange(len(mass)), mass, marker="o", markersize=2)
    ax.set_title(title)
    ax.set_xlabel("Timestep (0=oldest, T-1=most recent)")
    ax.set_ylabel("Mean sum_out |SHAP|")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_global_feature_shap(
    arr: np.ndarray, feature_names: list[str], title: str, path: Path
) -> None:
    vals = np.abs(arr).mean(axis=0).sum(axis=1)
    idx = np.argsort(vals)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.arange(len(vals))
    ax.barh(y, vals[idx])
    ax.set_yticks(y)
    ax.set_yticklabels([feature_names[i] for i in idx])
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel("Mean sum_out |SHAP|")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    splits = load_splits()
    train_w, test_w = splits["train_w"], splits["test_w"]
    test_mu, test_sd = splits["test_mu"], splits["test_sd"]

    F = test_w.shape[2]
    feature_names: list[str] = np.load(config.FEATURE_NAMES_PATH, allow_pickle=True).tolist()

    model = load_model(input_dim=F)
    lr_model = load_lr_model()
    std_mu = np.load(config.CLUSTER_STANDARDISE_MU_PATH).astype(np.float32).reshape(1, -1)
    std_sd = np.load(config.CLUSTER_STANDARDISE_SD_PATH).astype(np.float32).reshape(1, -1)

    background_idx = subsample_indices(len(train_w), min(N_BACKGROUND, len(train_w)), seed=RANDOM_SEED)
    baseline = train_w[background_idx].mean(axis=0).astype(np.float32)  # (T, F); baseline as mean of background windows (Bento et al., 2021)

    best_k = best_k_from_summary()
    labels_test = np.load(config.CLUSTERING_OUT_DIR / f"labels_test_k{best_k}.npy")
    explain_idx = stratified_indices_from_labels(labels_test, N_EXPLAIN_TEST, seed=config.EXPLAIN_SEED)

    assert len(test_mu) == len(test_w) and len(test_sd) == len(test_w), (
        "test_win_mu/sd must be aligned with test_windows (shuffle=False required during extraction)."
    )

    n_clusters = cluster_proba(
        model, lr_model,
        test_w[explain_idx[:1]], test_mu[explain_idx[:1]], test_sd[explain_idx[:1]],
        std_mu, std_sd,
    ).shape[1]

    explainer = GroupShapleyExplainer(baseline=baseline, out_dim=n_clusters)

    results: list[ExplainResult] = []
    for sample_idx, window_idx in enumerate(explain_idx):
        x = test_w[window_idx]
        mu_i = test_mu[window_idx : window_idx + 1]
        sd_i = test_sd[window_idx : window_idx + 1]

        # mu_i and sd_i fixed per iteration through default args to avoid late binding
        def predict_cluster(
            x_batch: np.ndarray,
            _mu: np.ndarray = mu_i,
            _sd: np.ndarray = sd_i,
        ) -> np.ndarray:
            B = x_batch.shape[0]
            return cluster_proba(
                model, lr_model,
                x_batch,
                np.repeat(_mu, B, axis=0),
                np.repeat(_sd, B, axis=0),
                std_mu, std_sd,
            )

        result = explainer.explain(
            predict_fn=predict_cluster,
            x=x,
            mc_event=MC_EVENT,
            eta=PRUNE_ETA,
            min_keep=MIN_TIMESTEPS_KEEP,
            rng=np.random.default_rng([RANDOM_SEED, sample_idx]),
        )
        results.append(result)

        top_clusters = np.argsort(result.y_full)[-3:][::-1]
        logger.info(
            "[cluster] %d/%d  idx=%d  pruned=%d  top=%s",
            sample_idx + 1, len(explain_idx), int(window_idx), result.n_pruned,
            [(int(c), float(result.y_full[c])) for c in top_clusters],
        )

    event_shap_all = np.stack([r.event_shap for r in results], axis=0)    # (N, T, K)
    feature_shap_all = np.stack([r.feature_shap for r in results], axis=0)  # (N, F, K)
    pruned_all = np.array([r.n_pruned for r in results], dtype=int)
    outputs_all = np.stack([r.y_full for r in results], axis=0)            # (N, K)

    np.save(OUT_DIR / "cluster_event_shap.npy",       event_shap_all)
    np.save(OUT_DIR / "cluster_feature_shap.npy",     feature_shap_all)
    np.save(OUT_DIR / "cluster_pruned_timesteps.npy", pruned_all)
    np.save(OUT_DIR / "cluster_outputs.npy",          outputs_all)

    plot_global_event_shap(
        event_shap_all, "Global event importance (Cluster probs)", OUT_DIR / "cluster_global_event.png"
    )
    plot_global_feature_shap(
        feature_shap_all, feature_names, "Global feature importance (Cluster probs)", OUT_DIR / "cluster_global_feature.png"
    )

    summary = {
        "n_explained_test": int(len(explain_idx)),
        "explain_indices_test": [int(i) for i in explain_idx],
        "background_train_n": int(len(background_idx)),
        "mc_event": int(MC_EVENT),
        "feature_shap_method": "exact_coalitions_2^F",
        "prune_eta": float(PRUNE_ETA),
        "min_timesteps_keep": int(MIN_TIMESTEPS_KEEP),
        "avg_pruned_cluster": float(pruned_all.mean()),
        "cluster_out_dim_k": int(n_clusters),
    }
    with open(OUT_DIR / "timeshap_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Saved TimeSHAP outputs to: %s", OUT_DIR.resolve())


if __name__ == "__main__":
    main()