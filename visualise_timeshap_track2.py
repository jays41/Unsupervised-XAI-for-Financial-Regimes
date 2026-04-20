from __future__ import annotations
import argparse
import logging
from pathlib import Path
from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np
import plot_config
import config
from config import INDICATOR_NAMES, N_CLUSTERS
from plot_config import BINARY_COLORS

logger = logging.getLogger(__name__)

plot_config.setup_style()


def _timestep_ticks(T: int, n_ticks: int = 10):
    positions = np.linspace(0, T - 1, min(n_ticks, T)).astype(int)
    labels = [f"t-{T - 1 - t}" for t in positions]
    return positions, labels

def load_timeshap_outputs(out_dir: Path):
    event_shap = np.load(out_dir / "track2_event_shap.npy")[:, :, 0]             # (N, T)
    feature_shap = np.load(out_dir / "track2_feature_shap.npy")[:, :, 0]         # (N, F)
    pruned = np.load(out_dir / "track2_pruned_timesteps.npy")                     # (N,)
    labels = np.load(out_dir / "track2_labels_precomputed.npy").astype(int)       # (N,)
    explain_idx = np.load(out_dir / "explain_indices_test.npy").astype(int)       # (N,)
    model_out = np.load(out_dir / "track2_outputs_model.npy").squeeze()           # (N,)

    logger.info("  event_shap   : %s", event_shap.shape)
    logger.info("  feature_shap : %s", feature_shap.shape)
    logger.info("  pruned       : %s", pruned.shape)
    logger.info("  labels       : %s  (normal=%d, unusual=%d)", labels.shape, np.sum(labels == 0), np.sum(labels == 1))
    logger.info("  explain_idx  : %s", explain_idx.shape)
    logger.info("  model_out    : %s", model_out.shape)

    return event_shap, feature_shap, pruned, labels, explain_idx, model_out


def load_cluster_membership(cluster_dir: Path, split: str, explain_idx: np.ndarray) -> Optional[np.ndarray]:
    """Load cluster labels for the explained samples."""
    path = cluster_dir / f"labels_{split}_k3.npy"
    if not path.exists():
        return None
    labels_all = np.load(path).astype(np.int32)
    return labels_all[explain_idx]


def plot_temporal_importance_binary(
    event_shap: np.ndarray,
    labels: np.ndarray,
    save_path: Path,
    title: str = "Temporal Importance: Normal vs Unusual Windows",
):
    _, T = event_shap.shape
    mask_n = labels == 0
    mask_u = labels == 1

    def _mean_se(arr):
        m = np.mean(np.abs(arr), axis=0)   # (T,)
        se = np.std(np.abs(arr), axis=0) / np.sqrt(len(arr))
        return m, se

    m_n, se_n = _mean_se(event_shap[mask_n])
    m_u, se_u = _mean_se(event_shap[mask_u])

    fig, ax = plt.subplots(figsize=(10, 5))
    timesteps = np.arange(T)

    ax.plot(timesteps, m_n, label=f'Normal (n={mask_n.sum()})',
            color=BINARY_COLORS["normal"], linewidth=2.5)
    ax.fill_between(timesteps, m_n - 1.96 * se_n, m_n + 1.96 * se_n,
                    color=BINARY_COLORS["normal"], alpha=0.2)

    ax.plot(timesteps, m_u, label=f'Unusual (n={mask_u.sum()})',
            color=BINARY_COLORS["unusual"], linewidth=2.5)
    ax.fill_between(timesteps, m_u - 1.96 * se_u, m_u + 1.96 * se_u,
                    color=BINARY_COLORS["unusual"], alpha=0.2)

    ax.axvspan(T - 10, T - 1, alpha=0.08, color='gray', label='Recent (last 10)')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Mean |Event SHAP|')
    ax.set_title(title, fontweight='bold')

    pos, labs = _timestep_ticks(T)
    ax.set_xticks(pos)
    ax.set_xticklabels(labs, rotation=45, ha='right')
    ax.legend(title='Window Type', loc='upper left')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_feature_importance(
    feature_shap: np.ndarray,
    feature_names: List[str],
    save_path: Path,
    title: str = "Feature Importance (Aggregated Over Time)",
):
    _, F = feature_shap.shape
    mean_imp = np.mean(np.abs(feature_shap), axis=0)  # (F,)
    order = np.argsort(mean_imp)
    sorted_imp = mean_imp[order]
    sorted_names = [feature_names[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.arange(F)
    ax.barh(y, sorted_imp, color='#4878CF', edgecolor='none', height=0.7)

    for i, val in enumerate(sorted_imp):
        ax.text(val + sorted_imp.max() * 0.02, i, f'{val:.4f}', va='center', fontsize=10)

    ax.set_yticks(y)
    ax.set_yticklabels(sorted_names)
    ax.set_xlabel('Mean |Feature SHAP|')
    ax.set_title(title, fontweight='bold')
    ax.set_xlim(0, sorted_imp.max() * 1.25)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_feature_importance_binary(
    feature_shap: np.ndarray,
    labels: np.ndarray,
    feature_names: List[str],
    save_path: Path,
    title: str = "Feature Importance: Normal vs Unusual",
):
    _, F = feature_shap.shape
    mask_n = labels == 0
    mask_u = labels == 1

    imp_n = np.mean(np.abs(feature_shap[mask_n]), axis=0)   # (F,)
    imp_u = np.mean(np.abs(feature_shap[mask_u]), axis=0)   # (F,)

    order = np.argsort(imp_u)[::-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(F)
    width = 0.35

    ax.bar(x - width / 2, imp_n[order], width, label='Normal',
           color=BINARY_COLORS["normal"], edgecolor='none')
    ax.bar(x + width / 2, imp_u[order], width, label='Unusual',
           color=BINARY_COLORS["unusual"], edgecolor='none')

    ax.set_xticks(x)
    ax.set_xticklabels([feature_names[i] for i in order], rotation=30, ha='right')
    ax.set_ylabel('Mean |Feature SHAP|')
    ax.set_title(title, fontweight='bold')
    ax.legend(title='Window Type')

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_pruning_distribution(
    pruned: np.ndarray,
    labels: np.ndarray,
    T: int,
    save_path: Path,
    title: str = "Distribution of Retained Timesteps After Pruning",
):
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.arange(0, T + 2) - 0.5

    for val, name in [(0, "normal"), (1, "unusual")]:
        mask = labels == val
        if mask.sum() == 0:
            continue
        ax.hist(pruned[mask], bins=bins, alpha=0.6,
                label=f'{name.capitalize()} (n={mask.sum()})',
                color=BINARY_COLORS[name], edgecolor='none')

    ax.axvline(pruned.mean(), color='black', linestyle='--',
               label=f'Overall mean: {pruned.mean():.1f}')
    ax.set_xlabel('Retained Timesteps')
    ax.set_ylabel('Count')
    ax.set_title(title, fontweight='bold')
    ax.legend()

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_recency_analysis(
    event_shap: np.ndarray,
    save_path: Path,
    title: str = "SHAP Concentration by Recency",
):
    _, T = event_shap.shape
    mean_per_t = np.mean(np.abs(event_shap), axis=0)   # (T,)
    total = mean_per_t.sum()
    cum_recent = np.cumsum(mean_per_t[::-1]) / total    # (T,) [0] = most recent 1 step

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.bar(np.arange(T), mean_per_t[::-1] / total, color='#4878CF', edgecolor='none')
    ax.set_xlabel('Steps from present (0 = most recent)')
    ax.set_ylabel('Fraction of Total |Event SHAP|')
    ax.set_title('Per-Step Contribution', fontweight='bold')

    ax = axes[1]
    ax.plot(np.arange(1, T + 1), cum_recent, color='#4878CF', linewidth=2)
    for thresh, color in [(0.5, '#6ACC65'), (0.75, '#FFB347'), (0.9, '#D65F5F')]:
        n_steps = int(np.searchsorted(cum_recent, thresh)) + 1
        ax.axhline(thresh, color=color, linestyle='--', alpha=0.7,
                   label=f'{int(thresh * 100)}% in {n_steps} steps')
    ax.set_xlabel('Most-Recent N Timesteps')
    ax.set_ylabel('Cumulative Fraction of |Event SHAP|')
    ax.set_title('Cumulative Recency', fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_xlim(1, T)
    ax.grid(axis='y', alpha=0.3)

    plt.suptitle(title, fontweight='bold')
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_exemplar_event_shap(
    event_shap: np.ndarray,
    model_out: np.ndarray,
    labels: np.ndarray,
    save_dir: Path,
    n_per_group: int = 2,
):
    _, T = event_shap.shape
    timesteps = np.arange(T)

    for val, name in [(1, "unusual"), (0, "normal")]:
        mask = labels == val
        if mask.sum() == 0:
            continue
        indices = np.where(mask)[0]
        if val == 1:
            sorted_idx = indices[np.argsort(model_out[indices])[::-1]]
        else:
            sorted_idx = indices[np.argsort(model_out[indices])]

        for rank, idx in enumerate(sorted_idx[:n_per_group]):
            vals = event_shap[idx]   # (T,)
            colors = ['#D65F5F' if v > 0 else '#4878CF' for v in vals]

            fig, ax = plt.subplots(figsize=(10, 3.5))
            ax.bar(timesteps, vals, color=colors, edgecolor='none', width=0.8)
            ax.axhline(0, color='black', linewidth=0.7)
            ax.set_xlabel('Timestep')
            ax.set_ylabel('Event SHAP')
            ax.set_title(
                f'{name.capitalize()} Exemplar {rank + 1} - Sample {idx} '
                f'(error={model_out[idx]:.4f})',
                fontweight='bold',
                color=BINARY_COLORS[name],
            )

            pos, labs = _timestep_ticks(T, n_ticks=10)
            ax.set_xticks(pos)
            ax.set_xticklabels(labs, rotation=45, ha='right')

            plt.tight_layout()
            fname = save_dir / f"exemplar_event_shap_{name}_rank{rank}_sample{idx}.png"
            fig.savefig(fname, dpi=200, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            logger.info("  Saved: %s", fname.name)


def plot_model_output_distribution(
    model_out: np.ndarray,
    labels: np.ndarray,
    save_path: Path,
    title: str = "Reconstruction Error Distribution by Label",
):
    fig, ax = plt.subplots(figsize=(8, 4))

    for val, name in [(0, "normal"), (1, "unusual")]:
        mask = labels == val
        if mask.sum() == 0:
            continue
        ax.hist(model_out[mask], bins=30, alpha=0.6,
                label=f'{name.capitalize()} (n={mask.sum()})',
                color=BINARY_COLORS[name], edgecolor='none')

    ax.set_xlabel('Reconstruction Error')
    ax.set_ylabel('Count')
    ax.set_title(title, fontweight='bold')
    ax.legend()

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def make_all_figures(
    out_dir: Path,
    cluster_dir: Path,
    split: str,
):
    event_shap, feature_shap, pruned, labels, explain_idx, model_out = load_timeshap_outputs(out_dir)

    N, T = event_shap.shape
    F = feature_shap.shape[1]
    logger.info("N=%d, T=%d, F=%d", N, T, F)

    feature_names = INDICATOR_NAMES if len(INDICATOR_NAMES) == F else [f"Feature {i}" for i in range(F)]

    membership = load_cluster_membership(cluster_dir, split, explain_idx)
    if membership is None:
        logger.warning("No cluster labels found - skipping cluster figures")
    else:
        logger.info("Membership counts: %s", np.bincount(membership, minlength=N_CLUSTERS))

    figs_dir = out_dir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    plot_temporal_importance_binary(
        event_shap, labels,
        save_path=figs_dir / "fig1_temporal_importance_binary.png",
        title="Temporal Importance: Normal vs Unusual Windows",
    )

    plot_feature_importance(
        feature_shap, feature_names,
        save_path=figs_dir / "fig3_feature_importance.png",
        title="Feature Importance (Aggregated Over Time)",
    )

    plot_feature_importance_binary(
        feature_shap, labels, feature_names,
        save_path=figs_dir / "fig4_feature_importance_binary.png",
        title="Feature Importance: Normal vs Unusual",
    )



def parse_args():
    ap = argparse.ArgumentParser(description="TimeSHAP Track 2 Visualisation")
    ap.add_argument("--out_dir", type=str,
                    default=str(config.TIMESHAP_TRACK2_OUT_DIR),
                    help="Directory containing TimeSHAP Track 2 outputs")
    ap.add_argument("--cluster_dir", type=str,
                    default=str(config.CLUSTERING_OUT_DIR),
                    help="Directory containing cluster label files")
    ap.add_argument("--split", type=str, default="test",
                    choices=["train", "val", "test"])
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    make_all_figures(
        out_dir=Path(args.out_dir),
        cluster_dir=Path(args.cluster_dir),
        split=args.split,
    )
