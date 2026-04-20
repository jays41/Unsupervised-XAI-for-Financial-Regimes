from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import plot_config
import config
from config import INDICATOR_NAMES, N_CLUSTERS
from plot_config import CLUSTER_COLORS

logger = logging.getLogger(__name__)

plot_config.setup_style()


def load_timeshap_outputs(out_dir: Path):
    event_shap = np.load(out_dir / "cluster_event_shap.npy")
    feature_shap = np.load(out_dir / "cluster_feature_shap.npy")
    outputs = np.load(out_dir / "cluster_outputs.npy")
    pruned = np.load(out_dir / "cluster_pruned_timesteps.npy")

    logger.info("  event_shap   : %s", event_shap.shape)
    logger.info("  feature_shap : %s", feature_shap.shape)
    logger.info("  outputs      : %s", outputs.shape)
    logger.info("  pruned       : %s", pruned.shape)

    return event_shap, feature_shap, outputs, pruned


def load_cluster_labels(cluster_dir: Path, split: str, n_samples: int) -> np.ndarray:
    path = cluster_dir / f"labels_{split}_k3.npy"
    if path.exists():
        labels = np.load(path).astype(np.int32)
        return labels[:n_samples]
    return None

def _timestep_ticks(T: int, n_ticks: int = 10):
    positions = np.linspace(0, T - 1, min(n_ticks, T)).astype(int)
    labels = [f"t-{T - 1 - t}" for t in positions]
    return positions, labels

def plot_temporal_importance_curve(
    event_shap: np.ndarray,
    save_path: Path,
    title: str = "Temporal Importance by Cluster Output",
):
    N, T, K = event_shap.shape
    # Mean |SHAP| over samples -> (T, K)
    temporal_imp = np.mean(np.abs(event_shap), axis=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    timesteps = np.arange(T)
    for k in range(K):
        ax.plot(timesteps, temporal_imp[:, k],
                label=f'P(Cluster {k})', color=CLUSTER_COLORS[k],
                linewidth=2, marker='o', markersize=3, alpha=0.85)

    ax.axvspan(T - 10, T - 1, alpha=0.1, color='gray', label='Recent (last 10)')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Mean |Event SHAP|')
    ax.set_title(title, fontweight='bold')

    pos, labs = _timestep_ticks(T)
    ax.set_xticks(pos)
    ax.set_xticklabels(labs, rotation=45, ha='right')
    ax.legend(title='Output', loc='upper left')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_temporal_importance_faceted(
    event_shap: np.ndarray,
    save_path: Path,
    title: str = "Temporal Importance per Cluster Output",
):
    N, T, K = event_shap.shape
    fig, axes = plt.subplots(1, K, figsize=(4 * K, 4), sharey=True)
    if K == 1:
        axes = [axes]

    timesteps = np.arange(T)
    for k, ax in enumerate(axes):
        imp = np.mean(np.abs(event_shap[:, :, k]), axis=0)  # (T,)
        ax.plot(timesteps, imp, color=CLUSTER_COLORS[k], linewidth=2)
        ax.fill_between(timesteps, 0, imp, color=CLUSTER_COLORS[k], alpha=0.25)

        ax.set_xlabel('Timestep')
        if k == 0:
            ax.set_ylabel('Mean |Event SHAP|')
        ax.set_title(f'P(Cluster {k})', fontweight='bold')

        pos, labs = _timestep_ticks(T, n_ticks=5)
        ax.set_xticks(pos)
        ax.set_xticklabels(labs, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle(title, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_feature_importance(
    feature_shap: np.ndarray,
    feature_names: List[str],
    save_path: Path,
    title: str = "Feature Importance (Aggregated Over Time)",
):

    N, F, K = feature_shap.shape
    # Mean |SHAP| over samples -> (F, K)
    mean_imp = np.mean(np.abs(feature_shap), axis=0)

    order = np.argsort(mean_imp.sum(axis=1))
    sorted_imp = mean_imp[order]
    sorted_names = [feature_names[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.arange(F)
    width = 0.25
    offsets = np.linspace(-(K - 1) * width / 2, (K - 1) * width / 2, K)

    for k in range(K):
        ax.barh(y + offsets[k], sorted_imp[:, k], width,
                       label=f'P(Cluster {k})', color=CLUSTER_COLORS[k],
                       edgecolor='none')

    ax.set_yticks(y)
    ax.set_yticklabels(sorted_names)
    ax.set_xlabel('Mean |Feature SHAP|')
    ax.set_title(title, fontweight='bold')
    ax.legend(title='Output', loc='lower right')

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_pruning_distribution(
    pruned: np.ndarray,
    membership: np.ndarray,
    T: int,
    save_path: Path,
    title: str = "Distribution of Retained Timesteps After Pruning",
):
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.arange(0, T + 2) - 0.5

    for c in range(N_CLUSTERS):
        mask = membership == c
        if mask.sum() == 0:
            continue
        ax.hist(pruned[mask], bins=bins, alpha=0.6,
                label=f'Cluster {c} (n={mask.sum()})',
                color=CLUSTER_COLORS[c], edgecolor='none')

    overall_mean = pruned.mean()
    ax.axvline(overall_mean, color='black', linestyle='--',
               label=f'Overall mean: {overall_mean:.1f}')

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
    N, T, _ = event_shap.shape

    # Mean |event SHAP| per timestep -> (T,)
    mean_per_t = np.mean(np.abs(event_shap), axis=(0, 2))
    total = mean_per_t.sum()
    cum_recent = np.cumsum(mean_per_t[::-1]) / total  # (T,) [0] = most recent 1 step

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
                   label=f'{int(thresh*100)}% in {n_steps} steps')
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
    outputs: np.ndarray,
    membership: np.ndarray,
    save_dir: Path,
    n_per_cluster: int = 2,
):
    N, T, K = event_shap.shape
    timesteps = np.arange(T)

    for c in range(N_CLUSTERS):
        mask = membership == c
        if mask.sum() == 0:
            continue
        indices = np.where(mask)[0]
        probs = outputs[indices, c]
        top_indices = indices[np.argsort(probs)[::-1][:n_per_cluster]]

        for rank, idx in enumerate(top_indices):
            fig, axes = plt.subplots(1, K, figsize=(4 * K, 3.5), sharey=False)
            if K == 1:
                axes = [axes]

            for k, ax in enumerate(axes):
                vals = event_shap[idx, :, k]  # (T,)
                colors = ['#D65F5F' if v > 0 else '#4878CF' for v in vals]
                ax.bar(timesteps, vals, color=colors, edgecolor='none', width=0.8)
                ax.axhline(0, color='black', linewidth=0.7)
                ax.set_title(f'P(Cluster {k})', fontweight='bold')
                ax.set_xlabel('Timestep')
                if k == 0:
                    ax.set_ylabel('Event SHAP')

                pos, labs = _timestep_ticks(T, n_ticks=8)
                ax.set_xticks(pos)
                ax.set_xticklabels(labs, rotation=45, ha='right')

            p_c = outputs[idx, c]
            plt.suptitle(
                f'Cluster {c} Exemplar {rank + 1} (sample {idx}, p={p_c:.3f})',
                fontweight='bold'
            )
            plt.tight_layout()
            fname = save_dir / f"exemplar_event_shap_cluster{c}_rank{rank}_sample{idx}.png"
            fig.savefig(fname, dpi=200, bbox_inches='tight', facecolor='white')
            plt.close(fig)


def make_all_figures(
    out_dir: Path,
    cluster_dir: Path,
    split: str,
):
    event_shap, feature_shap, outputs, pruned = load_timeshap_outputs(out_dir)

    N, T, K = event_shap.shape
    F = feature_shap.shape[1]
    logger.info("N=%d, T=%d, K=%d, F=%d", N, T, K, F)

    feature_names = INDICATOR_NAMES if len(INDICATOR_NAMES) == F else [f"Feature {i}" for i in range(F)]

    membership = load_cluster_labels(cluster_dir, split, N)
    if membership is None:
        logger.warning("No cluster labels found - using argmax(outputs) as proxy")
        membership = np.argmax(outputs, axis=1).astype(np.int32)
    logger.info("Membership counts: %s", np.bincount(membership, minlength=N_CLUSTERS))

    figs_dir = out_dir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    plot_temporal_importance_curve(
        event_shap,
        save_path=figs_dir / "fig1_temporal_importance_curve.png",
        title="Temporal Importance by Cluster Output",
    )

    plot_temporal_importance_faceted(
        event_shap,
        save_path=figs_dir / "fig3_temporal_importance_faceted.png",
        title="Temporal Importance per Cluster Output",
    )

    plot_feature_importance(
        feature_shap, feature_names,
        save_path=figs_dir / "fig4_feature_importance.png",
        title="Feature Importance (Aggregated Over Time)",
    )



def parse_args():
    ap = argparse.ArgumentParser(description="TimeSHAP Track 1 Visualisation")
    ap.add_argument("--out_dir", type=str,
                    default=str(config.TIMESHAP_TRACK1_OUT_DIR),
                    help="Directory containing TimeSHAP Track 1 outputs")
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