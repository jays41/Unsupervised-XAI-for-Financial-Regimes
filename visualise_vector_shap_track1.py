from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import plot_config
from config import INDICATOR_NAMES, FAMILY_NAMES, FAMILY_KEYS
from plot_config import CLUSTER_COLORS

logger = logging.getLogger(__name__)

plot_config.setup_style()

TOPN: int = 10
MAX_HEATMAP_SAMPLES: int = 30


def load_outputs(out_dir: Path, split: str) -> Tuple:
    meta_path = out_dir / f"vectorshap_meta_{split}.json"
    lvl1_path = out_dir / f"vectorshap_level1_{split}.npy"
    base_path = out_dir / f"base_value_{split}.npy"
    full_path = out_dir / f"full_value_{split}.npy"

    for p in (meta_path, lvl1_path, base_path, full_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    shap1 = np.load(lvl1_path)   # (N, D, K)
    base = np.load(base_path)    # (N, K)
    full = np.load(full_path)    # (N, K)

    families = meta.get("level2_families", {})
    shap2 = {}
    for fam_name in families.keys():
        p = out_dir / f"vectorshap_level2_{fam_name}_{split}.npy"
        if p.exists():
            shap2[fam_name] = np.load(p)  # (N, K)

    indicator_names = meta.get("indicator_names", INDICATOR_NAMES)
    return meta, shap1, base, full, shap2, indicator_names

def load_cluster_labels(cluster_dir: Path, split: str, n_samples: int) -> Optional[np.ndarray]:
    labels_path = cluster_dir / f"labels_{split}_k3.npy"
    if labels_path.exists():
        labels = np.load(labels_path).astype(np.int32)
        return labels[:n_samples]
    return None

def plot_feature_cluster_heatmap(
    shap1: np.ndarray,
    feature_names: List[str],
    save_path: Path,
    title: str = "Feature Importance by Cluster Output",
):
    mean_shap = np.mean(shap1, axis=0)
    
    D, K = mean_shap.shape
    
    vmax = np.max(np.abs(mean_shap)) * 1.1
    vmin = -vmax
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    im = ax.imshow(
        mean_shap,
        aspect='auto',
        cmap='RdBu_r',
        vmin=vmin,
        vmax=vmax,
    )
    
    for i in range(D):
        for j in range(K):
            val = mean_shap[i, j]
            color = 'white' if abs(val) > vmax * 0.6 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center', 
                    fontsize=10, color=color, fontweight='medium')
    
    ax.set_xticks(range(K))
    ax.set_xticklabels([f'P(Cluster {k})' for k in range(K)])
    ax.set_yticks(range(D))
    ax.set_yticklabels(feature_names)
    
    ax.set_xlabel('Cluster Output', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Mean SHAP Value', fontsize=11)
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_family_cluster_heatmap(
    shap2: Dict[str, np.ndarray],
    save_path: Path,
    title: str = "Feature Family Importance by Cluster Output",
):
    fam_names_display = []
    fam_means = []
    
    for key, display_name in zip(FAMILY_KEYS, FAMILY_NAMES):
        if key in shap2:
            fam_names_display.append(display_name)
            fam_means.append(np.mean(shap2[key], axis=0))  # (K,)    
    
    mean_shap = np.array(fam_means)  # (n_families, K)
    n_fam, K = mean_shap.shape
    
    vmax = np.max(np.abs(mean_shap)) * 1.1
    vmin = -vmax
    
    fig, ax = plt.subplots(figsize=(6, 3.5))
    
    im = ax.imshow(
        mean_shap,
        aspect='auto',
        cmap='RdBu_r',
        vmin=vmin,
        vmax=vmax,
    )
    
    for i in range(n_fam):
        for j in range(K):
            val = mean_shap[i, j]
            color = 'white' if abs(val) > vmax * 0.6 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=11, color=color, fontweight='medium')
    
    ax.set_xticks(range(K))
    ax.set_xticklabels([f'P(Cluster {k})' for k in range(K)])
    ax.set_yticks(range(n_fam))
    ax.set_yticklabels(fam_names_display)
    
    ax.set_xlabel('Cluster Output', fontsize=12)
    ax.set_ylabel('Feature Family', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    
    cbar = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
    cbar.set_label('Mean SHAP Value', fontsize=11)
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_regime_radar(
    shap1: np.ndarray,
    membership: np.ndarray,
    feature_names: List[str],
    save_path: Path,
    title: str = "Feature Attribution Profile by Regime",
):
    N, D, K = shap1.shape
    n_clusters = len(np.unique(membership))
    
    profiles = []
    for c in range(n_clusters):
        mask = membership == c
        if mask.sum() == 0:
            profiles.append(np.zeros(D))
            continue
        cluster_shap = np.mean(np.abs(shap1[mask]), axis=(0, 2))  # (D,)
        profiles.append(cluster_shap)
    
    profiles = np.array(profiles)  # (n_clusters, D)
    
    max_vals = profiles.max(axis=0, keepdims=True)
    max_vals = np.where(max_vals == 0, 1, max_vals)
    profiles_norm = profiles / max_vals
    
    angles = np.linspace(0, 2 * np.pi, D, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(7, 6), subplot_kw=dict(polar=True))
    
    for c in range(n_clusters):
        values = profiles_norm[c].tolist()
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=f'Cluster {c}',
                color=CLUSTER_COLORS[c], markersize=6)
        ax.fill(angles, values, alpha=0.15, color=CLUSTER_COLORS[c])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_names, fontsize=10)
    ax.set_ylim(0, 1.1)
    
    ax.set_yticklabels([])
    
    ax.set_title(title, fontsize=13, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.1))
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_global_importance_bar(
    shap1: np.ndarray,
    feature_names: List[str],
    save_path: Path,
    title: str = "Global Feature Importance",
):
    importance = np.mean(np.abs(shap1), axis=(0, 2))  # (D,)
    
    order = np.argsort(importance)[::-1]
    sorted_imp = importance[order]
    sorted_names = [feature_names[i] for i in order]
    
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    y_pos = np.arange(len(sorted_names))
    bars = ax.barh(y_pos, sorted_imp, color='#4878CF', edgecolor='none', height=0.7)
    
    for i, (bar, val) in enumerate(zip(bars, sorted_imp)):
        ax.text(val + sorted_imp.max() * 0.02, i, f'{val:.4f}', 
                va='center', fontsize=10)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names)
    ax.invert_yaxis()
    ax.set_xlabel('Mean |SHAP|', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlim(0, sorted_imp.max() * 1.2)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_per_cluster_importance(
    shap1: np.ndarray,
    feature_names: List[str],
    save_path: Path,
    title: str = "Feature Importance per Cluster Output",
):
    importance = np.mean(np.abs(shap1), axis=0)
    D, K = importance.shape
    
    overall = importance.mean(axis=1)
    order = np.argsort(overall)[::-1]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    x = np.arange(D)
    width = 0.25
    offsets = np.array([-1, 0, 1]) * width
    
    for k in range(K):
        vals = importance[order, k]
        ax.bar(x + offsets[k], vals, width, label=f'P(Cluster {k})',
               color=CLUSTER_COLORS[k], edgecolor='none')
    
    ax.set_xticks(x)
    ax.set_xticklabels([feature_names[i] for i in order], rotation=30, ha='right')
    ax.set_ylabel('Mean |SHAP|', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(title='Output', loc='upper right')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_sample_heatmap(
    shap1: np.ndarray,
    full: np.ndarray,
    feature_names: List[str],
    cluster_idx: int,
    save_path: Path,
    max_samples: int = 30,
    title: Optional[str] = None,
):
    N, D, K = shap1.shape
    
    order = np.argsort(full[:, cluster_idx])[::-1]
    take = order[:min(max_samples, N)]
    
    mat = shap1[take, :, cluster_idx]  # (n_samples, D)
    
    vmax = np.max(np.abs(mat)) * 1.0
    vmin = -vmax
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(mat, aspect='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax)
    
    ax.set_xticks(range(D))
    ax.set_xticklabels(feature_names, rotation=35, ha='right')
    ax.set_ylabel(f'Samples (ranked by P(Cluster {cluster_idx}))', fontsize=11)
    ax.set_xlabel('Feature', fontsize=11)
    
    if title is None:
        title = f'SHAP Values for Top Samples -> Cluster {cluster_idx}'
    ax.set_title(title, fontsize=13, fontweight='bold')
    
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('SHAP Value', fontsize=11)
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_waterfall(
    shap_vec: np.ndarray,
    feature_names: List[str],
    base_val: float,
    full_val: float,
    save_path: Path,
    title: str = "Feature Contributions",
    topn: int = 10,
):
    D = len(shap_vec)
    
    idx = np.argsort(np.abs(shap_vec))[::-1][:min(topn, D)]
    
    contribs = shap_vec[idx]
    labels = [feature_names[i] for i in idx]
    
    sort_order = np.argsort(contribs)
    contribs = contribs[sort_order]
    labels = [labels[i] for i in sort_order]
    
    fig, ax = plt.subplots(figsize=(9, 5))
    
    running = base_val
    y_positions = range(len(contribs))
    
    for i, (label, contrib) in enumerate(zip(labels, contribs)):
        start = running
        end = running + contrib
        color = '#D65F5F' if contrib > 0 else '#4878CF'
        
        ax.barh(i, contrib, left=start, color=color, edgecolor='none', height=0.6)
        
        text_x = end + 0.005 if contrib > 0 else end - 0.005
        ha = 'left' if contrib > 0 else 'right'
        ax.text(text_x, i, f'{contrib:+.4f}', va='center', ha=ha, fontsize=9)
        
        running = end
    
    ax.axvline(base_val, color='gray', linestyle='--', linewidth=1.5, label=f'Base: {base_val:.4f}')
    ax.axvline(full_val, color='black', linestyle='-', linewidth=1.5, label=f'Output: {full_val:.4f}')
    
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Probability', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_additivity_check(
    shap1: np.ndarray,
    base: np.ndarray,
    full: np.ndarray,
    save_path: Path,
):
    approx = base + shap1.sum(axis=1)  # (N, K)
    error = approx - full
    abs_error = np.abs(error)
    
    max_err_per_sample = abs_error.max(axis=1)
    
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    ax.hist(max_err_per_sample, bins=40, color='#4878CF', edgecolor='white', alpha=0.8)
    
    ax.axvline(max_err_per_sample.mean(), color='red', linestyle='--', 
               label=f'Mean: {max_err_per_sample.mean():.2e}')
    ax.axvline(max_err_per_sample.max(), color='darkred', linestyle=':',
               label=f'Max: {max_err_per_sample.max():.2e}')
    
    ax.set_xlabel('Max |Error| per Sample', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Additivity Check: base + ΣSHAP ≈ full', fontsize=13, fontweight='bold')
    ax.legend()
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_base_vs_full_scatter(
    base: np.ndarray,
    full: np.ndarray,
    cluster_idx: int,
    save_path: Path,
):
    fig, ax = plt.subplots(figsize=(6, 5))
    
    ax.scatter(base[:, cluster_idx], full[:, cluster_idx], s=15, alpha=0.6, 
               color=CLUSTER_COLORS[cluster_idx])
    
    lims = [
        min(base[:, cluster_idx].min(), full[:, cluster_idx].min()),
        max(base[:, cluster_idx].max(), full[:, cluster_idx].max()),
    ]
    ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=1)
    
    ax.set_xlabel('Base Value (masked input)', fontsize=11)
    ax.set_ylabel('Full Value (actual input)', fontsize=11)
    ax.set_title(f'Base vs Full: P(Cluster {cluster_idx})', fontsize=13, fontweight='bold')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def make_all_figures(
    out_dir: Path,
    split: str,
    cluster_dir: Path,
):    
    meta, shap1, base, full, shap2, indicator_names = load_outputs(out_dir, split)
    N, D, K = shap1.shape
    
    if D == len(INDICATOR_NAMES):
        feature_names = INDICATOR_NAMES
    else:
        feature_names = indicator_names
    
    membership = load_cluster_labels(cluster_dir, split, N)
    if membership is None:
        logger.warning("No cluster labels found, using argmax(full) as proxy")
        membership = np.argmax(full, axis=1)
    
    figs_dir = out_dir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    plot_feature_cluster_heatmap(
        shap1, feature_names,
        save_path=figs_dir / "fig_feature_cluster_heatmap.png",
        title="Mean SHAP: Feature × Cluster Output",
    )

    plot_family_cluster_heatmap(
        shap2,
        save_path=figs_dir / "fig_family_cluster_heatmap.png",
        title="Mean SHAP: Feature Family × Cluster Output",
    )

    plot_regime_radar(
        shap1, membership, feature_names,
        save_path=figs_dir / "fig_regime_radar.png",
        title="Feature Attribution Profile by Cluster Membership",
    )

    plot_global_importance_bar(
        shap1, feature_names,
        save_path=figs_dir / "fig_global_importance.png",
        title="Global Feature Importance (Mean |SHAP|)",
    )

    plot_per_cluster_importance(
        shap1, feature_names,
        save_path=figs_dir / "fig_per_cluster_importance.png",
        title="Feature Importance per Cluster Output",
    )
    

if __name__ == "__main__":
    make_all_figures(
        out_dir=Path("outputs/vector_shap_track1"),
        split="test",
        cluster_dir=Path("outputs/clustering_analysis"),
    )