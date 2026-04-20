import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import seaborn as sns
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, confusion_matrix
from scipy.stats import mode as scipy_mode
import config

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

CLUSTERING_DIR = config.CLUSTERING_OUT_DIR
VIX_VALIDATION_DIR = Path("outputs/vix_validation")
WINDOW_END_DATES_PATH = CLUSTERING_DIR / "window_end_dates.npy"

VIX_LOW_THRESHOLD = 15.0
VIX_HIGH_THRESHOLD = 25.0
PERSISTENCE_FLAG_THRESHOLD = 0.85

MAJOR_EVENTS: dict[str, str] = {
    "2000-03-10": "Dot-com Peak",
    "2001-09-11": "9/11",
    "2002-10-09": "Dot-com Trough",
    "2008-09-15": "Lehman Collapse",
    "2010-05-06": "Flash Crash",
    "2011-08-08": "US Debt Downgrade",
    "2015-08-24": "China Black Monday",
    "2018-02-05": "VIX Spike",
    "2018-10-10": "Oct Correction",
    "2018-12-24": "Dec Selloff",
    "2020-03-16": "COVID Crash",
    "2022-01-03": "Rate Hike Cycle",
}

_DISPLAY_LABELS: dict[str, str] = {
    "2000-03-10": "Dot-com Peak",
    "2001-09-11": "9/11",
    "2002-10-09": "Dot-com Trough",
    "2008-09-15": "Lehman",
    "2010-05-06": "Flash Crash",
    "2011-08-08": "US Downgrade",
    "2015-08-24": "China Black Mon.",
    "2018-02-05": "VIX Spike",
    "2018-10-10": "Oct Correction",
    "2018-12-24": "Dec Selloff",
    "2020-03-16": "COVID Crash",
    "2022-01-03": "Rate Hikes",
}



def load_vix_data(vix_file: str = "vix_data.csv") -> pd.DataFrame:
    path = Path(vix_file)
    if not path.exists():
        raise FileNotFoundError(f"{vix_file} not found. Run fetch_vix_data.py first.")

    vix = pd.read_csv(path, index_col=0, parse_dates=True)

    if "VIX" not in vix.columns:
        if len(vix.columns) == 1:
            vix.columns = ["VIX"]
        else:
            raise ValueError(
                f"Cannot identify VIX column. Available: {vix.columns.tolist()}"
            )

    return vix[["VIX"]].sort_index()


def load_window_end_dates() -> pd.DatetimeIndex:
    dates_raw = np.load(WINDOW_END_DATES_PATH, allow_pickle=True)
    return pd.to_datetime(dates_raw)


def load_selection_summary() -> dict[int, dict]:
    summary_path = CLUSTERING_DIR / "selection_summary.json"
    with open(summary_path) as f:
        return {int(k): v for k, v in json.load(f).items()}


def discover_available_ks() -> list[int]:
    label_files = sorted(CLUSTERING_DIR.glob("labels_all_k*.npy"))
    if not label_files:
        raise FileNotFoundError(
            f"No label files found in {CLUSTERING_DIR}. Run clustering_analysis.py first."
        )
    return [int(p.stem.replace("labels_all_k", "")) for p in label_files]



def create_vix_regime_labels(vix_values: np.ndarray) -> np.ndarray:
    """Assign discrete regime labels: 0=Low (<15), 1=Normal (15-25), 2=High (>=25)."""
    vix_values = np.asarray(vix_values).flatten()
    regimes = np.zeros(len(vix_values), dtype=int)
    regimes[(vix_values >= VIX_LOW_THRESHOLD) & (vix_values < VIX_HIGH_THRESHOLD)] = 1
    regimes[vix_values >= VIX_HIGH_THRESHOLD] = 2
    return regimes


def align_vix_with_dates(
    vix_df: pd.DataFrame, window_dates: pd.DatetimeIndex
) -> tuple[np.ndarray, np.ndarray]:
    indices = vix_df.index.get_indexer(window_dates, method="nearest")
    vix_aligned = vix_df["VIX"].iloc[indices].values
    return vix_aligned, create_vix_regime_labels(vix_aligned)


def load_splits_from_summary(
    summary: dict, total_windows: int
) -> dict[str, tuple[int, int]]:
    sizes = next(iter(summary.values()))["sizes"]
    n_train = sum(sizes["train"].values())
    n_val = sum(sizes["val"].values())
    n_test = sum(sizes["test"].values())

    assert n_train + n_val + n_test == total_windows, (
        f"Split lengths {n_train}+{n_val}+{n_test} don't match total windows {total_windows}."
    )

    return {
        "train": (0, n_train),
        "val": (n_train, n_train + n_val),
        "test": (n_train + n_val, n_train + n_val + n_test),
    }



def compute_cluster_purity(
    cluster_labels: np.ndarray, regime_labels: np.ndarray
) -> tuple[float, list[float]]:
    n_clusters = len(np.unique(cluster_labels))
    purities = []

    for c in range(n_clusters):
        mask = cluster_labels == c
        if mask.sum() == 0:
            continue
        regime_counts = np.bincount(regime_labels[mask])
        purities.append(regime_counts.max() / mask.sum())

    return float(np.mean(purities)), purities


def compute_transition_matrix(labels: np.ndarray) -> np.ndarray:
    unique_states = np.unique(labels)
    n_states = len(unique_states)

    if n_states == 1:
        return np.array([[1.0]])

    state_to_idx = {state: idx for idx, state in enumerate(unique_states)}
    transition_matrix = np.zeros((n_states, n_states))

    for i in range(len(labels) - 1):
        transition_matrix[state_to_idx[labels[i]], state_to_idx[labels[i + 1]]] += 1

    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return transition_matrix / row_sums



def plot_confusion_matrix(
    cluster_labels: np.ndarray, regime_labels: np.ndarray, save_path: Path
) -> None:
    cm = confusion_matrix(regime_labels, cluster_labels)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = cm.astype(float) / np.where(row_sums == 0, 1, row_sums)

    regime_ticklabels = ["Low VIX\n(<15)", "Normal VIX\n(15-25)", "High VIX\n(>=25)"]
    cluster_ticklabels = [f"Cluster {i}" for i in range(cm.shape[1])]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=True, ax=ax1,
        xticklabels=cluster_ticklabels,
        yticklabels=regime_ticklabels,
    )
    ax1.set_xlabel("Cluster Assignment", fontsize=12)
    ax1.set_ylabel("VIX Regime", fontsize=12)
    ax1.set_title("Raw Counts", fontsize=13, fontweight="bold")

    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues", cbar=True, ax=ax2,
        vmin=0, vmax=1,
        xticklabels=cluster_ticklabels,
        yticklabels=regime_ticklabels,
    )
    ax2.set_xlabel("Cluster Assignment", fontsize=12)
    ax2.set_ylabel("VIX Regime", fontsize=12)
    ax2.set_title("Row-Normalised (proportion per regime)", fontsize=13, fontweight="bold")

    fig.suptitle(
        "Confusion Matrix: VIX Regimes vs Cluster Assignments",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_transition_matrices(
    cluster_trans: np.ndarray, regime_trans: np.ndarray, save_path: Path
) -> None:
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(
        cluster_trans, annot=True, fmt=".3f", cmap="YlOrRd",
        ax=ax1, vmin=0, vmax=1, cbar_kws={"label": "Probability"},
    )
    ax1.set_xlabel("Next Cluster", fontsize=11)
    ax1.set_ylabel("Current Cluster", fontsize=11)
    ax1.set_title("Cluster Transition Matrix\nP(cluster_t+1 | cluster_t)", fontsize=12, fontweight="bold")

    sns.heatmap(
        regime_trans, annot=True, fmt=".3f", cmap="YlGnBu",
        ax=ax2, vmin=0, vmax=1, cbar_kws={"label": "Probability"},
        xticklabels=["Low", "Normal", "High"],
        yticklabels=["Low", "Normal", "High"],
    )
    ax2.set_xlabel("Next Regime", fontsize=11)
    ax2.set_ylabel("Current Regime", fontsize=11)
    ax2.set_title("VIX Regime Transition Matrix\nP(regime_t+1 | regime_t)", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def _majority_smooth(labels: np.ndarray, window: int) -> np.ndarray:
    """Iterative majority vote smoothing for display only"""

    result = labels.astype(int).copy()
    half = window // 2
    for _ in range(3):  # three passes to remove short bursts
        tmp = result.copy()
        for i in range(len(result)):
            lo = max(0, i - half)
            hi = min(len(result), i + half + 1)
            tmp[i] = int(scipy_mode(result[lo:hi], keepdims=False).mode)
        result = tmp
    return result


def _gantt_spans(
    dates: pd.DatetimeIndex,
    labels: np.ndarray,
) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    spans = []
    start_idx = 0
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            spans.append((dates[start_idx], dates[i - 1], int(labels[i - 1])))
            start_idx = i
    spans.append((dates[start_idx], dates[-1], int(labels[-1])))
    return spans


def _draw_gantt(
    ax: plt.Axes,
    dates: pd.DatetimeIndex,
    labels: np.ndarray,
    row_labels: list[str],
    colors: dict[int, str],
    title: str,
    smooth_window: int = 63,
) -> None:

    display_labels = _majority_smooth(labels, smooth_window)
    spans = _gantt_spans(dates, display_labels)

    for start, end, lbl in spans:
        ax.barh(
            y=0,
            width=end - start,
            left=start,
            height=0.8,
            color=colors[lbl],
            edgecolor="none",
            align="center",
        )

    patches = [
        mpatches.Patch(color=colors[i], label=row_labels[i])
        for i in sorted(colors)
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=9, framealpha=0.9,
              edgecolor="0.7")
    ax.set_yticks([])
    ax.set_ylim(-0.5, 0.5)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlim(dates[0], dates[-1])
    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.spines["bottom"].set_color("0.5")


def plot_temporal_visualisation(
    window_dates: pd.DatetimeIndex,
    cluster_labels: np.ndarray,
    vix_values: np.ndarray,
    regime_labels: np.ndarray,
    splits: dict[str, tuple[int, int]],
    save_path: Path,
    major_events: dict[str, str] | None = None,
) -> None:

    dates = pd.to_datetime(window_dates)
    n_clusters = int(cluster_labels.max()) + 1

    viridis = cm.get_cmap("viridis", n_clusters)
    cluster_colors = {i: viridis(i / max(n_clusters - 1, 1)) for i in range(n_clusters)}
    cluster_row_labels = [f"Cluster {i}" for i in range(n_clusters)]

    regime_colors = {0: "#a8d5a2", 1: "#fffacd", 2: "#f4a0a0"}

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(16, 8), sharex=True,
        gridspec_kw={"height_ratios": [1, 2.5]},
    )
    fig.subplots_adjust(hspace=0.06)

    _draw_gantt(ax1, dates, cluster_labels, cluster_row_labels, cluster_colors,
                "Cluster Regime Spans")

    for split_name, (_, end) in splits.items():
        if end < len(dates):
            for ax in (ax1, ax2):
                ax.axvline(dates[end], color="0.3", linestyle="--", alpha=0.6,
                           linewidth=1.2)

    ax2.plot(dates, vix_values, color="black", linewidth=1.5, label="VIX", zorder=3)
    ax2.axhline(VIX_LOW_THRESHOLD, color="green", linestyle="--", alpha=0.7,
                label=f"Low/Normal ({int(VIX_LOW_THRESHOLD)})")
    ax2.axhline(VIX_HIGH_THRESHOLD, color="orange", linestyle="--", alpha=0.7,
                label=f"Normal/High ({int(VIX_HIGH_THRESHOLD)})")

    for i in range(len(dates) - 1):
        ax2.axvspan(dates[i], dates[i + 1], alpha=0.13,
                    color=regime_colors[regime_labels[i]], linewidth=0)

    ax2.set_ylabel("VIX Level", fontsize=11)
    ax2.set_title("VIX Level with Regime Shading", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=9, loc="upper right", framealpha=0.9, edgecolor="0.7")
    ax2.grid(alpha=0.3, axis="y")
    ax2.set_xlabel("Date", fontsize=11)

    for split_name, (_, end) in splits.items():
        if end < len(dates):
            ax1.text(dates[end], 1.01, split_name,
                     rotation=90, va="bottom", ha="right", fontsize=8, color="0.4",
                     transform=ax1.get_xaxis_transform())

    _near_end = dates[-1] - pd.Timedelta(days=200)

    if major_events:
        for event_date_str, event_name in major_events.items():
            event_date = pd.to_datetime(event_date_str)
            if dates.min() <= event_date <= dates.max():
                for ax in (ax1, ax2):
                    ax.axvline(event_date, color="red", linestyle=":",
                               alpha=0.75, linewidth=1.2, zorder=4)
                if event_date > _near_end:
                    continue
                label = _DISPLAY_LABELS.get(event_date_str, event_name)
                ax2.text(
                    event_date, 0.97, label,
                    rotation=90, va="top", ha="right",
                    fontsize=9, color="red",
                    transform=ax2.get_xaxis_transform(),
                    clip_on=True,
                )

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()



def main() -> None:
    VIX_VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    window_dates = load_window_end_dates()
    summary = load_selection_summary()
    splits = load_splits_from_summary(summary, total_windows=len(window_dates))
    
    vix_df = load_vix_data()
    vix_aligned, regime_labels = align_vix_with_dates(vix_df, window_dates)

    regime_counts = np.bincount(regime_labels)
    n = len(regime_labels)
    logger.info("\n  VIX Regime distribution:")
    logger.info(f"    Low volatility (<{VIX_LOW_THRESHOLD}): {regime_counts[0]} ({100 * regime_counts[0] / n:.1f}%)")
    logger.info(f"    Normal ({VIX_LOW_THRESHOLD}-{VIX_HIGH_THRESHOLD}): {regime_counts[1]} ({100 * regime_counts[1] / n:.1f}%)")
    logger.info(f"    High volatility (>={VIX_HIGH_THRESHOLD}): {regime_counts[2]} ({100 * regime_counts[2] / n:.1f}%)")

    ks = discover_available_ks()

    results_all_k: dict[int, dict] = {}

    for k in ks:
        cluster_labels = np.load(CLUSTERING_DIR / f"labels_all_k{k}.npy")

        if len(cluster_labels) != len(window_dates):
            logger.warning(
                f"  K={k}: label length {len(cluster_labels)} != "
                f"window count {len(window_dates)}. Skipping."
            )
            continue

        logger.info(f"\n  Validating K={k}...")

        nmi = normalized_mutual_info_score(regime_labels, cluster_labels)
        ari = adjusted_rand_score(regime_labels, cluster_labels)
        avg_purity, purities = compute_cluster_purity(cluster_labels, regime_labels)
        cluster_trans = compute_transition_matrix(cluster_labels)
        regime_trans = compute_transition_matrix(regime_labels)
        cluster_persistence = np.diag(cluster_trans)
        regime_persistence = np.diag(regime_trans)

        per_split_metrics: dict[str, dict[str, float]] = {}
        for split_name, (start, end) in splits.items():
            cl_split = cluster_labels[start:end]
            rl_split = regime_labels[start:end]
            per_split_metrics[split_name] = {
                "nmi": float(normalized_mutual_info_score(rl_split, cl_split)),
                "ari": float(adjusted_rand_score(rl_split, cl_split)),
            }

        results_all_k[k] = {
            "nmi": float(nmi),
            "ari": float(ari),
            "avg_purity": float(avg_purity),
            "purities": [float(p) for p in purities],
            "cluster_persistence": cluster_persistence.tolist(),
            "regime_persistence": regime_persistence.tolist(),
            "avg_cluster_persistence": float(cluster_persistence.mean()),
            "avg_regime_persistence": float(regime_persistence.mean()),
            "per_split": per_split_metrics,
        }

        flag_cluster = " [HIGH]" if cluster_persistence.mean() > PERSISTENCE_FLAG_THRESHOLD else ""
        flag_regime = " [HIGH]" if regime_persistence.mean() > PERSISTENCE_FLAG_THRESHOLD else ""
        logger.info(f"    NMI (all):            {nmi:.4f}")
        logger.info(f"    ARI (all):            {ari:.4f}")
        for split_name, m in per_split_metrics.items():
            logger.info(f"    NMI ({split_name:<5}):        {m['nmi']:.4f}   ARI: {m['ari']:.4f}")
        logger.info(f"    Avg Purity:           {avg_purity:.4f}")
        logger.info(f"    Cluster Persistence:  {cluster_persistence.mean():.4f} (mean){flag_cluster}")
        logger.info(f"    Regime Persistence:   {regime_persistence.mean():.4f} (mean){flag_regime}")

        plot_confusion_matrix(cluster_labels, regime_labels, VIX_VALIDATION_DIR / f"confusion_matrix_k{k}.png")
        plot_transition_matrices(cluster_trans, regime_trans, VIX_VALIDATION_DIR / f"transition_matrices_k{k}.png")
        plot_temporal_visualisation(
            window_dates, cluster_labels, vix_aligned, regime_labels, splits,
            VIX_VALIDATION_DIR / f"temporal_viz_k{k}.png",
            major_events=MAJOR_EVENTS,
        )

    with open(VIX_VALIDATION_DIR / "vix_validation_summary.json", "w") as f:
        json.dump(results_all_k, f, indent=2)

    vix_data = {
        "window_dates": [str(d) for d in window_dates],
        "vix_values": vix_aligned.tolist(),
        "regime_labels": regime_labels.tolist(),
    }
    with open(VIX_VALIDATION_DIR / "vix_aligned_data.json", "w") as f:
        json.dump(vix_data, f, indent=2)

    np.save(VIX_VALIDATION_DIR / "vix_aligned.npy", vix_aligned)
    np.save(VIX_VALIDATION_DIR / "regime_labels.npy", regime_labels)

    if results_all_k:
        logger.info(f"\n{'K':<5} {'NMI':<10} {'ARI':<10} {'Purity':<10} {'Cluster Persist.':<20} {'Regime Persist.'}")
        for k in sorted(results_all_k.keys()):
            r = results_all_k[k]
            logger.info(
                f"{k:<5} {r['nmi']:<10.4f} {r['ari']:<10.4f} {r['avg_purity']:<10.4f} "
                f"{r['avg_cluster_persistence']:<20.4f} {r['avg_regime_persistence']:.4f}"
            )

        best_k = max(results_all_k, key=lambda k: results_all_k[k]["nmi"])
        logger.info(f"\nBest K by NMI: {best_k}")
        logger.info(f"  NMI:    {results_all_k[best_k]['nmi']:.4f}")
        logger.info(f"  Purity: {results_all_k[best_k]['avg_purity']:.4f}")


if __name__ == "__main__":
    main()
