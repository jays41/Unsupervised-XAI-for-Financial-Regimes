"""Segment reconstruction errors into normal and unusual windows."""

import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from config import (
    ANOMALY_PERCENTILE,
    CLUSTER_WINDOW_END_DATES_PATH,
    ERROR_ANALYSIS_OUT_DIR as OUT_DIR,
    TRAIN_ERR_PATH,
    VAL_ERR_PATH,
    TEST_ERR_PATH,
)

logger = logging.getLogger(__name__)


def load_split_errors() -> dict[str, np.ndarray]:
    splits = {
        "train": np.load(TRAIN_ERR_PATH, allow_pickle=False).astype(np.float64),
        "val": np.load(VAL_ERR_PATH, allow_pickle=False).astype(np.float64),
        "test": np.load(TEST_ERR_PATH, allow_pickle=False).astype(np.float64),
    }
    logger.info(
        "Loaded reconstruction errors:  train=%d  val=%d  test=%d  total=%d",
        len(splits["train"]),
        len(splits["val"]),
        len(splits["test"]),
        sum(len(v) for v in splits.values()),
    )
    return splits


def build_concatenated(split_errors: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, tuple[int, int]]]:
    """Concatenate split errors into one array and return the slice boundaries per split."""
    all_errors = np.concatenate(
        [split_errors["train"], split_errors["val"], split_errors["test"]]
    )
    n_train = len(split_errors["train"])
    n_val = len(split_errors["val"])
    boundaries = {
        "train": (0, n_train),
        "val": (n_train, n_train + n_val),
        "test": (n_train + n_val, len(all_errors)),
    }
    return all_errors, boundaries


def descriptive_stats(errors: np.ndarray) -> dict:
    q25, q75, q90, q95, q99 = np.percentile(errors, [25, 75, 90, 95, 99])
    return {
        "n": int(len(errors)),
        "mean": float(np.mean(errors)),
        "std": float(np.std(errors)),
        "min": float(np.min(errors)),
        "max": float(np.max(errors)),
        "median": float(np.median(errors)),
        "q25": float(q25),
        "q75": float(q75),
        "q90": float(q90),
        "q95": float(q95),
        "q99": float(q99),
    }


def compute_anomaly_labels(train_errors: np.ndarray, all_errors: np.ndarray, percentile: int) -> tuple[float, np.ndarray]:
    """Fit threshold on train split only (no data leakage), then label all windows"""
    threshold = float(np.percentile(train_errors, percentile))
    labels = (all_errors >= threshold).astype(np.int32)  # binary anomaly label (Malhotra et al., 2016)
    return threshold, labels


def compute_split_rates(labels: np.ndarray, boundaries: dict[str, tuple[int, int]]) -> dict[str, dict]:
    rates: dict[str, dict] = {}
    for name, (start, end) in boundaries.items():
        n = end - start
        unusual_count = int(labels[start:end].sum())
        rates[name] = {
            "n": n,
            "unusual_count": unusual_count,
            "unusual_rate_pct": float(100.0 * unusual_count / n),
        }
    return rates


def save_explainability_targets(all_errors: np.ndarray, labels: np.ndarray, out_dir: Path) -> None:
    """Save continuous (MSE) and binary (above-threshold) targets for explainability"""
    np.save(out_dir / "track2_target_continuous.npy", all_errors.astype(np.float32))
    np.save(out_dir / "track2_target_binary.npy", labels.astype(np.int32))


def plot_error_histogram(all_errors: np.ndarray, threshold: float, percentile: int, save_path: Path) -> None:
    plt.figure(figsize=(9, 5))
    plt.hist(all_errors, bins=60, density=True, alpha=0.8, edgecolor="black")
    plt.axvline(
        threshold,
        linestyle="--",
        linewidth=2,
        label=f"Train {percentile}th pct = {threshold:.6f}",
    )
    plt.title("Reconstruction Error Distribution")
    plt.xlabel("Window reconstruction error (MSE)")
    plt.ylabel("Density")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_error_timeseries(all_errors: np.ndarray, labels: np.ndarray, boundaries: dict[str, tuple[int, int]], threshold: float, percentile: int, save_path: Path, dates: np.ndarray | None = None) -> None:
    unusual = labels == 1
    split_colours = {"train": "#4C9BE8", "val": "#E87C4C", "test": "#4CAF7D"}

    if dates is not None:
        x = pd.to_datetime(dates)
        x_unusual = x[unusual]
        x_splits = {name: x[start] for name, (start, end) in boundaries.items()}
        x_label_centres = {name: x[start + (end - start) // 2] for name, (start, end) in boundaries.items()}
        use_dates = True
    else:
        x = np.arange(len(all_errors))
        x_unusual = x[unusual]
        x_splits = {name: start for name, (start, end) in boundaries.items()}
        x_label_centres = {name: start + (end - start) // 2 for name, (start, end) in boundaries.items()}
        use_dates = False

    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.plot(x, all_errors, linewidth=0.9, alpha=0.7, color="#555555", zorder=2)
    ax.scatter(
        x_unusual, all_errors[unusual],
        s=10, alpha=0.75, color="#E84C4C", label="Unusual (above threshold)", zorder=3,
    )
    ax.axhline(
        threshold,
        linestyle="--",
        linewidth=1.8,
        color="#9B59B6",
        label=f"Anomaly threshold (train {percentile}th pct = {threshold:.6f})",
        zorder=4,
    )

    y_max = float(np.max(all_errors))
    for name, (start, end) in boundaries.items():
        colour = split_colours.get(name, "#888888")
        ax.axvline(x_splits[name], linestyle="--", linewidth=1.2, color=colour, alpha=0.35, zorder=1)
        ax.text(
            x_label_centres[name],
            y_max * 0.95,
            name.upper(),
            ha="center",
            va="top",
            fontsize=9,
            color="#444444",
            fontweight="bold",
        )

    if use_dates:
        ax.set_xlim(x[0], x[-1])
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        fig.autofmt_xdate(rotation=0, ha="center")
        ax.set_xlabel("Date")
    else:
        ax.set_xlim(x[0], x[-1])
        ax.set_xlabel("Window index")

    ax.set_title("Reconstruction Error Over Windows")
    ax.set_ylabel("Reconstruction error (MSE)")
    ax.grid(alpha=0.25)
    ax.legend(framealpha=0.85)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_split_summary(all_errors: np.ndarray, rates: dict[str, dict], boundaries: dict[str, tuple[int, int]], save_path: Path) -> None:
    names = list(boundaries.keys())
    means = [float(all_errors[s:e].mean()) for s, e in boundaries.values()]
    unusual_pcts = [rates[n]["unusual_rate_pct"] for n in names]

    _, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.bar(names, means, alpha=0.8)
    ax1.set_ylabel("Mean reconstruction error (MSE)")
    ax1.set_title("Mean error and unusual-rate by split")
    ax1.grid(alpha=0.3, axis="y")

    ax2 = ax1.twinx()
    ax2.plot(names, unusual_pcts, marker="o")
    ax2.set_ylabel("Unusual rate (%)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def log_results(threshold: float, percentile: int, rates: dict[str, dict]) -> None:
    logger.info("Threshold (train %dth pct): %.6f", percentile, threshold)
    logger.info("Unusual rate by split:")
    for name, info in rates.items():
        logger.info(
            "  %s: %.2f%% (%d/%d)",
            name, info["unusual_rate_pct"], info["unusual_count"], info["n"],
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    split_errors = load_split_errors()
    all_errors, boundaries = build_concatenated(split_errors)

    threshold, labels = compute_anomaly_labels(split_errors["train"], all_errors, ANOMALY_PERCENTILE)
    rates = compute_split_rates(labels, boundaries)

    np.save(OUT_DIR / "unusual_labels.npy", labels)
    save_explainability_targets(all_errors, labels, OUT_DIR)

    threshold_meta = {"percentile": ANOMALY_PERCENTILE, "value": threshold, "fit_split": "train"}
    summary = {
        "threshold": threshold_meta,
        "stats": {
            name: descriptive_stats(split_errors[name]) for name in split_errors
        } | {"all": descriptive_stats(all_errors)},
        "unusual_rates": rates,
    }

    with open(OUT_DIR / "threshold.json", "w") as f:
        json.dump(threshold_meta, f, indent=2)
    with open(OUT_DIR / "track2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    dates = None
    if CLUSTER_WINDOW_END_DATES_PATH.exists():
        dates = np.load(CLUSTER_WINDOW_END_DATES_PATH, allow_pickle=True)
        if len(dates) != len(all_errors):
            logger.warning("Date array length (%d) does not match error array (%d); falling back to window index.", len(dates), len(all_errors))
            dates = None

    plot_error_histogram(all_errors, threshold, ANOMALY_PERCENTILE, OUT_DIR / "error_hist.png")
    plot_error_timeseries(all_errors, labels, boundaries, threshold, ANOMALY_PERCENTILE, OUT_DIR / "error_over_time.png", dates=dates)
    plot_split_summary(all_errors, rates, boundaries, OUT_DIR / "split_summary.png")

    logger.info("\nTrack 2 complete.")
    log_results(threshold, ANOMALY_PERCENTILE, rates)


if __name__ == "__main__":
    main()