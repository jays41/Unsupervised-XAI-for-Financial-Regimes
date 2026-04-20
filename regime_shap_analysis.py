"""Regime-specific analysis of Vector SHAP values.

Track 1: f(x) = P(cluster=k | encoder(x)) - grouped by KMeans hard label.
Track 2: f(x) = mean MSE - grouped by cluster membership and binary anomaly label.
"""

from __future__ import annotations
import argparse
import csv
import json
import logging
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from scipy.stats import kruskal, mannwhitneyu
import config
from config import N_CLUSTERS, INDICATOR_NAMES, FAMILY_KEYS

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

SIG_THRESHOLDS = [(0.001, "***"), (0.01, "**"), (0.05, "*")]

MEMBERSHIP_LABELS = [f"Cluster {k}" for k in range(N_CLUSTERS)]
BINARY_LABELS = ["Normal (0)", "Unusual (1)"]

CLUSTER_DIR = config.CLUSTERING_OUT_DIR

def sig_stars(p: float) -> str:
    for thresh, star in SIG_THRESHOLDS:
        if p < thresh:
            return star
    return "ns"


def rank_biserial_r(u: float, n1: int, n2: int) -> float:
    return float(1.0 - 2.0 * u / (n1 * n2))


def run_kruskal_wallis(groups: List[np.ndarray]) -> Tuple[float, float]:
    h, p = kruskal(*groups)
    return float(h), float(p)


def pairwise_mwu(groups: List[np.ndarray], n_comparisons: int = 3) -> List[Dict]:
    results = []
    for (i, j) in combinations(range(len(groups)), 2):
        g1, g2 = groups[i], groups[j]
        u, p_raw = mannwhitneyu(g1, g2, alternative="two-sided")
        p_bonf = min(float(p_raw) * n_comparisons, 1.0)
        results.append({
            "pair": f"{i}v{j}", "U": float(u), "p_raw": float(p_raw),
            "p_bonf": float(p_bonf), "stars": sig_stars(p_bonf),
            "effect_r": rank_biserial_r(float(u), len(g1), len(g2)),
        })
    return results


def analyse_by_cluster(shap_col: np.ndarray, membership: np.ndarray) -> Dict:
    groups = [shap_col[membership == k] for k in range(N_CLUSTERS)]
    means = [float(g.mean()) for g in groups]
    ses = [float(g.std(ddof=1) / np.sqrt(len(g))) for g in groups]
    ns = [int(len(g)) for g in groups]
    h, p_kw = run_kruskal_wallis(groups)
    mwu = pairwise_mwu(groups, n_comparisons=3)
    return {
        "group_means": means, "group_se": ses, "group_n": ns,
        "kruskal_H": h, "kruskal_p": p_kw, "kruskal_stars": sig_stars(p_kw),
        "pairwise_mwu": mwu,
    }


def analyse_by_binary(shap_col: np.ndarray, binary: np.ndarray) -> Dict:
    g0 = shap_col[binary == 0]
    g1 = shap_col[binary == 1]

    def _stats(g):
        return float(g.mean()), float(g.std(ddof=1) / np.sqrt(len(g))), int(len(g))

    mean0, se0, n0 = _stats(g0)
    mean1, se1, n1 = _stats(g1)

    u, p = mannwhitneyu(g0, g1, alternative="two-sided")
    u, p = float(u), float(p)
    r = rank_biserial_r(u, n0, n1)

    return {
        "mean_normal": mean0, "se_normal": se0, "n_normal": n0,
        "mean_unusual": mean1, "se_unusual": se1, "n_unusual": n1,
        "mwu_U": u, "mwu_p": p, "mwu_stars": sig_stars(p), "effect_r": r,
    }

def save_json(data: dict, path: Path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def save_cluster_csv(feature_names: List[str], summary: Dict[str, Dict], path: Path):
    fieldnames = ["feature"]
    for g in range(N_CLUSTERS):
        fieldnames += [f"cluster{g}_mean", f"cluster{g}_se", f"cluster{g}_n"]
    fieldnames += ["kruskal_H", "kruskal_p", "kruskal_stars"]
    for pair in ["0v1", "0v2", "1v2"]:
        fieldnames += [f"mwu_{pair}_p_bonf", f"mwu_{pair}_stars", f"mwu_{pair}_effect_r"]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for feat in feature_names:
            s = summary[feat]
            row = {"feature": feat}
            for g in range(N_CLUSTERS):
                row[f"cluster{g}_mean"] = f"{s['group_means'][g]:.6f}"
                row[f"cluster{g}_se"]   = f"{s['group_se'][g]:.6f}"
                row[f"cluster{g}_n"]    = s["group_n"][g]
            row["kruskal_H"]     = f"{s['kruskal_H']:.4f}"
            row["kruskal_p"]     = f"{s['kruskal_p']:.6f}"
            row["kruskal_stars"] = s["kruskal_stars"]
            for entry in s["pairwise_mwu"]:
                p = entry["pair"]
                row[f"mwu_{p}_p_bonf"]   = f"{entry['p_bonf']:.6f}"
                row[f"mwu_{p}_stars"]    = entry["stars"]
                row[f"mwu_{p}_effect_r"] = f"{entry['effect_r']:.4f}"
            writer.writerow(row)


def save_binary_csv(feature_names: List[str], summary: Dict[str, Dict], path: Path):
    fieldnames = [
        "feature",
        "mean_normal", "se_normal", "n_normal",
        "mean_unusual", "se_unusual", "n_unusual",
        "mwu_U", "mwu_p", "mwu_stars", "effect_r",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for feat in feature_names:
            s = summary[feat]
            writer.writerow({
                "feature":      feat,
                "mean_normal":  f"{s['mean_normal']:.6f}",
                "se_normal":    f"{s['se_normal']:.6f}",
                "n_normal":     s["n_normal"],
                "mean_unusual": f"{s['mean_unusual']:.6f}",
                "se_unusual":   f"{s['se_unusual']:.6f}",
                "n_unusual":    s["n_unusual"],
                "mwu_U":        f"{s['mwu_U']:.1f}",
                "mwu_p":        f"{s['mwu_p']:.6f}",
                "mwu_stars":    s["mwu_stars"],
                "effect_r":     f"{s['effect_r']:.4f}",
            })


def _print_cluster_row(feat: str, result: Dict):
    logger.info(
        "  %s | %s  | KW H=%.2f p=%.4f %s",
        f"{feat:25s}",
        "  ".join(f"cl{g}={result['group_means'][g]:+.4f}(n={result['group_n'][g]})" for g in range(N_CLUSTERS)),
        result['kruskal_H'], result['kruskal_p'], result['kruskal_stars'],
    )


def _sanity_check_softmax_shap(shap_l1: np.ndarray):
    row_sums = shap_l1.sum(axis=2)  # (N, D); should be around 0 (softmax constraint)
    max_dev = float(np.max(np.abs(row_sums)))
    mean_dev = float(np.mean(np.abs(row_sums)))
    logger.info("[Sanity] Softmax SHAP column-sum | max|dev|=%.2e | mean|dev|=%.2e", max_dev, mean_dev)
    if max_dev > 1e-2:
        logger.warning("[Sanity] Column sums deviate >1e-2 - check SHAP computation.")


def run_track1():
    in_dir = config.VECTOR_SHAP_TRACK1_OUT_DIR
    out_dir = config.REGIME_TRACK1_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    shap_path = in_dir / "vectorshap_level1_test.npy"
    shap_l1 = np.load(shap_path).astype(np.float64)
    N, D, K = shap_l1.shape

    assert D == len(INDICATOR_NAMES), f"D={D} does not match INDICATOR_NAMES length={len(INDICATOR_NAMES)}"
    assert K == N_CLUSTERS, f"K={K} does not match N_CLUSTERS={N_CLUSTERS}"

    labels_path = CLUSTER_DIR / "labels_test_k3.npy"
    membership = np.load(labels_path).astype(np.int32)[:N]
    logger.info("Loaded cluster membership (first %d of test split):", N)
    for u, c in zip(*np.unique(membership, return_counts=True)):
        logger.info("  Cluster %d: %d samples (%.1f%%)", u, c, 100*c/N)

    _sanity_check_softmax_shap(shap_l1)
    shap_l2: Dict[str, np.ndarray] = {}
    for fam in FAMILY_KEYS:
        p = in_dir / f"vectorshap_level2_{fam}_test.npy"
        if p.exists():
            shap_l2[fam] = np.load(p).astype(np.float64)
        else:
            logger.warning("  Missing %s, skipping '%s'", p.name, fam)

    all_summary_l2: Dict[int, Dict[str, Dict]] = {}

    for k in range(K):
        logger.info("Cluster output k=%d  (SHAP toward P(cluster=%d))", k, k)

        summary_l1: Dict[str, Dict] = {}
        for di, feat in enumerate(INDICATOR_NAMES):
            result = analyse_by_cluster(shap_l1[:, di, k], membership)
            summary_l1[feat] = result
            _print_cluster_row(feat, result)

        save_json(summary_l1, out_dir / f"regime_summary_k{k}.json")
        save_cluster_csv(INDICATOR_NAMES, summary_l1, out_dir / f"regime_table_k{k}.csv")

        if shap_l2:
            summary_l2: Dict[str, Dict] = {}
            for fam in FAMILY_KEYS:
                if fam not in shap_l2:
                    continue
                result_l2 = analyse_by_cluster(shap_l2[fam][:, k], membership)
                summary_l2[fam] = result_l2
                logger.info(
                    "  [L2] %s | %s  | KW p=%.4f %s",
                    f"{fam:30s}",
                    "  ".join(f"cl{g}={result_l2['group_means'][g]:+.4f}" for g in range(N_CLUSTERS)),
                    result_l2['kruskal_p'], result_l2['kruskal_stars'],
                )

            all_summary_l2[k] = summary_l2

            save_json(summary_l2, out_dir / f"regime_summary_level2_k{k}.json")
            save_cluster_csv(list(summary_l2.keys()), summary_l2, out_dir / f"regime_table_level2_k{k}.csv")

    logger.info("\nExamples\n")
    for k in range(K):
        for fam, s in all_summary_l2.get(k, {}).items():
            logger.info(
                "  P(cl=%d) | %s: %s  [KW H=%.2f, p=%.4f %s]",
                k, fam,
                ", ".join(f"cl{g}={s['group_means'][g]:+.3f}\u00b1{s['group_se'][g]:.3f}" for g in range(N_CLUSTERS)),
                s['kruskal_H'], s['kruskal_p'], s['kruskal_stars'],
            )


def run_track2():
    in_dir = config.VECTOR_SHAP_TRACK2_OUT_DIR
    out_dir = config.REGIME_TRACK2_OUT_DIR
    error_dir = config.ERROR_ANALYSIS_OUT_DIR
    prep_dir = config.PREP_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    shap_path = in_dir / "vectorshap_level1_test.npy"
    raw = np.load(shap_path).astype(np.float64)
    shap_l1 = raw[:, :, 0] if (raw.ndim == 3 and raw.shape[2] == 1) else raw
    N, D = shap_l1.shape
    logger.info("Loaded Level 1 SHAP: shape=%s  (N=%d, D=%d)", shap_l1.shape, N, D)

    assert D == len(INDICATOR_NAMES), f"D={D} does not match INDICATOR_NAMES length={len(INDICATOR_NAMES)}"

    idx_path = in_dir / "explain_indices_test.npy"
    explain_idx = np.load(idx_path).astype(np.int64)
    assert len(explain_idx) == N, f"explain_indices length {len(explain_idx)} != SHAP N={N}"

    labels_path = CLUSTER_DIR / "labels_test_k3.npy"
    membership = np.load(labels_path).astype(np.int32)[explain_idx]
    logger.info("Loaded cluster membership (%d explained samples from test split):", N)
    for u, c in zip(*np.unique(membership, return_counts=True)):
        logger.info("  Cluster %d: %d samples (%.1f%%)", u, c, 100*c/N)

    bin_path = error_dir / "track2_target_binary.npy"
    y_bin_all = np.load(bin_path).astype(np.int32).reshape(-1)
    n_train = len(np.load(prep_dir / "train_windows.npy", mmap_mode="r"))
    n_val = len(np.load(prep_dir / "val_windows.npy", mmap_mode="r"))
    binary = y_bin_all[n_train + n_val:][explain_idx]
    logger.info("Binary labels: normal=%d, unusual=%d (N=%d)",
                int((binary==0).sum()), int((binary==1).sum()), N)

    shap_l2: Dict[str, np.ndarray] = {}
    for fam in FAMILY_KEYS:
        p = in_dir / f"vectorshap_level2_{fam}_test.npy"
        if p.exists():
            shap_l2[fam] = np.load(p).astype(np.float64).squeeze()
        else:
            logger.warning("  Missing %s, skipping '%s'", p.name, fam)

    logger.info("Grouping A: SHAP by cluster membership (Kruskal-Wallis + MWU)")

    summary_cluster_l1: Dict[str, Dict] = {}
    for di, feat in enumerate(INDICATOR_NAMES):
        result = analyse_by_cluster(shap_l1[:, di], membership)
        summary_cluster_l1[feat] = result
        _print_cluster_row(feat, result)

    save_json(summary_cluster_l1, out_dir / "regime_summary_by_cluster.json")
    save_cluster_csv(INDICATOR_NAMES, summary_cluster_l1, out_dir / "regime_table_by_cluster.csv")

    if shap_l2:
        summary_cluster_l2: Dict[str, Dict] = {}
        for fam in FAMILY_KEYS:
            if fam not in shap_l2:
                continue
            result_l2 = analyse_by_cluster(shap_l2[fam], membership)
            summary_cluster_l2[fam] = result_l2
            logger.info(
                "  [L2] %s | %s  | KW p=%.4f %s",
                f"{fam:30s}",
                "  ".join(f"cl{g}={result_l2['group_means'][g]:+.4f}" for g in range(N_CLUSTERS)),
                result_l2['kruskal_p'], result_l2['kruskal_stars'],
            )
        save_json(summary_cluster_l2, out_dir / "regime_summary_level2_by_cluster.json")
        save_cluster_csv(list(summary_cluster_l2.keys()), summary_cluster_l2, out_dir / "regime_table_level2_by_cluster.csv")

    logger.info("Grouping B: SHAP by anomaly label \u2014 normal(0) vs unusual(1) (MWU)")

    summary_binary_l1: Dict[str, Dict] = {}
    for di, feat in enumerate(INDICATOR_NAMES):
        result = analyse_by_binary(shap_l1[:, di], binary)
        summary_binary_l1[feat] = result
        logger.info(
            "  %s | normal=%+.4f\u00b1%.4f(n=%d)  unusual=%+.4f\u00b1%.4f(n=%d)"
            "  | MWU p=%.4f %s  r=%+.3f",
            f"{feat:25s}",
            result['mean_normal'], result['se_normal'], result['n_normal'],
            result['mean_unusual'], result['se_unusual'], result['n_unusual'],
            result['mwu_p'], result['mwu_stars'], result['effect_r'],
        )

    save_json(summary_binary_l1, out_dir / "regime_summary_by_binary.json")
    save_binary_csv(INDICATOR_NAMES, summary_binary_l1, out_dir / "regime_table_by_binary.csv")

    if shap_l2:
        summary_binary_l2: Dict[str, Dict] = {}
        for fam in FAMILY_KEYS:
            if fam not in shap_l2:
                continue
            result_l2 = analyse_by_binary(shap_l2[fam], binary)
            summary_binary_l2[fam] = result_l2
            logger.info(
                "  [L2] %s | normal=%+.4f  unusual=%+.4f"
                "  | MWU p=%.4f %s  r=%+.3f",
                f"{fam:30s}",
                result_l2['mean_normal'], result_l2['mean_unusual'],
                result_l2['mwu_p'], result_l2['mwu_stars'], result_l2['effect_r'],
            )
        save_json(summary_binary_l2, out_dir / "regime_summary_level2_by_binary.json")
        save_binary_csv(list(summary_binary_l2.keys()), summary_binary_l2, out_dir / "regime_table_level2_by_binary.csv")

    logger.info("\nExamples\n")
    logger.info("  [Binary grouping]")
    for feat in INDICATOR_NAMES:
        s = summary_binary_l1[feat]
        logger.info(
            "  %s: normal=%+.3f\u00b1%.3f  unusual=%+.3f\u00b1%.3f  "
            "[MWU U=%.0f, p=%.4f %s, r=%+.3f]",
            feat,
            s['mean_normal'], s['se_normal'],
            s['mean_unusual'], s['se_unusual'],
            s['mwu_U'], s['mwu_p'], s['mwu_stars'], s['effect_r'],
        )
    logger.info("  [Cluster grouping]")
    for feat in INDICATOR_NAMES:
        s = summary_cluster_l1[feat]
        logger.info(
            "  %s: %s  [KW H=%.2f, p=%.4f %s]",
            feat,
            ", ".join(f"cl{g}={s['group_means'][g]:+.3f}\u00b1{s['group_se'][g]:.3f}" for g in range(N_CLUSTERS)),
            s['kruskal_H'], s['kruskal_p'], s['kruskal_stars'],
        )


def main():
    parser = argparse.ArgumentParser(
        description="Regime-specific SHAP analysis. Choose which track to analyse."
    )
    parser.add_argument(
        "--track", type=int, choices=[1, 2], required=True,
        help="Which track to run: 1 = cluster softmax probabilities, 2 = reconstruction error",
    )
    args = parser.parse_args()

    logger.info("  Regime SHAP Analysis - Track %d", args.track)

    if args.track == 1:
        run_track1()
    else:
        run_track2()


if __name__ == "__main__":
    main()
