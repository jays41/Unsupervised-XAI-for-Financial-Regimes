import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import plot_config

logger = logging.getLogger(__name__)

plot_config.setup_style()

OUT_DIR = Path("outputs/vector_shap_track2")
SPLIT = "test"
TOPN = 10
MAX_HEATMAP_SAMPLES = 40
SAMPLE_IDX = None


def load_outputs(out_dir: Path, split: str):
    meta_path = out_dir / f"vectorshap_meta_{split}.json"
    lvl1_path = out_dir / f"vectorshap_level1_{split}.npy"
    base_path = out_dir / f"base_value_{split}.npy"
    full_path = out_dir / f"full_value_{split}.npy"

    for p in (meta_path, lvl1_path, base_path, full_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    shap1 = np.load(lvl1_path)
    base = np.load(base_path)
    full = np.load(full_path)

    # squeeze trailing output dimension (N, D, 1) -> (N, D) and (N, 1) -> (N,)
    if shap1.ndim == 3 and shap1.shape[-1] == 1:
        shap1 = shap1[:, :, 0]
    if base.ndim == 2 and base.shape[-1] == 1:
        base = base[:, 0]
    if full.ndim == 2 and full.shape[-1] == 1:
        full = full[:, 0]

    families = meta.get("level2_families", {})
    shap2: Dict[str, np.ndarray] = {}
    for fam_name in families.keys():
        p = out_dir / f"vectorshap_level2_{fam_name}_{split}.npy"
        if p.exists():
            arr = np.load(p)
            if arr.ndim == 2 and arr.shape[-1] == 1:
                arr = arr[:, 0]
            shap2[fam_name] = arr

    indicator_names = meta.get("indicator_names", [f"x{i}" for i in range(shap1.shape[1])])

    return meta, shap1, base, full, shap2, indicator_names


def mean_abs_importance(shap: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(shap), axis=0)


def mean_signed_importance(shap: np.ndarray) -> np.ndarray:
    return np.mean(shap, axis=0)


def barh_plot(
    values: np.ndarray,
    names: List[str],
    title: str,
    xlabel: str,
    save_path: Path,
    topn: Optional[int] = None,
):
    values = values.copy()
    names = list(names)

    order = np.argsort(values)[::-1]
    if topn is not None:
        order = order[:topn]

    vals = values[order][::-1]
    nms = [names[i] for i in order][::-1]

    plt.figure(figsize=(9, 5.2))
    plt.barh(range(len(vals)), vals)
    plt.yticks(range(len(vals)), nms)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def heatmap(mat: np.ndarray, xlabels: List[str], title: str, save_path: Path, ylabel: str = "Sample index"):
    plt.figure(figsize=(10.5, 5.5))
    plt.imshow(mat, aspect="auto")
    plt.colorbar(label="SHAP contribution")
    plt.xticks(range(len(xlabels)), xlabels, rotation=25, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def scatter_xy(x: np.ndarray, y: np.ndarray, title: str, xlabel: str, ylabel: str, save_path: Path):
    plt.figure(figsize=(6.5, 5.5))
    plt.scatter(x, y, s=10)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def additivity_errors(shap1: np.ndarray, base: np.ndarray, full: np.ndarray) -> np.ndarray:
    return (base + shap1.sum(axis=1)) - full


def waterfall_contrib(
    shap_vec: np.ndarray,
    names: List[str],
    base_val: float,
    full_val: float,
    title: str,
    save_path: Path,
    topn: int = 10,
):
    D = shap_vec.shape[0]
    idx = np.argsort(np.abs(shap_vec))[::-1][: min(topn, D)]
    contribs = shap_vec[idx]
    labels = [names[i] for i in idx]

    order = np.argsort(contribs)
    contribs = contribs[order]
    labels = [labels[i] for i in order]

    running = base_val
    starts = []
    ends = []
    for c in contribs:
        starts.append(running)
        running = running + float(c)
        ends.append(running)

    plt.figure(figsize=(10, 5.2))
    for i, (s, e, c) in enumerate(zip(starts, ends, contribs)):
        plt.plot([s, e], [i, i], linewidth=8)
        plt.text(e, i, f"  {float(c):+.3f}", va="center", fontsize=9)

    plt.axvline(base_val, linestyle="--", linewidth=1)
    plt.axvline(full_val, linestyle="--", linewidth=1)

    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Reconstruction error")
    plt.title(title + f"\n(base={base_val:.4f}, approx(top{len(contribs)})={running:.4f}, full={full_val:.4f})")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def make_all_figures(out_dir: Path, split: str, sample_idx: Optional[int], topn: int, max_heatmap_samples: int):
    meta, shap1, base, full, shap2, indicator_names = load_outputs(out_dir, split)

    figs_dir = out_dir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    N, D = shap1.shape

    err = additivity_errors(shap1, base, full)
    abs_err = np.abs(err)

    plt.figure(figsize=(7.5, 5.2))
    plt.hist(abs_err, bins=40)
    plt.title("Additivity check: |error| per sample\n(base + ΣSHAP − full)")
    plt.xlabel("|error|")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(figs_dir / "additivity_abs_err_hist.png", dpi=200)
    plt.close()

    scatter_xy(
        x=base,
        y=full,
        title="Base vs Full reconstruction error",
        xlabel="Base value f(baseline-masked x)",
        ylabel="Full value f(x)",
        save_path=figs_dir / "base_vs_full.png",
    )

    imp_abs = mean_abs_importance(shap1)
    imp_signed = mean_signed_importance(shap1)

    barh_plot(
        imp_abs,
        indicator_names,
        title="Global Vector SHAP importance (Level 1 indicators)\nmean(|SHAP|) over samples",
        xlabel="mean |SHAP|",
        save_path=figs_dir / "level1_global_mean_abs.png",
        topn=topn,
    )
    barh_plot(
        np.abs(imp_signed),
        indicator_names,
        title="Global signed effect magnitude (Level 1 indicators)\n|mean(SHAP)| over samples",
        xlabel="|mean SHAP|",
        save_path=figs_dir / "level1_global_abs_mean_signed.png",
        topn=topn,
    )

    if shap2:
        fam_names = list(shap2.keys())
        fam_global = np.array([float(np.mean(np.abs(shap2[fn]))) for fn in fam_names])
        barh_plot(
            fam_global,
            fam_names,
            title="Global Vector SHAP importance (Level 2 families)\nmean(|SHAP|) over samples",
            xlabel="mean |SHAP|",
            save_path=figs_dir / "level2_global_mean_abs.png",
            topn=None,
        )

    order = np.argsort(full)[::-1]
    take = order[: min(max_heatmap_samples, len(order))]

    heatmap(
        shap1[take, :],
        indicator_names,
        title="Level 1 SHAP heatmap (Track 2)\nTop samples by full reconstruction error",
        save_path=figs_dir / "heatmap_level1_top_full_error.png",
        ylabel="Top samples (ranked by full error)",
    )

    if sample_idx is None:
        sample_idx = int(order[0]) if len(order) else 0

    if not (0 <= sample_idx < N):
        raise ValueError(f"SAMPLE_IDX {sample_idx} out of range [0, {N-1}]")

    waterfall_contrib(
        shap_vec=shap1[sample_idx, :],
        names=indicator_names,
        base_val=float(base[sample_idx]),
        full_val=float(full[sample_idx]),
        title=f"Waterfall contributions (sample {sample_idx})",
        save_path=figs_dir / f"waterfall_sample_{sample_idx}.png",
        topn=min(topn, D),
    )

    logger.info("Saved figures to: %s", figs_dir.resolve())


if __name__ == "__main__":
    make_all_figures(
        out_dir=OUT_DIR,
        split=SPLIT,
        sample_idx=SAMPLE_IDX,
        topn=TOPN,
        max_heatmap_samples=MAX_HEATMAP_SAMPLES,
    )
