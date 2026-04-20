import json
import logging
import pickle
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import config

warnings.filterwarnings("ignore", category=ConvergenceWarning)
log = logging.getLogger(__name__)

OUT_DIR = config.CLUSTERING_OUT_DIR


def load_split_features() -> tuple:
    train_x = np.load(config.TRAIN_CLUSTER_Z_PATH)
    val_x = np.load(config.VAL_CLUSTER_Z_PATH)
    test_x = np.load(config.TEST_CLUSTER_Z_PATH)

    all_x = np.vstack([train_x, val_x, test_x])
    splits = {
        "train": (0, len(train_x)),
        "val": (len(train_x), len(train_x) + len(val_x)),
        "test": (len(train_x) + len(val_x), len(all_x)),
    }
    return train_x, val_x, test_x, all_x, splits


def save_window_end_dates(splits: dict) -> np.ndarray:
    dates = np.load(config.WINDOW_END_DATES_PATH, allow_pickle=True)
    n_total = splits["test"][1]
    if len(dates) != n_total:
        raise ValueError(
            f"window_end_dates length {len(dates)} does not match total windows {n_total}."
        )
    np.save(OUT_DIR / "window_end_dates.npy", dates)
    return dates


def label_counts(labels: np.ndarray) -> dict:
    unique, counts = np.unique(labels, return_counts=True)
    return {int(k): int(v) for k, v in zip(unique, counts)}


def split_shares(labels: np.ndarray, k: int) -> dict:
    counts = np.bincount(labels, minlength=k).astype(float)
    shares = counts / counts.sum()
    return {i: float(shares[i]) for i in range(k)}


def normalised_entropy(proba: np.ndarray) -> float:
    """Mean normalised Shannon entropy: 0 = confident, 1 = uniform.
    Used as K-selection criterion: optimal K unknown a priori (Ang & Timmermann, 2011; Baldi, 2012).
    """
    k = proba.shape[1]
    p = np.clip(proba, 1e-12, None)
    entropy = -np.sum(p * np.log(p), axis=1)
    return float(entropy.mean() / np.log(k))


def fit_kmeans_and_softmax(
    train_x: np.ndarray, k: int
) -> tuple:
    # KMeans regime segmentation motivated by Hamilton (1989)
    km = KMeans(n_clusters=k, random_state=config.RANDOM_SEED, n_init=20, max_iter=500)
    km.fit(train_x)

    # Soft assignments via logistic regression softmax (Baldi, 2012)
    lr = LogisticRegression(
        solver="lbfgs", C=0.1, max_iter=1000, random_state=config.RANDOM_SEED
    )
    lr.fit(train_x, km.labels_)
    return km, lr


def kmeans_logreg_select_k(
    train_x: np.ndarray,
    val_x: np.ndarray,
    test_x: np.ndarray,
    k_range: list[int] = config.K_RANGE,
) -> tuple:
    """Fit KMeans+softmax for each K; select best K by lowest val entropy among stable candidates."""
    results = {}

    for k in k_range:
        km, lr = fit_kmeans_and_softmax(train_x, k)

        hard = {
            "train": km.predict(train_x),
            "val":   km.predict(val_x),
            "test":  km.predict(test_x),
        }
        soft = {
            "train": lr.predict_proba(train_x),
            "val":   lr.predict_proba(val_x),
            "test":  lr.predict_proba(test_x),
        }

        shares = {split: split_shares(hard[split], k) for split in hard}
        stable = all(min(shares[s].values()) >= config.MIN_CLUSTER_SHARE for s in shares)
        min_share_train = min(shares["train"].values())

        silhouette = {
            s: float(silhouette_score(x, hard[s], random_state=config.RANDOM_SEED))
            for s, x in (("train", train_x), ("val", val_x), ("test", test_x))
        }

        results[k] = {
            "km_model": km,
            "lr_model": lr,
            "inertia_train": float(km.inertia_),
            "min_share_train": float(min_share_train),
            "stable": stable,
            "shares": shares,
            "sizes": {s: label_counts(hard[s]) for s in hard},
            "entropy": {s: normalised_entropy(soft[s]) for s in soft},
            "silhouette": silhouette,
            "labels": hard,
            "proba": soft,
        }

        log.info(
            "K=%d: inertia=%.2f  min_share_train=%.4f  stable=%s  sizes_train=%s",
            k, km.inertia_, min_share_train, stable, results[k]["sizes"]["train"],
        )

    stable_ks = [k for k in k_range if results[k]["stable"]]
    candidate_ks = stable_ks or k_range

    if not stable_ks:
        log.warning("No K passed stability constraints: selecting by min VAL entropy.")

    best_k = min(candidate_ks, key=lambda k: results[k]["entropy"]["val"])
    log.info("Selected K=%d (min VAL entropy%s).", best_k, " among stable K" if stable_ks else "")

    return results, best_k


def save_outputs(results: dict, best_k: int) -> np.ndarray:
    best = results[best_k]

    with open(OUT_DIR / "best_km_model.pkl", "wb") as fh:
        pickle.dump(best["km_model"], fh)
    with open(OUT_DIR / "best_lr_model.pkl", "wb") as fh:
        pickle.dump(best["lr_model"], fh)

    for split in ("train", "val", "test"):
        np.save(OUT_DIR / f"labels_{split}_k{best_k}.npy", best["labels"][split])
        np.save(OUT_DIR / f"proba_{split}_k{best_k}.npy", best["proba"][split])

    labels_all = np.concatenate([best["labels"][s] for s in ("train", "val", "test")])
    np.save(OUT_DIR / f"labels_all_k{best_k}.npy", labels_all)

    summary = {
        k: {
            "inertia_train": results[k]["inertia_train"],
            "min_share_train": results[k]["min_share_train"],
            "stable": results[k]["stable"],
            "shares": results[k]["shares"],
            "sizes": results[k]["sizes"],
            "entropy": results[k]["entropy"],
            "silhouette": results[k]["silhouette"],
        }
        for k in results
    }
    with open(OUT_DIR / "selection_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    return labels_all


def plot_model_selection(results: dict, k_range: list[int] = config.K_RANGE) -> None:
    ks = [k for k in k_range if k in results]
    if not ks:
        return

    _, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    axes[0].plot(ks, [results[k]["inertia_train"] for k in ks], marker="o", color="tab:orange")
    axes[0].set(title="KMeans+Softmax: Train Inertia", xlabel="K", ylabel="Inertia")
    axes[0].grid(alpha=0.3)

    for split, label in (("train", "Train"), ("val", "Val"), ("test", "Test")):
        axes[1].plot(ks, [results[k]["entropy"][split] for k in ks], marker="o", label=label)
    axes[1].set(
        title="Soft Assignment Uncertainty",
        xlabel="K",
        ylabel="Normalised Entropy (0=confident, 1=uncertain)",
    )
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "model_selection_plots.png", dpi=300, bbox_inches="tight")
    plt.close()


def visualise_clusters_tsne(all_x: np.ndarray, labels_all: np.ndarray) -> None:
    # Dimensionality reduction for latent space visualisation (Cho et al., 2014; Fawaz et al., 2019)
    tsne = TSNE(n_components=2, random_state=config.RANDOM_SEED, perplexity=30, max_iter=1000)
    coords = tsne.fit_transform(all_x)
    np.save(OUT_DIR / "tsne_coords.npy", coords)

    plt.figure(figsize=(7, 6))
    unique_labels = np.unique(labels_all)
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / (len(unique_labels) - 1) if len(unique_labels) > 1 else 0.5) for i in unique_labels]

    for i, label in enumerate(unique_labels):
        idx = labels_all == label
        plt.scatter(coords[idx, 0], coords[idx, 1],
                    color=colors[i], s=12, alpha=0.6,
                    label=f"Cluster {label}")

    plt.title("t-SNE Projection of Latent Cluster Features")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(title="Cluster", loc="best")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "tsne_clusters.png", dpi=300, bbox_inches="tight")
    plt.close()


def log_temporal_distribution(labels_all: np.ndarray, splits: dict) -> None:
    log.info("Temporal distribution (counts by split):")
    for split, (start, end) in splits.items():
        log.info("  %s  total=%d  %s", split, end - start, label_counts(labels_all[start:end]))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_x, val_x, test_x, all_x, splits = load_split_features()
    save_window_end_dates(splits)

    results, best_k = kmeans_logreg_select_k(train_x, val_x, test_x)
    labels_all = save_outputs(results, best_k)

    log_temporal_distribution(labels_all, splits)
    plot_model_selection(results)
    visualise_clusters_tsne(all_x, labels_all)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()