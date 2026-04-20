"""Benchmark Markov-switching AR and Gaussian HMM against the AE K-Means regimes."""

import logging
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score, silhouette_score, confusion_matrix
from hmmlearn.hmm import GaussianHMM
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
import config

OUT_DIR = Path("outputs/baselines")
N_REGIMES = config.N_CLUSTERS
logging.basicConfig(level=logging.WARNING)


def _setup_file_logger() -> logging.Logger:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return logging.getLogger("baselines_file")


def _load_inputs() -> dict:
    X_train = np.load(config.TRAIN_WIN_MU_PATH)
    X_val = np.load(config.VAL_WIN_MU_PATH)
    X_test = np.load(config.TEST_WIN_MU_PATH)
    non_const = X_train.std(axis=0) > 0
    X_train, X_val, X_test = X_train[:, non_const], X_val[:, non_const], X_test[:, non_const]
    X_all = np.vstack([X_train, X_val, X_test])
    splits = {
        "train": (0, len(X_train)),
        "val":   (len(X_train), len(X_train) + len(X_val)),
        "test":  (len(X_train) + len(X_val), len(X_all)),
    }
    dates = pd.to_datetime(np.load(config.WINDOW_END_DATES_PATH, allow_pickle=True))
    ae_labels = np.concatenate([
        np.load(config.CLUSTERING_OUT_DIR / f"labels_{s}_k{N_REGIMES}.npy")
        for s in ("train", "val", "test")
    ])
    vix_labels = np.load(Path("outputs/vix_validation/regime_labels.npy"))
    log_returns = config.load_data()["Log_Returns"].dropna()
    return dict(X_train=X_train, X_val=X_val, X_test=X_test, X_all=X_all,
                splits=splits, dates=dates, ae_labels=ae_labels,
                vix_labels=vix_labels, log_returns=log_returns)


def _persistence(labels: np.ndarray) -> float:
    return float((labels[:-1] == labels[1:]).mean()) if len(labels) > 1 else 1.0


def _fit_ms_ar(log_returns: pd.Series, train_end: pd.Timestamp, flog: logging.Logger):
    train_ret = log_returns[log_returns.index <= train_end]
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            mod = MarkovAutoregression(train_ret, k_regimes=N_REGIMES, order=1,
                                       switching_ar=False, switching_variance=True)
            res = mod.fit(disp=False)
            if not res.mle_retvals.get("converged", True):
                raise RuntimeError("did not converge on first attempt")
        except Exception as e:
            flog.warning("MS-AR first fit failed (%s); retrying with search_reps=20", e)
            try:
                res = mod.fit(disp=False, search_reps=20)
            except Exception as e2:
                flog.error("MS-AR retry also failed: %s", e2)
                return None, None
        for w in caught:
            flog.warning("MS-AR fit warning: %s", str(w.message))
    # Re-apply trained params to full series for out-of-sample smoothed probs
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        mod_full = MarkovAutoregression(log_returns, k_regimes=N_REGIMES, order=1,
                                        switching_ar=False, switching_variance=True)
        res_full = mod_full.fit(start_params=res.params, maxiter=1, disp=False)
    flog.info("MS-AR fit time: %.2f s  |  converged: %s  |  log-likelihood (train): %.4f",
              time.perf_counter() - t0, res.mle_retvals.get("converged", False), res.llf)
    probs = res_full.smoothed_marginal_probabilities
    if hasattr(probs, "index"):
        return probs.values, probs.index
    return probs, log_returns.index[-len(probs):]


def _ms_window_labels(probs: np.ndarray, ret_idx: pd.Index, dates: pd.DatetimeIndex) -> np.ndarray:
    prob_df = pd.DataFrame(probs, index=ret_idx)
    idxs = prob_df.index.get_indexer(dates, method="nearest")
    return prob_df.values[idxs].argmax(axis=1).astype(int)


def _fit_hmm(X_train: np.ndarray, X_all: np.ndarray, flog: logging.Logger):
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            model = GaussianHMM(n_components=N_REGIMES, covariance_type="full",
                                n_iter=100, random_state=42, min_covar=1e-3)
            model.fit(X_train)
        except Exception as e:
            flog.error("HMM fit failed: %s", e)
            return None
        for w in caught:
            flog.warning("HMM fit warning: %s", str(w.message))
    flog.info("HMM fit time: %.2f s  |  converged: %s  |  log-likelihood (train): %.4f",
              time.perf_counter() - t0, model.monitor_.converged, model.score(X_train))
    return model.predict(X_all)


def _metrics_per_split(labels, ae, vix, X_all, splits, silh=False) -> list[dict]:
    rows = []
    for split, (s, e) in splits.items():
        ml, a, v, X = labels[s:e], ae[s:e], vix[s:e], X_all[s:e]
        row = {"split": split,
               "NMI_vs_AE":  normalized_mutual_info_score(a, ml, average_method="arithmetic"),
               "NMI_vs_VIX": normalized_mutual_info_score(v, ml, average_method="arithmetic"),
               "persistence": _persistence(ml), "n_windows": e - s}
        if silh and len(np.unique(ml)) > 1:
            row["silhouette"] = silhouette_score(X, ml)
        rows.append(row)
    return rows


def _proportions(labels: np.ndarray, splits: dict) -> list[dict]:
    rows = []
    for split, (s, e) in splits.items():
        props = np.bincount(labels[s:e], minlength=N_REGIMES).astype(float)
        props /= props.sum()
        rows += [{"split": split, "regime_id": r, "proportion": props[r]} for r in range(N_REGIMES)]
    return rows


def _confusion_df(a, b, na, nb) -> pd.DataFrame:
    cm = confusion_matrix(a, b)
    return pd.DataFrame(cm, index=[f"{na}_{i}" for i in range(cm.shape[0])],
                        columns=[f"{nb}_{i}" for i in range(cm.shape[1])])


def main() -> None:
    flog = _setup_file_logger()
    d = _load_inputs()
    train_end = d["dates"][d["splits"]["train"][1] - 1]
    ts, te = d["splits"]["test"]

    probs, ret_idx = _fit_ms_ar(d["log_returns"], train_end, flog)
    if probs is not None:
        ms_labels = _ms_window_labels(probs, ret_idx, d["dates"])
        ms_sum = _metrics_per_split(ms_labels, d["ae_labels"], d["vix_labels"], d["X_all"], d["splits"])
        ms_prop = _proportions(ms_labels, d["splits"])
    else:
        ms_labels, ms_sum, ms_prop = None, [], []

    hmm_labels = _fit_hmm(d["X_train"], d["X_all"], flog)
    if hmm_labels is not None:
        hmm_sum = _metrics_per_split(hmm_labels, d["ae_labels"], d["vix_labels"], d["X_all"], d["splits"], silh=True)
        hmm_prop = _proportions(hmm_labels, d["splits"])
    else:
        hmm_sum, hmm_prop = [], []

    pd.DataFrame(
        [{"method": "MS-AR", **r} for r in ms_sum] + [{"method": "HMM", **r} for r in hmm_sum]
    ).to_csv(OUT_DIR / "summary_metrics.csv", index=False)

    pd.DataFrame(
        [{"method": "MS-AR", **r} for r in ms_prop] + [{"method": "HMM", **r} for r in hmm_prop]
    ).to_csv(OUT_DIR / "regime_proportions.csv", index=False)

    ae_test = d["ae_labels"][ts:te]
    if ms_labels is not None:
        _confusion_df(ms_labels[ts:te], ae_test, "MS", "AE").to_csv(OUT_DIR / "confusion_ms_vs_ae.csv")
    if hmm_labels is not None:
        _confusion_df(hmm_labels[ts:te], ae_test, "HMM", "AE").to_csv(OUT_DIR / "confusion_hmm_vs_ae.csv")


if __name__ == "__main__":
    main()
