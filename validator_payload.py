from __future__ import annotations
import json
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

FEATURE_NAMES: list[str] = [
    "Log_Returns", "Realised_Volatility", "RSI",
    "MACD", "Normalised_Volume", "Momentum",
]

CLUSTER_LABELS: dict[int, str] = {0: "high_vol_bull", 1: "corrective", 2: "steady_bull"}

_N_TRAIN_VAL: int = 4951  # 4197 train + 754 val


class _Cache:
    def __init__(self, d: Path) -> None:
        self._d = d
        self._store: dict = {}

    def _load(self, key: str, path: Path, **kw) -> np.ndarray:
        if key not in self._store:
            self._store[key] = np.load(path, **kw)
        return self._store[key]

    @property
    def labels(self):   return self._load("labels", self._d / "clustering_analysis/labels_test_k3.npy")
    @property
    def proba(self):    return self._load("proba",  self._d / "clustering_analysis/proba_test_k3.npy")
    @property
    def dates(self):    return self._load("dates",  self._d / "clustering_analysis/window_end_dates.npy", allow_pickle=True)
    @property
    def errors(self):   return self._load("errors", self._d / "test_errors.npy")
    @property
    def vshap(self):    return self._load("vshap",  self._d / "vector_shap_track1/vectorshap_level1_test.npy")
    @property
    def ts_shap(self):  return self._load("ts_shap",self._d / "timeshap_track2/track2_feature_shap.npy")
    @property
    def ts_idx(self):   return self._load("ts_idx", self._d / "timeshap_track2/explain_indices_test.npy")

    @property
    def threshold(self) -> float:
        if "thresh" not in self._store:
            with open(self._d / "error_analysis/threshold.json") as f:
                self._store["thresh"] = float(json.load(f)["value"])
        return self._store["thresh"]


_caches: dict[str, _Cache] = {}

def _cache(outputs_dir: Path) -> _Cache:
    key = str(outputs_dir.resolve())
    if key not in _caches:
        _caches[key] = _Cache(outputs_dir)
    return _caches[key]


def build_validator_payload(
    window_idx: int,
    outputs_dir: str | Path = "outputs",
    n_train_val: int = _N_TRAIN_VAL,
) -> dict:
    c = _cache(Path(outputs_dir))

    if not 0 <= window_idx < len(c.labels):
        raise IndexError(f"window_idx={window_idx} out of range [0, {len(c.labels)})")

    date = str(c.dates[n_train_val + window_idx])
    cluster = int(c.labels[window_idx])
    label = CLUSTER_LABELS[cluster]
    confidence = float(c.proba[window_idx, cluster])

    # Vector SHAP
    if window_idx < len(c.vshap):
        shap_vec = c.vshap[window_idx, :, cluster]
    else:
        shap_vec = np.full(len(FEATURE_NAMES), float("nan"))
        logger.warning("window_idx=%d outside Vector SHAP range; attributions will be NaN.", window_idx)

    attributions = {name: float(shap_vec[i]) for i, name in enumerate(FEATURE_NAMES)}
    dominant = FEATURE_NAMES[int(np.argmax(np.abs(shap_vec)))] if not np.all(np.isnan(shap_vec)) else "N/A"

    # Anomaly
    error = float(c.errors[window_idx])
    threshold = c.threshold
    flagged = error >= threshold

    # TimeSHAP
    ts_match = np.where(c.ts_idx == window_idx)[0]
    if ts_match.size:
        ts_vec = c.ts_shap[int(ts_match[0]), :, 0]
        top_drivers = [FEATURE_NAMES[int(j)] for j in np.argsort(np.abs(ts_vec))[::-1][:2]]
    else:
        top_drivers = []
        logger.warning("window_idx=%d not in TimeSHAP explain set; top_error_drivers will be empty.", window_idx)


    return {
        "window_end_date": date,
        "regime": {
            "assigned_cluster": cluster,
            "cluster_label": label,
            "assignment_confidence": round(confidence, 6),
            "feature_attributions": {k: round(v, 8) for k, v in attributions.items()},
            "dominant_driver": dominant,
        },
        "anomaly": {
            "reconstruction_error": round(error, 10),
            "threshold": round(threshold, 10),
            "flagged": flagged,
            "top_error_drivers": top_drivers,
        }
    }


def dump_payload(payload: dict, path: str | Path | None = None) -> str:
    text = json.dumps(payload, indent=2)
    if path is not None:
        Path(path).write_text(text, encoding="utf-8")
    return text


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    DEMOS: list[tuple[int, str]] = [
        (39,  "covid_onset__corrective__flagged"),  # 2020-02-28  COVID crash
        (3,   "jan2020_calm__high_vol_bull__normal"),   # 2020-01-07  calm bull
        (229, "nov2020_postelection__steady_bull__flagged"),    # 2020-11-27  post-election
    ]

    out_dir = Path("outputs/validator_payload")
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, name in DEMOS:
        payload = build_validator_payload(idx)
        text = dump_payload(payload, path=out_dir / f"{name}.json")
        print(f"\n{'='*70}\nwindow_idx={idx}  ({name})\n{'='*70}")
        print(text)