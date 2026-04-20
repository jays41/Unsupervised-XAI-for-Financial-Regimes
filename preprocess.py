import numpy as np
import pandas as pd
import logging
from pathlib import Path
import config

logger = logging.getLogger(__name__)


def preprocess_chrono(
    data: pd.DataFrame,
    window_size: int = 60,  # sliding window length (Malhotra et al., 2016)
    stride: int = 1,
    train_end: str = "2016-12-31",
    val_end: str = "2019-12-31",
    winsor_q: float = 0.01,
    per_window_norm: bool = True,
    save_dir: str | None = None,
    prefix: str = "",
    save_window_stats: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Scaler parameters are fit on the training partition only to prevent data leakage.
    Per-window z-scoring (per_window_norm) removes absolute-level information, preserving shape for the autoencoder.
    """
    data = data.sort_index()
    train_end_dt = pd.to_datetime(train_end)
    val_end_dt = pd.to_datetime(val_end)

    assert train_end_dt < val_end_dt, "train_end must be before val_end."

    train_raw = data.loc[data.index <= train_end_dt]
    val_raw = data.loc[(data.index > train_end_dt) & (data.index <= val_end_dt)]
    test_raw = data.loc[data.index > val_end_dt]

    for name, split in [("train", train_raw), ("val", val_raw), ("test", test_raw)]:
        assert len(split) >= window_size, (f"{name} partition ({len(split)} rows) is shorter than window_size ({window_size}).")

    train_median = train_raw.median(axis=0)
    train_iqr = (train_raw.quantile(0.75, axis=0) - train_raw.quantile(0.25, axis=0)).replace(0, 1.0)

    def robust_scale(df: pd.DataFrame) -> pd.DataFrame:
        return (df - train_median) / train_iqr

    train_scaled = robust_scale(train_raw)
    val_scaled = robust_scale(val_raw)
    test_scaled = robust_scale(test_raw)

    winsor_lower = train_scaled.quantile(winsor_q, axis=0)
    winsor_upper = train_scaled.quantile(1.0 - winsor_q, axis=0)

    def winsorise(df: pd.DataFrame) -> pd.DataFrame:
        return df.clip(lower=winsor_lower, upper=winsor_upper, axis=1)

    train_scaled = winsorise(train_scaled)
    val_scaled = winsorise(val_scaled)
    test_scaled = winsorise(test_scaled)

    scaled_series = pd.concat([train_scaled, val_scaled, test_scaled], axis=0)
    scaled_array = scaled_series.to_numpy(dtype=np.float32)
    date_index = scaled_series.index

    n_windows = (len(scaled_array) - window_size) // stride + 1
    start_offsets = stride * np.arange(n_windows)
    indices = start_offsets[:, None] + np.arange(window_size)[None, :]
    windows = scaled_array[indices]
    window_end_dates = date_index[(window_size - 1) + start_offsets]
    window_end_dates_str = window_end_dates.to_numpy("datetime64[D]").astype(str)

    if save_window_stats:
        window_means = windows.mean(axis=1)
        window_stds = windows.std(axis=1)
        window_stds = np.where(window_stds == 0, 1.0, window_stds)

    if per_window_norm:
        # Per-window z-normalisation: removes absolute-level information, critical for DNN performance (Fawaz et al., 2019)
        mu = windows.mean(axis=1, keepdims=True)
        sd = windows.std(axis=1, keepdims=True)
        sd = np.where(sd == 0, 1.0, sd)
        windows = (windows - mu) / sd

    train_mask = window_end_dates <= train_end_dt
    val_mask = (window_end_dates > train_end_dt) & (window_end_dates <= val_end_dt)
    test_mask = window_end_dates > val_end_dt

    train_windows = windows[train_mask]
    val_windows = windows[val_mask]
    test_windows = windows[test_mask]

    if save_dir:
        out = Path(save_dir)
        out.mkdir(parents=True, exist_ok=True)

        for split_name, arr in [("train_windows", train_windows), ("val_windows", val_windows), ("test_windows", test_windows)]:
            np.save(out / f"{prefix}{split_name}.npy", arr)

        np.save(out / f"{prefix}window_end_dates.npy", window_end_dates_str)
        np.save(out / f"{prefix}feature_names.npy", np.array(list(scaled_series.columns)))
        np.save(out / f"{prefix}median.npy", train_median.to_numpy())
        np.save(out / f"{prefix}iqr.npy", train_iqr.to_numpy())
        np.save(out / f"{prefix}winsor_lo.npy", winsor_lower.to_numpy())
        np.save(out / f"{prefix}winsor_hi.npy", winsor_upper.to_numpy())

        if save_window_stats:
            for split_name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
                np.save(out / f"{prefix}{split_name}_win_mu.npy", window_means[mask])
                np.save(out / f"{prefix}{split_name}_win_sd.npy", window_stds[mask])

        logger.info("Artefacts saved to %s", out)

    return train_windows, val_windows, test_windows, {
        "feature_names": list(scaled_series.columns),
        "train_end": train_end_dt,
        "val_end": val_end_dt,
        "window_size": window_size,
        "stride": stride,
        "per_window_norm": per_window_norm,
        "save_window_stats": save_window_stats,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    train_windows, val_windows, test_windows, params = preprocess_chrono(
        data=config.load_data(),
        window_size=60,
        stride=1,
        train_end="2016-12-31",
        val_end="2019-12-31",
        winsor_q=0.01,
        per_window_norm=True,
        save_window_stats=True,
        save_dir="outputs_preprocess",
    )

    logger.info(
        "Windows - train: %s | val: %s | test: %s",
        train_windows.shape,
        val_windows.shape,
        test_windows.shape,
    )
    logger.info("Features: %s", params["feature_names"])


if __name__ == "__main__":
    main()
