import pandas as pd
from pathlib import Path

CLUSTERING_OUT_DIR = Path("outputs/clustering_analysis")
TRAIN_CLUSTER_Z_PATH = Path("outputs/train_cluster_features_z.npy")
VAL_CLUSTER_Z_PATH = Path("outputs/val_cluster_features_z.npy")
TEST_CLUSTER_Z_PATH = Path("outputs/test_cluster_features_z.npy")
WINDOW_END_DATES_PATH = Path("outputs_preprocess/window_end_dates.npy")
CLUSTER_WINDOW_END_DATES_PATH = Path("outputs/clustering_analysis/window_end_dates.npy")

ERROR_ANALYSIS_OUT_DIR: Path = Path("outputs/error_analysis")
TRACK2_TARGET_CONT_PATH: Path = ERROR_ANALYSIS_OUT_DIR / "track2_target_continuous.npy"
TRACK2_TARGET_BIN_PATH: Path = ERROR_ANALYSIS_OUT_DIR / "track2_target_binary.npy"
TRACK2_THRESH_PATH: Path = ERROR_ANALYSIS_OUT_DIR / "threshold.json"
ANOMALY_PERCENTILE: int = 90    # threshold fit on train set (Malhotra et al., 2016)
TRAIN_ERR_PATH: Path = Path("outputs/train_errors.npy")
VAL_ERR_PATH: Path = Path("outputs/val_errors.npy")
TEST_ERR_PATH: Path = Path("outputs/test_errors.npy")

PREP_DIR: Path = Path("outputs_preprocess")
TRAIN_W_PATH: Path = PREP_DIR / "train_windows.npy"
VAL_W_PATH: Path = PREP_DIR / "val_windows.npy"
TEST_W_PATH: Path = PREP_DIR / "test_windows.npy"
TRAIN_WIN_MU_PATH: Path = PREP_DIR / "train_win_mu.npy"
VAL_WIN_MU_PATH: Path = PREP_DIR / "val_win_mu.npy"
TEST_WIN_MU_PATH: Path = PREP_DIR / "test_win_mu.npy"
TRAIN_WIN_SD_PATH: Path = PREP_DIR / "train_win_sd.npy"
VAL_WIN_SD_PATH: Path = PREP_DIR / "val_win_sd.npy"
TEST_WIN_SD_PATH: Path = PREP_DIR / "test_win_sd.npy"

BEST_MODEL_PATH: Path = Path("outputs/best_model.pth")
KM_MODEL_PATH: Path = CLUSTERING_OUT_DIR / "best_km_model.pkl"
LR_MODEL_PATH: Path = CLUSTERING_OUT_DIR / "best_lr_model.pkl"
CLUSTER_STANDARDISE_MU_PATH: Path = Path("outputs/cluster_standardise_mu.npy")
CLUSTER_STANDARDISE_SD_PATH: Path = Path("outputs/cluster_standardise_sd.npy")

TIMESHAP_TRACK1_OUT_DIR: Path = Path("outputs/timeshap_track1")
TIMESHAP_TRACK2_OUT_DIR: Path = Path("outputs/timeshap_track2")
VECTOR_SHAP_TRACK1_OUT_DIR: Path = Path("outputs/vector_shap_track1")
VECTOR_SHAP_TRACK2_OUT_DIR: Path = Path("outputs/vector_shap_track2")
REGIME_TRACK1_OUT_DIR: Path = Path("outputs/regime_analysis/track1")
REGIME_TRACK2_OUT_DIR: Path = Path("outputs/regime_analysis/track2")

FEATURE_NAMES_PATH: Path = PREP_DIR / "feature_names.npy"


N_FEATURES = 6  # number of input features per timestep
RANDOM_SEED: int = 42
EXPLAIN_SEED: int = 123 # distinct from RANDOM_SEED to decouple explain-set from background sampling

INDICATOR_NAMES: list[str] = [
    "Log_Returns",
    "Realised_Volatility",
    "RSI",
    "MACD",
    "Normalised_Volume",
    "Momentum",
]

FAMILY_NAMES: list[str] = [
    "Price Dynamics",
    "Volatility Indicators",
    "Technical Indicators",
]

FAMILY_KEYS: list[str] = [
    "price_dynamics",
    "volatility_indicators", 
    "technical_indicators",
]

LEVEL2_FAMILIES: dict[str, list[str]] = {
    "price_dynamics": ["Log_Returns", "Momentum"],
    "volatility_indicators": ["Realised_Volatility", "Normalised_Volume"],
    "technical_indicators": ["RSI", "MACD"],
}

DATA_PATH = "sp500_daily_2000_2025.csv"

# Model architecture
DEVICE: str = "cpu"
HIDDEN_DIM: int = 64    # consistent with Franco De La Peña et al. (2025); moderate capacity relative to input dim 6 and latent dim 32 (Fawaz et al., 2019)
LATENT_DIM: int = 32
NUM_LAYERS: int = 2 # Sutskever et al. (2014)
DROPOUT: float = 0.2    # Fawaz et al. (2019)

# Training
BATCH_SIZE: int = 64
EPOCHS: int = 500
LEARNING_RATE: float = 1e-3 # Fawaz et al. (2019)
PATIENCE_EARLY_STOP: int = 20   # Malhotra et al. (2016)
PATIENCE_LR_REDUCE: int = 50    # Fawaz et al. (2019)
LR_REDUCTION_FACTOR: float = 0.5    # halve LR after patience epochs without improvement (Fawaz et al., 2019)

# Clustering
N_CLUSTERS: int = 3
K_RANGE: list[int] = [2, 3, 4, 5, 6]
MIN_CLUSTER_SHARE: float = 0.01 # min share per cluster on every split (train/val/test)

# Vector SHAP
N_PERM: int = 32_000
BACKGROUND_K: int = 500 # stratified background set (Antwarg et al., 2020; Cohen et al., 2023)
MAX_EXPLAIN: int = 300

# TimeSHAP
TIMESHAP_PRUNE_ETA: float = 0.025   # temporal pruning tolerance (Bento et al., 2021)
TIMESHAP_MIN_TIMESTEPS_KEEP: int = 5    # minimum timesteps retained after pruning (Bento et al., 2021)
TIMESHAP_MC_EVENT: int = 100 # Monte Carlo permutations for event-level SHAP (Bento et al., 2021)
TIMESHAP_N_EXPLAIN_TEST: int = 100
TIMESHAP_N_BACKGROUND: int = 100    # background windows from train set for baseline (Bento et al., 2021)

def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, skiprows=[1])

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()

    df = df.ffill().bfill()
    df = df.apply(pd.to_numeric, errors="coerce").dropna()

    return df