import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

import config
from model import LSTMAutoencoder

NUM_WORKERS: int = 0
GRAD_CLIP_NORM: float = 5.0     # gradient clipping norm (Sutskever et al., 2014)
STANDARDISATION_EPS: float = 1e-8

OUTPUT_DIR: Path = Path("outputs")
PREPROCESS_DIR: Path = Path("outputs_preprocess")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


class TimeSeriesDataset(Dataset):

    def __init__(self, windows: np.ndarray) -> None:
        self.windows = torch.FloatTensor(windows)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.windows[idx]


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimiser: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_norm: float = GRAD_CLIP_NORM,
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0
    for batch in dataloader:
        batch = batch.to(device)
        optimiser.zero_grad()
        reconstruction, _ = model(batch)
        loss = criterion(reconstruction, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimiser.step()
        total_loss += loss.item()
        num_batches += 1
    return total_loss / num_batches


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            reconstruction, _ = model(batch)
            total_loss += criterion(reconstruction, batch).item()
            num_batches += 1
    return total_loss / num_batches


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = config.EPOCHS,
    lr: float = config.LEARNING_RATE,
    patience_early_stop: int = config.PATIENCE_EARLY_STOP,
    patience_lr_reduce: int = config.PATIENCE_LR_REDUCE,
    lr_factor: float = config.LR_REDUCTION_FACTOR,
    checkpoint_path: Path = OUTPUT_DIR / "best_model.pth",
    history_path: Path = OUTPUT_DIR / "training_history.json",
) -> dict[str, list[float]]:
    """Train the model with Adam optimisation, ReduceLROnPlateau scheduling, gradient clipping, and early stopping"""
    criterion = nn.MSELoss()        # reconstruction objective (Malhotra et al., 2016)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)     # Adam (Kingma & Ba, 2015); lr=1e-3 (Fawaz et al., 2019)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(     # factor=0.5, patience=50 (Fawaz et al., 2019)
        optimiser, mode="min", factor=lr_factor, patience=patience_lr_reduce
    )

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "lr": []}
    best_val_loss = float("inf")
    best_epoch = 1
    epochs_no_improve = 0

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimiser, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        current_lr = optimiser.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        log.info(
            "Epoch %3d/%d | Train: %.6f | Val: %.6f | LR: %.2e",
            epoch + 1, epochs, train_loss, val_loss, current_lr,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            epochs_no_improve = 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict()}, checkpoint_path)
            log.info("  → Checkpoint saved (val_loss: %.6f)", val_loss)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience_early_stop:    # early stopping (Malhotra et al., 2016)
            log.info("Early stopping at epoch %d. Best val_loss=%.6f", epoch + 1, best_val_loss)
            break

    history["best_epoch"] = best_epoch

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    return history


def plot_loss_curves(history: dict[str, list[float]], save_path: Path = OUTPUT_DIR / "loss_curves.png") -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    _, (ax_loss, ax_lr) = plt.subplots(1, 2, figsize=(14, 5))

    ax_loss.plot(epochs, history["train_loss"], label="Train", linewidth=2)
    ax_loss.plot(epochs, history["val_loss"], label="Validation", linewidth=2)
    if "best_epoch" in history:
        ax_loss.axvline(history["best_epoch"], color="red", linestyle="--", linewidth=1.5, label=f"Best checkpoint (epoch {history['best_epoch']})")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("MSE")
    ax_loss.set_title("Loss Curves")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    ax_lr.plot(epochs, history["lr"], linewidth=2)
    if "best_epoch" in history:
        ax_lr.axvline(history["best_epoch"], color="red", linestyle="--", linewidth=1.5, label=f"Best checkpoint (epoch {history['best_epoch']})")
        ax_lr.legend()
    ax_lr.set_xlabel("Epoch")
    ax_lr.set_ylabel("Learning Rate")
    ax_lr.set_yscale("log")
    ax_lr.set_title("LR Schedule")
    ax_lr.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def extract_latents(model: nn.Module, dataloader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    latents = []
    with torch.no_grad():
        for batch in dataloader:
            _, latent = model(batch.to(device))
            latents.append(latent.cpu().numpy())
    return np.vstack(latents)


def extract_reconstruction_errors(model: nn.Module, dataloader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    per_sample_mse = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            reconstruction, _ = model(batch)
            mse = ((reconstruction - batch) ** 2).mean(dim=[1, 2])
            per_sample_mse.append(mse.cpu().numpy())
    return np.concatenate(per_sample_mse)


def fit_standardiser(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True) + STANDARDISATION_EPS
    return mean, std


def apply_standardiser(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def load_data(base: Path):
    train_windows = np.load(base / "train_windows.npy")
    val_windows = np.load(base / "val_windows.npy")
    test_windows = np.load(base / "test_windows.npy")

    train_win_mu = np.load(base / "train_win_mu.npy")
    val_win_mu = np.load(base / "val_win_mu.npy")
    test_win_mu = np.load(base / "test_win_mu.npy")
    train_win_sd = np.load(base / "train_win_sd.npy")
    val_win_sd = np.load(base / "val_win_sd.npy")
    test_win_sd = np.load(base / "test_win_sd.npy")

    return (
        train_windows, val_windows, test_windows,
        train_win_mu, val_win_mu, test_win_mu,
        train_win_sd, val_win_sd, test_win_sd,
    )


def make_eval_loader(dataset: Dataset) -> DataLoader:
    # shuffle=False is required to maintain temporal alignment with window metadata.
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)


def build_cluster_features(
    latents: np.ndarray,
    win_mu: np.ndarray,
    win_sd: np.ndarray,
    split_label: str,
) -> np.ndarray:
    # Horizontal composition of latent vectors with window-level statistics (Baldi, 2012)
    # Volatility level (win_sd) is the primary regime identifier (Ang & Timmermann, 2011)
    assert len(win_mu) == len(latents) and len(win_sd) == len(latents), (
        f"{split_label} window statistics are not aligned with latent vectors. "
        "Ensure DataLoaders used for extraction have shuffle=False."
    )
    return np.hstack([latents, win_mu, win_sd])


def main() -> None:
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    device = torch.device("cpu")
    OUTPUT_DIR.mkdir(exist_ok=True)

    base = PREPROCESS_DIR
    (
        train_windows, val_windows, test_windows,
        train_win_mu, val_win_mu, test_win_mu,
        train_win_sd, val_win_sd, test_win_sd,
    ) = load_data(base)

    train_ds = TimeSeriesDataset(train_windows)
    val_ds = TimeSeriesDataset(val_windows)
    test_ds = TimeSeriesDataset(test_windows)

    train_loader_shuffled = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    train_loader = make_eval_loader(train_ds)
    val_loader = make_eval_loader(val_ds)
    test_loader = make_eval_loader(test_ds)

    input_dim = train_windows.shape[2]
    model = LSTMAutoencoder(
        input_dim=input_dim,
        hidden_dim=config.HIDDEN_DIM,
        latent_dim=config.LATENT_DIM,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
    ).to(device)

    history = train_model(
        model,
        train_loader_shuffled,
        val_loader,
        device,
        checkpoint_path=OUTPUT_DIR / "best_model.pth",
        history_path=OUTPUT_DIR / "training_history.json",
    )
    plot_loss_curves(history, save_path=OUTPUT_DIR / "loss_curves.png")

    ckpt = torch.load(OUTPUT_DIR / "best_model.pth", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    train_latents = extract_latents(model, train_loader, device)
    val_latents = extract_latents(model, val_loader, device)
    test_latents = extract_latents(model, test_loader, device)

    np.save(OUTPUT_DIR / "train_latents.npy", train_latents)
    np.save(OUTPUT_DIR / "val_latents.npy", val_latents)
    np.save(OUTPUT_DIR / "test_latents.npy", test_latents)

    train_errors = extract_reconstruction_errors(model, train_loader, device)
    val_errors = extract_reconstruction_errors(model, val_loader, device)
    test_errors = extract_reconstruction_errors(model, test_loader, device)

    np.save(OUTPUT_DIR / "train_errors.npy", train_errors)
    np.save(OUTPUT_DIR / "val_errors.npy", val_errors)
    np.save(OUTPUT_DIR / "test_errors.npy", test_errors)

    train_cluster_x = build_cluster_features(train_latents, train_win_mu, train_win_sd, "train")
    val_cluster_x = build_cluster_features(val_latents, val_win_mu, val_win_sd, "val")
    test_cluster_x = build_cluster_features(test_latents, test_win_mu, test_win_sd, "test")

    scale_mean, scale_std = fit_standardiser(train_cluster_x)
    train_cluster_z = apply_standardiser(train_cluster_x, scale_mean, scale_std)
    val_cluster_z = apply_standardiser(val_cluster_x, scale_mean, scale_std)
    test_cluster_z = apply_standardiser(test_cluster_x, scale_mean, scale_std)

    np.save(OUTPUT_DIR / "train_cluster_features.npy", train_cluster_x)
    np.save(OUTPUT_DIR / "val_cluster_features.npy", val_cluster_x)
    np.save(OUTPUT_DIR / "test_cluster_features.npy", test_cluster_x)
    np.save(OUTPUT_DIR / "train_cluster_features_z.npy", train_cluster_z)
    np.save(OUTPUT_DIR / "val_cluster_features_z.npy", val_cluster_z)
    np.save(OUTPUT_DIR / "test_cluster_features_z.npy", test_cluster_z)
    np.save(OUTPUT_DIR / "cluster_standardise_mu.npy", scale_mean)
    np.save(OUTPUT_DIR / "cluster_standardise_sd.npy", scale_std)

    log.info("Latent shapes: %s %s %s", train_latents.shape, val_latents.shape, test_latents.shape)
    log.info("Mean errors — train: %.6f | val: %.6f | test: %.6f",
             train_errors.mean(), val_errors.mean(), test_errors.mean())
    log.info("Cluster feature shape (z): %s", train_cluster_z.shape)


if __name__ == "__main__":
    main()
