from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

from config import N_FEATURES, HIDDEN_DIM, LATENT_DIM, NUM_LAYERS, DROPOUT, RANDOM_SEED, DEVICE


def _init_lstm_weights(lstm: nn.LSTM) -> None:
    # Forget gate biases initialised to 1 to aid gradient flow early in training (Hochreiter & Schmidhuber, 1997)
    for name, param in lstm.named_parameters():
        if "weight_ih" in name:
            nn.init.xavier_uniform_(param)      # Xavier uniform (Glorot & Bengio, 2010; Fawaz et al., 2019)
        elif "weight_hh" in name:
            nn.init.orthogonal_(param)          # Orthogonal init (Glorot & Bengio, 2010; Fawaz et al., 2019)
        elif "bias" in name:
            nn.init.zeros_(param)
            n = param.size(0)
            param.data[n // 4 : n // 2].fill_(1.0)


class LSTMEncoder(nn.Module):
    # LSTM gate equations (Hochreiter & Schmidhuber, 1997)
    def __init__(
        self,
        input_dim: int = N_FEATURES,
        hidden_dim: int = HIDDEN_DIM,
        latent_dim: int = LATENT_DIM,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.hidden_to_latent = nn.Linear(hidden_dim, latent_dim)
        _init_lstm_weights(self.lstm)
        nn.init.xavier_uniform_(self.hidden_to_latent.weight)
        nn.init.zeros_(self.hidden_to_latent.bias)

    def forward(self, x: Tensor) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        _, (h_n, c_n) = self.lstm(x)
        latent = torch.tanh(self.hidden_to_latent(h_n[-1]))    # tanh on latent vector (Cho et al., 2014)
        return latent, (h_n, c_n)


class LSTMDecoder(nn.Module):
    # Encoder-decoder separation (Cho et al., 2014; Sutskever et al., 2014)
    def __init__(
        self,
        output_dim: int = N_FEATURES,
        hidden_dim: int = HIDDEN_DIM,
        latent_dim: int = LATENT_DIM,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(output_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        _init_lstm_weights(self.lstm)
        nn.init.xavier_uniform_(self.latent_to_hidden.weight)
        nn.init.zeros_(self.latent_to_hidden.bias)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x_reversed: Tensor, hidden_state: tuple[Tensor, Tensor]) -> Tensor:
        output, _ = self.lstm(x_reversed, hidden_state)
        return self.output_layer(output)


class LSTMAutoencoder(nn.Module):
    # Decodes in reverse temporal order (Malhotra et al., 2016; Sutskever et al., 2014)
    # Latent space used as basis for clustering and regime labelling (Baldi, 2012)

    def __init__(
        self,
        input_dim: int = N_FEATURES,
        hidden_dim: int = HIDDEN_DIM,
        latent_dim: int = LATENT_DIM,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.encoder = LSTMEncoder(input_dim, hidden_dim, latent_dim, num_layers, dropout)
        self.decoder = LSTMDecoder(input_dim, hidden_dim, latent_dim, num_layers, dropout)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        latent, hidden_state = self.encoder(x)
        reconstruction_reversed = self.decoder(torch.flip(x, dims=[1]), hidden_state)
        return torch.flip(reconstruction_reversed, dims=[1]), latent


def load_autoencoder(
    model_path: Path | str,
    input_dim: int = N_FEATURES,
    device: str = DEVICE,
) -> "LSTMAutoencoder":
    model = LSTMAutoencoder(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)
    ckpt = torch.load(model_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model


if __name__ == "__main__":
    torch.manual_seed(RANDOM_SEED)
    batch = 32
    seq_len = 60
    dummy_input = torch.randn(batch, seq_len, N_FEATURES)
    model = LSTMAutoencoder()

    model.eval()
    with torch.no_grad():
        reconstruction, latent = model(dummy_input)

    print(f"Input:          {dummy_input.shape}")
    print(f"Reconstruction: {reconstruction.shape}")
    print(f"Latent: {latent.shape}")
    print(f"MSE Loss: {nn.MSELoss()(reconstruction, dummy_input).item():.4f}")
