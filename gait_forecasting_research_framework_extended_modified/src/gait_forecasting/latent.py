
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class PCALatentState:
    pca: PCA
    scaler: StandardScaler

    @property
    def n_components(self) -> int:
        return int(self.pca.n_components_)

    def transform(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X)
        return self.pca.transform(Xs)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        Xs = self.pca.inverse_transform(Z)
        return self.scaler.inverse_transform(Xs)


def fit_pca_latent_state(X: np.ndarray, n_components: int = 8, random_state: int = 42) -> PCALatentState:
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    pca = PCA(n_components=min(n_components, Xs.shape[1]), random_state=random_state)
    pca.fit(Xs)
    return PCALatentState(pca=pca, scaler=scaler)


class _Autoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_sizes: Sequence[int]):
        super().__init__()
        enc = []
        prev = input_dim
        for h in hidden_sizes:
            enc.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        enc.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*enc)

        dec = []
        prev = latent_dim
        for h in reversed(hidden_sizes):
            dec.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        dec.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


@dataclass
class AutoencoderLatentState:
    model: _Autoencoder
    scaler: StandardScaler
    latent_dim: int

    def transform(self, X: np.ndarray, device: str | None = None) -> np.ndarray:
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        self.model.eval()
        Xs = self.scaler.transform(X)
        with torch.no_grad():
            _, z = self.model(torch.tensor(Xs, dtype=torch.float32, device=device))
        return z.cpu().numpy()

    def inverse_transform(self, Z: np.ndarray, device: str | None = None) -> np.ndarray:
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        self.model.eval()
        with torch.no_grad():
            Xs = self.model.decoder(torch.tensor(Z, dtype=torch.float32, device=device))
        return self.scaler.inverse_transform(Xs.cpu().numpy())


def fit_autoencoder_latent_state(
    X: np.ndarray,
    latent_dim: int = 8,
    hidden_sizes: Sequence[int] = (64, 32),
    epochs: int = 50,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 8,
    random_state: int = 42,
    device: str | None = None,
) -> AutoencoderLatentState:
    torch.manual_seed(random_state)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X).astype(np.float32)
    model = _Autoencoder(input_dim=Xs.shape[1], latent_dim=latent_dim, hidden_sizes=hidden_sizes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    ds = TensorDataset(torch.tensor(Xs, dtype=torch.float32))
    n_val = max(1, int(0.15 * len(ds)))
    n_train = max(1, len(ds) - n_val)
    train_ds, val_ds = torch.utils.data.random_split(
        ds,
        [n_train, len(ds) - n_train],
        generator=torch.Generator().manual_seed(random_state),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_state = None
    best_val = float("inf")
    bad = 0
    for _ in range(epochs):
        model.train()
        for (xb,) in train_loader:
            xb = xb.to(device)
            opt.zero_grad()
            recon, _ = model(xb)
            loss = loss_fn(recon, xb)
            loss.backward()
            opt.step()
        model.eval()
        losses = []
        with torch.no_grad():
            for (xb,) in val_loader:
                xb = xb.to(device)
                recon, _ = model(xb)
                losses.append(loss_fn(recon, xb).item())
        val_loss = float(np.mean(losses)) if losses else float("inf")
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return AutoencoderLatentState(model=model, scaler=scaler, latent_dim=latent_dim)


def build_latent_state(H: np.ndarray, dH: np.ndarray | None = None, d2H: np.ndarray | None = None) -> np.ndarray:
    parts = [H]
    if dH is not None:
        parts.append(dH)
    if d2H is not None:
        parts.append(d2H)
    return np.concatenate(parts, axis=1)


def compute_ordered_differences(H: np.ndarray, order: int = 2) -> tuple[np.ndarray, ...]:
    diffs = []
    current = H
    for _ in range(order):
        d = np.vstack([np.zeros((1, current.shape[1])), np.diff(current, axis=0)])
        diffs.append(d)
        current = d
    return tuple(diffs)
