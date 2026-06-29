from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Device / AMP helpers
# ---------------------------------------------------------------------------

def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _amp_enabled() -> bool:
    return torch.cuda.is_available()


@contextlib.contextmanager
def _autocast_ctx():
    if _amp_enabled():
        with autocast():
            yield
    else:
        yield


def _to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def _maybe_compile(model: nn.Module) -> nn.Module:
    if _amp_enabled() and hasattr(torch, "compile"):
        try:
            return torch.compile(model)
        except Exception:
            pass
    return model


# ---------------------------------------------------------------------------
# PCA latent state (sklearn — CPU, unchanged interface)
# ---------------------------------------------------------------------------

@dataclass
class PCALatentState:
    pca: PCA
    scaler: StandardScaler

    @property
    def n_components(self) -> int:
        return int(self.pca.n_components_)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.pca.transform(self.scaler.transform(X))

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(self.pca.inverse_transform(Z))


def fit_pca_latent_state(
    X: np.ndarray,
    n_components: int = 8,
    random_state: int = 42,
) -> PCALatentState:
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    pca = PCA(n_components=min(n_components, Xs.shape[1]), random_state=random_state)
    pca.fit(Xs)
    return PCALatentState(pca=pca, scaler=scaler)


# ---------------------------------------------------------------------------
# Autoencoder
# ---------------------------------------------------------------------------

class _Autoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_sizes: Sequence[int],
    ):
        super().__init__()
        enc: list[nn.Module] = []
        prev = input_dim
        for h in hidden_sizes:
            enc.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        enc.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*enc)

        dec: list[nn.Module] = []
        prev = latent_dim
        for h in reversed(hidden_sizes):
            dec.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        dec.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        return self.decoder(z), z


@dataclass
class AutoencoderLatentState:
    model: _Autoencoder
    scaler: StandardScaler
    latent_dim: int
    _device: torch.device = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self._device is None:
            object.__setattr__(self, "_device", _get_device())

    def _get_base_model(self) -> _Autoencoder:
        """Unwrap torch.compile wrapper if present."""
        return getattr(self.model, "_orig_mod", self.model)

    def transform(self, X: np.ndarray, device: Optional[str] = None) -> np.ndarray:
        dev = torch.device(device) if device else self._device
        self.model.to(dev)
        self.model.eval()
        Xs = _to_tensor(self.scaler.transform(X), dev)
        with torch.no_grad(), _autocast_ctx():
            _, z = self.model(Xs)
        return _to_numpy(z)

    def inverse_transform(self, Z: np.ndarray, device: Optional[str] = None) -> np.ndarray:
        dev = torch.device(device) if device else self._device
        base = self._get_base_model()
        base.to(dev)
        base.eval()
        Zt = _to_tensor(Z, dev)
        with torch.no_grad(), _autocast_ctx():
            Xs = base.decoder(Zt)
        return self.scaler.inverse_transform(_to_numpy(Xs))


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
    device: Optional[str] = None,
) -> AutoencoderLatentState:
    torch.manual_seed(random_state)
    dev = torch.device(device) if device else _get_device()
    pin = dev.type == "cuda"

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X).astype(np.float32)

    model = _Autoencoder(
        input_dim=Xs.shape[1],
        latent_dim=latent_dim,
        hidden_sizes=hidden_sizes,
    ).to(dev)
    model = _maybe_compile(model)

    opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    grad_scaler = GradScaler(enabled=_amp_enabled())

    full_ds = TensorDataset(torch.as_tensor(Xs))
    n_val = max(1, int(0.15 * len(full_ds)))
    n_train = max(1, len(full_ds) - n_val)
    train_ds, val_ds = torch.utils.data.random_split(
        full_ds,
        [n_train, len(full_ds) - n_train],
        generator=torch.Generator().manual_seed(random_state),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin,
        num_workers=0,
    )

    best_state: Optional[dict] = None
    best_val = float("inf")
    bad = 0

    for _ in range(epochs):
        model.train()
        for (xb,) in train_loader:
            xb = xb.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with _autocast_ctx():
                recon, _ = model(xb)
                loss = loss_fn(recon, xb)
            grad_scaler.scale(loss).backward()
            grad_scaler.step(opt)
            grad_scaler.update()

        model.eval()
        losses: list[float] = []
        with torch.no_grad():
            for (xb,) in val_loader:
                xb = xb.to(dev, non_blocking=True)
                with _autocast_ctx():
                    recon, _ = model(xb)
                    losses.append(float(loss_fn(recon, xb).item()))

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

    return AutoencoderLatentState(model=model, scaler=scaler, latent_dim=latent_dim, _device=dev)


# ---------------------------------------------------------------------------
# Utility functions (public API preserved)
# ---------------------------------------------------------------------------

def build_latent_state(
    H: np.ndarray,
    dH: Optional[np.ndarray] = None,
    d2H: Optional[np.ndarray] = None,
) -> np.ndarray:
    parts = [H]
    if dH is not None:
        parts.append(dH)
    if d2H is not None:
        parts.append(d2H)
    return np.concatenate(parts, axis=1)


def compute_ordered_differences(
    H: np.ndarray,
    order: int = 2,
) -> tuple[np.ndarray, ...]:
    device = _get_device()
    Ht = torch.as_tensor(H, dtype=torch.float32, device=device)
    diffs: list[np.ndarray] = []
    current = Ht
    for _ in range(order):
        d = torch.cat(
            [torch.zeros(1, current.shape[1], device=device), torch.diff(current, dim=0)],
            dim=0,
        )
        diffs.append(_to_numpy(d))
        current = d
    return tuple(diffs)
