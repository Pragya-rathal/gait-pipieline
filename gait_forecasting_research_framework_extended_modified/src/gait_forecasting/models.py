from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Device helpers
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


def _to_tensor(x: np.ndarray, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.as_tensor(x, dtype=dtype, device=device)


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Sklearn baselines (CPU — unchanged interface)
# ---------------------------------------------------------------------------

def make_rf(random_state: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=random_state,
        class_weight="balanced_subsample",
    )


def make_sklearn_mlp(
    hidden_layers: Sequence[int] = (128, 64),
    random_state: int = 42,
) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=tuple(hidden_layers),
                    activation="relu",
                    solver="adam",
                    alpha=1e-4,
                    batch_size=256,
                    learning_rate_init=1e-3,
                    max_iter=300,
                    random_state=random_state,
                    early_stopping=True,
                    n_iter_no_change=10,
                    validation_fraction=0.15,
                ),
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Torch model definitions
# ---------------------------------------------------------------------------

class TorchMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        hidden_sizes: Sequence[int] = (128, 64),
        dropout: float = 0.2,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_sizes:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TorchGRUClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.head(out[:, -1])


class TorchBiLSTMClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        out_dim = hidden_size * 2
        self.head = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1])


class _TCNBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.downsample(x)
        y = self.relu(self.dropout(self.conv1(x)[..., : residual.shape[-1]]))
        y = self.relu(self.dropout(self.conv2(y)[..., : residual.shape[-1]]))
        return self.relu(y + residual)


class TorchTCNClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        channels: Sequence[int] = (64, 64, 64),
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        blocks: list[nn.Module] = []
        prev = input_dim
        for i, ch in enumerate(channels):
            blocks.append(_TCNBlock(prev, ch, kernel_size=kernel_size, dilation=2 ** i, dropout=dropout))
            prev = ch
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.LayerNorm(prev),
            nn.Linear(prev, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) -> (B, C, T)
        return self.head(self.tcn(x.transpose(1, 2)))


# ---------------------------------------------------------------------------
# Training result
# ---------------------------------------------------------------------------

@dataclass
class TorchTrainResult:
    model: nn.Module
    best_val_loss: float


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> Tuple[DataLoader, DataLoader]:
    # Pin memory only when CUDA is available so transfers are async
    pin = device.type == "cuda"

    def _ds(X: np.ndarray, y: np.ndarray) -> TensorDataset:
        return TensorDataset(
            torch.as_tensor(X, dtype=torch.float32),
            torch.as_tensor(y, dtype=torch.long),
        )

    train_loader = DataLoader(
        _ds(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin,
        num_workers=0,
        persistent_workers=False,
    )
    val_loader = DataLoader(
        _ds(X_val, y_val),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin,
        num_workers=0,
        persistent_workers=False,
    )
    return train_loader, val_loader


def _maybe_compile(model: nn.Module) -> nn.Module:
    """Apply torch.compile when supported (PyTorch >= 2.0, CUDA available)."""
    if _amp_enabled() and hasattr(torch, "compile"):
        try:
            return torch.compile(model)
        except Exception:
            pass
    return model


# ---------------------------------------------------------------------------
# Core training loop
# ---------------------------------------------------------------------------

def _train_torch_classifier(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 256,
    epochs: int = 40,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 8,
    random_state: int = 42,
    device: Optional[str] = None,
) -> TorchTrainResult:
    torch.manual_seed(random_state)
    dev = torch.device(device) if device else _get_device()
    model = model.to(dev)
    model = _maybe_compile(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=_amp_enabled())

    train_loader, val_loader = _make_loaders(X_train, y_train, X_val, y_val, batch_size, dev)

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val = float("inf")
    bad = 0

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(dev, non_blocking=True)
            yb = yb.to(dev, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _autocast_ctx():
                loss = criterion(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        model.eval()
        losses: list[float] = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(dev, non_blocking=True)
                yb = yb.to(dev, non_blocking=True)
                with _autocast_ctx():
                    losses.append(float(criterion(model(xb), yb).item()))

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
    return TorchTrainResult(model=model, best_val_loss=best_val)


# ---------------------------------------------------------------------------
# Public training functions
# ---------------------------------------------------------------------------

def train_torch_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_dim: int,
    n_classes: int,
    hidden_sizes: Sequence[int] = (128, 64),
    dropout: float = 0.2,
    batch_size: int = 256,
    epochs: int = 40,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 8,
    random_state: int = 42,
    device: Optional[str] = None,
) -> TorchTrainResult:
    model = TorchMLP(
        input_dim=input_dim,
        n_classes=n_classes,
        hidden_sizes=hidden_sizes,
        dropout=dropout,
    )
    return _train_torch_classifier(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        patience=patience,
        random_state=random_state,
        device=device,
    )


def train_gru_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_dim: int,
    n_classes: int,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.2,
    batch_size: int = 256,
    epochs: int = 40,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 8,
    random_state: int = 42,
    device: Optional[str] = None,
    bidirectional: bool = False,
) -> TorchTrainResult:
    model = TorchGRUClassifier(
        input_dim=input_dim,
        n_classes=n_classes,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
    )
    return _train_torch_classifier(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        patience=patience,
        random_state=random_state,
        device=device,
    )


def train_bilstm_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_dim: int,
    n_classes: int,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.2,
    batch_size: int = 256,
    epochs: int = 40,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 8,
    random_state: int = 42,
    device: Optional[str] = None,
) -> TorchTrainResult:
    model = TorchBiLSTMClassifier(
        input_dim=input_dim,
        n_classes=n_classes,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )
    return _train_torch_classifier(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        patience=patience,
        random_state=random_state,
        device=device,
    )


def train_tcn_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_dim: int,
    n_classes: int,
    channels: Sequence[int] = (64, 64, 64),
    kernel_size: int = 3,
    dropout: float = 0.2,
    batch_size: int = 256,
    epochs: int = 40,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 8,
    random_state: int = 42,
    device: Optional[str] = None,
) -> TorchTrainResult:
    model = TorchTCNClassifier(
        input_dim=input_dim,
        n_classes=n_classes,
        channels=channels,
        kernel_size=kernel_size,
        dropout=dropout,
    )
    return _train_torch_classifier(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        patience=patience,
        random_state=random_state,
        device=device,
    )


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_torch(
    model: nn.Module,
    X: np.ndarray,
    device: Optional[str] = None,
    batch_size: int = 2048,
) -> np.ndarray:
    dev = torch.device(device) if device else _get_device()
    model = model.to(dev)
    model.eval()

    Xt = torch.as_tensor(X, dtype=torch.float32)
    preds_list: list[torch.Tensor] = []

    with torch.no_grad():
        for start in range(0, len(Xt), batch_size):
            xb = Xt[start : start + batch_size].to(dev, non_blocking=True)
            with _autocast_ctx():
                logits = model(xb)
            preds_list.append(torch.argmax(logits, dim=1).cpu())

    return torch.cat(preds_list).numpy()


# ---------------------------------------------------------------------------
# Parameter counting
# ---------------------------------------------------------------------------

def model_parameter_count(model) -> int:
    if isinstance(model, nn.Module):
        # unwrap torch.compile wrapper if present
        underlying = getattr(model, "_orig_mod", model)
        return sum(p.numel() for p in underlying.parameters())
    if hasattr(model, "estimators_"):  # random forest
        return int(sum(getattr(est.tree_, "node_count", 0) for est in model.estimators_))
    if hasattr(model, "coefs_"):
        return int(sum(w.size for w in model.coefs_))
    return 0
