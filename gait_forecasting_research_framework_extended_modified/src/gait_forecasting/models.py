
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


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


def make_sklearn_mlp(hidden_layers: Sequence[int] = (128, 64), random_state: int = 42) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(
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
            )),
        ]
    )


class TorchMLP(nn.Module):
    def __init__(self, input_dim: int, n_classes: int, hidden_sizes=(128, 64), dropout: float = 0.2):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TorchGRUClassifier(nn.Module):
    def __init__(self, input_dim: int, n_classes: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2, bidirectional: bool = False):
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

    def forward(self, x):
        out, _ = self.gru(x)
        feat = out[:, -1]
        return self.head(feat)


class TorchBiLSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, n_classes: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2):
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

    def forward(self, x):
        out, _ = self.lstm(x)
        feat = out[:, -1]
        return self.head(feat)


class _TCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        residual = self.downsample(x)
        y = self.conv1(x)
        y = y[..., : residual.shape[-1]]
        y = self.relu(y)
        y = self.dropout(y)
        y = self.conv2(y)
        y = y[..., : residual.shape[-1]]
        y = self.relu(y)
        y = self.dropout(y)
        return self.relu(y + residual)


class TorchTCNClassifier(nn.Module):
    def __init__(self, input_dim: int, n_classes: int, channels: Sequence[int] = (64, 64, 64), kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        blocks = []
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

    def forward(self, x):
        # x: (B, T, C) -> (B, C, T)
        x = x.transpose(1, 2)
        y = self.tcn(x)
        return self.head(y)


@dataclass
class TorchTrainResult:
    model: nn.Module
    best_val_loss: float


def _sequence_model_input(x: np.ndarray) -> np.ndarray:
    if x.ndim != 3:
        raise ValueError("Sequence models expect input shaped (N, T, C)")
    return x


def _make_loaders(X_train, y_train, X_val, y_val, batch_size: int):
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


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
    device: str | None = None,
) -> TorchTrainResult:
    torch.manual_seed(random_state)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    train_loader, val_loader = _make_loaders(X_train, y_train, X_val, y_val, batch_size=batch_size)

    best_state = None
    best_val = float("inf")
    bad = 0
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
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


def train_torch_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_dim: int,
    n_classes: int,
    hidden_sizes=(128, 64),
    dropout: float = 0.2,
    batch_size: int = 256,
    epochs: int = 40,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 8,
    random_state: int = 42,
    device: str | None = None,
) -> TorchTrainResult:
    model = TorchMLP(input_dim=input_dim, n_classes=n_classes, hidden_sizes=hidden_sizes, dropout=dropout)
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
    device: str | None = None,
    bidirectional: bool = False,
) -> TorchTrainResult:
    model = TorchGRUClassifier(input_dim=input_dim, n_classes=n_classes, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)
    return _train_torch_classifier(
        model, X_train, y_train, X_val, y_val, batch_size=batch_size, epochs=epochs,
        learning_rate=learning_rate, weight_decay=weight_decay, patience=patience, random_state=random_state, device=device
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
    device: str | None = None,
) -> TorchTrainResult:
    model = TorchBiLSTMClassifier(input_dim=input_dim, n_classes=n_classes, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
    return _train_torch_classifier(
        model, X_train, y_train, X_val, y_val, batch_size=batch_size, epochs=epochs,
        learning_rate=learning_rate, weight_decay=weight_decay, patience=patience, random_state=random_state, device=device
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
    device: str | None = None,
) -> TorchTrainResult:
    model = TorchTCNClassifier(input_dim=input_dim, n_classes=n_classes, channels=channels, kernel_size=kernel_size, dropout=dropout)
    return _train_torch_classifier(
        model, X_train, y_train, X_val, y_val, batch_size=batch_size, epochs=epochs,
        learning_rate=learning_rate, weight_decay=weight_decay, patience=patience, random_state=random_state, device=device
    )


def predict_torch(model: nn.Module, X: np.ndarray, device: str | None = None) -> np.ndarray:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(X, dtype=torch.float32, device=device)
        logits = model(xb)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
    return preds


def model_parameter_count(model) -> int:
    if isinstance(model, nn.Module):
        return sum(p.numel() for p in model.parameters())
    if hasattr(model, "estimators_"):  # random forest
        return int(sum(getattr(est.tree_, "node_count", 0) for est in model.estimators_))
    if hasattr(model, "coefs_"):
        return int(sum(w.size for w in model.coefs_))
    return 0
