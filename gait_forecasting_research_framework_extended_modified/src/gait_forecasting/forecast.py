from __future__ import annotations

import contextlib
import math
from typing import Literal, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _amp_enabled() -> bool:
    return torch.cuda.is_available()


@contextlib.contextmanager
def _autocast_ctx():
    if _amp_enabled():
        with torch.cuda.amp.autocast():
            yield
    else:
        yield


BackboneKind = Literal["gru", "bilstm", "tcn", "transformer"]


# ---------------------------------------------------------------------------
# Positional encoding for Transformer
# ---------------------------------------------------------------------------

class _SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))     # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# TCN block
# ---------------------------------------------------------------------------

class _TCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.downsample(x)
        y = self.relu(self.drop(self.conv1(x)[..., : res.shape[-1]]))
        y = self.relu(self.drop(self.conv2(y)[..., : res.shape[-1]]))
        return self.relu(y + res)


# ---------------------------------------------------------------------------
# Backbone implementations
# ---------------------------------------------------------------------------

class _GRUBackbone(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False,
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.norm(out[:, -1])


class _BiLSTMBackbone(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        self.norm = nn.LayerNorm(hidden_size * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.norm(out[:, -1])


class _TCNBackbone(nn.Module):
    def __init__(self, input_dim: int, channels: Sequence[int], kernel_size: int, dropout: float) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        prev = input_dim
        for i, ch in enumerate(channels):
            blocks.append(_TCNBlock(prev, ch, kernel_size=kernel_size, dilation=2 ** i, dropout=dropout))
            prev = ch
        self.tcn = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.norm = nn.LayerNorm(prev)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.tcn(x.transpose(1, 2))              # (B, C, T)
        y = self.pool(y).squeeze(-1)                  # (B, C)
        return self.norm(y)


class _TransformerBackbone(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        max_len: int = 2048,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pe = _SinusoidalPE(d_model, max_len=max_len, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pe(self.input_proj(x))
        out = self.encoder(x)
        return self.norm(out[:, -1])


# ---------------------------------------------------------------------------
# ForecastModel — public interface
# ---------------------------------------------------------------------------

class ForecastModel(nn.Module):
    """
    Temporal Intent Modeling Network.

    Consumes a sequence of latent motor states or physiological fusion vectors
    and outputs a shared temporal representation. Contains no prediction heads.

    Parameters
    ----------
    input_dim : int
        Feature dimension of each timestep (C in input (B, T, C)).
    output_dim : int
        Dimension of the shared temporal representation.
    backbone : BackboneKind
        One of ``"gru"``, ``"bilstm"``, ``"tcn"``, ``"transformer"``.
    hidden_size : int
        Hidden size for GRU / BiLSTM backbones. Also used as d_model for Transformer.
    num_layers : int
        Number of recurrent / transformer layers.
    tcn_channels : Sequence[int]
        Channel progression for the TCN backbone.
    tcn_kernel_size : int
    nhead : int
        Number of attention heads (Transformer only).
    dim_feedforward : int
        FFN width (Transformer only).
    dropout : float
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        backbone: BackboneKind = "gru",
        hidden_size: int = 128,
        num_layers: int = 2,
        tcn_channels: Sequence[int] = (64, 128, 128),
        tcn_kernel_size: int = 3,
        nhead: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone_kind = backbone
        self.input_dim = input_dim
        self.output_dim = output_dim

        if backbone == "gru":
            self.backbone = _GRUBackbone(input_dim, hidden_size, num_layers, dropout)
            backbone_out = hidden_size
        elif backbone == "bilstm":
            self.backbone = _BiLSTMBackbone(input_dim, hidden_size, num_layers, dropout)
            backbone_out = hidden_size * 2
        elif backbone == "tcn":
            self.backbone = _TCNBackbone(input_dim, tcn_channels, tcn_kernel_size, dropout)
            backbone_out = tcn_channels[-1]
        elif backbone == "transformer":
            self.backbone = _TransformerBackbone(
                input_dim, hidden_size, nhead, num_layers, dim_feedforward, dropout
            )
            backbone_out = hidden_size
        else:
            raise ValueError(f"Unknown backbone: {backbone!r}")

        self.output_proj = nn.Sequential(
            nn.Linear(backbone_out, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T, input_dim) tensor

        Returns
        -------
        representation : (B, output_dim) tensor
        """
        return self.output_proj(self.backbone(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for forward — returns shared temporal representation."""
        return self(x)


def build_forecast_model(
    input_dim: int,
    output_dim: int,
    backbone: BackboneKind = "gru",
    hidden_size: int = 128,
    num_layers: int = 2,
    tcn_channels: Sequence[int] = (64, 128, 128),
    tcn_kernel_size: int = 3,
    nhead: int = 4,
    dim_feedforward: int = 256,
    dropout: float = 0.1,
) -> ForecastModel:
    return ForecastModel(
        input_dim=input_dim,
        output_dim=output_dim,
        backbone=backbone,
        hidden_size=hidden_size,
        num_layers=num_layers,
        tcn_channels=tcn_channels,
        tcn_kernel_size=tcn_kernel_size,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    )
