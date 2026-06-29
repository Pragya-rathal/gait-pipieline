
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


@dataclass
class SynergyConfig:
    n_synergies: int = 5
    max_iter: int = 1000
    random_state: int = 42
    threshold: float = 0.90


@dataclass
class ModelConfig:
    hidden_sizes: Sequence[int] = field(default_factory=lambda: (128, 64))
    dropout: float = 0.2
    batch_size: int = 128
    epochs: int = 40
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 8
    random_state: int = 42
    gru_hidden_size: int = 64
    gru_layers: int = 1
    lstm_hidden_size: int = 64
    lstm_layers: int = 1
    tcn_channels: Sequence[int] = field(default_factory=lambda: (32, 32, 32))
    tcn_kernel_size: int = 3
    ae_latent_dim: int = 8
    ae_hidden_sizes: Sequence[int] = field(default_factory=lambda: (64, 32))


@dataclass
class EvalConfig:
    cross_validation: str = "loso"  # "loso", "groupkfold", or "holdout"
    n_splits: int = 5
    test_size: float = 0.2
    val_size: float = 0.2
    random_state: int = 42


@dataclass
class WindowConfig:
    window_ms: tuple[int, ...] = (50, 100, 150, 200)
    forecast_ms: tuple[int, ...] = (50, 100, 200, 300)
    overlap: float = 0.5
    sample_rate_hz: int = 1000
    use_center_label: bool = False


@dataclass
class PipelineConfig:
    data_dir: Path
    output_dir: Path
    normalize: bool = True
    smooth: bool = False
    use_synergies: bool = True
    use_dh: bool = True
    use_d2h: bool = True
    demo: bool = False
    random_state: int = 42
    synergy: SynergyConfig = field(default_factory=SynergyConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    windows: WindowConfig = field(default_factory=WindowConfig)
