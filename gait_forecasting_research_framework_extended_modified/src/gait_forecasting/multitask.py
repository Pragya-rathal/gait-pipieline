from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Task descriptors
# ---------------------------------------------------------------------------

@dataclass
class TaskSpec:
    name: str
    kind: str  # "classification" | "binary" | "regression"
    output_dim: int
    weight: float = 1.0
    enabled: bool = True


# ---------------------------------------------------------------------------
# Individual prediction heads
# ---------------------------------------------------------------------------

class _ClassificationHead(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _BinaryHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class _RegressionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Shared encoder (optional projection on top of ForecastModel output)
# ---------------------------------------------------------------------------

class _SharedEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Default task layout matching the pipeline's physiological intent targets
# ---------------------------------------------------------------------------

def _default_task_specs(n_activity_classes: int, n_transition_types: int) -> List[TaskSpec]:
    return [
        TaskSpec("current_activity",  kind="classification", output_dim=n_activity_classes),
        TaskSpec("future_activity",   kind="classification", output_dim=n_activity_classes),
        TaskSpec("transition_flag",   kind="binary",         output_dim=1),
        TaskSpec("transition_type",   kind="classification", output_dim=n_transition_types),
        TaskSpec("time_to_transition", kind="regression",    output_dim=1),
    ]


# ---------------------------------------------------------------------------
# Multi-task prediction module
# ---------------------------------------------------------------------------

class MultiTaskPredictor(nn.Module):
    """
    Applies independent prediction heads to a shared temporal representation
    produced by ForecastModel.  Does not perform any temporal modelling.

    Parameters
    ----------
    repr_dim : int
        Dimension of the shared representation vector (ForecastModel.output_dim).
    task_specs : Sequence[TaskSpec]
        One descriptor per task.  Tasks can be added, removed, or toggled
        without changing any other task's weights.
    shared_dim : int
        Width of the shared encoder that sits between the forecast representation
        and the individual heads.  Set equal to repr_dim to use a single linear
        projection; set to 0 to bypass the shared encoder entirely.
    head_hidden_dim : int
        Hidden width inside every prediction head.
    dropout : float
        Applied inside the shared encoder and inside every head.
    """

    def __init__(
        self,
        repr_dim: int,
        task_specs: Sequence[TaskSpec],
        shared_dim: int = 128,
        head_hidden_dim: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.repr_dim = repr_dim
        self.shared_dim = shared_dim if shared_dim > 0 else repr_dim
        self.head_hidden_dim = head_hidden_dim
        self.dropout = dropout

        if shared_dim > 0:
            self.shared_encoder: Optional[nn.Module] = _SharedEncoder(repr_dim, shared_dim, dropout)
            head_in = shared_dim
        else:
            self.shared_encoder = None
            head_in = repr_dim

        self._task_specs: Dict[str, TaskSpec] = {}
        self._heads = nn.ModuleDict()

        for spec in task_specs:
            self._register_task(spec, head_in)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_head(self, spec: TaskSpec, head_in: int) -> nn.Module:
        if spec.kind == "classification":
            return _ClassificationHead(head_in, spec.output_dim, self.head_hidden_dim, self.dropout)
        elif spec.kind == "binary":
            return _BinaryHead(head_in, self.head_hidden_dim, self.dropout)
        elif spec.kind == "regression":
            return _RegressionHead(head_in, spec.output_dim, self.head_hidden_dim, self.dropout)
        else:
            raise ValueError(f"Unknown task kind: {spec.kind!r}. Use 'classification', 'binary', or 'regression'.")

    def _register_task(self, spec: TaskSpec, head_in: int) -> None:
        self._task_specs[spec.name] = spec
        self._heads[spec.name] = self._build_head(spec, head_in)

    # ------------------------------------------------------------------
    # Public API for dynamic task management
    # ------------------------------------------------------------------

    def add_task(self, spec: TaskSpec) -> None:
        """Register a new task and initialise its head. Safe to call at any time."""
        head_in = self.shared_dim
        self._register_task(spec, head_in)

    def enable_task(self, name: str) -> None:
        if name not in self._task_specs:
            raise KeyError(f"Task {name!r} not registered.")
        self._task_specs[name].enabled = True

    def disable_task(self, name: str) -> None:
        if name not in self._task_specs:
            raise KeyError(f"Task {name!r} not registered.")
        self._task_specs[name].enabled = False

    def set_task_weight(self, name: str, weight: float) -> None:
        if name not in self._task_specs:
            raise KeyError(f"Task {name!r} not registered.")
        self._task_specs[name].weight = weight

    @property
    def task_names(self) -> List[str]:
        return list(self._task_specs.keys())

    @property
    def enabled_task_names(self) -> List[str]:
        return [n for n, s in self._task_specs.items() if s.enabled]

    def task_spec(self, name: str) -> TaskSpec:
        return self._task_specs[name]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, representation: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        representation : (B, repr_dim) tensor
            Shared temporal representation from ForecastModel.forward().

        Returns
        -------
        predictions : dict[task_name -> tensor]
            Only enabled tasks are included.
            - classification : (B, n_classes) logits
            - binary         : (B,) logits
            - regression     : (B, out_dim) values
        """
        shared = self.shared_encoder(representation) if self.shared_encoder is not None else representation

        predictions: Dict[str, torch.Tensor] = {}
        for name, spec in self._task_specs.items():
            if spec.enabled:
                predictions[name] = self._heads[name](shared)

        return predictions

    # ------------------------------------------------------------------
    # Loss helpers
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        reduction: str = "mean",
    ) -> Dict[str, torch.Tensor]:
        """
        Compute per-task and total weighted loss.

        Parameters
        ----------
        predictions : output of forward()
        targets : dict mapping task name -> ground-truth tensor
        reduction : passed to torch loss functions

        Returns
        -------
        losses : dict with per-task keys plus ``"total"``
        """
        losses: Dict[str, torch.Tensor] = {}
        total = torch.zeros(1, device=next(self.parameters()).device)

        for name, pred in predictions.items():
            if name not in targets:
                continue
            spec = self._task_specs[name]
            tgt = targets[name]

            if spec.kind == "classification":
                loss = F.cross_entropy(pred, tgt.long(), reduction=reduction)
            elif spec.kind == "binary":
                loss = F.binary_cross_entropy_with_logits(pred, tgt.float(), reduction=reduction)
            elif spec.kind == "regression":
                loss = F.mse_loss(pred, tgt.float(), reduction=reduction)
            else:
                raise ValueError(f"Unknown task kind: {spec.kind!r}")

            losses[name] = loss
            total = total + spec.weight * loss

        losses["total"] = total.squeeze()
        return losses

    def predict(
        self,
        representation: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Inference-only forward pass.  Returns decoded predictions rather than
        raw logits: argmax for classification/binary, raw values for regression.

        Parameters
        ----------
        representation : (B, repr_dim) tensor

        Returns
        -------
        dict[task_name -> tensor]
        """
        with torch.no_grad():
            raw = self.forward(representation)

        decoded: Dict[str, Any] = {}
        for name, pred in raw.items():
            spec = self._task_specs[name]
            if spec.kind == "classification":
                decoded[name] = pred.argmax(dim=-1)
            elif spec.kind == "binary":
                decoded[name] = (pred.sigmoid() > 0.5).long()
            else:
                decoded[name] = pred

        return decoded


# ---------------------------------------------------------------------------
# Convenience constructor mirroring the pipeline's default activity targets
# ---------------------------------------------------------------------------

def build_multitask_predictor(
    repr_dim: int,
    n_activity_classes: int,
    n_transition_types: int,
    shared_dim: int = 128,
    head_hidden_dim: int = 64,
    dropout: float = 0.2,
    task_specs: Optional[Sequence[TaskSpec]] = None,
) -> MultiTaskPredictor:
    """
    Build a MultiTaskPredictor with the standard physiological intent heads.

    Parameters
    ----------
    repr_dim : int
        Output dimension of ForecastModel.
    n_activity_classes : int
        Number of gait / activity categories for current and future activity heads.
    n_transition_types : int
        Number of distinct transition types for the transition-type head.
    shared_dim : int
        Width of the shared encoder.  Pass 0 to bypass.
    head_hidden_dim : int
        Hidden width of each prediction head.
    dropout : float
    task_specs : optional override — pass a custom list of TaskSpec objects to
        replace the default five-task layout.
    """
    specs = task_specs if task_specs is not None else _default_task_specs(n_activity_classes, n_transition_types)
    return MultiTaskPredictor(
        repr_dim=repr_dim,
        task_specs=specs,
        shared_dim=shared_dim,
        head_hidden_dim=head_hidden_dim,
        dropout=dropout,
    )
