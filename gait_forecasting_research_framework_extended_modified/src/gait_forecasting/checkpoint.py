from __future__ import annotations

import json
import random
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class CheckpointState:
    epoch: int
    best_val_metric: float
    model_state_dict: Dict[str, torch.Tensor]
    optimizer_state_dict: Dict[str, Any]
    scheduler_state_dict: Optional[Dict[str, Any]]
    config: Dict[str, Any]
    random_seed: int
    metric_history: List[Dict[str, float]] = field(default_factory=list)


@dataclass
class CheckpointMetadata:
    model_architecture: str
    input_shape: Sequence[int]
    output_names: Sequence[str]
    label_mappings: Dict[str, Any]
    normalization_method: str
    framework_version: str
    creation_timestamp: str
    extra: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _torch_version() -> str:
    return torch.__version__


def _rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state()[1].tolist(),
        "torch_cpu": torch.get_rng_state().tolist(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = [s.tolist() for s in torch.cuda.get_rng_state_all()]
    return state


def _set_rng_state(state: Dict[str, Any]) -> None:
    random.setstate(tuple(state["python_random"]))
    np.random.set_state(("MT19937", np.array(state["numpy_random"], dtype=np.uint32), 624, 0, 0.0))
    torch.set_rng_state(torch.tensor(state["torch_cpu"], dtype=torch.uint8))
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all([torch.tensor(s, dtype=torch.uint8) for s in state["torch_cuda"]])


def _model_architecture_str(model: nn.Module) -> str:
    underlying = getattr(model, "_orig_mod", model)
    return underlying.__class__.__name__


def _param_count(model: nn.Module) -> int:
    underlying = getattr(model, "_orig_mod", model)
    return sum(p.numel() for p in underlying.parameters())


def _safe_write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Main checkpoint manager
# ---------------------------------------------------------------------------

class CheckpointManager:
    """
    Production-grade checkpoint manager that is completely independent of
    model architecture.  Works with any ``nn.Module``.

    Automatically saves:
    - ``best_model.pt``     — checkpoint at best validation metric
    - ``last_model.pt``     — checkpoint at most recent epoch
    - ``history.json``      — per-epoch metric log
    - ``metrics.json``      — best and final summary metrics
    - ``config.json``       — configuration dictionary
    - ``metadata.json``     — model / framework metadata for deployment

    Optionally exports:
    - ``best_model.onnx``   — ONNX export of the best checkpoint

    Parameters
    ----------
    checkpoint_dir : Path
        Directory where all checkpoint files are written.
    metric_name : str
        Name of the scalar metric (key in the ``metrics`` dict passed to
        :meth:`step`) used to decide which checkpoint is "best".
    mode : ``"min"`` | ``"max"``
        Whether lower or higher ``metric_name`` is better.
    save_every : int
        Additionally save a versioned checkpoint every ``save_every`` epochs.
        Set to 0 to disable versioned saves.
    patience : int
        If > 0, :attr:`should_stop` becomes ``True`` after this many epochs
        without improvement (early stopping).
    min_delta : float
        Minimum change in metric to count as an improvement.
    config : dict
        Arbitrary configuration to embed in every checkpoint and ``config.json``.
    random_seed : int
        RNG seed embedded in checkpoints so training can be reproduced.
    """

    BEST_PT = "best_model.pt"
    LAST_PT = "last_model.pt"
    HISTORY_JSON = "history.json"
    METRICS_JSON = "metrics.json"
    CONFIG_JSON = "config.json"
    METADATA_JSON = "metadata.json"
    BEST_ONNX = "best_model.onnx"

    def __init__(
        self,
        checkpoint_dir: Path,
        metric_name: str = "val_loss",
        mode: str = "min",
        save_every: int = 0,
        patience: int = 0,
        min_delta: float = 1e-6,
        config: Optional[Dict[str, Any]] = None,
        random_seed: int = 42,
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.metric_name = metric_name
        self.mode = mode
        self.save_every = save_every
        self.patience = patience
        self.min_delta = min_delta
        self.config: Dict[str, Any] = config or {}
        self.random_seed = random_seed

        self._best_metric: float = float("inf") if mode == "min" else float("-inf")
        self._epochs_without_improvement: int = 0
        self._history: List[Dict[str, float]] = []
        self._epoch: int = 0
        self.should_stop: bool = False

        _safe_write_json(self.checkpoint_dir / self.CONFIG_JSON, self.config)

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def _is_better(self, value: float) -> bool:
        if self.mode == "min":
            return value < self._best_metric - self.min_delta
        return value > self._best_metric + self.min_delta

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def _pack(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        epoch: int,
        metrics: Dict[str, float],
    ) -> Dict[str, Any]:
        underlying = getattr(model, "_orig_mod", model)
        return {
            "epoch": epoch,
            "best_val_metric": float(self._best_metric),
            "metric_name": self.metric_name,
            "model_state_dict": {k: v.cpu() for k, v in underlying.state_dict().items()},
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "config": self.config,
            "random_seed": self.random_seed,
            "rng_state": _rng_state(),
            "metrics": metrics,
            "metric_history": self._history,
        }

    def _save(self, payload: Dict[str, Any], filename: str) -> Path:
        path = self.checkpoint_dir / filename
        tmp = path.with_suffix(".tmp")
        torch.save(payload, tmp)
        tmp.replace(path)
        return path

    # ------------------------------------------------------------------
    # Public step method
    # ------------------------------------------------------------------

    def step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        metrics: Dict[str, float],
        scheduler: Optional[Any] = None,
    ) -> bool:
        """
        Call once per epoch after validation.

        Parameters
        ----------
        model : nn.Module
        optimizer : torch.optim.Optimizer
        metrics : dict
            Must contain ``self.metric_name``.
        scheduler : optional LR scheduler

        Returns
        -------
        improved : bool
            ``True`` if this epoch produced a new best checkpoint.
        """
        self._epoch += 1
        epoch = self._epoch

        metric_value = metrics[self.metric_name]
        record = {"epoch": epoch, **{k: float(v) for k, v in metrics.items()}}
        self._history.append(record)

        payload = self._pack(model, optimizer, scheduler, epoch, metrics)
        self._save(payload, self.LAST_PT)

        improved = self._is_better(metric_value)
        if improved:
            self._best_metric = metric_value
            self._epochs_without_improvement = 0
            self._save(payload, self.BEST_PT)
        else:
            self._epochs_without_improvement += 1
            if self.patience > 0 and self._epochs_without_improvement >= self.patience:
                self.should_stop = True

        if self.save_every > 0 and epoch % self.save_every == 0:
            self._save(payload, f"checkpoint_epoch_{epoch:05d}.pt")

        _safe_write_json(self.checkpoint_dir / self.HISTORY_JSON, self._history)
        _safe_write_json(
            self.checkpoint_dir / self.METRICS_JSON,
            {
                "best_epoch": epoch if improved else self._best_epoch(),
                "best_val_metric": float(self._best_metric),
                "metric_name": self.metric_name,
                "last_epoch": epoch,
                "last_metrics": record,
            },
        )

        return improved

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------

    def resume(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        from_best: bool = False,
        restore_rng: bool = False,
    ) -> int:
        """
        Load a saved checkpoint into ``model``, ``optimizer``, and optionally
        ``scheduler``.

        Parameters
        ----------
        model : nn.Module
        optimizer : torch.optim.Optimizer
        scheduler : optional
        from_best : bool
            Load ``best_model.pt`` instead of ``last_model.pt``.
        restore_rng : bool
            Restore the full RNG state saved in the checkpoint.

        Returns
        -------
        epoch : int
            The epoch at which training was interrupted.
        """
        filename = self.BEST_PT if from_best else self.LAST_PT
        path = self.checkpoint_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        ckpt = torch.load(path, map_location="cpu", weights_only=False)

        underlying = getattr(model, "_orig_mod", model)
        underlying.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        self._epoch = ckpt["epoch"]
        self._best_metric = ckpt.get("best_val_metric", self._best_metric)
        self._history = ckpt.get("metric_history", [])

        if restore_rng and "rng_state" in ckpt:
            _set_rng_state(ckpt["rng_state"])

        return self._epoch

    # ------------------------------------------------------------------
    # Best checkpoint selection
    # ------------------------------------------------------------------

    def best_checkpoint_path(self) -> Path:
        path = self.checkpoint_dir / self.BEST_PT
        if not path.exists():
            raise FileNotFoundError(f"Best checkpoint not found: {path}")
        return path

    def load_best(self, model: nn.Module) -> None:
        """Load only model weights from the best checkpoint (inference use)."""
        ckpt = torch.load(self.best_checkpoint_path(), map_location="cpu", weights_only=False)
        underlying = getattr(model, "_orig_mod", model)
        underlying.load_state_dict(ckpt["model_state_dict"])

    # ------------------------------------------------------------------
    # ONNX export
    # ------------------------------------------------------------------

    def export_onnx(
        self,
        model: nn.Module,
        dummy_input: torch.Tensor,
        input_names: Optional[Sequence[str]] = None,
        output_names: Optional[Sequence[str]] = None,
        dynamic_axes: Optional[Dict[str, Any]] = None,
        opset_version: int = 17,
    ) -> Path:
        """
        Export the best checkpoint to ONNX.

        Parameters
        ----------
        model : nn.Module
            Architecture instance (weights will be loaded from best_model.pt).
        dummy_input : torch.Tensor
            Representative input tensor (batch of 1 recommended).
        input_names : list[str]
        output_names : list[str]
        dynamic_axes : dict
        opset_version : int

        Returns
        -------
        Path to the exported ONNX file.
        """
        self.load_best(model)
        underlying = getattr(model, "_orig_mod", model)
        underlying.eval()

        onnx_path = self.checkpoint_dir / self.BEST_ONNX
        dynamic_axes = dynamic_axes or {"input": {0: "batch_size"}, "output": {0: "batch_size"}}
        input_names = list(input_names or ["input"])
        output_names = list(output_names or ["output"])

        with torch.no_grad():
            torch.onnx.export(
                underlying,
                dummy_input.cpu(),
                str(onnx_path),
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=opset_version,
                do_constant_folding=True,
            )

        return onnx_path

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def write_metadata(
        self,
        model: nn.Module,
        input_shape: Sequence[int],
        output_names: Sequence[str],
        label_mappings: Optional[Dict[str, Any]] = None,
        normalization_method: str = "standard",
        extra: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Write ``metadata.json`` describing the model for deployment.

        Parameters
        ----------
        model : nn.Module
        input_shape : tuple[int, ...]
            Shape of a single input sample (excluding batch dimension).
        output_names : list[str]
            Names of model outputs (e.g. task names for multi-task models).
        label_mappings : dict
            Maps output names to integer→class-name dicts.
        normalization_method : str
            E.g. ``"standard"``, ``"minmax"``, or ``"none"``.
        extra : dict
            Any additional deployment metadata.
        """
        meta = CheckpointMetadata(
            model_architecture=_model_architecture_str(model),
            input_shape=list(input_shape),
            output_names=list(output_names),
            label_mappings=label_mappings or {},
            normalization_method=normalization_method,
            framework_version=f"torch=={_torch_version()}",
            creation_timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            extra={
                "param_count": _param_count(model),
                "best_val_metric": float(self._best_metric),
                "metric_name": self.metric_name,
                **(extra or {}),
            },
        )
        path = self.checkpoint_dir / self.METADATA_JSON
        _safe_write_json(path, asdict(meta))
        return path

    # ------------------------------------------------------------------
    # Versioning helpers
    # ------------------------------------------------------------------

    def list_versioned_checkpoints(self) -> List[Path]:
        """Return all epoch-versioned checkpoint files, sorted by epoch."""
        return sorted(
            self.checkpoint_dir.glob("checkpoint_epoch_*.pt"),
            key=lambda p: int(p.stem.split("_")[-1]),
        )

    def prune_versioned_checkpoints(self, keep: int = 5) -> None:
        """Keep only the ``keep`` most recent versioned checkpoints."""
        checkpoints = self.list_versioned_checkpoints()
        for path in checkpoints[: max(0, len(checkpoints) - keep)]:
            path.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _best_epoch(self) -> int:
        path = self.checkpoint_dir / self.BEST_PT
        if not path.exists():
            return 0
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            return int(ckpt.get("epoch", 0))
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def best_metric(self) -> float:
        return self._best_metric

    @property
    def current_epoch(self) -> int:
        return self._epoch

    @property
    def epochs_without_improvement(self) -> int:
        return self._epochs_without_improvement

    @property
    def history(self) -> List[Dict[str, float]]:
        return list(self._history)
