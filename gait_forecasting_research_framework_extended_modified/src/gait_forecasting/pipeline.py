"""
Orchestration layer for the physiological intent forecasting architecture.

Execution graph
---------------
JSON Dataset
    -> Window Quality Check
    -> NMF
    -> Synergy Dynamics
    -> Physiological Feature Fusion
    -> Latent Motor State Encoder
    -> ForecastModel
    -> MultiTaskPredictor
    -> Loss
    -> Metrics
    -> Checkpoint

Every stage below is a small, independently replaceable callable. None of
the stages reimplement logic that already exists elsewhere in the package;
this module only coordinates calls into config.py, data.py, preprocessing.py,
synergies.py, dynamics.py, physiology.py, latent.py, forecast.py,
multitask.py, metrics.py, checkpoint.py, and benchmark.py.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .config import PipelineConfig
from .data import (
    SubjectDataset,
    group_kfold_splits,
    leave_one_subject_out,
    load_dataset,
    make_forecast_target,
    temporal_train_val_test_split,
)
from .preprocessing import (
    build_horizon_steps,
    build_windowed_dataset,
    condition_emg,
    fit_scaler,
    ms_to_samples,
)
from .synergies import NMFSynergyExtractor, compute_dH, compute_d2H

def _load_dynamics_module():
    """
    Load the synergy-dynamics module. Imported defensively because the
    on-disk filename for this module may contain non-standard characters
    that prevent a normal ``from .dynamics import ...`` statement.
    """
    try:
        return importlib.import_module(".dynamics", package=__package__)
    except ImportError:
        pass

    import importlib as _importlib_inner
    import importlib.util as _importlib_util

    pkg_dir = Path(__file__).resolve().parent
    candidates = [p for p in pkg_dir.glob("dynamics*.py")]
    for candidate in candidates:
        spec = _importlib_util.spec_from_file_location(f"{__package__}.dynamics", candidate)
        if spec is None or spec.loader is None:
            continue
        module = _importlib_util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    raise ImportError("Could not locate the synergy-dynamics module in this package.")


_dynamics_module = _load_dynamics_module()
SynergyDynamics = _dynamics_module.SynergyDynamics

from .physiology import build_physiological_fusion
from .latent import build_latent_state, fit_pca_latent_state, fit_autoencoder_latent_state
from .forecast import build_forecast_model, ForecastModel
from .multitask import build_multitask_predictor, MultiTaskPredictor, TaskSpec
from .metrics import compute_multitask_metrics, aggregate_metrics_across_folds
from .checkpoint import CheckpointManager
from .utils import ensure_dir, save_json, set_seed

try:
    from . import benchmark as benchmark_module
except Exception:
    benchmark_module = None


def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _amp_enabled(cfg: PipelineConfig) -> bool:
    return torch.cuda.is_available() and bool(getattr(cfg, "use_amp", True))


# ===========================================================================
# Stage 0 — JSON Dataset (loading)
# ===========================================================================

def stage_load_dataset(cfg: PipelineConfig) -> List[SubjectDataset]:
    """Load all subject records from cfg.data_dir (JSON/NPZ/tabular formats)."""
    if cfg.demo:
        from .synthetic import generate_demo_dataset
        data_dir = ensure_dir(cfg.data_dir)
        generate_demo_dataset(data_dir, n_subjects=6, random_state=cfg.random_state)
    return load_dataset(cfg.data_dir)


# ===========================================================================
# Stage 1 — Window Quality Check
# ===========================================================================

@dataclass
class WindowQualityReport:
    subject_id: str
    n_samples: int
    n_channels: int
    n_nan: int
    n_inf: int
    flat_channels: int
    accepted: bool
    reason: Optional[str] = None


def stage_window_quality_check(
    subjects: Sequence[SubjectDataset],
    min_samples: int = 32,
    max_nan_frac: float = 0.05,
    flat_std_eps: float = 1e-8,
) -> Tuple[List[SubjectDataset], List[WindowQualityReport]]:
    """
    Reject subjects whose raw signal windows are degenerate before any
    feature extraction runs (too short, NaN/Inf contaminated, or constant
    channels). This is purely a gate; it does not alter accepted data.
    """
    accepted: List[SubjectDataset] = []
    reports: List[WindowQualityReport] = []

    for subj in subjects:
        X = np.asarray(subj.X, dtype=float)
        n_samples, n_channels = X.shape if X.ndim == 2 else (0, 0)
        n_nan = int(np.isnan(X).sum())
        n_inf = int(np.isinf(X).sum())
        flat_channels = int((np.nanstd(X, axis=0) < flat_std_eps).sum()) if n_samples else n_channels

        reason = None
        ok = True
        if n_samples < min_samples:
            ok, reason = False, f"too few samples ({n_samples} < {min_samples})"
        elif n_samples and (n_nan / (n_samples * max(n_channels, 1))) > max_nan_frac:
            ok, reason = False, "excessive NaN fraction"
        elif n_inf > 0:
            ok, reason = False, "contains Inf values"
        elif n_channels and flat_channels == n_channels:
            ok, reason = False, "all channels constant"

        reports.append(WindowQualityReport(
            subject_id=subj.subject_id, n_samples=n_samples, n_channels=n_channels,
            n_nan=n_nan, n_inf=n_inf, flat_channels=flat_channels,
            accepted=ok, reason=reason,
        ))
        if ok:
            accepted.append(subj)

    return accepted, reports


# ===========================================================================
# Stage 2 — NMF (synergy extraction)
# ===========================================================================

@dataclass
class SynergyArtifacts:
    extractor: NMFSynergyExtractor
    raw_scaler: Any
    vaf: float
    H_by_subject: Dict[str, np.ndarray] = field(default_factory=dict)


def stage_fit_nmf(
    train_subjects: Sequence[SubjectDataset],
    all_subjects: Sequence[SubjectDataset],
    cfg: PipelineConfig,
) -> SynergyArtifacts:
    """Fit synergy extraction (NMF) on training data, transform all subjects."""
    raw_train_X = np.vstack([s.X for s in train_subjects if len(s.X) > 0])
    raw_scaler = fit_scaler(raw_train_X) if cfg.normalize else None

    conditioned_train = [
        condition_emg(s.X, smooth=cfg.smooth, scaler=raw_scaler, rectify=False)
        for s in train_subjects
    ]

    extractor = NMFSynergyExtractor(
        n_synergies=cfg.synergy.n_synergies,
        max_iter=cfg.synergy.max_iter,
        random_state=cfg.synergy.random_state,
    )
    fit_result = extractor.fit_transform(np.vstack(conditioned_train))

    H_by_subject: Dict[str, np.ndarray] = {}
    for subj in all_subjects:
        Xc = condition_emg(subj.X, smooth=cfg.smooth, scaler=raw_scaler, rectify=False)
        H_by_subject[subj.subject_id] = extractor.transform(Xc)

    return SynergyArtifacts(
        extractor=extractor, raw_scaler=raw_scaler, vaf=fit_result.vaf, H_by_subject=H_by_subject
    )


# ===========================================================================
# Stage 3 — Synergy Dynamics
# ===========================================================================

def stage_synergy_dynamics(
    synergy: SynergyArtifacts,
    include_cross_synergy: bool = True,
) -> Tuple[Dict[str, np.ndarray], int]:
    """
    Compute a per-subject dynamic feature vector (mean/var/energy/rms/...)
    over each subject's full activation sequence, plus per-timestep dH/d2H
    used downstream by the latent motor-state encoder.
    """
    dyn = SynergyDynamics(include_cross_synergy=include_cross_synergy)
    dynamic_features: Dict[str, np.ndarray] = {}
    for subject_id, H in synergy.H_by_subject.items():
        dynamic_features[subject_id] = dyn.compute(H)
    feature_dim = dyn.feature_dim(synergy.extractor.n_synergies)
    return dynamic_features, feature_dim


# ===========================================================================
# Stage 4 — Physiological Feature Fusion
# ===========================================================================

@dataclass
class FusionArtifacts:
    fusion_module: nn.Module
    fused_by_subject: Dict[str, np.ndarray] = field(default_factory=dict)


def stage_physiological_fusion(
    synergy: SynergyArtifacts,
    dynamic_features: Dict[str, np.ndarray],
    cfg: PipelineConfig,
    output_dim: int = 64,
    method: str = "learnable",
) -> FusionArtifacts:
    """
    Fuse (W, per-timestep H summary, dynamic feature vector) into a single
    physiological feature vector per timestep, via PhysiologicalFusion.
    """
    device = _get_device()
    n_muscles = synergy.extractor.model.components_.shape[1]
    n_synergies = synergy.extractor.n_synergies
    dynamic_dim = next(iter(dynamic_features.values())).shape[0] if dynamic_features else 0

    W_flat = synergy.extractor.model.components_.reshape(-1)

    fused_by_subject: Dict[str, np.ndarray] = {}
    fusion_module: Optional[nn.Module] = None

    for subject_id, H in synergy.H_by_subject.items():
        T = H.shape[0]
        h_summary_dim = H.shape[1]

        if fusion_module is None:
            fusion_module = build_physiological_fusion(
                n_muscles=n_muscles,
                n_synergies=n_synergies,
                h_summary_dim=h_summary_dim,
                dynamic_dim=max(dynamic_dim, 1),
                output_dim=output_dim,
                method=method,
            ).to(device)
            fusion_module.eval()

        dyn_vec = dynamic_features.get(subject_id, np.zeros(max(dynamic_dim, 1)))
        W_t = torch.as_tensor(W_flat, dtype=torch.float32, device=device).unsqueeze(0).expand(T, -1)
        H_t = torch.as_tensor(H, dtype=torch.float32, device=device)
        dyn_t = torch.as_tensor(dyn_vec, dtype=torch.float32, device=device).unsqueeze(0).expand(T, -1)

        with torch.no_grad():
            fused = fusion_module(W_t, H_t, dyn_t)
        fused_by_subject[subject_id] = fused.detach().cpu().numpy()

    assert fusion_module is not None, "No subjects available for fusion stage"
    return FusionArtifacts(fusion_module=fusion_module, fused_by_subject=fused_by_subject)


# ===========================================================================
# Stage 5 — Latent Motor State Encoder
# ===========================================================================

@dataclass
class LatentArtifacts:
    encoder_kind: str
    encoder: Any
    latent_by_subject: Dict[str, np.ndarray] = field(default_factory=dict)


def stage_latent_motor_state(
    fusion: FusionArtifacts,
    train_subject_ids: Sequence[str],
    cfg: PipelineConfig,
    kind: str = "ae",
) -> LatentArtifacts:
    """
    Encode the fused physiological feature stream into a compact latent
    motor-state trajectory using either PCA or an autoencoder.
    """
    train_ids = set(train_subject_ids)
    train_X = np.vstack([
        fused for sid, fused in fusion.fused_by_subject.items() if sid in train_ids
    ])

    latent_dim = min(cfg.model.ae_latent_dim, train_X.shape[1])

    if kind == "pca":
        encoder = fit_pca_latent_state(train_X, n_components=latent_dim, random_state=cfg.random_state)
        transform = encoder.transform
    elif kind == "ae":
        rng = np.random.default_rng(cfg.random_state)
        sample_X = train_X if len(train_X) <= 4000 else train_X[rng.choice(len(train_X), 4000, replace=False)]
        encoder = fit_autoencoder_latent_state(
            sample_X,
            latent_dim=latent_dim,
            hidden_sizes=cfg.model.ae_hidden_sizes,
            epochs=max(3, min(8, cfg.model.epochs)),
            batch_size=min(cfg.model.batch_size, 128),
            learning_rate=cfg.model.learning_rate,
            weight_decay=cfg.model.weight_decay,
            patience=max(2, cfg.model.patience),
            random_state=cfg.random_state,
        )
        transform = encoder.transform
    else:
        raise ValueError(f"Unknown latent encoder kind: {kind!r}")

    latent_by_subject = {sid: transform(fused) for sid, fused in fusion.fused_by_subject.items()}
    return LatentArtifacts(encoder_kind=kind, encoder=encoder, latent_by_subject=latent_by_subject)


# ===========================================================================
# Windowing helper bridging latent trajectories into sequence tensors
# ===========================================================================

def _build_subjects_from_latent(
    subjects: Sequence[SubjectDataset],
    latent: LatentArtifacts,
) -> List[SubjectDataset]:
    out: List[SubjectDataset] = []
    for subj in subjects:
        Z = latent.latent_by_subject[subj.subject_id]
        n = min(len(Z), len(subj.y))
        out.append(SubjectDataset(
            subject_id=subj.subject_id,
            X=Z[:n],
            y=np.asarray(subj.y[:n], dtype=int),
            channel_names=[f"z{i+1}" for i in range(Z.shape[1])],
            cycle_id=subj.cycle_id[:n] if subj.cycle_id is not None else None,
            gait_percent=subj.gait_percent[:n] if subj.gait_percent is not None else None,
            sample_index=subj.sample_index[:n] if subj.sample_index is not None else None,
            source_file=subj.source_file,
            metadata={**subj.metadata, "representation": "latent_motor_state"},
        ))
    return out


# ===========================================================================
# Multitask target construction
# ===========================================================================

def _build_multitask_targets(
    y_current: np.ndarray,
    y_future: np.ndarray,
    n_activity_classes: int,
) -> Dict[str, np.ndarray]:
    """
    Builds all multitask targets. transition_type uses a deterministic
    encoding (0 = no transition, 1..K = current*n_classes+future for the
    K possible class-pairs) so the label space is identical across
    train/val/test splits regardless of which transitions actually occur.
    """
    transition_flag = (y_current != y_future).astype(np.int64)
    pair_id = y_current.astype(np.int64) * n_activity_classes + y_future.astype(np.int64)
    transition_type_idx = np.where(transition_flag == 1, pair_id + 1, 0).astype(np.int64)

    time_to_transition = np.zeros(len(y_current), dtype=np.float32)
    next_change = len(y_current)
    for i in range(len(y_current) - 1, -1, -1):
        if i < len(y_current) - 1 and y_current[i] != y_current[i + 1]:
            next_change = i + 1
        time_to_transition[i] = float(next_change - i)

    return {
        "current_activity": y_current.astype(np.int64),
        "future_activity": y_future.astype(np.int64),
        "transition_flag": transition_flag,
        "transition_type": transition_type_idx,
        "time_to_transition": time_to_transition,
    }


def _n_transition_types(n_activity_classes: int) -> int:
    """Fixed-size transition-type label space: 0 (none) + n_classes^2 pairs."""
    return n_activity_classes * n_activity_classes + 1


# ===========================================================================
# Stage 6/7 — ForecastModel + MultiTaskPredictor (architecture assembly)
# ===========================================================================

class IntentForecastingArchitecture(nn.Module):
    """Thin composite wrapping ForecastModel -> MultiTaskPredictor."""

    def __init__(self, forecast_model: ForecastModel, predictor: MultiTaskPredictor) -> None:
        super().__init__()
        self.forecast_model = forecast_model
        self.predictor = predictor

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        representation = self.forecast_model(x)
        return self.predictor(representation)


def build_architecture(
    input_dim: int,
    n_activity_classes: int,
    n_transition_types: int,
    cfg: PipelineConfig,
    backbone: str = "gru",
    repr_dim: int = 128,
) -> IntentForecastingArchitecture:
    """Assemble ForecastModel + MultiTaskPredictor without reimplementing either."""
    forecast_model = build_forecast_model(
        input_dim=input_dim,
        output_dim=repr_dim,
        backbone=backbone,
        hidden_size=cfg.model.gru_hidden_size if backbone == "gru" else cfg.model.lstm_hidden_size,
        num_layers=cfg.model.gru_layers if backbone == "gru" else cfg.model.lstm_layers,
        tcn_channels=cfg.model.tcn_channels,
        tcn_kernel_size=cfg.model.tcn_kernel_size,
        dropout=cfg.model.dropout,
    )
    predictor = build_multitask_predictor(
        repr_dim=repr_dim,
        n_activity_classes=n_activity_classes,
        n_transition_types=n_transition_types,
        shared_dim=repr_dim,
        head_hidden_dim=max(32, repr_dim // 2),
        dropout=cfg.model.dropout,
    )
    return IntentForecastingArchitecture(forecast_model=forecast_model, predictor=predictor)


# ===========================================================================
# Stage 8 — Loss
# ===========================================================================

def compute_total_loss(
    predictor: MultiTaskPredictor,
    outputs: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    return predictor.compute_loss(outputs, targets)


# ===========================================================================
# Data loader assembly for one fold/window/horizon combination
# ===========================================================================

@dataclass
class FoldTensors:
    X: torch.Tensor
    targets: Dict[str, torch.Tensor]
    n_activity_classes: int
    n_transition_types: int


def _prepare_fold_tensors(
    subjects: Sequence[SubjectDataset],
    window_ms: int,
    horizon_ms: int,
    cfg: PipelineConfig,
    n_activity_classes: int,
) -> FoldTensors:
    window_size = ms_to_samples(window_ms, cfg.windows.sample_rate_hz)
    horizon_steps = build_horizon_steps(horizon_ms, cfg.windows.sample_rate_hz)

    # make_forecast_target fills unreachable trailing positions with -1;
    # truncate each subject so every remaining sample has a valid future label.
    truncated_subjects: List[SubjectDataset] = []
    future_labels: Dict[str, np.ndarray] = {}
    for subj in subjects:
        y_future_full = make_forecast_target(subj.y, horizon_steps)
        valid_len = len(subj.y) - horizon_steps if horizon_steps > 0 else len(subj.y)
        valid_len = max(valid_len, 0)
        if valid_len == 0:
            continue
        truncated_subjects.append(SubjectDataset(
            subject_id=subj.subject_id, X=subj.X[:valid_len], y=subj.y[:valid_len],
            channel_names=subj.channel_names,
            cycle_id=subj.cycle_id[:valid_len] if subj.cycle_id is not None else None,
            gait_percent=subj.gait_percent[:valid_len] if subj.gait_percent is not None else None,
            sample_index=subj.sample_index[:valid_len] if subj.sample_index is not None else None,
            source_file=subj.source_file, metadata=subj.metadata,
        ))
        future_labels[subj.subject_id] = y_future_full[:valid_len]

    win = build_windowed_dataset(
        truncated_subjects, window_size=window_size, horizon_steps=0,
        overlap=cfg.windows.overlap, use_center_label=cfg.windows.use_center_label,
    )

    future_win = build_windowed_dataset(
        [SubjectDataset(
            subject_id=s.subject_id, X=s.X, y=future_labels[s.subject_id],
            channel_names=s.channel_names, cycle_id=s.cycle_id,
            gait_percent=s.gait_percent, sample_index=s.sample_index,
            source_file=s.source_file, metadata=s.metadata,
        ) for s in truncated_subjects],
        window_size=window_size, horizon_steps=0,
        overlap=cfg.windows.overlap, use_center_label=cfg.windows.use_center_label,
    )

    n = min(len(win.y), len(future_win.y))
    y_current = win.y[:n]
    y_future = future_win.y[:n]
    X_seq = win.X_seq[:n]

    targets_np = _build_multitask_targets(y_current, y_future, n_activity_classes)
    n_types = _n_transition_types(n_activity_classes)

    device = _get_device()
    X_t = torch.as_tensor(X_seq, dtype=torch.float32, device=device)
    targets_t = {k: torch.as_tensor(v, device=device) for k, v in targets_np.items()}

    return FoldTensors(X=X_t, targets=targets_t, n_activity_classes=n_activity_classes, n_transition_types=n_types)


# ===========================================================================
# Train / validate / test / infer loops
# ===========================================================================

@dataclass
class EpochResult:
    loss: float
    task_losses: Dict[str, float]


def _iterate_batches(n: int, batch_size: int, shuffle: bool, generator: Optional[torch.Generator] = None):
    idx = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
    for start in range(0, n, batch_size):
        yield idx[start:start + batch_size]


def run_train_epoch(
    architecture: IntentForecastingArchitecture,
    data: FoldTensors,
    optimizer: torch.optim.Optimizer,
    cfg: PipelineConfig,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> EpochResult:
    architecture.train()
    amp_active = _amp_enabled(cfg) and scaler is not None
    n = data.X.shape[0]
    total_loss = 0.0
    n_batches = 0
    task_loss_accum: Dict[str, float] = {}

    for batch_idx in _iterate_batches(n, cfg.model.batch_size, shuffle=True):
        if len(batch_idx) == 0:
            continue
        x_batch = data.X[batch_idx]
        targets_batch = {k: v[batch_idx] for k, v in data.targets.items()}

        optimizer.zero_grad(set_to_none=True)
        if amp_active:
            with torch.cuda.amp.autocast():
                outputs = architecture(x_batch)
                losses = compute_total_loss(architecture.predictor, outputs, targets_batch)
            scaler.scale(losses["total"]).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = architecture(x_batch)
            losses = compute_total_loss(architecture.predictor, outputs, targets_batch)
            losses["total"].backward()
            optimizer.step()

        total_loss += float(losses["total"].detach().cpu())
        for k, v in losses.items():
            if k == "total":
                continue
            task_loss_accum[k] = task_loss_accum.get(k, 0.0) + float(v.detach().cpu())
        n_batches += 1

    n_batches = max(n_batches, 1)
    return EpochResult(
        loss=total_loss / n_batches,
        task_losses={k: v / n_batches for k, v in task_loss_accum.items()},
    )


@torch.no_grad()
def run_eval_epoch(
    architecture: IntentForecastingArchitecture,
    data: FoldTensors,
    cfg: PipelineConfig,
) -> Tuple[EpochResult, Dict[str, np.ndarray]]:
    architecture.eval()
    n = data.X.shape[0]
    total_loss = 0.0
    n_batches = 0
    task_loss_accum: Dict[str, float] = {}
    predictions: Dict[str, List[np.ndarray]] = {}

    amp_active = _amp_enabled(cfg)
    for batch_idx in _iterate_batches(n, cfg.model.batch_size, shuffle=False):
        if len(batch_idx) == 0:
            continue
        x_batch = data.X[batch_idx]
        targets_batch = {k: v[batch_idx] for k, v in data.targets.items()}

        if amp_active:
            with torch.cuda.amp.autocast():
                outputs = architecture(x_batch)
                losses = compute_total_loss(architecture.predictor, outputs, targets_batch)
        else:
            outputs = architecture(x_batch)
            losses = compute_total_loss(architecture.predictor, outputs, targets_batch)

        total_loss += float(losses["total"].detach().cpu())
        for k, v in losses.items():
            if k == "total":
                continue
            task_loss_accum[k] = task_loss_accum.get(k, 0.0) + float(v.detach().cpu())
        n_batches += 1

        decoded = architecture.predictor.predict(architecture.forecast_model(x_batch))
        for task_name, pred in decoded.items():
            predictions.setdefault(task_name, []).append(pred.detach().cpu().numpy())

    n_batches = max(n_batches, 1)
    pred_arrays = {k: np.concatenate(v, axis=0) for k, v in predictions.items()}
    return (
        EpochResult(loss=total_loss / n_batches, task_losses={k: v / n_batches for k, v in task_loss_accum.items()}),
        pred_arrays,
    )


@torch.no_grad()
def run_inference(
    architecture: IntentForecastingArchitecture,
    X: torch.Tensor,
    cfg: PipelineConfig,
) -> Dict[str, np.ndarray]:
    """Pure inference: returns decoded predictions for new (unlabeled) data."""
    architecture.eval()
    device = _get_device()
    X = X.to(device)
    amp_active = _amp_enabled(cfg)
    if amp_active:
        with torch.cuda.amp.autocast():
            decoded = architecture.predictor.predict(architecture.forecast_model(X))
    else:
        decoded = architecture.predictor.predict(architecture.forecast_model(X))
    return {k: v.detach().cpu().numpy() for k, v in decoded.items()}


# ===========================================================================
# Single-fold training + evaluation + checkpointing
# ===========================================================================

@dataclass
class FoldOutcome:
    fold_name: str
    history: List[Dict[str, float]]
    test_metrics: Dict[str, Any]
    checkpoint_dir: Path


def run_fold(
    fold_name: str,
    train_subjects: Sequence[SubjectDataset],
    val_subjects: Sequence[SubjectDataset],
    test_subjects: Sequence[SubjectDataset],
    cfg: PipelineConfig,
    window_ms: int,
    horizon_ms: int,
    n_activity_classes: int,
) -> FoldOutcome:
    """
    Runs the full Window Quality Check -> NMF -> Synergy Dynamics ->
    Physiological Fusion -> Latent Motor State -> ForecastModel ->
    MultiTaskPredictor -> Loss -> Metrics -> Checkpoint graph for one fold.
    """
    device = _get_device()
    all_subjects = list(train_subjects) + list(val_subjects) + list(test_subjects)

    qc_subjects, qc_reports = stage_window_quality_check(all_subjects)
    qc_ids = {s.subject_id for s in qc_subjects}
    train_subjects = [s for s in train_subjects if s.subject_id in qc_ids]
    val_subjects = [s for s in val_subjects if s.subject_id in qc_ids]
    test_subjects = [s for s in test_subjects if s.subject_id in qc_ids]

    synergy = stage_fit_nmf(train_subjects, qc_subjects, cfg)
    dynamic_features, _ = stage_synergy_dynamics(synergy, include_cross_synergy=True)
    fusion = stage_physiological_fusion(synergy, dynamic_features, cfg, output_dim=64, method="learnable")
    train_ids = [s.subject_id for s in train_subjects]
    latent = stage_latent_motor_state(fusion, train_ids, cfg, kind="ae")

    latent_train = _build_subjects_from_latent(train_subjects, latent)
    latent_val = _build_subjects_from_latent(val_subjects, latent)
    latent_test = _build_subjects_from_latent(test_subjects, latent)

    train_data = _prepare_fold_tensors(latent_train, window_ms, horizon_ms, cfg, n_activity_classes)
    n_transition_types = train_data.n_transition_types
    val_data = _prepare_fold_tensors(latent_val, window_ms, horizon_ms, cfg, n_activity_classes)
    test_data = _prepare_fold_tensors(latent_test, window_ms, horizon_ms, cfg, n_activity_classes)

    input_dim = train_data.X.shape[-1]
    architecture = build_architecture(
        input_dim=input_dim,
        n_activity_classes=n_activity_classes,
        n_transition_types=n_transition_types,
        cfg=cfg,
        backbone=getattr(cfg.model, "forecast_backbone", "gru"),
        repr_dim=getattr(cfg.model, "forecast_repr_dim", 128),
    ).to(device)

    if _amp_enabled(cfg) and hasattr(torch, "compile"):
        try:
            architecture = torch.compile(architecture)
        except Exception:
            pass

    optimizer = torch.optim.Adam(
        architecture.parameters(), lr=cfg.model.learning_rate, weight_decay=cfg.model.weight_decay
    )
    scaler = torch.cuda.amp.GradScaler(enabled=_amp_enabled(cfg))

    ckpt_dir = ensure_dir(Path(cfg.output_dir) / fold_name / f"window_{window_ms}ms" / f"horizon_{horizon_ms}ms" / "checkpoints")
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=ckpt_dir,
        metric_name="val_loss",
        mode="min",
        patience=cfg.model.patience,
        config={"window_ms": window_ms, "horizon_ms": horizon_ms, "fold": fold_name},
        random_seed=cfg.random_state,
    )

    history: List[Dict[str, float]] = []
    for epoch in range(cfg.model.epochs):
        train_result = run_train_epoch(architecture, train_data, optimizer, cfg, scaler)
        val_result, val_preds = run_eval_epoch(architecture, val_data, cfg)

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_result.loss,
            "val_loss": val_result.loss,
            **{f"train_{k}": v for k, v in train_result.task_losses.items()},
            **{f"val_{k}": v for k, v in val_result.task_losses.items()},
        }
        history.append(epoch_metrics)

        checkpoint_manager.step(architecture, optimizer, {"val_loss": val_result.loss})
        if checkpoint_manager.should_stop:
            break

    checkpoint_manager.load_best(architecture)
    _, test_preds = run_eval_epoch(architecture, test_data, cfg)

    task_outputs = {
        task: {"y_true": test_data.targets[task].detach().cpu().numpy(), "y_pred": test_preds[task]}
        for task in test_preds
        if task in test_data.targets
    }
    test_metrics = compute_multitask_metrics(task_outputs)

    checkpoint_manager.write_metadata(
        architecture,
        input_shape=(train_data.X.shape[1], train_data.X.shape[2]),
        output_names=list(architecture.predictor.task_names),
        label_mappings={"n_activity_classes": n_activity_classes, "n_transition_types": n_transition_types},
        normalization_method="autoencoder_latent" if latent.encoder_kind == "ae" else "pca_latent",
    )

    return FoldOutcome(fold_name=fold_name, history=history, test_metrics=test_metrics, checkpoint_dir=ckpt_dir)


# ===========================================================================
# Cross-validation orchestration
# ===========================================================================

def _split_train_val_subjects(
    subjects: Sequence[SubjectDataset],
    val_size: float,
    random_state: int,
) -> Tuple[List[SubjectDataset], List[SubjectDataset]]:
    """Subject-level train/val split that works even with very few subjects."""
    ids = sorted({s.subject_id for s in subjects})
    if len(ids) < 2 or val_size <= 0.0:
        return list(subjects), []

    from sklearn.model_selection import train_test_split as _tts
    n_val = max(1, int(round(len(ids) * val_size)))
    n_val = min(n_val, len(ids) - 1)
    train_ids, val_ids = _tts(ids, test_size=n_val, random_state=random_state, shuffle=True)
    train_ids, val_ids = set(train_ids), set(val_ids)
    train = [s for s in subjects if s.subject_id in train_ids]
    val = [s for s in subjects if s.subject_id in val_ids]
    return train, val


def _resolve_folds(
    subjects: Sequence[SubjectDataset], cfg: PipelineConfig
) -> List[Tuple[str, List[SubjectDataset], List[SubjectDataset], List[SubjectDataset]]]:
    cv = cfg.eval.cross_validation.lower()
    resolved: List[Tuple[str, List[SubjectDataset], List[SubjectDataset], List[SubjectDataset]]] = []

    if cv == "loso":
        for held_out, train_subj, test_subj in leave_one_subject_out(subjects):
            train_subj, val_subj = _split_train_val_subjects(
                train_subj, val_size=cfg.eval.val_size, random_state=cfg.eval.random_state
            )
            resolved.append((f"loso_{held_out}", list(train_subj), list(val_subj), list(test_subj)))
    elif cv == "groupkfold":
        for i, (train_subj, test_subj) in enumerate(
            group_kfold_splits(subjects, n_splits=cfg.eval.n_splits)
        ):
            train_subj, val_subj = _split_train_val_subjects(
                train_subj, val_size=cfg.eval.val_size, random_state=cfg.eval.random_state
            )
            resolved.append((f"fold_{i+1}", list(train_subj), list(val_subj), list(test_subj)))
    else:
        train_subj, val_subj, test_subj = temporal_train_val_test_split(
            subjects, test_size=cfg.eval.test_size, val_size=cfg.eval.val_size, random_state=cfg.eval.random_state
        )
        resolved.append(("holdout", list(train_subj), list(val_subj), list(test_subj)))

    return resolved


# ===========================================================================
# Top-level orchestrator
# ===========================================================================

def run_pipeline(cfg: PipelineConfig) -> Dict[str, Any]:
    """
    Coordinates the full physiological intent forecasting graph across all
    requested folds, window sizes, and forecast horizons. Each fold runs:

        JSON Dataset -> Window Quality Check -> NMF -> Synergy Dynamics
        -> Physiological Feature Fusion -> Latent Motor State Encoder
        -> ForecastModel -> MultiTaskPredictor -> Loss -> Metrics -> Checkpoint
    """
    set_seed(cfg.random_state)
    ensure_dir(cfg.output_dir)

    subjects = stage_load_dataset(cfg)
    n_activity_classes = len(sorted(set(np.concatenate([s.y for s in subjects]).astype(int).tolist())))

    folds = _resolve_folds(subjects, cfg)

    all_fold_metrics: List[Dict[str, Any]] = []
    fold_outcomes: Dict[str, Any] = {}

    for fold_name, train_subj, val_subj, test_subj in folds:
        for window_ms in cfg.windows.window_ms:
            for horizon_ms in cfg.windows.forecast_ms:
                outcome = run_fold(
                    fold_name=fold_name,
                    train_subjects=train_subj,
                    val_subjects=val_subj,
                    test_subjects=test_subj,
                    cfg=cfg,
                    window_ms=window_ms,
                    horizon_ms=horizon_ms,
                    n_activity_classes=n_activity_classes,
                )
                key = f"{fold_name}/{window_ms}ms/{horizon_ms}ms"
                fold_outcomes[key] = {
                    "history": outcome.history,
                    "test_metrics": {
                        k: (v if not hasattr(v, "tolist") else v.tolist())
                        for k, v in outcome.test_metrics.items()
                        if k != "_aggregate"
                    },
                    "aggregate": outcome.test_metrics.get("_aggregate", {}),
                    "checkpoint_dir": str(outcome.checkpoint_dir),
                }
                all_fold_metrics.append(outcome.test_metrics)

    aggregate = aggregate_metrics_across_folds(
        [{"_aggregate": m.get("_aggregate", {})} for m in all_fold_metrics]
    ) if all_fold_metrics else {}

    summary = {
        "n_subjects": len(subjects),
        "subjects": [s.subject_id for s in subjects],
        "n_activity_classes": n_activity_classes,
        "cross_validation": cfg.eval.cross_validation,
        "folds": fold_outcomes,
        "aggregate_metrics": aggregate,
    }

    save_json(Path(cfg.output_dir) / "results.json", fold_outcomes)
    save_json(Path(cfg.output_dir) / "final_summary.json", summary)

    return summary


# ===========================================================================
# Benchmarking entry point (delegates to benchmark.py)
# ===========================================================================

def run_benchmark(cfg: PipelineConfig, **benchmark_kwargs: Any) -> Optional[List[Any]]:
    """Run the architecture benchmark suite and write results under cfg.output_dir."""
    if benchmark_module is None:
        return None
    out_dir = ensure_dir(Path(cfg.output_dir) / "benchmark")
    return benchmark_module.run_and_export(out_dir, **benchmark_kwargs)


# ===========================================================================
# Inference-only entry point (no labels required)
# ===========================================================================

def run_inference_pipeline(
    architecture: IntentForecastingArchitecture,
    raw_subject: SubjectDataset,
    synergy: SynergyArtifacts,
    fusion_module: nn.Module,
    latent: LatentArtifacts,
    cfg: PipelineConfig,
    window_ms: int,
) -> Dict[str, np.ndarray]:
    """
    Runs a single unlabeled subject through the full feature graph
    (NMF -> Synergy Dynamics -> Fusion -> Latent -> ForecastModel ->
    MultiTaskPredictor) and returns decoded multitask predictions.
    """
    device = _get_device()
    Xc = condition_emg(raw_subject.X, smooth=cfg.smooth, scaler=synergy.raw_scaler, rectify=False)
    H = synergy.extractor.transform(Xc)

    dyn = SynergyDynamics(include_cross_synergy=True)
    dyn_vec = dyn.compute(H)

    T = H.shape[0]
    W_flat = synergy.extractor.model.components_.reshape(-1)
    W_t = torch.as_tensor(W_flat, dtype=torch.float32, device=device).unsqueeze(0).expand(T, -1)
    H_t = torch.as_tensor(H, dtype=torch.float32, device=device)
    dyn_t = torch.as_tensor(dyn_vec, dtype=torch.float32, device=device).unsqueeze(0).expand(T, -1)

    with torch.no_grad():
        fused = fusion_module(W_t, H_t, dyn_t).detach().cpu().numpy()

    Z = latent.encoder.transform(fused)

    window_size = ms_to_samples(window_ms, cfg.windows.sample_rate_hz)
    if len(Z) < window_size:
        raise ValueError("Subject sequence shorter than window size; cannot run inference.")
    X_seq = np.stack([Z[i:i + window_size] for i in range(len(Z) - window_size + 1)], axis=0)
    X_t = torch.as_tensor(X_seq, dtype=torch.float32, device=device)

    return run_inference(architecture, X_t, cfg)
