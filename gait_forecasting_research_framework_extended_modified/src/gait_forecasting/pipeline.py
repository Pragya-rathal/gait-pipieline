from __future__ import annotations

import importlib
import json
import platform
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import joblib
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
from .synergies import (
    NMFSynergyExtractor,
    compute_dH,
    compute_d2H,
)


def _load_dynamics_module():
    """
    Load the synergy-dynamics module.

    Imported defensively because the on-disk filename for this module may
    contain non-standard characters that prevent a normal
    `from .dynamics import ...` statement.
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
        spec = _importlib_util.spec_from_file_location(
            f"{__package__}.dynamics",
            candidate,
        )
        if spec is None or spec.loader is None:
            continue

        module = _importlib_util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    raise ImportError(
        "Could not locate the synergy-dynamics module in this package."
    )


_dynamics_module = _load_dynamics_module()
SynergyDynamics = _dynamics_module.SynergyDynamics

from .physiology import build_physiological_fusion
from .latent import (
    build_latent_state,
    fit_pca_latent_state,
    fit_autoencoder_latent_state,
)
from .forecast import (
    build_forecast_model,
    ForecastModel,
)
from .multitask import (
    build_multitask_predictor,
    MultiTaskPredictor,
    TaskSpec,
)
from .metrics import (
    compute_multitask_metrics,
    aggregate_metrics_across_folds,
)
from .checkpoint import CheckpointManager
from .utils import (
    ensure_dir,
    save_json,
    set_seed,
)

try:
    from . import benchmark as benchmark_module
except Exception:
    benchmark_module = None


def _get_device() -> torch.device:
    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def _amp_enabled(cfg: PipelineConfig) -> bool:
    return (
        torch.cuda.is_available()
        and bool(getattr(cfg, "use_amp", False))
    )

# ============================================================================
# Stage 0 — JSON Dataset (loading)
# ============================================================================

def stage_load_dataset(cfg: PipelineConfig) -> List[SubjectDataset]:
    """
    Load all subject records from cfg.data_dir
    (JSON/NPZ/tabular formats).
    """
    if cfg.demo:
        from .synthetic import generate_demo_dataset

        data_dir = ensure_dir(cfg.data_dir)
        generate_demo_dataset(
            data_dir,
            n_subjects=6,
            random_state=cfg.random_state,
        )

    return load_dataset(cfg.data_dir)


# ============================================================================
# Stage 1 — Window Quality Check
# ============================================================================


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

        if X.ndim == 2:
            n_samples, n_channels = X.shape
        else:
            n_samples, n_channels = 0, 0

        n_nan = int(np.isnan(X).sum())
        n_inf = int(np.isinf(X).sum())

        if n_samples:
            flat_channels = int(
                (np.nanstd(X, axis=0) < flat_std_eps).sum()
            )
        else:
            flat_channels = n_channels

        reason = None
        ok = True

        if n_samples < min_samples:
            ok = False
            reason = f"too few samples ({n_samples} < {min_samples})"

        elif (
            n_samples
            and (n_nan / (n_samples * max(n_channels, 1))) > max_nan_frac
        ):
            ok = False
            reason = "excessive NaN fraction"

        elif n_inf > 0:
            ok = False
            reason = "contains Inf values"

        elif n_channels and flat_channels == n_channels:
            ok = False
            reason = "all channels constant"

        reports.append(
            WindowQualityReport(
                subject_id=subj.subject_id,
                n_samples=n_samples,
                n_channels=n_channels,
                n_nan=n_nan,
                n_inf=n_inf,
                flat_channels=flat_channels,
                accepted=ok,
                reason=reason,
            )
        )

        if ok:
            accepted.append(subj)

    return accepted, reports

# ============================================================================
# Stage 2 — NMF (Synergy Extraction)
# ============================================================================


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
    """
    Fit synergy extraction (NMF) on training data and transform all
    subjects into their synergy activation matrices.
    """
    raw_train_X = np.vstack(
        [s.X for s in train_subjects if len(s.X) > 0]
    )

    raw_scaler = (
        fit_scaler(raw_train_X)
        if cfg.normalize
        else None
    )

    conditioned_train = [
        condition_emg(
            s.X,
            smooth=cfg.smooth,
            scaler=raw_scaler,
            rectify=False,
        )
        for s in train_subjects
    ]

    extractor = NMFSynergyExtractor(
        n_synergies=cfg.synergy.n_synergies,
        max_iter=cfg.synergy.max_iter,
        random_state=cfg.synergy.random_state,
    )

    fit_result = extractor.fit_transform(
        np.vstack(conditioned_train)
    )

    H_by_subject: Dict[str, np.ndarray] = {}

    for subj in all_subjects:
        Xc = condition_emg(
            subj.X,
            smooth=cfg.smooth,
            scaler=raw_scaler,
            rectify=False,
        )

        H_by_subject[subj.subject_id] = extractor.transform(Xc)

    return SynergyArtifacts(
        extractor=extractor,
        raw_scaler=raw_scaler,
        vaf=fit_result.vaf,
        H_by_subject=H_by_subject,
    )

# ============================================================================
# Stage 3 — Synergy Dynamics
# ============================================================================


def stage_synergy_dynamics(
    synergy: SynergyArtifacts,
    include_cross_synergy: bool = True,
) -> Tuple[Dict[str, np.ndarray], int]:
    """
    Compute a per-subject dynamic feature vector
    (mean/var/energy/rms/...) over each subject's full activation
    sequence, plus per-timestep dH/d2H used downstream by the latent
    motor-state encoder.
    """
    dyn = SynergyDynamics(
        include_cross_synergy=include_cross_synergy
    )

    dynamic_features: Dict[str, np.ndarray] = {}

    for subject_id, H in synergy.H_by_subject.items():
        dynamic_features[subject_id] = dyn.compute(H)

    feature_dim = dyn.feature_dim(
        synergy.extractor.n_synergies
    )

    return dynamic_features, feature_dim


# ============================================================================
# Stage 4 — Physiological Feature Fusion
# ============================================================================


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
    Fuse (W, per-timestep H summary, dynamic feature vector) into a
    single physiological feature vector per timestep using the
    PhysiologicalFusion module.
    """
    device = _get_device()

    n_muscles = synergy.extractor.model.components_.shape[1]
    n_synergies = synergy.extractor.n_synergies

    dynamic_dim = (
        next(iter(dynamic_features.values())).shape[0]
        if dynamic_features
        else 0
    )

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

        dyn_vec = dynamic_features.get(
            subject_id,
            np.zeros(max(dynamic_dim, 1)),
        )

        W_t = (
            torch.as_tensor(
                W_flat,
                dtype=torch.float32,
                device=device,
            )
            .unsqueeze(0)
            .expand(T, -1)
        )

        H_t = torch.as_tensor(
            H,
            dtype=torch.float32,
            device=device,
        )

        dyn_t = (
            torch.as_tensor(
                dyn_vec,
                dtype=torch.float32,
                device=device,
            )
            .unsqueeze(0)
            .expand(T, -1)
        )

        with torch.no_grad():
            fused = fusion_module(W_t, H_t, dyn_t)

        fused_by_subject[subject_id] = (
            fused.detach()
            .cpu()
            .numpy()
        )

    assert (
        fusion_module is not None
    ), "No subjects available for fusion stage"

    return FusionArtifacts(
        fusion_module=fusion_module,
        fused_by_subject=fused_by_subject,
    )
# ============================================================================
# Stage 5 — Latent Motor State Encoder
# ============================================================================


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

    train_X = np.vstack(
        [
            fused
            for sid, fused in fusion.fused_by_subject.items()
            if sid in train_ids
        ]
    )

    latent_dim = min(
        cfg.model.ae_latent_dim,
        train_X.shape[1],
    )

    if kind == "pca":
        encoder = fit_pca_latent_state(
            train_X,
            n_components=latent_dim,
            random_state=cfg.random_state,
        )
        transform = encoder.transform

    elif kind == "ae":
        rng = np.random.default_rng(cfg.random_state)

        sample_X = (
            train_X
            if len(train_X) <= 4000
            else train_X[
                rng.choice(
                    len(train_X),
                    4000,
                    replace=False,
                )
            ]
        )

        encoder = fit_autoencoder_latent_state(
            sample_X,
            latent_dim=latent_dim,
            hidden_sizes=cfg.model.ae_hidden_sizes,
            epochs=max(
                3,
                min(8, cfg.model.epochs),
            ),
            batch_size=min(
                cfg.model.batch_size,
                128,
            ),
            learning_rate=cfg.model.learning_rate,
            weight_decay=cfg.model.weight_decay,
            patience=max(
                2,
                cfg.model.patience,
            ),
            random_state=cfg.random_state,
        )

        transform = encoder.transform

    else:
        raise ValueError(
            f"Unknown latent encoder kind: {kind!r}"
        )

    latent_by_subject = {
        sid: transform(fused)
        for sid, fused in fusion.fused_by_subject.items()
    }

    return LatentArtifacts(
        encoder_kind=kind,
        encoder=encoder,
        latent_by_subject=latent_by_subject,
    )


# ============================================================================
# Windowing helper bridging latent trajectories into sequence tensors
# ============================================================================


def _build_subjects_from_latent(
    subjects: Sequence[SubjectDataset],
    latent: LatentArtifacts,
) -> List[SubjectDataset]:
    out: List[SubjectDataset] = []

    for subj in subjects:
        Z = latent.latent_by_subject[subj.subject_id]
        n = min(len(Z), len(subj.y))

        out.append(
            SubjectDataset(
                subject_id=subj.subject_id,
                X=Z[:n],
                y=np.asarray(subj.y[:n], dtype=int),
                channel_names=[
                    f"z{i + 1}"
                    for i in range(Z.shape[1])
                ],
                cycle_id=(
                    subj.cycle_id[:n]
                    if subj.cycle_id is not None
                    else None
                ),
                gait_percent=(
                    subj.gait_percent[:n]
                    if subj.gait_percent is not None
                    else None
                ),
                sample_index=(
                    subj.sample_index[:n]
                    if subj.sample_index is not None
                    else None
                ),
                source_file=subj.source_file,
                metadata={
                    **subj.metadata,
                    "representation": "latent_motor_state",
                },
            )
        )

    return out


# ============================================================================
# Multitask target construction
# ============================================================================


def _build_multitask_targets(
    y_current: np.ndarray,
    y_future: np.ndarray,
    n_activity_classes: int,
) -> Dict[str, np.ndarray]:
    """
    Builds all multitask targets.

    transition_type uses a deterministic encoding
    (0 = no transition, 1..K = current*n_classes + future)
    so the label space is identical across train/val/test splits
    regardless of which transitions actually occur.
    """
    transition_flag = (
        y_current != y_future
    ).astype(np.int64)

    pair_id = (
        y_current.astype(np.int64)
        * n_activity_classes
        + y_future.astype(np.int64)
    )

    transition_type_idx = np.where(
        transition_flag == 1,
        pair_id + 1,
        0,
    ).astype(np.int64)

    time_to_transition = np.zeros(
        len(y_current),
        dtype=np.float32,
    )

    next_change = len(y_current)

    for i in range(len(y_current) - 1, -1, -1):
        if (
            i < len(y_current) - 1
            and y_current[i] != y_current[i + 1]
        ):
            next_change = i + 1

        time_to_transition[i] = float(
            next_change - i
        )

    return {
        "current_activity": y_current.astype(np.int64),
        "future_activity": y_future.astype(np.int64),
        "transition_flag": transition_flag,
        "transition_type": transition_type_idx,
        "time_to_transition": time_to_transition,
    }


def _n_transition_types(
    n_activity_classes: int,
) -> int:
    """
    Fixed-size transition-type label space:
    0 (none) + n_classes² pairs.
    """
    return (
        n_activity_classes
        * n_activity_classes
        + 1
    )


# ============================================================================
# Stage 6/7 — ForecastModel + MultiTaskPredictor (architecture assembly)
# ============================================================================

# ============================================================================
# Stage 6/7 — ForecastModel + MultiTaskPredictor (architecture assembly)
# ============================================================================


class IntentForecastingArchitecture(nn.Module):
    """
    Thin composite wrapping ForecastModel -> MultiTaskPredictor.
    """

    def __init__(
        self,
        forecast_model: ForecastModel,
        predictor: MultiTaskPredictor,
    ) -> None:
        super().__init__()
        self.forecast_model = forecast_model
        self.predictor = predictor

    def forward(
        self,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
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
    """
    Assemble ForecastModel + MultiTaskPredictor without
    reimplementing either.
    """
    forecast_model = build_forecast_model(
        input_dim=input_dim,
        output_dim=repr_dim,
        backbone=backbone,
        hidden_size=(
            cfg.model.gru_hidden_size
            if backbone == "gru"
            else cfg.model.lstm_hidden_size
        ),
        num_layers=(
            cfg.model.gru_layers
            if backbone == "gru"
            else cfg.model.lstm_layers
        ),
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

    return IntentForecastingArchitecture(
        forecast_model=forecast_model,
        predictor=predictor,
    )


# ============================================================================
# Stage 8 — Loss
# ============================================================================


def compute_total_loss(
    predictor: MultiTaskPredictor,
    outputs: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    return predictor.compute_loss(
        outputs,
        targets,
    )


# ============================================================================
# Data loader assembly for one fold/window/horizon combination
# ============================================================================
# ============================================================================
# Data loader assembly for one fold/window/horizon combination
# ============================================================================


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
    window_size = ms_to_samples(
        window_ms,
        cfg.windows.sample_rate_hz,
    )

    horizon_steps = build_horizon_steps(
        horizon_ms,
        cfg.windows.sample_rate_hz,
    )

    # make_forecast_target fills unreachable trailing positions with -1;
    # truncate each subject so every remaining sample has a valid future label.
    truncated_subjects: List[SubjectDataset] = []
    future_labels: Dict[str, np.ndarray] = {}

    for subj in subjects:
        y_future_full = make_forecast_target(
            subj.y,
            horizon_steps,
        )

        valid_len = (
            len(subj.y) - horizon_steps
            if horizon_steps > 0
            else len(subj.y)
        )

        valid_len = max(valid_len, 0)

        if valid_len == 0:
            continue

        truncated_subjects.append(
            SubjectDataset(
                subject_id=subj.subject_id,
                X=subj.X[:valid_len],
                y=subj.y[:valid_len],
                channel_names=subj.channel_names,
                cycle_id=(
                    subj.cycle_id[:valid_len]
                    if subj.cycle_id is not None
                    else None
                ),
                gait_percent=(
                    subj.gait_percent[:valid_len]
                    if subj.gait_percent is not None
                    else None
                ),
                sample_index=(
                    subj.sample_index[:valid_len]
                    if subj.sample_index is not None
                    else None
                ),
                source_file=subj.source_file,
                metadata=subj.metadata,
            )
        )

        future_labels[subj.subject_id] = (
            y_future_full[:valid_len]
        )

    win = build_windowed_dataset(
        truncated_subjects,
        window_size=window_size,
        horizon_steps=0,
        overlap=cfg.windows.overlap,
        use_center_label=cfg.windows.use_center_label,
    )

    future_win = build_windowed_dataset(
        [
            SubjectDataset(
                subject_id=s.subject_id,
                X=s.X,
                y=future_labels[s.subject_id],
                channel_names=s.channel_names,
                cycle_id=s.cycle_id,
                gait_percent=s.gait_percent,
                sample_index=s.sample_index,
                source_file=s.source_file,
                metadata=s.metadata,
            )
            for s in truncated_subjects
        ],
        window_size=window_size,
        horizon_steps=0,
        overlap=cfg.windows.overlap,
        use_center_label=cfg.windows.use_center_label,
    )

    n = min(
        len(win.y),
        len(future_win.y),
    )

    y_current = win.y[:n]
    y_future = future_win.y[:n]
    X_seq = win.X_seq[:n]

    targets_np = _build_multitask_targets(
        y_current,
        y_future,
        n_activity_classes,
    )

    n_types = _n_transition_types(
        n_activity_classes
    )

    device = _get_device()

    X_t = torch.as_tensor(
        X_seq,
        dtype=torch.float32,
        device=device,
    )

    targets_t = {
        k: torch.as_tensor(
            v,
            device=device,
        )
        for k, v in targets_np.items()
    }

    return FoldTensors(
        X=X_t,
        targets=targets_t,
        n_activity_classes=n_activity_classes,
        n_transition_types=n_types,
    )


# ============================================================================
# Train / validate / test / infer loops
# ============================================================================

# ============================================================================
# Train / validate / test / infer loops
# ============================================================================


@dataclass
class EpochResult:
    loss: float
    task_losses: Dict[str, float]


def _iterate_batches(
    n: int,
    batch_size: int,
    shuffle: bool,
    generator: Optional[torch.Generator] = None,
):
    idx = (
        torch.randperm(n, generator=generator)
        if shuffle
        else torch.arange(n)
    )

    for start in range(0, n, batch_size):
        yield idx[start : start + batch_size]


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

    for batch_idx in _iterate_batches(
        n,
        cfg.model.batch_size,
        shuffle=True,
    ):
        if len(batch_idx) == 0:
            continue

        x_batch = data.X[batch_idx]
        targets_batch = {
            k: v[batch_idx]
            for k, v in data.targets.items()
        }

        optimizer.zero_grad(set_to_none=True)

        if amp_active:
            with torch.cuda.amp.autocast():
                outputs = architecture(x_batch)
                losses = compute_total_loss(
                    architecture.predictor,
                    outputs,
                    targets_batch,
                )

            scaler.scale(losses["total"]).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            outputs = architecture(x_batch)

            losses = compute_total_loss(
                architecture.predictor,
                outputs,
                targets_batch,
            )

            losses["total"].backward()
            optimizer.step()

        total_loss += float(
            losses["total"].detach().cpu()
        )

        for k, v in losses.items():
            if k == "total":
                continue

            task_loss_accum[k] = (
                task_loss_accum.get(k, 0.0)
                + float(v.detach().cpu())
            )

        n_batches += 1

    n_batches = max(n_batches, 1)

    return EpochResult(
        loss=total_loss / n_batches,
        task_losses={
            k: v / n_batches
            for k, v in task_loss_accum.items()
        },
    )
# ============================================================================
# Train / validate / test / infer loops
# ============================================================================


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

    for batch_idx in _iterate_batches(
        n,
        cfg.model.batch_size,
        shuffle=False,
    ):
        if len(batch_idx) == 0:
            continue

        x_batch = data.X[batch_idx]
        targets_batch = {
            k: v[batch_idx]
            for k, v in data.targets.items()
        }

        if amp_active:
            with torch.cuda.amp.autocast():
                outputs = architecture(x_batch)
                losses = compute_total_loss(
                    architecture.predictor,
                    outputs,
                    targets_batch,
                )
        else:
            outputs = architecture(x_batch)
            losses = compute_total_loss(
                architecture.predictor,
                outputs,
                targets_batch,
            )

        total_loss += float(
            losses["total"].detach().cpu()
        )

        for k, v in losses.items():
            if k == "total":
                continue

            task_loss_accum[k] = (
                task_loss_accum.get(k, 0.0)
                + float(v.detach().cpu())
            )

        n_batches += 1

        representation = architecture.forecast_model(x_batch)
        decoded = architecture.predictor.predict(
            representation
        )

        for task_name, pred in decoded.items():
            predictions.setdefault(
                task_name,
                [],
            ).append(
                pred.detach().cpu().numpy()
            )

    n_batches = max(n_batches, 1)

    pred_arrays = {
        k: np.concatenate(v, axis=0)
        for k, v in predictions.items()
    }

    return (
        EpochResult(
            loss=total_loss / n_batches,
            task_losses={
                k: v / n_batches
                for k, v in task_loss_accum.items()
            },
        ),
        pred_arrays,
    )


@torch.no_grad()
def run_inference(
    architecture: IntentForecastingArchitecture,
    X: torch.Tensor,
    cfg: PipelineConfig,
) -> Dict[str, np.ndarray]:
    """
    Pure inference: returns decoded predictions for
    new (unlabeled) data.
    """
    architecture.eval()

    device = _get_device()
    X = X.to(device)

    amp_active = _amp_enabled(cfg)

    if amp_active:
        with torch.cuda.amp.autocast():
            representation = architecture.forecast_model(X)
            decoded = architecture.predictor.predict(
                representation
            )
    else:
        representation = architecture.forecast_model(X)
        decoded = architecture.predictor.predict(
            representation
        )

    return {
        k: v.detach().cpu().numpy()
        for k, v in decoded.items()
    }


# ============================================================================
# Single-fold training + evaluation + checkpointing
# ============================================================================

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
    Runs the full pipeline for one fold:

        Window Quality Check
            ↓
        NMF Synergy Extraction
            ↓
        Synergy Dynamics
            ↓
        Physiological Fusion
            ↓
        Latent Motor State
            ↓
        ForecastModel
            ↓
        MultiTaskPredictor
            ↓
        Loss
            ↓
        Metrics
            ↓
        Checkpoint
    """
    device = _get_device()

    all_subjects = (
        list(train_subjects)
        + list(val_subjects)
        + list(test_subjects)
    )

    # ------------------------------------------------------------------
    # Stage 1: Window Quality Check
    # ------------------------------------------------------------------
    qc_subjects, qc_reports = stage_window_quality_check(
        all_subjects
    )

    qc_ids = {
        s.subject_id
        for s in qc_subjects
    }

    train_subjects = [
        s
        for s in train_subjects
        if s.subject_id in qc_ids
    ]

    val_subjects = [
        s
        for s in val_subjects
        if s.subject_id in qc_ids
    ]

    test_subjects = [
        s
        for s in test_subjects
        if s.subject_id in qc_ids
    ]

    # ------------------------------------------------------------------
    # Stage 2: NMF
    # ------------------------------------------------------------------
    synergy = stage_fit_nmf(
        train_subjects,
        qc_subjects,
        cfg,
    )

    # ------------------------------------------------------------------
    # Stage 3: Synergy Dynamics
    # ------------------------------------------------------------------
    dynamic_features, _ = stage_synergy_dynamics(
        synergy,
        include_cross_synergy=True,
    )

    # ------------------------------------------------------------------
    # Stage 4: Physiological Fusion
    # ------------------------------------------------------------------
    fusion = stage_physiological_fusion(
        synergy,
        dynamic_features,
        cfg,
        output_dim=64,
        method="learnable",
    )

    # ------------------------------------------------------------------
    # Stage 5: Latent Motor State
    # ------------------------------------------------------------------
    train_ids = [
        s.subject_id
        for s in train_subjects
    ]

    latent = stage_latent_motor_state(
        fusion,
        train_ids,
        cfg,
        kind="ae",
    )

    latent_train = _build_subjects_from_latent(
        train_subjects,
        latent,
    )

    latent_val = _build_subjects_from_latent(
        val_subjects,
        latent,
    )

    latent_test = _build_subjects_from_latent(
        test_subjects,
        latent,
    )

    # ------------------------------------------------------------------
    # Window generation
    # ------------------------------------------------------------------
    train_data = _prepare_fold_tensors(
        latent_train,
        window_ms,
        horizon_ms,
        cfg,
        n_activity_classes,
    )

    n_transition_types = (
        train_data.n_transition_types
    )

    val_data = _prepare_fold_tensors(
        latent_val,
        window_ms,
        horizon_ms,
        cfg,
        n_activity_classes,
    )

    test_data = _prepare_fold_tensors(
        latent_test,
        window_ms,
        horizon_ms,
        cfg,
        n_activity_classes,
    )

    # ------------------------------------------------------------------
    # Stage 6/7: ForecastModel + MultiTaskPredictor
    # ------------------------------------------------------------------
    input_dim = train_data.X.shape[-1]

    architecture = build_architecture(
        input_dim=input_dim,
        n_activity_classes=n_activity_classes,
        n_transition_types=n_transition_types,
        cfg=cfg,
        backbone=getattr(
            cfg.model,
            "forecast_backbone",
            "gru",
        ),
        repr_dim=getattr(
            cfg.model,
            "forecast_repr_dim",
            128,
        ),
    ).to(device)

    if (
        _amp_enabled(cfg)
        and hasattr(torch, "compile")
    ):
        try:
            architecture = torch.compile(
                architecture
            )
        except Exception:
            pass

    optimizer = torch.optim.Adam(
        architecture.parameters(),
        lr=cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay,
    )

    scaler = torch.cuda.amp.GradScaler(
        enabled=_amp_enabled(cfg)
    )

    # ------------------------------------------------------------------
    # Checkpoint manager
    # ------------------------------------------------------------------
    ckpt_dir = ensure_dir(
        Path(cfg.output_dir)
        / fold_name
        / f"window_{window_ms}ms"
        / f"horizon_{horizon_ms}ms"
        / "checkpoints"
    )

    checkpoint_manager = CheckpointManager(
        checkpoint_dir=ckpt_dir,
        metric_name="val_loss",
        mode="min",
        patience=cfg.model.patience,
        config={
            "window_ms": window_ms,
            "horizon_ms": horizon_ms,
            "fold": fold_name,
        },
        random_seed=cfg.random_state,
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    history: List[Dict[str, float]] = []

    for epoch in range(cfg.model.epochs):
        train_result = run_train_epoch(
            architecture,
            train_data,
            optimizer,
            cfg,
            scaler,
        )

        val_result, val_preds = run_eval_epoch(
            architecture,
            val_data,
            cfg,
        )

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_result.loss,
            "val_loss": val_result.loss,
            **{
                f"train_{k}": v
                for k, v in train_result.task_losses.items()
            },
            **{
                f"val_{k}": v
                for k, v in val_result.task_losses.items()
            },
        }

        history.append(epoch_metrics)

        checkpoint_manager.step(
            architecture,
            optimizer,
            {"val_loss": val_result.loss},
        )

        if checkpoint_manager.should_stop:
            break

    # ------------------------------------------------------------------
    # Testing
    # ------------------------------------------------------------------
    checkpoint_manager.load_best(
        architecture
    )

    _, test_preds = run_eval_epoch(
        architecture,
        test_data,
        cfg,
    )

    task_outputs = {
        task: {
            "y_true": test_data.targets[task]
            .detach()
            .cpu()
            .numpy(),
            "y_pred": test_preds[task],
        }
        for task in test_preds
        if task in test_data.targets
    }

    test_metrics = compute_multitask_metrics(
        task_outputs
    )

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    checkpoint_manager.write_metadata(
        architecture,
        input_shape=(
            train_data.X.shape[1],
            train_data.X.shape[2],
        ),
        output_names=list(
            architecture.predictor.task_names
        ),
        label_mappings={
            "n_activity_classes": n_activity_classes,
            "n_transition_types": n_transition_types,
        },
        normalization_method=(
            "autoencoder_latent"
            if latent.encoder_kind == "ae"
            else "pca_latent"
        ),
    )

    # Persist the complete preprocessing pipeline.
    save_pipeline_state(
        ckpt_dir,
        synergy=synergy,
        fusion=fusion,
        latent=latent,
        architecture=architecture,
        cfg=cfg,
        n_activity_classes=n_activity_classes,
        n_transition_types=n_transition_types,
    )

    return FoldOutcome(
        fold_name=fold_name,
        history=history,
        test_metrics=test_metrics,
        checkpoint_dir=ckpt_dir,
    )

#===========================================================================

#Preprocessing-pipeline serialization

#===========================================================================

# ============================================================================
# Pipeline serialization
# ============================================================================

PREPROCESSING_JOBLIB = "preprocessing_state.joblib"
PIPELINE_CONFIG_JSON = "pipeline_config.json"


def save_pipeline_state(
    checkpoint_dir: Path,
    synergy: SynergyArtifacts,
    fusion: FusionArtifacts,
    latent: LatentArtifacts,
    architecture: nn.Module,
    cfg: PipelineConfig,
    n_activity_classes: int,
    n_transition_types: int,
) -> Path:
    """
    Serializes every non-architecture preprocessing stage
    (NMF extractor, raw scaler, physiological fusion weights,
    latent encoder, and the PipelineConfig itself) into a
    single joblib bundle plus a companion JSON config, so that
    deployment can reconstruct an identical feature pipeline
    without re-deriving any fitted parameters.
    """
    checkpoint_dir = ensure_dir(Path(checkpoint_dir))

    # torch.compile wraps the original model in _orig_mod
    base_arch = getattr(architecture, "_orig_mod", architecture)

    fusion_module = fusion.fusion_module
    fusion_state = fusion_module.state_dict()

    latent_encoder = latent.encoder

    if latent.encoder_kind == "ae":
        latent_state = {
            "model_state_dict": latent_encoder.model.state_dict(),
            "scaler": latent_encoder.scaler,
            "latent_dim": latent_encoder.latent_dim,
        }
    else:
        latent_state = {
            "pca": latent_encoder.pca,
            "scaler": latent_encoder.scaler,
        }

    bundle = {
        # -------------------------------
        # NMF
        # -------------------------------
        "nmf_model": synergy.extractor.model,
        "nmf_n_synergies": synergy.extractor.n_synergies,
        "nmf_max_iter": synergy.extractor.max_iter,
        "nmf_random_state": synergy.extractor.random_state,
        "raw_scaler": synergy.raw_scaler,
        "vaf": synergy.vaf,

        # -------------------------------
        # Physiological fusion
        # -------------------------------
        "fusion_state_dict": fusion_state,
        "fusion_method": getattr(
            fusion_module,
            "method",
            "learnable",
        ),
        "fusion_w_dim": getattr(
            fusion_module,
            "w_dim",
            None,
        ),
        "fusion_h_dim": getattr(
            fusion_module,
            "h_dim",
            None,
        ),
        "fusion_dynamic_dim": getattr(
            fusion_module,
            "dynamic_dim",
            None,
        ),
        "fusion_output_dim": getattr(
            fusion_module,
            "output_dim",
            None,
        ),

        # -------------------------------
        # Latent encoder
        # -------------------------------
        "latent_encoder_kind": latent.encoder_kind,
        "latent_state": latent_state,

        # -------------------------------
        # Label information
        # -------------------------------
        "n_activity_classes": n_activity_classes,
        "n_transition_types": n_transition_types,

        # -------------------------------
        # Forecast architecture metadata
        # -------------------------------
        "input_dim": base_arch.forecast_model.input_dim,
        "repr_dim": base_arch.forecast_model.output_dim,
        "forecast_backbone": (
            base_arch.forecast_model.backbone_kind
        ),
    }

    joblib.dump(
        bundle,
        checkpoint_dir / PREPROCESSING_JOBLIB,
    )

    save_json(
        checkpoint_dir / PIPELINE_CONFIG_JSON,
        asdict(cfg),
    )

    return checkpoint_dir / PREPROCESSING_JOBLIB


def load_pipeline_state(
    checkpoint_dir: Path,
) -> Dict[str, Any]:
    """
    Loads the preprocessing bundle written by
    save_pipeline_state().
    """
    checkpoint_dir = Path(checkpoint_dir)

    bundle_path = (
        checkpoint_dir / PREPROCESSING_JOBLIB
    )

    if not bundle_path.exists():
        raise FileNotFoundError(
            f"No preprocessing state found at {bundle_path}"
        )

    return joblib.load(bundle_path)
def rebuild_feature_pipeline(
    checkpoint_dir: Path,
    architecture: nn.Module,
    device: Optional[torch.device] = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Reconstructs the exact

        NMF
          ↓
        Synergy Dynamics
          ↓
        Physiological Fusion
          ↓
        Latent Motor State

    feature graph used during training, and returns a single callable

        feature_pipeline(raw_window) -> latent_sequence

    that deployment.py can use without knowing about any
    individual stage.

    ``architecture`` must already be loaded (e.g. via
    CheckpointManager) so that its forecast_model.input_dim matches
    the encoder's latent dimension.
    """
    dev = device or _get_device()

    state = load_pipeline_state(checkpoint_dir)

    # ------------------------------------------------------------------
    # Restore NMF
    # ------------------------------------------------------------------
    nmf_extractor = NMFSynergyExtractor(
        n_synergies=state["nmf_n_synergies"],
        max_iter=state["nmf_max_iter"],
        random_state=state["nmf_random_state"],
    )

    nmf_extractor.model = state["nmf_model"]

    if hasattr(nmf_extractor, "_cache_components"):
        nmf_extractor._cache_components()

    raw_scaler = state["raw_scaler"]

    # ------------------------------------------------------------------
    # Restore physiological fusion module
    # ------------------------------------------------------------------
    fusion_module = build_physiological_fusion(
        n_muscles=nmf_extractor.model.components_.shape[1],
        n_synergies=nmf_extractor.n_synergies,
        h_summary_dim=(
            state["fusion_h_dim"]
            or nmf_extractor.n_synergies
        ),
        dynamic_dim=(
            state["fusion_dynamic_dim"]
            or 1
        ),
        output_dim=(
            state["fusion_output_dim"]
            or 64
        ),
        method=state["fusion_method"],
    )

    fusion_module.load_state_dict(
        state["fusion_state_dict"]
    )

    fusion_module.to(dev)
    fusion_module.eval()

    # ------------------------------------------------------------------
    # Restore latent encoder
    # ------------------------------------------------------------------
    latent_kind = state["latent_encoder_kind"]
    latent_state = state["latent_state"]

    if latent_kind == "ae":
        from .latent import (
            _Autoencoder,
            AutoencoderLatentState,
        )

        scaler = latent_state["scaler"]
        latent_dim = latent_state["latent_dim"]

        n_features = scaler.mean_.shape[0]

        ae_model = _Autoencoder(
            input_dim=n_features,
            latent_dim=latent_dim,
            hidden_sizes=(64, 32),
        )

        ae_model.load_state_dict(
            latent_state["model_state_dict"]
        )

        latent_encoder = AutoencoderLatentState(
            model=ae_model,
            scaler=scaler,
            latent_dim=latent_dim,
            _device=dev,
        )

        latent_transform = latent_encoder.transform

    else:
        from .latent import PCALatentState

        latent_encoder = PCALatentState(
            pca=latent_state["pca"],
            scaler=latent_state["scaler"],
        )

        latent_transform = latent_encoder.transform

    # ------------------------------------------------------------------
    # Restore dynamics
    # ------------------------------------------------------------------
    dyn = SynergyDynamics(
        include_cross_synergy=True
    )

    W_flat = nmf_extractor.model.components_.reshape(-1)

    # ------------------------------------------------------------------
    # Deployment feature pipeline
    # ------------------------------------------------------------------
    def feature_pipeline(
        window: np.ndarray,
    ) -> np.ndarray:
        Xc = condition_emg(
            (
                window.T
                if window.shape[0] != window.shape[-1]
                else window
            ),
            scaler=raw_scaler,
            rectify=False,
        )

        H = nmf_extractor.transform(Xc)

        dyn_vec = dyn.compute(H)

        T = H.shape[0]

        W_t = (
            torch.as_tensor(
                W_flat,
                dtype=torch.float32,
                device=dev,
            )
            .unsqueeze(0)
            .expand(T, -1)
        )

        H_t = torch.as_tensor(
            H,
            dtype=torch.float32,
            device=dev,
        )

        dyn_t = (
            torch.as_tensor(
                dyn_vec,
                dtype=torch.float32,
                device=dev,
            )
            .unsqueeze(0)
            .expand(T, -1)
        )

        with torch.no_grad():
            fused = (
                fusion_module(
                    W_t,
                    H_t,
                    dyn_t,
                )
                .detach()
                .cpu()
                .numpy()
            )

        return latent_transform(fused)

    return feature_pipeline



def _split_train_val_subjects(
    subjects: Sequence[SubjectDataset],
    val_size: float,
    random_state: int,
) -> Tuple[List[SubjectDataset], List[SubjectDataset]]:
    """
    Subject-level train/validation split that works even
    with very few subjects.
    """
    ids = sorted({s.subject_id for s in subjects})

    if len(ids) < 2 or val_size <= 0.0:
        return list(subjects), []

    from sklearn.model_selection import train_test_split as _tts

    n_val = max(
        1,
        int(round(len(ids) * val_size)),
    )
    n_val = min(
        n_val,
        len(ids) - 1,
    )

    train_ids, val_ids = _tts(
        ids,
        test_size=n_val,
        random_state=random_state,
        shuffle=True,
    )

    train_ids = set(train_ids)
    val_ids = set(val_ids)

    train = [
        s
        for s in subjects
        if s.subject_id in train_ids
    ]

    val = [
        s
        for s in subjects
        if s.subject_id in val_ids
    ]

    return train, val


def _resolve_folds(
    subjects: Sequence[SubjectDataset],
    cfg: PipelineConfig,
) -> List[
    Tuple[
        str,
        List[SubjectDataset],
        List[SubjectDataset],
        List[SubjectDataset],
    ]
]:
    cv = cfg.eval.cross_validation.lower()

    resolved: List[
        Tuple[
            str,
            List[SubjectDataset],
            List[SubjectDataset],
            List[SubjectDataset],
        ]
    ] = []

    if cv == "loso":
        for (
            held_out,
            train_subj,
            test_subj,
        ) in leave_one_subject_out(subjects):

            train_subj, val_subj = _split_train_val_subjects(
                train_subj,
                val_size=cfg.eval.val_size,
                random_state=cfg.eval.random_state,
            )

            resolved.append(
                (
                    f"loso_{held_out}",
                    list(train_subj),
                    list(val_subj),
                    list(test_subj),
                )
            )

    elif cv == "groupkfold":
        for i, (
            train_subj,
            test_subj,
        ) in enumerate(
            group_kfold_splits(
                subjects,
                n_splits=cfg.eval.n_splits,
            )
        ):
            train_subj, val_subj = _split_train_val_subjects(
                train_subj,
                val_size=cfg.eval.val_size,
                random_state=cfg.eval.random_state,
            )

            resolved.append(
                (
                    f"fold_{i + 1}",
                    list(train_subj),
                    list(val_subj),
                    list(test_subj),
                )
            )

    else:
        train_subj, val_subj, test_subj = (
            temporal_train_val_test_split(
                subjects,
                test_size=cfg.eval.test_size,
                val_size=cfg.eval.val_size,
                random_state=cfg.eval.random_state,
            )
        )

        resolved.append(
            (
                "holdout",
                list(train_subj),
                list(val_subj),
                list(test_subj),
            )
        )

    return resolved


def run_pipeline(cfg: PipelineConfig) -> Dict[str, Any]:
    """
    Coordinates the full physiological intent forecasting graph across
    all requested folds, window sizes, and forecast horizons.

    Each fold runs:

        JSON Dataset
            ↓
        Window Quality Check
            ↓
        NMF
            ↓
        Synergy Dynamics
            ↓
        Physiological Feature Fusion
            ↓
        Latent Motor State Encoder
            ↓
        ForecastModel
            ↓
        MultiTaskPredictor
            ↓
        Loss
            ↓
        Metrics
            ↓
        Checkpoint
    """
    set_seed(cfg.random_state)
    ensure_dir(cfg.output_dir)

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    subjects = stage_load_dataset(cfg)

    n_activity_classes = len(
        sorted(
            set(
                np.concatenate(
                    [s.y for s in subjects]
                )
                .astype(int)
                .tolist()
            )
        )
    )

    # ------------------------------------------------------------------
    # Resolve cross-validation folds
    # ------------------------------------------------------------------
    folds = _resolve_folds(subjects, cfg)

    all_fold_metrics: List[Dict[str, Any]] = []
    fold_outcomes: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Run every fold × window × horizon combination
    # ------------------------------------------------------------------
    for (
        fold_name,
        train_subj,
        val_subj,
        test_subj,
    ) in folds:

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

                key = (
                    f"{fold_name}/"
                    f"{window_ms}ms/"
                    f"{horizon_ms}ms"
                )

                fold_outcomes[key] = {
                    "history": outcome.history,
                    "test_metrics": {
                        k: (
                            v
                            if not hasattr(v, "tolist")
                            else v.tolist()
                        )
                        for k, v in outcome.test_metrics.items()
                        if k != "_aggregate"
                    },
                    "aggregate": outcome.test_metrics.get(
                        "_aggregate",
                        {},
                    ),
                    "checkpoint_dir": str(
                        outcome.checkpoint_dir
                    ),
                }

                all_fold_metrics.append(
                    outcome.test_metrics
                )

    # ------------------------------------------------------------------
    # Aggregate metrics across folds
    # ------------------------------------------------------------------
    if all_fold_metrics:
        aggregate = aggregate_metrics_across_folds(
            [
                {
                    "_aggregate": m.get(
                        "_aggregate",
                        {},
                    )
                }
                for m in all_fold_metrics
            ]
        )
    else:
        aggregate = {}

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    summary = {
        "n_subjects": len(subjects),
        "subjects": [
            s.subject_id
            for s in subjects
        ],
        "n_activity_classes": n_activity_classes,
        "cross_validation": cfg.eval.cross_validation,
        "folds": fold_outcomes,
        "aggregate_metrics": aggregate,
    }

    save_json(
        Path(cfg.output_dir) / "results.json",
        fold_outcomes,
    )

    save_json(
        Path(cfg.output_dir) / "final_summary.json",
        summary,
    )

    return summary



def run_benchmark(
    cfg: PipelineConfig,
    **benchmark_kwargs: Any,
) -> Optional[List[Any]]:
    """
    Run the architecture benchmark suite and write results under
    cfg.output_dir.
    """
    if benchmark_module is None:
        return None

    out_dir = ensure_dir(
        Path(cfg.output_dir) / "benchmark"
    )

    return benchmark_module.run_and_export(
        out_dir,
        **benchmark_kwargs,
    )


# ============================================================================
# Inference-only entry point (no labels required)
# ============================================================================


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
    Runs a single unlabeled subject through the complete feature graph

        NMF
          ↓
        Synergy Dynamics
          ↓
        Physiological Fusion
          ↓
        Latent Motor State
          ↓
        ForecastModel
          ↓
        MultiTaskPredictor

    and returns decoded multitask predictions.
    """
    device = _get_device()

    # --------------------------------------------------------------
    # EMG conditioning
    # --------------------------------------------------------------
    Xc = condition_emg(
        raw_subject.X,
        smooth=cfg.smooth,
        scaler=synergy.raw_scaler,
        rectify=False,
    )

    # --------------------------------------------------------------
    # NMF
    # --------------------------------------------------------------
    H = synergy.extractor.transform(Xc)

    # --------------------------------------------------------------
    # Synergy dynamics
    # --------------------------------------------------------------
    dyn = SynergyDynamics(
        include_cross_synergy=True
    )

    dyn_vec = dyn.compute(H)

    # --------------------------------------------------------------
    # Physiological fusion
    # --------------------------------------------------------------
    T = H.shape[0]

    W_flat = synergy.extractor.model.components_.reshape(-1)

    W_t = (
        torch.as_tensor(
            W_flat,
            dtype=torch.float32,
            device=device,
        )
        .unsqueeze(0)
        .expand(T, -1)
    )

    H_t = torch.as_tensor(
        H,
        dtype=torch.float32,
        device=device,
    )

    dyn_t = (
        torch.as_tensor(
            dyn_vec,
            dtype=torch.float32,
            device=device,
        )
        .unsqueeze(0)
        .expand(T, -1)
    )

    with torch.no_grad():
        fused = (
            fusion_module(
                W_t,
                H_t,
                dyn_t,
            )
            .detach()
            .cpu()
            .numpy()
        )

    # --------------------------------------------------------------
    # Latent motor state
    # --------------------------------------------------------------
    Z = latent.encoder.transform(fused)

    # --------------------------------------------------------------
    # Windowing
    # --------------------------------------------------------------
    window_size = ms_to_samples(
        window_ms,
        cfg.windows.sample_rate_hz,
    )

    if len(Z) < window_size:
        raise ValueError(
            "Subject sequence shorter than window size; "
            "cannot run inference."
        )

    X_seq = np.stack(
        [
            Z[i : i + window_size]
            for i in range(
                len(Z) - window_size + 1
            )
        ],
        axis=0,
    )

    X_t = torch.as_tensor(
        X_seq,
        dtype=torch.float32,
        device=device,
    )

    # --------------------------------------------------------------
    # Forecasting
    # --------------------------------------------------------------
    return run_inference(
        architecture,
        X_t,
        cfg,
    )

# ============================================================================
# This class is the single sanctioned entry point for evaluate.py and
# deployment.py.
#
# Neither module should construct NMF / PhysiologicalFusion /
# latent encoders / ForecastModel / MultiTaskPredictor directly.
#
# They should instead call:
#
#     train()
#     validate()
#     test()
#     predict()
#     save_checkpoint()
#     load_checkpoint()
#
# on an instance of GaitForecastingPipeline.
# ============================================================================


@dataclass
class GaitForecastingPipeline:
    """
    Object-oriented façade over the functional execution graph above.

    Owns one fold's worth of fitted preprocessing stages plus the
    trained architecture, and exposes a clean

        train()
        validate()
        test()
        predict()
        checkpoint()

    interface so the rest of the repository never has to touch

        • NMF
        • PhysiologicalFusion
        • Latent Motor State Encoder
        • ForecastModel
        • MultiTaskPredictor

    directly.
    """

    cfg: PipelineConfig
    n_activity_classes: int
    architecture: IntentForecastingArchitecture
    synergy: SynergyArtifacts
    fusion: FusionArtifacts
    latent: LatentArtifacts
    n_transition_types: int

    optimizer: torch.optim.Optimizer = field(init=False)
    scaler: torch.cuda.amp.GradScaler = field(init=False)

    checkpoint_manager: Optional[
        CheckpointManager
    ] = None

    device: torch.device = field(
        default_factory=_get_device
    )

    def __post_init__(self) -> None:
        self.architecture = self.architecture.to(
            self.device
        )

        self.optimizer = torch.optim.Adam(
            self.architecture.parameters(),
            lr=self.cfg.model.learning_rate,
            weight_decay=self.cfg.model.weight_decay,
        )

        self.scaler = torch.cuda.amp.GradScaler(
            enabled=_amp_enabled(self.cfg)
        )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def fit_preprocessing(
        cls,
        train_subjects: Sequence[SubjectDataset],
        all_subjects: Sequence[SubjectDataset],
        cfg: PipelineConfig,
        n_activity_classes: int,
        backbone: str = "gru",
        repr_dim: int = 128,
    ) -> "GaitForecastingPipeline":
        """
        Fits

            NMF
              ↓
            Synergy Dynamics
              ↓
            Physiological Fusion
              ↓
            Latent Motor State

        and builds the forecasting architecture.
        """

        synergy = stage_fit_nmf(
            train_subjects,
            all_subjects,
            cfg,
        )

        dynamic_features, _ = stage_synergy_dynamics(
            synergy,
            include_cross_synergy=True,
        )

        fusion = stage_physiological_fusion(
            synergy,
            dynamic_features,
            cfg,
            output_dim=64,
            method="learnable",
        )

        train_ids = [
            s.subject_id
            for s in train_subjects
        ]

        latent = stage_latent_motor_state(
            fusion,
            train_ids,
            cfg,
            kind="ae",
        )

        sample_subj = _build_subjects_from_latent(
            train_subjects[:1],
            latent,
        )

        if sample_subj:
            input_dim = sample_subj[0].X.shape[1]
        else:
            input_dim = cfg.model.ae_latent_dim

        n_transition_types = _n_transition_types(
            n_activity_classes
        )

        architecture = build_architecture(
            input_dim=input_dim,
            n_activity_classes=n_activity_classes,
            n_transition_types=n_transition_types,
            cfg=cfg,
            backbone=backbone,
            repr_dim=repr_dim,
        )

        return cls(
            cfg=cfg,
            n_activity_classes=n_activity_classes,
            architecture=architecture,
            synergy=synergy,
            fusion=fusion,
            latent=latent,
            n_transition_types=n_transition_types,
        )

    # ------------------------------------------------------------------
    # Data preparation through the fitted preprocessing stages
    # ------------------------------------------------------------------

    def prepare_fold_tensors(
        self,
        subjects: Sequence[SubjectDataset],
        window_ms: int,
        horizon_ms: int,
    ) -> FoldTensors:
        latent_subjects = _build_subjects_from_latent(
            subjects,
            self.latent,
        )

        return _prepare_fold_tensors(
            latent_subjects,
            window_ms,
            horizon_ms,
            self.cfg,
            self.n_activity_classes,
        )

    # ------------------------------------------------------------------
    # train / validate / test / predict
    # ------------------------------------------------------------------

    def train(
        self,
        data: FoldTensors,
    ) -> EpochResult:
        """
        Runs a single training epoch over the given
        (already-featurized) data.
        """
        return run_train_epoch(
            self.architecture,
            data,
            self.optimizer,
            self.cfg,
            self.scaler,
        )

    def train_epochs(
        self,
        train_data: FoldTensors,
        val_data: FoldTensors,
        n_epochs: Optional[int] = None,
        checkpoint_dir: Optional[Path] = None,
    ) -> List[Dict[str, float]]:
        """
        Runs the full train/validate loop with early stopping
        and checkpointing.
        """
        n_epochs = (
            n_epochs
            if n_epochs is not None
            else self.cfg.model.epochs
        )

        if (
            checkpoint_dir is not None
            and self.checkpoint_manager is None
        ):
            self.checkpoint_manager = CheckpointManager(
                checkpoint_dir=ensure_dir(
                    Path(checkpoint_dir)
                ),
                metric_name="val_loss",
                mode="min",
                patience=self.cfg.model.patience,
                config={
                    "window_size": train_data.X.shape[1],
                },
                random_seed=self.cfg.random_state,
            )

        history: List[Dict[str, float]] = []

        for epoch in range(n_epochs):
            train_result = self.train(train_data)
            val_result = self.validate(val_data)

            epoch_metrics = {
                "epoch": epoch,
                "train_loss": train_result.loss,
                "val_loss": val_result.loss,
                **{
                    f"train_{k}": v
                    for k, v in train_result.task_losses.items()
                },
                **{
                    f"val_{k}": v
                    for k, v in val_result.task_losses.items()
                },
            }

            history.append(epoch_metrics)

            if self.checkpoint_manager is not None:
                self.checkpoint_manager.step(
                    self.architecture,
                    self.optimizer,
                    {"val_loss": val_result.loss},
                )

                if self.checkpoint_manager.should_stop:
                    break

        if self.checkpoint_manager is not None:
            self.checkpoint_manager.load_best(
                self.architecture
            )

        return history

    def validate(
        self,
        data: FoldTensors,
    ) -> EpochResult:
        """
        Runs one validation pass; returns loss only
        (no decoded predictions).
        """
        result, _ = run_eval_epoch(
            self.architecture,
            data,
            self.cfg,
        )

        return result

    def test(
        self,
        data: FoldTensors,
    ) -> Dict[str, Any]:
        """
        Runs the architecture on held-out data and returns
        aggregated multitask metrics.
        """
        _, test_preds = run_eval_epoch(
            self.architecture,
            data,
            self.cfg,
        )

        task_outputs = {
            task: {
                "y_true": (
                    data.targets[task]
                    .detach()
                    .cpu()
                    .numpy()
                ),
                "y_pred": test_preds[task],
            }
            for task in test_preds
            if task in data.targets
        }

        return compute_multitask_metrics(
            task_outputs
        )

    def predict(
        self,
        raw_window: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Runs a single raw EMG window through the complete feature graph

            NMF
              ↓
            Synergy Dynamics
              ↓
            Physiological Fusion
              ↓
            Latent Motor State
              ↓
            ForecastModel
              ↓
            MultiTaskPredictor

        and returns decoded multitask predictions.

        Accepts either a

            (channels, samples)

        or

            (samples, channels)

        NumPy array.
        """
        device = self.device

        # ----------------------------------------------------------
        # EMG conditioning
        # ----------------------------------------------------------
        Xc = condition_emg(
            raw_window,
            smooth=self.cfg.smooth,
            scaler=self.synergy.raw_scaler,
            rectify=False,
        )

        # ----------------------------------------------------------
        # NMF
        # ----------------------------------------------------------
        H = self.synergy.extractor.transform(
            Xc
        )

        # ----------------------------------------------------------
        # Synergy dynamics
        # ----------------------------------------------------------
        dyn = SynergyDynamics(
            include_cross_synergy=True
        )

        dyn_vec = dyn.compute(H)

        # ----------------------------------------------------------
        # Physiological fusion
        # ----------------------------------------------------------
        T = H.shape[0]

        W_flat = (
            self.synergy.extractor.model
            .components_
            .reshape(-1)
        )

        W_t = (
            torch.as_tensor(
                W_flat,
                dtype=torch.float32,
                device=device,
            )
            .unsqueeze(0)
            .expand(T, -1)
        )

        H_t = torch.as_tensor(
            H,
            dtype=torch.float32,
            device=device,
        )

        dyn_t = (
            torch.as_tensor(
                dyn_vec,
                dtype=torch.float32,
                device=device,
            )
            .unsqueeze(0)
            .expand(T, -1)
        )

        with torch.no_grad():
            fused = (
                self.fusion.fusion_module(
                    W_t,
                    H_t,
                    dyn_t,
                )
                .detach()
                .cpu()
                .numpy()
            )

        # ----------------------------------------------------------
        # Latent encoding
        # ----------------------------------------------------------
        Z = self.latent.encoder.transform(
            fused
        )

        # ----------------------------------------------------------
        # Windowing
        # ----------------------------------------------------------
        window_size = ms_to_samples(
            self.cfg.windows.window_ms[0],
            self.cfg.windows.sample_rate_hz,
        )

        if len(Z) < window_size:
            raise ValueError(
                "Subject sequence shorter than window size; "
                "cannot run inference."
            )

        X_seq = np.stack(
            [
                Z[i : i + window_size]
                for i in range(
                    len(Z) - window_size + 1
                )
            ],
            axis=0,
        )

        X_t = torch.as_tensor(
            X_seq,
            dtype=torch.float32,
            device=device,
        )

        return run_inference(
            self.architecture,
            X_t,
            self.cfg,
        )

    # ------------------------------------------------------------------
    # Checkpointing — full pipeline (architecture + preprocessing)
    # ------------------------------------------------------------------

    def save_checkpoint(
        self,
        checkpoint_dir: Path,
        metrics: Optional[Dict[str, float]] = None,
    ) -> Path:
        """
        Saves the trained architecture (via CheckpointManager) and the
        fitted preprocessing stages (NMF, Physiological Fusion, latent
        encoder, normalization, and configuration) so that deployment
        can reconstruct an identical feature pipeline from this single
        directory.
        """
        checkpoint_dir = ensure_dir(
            Path(checkpoint_dir)
        )

        if self.checkpoint_manager is None:
            self.checkpoint_manager = CheckpointManager(
                checkpoint_dir=checkpoint_dir,
                metric_name="val_loss",
                mode="min",
                patience=self.cfg.model.patience,
                config={},
                random_seed=self.cfg.random_state,
            )

        if metrics is not None:
            self.checkpoint_manager.step(
                self.architecture,
                self.optimizer,
                metrics,
            )
        else:
            self.checkpoint_manager.save_last(
                self.architecture,
                self.optimizer,
            )

        save_pipeline_state(
            checkpoint_dir,
            synergy=self.synergy,
            fusion=self.fusion,
            latent=self.latent,
            architecture=self.architecture,
            cfg=self.cfg,
            n_activity_classes=self.n_activity_classes,
            n_transition_types=self.n_transition_types,
        )

        return checkpoint_dir

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_dir: Path,
        cfg: Optional[PipelineConfig] = None,
        backbone: str = "gru",
        repr_dim: int = 128,
        device: Optional[torch.device] = None,
        from_best: bool = True,
    ) -> "GaitForecastingPipeline":
        """
        Reconstructs a complete GaitForecastingPipeline
        (architecture + preprocessing stages) from a checkpoint
        directory produced by ``save_checkpoint()``.

        The returned instance is ready for

            validate()
            test()
            predict()

        without re-fitting anything.
        """
        checkpoint_dir = Path(checkpoint_dir)
        dev = device or _get_device()

        state = load_pipeline_state(checkpoint_dir)

        resolved_cfg = cfg

        if resolved_cfg is None:
            config_path = checkpoint_dir / PIPELINE_CONFIG_JSON

            if config_path.exists():
                raw_cfg = json.loads(
                    config_path.read_text()
                )
                resolved_cfg = _pipeline_config_from_dict(
                    raw_cfg
                )
            else:
                raise ValueError(
                    "PipelineConfig not provided and no "
                    "pipeline_config.json found."
                )

        n_activity_classes = state["n_activity_classes"]
        n_transition_types = state["n_transition_types"]
        input_dim = state["input_dim"]

        arch_repr_dim = state.get(
            "repr_dim",
            repr_dim,
        )

        arch_backbone = state.get(
            "forecast_backbone",
            backbone,
        )

        architecture = build_architecture(
            input_dim=input_dim,
            n_activity_classes=n_activity_classes,
            n_transition_types=n_transition_types,
            cfg=resolved_cfg,
            backbone=arch_backbone,
            repr_dim=arch_repr_dim,
        ).to(dev)

        manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir
        )

        if from_best:
            manager.load_best(architecture)
        else:
            manager.resume(
                architecture,
                torch.optim.Adam(
                    architecture.parameters(),
                    lr=1e-3,
                ),
            )

        # ----------------------------------------------------------
        # Restore NMF
        # ----------------------------------------------------------
        nmf_extractor = NMFSynergyExtractor(
            n_synergies=state["nmf_n_synergies"],
            max_iter=state["nmf_max_iter"],
            random_state=state["nmf_random_state"],
        )

        nmf_extractor.model = state["nmf_model"]

        if hasattr(
            nmf_extractor,
            "_cache_components",
        ):
            nmf_extractor._cache_components()

        synergy = SynergyArtifacts(
            extractor=nmf_extractor,
            raw_scaler=state["raw_scaler"],
            vaf=state["vaf"],
            H_by_subject={},
        )

        # ----------------------------------------------------------
        # Restore physiological fusion
        # ----------------------------------------------------------
        fusion_module = build_physiological_fusion(
            n_muscles=(
                nmf_extractor.model.components_.shape[1]
            ),
            n_synergies=nmf_extractor.n_synergies,
            h_summary_dim=(
                state["fusion_h_dim"]
                or nmf_extractor.n_synergies
            ),
            dynamic_dim=(
                state["fusion_dynamic_dim"]
                or 1
            ),
            output_dim=(
                state["fusion_output_dim"]
                or 64
            ),
            method=state["fusion_method"],
        )

        fusion_module.load_state_dict(
            state["fusion_state_dict"]
        )

        fusion_module.to(dev)
        fusion_module.eval()

        fusion = FusionArtifacts(
            fusion_module=fusion_module,
            fused_by_subject={},
        )

        # ----------------------------------------------------------
        # Restore latent encoder
        # ----------------------------------------------------------
        latent_kind = state["latent_encoder_kind"]
        latent_state = state["latent_state"]

        if latent_kind == "ae":
            from .latent import (
                _Autoencoder,
                AutoencoderLatentState,
            )

            scaler = latent_state["scaler"]
            latent_dim = latent_state["latent_dim"]

            n_features = scaler.mean_.shape[0]

            ae_model = _Autoencoder(
                input_dim=n_features,
                latent_dim=latent_dim,
                hidden_sizes=resolved_cfg.model.ae_hidden_sizes,
            )

            ae_model.load_state_dict(
                latent_state["model_state_dict"]
            )

            encoder = AutoencoderLatentState(
                model=ae_model,
                scaler=scaler,
                latent_dim=latent_dim,
                _device=dev,
            )

        else:
            from .latent import PCALatentState

            encoder = PCALatentState(
                pca=latent_state["pca"],
                scaler=latent_state["scaler"],
            )

        latent = LatentArtifacts(
            encoder_kind=latent_kind,
            encoder=encoder,
            latent_by_subject={},
        )

        instance = cls(
            cfg=resolved_cfg,
            n_activity_classes=n_activity_classes,
            architecture=architecture,
            synergy=synergy,
            fusion=fusion,
            latent=latent,
            n_transition_types=n_transition_types,
            checkpoint_manager=manager,
            device=dev,
        )

        return instance

def _pipeline_config_from_dict(
    raw: Dict[str, Any],
) -> PipelineConfig:
    """
    Reconstructs a PipelineConfig (with nested dataclasses)
    from a plain dictionary.
    """
    from .config import (
        SynergyConfig,
        ModelConfig,
        EvalConfig,
        WindowConfig,
    )

    def _sub(cls, key):
        val = raw.get(key, {})

        if isinstance(val, dict):
            return cls(**val)

        return val

    kwargs = dict(raw)

    kwargs["data_dir"] = Path(
        raw["data_dir"]
    )

    kwargs["output_dir"] = Path(
        raw["output_dir"]
    )

    kwargs["synergy"] = _sub(
        SynergyConfig,
        "synergy",
    )

    kwargs["model"] = _sub(
        ModelConfig,
        "model",
    )

    kwargs["eval"] = _sub(
        EvalConfig,
        "eval",
    )

    kwargs["windows"] = _sub(
        WindowConfig,
        "windows",
    )

    return PipelineConfig(**kwargs)
