from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

from .data import SubjectDataset, make_forecast_target


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Signal conditioning
# ---------------------------------------------------------------------------

def smooth_signal(X: np.ndarray, window: int = 5) -> np.ndarray:
    if window <= 1:
        return X.astype(float, copy=True)
    device = _get_device()
    Xt = _to_tensor(X, device)                        # (T, C)
    # Conv1d expects (N, C_in, L); treat each channel independently
    T, C = Xt.shape
    Xt_t = Xt.T.unsqueeze(0)                          # (1, C, T)
    kernel = torch.ones(1, 1, window, device=device, dtype=torch.float32) / window
    pad = window // 2
    # group conv: each channel convolved with the same box kernel
    Xt_t = Xt_t.view(C, 1, T)
    smoothed = F.conv1d(Xt_t, kernel, padding=pad)    # (C, 1, T)
    # Trim or pad to original length to match numpy "same" semantics
    smoothed = smoothed[:, 0, :T]                     # (C, T)
    return _to_numpy(smoothed.T)                      # (T, C)


def fit_scaler(X: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler


def transform_scaler(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    return scaler.transform(X)


def minmax_positive(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    device = _get_device()
    Xt = _to_tensor(
        np.nan_to_num(np.asarray(X, dtype=float), nan=0.0, posinf=0.0, neginf=0.0),
        device,
    )
    Xt = Xt - Xt.min(dim=0, keepdim=True).values
    Xt = torch.clamp(Xt, min=0.0)
    return _to_numpy(Xt + eps)


def condition_emg(
    X: np.ndarray,
    smooth: bool = False,
    smooth_window: int = 5,
    scaler: Optional[StandardScaler] = None,
    rectify: bool = False,
) -> np.ndarray:
    device = _get_device()
    out = _to_tensor(np.asarray(X, dtype=float), device)

    if smooth:
        # Reuse smooth_signal but stay on GPU via the tensor path
        T, C = out.shape
        kernel = torch.ones(1, 1, smooth_window, device=device) / smooth_window
        pad = smooth_window // 2
        out_t = out.T.unsqueeze(0).view(C, 1, T)
        out_t = F.conv1d(out_t, kernel, padding=pad)[:, 0, :T]
        out = out_t.T

    if rectify:
        out = torch.abs(out)

    result = _to_numpy(out)
    if scaler is not None:
        result = transform_scaler(result, scaler)
    return result


def ms_to_samples(ms: int, sample_rate_hz: int) -> int:
    return max(1, int(round((ms / 1000.0) * sample_rate_hz)))


def overlap_to_stride(window_size: int, overlap: float) -> int:
    overlap = float(np.clip(overlap, 0.0, 0.99))
    return max(1, int(round(window_size * (1.0 - overlap))))


# ---------------------------------------------------------------------------
# Windowed dataset
# ---------------------------------------------------------------------------

@dataclass
class WindowedDataset:
    X_seq: np.ndarray
    X_flat: np.ndarray
    y: np.ndarray
    subject_ids: np.ndarray
    cycle_ids: Optional[np.ndarray] = None
    gait_percent: Optional[np.ndarray] = None
    start_indices: Optional[np.ndarray] = None
    end_indices: Optional[np.ndarray] = None
    source_shapes: Optional[List[Tuple[int, int]]] = None


def build_windows_from_subject(
    subject: SubjectDataset,
    window_size: int,
    horizon_steps: int = 0,
    stride: int = 1,
    use_center_label: bool = False,
) -> WindowedDataset:
    X = np.asarray(subject.X, dtype=float)
    y = np.asarray(subject.y)
    if len(X) != len(y):
        raise ValueError(f"X/y length mismatch for {subject.subject_id}")
    n, n_feat = len(X), X.shape[1]
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")

    device = _get_device()
    Xt = _to_tensor(X, device)                        # (T, C) on GPU

    # Compute valid start indices on CPU (indexing logic is trivial)
    starts_np = np.arange(0, n - window_size + 1, stride, dtype=np.int32)
    if use_center_label:
        target_indices = starts_np + window_size // 2 + horizon_steps
    else:
        target_indices = starts_np + window_size - 1 + horizon_steps

    valid_mask = target_indices < n
    starts_np = starts_np[valid_mask]
    target_indices = target_indices[valid_mask]

    if len(starts_np) == 0:
        return WindowedDataset(
            X_seq=np.empty((0, window_size, n_feat)),
            X_flat=np.empty((0, window_size * n_feat)),
            y=np.empty((0,), dtype=y.dtype),
            subject_ids=np.empty((0,), dtype=object),
        )

    # Build sequence tensor on GPU using advanced indexing
    # idx: (N_windows, window_size)
    idx = (
        torch.as_tensor(starts_np, dtype=torch.long, device=device).unsqueeze(1)
        + torch.arange(window_size, device=device).unsqueeze(0)
    )
    seqs_t = Xt[idx]                                  # (N, W, C)
    seqs_np = _to_numpy(seqs_t)
    flats_np = seqs_np.reshape(len(seqs_np), -1)

    targets = y[target_indices]
    ends_np = starts_np + window_size - 1

    subject_ids_arr = np.full(len(starts_np), subject.subject_id, dtype=object)
    cycle_ids_arr = (
        subject.cycle_id[ends_np] if subject.cycle_id is not None else None
    )
    gait_arr = (
        subject.gait_percent[ends_np].astype(float) if subject.gait_percent is not None else None
    )

    return WindowedDataset(
        X_seq=seqs_np,
        X_flat=flats_np,
        y=targets.astype(int),
        subject_ids=subject_ids_arr,
        cycle_ids=np.asarray(cycle_ids_arr, dtype=object) if cycle_ids_arr is not None else None,
        gait_percent=gait_arr,
        start_indices=starts_np.astype(int),
        end_indices=ends_np.astype(int),
        source_shapes=[X.shape],
    )


def build_windowed_dataset(
    subjects: Sequence[SubjectDataset],
    window_size: int,
    horizon_steps: int = 0,
    overlap: float = 0.5,
    use_center_label: bool = False,
) -> WindowedDataset:
    all_seq: List[np.ndarray] = []
    all_flat: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    all_subject_ids: List[np.ndarray] = []
    all_cycle: List[np.ndarray] = []
    all_gait: List[np.ndarray] = []
    all_start: List[np.ndarray] = []
    all_end: List[np.ndarray] = []

    stride = overlap_to_stride(window_size, overlap)

    for subject in subjects:
        wd = build_windows_from_subject(
            subject,
            window_size=window_size,
            horizon_steps=horizon_steps,
            stride=stride,
            use_center_label=use_center_label,
        )
        if len(wd.y) == 0:
            continue
        all_seq.append(wd.X_seq)
        all_flat.append(wd.X_flat)
        all_y.append(wd.y)
        all_subject_ids.append(wd.subject_ids)
        if wd.cycle_ids is not None:
            all_cycle.append(wd.cycle_ids)
        if wd.gait_percent is not None:
            all_gait.append(wd.gait_percent)
        if wd.start_indices is not None:
            all_start.append(wd.start_indices)
        if wd.end_indices is not None:
            all_end.append(wd.end_indices)

    if not all_seq:
        n_feat = subjects[0].X.shape[1] if subjects else 0
        return WindowedDataset(
            X_seq=np.empty((0, window_size, n_feat)),
            X_flat=np.empty((0, window_size * n_feat)),
            y=np.empty((0,), dtype=int),
            subject_ids=np.empty((0,), dtype=object),
        )

    return WindowedDataset(
        X_seq=np.concatenate(all_seq, axis=0),
        X_flat=np.concatenate(all_flat, axis=0),
        y=np.concatenate(all_y, axis=0),
        subject_ids=np.concatenate(all_subject_ids, axis=0),
        cycle_ids=np.concatenate(all_cycle, axis=0) if all_cycle else None,
        gait_percent=np.concatenate(all_gait, axis=0) if all_gait else None,
        start_indices=np.concatenate(all_start, axis=0) if all_start else None,
        end_indices=np.concatenate(all_end, axis=0) if all_end else None,
        source_shapes=[s.X.shape for s in subjects],
    )


def build_horizon_steps(ms: int, sample_rate_hz: int) -> int:
    return ms_to_samples(ms, sample_rate_hz)


def make_forecast_target_from_windows(y: np.ndarray, horizon_steps: int) -> np.ndarray:
    return make_forecast_target(y, horizon_steps)
