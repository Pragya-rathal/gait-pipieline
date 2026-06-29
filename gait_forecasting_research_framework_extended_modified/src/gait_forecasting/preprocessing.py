
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler

from .data import SubjectDataset, make_forecast_target


def smooth_signal(X: np.ndarray, window: int = 5) -> np.ndarray:
    if window <= 1:
        return X.astype(float, copy=True)
    kernel = np.ones(window, dtype=float) / float(window)
    out = np.empty_like(X, dtype=float)
    for i in range(X.shape[1]):
        out[:, i] = np.convolve(X[:, i], kernel, mode="same")
    return out


def fit_scaler(X: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler


def transform_scaler(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    return scaler.transform(X)


def minmax_positive(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    X = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    X = X - X.min(axis=0, keepdims=True)
    X = np.maximum(X, 0.0)
    return X + eps


def condition_emg(
    X: np.ndarray,
    smooth: bool = False,
    smooth_window: int = 5,
    scaler: StandardScaler | None = None,
    rectify: bool = False,
) -> np.ndarray:
    out = np.asarray(X, dtype=float)
    if smooth:
        out = smooth_signal(out, window=smooth_window)
    if rectify:
        out = np.abs(out)
    if scaler is not None:
        out = transform_scaler(out, scaler)
    return out


def ms_to_samples(ms: int, sample_rate_hz: int) -> int:
    return max(1, int(round((ms / 1000.0) * sample_rate_hz)))


def overlap_to_stride(window_size: int, overlap: float) -> int:
    overlap = float(np.clip(overlap, 0.0, 0.99))
    stride = int(round(window_size * (1.0 - overlap)))
    return max(1, stride)


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
    n = len(X)
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")

    seqs, flats, targets = [], [], []
    subject_ids, cycle_ids, gait_percent, starts, ends = [], [], [], [], []

    for start in range(0, n - window_size + 1, stride):
        end = start + window_size - 1
        target_idx = start + window_size - 1 + horizon_steps
        if use_center_label:
            target_idx = start + window_size // 2 + horizon_steps
        if target_idx >= n:
            break
        seq = X[start : start + window_size]
        seqs.append(seq)
        flats.append(seq.reshape(-1))
        targets.append(y[target_idx])
        subject_ids.append(subject.subject_id)
        if subject.cycle_id is not None:
            cycle_ids.append(subject.cycle_id[end])
        if subject.gait_percent is not None:
            gait_percent.append(float(subject.gait_percent[end]))
        starts.append(start)
        ends.append(end)

    if not seqs:
        return WindowedDataset(
            X_seq=np.empty((0, window_size, X.shape[1])),
            X_flat=np.empty((0, window_size * X.shape[1])),
            y=np.empty((0,), dtype=y.dtype),
            subject_ids=np.empty((0,), dtype=object),
        )

    return WindowedDataset(
        X_seq=np.asarray(seqs, dtype=float),
        X_flat=np.asarray(flats, dtype=float),
        y=np.asarray(targets, dtype=int),
        subject_ids=np.asarray(subject_ids, dtype=object),
        cycle_ids=np.asarray(cycle_ids, dtype=object) if cycle_ids else None,
        gait_percent=np.asarray(gait_percent, dtype=float) if gait_percent else None,
        start_indices=np.asarray(starts, dtype=int),
        end_indices=np.asarray(ends, dtype=int),
        source_shapes=[X.shape],
    )


def build_windowed_dataset(
    subjects: Sequence[SubjectDataset],
    window_size: int,
    horizon_steps: int = 0,
    overlap: float = 0.5,
    use_center_label: bool = False,
) -> WindowedDataset:
    all_seq, all_flat, all_y, all_subject_ids = [], [], [], []
    all_cycle, all_gait, all_start, all_end = [], [], [], []
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
        return WindowedDataset(
            X_seq=np.empty((0, window_size, subjects[0].X.shape[1] if subjects else 0)),
            X_flat=np.empty((0, window_size * (subjects[0].X.shape[1] if subjects else 0))),
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
