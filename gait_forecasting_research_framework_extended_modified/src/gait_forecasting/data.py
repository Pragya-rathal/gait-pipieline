from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import json
import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold, KFold, train_test_split


KNOWN_LABEL_NAMES = {
    "phase", "label", "labels", "target", "y", "class", "classes",
    "phase2", "phase4", "phase7", "phase_label", "phase_labels",
    "gait_phase", "gait_label", "gait_state", "state", "class_label",
}
KNOWN_META_NAMES = {
    "subject_id", "subject", "subjectid", "participant", "participant_id",
    "cycle_id", "cycle", "cycleid", "trial", "trial_id",
    "sample_id", "sample", "sample_idx", "sample_index", "index", "idx",
    "gait_percent", "gait%", "percent_gait", "percent", "time", "time_sec", "timestamp",
    "file", "filepath", "path", "source", "session", "session_id", "recording",
    "analog_idx", "frame", "ftc_state", "ftc_axis", "ftc_axis_name", "ftc_threshold",
    "event_toe_off_frame", "event_heel_strike_frame", "event_peak_frame",
    "confidence", "label_source", "label_quality", "ftc_marker",
    "force_used_force_z_channels", "force_force_rising_edges", "force_force_falling_edges",
    "phase2_name", "phase4_name", "phase7_name", "c3d_file",
}
KNOWN_PATH_NAMES = {"file", "filepath", "path", "source", "csv_path", "emg_path", "data_path"}

EMG_WINDOW_SHAPE = (11, 400)


@dataclass
class SubjectDataset:
    subject_id: str
    X: np.ndarray
    y: np.ndarray
    channel_names: List[str]
    cycle_id: Optional[np.ndarray] = None
    gait_percent: Optional[np.ndarray] = None
    sample_index: Optional[np.ndarray] = None
    source_file: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def _is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s)


def _unique_nonnull_count(s: pd.Series) -> int:
    return int(pd.Series(s).dropna().nunique())


def _is_identifier_like(series: pd.Series) -> bool:
    if series.dtype == object:
        return True
    if _is_numeric_series(series) and _unique_nonnull_count(series) > max(20, int(0.5 * len(series))):
        return True
    return False


def _detect_subject_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if _normalize_name(c) in {"subject_id", "subject", "subjectid", "participant", "participant_id"}:
            return c
    return None


def _detect_cycle_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if _normalize_name(c) in {"cycle_id", "cycle", "cycleid", "trial", "trial_id"}:
            return c
    return None


def _detect_gait_percent_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if _normalize_name(c) in {"gait_percent", "gait", "percent_gait", "gaitpct", "percent"}:
            return c
    return None


def _detect_sample_index_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if _normalize_name(c) in {"sample_index", "sample_idx", "sampleid", "sample_id", "index", "idx"}:
            return c
    return None


def _candidate_label_columns(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in df.columns:
        nc = _normalize_name(c)
        if nc in KNOWN_LABEL_NAMES or nc.startswith("phase_") or nc.endswith("_label") or nc.endswith("_labels"):
            cols.append(c)
    for c in df.columns:
        if c in cols:
            continue
        nc = _normalize_name(c)
        if nc in KNOWN_META_NAMES:
            continue
        s = df[c]
        if _is_numeric_series(s) and 2 <= _unique_nonnull_count(s) <= max(20, int(0.25 * len(df))):
            cols.append(c)
        elif s.dtype == object and 2 <= _unique_nonnull_count(s) <= max(20, int(0.25 * len(df))):
            cols.append(c)
    out = []
    seen = set()
    for c in cols:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def _candidate_emg_columns(df: pd.DataFrame, label_cols: Sequence[str], meta_cols: Sequence[str]) -> List[str]:
    label_norm = {_normalize_name(c) for c in label_cols}
    meta_norm = {_normalize_name(c) for c in meta_cols}

    def _eligible(c: str) -> bool:
        nc = _normalize_name(c)
        if nc in label_norm or nc in meta_norm:
            return False
        return _is_numeric_series(df[c])

    for suffixes in (("_normalized",), ("_raw", "_bandpassed", "_rectified", "_envelope")):
        emg = [
            c for c in df.columns
            if _eligible(c) and any(_normalize_name(c).endswith(sfx) for sfx in suffixes)
        ]
        if emg:
            return emg

    emg = []
    for c in df.columns:
        if not _eligible(c):
            continue
        if not _is_identifier_like(df[c]):
            emg.append(c)
    return emg


def _coerce_str_series(s: pd.Series) -> np.ndarray:
    return s.astype(str).to_numpy()


def _ensure_int_labels(y: np.ndarray) -> np.ndarray:
    """Convert label array to int, raising ValueError on NaN/non-finite values."""
    y = np.asarray(y)
    if y.dtype.kind in {"U", "S", "O"}:
        _, inv = np.unique(y.astype(str), return_inverse=True)
        return inv.astype(int)
    if y.dtype.kind == "f":
        if np.any(~np.isfinite(y)):
            raise ValueError(
                f"Label array contains NaN or Inf values: {y[~np.isfinite(y)]}. "
                "Remove or impute invalid labels before constructing SubjectDataset."
            )
        return np.rint(y).astype(int)
    return y.astype(int)


def _ensure_int_labels_safe(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert label array to int, returning (int_labels, valid_mask).
    Rows with NaN/Inf are flagged in the mask (False = invalid)."""
    y = np.asarray(y)
    if y.dtype.kind in {"U", "S", "O"}:
        int_labels = np.zeros(len(y), dtype=int)
        valid_mask = np.ones(len(y), dtype=bool)
        try:
            _, inv = np.unique(y.astype(str), return_inverse=True)
            int_labels = inv.astype(int)
        except Exception:
            valid_mask[:] = False
        return int_labels, valid_mask
    if y.dtype.kind == "f":
        valid_mask = np.isfinite(y)
        int_labels = np.where(valid_mask, np.rint(y).astype(int), -1)
        return int_labels, valid_mask
    return y.astype(int), np.ones(len(y), dtype=bool)


def _load_npz(path: Path) -> List[SubjectDataset]:
    data = np.load(path, allow_pickle=True)
    if "X" not in data or "y" not in data:
        raise ValueError(f"{path} must contain X and y arrays")
    X = np.asarray(data["X"], dtype=float)
    y = np.asarray(data["y"])
    if y.ndim > 1:
        y = y.reshape(-1)
    channel_names = list(data["channel_names"]) if "channel_names" in data else [f"ch_{i+1}" for i in range(X.shape[1])]
    subject_id = str(data["subject_id"]) if "subject_id" in data else path.stem
    cycle_id = np.asarray(data["cycle_id"]) if "cycle_id" in data else None
    gait_percent = np.asarray(data["gait_percent"], dtype=float) if "gait_percent" in data else None
    sample_index = np.asarray(data["sample_index"], dtype=int) if "sample_index" in data else None
    int_labels, valid_mask = _ensure_int_labels_safe(y)
    if not np.all(valid_mask):
        X = X[valid_mask]
        int_labels = int_labels[valid_mask]
        if cycle_id is not None:
            cycle_id = cycle_id[valid_mask]
        if gait_percent is not None:
            gait_percent = gait_percent[valid_mask]
        if sample_index is not None:
            sample_index = sample_index[valid_mask]
    if len(X) == 0:
        raise ValueError(f"{path}: no valid samples remain after filtering NaN labels")
    return [SubjectDataset(subject_id=subject_id, X=X, y=int_labels, channel_names=channel_names, cycle_id=cycle_id, gait_percent=gait_percent, sample_index=sample_index, source_file=str(path))]


def _pick_label_column(df: pd.DataFrame, preferred: Optional[str] = None) -> Optional[str]:
    if preferred and preferred in df.columns:
        return preferred
    candidates = _candidate_label_columns(df)
    if not candidates:
        return None

    def score(c: str) -> tuple[int, int, int]:
        nc = _normalize_name(c)
        exact = 0 if nc in KNOWN_LABEL_NAMES or nc.endswith("_label") or nc.endswith("_labels") else 1
        cardinality = _unique_nonnull_count(df[c])
        return (exact, cardinality, len(c))

    candidates = sorted(candidates, key=score)
    return candidates[0]


def _detect_emg_and_metadata(df: pd.DataFrame, label_col: Optional[str] = None):
    subject_col = _detect_subject_col(df)
    cycle_col = _detect_cycle_col(df)
    gait_percent_col = _detect_gait_percent_col(df)
    sample_index_col = _detect_sample_index_col(df)
    meta_cols = [c for c in [subject_col, cycle_col, gait_percent_col, sample_index_col] if c is not None]
    if label_col is None:
        label_col = _pick_label_column(df)
    label_cols = [label_col] if label_col is not None else []
    emg_cols = _candidate_emg_columns(df, label_cols=label_cols, meta_cols=meta_cols)
    return emg_cols, label_col, subject_col, cycle_col, gait_percent_col, sample_index_col, meta_cols


def _sort_df(df: pd.DataFrame, cycle_col: Optional[str], gait_percent_col: Optional[str], sample_index_col: Optional[str]) -> pd.DataFrame:
    sort_cols = [c for c in [cycle_col, gait_percent_col, sample_index_col] if c is not None]
    if sort_cols:
        return df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    return df.reset_index(drop=True)


def _load_tabular_as_subjects(path: Path, preferred_label: Optional[str] = None) -> List[SubjectDataset]:
    df = pd.read_csv(path)

    if preferred_label is None and path.name.lower() == "sample_level_dataset.csv" and "phase7" in df.columns:
        preferred_label = "phase7"

    emg_cols, label_col, subject_col, cycle_col, gait_percent_col, sample_index_col, meta_cols = _detect_emg_and_metadata(df, preferred_label)

    if label_col is None:
        raise ValueError(f"Could not detect a label column in {path.name}")

    path_like_cols = [c for c in df.columns if _normalize_name(c) in KNOWN_PATH_NAMES]
    if path_like_cols and len(emg_cols) < 3:
        records: List[SubjectDataset] = []
        for _, row in df.iterrows():
            ref = None
            for c in path_like_cols:
                val = row[c]
                if isinstance(val, str) and val.strip():
                    ref = val
                    break
            if ref is None:
                continue
            ref_path = (path.parent / str(ref)).resolve() if not Path(str(ref)).is_absolute() else Path(str(ref))
            if ref_path.exists():
                records.extend(load_subject_file(ref_path))
        if records:
            return records

    raw_labels = df[label_col].to_numpy()
    int_labels, valid_mask = _ensure_int_labels_safe(raw_labels)

    if not np.all(valid_mask):
        df = df[valid_mask].reset_index(drop=True)
        int_labels = int_labels[valid_mask]

    if subject_col:
        subject_vals = _coerce_str_series(df[subject_col])
        if (
            len(np.unique(subject_vals)) == 1
            and str(subject_vals[0]).strip().lower() in {"unknown", "nan", "none", ""}
            and "trial_id" in df.columns
        ):
            subject_vals = _coerce_str_series(df["trial_id"])
    else:
        subject_vals = np.array([path.stem] * len(df), dtype=object)
    cycle_vals = _coerce_str_series(df[cycle_col]) if cycle_col else None
    gait_vals = df[gait_percent_col].to_numpy(dtype=float) if gait_percent_col else None
    sample_vals = df[sample_index_col].to_numpy(dtype=int) if sample_index_col else np.arange(len(df), dtype=int)

    out: List[SubjectDataset] = []
    grouped = df.assign(_subject_id=subject_vals, _label=int_labels)
    if cycle_vals is not None:
        grouped["_cycle_id"] = cycle_vals
    if gait_vals is not None:
        grouped["_gait_percent"] = gait_vals
    if sample_index_col is not None:
        grouped["_sample_index"] = sample_vals

    for subject_id, subdf in grouped.groupby("_subject_id", sort=False):
        subdf = _sort_df(subdf, "_cycle_id" if "_cycle_id" in subdf.columns else None, "_gait_percent" if "_gait_percent" in subdf.columns else None, "_sample_index" if "_sample_index" in subdf.columns else None)
        X = subdf[emg_cols].to_numpy(dtype=float)
        y = subdf["_label"].to_numpy(dtype=int)
        if len(X) == 0:
            continue
        cycle_id = subdf["_cycle_id"].to_numpy(dtype=object) if "_cycle_id" in subdf.columns else None
        gait_percent = subdf["_gait_percent"].to_numpy(dtype=float) if "_gait_percent" in subdf.columns else None
        sample_index = subdf["_sample_index"].to_numpy(dtype=int) if "_sample_index" in subdf.columns else None
        metadata = {
            "source_file": str(path),
            "label_column": label_col,
            "emg_columns": emg_cols,
            "all_columns": list(df.columns),
        }
        for extra_col in _candidate_label_columns(df):
            if extra_col == label_col:
                continue
            try:
                metadata[f"label::{extra_col}"] = _ensure_int_labels(subdf[extra_col].to_numpy())
            except Exception:
                pass
        out.append(
            SubjectDataset(
                subject_id=str(subject_id),
                X=X,
                y=y,
                channel_names=list(emg_cols),
                cycle_id=cycle_id,
                gait_percent=gait_percent,
                sample_index=sample_index,
                source_file=str(path),
                metadata=metadata,
            )
        )
    return out


# ===========================================================================
# JSON manifest-driven windowed dataset (primary target format)
# ===========================================================================
#
# Layout convention:
#   data_dir/
#     manifest.csv                 (or *_manifest.csv)
#       columns: window_path, subject_id, recording_id, window_index,
#                current_activity, future_activity, transition_flag,
#                transition_type, time_to_transition, [extra metadata...]
#     windows/<subject_id>/<recording_id>/<window_index>.json
#       { "emg": [[...11x400...]], "channel_names": [...] (optional) }
#
# Each JSON window tensor is read from disk lazily, on `__getitem__`, so the
# full 25,338-window corpus is never materialized in RAM at once.

MANIFEST_LABEL_COLUMNS = {
    "current_activity", "future_activity", "transition_flag",
    "transition_type", "time_to_transition",
}
MANIFEST_REQUIRED_COLUMNS = {"window_path", "subject_id"}


def _find_manifest_csv(data_dir: Path) -> Optional[Path]:
    data_dir = Path(data_dir)

    preferred = [
        data_dir / "manifest" / "dataset_manifest.csv",
        data_dir / "manifest" / "windows_manifest.csv",
        data_dir / "metadata" / "window_metadata.csv",
        data_dir / "dataset_manifest.csv",
        data_dir / "windows_manifest.csv",
        data_dir / "manifest.csv",
    ]

    for path in preferred:
        if path.exists():
            return path

    candidates = sorted(data_dir.rglob("*manifest*.csv"))

    if candidates:
        return candidates[0]

    return None


def _resolve_window_path(data_dir: Path, raw_path: str) -> Path:
    p = Path(str(raw_path))
    return p if p.is_absolute() else (data_dir / p)


@dataclass
class JSONWindowRecord:
    """One row of the manifest: pointer to a JSON tensor plus its labels."""
    window_path: Path
    subject_id: str
    recording_id: str
    window_index: int
    current_activity: int
    future_activity: int
    transition_flag: int
    transition_type: int
    time_to_transition: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class WindowValidationError(ValueError):
    pass


def _validate_emg_tensor(arr: np.ndarray, expected_shape: Tuple[int, int] = EMG_WINDOW_SHAPE, *, source: str = "") -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        raise WindowValidationError(f"{source}: empty array (size=0)")
    if arr.ndim != 2:
        raise WindowValidationError(f"{source}: expected 2D array, got {arr.ndim}D with shape {arr.shape}")
    if arr.shape != expected_shape:
        raise WindowValidationError(f"{source}: expected shape {expected_shape}, got {arr.shape}")
    if np.isnan(arr).any():
        raise WindowValidationError(f"{source}: contains NaN values")
    if np.isinf(arr).any():
        raise WindowValidationError(f"{source}: contains Inf values")
    return arr


def _validate_manifest_row(row: pd.Series, source: str = "") -> None:
    for col in MANIFEST_REQUIRED_COLUMNS:
        if col not in row or pd.isna(row[col]):
            raise WindowValidationError(f"{source}: malformed metadata, missing required field {col!r}")
    # Validate subject_id
    subject_id = row.get("subject_id", None)
    if subject_id is None or (isinstance(subject_id, float) and np.isnan(subject_id)):
        raise WindowValidationError(f"{source}: missing or NaN subject_id")
    # Validate recording_id if present
    if "recording_id" in row:
        rec_id = row["recording_id"]
        if pd.isna(rec_id):
            raise WindowValidationError(f"{source}: NaN recording_id")
    # Validate window_index if present
    if "window_index" in row:
        w_idx = row["window_index"]
        if pd.isna(w_idx):
            raise WindowValidationError(f"{source}: NaN window_index")
    # Validate window_path exists
    window_path_raw = row.get("window_path", None)
    if window_path_raw is None or (isinstance(window_path_raw, float) and np.isnan(window_path_raw)):
        raise WindowValidationError(f"{source}: missing window_path")
    for col in MANIFEST_LABEL_COLUMNS:
        if col in row and pd.notna(row[col]):
            val = row[col]
            if col == "time_to_transition":
                if not np.isfinite(float(val)):
                    raise WindowValidationError(f"{source}: non-finite time_to_transition")
            else:
                try:
                    int(val)
                except (TypeError, ValueError):
                    raise WindowValidationError(f"{source}: non-integer label in column {col!r}")


def discover_json_manifest(data_dir: Path) -> Optional[Path]:
    """Automatic dataset discovery: locate the manifest CSV that indexes a JSON window corpus."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return None
    return _find_manifest_csv(data_dir)


def load_json_manifest(
    data_dir: Path,
    manifest_path: Optional[Path] = None,
    validate: bool = True,
) -> List[JSONWindowRecord]:
    """
    Parses the manifest CSV into a list of lightweight ``JSONWindowRecord``
    pointers (no tensor data is loaded here — that happens lazily inside
    ``JSONDataset.__getitem__``).
    """
    data_dir = Path(data_dir)
    manifest_path = Path(manifest_path) if manifest_path is not None else _find_manifest_csv(data_dir)
    if manifest_path is None or not manifest_path.exists():
        raise FileNotFoundError(f"No manifest CSV found under {data_dir}")

    df = pd.read_csv(manifest_path)
    df.columns = [_normalize_name(c) if _normalize_name(c) in (MANIFEST_LABEL_COLUMNS | MANIFEST_REQUIRED_COLUMNS | {"recording_id", "window_index"}) else c for c in df.columns]

    records: List[JSONWindowRecord] = []
    known_cols = MANIFEST_REQUIRED_COLUMNS | MANIFEST_LABEL_COLUMNS | {"recording_id", "window_index"}

    skipped = 0
    for i, row in df.iterrows():
        source = f"{manifest_path.name}:row{i}"
        if validate:
            try:
                _validate_manifest_row(row, source=source)
            except WindowValidationError as exc:
                skipped += 1
                continue

        # Verify window_path is not NaN/empty before resolving
        window_path_raw = row.get("window_path", None)
        if window_path_raw is None or (isinstance(window_path_raw, float) and np.isnan(window_path_raw)):
            skipped += 1
            continue

        resolved_path = _resolve_window_path(data_dir, str(window_path_raw))

        extra_meta = {c: row[c] for c in df.columns if c not in known_cols and pd.notna(row[c])}

        records.append(JSONWindowRecord(
            window_path=resolved_path,
            subject_id=str(row["subject_id"]),
            recording_id=str(row.get("recording_id", row["subject_id"])),
            window_index=int(row.get("window_index", i)),
            current_activity=int(row.get("current_activity", -1)) if pd.notna(row.get("current_activity", np.nan)) else -1,
            future_activity=int(row.get("future_activity", -1)) if pd.notna(row.get("future_activity", np.nan)) else -1,
            transition_flag=int(row.get("transition_flag", 0)) if pd.notna(row.get("transition_flag", np.nan)) else 0,
            transition_type=int(row.get("transition_type", 0)) if pd.notna(row.get("transition_type", np.nan)) else 0,
            time_to_transition=float(row.get("time_to_transition", 0.0)) if pd.notna(row.get("time_to_transition", np.nan)) else 0.0,
            metadata=extra_meta,
        ))

    if not records:
        raise ValueError(
            f"No valid manifest rows found in {manifest_path} "
            f"({skipped} rows skipped due to validation errors)."
        )

    return records


def _read_json_window(path: Path, expected_shape: Tuple[int, int] = EMG_WINDOW_SHAPE) -> Tuple[np.ndarray, List[str]]:
    if not path.exists():
        raise FileNotFoundError(f"JSON window tensor not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        emg = payload.get("emg", payload.get("X"))
        channel_names = payload.get("channel_names", [f"ch_{i+1}" for i in range(expected_shape[0])])
    else:
        emg = payload
        channel_names = [f"ch_{i+1}" for i in range(expected_shape[0])]
    if emg is None:
        raise WindowValidationError(f"{path}: JSON payload missing 'emg' or 'X' key")
    arr = _validate_emg_tensor(np.asarray(emg), expected_shape=expected_shape, source=str(path))
    # Guarantee channel_names length matches expected_shape[0]
    if len(channel_names) != expected_shape[0]:
        channel_names = [f"ch_{i+1}" for i in range(expected_shape[0])]
    return arr, list(channel_names)


class JSONDataset(Dataset):
    """
    PyTorch ``Dataset`` over a JSON-tensor EMG window corpus indexed by a
    manifest CSV. Designed for the 25,338-window / 11×400 corpus, but works
    for any manifest-described JSON window collection.

    Tensors are read from disk lazily inside ``__getitem__`` — nothing is
    preloaded into RAM at construction time, so this scales to corpora far
    larger than available memory. Combine with ``num_workers > 0`` and
    ``pin_memory=True`` (see ``build_dataloader``) for GPU-friendly batching.

    Each sample is a dict with keys:
        emg, current_activity, future_activity, transition_flag,
        transition_type, time_to_transition, subject_id, recording_id,
        window_index, metadata
    """

    def __init__(
        self,
        records: Sequence[JSONWindowRecord],
        expected_shape: Tuple[int, int] = EMG_WINDOW_SHAPE,
        transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        validate_on_load: bool = True,
    ) -> None:
        self.records = list(records)
        self.expected_shape = expected_shape
        self.transform = transform
        self.validate_on_load = validate_on_load

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        emg, channel_names = _read_json_window(rec.window_path, expected_shape=self.expected_shape)
        if self.transform is not None:
            emg = self.transform(emg)

        return {
            "emg": torch.from_numpy(np.ascontiguousarray(emg, dtype=np.float32)),
            "current_activity": torch.tensor(rec.current_activity, dtype=torch.long),
            "future_activity": torch.tensor(rec.future_activity, dtype=torch.long),
            "transition_flag": torch.tensor(rec.transition_flag, dtype=torch.long),
            "transition_type": torch.tensor(rec.transition_type, dtype=torch.long),
            "time_to_transition": torch.tensor(rec.time_to_transition, dtype=torch.float32),
            "subject_id": rec.subject_id,
            "recording_id": rec.recording_id,
            "window_index": rec.window_index,
            "metadata": rec.metadata,
            "channel_names": channel_names,
        }

    # ------------------------------------------------------------------
    # Subject-aware helpers (mirrors SubjectDataset-level grouping)
    # ------------------------------------------------------------------

    def subject_ids(self) -> np.ndarray:
        return np.array([r.subject_id for r in self.records], dtype=object)

    def filter_by_subjects(self, subject_ids: Iterable[str]) -> "JSONDataset":
        keep = set(subject_ids)
        filtered = [r for r in self.records if r.subject_id in keep]
        return JSONDataset(filtered, expected_shape=self.expected_shape, transform=self.transform, validate_on_load=self.validate_on_load)

    def to_subject_datasets(self) -> List[SubjectDataset]:
        """
        Materializes this JSON corpus into the legacy ``SubjectDataset``
        in-memory representation, for code paths that have not yet been
        ported to the lazy ``JSONDataset`` (e.g. NMF fitting). This call
        does load all tensors into RAM — only use for corpora that fit.

        Each 11×400 EMG window is reduced to an 11-dimensional feature
        vector (per-channel RMS amplitude), so X always has shape
        (number_of_windows, 11) and channel_names always has length 11.
        """
        n_channels = self.expected_shape[0]
        standard_channel_names = [f"ch_{i+1}" for i in range(n_channels)]

        by_subject: Dict[str, List[int]] = {}
        for i, r in enumerate(self.records):
            by_subject.setdefault(r.subject_id, []).append(i)

        out: List[SubjectDataset] = []
        for subject_id, indices in by_subject.items():
            ordered = sorted(indices, key=lambda i: (self.records[i].recording_id, self.records[i].window_index))
            X_rows: List[np.ndarray] = []
            y_rows: List[int] = []
            skipped = 0
            for i in ordered:
                rec = self.records[i]
                try:
                    emg, _ = _read_json_window(rec.window_path, expected_shape=self.expected_shape)
                except (WindowValidationError, FileNotFoundError, ValueError):
                    skipped += 1
                    continue
                # Validate label: skip windows with invalid current_activity
                label = rec.current_activity
                if label < 0:
                    skipped += 1
                    continue
                # Reduce 11×400 → 11-dim feature vector via per-channel RMS
                feature_vec = np.sqrt(np.mean(emg ** 2, axis=1))  # shape (11,)
                if feature_vec.shape != (n_channels,):
                    skipped += 1
                    continue
                X_rows.append(feature_vec)
                y_rows.append(label)

            if not X_rows:
                # Subject has zero valid windows; exclude completely
                continue

            X = np.vstack(X_rows)  # (N, 11) guaranteed
            y = np.asarray(y_rows, dtype=int)

            # Invariant checks before appending
            if X.shape[0] != len(y):
                raise ValueError(
                    f"Subject {subject_id!r}: X.shape[0]={X.shape[0]} != len(y)={len(y)}"
                )
            if X.shape[1] != n_channels:
                raise ValueError(
                    f"Subject {subject_id!r}: feature dimension {X.shape[1]} != expected {n_channels}"
                )

            out.append(SubjectDataset(
                subject_id=subject_id,
                X=X,
                y=y,
                channel_names=standard_channel_names,
                source_file=str(self.records[ordered[0]].window_path) if ordered else None,
                metadata={"format": "json_manifest", "skipped_windows": skipped},
            ))

        # Final cross-subject dimension consistency check
        if out:
            dims = {s.X.shape[1] for s in out}
            if len(dims) > 1:
                raise ValueError(
                    f"Inconsistent feature dimensions across subjects after materialization: {dims}. "
                    "All subjects must have the same X.shape[1]."
                )

        return out


def json_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom ``collate_fn`` for ``JSONDataset`` batches. Stacks tensor fields
    and groups string/dict metadata fields into lists, since the default
    PyTorch collate cannot stack heterogeneous metadata dicts.
    """
    out: Dict[str, Any] = {}
    tensor_keys = [
        "emg", "current_activity", "future_activity",
        "transition_flag", "transition_type", "time_to_transition",
    ]
    for k in tensor_keys:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    out["subject_id"] = [b["subject_id"] for b in batch]
    out["recording_id"] = [b["recording_id"] for b in batch]
    out["window_index"] = [b["window_index"] for b in batch]
    out["metadata"] = [b["metadata"] for b in batch]
    out["channel_names"] = batch[0]["channel_names"] if batch else []
    return out


def build_dataloader(
    dataset: JSONDataset,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
    persistent_workers: Optional[bool] = None,
) -> DataLoader:
    """
    Constructs a GPU-friendly ``DataLoader`` for a ``JSONDataset``:
    multi-worker lazy disk reads, pinned host memory for fast
    host→device transfer, and the dedicated ``json_collate_fn``.
    """
    persistent = persistent_workers if persistent_workers is not None else (num_workers > 0)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=drop_last,
        collate_fn=json_collate_fn,
        persistent_workers=persistent and num_workers > 0,
    )


# ===========================================================================
# Cross-validation splitting over JSON manifest records (subject-level)
# ===========================================================================

def json_leave_one_subject_out(
    records: Sequence[JSONWindowRecord],
) -> Iterable[Tuple[str, List[JSONWindowRecord], List[JSONWindowRecord]]]:
    unique = sorted({r.subject_id for r in records})
    for held_out in unique:
        train = [r for r in records if r.subject_id != held_out]
        test = [r for r in records if r.subject_id == held_out]
        yield held_out, train, test


def json_group_kfold_splits(
    records: Sequence[JSONWindowRecord],
    n_splits: int = 5,
) -> Iterable[Tuple[List[JSONWindowRecord], List[JSONWindowRecord]]]:
    groups = np.array([r.subject_id for r in records], dtype=object)
    n_splits = min(n_splits, max(2, len(set(groups.tolist()))))
    splitter = GroupKFold(n_splits=n_splits)
    X_dummy = np.zeros((len(records), 1))
    for train_idx, test_idx in splitter.split(X_dummy, groups=groups):
        yield [records[i] for i in train_idx], [records[i] for i in test_idx]


def json_holdout_split(
    records: Sequence[JSONWindowRecord],
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[List[JSONWindowRecord], List[JSONWindowRecord], List[JSONWindowRecord]]:
    subject_ids = sorted({r.subject_id for r in records})
    if len(subject_ids) < 3:
        train, test = train_test_split(list(records), test_size=test_size, random_state=random_state, shuffle=True)
        train, val = train_test_split(train, test_size=val_size, random_state=random_state, shuffle=True)
        return list(train), list(val), list(test)

    train_ids, test_ids = train_test_split(subject_ids, test_size=test_size, random_state=random_state, shuffle=True)
    train_ids, val_ids = train_test_split(train_ids, test_size=val_size, random_state=random_state, shuffle=True)
    train_ids, val_ids, test_ids = set(train_ids), set(val_ids), set(test_ids)
    train = [r for r in records if r.subject_id in train_ids]
    val = [r for r in records if r.subject_id in val_ids]
    test = [r for r in records if r.subject_id in test_ids]
    return train, val, test


# ===========================================================================
# DatasetFactory — unified entry point across JSON / CSV / NPZ
# ===========================================================================

class DatasetFactory:
    """
    Single entry point for dataset discovery and construction. Detects
    whether ``data_dir`` contains a JSON-manifest corpus, legacy CSV
    manifests, or NPZ files, and dispatches accordingly. ``load_dataset``
    (the original public API) remains a thin wrapper around this factory
    for backward compatibility.
    """

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)

    def detect_format(self) -> str:
        if discover_json_manifest(self.data_dir) is not None:
            return "json"
        files = _candidate_data_files(self.data_dir)
        if any(f.suffix.lower() == ".npz" for f in files):
            return "npz"
        if any(f.suffix.lower() == ".csv" for f in files):
            return "csv"
        raise FileNotFoundError(f"No supported dataset files found in {self.data_dir}")

    

    def load_subject_datasets(self) -> List[SubjectDataset]:
        fmt = self.detect_format()

        print("=" * 60)
        print("Detected dataset format:", fmt)
        print("Dataset directory:", self.data_dir)
        print("=" * 60)

        if fmt == "json":
            print(">>> Using JSONDataset loader")
            records = load_json_manifest(self.data_dir)
            print(f"Loaded {len(records)} manifest records")
            return JSONDataset(records).to_subject_datasets()

        print(">>> Using legacy dataset loader")
        return _load_legacy_dataset(self.data_dir)

        print("Using legacy loader")
        return _load_legacy_dataset(self.data_dir)

    def load_json_dataset(
        self,
        manifest_path: Optional[Path] = None,
        validate: bool = True,
    ) -> JSONDataset:
        records = load_json_manifest(self.data_dir, manifest_path=manifest_path, validate=validate)
        return JSONDataset(records)

    def build_dataloaders(
        self,
        batch_size: int = 64,
        cv: str = "loso",
        n_splits: int = 5,
        val_size: float = 0.2,
        test_size: float = 0.2,
        random_state: int = 42,
        num_workers: int = 4,
        pin_memory: bool = True,
    ) -> Iterable[Tuple[str, DataLoader, DataLoader, DataLoader]]:
        """
        Yields ``(fold_name, train_loader, val_loader, test_loader)`` tuples
        for the requested cross-validation scheme, operating directly on the
        lazy JSON dataset (no full in-RAM materialization).
        """
        full = self.load_json_dataset()
        records = full.records

        def _make_loaders(train_r, val_r, test_r, shuffle_train=True):
            train_ds = JSONDataset(train_r, expected_shape=full.expected_shape)
            val_ds = JSONDataset(val_r, expected_shape=full.expected_shape)
            test_ds = JSONDataset(test_r, expected_shape=full.expected_shape)
            return (
                build_dataloader(train_ds, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers, pin_memory=pin_memory),
                build_dataloader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory),
                build_dataloader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory),
            )

        if cv == "loso":
            for held_out, train_r, test_r in json_leave_one_subject_out(records):
                train_ids = sorted({r.subject_id for r in train_r})
                if len(train_ids) >= 2:
                    val_subj = train_ids[: max(1, int(round(len(train_ids) * val_size)))]
                    val_r = [r for r in train_r if r.subject_id in set(val_subj)]
                    train_r2 = [r for r in train_r if r.subject_id not in set(val_subj)]
                else:
                    val_r, train_r2 = [], train_r
                yield (f"loso_{held_out}",) + _make_loaders(train_r2, val_r, test_r)
        elif cv == "groupkfold":
            for i, (train_r, test_r) in enumerate(json_group_kfold_splits(records, n_splits=n_splits)):
                train_ids = sorted({r.subject_id for r in train_r})
                val_subj = train_ids[: max(1, int(round(len(train_ids) * val_size)))] if len(train_ids) >= 2 else []
                val_r = [r for r in train_r if r.subject_id in set(val_subj)]
                train_r2 = [r for r in train_r if r.subject_id not in set(val_subj)]
                yield (f"fold_{i+1}",) + _make_loaders(train_r2, val_r, test_r)
        else:
            train_r, val_r, test_r = json_holdout_split(records, test_size=test_size, val_size=val_size, random_state=random_state)
            yield ("holdout",) + _make_loaders(train_r, val_r, test_r)


def _load_legacy_dataset(data_dir: Path) -> List[SubjectDataset]:
    if not data_dir.exists():
        raise FileNotFoundError(f"{data_dir} does not exist")
    files = _candidate_data_files(data_dir)
    if not files:
        raise FileNotFoundError(f"No CSV/NPZ files found in {data_dir}")

    sample_level = [f for f in files if f.name.lower() == "sample_level_dataset.csv"]
    if sample_level:
        files = sample_level

    subjects: List[SubjectDataset] = []
    for path in files:
        try:
            subjects.extend(load_subject_file(path))
        except Exception as exc:
            subjects.append(
                SubjectDataset(
                    subject_id=path.stem,
                    X=np.empty((0, 0)),
                    y=np.empty((0,), dtype=int),
                    channel_names=[],
                    source_file=str(path),
                    metadata={"load_error": str(exc)},
                )
            )

    subjects = [s for s in subjects if s.X.size > 0 and len(s.y) > 0]
    if not subjects:
        raise FileNotFoundError(f"No usable datasets could be loaded from {data_dir}")
    return _merge_subject_datasets(subjects)


def load_subject_file(path: Path, preferred_label: Optional[str] = None) -> List[SubjectDataset]:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        return _load_npz(path)
    if suffix == ".csv":
        return _load_tabular_as_subjects(path, preferred_label=preferred_label)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def _candidate_data_files(data_dir: Path) -> List[Path]:
    files = []
    for p in data_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".csv", ".npz"}:
            files.append(p)
    priority = {"cycle_manifest.csv": 0, "sample_level_dataset.csv": 1, "phase2_labels.csv": 2, "phase4_labels.csv": 2, "phase7_labels.csv": 2}
    return sorted(files, key=lambda p: (priority.get(p.name.lower(), 99), str(p).lower()))


def _merge_subject_datasets(subjects: List[SubjectDataset]) -> List[SubjectDataset]:
    merged: Dict[str, List[SubjectDataset]] = {}
    for s in subjects:
        merged.setdefault(s.subject_id, []).append(s)

    out: List[SubjectDataset] = []
    for subject_id, items in merged.items():
        # Filter out zero-size items
        items = [x for x in items if x.X.size > 0 and x.X.ndim == 2 and x.X.shape[0] > 0 and x.X.shape[1] > 0 and len(x.y) > 0]
        if not items:
            continue

        if len(items) == 1:
            out.append(items[0])
            continue

        channel_sets = [tuple(x.channel_names) for x in items]
        if len(set(channel_sets)) == 1:
            channel_names = list(items[0].channel_names)
            X = np.vstack([x.X for x in items])
            y = np.concatenate([x.y for x in items])
            cycle_id = np.concatenate([x.cycle_id for x in items if x.cycle_id is not None]) if any(x.cycle_id is not None for x in items) else None
            gait_percent = np.concatenate([x.gait_percent for x in items if x.gait_percent is not None]) if any(x.gait_percent is not None for x in items) else None
            sample_index = np.concatenate([x.sample_index for x in items if x.sample_index is not None]) if any(x.sample_index is not None for x in items) else None
            metadata = {}
            for x in items:
                metadata.update(x.metadata)
            out.append(SubjectDataset(subject_id, X, y, channel_names, cycle_id, gait_percent, sample_index, items[0].source_file, metadata))
            continue

        common = list(set(items[0].channel_names).intersection(*[set(x.channel_names) for x in items[1:]]))
        common = sorted(common, key=lambda c: items[0].channel_names.index(c))
        if not common:
            # No common channels: use the first item only to avoid shape mismatch
            out.append(items[0])
            continue
        aligned_X = []
        aligned_y = []
        aligned_cycle = []
        aligned_gait = []
        aligned_idx = []
        metadata = {}
        for item in items:
            idx = [item.channel_names.index(c) for c in common]
            aligned_X.append(item.X[:, idx])
            aligned_y.append(item.y)
            if item.cycle_id is not None:
                aligned_cycle.append(item.cycle_id)
            if item.gait_percent is not None:
                aligned_gait.append(item.gait_percent)
            if item.sample_index is not None:
                aligned_idx.append(item.sample_index)
            metadata.update(item.metadata)
        X = np.vstack(aligned_X)
        y = np.concatenate(aligned_y)
        cycle_id = np.concatenate(aligned_cycle) if aligned_cycle else None
        gait_percent = np.concatenate(aligned_gait) if aligned_gait else None
        sample_index = np.concatenate(aligned_idx) if aligned_idx else None
        out.append(SubjectDataset(subject_id, X, y, common, cycle_id, gait_percent, sample_index, items[0].source_file, metadata))

    # Final invariant: verify len(X)==len(y) and consistent feature dims for all subjects
    feature_dims = set()
    for s in out:
        if s.X.shape[0] != len(s.y):
            raise ValueError(
                f"Subject {s.subject_id!r}: len(X)={s.X.shape[0]} != len(y)={len(s.y)}"
            )
        feature_dims.add(s.X.shape[1])
    if len(feature_dims) > 1:
        raise ValueError(
            f"Inconsistent feature dimensions across subjects: {feature_dims}. "
            "All subjects must have the same X.shape[1]."
        )

    return out


def load_dataset(data_dir: Path) -> List[SubjectDataset]:
    """
    Public, backward-compatible entry point. Auto-detects JSON-manifest,
    CSV, or NPZ format under ``data_dir`` and returns the legacy
    ``SubjectDataset`` in-memory representation consumed by pipeline.py.
    """
    factory = DatasetFactory(Path(data_dir))
    return factory.load_subject_datasets()


def make_forecast_target(y: np.ndarray, horizon_steps: int) -> np.ndarray:
    if horizon_steps < 0:
        raise ValueError("horizon_steps must be non-negative")
    if horizon_steps == 0:
        return y.copy()
    out = np.full_like(y, fill_value=-1)
    if horizon_steps < len(y):
        out[:-horizon_steps] = y[horizon_steps:]
    return out


def _subject_group_arrays(subjects: Sequence[SubjectDataset]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_all, y_all, gid_all = [], [], []
    for s in subjects:
        X_all.append(s.X)
        y_all.append(s.y)
        gid_all.append(np.full(len(s.y), s.subject_id, dtype=object))
    return np.vstack(X_all), np.concatenate(y_all), np.concatenate(gid_all)


def temporal_train_val_test_split(
    subjects: Sequence[SubjectDataset],
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
):
    subject_ids = np.array([s.subject_id for s in subjects], dtype=object)
    unique = np.array(sorted(set(subject_ids.tolist())), dtype=object)
    if len(unique) < 3:
        train_subjects, test_subjects = train_test_split(list(subjects), test_size=test_size, random_state=random_state, shuffle=True)
        train_subjects, val_subjects = train_test_split(train_subjects, test_size=val_size, random_state=random_state, shuffle=True)
        return list(train_subjects), list(val_subjects), list(test_subjects)

    train_ids, test_ids = train_test_split(unique, test_size=test_size, random_state=random_state, shuffle=True)
    train_subjects = [s for s in subjects if s.subject_id in set(train_ids.tolist())]
    test_subjects = [s for s in subjects if s.subject_id in set(test_ids.tolist())]
    train_ids2, val_ids = train_test_split(np.array([s.subject_id for s in train_subjects], dtype=object), test_size=val_size, random_state=random_state, shuffle=True)
    train_subjects = [s for s in train_subjects if s.subject_id in set(train_ids2.tolist())]
    val_subjects = [s for s in subjects if s.subject_id in set(val_ids.tolist())]
    return train_subjects, val_subjects, test_subjects


def leave_one_subject_out(subjects: Sequence[SubjectDataset]):
    unique = sorted(set(s.subject_id for s in subjects))
    for held_out in unique:
        train = [s for s in subjects if s.subject_id != held_out]
        test = [s for s in subjects if s.subject_id == held_out]
        yield held_out, train, test


def group_kfold_splits(subjects: Sequence[SubjectDataset], n_splits: int = 5, shuffle: bool = True, random_state: int = 42):
    groups = np.array([s.subject_id for s in subjects], dtype=object)
    if len(set(groups.tolist())) < n_splits:
        n_splits = max(2, len(set(groups.tolist())))
    splitter = GroupKFold(n_splits=n_splits)
    X_dummy = np.zeros((len(subjects), 1))
    for train_idx, test_idx in splitter.split(X_dummy, groups=groups):
        yield [subjects[i] for i in train_idx], [subjects[i] for i in test_idx]


def concatenate_subjects(subjects: Sequence[SubjectDataset]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_all, y_all, sid_all = [], [], []
    for s in subjects:
        X_all.append(s.X)
        y_all.append(s.y)
        sid_all.append(np.full(len(s.y), s.subject_id, dtype=object))
    return np.vstack(X_all), np.concatenate(y_all), np.concatenate(sid_all)
