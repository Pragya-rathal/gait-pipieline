
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import re

import numpy as np
import pandas as pd
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
    # numeric identifiers with many unique values but not likely EMG
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
    # favor low-cardinality non-EMG numeric or categorical columns
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
    # de-duplicate preserve order
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

    # Prefer standardized EMG channels for this dataset.
    for suffixes in (("_normalized",), ("_raw", "_bandpassed", "_rectified", "_envelope")):
        emg = [
            c for c in df.columns
            if _eligible(c) and any(_normalize_name(c).endswith(sfx) for sfx in suffixes)
        ]
        if emg:
            return emg

    # Fallback: any numeric feature that is not a known label/meta field.
    emg = []
    for c in df.columns:
        if not _eligible(c):
            continue
        if not _is_identifier_like(df[c]):
            emg.append(c)
    return emg


def _coerce_str_series(s: pd.Series) -> np.ndarray:
    return s.astype(str).to_numpy()


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
    return [SubjectDataset(subject_id=subject_id, X=X, y=_ensure_int_labels(y), channel_names=channel_names, cycle_id=cycle_id, gait_percent=gait_percent, sample_index=sample_index, source_file=str(path))]


def _ensure_int_labels(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if y.dtype.kind in {"U", "S", "O"}:
        _, inv = np.unique(y.astype(str), return_inverse=True)
        return inv.astype(int)
    if y.dtype.kind == "f":
        # labels should be discrete; coerce to int safely
        return np.rint(y).astype(int)
    return y.astype(int)


def _pick_label_column(df: pd.DataFrame, preferred: Optional[str] = None) -> Optional[str]:
    if preferred and preferred in df.columns:
        return preferred
    candidates = _candidate_label_columns(df)
    if not candidates:
        return None
    # prioritize common names and low cardinality
    def score(c: str) -> tuple[int, int, int]:
        nc = _normalize_name(c)
        exact = 0 if nc in KNOWN_LABEL_NAMES or nc.endswith("_label") or nc.endswith("_labels") else 1
        cardinality = _unique_nonnull_count(df[c])
        # lower is better for labels, but avoid trivial binary meta columns if possible
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

    # special case: if file is a manifest with file paths, resolve and load referenced files
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

    # generic sample-level/subject-level CSV
    label_vals = _ensure_int_labels(df[label_col].to_numpy())
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

    # some files may store multiple subjects; group by subject_id
    out: List[SubjectDataset] = []
    grouped = df.assign(_subject_id=subject_vals, _label=label_vals)
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
        cycle_id = subdf["_cycle_id"].to_numpy(dtype=object) if "_cycle_id" in subdf.columns else None
        gait_percent = subdf["_gait_percent"].to_numpy(dtype=float) if "_gait_percent" in subdf.columns else None
        sample_index = subdf["_sample_index"].to_numpy(dtype=int) if "_sample_index" in subdf.columns else None
        metadata = {
            "source_file": str(path),
            "label_column": label_col,
            "emg_columns": emg_cols,
            "all_columns": list(df.columns),
        }
        # attach any extra phase label columns for later use
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
    # prefer manifests first; loader will resolve referenced files if present
    priority = {"cycle_manifest.csv": 0, "sample_level_dataset.csv": 1, "phase2_labels.csv": 2, "phase4_labels.csv": 2, "phase7_labels.csv": 2}
    return sorted(files, key=lambda p: (priority.get(p.name.lower(), 99), str(p).lower()))


def _merge_subject_datasets(subjects: List[SubjectDataset]) -> List[SubjectDataset]:
    merged: Dict[str, List[SubjectDataset]] = {}
    for s in subjects:
        merged.setdefault(s.subject_id, []).append(s)

    out: List[SubjectDataset] = []
    for subject_id, items in merged.items():
        if len(items) == 1:
            out.append(items[0])
            continue

        # ensure same channel layout; if not, intersect common columns
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

        # align by common columns
        common = list(set(items[0].channel_names).intersection(*[set(x.channel_names) for x in items[1:]]))
        common = sorted(common, key=lambda c: items[0].channel_names.index(c))
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
    return out


def load_dataset(data_dir: Path) -> List[SubjectDataset]:
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
            # keep going for discovery; raise only if everything fails later
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
        # fallback: split samples within available subjects
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
