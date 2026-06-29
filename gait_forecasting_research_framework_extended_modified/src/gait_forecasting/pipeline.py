from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import json
import math
import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from .config import PipelineConfig
from .data import (
    SubjectDataset,
    group_kfold_splits,
    leave_one_subject_out,
    load_dataset,
    make_forecast_target,
    temporal_train_val_test_split,
)
from .deployment import deployment_table, summarize_deployment
from .evaluate import (
    Metrics,
    aggregate_metrics_across_folds,
    compute_metrics,
    metrics_dataframe,
    metrics_table,
    save_confusion_matrix,
    save_metrics_csv,
)
from .latent import fit_autoencoder_latent_state, fit_pca_latent_state
from .models import (
    _get_device,
    make_rf,
    make_sklearn_mlp,
    predict_torch,
    train_bilstm_classifier,
    train_gru_classifier,
    train_tcn_classifier,
    train_torch_mlp,
)
from .plots import (
    plot_deployment_comparison,
    plot_forecast_horizon,
    plot_latent_trajectories,
    plot_model_comparison,
    plot_pca_scatter,
    plot_synergy_activations,
    plot_umap_scatter,
    plot_vaf,
    plot_weights,
)
from .preprocessing import (
    WindowedDataset,
    build_horizon_steps,
    build_windowed_dataset,
    condition_emg,
    fit_scaler,
    ms_to_samples,
    transform_scaler,
)
from .state_space import (
    forecast_horizon_sequence,
    fit_linear_state_space_from_sequences,
)
from .research import adaptation_experiment, generate_research_artifacts
from .synergies import (
    NMFSynergyExtractor,
    compute_d2H,
    compute_dH,
    compute_synergy_state,
    choose_n_synergies,
)
from .utils import ensure_dir, save_json, set_seed


# ---------------------------------------------------------------------------
# Subject cloning helpers
# ---------------------------------------------------------------------------

def _clone_subject(
    subject: SubjectDataset,
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    suffix: str = "",
    source_file: Optional[str] = None,
    metadata_extra: Optional[Dict] = None,
) -> SubjectDataset:
    return SubjectDataset(
        subject_id=subject.subject_id,
        X=np.asarray(X, dtype=float),
        y=np.asarray(subject.y if y is None else y, dtype=int),
        channel_names=[
            f"{suffix}{c}" if suffix else c
            for c in range(X.shape[1] and len(subject.channel_names) or len(subject.channel_names))
        ],
        cycle_id=subject.cycle_id.copy() if subject.cycle_id is not None else None,
        gait_percent=subject.gait_percent.copy() if subject.gait_percent is not None else None,
        sample_index=subject.sample_index.copy() if subject.sample_index is not None else None,
        source_file=source_file or subject.source_file,
        metadata={**subject.metadata, **(metadata_extra or {})},
    )


def _clone_subject_with_channel_names(
    subject: SubjectDataset,
    X: np.ndarray,
    channel_names: Sequence[str],
    y: Optional[np.ndarray] = None,
    source_file: Optional[str] = None,
    metadata_extra: Optional[Dict] = None,
) -> SubjectDataset:
    return SubjectDataset(
        subject_id=subject.subject_id,
        X=np.asarray(X, dtype=float),
        y=np.asarray(subject.y if y is None else y, dtype=int),
        channel_names=list(channel_names),
        cycle_id=subject.cycle_id.copy() if subject.cycle_id is not None else None,
        gait_percent=subject.gait_percent.copy() if subject.gait_percent is not None else None,
        sample_index=subject.sample_index.copy() if subject.sample_index is not None else None,
        source_file=source_file or subject.source_file,
        metadata={**subject.metadata, **(metadata_extra or {})},
    )


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------

def _train_val_split_subjects(
    subjects: Sequence[SubjectDataset],
    val_size: float,
    random_state: int,
) -> Tuple[List[SubjectDataset], List[SubjectDataset]]:
    ids = np.array([s.subject_id for s in subjects], dtype=object)
    unique = np.array(sorted(set(ids.tolist())), dtype=object)
    if len(unique) <= 2:
        train, val = temporal_train_val_test_split(
            subjects,
            test_size=max(0.2, val_size),
            val_size=val_size,
            random_state=random_state,
        )[:2]
        return train, val
    from sklearn.model_selection import train_test_split
    train_ids, val_ids = train_test_split(
        unique, test_size=val_size, random_state=random_state, shuffle=True
    )
    train = [s for s in subjects if s.subject_id in set(train_ids.tolist())]
    val = [s for s in subjects if s.subject_id in set(val_ids.tolist())]
    return train, val


def _stack_subject_arrays(
    subjects: Sequence[SubjectDataset],
) -> Tuple[np.ndarray, np.ndarray]:
    X = np.vstack([s.X for s in subjects if len(s.X) > 0])
    y = np.concatenate([s.y for s in subjects if len(s.y) > 0]).astype(int)
    return X, y


# ---------------------------------------------------------------------------
# Representation fitting
# ---------------------------------------------------------------------------

def _fit_representations(
    train_subjects: Sequence[SubjectDataset],
    all_subjects: Sequence[SubjectDataset],
    cfg: PipelineConfig,
) -> Tuple[Dict[str, List[SubjectDataset]], Dict]:
    raw_train_X, _ = _stack_subject_arrays(train_subjects)
    raw_scaler = fit_scaler(raw_train_X) if cfg.normalize else None

    def condition(subject: SubjectDataset) -> np.ndarray:
        return condition_emg(subject.X, smooth=cfg.smooth, scaler=raw_scaler, rectify=False)

    conditioned_train = [condition(s) for s in train_subjects]
    conditioned_all = [condition(s) for s in all_subjects]

    nmf = NMFSynergyExtractor(
        n_synergies=cfg.synergy.n_synergies,
        max_iter=cfg.synergy.max_iter,
        random_state=cfg.synergy.random_state,
    )
    fit = nmf.fit_transform(np.vstack(conditioned_train))

    rep_subjects: Dict[str, List[SubjectDataset]] = {
        "raw": [],
        "H": [],
        "H_dH": [],
        "H_dH_d2H": [],
        "pca_latent": [],
        "ae_latent": [],
    }
    all_H: List[np.ndarray] = []
    all_z: List[np.ndarray] = []

    for subj, Xc in zip(all_subjects, conditioned_all):
        H = nmf.transform(Xc)
        dH = compute_dH(H)
        d2H = compute_d2H(H)
        z = compute_synergy_state(
            H, order=2 if cfg.use_d2h else 1 if cfg.use_dh else 0
        )
        all_H.append(H)
        all_z.append(z)

        n_syn = H.shape[1]
        h_names = [f"H{i+1}" for i in range(n_syn)]
        dh_names = [f"dH{i+1}" for i in range(n_syn)]
        d2h_names = [f"d2H{i+1}" for i in range(n_syn)]
        z_names = [f"z{i+1}" for i in range(z.shape[1])]

        rep_subjects["raw"].append(
            _clone_subject_with_channel_names(subj, Xc, subj.channel_names, metadata_extra={"representation": "raw_conditioned"})
        )
        rep_subjects["H"].append(
            _clone_subject_with_channel_names(subj, H, h_names, metadata_extra={"representation": "H"})
        )
        rep_subjects["H_dH"].append(
            _clone_subject_with_channel_names(subj, np.concatenate([H, dH], axis=1), h_names + dh_names, metadata_extra={"representation": "H_dH"})
        )
        rep_subjects["H_dH_d2H"].append(
            _clone_subject_with_channel_names(subj, np.concatenate([H, dH, d2H], axis=1), h_names + dh_names + d2h_names, metadata_extra={"representation": "H_dH_d2H"})
        )
        # placeholders — overwritten below after latent encoders are ready
        rep_subjects["pca_latent"].append(
            _clone_subject_with_channel_names(subj, z, z_names, metadata_extra={"representation": "synergy_state"})
        )
        rep_subjects["ae_latent"].append(
            _clone_subject_with_channel_names(subj, z, z_names, metadata_extra={"representation": "synergy_state"})
        )

    train_ids = {s.subject_id for s in train_subjects}
    z_train = np.vstack([z for s, z in zip(all_subjects, all_z) if s.subject_id in train_ids])

    pca = fit_pca_latent_state(
        z_train,
        n_components=min(cfg.model.ae_latent_dim, z_train.shape[1]),
        random_state=cfg.random_state,
    )

    ae_epochs = max(3, min(8, cfg.model.epochs))
    rng = np.random.default_rng(cfg.random_state)
    if len(z_train) > 2000:
        idx = rng.choice(len(z_train), size=2000, replace=False)
        z_train_ae = z_train[idx]
    else:
        z_train_ae = z_train

    ae = fit_autoencoder_latent_state(
        z_train_ae,
        latent_dim=min(cfg.model.ae_latent_dim, z_train.shape[1]),
        hidden_sizes=cfg.model.ae_hidden_sizes,
        epochs=ae_epochs,
        batch_size=min(cfg.model.batch_size, 128),
        learning_rate=cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay,
        patience=max(2, cfg.model.patience),
        random_state=cfg.random_state,
    )

    # Overwrite pca/ae placeholder subjects
    for i, subj in enumerate(all_subjects):
        z = all_z[i]
        zp = pca.transform(z)
        za = ae.transform(z)
        rep_subjects["pca_latent"][i] = _clone_subject_with_channel_names(
            subj, zp, [f"pca{j+1}" for j in range(zp.shape[1])], metadata_extra={"representation": "pca_latent"}
        )
        rep_subjects["ae_latent"][i] = _clone_subject_with_channel_names(
            subj, za, [f"ae{j+1}" for j in range(za.shape[1])], metadata_extra={"representation": "ae_latent"}
        )

    fit_objects = {
        "raw_scaler": raw_scaler,
        "nmf": nmf,
        "nmf_fit": fit,
        "pca": pca,
        "ae": ae,
        "all_z": all_z,
    }
    return rep_subjects, fit_objects


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

def _window_subjects(
    subjects: Sequence[SubjectDataset],
    window_ms: int,
    horizon_ms: int,
    cfg: PipelineConfig,
) -> WindowedDataset:
    window_size = ms_to_samples(window_ms, cfg.windows.sample_rate_hz)
    horizon_steps = build_horizon_steps(horizon_ms, cfg.windows.sample_rate_hz)
    return build_windowed_dataset(
        subjects,
        window_size=window_size,
        horizon_steps=horizon_steps,
        overlap=cfg.windows.overlap,
        use_center_label=cfg.windows.use_center_label,
    )


# ---------------------------------------------------------------------------
# Model training helpers
# ---------------------------------------------------------------------------

def _train_torch_model(
    model_kind: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
    cfg: PipelineConfig,
    seq: bool = False,
):
    label_set = sorted(np.unique(np.concatenate([y_train, y_val])).tolist())
    label_to_idx = {lab: i for i, lab in enumerate(label_set)}
    idx_to_label = {i: lab for lab, i in label_to_idx.items()}
    y_train_i = np.array([label_to_idx[v] for v in y_train], dtype=int)
    y_val_i = np.array([label_to_idx[v] for v in y_val], dtype=int)

    common_kwargs = dict(
        batch_size=cfg.model.batch_size,
        epochs=cfg.model.epochs,
        learning_rate=cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay,
        patience=cfg.model.patience,
        random_state=cfg.random_state,
    )

    if model_kind == "mlp":
        tr = train_torch_mlp(
            X_train, y_train_i, X_val, y_val_i,
            input_dim=X_train.shape[1], n_classes=len(label_set),
            hidden_sizes=cfg.model.hidden_sizes, dropout=cfg.model.dropout,
            **common_kwargs,
        )
    elif model_kind == "gru":
        tr = train_gru_classifier(
            X_train, y_train_i, X_val, y_val_i,
            input_dim=X_train.shape[2], n_classes=len(label_set),
            hidden_size=cfg.model.gru_hidden_size, num_layers=cfg.model.gru_layers,
            dropout=cfg.model.dropout, **common_kwargs,
        )
    elif model_kind == "bilstm":
        tr = train_bilstm_classifier(
            X_train, y_train_i, X_val, y_val_i,
            input_dim=X_train.shape[2], n_classes=len(label_set),
            hidden_size=cfg.model.lstm_hidden_size, num_layers=cfg.model.lstm_layers,
            dropout=cfg.model.dropout, **common_kwargs,
        )
    elif model_kind == "tcn":
        tr = train_tcn_classifier(
            X_train, y_train_i, X_val, y_val_i,
            input_dim=X_train.shape[2], n_classes=len(label_set),
            channels=cfg.model.tcn_channels, kernel_size=cfg.model.tcn_kernel_size,
            dropout=cfg.model.dropout, **common_kwargs,
        )
    else:
        raise ValueError(model_kind)

    y_pred_i = predict_torch(tr.model, X_val)
    y_pred = np.array([idx_to_label[i] for i in y_pred_i], dtype=int)
    return tr, y_pred, label_set


def _fit_predict_sklearn_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    cfg: PipelineConfig,
):
    model = make_sklearn_mlp(cfg.model.hidden_sizes, random_state=cfg.random_state)
    model.fit(X_train, y_train)
    return model, model.predict(X_eval)


def _fit_predict_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    cfg: PipelineConfig,
):
    model = make_rf(cfg.random_state)
    model.fit(X_train, y_train)
    return model, model.predict(X_eval)


def _fit_state_space_model(
    train_subjects: Sequence[SubjectDataset],
    cfg: PipelineConfig,
    rep_subjects: Dict[str, List[SubjectDataset]],
):
    train_ids = {s.subject_id for s in train_subjects}
    z_train = [s.X for s in rep_subjects["H_dH_d2H"] if s.subject_id in train_ids]
    raw_train = [s.X for s in rep_subjects["raw"] if s.subject_id in train_ids]
    return fit_linear_state_space_from_sequences(
        z_train, inputs_list=raw_train, ridge=1e-5, include_bias=True
    )


def _forecast_subjects(
    rep_subjects: Dict[str, List[SubjectDataset]],
    train_subjects: Sequence[SubjectDataset],
    state_model,
    horizon_steps: int,
) -> List[SubjectDataset]:
    forecasted: List[SubjectDataset] = []
    for raw_subj, z_subj in zip(rep_subjects["raw"], rep_subjects["H_dH_d2H"]):
        z_pred = forecast_horizon_sequence(
            state_model, z_subj.X, raw_subj.X, horizon_steps=horizon_steps
        )
        y_future = make_forecast_target(raw_subj.y, horizon_steps)
        valid = ~np.isnan(z_pred).any(axis=1)
        z_pred = z_pred[valid]
        y_future = y_future[valid]
        forecasted.append(
            _clone_subject_with_channel_names(
                raw_subj,
                z_pred,
                [f"state{i+1}" for i in range(z_pred.shape[1])],
                y=y_future,
                metadata_extra={"representation": "state_space_forecast"},
            )
        )
    return forecasted


# ---------------------------------------------------------------------------
# Model evaluation bundle
# ---------------------------------------------------------------------------

def _evaluate_model_bundle(
    model_name: str,
    feature_kind: str,
    train_win: WindowedDataset,
    val_win: WindowedDataset,
    test_win: WindowedDataset,
    cfg: PipelineConfig,
    labels: List[int],
    out_dir: Path,
):
    """Returns (metrics, trained_model, deployment_metrics)."""
    if feature_kind == "sequence":
        X_train, X_val, X_test = train_win.X_seq, val_win.X_seq, test_win.X_seq
    else:
        X_train, X_val, X_test = train_win.X_flat, val_win.X_flat, test_win.X_flat

    # ---- Classical baselines (CPU sklearn) --------------------------------
    if model_name == "baseline_rf_raw":
        model, y_pred = _fit_predict_rf(X_train, train_win.y, X_test, cfg)
        metrics = compute_metrics(test_win.y, y_pred, labels=labels)
        deployment = summarize_deployment(model, X_test[: min(8, len(X_test))])
        return metrics, model, deployment

    if model_name == "baseline_mlp_raw":
        model, y_pred = _fit_predict_sklearn_mlp(X_train, train_win.y, X_test, cfg)
        metrics = compute_metrics(test_win.y, y_pred, labels=labels)
        deployment = summarize_deployment(model, X_test[: min(8, len(X_test))])
        return metrics, model, deployment

    # ---- Torch models: build zero-based label mappings --------------------
    torch_labels = sorted(
        np.unique(np.concatenate([train_win.y, val_win.y, test_win.y])).tolist()
    )
    label_to_idx = {lab: i for i, lab in enumerate(torch_labels)}
    idx_to_label = {i: lab for lab, i in label_to_idx.items()}
    y_train_i = np.array([label_to_idx[v] for v in train_win.y], dtype=int)
    y_val_i = np.array([label_to_idx[v] for v in val_win.y], dtype=int)
    y_test_i = np.array([label_to_idx[v] for v in test_win.y], dtype=int)

    # ---- Sequence & MLP torch models with StandardScaler ------------------
    if model_name in {"raw_emg_gru", "raw_emg_bilstm", "raw_emg_tcn", "forecasted_state_mlp"}:
        if feature_kind == "sequence":
            flat_train = X_train.reshape(len(X_train), -1)
            scaler = StandardScaler().fit(flat_train)
            X_train_s = scaler.transform(flat_train).reshape(X_train.shape)
            X_val_s = scaler.transform(X_val.reshape(len(X_val), -1)).reshape(X_val.shape)
            X_test_s = scaler.transform(X_test.reshape(len(X_test), -1)).reshape(X_test.shape)
        else:
            scaler = StandardScaler().fit(X_train)
            X_train_s = scaler.transform(X_train)
            X_val_s = scaler.transform(X_val)
            X_test_s = scaler.transform(X_test)

        common = dict(
            batch_size=cfg.model.batch_size,
            epochs=cfg.model.epochs,
            learning_rate=cfg.model.learning_rate,
            weight_decay=cfg.model.weight_decay,
            patience=cfg.model.patience,
            random_state=cfg.random_state,
        )

        if model_name == "raw_emg_gru":
            tr = train_gru_classifier(
                X_train_s, y_train_i, X_val_s, y_val_i,
                input_dim=X_train_s.shape[2], n_classes=len(torch_labels),
                hidden_size=cfg.model.gru_hidden_size, num_layers=cfg.model.gru_layers,
                dropout=cfg.model.dropout, **common,
            )
        elif model_name == "raw_emg_bilstm":
            tr = train_bilstm_classifier(
                X_train_s, y_train_i, X_val_s, y_val_i,
                input_dim=X_train_s.shape[2], n_classes=len(torch_labels),
                hidden_size=cfg.model.lstm_hidden_size, num_layers=cfg.model.lstm_layers,
                dropout=cfg.model.dropout, **common,
            )
        elif model_name == "raw_emg_tcn":
            tr = train_tcn_classifier(
                X_train_s, y_train_i, X_val_s, y_val_i,
                input_dim=X_train_s.shape[2], n_classes=len(torch_labels),
                channels=cfg.model.tcn_channels, kernel_size=cfg.model.tcn_kernel_size,
                dropout=cfg.model.dropout, **common,
            )
        else:  # forecasted_state_mlp
            tr = train_torch_mlp(
                X_train_s, y_train_i, X_val_s, y_val_i,
                input_dim=X_train_s.shape[1], n_classes=len(torch_labels),
                hidden_sizes=cfg.model.hidden_sizes, dropout=cfg.model.dropout, **common,
            )

        y_pred_i = predict_torch(tr.model, X_test_s)
        y_pred = np.array([idx_to_label[i] for i in y_pred_i], dtype=int)
        metrics = compute_metrics(test_win.y, y_pred, labels=labels)
        deployment = summarize_deployment(tr.model, X_test_s[: min(8, len(X_test_s))])
        return metrics, tr.model, deployment

    # ---- Tabular latent models --------------------------------------------
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    model, y_pred = _fit_predict_sklearn_mlp(X_train_s, train_win.y, X_test_s, cfg)
    metrics = compute_metrics(test_win.y, y_pred, labels=labels)
    deployment = summarize_deployment(model, X_test_s[: min(8, len(X_test_s))])
    return metrics, model, deployment


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(cfg: PipelineConfig) -> Dict[str, object]:
    set_seed(cfg.random_state)
    ensure_dir(cfg.output_dir)

    if cfg.demo:
        from .synthetic import generate_demo_dataset
        data_dir = ensure_dir(cfg.data_dir)
        generate_demo_dataset(data_dir, n_subjects=6, random_state=cfg.random_state)

    subjects = load_dataset(cfg.data_dir)
    labels = sorted(set(np.concatenate([s.y for s in subjects]).astype(int).tolist()))

    summary_rows: List[Dict] = []
    deployment_rows: List[Dict] = []
    metrics_by_fold: List[Dict] = []
    results_nested: Dict[str, dict] = {}
    representative_model_bundle: Optional[Dict[str, object]] = None
    rep_subjects: Optional[Dict[str, List[SubjectDataset]]] = None
    fit_objs: Optional[Dict] = None

    # ---- Cross-subject splits --------------------------------------------
    cv = cfg.eval.cross_validation.lower()
    if cv == "loso":
        folds = list(leave_one_subject_out(subjects))
        split_mode = "loso"
    elif cv == "groupkfold":
        folds = [
            (f"fold_{i+1}", train, test)
            for i, (train, test) in enumerate(
                group_kfold_splits(subjects, n_splits=cfg.eval.n_splits)
            )
        ]
        split_mode = "groupkfold"
    else:
        train_subjects, val_subjects, test_subjects = temporal_train_val_test_split(
            subjects,
            test_size=cfg.eval.test_size,
            val_size=cfg.eval.val_size,
            random_state=cfg.eval.random_state,
        )
        folds = [("holdout", train_subjects, test_subjects)]
        split_mode = "holdout"

    for fold_idx, fold in enumerate(folds):
        if cv == "loso":
            held_out, train_subjects, test_subjects = fold
            train_subjects, val_subjects = _train_val_split_subjects(
                train_subjects, val_size=cfg.eval.val_size, random_state=cfg.eval.random_state
            )
            fold_name = f"loso_{held_out}"
        elif cv == "groupkfold":
            fold_name, train_subjects, test_subjects = fold
            train_subjects, val_subjects = _train_val_split_subjects(
                train_subjects, val_size=cfg.eval.val_size, random_state=cfg.eval.random_state
            )
        else:
            fold_name = "holdout"
            train_subjects, val_subjects, test_subjects = temporal_train_val_test_split(
                subjects,
                test_size=cfg.eval.test_size,
                val_size=cfg.eval.val_size,
                random_state=cfg.eval.random_state,
            )

        fold_dir = ensure_dir(cfg.output_dir / fold_name)
        rep_subjects, fit_objs = _fit_representations(train_subjects, subjects, cfg)
        nmf_fit = fit_objs["nmf_fit"]

        plot_vaf(
            [(cfg.synergy.n_synergies, nmf_fit.vaf)],
            cfg.synergy.n_synergies,
            cfg.synergy.threshold,
            fold_dir / "vaf.png",
        )
        plot_synergy_activations(
            fit_objs["nmf_fit"].H[: min(1000, len(fit_objs["nmf_fit"].H))],
            fold_dir / "synergy_activations.png",
        )
        plot_weights(
            fit_objs["nmf_fit"].W,
            train_subjects[0].channel_names,
            fold_dir / "synergy_weights.png",
        )

        train_ids = {t.subject_id for t in train_subjects}
        pca_train_X = np.vstack(
            [s.X for s in rep_subjects["H_dH_d2H"] if s.subject_id in train_ids]
        )
        plot_pca_scatter(fit_objs["pca"].transform(pca_train_X), None, fold_dir / "pca_train.png")
        plot_umap_scatter(pca_train_X, None, fold_dir / "umap_train.png")

        # Fit linear state-space on training subjects
        state_model = fit_linear_state_space_from_sequences(
            [s.X for s in rep_subjects["H_dH_d2H"] if s.subject_id in train_ids],
            inputs_list=[s.X for s in rep_subjects["raw"] if s.subject_id in train_ids],
            ridge=1e-5,
            include_bias=True,
        )

        fold_results: Dict[str, Metrics] = {}
        fold_deploy: Dict[str, object] = {}

        for window_ms in cfg.windows.window_ms:
            for horizon_ms in cfg.windows.forecast_ms:
                window_dir = ensure_dir(
                    fold_dir / f"window_{window_ms}ms" / f"horizon_{horizon_ms}ms"
                )

                val_ids = {s.subject_id for s in val_subjects}
                test_ids = {s.subject_id for s in test_subjects}

                def _select(rep_name: str, ids: set) -> List[SubjectDataset]:
                    return [s for s in rep_subjects[rep_name] if s.subject_id in ids]

                current_train_raw = _select("raw", train_ids)
                current_val_raw = _select("raw", val_ids)
                current_test_raw = _select("raw", test_ids)

                current_train_h = _select("H", train_ids)
                current_val_h = _select("H", val_ids)
                current_test_h = _select("H", test_ids)

                current_train_hdH = _select("H_dH", train_ids)
                current_val_hdH = _select("H_dH", val_ids)
                current_test_hdH = _select("H_dH", test_ids)

                current_train_hdH2 = _select("H_dH_d2H", train_ids)
                current_val_hdH2 = _select("H_dH_d2H", val_ids)
                current_test_hdH2 = _select("H_dH_d2H", test_ids)

                current_train_pca = _select("pca_latent", train_ids)
                current_val_pca = _select("pca_latent", val_ids)
                current_test_pca = _select("pca_latent", test_ids)

                current_train_ae = _select("ae_latent", train_ids)
                current_val_ae = _select("ae_latent", val_ids)
                current_test_ae = _select("ae_latent", test_ids)

                # Build forecasted latent states
                horizon_steps_now = build_horizon_steps(horizon_ms, cfg.windows.sample_rate_hz)

                def _build_forecast(subset: Sequence[SubjectDataset]) -> List[SubjectDataset]:
                    out: List[SubjectDataset] = []
                    for subj in subset:
                        raw_subj = next(
                            s for s in rep_subjects["raw"] if s.subject_id == subj.subject_id
                        )
                        z_subj = next(
                            s for s in rep_subjects["H_dH_d2H"] if s.subject_id == subj.subject_id
                        )
                        z_pred = forecast_horizon_sequence(
                            state_model, z_subj.X, raw_subj.X, horizon_steps=horizon_steps_now
                        )
                        y_future = make_forecast_target(subj.y, horizon_steps_now)
                        valid = ~np.isnan(z_pred).any(axis=1)
                        out.append(
                            _clone_subject_with_channel_names(
                                subj,
                                z_pred[valid],
                                [f"state{i+1}" for i in range(z_pred.shape[1])],
                                y=y_future[valid],
                                metadata_extra={"representation": "forecasted_state"},
                            )
                        )
                    return out

                forecast_train = _build_forecast(train_subjects)
                forecast_val = _build_forecast(val_subjects)
                forecast_test = _build_forecast(test_subjects)

                reps_for_windows = {
                    "raw": (current_train_raw, current_val_raw, current_test_raw, "sequence"),
                    "H": (current_train_h, current_val_h, current_test_h, "tabular"),
                    "H_dH": (current_train_hdH, current_val_hdH, current_test_hdH, "tabular"),
                    "H_dH_d2H": (current_train_hdH2, current_val_hdH2, current_test_hdH2, "tabular"),
                    "pca_latent": (current_train_pca, current_val_pca, current_test_pca, "tabular"),
                    "ae_latent": (current_train_ae, current_val_ae, current_test_ae, "tabular"),
                    "forecasted_state": (forecast_train, forecast_val, forecast_test, "tabular"),
                }

                model_specs = [
                    ("baseline_rf_raw", "raw", "tabular"),
                    ("baseline_mlp_raw", "raw", "tabular"),
                    ("raw_emg_gru", "raw", "sequence"),
                    ("raw_emg_bilstm", "raw", "sequence"),
                    ("raw_emg_tcn", "raw", "sequence"),
                    ("synergy_h_mlp", "H", "tabular"),
                    ("synergy_hdH_mlp", "H_dH", "tabular"),
                    ("synergy_hdH2_mlp", "H_dH_d2H", "tabular"),
                    ("pca_latent_mlp", "pca_latent", "tabular"),
                    ("ae_latent_mlp", "ae_latent", "tabular"),
                    ("forecasted_state_mlp", "forecasted_state", "tabular"),
                ]

                for model_name, rep_name, feature_kind in model_specs:
                    tr_subj, va_subj, te_subj, _ = reps_for_windows[rep_name]
                    horizon_arg = horizon_ms if rep_name != "forecasted_state" else 0
                    train_win = _window_subjects(tr_subj, window_ms, horizon_arg, cfg)
                    val_win = _window_subjects(va_subj, window_ms, horizon_arg, cfg)
                    test_win = _window_subjects(te_subj, window_ms, horizon_arg, cfg)

                    if not (len(train_win.y) and len(val_win.y) and len(test_win.y)):
                        continue

                    metrics, trained_model, deployment = _evaluate_model_bundle(
                        model_name, feature_kind, train_win, val_win, test_win, cfg, labels, window_dir
                    )

                    key = f"{window_ms}ms/{horizon_ms}ms/{model_name}"
                    fold_results[key] = metrics
                    fold_deploy[key] = deployment

                    summary_rows.append({
                        "fold": fold_name,
                        "window_ms": window_ms,
                        "horizon_ms": horizon_ms,
                        "model": model_name,
                        "accuracy": metrics.accuracy,
                        "macro_f1": metrics.macro_f1,
                        "weighted_f1": metrics.weighted_f1,
                        "n_train": int(len(train_win.y)),
                        "n_val": int(len(val_win.y)),
                        "n_test": int(len(test_win.y)),
                    })
                    deployment_rows.append({
                        "fold": fold_name,
                        "window_ms": window_ms,
                        "horizon_ms": horizon_ms,
                        "model": model_name,
                        "params": deployment.params,
                        "memory_bytes": deployment.memory_bytes,
                        "latency_ms": deployment.latency_ms,
                        "flops": deployment.flops,
                    })
                    save_confusion_matrix(
                        metrics.confusion,
                        labels,
                        f"{model_name} - {window_ms}ms - {horizon_ms}ms",
                        window_dir / f"{model_name}_cm.png",
                    )

                    # Save model artifacts
                    try:
                        if hasattr(trained_model, "state_dict"):
                            import torch as _torch
                            _torch.save(
                                trained_model.state_dict(),
                                window_dir / f"{model_name}.pt",
                            )
                        else:
                            joblib.dump(trained_model, window_dir / f"{model_name}.joblib")
                    except Exception:
                        pass

                    # Capture representative model for research validation (fold 0 only)
                    if (
                        representative_model_bundle is None
                        and fold_idx == 0
                        and window_ms == cfg.windows.window_ms[1 if len(cfg.windows.window_ms) > 1 else 0]
                        and horizon_ms == cfg.windows.forecast_ms[1 if len(cfg.windows.forecast_ms) > 1 else 0]
                        and model_name == "synergy_hdH2_mlp"
                    ):
                        try:
                            train_rep = [
                                s for s in rep_subjects["H_dH_d2H"] if s.subject_id in train_ids
                            ]
                            test_rep = (
                                next(
                                    (s for s in rep_subjects["H_dH_d2H"] if s.subject_id == test_subjects[0].subject_id),
                                    None,
                                )
                                if test_subjects
                                else None
                            )
                            adaptation_df = (
                                adaptation_experiment(
                                    train_rep,
                                    test_rep,
                                    window_ms=window_ms,
                                    horizon_ms=horizon_ms,
                                    cfg=cfg,
                                    adaptation_cycles=(0, 1, 5, 10),
                                    random_state=cfg.random_state,
                                )
                                if train_rep and test_rep is not None
                                else pd.DataFrame()
                            )
                            actual_latent = None
                            forecasted_latent = None
                            if test_subjects:
                                raw_subj = next(
                                    (s for s in rep_subjects["raw"] if s.subject_id == test_subjects[0].subject_id),
                                    None,
                                )
                                z_subj = next(
                                    (s for s in rep_subjects["H_dH_d2H"] if s.subject_id == test_subjects[0].subject_id),
                                    None,
                                )
                                if raw_subj is not None and z_subj is not None:
                                    z_pred = forecast_horizon_sequence(
                                        state_model,
                                        z_subj.X,
                                        raw_subj.X,
                                        horizon_steps=build_horizon_steps(horizon_ms, cfg.windows.sample_rate_hz),
                                    )
                                    valid = ~np.isnan(z_pred).any(axis=1)
                                    actual_latent = z_subj.X[valid]
                                    forecasted_latent = z_pred[valid]

                            representative_model_bundle = {
                                "model": trained_model,
                                "X_train": train_win.X_flat,
                                "feature_names": [f"f{i+1}" for i in range(train_win.X_flat.shape[1])],
                                "adaptation_df": adaptation_df,
                                "actual_latent": actual_latent,
                                "forecasted_latent": forecasted_latent,
                            }
                        except Exception:
                            pass

        metrics_by_fold.append(fold_results)
        results_nested[fold_name] = {
            k: {"accuracy": v.accuracy, "macro_f1": v.macro_f1, "weighted_f1": v.weighted_f1}
            for k, v in fold_results.items()
        }

    # ---- Aggregate and save outputs --------------------------------------
    metrics_agg = aggregate_metrics_across_folds(metrics_by_fold) if metrics_by_fold else {}
    metrics_df = pd.DataFrame(summary_rows)
    deploy_df = pd.DataFrame(deployment_rows)

    save_json(cfg.output_dir / "results.json", results_nested)
    save_json(
        cfg.output_dir / "summary.json",
        {
            "n_subjects": len(subjects),
            "subjects": [s.subject_id for s in subjects],
            "labels": labels,
            "config": asdict(cfg),
            "aggregate_metrics": metrics_agg,
        },
    )
    metrics_df.to_csv(cfg.output_dir / "metrics_summary.csv", index=False)
    deploy_df.to_csv(cfg.output_dir / "deployment_summary.csv", index=False)

    _metrics_map = {
        f"{r['fold']}/{r['window_ms']}ms/{r['horizon_ms']}ms/{r['model']}": Metrics(
            r["accuracy"], r["macro_f1"], r["weighted_f1"], {}, np.array([])
        )
        for r in summary_rows
    }
    (cfg.output_dir / "metrics.md").write_text(metrics_table(_metrics_map), encoding="utf-8")
    (cfg.output_dir / "deployment.md").write_text(
        deployment_table(
            {
                f"{r['fold']}/{r['window_ms']}ms/{r['horizon_ms']}ms/{r['model']}": type("D", (), r)()
                for r in deployment_rows
            }
        )
        if deployment_rows
        else "",
        encoding="utf-8",
    )
    save_metrics_csv(_metrics_map, cfg.output_dir / "metrics_compact.csv")

    # ---- Plots -----------------------------------------------------------
    if summary_rows:
        plot_model_comparison(
            metrics_df[
                metrics_df["model"].isin([
                    "baseline_mlp_raw",
                    "synergy_h_mlp",
                    "synergy_hdH_mlp",
                    "synergy_hdH2_mlp",
                    "pca_latent_mlp",
                    "ae_latent_mlp",
                    "forecasted_state_mlp",
                ])
            ],
            "macro_f1",
            cfg.output_dir / "model_comparison.png",
        )
        plot_deployment_comparison(
            deploy_df.drop_duplicates("model")[["model", "params", "latency_ms", "flops"]],
            cfg.output_dir / "deployment_comparison.png",
        )
        horizon_results: Dict[str, Dict] = {}
        for r in summary_rows:
            key = f"{r['horizon_ms']}ms"
            horizon_results.setdefault(key, {})
            horizon_results[key].setdefault(r["model"], {})
            horizon_results[key][r["model"]]["macro_f1"] = r["macro_f1"]
            horizon_results[key][r["model"]]["accuracy"] = r["accuracy"]
        plot_forecast_horizon(
            horizon_results, "macro_f1", cfg.output_dir / "forecast_horizon_macro_f1.png"
        )

    summary = {
        "n_subjects": len(subjects),
        "labels": labels,
        "cross_validation": cfg.eval.cross_validation,
        "results_file": str(cfg.output_dir / "results.json"),
        "metrics_csv": str(cfg.output_dir / "metrics_summary.csv"),
        "deployment_csv": str(cfg.output_dir / "deployment_summary.csv"),
        "aggregate_metrics": metrics_agg,
    }

    # ---- Research-grade validation and manuscript assets -----------------
    try:
        generate_research_artifacts(
            cfg.output_dir,
            cfg,
            subjects=subjects,
            rep_subjects=rep_subjects,
            fit_objects=fit_objs,
            metrics_df=metrics_df,
            deploy_df=deploy_df,
            representative_model_bundle=representative_model_bundle,
        )
    except Exception as exc:
        save_json(cfg.output_dir / "research_artifacts_error.json", {"error": str(exc)})

    save_json(cfg.output_dir / "final_summary.json", summary)
    return summary
