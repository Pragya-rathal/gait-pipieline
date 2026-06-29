
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple
import copy
import json
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.stats import ttest_rel, wilcoxon, sem, t
from sklearn.inspection import permutation_importance
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils import resample

from .config import PipelineConfig
from .deployment import summarize_deployment
from .evaluate import Metrics
from .latent import build_latent_state
from .models import TorchMLP, predict_torch, train_torch_mlp
from .plots import _finalize
from .preprocessing import build_horizon_steps, build_windowed_dataset
from .state_space import forecast_latent_states, fit_linear_state_space_from_sequences
from .synergies import NMFSynergyExtractor, choose_n_synergies, compute_d2H, compute_dH, variance_accounted_for
from .utils import ensure_dir, save_json


# -----------------------------
# small helpers
# -----------------------------
def _safe_labels(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    return y[np.isfinite(y)] if y.dtype.kind == "f" else y


def _confidence_interval(x: Sequence[float], confidence: float = 0.95) -> Tuple[float, float]:
    arr = np.asarray(list(x), dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return (float("nan"), float("nan"))
    if len(arr) == 1:
        return (float(arr[0]), float(arr[0]))
    m = float(np.mean(arr))
    s = float(sem(arr))
    h = float(t.ppf((1 + confidence) / 2.0, len(arr) - 1) * s)
    return (m - h, m + h)


def _bootstrap_ci(x: Sequence[float], confidence: float = 0.95, n_boot: int = 2000, random_state: int = 42) -> Tuple[float, float]:
    arr = np.asarray(list(x), dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(random_state)
    boots = [float(np.mean(rng.choice(arr, size=len(arr), replace=True))) for _ in range(n_boot)]
    alpha = (1.0 - confidence) / 2.0
    return (float(np.quantile(boots, alpha)), float(np.quantile(boots, 1.0 - alpha)))


def _to_2d_rows(W: np.ndarray) -> np.ndarray:
    W = np.asarray(W, dtype=float)
    if W.ndim != 2:
        raise ValueError("Expected 2D matrix")
    return W


# -----------------------------
# Synergy validation
# -----------------------------
def cosine_similarity_matrix(W: np.ndarray) -> np.ndarray:
    W = _to_2d_rows(W)
    Wn = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)
    S = cosine_similarity(Wn)
    return np.clip(S, -1.0, 1.0)


def match_synergies(W_ref: np.ndarray, W_other: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Match rows of W_other to W_ref by maximal absolute cosine similarity.
    Returns (matched_similarity_vector, permuted_other, assignment)
    """
    W_ref = _to_2d_rows(W_ref)
    W_other = _to_2d_rows(W_other)
    S = np.abs(cosine_similarity_matrix(np.vstack([W_ref, W_other]))[: len(W_ref), len(W_ref):])
    row_ind, col_ind = linear_sum_assignment(-S)
    matched = S[row_ind, col_ind]
    permuted = W_other[col_ind]
    return matched, permuted, col_ind


def subject_to_subject_synergy_similarity(
    subjects: Sequence[Any],
    n_synergies: int,
    max_iter: int = 1000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Fit NMF per subject and return a pairwise synergy similarity matrix.
    """
    models = []
    names = []
    for i, subj in enumerate(subjects):
        if getattr(subj, "X", None) is None or len(subj.X) < 2:
            continue
        nmf = NMFSynergyExtractor(n_synergies=n_synergies, max_iter=max_iter, random_state=random_state + i)
        fit = nmf.fit_transform(np.asarray(subj.X, dtype=float))
        models.append(fit.W)
        names.append(str(subj.subject_id))
    if not models:
        return pd.DataFrame()
    n = len(models)
    sim = np.eye(n, dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            matched, _, _ = match_synergies(models[i], models[j])
            val = float(np.mean(matched))
            sim[i, j] = sim[j, i] = val
    return pd.DataFrame(sim, index=names, columns=names)


def cycle_to_cycle_synergy_similarity(
    subject: Any,
    n_synergies: int,
    max_iter: int = 1000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Similarity of cycle-wise mean latent activation vectors.
    """
    if getattr(subject, "cycle_id", None) is None:
        return pd.DataFrame()
    X = np.asarray(subject.X, dtype=float)
    cycle_ids = np.asarray(subject.cycle_id)
    nmf = NMFSynergyExtractor(n_synergies=n_synergies, max_iter=max_iter, random_state=random_state)
    H = nmf.fit_transform(X).H
    cycle_order = []
    cycle_vecs = []
    for c in pd.unique(cycle_ids):
        idx = np.where(cycle_ids == c)[0]
        if len(idx) == 0:
            continue
        cycle_order.append(str(c))
        cycle_vecs.append(np.mean(H[idx], axis=0))
    if len(cycle_vecs) < 2:
        return pd.DataFrame()
    V = np.vstack(cycle_vecs)
    S = cosine_similarity(V)
    return pd.DataFrame(S, index=cycle_order, columns=cycle_order)


def bootstrap_nmf_stability(
    X: np.ndarray,
    n_synergies: int,
    n_boot: int = 100,
    max_iter: int = 1000,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Bootstrap rows, fit NMF, and compare each fit with the reference solution.
    """
    X = np.asarray(X, dtype=float)
    rng = np.random.default_rng(random_state)
    ref = NMFSynergyExtractor(n_synergies=n_synergies, max_iter=max_iter, random_state=random_state).fit_transform(X)
    scores = []
    for b in range(n_boot):
        idx = rng.integers(0, len(X), size=len(X))
        boot = X[idx]
        fit = NMFSynergyExtractor(n_synergies=n_synergies, max_iter=max_iter, random_state=random_state + b + 1).fit_transform(boot)
        matched, _, _ = match_synergies(ref.W, fit.W)
        scores.append(float(np.mean(matched)))
    ci_low, ci_high = _bootstrap_ci(scores, confidence=0.95, n_boot=max(1000, n_boot * 20), random_state=random_state)
    return {
        "reference_vaf": float(ref.vaf),
        "scores": scores,
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "reference_W": ref.W,
        "reference_H": ref.H,
    }


def phase_conditioned_activation_summary(
    H: np.ndarray,
    labels: np.ndarray,
    confidence: float = 0.95,
) -> pd.DataFrame:
    H = np.asarray(H, dtype=float)
    labels = np.asarray(labels)
    phases = np.unique(labels.astype(int))
    rows = []
    for p in phases:
        idx = np.where(labels == p)[0]
        if len(idx) == 0:
            continue
        vals = H[idx]
        mean = np.mean(vals, axis=0)
        lo = np.quantile(vals, (1.0 - confidence) / 2.0, axis=0)
        hi = np.quantile(vals, 1.0 - (1.0 - confidence) / 2.0, axis=0)
        for j in range(H.shape[1]):
            rows.append({
                "phase": int(p),
                "synergy": int(j + 1),
                "mean": float(mean[j]),
                "ci_low": float(lo[j]),
                "ci_high": float(hi[j]),
                "n": int(len(idx)),
            })
    return pd.DataFrame(rows)


def transition_importance_summary(
    H: np.ndarray,
    labels: np.ndarray,
    dH: np.ndarray | None = None,
    d2H: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Compare features near phase transitions vs stable regions.
    """
    H = np.asarray(H, dtype=float)
    labels = np.asarray(labels)
    transitions = np.zeros(len(labels), dtype=bool)
    transitions[1:] = labels[1:] != labels[:-1]
    stable = ~transitions
    feats = [("H", H)]
    if dH is not None:
        feats.append(("dH", np.asarray(dH, dtype=float)))
    if d2H is not None:
        feats.append(("d2H", np.asarray(d2H, dtype=float)))
    rows = []
    for prefix, X in feats:
        tr = np.mean(np.abs(X[transitions]), axis=0) if np.any(transitions) else np.zeros(X.shape[1])
        st = np.mean(np.abs(X[stable]), axis=0) if np.any(stable) else np.zeros(X.shape[1])
        denom = np.maximum(st, 1e-12)
        gain = tr / denom
        for j in range(X.shape[1]):
            rows.append({
                "feature_group": prefix,
                "feature": f"{prefix}{j+1}",
                "transition_mean_abs": float(tr[j]),
                "stable_mean_abs": float(st[j]),
                "transition_gain": float(gain[j]),
            })
    return pd.DataFrame(rows)


# -----------------------------
# SHAP / feature attribution
# -----------------------------
def _predict_proba_wrapper(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    if hasattr(model, "state_dict") or "torch" in type(model).__module__:
        import torch
        with torch.no_grad():
            x = torch.tensor(np.asarray(X, dtype=np.float32))
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs
    # fallback: use predict and one-hot encode labels
    preds = np.asarray(model.predict(X))
    classes = np.unique(preds)
    out = np.zeros((len(preds), len(classes)), dtype=float)
    for i, c in enumerate(classes):
        out[:, i] = (preds == c).astype(float)
    return out


def shap_feature_attribution(
    model,
    X: np.ndarray,
    feature_names: Sequence[str],
    background_size: int = 64,
    sample_size: int = 128,
    random_state: int = 42,
) -> Dict[str, Any]:
    X = np.asarray(X, dtype=float)
    if len(X) == 0:
        return {"importance": pd.DataFrame(), "shap_values": None, "method": "empty"}
    rng = np.random.default_rng(random_state)
    bg_idx = rng.choice(len(X), size=min(background_size, len(X)), replace=False)
    eval_idx = rng.choice(len(X), size=min(sample_size, len(X)), replace=False)
    background = X[bg_idx]
    sample = X[eval_idx]

    method = "shap"
    shap_values = None
    importance = None
    try:
        import shap
        # Try model-specific explainers first
        if hasattr(model, "estimators_"):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(sample)
        else:
            explainer = shap.Explainer(lambda z: _predict_proba_wrapper(model, z), background)
            shap_values = explainer(sample).values
        arr = np.asarray(shap_values)
        if arr.ndim == 3:
            arr = np.mean(np.abs(arr), axis=0)  # (samples, features)
        elif arr.ndim == 2:
            arr = np.abs(arr)
        else:
            arr = np.abs(arr.reshape(len(sample), -1))
        imp = np.mean(arr, axis=0)
        importance = pd.DataFrame({"feature": list(feature_names), "mean_abs_shap": imp})
    except Exception:
        method = "permutation"
        try:
            scoring = "f1_macro"
            res = permutation_importance(model, sample, np.argmax(_predict_proba_wrapper(model, sample), axis=1), n_repeats=5, random_state=random_state, scoring=scoring)
            importance = pd.DataFrame({"feature": list(feature_names), "mean_abs_shap": res.importances_mean})
        except Exception:
            importance = pd.DataFrame({"feature": list(feature_names), "mean_abs_shap": np.zeros(len(feature_names))})
    if importance is None:
        importance = pd.DataFrame({"feature": list(feature_names), "mean_abs_shap": np.zeros(len(feature_names))})
    importance = importance.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return {"importance": importance, "shap_values": shap_values, "method": method, "sample": sample}


# -----------------------------
# Statistical validation
# -----------------------------
def paired_comparison_stats(a: Sequence[float], b: Sequence[float], confidence: float = 0.95) -> Dict[str, Any]:
    a = np.asarray(list(a), dtype=float)
    b = np.asarray(list(b), dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if len(a) == 0:
        return {"n": 0}
    diff = a - b
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
    effect_dz = float(mean_diff / (std_diff + 1e-12))
    t_stat, t_p = ttest_rel(a, b, nan_policy="omit")
    try:
        w_stat, w_p = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided", mode="auto")
    except Exception:
        w_stat, w_p = np.nan, np.nan
    ci = _bootstrap_ci(diff, confidence=confidence, n_boot=4000)
    return {
        "n": int(len(a)),
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "cohen_dz": effect_dz,
        "paired_t_stat": float(t_stat),
        "paired_t_p": float(t_p),
        "wilcoxon_stat": float(w_stat) if w_stat == w_stat else np.nan,
        "wilcoxon_p": float(w_p) if w_p == w_p else np.nan,
        "ci_low": float(ci[0]),
        "ci_high": float(ci[1]),
    }


def major_model_comparisons(metrics_df: pd.DataFrame, score_col: str = "macro_f1") -> pd.DataFrame:
    """
    Compare a small set of scientifically relevant pairs across matched folds/windows/horizons.
    """
    if metrics_df.empty:
        return pd.DataFrame()
    pairs = [
        ("synergy_hdH2_mlp", "baseline_mlp_raw"),
        ("forecasted_state_mlp", "synergy_hdH2_mlp"),
        ("synergy_hdH_mlp", "synergy_h_mlp"),
        ("raw_emg_gru", "baseline_mlp_raw"),
    ]
    rows = []
    keys = ["fold", "window_ms", "horizon_ms"]
    for a, b in pairs:
        sub_a = metrics_df[metrics_df["model"] == a].copy()
        sub_b = metrics_df[metrics_df["model"] == b].copy()
        merged = pd.merge(sub_a, sub_b, on=keys, suffixes=("_a", "_b"))
        if merged.empty:
            continue
        stats = paired_comparison_stats(merged[f"{score_col}_a"], merged[f"{score_col}_b"])
        rows.append({
            "model_a": a,
            "model_b": b,
            "score_col": score_col,
            **stats,
        })
    return pd.DataFrame(rows)


# -----------------------------
# Subject adaptation
# -----------------------------
def _split_cycles(subject: Any) -> List[np.ndarray]:
    if getattr(subject, "cycle_id", None) is None:
        return []
    cycle_ids = np.asarray(subject.cycle_id)
    order = []
    seen = set()
    for i, c in enumerate(cycle_ids):
        if c not in seen:
            seen.add(c)
            order.append(c)
    return [np.where(cycle_ids == c)[0] for c in order]


def adaptation_experiment(
    train_subjects: Sequence[Any],
    test_subject: Any,
    window_ms: int,
    horizon_ms: int,
    cfg: PipelineConfig,
    feature_representation: str = "H_dH_d2H",
    adaptation_cycles: Sequence[int] = (0, 1, 5, 10),
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Zero-shot vs 1/5/10 cycle adaptation using a small Torch MLP on latent features.
    """
    rep_map = {
        "H": lambda s: s,
        "H_dH": lambda s: s,
        "H_dH_d2H": lambda s: s,
        "pca_latent": lambda s: s,
        "ae_latent": lambda s: s,
        "forecasted_state": lambda s: s,
    }
    if feature_representation not in rep_map:
        raise ValueError(f"Unsupported feature_representation={feature_representation}")

    def _prep(subjects):
        return build_windowed_dataset(
            subjects,
            window_size=build_horizon_steps(window_ms, cfg.windows.sample_rate_hz),
            horizon_steps=build_horizon_steps(horizon_ms, cfg.windows.sample_rate_hz),
            overlap=cfg.windows.overlap,
            use_center_label=cfg.windows.use_center_label,
        )

    train_win = _prep(train_subjects)
    test_win = _prep([test_subject])
    if len(train_win.y) == 0 or len(test_win.y) == 0:
        return pd.DataFrame()

    # use a compact Torch MLP for adaptation so the same architecture is reused
    y_classes = sorted(np.unique(np.concatenate([train_win.y, test_win.y])).tolist())
    label_to_idx = {lab: i for i, lab in enumerate(y_classes)}
    y_train = np.array([label_to_idx[v] for v in train_win.y], dtype=int)
    y_test = np.array([label_to_idx.get(v, -1) for v in test_win.y], dtype=int)
    valid_mask = y_test >= 0
    y_test = y_test[valid_mask]
    X_test = test_win.X_flat[valid_mask]

    # scale on source train only
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(train_win.X_flat)
    X_train = scaler.transform(train_win.X_flat)
    X_test = scaler.transform(X_test)

    # train base model
    base = train_torch_mlp(
        X_train, y_train,
        X_train, y_train,
        input_dim=X_train.shape[1],
        n_classes=len(y_classes),
        hidden_sizes=cfg.model.hidden_sizes,
        dropout=cfg.model.dropout,
        batch_size=cfg.model.batch_size,
        epochs=max(5, min(cfg.model.epochs, 20)),
        learning_rate=cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay,
        patience=max(2, cfg.model.patience),
        random_state=random_state,
    ).model

    # use available cycles
    cycle_sets = _split_cycles(test_subject)
    if not cycle_sets:
        # fallback: split by ordered windows
        cycle_sets = np.array_split(np.arange(len(test_win.y)), max(1, max(adaptation_cycles)))

    rows = []
    for n_cycles in adaptation_cycles:
        model = copy.deepcopy(base)
        if n_cycles > 0:
            adapt_idx = np.concatenate(cycle_sets[: min(n_cycles, len(cycle_sets))]) if len(cycle_sets) else np.array([], dtype=int)
            adapt_idx = adapt_idx[adapt_idx < len(test_win.y)]
            if len(adapt_idx) > 0:
                X_adapt = X_test[adapt_idx]
                y_adapt = y_test[adapt_idx]
                # a few steps of fine-tuning
                _fine_tune_mlp(model, X_adapt, y_adapt, random_state=random_state)
        y_pred = predict_torch(model, X_test)
        pred_labels = np.array([y_classes[i] for i in y_pred], dtype=int)
        from .evaluate import compute_metrics
        m = compute_metrics(y_test, pred_labels, labels=y_classes)
        rows.append({
            "adaptation_cycles": int(n_cycles),
            "accuracy": m.accuracy,
            "macro_f1": m.macro_f1,
            "weighted_f1": m.weighted_f1,
        })
    return pd.DataFrame(rows)


def _fine_tune_mlp(model: TorchMLP, X: np.ndarray, y: np.ndarray, random_state: int = 42, epochs: int = 5, lr: float = 5e-4):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    torch.manual_seed(random_state)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    loader = DataLoader(ds, batch_size=min(64, len(ds)), shuffle=True)
    device = next(model.parameters()).device
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
    return model


# -----------------------------
# Deployment simulation
# -----------------------------
HARDWARE_PROFILES = {
    "Cortex-M7": {"memory_bytes": 512 * 1024, "latency_scale": 12.0, "throughput_scale": 0.08},
    "Raspberry Pi": {"memory_bytes": 2 * 1024**3, "latency_scale": 2.5, "throughput_scale": 0.4},
}


def simulate_deployment(model_metrics: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    for hw, profile in HARDWARE_PROFILES.items():
        latency = float(model_metrics["latency_ms"]) * float(profile["latency_scale"])
        throughput = 1000.0 / max(latency, 1e-6)
        rows.append({
            "hardware": hw,
            "memory_budget_bytes": int(profile["memory_bytes"]),
            "estimated_latency_ms": latency,
            "estimated_throughput_hz": throughput,
            "fits_memory": bool(model_metrics["memory_bytes"] <= profile["memory_bytes"]),
            "params": int(model_metrics["params"]),
            "model_memory_bytes": int(model_metrics["memory_bytes"]),
            "flops": float(model_metrics["flops"]),
        })
    return pd.DataFrame(rows)


# -----------------------------
# Tables / exports
# -----------------------------
def export_table_bundle(df: pd.DataFrame, base_path: Path, index: bool = False) -> Dict[str, str]:
    base_path = Path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = base_path.with_suffix(".csv")
    md_path = base_path.with_suffix(".md")
    tex_path = base_path.with_suffix(".tex")
    df.to_csv(csv_path, index=index)
    md_path.write_text(df.to_markdown(index=index), encoding="utf-8")
    try:
        tex_path.write_text(df.to_latex(index=index, escape=False), encoding="utf-8")
    except Exception:
        tex_path.write_text("", encoding="utf-8")
    return {"csv": str(csv_path), "md": str(md_path), "tex": str(tex_path)}


# -----------------------------
# Figures
# -----------------------------
def plot_pipeline_overview(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 2.8))
    ax.axis("off")
    boxes = [
        "EMG",
        "Conditioning",
        "Windowing",
        "NMF",
        "Synergies",
        "Dynamics",
        "Latent State",
        "State Space",
        "Forecast Layer",
        "Future Phase",
    ]
    xs = np.linspace(0.05, 0.95, len(boxes))
    for i, (x, b) in enumerate(zip(xs, boxes)):
        ax.text(x, 0.5, b, ha="center", va="center", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="black", lw=1.2))
        if i < len(boxes) - 1:
            ax.annotate("", xy=(xs[i + 1] - 0.03, 0.5), xytext=(x + 0.03, 0.5),
                        arrowprops=dict(arrowstyle="->", lw=1.4))
    ax.set_title("Research pipeline overview", pad=12)
    _finalize(fig, path)


def plot_bootstrap_stability(scores: Sequence[float], ci: Tuple[float, float], path: Path, title: str = "Bootstrap NMF stability") -> None:
    scores = np.asarray(scores, dtype=float)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(scores, bins=min(25, max(10, len(scores) // 2)), alpha=0.8)
    ax.axvline(np.mean(scores), linestyle="--", label=f"mean={np.mean(scores):.3f}")
    ax.axvspan(ci[0], ci[1], alpha=0.2, label=f"95% CI [{ci[0]:.3f}, {ci[1]:.3f}]")
    ax.set_xlabel("Matched cosine similarity")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend(frameon=False)
    _finalize(fig, path)


def plot_similarity_heatmap(S: pd.DataFrame, path: Path, title: str) -> None:
    if S.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(S.values, cmap="viridis", vmin=-1, vmax=1)
    ax.set_xticks(range(len(S.columns)))
    ax.set_yticks(range(len(S.index)))
    ax.set_xticklabels(S.columns, rotation=90)
    ax.set_yticklabels(S.index)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Cosine similarity")
    _finalize(fig, path)


def plot_phase_conditioned_activation(summary: pd.DataFrame, path: Path) -> None:
    if summary.empty:
        return
    phases = sorted(summary["phase"].unique())
    synergies = sorted(summary["synergy"].unique())
    fig, ax = plt.subplots(figsize=(12, 5))
    for syn in synergies:
        sub = summary[summary["synergy"] == syn].set_index("phase").reindex(phases)
        ax.plot(phases, sub["mean"].values, marker="o", label=f"H{syn}")
    ax.set_xlabel("Phase")
    ax.set_ylabel("Mean activation")
    ax.set_title("Phase-conditioned H(t)")
    ax.legend(ncols=min(4, len(synergies)), frameon=False)
    _finalize(fig, path)


def plot_feature_importance(importance: pd.DataFrame, path: Path, title: str = "Feature importance / SHAP") -> None:
    if importance.empty:
        return
    imp = importance.sort_values("mean_abs_shap", ascending=True).tail(20)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(imp) + 2)))
    ax.barh(imp["feature"], imp["mean_abs_shap"])
    ax.set_title(title)
    ax.set_xlabel("Mean |SHAP|")
    _finalize(fig, path)


def plot_adaptation_curve(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["adaptation_cycles"], df["macro_f1"], marker="o")
    ax.set_xlabel("Adaptation cycles")
    ax.set_ylabel("Macro F1")
    ax.set_title("Subject adaptation")
    _finalize(fig, path)


def plot_significance_summary(stats_df: pd.DataFrame, path: Path) -> None:
    if stats_df.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 4))
    y = np.arange(len(stats_df))
    ax.barh(y, -np.log10(np.maximum(stats_df["wilcoxon_p"].astype(float).fillna(1.0).values, 1e-300)))
    ax.set_yticks(y)
    ax.set_yticklabels([f"{a} vs {b}" for a, b in zip(stats_df["model_a"], stats_df["model_b"])])
    ax.set_xlabel("-log10(p)")
    ax.set_title("Statistical significance summary")
    _finalize(fig, path)


def plot_state_space_forecast(actual: np.ndarray, forecast: np.ndarray, path: Path, title: str = "State-space forecast") -> None:
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(actual[:, 0], label="actual z1")
    ax.plot(forecast[:, 0], label="forecast z1", linestyle="--")
    if actual.shape[1] > 1 and forecast.shape[1] > 1:
        ax.plot(actual[:, 1], label="actual z2")
        ax.plot(forecast[:, 1], label="forecast z2", linestyle="--")
    ax.set_title(title)
    ax.legend(frameon=False)
    _finalize(fig, path)


# -----------------------------
# Orchestration
# -----------------------------
def generate_research_artifacts(
    output_dir: Path,
    cfg: PipelineConfig,
    subjects: Sequence[Any],
    rep_subjects: Mapping[str, Sequence[Any]] | None,
    fit_objects: Mapping[str, Any] | None,
    metrics_df: pd.DataFrame,
    deploy_df: pd.DataFrame,
    representative_model_bundle: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Create research-grade validation assets without altering the core MVP flow.
    """
    output_dir = ensure_dir(Path(output_dir))
    artifacts: Dict[str, Any] = {}

    # 1) Synergy validation
    if fit_objects is not None and "nmf_fit" in fit_objects:
        nmf_fit = fit_objects["nmf_fit"]
        W = np.asarray(nmf_fit.W)
        S = cosine_similarity_matrix(W)
        pd.DataFrame(S, index=[f"H{i+1}" for i in range(W.shape[0])], columns=[f"H{i+1}" for i in range(W.shape[0])]).to_csv(output_dir / "synergy_cosine_similarity.csv")
        plot_similarity_heatmap(pd.DataFrame(S), output_dir / "synergy_cosine_similarity.png", "Synergy cosine similarity")
        if subjects:
            subj_sim = subject_to_subject_synergy_similarity(subjects, n_synergies=cfg.synergy.n_synergies, max_iter=cfg.synergy.max_iter, random_state=cfg.random_state)
            if not subj_sim.empty:
                subj_sim.to_csv(output_dir / "subject_synergy_similarity.csv")
                plot_similarity_heatmap(subj_sim, output_dir / "subject_synergy_similarity.png", "Subject-to-subject synergy similarity")

        X_train = np.vstack([np.asarray(s.X, dtype=float) for s in subjects]) if subjects else None
        if X_train is not None and len(X_train) > 5:
            boot = bootstrap_nmf_stability(X_train, n_synergies=cfg.synergy.n_synergies, n_boot=50, max_iter=cfg.synergy.max_iter, random_state=cfg.random_state)
            artifacts["bootstrap_nmf"] = boot
            pd.DataFrame({"score": boot["scores"]}).to_csv(output_dir / "bootstrap_nmf_scores.csv", index=False)
            pd.DataFrame([{"mean": boot["mean"], "std": boot["std"], "ci_low": boot["ci_low"], "ci_high": boot["ci_high"], "reference_vaf": boot["reference_vaf"]}]).to_csv(output_dir / "bootstrap_nmf_summary.csv", index=False)
            plot_bootstrap_stability(boot["scores"], (boot["ci_low"], boot["ci_high"]), output_dir / "bootstrap_nmf_stability.png")

    # 2) Motor primitive analysis
    if fit_objects is not None and rep_subjects is not None and "H" in rep_subjects:
        H_concat = np.vstack([np.asarray(s.X, dtype=float) for s in rep_subjects["H"]])
        y_concat = np.concatenate([np.asarray(s.y, dtype=int) for s in rep_subjects["H"]])
        phase_summary = phase_conditioned_activation_summary(H_concat, y_concat)
        if not phase_summary.empty:
            phase_summary.to_csv(output_dir / "phase_conditioned_H.csv", index=False)
            plot_phase_conditioned_activation(phase_summary, output_dir / "phase_conditioned_H.png")
        if "H_dH" in rep_subjects:
            Hdh = np.vstack([np.asarray(s.X, dtype=float) for s in rep_subjects["H_dH"]])
            dH = Hdh[:, H_concat.shape[1]:]
            transition_summary = transition_importance_summary(H_concat, y_concat, dH=dH)
            if not transition_summary.empty:
                transition_summary.to_csv(output_dir / "transition_importance.csv", index=False)
                plot_feature_importance(
                    transition_summary.rename(columns={"transition_gain": "mean_abs_shap"}),
                    output_dir / "transition_importance.png",
                    title="Phase-transition importance",
                )

    # 3) True latent forecasting and state-space preview
    if representative_model_bundle is not None:
        try:
            actual = np.asarray(representative_model_bundle["actual_latent"])
            forecast = np.asarray(representative_model_bundle["forecasted_latent"])
            plot_state_space_forecast(actual, forecast, output_dir / "state_space_forecast.png")
        except Exception:
            pass

    # 4) Subject generalization / adaptation
    if representative_model_bundle is not None and "adaptation_df" in representative_model_bundle:
        adaptation_df = representative_model_bundle["adaptation_df"]
        if isinstance(adaptation_df, pd.DataFrame) and not adaptation_df.empty:
            adaptation_df.to_csv(output_dir / "subject_adaptation.csv", index=False)
            plot_adaptation_curve(adaptation_df, output_dir / "subject_adaptation.png")

    # 5) SHAP / attribution on representative model
    if representative_model_bundle is not None and "model" in representative_model_bundle and "X_train" in representative_model_bundle:
        model = representative_model_bundle["model"]
        X_train = np.asarray(representative_model_bundle["X_train"], dtype=float)
        feature_names = representative_model_bundle.get("feature_names", [f"x{i+1}" for i in range(X_train.shape[1])])
        shap_out = shap_feature_attribution(model, X_train, feature_names)
        artifacts["shap_method"] = shap_out["method"]
        if isinstance(shap_out["importance"], pd.DataFrame) and not shap_out["importance"].empty:
            shap_out["importance"].to_csv(output_dir / "feature_importance_shap.csv", index=False)
            plot_feature_importance(shap_out["importance"], output_dir / "feature_importance_shap.png", title="SHAP feature importance")

    # 6) Stat validation across major comparisons
    stats_df = major_model_comparisons(metrics_df, score_col="macro_f1")
    if not stats_df.empty:
        stats_df.to_csv(output_dir / "major_comparisons.csv", index=False)
        plot_significance_summary(stats_df, output_dir / "statistical_significance_summary.png")
        export_table_bundle(stats_df, output_dir / "major_comparisons", index=False)

    # 7) Deployment simulation
    if not deploy_df.empty:
        # pick per-model median metrics then simulate
        agg = deploy_df.groupby("model", as_index=False).median(numeric_only=True)
        sim_rows = []
        for _, row in agg.iterrows():
            sim = simulate_deployment(row.to_dict())
            sim.insert(0, "model", row["model"])
            sim_rows.append(sim)
        if sim_rows:
            sim_df = pd.concat(sim_rows, ignore_index=True)
            sim_df.to_csv(output_dir / "deployment_hardware_simulation.csv", index=False)
            export_table_bundle(sim_df, output_dir / "deployment_hardware_simulation", index=False)

    # 8) Paper tables
    if not metrics_df.empty:
        metrics_df.to_csv(output_dir / "metrics_summary_full.csv", index=False)
        export_table_bundle(metrics_df, output_dir / "metrics_summary_full", index=False)
    if not deploy_df.empty:
        deploy_df.to_csv(output_dir / "deployment_summary_full.csv", index=False)
        export_table_bundle(deploy_df, output_dir / "deployment_summary_full", index=False)

    # 9) Overview figure
    plot_pipeline_overview(output_dir / "pipeline_overview.png")

    save_json(output_dir / "research_artifacts.json", {"keys": sorted(list(artifacts.keys()))})
    return artifacts
