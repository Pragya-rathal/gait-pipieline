from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef,
    cohen_kappa_score,
    mean_absolute_error,
)


# ---------------------------------------------------------------------------
# Rich metric container
# ---------------------------------------------------------------------------

@dataclass
class MetricBundle:
    accuracy: float = 0.0
    balanced_accuracy: float = 0.0
    precision_macro: float = 0.0
    recall_macro: float = 0.0
    f1_macro: float = 0.0
    f1_weighted: float = 0.0
    f1_micro: float = 0.0
    roc_auc: float = 0.0
    pr_auc: float = 0.0
    confusion: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=int))
    matthews_corr: float = 0.0
    cohen_kappa: float = 0.0
    mae: float = 0.0
    rmse: float = 0.0
    transition_detection_accuracy: float = 0.0
    future_activity_accuracy: float = 0.0
    forecast_horizon_accuracy: Dict[str, float] = field(default_factory=dict)
    per_class_f1: Dict[str, float] = field(default_factory=dict)
    per_class_precision: Dict[str, float] = field(default_factory=dict)
    per_class_recall: Dict[str, float] = field(default_factory=dict)
    n_samples: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["confusion"] = self.confusion.tolist()
        return d

    def to_flat_dict(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for k, v in self.to_dict().items():
            if k in ("confusion", "per_class_f1", "per_class_precision",
                     "per_class_recall", "forecast_horizon_accuracy"):
                continue
            if isinstance(v, (int, float)):
                out[k] = float(v)
        for horizon, acc in self.forecast_horizon_accuracy.items():
            out[f"horizon_{horizon}_accuracy"] = float(acc)
        return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_roc_auc(
    y_true: np.ndarray,
    y_prob: Optional[np.ndarray],
    labels: Optional[Sequence[int]],
) -> float:
    if y_prob is None:
        return 0.0
    unique = np.unique(y_true)
    if len(unique) < 2:
        return 0.0
    try:
        if y_prob.ndim == 2 and y_prob.shape[1] > 2:
            return float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro",
                                       labels=labels))
        col = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
        return float(roc_auc_score(y_true, col))
    except Exception:
        return 0.0


def _safe_pr_auc(
    y_true: np.ndarray,
    y_prob: Optional[np.ndarray],
) -> float:
    if y_prob is None:
        return 0.0
    unique = np.unique(y_true)
    if len(unique) < 2:
        return 0.0
    try:
        if y_prob.ndim == 2 and y_prob.shape[1] > 2:
            return float(average_precision_score(y_true, y_prob, average="macro"))
        col = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
        return float(average_precision_score(y_true, col))
    except Exception:
        return 0.0


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true, float) - np.asarray(y_pred, float)) ** 2)))


# ---------------------------------------------------------------------------
# Core classification metric computation
# ---------------------------------------------------------------------------

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[Sequence[int]] = None,
    y_prob: Optional[np.ndarray] = None,
) -> MetricBundle:
    """
    Compute the full suite of classification metrics.

    Parameters
    ----------
    y_true : (N,) integer array of ground-truth class indices.
    y_pred : (N,) integer array of predicted class indices.
    labels : optional list of all class labels (for confusion matrix ordering).
    y_prob : optional (N, C) or (N,) probability array for ROC/PR AUC.

    Returns
    -------
    MetricBundle
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if labels is None:
        labels = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
    labels_list = list(labels)

    zd = dict(zero_division=0)

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    prec_mac = float(precision_score(y_true, y_pred, average="macro", labels=labels_list, **zd))
    rec_mac = float(recall_score(y_true, y_pred, average="macro", labels=labels_list, **zd))
    f1_mac = float(f1_score(y_true, y_pred, average="macro", labels=labels_list, **zd))
    f1_wei = float(f1_score(y_true, y_pred, average="weighted", labels=labels_list, **zd))
    f1_mic = float(f1_score(y_true, y_pred, average="micro", labels=labels_list, **zd))
    cm = confusion_matrix(y_true, y_pred, labels=labels_list)
    mcc = float(matthews_corrcoef(y_true, y_pred))

    try:
        kappa = float(cohen_kappa_score(y_true, y_pred, labels=labels_list))
    except Exception:
        kappa = 0.0

    roc = _safe_roc_auc(y_true, y_prob, labels_list)
    pr = _safe_pr_auc(y_true, y_prob)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = _rmse(y_true, y_pred)

    f1_per = f1_score(y_true, y_pred, average=None, labels=labels_list, **zd)
    prec_per = precision_score(y_true, y_pred, average=None, labels=labels_list, **zd)
    rec_per = recall_score(y_true, y_pred, average=None, labels=labels_list, **zd)

    return MetricBundle(
        accuracy=acc,
        balanced_accuracy=bal_acc,
        precision_macro=prec_mac,
        recall_macro=rec_mac,
        f1_macro=f1_mac,
        f1_weighted=f1_wei,
        f1_micro=f1_mic,
        roc_auc=roc,
        pr_auc=pr,
        confusion=cm,
        matthews_corr=mcc,
        cohen_kappa=kappa,
        mae=mae,
        rmse=rmse,
        per_class_f1={str(labels_list[i]): float(f1_per[i]) for i in range(len(labels_list))},
        per_class_precision={str(labels_list[i]): float(prec_per[i]) for i in range(len(labels_list))},
        per_class_recall={str(labels_list[i]): float(rec_per[i]) for i in range(len(labels_list))},
        n_samples=int(len(y_true)),
    )


# ---------------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------------

def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """MAE and RMSE for regression targets (e.g. time-to-transition)."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": _rmse(y_true, y_pred),
        "n_samples": float(len(y_true)),
    }


# ---------------------------------------------------------------------------
# Domain-specific metrics
# ---------------------------------------------------------------------------

def transition_detection_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tolerance_samples: int = 5,
) -> float:
    """
    Fraction of true transition boundaries detected within ±tolerance_samples.

    Parameters
    ----------
    y_true : (N,) integer label sequence
    y_pred : (N,) integer label sequence
    tolerance_samples : symmetric window (samples) around each true transition

    Returns
    -------
    float in [0, 1]; 1.0 when no ground-truth transitions exist.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    true_transitions = np.where(np.diff(y_true) != 0)[0] + 1
    if len(true_transitions) == 0:
        return 1.0

    pred_transitions = set(np.where(np.diff(y_pred) != 0)[0] + 1)
    detected = sum(
        any(abs(t - p) <= tolerance_samples for p in pred_transitions)
        for t in true_transitions
    )
    return float(detected) / float(len(true_transitions))


def future_activity_accuracy(
    y_true_future: np.ndarray,
    y_pred_future: np.ndarray,
) -> float:
    """Accuracy of the future-activity prediction head."""
    return float(accuracy_score(
        np.asarray(y_true_future, dtype=int),
        np.asarray(y_pred_future, dtype=int),
    ))


def forecast_horizon_accuracy(
    horizon_results: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> Dict[str, float]:
    """
    Compute accuracy at each forecast horizon.

    Parameters
    ----------
    horizon_results : dict mapping horizon label (e.g. "50ms") to
        (y_true, y_pred) arrays.

    Returns
    -------
    dict mapping horizon label to accuracy float.
    """
    return {
        horizon: float(accuracy_score(
            np.asarray(y_true, dtype=int),
            np.asarray(y_pred, dtype=int),
        ))
        for horizon, (y_true, y_pred) in horizon_results.items()
    }


# ---------------------------------------------------------------------------
# Multi-task metrics
# ---------------------------------------------------------------------------

def compute_multitask_metrics(
    task_outputs: Dict[str, Dict[str, np.ndarray]],
    labels: Optional[Dict[str, Sequence[int]]] = None,
    y_probs: Optional[Dict[str, np.ndarray]] = None,
    horizon_results: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
    transition_tolerance: int = 5,
) -> Dict[str, Any]:
    """
    Evaluate all tasks from a MultiTaskPredictor in one call.

    Parameters
    ----------
    task_outputs : dict mapping task_name -> {"y_true": array, "y_pred": array}.
        For regression tasks: {"y_true": array, "y_pred": array}.
        For binary tasks: {"y_true": array, "y_pred": array} with 0/1 values.
    labels : optional dict mapping task_name -> list of class labels.
    y_probs : optional dict mapping task_name -> probability array for AUC.
    horizon_results : optional dict for forecast_horizon_accuracy computation.
    transition_tolerance : sample tolerance for transition detection.

    Returns
    -------
    dict with one entry per task plus aggregate summary fields.
    """
    labels = labels or {}
    y_probs = y_probs or {}
    results: Dict[str, Any] = {}

    for task_name, arrays in task_outputs.items():
        y_true = np.asarray(arrays["y_true"])
        y_pred = np.asarray(arrays["y_pred"])
        task_labels = labels.get(task_name)
        task_prob = y_probs.get(task_name)

        if y_true.dtype.kind == "f" and task_labels is None:
            reg = compute_regression_metrics(y_true, y_pred)
            results[task_name] = reg
        else:
            bundle = compute_classification_metrics(y_true, y_pred, task_labels, task_prob)

            if task_name == "transition_flag":
                bundle.transition_detection_accuracy = float(
                    accuracy_score(y_true.astype(int), y_pred.astype(int))
                )
            if task_name == "future_activity":
                bundle.future_activity_accuracy = future_activity_accuracy(y_true, y_pred)

            results[task_name] = bundle.to_dict()

    if horizon_results is not None:
        results["forecast_horizon"] = forecast_horizon_accuracy(horizon_results)

    accuracy_vals = [
        v["accuracy"] for v in results.values()
        if isinstance(v, dict) and "accuracy" in v
    ]
    f1_vals = [
        v["f1_macro"] for v in results.values()
        if isinstance(v, dict) and "f1_macro" in v
    ]
    results["_aggregate"] = {
        "mean_accuracy": float(np.mean(accuracy_vals)) if accuracy_vals else 0.0,
        "mean_f1_macro": float(np.mean(f1_vals)) if f1_vals else 0.0,
        "n_tasks": len(task_outputs),
    }
    return results


# ---------------------------------------------------------------------------
# Fold aggregation
# ---------------------------------------------------------------------------

@dataclass
class FoldAggregation:
    mean: float
    std: float
    min: float
    max: float
    values: List[float]


def aggregate_metrics_across_folds(
    fold_metrics: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate scalar metrics across cross-validation folds.

    Parameters
    ----------
    fold_metrics : list of dicts (one per fold), each mapping
        metric_name -> scalar value or MetricBundle.to_dict().

    Returns
    -------
    dict mapping metric_name -> {"mean", "std", "min", "max"}.
    """
    if not fold_metrics:
        return {}

    def _flatten(d: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
        out: Dict[str, float] = {}
        for k, v in d.items():
            full_key = f"{prefix}{k}" if prefix else k
            if isinstance(v, (int, float)):
                out[full_key] = float(v)
            elif isinstance(v, dict):
                out.update(_flatten(v, f"{full_key}/"))
        return out

    all_flat: List[Dict[str, float]] = [_flatten(fm) for fm in fold_metrics]
    all_keys = set().union(*[set(f.keys()) for f in all_flat])

    aggregated: Dict[str, Dict[str, float]] = {}
    for key in sorted(all_keys):
        values = [f[key] for f in all_flat if key in f]
        if not values:
            continue
        arr = np.array(values, dtype=float)
        aggregated[key] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "n_folds": float(len(arr)),
        }
    return aggregated


def aggregate_metric_bundles_across_folds(
    fold_bundles: Sequence[Dict[str, MetricBundle]],
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate MetricBundle objects (keyed by model/config name) across folds.

    Parameters
    ----------
    fold_bundles : list (one per fold) of dicts mapping name -> MetricBundle.

    Returns
    -------
    dict mapping name -> aggregated scalar metrics.
    """
    grouped: Dict[str, List[MetricBundle]] = {}
    for fold in fold_bundles:
        for name, bundle in fold.items():
            grouped.setdefault(name, []).append(bundle)

    summary: Dict[str, Dict[str, float]] = {}
    for name, bundles in grouped.items():
        scalar_fields = [
            "accuracy", "balanced_accuracy", "precision_macro", "recall_macro",
            "f1_macro", "f1_weighted", "f1_micro", "roc_auc", "pr_auc",
            "matthews_corr", "cohen_kappa", "mae", "rmse",
        ]
        result: Dict[str, float] = {}
        for sf in scalar_fields:
            vals = np.array([getattr(b, sf) for b in bundles], dtype=float)
            result[f"{sf}_mean"] = float(vals.mean())
            result[f"{sf}_std"] = float(vals.std())
        result["n_folds"] = float(len(bundles))
        summary[name] = result
    return summary


# ---------------------------------------------------------------------------
# evaluate.py-compatible output
# ---------------------------------------------------------------------------

def to_evaluate_metrics(
    bundle: MetricBundle,
    labels: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """
    Convert a MetricBundle to a dict compatible with evaluate.py's Metrics dataclass.

    Returns a plain dict with the same fields as evaluate.Metrics so callers
    can either use it directly or construct a Metrics object from it.
    """
    from sklearn.metrics import classification_report
    try:
        report: Dict = {}
    except Exception:
        report = {}

    return {
        "accuracy": bundle.accuracy,
        "macro_f1": bundle.f1_macro,
        "weighted_f1": bundle.f1_weighted,
        "report": report,
        "confusion": bundle.confusion,
        "balanced_accuracy": bundle.balanced_accuracy,
        "roc_auc": bundle.roc_auc,
        "pr_auc": bundle.pr_auc,
        "matthews_corr": bundle.matthews_corr,
        "cohen_kappa": bundle.cohen_kappa,
        "mae": bundle.mae,
        "rmse": bundle.rmse,
        "precision_macro": bundle.precision_macro,
        "recall_macro": bundle.recall_macro,
        "f1_micro": bundle.f1_micro,
    }


# ---------------------------------------------------------------------------
# Convenience: single-call full evaluation
# ---------------------------------------------------------------------------

def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[Sequence[int]] = None,
    y_prob: Optional[np.ndarray] = None,
    y_true_future: Optional[np.ndarray] = None,
    y_pred_future: Optional[np.ndarray] = None,
    horizon_results: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
    transition_tolerance: int = 5,
) -> MetricBundle:
    """
    Full evaluation in a single call.

    Parameters
    ----------
    y_true : ground-truth current-activity labels.
    y_pred : predicted current-activity labels.
    labels : all class labels.
    y_prob : probability matrix for AUC metrics.
    y_true_future : ground-truth future-activity labels (optional).
    y_pred_future : predicted future-activity labels (optional).
    horizon_results : mapping horizon_label -> (y_true, y_pred) for multi-horizon eval.
    transition_tolerance : window (samples) for transition detection accuracy.

    Returns
    -------
    MetricBundle with all applicable fields populated.
    """
    bundle = compute_classification_metrics(y_true, y_pred, labels, y_prob)

    bundle.transition_detection_accuracy = transition_detection_accuracy(
        y_true, y_pred, tolerance_samples=transition_tolerance
    )

    if y_true_future is not None and y_pred_future is not None:
        bundle.future_activity_accuracy = future_activity_accuracy(y_true_future, y_pred_future)

    if horizon_results is not None:
        bundle.forecast_horizon_accuracy = forecast_horizon_accuracy(horizon_results)

    return bundle
