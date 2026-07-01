from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .config import PipelineConfig
from .data import SubjectDataset, group_kfold_splits, leave_one_subject_out, load_dataset
from .checkpoint import CheckpointManager
from .metrics import (
    MetricBundle,
    compute_classification_metrics,
    compute_multitask_metrics,
    aggregate_metrics_across_folds as _metrics_aggregate,
    to_evaluate_metrics,
)
from .utils import ensure_dir, save_json


# ---------------------------------------------------------------------------
# Legacy dataclass kept for backward-compat with research.py and pipeline.py
# ---------------------------------------------------------------------------

@dataclass
class Metrics:
    accuracy: float
    macro_f1: float
    weighted_f1: float
    report: dict
    confusion: np.ndarray


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int]) -> Metrics:
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return Metrics(acc, macro_f1, weighted_f1, report, cm)


def save_confusion_matrix(cm: np.ndarray, labels: list[int], title: str, path: Path) -> None:
    from . import plots as _plots
    _plots.plot_confusion_matrix(matrix=cm, labels=labels, task=title, path=path)


def metrics_table(metrics_by_name: Dict[str, Metrics]) -> str:
    lines = ["| Model | Accuracy | Macro F1 | Weighted F1 |", "|---|---:|---:|---:|"]
    for name, m in metrics_by_name.items():
        lines.append(f"| {name} | {m.accuracy:.4f} | {m.macro_f1:.4f} | {m.weighted_f1:.4f} |")
    return "\n".join(lines)


def metrics_dataframe(metrics_by_name: Dict[str, Metrics]) -> pd.DataFrame:
    rows = [
        {"model": name, "accuracy": m.accuracy, "macro_f1": m.macro_f1, "weighted_f1": m.weighted_f1}
        for name, m in metrics_by_name.items()
    ]
    return pd.DataFrame(rows)


def save_metrics_csv(metrics_by_name: Dict[str, Metrics], path: Path) -> None:
    metrics_dataframe(metrics_by_name).to_csv(path, index=False)


def aggregate_metrics_across_folds(
    all_fold_metrics: Sequence[Dict[str, Metrics]],
) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[Metrics]] = {}
    for fold_metrics in all_fold_metrics:
        for name, m in fold_metrics.items():
            grouped.setdefault(name, []).append(m)
    summary: Dict[str, Dict[str, float]] = {}
    for name, items in grouped.items():
        summary[name] = {
            "accuracy_mean": float(np.mean([m.accuracy for m in items])),
            "accuracy_std": float(np.std([m.accuracy for m in items])),
            "macro_f1_mean": float(np.mean([m.macro_f1 for m in items])),
            "macro_f1_std": float(np.std([m.macro_f1 for m in items])),
            "weighted_f1_mean": float(np.mean([m.weighted_f1 for m in items])),
            "weighted_f1_std": float(np.std([m.weighted_f1 for m in items])),
        }
    return summary


# ---------------------------------------------------------------------------
# Structured evaluation report
# ---------------------------------------------------------------------------

@dataclass
class EvalReport:
    name: str
    task_metrics: Dict[str, Any]
    aggregate: Dict[str, float]
    history: List[Dict[str, float]]
    n_samples: int
    device: str
    checkpoint_dir: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for k, v in d["task_metrics"].items():
            if isinstance(v, np.ndarray):
                d["task_metrics"][k] = v.tolist()
        return d

    @property
    def accuracy(self) -> float:
        ca = self.task_metrics.get("current_activity", {})
        return float(ca.get("accuracy", 0.0)) if isinstance(ca, dict) else 0.0

    @property
    def macro_f1(self) -> float:
        ca = self.task_metrics.get("current_activity", {})
        return float(ca.get("f1_macro", 0.0)) if isinstance(ca, dict) else 0.0

    def as_legacy_metrics(self) -> Metrics:
        ca = self.task_metrics.get("current_activity", {})
        if not isinstance(ca, dict):
            return Metrics(0.0, 0.0, 0.0, {}, np.zeros((1, 1), dtype=int))
        cm = np.array(ca.get("confusion", [[0]]))
        return Metrics(
            accuracy=float(ca.get("accuracy", 0.0)),
            macro_f1=float(ca.get("f1_macro", 0.0)),
            weighted_f1=float(ca.get("f1_weighted", 0.0)),
            report={},
            confusion=cm,
        )


# ---------------------------------------------------------------------------
# Core evaluation function — delegates inference to pipeline.run_fold /
# pipeline.run_pipeline; never instantiates model classes directly.
# ---------------------------------------------------------------------------

def _fold_outcome_to_eval_report(fold_key: str, fold_data: Dict[str, Any]) -> EvalReport:
    task_metrics = fold_data.get("test_metrics", {})
    aggregate = fold_data.get("aggregate", {})
    history = fold_data.get("history", [])
    n_samples = int(aggregate.get("n_tasks", 0))
    return EvalReport(
        name=fold_key,
        task_metrics=task_metrics,
        aggregate=aggregate,
        history=history,
        n_samples=n_samples,
        checkpoint_dir=fold_data.get("checkpoint_dir"),
    )


def evaluate(cfg: PipelineConfig) -> Dict[str, Any]:
    """
    Run the complete pipeline (all folds, windows, horizons) and return
    structured evaluation results. All computation flows through pipeline.run_pipeline.
    """
    from . import pipeline as _pipeline
    summary = _pipeline.run_pipeline(cfg)
    return summary


def evaluate_holdout(cfg: PipelineConfig) -> Tuple[List[EvalReport], Dict[str, Any]]:
    """Run evaluation in holdout mode. Returns (reports, summary)."""
    from dataclasses import replace
    from .config import EvalConfig
    cfg_holdout = PipelineConfig(
        data_dir=cfg.data_dir,
        output_dir=cfg.output_dir,
        normalize=cfg.normalize,
        smooth=cfg.smooth,
        use_synergies=cfg.use_synergies,
        use_dh=cfg.use_dh,
        use_d2h=cfg.use_d2h,
        demo=cfg.demo,
        random_state=cfg.random_state,
        synergy=cfg.synergy,
        model=cfg.model,
        eval=EvalConfig(
            cross_validation="holdout",
            n_splits=cfg.eval.n_splits,
            test_size=cfg.eval.test_size,
            val_size=cfg.eval.val_size,
            random_state=cfg.eval.random_state,
        ),
        windows=cfg.windows,
    )
    summary = evaluate(cfg_holdout)
    reports = [_fold_outcome_to_eval_report(k, v) for k, v in summary.get("folds", {}).items()]
    return reports, summary


def evaluate_loso(cfg: PipelineConfig) -> Tuple[List[EvalReport], Dict[str, Any]]:
    """Run Leave-One-Subject-Out evaluation. Returns (reports, summary)."""
    from .config import EvalConfig
    cfg_loso = PipelineConfig(
        data_dir=cfg.data_dir,
        output_dir=cfg.output_dir,
        normalize=cfg.normalize,
        smooth=cfg.smooth,
        use_synergies=cfg.use_synergies,
        use_dh=cfg.use_dh,
        use_d2h=cfg.use_d2h,
        demo=cfg.demo,
        random_state=cfg.random_state,
        synergy=cfg.synergy,
        model=cfg.model,
        eval=EvalConfig(
            cross_validation="loso",
            n_splits=cfg.eval.n_splits,
            test_size=cfg.eval.test_size,
            val_size=cfg.eval.val_size,
            random_state=cfg.eval.random_state,
        ),
        windows=cfg.windows,
    )
    summary = evaluate(cfg_loso)
    reports = [_fold_outcome_to_eval_report(k, v) for k, v in summary.get("folds", {}).items()]
    return reports, summary


def evaluate_groupkfold(cfg: PipelineConfig, n_splits: Optional[int] = None) -> Tuple[List[EvalReport], Dict[str, Any]]:
    """Run Group K-Fold evaluation. Returns (reports, summary)."""
    from .config import EvalConfig
    cfg_gkf = PipelineConfig(
        data_dir=cfg.data_dir,
        output_dir=cfg.output_dir,
        normalize=cfg.normalize,
        smooth=cfg.smooth,
        use_synergies=cfg.use_synergies,
        use_dh=cfg.use_dh,
        use_d2h=cfg.use_d2h,
        demo=cfg.demo,
        random_state=cfg.random_state,
        synergy=cfg.synergy,
        model=cfg.model,
        eval=EvalConfig(
            cross_validation="groupkfold",
            n_splits=n_splits or cfg.eval.n_splits,
            test_size=cfg.eval.test_size,
            val_size=cfg.eval.val_size,
            random_state=cfg.eval.random_state,
        ),
        windows=cfg.windows,
    )
    summary = evaluate(cfg_gkf)
    reports = [_fold_outcome_to_eval_report(k, v) for k, v in summary.get("folds", {}).items()]
    return reports, summary


def load_eval_report_from_checkpoint(checkpoint_dir: Path) -> Optional[EvalReport]:
    """Load an EvalReport from a CheckpointManager's saved metrics.json."""
    checkpoint_dir = Path(checkpoint_dir)
    metrics_path = checkpoint_dir / CheckpointManager.METRICS_JSON
    history_path = checkpoint_dir / CheckpointManager.HISTORY_JSON
    metadata_path = checkpoint_dir / CheckpointManager.METADATA_JSON
    if not metrics_path.exists():
        return None
    import json
    metrics_data = json.loads(metrics_path.read_text())
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    return EvalReport(
        name=str(checkpoint_dir.name),
        task_metrics=metrics_data.get("last_metrics", {}),
        aggregate={"best_val_metric": metrics_data.get("best_val_metric", 0.0)},
        history=history,
        n_samples=0,
        checkpoint_dir=str(checkpoint_dir),
    )


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_reports(
    summary: Dict[str, Any],
    output_dir: Path,
    cfg: Optional[PipelineConfig] = None,
    formats: Sequence[str] = ("png",),
    run_benchmark: bool = False,
    benchmark_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    From a summary dict returned by evaluate() / run_pipeline(), produce
    CSV, JSON, Markdown, and plot outputs.
    """
    from . import plots as _plots
    from . import benchmark as _benchmark

    output_dir = ensure_dir(Path(output_dir))
    reports = [_fold_outcome_to_eval_report(k, v) for k, v in summary.get("folds", {}).items()]

    # --- JSON ---
    save_json(output_dir / "evaluation_results.json", {r.name: r.to_dict() for r in reports})
    save_json(output_dir / "evaluation_aggregate.json", summary.get("aggregate_metrics", {}))

    # --- CSV / Markdown per-task accuracy table ---
    rows = []
    for r in reports:
        row: Dict[str, Any] = {"fold_key": r.name}
        for task, tm in r.task_metrics.items():
            if isinstance(tm, dict):
                row[f"{task}_accuracy"] = tm.get("accuracy", "")
                row[f"{task}_f1_macro"] = tm.get("f1_macro", "")
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "evaluation_per_fold.csv", index=False)
    (output_dir / "evaluation_per_fold.md").write_text(df.to_markdown(index=False) or "", encoding="utf-8")

    # --- Training curves per fold ---
    plots_dir = ensure_dir(output_dir / "plots")
    for r in reports:
        if not r.history:
            continue
        train_loss = [h.get("train_loss", float("nan")) for h in r.history]
        val_loss = [h.get("val_loss", float("nan")) for h in r.history]
        safe_name = r.name.replace("/", "_")
        _plots.plot_training_curves(
            train_loss=train_loss,
            val_loss=val_loss,
            path=plots_dir / f"{safe_name}_training_curves.png",
            formats=list(formats),
        )

    # --- Multitask metric summary across all folds ---
    task_accs: Dict[str, List[float]] = {}
    for r in reports:
        for task, tm in r.task_metrics.items():
            if isinstance(tm, dict) and "accuracy" in tm:
                task_accs.setdefault(task, []).append(float(tm["accuracy"]))
    if task_accs:
        mean_by_task = {t: {"accuracy": float(np.mean(v)), "std": float(np.std(v))} for t, v in task_accs.items()}
        _plots.plot_multitask_metrics(
            {t: {"accuracy": v["accuracy"]} for t, v in mean_by_task.items()},
            path=plots_dir / "multitask_accuracy_summary.png",
            formats=list(formats),
        )
        save_json(output_dir / "multitask_accuracy_summary.json", mean_by_task)

    # --- Confusion matrices for current_activity ---
    for r in reports:
        ca = r.task_metrics.get("current_activity")
        if isinstance(ca, dict) and "confusion" in ca:
            cm = np.array(ca["confusion"])
            labels_int = list(range(cm.shape[0]))
            safe_name = r.name.replace("/", "_")
            _plots.plot_confusion_matrix(
                matrix=cm,
                labels=labels_int,
                task=f"{r.name} — current_activity",
                path=plots_dir / f"{safe_name}_current_activity_cm.png",
                formats=list(formats),
            )

    # --- Benchmark ---
    benchmark_results = None
    if run_benchmark and cfg is not None:
        from . import pipeline as _pipeline
        benchmark_results = _pipeline.run_benchmark(cfg, **(benchmark_kwargs or {}))

    artifact_paths = {
        "evaluation_results_json": str(output_dir / "evaluation_results.json"),
        "evaluation_aggregate_json": str(output_dir / "evaluation_aggregate.json"),
        "evaluation_per_fold_csv": str(output_dir / "evaluation_per_fold.csv"),
        "plots_dir": str(plots_dir),
    }
    if benchmark_results:
        artifact_paths["benchmark_results"] = [b.to_dict() for b in benchmark_results]

    save_json(output_dir / "report_manifest.json", artifact_paths)
    return artifact_paths
