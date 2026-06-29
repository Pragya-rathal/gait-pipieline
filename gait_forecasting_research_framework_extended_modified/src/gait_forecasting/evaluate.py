
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


@dataclass
class Metrics:
    accuracy: float
    macro_f1: float
    weighted_f1: float
    report: dict
    confusion: np.ndarray


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int]) -> Metrics:
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return Metrics(acc, macro_f1, weighted_f1, report, cm)


def save_confusion_matrix(cm: np.ndarray, labels: list[int], title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted phase")
    ax.set_ylabel("True phase")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black", fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def metrics_table(metrics_by_name: Dict[str, Metrics]) -> str:
    lines = ["| Model | Accuracy | Macro F1 | Weighted F1 |", "|---|---:|---:|---:|"]
    for name, m in metrics_by_name.items():
        lines.append(f"| {name} | {m.accuracy:.4f} | {m.macro_f1:.4f} | {m.weighted_f1:.4f} |")
    return "\n".join(lines)


def metrics_dataframe(metrics_by_name: Dict[str, Metrics]) -> pd.DataFrame:
    rows = []
    for name, m in metrics_by_name.items():
        rows.append({
            "model": name,
            "accuracy": m.accuracy,
            "macro_f1": m.macro_f1,
            "weighted_f1": m.weighted_f1,
        })
    return pd.DataFrame(rows)


def save_metrics_csv(metrics_by_name: Dict[str, Metrics], path: Path) -> None:
    metrics_dataframe(metrics_by_name).to_csv(path, index=False)


def aggregate_metrics_across_folds(all_fold_metrics: Sequence[Dict[str, Metrics]]) -> Dict[str, Dict[str, float]]:
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
