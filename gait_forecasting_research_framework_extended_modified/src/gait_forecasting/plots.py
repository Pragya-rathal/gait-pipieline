from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

_DEFAULT_STYLE = {
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "legend.frameon": False,
}
_COLOR_CYCLE = plt.get_cmap("tab10").colors


def _as_path(path: str | Path) -> Path:
    return Path(path)


def _apply_style() -> None:
    plt.rcParams.update(_DEFAULT_STYLE)


def _coerce_1d(values: Sequence[float] | np.ndarray | None) -> np.ndarray | None:
    if values is None:
        return None
    return np.asarray(values, dtype=float).ravel()


def _coerce_labels(values: Sequence[Any] | np.ndarray | None) -> np.ndarray | None:
    if values is None:
        return None
    return np.asarray(values)


def _save_figure(fig: plt.Figure, path: str | Path, dpi: int = 300, formats: Sequence[str] | None = None) -> None:
    """Save a figure to one or more publication formats and close it."""
    out = _as_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_formats = [out.suffix.lstrip(".").lower() or "png"] if formats is None else [f.lower().lstrip(".") for f in formats]
    stem = out.with_suffix("")
    fig.tight_layout()
    for fmt in save_formats:
        target = out if out.suffix.lower() == f".{fmt}" and len(save_formats) == 1 else stem.with_suffix(f".{fmt}")
        fig.savefig(target, dpi=dpi, bbox_inches="tight", format=fmt)
    plt.close(fig)


def _finalize(fig: plt.Figure, path: str | Path, dpi: int = 300, formats: Sequence[str] | None = None) -> None:
    _save_figure(fig, path, dpi=dpi, formats=formats)


def _bar_labels(ax: plt.Axes, fmt: str = "{:.3g}") -> None:
    for container in ax.containers:
        ax.bar_label(container, labels=[fmt.format(v.get_height()) if np.isfinite(v.get_height()) else "" for v in container], padding=2, fontsize=8)


def _metric_title(metric: str) -> str:
    return metric.replace("_", " ").replace("ms", "(ms)").title()


def _mapping_to_matrix(data: Mapping[str, Mapping[str, float]] | Mapping[str, Sequence[float]] | np.ndarray) -> tuple[list[str], list[str], np.ndarray]:
    if isinstance(data, np.ndarray):
        arr = np.asarray(data, dtype=float)
        return [str(i) for i in range(arr.shape[0])], [str(i) for i in range(arr.shape[1])], arr
    rows = list(data.keys())
    first = next(iter(data.values())) if data else {}
    if isinstance(first, Mapping):
        cols = sorted({str(k) for row in data.values() for k in row.keys()})  # type: ignore[union-attr]
        arr = np.array([[float(data[r].get(c, np.nan)) for c in cols] for r in rows])  # type: ignore[index]
    else:
        arr = np.array([list(v) for v in data.values()], dtype=float)  # type: ignore[union-attr]
        cols = [str(i) for i in range(arr.shape[1])]
    return [str(r) for r in rows], cols, arr


def plot_vaf(vaf_pairs: Sequence[tuple[int, float]], chosen_k: int, threshold: float, path: Path, dpi: int = 300, formats: Sequence[str] | None = None) -> None:
    _apply_style()
    ks = [k for k, _ in vaf_pairs]
    vafs = [v for _, v in vaf_pairs]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, vafs, marker="o", linewidth=2.2, color=_COLOR_CYCLE[0])
    ax.axhline(threshold, linestyle="--", linewidth=1.5, color=_COLOR_CYCLE[3], label=f"Threshold {threshold:.2f}")
    ax.axvline(chosen_k, linestyle=":", linewidth=1.8, color=_COLOR_CYCLE[2], label=f"Chosen k={chosen_k}")
    ax.set_xlabel("Number of synergies")
    ax.set_ylabel("VAF")
    ax.set_title("Variance Accounted For vs Number of Synergies")
    ax.legend()
    _finalize(fig, path, dpi=dpi, formats=formats)


def plot_synergy_activations(H: np.ndarray, path: Path, title: str = "Synergy activations H(t)", dpi: int = 300, formats: Sequence[str] | None = None) -> None:
    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.linspace(0, 1, H.shape[0])
    for i in range(H.shape[1]):
        ax.plot(x, H[:, i], linewidth=2, label=f"H{i+1}")
    ax.set_xlabel("Normalized gait cycle")
    ax.set_ylabel("Activation")
    ax.set_title(title)
    ax.legend(ncols=min(4, H.shape[1]))
    _finalize(fig, path, dpi=dpi, formats=formats)


def plot_weights(W: np.ndarray, channel_names: Sequence[str], path: Path, title: str = "Synergy weight vectors", dpi: int = 300, formats: Sequence[str] | None = None) -> None:
    _apply_style()
    n_synergies = W.shape[0]
    fig, axes = plt.subplots(1, n_synergies, figsize=(max(4, 3.5 * n_synergies), 4.5), sharey=True)
    axes = np.atleast_1d(axes)
    for i, ax in enumerate(axes):
        ax.bar(range(len(channel_names)), W[i], color=_COLOR_CYCLE[i % len(_COLOR_CYCLE)])
        ax.set_title(f"Synergy {i+1}")
        ax.set_xticks(range(len(channel_names)))
        ax.set_xticklabels(channel_names, rotation=90)
        ax.set_ylabel("Weight" if i == 0 else "")
    fig.suptitle(title)
    _finalize(fig, path, dpi=dpi, formats=formats)


def plot_latent_trajectories(Z: np.ndarray, labels: np.ndarray | None, path: Path, title: str = "Latent state trajectories", dpi: int = 300, formats: Sequence[str] | None = None) -> None:
    _apply_style()
    Z = np.asarray(Z)
    fig, ax = plt.subplots(figsize=(7, 6))
    if Z.shape[1] >= 2:
        if labels is None:
            ax.plot(Z[:, 0], Z[:, 1], linewidth=1.6, color=_COLOR_CYCLE[0])
        else:
            sc = ax.scatter(Z[:, 0], Z[:, 1], c=labels, s=14, cmap="viridis", alpha=0.85, edgecolors="none")
            fig.colorbar(sc, ax=ax, label="Class")
        ax.set_xlabel("Latent 1")
        ax.set_ylabel("Latent 2")
    else:
        ax.plot(Z[:, 0], linewidth=1.6, color=_COLOR_CYCLE[0])
        ax.set_xlabel("Index")
        ax.set_ylabel("Latent 1")
    ax.set_title(title)
    _finalize(fig, path, dpi=dpi, formats=formats)


def plot_pca_scatter(Z: np.ndarray, labels: np.ndarray | None, path: Path, title: str = "PCA latent state", dpi: int = 300, formats: Sequence[str] | None = None) -> None:
    plot_latent_trajectories(Z, labels, path, title=title, dpi=dpi, formats=formats)


def plot_umap_scatter(Z: np.ndarray, labels: np.ndarray | None, path: Path, title: str = "UMAP latent state", dpi: int = 300, formats: Sequence[str] | None = None) -> None:
    _apply_style()
    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=42)
        Z2 = reducer.fit_transform(Z)
    except Exception:
        Z2 = Z[:, :2] if Z.shape[1] >= 2 else np.c_[np.arange(len(Z)), Z[:, 0]]
    fig, ax = plt.subplots(figsize=(7, 6))
    if labels is None:
        ax.scatter(Z2[:, 0], Z2[:, 1], s=14, alpha=0.85, edgecolors="none")
    else:
        sc = ax.scatter(Z2[:, 0], Z2[:, 1], c=labels, s=14, cmap="viridis", alpha=0.85, edgecolors="none")
        fig.colorbar(sc, ax=ax, label="Class")
    ax.set_title(title)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    _finalize(fig, path, dpi=dpi, formats=formats)


def plot_tsne_scatter(Z: np.ndarray, labels: np.ndarray | None, path: Path, title: str = "t-SNE latent state", perplexity: float = 30.0, dpi: int = 300, formats: Sequence[str] | None = None) -> None:
    _apply_style()
    try:
        from sklearn.manifold import TSNE
        Z2 = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=42).fit_transform(Z)
    except Exception:
        Z2 = Z[:, :2] if Z.shape[1] >= 2 else np.c_[np.arange(len(Z)), Z[:, 0]]
    fig, ax = plt.subplots(figsize=(7, 6))
    if labels is None:
        ax.scatter(Z2[:, 0], Z2[:, 1], color=_COLOR_CYCLE[0], s=14, alpha=0.85, edgecolors="none")
    else:
        sc = ax.scatter(Z2[:, 0], Z2[:, 1], c=labels, s=14, cmap="viridis", alpha=0.85, edgecolors="none")
        fig.colorbar(sc, ax=ax, label="Class")
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    _finalize(fig, path, dpi=dpi, formats=formats)


def plot_training_curves(train_loss: Sequence[float], val_loss: Sequence[float] | None = None, train_accuracy: Sequence[float] | None = None, val_accuracy: Sequence[float] | None = None, epochs: Sequence[int] | None = None, path: str | Path = "training_curves.png", learning_rate: Sequence[float] | None = None, dpi: int = 300, formats: Sequence[str] | None = None) -> None:
    _apply_style()
    train_loss = _coerce_1d(train_loss)
    val_loss = _coerce_1d(val_loss)
    train_accuracy = _coerce_1d(train_accuracy)
    val_accuracy = _coerce_1d(val_accuracy)
    learning_rate = _coerce_1d(learning_rate)
    x = np.asarray(epochs if epochs is not None else np.arange(1, len(train_loss) + 1))
    ncols = 3 if learning_rate is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5.2 * ncols, 4.2))
    axes = np.atleast_1d(axes)
    axes[0].plot(x, train_loss, label="Train", linewidth=2.2)
    if val_loss is not None:
        axes[0].plot(x, val_loss, label="Validation", linewidth=2.2)
    axes[0].set_title("Loss vs Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    if train_accuracy is not None:
        axes[1].plot(x, train_accuracy, label="Train", linewidth=2.2)
    if val_accuracy is not None:
        axes[1].plot(x, val_accuracy, label="Validation", linewidth=2.2)
    axes[1].set_title("Accuracy vs Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    if learning_rate is not None:
        axes[2].plot(x, learning_rate, color=_COLOR_CYCLE[2], linewidth=2.2)
        axes[2].set_title("Learning Rate Schedule")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Learning rate")
        axes[2].set_yscale("log")
    _finalize(fig, path, dpi=dpi, formats=formats)


def plot_confusion_matrix(y_true: Sequence[Any] | np.ndarray | None = None, y_pred: Sequence[Any] | np.ndarray | None = None, path: str | Path = "confusion_matrix.png", labels: Sequence[Any] | None = None, matrix: np.ndarray | None = None, normalize: bool = False, task: str = "Current Activity", cmap: str = "Blues", dpi: int = 300, formats: Sequence[str] | None = None) -> None:
    _apply_style()
    if matrix is None:
        yt, yp = _coerce_labels(y_true), _coerce_labels(y_pred)
        if yt is None or yp is None:
            raise ValueError("Provide either matrix or both y_true and y_pred.")
        labels = list(labels) if labels is not None else list(dict.fromkeys(np.concatenate([yt, yp]).tolist()))
        idx = {label: i for i, label in enumerate(labels)}
        matrix = np.zeros((len(labels), len(labels)), dtype=float)
        for t, p in zip(yt, yp):
            matrix[idx[t], idx[p]] += 1
    else:
        matrix = np.asarray(matrix, dtype=float)
        labels = list(labels) if labels is not None else [str(i) for i in range(matrix.shape[0])]
    values = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1) if normalize else matrix
    fig, ax = plt.subplots(figsize=(max(6, 0.55 * len(labels)), max(5, 0.5 * len(labels))))
    im = ax.imshow(values, cmap=cmap, aspect="auto")
    fig.colorbar(im, ax=ax, label="Normalized count" if normalize else "Count")
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"{task} Confusion Matrix")
    fmt = ".2f" if normalize else ".0f"
    threshold = np.nanmax(values) / 2 if values.size else 0
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, format(values[i, j], fmt), ha="center", va="center", color="white" if values[i, j] > threshold else "black", fontsize=8)
    _finalize(fig, path, dpi=dpi, formats=formats)


def _binary_curve(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(-scores)
    y = y_true[order].astype(int)
    s = scores[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    pos = max(tp[-1] if len(tp) else 0, 1)
    neg = max(fp[-1] if len(fp) else 0, 1)
    return np.r_[0, fp / neg], np.r_[0, tp / pos], np.r_[np.inf, s]


def plot_roc_curve(y_true: Sequence[int] | np.ndarray, y_score: Sequence[float] | np.ndarray, path: str | Path = "roc_curve.png", class_names: Sequence[str] | None = None, dpi: int = 300, formats: Sequence[str] | None = None) -> None:
    _apply_style()
    y_true = np.asarray(y_true)
    scores = np.asarray(y_score, dtype=float)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    if scores.ndim == 1:
        fpr, tpr, _ = _binary_curve(y_true, scores)
        ax.plot(fpr, tpr, linewidth=2.2, label="ROC")
    else:
        classes = np.arange(scores.shape[1])
        names = class_names if class_names is not None else [str(c) for c in classes]
        for c, name in zip(classes, names):
            fpr, tpr, _ = _binary_curve((y_true == c).astype(int), scores[:, c])
            ax.plot(fpr, tpr, linewidth=2, label=str(name))
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.5", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    _finalize(fig, path, dpi=dpi, formats=formats)


def plot_precision_recall_curve(y_true: Sequence[int] | np.ndarray, y_score: Sequence[float] | np.ndarray, path: str | Path = "precision_recall_curve.png", class_names: Sequence[str] | None = None, dpi: int = 300, formats: Sequence[str] | None = None) -> None:
    _apply_style()
    y_true = np.asarray(y_true)
    scores = np.asarray(y_score, dtype=float)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    series = [(y_true.astype(int), scores, "PR")] if scores.ndim == 1 else [((y_true == c).astype(int), scores[:, c], (class_names[c] if class_names else str(c))) for c in range(scores.shape[1])]
    for y, s, name in series:
        _, recall, thresholds = _binary_curve(y, s)
        order = np.argsort(-s)
        yy = y[order].astype(int)
        precision = np.cumsum(yy == 1) / np.maximum(np.arange(1, len(yy) + 1), 1)
        ax.plot(np.r_[0, recall[1:]], np.r_[1, precision], linewidth=2, label=str(name))
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    _finalize(fig, path, dpi=dpi, formats=formats)


def plot_class_distribution(labels: Sequence[Any] | Mapping[str, int], path: str | Path = "class_distribution.png", title: str = "Class Distribution", dpi: int = 300, formats: Sequence[str] | None = None) -> None:
    _apply_style()
    if isinstance(labels, Mapping):
        names, counts = list(labels.keys()), np.asarray(list(labels.values()), dtype=float)
    else:
        names, counts = np.unique(np.asarray(labels), return_counts=True)
        names = [str(n) for n in names]
    fig, ax = plt.subplots(figsize=(max(7, 0.5 * len(names)), 4.8))
    ax.bar(names, counts, color=_COLOR_CYCLE[0])
    ax.set_title(title)
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)
    _bar_labels(ax, "{:.0f}")
    _finalize(fig, path, dpi=dpi, formats=formats)


def plot_prediction_confidence(confidences: Sequence[float] | np.ndarray, path: str | Path = "prediction_confidence.png", correct: Sequence[bool] | np.ndarray | None = None, bins: int = 20, dpi: int = 300, formats: Sequence[str] | None = None) -> None:
    _apply_style()
    c = np.asarray(confidences, dtype=float)
    fig, ax = plt.subplots(figsize=(7, 4.8))
    if correct is None:
        ax.hist(c, bins=bins, color=_COLOR_CYCLE[0], alpha=0.85)
    else:
        correct = np.asarray(correct, dtype=bool)
        ax.hist([c[correct], c[~correct]], bins=bins, label=["Correct", "Incorrect"], color=[_COLOR_CYCLE[2], _COLOR_CYCLE[3]], alpha=0.8, stacked=True)
        ax.legend()
    ax.set_title("Prediction Confidence")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Count")
    _finalize(fig, path, dpi=dpi, formats=formats)


def plot_multitask_metrics(metrics: Mapping[str, Mapping[str, float]], path: str | Path = "multitask_metrics.png", dpi: int = 300, formats: Sequence[str] | None = None) -> None:
    _apply_style()
    tasks, metric_names, arr = _mapping_to_matrix(metrics)
    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(tasks)), 5))
    x = np.arange(len(tasks)); width = 0.8 / max(len(metric_names), 1)
    for i, m in enumerate(metric_names):
        ax.bar(x + (i - (len(metric_names) - 1) / 2) * width, arr[:, i], width, label=_metric_title(m))
    ax.set_xticks(x, tasks, rotation=20, ha="right")
    ax.set_ylabel("Metric value")
    ax.set_title("Multi-task Performance Metrics")
    ax.legend(ncols=min(4, len(metric_names)))
    _finalize(fig, path, dpi=dpi, formats=formats)


def plot_baseline_comparison(results: Mapping[str, Mapping[str, float]] | Any, path: str | Path = "baseline_comparison.png", metrics: Sequence[str] = ("accuracy", "f1", "latency", "parameters", "model_size", "gpu_memory"), dpi: int = 300, formats: Sequence[str] | None = None) -> None:
    _apply_style()
    if hasattr(results, "set_index"):
        data = results.set_index("model")[[m for m in metrics if m in results.columns]].to_dict("index")
    else:
        data = results
    models, metric_names, arr = _mapping_to_matrix(data)
    fig, axes = plt.subplots(1, len(metric_names), figsize=(max(5 * len(metric_names), 8), 4.8), squeeze=False)
    for i, metric in enumerate(metric_names):
        ax = axes[0, i]
        ax.bar(models, arr[:, i], color=_COLOR_CYCLE[i % len(_COLOR_CYCLE)])
        ax.set_title(_metric_title(metric))
        ax.tick_params(axis="x", rotation=60)
    fig.suptitle("Baseline Model Comparison")
    _finalize(fig, path, dpi=dpi, formats=formats)


def plot_ablation_results(results: Mapping[str, Mapping[str, float]], path: str | Path = "ablation_results.png", dpi: int = 300, formats: Sequence[str] | None = None) -> None:
    _apply_style()
    variants, metrics, arr = _mapping_to_matrix(results)
    fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(variants)), 5))
    x = np.arange(len(variants)); width = 0.8 / max(len(metrics), 1)
    for i, metric in enumerate(metrics):
        ax.bar(x + (i - (len(metrics) - 1) / 2) * width, arr[:, i], width, label=_metric_title(metric))
    ax.set_xticks(x, variants, rotation=25, ha="right")
    ax.set_ylabel("Metric value")
    ax.set_title("Ablation Study")
    ax.legend(ncols=min(4, len(metrics)))
    _finalize(fig, path, dpi=dpi, formats=formats)


def plot_forecast_horizon_accuracy(horizons: Sequence[float], accuracy: Sequence[float] | Mapping[str, Sequence[float]], path: str | Path = "forecast_horizon_accuracy.png", metric_name: str = "Accuracy", dpi: int = 300, formats: Sequence[str] | None = None) -> None:
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    if isinstance(accuracy, Mapping):
        for name, vals in accuracy.items():
            ax.plot(horizons, vals, marker="o", linewidth=2.2, label=name)
        ax.legend()
    else:
        ax.plot(horizons, accuracy, marker="o", linewidth=2.2)
    ax.set_xlabel("Forecast horizon (ms)")
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name} Across Forecast Horizons")
    _finalize(fig, path, dpi=dpi, formats=formats)


def plot_transition_timeline(time: Sequence[float], true_transition: Sequence[float], predicted_transition: Sequence[float] | None = None, path: str | Path = "transition_timeline.png", probabilities: Sequence[float] | None = None, dpi: int = 300, formats: Sequence[str] | None = None) -> None:
    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(time, true_transition, drawstyle="steps-post", linewidth=2.2, label="Observed transition")
    if predicted_transition is not None:
        ax.plot(time, predicted_transition, drawstyle="steps-post", linewidth=2, label="Predicted transition")
    if probabilities is not None:
        ax2 = ax.twinx()
        ax2.plot(time, probabilities, color=_COLOR_CYCLE[3], alpha=0.75, label="Transition probability")
        ax2.set_ylabel("Probability")
        ax2.set_ylim(0, 1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Transition state")
    ax.set_title("Anticipatory Transition Timeline")
    ax.legend(loc="upper left")
    _finalize(fig, path, dpi=dpi, formats=formats)


def plot_time_to_transition_error(errors: Sequence[float], path: str | Path = "time_to_transition_error.png", bins: int = 25, dpi: int = 300, formats: Sequence[str] | None = None) -> None:
    _apply_style()
    errors = np.asarray(errors, dtype=float)
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.hist(errors, bins=bins, color=_COLOR_CYCLE[0], alpha=0.85)
    ax.axvline(0, color="0.25", linestyle="--", linewidth=1.4)
    ax.axvline(np.nanmean(errors), color=_COLOR_CYCLE[3], linewidth=2, label=f"Mean={np.nanmean(errors):.2f}")
    ax.set_xlabel("Prediction error (s)")
    ax.set_ylabel("Count")
    ax.set_title("Time-to-Transition Error")
    ax.legend()
    _finalize(fig, path, dpi=dpi, formats=formats)


def plot_transition_probability(time: Sequence[float], probability: Sequence[float] | Mapping[str, Sequence[float]], path: str | Path = "transition_probability.png", threshold: float | None = 0.5, dpi: int = 300, formats: Sequence[str] | None = None) -> None:
    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 4.8))
    if isinstance(probability, Mapping):
        for name, vals in probability.items():
            ax.plot(time, vals, linewidth=2, label=name)
        ax.legend()
    else:
        ax.plot(time, probability, linewidth=2.2, label="Transition probability")
    if threshold is not None:
        ax.axhline(threshold, linestyle="--", color=_COLOR_CYCLE[3], label=f"Threshold={threshold:.2f}")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Probability")
    ax.set_title("Forecast Transition Probability")
    _finalize(fig, path, dpi=dpi, formats=formats)


def plot_forecast_horizon(results: dict, metric: str, path: Path, title: str = "Forecast horizon comparison", dpi: int = 300, formats: Sequence[str] | None = None) -> None:
    _apply_style()
    horizons = sorted(results.keys(), key=lambda x: int(str(x).replace("ms", "")))
    models = sorted({m for h in horizons for m in results[h].keys()})
    fig, ax = plt.subplots(figsize=(10, 5))
    for model in models:
        vals = [results[h].get(model, {}).get(metric, np.nan) for h in horizons]
        ax.plot([int(str(h).replace("ms", "")) for h in horizons], vals, marker="o", linewidth=2, label=model)
    ax.set_xlabel("Forecast horizon (ms)")
    ax.set_ylabel(_metric_title(metric))
    ax.set_title(title)
    ax.legend(ncols=2)
    _finalize(fig, path, dpi=dpi, formats=formats)


def plot_model_comparison(df, metric: str, path: Path, title: str = "Model comparison", dpi: int = 300, formats: Sequence[str] | None = None) -> None:
    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    for model, g in df.groupby("model"):
        g = g.sort_values("window_ms")
        ax.plot(g["window_ms"], g[metric], marker="o", linewidth=2, label=model)
    ax.set_xlabel("Window (ms)")
    ax.set_ylabel(_metric_title(metric))
    ax.set_title(title)
    ax.legend(ncols=2)
    _finalize(fig, path, dpi=dpi, formats=formats)


def plot_deployment_comparison(df, path: Path, title: str = "Deployment comparison", dpi: int = 300, formats: Sequence[str] | None = None) -> None:
    _apply_style()
    candidate_metrics = ["latency_ms", "training_time", "gpu_utilization", "cpu_utilization", "vram", "ram", "model_size", "params", "parameters", "flops"]
    model_names = list(df["model"])
    metrics = [m for m in candidate_metrics if m in df]
    if not metrics:
        metrics = [c for c in df.columns if c != "model"]
    ncols = min(3, len(metrics))
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.2 * nrows), squeeze=False)
    for ax, metric in zip(axes.ravel(), metrics):
        ax.bar(model_names, df[metric], color=_COLOR_CYCLE[metrics.index(metric) % len(_COLOR_CYCLE)])
        ax.set_title(_metric_title(metric))
        ax.tick_params(axis="x", rotation=60)
    for ax in axes.ravel()[len(metrics):]:
        ax.axis("off")
    fig.suptitle(title)
    _finalize(fig, path, dpi=dpi, formats=formats)


def plot_cross_validation_summary(
    fold_metrics: Sequence[Mapping[str, float]],
    path: str | Path = "cv_summary.png",
    metric_keys: Sequence[str] = ("accuracy", "macro_f1", "weighted_f1"),
    title: str = "Cross-Validation Summary",
    dpi: int = 300,
    formats: Sequence[str] | None = None,
) -> None:
    _apply_style()
    if not fold_metrics:
        return
    fig, axes = plt.subplots(1, len(metric_keys), figsize=(5 * len(metric_keys), 4.5), squeeze=False)
    for col, key in enumerate(metric_keys):
        ax = axes[0, col]
        vals = [float(m.get(key, float("nan"))) for m in fold_metrics]
        folds = [f"F{i+1}" for i in range(len(vals))]
        ax.bar(folds, vals, color=_COLOR_CYCLE[col % len(_COLOR_CYCLE)])
        mean = np.nanmean(vals)
        ax.axhline(mean, linestyle="--", linewidth=1.5, color="0.3", label=f"mean={mean:.3f}")
        ax.set_title(_metric_title(key))
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
    fig.suptitle(title)
    _finalize(fig, path, dpi=dpi, formats=formats)


def plot_loso_summary(
    subject_metrics: Mapping[str, Mapping[str, float]],
    path: str | Path = "loso_summary.png",
    metric_key: str = "accuracy",
    title: str = "LOSO Summary",
    dpi: int = 300,
    formats: Sequence[str] | None = None,
) -> None:
    _apply_style()
    if not subject_metrics:
        return
    subjects = list(subject_metrics.keys())
    vals = [float(subject_metrics[s].get(metric_key, float("nan"))) for s in subjects]
    fig, ax = plt.subplots(figsize=(max(8, 0.7 * len(subjects)), 5))
    ax.bar(subjects, vals, color=_COLOR_CYCLE[0])
    mean = np.nanmean(vals)
    ax.axhline(mean, linestyle="--", linewidth=1.5, color="0.3", label=f"mean={mean:.3f}")
    ax.set_title(title)
    ax.set_xlabel("Subject")
    ax.set_ylabel(_metric_title(metric_key))
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylim(0, 1)
    ax.legend()
    _finalize(fig, path, dpi=dpi, formats=formats)


def plot_learning_rate_schedule(
    lr_values: Sequence[float],
    path: str | Path = "lr_schedule.png",
    epochs: Sequence[int] | None = None,
    dpi: int = 300,
    formats: Sequence[str] | None = None,
) -> None:
    _apply_style()
    x = list(epochs) if epochs is not None else list(range(1, len(lr_values) + 1))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, lr_values, linewidth=2.2, color=_COLOR_CYCLE[2])
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate")
    ax.set_title("Learning Rate Schedule")
    _finalize(fig, path, dpi=dpi, formats=formats)


def plot_synergy_heatmap(
    H: np.ndarray,
    path: str | Path = "synergy_heatmap.png",
    title: str = "Synergy Activation Heatmap",
    dpi: int = 300,
    formats: Sequence[str] | None = None,
) -> None:
    _apply_style()
    H = np.asarray(H, dtype=float)
    fig, ax = plt.subplots(figsize=(max(8, H.shape[0] / 50), max(4, H.shape[1] * 0.5 + 1)))
    im = ax.imshow(H.T, aspect="auto", origin="lower", cmap="hot")
    ax.set_xlabel("Time")
    ax.set_ylabel("Synergy")
    ax.set_yticks(np.arange(H.shape[1]))
    ax.set_yticklabels([f"H{i+1}" for i in range(H.shape[1])])
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Activation")
    _finalize(fig, path, dpi=dpi, formats=formats)


def plot_dynamic_synergy_features(
    H: np.ndarray,
    dH: np.ndarray,
    path: str | Path = "dynamic_synergy_features.png",
    d2H: np.ndarray | None = None,
    dpi: int = 300,
    formats: Sequence[str] | None = None,
) -> None:
    _apply_style()
    H = np.asarray(H, dtype=float)
    dH = np.asarray(dH, dtype=float)
    nrows = 3 if d2H is not None else 2
    fig, axes = plt.subplots(nrows, 1, figsize=(10, 3 * nrows), sharex=True)
    x = np.arange(H.shape[0])
    for i in range(H.shape[1]):
        axes[0].plot(x, H[:, i], linewidth=1.4, label=f"H{i+1}")
        axes[1].plot(x, dH[:, i], linewidth=1.2, label=f"dH{i+1}")
    axes[0].set_title("Synergy Activations H(t)")
    axes[0].set_ylabel("H")
    axes[1].set_title("Synergy Velocities dH/dt")
    axes[1].set_ylabel("dH")
    if d2H is not None:
        d2H = np.asarray(d2H, dtype=float)
        for i in range(d2H.shape[1]):
            axes[2].plot(x, d2H[:, i], linewidth=1.2, label=f"d²H{i+1}")
        axes[2].set_title("Synergy Accelerations d²H/dt²")
        axes[2].set_ylabel("d²H")
    axes[-1].set_xlabel("Sample")
    for ax in axes:
        ax.legend(ncols=min(4, H.shape[1]), fontsize=8)
    _finalize(fig, path, dpi=dpi, formats=formats)


def plot_benchmark_comparison(
    results: Sequence[Any],
    path: str | Path = "benchmark_comparison.png",
    metrics: Sequence[str] = (
        "inference_latency_ms", "throughput_samples_per_s",
        "n_params", "model_size_mb", "vram_used_mb",
    ),
    dpi: int = 300,
    formats: Sequence[str] | None = None,
) -> None:
    _apply_style()
    if not results:
        return
    rows = [r.to_dict() if hasattr(r, "to_dict") else dict(r) for r in results]
    model_names = [r.get("model_name", str(i)) for i, r in enumerate(rows)]
    valid_metrics = [m for m in metrics if any(m in r for r in rows)]
    if not valid_metrics:
        return
    ncols = min(3, len(valid_metrics))
    nrows = int(np.ceil(len(valid_metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.2 * nrows), squeeze=False)
    for idx, metric in enumerate(valid_metrics):
        ax = axes[idx // ncols, idx % ncols]
        vals = [float(r.get(metric, 0) or 0) for r in rows]
        ax.bar(model_names, vals, color=_COLOR_CYCLE[idx % len(_COLOR_CYCLE)])
        ax.set_title(_metric_title(metric))
        ax.tick_params(axis="x", rotation=55)
    for idx in range(len(valid_metrics), nrows * ncols):
        axes[idx // ncols, idx % ncols].axis("off")
    fig.suptitle("Architecture Benchmark Comparison")
    _finalize(fig, path, dpi=dpi, formats=formats)


def plot_inference_latency_distribution(
    latencies_ms: Sequence[float],
    path: str | Path = "inference_latency_distribution.png",
    bins: int = 30,
    title: str = "Inference Latency Distribution",
    dpi: int = 300,
    formats: Sequence[str] | None = None,
) -> None:
    _apply_style()
    latencies = np.asarray(latencies_ms, dtype=float)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(latencies, bins=bins, color=_COLOR_CYCLE[0], alpha=0.85)
    mean = float(np.mean(latencies))
    p95 = float(np.percentile(latencies, 95))
    ax.axvline(mean, linestyle="--", color=_COLOR_CYCLE[1], linewidth=1.8, label=f"mean={mean:.2f} ms")
    ax.axvline(p95, linestyle=":", color=_COLOR_CYCLE[3], linewidth=1.8, label=f"p95={p95:.2f} ms")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    _finalize(fig, path, dpi=dpi, formats=formats)


def plot_gpu_memory_usage(
    model_names: Sequence[str],
    vram_mb: Sequence[float],
    path: str | Path = "gpu_memory_usage.png",
    ram_mb: Sequence[float] | None = None,
    dpi: int = 300,
    formats: Sequence[str] | None = None,
) -> None:
    _apply_style()
    x = np.arange(len(model_names))
    width = 0.4 if ram_mb is not None else 0.7
    fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(model_names)), 5))
    bars = ax.bar(x - (width / 2 if ram_mb is not None else 0), vram_mb, width, label="VRAM (MB)", color=_COLOR_CYCLE[0])
    if ram_mb is not None:
        ax.bar(x + width / 2, ram_mb, width, label="RAM (MB)", color=_COLOR_CYCLE[1])
        ax.legend()
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha="right")
    ax.set_ylabel("Memory (MB)")
    ax.set_title("GPU / CPU Memory Usage")
    _finalize(fig, path, dpi=dpi, formats=formats)


def plot_parameter_count(
    model_names: Sequence[str],
    param_counts: Sequence[int],
    path: str | Path = "parameter_count.png",
    model_sizes_mb: Sequence[float] | None = None,
    dpi: int = 300,
    formats: Sequence[str] | None = None,
) -> None:
    _apply_style()
    x = np.arange(len(model_names))
    width = 0.4 if model_sizes_mb is not None else 0.7
    fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(model_names)), 5))
    ax.bar(x - (width / 2 if model_sizes_mb is not None else 0), param_counts, width, label="Parameters", color=_COLOR_CYCLE[0])
    ax.set_ylabel("Parameter Count")
    ax.set_title("Model Parameter Counts")
    if model_sizes_mb is not None:
        ax2 = ax.twinx()
        ax2.bar(x + width / 2, model_sizes_mb, width, label="Size (MB)", color=_COLOR_CYCLE[1], alpha=0.7)
        ax2.set_ylabel("Model size (MB)")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha="right")
    _finalize(fig, path, dpi=dpi, formats=formats)


def plot_flops_comparison(
    model_names: Sequence[str],
    flops: Sequence[float],
    path: str | Path = "flops_comparison.png",
    title: str = "Approximate FLOPs Comparison",
    dpi: int = 300,
    formats: Sequence[str] | None = None,
) -> None:
    _apply_style()
    fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(model_names)), 5))
    ax.bar(model_names, flops, color=_COLOR_CYCLE[2])
    ax.set_yscale("log")
    ax.set_ylabel("FLOPs (log scale)")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    _finalize(fig, path, dpi=dpi, formats=formats)


def plot_ablation_study(
    results: Mapping[str, Mapping[str, float]],
    path: str | Path = "ablation_study.png",
    baseline_key: str | None = None,
    metric_key: str = "accuracy",
    title: str = "Ablation Study",
    dpi: int = 300,
    formats: Sequence[str] | None = None,
) -> None:
    _apply_style()
    variants = list(results.keys())
    vals = [float(results[v].get(metric_key, float("nan"))) for v in variants]
    colors = []
    baseline_val = None
    if baseline_key and baseline_key in results:
        baseline_val = float(results[baseline_key].get(metric_key, float("nan")))
    for v, val in zip(variants, vals):
        if v == baseline_key:
            colors.append(_COLOR_CYCLE[0])
        elif baseline_val is not None and val < baseline_val:
            colors.append(_COLOR_CYCLE[3])
        else:
            colors.append(_COLOR_CYCLE[2])
    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(variants)), 5))
    ax.bar(variants, vals, color=colors)
    if baseline_val is not None:
        ax.axhline(baseline_val, linestyle="--", color="0.3", linewidth=1.5, label=f"Baseline={baseline_val:.3f}")
        ax.legend()
    ax.set_ylabel(_metric_title(metric_key))
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=40)
    _finalize(fig, path, dpi=dpi, formats=formats)
