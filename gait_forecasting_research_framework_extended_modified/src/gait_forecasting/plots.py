
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np


def _finalize(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_vaf(vaf_pairs: Sequence[tuple[int, float]], chosen_k: int, threshold: float, path: Path) -> None:
    ks = [k for k, _ in vaf_pairs]
    vafs = [v for _, v in vaf_pairs]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, vafs, marker="o", linewidth=2)
    ax.axhline(threshold, linestyle="--", linewidth=1.5, label=f"Threshold {threshold:.2f}")
    ax.axvline(chosen_k, linestyle=":", linewidth=1.5, label=f"Chosen k={chosen_k}")
    ax.set_xlabel("Number of synergies")
    ax.set_ylabel("VAF")
    ax.set_title("Variance Accounted For vs Number of Synergies")
    ax.legend()
    _finalize(fig, path)


def plot_synergy_activations(H: np.ndarray, path: Path, title: str = "Synergy activations H(t)") -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.linspace(0, 1, H.shape[0])
    for i in range(H.shape[1]):
        ax.plot(x, H[:, i], linewidth=2, label=f"H{i+1}")
    ax.set_xlabel("Normalized gait cycle")
    ax.set_ylabel("Activation")
    ax.set_title(title)
    ax.legend(ncols=min(4, H.shape[1]), frameon=False)
    _finalize(fig, path)


def plot_weights(W: np.ndarray, channel_names: Sequence[str], path: Path, title: str = "Synergy weight vectors") -> None:
    n_synergies = W.shape[0]
    fig, axes = plt.subplots(1, n_synergies, figsize=(4 * n_synergies, 4), sharey=True)
    if n_synergies == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.bar(range(len(channel_names)), W[i])
        ax.set_title(f"Synergy {i+1}")
        ax.set_xticks(range(len(channel_names)))
        ax.set_xticklabels(channel_names, rotation=90)
    fig.suptitle(title)
    _finalize(fig, path)


def plot_latent_trajectories(Z: np.ndarray, labels: np.ndarray | None, path: Path, title: str = "Latent state trajectories") -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    if Z.shape[1] >= 2:
        if labels is None:
            ax.plot(Z[:, 0], Z[:, 1], linewidth=1.5)
        else:
            sc = ax.scatter(Z[:, 0], Z[:, 1], c=labels, s=8, cmap="viridis")
            fig.colorbar(sc, ax=ax, label="Class")
        ax.set_xlabel("Latent 1")
        ax.set_ylabel("Latent 2")
    else:
        ax.plot(Z[:, 0], linewidth=1.5)
        ax.set_xlabel("Index")
        ax.set_ylabel("Latent 1")
    ax.set_title(title)
    _finalize(fig, path)


def plot_pca_scatter(Z: np.ndarray, labels: np.ndarray | None, path: Path, title: str = "PCA latent state") -> None:
    plot_latent_trajectories(Z, labels, path, title=title)


def plot_umap_scatter(Z: np.ndarray, labels: np.ndarray | None, path: Path, title: str = "UMAP latent state") -> None:
    # Try UMAP if available; otherwise fall back to first two dimensions
    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=42)
        Z2 = reducer.fit_transform(Z)
    except Exception:
        Z2 = Z[:, :2] if Z.shape[1] >= 2 else np.c_[np.arange(len(Z)), Z[:, 0]]
    fig, ax = plt.subplots(figsize=(7, 6))
    if labels is None:
        ax.scatter(Z2[:, 0], Z2[:, 1], s=8)
    else:
        sc = ax.scatter(Z2[:, 0], Z2[:, 1], c=labels, s=8, cmap="viridis")
        fig.colorbar(sc, ax=ax, label="Class")
    ax.set_title(title)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    _finalize(fig, path)


def plot_forecast_horizon(results: dict, metric: str, path: Path, title: str = "Forecast horizon comparison") -> None:
    horizons = sorted(results.keys(), key=lambda x: int(str(x).replace("ms", "")))
    models = sorted({m for h in horizons for m in results[h].keys()})
    fig, ax = plt.subplots(figsize=(10, 5))
    for model in models:
        vals = [results[h].get(model, {}).get(metric, np.nan) for h in horizons]
        ax.plot([int(str(h).replace("ms", "")) for h in horizons], vals, marker="o", linewidth=2, label=model)
    ax.set_xlabel("Forecast horizon (ms)")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title)
    ax.legend(frameon=False, ncols=2)
    _finalize(fig, path)


def plot_model_comparison(df, metric: str, path: Path, title: str = "Model comparison") -> None:
    # expects columns: model, window_ms, metric
    fig, ax = plt.subplots(figsize=(10, 5))
    for model, g in df.groupby("model"):
        g = g.sort_values("window_ms")
        ax.plot(g["window_ms"], g[metric], marker="o", linewidth=2, label=model)
    ax.set_xlabel("Window (ms)")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title)
    ax.legend(frameon=False, ncols=2)
    _finalize(fig, path)


def plot_deployment_comparison(df, path: Path, title: str = "Deployment comparison") -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].bar(df["model"], df["params"])
    axes[0].set_title("Parameters")
    axes[0].tick_params(axis="x", rotation=90)
    axes[1].bar(df["model"], df["latency_ms"])
    axes[1].set_title("Latency (ms)")
    axes[1].tick_params(axis="x", rotation=90)
    axes[2].bar(df["model"], df["flops"])
    axes[2].set_title("FLOPs (approx)")
    axes[2].tick_params(axis="x", rotation=90)
    fig.suptitle(title)
    _finalize(fig, path)
