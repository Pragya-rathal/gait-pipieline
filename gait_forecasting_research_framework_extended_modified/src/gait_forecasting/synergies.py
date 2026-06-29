
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.decomposition import NMF

from .preprocessing import minmax_positive


@dataclass
class SynergyResult:
    H: np.ndarray
    W: np.ndarray
    vaf: float


def variance_accounted_for(X: np.ndarray, X_hat: np.ndarray) -> float:
    num = np.sum((X - X_hat) ** 2)
    den = np.sum(X ** 2) + 1e-12
    return 1.0 - num / den


class NMFSynergyExtractor:
    def __init__(self, n_synergies: int = 5, max_iter: int = 1000, random_state: int = 42):
        self.n_synergies = n_synergies
        self.max_iter = max_iter
        self.random_state = random_state
        self.model: NMF | None = None

    def fit(self, X: np.ndarray) -> "NMFSynergyExtractor":
        Xp = minmax_positive(X)
        self.model = NMF(
            n_components=self.n_synergies,
            init="nndsvda",
            random_state=self.random_state,
            max_iter=self.max_iter,
        )
        self.model.fit(Xp)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted")
        Xp = minmax_positive(X)
        return self.model.transform(Xp)

    def fit_transform(self, X: np.ndarray) -> SynergyResult:
        self.fit(X)
        assert self.model is not None
        Xp = minmax_positive(X)
        H = self.model.transform(Xp)
        W = self.model.components_
        X_hat = H @ W
        return SynergyResult(H=H, W=W, vaf=variance_accounted_for(Xp, X_hat))

    def reconstruct(self, H: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted")
        return H @ self.model.components_


def compute_dH(H: np.ndarray) -> np.ndarray:
    return np.vstack([np.zeros((1, H.shape[1])), np.diff(H, axis=0)])


def compute_d2H(H: np.ndarray) -> np.ndarray:
    dH = compute_dH(H)
    return compute_dH(dH)


def choose_n_synergies(X: np.ndarray, k_values=range(2, 9), threshold: float = 0.90, random_state: int = 42):
    results = []
    for k in k_values:
        nmf = NMFSynergyExtractor(n_synergies=k, random_state=random_state)
        fit = nmf.fit_transform(X)
        results.append((k, fit.vaf))
        if fit.vaf >= threshold:
            return results, k
    return results, k_values[-1]


def compute_synergy_state(H: np.ndarray, order: int = 2) -> np.ndarray:
    parts = [H]
    if order >= 1:
        dH = compute_dH(H)
        parts.append(dH)
    if order >= 2:
        d2H = compute_d2H(H)
        parts.append(d2H)
    return np.concatenate(parts, axis=1)
