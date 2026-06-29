from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import NMF

from .preprocessing import minmax_positive


def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


@dataclass
class SynergyResult:
    H: np.ndarray
    W: np.ndarray
    vaf: float


def variance_accounted_for(X: np.ndarray, X_hat: np.ndarray) -> float:
    device = _get_device()
    Xt = _to_tensor(X, device)
    Xht = _to_tensor(X_hat, device)
    num = torch.sum((Xt - Xht) ** 2)
    den = torch.sum(Xt ** 2) + 1e-12
    return float((1.0 - num / den).item())


def _variance_accounted_for_tensors(X: torch.Tensor, X_hat: torch.Tensor) -> float:
    num = torch.sum((X - X_hat) ** 2)
    den = torch.sum(X ** 2) + 1e-12
    return float((1.0 - num / den).item())


class NMFSynergyExtractor:
    def __init__(self, n_synergies: int = 5, max_iter: int = 1000, random_state: int = 42):
        self.n_synergies = n_synergies
        self.max_iter = max_iter
        self.random_state = random_state
        self.model: NMF | None = None
        self._W_tensor: torch.Tensor | None = None
        self._device: torch.device = _get_device()

    def fit(self, X: np.ndarray) -> "NMFSynergyExtractor":
        Xp = minmax_positive(X)
        self.model = NMF(
            n_components=self.n_synergies,
            init="nndsvda",
            random_state=self.random_state,
            max_iter=self.max_iter,
        )
        self.model.fit(Xp)
        self._cache_components()
        return self

    def _cache_components(self) -> None:
        if self.model is not None:
            self._W_tensor = _to_tensor(self.model.components_, self._device)

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted")
        Xp = minmax_positive(X)
        return self.model.transform(Xp)

    def transform_tensor(self, X: torch.Tensor) -> torch.Tensor:
        """Transform already-positive tensor using cached W; avoids sklearn overhead for GPU paths."""
        if self._W_tensor is None:
            raise RuntimeError("Model not fitted")
        # Non-negative least squares projection: H = X @ W^T @ (W @ W^T)^{-1}
        # For inference we use the sklearn path to stay numerically consistent,
        # but keep result on GPU.
        H_np = self.model.transform(_to_numpy(X.cpu()))
        return _to_tensor(H_np, X.device)

    def fit_transform(self, X: np.ndarray) -> SynergyResult:
        self.fit(X)
        assert self.model is not None
        device = self._device
        Xp = minmax_positive(X)
        H = self.model.transform(Xp)
        W = self.model.components_

        Ht = _to_tensor(H, device)
        Wt = _to_tensor(W, device)
        Xpt = _to_tensor(Xp, device)
        X_hat_t = Ht @ Wt
        vaf = _variance_accounted_for_tensors(Xpt, X_hat_t)
        return SynergyResult(H=H, W=W, vaf=vaf)

    def reconstruct(self, H: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted")
        if self._W_tensor is not None:
            Ht = _to_tensor(H, self._device)
            return _to_numpy(Ht @ self._W_tensor)
        return H @ self.model.components_

    def reconstruct_tensor(self, H: torch.Tensor) -> torch.Tensor:
        if self._W_tensor is None:
            raise RuntimeError("Model not fitted")
        return H @ self._W_tensor.to(H.device)


def compute_dH(H: np.ndarray) -> np.ndarray:
    device = _get_device()
    Ht = _to_tensor(H, device)
    diff = torch.diff(Ht, dim=0)
    dH = torch.cat([torch.zeros(1, Ht.shape[1], device=device), diff], dim=0)
    return _to_numpy(dH)


def compute_dH_tensor(H: torch.Tensor) -> torch.Tensor:
    diff = torch.diff(H, dim=0)
    return torch.cat([torch.zeros(1, H.shape[1], device=H.device, dtype=H.dtype), diff], dim=0)


def compute_d2H(H: np.ndarray) -> np.ndarray:
    device = _get_device()
    Ht = _to_tensor(H, device)
    dH = compute_dH_tensor(Ht)
    d2H = compute_dH_tensor(dH)
    return _to_numpy(d2H)


def compute_d2H_tensor(H: torch.Tensor) -> torch.Tensor:
    dH = compute_dH_tensor(H)
    return compute_dH_tensor(dH)


def compute_synergy_state(H: np.ndarray, order: int = 2) -> np.ndarray:
    device = _get_device()
    Ht = _to_tensor(H, device)
    parts = [Ht]
    if order >= 1:
        dH = compute_dH_tensor(Ht)
        parts.append(dH)
    if order >= 2:
        d2H = compute_dH_tensor(parts[1])
        parts.append(d2H)
    return _to_numpy(torch.cat(parts, dim=1))


def compute_synergy_state_tensor(H: torch.Tensor, order: int = 2) -> torch.Tensor:
    parts = [H]
    if order >= 1:
        dH = compute_dH_tensor(H)
        parts.append(dH)
    if order >= 2:
        d2H = compute_dH_tensor(parts[1])
        parts.append(d2H)
    return torch.cat(parts, dim=1)


def choose_n_synergies(
    X: np.ndarray,
    k_values: Sequence[int] = range(2, 9),
    threshold: float = 0.90,
    random_state: int = 42,
) -> Tuple[List[Tuple[int, float]], int]:
    results: List[Tuple[int, float]] = []
    last_k = list(k_values)[-1]
    for k in k_values:
        nmf = NMFSynergyExtractor(n_synergies=k, random_state=random_state)
        fit = nmf.fit_transform(X)
        results.append((k, fit.vaf))
        if fit.vaf >= threshold:
            return results, k
    return results, last_k
