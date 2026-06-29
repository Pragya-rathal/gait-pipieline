from __future__ import annotations

import contextlib
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np


def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _amp_enabled() -> bool:
    return torch.cuda.is_available()


@contextlib.contextmanager
def _autocast_ctx():
    if _amp_enabled():
        with torch.cuda.amp.autocast():
            yield
    else:
        yield


def _to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Derivative helpers (GPU tensors in, GPU tensors out)
# ---------------------------------------------------------------------------

def _first_derivative(H: torch.Tensor) -> torch.Tensor:
    diff = torch.diff(H, dim=0)
    return torch.cat([torch.zeros(1, H.shape[1], device=H.device, dtype=H.dtype), diff], dim=0)


def _second_derivative(H: torch.Tensor) -> torch.Tensor:
    return _first_derivative(_first_derivative(H))


# ---------------------------------------------------------------------------
# Individual feature functions — all operate on (T, K) GPU tensors
# ---------------------------------------------------------------------------

def _mean_activation(H: torch.Tensor) -> torch.Tensor:
    return H.mean(dim=0)


def _variance_activation(H: torch.Tensor) -> torch.Tensor:
    return H.var(dim=0, unbiased=False)


def _energy(H: torch.Tensor) -> torch.Tensor:
    return (H ** 2).sum(dim=0)


def _rms(H: torch.Tensor) -> torch.Tensor:
    return torch.sqrt((H ** 2).mean(dim=0) + 1e-12)


def _peak_activation(H: torch.Tensor) -> torch.Tensor:
    return H.max(dim=0).values


def _activation_duration(H: torch.Tensor, threshold_frac: float = 0.1) -> torch.Tensor:
    threshold = H.max(dim=0).values * threshold_frac
    active = (H > threshold.unsqueeze(0)).float()
    return active.sum(dim=0) / max(H.shape[0], 1)


def _spectral_entropy(H: torch.Tensor) -> torch.Tensor:
    T, K = H.shape
    nfft = max(2, int(2 ** torch.ceil(torch.log2(torch.tensor(T, dtype=torch.float32))).item()))
    Hf = torch.fft.rfft(H, n=nfft, dim=0)
    power = (Hf.real ** 2 + Hf.imag ** 2)
    power_sum = power.sum(dim=0, keepdim=True) + 1e-12
    p = power / power_sum
    return -(p * torch.log(p + 1e-12)).sum(dim=0)


def _zero_crossing_rate(H: torch.Tensor) -> torch.Tensor:
    signs = torch.sign(H)
    crossings = (signs[1:] != signs[:-1]).float().sum(dim=0)
    return crossings / max(H.shape[0] - 1, 1)


def _cross_synergy_correlation(H: torch.Tensor) -> torch.Tensor:
    T, K = H.shape
    if K < 2:
        return torch.zeros(0, device=H.device, dtype=H.dtype)
    Hn = H - H.mean(dim=0, keepdim=True)
    std = Hn.std(dim=0, unbiased=False) + 1e-12
    Hn = Hn / std.unsqueeze(0)
    corr = (Hn.T @ Hn) / T
    indices = torch.triu_indices(K, K, offset=1)
    return corr[indices[0], indices[1]]


def _temporal_smoothness(H: torch.Tensor) -> torch.Tensor:
    dH = _first_derivative(H)
    return (dH ** 2).mean(dim=0)


def _persistence(H: torch.Tensor, threshold_frac: float = 0.1) -> torch.Tensor:
    threshold = H.max(dim=0).values * threshold_frac
    active = (H > threshold.unsqueeze(0)).float()
    changes = (active[1:] - active[:-1]).abs().sum(dim=0)
    return 1.0 / (changes + 1.0)


def _synergy_similarity(H: torch.Tensor) -> torch.Tensor:
    T, K = H.shape
    if K < 2:
        return torch.zeros(0, device=H.device, dtype=H.dtype)
    norm = F.normalize(H.T, dim=1)
    sim = norm @ norm.T
    indices = torch.triu_indices(K, K, offset=1)
    return sim[indices[0], indices[1]]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class SynergyDynamics:
    """
    Computes a rich dynamic feature vector from a synergy activation matrix H.

    Parameters
    ----------
    include_cross_synergy : bool
        Whether to include cross-synergy correlation and similarity features.
        These are O(K^2) features and may be disabled for single-synergy inputs.
    """

    def __init__(self, include_cross_synergy: bool = True) -> None:
        self.include_cross_synergy = include_cross_synergy
        self._feature_names: Optional[List[str]] = None

    def compute(self, H: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        H : (T, K) ndarray
            Synergy activation matrix.

        Returns
        -------
        features : (D,) ndarray
            Flat dynamic feature vector.
        """
        device = _get_device()
        with _autocast_ctx():
            feat = self._compute_tensor(_to_tensor(H, device))
        return _to_numpy(feat)

    def compute_batch(self, H_batch: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        H_batch : (N, T, K) ndarray

        Returns
        -------
        features : (N, D) ndarray
        """
        device = _get_device()
        N = H_batch.shape[0]
        results: List[torch.Tensor] = []
        Hb = _to_tensor(H_batch, device)
        with _autocast_ctx():
            for i in range(N):
                results.append(self._compute_tensor(Hb[i]))
        return _to_numpy(torch.stack(results, dim=0))

    def _compute_tensor(self, H: torch.Tensor) -> torch.Tensor:
        T, K = H.shape
        dH = _first_derivative(H)
        d2H = _second_derivative(H)
        parts: List[torch.Tensor] = [
            _mean_activation(H),
            _variance_activation(H),
            _energy(H),
            _rms(H),
            _peak_activation(H),
            _activation_duration(H),
            _spectral_entropy(H),
            _zero_crossing_rate(H),
            _temporal_smoothness(H),
            _persistence(H),
            _mean_activation(dH),
            _rms(dH),
            _mean_activation(d2H),
            _rms(d2H),
        ]
        if self.include_cross_synergy and K >= 2:
            parts.append(_cross_synergy_correlation(H))
            parts.append(_synergy_similarity(H))
        return torch.cat(parts, dim=0)

    def feature_dim(self, n_synergies: int) -> int:
        per_synergy = 14
        cross = n_synergies * (n_synergies - 1) // 2
        extra = cross * 2 if self.include_cross_synergy and n_synergies >= 2 else 0
        return per_synergy * n_synergies + extra

    def feature_names(self, n_synergies: int) -> List[str]:
        K = n_synergies
        names: List[str] = []
        for prefix, dims in [
            ("H", K), ("dH", K), ("d2H", K),
        ]:
            pass
        scalar_feats = [
            "mean", "var", "energy", "rms", "peak", "duration",
            "spec_entropy", "zcr", "smoothness", "persistence",
        ]
        dscalar_feats = ["d_mean", "d_rms", "d2_mean", "d2_rms"]
        for f in scalar_feats:
            for k in range(K):
                names.append(f"{f}_syn{k}")
        for f in dscalar_feats:
            for k in range(K):
                names.append(f"{f}_syn{k}")
        if self.include_cross_synergy and K >= 2:
            for i in range(K):
                for j in range(i + 1, K):
                    names.append(f"corr_syn{i}_syn{j}")
            for i in range(K):
                for j in range(i + 1, K):
                    names.append(f"sim_syn{i}_syn{j}")
        return names


def compute_synergy_dynamics(H: np.ndarray, include_cross_synergy: bool = True) -> np.ndarray:
    """Convenience wrapper; returns (D,) feature vector."""
    return SynergyDynamics(include_cross_synergy=include_cross_synergy).compute(H)
