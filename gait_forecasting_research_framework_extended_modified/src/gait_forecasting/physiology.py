from __future__ import annotations

import contextlib
from typing import Literal, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


FusionMethod = Literal["concat", "weighted", "learnable", "attention"]


# ---------------------------------------------------------------------------
# Fusion modules
# ---------------------------------------------------------------------------

class _WeightedFusion(nn.Module):
    def __init__(self, n_streams: int, stream_dim: int) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.ones(n_streams) / n_streams)

    def forward(self, streams: list[torch.Tensor]) -> torch.Tensor:
        w = torch.softmax(self.weights, dim=0)
        return sum(w[i] * streams[i] for i in range(len(streams)))


class _LearnableFusion(nn.Module):
    def __init__(self, stream_dims: Sequence[int], output_dim: int) -> None:
        super().__init__()
        total = sum(stream_dims)
        self.proj = nn.Sequential(
            nn.Linear(total, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
        )

    def forward(self, streams: list[torch.Tensor]) -> torch.Tensor:
        return self.proj(torch.cat(streams, dim=-1))


class _AttentionFusion(nn.Module):
    def __init__(self, stream_dims: Sequence[int], output_dim: int) -> None:
        super().__init__()
        self.projs = nn.ModuleList([nn.Linear(d, output_dim) for d in stream_dims])
        self.attn = nn.Linear(output_dim, 1)

    def forward(self, streams: list[torch.Tensor]) -> torch.Tensor:
        projected = [p(s) for p, s in zip(self.projs, streams)]
        stacked = torch.stack(projected, dim=-2)      # (..., n_streams, D)
        scores = self.attn(torch.tanh(stacked))       # (..., n_streams, 1)
        weights = torch.softmax(scores, dim=-2)
        return (weights * stacked).sum(dim=-2)        # (..., D)


# ---------------------------------------------------------------------------
# PhysiologicalFusion
# ---------------------------------------------------------------------------

class PhysiologicalFusion(nn.Module):
    """
    Fuses W (synergy weight matrix), H (activation matrix), and dynamic features
    into a single physiological feature vector.

    Parameters
    ----------
    w_dim : int
        Flattened size of W (n_muscles × n_synergies).
    h_dim : int
        Flattened size of H (window_length × n_synergies) or summary stats dim.
    dynamic_dim : int
        Size of the dynamic feature vector from ``dynamics.py``.
    output_dim : int
        Output feature dimension.
    method : FusionMethod
        One of ``"concat"``, ``"weighted"``, ``"learnable"``, ``"attention"``.
    """

    def __init__(
        self,
        w_dim: int,
        h_dim: int,
        dynamic_dim: int,
        output_dim: int,
        method: FusionMethod = "learnable",
    ) -> None:
        super().__init__()
        self.w_dim = w_dim
        self.h_dim = h_dim
        self.dynamic_dim = dynamic_dim
        self.output_dim = output_dim
        self.method = method

        stream_dims = [w_dim, h_dim, dynamic_dim]

        if method == "concat":
            self.proj = nn.Sequential(
                nn.Linear(sum(stream_dims), output_dim),
                nn.LayerNorm(output_dim),
                nn.ReLU(),
            )
        elif method == "weighted":
            self._fuse = _WeightedFusion(n_streams=3, stream_dim=output_dim)
            self.stream_projs = nn.ModuleList(
                [nn.Linear(d, output_dim) for d in stream_dims]
            )
        elif method == "learnable":
            self._fuse = _LearnableFusion(stream_dims, output_dim)
        elif method == "attention":
            self._fuse = _AttentionFusion(stream_dims, output_dim)
        else:
            raise ValueError(f"Unknown fusion method: {method!r}")

    def forward(
        self,
        W_flat: torch.Tensor,
        H_flat: torch.Tensor,
        dynamic: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        W_flat : (..., w_dim)
        H_flat : (..., h_dim)
        dynamic : (..., dynamic_dim)

        Returns
        -------
        fused : (..., output_dim)
        """
        streams = [W_flat, H_flat, dynamic]
        if self.method == "concat":
            return self.proj(torch.cat(streams, dim=-1))
        elif self.method == "weighted":
            projected = [p(s) for p, s in zip(self.stream_projs, streams)]
            return self._fuse(projected)
        else:
            return self._fuse(streams)

    def fuse_numpy(
        self,
        W: np.ndarray,
        H: np.ndarray,
        dynamic: np.ndarray,
    ) -> np.ndarray:
        """
        Convenience method for numpy inputs. Moves to device, runs forward, returns numpy.

        Parameters
        ----------
        W : (n_muscles, n_synergies) ndarray — synergy weight matrix
        H : (T, n_synergies) or (n_synergies,) ndarray — activation or summary
        dynamic : (D,) ndarray — dynamic feature vector
        """
        device = _get_device()
        self.to(device)
        self.eval()

        W_t = torch.as_tensor(W.ravel(), dtype=torch.float32, device=device).unsqueeze(0)
        H_t = torch.as_tensor(H.ravel(), dtype=torch.float32, device=device).unsqueeze(0)
        d_t = torch.as_tensor(dynamic.ravel(), dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad(), _autocast_ctx():
            out = self(W_t, H_t, d_t)
        return out.squeeze(0).detach().cpu().numpy()


def build_physiological_fusion(
    n_muscles: int,
    n_synergies: int,
    h_summary_dim: int,
    dynamic_dim: int,
    output_dim: int = 64,
    method: FusionMethod = "learnable",
) -> PhysiologicalFusion:
    """
    Factory for PhysiologicalFusion with standard dimensions.

    Parameters
    ----------
    n_muscles : int
    n_synergies : int
    h_summary_dim : int
        Dimension of the H summary (e.g. n_synergies for mean, or full T×K for window).
    dynamic_dim : int
        Output dimension of SynergyDynamics.
    output_dim : int
    method : FusionMethod
    """
    w_dim = n_muscles * n_synergies
    return PhysiologicalFusion(
        w_dim=w_dim,
        h_dim=h_summary_dim,
        dynamic_dim=dynamic_dim,
        output_dim=output_dim,
        method=method,
    )
