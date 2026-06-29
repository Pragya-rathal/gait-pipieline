from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_f64(x: np.ndarray, device: torch.device) -> torch.Tensor:
    """Cast to float64 on device; ridge regression benefits from full precision."""
    return torch.as_tensor(x, dtype=torch.float64, device=device)


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@dataclass
class LinearStateSpaceModel:
    A: np.ndarray
    B: Optional[np.ndarray] = None
    bias: Optional[np.ndarray] = None
    ridge: float = 1e-6

    # Cached GPU tensors — populated lazily
    _A_t: Optional[torch.Tensor] = field(default=None, repr=False, compare=False)
    _B_t: Optional[torch.Tensor] = field(default=None, repr=False, compare=False)
    _bias_t: Optional[torch.Tensor] = field(default=None, repr=False, compare=False)
    _device: Optional[torch.device] = field(default=None, repr=False, compare=False)

    def _ensure_tensors(self, device: torch.device) -> None:
        if self._device == device and self._A_t is not None:
            return
        self._A_t = torch.as_tensor(self.A, dtype=torch.float64, device=device)
        self._B_t = (
            torch.as_tensor(self.B, dtype=torch.float64, device=device)
            if self.B is not None
            else None
        )
        self._bias_t = (
            torch.as_tensor(self.bias, dtype=torch.float64, device=device)
            if self.bias is not None
            else None
        )
        self._device = device

    def predict_next(
        self,
        x: np.ndarray,
        u: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        device = _get_device()
        self._ensure_tensors(device)
        xt = torch.as_tensor(x, dtype=torch.float64, device=device).reshape(-1)
        y = self._A_t @ xt
        if self._B_t is not None and u is not None:
            y = y + self._B_t @ torch.as_tensor(u, dtype=torch.float64, device=device).reshape(-1)
        if self._bias_t is not None:
            y = y + self._bias_t
        return _to_numpy(y)

    def forecast(
        self,
        x0: np.ndarray,
        u_seq: Optional[np.ndarray] = None,
        steps: int = 1,
    ) -> np.ndarray:
        device = _get_device()
        self._ensure_tensors(device)
        x = torch.as_tensor(x0, dtype=torch.float64, device=device).reshape(-1)
        out = [_to_numpy(x)]
        U = (
            torch.as_tensor(u_seq, dtype=torch.float64, device=device)
            if u_seq is not None
            else None
        )
        for k in range(steps):
            y = self._A_t @ x
            if self._B_t is not None and U is not None:
                ui = U[min(k, len(U) - 1)]
                y = y + self._B_t @ ui
            if self._bias_t is not None:
                y = y + self._bias_t
            x = y
            out.append(_to_numpy(x))
        return np.asarray(out)


# ---------------------------------------------------------------------------
# Fitting helpers
# ---------------------------------------------------------------------------

def _ridge_solve(Z: torch.Tensor, Y: torch.Tensor, ridge: float) -> torch.Tensor:
    """Theta = (Z^T Z + ridge * I)^{-1} Z^T Y  — on GPU with float64."""
    reg = ridge * torch.eye(Z.shape[1], dtype=torch.float64, device=Z.device)
    return torch.linalg.solve(Z.T @ Z + reg, Z.T @ Y)


def _build_theta(
    X: torch.Tensor,
    Y: torch.Tensor,
    U: Optional[torch.Tensor],
    include_bias: bool,
    ridge: float,
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Returns (A, B, bias) as numpy arrays via a single GPU solve."""
    Z = X
    if U is not None:
        Z = torch.cat([Z, U], dim=1)
    if include_bias:
        Z = torch.cat([Z, torch.ones(len(Z), 1, dtype=torch.float64, device=Z.device)], dim=1)

    Theta = _ridge_solve(Z, Y, ridge)          # (features, n_state)

    n_state = X.shape[1]
    A = Theta[:n_state].T

    B: Optional[torch.Tensor] = None
    if U is not None:
        n_input = U.shape[1]
        B = Theta[n_state : n_state + n_input].T

    bias: Optional[torch.Tensor] = None
    if include_bias:
        bias = Theta[-1]

    return A, B, bias


def fit_linear_state_space(
    states: np.ndarray,
    inputs: Optional[np.ndarray] = None,
    ridge: float = 1e-6,
    include_bias: bool = True,
) -> LinearStateSpaceModel:
    states = np.asarray(states, dtype=float)
    if states.ndim != 2 or len(states) < 2:
        raise ValueError("states must be 2D with at least 2 timesteps")

    device = _get_device()
    St = _to_f64(states, device)
    Xt, Yt = St[:-1], St[1:]

    Ut: Optional[torch.Tensor] = None
    if inputs is not None:
        inp = np.asarray(inputs, dtype=float)
        if len(inp) != len(states):
            raise ValueError("inputs must match states length")
        Ut = _to_f64(inp, device)[:-1]

    A_t, B_t, bias_t = _build_theta(Xt, Yt, Ut, include_bias, ridge)

    return LinearStateSpaceModel(
        A=_to_numpy(A_t),
        B=_to_numpy(B_t) if B_t is not None else None,
        bias=_to_numpy(bias_t) if bias_t is not None else None,
        ridge=ridge,
    )


def fit_linear_state_space_from_sequences(
    states_list: Sequence[np.ndarray],
    inputs_list: Optional[Sequence[np.ndarray]] = None,
    ridge: float = 1e-6,
    include_bias: bool = True,
) -> LinearStateSpaceModel:
    if inputs_list is not None and len(inputs_list) != len(states_list):
        raise ValueError("inputs_list must match states_list length")

    device = _get_device()
    Xs_list: list[torch.Tensor] = []
    Ys_list: list[torch.Tensor] = []
    Us_list: list[torch.Tensor] = []

    for i, states in enumerate(states_list):
        s = np.asarray(states, dtype=float)
        if len(s) < 2:
            continue
        St = _to_f64(s, device)
        Xs_list.append(St[:-1])
        Ys_list.append(St[1:])
        if inputs_list is not None:
            u = np.asarray(inputs_list[i], dtype=float)
            if len(u) != len(s):
                raise ValueError("inputs and states must have same length per sequence")
            Us_list.append(_to_f64(u, device)[:-1])

    if not Xs_list:
        raise ValueError("Need at least one sequence with length > 1")

    X = torch.cat(Xs_list, dim=0)
    Y = torch.cat(Ys_list, dim=0)
    U = torch.cat(Us_list, dim=0) if Us_list else None

    A_t, B_t, bias_t = _build_theta(X, Y, U, include_bias, ridge)

    return LinearStateSpaceModel(
        A=_to_numpy(A_t),
        B=_to_numpy(B_t) if B_t is not None else None,
        bias=_to_numpy(bias_t) if bias_t is not None else None,
        ridge=ridge,
    )


# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------

def forecast_horizon_sequence(
    model: LinearStateSpaceModel,
    states: np.ndarray,
    inputs: Optional[np.ndarray],
    horizon_steps: int,
) -> np.ndarray:
    states = np.asarray(states, dtype=float)
    T, n_state = states.shape
    device = _get_device()
    model._ensure_tensors(device)

    if horizon_steps == 0:
        return states.copy()

    St = torch.as_tensor(states, dtype=torch.float64, device=device)
    Ut = (
        torch.as_tensor(inputs, dtype=torch.float64, device=device)
        if inputs is not None
        else None
    )

    out = torch.full((T, n_state), float("nan"), dtype=torch.float64, device=device)

    # Vectorise over the horizon loop; for each valid t, roll out h steps.
    # For typical horizon_steps (1–20) a Python loop over h is fine; the
    # heavy work (matrix-vector products) runs on GPU.
    A = model._A_t           # (n_state, n_state)
    B = model._B_t           # (n_state, n_input) or None
    bias = model._bias_t     # (n_state,) or None

    valid_T = T - horizon_steps
    if valid_T <= 0:
        return _to_numpy(out)

    # x: (valid_T, n_state) — batch of initial states
    x = St[:valid_T]

    for h in range(horizon_steps):
        # y = x @ A^T  (batched matmul)
        y = x @ A.T
        if B is not None and Ut is not None:
            u_idx = torch.clamp(
                torch.arange(h, valid_T + h, device=device),
                max=T - 1,
            )
            y = y + Ut[u_idx] @ B.T
        if bias is not None:
            y = y + bias
        x = y

    out[:valid_T] = x
    return _to_numpy(out)


def forecast_latent_states(
    model: LinearStateSpaceModel,
    x0: np.ndarray,
    u_seq: Optional[np.ndarray],
    steps: int,
) -> np.ndarray:
    return model.forecast(x0=x0, u_seq=u_seq, steps=steps)
