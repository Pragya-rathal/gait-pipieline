
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class LinearStateSpaceModel:
    A: np.ndarray
    B: np.ndarray | None = None
    bias: np.ndarray | None = None
    ridge: float = 1e-6

    def predict_next(self, x: np.ndarray, u: np.ndarray | None = None) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        y = self.A @ x
        if self.B is not None and u is not None:
            y = y + self.B @ np.asarray(u, dtype=float).reshape(-1)
        if self.bias is not None:
            y = y + self.bias
        return y

    def forecast(self, x0: np.ndarray, u_seq: np.ndarray | None = None, steps: int = 1) -> np.ndarray:
        x = np.asarray(x0, dtype=float).reshape(-1)
        out = [x.copy()]
        for k in range(steps):
            u = None if u_seq is None else np.asarray(u_seq[min(k, len(u_seq) - 1)], dtype=float).reshape(-1)
            x = self.predict_next(x, u)
            out.append(x.copy())
        return np.asarray(out)


def fit_linear_state_space(
    states: np.ndarray,
    inputs: np.ndarray | None = None,
    ridge: float = 1e-6,
    include_bias: bool = True,
) -> LinearStateSpaceModel:
    """
    Fit x(k+1) = A x(k) + B u(k) (+ b) using ridge-regularized least squares.
    states: (T, n_state)
    inputs: (T, n_input) or None
    """
    states = np.asarray(states, dtype=float)
    if states.ndim != 2 or len(states) < 2:
        raise ValueError("states must be 2D with at least 2 timesteps")

    X = states[:-1]
    Y = states[1:]
    if inputs is not None:
        U = np.asarray(inputs, dtype=float)
        if len(U) != len(states):
            raise ValueError("inputs must match states length")
        U = U[:-1]
        Z = np.hstack([X, U])
    else:
        U = None
        Z = X

    if include_bias:
        Z = np.hstack([Z, np.ones((len(Z), 1), dtype=float)])

    reg = ridge * np.eye(Z.shape[1], dtype=float)
    Theta = np.linalg.solve(Z.T @ Z + reg, Z.T @ Y)  # (features, state)

    if inputs is not None:
        n_state = X.shape[1]
        n_input = U.shape[1]
        A = Theta[:n_state].T
        B = Theta[n_state:n_state + n_input].T
        bias = Theta[-1].T if include_bias else None
        return LinearStateSpaceModel(A=A, B=B, bias=bias, ridge=ridge)
    else:
        A = Theta[:X.shape[1]].T
        bias = Theta[-1].T if include_bias else None
        return LinearStateSpaceModel(A=A, B=None, bias=bias, ridge=ridge)


def fit_linear_state_space_from_sequences(
    states_list: Sequence[np.ndarray],
    inputs_list: Sequence[np.ndarray] | None = None,
    ridge: float = 1e-6,
    include_bias: bool = True,
) -> LinearStateSpaceModel:
    Xs, Ys, Us = [], [], []
    if inputs_list is not None and len(inputs_list) != len(states_list):
        raise ValueError("inputs_list must match states_list length")
    for i, states in enumerate(states_list):
        states = np.asarray(states, dtype=float)
        if len(states) < 2:
            continue
        Xs.append(states[:-1])
        Ys.append(states[1:])
        if inputs_list is not None:
            u = np.asarray(inputs_list[i], dtype=float)
            if len(u) != len(states):
                raise ValueError("inputs and states must have same length per sequence")
            Us.append(u[:-1])
    if not Xs:
        raise ValueError("Need at least one sequence with length > 1")
    X = np.vstack(Xs)
    Y = np.vstack(Ys)
    if inputs_list is not None:
        U = np.vstack(Us)
        Z = np.hstack([X, U])
    else:
        Z = X
    if include_bias:
        Z = np.hstack([Z, np.ones((len(Z), 1), dtype=float)])
    reg = ridge * np.eye(Z.shape[1], dtype=float)
    Theta = np.linalg.solve(Z.T @ Z + reg, Z.T @ Y)
    if inputs_list is not None:
        n_state = X.shape[1]
        n_input = U.shape[1]
        A = Theta[:n_state].T
        B = Theta[n_state:n_state + n_input].T
        bias = Theta[-1].T if include_bias else None
        return LinearStateSpaceModel(A=A, B=B, bias=bias, ridge=ridge)
    A = Theta[:X.shape[1]].T
    bias = Theta[-1].T if include_bias else None
    return LinearStateSpaceModel(A=A, B=None, bias=bias, ridge=ridge)


def forecast_horizon_sequence(
    model: LinearStateSpaceModel,
    states: np.ndarray,
    inputs: np.ndarray | None,
    horizon_steps: int,
) -> np.ndarray:
    states = np.asarray(states, dtype=float)
    T = len(states)
    out = np.full_like(states, np.nan, dtype=float)
    for t in range(T):
        if t + horizon_steps >= T:
            break
        x = states[t]
        for h in range(horizon_steps):
            u = None
            if inputs is not None:
                idx = min(t + h, len(inputs) - 1)
                u = inputs[idx]
            x = model.predict_next(x, u)
        out[t] = x
    return out


def forecast_latent_states(
    model: LinearStateSpaceModel,
    x0: np.ndarray,
    u_seq: np.ndarray | None,
    steps: int,
) -> np.ndarray:
    return model.forecast(x0=x0, u_seq=u_seq, steps=steps)
