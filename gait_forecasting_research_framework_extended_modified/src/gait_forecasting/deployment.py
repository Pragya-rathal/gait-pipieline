
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence
import io
import tempfile
import time

import numpy as np
import joblib

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None
    nn = None


@dataclass
class DeploymentMetrics:
    params: int
    memory_bytes: int
    latency_ms: float
    flops: float
    extra: Dict[str, Any]


def _torch_param_count(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def _torch_memory_bytes(model) -> int:
    return int(sum(p.numel() * p.element_size() for p in model.parameters()) + sum(b.numel() * b.element_size() for b in model.buffers()))


def _file_size_bytes(obj) -> int:
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=True) as f:
        joblib.dump(obj, f.name)
        return Path(f.name).stat().st_size


def _estimate_torch_flops(model, sample_input: np.ndarray) -> float:
    # Rough, architecture-aware estimate; intended for reporting and comparison.
    if torch is None or nn is None:
        return 0.0
    if hasattr(model, "net"):  # MLP
        dims = []
        for layer in model.net:
            if isinstance(layer, nn.Linear):
                dims.append((layer.in_features, layer.out_features))
        return float(sum(2 * a * b for a, b in dims))
    if hasattr(model, "gru"):
        seq_len, input_dim = sample_input.shape[1], sample_input.shape[2]
        h = model.gru.hidden_size
        layers = model.gru.num_layers
        bidir = 2 if getattr(model.gru, "bidirectional", False) else 1
        # per time step: 3 gates * (input*hidden + hidden*hidden)
        return float(seq_len * layers * bidir * 2 * 3 * (input_dim * h + h * h))
    if hasattr(model, "lstm"):
        seq_len, input_dim = sample_input.shape[1], sample_input.shape[2]
        h = model.lstm.hidden_size
        layers = model.lstm.num_layers
        bidir = 2 if getattr(model.lstm, "bidirectional", False) else 1
        # 4 gates for LSTM
        return float(seq_len * layers * bidir * 2 * 4 * (input_dim * h + h * h))
    if hasattr(model, "tcn"):
        # approx for 1D conv blocks
        flops = 0.0
        in_ch = sample_input.shape[2]
        seq_len = sample_input.shape[1]
        for block in model.tcn:
            if hasattr(block, "conv1"):
                out_ch = block.conv1.out_channels
                k = block.conv1.kernel_size[0]
                flops += 2.0 * seq_len * in_ch * out_ch * k
                flops += 2.0 * seq_len * out_ch * out_ch * k
                in_ch = out_ch
        return float(flops)
    return float(_torch_param_count(model) * 2)


def _estimate_sklearn_rf_flops(model, sample_input: np.ndarray) -> float:
    # each node visit involves a comparison; rough average across all trees
    total_nodes = 0
    for est in getattr(model, "estimators_", []):
        total_nodes += int(getattr(est.tree_, "node_count", 0))
    return float(total_nodes)


def measure_latency_ms(model, sample_input: np.ndarray, predict_fn=None, n_warmup: int = 5, n_runs: int = 30) -> float:
    predict_fn = predict_fn or (lambda m, x: m.predict(x))
    for _ in range(n_warmup):
        try:
            predict_fn(model, sample_input)
        except Exception:
            pass
    t0 = time.perf_counter()
    for _ in range(n_runs):
        predict_fn(model, sample_input)
    t1 = time.perf_counter()
    return 1000.0 * (t1 - t0) / max(1, n_runs)


def summarize_deployment(model, sample_input: np.ndarray, model_kind: str = "auto") -> DeploymentMetrics:
    extra: Dict[str, Any] = {}
    if torch is not None and isinstance(model, nn.Module):
        params = _torch_param_count(model)
        memory = _torch_memory_bytes(model)
        flops = _estimate_torch_flops(model, sample_input)
        device = next(model.parameters()).device if any(True for _ in model.parameters()) else torch.device("cpu")
        latency = measure_latency_ms(
            model,
            sample_input,
            predict_fn=lambda m, x: m(torch.tensor(x, dtype=torch.float32, device=device)).detach().cpu().numpy(),
        )
        extra["framework"] = "torch"
    else:
        params = int(getattr(model, "n_params_", 0) or getattr(model, "n_features_in_", 0) or 0)
        try:
            memory = _file_size_bytes(model)
        except Exception:
            memory = int(params * 8)
        if hasattr(model, "estimators_"):
            flops = _estimate_sklearn_rf_flops(model, sample_input)
        elif hasattr(model, "coefs_"):
            flops = float(sum(w.size for w in model.coefs_) * 2)
        else:
            flops = float(params * 2)
        latency = measure_latency_ms(model, sample_input, predict_fn=lambda m, x: m.predict(x))
        extra["framework"] = "sklearn"
    return DeploymentMetrics(params=params, memory_bytes=memory, latency_ms=float(latency), flops=float(flops), extra=extra)


def deployment_table(metrics_by_name: Dict[str, DeploymentMetrics]) -> str:
    lines = ["| Model | Params | Memory (KB) | Latency (ms) | FLOPs (approx) |", "|---|---:|---:|---:|---:|"]
    for name, m in metrics_by_name.items():
        lines.append(f"| {name} | {m.params} | {m.memory_bytes/1024:.1f} | {m.latency_ms:.2f} | {m.flops:.1f} |")
    return "\n".join(lines)
