from __future__ import annotations

import csv
import gc
import json
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Sequence, Tuple

import numpy as np
import psutil
import torch
import torch.nn as nn
from torch.cuda.amp import autocast


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _amp_enabled() -> bool:
    return torch.cuda.is_available()


def _maybe_compile(model: nn.Module) -> nn.Module:
    if _amp_enabled() and hasattr(torch, "compile"):
        try:
            return torch.compile(model)
        except Exception:
            pass
    return model


# ---------------------------------------------------------------------------
# GPU / CPU / Memory utilities
# ---------------------------------------------------------------------------

def _gpu_vram_used_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / 1024 ** 2


def _gpu_vram_reserved_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_reserved() / 1024 ** 2


def _ram_used_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2


def _cpu_percent() -> float:
    return psutil.cpu_percent(interval=None)


def _try_nvml_gpu_util() -> float:
    """Return GPU utilisation % via pynvml; 0.0 when unavailable."""
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return float(util.gpu)
    except Exception:
        return 0.0


def _model_params(model: nn.Module) -> int:
    underlying = getattr(model, "_orig_mod", model)
    return sum(p.numel() for p in underlying.parameters())


def _model_size_mb(model: nn.Module) -> float:
    underlying = getattr(model, "_orig_mod", model)
    total = sum(
        p.numel() * p.element_size()
        for p in list(underlying.parameters()) + list(underlying.buffers())
    )
    return total / 1024 ** 2


def _estimate_flops(model: nn.Module, dummy: torch.Tensor) -> float:
    """
    Lightweight FLOPs estimate via hook-based MAC counting.
    Falls back to a parameter-proportional heuristic if hooks fail.
    """
    underlying = getattr(model, "_orig_mod", model)
    mac_count: List[float] = [0.0]

    def _linear_hook(m: nn.Linear, inp: Tuple, out: torch.Tensor) -> None:
        mac_count[0] += float(inp[0].numel() * m.out_features / inp[0].shape[-1])

    def _conv1d_hook(m: nn.Conv1d, inp: Tuple, out: torch.Tensor) -> None:
        B, C_in, L = inp[0].shape
        C_out, _, K = m.weight.shape
        groups = m.groups
        mac_count[0] += float(B * C_out * out.shape[-1] * (C_in // groups) * K)

    handles = []
    try:
        for mod in underlying.modules():
            if isinstance(mod, nn.Linear):
                handles.append(mod.register_forward_hook(_linear_hook))
            elif isinstance(mod, nn.Conv1d):
                handles.append(mod.register_forward_hook(_conv1d_hook))

        underlying.eval()
        with torch.no_grad():
            try:
                underlying(dummy.to(next(underlying.parameters()).device, non_blocking=True))
            except Exception:
                pass
    finally:
        for h in handles:
            h.remove()

    macs = mac_count[0]
    flops = macs * 2.0
    return flops if flops > 0.0 else float(_model_params(model)) * 2.0


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    model_name: str
    n_params: int
    model_size_mb: float
    train_time_s: float
    inference_latency_ms: float
    inference_latency_std_ms: float
    throughput_samples_per_s: float
    gpu_util_pct: float
    cpu_util_pct: float
    vram_used_mb: float
    ram_used_mb: float
    flops: float
    batch_size: int
    input_shape: Tuple[int, ...]
    device: str
    amp: bool
    compiled: bool
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["input_shape"] = list(self.input_shape)
        return d


# ---------------------------------------------------------------------------
# Timing context
# ---------------------------------------------------------------------------

@contextmanager
def _cuda_timer() -> Generator[Callable[[], float], None, None]:
    if torch.cuda.is_available():
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()
        elapsed: List[float] = [0.0]
        yield lambda: elapsed[0]
        end_evt.record()
        torch.cuda.synchronize()
        elapsed[0] = start_evt.elapsed_time(end_evt)
    else:
        t0 = time.perf_counter()
        elapsed = [0.0]
        yield lambda: elapsed[0]
        elapsed[0] = (time.perf_counter() - t0) * 1000.0


# ---------------------------------------------------------------------------
# CPU utilisation sampler (background thread)
# ---------------------------------------------------------------------------

class _CpuSampler:
    def __init__(self, interval: float = 0.1) -> None:
        self._samples: List[float] = []
        self._interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._samples.append(psutil.cpu_percent(interval=None))
            time.sleep(self._interval)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> float:
        self._stop.set()
        self._thread.join(timeout=2.0)
        return float(np.mean(self._samples)) if self._samples else 0.0


# ---------------------------------------------------------------------------
# Single-model benchmark
# ---------------------------------------------------------------------------

def benchmark_model(
    model: nn.Module,
    dummy_input: torch.Tensor,
    n_warmup: int = 10,
    n_runs: int = 100,
    compile_model: bool = False,
    use_amp: bool = True,
    model_name: str = "model",
) -> BenchmarkResult:
    """
    Benchmark a single nn.Module.

    Parameters
    ----------
    model : nn.Module
    dummy_input : representative input tensor (batch dimension first)
    n_warmup : number of warm-up forward passes before timing
    n_runs : number of timed forward passes
    compile_model : whether to call torch.compile
    use_amp : whether to use automatic mixed precision (GPU only)
    model_name : display name in results

    Returns
    -------
    BenchmarkResult
    """
    device = _get_device()
    amp_active = use_amp and _amp_enabled()

    if compile_model:
        model = _maybe_compile(model)

    model = model.to(device)
    model.eval()
    x = dummy_input.to(device, non_blocking=True)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    flops = _estimate_flops(model, x)

    for _ in range(n_warmup):
        with torch.no_grad():
            if amp_active:
                with autocast():
                    model(x)
            else:
                model(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    vram_before = _gpu_vram_used_mb()
    ram_before = _ram_used_mb()
    cpu_sampler = _CpuSampler()
    cpu_sampler.start()

    latencies: List[float] = []
    for _ in range(n_runs):
        with _cuda_timer() as get_ms:
            with torch.no_grad():
                if amp_active:
                    with autocast():
                        model(x)
                else:
                    model(x)
        latencies.append(get_ms())

    cpu_mean = cpu_sampler.stop()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    vram_after = _gpu_vram_used_mb()
    ram_after = _ram_used_mb()
    gpu_util = _try_nvml_gpu_util()

    lat_arr = np.array(latencies, dtype=float)
    mean_lat_ms = float(lat_arr.mean())
    std_lat_ms = float(lat_arr.std())
    batch = int(x.shape[0])
    throughput = (batch / (mean_lat_ms / 1000.0)) if mean_lat_ms > 0 else 0.0

    return BenchmarkResult(
        model_name=model_name,
        n_params=_model_params(model),
        model_size_mb=_model_size_mb(model),
        train_time_s=0.0,
        inference_latency_ms=mean_lat_ms,
        inference_latency_std_ms=std_lat_ms,
        throughput_samples_per_s=throughput,
        gpu_util_pct=gpu_util,
        cpu_util_pct=cpu_mean,
        vram_used_mb=max(0.0, vram_after - vram_before),
        ram_used_mb=max(0.0, ram_after - ram_before),
        flops=flops,
        batch_size=batch,
        input_shape=tuple(x.shape[1:]),
        device=str(device),
        amp=amp_active,
        compiled=compile_model,
    )


def benchmark_training(
    model: nn.Module,
    dummy_input: torch.Tensor,
    dummy_target: torch.Tensor,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    n_steps: int = 50,
    use_amp: bool = True,
    model_name: str = "model",
    **benchmark_kwargs: Any,
) -> BenchmarkResult:
    """
    Measure training throughput (forward + backward + optimizer step).

    Parameters
    ----------
    model : nn.Module
    dummy_input : input tensor (B, ...)
    dummy_target : target tensor (B, ...) for the loss function
    loss_fn : callable(output, target) -> scalar loss
    n_steps : number of training steps to time
    use_amp : enable AMP
    model_name : display name
    **benchmark_kwargs : forwarded to benchmark_model for inference measurement

    Returns
    -------
    BenchmarkResult with train_time_s populated
    """
    device = _get_device()
    amp_active = use_amp and _amp_enabled()

    model = model.to(device).train()
    x = dummy_input.to(device, non_blocking=True)
    y = dummy_target.to(device, non_blocking=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_active)

    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        if amp_active:
            with autocast():
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    cpu_sampler = _CpuSampler()
    cpu_sampler.start()
    t0 = time.perf_counter()

    for _ in range(n_steps):
        optimizer.zero_grad(set_to_none=True)
        if amp_active:
            with autocast():
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    train_time = time.perf_counter() - t0
    cpu_sampler.stop()

    model.eval()
    result = benchmark_model(model, dummy_input, model_name=model_name,
                              use_amp=use_amp, **benchmark_kwargs)
    result.train_time_s = train_time
    return result


# ---------------------------------------------------------------------------
# Standard model constructors for the framework benchmark suite
# ---------------------------------------------------------------------------

def _build_cnn(input_dim: int, seq_len: int, n_classes: int, dropout: float) -> nn.Module:
    return nn.Sequential(
        nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm1d(64),
        nn.Conv1d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm1d(128),
        nn.AdaptiveAvgPool1d(1),
        nn.Flatten(),
        nn.Dropout(dropout),
        nn.Linear(128, n_classes),
    )


class _CNNWrapper(nn.Module):
    def __init__(self, cnn: nn.Module) -> None:
        super().__init__()
        self.cnn = cnn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cnn(x.transpose(1, 2))


def _build_gru(input_dim: int, n_classes: int, hidden: int, layers: int, dropout: float) -> nn.Module:
    class _GRU(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gru = nn.GRU(input_dim, hidden, layers,
                              batch_first=True,
                              dropout=dropout if layers > 1 else 0.0)
            self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, n_classes))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out, _ = self.gru(x)
            return self.head(out[:, -1])

    return _GRU()


def _build_bilstm(input_dim: int, n_classes: int, hidden: int, layers: int, dropout: float) -> nn.Module:
    class _BiLSTM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden, layers,
                                batch_first=True, bidirectional=True,
                                dropout=dropout if layers > 1 else 0.0)
            self.head = nn.Sequential(nn.LayerNorm(hidden * 2), nn.Linear(hidden * 2, n_classes))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out, _ = self.lstm(x)
            return self.head(out[:, -1])

    return _BiLSTM()


def _build_tcn(input_dim: int, n_classes: int, channels: Sequence[int], kernel: int, dropout: float) -> nn.Module:
    class _Block(nn.Module):
        def __init__(self, ic: int, oc: int, k: int, d: int) -> None:
            super().__init__()
            pad = (k - 1) * d
            self.c1 = nn.Conv1d(ic, oc, k, padding=pad, dilation=d)
            self.c2 = nn.Conv1d(oc, oc, k, padding=pad, dilation=d)
            self.relu = nn.ReLU()
            self.drop = nn.Dropout(dropout)
            self.ds = nn.Conv1d(ic, oc, 1) if ic != oc else nn.Identity()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            r = self.ds(x)
            y = self.relu(self.drop(self.c1(x)[..., :r.shape[-1]]))
            y = self.relu(self.drop(self.c2(y)[..., :r.shape[-1]]))
            return self.relu(y + r)

    class _TCN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            blocks: List[nn.Module] = []
            prev = input_dim
            for i, ch in enumerate(channels):
                blocks.append(_Block(prev, ch, kernel, 2 ** i))
                prev = ch
            self.tcn = nn.Sequential(*blocks)
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                nn.LayerNorm(prev), nn.Linear(prev, n_classes)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.head(self.tcn(x.transpose(1, 2)))

    return _TCN()


def _build_transformer(
    input_dim: int, n_classes: int, d_model: int, nhead: int,
    num_layers: int, dim_ff: int, dropout: float, seq_len: int,
) -> nn.Module:
    class _Transformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(input_dim, d_model)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
                dropout=dropout, batch_first=True, norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
            self.norm = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, n_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.head(self.norm(self.encoder(self.proj(x))[:, -1]))

    return _Transformer()


def _build_physiological_model(
    input_dim: int, n_classes: int, repr_dim: int, hidden: int,
    num_layers: int, dropout: float,
) -> nn.Module:
    """
    Proposed physiological forecasting model:
    ForecastModel (GRU backbone) → MultiTaskPredictor (current-activity head).
    Self-contained so benchmark.py imports no sibling modules.
    """
    class _ForecastBackbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gru = nn.GRU(input_dim, hidden, num_layers,
                              batch_first=True,
                              dropout=dropout if num_layers > 1 else 0.0)
            self.proj = nn.Sequential(
                nn.Linear(hidden, repr_dim), nn.LayerNorm(repr_dim), nn.ReLU()
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out, _ = self.gru(x)
            return self.proj(out[:, -1])

    class _Head(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(repr_dim, hidden), nn.LayerNorm(hidden),
                nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, n_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    class _PhysModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = _ForecastBackbone()
            self.head = _Head()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.head(self.backbone(x))

    return _PhysModel()


# ---------------------------------------------------------------------------
# Standard benchmark suite
# ---------------------------------------------------------------------------

def run_benchmark_suite(
    input_dim: int = 16,
    seq_len: int = 100,
    n_classes: int = 6,
    batch_size: int = 64,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.2,
    n_warmup: int = 10,
    n_runs: int = 100,
    n_train_steps: int = 50,
    use_amp: bool = True,
    compile_models: bool = False,
    include_flops: bool = True,
    models_to_run: Optional[Sequence[str]] = None,
) -> List[BenchmarkResult]:
    """
    Run the full benchmark suite across all framework architectures.

    Parameters
    ----------
    input_dim : feature dimension per timestep
    seq_len : sequence length (time steps)
    n_classes : number of output classes
    batch_size : inference and training batch size
    hidden_size : RNN/Transformer hidden dimension
    num_layers : number of RNN/Transformer layers
    dropout : dropout rate
    n_warmup : warm-up passes before timing
    n_runs : timed inference passes
    n_train_steps : training steps for training-time measurement
    use_amp : enable AMP
    compile_models : apply torch.compile
    include_flops : compute FLOPs (adds one forward pass)
    models_to_run : subset of model names; None = all

    Returns
    -------
    list of BenchmarkResult, one per architecture
    """
    device = _get_device()
    dummy = torch.randn(batch_size, seq_len, input_dim)
    dummy_target = torch.randint(0, n_classes, (batch_size,))
    loss_fn = nn.CrossEntropyLoss()

    tcn_channels = (64, 128, 128)

    model_factories: Dict[str, Callable[[], nn.Module]] = {
        "CNN": lambda: _CNNWrapper(_build_cnn(input_dim, seq_len, n_classes, dropout)),
        "GRU": lambda: _build_gru(input_dim, n_classes, hidden_size, num_layers, dropout),
        "BiLSTM": lambda: _build_bilstm(input_dim, n_classes, hidden_size, num_layers, dropout),
        "TCN": lambda: _build_tcn(input_dim, n_classes, tcn_channels, kernel=3, dropout=dropout),
        "Transformer": lambda: _build_transformer(
            input_dim, n_classes, d_model=hidden_size, nhead=max(1, hidden_size // 64),
            num_layers=num_layers, dim_ff=hidden_size * 4, dropout=dropout, seq_len=seq_len,
        ),
        "PhysiologicalForecast": lambda: _build_physiological_model(
            input_dim, n_classes, repr_dim=hidden_size, hidden=hidden_size,
            num_layers=num_layers, dropout=dropout,
        ),
    }

    if models_to_run is not None:
        model_factories = {k: v for k, v in model_factories.items() if k in models_to_run}

    results: List[BenchmarkResult] = []

    for name, factory in model_factories.items():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model = factory()
        result = benchmark_training(
            model=model,
            dummy_input=dummy,
            dummy_target=dummy_target,
            loss_fn=loss_fn,
            n_steps=n_train_steps,
            use_amp=use_amp,
            model_name=name,
            n_warmup=n_warmup,
            n_runs=n_runs,
            compile_model=compile_models,
        )
        if not include_flops:
            result.flops = 0.0
        results.append(result)

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Export utilities
# ---------------------------------------------------------------------------

_TABLE_COLUMNS: List[Tuple[str, str, str]] = [
    ("model_name",                 "Model",              "s"),
    ("n_params",                   "Params",             ",d"),
    ("model_size_mb",              "Size (MB)",          ".2f"),
    ("train_time_s",               "Train (s)",          ".3f"),
    ("inference_latency_ms",       "Latency (ms)",       ".3f"),
    ("inference_latency_std_ms",   "Lat. σ (ms)",        ".3f"),
    ("throughput_samples_per_s",   "Throughput (s/s)",   ".1f"),
    ("gpu_util_pct",               "GPU (%)",            ".1f"),
    ("cpu_util_pct",               "CPU (%)",            ".1f"),
    ("vram_used_mb",               "VRAM (MB)",          ".2f"),
    ("ram_used_mb",                "RAM (MB)",           ".2f"),
    ("flops",                      "FLOPs",              ".2e"),
]


def results_to_dataframe(results: Sequence[BenchmarkResult]):
    """Convert benchmark results to a pandas DataFrame."""
    import pandas as pd
    return pd.DataFrame([r.to_dict() for r in results])


def results_to_csv(results: Sequence[BenchmarkResult], path: Path) -> None:
    """Write benchmark results to a CSV file."""
    path = Path(path)
    rows = [r.to_dict() for r in results]
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row["input_shape"] = str(row["input_shape"])
            writer.writerow(row)


def results_to_json(results: Sequence[BenchmarkResult], path: Path) -> None:
    """Write benchmark results to a JSON file."""
    Path(path).write_text(
        json.dumps([r.to_dict() for r in results], indent=2, default=str),
        encoding="utf-8",
    )


def comparison_table(
    results: Sequence[BenchmarkResult],
    columns: Optional[Sequence[str]] = None,
) -> str:
    """
    Generate a publication-ready Markdown comparison table.

    Parameters
    ----------
    results : benchmark results
    columns : subset of column keys from _TABLE_COLUMNS; None = all

    Returns
    -------
    Markdown string
    """
    col_defs = (
        [(k, h, f) for k, h, f in _TABLE_COLUMNS if k in columns]
        if columns is not None
        else _TABLE_COLUMNS
    )

    header = "| " + " | ".join(h for _, h, _ in col_defs) + " |"
    sep = "| " + " | ".join(":---" if i == 0 else "---:" for i, _ in enumerate(col_defs)) + " |"

    rows = [header, sep]
    for r in results:
        d = r.to_dict()
        cells = []
        for key, _, fmt in col_defs:
            val = d.get(key, "")
            try:
                cells.append(format(val, fmt))
            except (TypeError, ValueError):
                cells.append(str(val))
        rows.append("| " + " | ".join(cells) + " |")

    return "\n".join(rows)


def latex_table(results: Sequence[BenchmarkResult]) -> str:
    """
    Generate a LaTeX booktabs table for publication.

    Returns
    -------
    LaTeX string suitable for inclusion in a paper.
    """
    col_defs = _TABLE_COLUMNS
    col_spec = "l" + "r" * (len(col_defs) - 1)
    header = " & ".join(h for _, h, _ in col_defs) + r" \\"

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
        header,
        r"\midrule",
    ]
    for r in results:
        d = r.to_dict()
        cells = []
        for key, _, fmt in col_defs:
            val = d.get(key, "")
            try:
                cells.append(format(val, fmt))
            except (TypeError, ValueError):
                cells.append(str(val))
        lines.append(" & ".join(cells) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Model benchmark results.}",
        r"\label{tab:benchmark}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def summary_report(results: Sequence[BenchmarkResult]) -> str:
    """
    Generate a human-readable text summary report.

    Returns
    -------
    Multi-line string with rankings and highlights.
    """
    if not results:
        return "No benchmark results."

    lines = ["=" * 72, "BENCHMARK SUMMARY REPORT", "=" * 72, ""]

    best_lat = min(results, key=lambda r: r.inference_latency_ms)
    best_thr = max(results, key=lambda r: r.throughput_samples_per_s)
    smallest = min(results, key=lambda r: r.n_params)

    lines += [
        f"  Architectures benchmarked : {len(results)}",
        f"  Device                    : {results[0].device}",
        f"  AMP enabled               : {results[0].amp}",
        f"  Batch size                : {results[0].batch_size}",
        f"  Input shape               : {results[0].input_shape}",
        "",
        "Rankings:",
        f"  Fastest inference   : {best_lat.model_name} "
        f"({best_lat.inference_latency_ms:.3f} ms ± {best_lat.inference_latency_std_ms:.3f} ms)",
        f"  Highest throughput  : {best_thr.model_name} "
        f"({best_thr.throughput_samples_per_s:.1f} samples/s)",
        f"  Fewest parameters   : {smallest.model_name} "
        f"({smallest.n_params:,})",
        "",
        "Per-model summary:",
    ]

    for r in results:
        lines += [
            f"  [{r.model_name}]",
            f"    Params          : {r.n_params:,}",
            f"    Size            : {r.model_size_mb:.2f} MB",
            f"    Train time      : {r.train_time_s:.3f} s",
            f"    Inference       : {r.inference_latency_ms:.3f} ms ± "
            f"{r.inference_latency_std_ms:.3f} ms",
            f"    Throughput      : {r.throughput_samples_per_s:.1f} samples/s",
            f"    GPU util        : {r.gpu_util_pct:.1f} %",
            f"    CPU util        : {r.cpu_util_pct:.1f} %",
            f"    VRAM            : {r.vram_used_mb:.2f} MB",
            f"    RAM             : {r.ram_used_mb:.2f} MB",
            f"    FLOPs           : {r.flops:.3e}",
            "",
        ]

    lines.append("=" * 72)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convenience: run + export in one call
# ---------------------------------------------------------------------------

def run_and_export(
    output_dir: Path,
    input_dim: int = 16,
    seq_len: int = 100,
    n_classes: int = 6,
    batch_size: int = 64,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.2,
    n_warmup: int = 10,
    n_runs: int = 100,
    n_train_steps: int = 50,
    use_amp: bool = True,
    compile_models: bool = False,
    models_to_run: Optional[Sequence[str]] = None,
) -> List[BenchmarkResult]:
    """
    Run the full benchmark suite and write all outputs to ``output_dir``.

    Writes
    ------
    benchmark_results.csv
    benchmark_results.json
    benchmark_table.md
    benchmark_table.tex
    benchmark_report.txt
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = run_benchmark_suite(
        input_dim=input_dim,
        seq_len=seq_len,
        n_classes=n_classes,
        batch_size=batch_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        n_warmup=n_warmup,
        n_runs=n_runs,
        n_train_steps=n_train_steps,
        use_amp=use_amp,
        compile_models=compile_models,
        models_to_run=models_to_run,
    )

    results_to_csv(results, output_dir / "benchmark_results.csv")
    results_to_json(results, output_dir / "benchmark_results.json")
    (output_dir / "benchmark_table.md").write_text(comparison_table(results), encoding="utf-8")
    (output_dir / "benchmark_table.tex").write_text(latex_table(results), encoding="utf-8")
    (output_dir / "benchmark_report.txt").write_text(summary_report(results), encoding="utf-8")

    return results
