from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def generate_demo_dataset(out_dir: Path, n_subjects: int = 6, n_samples: int = 4000, n_channels: int = 10, n_classes: int = 7, random_state: int = 42):
    rng = np.random.default_rng(random_state)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_t = np.linspace(0, 2 * np.pi, n_samples)
    for sid in range(1, n_subjects + 1):
        phase = (np.floor((np.linspace(0, n_classes, n_samples, endpoint=False)) % n_classes) + 1).astype(int)
        X = []
        for ch in range(n_channels):
            freq = rng.uniform(0.5, 4.0)
            amp = rng.uniform(0.4, 1.5)
            shift = rng.uniform(0, np.pi)
            signal = amp * (np.sin(freq * base_t + shift) ** 2)
            modulation = 0.15 * phase + 0.1 * np.sin(base_t * (ch + 1))
            noise = rng.normal(0, 0.1, n_samples)
            X.append(signal + modulation + noise)
        X = np.vstack(X).T
        cols = {f"ch_{i+1}": X[:, i] for i in range(n_channels)}
        df = pd.DataFrame(cols)
        df.insert(0, "phase", phase)
        df.to_csv(out_dir / f"subject_{sid}.csv", index=False)
