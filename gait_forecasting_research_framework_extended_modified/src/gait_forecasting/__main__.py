
from __future__ import annotations

import argparse
from pathlib import Path

from .config import PipelineConfig
from .pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Research-grade gait forecasting framework")
    parser.add_argument("--data_dir", type=Path, required=True, help="Directory containing CSV/NPZ files")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory for outputs")
    parser.add_argument("--demo", action="store_true", help="Generate synthetic demo data in data_dir before running")
    parser.add_argument("--no_synergies", action="store_true", help="Disable NMF synergy features")
    parser.add_argument("--no_dh", action="store_true", help="Disable dH features")
    parser.add_argument("--no_d2h", action="store_true", help="Disable d2H features")
    parser.add_argument("--cv", type=str, default="loso", choices=["loso", "groupkfold", "holdout"], help="Cross-subject evaluation mode")
    args = parser.parse_args()

    cfg = PipelineConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        demo=args.demo,
        use_synergies=not args.no_synergies,
        use_dh=not args.no_dh,
        use_d2h=not args.no_d2h,
    )
    cfg.eval.cross_validation = args.cv
    summary = run_pipeline(cfg)
    print(summary)


if __name__ == "__main__":
    main()
