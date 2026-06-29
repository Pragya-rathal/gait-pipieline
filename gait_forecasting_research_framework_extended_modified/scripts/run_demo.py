
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gait_forecasting.config import PipelineConfig
from gait_forecasting.pipeline import run_pipeline

if __name__ == "__main__":
    cfg = PipelineConfig(
        data_dir=Path("./demo_data"),
        output_dir=Path("./demo_outputs"),
        demo=True,
    )
    cfg.eval.cross_validation = "holdout"
    run_pipeline(cfg)
    print("Done.")
