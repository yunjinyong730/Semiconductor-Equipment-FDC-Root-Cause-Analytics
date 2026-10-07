from __future__ import annotations

import argparse
import json

from src.fdc_analytics.config import PipelineConfig
from src.fdc_analytics.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SECOM FDC, drift, yield-risk and RCA analytics.")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--drift-window", type=int, default=30)
    parser.add_argument("--fdc-quantile", type=float, default=0.99)
    args = parser.parse_args()
    config = PipelineConfig(drift_window=args.drift_window, fdc_quantile=args.fdc_quantile)
    metrics = run_pipeline(args.data_dir, args.output_dir, config)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
