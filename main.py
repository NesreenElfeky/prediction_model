from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ml_pipeline.orchestrator import run_ml_pipeline
from utils.logging import configure_logging


log = logging.getLogger(__name__)


def run_project(config_path: str | Path = "configs/ml.yaml") -> None:
    results = run_ml_pipeline(str(config_path))
    log.info("Best model: %s", results["best_model_name"])
    log.info("Metrics: %s", results["metrics"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the cyber financial loss prediction pipeline.")
    parser.add_argument("--config", default="configs/ml.yaml")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    configure_logging(args.log_level)
    run_project(args.config)


if __name__ == "__main__":
    main()
