"""
threat_pipeline/pipeline.py
Orchestrates all TI processing steps: validate → clean → impute → engineer → normalize → aggregate.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from omegaconf import DictConfig

from threat_pipeline.aggregator import ThreatAggregator
from threat_pipeline.cleaner import ThreatCleaner
from threat_pipeline.feature_engineer import FeatureEngineer
from threat_pipeline.imputer import ThreatImputer
from threat_pipeline.normalizer import ThreatNormalizer
from threat_pipeline.validator import ThreatValidator

logger = logging.getLogger(__name__)


class ThreatPipeline:
    """
    End-to-end TI pipeline.

    Usage:
        pipeline = ThreatPipeline(cfg)
        asset_df = pipeline.run(raw_csv_path)
    """

    def __init__(self, cfg: Optional[DictConfig] = None) -> None:
        self.cfg = cfg or {}
        self._validator = ThreatValidator(strict=False)
        self._cleaner = ThreatCleaner()
        self._imputer = ThreatImputer(knn_neighbors=5)
        self._feature_engineer = FeatureEngineer()
        self._normalizer = ThreatNormalizer()
        self._aggregator = ThreatAggregator(group_key="asset_id")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, input_path: str | Path) -> pd.DataFrame:
        """
        Full pipeline run from raw CSV to per-asset ML-ready DataFrame.

        Args:
            input_path: Path to raw threat_intelligence.csv

        Returns:
            Per-asset aggregated DataFrame ready for integration layer.
        """
        logger.info("=== ThreatPipeline START ===")

        # 1. Load
        df = self._load(input_path)

        # 2. Validate
        logger.info("Step 1/6 — Validation")
        df = self._validator.validate(df)
        if self._validator.errors:
            logger.warning("%d rows dropped during validation.", len(self._validator.errors))

        # 3. Clean
        logger.info("Step 2/6 — Cleaning")
        df = self._cleaner.clean(df)

        # 4. Impute
        logger.info("Step 3/6 — Imputation")
        df = self._imputer.fit_transform(df)

        # 5. Feature engineering
        logger.info("Step 4/6 — Feature Engineering")
        df = self._feature_engineer.transform(df)

        # 6. Normalize
        logger.info("Step 5/6 — Normalization")
        df = self._normalizer.fit_transform(df)

        # 7. Aggregate to per-asset
        logger.info("Step 6/6 — Aggregation")
        asset_df = self._aggregator.aggregate(df)

        logger.info("=== ThreatPipeline COMPLETE — %d asset rows ===", len(asset_df))
        return asset_df

    def save(self, df: pd.DataFrame, output_path: str | Path) -> None:
        """Persist the processed dataset to parquet."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False, engine="pyarrow")
        logger.info("Saved processed TI data → %s (%d rows)", output_path, len(df))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load(path: str | Path) -> pd.DataFrame:
        path = Path(path)
        logger.info("Loading raw TI data from %s", path)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        df = pd.read_csv(path, low_memory=False)
        logger.info("Loaded %d rows × %d columns.", *df.shape)
        return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv")
    parser.add_argument("output_parquet")

    args = parser.parse_args()

    pipeline = ThreatPipeline()

    asset_df = pipeline.run(args.input_csv)

    pipeline.save(asset_df, args.output_parquet)