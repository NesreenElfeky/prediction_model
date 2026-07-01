"""
threat_pipeline/cleaner.py
Deduplication, outlier flagging, and type coercion for TI records.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# Columns to use for deduplication
DEDUP_KEYS = ["asset_id", "cve_id", "source_feed"]

# Numeric columns susceptible to outliers
NUMERIC_COLS = ["cvss_score", "epss_score", "exploit_count", "affected_systems_count"]


class ThreatCleaner:
    """
    Cleans raw TI DataFrame:
      1. Type coercion (strings → correct dtypes)
      2. Deduplication
      3. Outlier flagging via IQR / Z-score
    """

    def __init__(
        self,
        dedup_keys: list[str] = DEDUP_KEYS,
        outlier_method: str = "iqr",   # "iqr" or "zscore"
        zscore_threshold: float = 3.5,
        iqr_multiplier: float = 1.5,
    ) -> None:
        self.dedup_keys = dedup_keys
        self.outlier_method = outlier_method
        self.zscore_threshold = zscore_threshold
        self.iqr_multiplier = iqr_multiplier

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Starting cleaning on %d rows.", len(df))
        df = self._coerce_types(df)
        df = self._deduplicate(df)
        df = self._flag_outliers(df)
        logger.info("Cleaning complete. %d rows remain.", len(df))
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _coerce_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Force expected dtypes; log columns that cannot be coerced."""
        df = df.copy()

        # Boolean
        for col in ["kev_listed"]:
            if col in df.columns:
                df[col] = df[col].map(
                    lambda x: True if str(x).strip().lower() in {"true", "1", "yes"} else False
                )

        # Float
        for col in ["cvss_score", "epss_score"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Integer
        for col in ["exploit_count", "affected_systems_count"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

        # Date strings are left intact. The project CSV already contains
        # numeric date-delta columns such as days_since_published.

        # String normalisation
        for col in ["severity", "attack_vector", "attack_complexity"]:
            if col in df.columns:
                df[col] = df[col].str.strip().str.lower()

        return df

    def _deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep the latest record per (asset_id, cve_id, source_feed)."""
        available_keys = [k for k in self.dedup_keys if k in df.columns]
        before = len(df)

        if "last_seen" in df.columns:
            df = df.sort_values("last_seen", ascending=False)

        df = df.drop_duplicates(subset=available_keys, keep="first")
        logger.info("Deduplication removed %d duplicate rows.", before - len(df))
        return df.reset_index(drop=True)

    def _flag_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add boolean 'is_outlier' column; does NOT remove rows."""
        df = df.copy()
        df["is_outlier"] = False

        for col in NUMERIC_COLS:
            if col not in df.columns:
                continue
            series = df[col].dropna()

            if self.outlier_method == "zscore":
                z = np.abs(stats.zscore(series))
                mask = z > self.zscore_threshold
            else:  # IQR
                q1, q3 = series.quantile(0.25), series.quantile(0.75)
                iqr = q3 - q1
                lower, upper = q1 - self.iqr_multiplier * iqr, q3 + self.iqr_multiplier * iqr
                mask = (series < lower) | (series > upper)

            outlier_idx = series[mask].index
            df.loc[outlier_idx, "is_outlier"] = True
            logger.debug("Column '%s': %d outliers flagged.", col, mask.sum())

        total_outliers = df["is_outlier"].sum()
        logger.info("Total outlier rows flagged: %d", total_outliers)
        return df
