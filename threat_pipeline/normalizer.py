"""
threat_pipeline/normalizer.py
Per-column MinMax or Standard scaling for TI numeric features.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Literal

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logger = logging.getLogger(__name__)

ScalerType = Literal["minmax", "standard"]

# Columns that use MinMax (bounded, e.g. scores 0-10, 0-1)
MINMAX_COLS = [
    "cvss_score",
    "epss_score",
    "composite_risk_score",
    "epss_cvss_interaction",
]

# Columns that use Standard scaling (unbounded counts, deltas)
STANDARD_COLS = [
    "exploit_count",
    "affected_systems_count",
    "days_since_published",
    "days_since_first_seen",
    "days_active",
    "risk_exploit_product",
]


class ThreatNormalizer:
    """Fits and applies MinMax / Standard scalers per column group."""

    def __init__(
        self,
        minmax_cols: list[str] = MINMAX_COLS,
        standard_cols: list[str] = STANDARD_COLS,
    ) -> None:
        self.minmax_cols = minmax_cols
        self.standard_cols = standard_cols
        self._scalers: dict[str, MinMaxScaler | StandardScaler] = {}
        self._fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> ThreatNormalizer:
        for col in self._available(self.minmax_cols, df):
            scaler = MinMaxScaler()
            scaler.fit(df[[col]].dropna())
            self._scalers[col] = scaler
            logger.debug("Fitted MinMaxScaler for '%s'.", col)

        for col in self._available(self.standard_cols, df):
            scaler = StandardScaler()
            scaler.fit(df[[col]].dropna())
            self._scalers[col] = scaler
            logger.debug("Fitted StandardScaler for '%s'.", col)

        self._fitted = True
        logger.info("Normalizer fitted on %d columns.", len(self._scalers))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        df = df.copy()
        for col, scaler in self._scalers.items():
            if col in df.columns:
                non_null_mask = df[col].notna()
                df[col] = df[col].astype(float)
                df.loc[non_null_mask, col] = scaler.transform(
                    df.loc[non_null_mask, [col]]
                ).ravel()
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._scalers, f)
        logger.info("Scalers saved to %s", path)

    def load(self, path: str | Path) -> ThreatNormalizer:
        with open(path, "rb") as f:
            self._scalers = pickle.load(f)
        self._fitted = True
        logger.info("Scalers loaded from %s", path)
        return self

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _available(cols: list[str], df: pd.DataFrame) -> list[str]:
        return [c for c in cols if c in df.columns]
