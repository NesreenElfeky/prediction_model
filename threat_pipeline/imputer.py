"""
threat_pipeline/imputer.py
Missing-value imputation: KNN for numeric, median fallback, domain-rule overrides.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer

logger = logging.getLogger(__name__)

# Columns imputed by KNN (require numeric context)
KNN_COLS = ["cvss_score", "epss_score"]

# Columns imputed by median
MEDIAN_COLS = ["exploit_count", "affected_systems_count"]

# Domain-rule defaults applied last
DOMAIN_DEFAULTS: dict[str, object] = {
    "kev_listed": False,
    "exploit_count": 0,
    "affected_systems_count": 1,
    "severity": "low",
    "attack_complexity": "high",
    "privileges_required": "high",
    "user_interaction": "required",
}


class ThreatImputer:
    """
    Multi-strategy imputer for TI records:
      1. KNN imputation for cvss_score, epss_score (using numeric neighbors)
      2. Median imputation for count columns
      3. Domain-rule fill-in for remaining nulls
    """

    def __init__(self, knn_neighbors: int = 5) -> None:
        self.knn_neighbors = knn_neighbors
        self._knn_imputer: Optional[KNNImputer] = None
        self._median_imputer: Optional[SimpleImputer] = None
        self._fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> ThreatImputer:
        numeric_df = df[self._available(KNN_COLS, df)].select_dtypes(include=np.number)
        if not numeric_df.empty:
            self._knn_imputer = KNNImputer(n_neighbors=self.knn_neighbors)
            self._knn_imputer.fit(numeric_df)

        median_cols = self._available(MEDIAN_COLS, df)
        if median_cols:
            self._median_imputer = SimpleImputer(strategy="median")
            self._median_imputer.fit(df[median_cols])

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")

        df = df.copy()
        df = self._knn_impute(df)
        df = self._median_impute(df)
        df = self._domain_rule_impute(df)
        self._report_nulls(df)
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _available(cols: list[str], df: pd.DataFrame) -> list[str]:
        return [c for c in cols if c in df.columns]

    def _knn_impute(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = self._available(KNN_COLS, df)
        if self._knn_imputer is None or not cols:
            return df
        before = df[cols].isna().sum().sum()
        df[cols] = self._knn_imputer.transform(df[cols])
        after = df[cols].isna().sum().sum()
        logger.info("KNN imputation filled %d values across %s.", before - after, cols)
        return df

    def _median_impute(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = self._available(MEDIAN_COLS, df)
        if self._median_imputer is None or not cols:
            return df
        before = df[cols].isna().sum().sum()
        df[cols] = self._median_imputer.transform(df[cols])
        after = df[cols].isna().sum().sum()
        logger.info("Median imputation filled %d values across %s.", before - after, cols)
        return df

    def _domain_rule_impute(self, df: pd.DataFrame) -> pd.DataFrame:
        for col, default in DOMAIN_DEFAULTS.items():
            if col in df.columns:
                before = df[col].isna().sum()
                df[col] = df[col].fillna(default)
                if before:
                    logger.debug("Domain rule: filled %d nulls in '%s' with %r.", before, col, default)
        return df

    @staticmethod
    def _report_nulls(df: pd.DataFrame) -> None:
        remaining = df.isna().sum()
        remaining = remaining[remaining > 0]
        if not remaining.empty:
            logger.warning("Remaining nulls after imputation:\n%s", remaining.to_string())
        else:
            logger.info("No remaining nulls after imputation.")
