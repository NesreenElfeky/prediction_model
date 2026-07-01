"""
threat_pipeline/feature_engineer.py
Derived risk scores, date-delta features, and binary flags from cleaned TI data.
"""

from __future__ import annotations

import logging
from datetime import timezone

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Severity → numeric weight
SEVERITY_WEIGHT = {"critical": 4, "high": 3, "medium": 2, "low": 1, "info": 0}

# Attack vector → exposure weight
VECTOR_WEIGHT = {"network": 4, "adjacent": 3, "local": 2, "physical": 1}


class FeatureEngineer:
    """
    Adds derived columns to the cleaned TI DataFrame:
      - composite_risk_score
      - days_since_published / days_since_first_seen / days_active
      - is_exploitable / is_critical / is_network_exposed flags
      - epss_cvss_interaction
    """

    def __init__(self, reference_date: pd.Timestamp | None = None) -> None:
        self.reference_date = reference_date or pd.Timestamp.now(tz=timezone.utc)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Engineering features on %d rows.", len(df))
        df = df.copy()
        df = self._risk_score(df)
        df = self._date_deltas(df)
        df = self._binary_flags(df)
        df = self._interaction_terms(df)
        logger.info("Feature engineering added %d new columns.", len(df.columns))
        return df

    # ------------------------------------------------------------------
    # Feature groups
    # ------------------------------------------------------------------

    def _risk_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        composite_risk_score = (cvss_norm × 0.4) + (epss × 0.3)
                               + (severity_weight_norm × 0.2)
                               + (vector_weight_norm × 0.1)
        Scaled 0–10.
        """
        cvss_norm = df.get("cvss_score", pd.Series(0, index=df.index)) / 10.0
        epss = df.get("epss_score", pd.Series(0, index=df.index)).fillna(0)
        sev_w = df.get("severity", pd.Series("low", index=df.index)).map(SEVERITY_WEIGHT).fillna(0) / 4.0
        vec_w = df.get("attack_vector", pd.Series("local", index=df.index)).map(VECTOR_WEIGHT).fillna(0) / 4.0
        kev_bonus = df.get("kev_listed", pd.Series(False, index=df.index)).astype(float) * 0.5

        score = (cvss_norm * 0.4 + epss * 0.3 + sev_w * 0.2 + vec_w * 0.1) * 10 + kev_bonus
        df["composite_risk_score"] = score.clip(0, 10).round(4)
        logger.debug("composite_risk_score: min=%.2f max=%.2f mean=%.2f",
                     df["composite_risk_score"].min(),
                     df["composite_risk_score"].max(),
                     df["composite_risk_score"].mean())
        return df

    def _date_deltas(self, df: pd.DataFrame) -> pd.DataFrame:
        ref = self.reference_date

        for col, new_col in [
            ("published_date", "days_since_published"),
            ("first_seen", "days_since_first_seen"),
        ]:
            if col in df.columns and new_col not in df.columns:
                parsed = pd.to_datetime(df[col], errors="coerce", utc=True)
                delta = (ref - parsed).dt.days
                df[new_col] = delta.clip(lower=0)

        if "first_seen" in df.columns and "last_seen" in df.columns and "days_active" not in df.columns:
            first = pd.to_datetime(df["first_seen"], errors="coerce", utc=True)
            last = pd.to_datetime(df["last_seen"], errors="coerce", utc=True)
            df["days_active"] = (last - first).dt.days.clip(lower=0)

        return df

    def _binary_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        if "epss_score" in df.columns:
            df["is_exploitable"] = (df["epss_score"] >= 0.1).astype(int)

        if "severity" in df.columns:
            df["is_critical"] = (df["severity"] == "critical").astype(int)

        if "attack_vector" in df.columns:
            df["is_network_exposed"] = (df["attack_vector"] == "network").astype(int)

        if "kev_listed" in df.columns:
            df["kev_flag"] = df["kev_listed"].astype(int)

        return df

    def _interaction_terms(self, df: pd.DataFrame) -> pd.DataFrame:
        if "epss_score" in df.columns and "cvss_score" in df.columns:
            df["epss_cvss_interaction"] = (df["epss_score"] * df["cvss_score"]).round(4)

        if "composite_risk_score" in df.columns and "exploit_count" in df.columns:
            df["risk_exploit_product"] = (
                df["composite_risk_score"] * np.log1p(df["exploit_count"])
            ).round(4)

        return df
