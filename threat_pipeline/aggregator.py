"""
threat_pipeline/aggregator.py
Collapse multiple vulnerability records → one row per asset_id.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ThreatAggregator:
    """
    Aggregates the cleaned, feature-engineered TI DataFrame from
    one-row-per-vulnerability to one-row-per-asset.

    Aggregation strategy:
      - Risk/score columns   → max, mean
      - Count columns        → sum, max
      - Boolean flags        → max (any-true)
      - Days delta columns   → min (most recent), max (oldest)
      - Categorical counts   → number of distinct values
    """

    # Columns and their aggregation functions
    AGG_MAP: dict[str, list[str]] = {
        "cvss_score":            ["max", "mean"],
        "epss_score":            ["max", "mean"],
        "composite_risk_score":  ["max", "mean"],
        "epss_cvss_interaction": ["max", "mean"],
        "risk_exploit_product":  ["max", "mean"],
        "exploit_count":         ["sum", "max"],
        "affected_systems_count":["sum", "max"],
        "is_exploitable":        ["max", "sum"],
        "is_critical":           ["max", "sum"],
        "is_network_exposed":    ["max"],
        "kev_flag":              ["max", "sum"],
        "days_since_published":  ["min", "max"],
        "days_since_first_seen": ["min"],
        "days_active":           ["max", "mean"],
    }

    def __init__(self, group_key: str = "asset_id") -> None:
        self.group_key = group_key

    def aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.group_key not in df.columns:
            raise ValueError(f"group_key '{self.group_key}' not found in DataFrame.")

        logger.info(
            "Aggregating %d rows → per-asset rows (group_key='%s').",
            len(df), self.group_key,
        )

        available_agg = {
            col: funcs
            for col, funcs in self.AGG_MAP.items()
            if col in df.columns
        }

        agg_df = df.groupby(self.group_key).agg(available_agg)

        # Flatten multi-level column names: ("cvss_score", "max") → "cvss_score_max"
        agg_df.columns = ["_".join(col).strip() for col in agg_df.columns]

        # Vulnerability count per asset
        agg_df["vuln_count"] = df.groupby(self.group_key).size()

        # Unique CVE count
        if "cve_id" in df.columns:
            agg_df["unique_cve_count"] = df.groupby(self.group_key)["cve_id"].nunique()

        # Unique threat actors
        if "threat_actor" in df.columns:
            agg_df["unique_threat_actor_count"] = df.groupby(self.group_key)["threat_actor"].nunique()

        # Severity distribution
        if "severity" in df.columns:
            sev_counts = (
                df.groupby([self.group_key, "severity"])
                .size()
                .unstack(fill_value=0)
                .add_prefix("sev_count_")
            )
            agg_df = agg_df.join(sev_counts, how="left")

        agg_df = agg_df.reset_index()
        logger.info("Aggregation produced %d asset rows with %d columns.", len(agg_df), len(agg_df.columns))
        return agg_df
