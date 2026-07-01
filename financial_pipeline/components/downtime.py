from __future__ import annotations

import pandas as pd


def estimate_downtime_loss(row: pd.Series, cfg: dict) -> float:
    constants = cfg.get("downtime", {})
    criticality = str(row.get("business_criticality", row.get("business_criticality_mode", "medium"))).lower()
    per_min = float(constants.get(f"{criticality}_systems_cost_per_min_usd", constants.get("medium_systems_cost_per_min_usd", 840)))
    downtime_hours = float(row.get("estimated_downtime_hours", 8 + row.get("sev_count_CRITICAL", 0) * 12) or 0)
    return per_min * downtime_hours * 60

