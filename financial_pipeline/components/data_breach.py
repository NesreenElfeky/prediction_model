from __future__ import annotations

import pandas as pd


def estimate_data_breach_loss(row: pd.Series, cfg: dict) -> float:
    constants = cfg.get("data_breach", {})
    baseline = float(constants.get("baseline_cost_usd", 4_880_000))
    per_record = float(constants.get("cost_per_record_usd", 165))
    records = float(row.get("estimated_records_at_risk", row.get("vuln_count", 1) * 1000) or 0)
    multiplier = 1.0 + min(float(row.get("vulnerability_impact_score_max", row.get("cvss_score_max", 5))) / 20, 1.0)
    return max(baseline * 0.15, records * per_record) * multiplier

