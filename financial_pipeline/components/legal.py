from __future__ import annotations

import pandas as pd


def estimate_legal_loss(row: pd.Series, cfg: dict) -> float:
    constants = cfg.get("legal", {})
    litigation = float(constants.get("avg_litigation_cost_usd", 1_200_000))
    fine = float(constants.get("avg_regulatory_fine_usd", 740_000))
    records = float(row.get("estimated_records_at_risk", row.get("vuln_count", 1) * 1000) or 0)
    notification = float(constants.get("notification_cost_per_record_usd", 3.5)) * records
    return (litigation + fine) * min(records / 100_000, 1.0) + notification

