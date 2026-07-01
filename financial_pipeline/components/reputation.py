from __future__ import annotations

import pandas as pd


def estimate_reputation_loss(row: pd.Series, cfg: dict) -> float:
    constants = cfg.get("reputation", {})
    churn = float(constants.get("avg_customer_churn_rate_post_breach", 0.038))
    ltv = float(constants.get("avg_customer_ltv_usd", 4200))
    customers = float(row.get("estimated_customer_count", row.get("vuln_count", 1) * 100) or 0)
    return customers * churn * ltv

