from __future__ import annotations

import pandas as pd


def estimate_ransomware_loss(row: pd.Series, cfg: dict) -> float:
    constants = cfg.get("ransomware", {})
    payment = float(constants.get("avg_ransom_payment_usd", 2_540_000))
    recovery = float(constants.get("avg_recovery_cost_usd", 1_850_000))
    rate = float(constants.get("payment_rate", 0.46))
    ransomware_flag = float(row.get("known_ransomware_max", row.get("kev_flag_max", 0)) or 0)
    exploit_pressure = float(row.get("epss_score_max", row.get("exploit_risk_score_max", 0)) or 0)
    return (payment * rate + recovery) * max(ransomware_flag, exploit_pressure, 0.15)

