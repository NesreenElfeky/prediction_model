from __future__ import annotations

import numpy as np
import pandas as pd


def exploit_probability(row: pd.Series) -> float:
    epss = float(row.get("epss_score_max", row.get("epss_score_mean", 0.01)) or 0.01)
    cvss = float(row.get("cvss_score_max", row.get("cvss_score_mean", 5.0)) or 5.0) / 10
    kev = float(row.get("kev_flag_max", 0) or 0)
    public_exploit = float(row.get("has_public_exploit_max", row.get("is_exploitable_max", 0)) or 0)
    pressure = float(row.get("threat_pressure_factor_max", row.get("pentest_risk_score", 0) / 100) or 0)
    p = epss * (1 + kev * 1.5 + public_exploit * 0.75) + cvss * 0.08 + pressure * 0.05
    return float(np.clip(p, 0.001, 0.95))

