from __future__ import annotations

from pathlib import Path

import pandas as pd

from financial_pipeline.components.data_breach import estimate_data_breach_loss
from financial_pipeline.components.downtime import estimate_downtime_loss
from financial_pipeline.components.legal import estimate_legal_loss
from financial_pipeline.components.ransomware import estimate_ransomware_loss
from financial_pipeline.components.reputation import estimate_reputation_loss
from financial_pipeline.eal_calculator import expected_annual_loss
from financial_pipeline.probability_engine import exploit_probability
from financial_pipeline.report_constants import load_financial_constants
from utils.io import write_table


def estimate_losses(df: pd.DataFrame, financial_config: str | Path = "configs/financial.yaml") -> pd.DataFrame:
    cfg = load_financial_constants(financial_config)
    out = df.copy()
    out["loss_data_breach_usd"] = out.apply(lambda row: estimate_data_breach_loss(row, cfg), axis=1)
    out["loss_ransomware_usd"] = out.apply(lambda row: estimate_ransomware_loss(row, cfg), axis=1)
    out["loss_downtime_usd"] = out.apply(lambda row: estimate_downtime_loss(row, cfg), axis=1)
    out["loss_legal_usd"] = out.apply(lambda row: estimate_legal_loss(row, cfg), axis=1)
    out["loss_reputation_usd"] = out.apply(lambda row: estimate_reputation_loss(row, cfg), axis=1)
    components = [
        "loss_data_breach_usd",
        "loss_ransomware_usd",
        "loss_downtime_usd",
        "loss_legal_usd",
        "loss_reputation_usd",
    ]
    out["financial_impact_usd"] = out[components].sum(axis=1)
    out["exploit_probability"] = out.apply(exploit_probability, axis=1)
    out["expected_annual_loss_usd"] = out.apply(
        lambda row: expected_annual_loss(row["exploit_probability"], row["financial_impact_usd"]),
        axis=1,
    )
    return out


def run_financial_pipeline(input_df: pd.DataFrame, output_path: str | Path | None = None) -> pd.DataFrame:
    out = estimate_losses(input_df)
    if output_path:
        write_table(out, output_path)
    return out

