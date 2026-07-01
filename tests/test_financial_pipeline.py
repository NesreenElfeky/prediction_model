from financial_pipeline.loss_engine import estimate_losses
import pandas as pd


def test_estimate_losses_adds_target_column():
    df = pd.DataFrame({"asset_id": ["a1"], "cvss_score_max": [9.8], "epss_score_max": [0.2], "vuln_count": [3]})
    out = estimate_losses(df)
    assert "expected_annual_loss_usd" in out.columns
    assert out["expected_annual_loss_usd"].iloc[0] > 0

