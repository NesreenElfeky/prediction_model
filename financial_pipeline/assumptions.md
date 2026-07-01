# Financial Loss Assumptions

The loss engine uses constants from `configs/financial.yaml` and turns each asset row into five component estimates:

- data breach response and records at risk
- ransomware payment and recovery exposure
- downtime cost by business criticality
- legal, regulatory, and notification costs
- reputation-driven customer churn

`expected_annual_loss_usd = exploit_probability * financial_impact_usd`.

These values are designed for modeling and prioritization, not accounting-grade forecasts. Replace the defaults with organization-specific revenue, customer count, downtime, and records-at-risk values when available.

