# Pipeline

1. `threat_pipeline` ingests `threat_intelligence.csv`, validates fields, cleans values, engineers threat features, normalizes scores, and aggregates to one row per `asset_id`.
2. `asset_pipeline` can build or enrich asset inventory rows.
3. `pentest_pipeline` parses external scanner outputs and computes `pentest_risk_score`.
4. `financial_pipeline` estimates component impact and expected annual loss.
5. `integration` merges all asset-keyed frames into `datasets/final/ml_dataset.parquet`.
6. `preprocessing`, `training`, `evaluation`, and `explainability` support model development.
