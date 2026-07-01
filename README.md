# Cyber Financial Loss Prediction

Production-style pipeline for turning vulnerability intelligence into asset-level financial-loss predictions.

## Data

The raw input is expected at:

```text
datasets/raw/threat_intelligence.csv
```

The current dataset columns include asset metadata, CVE details, CVSS/EPSS scores, exploit flags, threat scores, and financial-risk starter fields.

## Run

```powershell
.\.venv\Scripts\python.exe main.py
```

`main.py` runs the enterprise ML lifecycle on `datasets/final/ml_dataset.parquet`:

1. data loading and schema validation
2. data profiling, missing-value, outlier, correlation, and target analysis
3. cybersecurity feature engineering
4. feature selection with variance filtering, correlation filtering, mutual information, random forest importance, SHAP importance, and RFE
5. sklearn preprocessing pipelines
6. model comparison across Linear Regression, ElasticNet, Random Forest, Extra Trees, Gradient Boosting, XGBoost, LightGBM, CatBoost, and HistGradientBoosting
7. KFold, RepeatedKFold, RandomizedSearchCV, and Optuna tuning
8. evaluation, uncertainty intervals, explainability plots, and reports

Dataset-generation outputs:

- `datasets/processed/ti_cleaned.parquet`
- `datasets/financial/loss_estimates.parquet`
- `datasets/final/ml_dataset.parquet`

ML outputs:

- `artifacts/ml/trained_model.pkl`
- `artifacts/ml/preprocessor.pkl`
- `artifacts/ml/feature_list.json`
- `artifacts/ml/metrics.json`
- `artifacts/ml/training_config.json`
- `artifacts/ml/best_model_report.md`
- `artifacts/ml/leaderboard.csv`
- `reports/ml/model_report.html`
- `reports/ml/model_report.pdf`
- `reports/ml/plots/`

## Structure

- `threat_pipeline/`: validation, cleaning, imputation, feature engineering, normalization, aggregation
- `asset_pipeline/`: asset inventory loading and enrichment
- `pentest_pipeline/`: scanner parsers, feature builder, risk scoring
- `financial_pipeline/`: loss components and expected annual loss
- `integration/`: asset-keyed merge and integrity checks
- `preprocessing/`: encoding, scaling, selection, train/validation/test split
- `models/`, `training/`, `evaluation/`, `explainability/`: ML workflow helpers
- `ml_pipeline/`: production ML orchestration, validation, feature engineering, selection, training, evaluation, explainability, and reporting
