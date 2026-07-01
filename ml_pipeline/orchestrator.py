from __future__ import annotations

import json
import logging

import joblib
import pandas as pd

from ml_pipeline.config import MLConfig, load_ml_config
from ml_pipeline.data_validation.loader import load_ml_dataset
from ml_pipeline.data_validation.profiler import profile_dataframe
from ml_pipeline.data_validation.quality import feature_quality_report
from ml_pipeline.data_validation.schema import validate_schema
from ml_pipeline.evaluation.plots import generate_evaluation_plots
from ml_pipeline.explainability.explainer import generate_shap_artifacts
from ml_pipeline.feature_engineering.builder import build_cyber_features
from ml_pipeline.feature_engineering.selector import remove_leakage, select_features
from ml_pipeline.reports.generator import write_reports
from ml_pipeline.training.trainer import train_and_select_model


log = logging.getLogger(__name__)


def run_ml_pipeline(config_path: str = "configs/ml.yaml") -> dict:
    cfg: MLConfig = load_ml_config(config_path)
    cfg.artifact_dir.mkdir(parents=True, exist_ok=True)
    cfg.report_dir.mkdir(parents=True, exist_ok=True)
    cfg.plot_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading final integrated ML dataset: %s", cfg.input_path)
    df = load_ml_dataset(cfg.input_path, cfg.target_column, cfg.source_target_column)
    schema_report = validate_schema(df, cfg.target_column, cfg.id_column)
    profile = profile_dataframe(df, cfg.target_column)
    quality = feature_quality_report(df, cfg.target_column, cfg.id_column)

    log.info("Engineering cybersecurity features.")
    engineered = build_cyber_features(df)
    y = pd.to_numeric(engineered[cfg.target_column], errors="raise")
    X_all = remove_leakage(engineered, cfg.leakage_columns, cfg.id_column)

    log.info("Selecting features with variance, correlation, MI, RF, SHAP, and RFE.")
    X_selected, feature_report = select_features(
        X_all,
        y,
        cfg.correlation_threshold,
        cfg.variance_threshold,
        cfg.max_selected_features,
        cfg.random_state,
        cfg.artifact_dir / "feature_selection_report.json",
    )
    (cfg.artifact_dir / "feature_list.json").write_text(json.dumps(list(X_selected.columns), indent=2), encoding="utf-8")

    log.info("Training and comparing regression models.")
    results = train_and_select_model(X_selected, y, cfg)

    intervals_path = cfg.artifact_dir / "prediction_intervals.csv"
    results["prediction_intervals"].to_csv(intervals_path, index=False)
    joblib.dump({"features": list(X_selected.columns), "target": cfg.target_column}, cfg.artifact_dir / "model_metadata.pkl")

    log.info("Generating evaluation plots and explainability artifacts.")
    plots = generate_evaluation_plots(results, X_selected, y, cfg.plot_dir)
    shap_paths = generate_shap_artifacts(results["best_model"], results["X_test"], cfg.plot_dir, cfg.shap_sample_size)

    context = {
        **results,
        "schema_report": schema_report,
        "profile": profile,
        "quality": quality,
        "feature_report": feature_report,
        "selected_features": list(X_selected.columns),
        "plots": plots,
        "shap_paths": shap_paths,
    }
    write_reports(context, cfg.report_dir, cfg.artifact_dir)
    (cfg.artifact_dir / "data_profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")
    (cfg.artifact_dir / "data_quality_report.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    log.info("ML pipeline complete. Best model: %s", results["best_model_name"])
    return context

