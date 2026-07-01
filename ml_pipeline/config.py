from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from utils.io import load_yaml


@dataclass(frozen=True)
class MLConfig:
    input_path: Path
    target_column: str
    source_target_column: str
    id_column: str
    test_size: float
    validation_size: float
    random_state: int
    artifact_dir: Path
    report_dir: Path
    plot_dir: Path
    leakage_columns: list[str]
    correlation_threshold: float
    variance_threshold: float
    max_selected_features: int
    cv_folds: int
    repeated_cv_repeats: int
    random_search_iter: int
    optuna_trials: int
    shap_sample_size: int
    n_jobs: int
    raw: dict[str, Any]


def load_ml_config(path: str | Path = "configs/ml.yaml") -> MLConfig:
    raw = load_yaml(path)
    data = raw["data"]
    outputs = raw["outputs"]
    features = raw["features"]
    models = raw["models"]
    return MLConfig(
        input_path=Path(data["input_path"]),
        target_column=data["target_column"],
        source_target_column=data["source_target_column"],
        id_column=data["id_column"],
        test_size=float(data["test_size"]),
        validation_size=float(data["validation_size"]),
        random_state=int(data["random_state"]),
        artifact_dir=Path(outputs["root"]),
        report_dir=Path(outputs["reports"]),
        plot_dir=Path(outputs["plots"]),
        leakage_columns=list(features["leakage_columns"]),
        correlation_threshold=float(features["correlation_threshold"]),
        variance_threshold=float(features["variance_threshold"]),
        max_selected_features=int(features["max_selected_features"]),
        cv_folds=int(models["cv_folds"]),
        repeated_cv_repeats=int(models["repeated_cv_repeats"]),
        random_search_iter=int(models["random_search_iter"]),
        optuna_trials=int(models["optuna_trials"]),
        shap_sample_size=int(models["shap_sample_size"]),
        n_jobs=int(models["n_jobs"]),
        raw=raw,
    )

