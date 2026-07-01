from __future__ import annotations

import json
import logging
import math
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

log = logging.getLogger(__name__)


class PredictionError(ValueError):
    """Raised when prediction input or artifacts are invalid."""


@dataclass(frozen=True)
class FeatureRule:
    """Validation rule for one model feature."""

    minimum: float | None = None
    maximum: float | None = None
    integer: bool = False
    binary: bool = False


DEFAULT_FEATURE_RULES: dict[str, FeatureRule] = {
    "epss_score_max": FeatureRule(0.0, 1.0),
    "epss_score_mean": FeatureRule(0.0, 1.0),
    "cvss_score_mean": FeatureRule(0.0, 1.0),
    "cvss_score_max": FeatureRule(0.0, 1.0),
    "is_exploitable_max": FeatureRule(0.0, 1.0, binary=True),
    "is_critical_max": FeatureRule(0.0, 1.0, binary=True),
    "kev_flag_max": FeatureRule(0.0, 1.0, binary=True),
    "vuln_count": FeatureRule(0.0, None, integer=True),
    "is_exploitable_sum": FeatureRule(0.0, None, integer=True),
    "is_critical_sum": FeatureRule(0.0, None, integer=True),
    "kev_flag_sum": FeatureRule(0.0, None, integer=True),
    "sev_count_low": FeatureRule(0.0, None, integer=True),
    "weighted_risk_index": FeatureRule(0.0, None),
    "composite_vulnerability_score": FeatureRule(0.0, None),
    "attack_surface_score": FeatureRule(0.0, None),
    "recent_threat_score": FeatureRule(0.0, 1.0),
    "vulnerability_age": FeatureRule(0.0, None),
    "epss_exploit_count_interaction": FeatureRule(0.0, None),
}

MODEL_READY_SIGNED_FEATURES = {
    "exploit_count_sum",
    "days_since_published_max",
    "days_since_published_min",
    "risk_exploit_product_max",
    "risk_exploit_product_mean",
}

FEATURE_EXPORT_SCHEMA = {
    "epss_score_max": ("raw_feature__epss_score_max", "raw"),
    "weighted_risk_index": ("derived_feature__weighted_risk_index", "derived"),
    "composite_vulnerability_score": ("derived_feature__composite_vulnerability_score", "derived"),
    "epss_score_mean": ("raw_feature__epss_score_mean", "raw"),
    "attack_surface_score": ("derived_feature__attack_surface_score", "derived"),
    "recent_threat_score": ("derived_feature__recent_threat_score", "derived"),
    "cvss_score_mean": ("raw_feature__cvss_score_mean_0_to_10", "cvss_0_to_10"),
    "vuln_count": ("raw_feature__vulnerability_count", "raw"),
    "exploit_count_sum": ("scaled_feature__exploit_count_sum", "scaled"),
    "is_exploitable_sum": ("raw_feature__exploitable_vulnerability_count", "raw"),
    "is_exploitable_max": ("raw_feature__has_exploitable_vulnerability", "raw"),
    "cvss_score_max": ("raw_feature__cvss_score_max_0_to_10", "cvss_0_to_10"),
    "days_since_published_max": ("scaled_feature__days_since_published_max", "scaled"),
    "days_since_published_min": ("scaled_feature__days_since_published_min", "scaled"),
    "vulnerability_age": ("derived_feature__vulnerability_age", "derived"),
    "is_critical_sum": ("raw_feature__critical_vulnerability_count", "raw"),
    "is_critical_max": ("raw_feature__has_critical_vulnerability", "raw"),
    "epss_exploit_count_interaction": ("derived_feature__epss_exploit_count_interaction", "derived"),
    "risk_exploit_product_max": ("scaled_feature__risk_exploit_product_max", "scaled"),
    "risk_exploit_product_mean": ("scaled_feature__risk_exploit_product_mean", "scaled"),
    "kev_flag_max": ("raw_feature__has_known_exploited_vulnerability", "raw"),
    "kev_flag_sum": ("raw_feature__known_exploited_vulnerability_count", "raw"),
    "sev_count_low": ("raw_feature__low_severity_vulnerability_count", "raw"),
}

FEATURE_LABELS = {
    "epss_score_max": "High EPSS Score",
    "epss_score_mean": "Average EPSS Score",
    "cvss_score_max": "High CVSS Score",
    "cvss_score_mean": "Average CVSS Score",
    "vuln_count": "Vulnerability Count",
    "is_critical_sum": "Critical Vulnerabilities",
    "is_critical_max": "Critical Vulnerability Present",
    "is_exploitable_sum": "Exploitable Vulnerabilities",
    "is_exploitable_max": "Exploitability Present",
    "kev_flag_sum": "Known Exploited Vulnerabilities",
    "kev_flag_max": "Known Exploited Vulnerability Present",
    "attack_surface_score": "Large Attack Surface",
    "weighted_risk_index": "Weighted Risk Index",
    "composite_vulnerability_score": "Composite Vulnerability Score",
    "recent_threat_score": "Recent Threat Activity",
    "epss_exploit_count_interaction": "Exploit Likelihood Interaction",
}


@dataclass(frozen=True)
class Explanation:
    """Per-feature contribution details for one prediction."""

    feature: str
    label: str
    contribution: float
    impact: str


class Predictor:
    """Loads trained artifacts and produces production predictions.

    The class has no CLI assumptions and can be imported directly from FastAPI,
    Flask, Streamlit, batch jobs, or notebooks.
    """

    def __init__(
        self,
        model_dir: str | Path = "models",
        risk_thresholds_path: str | Path = "config/risk_thresholds.yaml",
        plots_dir: str | Path = "plots",
    ) -> None:
        self.model_dir = Path(model_dir)
        self.risk_thresholds_path = Path(risk_thresholds_path)
        self.plots_dir = Path(plots_dir)
        self.model: Any | None = None
        self.preprocessing_pipeline: Any | None = None
        self.feature_columns: list[str] = []
        self.metadata: dict[str, Any] = {}
        self.risk_thresholds: dict[str, float] = {}

    def load(self) -> "Predictor":
        """Load model, preprocessing, feature metadata, and risk thresholds."""
        log.info("Loading model...")
        self.model = self._load_pickle(self.model_dir / "best_model.pkl")
        log.info("Loading preprocessing pipeline...")
        self.preprocessing_pipeline = self._load_pickle(self.model_dir / "preprocessing_pipeline.pkl")
        self.feature_columns = self._load_feature_columns(self.model_dir / "feature_columns.json")
        self.metadata = self._load_json(self.model_dir / "metadata.json", required=False)
        self.risk_thresholds = self._load_risk_thresholds(self.risk_thresholds_path)
        self._patch_runtime_metadata()
        log.info("Loaded %d feature columns.", len(self.feature_columns))
        return self

    def predict(self, data: Mapping[str, Any] | pd.DataFrame) -> dict[str, Any] | pd.DataFrame:
        """Predict from either a single mapping or a dataframe."""
        if isinstance(data, pd.DataFrame):
            return self.predict_dataframe(data)
        return self.predict_single(data)

    def predict_single(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """Predict financial loss for one record.

        Args:
            record: Mapping containing every required model feature.

        Returns:
            Prediction details including loss, risk, interval, confidence,
            currency, timestamp, and top contributing features.
        """
        result_df, explanations = self._predict_internal(pd.DataFrame([dict(record)]), explain=True)
        row = result_df.iloc[0].to_dict()
        row["prediction_interval_95"] = {
            "lower": row.pop("prediction_interval_lower"),
            "upper": row.pop("prediction_interval_upper"),
        }
        row["top_features"] = [item.__dict__ for item in explanations[0][:10]]
        return row

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict financial loss for a dataframe."""
        result_df, _ = self._predict_internal(df, explain=True)
        return result_df

    def explain_prediction(self, record: Mapping[str, Any]) -> list[Explanation]:
        """Generate SHAP or fallback feature-contribution explanation."""
        _, explanations = self._predict_internal(pd.DataFrame([dict(record)]), explain=True)
        return explanations[0]

    def _predict_internal(self, df: pd.DataFrame, explain: bool) -> tuple[pd.DataFrame, list[list[Explanation]]]:
        self._ensure_loaded()
        log.info("Validating input...")
        validated = self._validate_dataframe(df)

        log.info("Generating prediction...")
        predictions = np.maximum(np.asarray(self.model.predict(validated), dtype=float), 0.0)

        log.info("Calculating confidence...")
        uncertainty = self._prediction_uncertainty(validated, predictions)
        lower = np.maximum(predictions - 1.96 * uncertainty, 0.0)
        upper = predictions + 1.96 * uncertainty
        confidence = self._confidence_scores(predictions, uncertainty)

        explanations: list[list[Explanation]] = [[] for _ in range(len(validated))]
        if explain:
            log.info("Generating SHAP explanation...")
            explanations = self._generate_explanations(validated, write_plot=True)

        result = self._display_input_dataframe(df)
        result["predicted_loss"] = predictions
        result["currency"] = self.metadata.get("currency", "USD")
        result["risk_level"] = [self._risk_level(value) for value in predictions]
        result["confidence_score"] = confidence
        result["prediction_interval_lower"] = lower
        result["prediction_interval_upper"] = upper
        result["top_feature_1"] = [self._top_label(items, 0) for items in explanations]
        result["top_feature_2"] = [self._top_label(items, 1) for items in explanations]
        result["top_feature_3"] = [self._top_label(items, 2) for items in explanations]
        result["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

        log.info("Prediction completed successfully.")
        return result, explanations

    def _validate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            raise PredictionError("Input data is empty.")

        missing = [feature for feature in self.feature_columns if feature not in df.columns]
        if missing:
            raise PredictionError("Missing required feature(s): " + ", ".join(missing))

        unexpected_categorical = [
            column
            for column in df.columns
            if column in self.feature_columns and not pd.api.types.is_numeric_dtype(df[column])
        ]
        if unexpected_categorical:
            log.debug("Validating possible non-numeric feature columns: %s", unexpected_categorical)

        validated = pd.DataFrame(index=df.index)
        errors: list[str] = []
        for feature in self.feature_columns:
            series = df[feature]
            numeric = pd.to_numeric(series, errors="coerce")

            bad_type = series.notna() & numeric.isna()
            if bad_type.any():
                values = series[bad_type].astype(str).head(3).tolist()
                errors.append(f"{feature} must be numeric; invalid value(s): {values}.")

            if series.isna().any() or numeric.isna().any():
                rows = numeric[numeric.isna()].index.astype(str).tolist()[:5]
                errors.append(f"{feature} contains missing or NaN value(s) at row(s): {rows}.")

            finite = numeric.replace([np.inf, -np.inf], np.nan)
            if finite.isna().any():
                rows = finite[finite.isna()].index.astype(str).tolist()[:5]
                errors.append(f"{feature} must be finite at row(s): {rows}.")

            rule = DEFAULT_FEATURE_RULES.get(feature, FeatureRule())
            if rule.minimum is not None and (numeric < rule.minimum).any():
                errors.append(f"{feature} must be greater than or equal to {rule.minimum}.")
            if rule.maximum is not None and (numeric > rule.maximum).any():
                errors.append(f"{feature} must be less than or equal to {rule.maximum}.")
            if rule.binary and (~numeric.isin([0, 1])).any():
                errors.append(f"{feature} must be binary: 0 or 1.")
            if rule.integer and ((numeric % 1) != 0).any():
                errors.append(f"{feature} must be an integer count.")

            validated[feature] = numeric.astype(float)

        if errors:
            preview = " ".join(errors[:8])
            if len(errors) > 8:
                preview += f" Plus {len(errors) - 8} more validation error(s)."
            raise PredictionError(preview)
        return validated[self.feature_columns]

    def _risk_level(self, prediction: float) -> str:
        ordered = [
            ("Very Low", self.risk_thresholds["very_low"]),
            ("Low", self.risk_thresholds["low"]),
            ("Medium", self.risk_thresholds["medium"]),
            ("High", self.risk_thresholds["high"]),
        ]
        for label, threshold in ordered:
            if prediction < threshold:
                return label
        return "Critical"

    def _prediction_uncertainty(self, validated: pd.DataFrame, predictions: np.ndarray) -> np.ndarray:
        """Estimate row-level uncertainty from ensemble spread and distribution distance."""
        transformed = self.preprocessing_pipeline.transform(validated)
        estimator = self.model.named_steps["regressor"].regressor_
        staged_predict = getattr(estimator, "staged_predict", None)
        mape = float(self.metadata.get("metrics", {}).get("mape", 3.0) or 3.0) / 100.0
        min_error = np.maximum(predictions * max(mape, 0.01), 1.0)
        distance = self._training_distribution_distance(validated)
        distribution_multiplier = 1.0 + np.minimum(distance, 12.0) / 2.5
        if staged_predict is None:
            return min_error * distribution_multiplier

        stage_values = np.vstack([np.expm1(stage) for stage in staged_predict(transformed)])
        tail_start = max(0, int(stage_values.shape[0] * 0.5))
        tree_std = np.std(stage_values[tail_start:], axis=0)
        return np.maximum(tree_std, min_error) * distribution_multiplier

    def _confidence_scores(self, predictions: np.ndarray, uncertainty: np.ndarray) -> list[float]:
        scale = np.maximum(predictions, float(self.metadata.get("metrics", {}).get("mae", 1.0) or 1.0))
        relative_uncertainty = uncertainty / scale
        interval_width_ratio = (1.96 * 2.0 * uncertainty) / scale
        scores = 100.0 * np.exp(-relative_uncertainty) * np.exp(-0.08 * interval_width_ratio)
        return [round(float(np.clip(score, 0.0, 100.0)), 2) for score in scores]

    def _training_distribution_distance(self, validated: pd.DataFrame) -> np.ndarray:
        """Calculate robust distance from the training feature distribution."""
        profiles = self.metadata.get("feature_profiles", {})
        if not profiles:
            return np.zeros(len(validated), dtype=float)

        distances = []
        for _, row in validated.iterrows():
            feature_distances = []
            for feature in self.feature_columns:
                profile = profiles.get(feature, {})
                median = float(profile.get("median", 0.0))
                iqr = max(float(profile.get("iqr", 1.0)), 1e-6)
                p01 = float(profile.get("p01", -np.inf))
                p99 = float(profile.get("p99", np.inf))
                value = float(row[feature])
                robust_z = abs(value - median) / iqr
                tail_penalty = 2.0 if value < p01 or value > p99 else 0.0
                feature_distances.append(min(robust_z + tail_penalty, 12.0))
            distances.append(float(np.mean(feature_distances)))
        return np.asarray(distances, dtype=float)

    def _generate_explanations(self, validated: pd.DataFrame, write_plot: bool) -> list[list[Explanation]]:
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        plot_path = self.plots_dir / "prediction_shap.png"
        try:
            import shap

            transformed = self.preprocessing_pipeline.transform(validated)
            estimator = self.model.named_steps["regressor"].regressor_
            explainer = shap.Explainer(estimator)
            shap_values = explainer(transformed)
            values = np.asarray(shap_values.values, dtype=float)

            if write_plot and len(validated) > 0:
                shap.plots.waterfall(shap_values[0], max_display=10, show=False)
                plt.tight_layout()
                plt.savefig(plot_path, dpi=170, bbox_inches="tight")
                plt.close()
            return [self._rank_contributions(row_values) for row_values in values]
        except Exception as exc:
            log.warning("SHAP explanation failed; using feature importance fallback: %s", exc)
            return self._fallback_explanations(validated, plot_path)

    def _fallback_explanations(self, validated: pd.DataFrame, plot_path: Path) -> list[list[Explanation]]:
        estimator = self.model.named_steps["regressor"].regressor_
        importances = np.asarray(getattr(estimator, "feature_importances_", np.ones(len(self.feature_columns))), dtype=float)
        rows = validated.to_numpy(dtype=float) * importances
        explanations = [self._rank_contributions(row) for row in rows]

        if explanations:
            top = explanations[0][:10]
            plt.figure(figsize=(10, 6))
            plt.barh([item.label for item in reversed(top)], [item.contribution for item in reversed(top)])
            plt.xlabel("Contribution")
            plt.title("Top prediction contributors")
            plt.tight_layout()
            plt.savefig(plot_path, dpi=170, bbox_inches="tight")
            plt.close()
        return explanations

    def _rank_contributions(self, contributions: np.ndarray) -> list[Explanation]:
        ranked_indices = np.argsort(np.abs(contributions))[::-1]
        explanations = []
        for idx in ranked_indices:
            value = float(contributions[idx])
            feature = self.feature_columns[int(idx)]
            explanations.append(
                Explanation(
                    feature=feature,
                    label=FEATURE_LABELS.get(feature, feature.replace("_", " ").title()),
                    contribution=round(value, 6),
                    impact="Increases risk" if value >= 0 else "Decreases risk",
                )
            )
        return explanations

    def _display_input_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        display = pd.DataFrame(index=df.index)
        for column in df.columns:
            export_name, export_kind = FEATURE_EXPORT_SCHEMA.get(column, (f"derived_feature__{column}", "derived"))
            numeric = pd.to_numeric(df[column], errors="coerce")
            if export_kind == "cvss_0_to_10":
                display[export_name] = (numeric * 10.0).round(2)
            elif export_kind == "raw":
                display[export_name] = numeric.clip(lower=0) if self._should_clip_for_display(column) else df[column]
            elif export_kind == "scaled":
                display[export_name] = numeric
            else:
                display[export_name] = df[column]
        return display

    def _should_clip_for_display(self, column: str) -> bool:
        if column in MODEL_READY_SIGNED_FEATURES:
            return True
        return column.endswith(("_count", "_sum", "_max")) and column not in {
            "epss_score_max",
            "cvss_score_max",
            "kev_flag_max",
            "is_critical_max",
            "is_exploitable_max",
        }

    def _top_label(self, explanations: list[Explanation], index: int) -> str:
        if index >= len(explanations):
            return ""
        return explanations[index].label

    def _patch_runtime_metadata(self) -> None:
        self.metadata.setdefault("currency", "USD")
        self.metadata.setdefault("currency_symbol", "$")
        self.metadata.setdefault("feature_count", len(self.feature_columns))
        self.metadata.setdefault("feature_names", self.feature_columns)
        self.metadata.setdefault("python_version", platform.python_version())
        self.metadata.setdefault("dataset_version", "datasets/final/ml_dataset.parquet")

    def _ensure_loaded(self) -> None:
        if self.model is None or self.preprocessing_pipeline is None or not self.feature_columns:
            raise PredictionError("Predictor is not loaded. Call Predictor().load() before predicting.")

    def _load_pickle(self, path: Path) -> Any:
        if not path.exists():
            raise PredictionError(f"Missing model artifact: {path}")
        try:
            return joblib.load(path)
        except Exception as exc:
            raise PredictionError(f"Could not load pickle artifact {path}: {exc}") from exc

    def _load_feature_columns(self, path: Path) -> list[str]:
        data = self._load_json(path)
        if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
            raise PredictionError(f"{path} must contain a JSON list of feature names.")
        return data

    def _load_json(self, path: Path, required: bool = True) -> Any:
        if not path.exists():
            if required:
                raise PredictionError(f"Missing JSON artifact: {path}")
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise PredictionError(f"Could not read JSON artifact {path}: {exc}") from exc

    def _load_risk_thresholds(self, path: Path) -> dict[str, float]:
        if not path.exists():
            raise PredictionError(f"Missing risk threshold configuration: {path}")
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            raise PredictionError(f"Could not read risk thresholds {path}: {exc}") from exc

        thresholds = raw.get("thresholds", raw)
        required = ["very_low", "low", "medium", "high"]
        missing = [key for key in required if key not in thresholds]
        if missing:
            raise PredictionError("Missing risk threshold(s): " + ", ".join(missing))

        parsed = {key: float(thresholds[key]) for key in required}
        values = [parsed["very_low"], parsed["low"], parsed["medium"], parsed["high"]]
        if any(math.isnan(value) or value < 0 for value in values) or values != sorted(values):
            raise PredictionError("Risk thresholds must be non-negative and increasing.")
        return parsed
