from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, RepeatedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from ml_pipeline.config import MLConfig
from ml_pipeline.models.registry import model_registry, random_search_spaces


log = logging.getLogger(__name__)


def build_preprocessor() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )


def build_pipeline(model) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("regressor", TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1)),
        ]
    )


def _metrics(y_true, y_pred) -> dict[str, float]:
    denom = np.where(np.asarray(y_true) == 0, 1, np.asarray(y_true))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": float(np.mean(np.abs((np.asarray(y_true) - np.asarray(y_pred)) / denom)) * 100),
        "median_absolute_error": float(np.median(np.abs(np.asarray(y_true) - np.asarray(y_pred)))),
        "explained_variance": float(1 - np.var(np.asarray(y_true) - np.asarray(y_pred)) / np.var(y_true)),
    }


def train_and_select_model(X: pd.DataFrame, y: pd.Series, cfg: MLConfig) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=cfg.test_size, random_state=cfg.random_state)
    registry = model_registry(cfg.random_state, cfg.n_jobs)
    cv = KFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.random_state)
    repeated_cv = RepeatedKFold(n_splits=cfg.cv_folds, n_repeats=cfg.repeated_cv_repeats, random_state=cfg.random_state)
    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }
    leaderboard_rows = []
    fitted_models = {}
    for name, model in registry.items():
        log.info("Training model: %s", name)
        pipe = build_pipeline(model)
        cv_scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
        repeated_scores = cross_validate(pipe, X_train, y_train, cv=repeated_cv, scoring=scoring, n_jobs=1)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        holdout = _metrics(y_test, pred)
        leaderboard_rows.append(
            {
                "model": name,
                "cv_rmse_mean": float(-cv_scores["test_rmse"].mean()),
                "cv_mae_mean": float(-cv_scores["test_mae"].mean()),
                "cv_r2_mean": float(cv_scores["test_r2"].mean()),
                "repeated_cv_rmse_mean": float(-repeated_scores["test_rmse"].mean()),
                **{f"holdout_{k}": v for k, v in holdout.items()},
            }
        )
        fitted_models[name] = pipe

    leaderboard = pd.DataFrame(leaderboard_rows).sort_values(
        ["holdout_r2", "holdout_rmse", "holdout_mae"],
        ascending=[False, True, True],
    )
    top_model_name = str(leaderboard.iloc[0]["model"])

    tuned_name, tuned_model, tuning_report = _tune_best_models(top_model_name, registry, X_train, y_train, cfg)
    tuned_pred = tuned_model.predict(X_test)
    tuned_metrics = _metrics(y_test, tuned_pred)
    tuned_row = {
        "model": f"{tuned_name} Tuned",
        "cv_rmse_mean": tuning_report.get("best_cv_rmse"),
        "cv_mae_mean": np.nan,
        "cv_r2_mean": np.nan,
        "repeated_cv_rmse_mean": np.nan,
        **{f"holdout_{k}": v for k, v in tuned_metrics.items()},
    }
    leaderboard = pd.concat([leaderboard, pd.DataFrame([tuned_row])], ignore_index=True).sort_values(
        ["holdout_r2", "holdout_rmse", "holdout_mae"],
        ascending=[False, True, True],
    )

    best_name = str(leaderboard.iloc[0]["model"])
    best_model = tuned_model if best_name.endswith("Tuned") else fitted_models[best_name]
    best_pred = best_model.predict(X_test)
    residuals = y_test.to_numpy() - best_pred
    prediction_intervals = _prediction_intervals(best_pred, residuals)

    cfg.artifact_dir.mkdir(parents=True, exist_ok=True)
    cfg.report_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, cfg.artifact_dir / "trained_model.pkl")
    joblib.dump(best_model.named_steps["preprocessor"], cfg.artifact_dir / "preprocessor.pkl")
    leaderboard.to_csv(cfg.artifact_dir / "leaderboard.csv", index=False)
    metrics = _metrics(y_test, best_pred)
    (cfg.artifact_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (cfg.artifact_dir / "training_config.json").write_text(json.dumps(cfg.raw, indent=2), encoding="utf-8")
    return {
        "best_model_name": best_name,
        "best_model": best_model,
        "leaderboard": leaderboard,
        "metrics": metrics,
        "tuning_report": tuning_report,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": best_pred,
        "residuals": residuals,
        "prediction_intervals": prediction_intervals,
    }


def _tune_best_models(top_model_name: str, registry: dict, X_train: pd.DataFrame, y_train: pd.Series, cfg: MLConfig) -> tuple[str, Pipeline, dict]:
    spaces = random_search_spaces()
    candidate_name = top_model_name if top_model_name in spaces else next((name for name in spaces if name in registry), top_model_name)
    base = build_pipeline(registry[candidate_name])
    report = {"candidate": candidate_name}
    if candidate_name in spaces:
        search = RandomizedSearchCV(
            base,
            spaces[candidate_name],
            n_iter=cfg.random_search_iter,
            scoring="neg_root_mean_squared_error",
            cv=3,
            random_state=cfg.random_state,
            n_jobs=1,
        )
        search.fit(X_train, y_train)
        base = search.best_estimator_
        report["randomized_search_best_params"] = search.best_params_
        report["best_cv_rmse"] = float(-search.best_score_)

    def objective(trial: optuna.Trial) -> float:
        model = registry[candidate_name]
        params = {}
        if candidate_name in {"Random Forest", "Extra Trees"}:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 80, 180),
                "max_depth": trial.suggest_int("max_depth", 5, 24),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            }
        elif candidate_name == "XGBoost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 80, 180),
                "max_depth": trial.suggest_int("max_depth", 3, 7),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.1),
            }
        elif candidate_name == "LightGBM":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 80, 180),
                "num_leaves": trial.suggest_int("num_leaves", 15, 80),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.1),
            }
        else:
            return report.get("best_cv_rmse", 0.0)
        model.set_params(**params)
        pipe = build_pipeline(model)
        scores = cross_validate(pipe, X_train, y_train, cv=3, scoring="neg_root_mean_squared_error", n_jobs=1)
        return float(-scores["test_score"].mean())

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=cfg.random_state))
    study.optimize(objective, n_trials=cfg.optuna_trials, show_progress_bar=False)
    report["optuna_best_params"] = study.best_params
    report["optuna_best_rmse"] = float(study.best_value)

    tuned_model = registry[candidate_name]
    if study.best_params:
        tuned_model.set_params(**study.best_params)
    tuned_pipe = build_pipeline(tuned_model)
    tuned_pipe.fit(X_train, y_train)
    return candidate_name, tuned_pipe, report


def _prediction_intervals(predictions: np.ndarray, residuals: np.ndarray) -> pd.DataFrame:
    lower_error, upper_error = np.quantile(residuals, [0.05, 0.95])
    return pd.DataFrame(
        {
            "prediction": predictions,
            "p05": np.maximum(predictions + lower_error, 0),
            "p95": np.maximum(predictions + upper_error, 0),
        }
    )
