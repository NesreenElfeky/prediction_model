from __future__ import annotations

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression
from xgboost import XGBRegressor


def model_registry(random_state: int, n_jobs: int) -> dict:
    return {
        "Linear Regression": LinearRegression(),
        "ElasticNet": ElasticNet(alpha=0.005, l1_ratio=0.2, random_state=random_state, max_iter=5000),
        "Random Forest": RandomForestRegressor(n_estimators=120, min_samples_leaf=2, random_state=random_state, n_jobs=n_jobs),
        "Extra Trees": ExtraTreesRegressor(n_estimators=120, min_samples_leaf=2, random_state=random_state, n_jobs=n_jobs),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=120, learning_rate=0.05, max_depth=3, random_state=random_state),
        "XGBoost": XGBRegressor(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=n_jobs,
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=120,
            learning_rate=0.05,
            num_leaves=31,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=-1,
        ),
        "CatBoost": CatBoostRegressor(
            iterations=120,
            depth=5,
            learning_rate=0.05,
            random_seed=random_state,
            verbose=False,
            allow_writing_files=False,
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=120, learning_rate=0.05, random_state=random_state),
    }


def random_search_spaces() -> dict:
    return {
        "Random Forest": {
            "regressor__regressor__n_estimators": [80, 120, 180],
            "regressor__regressor__max_depth": [None, 8, 14, 20],
            "regressor__regressor__min_samples_leaf": [1, 2, 4],
        },
        "Extra Trees": {
            "regressor__regressor__n_estimators": [80, 120, 180],
            "regressor__regressor__max_depth": [None, 8, 14, 20],
            "regressor__regressor__min_samples_leaf": [1, 2, 4],
        },
        "XGBoost": {
            "regressor__regressor__n_estimators": [80, 120, 180],
            "regressor__regressor__max_depth": [3, 4, 6],
            "regressor__regressor__learning_rate": [0.03, 0.05, 0.08],
        },
        "LightGBM": {
            "regressor__regressor__n_estimators": [80, 120, 180],
            "regressor__regressor__num_leaves": [15, 31, 63],
            "regressor__regressor__learning_rate": [0.03, 0.05, 0.08],
        },
        "CatBoost": {
            "regressor__regressor__iterations": [80, 120, 180],
            "regressor__regressor__depth": [4, 5, 6],
            "regressor__regressor__learning_rate": [0.03, 0.05, 0.08],
        },
    }
