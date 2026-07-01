from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from models.baselines import random_forest


def train_regressor(df: pd.DataFrame, target: str = "expected_annual_loss_usd", model_path: str | Path | None = None):
    X = df.drop(columns=[target, "asset_id"], errors="ignore").select_dtypes(include="number")
    y = df[target]
    model = random_forest()
    model.fit(X, y)
    if model_path:
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": model, "features": list(X.columns), "target": target}, model_path)
    return model

