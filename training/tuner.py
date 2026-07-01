from __future__ import annotations

import pandas as pd
from sklearn.model_selection import RandomizedSearchCV

from models.baselines import random_forest


def tune_random_forest(df: pd.DataFrame, target: str = "expected_annual_loss_usd", cv: int = 3):
    X = df.drop(columns=[target, "asset_id"], errors="ignore").select_dtypes(include="number")
    y = df[target]
    search = RandomizedSearchCV(
        random_forest(),
        {"max_depth": [6, 10, 15, None], "min_samples_leaf": [1, 2, 5]},
        n_iter=6,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        random_state=42,
    )
    return search.fit(X, y)

