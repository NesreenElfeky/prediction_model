from __future__ import annotations

import pandas as pd


def one_hot_encode(df: pd.DataFrame, target: str = "expected_annual_loss_usd") -> pd.DataFrame:
    feature_df = df.copy()
    categorical = [c for c in feature_df.select_dtypes(include=["object", "category", "bool"]).columns if c != target]
    return pd.get_dummies(feature_df, columns=categorical, dummy_na=True)

