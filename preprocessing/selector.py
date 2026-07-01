from __future__ import annotations

import numpy as np
import pandas as pd


def correlation_filter(df: pd.DataFrame, target: str, threshold: float = 0.98) -> pd.DataFrame:
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        return df
    corr = numeric.drop(columns=[target], errors="ignore").corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_cols = [column for column in upper.columns if any(upper[column] > threshold)]
    return df.drop(columns=drop_cols)
