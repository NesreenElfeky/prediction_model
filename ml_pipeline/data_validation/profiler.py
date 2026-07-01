from __future__ import annotations

import pandas as pd


def profile_dataframe(df: pd.DataFrame, target_column: str) -> dict:
    numeric = df.select_dtypes(include="number")
    missing = df.isna().sum().sort_values(ascending=False)
    return {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "numeric_columns": int(len(numeric.columns)),
        "categorical_columns": int(len(df.columns) - len(numeric.columns)),
        "missing_values": {k: int(v) for k, v in missing[missing > 0].items()},
        "target_distribution": {
            k: float(v)
            for k, v in df[target_column].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.95, 0.99]).items()
        },
    }

