from __future__ import annotations

import numpy as np
import pandas as pd


def feature_quality_report(df: pd.DataFrame, target_column: str, id_column: str) -> dict:
    feature_df = df.drop(columns=[target_column, id_column], errors="ignore")
    numeric = feature_df.select_dtypes(include="number")
    outliers: dict[str, int] = {}
    for col in numeric.columns:
        series = numeric[col].dropna()
        if series.empty:
            continue
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            outliers[col] = 0
            continue
        outliers[col] = int(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum())
    corr = numeric.corr(numeric_only=True).abs()
    high_corr_pairs = []
    for i, col in enumerate(corr.columns):
        for other in corr.columns[i + 1 :]:
            value = corr.loc[col, other]
            if pd.notna(value) and value >= 0.95:
                high_corr_pairs.append({"feature_a": col, "feature_b": other, "correlation": float(value)})
    return {
        "missing_by_column": {k: int(v) for k, v in feature_df.isna().sum().items() if v > 0},
        "outliers_iqr": outliers,
        "high_correlation_pairs": high_corr_pairs[:100],
    }

