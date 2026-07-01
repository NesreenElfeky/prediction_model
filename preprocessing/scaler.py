from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import RobustScaler


def robust_scale(df: pd.DataFrame, exclude: list[str] | None = None) -> tuple[pd.DataFrame, RobustScaler]:
    exclude = exclude or []
    out = df.copy()
    numeric = [c for c in out.select_dtypes(include="number").columns if c not in exclude]
    scaler = RobustScaler()
    if numeric:
        out[numeric] = scaler.fit_transform(out[numeric])
    return out, scaler

