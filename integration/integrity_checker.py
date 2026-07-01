from __future__ import annotations

import pandas as pd


def integrity_report(df: pd.DataFrame, key: str = "asset_id") -> dict:
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "missing_key": int(df[key].isna().sum()) if key in df.columns else None,
        "duplicate_keys": int(df.duplicated(key).sum()) if key in df.columns else None,
        "null_cells": int(df.isna().sum().sum()),
    }


def require_key(df: pd.DataFrame, key: str = "asset_id") -> None:
    if key not in df.columns:
        raise ValueError(f"Required key column not found: {key}")
    if df[key].isna().any():
        raise ValueError(f"Null values found in key column: {key}")

