from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_FEATURES = [
    "cvss_score_max",
    "epss_score_max",
    "vuln_count",
    "unique_cve_count",
]


def validate_schema(df: pd.DataFrame, target_column: str, id_column: str) -> dict:
    errors: list[str] = []
    if id_column not in df.columns:
        errors.append(f"Missing id column: {id_column}")
    if target_column not in df.columns:
        errors.append(f"Missing target column: {target_column}")
    for col in REQUIRED_FEATURES:
        if col not in df.columns:
            errors.append(f"Missing required cybersecurity feature: {col}")
    if target_column in df.columns:
        y = pd.to_numeric(df[target_column], errors="coerce")
        if y.isna().any():
            errors.append(f"Target contains {int(y.isna().sum())} null/non-numeric values.")
        if np.isinf(y).any():
            errors.append("Target contains infinite values.")
        if (y < 0).any():
            errors.append("Target contains negative financial losses.")
    if errors:
        raise ValueError("Schema validation failed: " + " | ".join(errors))
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "target": target_column,
        "id_column": id_column,
        "required_features": REQUIRED_FEATURES,
    }

