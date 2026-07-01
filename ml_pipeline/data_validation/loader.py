from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_ml_dataset(path: str | Path, target_column: str, source_target_column: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ML dataset not found: {path}")
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported dataset format: {path}")
    if target_column not in df.columns:
        if source_target_column not in df.columns:
            raise ValueError(f"Neither {target_column!r} nor {source_target_column!r} exists in dataset.")
        df[target_column] = df[source_target_column]
    return df

