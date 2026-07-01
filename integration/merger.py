from __future__ import annotations

from functools import reduce
from pathlib import Path

import pandas as pd

from integration.integrity_checker import require_key
from utils.io import write_table


def merge_on_asset_id(frames: list[pd.DataFrame], how: str = "left") -> pd.DataFrame:
    non_empty = [df.copy() for df in frames if df is not None and not df.empty]
    if not non_empty:
        return pd.DataFrame()
    for df in non_empty:
        require_key(df, "asset_id")
    merged = reduce(lambda left, right: left.merge(right, on="asset_id", how=how, suffixes=("", "_dup")), non_empty)
    duplicate_cols = [col for col in merged.columns if col.endswith("_dup")]
    return merged.drop(columns=duplicate_cols)


def run_integration(frames: list[pd.DataFrame], output_path: str | Path | None = None) -> pd.DataFrame:
    merged = merge_on_asset_id(frames)
    if output_path:
        write_table(merged, output_path)
    return merged

