from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def train_val_test_split(
    df: pd.DataFrame,
    target: str = "expected_annual_loss_usd",
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if target not in df.columns:
        raise ValueError(f"Target column not found: {target}")
    train, temp = train_test_split(df, test_size=0.3, random_state=seed)
    val, test = train_test_split(temp, test_size=0.5, random_state=seed)
    return train, val, test

