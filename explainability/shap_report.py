from __future__ import annotations

from pathlib import Path

import pandas as pd


def feature_importance_report(model, feature_names: list[str], output_path: str | Path | None = None) -> pd.DataFrame:
    values = getattr(model, "feature_importances_", None)
    if values is None:
        values = getattr(model, "coef_", None)
    if values is None:
        return pd.DataFrame(columns=["feature", "importance"])
    report = pd.DataFrame({"feature": feature_names, "importance": values}).sort_values("importance", ascending=False)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(output_path, index=False)
    return report

