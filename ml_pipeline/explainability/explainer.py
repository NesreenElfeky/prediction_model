from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def generate_shap_artifacts(model, X: pd.DataFrame, plot_dir: str | Path, sample_size: int) -> dict:
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    sample = X.sample(min(sample_size, len(X)), random_state=42)
    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["regressor"].regressor_
    transformed = preprocessor.transform(sample)
    paths = {}
    try:
        explainer = shap.Explainer(estimator, transformed, feature_names=list(sample.columns))
        shap_values = explainer(transformed)
        summary_path = plot_dir / "shap_summary.png"
        shap.plots.beeswarm(shap_values, show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(summary_path, dpi=170)
        plt.close()
        paths["shap_summary"] = str(summary_path)

        waterfall_path = plot_dir / "shap_waterfall.png"
        shap.plots.waterfall(shap_values[0], show=False, max_display=15)
        plt.tight_layout()
        plt.savefig(waterfall_path, dpi=170)
        plt.close()
        paths["shap_waterfall"] = str(waterfall_path)

        values = np.abs(shap_values.values).mean(axis=0)
        importance = pd.DataFrame({"feature": sample.columns, "mean_abs_shap": values}).sort_values("mean_abs_shap", ascending=False)
        importance.to_csv(plot_dir / "shap_importance.csv", index=False)
        paths["shap_importance_csv"] = str(plot_dir / "shap_importance.csv")
    except Exception as exc:
        (plot_dir / "shap_error.txt").write_text(str(exc), encoding="utf-8")
        paths["shap_error"] = str(plot_dir / "shap_error.txt")
    return paths

