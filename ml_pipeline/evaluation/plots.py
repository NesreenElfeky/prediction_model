from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.model_selection import learning_curve


def generate_evaluation_plots(results: dict, X: pd.DataFrame, y: pd.Series, plot_dir: str | Path) -> dict[str, str]:
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    y_test = results["y_test"]
    y_pred = results["y_pred"]
    residuals = results["residuals"]

    paths["target_distribution"] = _hist(y, plot_dir / "target_distribution.png", "Financial Loss Distribution", "financial_loss_usd")
    paths["prediction_vs_actual"] = _scatter(y_test, y_pred, plot_dir / "prediction_vs_actual.png")
    paths["residual_distribution"] = _hist(
        pd.Series(residuals),
        plot_dir / "residual_distribution.png",
        "Residual Distribution",
        "Residual",
        log_scale=False,
    )
    paths["error_histogram"] = _hist(pd.Series(np.abs(residuals)), plot_dir / "error_histogram.png", "Absolute Error Distribution", "Absolute Error")
    paths["correlation_heatmap"] = _correlation_heatmap(X, plot_dir / "correlation_heatmap.png")
    paths["feature_importance"] = _permutation_importance(results["best_model"], results["X_test"], results["y_test"], plot_dir / "feature_importance.png")
    paths["learning_curve"] = _learning_curve(results["best_model"], X, y, plot_dir / "learning_curve.png")
    paths["partial_dependence"] = _partial_dependence(results["best_model"], X, plot_dir / "partial_dependence.png")
    return paths


def _hist(series: pd.Series, path: Path, title: str, xlabel: str, log_scale: bool = True) -> str:
    plt.figure(figsize=(9, 5))
    values = series.dropna()
    plot_values = np.log1p(values.clip(lower=0)) if log_scale else values
    plt.hist(plot_values, bins=50, color="#2563eb", alpha=0.82)
    plt.title(title)
    plt.xlabel(f"log1p({xlabel})" if log_scale else xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return str(path)


def _scatter(y_true, y_pred, path: Path) -> str:
    plt.figure(figsize=(6, 6))
    plt.scatter(np.log1p(y_true), np.log1p(y_pred), s=10, alpha=0.45, color="#0891b2")
    low = min(np.log1p(y_true).min(), np.log1p(y_pred).min())
    high = max(np.log1p(y_true).max(), np.log1p(y_pred).max())
    plt.plot([low, high], [low, high], color="#dc2626", linewidth=1.5)
    plt.title("Prediction vs Actual")
    plt.xlabel("Actual log1p(loss)")
    plt.ylabel("Predicted log1p(loss)")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return str(path)


def _correlation_heatmap(X: pd.DataFrame, path: Path) -> str:
    corr = X.corr().fillna(0)
    plt.figure(figsize=(12, 10))
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Correlation")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=6)
    plt.yticks(range(len(corr.columns)), corr.columns, fontsize=6)
    plt.title("Selected Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return str(path)


def _permutation_importance(model, X_test: pd.DataFrame, y_test: pd.Series, path: Path) -> str:
    result = permutation_importance(model, X_test, y_test, scoring="neg_root_mean_squared_error", n_repeats=5, random_state=42)
    imp = pd.Series(result.importances_mean, index=X_test.columns).sort_values(ascending=False).head(20)
    plt.figure(figsize=(9, 6))
    plt.barh(imp.index[::-1], imp.values[::-1], color="#16a34a")
    plt.title("Permutation Importance")
    plt.xlabel("Mean RMSE Degradation")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return str(path)


def _learning_curve(model, X: pd.DataFrame, y: pd.Series, path: Path) -> str:
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=3, scoring="neg_root_mean_squared_error", train_sizes=np.linspace(0.2, 1.0, 5), n_jobs=1
    )
    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, -train_scores.mean(axis=1), marker="o", label="Training RMSE")
    plt.plot(train_sizes, -val_scores.mean(axis=1), marker="o", label="Validation RMSE")
    plt.title("Learning Curve")
    plt.xlabel("Training Rows")
    plt.ylabel("RMSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return str(path)


def _partial_dependence(model, X: pd.DataFrame, path: Path) -> str:
    features = list(X.columns[: min(3, len(X.columns))])
    if not features:
        return ""
    fig, ax = plt.subplots(figsize=(10, 4))
    PartialDependenceDisplay.from_estimator(model, X.sample(min(1000, len(X)), random_state=42), features, ax=ax)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return str(path)
