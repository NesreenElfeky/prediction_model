from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE, VarianceThreshold, mutual_info_regression


def remove_leakage(df: pd.DataFrame, leakage_columns: list[str], id_column: str) -> pd.DataFrame:
    return df.drop(columns=[id_column, *leakage_columns], errors="ignore")


def select_features(
    X: pd.DataFrame,
    y: pd.Series,
    correlation_threshold: float,
    variance_threshold: float,
    max_features: int,
    random_state: int,
    output_path: str | Path,
) -> tuple[pd.DataFrame, dict]:
    numeric = X.select_dtypes(include="number").copy()
    numeric = numeric.replace([np.inf, -np.inf], np.nan).fillna(numeric.median(numeric_only=True))

    variance = VarianceThreshold(threshold=variance_threshold)
    vt_values = variance.fit_transform(numeric)
    vt_features = list(numeric.columns[variance.get_support()])
    vt_df = pd.DataFrame(vt_values, columns=vt_features, index=numeric.index)

    corr = vt_df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    corr_drop = [col for col in upper.columns if any(upper[col] > correlation_threshold)]
    corr_df = vt_df.drop(columns=corr_drop)

    mi = mutual_info_regression(corr_df, y, random_state=random_state)
    mi_scores = pd.Series(mi, index=corr_df.columns).sort_values(ascending=False)

    rf = RandomForestRegressor(n_estimators=90, min_samples_leaf=2, random_state=random_state, n_jobs=-1)
    rf.fit(corr_df, np.log1p(y))
    rf_scores = pd.Series(rf.feature_importances_, index=corr_df.columns).sort_values(ascending=False)

    rfe_feature_count = min(max_features, max(5, len(corr_df.columns) // 2), len(corr_df.columns))
    rfe = RFE(
        estimator=RandomForestRegressor(n_estimators=45, min_samples_leaf=3, random_state=random_state, n_jobs=-1),
        n_features_to_select=rfe_feature_count,
        step=0.2,
    )
    rfe.fit(corr_df, np.log1p(y))
    rfe_features = list(corr_df.columns[rfe.support_])

    shap_sample = corr_df.sample(min(250, len(corr_df)), random_state=random_state)
    try:
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(shap_sample, check_additivity=False)
        shap_scores = pd.Series(np.abs(shap_values).mean(axis=0), index=shap_sample.columns).sort_values(ascending=False)
    except Exception:
        shap_scores = rf_scores.copy()

    rank_table = pd.DataFrame(index=corr_df.columns)
    rank_table["mutual_information"] = mi_scores
    rank_table["random_forest_importance"] = rf_scores
    rank_table["shap_importance"] = shap_scores
    rank_table["rfe_selected"] = rank_table.index.isin(rfe_features).astype(int)
    for col in ["mutual_information", "random_forest_importance", "shap_importance"]:
        denom = rank_table[col].max()
        rank_table[col] = rank_table[col] / denom if denom and pd.notna(denom) else 0
    rank_table["selection_score"] = (
        0.3 * rank_table["mutual_information"]
        + 0.3 * rank_table["random_forest_importance"]
        + 0.25 * rank_table["shap_importance"]
        + 0.15 * rank_table["rfe_selected"]
    )
    selected = list(rank_table.sort_values("selection_score", ascending=False).head(max_features).index)
    report = {
        "initial_numeric_features": int(len(numeric.columns)),
        "after_variance_threshold": int(len(vt_features)),
        "correlation_dropped": corr_drop,
        "after_correlation_filter": int(len(corr_df.columns)),
        "selected_features": selected,
        "feature_scores": rank_table.sort_values("selection_score", ascending=False).reset_index(names="feature").to_dict("records"),
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return X[selected].copy(), report
