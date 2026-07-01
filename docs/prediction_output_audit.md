# Prediction Output Quality Audit

## Scope

This audit covers the production prediction path from CSV/API input through `prediction.Predictor` to `outputs/predictions.csv`. The trained Gradient Boosting model, training pipeline, and dataset were not retrained or regenerated.

## Exported Feature Classes

`predictions.csv` now labels feature columns by source and user-readability:

| Export prefix | Meaning |
| --- | --- |
| `raw_feature__` | User-friendly raw or direct business/count value. CVSS model inputs are stored as 0-1 values, so export converts them back to the familiar 0-10 CVSS scale. |
| `derived_feature__` | Engineered model feature with business meaning and no single raw equivalent. |
| `scaled_feature__` | Model-ready normalized/standardized upstream value. These are preserved only where the saved model requires them and no inverse raw value exists in the artifacts. |

Raw exports:

- `raw_feature__epss_score_max`
- `raw_feature__epss_score_mean`
- `raw_feature__cvss_score_mean_0_to_10`
- `raw_feature__vulnerability_count`
- `raw_feature__exploitable_vulnerability_count`
- `raw_feature__has_exploitable_vulnerability`
- `raw_feature__cvss_score_max_0_to_10`
- `raw_feature__critical_vulnerability_count`
- `raw_feature__has_critical_vulnerability`
- `raw_feature__has_known_exploited_vulnerability`
- `raw_feature__known_exploited_vulnerability_count`
- `raw_feature__low_severity_vulnerability_count`

Derived exports:

- `derived_feature__weighted_risk_index`
- `derived_feature__composite_vulnerability_score`
- `derived_feature__attack_surface_score`
- `derived_feature__recent_threat_score`
- `derived_feature__vulnerability_age`
- `derived_feature__epss_exploit_count_interaction`

Scaled/model-ready exports:

- `scaled_feature__exploit_count_sum`
- `scaled_feature__days_since_published_max`
- `scaled_feature__days_since_published_min`
- `scaled_feature__risk_exploit_product_max`
- `scaled_feature__risk_exploit_product_mean`

These scaled columns are explicitly named because the saved feature schema contains upstream normalized values and the raw inverse values are not available in the production artifacts.

## Composite Vulnerability Score

The training feature engineering code defines:

```text
composite_vulnerability_score =
    0.35 * cvss_score_max
  + 3.00 * epss_score_max
  + 0.20 * epss_cvss_interaction_max
  + 0.15 * sev_count_critical
```

Meaning: an overall vulnerability severity score combining technical severity, exploit likelihood, the interaction between the two, and critical vulnerability count.

`weighted_risk_index` is defined separately:

```text
weighted_risk_index =
    composite_vulnerability_score * (1 + kev_feature_strength)

kev_feature_strength = log1p(kev_flag_sum) + kev_flag_max
```

Meaning: the vulnerability score re-weighted by known exploited vulnerability evidence. The audit found these two columns are not mathematically identical:

- Correlation: `0.8603895812442393`
- Max absolute difference: `31.17688928002487`
- Decision: both remain because removing either would break the saved trained model schema and would require retraining.

## Confidence Score

The old confidence implementation could collapse to nearly constant values. The current implementation is row-specific and uses:

1. Gradient Boosting staged prediction standard deviation on the transformed feature matrix.
2. Holdout MAPE as a minimum uncertainty floor.
3. Robust distance from the training feature distribution using median, IQR, and p01/p99 tail penalties.
4. Prediction interval width as an additional confidence penalty.

Formula summary:

```text
uncertainty =
    max(staged_prediction_std, predicted_loss * holdout_mape)
    * (1 + training_distribution_distance / 2.5)

confidence =
    100
    * exp(-uncertainty / max(predicted_loss, holdout_mae))
    * exp(-0.08 * interval_width_ratio)
```

The final score is clipped to `[0, 100]`. This is statistically meaningful because it combines model disagreement, empirical holdout error, and distance from the training distribution.

## Validation Results

Synthetic audit cases were generated from real model-ready feature distributions without retraining. Results:

| Case | Risk Level | Confidence |
| --- | --- | ---: |
| Very Low Risk | Very Low | 96.00 |
| Low Risk | Low | 94.59 |
| Medium Risk | Medium | 91.88 |
| High Risk | High | 88.15 |
| Critical Risk | Critical | 86.31 |

An additional extreme outlier-style case produced confidence near `51.68`, confirming that unknown patterns receive lower confidence.

## Output Contract

`outputs/predictions.csv` contains:

- Clearly labeled raw, derived, and scaled feature columns.
- `predicted_loss`
- `currency`
- `risk_level`
- `confidence_score`
- `prediction_interval_lower`
- `prediction_interval_upper`
- `top_feature_1`
- `top_feature_2`
- `top_feature_3`
- `timestamp`

Risk levels are derived only from `predicted_loss` using `config/risk_thresholds.yaml`.
