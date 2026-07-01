from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_EXPLANATIONS = {
    "cvss_severity_score": "Maximum and mean CVSS summarize exploit technical severity.",
    "asset_criticality_encoding": "Asset exposure and affected-system counts proxy asset operational importance.",
    "kev_feature_strength": "Known exploited vulnerability count and flags capture real-world exploitation evidence.",
    "exploit_activity_score": "Exploit count weighted by EPSS measures active exploit availability and likelihood.",
    "exposure_score": "Network exposure flags and affected systems approximate external attack surface.",
    "threat_density": "Vulnerability count per unique CVE indicates concentration of threats on the asset.",
    "attack_surface_score": "Open exposure, severity, and vulnerability count combine into attack surface pressure.",
    "business_risk_score": "Business impact fields from upstream risk generation summarize business-facing cyber risk.",
    "composite_vulnerability_score": "CVSS, EPSS, exploit interaction, and critical counts produce a technical risk composite.",
    "threat_frequency": "Vulnerability observations per asset estimate recurring threat frequency.",
    "cvss_asset_criticality_interaction": "High severity on important assets should raise potential financial impact.",
    "epss_exploit_count_interaction": "Exploit probability multiplied by exploit counts captures exploitation momentum.",
    "attack_vector_encoding": "Network-exposed assets carry higher remote compromise risk.",
    "vendor_product_risk": "Upstream aggregation can proxy vendor/product concentration via CVE and vulnerability density.",
    "vulnerability_age": "Older unresolved vulnerabilities can increase accumulated exposure.",
    "risk_percentile": "Ranks assets relative to portfolio risk for model-friendly ordinal risk signal.",
}


def _safe_divide(numerator, denominator):
    return numerator / denominator.replace(0, np.nan)


def _nonnegative(series: pd.Series) -> pd.Series:
    return series.fillna(0).clip(lower=0)


def build_cyber_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    idx = out.index
    cvss_max = _nonnegative(out.get("cvss_score_max", pd.Series(0, index=idx)))
    cvss_mean = _nonnegative(out.get("cvss_score_mean", pd.Series(0, index=idx)))
    epss_max = _nonnegative(out.get("epss_score_max", pd.Series(0, index=idx)))
    epss_mean = _nonnegative(out.get("epss_score_mean", pd.Series(0, index=idx)))
    vuln_count = _nonnegative(out.get("vuln_count", pd.Series(0, index=idx)))
    unique_cves = _nonnegative(out.get("unique_cve_count", pd.Series(0, index=idx)))
    exploit_sum = _nonnegative(out.get("exploit_count_sum", pd.Series(0, index=idx)))
    affected_sum = _nonnegative(out.get("affected_systems_count_sum", pd.Series(0, index=idx)))
    network = _nonnegative(out.get("is_network_exposed_max", pd.Series(0, index=idx)))
    kev_sum = _nonnegative(out.get("kev_flag_sum", pd.Series(0, index=idx)))
    critical = _nonnegative(out.get("sev_count_critical", pd.Series(0, index=idx)))
    high = _nonnegative(out.get("sev_count_high", pd.Series(0, index=idx)))
    age = _nonnegative(out.get("days_since_published_max", pd.Series(0, index=idx)))

    out["cvss_severity_score"] = (0.7 * cvss_max + 0.3 * cvss_mean).clip(0, 10)
    out["asset_criticality_encoding"] = np.log1p(affected_sum)
    out["kev_feature_strength"] = np.log1p(kev_sum) + out.get("kev_flag_max", pd.Series(0, index=idx)).fillna(0)
    out["exploit_activity_score"] = np.log1p(exploit_sum) * (1 + epss_max)
    out["exposure_score"] = network + np.log1p(affected_sum)
    out["threat_density"] = _safe_divide(vuln_count, unique_cves).fillna(0)
    out["attack_surface_score"] = np.log1p(vuln_count) * (1 + network) * (1 + cvss_max / 10)
    out["business_risk_score"] = out.get("composite_risk_score_max", pd.Series(0, index=idx)).fillna(0) * np.log1p(affected_sum)
    out["composite_vulnerability_score"] = (
        0.35 * cvss_max
        + 3.0 * epss_max
        + 0.2 * out.get("epss_cvss_interaction_max", pd.Series(0, index=idx)).fillna(0)
        + 0.15 * critical
    )
    out["threat_frequency"] = np.log1p(vuln_count)
    out["cvss_asset_criticality_interaction"] = cvss_max * np.log1p(affected_sum)
    out["epss_exploit_count_interaction"] = epss_max * np.log1p(exploit_sum)
    out["attack_vector_encoding"] = network
    out["vendor_product_risk"] = np.log1p(unique_cves) * (1 + high + critical)
    out["mean_cvss_per_asset"] = cvss_mean
    out["maximum_cvss"] = cvss_max
    out["average_epss"] = epss_mean
    out["recent_threat_score"] = out.get("composite_risk_score_max", pd.Series(0, index=idx)).fillna(0) / (1 + np.log1p(age))
    out["business_exposure_index"] = out["asset_criticality_encoding"] * (1 + network)
    out["weighted_risk_index"] = out["composite_vulnerability_score"] * (1 + out["kev_feature_strength"])
    out["asset_importance_score"] = np.log1p(affected_sum + vuln_count)
    out["threat_intelligence_confidence"] = out.get("composite_risk_score_mean", pd.Series(0, index=idx)).fillna(0) * (
        1 + out.get("epss_cvss_interaction_mean", pd.Series(0, index=idx)).fillna(0)
    )
    out["vulnerability_age"] = age
    out["exploit_availability"] = ((exploit_sum > 0) | (kev_sum > 0)).astype(int)
    out["risk_percentile"] = out["weighted_risk_index"].rank(pct=True)
    return out.replace([np.inf, -np.inf], np.nan)
