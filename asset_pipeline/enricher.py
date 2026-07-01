"""
asset_pipeline/enricher.py
==========================
Cybersecurity feature enrichment pipeline.

All derived features are computed EXCLUSIVELY from data present in the
validated asset record — no synthetic values, no random generation.

Derived features
----------------
Counting
  number_of_ports         : int   — len(ports)
  number_of_services      : int   — len(services)

Boolean flags  (all derived from services / tags / ports)
  has_web_service         : bool  — http or https in services
  has_database_service    : bool  — mysql, postgresql, mssql, oracle, mongodb…
  has_tls                 : bool  — 'tls' tag OR port 443 present
  has_remote_access_service: bool — ssh, rdp, telnet, vnc, winrm…
  is_public_asset         : bool  — internet_exposed == True

Scoring
  exposure_score          : float [0, 1]
      Weighted combination of internet_exposed + open ports + sensitive services.
  asset_risk_score        : float [0, 10]
      Multi-factor score: exposure + open ports + remote access + database
      + web services.  Higher = riskier.

Inference
  asset_type              : str
      Inferred from services and tags using a deterministic rule chain.

Part of: cyber_financial_loss_prediction / asset_pipeline (Part 2)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Service-to-category catalogues (deterministic, no randomness)
# ---------------------------------------------------------------------------

WEB_SERVICES: frozenset[str] = frozenset({
    "http", "https", "http-proxy", "http-alt", "webcache",
    "www", "www-http", "ssl/http", "ssl/https",
})

DATABASE_SERVICES: frozenset[str] = frozenset({
    "mysql", "postgresql", "postgres", "mssql", "ms-sql-s",
    "microsoft-sql-server", "oracle", "oracle-tns", "mongodb",
    "redis", "memcached", "cassandra", "couchdb", "elasticsearch",
    "ms-sql-m", "db2",
})

REMOTE_ACCESS_SERVICES: frozenset[str] = frozenset({
    "ssh", "rdp", "ms-wbt-server", "telnet", "vnc",
    "winrm", "wsman", "msrdp", "netsupport", "teamviewer",
    "x11",
})

FIREWALL_SERVICES: frozenset[str] = frozenset({
    "checkpoint-fw", "fortigate", "palo-alto-updates",
    "fw1-secureremote", "cisco-svcs",
})

MAIL_SERVICES: frozenset[str] = frozenset({
    "smtp", "smtps", "pop3", "pop3s", "imap", "imaps",
    "submission", "esmtp",
})

DNS_SERVICES: frozenset[str] = frozenset({"domain", "dns", "mdns"})

SMB_SERVICES: frozenset[str] = frozenset({
    "microsoft-ds", "netbios-ssn", "netbios-ns", "msrpc",
    "smb",
})

# Well-known high-risk ports for exposure scoring
HIGH_RISK_PORTS: frozenset[int] = frozenset({
    22,    # SSH
    23,    # Telnet
    3389,  # RDP
    5900,  # VNC
    1433,  # MSSQL
    3306,  # MySQL
    5432,  # PostgreSQL
    27017, # MongoDB
    6379,  # Redis
    9200,  # Elasticsearch
    445,   # SMB
    135,   # MSRPC
    139,   # NetBIOS
})

# ---------------------------------------------------------------------------
# Feature computation — pure functions (each takes a row dict or Series)
# ---------------------------------------------------------------------------


def _service_names(row: dict) -> frozenset[str]:
    """Extract a frozenset of normalised service names from the row."""
    services = row.get("services", [])
    names: set[str] = set()
    for svc in services:
        if isinstance(svc, dict):
            name = str(svc.get("service", "")).strip().lower()
            if name:
                names.add(name)
    return frozenset(names)


def _port_set(row: dict) -> frozenset[int]:
    """Return the set of open ports for this asset."""
    ports = row.get("ports", [])
    return frozenset(int(p) for p in ports if str(p).isdigit())


# -- Counting features -------------------------------------------------------


def compute_number_of_ports(row: dict) -> int:
    """Count of distinct open ports discovered."""
    return len(_port_set(row))


def compute_number_of_services(row: dict) -> int:
    """Count of distinct service entries."""
    services = row.get("services", [])
    return len(services) if isinstance(services, list) else 0


# -- Boolean flags -----------------------------------------------------------


def compute_has_web_service(row: dict) -> bool:
    """True when HTTP or HTTPS service is present, or web-related port (80/443/8080)."""
    svc = _service_names(row)
    ports = _port_set(row)
    return bool(svc & WEB_SERVICES) or bool(ports & {80, 443, 8080, 8443, 8000})


def compute_has_database_service(row: dict) -> bool:
    """True when any database service is present on any open port."""
    svc = _service_names(row)
    ports = _port_set(row)
    db_ports = {1433, 3306, 5432, 27017, 6379, 9200, 5984, 7474, 9042}
    return bool(svc & DATABASE_SERVICES) or bool(ports & db_ports)


def compute_has_tls(row: dict) -> bool:
    """
    True when TLS is evidenced by:
    - 'tls' in tags, OR
    - Port 443 is open, OR
    - https/ssl service detected.
    """
    tags = set(str(t).lower() for t in row.get("tags", []))
    ports = _port_set(row)
    svc = _service_names(row)
    tls_services = {"https", "ssl", "ssl/http", "ssl/https", "tls"}
    return (
        "tls" in tags
        or 443 in ports
        or bool(svc & tls_services)
    )


def compute_has_remote_access_service(row: dict) -> bool:
    """True when SSH, RDP, Telnet, VNC, or WinRM are in services or ports."""
    svc = _service_names(row)
    ports = _port_set(row)
    remote_ports = {22, 23, 3389, 5900, 5901, 5902, 5985, 5986}
    return bool(svc & REMOTE_ACCESS_SERVICES) or bool(ports & remote_ports)


def compute_is_public_asset(row: dict) -> bool:
    """Mirror of internet_exposed — explicit boolean for feature vector clarity."""
    v = row.get("internet_exposed", False)
    return bool(v)


# -- Exposure score ----------------------------------------------------------


def compute_exposure_score(row: dict) -> float:
    """
    Continuous exposure score in [0.0, 1.0].

    Components (each normalised to [0, 1], then weighted):
    ┌─────────────────────────────────┬────────┐
    │ Component                       │ Weight │
    ├─────────────────────────────────┼────────┤
    │ internet_exposed                │  0.40  │
    │ high-risk ports present         │  0.30  │
    │ remote access service present   │  0.15  │
    │ port count (capped at 20)       │  0.10  │
    │ database service present        │  0.05  │
    └─────────────────────────────────┴────────┘
    """
    ports = _port_set(row)

    w_exposed = 0.40 * (1.0 if row.get("internet_exposed") else 0.0)
    w_high_risk = 0.30 * (1.0 if ports & HIGH_RISK_PORTS else 0.0)
    w_remote = 0.15 * (1.0 if compute_has_remote_access_service(row) else 0.0)
    w_port_count = 0.10 * min(len(ports) / 20.0, 1.0)
    w_database = 0.05 * (1.0 if compute_has_database_service(row) else 0.0)

    score = w_exposed + w_high_risk + w_remote + w_port_count + w_database
    return round(min(score, 1.0), 4)


# -- Risk score --------------------------------------------------------------


def compute_asset_risk_score(row: dict) -> float:
    """
    Composite risk score in [0.0, 10.0].

    Built from weighted boolean + count signals:
    ┌──────────────────────────────────────────────────┬───────┐
    │ Signal                                           │ Score │
    ├──────────────────────────────────────────────────┼───────┤
    │ internet_exposed                                 │  3.0  │
    │ has_remote_access_service                        │  2.5  │
    │ has_database_service                             │  2.0  │
    │ high-risk port present                           │  1.5  │
    │ has_web_service                                  │  0.5  │
    │ port_count > 5                                   │  0.3  │
    │ port_count > 10                                  │  0.2  │
    └──────────────────────────────────────────────────┴───────┘
    Total possible: 10.0 (capped)
    """
    ports = _port_set(row)
    port_count = len(ports)

    score = 0.0
    if row.get("internet_exposed"):
        score += 3.0
    if compute_has_remote_access_service(row):
        score += 2.5
    if compute_has_database_service(row):
        score += 2.0
    if ports & HIGH_RISK_PORTS:
        score += 1.5
    if compute_has_web_service(row):
        score += 0.5
    if port_count > 5:
        score += 0.3
    if port_count > 10:
        score += 0.2

    return round(min(score, 10.0), 2)


# -- Asset type inference ----------------------------------------------------


def infer_asset_type(row: dict) -> str:
    """
    Infer a human-readable asset type using a deterministic rule chain.

    Rules are evaluated in priority order; the first match wins.

    Priority chain
    ~~~~~~~~~~~~~~
    1. Firewall tags / services
    2. RDP / Windows-specific signals → Windows Server
    3. SSH (no RDP, no web) → Linux Server
    4. Database services → Database Server
    5. HTTP/HTTPS (with domain) → Web Server
    6. HTTP/HTTPS (no domain, private IP) → Internal Web Service
    7. SMTP / mail services → Mail Server
    8. DNS services → DNS Server
    9. SMB / NetBIOS → Windows File Server
    10. Domain-only asset (crt.sh) → Domain Asset
    11. IP-only, no services → Network Host
    12. Fallback → Unknown
    """
    svc = _service_names(row)
    ports = _port_set(row)
    tags = set(str(t).lower() for t in row.get("tags", []))
    has_domain = bool(row.get("domain", "").strip())
    has_host = bool(row.get("host", "").strip())
    discovered_by = [str(d).lower() for d in row.get("discovered_by", [])]

    # 1. Firewall
    if "firewall" in tags or bool(svc & FIREWALL_SERVICES):
        return "Firewall"

    # 2. Windows Server (RDP or SMB-heavy)
    if compute_has_remote_access_service(row) and (
        bool(svc & {"rdp", "ms-wbt-server", "microsoft-ds"})
        or bool(ports & {3389})
    ):
        return "Windows Server"

    # 3. Linux Server (SSH, no RDP)
    if (
        "ssh" in svc or 22 in ports
    ) and not (bool(svc & {"rdp", "ms-wbt-server"}) or 3389 in ports):
        return "Linux Server"

    # 4. Database Server
    if compute_has_database_service(row):
        return "Database Server"

    # 5. Windows File Server (SMB without clear OS from above)
    if bool(svc & SMB_SERVICES):
        return "Windows File Server"

    # 6. Web Server
    if compute_has_web_service(row):
        return "Web Server"

    # 7. Mail Server
    if bool(svc & MAIL_SERVICES) or bool(ports & {25, 465, 587, 110, 995, 143, 993}):
        return "Mail Server"

    # 8. DNS Server
    if bool(svc & DNS_SERVICES) or 53 in ports:
        return "DNS Server"

    # 9. Domain-only asset (discovered by crt.sh, no host or no ports)
    if has_domain and not has_host and "crtsh" in discovered_by:
        return "Domain Asset"

    # 10. Domain + IP but no open services (crt.sh with resolved IP)
    if has_domain and has_host and not ports:
        return "Domain Asset"

    # 11. IP-only host, no services
    if has_host and not has_domain and not ports:
        return "Network Host"

    # 12. Fallback
    return "Unknown"


# ---------------------------------------------------------------------------
# Row-level enrichment dispatcher
# ---------------------------------------------------------------------------


def enrich_row(row: dict) -> dict:
    """
    Compute all derived features for a single asset row dict.

    Parameters
    ----------
    row : dict
        A single validated asset record (as returned by model_dump).

    Returns
    -------
    dict
        The original row dict augmented with all derived feature keys.
    """
    enriched = dict(row)

    enriched["number_of_ports"] = compute_number_of_ports(row)
    enriched["number_of_services"] = compute_number_of_services(row)
    enriched["has_web_service"] = compute_has_web_service(row)
    enriched["has_database_service"] = compute_has_database_service(row)
    enriched["has_tls"] = compute_has_tls(row)
    enriched["has_remote_access_service"] = compute_has_remote_access_service(row)
    enriched["is_public_asset"] = compute_is_public_asset(row)
    enriched["exposure_score"] = compute_exposure_score(row)
    enriched["asset_risk_score"] = compute_asset_risk_score(row)
    enriched["asset_type"] = infer_asset_type(row)

    return enriched


# ---------------------------------------------------------------------------
# DataFrame-level enrichment
# ---------------------------------------------------------------------------


def enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply enrichment to every row of a validated asset DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Clean, validated DataFrame from :func:`~asset_pipeline.loader.load_assets`.

    Returns
    -------
    pd.DataFrame
        Enriched DataFrame with all derived feature columns appended.
    """
    log.info("Enriching %d assets …", len(df))

    records = df.to_dict(orient="records")
    enriched_records = [enrich_row(r) for r in records]
    result = pd.DataFrame(enriched_records)

    # Report derived feature distributions
    bool_features = [
        "has_web_service", "has_database_service", "has_tls",
        "has_remote_access_service", "is_public_asset",
    ]
    for feat in bool_features:
        if feat in result.columns:
            count = result[feat].sum()
            log.info("  %-30s %d / %d  (%.1f%%)", feat, count, len(result),
                     100 * count / len(result))

    log.info("Enrichment complete. Total columns: %d", len(result.columns))
    return result


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    """Save DataFrame as Parquet, serialising list/dict columns to JSON strings."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    export = df.copy()
    for col in ("ports", "tags", "discovered_by", "services"):
        if col in export.columns:
            export[col] = export[col].apply(json.dumps, default=str)
    log.info("Saving Parquet → %s", path)
    export.to_parquet(path, index=False)
    log.info("Parquet saved: %d rows × %d cols.", len(export), len(export.columns))


def save_csv(df: pd.DataFrame, path: Path) -> None:
    """Save DataFrame as CSV, serialising list/dict columns to JSON strings."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    export = df.copy()
    for col in ("ports", "tags", "discovered_by", "services"):
        if col in export.columns:
            export[col] = export[col].apply(json.dumps, default=str)
    log.info("Saving CSV → %s", path)
    export.to_csv(path, index=False)
    log.info("CSV saved: %d rows × %d cols.", len(export), len(export.columns))


def build_summary(
    df: pd.DataFrame,
    rejections: list[dict],
) -> dict[str, Any]:
    """
    Build a JSON-serialisable summary of the enriched asset inventory.

    Parameters
    ----------
    df : pd.DataFrame
        Enriched DataFrame.
    rejections : list[dict]
        Rejection records from schema validation.

    Returns
    -------
    dict[str, Any]
    """
    total = len(df)

    def _vc(col: str) -> dict:
        return df[col].value_counts().to_dict() if col in df.columns else {}

    summary: dict[str, Any] = {
        "pipeline_metadata": {
            "total_records_processed": total + len(rejections),
            "accepted": total,
            "rejected": len(rejections),
            "pass_rate_pct": round(100 * total / max(1, total + len(rejections)), 2),
        },
        "asset_type_distribution": _vc("asset_type"),
        "discovery_source": {},
        "internet_exposure": {
            "exposed": int(df["is_public_asset"].sum()),
            "internal": int((~df["is_public_asset"]).sum()),
            "exposure_pct": round(100 * df["is_public_asset"].mean(), 2),
        },
        "service_flags": {
            "has_web_service": int(df["has_web_service"].sum()),
            "has_database_service": int(df["has_database_service"].sum()),
            "has_tls": int(df["has_tls"].sum()),
            "has_remote_access_service": int(df["has_remote_access_service"].sum()),
        },
        "port_statistics": {
            "assets_with_open_ports": int((df["number_of_ports"] > 0).sum()),
            "max_ports_on_single_asset": int(df["number_of_ports"].max()),
            "mean_ports": round(float(df["number_of_ports"].mean()), 3),
        },
        "risk_score_distribution": {
            "mean": round(float(df["asset_risk_score"].mean()), 3),
            "median": round(float(df["asset_risk_score"].median()), 3),
            "max": round(float(df["asset_risk_score"].max()), 3),
            "high_risk_count": int((df["asset_risk_score"] >= 5.0).sum()),
        },
        "exposure_score_distribution": {
            "mean": round(float(df["exposure_score"].mean()), 4),
            "max": round(float(df["exposure_score"].max()), 4),
        },
        "rejected_records": rejections,
    }

    # discovered_by distribution (list column — need to explode)
    if "discovered_by" in df.columns:
        exploded = df["discovered_by"].explode().dropna()
        summary["discovery_source"] = exploded.value_counts().to_dict()

    return summary


def save_summary(summary: dict[str, Any], path: Path) -> None:
    """Write the summary dictionary to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Saving summary JSON → %s", path)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    log.info("Summary JSON saved.")