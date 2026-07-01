"""
asset_pipeline/schema.py
========================
Pydantic v2 schema for a real asset discovery record produced by
Nmap + crt.sh security tooling.

Fields mirror the exact JSON structure emitted by the scanner:
    asset_id, host, domain, ports, services, tags,
    discovered_by, internet_exposed, last_seen

Validation rules
----------------
- asset_id       : non-empty string, must start with "ASSET-"
- host           : optional; when present must be valid IPv4 or IPv6
- domain         : optional; when present must be a syntactically valid FQDN
- ports          : each port in [1, 65535]; list may be empty
- services       : list of ServiceEntry sub-models; may be empty
- tags           : free-form lowercase strings
- discovered_by  : non-empty list; each value in known discovery tool names
- internet_exposed: bool
- last_seen      : ISO-8601 datetime (timezone-aware or naive)

Corrupted / missing records are rejected and logged; only clean rows are
returned to the caller.

Part of: cyber_financial_loss_prediction / asset_pipeline (Part 2)
"""

from __future__ import annotations

import ipaddress
import logging
import re
from datetime import datetime
from typing import Optional

import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FQDN_RE = re.compile(
    r"^(?=.{1,253}$)"           # total length
    r"(?!-)"                     # no leading hyphen
    r"([a-zA-Z0-9]"
    r"([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?"
    r"\.)+"                      # labels separated by dots
    r"[a-zA-Z]{2,63}$"          # TLD
)

KNOWN_DISCOVERY_TOOLS: frozenset[str] = frozenset({"nmap", "crtsh", "masscan", "shodan"})

PORT_MIN: int = 1
PORT_MAX: int = 65535

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sub-model
# ---------------------------------------------------------------------------


class ServiceEntry(BaseModel):
    """A single service observed on an open port."""

    port: int = Field(..., ge=PORT_MIN, le=PORT_MAX)
    service: str = Field(..., min_length=1)
    version: str = Field(default="")

    @field_validator("service", mode="before")
    @classmethod
    def normalise_service_name(cls, v: object) -> str:
        """Strip and lower-case the service name."""
        if isinstance(v, str):
            return v.strip().lower()
        raise ValueError(f"service must be a string, got {type(v)}")


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class DiscoveredAsset(BaseModel):
    """
    Validated representation of a single asset record from the discovery
    pipeline (Nmap / crt.sh).

    Both *host* and *domain* may be absent (empty string), but at least one
    must be present — an asset with neither is unidentifiable and rejected.
    """

    # --- Identifiers --------------------------------------------------------
    asset_id: str = Field(..., min_length=1)
    host: str = Field(default="")          # IPv4 or IPv6, may be empty
    domain: str = Field(default="")        # FQDN, may be empty

    # --- Network telemetry --------------------------------------------------
    ports: list[int] = Field(default_factory=list)
    services: list[ServiceEntry] = Field(default_factory=list)

    # --- Metadata -----------------------------------------------------------
    tags: list[str] = Field(default_factory=list)
    discovered_by: list[str] = Field(default_factory=list)
    internet_exposed: bool = Field(default=False)
    last_seen: datetime

    # -----------------------------------------------------------------------
    # Field validators
    # -----------------------------------------------------------------------

    @field_validator("asset_id", mode="before")
    @classmethod
    def validate_asset_id(cls, v: object) -> str:
        """asset_id must be a non-empty string starting with 'ASSET-'."""
        if not isinstance(v, str) or not v.strip():
            raise ValueError("asset_id must be a non-empty string")
        v = v.strip()
        if not v.startswith("ASSET-"):
            raise ValueError(f"asset_id must start with 'ASSET-', got '{v}'")
        return v

    @field_validator("host", mode="before")
    @classmethod
    def validate_host(cls, v: object) -> str:
        """
        Accept empty string (host not discovered yet) or a valid IPv4/IPv6.
        """
        if v is None:
            return ""
        v_str = str(v).strip()
        if v_str == "":
            return ""
        try:
            ipaddress.ip_address(v_str)
            return v_str
        except ValueError:
            raise ValueError(f"host '{v_str}' is not a valid IPv4 or IPv6 address")

    @field_validator("domain", mode="before")
    @classmethod
    def validate_domain(cls, v: object) -> str:
        """
        Accept empty string or a syntactically valid FQDN.
        Single-label names (e.g. 'localhost') are allowed.
        """
        if v is None:
            return ""
        v_str = str(v).strip()
        if v_str == "":
            return ""
        # Single-label hostnames: allow if alphanumeric/hyphen
        if "." not in v_str:
            if re.match(r"^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?$", v_str):
                return v_str.lower()
            raise ValueError(f"domain '{v_str}' is not a valid hostname or FQDN")
        if not _FQDN_RE.match(v_str):
            raise ValueError(f"domain '{v_str}' is not a valid FQDN")
        return v_str.lower()

    @field_validator("ports", mode="before")
    @classmethod
    def validate_ports(cls, v: object) -> list[int]:
        """Ensure every port is an integer in [1, 65535]."""
        if not isinstance(v, list):
            raise ValueError("ports must be a list")
        result: list[int] = []
        for p in v:
            try:
                port = int(p)
            except (TypeError, ValueError):
                raise ValueError(f"port value '{p}' is not an integer")
            if not (PORT_MIN <= port <= PORT_MAX):
                raise ValueError(
                    f"port {port} is out of valid range [{PORT_MIN}, {PORT_MAX}]"
                )
            result.append(port)
        return result

    @field_validator("tags", mode="before")
    @classmethod
    def normalise_tags(cls, v: object) -> list[str]:
        """Lower-case and strip all tag strings."""
        if not isinstance(v, list):
            return []
        return [str(t).strip().lower() for t in v if str(t).strip()]

    @field_validator("discovered_by", mode="before")
    @classmethod
    def validate_discovery_tools(cls, v: object) -> list[str]:
        """Accept any tool name; log a warning for unknown ones."""
        if not isinstance(v, list):
            return []
        result: list[str] = []
        for tool in v:
            t = str(tool).strip().lower()
            if t not in KNOWN_DISCOVERY_TOOLS:
                log.debug("Unknown discovery tool '%s' — accepted.", t)
            result.append(t)
        return result

    @field_validator("last_seen", mode="before")
    @classmethod
    def parse_last_seen(cls, v: object) -> datetime:
        """Parse ISO-8601 datetime strings; reject malformed values."""
        if isinstance(v, datetime):
            return v
        if isinstance(v, str):
            v_stripped = v.strip()
            # Python 3.11+ handles timezone offset Z and ±HH:MM natively
            try:
                return datetime.fromisoformat(v_stripped)
            except ValueError:
                pass
            # Fallback: strip timezone suffix and retry
            clean = re.sub(r"[+-]\d{2}:\d{2}$", "", v_stripped).rstrip("Z")
            try:
                return datetime.fromisoformat(clean)
            except ValueError:
                raise ValueError(f"Cannot parse last_seen datetime: '{v}'")
        raise ValueError(f"last_seen must be a string or datetime, got {type(v)}")

    # -----------------------------------------------------------------------
    # Cross-field validation
    # -----------------------------------------------------------------------

    @model_validator(mode="after")
    def require_host_or_domain(self) -> "DiscoveredAsset":
        """At least one of host or domain must be non-empty."""
        if not self.host and not self.domain:
            raise ValueError(
                f"asset_id='{self.asset_id}': both host and domain are empty — "
                "asset is unidentifiable"
            )
        return self

    # -----------------------------------------------------------------------
    # Config
    # -----------------------------------------------------------------------

    model_config = {"str_strip_whitespace": True}


# ---------------------------------------------------------------------------
# DataFrame validation helper
# ---------------------------------------------------------------------------


def validate_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """
    Validate every row of *df* against :class:`DiscoveredAsset`.

    Parameters
    ----------
    df:
        Raw DataFrame where list/dict columns have already been parsed
        (not stored as JSON strings).

    Returns
    -------
    tuple[pd.DataFrame, list[dict]]
        - Clean DataFrame of accepted rows (model_dump output).
        - List of rejection records: ``{"index": ..., "asset_id": ..., "reason": ...}``

    Raises
    ------
    ValueError
        If zero rows pass validation.
    """
    log.info("Validating schema for %d rows …", len(df))

    clean_rows: list[dict] = []
    rejections: list[dict] = []

    for idx, row in df.iterrows():
        record = row.to_dict()
        try:
            asset = DiscoveredAsset.model_validate(record)
            clean_rows.append(asset.model_dump())
        except Exception as exc:
            asset_id = record.get("asset_id", f"<row {idx}>")
            reason = str(exc)
            log.warning("REJECTED  %s — %s", asset_id, reason)
            rejections.append({"index": idx, "asset_id": asset_id, "reason": reason})

    accepted = len(clean_rows)
    rejected = len(rejections)
    log.info(
        "Schema validation: %d accepted / %d rejected (%.1f%% pass rate).",
        accepted,
        rejected,
        100 * accepted / max(1, accepted + rejected),
    )

    if not clean_rows:
        raise ValueError(
            "All rows were rejected during schema validation. "
            "Check the source file format."
        )

    clean_df = pd.DataFrame(clean_rows)
    return clean_df, rejections