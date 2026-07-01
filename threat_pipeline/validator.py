"""
threat_pipeline/validator.py
Pydantic schema validation: type checks and range constraints for raw TI records.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic schema
# ---------------------------------------------------------------------------

class ThreatRecord(BaseModel):
    """Single row of the raw threat-intelligence CSV after basic coercion."""

    # --- identifiers --------------------------------------------------------
    record_id: str
    asset_id: str
    source_feed: str

    # --- vulnerability fields -----------------------------------------------
    cve_id: Optional[str] = None
    cvss_score: Optional[float] = Field(default=None, ge=0.0, le=10.0)
    epss_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    severity: Optional[str] = None          # critical / high / medium / low / info
    kev_listed: bool = False

    # --- threat actor / campaign -------------------------------------------
    threat_actor: Optional[str] = None
    attack_vector: Optional[str] = None     # network / adjacent / local / physical
    attack_complexity: Optional[str] = None # low / high
    privileges_required: Optional[str] = None
    user_interaction: Optional[str] = None

    # --- temporal -----------------------------------------------------------
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    published_date: Optional[datetime] = None

    # --- impact flags -------------------------------------------------------
    confidentiality_impact: Optional[str] = None
    integrity_impact: Optional[str] = None
    availability_impact: Optional[str] = None

    # --- counts -------------------------------------------------------------
    exploit_count: int = Field(default=0, ge=0)
    affected_systems_count: int = Field(default=0, ge=0)

    @field_validator("severity")
    @classmethod
    def validate_severity(cls, v: Optional[str]) -> Optional[str]:
        allowed = {"critical", "high", "medium", "low", "info", None}
        if v is not None:
            v = v.strip().lower()
            if v not in allowed:
                raise ValueError(f"severity must be one of {allowed}, got '{v}'")
        return v

    @field_validator("attack_vector")
    @classmethod
    def validate_attack_vector(cls, v: Optional[str]) -> Optional[str]:
        allowed = {"network", "adjacent", "local", "physical", None}
        if v is not None:
            v = v.strip().lower()
            if v not in allowed:
                raise ValueError(f"attack_vector must be one of {allowed}, got '{v}'")
        return v

    @model_validator(mode="after")
    def validate_date_order(self) -> ThreatRecord:
        if self.first_seen and self.last_seen:
            if self.first_seen > self.last_seen:
                raise ValueError("first_seen cannot be after last_seen")
        return self


# ---------------------------------------------------------------------------
# Batch validation
# ---------------------------------------------------------------------------

class ThreatValidator:
    """Validates a DataFrame of raw TI records against ThreatRecord schema."""

    def __init__(self, strict: bool = False) -> None:
        self.strict = strict
        self._errors: list[dict] = []

    @property
    def errors(self) -> list[dict]:
        return self._errors

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and coerce the project CSV schema.

        The supplied threat_intelligence.csv is already feature-rich and does not
        use the older record_id/source_feed feed columns. This method normalizes
        compatible columns, adds aliases expected by downstream steps, and drops
        only rows that fail required identifier/range checks.
        """
        self._errors = []
        out = df.copy()

        if "record_id" not in out.columns:
            cve = out.get("cve_id", pd.Series("", index=out.index)).fillna("")
            out["record_id"] = out.get("asset_id", pd.Series("", index=out.index)).astype(str) + "::" + cve.astype(str)

        if "source_feed" not in out.columns:
            out["source_feed"] = out.get("source", pd.Series("unknown", index=out.index)).fillna("unknown")

        if "published_date" not in out.columns and "published" in out.columns:
            out["published_date"] = out["published"]

        if "first_seen" not in out.columns and "date_added" in out.columns:
            out["first_seen"] = out["date_added"]
            if "published" in out.columns:
                out["first_seen"] = out["first_seen"].fillna(out["published"])

        if "last_seen" not in out.columns:
            out["last_seen"] = pd.Timestamp.utcnow().isoformat()

        if "days_since_first_seen" not in out.columns and "days_since_published" in out.columns:
            out["days_since_first_seen"] = out["days_since_published"]

        if "days_active" not in out.columns:
            out["days_active"] = 0

        if "kev_listed" not in out.columns:
            if "date_added" in out.columns:
                out["kev_listed"] = out["date_added"].notna() & (out["date_added"].astype(str).str.len() > 0)
            elif "known_ransomware" in out.columns:
                out["kev_listed"] = out["known_ransomware"].fillna("").astype(str).str.len() > 0
            else:
                out["kev_listed"] = False

        if "affected_systems_count" not in out.columns:
            out["affected_systems_count"] = out.get("asset_exposure_score", pd.Series(1, index=out.index))

        if "attack_vector" not in out.columns:
            scope = out.get("scope", pd.Series("", index=out.index)).fillna("").astype(str).str.upper()
            out["attack_vector"] = scope.map(lambda value: "network" if value in {"GLOBAL", "EXTERNAL", "INTERNET"} else "local")

        for col in ["cvss_score", "epss_score", "exploit_count", "affected_systems_count"]:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")

        if "severity" in out.columns:
            severity = out["severity"].fillna("LOW").astype(str).str.strip().str.upper()
            severity = severity.replace({"INFORMATIONAL": "INFO", "NONE": "INFO"})
            out["severity"] = severity.str.lower()

        invalid_mask = pd.Series(False, index=out.index)
        if "asset_id" not in out.columns:
            raise ValueError("Required column missing: asset_id")
        invalid_mask |= out["asset_id"].isna() | (out["asset_id"].astype(str).str.len() == 0)

        if "cvss_score" in out.columns:
            invalid_mask |= out["cvss_score"].notna() & ~out["cvss_score"].between(0, 10)
        if "epss_score" in out.columns:
            invalid_mask |= out["epss_score"].notna() & ~out["epss_score"].between(0, 1)

        bad_indices = out.index[invalid_mask].tolist()
        if bad_indices:
            self._errors = [{"row_index": int(idx), "error": "required field/range validation failed"} for idx in bad_indices]
            logger.warning("%d rows failed validation and were excluded.", len(self._errors))
            if self.strict:
                raise ValueError(self._errors[0])
            out = out.loc[~invalid_mask].copy()

        logger.info("Validation accepted %d/%d rows.", len(out), len(df))
        return out.reset_index(drop=True)
