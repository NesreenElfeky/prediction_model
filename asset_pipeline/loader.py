"""
asset_pipeline/loader.py
========================
Production-ready multi-format asset loader.

Supports
--------
- JSON  (primary format — output of Nmap + crt.sh scanner)
- CSV   (future: flat exports from CMDB or spreadsheets)
- Parquet (future: data-warehouse / pipeline hand-offs)

The loader:
1. Detects or accepts an explicit file format.
2. Reads the file with the appropriate pandas reader.
3. Handles the JSON-specific challenge of list/dict columns
   (``ports``, ``services``, ``tags``, ``discovered_by``) that arrive
   as Python objects from ``pd.read_json`` but as serialised strings
   from CSV/Parquet round-trips.
4. Normalises column types before Pydantic validation.
5. Returns a clean DataFrame (accepted rows only) plus a rejection log.

Part of: cyber_financial_loss_prediction / asset_pipeline (Part 2)
"""

from __future__ import annotations

import ast
import json
import logging
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd

from asset_pipeline.schema import validate_dataframe

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported formats
# ---------------------------------------------------------------------------


class FileFormat(str, Enum):
    JSON = "json"
    CSV = "csv"
    PARQUET = "parquet"


# ---------------------------------------------------------------------------
# Helpers — column normalisation
# ---------------------------------------------------------------------------

_LIST_COLUMNS: tuple[str, ...] = ("ports", "tags", "discovered_by")
_NESTED_LIST_COLUMNS: tuple[str, ...] = ("services",)  # list-of-dicts


def _coerce_list_column(series: pd.Series, col: str) -> pd.Series:
    """
    Ensure every cell in *series* is a Python ``list``.

    Handles:
    - Already a list (JSON load)
    - JSON string ``"[1, 2]"``
    - Python repr string ``"[1, 2]"`` (CSV round-trip)
    - ``None`` / NaN → empty list
    """

    def _parse(v: Any) -> list:
        if isinstance(v, list):
            return v
        if v is None or (isinstance(v, float)):
            return []
        if isinstance(v, str):
            v = v.strip()
            if v == "" or v == "[]":
                return []
            # Try JSON first (handles double-quoted strings)
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
            except (json.JSONDecodeError, ValueError):
                pass
            # Fallback: Python literal eval
            try:
                parsed = ast.literal_eval(v)
                if isinstance(parsed, list):
                    return parsed
            except (ValueError, SyntaxError):
                pass
            log.warning("Could not parse list column '%s' value: %r", col, v)
            return []
        return []

    return series.apply(_parse)


def _coerce_nested_list_column(series: pd.Series, col: str) -> pd.Series:
    """
    Ensure every cell is a ``list[dict]``.

    Handles the same representations as :func:`_coerce_list_column` but
    validates that each element is a dict (service entry).
    """

    def _parse(v: Any) -> list[dict]:
        raw: list = _coerce_list_column(pd.Series([v]), col).iloc[0]
        result: list[dict] = []
        for item in raw:
            if isinstance(item, dict):
                result.append(item)
            else:
                log.debug("Skipping non-dict service entry: %r", item)
        return result

    return series.apply(_parse)


def _normalise_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce DataFrame column types before Pydantic validation.

    - List columns: ``ports``, ``tags``, ``discovered_by``
    - Nested list-of-dict columns: ``services``
    - ``internet_exposed`` → bool
    - ``last_seen`` → kept as string (Pydantic validator parses it)
    - ``host``, ``domain``, ``asset_id`` → str, NaN → ""
    """
    df = df.copy()

    # String columns — replace NaN with empty string
    for col in ("asset_id", "host", "domain"):
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()

    # List columns
    for col in _LIST_COLUMNS:
        if col in df.columns:
            df[col] = _coerce_list_column(df[col], col)
        else:
            df[col] = [[] for _ in range(len(df))]

    # Nested list-of-dict columns
    for col in _NESTED_LIST_COLUMNS:
        if col in df.columns:
            df[col] = _coerce_nested_list_column(df[col], col)
        else:
            df[col] = [[] for _ in range(len(df))]

    # Boolean
    if "internet_exposed" in df.columns:
        df["internet_exposed"] = df["internet_exposed"].map(
            lambda v: str(v).strip().lower() in {"true", "1", "yes"}
            if not isinstance(v, bool) else v
        ).astype(bool)

    # last_seen — keep as string; Pydantic parses it
    if "last_seen" in df.columns:
        df["last_seen"] = df["last_seen"].fillna("").astype(str).str.strip()

    return df


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------


def detect_format(path: Path) -> FileFormat:
    """
    Infer file format from the file extension.

    Parameters
    ----------
    path : Path

    Returns
    -------
    FileFormat

    Raises
    ------
    ValueError
        For unsupported extensions.
    """
    suffix = path.suffix.lstrip(".").lower()
    try:
        return FileFormat(suffix)
    except ValueError:
        supported = [f.value for f in FileFormat]
        raise ValueError(
            f"Unsupported extension '.{suffix}' for '{path.name}'. "
            f"Supported formats: {supported}"
        )


# ---------------------------------------------------------------------------
# Raw readers
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> pd.DataFrame:
    """Load a JSON array of asset objects into a DataFrame."""
    log.info("Reading JSON: %s", path)
    with path.open("r", encoding="utf-8") as fh:
        raw: list[dict] = json.load(fh)
    if not isinstance(raw, list):
        raise ValueError(
            f"Expected a JSON array at the top level, got {type(raw).__name__}"
        )
    log.info("Parsed %d raw JSON records.", len(raw))
    return pd.DataFrame(raw)


def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    """Load a CSV asset file."""
    log.info("Reading CSV: %s", path)
    df = pd.read_csv(path, **kwargs)
    log.info("Loaded %d rows from CSV.", len(df))
    return df


def _read_parquet(path: Path, **kwargs) -> pd.DataFrame:
    """Load a Parquet asset file."""
    log.info("Reading Parquet: %s", path)
    df = pd.read_parquet(path, **kwargs)
    log.info("Loaded %d rows from Parquet.", len(df))
    return df


_READERS = {
    FileFormat.JSON: _read_json,
    FileFormat.CSV: _read_csv,
    FileFormat.PARQUET: _read_parquet,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_assets(
    path: str | Path,
    fmt: str | FileFormat | None = None,
    validate: bool = True,
    **reader_kwargs: Any,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Load, normalise, and validate an asset discovery file.

    Parameters
    ----------
    path : str | Path
        Path to the asset file (JSON, CSV, or Parquet).
    fmt : str | FileFormat | None
        Explicit format override. When ``None`` the format is inferred
        from the file extension.
    validate : bool
        When ``True`` (default), validate every row against the Pydantic
        :class:`~asset_pipeline.schema.DiscoveredAsset` schema.
    **reader_kwargs
        Forwarded to the underlying pandas reader.

    Returns
    -------
    tuple[pd.DataFrame, list[dict]]
        - ``clean_df`` : validated DataFrame (only accepted rows).
        - ``rejections`` : list of ``{index, asset_id, reason}`` dicts.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the format is unsupported or all rows fail validation.
    RuntimeError
        If the reader raises an unexpected error.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Asset file not found: '{path}'. "
            "Verify the path or run the scanner to generate it."
        )

    # Resolve format
    if fmt is None:
        file_fmt = detect_format(path)
    elif isinstance(fmt, str):
        file_fmt = FileFormat(fmt.lower())
    else:
        file_fmt = fmt

    log.info("Loading assets from %s (%s) …", path.name, file_fmt.value.upper())

    # Read raw
    try:
        reader = _READERS[file_fmt]
        raw_df = reader(path, **reader_kwargs)
    except (FileNotFoundError, ValueError):
        raise
    except Exception as exc:
        raise RuntimeError(
            f"Unexpected error reading '{path}' as {file_fmt.value.upper()}: {exc}"
        ) from exc

    log.info("Raw shape: %d rows × %d columns.", len(raw_df), len(raw_df.columns))

    # Normalise types
    log.info("Normalising column types …")
    norm_df = _normalise_types(raw_df)

    # Schema validation
    if validate:
        log.info("Validating schema …")
        clean_df, rejections = validate_dataframe(norm_df)
    else:
        clean_df = norm_df
        rejections = []

    log.info(
        "Load complete: %d clean records returned (%d rejected).",
        len(clean_df),
        len(rejections),
    )
    return clean_df, rejections