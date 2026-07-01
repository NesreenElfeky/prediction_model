"""
asset_pipeline/pipeline.py
==========================
End-to-end asset pipeline entry point.

Flow
----
JSON file
    ↓  loader.load_assets()     — read, type-normalise, schema-validate
    ↓  enricher.enrich_dataframe() — derive cybersecurity features
    ↓  save outputs
    ↓
outputs/
    assets_clean.parquet
    assets_clean.csv
    assets_summary.json

Usage
-----
From Python:
    from asset_pipeline.pipeline import run_pipeline
    df, summary = run_pipeline("datasets/assets/targets.json")

From command line:
    python -m asset_pipeline.pipeline datasets/assets/targets.json

Part of: cyber_financial_loss_prediction / asset_pipeline (Part 2)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

from asset_pipeline.loader import load_assets
from asset_pipeline.enricher import (
    enrich_dataframe,
    save_parquet,
    save_csv,
    build_summary,
    save_summary,
)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("asset_pipeline.pipeline")

# ---------------------------------------------------------------------------
# Default output paths
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_PARQUET = DEFAULT_OUTPUT_DIR / "assets_clean.parquet"
DEFAULT_CSV = DEFAULT_OUTPUT_DIR / "assets_clean.csv"
DEFAULT_SUMMARY = DEFAULT_OUTPUT_DIR / "assets_summary.json"

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    input_path: str | Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    validate: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Execute the full asset pipeline.

    Parameters
    ----------
    input_path : str | Path
        Path to the raw asset JSON file (or CSV / Parquet).
    output_dir : Path
        Directory where output files are written.
    validate : bool
        When ``True``, validate each record against the Pydantic schema.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, Any]]
        ``(enriched_df, summary_dict)``
    """
    t0 = time.perf_counter()
    output_dir = Path(output_dir)

    log.info("=" * 65)
    log.info("  Asset Pipeline — START")
    log.info("  Input : %s", input_path)
    log.info("  Output: %s/", output_dir)
    log.info("=" * 65)

    # ── Step 1: Load & validate ───────────────────────────────────────────
    log.info("Loading assets …")
    clean_df, rejections = load_assets(input_path, validate=validate)

    log.info(
        "Loaded %d clean assets (%d rejected).",
        len(clean_df),
        len(rejections),
    )

    # ── Step 2: Enrich ────────────────────────────────────────────────────
    log.info("Enriching assets …")
    enriched_df = enrich_dataframe(clean_df)

    # ── Step 3: Save Parquet ──────────────────────────────────────────────
    parquet_path = output_dir / "assets_clean.parquet"
    log.info("Saving parquet …")
    save_parquet(enriched_df, parquet_path)

    # ── Step 4: Save CSV ──────────────────────────────────────────────────
    csv_path = output_dir / "assets_clean.csv"
    log.info("Saving CSV …")
    save_csv(enriched_df, csv_path)

    # ── Step 5: Build & save summary ──────────────────────────────────────
    log.info("Saving summary …")
    summary = build_summary(enriched_df, rejections)
    summary_path = output_dir / "assets_summary.json"
    save_summary(summary, summary_path)

    elapsed = time.perf_counter() - t0
    log.info("=" * 65)
    log.info("  Pipeline completed in %.2f s.", elapsed)
    log.info("  Assets accepted : %d", len(enriched_df))
    log.info("  Assets rejected : %d", len(rejections))
    log.info("  Output columns  : %d", len(enriched_df.columns))
    log.info("=" * 65)

    return enriched_df, summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="asset_pipeline.pipeline",
        description="Cyber-financial asset pipeline — load, validate, enrich, save.",
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to the raw asset JSON (or CSV/Parquet) file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip Pydantic schema validation (faster, less safe).",
    )
    return parser


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()

    df, summary = run_pipeline(
        input_path=args.input,
        output_dir=Path(args.output_dir),
        validate=not args.no_validate,
    )

    # Print compact summary to stdout
    print("\n" + "=" * 65)
    print("  PIPELINE SUMMARY")
    print("=" * 65)
    meta = summary["pipeline_metadata"]
    print(f"  Records processed : {meta['total_records_processed']}")
    print(f"  Accepted          : {meta['accepted']}")
    print(f"  Rejected          : {meta['rejected']}")
    print(f"  Pass rate         : {meta['pass_rate_pct']}%")
    print(f"\n  Asset types discovered:")
    for atype, count in summary["asset_type_distribution"].items():
        print(f"    {atype:<25} {count}")
    print(f"\n  Internet-exposed  : {summary['internet_exposure']['exposed']}")
    print(f"  Risk score (mean) : {summary['risk_score_distribution']['mean']}")
    print(f"  High-risk assets  : {summary['risk_score_distribution']['high_risk_count']}")
    if summary["rejected_records"]:
        print(f"\n  Rejected records:")
        for r in summary["rejected_records"]:
            print(f"    {r['asset_id']} — {r['reason'][:80]}")
    print("=" * 65 + "\n")

    sys.exit(0)