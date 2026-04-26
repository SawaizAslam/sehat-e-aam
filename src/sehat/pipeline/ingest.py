"""Notebook 01 equivalent: ingest the raw CSV into a Bronze parquet table.

Fixes vs the original guide:

* ``facility_id`` hashes ``name + city + zip`` so two ``Apollo Hospital`` rows
  do not collide.
* ``composite_text_length`` is **persisted** in Bronze (not just a temp view).
* ``facilityTypeId`` is normalised (``farmacy`` -> ``pharmacy``).
* ``address_zipOrPostcode`` is forced to a clean string.
* CSV reader auto-detects ``.xlsx``.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Iterable

import pandas as pd
from rich.console import Console
from rich.table import Table

from ..config import Settings, get_settings
from ..schemas import FacilityType
from ..storage import read_parquet, write_parquet

LOGGER = logging.getLogger(__name__)
console = Console()


# Columns we keep from the raw dataset (others stay ignored).
_KEPT_COLUMNS: tuple[str, ...] = (
    "name",
    "phone_numbers",
    "officialPhone",
    "email",
    "websites",
    "officialWebsite",
    "yearEstablished",
    "address_line1",
    "address_line2",
    "address_line3",
    "address_city",
    "address_stateOrRegion",
    "address_zipOrPostcode",
    "address_country",
    "facilityTypeId",
    "operatorTypeId",
    "affiliationTypeIds",
    "description",
    "numberDoctors",
    "capacity",
    "specialties",
    "procedure",
    "equipment",
    "capability",
    "latitude",
    "longitude",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hash_id(*parts: object) -> str:
    joined = "|".join("" if p is None else str(p).strip().lower() for p in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _coerce_str(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _build_composite_text(row: pd.Series) -> str:
    fields: Iterable[str] = (
        _coerce_str(row.get("name")),
        _coerce_str(row.get("description")),
        _coerce_str(row.get("specialties")),
        _coerce_str(row.get("procedure")),
        _coerce_str(row.get("equipment")),
        _coerce_str(row.get("capability")),
    )
    return " | ".join(p for p in fields if p)


def _read_raw(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. Set RAW_DATASET_PATH in .env or place the file there."
        )
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path, low_memory=False)


def _quality_report(df: pd.DataFrame) -> Table:
    table = Table(title="Bronze ingestion quality report")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta", justify="right")
    table.add_row("Total rows", f"{len(df):,}")
    table.add_row("With description", f"{df['description'].astype(str).str.len().gt(0).sum():,}")
    table.add_row(
        "With equipment list",
        f"{df['equipment'].astype(str).str.strip().ne('').sum() - df['equipment'].astype(str).str.strip().eq('[]').sum():,}",
    )
    table.add_row("With lat/lon", f"{df[['latitude', 'longitude']].notna().all(axis=1).sum():,}")
    table.add_row(
        "With pin code",
        f"{df['address_zipOrPostcode'].astype(str).str.strip().ne('').sum():,}",
    )
    table.add_row("Distinct facility_ids", f"{df['facility_id'].nunique():,}")
    return table


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_ingest(settings: Settings | None = None) -> pd.DataFrame:
    """Read the raw dataset, normalise it, and write Bronze parquet."""

    s = settings or get_settings()
    s.ensure_dirs()

    console.log(f"Reading raw dataset from {s.raw_dataset_path}")
    raw = _read_raw(s.raw_dataset_path)

    # Keep only known columns; fill with empty strings if missing
    for col in _KEPT_COLUMNS:
        if col not in raw.columns:
            raw[col] = pd.NA
    df = raw[list(_KEPT_COLUMNS)].copy()

    # Normalise primitive fields
    df["address_zipOrPostcode"] = (
        df["address_zipOrPostcode"]
        .astype("string")
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
        .replace({"<NA>": pd.NA, "nan": pd.NA, "None": pd.NA})
    )
    df["facilityTypeId"] = df["facilityTypeId"].apply(
        lambda v: FacilityType.normalise(v if isinstance(v, str) else None).value
    )
    for c in ("latitude", "longitude"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["numberDoctors"] = pd.to_numeric(df["numberDoctors"], errors="coerce").astype("Int64")
    df["capacity"] = pd.to_numeric(df["capacity"], errors="coerce").astype("Int64")

    # Composite ID + text
    df["facility_id"] = df.apply(
        lambda r: _hash_id(r["name"], r.get("address_city"), r.get("address_zipOrPostcode")),
        axis=1,
    )
    df["composite_text"] = df.apply(_build_composite_text, axis=1)
    df["composite_text_length"] = df["composite_text"].str.len().astype("int64")
    df["_ingest_ts"] = pd.Timestamp.utcnow()

    # Drop facility_id collisions defensively (none expected for this dataset)
    before = len(df)
    df = df.drop_duplicates(subset=["facility_id"], keep="first").reset_index(drop=True)
    after = len(df)
    if before != after:
        LOGGER.warning("Dropped %d duplicate facility_ids", before - after)

    write_parquet(df, s.bronze_path, overwrite=True)
    console.print(_quality_report(df))
    console.log(f":white_check_mark: Bronze written: {s.bronze_path} ({len(df):,} rows)")
    return df


# Convenience: load Bronze for downstream stages
def load_bronze(settings: Settings | None = None) -> pd.DataFrame:
    s = settings or get_settings()
    return read_parquet(s.bronze_path)


__all__ = ["run_ingest", "load_bronze"]


# ---------------------------------------------------------------------------
# CLI shim (``python -m sehat.pipeline.ingest``)
# ---------------------------------------------------------------------------


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_ingest()
