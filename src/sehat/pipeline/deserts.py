"""Notebook 07 equivalent: medical desert detection by PIN code.

Fixes vs the original guide:

* Uses DuckDB JSON extraction instead of 5 Python UDFs (one query, native).
* Uses ``not_present``-aware semantics (does not double-count "uncertain").
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pandas as pd
from rich.console import Console

from ..config import Settings, get_settings
from ..storage import duck, parquet_exists, write_parquet
from ..tracing import init_tracing, log_metrics, run

LOGGER = logging.getLogger(__name__)
console = Console()


# Capability weights — higher = bigger gap penalty
ICU_WEIGHT = 0.35
EMERGENCY_WEIGHT = 0.30
DIALYSIS_WEIGHT = 0.20
SURGERY_WEIGHT = 0.15
DESERT_THRESHOLD = 0.10


_SQL = """
WITH parsed AS (
    SELECT
        facility_id,
        address_state,
        address_zip,
        latitude,
        longitude,
        trust_score,
        json_extract_string(extraction_json, '$.icu.present') AS icu_present,
        json_extract_string(extraction_json, '$.dialysis.present') AS dialysis_present,
        json_extract_string(extraction_json, '$.emergency.emergency_care') AS emergency_present,
        json_extract_string(extraction_json, '$.surgery.general_surgery') AS surgery_present
    FROM gold
    WHERE address_zip IS NOT NULL AND length(trim(address_zip)) > 0
),
flagged AS (
    SELECT
        *,
        CASE WHEN icu_present       IN ('confirmed','claimed') THEN 1 ELSE 0 END AS has_icu,
        CASE WHEN dialysis_present  IN ('confirmed','claimed') THEN 1 ELSE 0 END AS has_dialysis,
        CASE WHEN emergency_present IN ('confirmed','claimed') THEN 1 ELSE 0 END AS has_emergency,
        CASE WHEN surgery_present   IN ('confirmed','claimed') THEN 1 ELSE 0 END AS has_surgery
    FROM parsed
)
SELECT
    address_zip                 AS pin_code,
    address_state               AS state,
    COUNT(*)                    AS facility_count,
    AVG(trust_score)            AS avg_trust_score,
    AVG(has_icu)                AS icu_coverage,
    AVG(has_dialysis)           AS dialysis_coverage,
    AVG(has_emergency)          AS emergency_coverage,
    AVG(has_surgery)            AS surgery_coverage,
    AVG(latitude)               AS centroid_lat,
    AVG(longitude)              AS centroid_lon
FROM flagged
GROUP BY address_zip, address_state
"""


def _categorise(row: pd.Series) -> list[str]:
    cats: list[str] = []
    if row["icu_coverage"] < DESERT_THRESHOLD:
        cats.append("ICU_DESERT")
    if row["dialysis_coverage"] < DESERT_THRESHOLD:
        cats.append("DIALYSIS_DESERT")
    if row["emergency_coverage"] < DESERT_THRESHOLD:
        cats.append("EMERGENCY_DESERT")
    if row["surgery_coverage"] < DESERT_THRESHOLD:
        cats.append("SURGERY_DESERT")
    if not cats:
        cats.append("ADEQUATE_COVERAGE")
    return cats


def run_deserts(settings: Settings | None = None) -> pd.DataFrame:
    s = settings or get_settings()
    init_tracing()

    if not parquet_exists(s.gold_path):
        raise FileNotFoundError("Gold parquet missing; run `sehat trust` first.")

    with run("medical_deserts"):
        with duck(s) as con:
            df = con.execute(_SQL).df()

        if df.empty:
            console.log(":warning: No PIN-coded rows in Gold; nothing to aggregate.")
            return df

        df = df.fillna({"icu_coverage": 0.0, "dialysis_coverage": 0.0, "emergency_coverage": 0.0, "surgery_coverage": 0.0})

        df["desert_risk_score"] = (
            ICU_WEIGHT * (1.0 - df["icu_coverage"])
            + EMERGENCY_WEIGHT * (1.0 - df["emergency_coverage"])
            + DIALYSIS_WEIGHT * (1.0 - df["dialysis_coverage"])
            + SURGERY_WEIGHT * (1.0 - df["surgery_coverage"])
        ).round(3)
        df["is_high_risk"] = df["desert_risk_score"] >= s.desert_high_risk_threshold
        df["desert_categories"] = df.apply(_categorise, axis=1)

        df = df.sort_values("desert_risk_score", ascending=False).reset_index(drop=True)
        write_parquet(df, s.deserts_path, overwrite=True)

        log_metrics(
            desert_pin_codes=len(df),
            high_risk_pin_codes=int(df["is_high_risk"].sum()),
            avg_desert_risk=float(df["desert_risk_score"].mean()),
        )
        console.log(
            f":white_check_mark: Deserts table: {len(df):,} PIN codes "
            f"({int(df['is_high_risk'].sum()):,} high-risk)"
        )
    return df


__all__ = ["run_deserts"]


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_deserts()
