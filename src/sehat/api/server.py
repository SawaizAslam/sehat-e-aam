"""FastAPI service exposing the four endpoints from Notebook 08."""

from __future__ import annotations

import json
import logging
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..config import get_settings
from ..pipeline.reasoning import query_facilities
from ..storage import duck, parquet_exists

LOGGER = logging.getLogger(__name__)

app = FastAPI(
    title="Sehat-e-Aam Healthcare Intelligence API",
    description=(
        "Search, evaluate trust, and analyse medical deserts across Indian healthcare facilities."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=2)
    state: str | None = None
    city: str | None = None
    facility_type: str | None = None
    min_trust_score: float | None = Field(default=None, ge=0.0, le=1.0)
    top_k: int = Field(default=5, ge=1, le=20)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _trust_grade(score: float) -> str:
    if score >= 0.85:
        return "A"
    if score >= 0.70:
        return "B"
    if score >= 0.55:
        return "C"
    if score >= 0.40:
        return "D"
    return "F"


def _load_gold_row(facility_id: str) -> pd.Series | None:
    s = get_settings()
    if not parquet_exists(s.gold_path):
        raise HTTPException(503, "Gold table not built yet. Run the pipeline first.")
    with duck(s) as con:
        df = con.execute(
            "SELECT * FROM gold WHERE facility_id = ?", [facility_id]
        ).df()
    if df.empty:
        return None
    return df.iloc[0]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, Any]:
    s = get_settings()
    return {
        "status": "ok",
        "bronze_ready": parquet_exists(s.bronze_path),
        "silver_ready": parquet_exists(s.silver_path),
        "gold_ready": parquet_exists(s.gold_path),
        "vector_ready": s.vector_index_path.exists(),
        "deserts_ready": parquet_exists(s.deserts_path),
        "llm_backend": s.llm_backend,
        "llm_model": s.llm_model,
        "embedding_backend": s.embedding_backend,
    }


@app.post("/api/query")
def api_query_facilities(req: QueryRequest) -> dict[str, Any]:
    return query_facilities(
        user_query=req.query,
        state_filter=req.state,
        city_filter=req.city,
        facility_type_filter=req.facility_type,
        min_trust_score=req.min_trust_score,
        top_k_final=req.top_k,
    )


@app.get("/api/facility/{facility_id}/trust")
def api_get_trust_report(facility_id: str) -> dict[str, Any]:
    row = _load_gold_row(facility_id)
    if row is None:
        raise HTTPException(404, f"Facility {facility_id} not found")

    trust_score = float(row["trust_score"])
    extraction = json.loads(row["extraction_json"])
    return {
        "facility_id": facility_id,
        "name": row["name"],
        "location": f"{row.get('address_city')}, {row.get('address_state')}",
        "trust_score": trust_score,
        "trust_grade": _trust_grade(trust_score),
        "trust_flags": json.loads(row["trust_flags_json"]),
        "confidence": json.loads(row["confidence_json"]),
        "correction_iterations": int(row["correction_iterations"] or 0),
        "extraction_summary": {
            k: extraction.get(k, {})
            for k in ("icu", "ventilator", "staff", "emergency", "surgery", "dialysis")
        },
        "extraction_notes": extraction.get("extraction_notes"),
    }


@app.get("/api/facility/{facility_id}")
def api_get_facility_profile(facility_id: str) -> dict[str, Any]:
    row = _load_gold_row(facility_id)
    if row is None:
        raise HTTPException(404, f"Facility {facility_id} not found")

    return {
        "facility_id": facility_id,
        "name": row["name"],
        "address": {
            "city": row.get("address_city"),
            "state": row.get("address_state"),
            "pin_code": row.get("address_zip"),
        },
        "coordinates": {
            "latitude": float(row["latitude"]) if pd.notna(row.get("latitude")) else None,
            "longitude": float(row["longitude"]) if pd.notna(row.get("longitude")) else None,
        },
        "facility_type": row.get("facility_type"),
        "operator_type": row.get("operator_type"),
        "trust_score": float(row["trust_score"]),
        "trust_flags": json.loads(row["trust_flags_json"]),
        "confidence": json.loads(row["confidence_json"]),
        "capabilities": json.loads(row["extraction_json"]),
        "correction_iterations": int(row["correction_iterations"] or 0),
    }


@app.get("/api/deserts")
def api_get_desert_map(
    state: str | None = Query(None),
    high_risk_only: bool = Query(False),
    desert_type: str | None = Query(None, description="ICU_DESERT | DIALYSIS_DESERT | EMERGENCY_DESERT | SURGERY_DESERT"),
    limit: int = Query(100, ge=1, le=2000),
) -> dict[str, Any]:
    s = get_settings()
    if not parquet_exists(s.deserts_path):
        raise HTTPException(503, "Deserts table not built. Run `sehat deserts` first.")

    df = pd.read_parquet(s.deserts_path)
    if state:
        df = df[df["state"].str.lower() == state.lower()]
    if high_risk_only:
        df = df[df["is_high_risk"]]
    if desert_type:
        df = df[df["desert_categories"].apply(lambda c: desert_type in (c or []))]

    df = df.sort_values("desert_risk_score", ascending=False).head(limit)

    regions = []
    for _, row in df.iterrows():
        regions.append(
            {
                "pin_code": row["pin_code"],
                "state": row["state"],
                "facility_count": int(row["facility_count"]),
                "desert_risk_score": float(row["desert_risk_score"]),
                "is_high_risk": bool(row["is_high_risk"]),
                "desert_categories": list(row["desert_categories"] or []),
                "centroid_lat": float(row["centroid_lat"]) if pd.notna(row.get("centroid_lat")) else None,
                "centroid_lon": float(row["centroid_lon"]) if pd.notna(row.get("centroid_lon")) else None,
                "coverage": {
                    "icu": float(row["icu_coverage"]),
                    "dialysis": float(row["dialysis_coverage"]),
                    "emergency": float(row["emergency_coverage"]),
                    "surgery": float(row["surgery_coverage"]),
                },
                "avg_trust_score": float(row["avg_trust_score"]),
            }
        )

    high_risk = sum(1 for r in regions if r["is_high_risk"])
    return {
        "regions": regions,
        "summary": {
            "total_regions": len(regions),
            "high_risk_regions": high_risk,
            "high_risk_percentage": round(high_risk / len(regions) * 100, 1) if regions else 0.0,
            "filter_state": state,
            "filter_desert_type": desert_type,
        },
    }


__all__ = ["app"]
