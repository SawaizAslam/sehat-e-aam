"""FastAPI service exposing the four endpoints from Notebook 08."""

from __future__ import annotations

import json
import logging
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from ..config import get_settings
from ..pipeline.reasoning import query_facilities
from ..storage import duck, parquet_exists, read_parquet

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


_LANDING_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Sehat-e-Aam · API</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>
  :root { color-scheme: light dark; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    max-width: 720px; margin: 4rem auto; padding: 0 1.5rem; line-height: 1.55;
  }
  h1 { margin-bottom: .25rem; }
  .tag { color: #888; font-size: .9rem; margin-bottom: 2rem; }
  code, pre { background: rgba(127,127,127,.12); border-radius: 6px; padding: .15rem .4rem; }
  pre { padding: 1rem; overflow-x: auto; }
  .endpoints { display: grid; grid-template-columns: 5rem 1fr; gap: .35rem 1rem; margin: 1rem 0; }
  .method { font-weight: 600; }
  .ok    { color: #1a7f37; }
  .post  { color: #6f42c1; }
  a { color: #0969da; }
  footer { margin-top: 3rem; font-size: .85rem; color: #888; }
</style>
</head>
<body>
  <h1>Sehat-e-Aam · Healthcare Intelligence API</h1>
  <p class="tag">FastAPI mirror, public CORS, LLM via Databricks Foundation Model API.</p>

  <p>
    <a href="/docs"><strong>Open Swagger UI &rarr;</strong></a>
    &nbsp;·&nbsp;
    <a href="/health">/health</a>
    &nbsp;·&nbsp;
    <a href="/openapi.json">openapi.json</a>
  </p>

  <h2>Endpoints</h2>
  <div class="endpoints">
    <span class="method ok">GET</span>   <code>/health</code>
    <span class="method post">POST</span><code>/api/query</code>
    <span class="method ok">GET</span>   <code>/api/facility/{facility_id}</code>
    <span class="method ok">GET</span>   <code>/api/facility/{facility_id}/trust</code>
    <span class="method ok">GET</span>   <code>/api/deserts</code>
  </div>

  <h2>Quick example</h2>
<pre>fetch("/api/query", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    query: "ICU and dialysis in Lucknow",
    top_k: 5
  })
}).then(r =&gt; r.json())</pre>

  <footer>
    Source: <a href="https://github.com/SawaizAslam/sehat-e-aam">github.com/SawaizAslam/sehat-e-aam</a>
  </footer>
</body>
</html>
"""


@app.get("/", include_in_schema=False)
def root() -> HTMLResponse:
    return HTMLResponse(_LANDING_PAGE)


@app.get("/health")
def health() -> dict[str, Any]:
    s = get_settings()
    row_count = 0
    if parquet_exists(s.gold_path):
        try:
            with duck(s) as con:
                row_count = int(con.execute("SELECT COUNT(*) FROM gold").fetchone()[0])
        except Exception as exc:  # noqa: BLE001 - degrade gracefully on health check
            LOGGER.warning("Failed to count gold rows for /health: %s", exc)
    return {
        "status": "ok",
        "row_count": row_count,
        "embedding_model": s.embedding_model,
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
    response = query_facilities(
        user_query=req.query,
        state_filter=req.state,
        city_filter=req.city,
        facility_type_filter=req.facility_type,
        min_trust_score=req.min_trust_score,
        top_k_final=req.top_k,
    )
    _enrich_ranked_results(response)
    return response


def _enrich_ranked_results(response: dict[str, Any]) -> None:
    """Mutate ``response['ranked_results']`` so each item includes a ``facility_meta``
    block with location, coordinates, trust score and trust flags from the Gold
    table. This lets the public web frontend render full result cards (map,
    trust badge, breakdown) without an N+1 round-trip per result.
    """
    results = response.get("ranked_results") or []
    if not results:
        return

    s = get_settings()
    if not parquet_exists(s.gold_path):
        LOGGER.warning("Gold parquet missing; skipping result enrichment.")
        return

    facility_ids = [r["facility_id"] for r in results if r.get("facility_id")]
    if not facility_ids:
        return

    placeholders = ", ".join(["?"] * len(facility_ids))
    sql = f"""
        SELECT facility_id, name, address_city, address_state, address_zip,
               latitude, longitude, facility_type, operator_type, trust_score,
               trust_flags_json
        FROM gold
        WHERE facility_id IN ({placeholders})
    """
    with duck(s) as con:
        df = con.execute(sql, facility_ids).df()

    meta_by_id: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        try:
            trust_flags = json.loads(row["trust_flags_json"]) if row.get("trust_flags_json") else []
        except (TypeError, ValueError):
            trust_flags = []
        trust_score = float(row["trust_score"]) if pd.notna(row.get("trust_score")) else 0.0
        meta_by_id[str(row["facility_id"])] = {
            "name": row["name"],
            "city": row.get("address_city"),
            "state": row.get("address_state"),
            "pin_code": row.get("address_zip"),
            "latitude": float(row["latitude"]) if pd.notna(row.get("latitude")) else None,
            "longitude": float(row["longitude"]) if pd.notna(row.get("longitude")) else None,
            "facility_type": row.get("facility_type"),
            "operator_type": row.get("operator_type"),
            "trust_score": trust_score,
            "trust_grade": _trust_grade(trust_score),
            "trust_flags": trust_flags,
        }

    for result in results:
        meta = meta_by_id.get(str(result.get("facility_id")))
        if meta is not None:
            result["facility_meta"] = meta


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

    df = read_parquet(s.deserts_path)
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
