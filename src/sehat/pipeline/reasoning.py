"""Notebook 06 equivalent: vector retrieval -> SQL filter -> LLM ranker."""

from __future__ import annotations

import json
import logging
from typing import Any

import pandas as pd

from ..config import Settings, get_settings
from ..llm import LLMClient, LLMError
from ..prompts import REASONING_SYSTEM_PROMPT
from ..schemas import ReasoningResponse
from ..storage import duck
from ..tracing import init_tracing, run, span
from .vector_search import FacilityVectorIndex

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Build the structured "candidate" list
# ---------------------------------------------------------------------------


def _summarise_candidate(meta: dict[str, Any]) -> dict[str, Any]:
    extraction = json.loads(meta.get("extraction_json") or "{}")
    confidence = json.loads(meta.get("confidence_json") or "{}")
    return {
        "facility_id": meta["facility_id"],
        "name": meta.get("name"),
        "location": f"{meta.get('address_city') or ''}, {meta.get('address_state') or ''} - {meta.get('address_zip') or ''}",
        "facility_type": meta.get("facility_type"),
        "trust_score": round(float(meta.get("trust_score") or 0.0), 3),
        "confidence_overall": confidence.get("overall"),
        "capabilities": {
            "icu": (extraction.get("icu") or {}).get("present"),
            "icu_functional": (extraction.get("icu") or {}).get("functional_status"),
            "surgery": extraction.get("surgery") or {},
            "emergency_24_7": (extraction.get("emergency") or {}).get("is_24_7"),
            "emergency_status": (extraction.get("emergency") or {}).get("emergency_care"),
            "anesthesiologist": (extraction.get("staff") or {}).get("anesthesiologist"),
            "surgeon_type": (extraction.get("staff") or {}).get("surgeon"),
            "dialysis": (extraction.get("dialysis") or {}).get("present"),
            "specialties": (extraction.get("specialties_extracted") or [])[:10],
        },
        "key_source_texts": [
            (extraction.get("surgery") or {}).get("source_text"),
            (extraction.get("staff") or {}).get("source_text"),
            (extraction.get("emergency") or {}).get("source_text"),
        ],
    }


def query_facilities(
    *,
    user_query: str,
    state_filter: str | None = None,
    city_filter: str | None = None,
    facility_type_filter: str | None = None,
    min_trust_score: float | None = None,
    top_k_vector: int | None = None,
    top_k_final: int | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Full multi-attribute reasoning pipeline."""

    s = settings or get_settings()
    init_tracing()
    min_trust = s.min_trust_for_reasoning if min_trust_score is None else min_trust_score
    top_k_v = top_k_vector or s.vector_top_k
    top_k_f = top_k_final or s.reasoning_top_k

    with run(
        "reasoning",
        query=user_query,
        state=state_filter or "",
        city=city_filter or "",
        min_trust=min_trust,
    ):
        # Step 1: vector retrieval
        with span("vector_retrieval", top_k=top_k_v):
            index = FacilityVectorIndex(s)
            hits = index.search(
                user_query,
                top_k=top_k_v,
                state=state_filter,
                city=city_filter,
                facility_type=facility_type_filter,
                min_trust=min_trust,
            )

        if not hits:
            return {
                "query": user_query,
                "ranked_results": [],
                "recommendation_summary": "No candidates met the filter and trust criteria.",
                "uncertainty_note": "Try lowering min_trust_score or broadening location filter.",
                "candidates_retrieved": 0,
            }

        # Step 2: structured re-validation against Gold (in case index is stale)
        with span("structured_filter", candidates=len(hits)):
            ids = [h.facility_id for h in hits]
            with duck(s) as con:
                placeholders = ", ".join(["?"] * len(ids))
                df = con.execute(
                    f"""
                    SELECT facility_id, trust_score, extraction_json, confidence_json
                    FROM gold
                    WHERE facility_id IN ({placeholders}) AND trust_score >= ?
                    """,
                    [*ids, min_trust],
                ).df()
            valid_ids = set(df["facility_id"].tolist())
            hits = [h for h in hits if h.facility_id in valid_ids]
            for h in hits:
                row = df[df["facility_id"] == h.facility_id].iloc[0]
                h.metadata["trust_score"] = float(row["trust_score"])
                h.metadata["extraction_json"] = row["extraction_json"]
                h.metadata["confidence_json"] = row["confidence_json"]

        if not hits:
            return {
                "query": user_query,
                "ranked_results": [],
                "recommendation_summary": "Candidates were filtered out by the trust threshold.",
                "uncertainty_note": f"Try lowering min_trust_score below {min_trust}.",
                "candidates_retrieved": 0,
            }

        # Step 3: build LLM context
        summaries = [_summarise_candidate(h.metadata) for h in hits]

        reasoning_user = (
            f"User Query: \"{user_query}\"\n\n"
            f"Trust threshold for recommendation: {min_trust}\n\n"
            f"Candidate Facilities ({len(summaries)} total):\n"
            f"{json.dumps(summaries, indent=2, default=str)}\n\n"
            f"Rank and evaluate these candidates. Return top {top_k_f} with detailed reasoning."
        )
        messages = [
            {"role": "system", "content": REASONING_SYSTEM_PROMPT},
            {"role": "user", "content": reasoning_user},
        ]

        client = LLMClient(s)
        with span("llm_reasoning", candidates=len(summaries), top_k_final=top_k_f):
            try:
                data, resp = client.complete_json(messages, temperature=0.1, max_tokens=2000)
            except LLMError as e:
                return {
                    "query": user_query,
                    "ranked_results": [],
                    "recommendation_summary": "LLM ranker failed.",
                    "uncertainty_note": str(e),
                    "candidates_retrieved": len(summaries),
                }

        try:
            parsed = ReasoningResponse.model_validate(data).model_dump(mode="json")
        except Exception as e:  # pragma: no cover
            LOGGER.warning("Reasoning response failed schema validation: %s", e)
            parsed = data

        parsed.update(
            {
                "query": user_query,
                "candidates_retrieved": len(summaries),
                "trust_threshold": min_trust,
                "filters": {
                    "state": state_filter,
                    "city": city_filter,
                    "facility_type": facility_type_filter,
                },
                "tokens": {
                    "prompt": resp.prompt_tokens,
                    "completion": resp.completion_tokens,
                },
            }
        )
        return parsed


__all__ = ["query_facilities"]
