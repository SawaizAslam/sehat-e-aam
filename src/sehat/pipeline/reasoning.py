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
from .vector_search import FacilityVectorIndex, VectorHit

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Heuristic fallback ranker (used when the LLM is unreachable)
# ---------------------------------------------------------------------------


_QUERY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "icu": ("icu", "intensive care", "critical care"),
    "ventilator": ("ventilator", "vent"),
    "dialysis": ("dialysis", "kidney", "renal"),
    "emergency": ("emergency", "casualty", "er ", "24/7", "24x7", "24*7", "trauma"),
    "trauma": ("trauma", "accident"),
    "surgery": ("surgery", "surgical", "operation", "ot ", "operation theatre"),
    "ambulance": ("ambulance", "108", "102"),
    "maternity": ("maternity", "obstetric", "delivery", "labour", "labor", "caesarean", "c-section"),
    "pharmacy": ("pharmacy", "chemist", "medicine"),
    "cardiac": ("cardiac", "cardiology", "heart"),
    "orthopedic": ("ortho", "bone", "fracture"),
    "neonatal": ("neonatal", "newborn", "nicu", "baby"),
    "dentist": ("dentist", "dental", "tooth"),
}


def _query_intents(user_query: str) -> list[str]:
    """Pick the capability tags a user query is asking about (lowercase substring match)."""
    q = (user_query or "").lower()
    intents = [tag for tag, kws in _QUERY_KEYWORDS.items() if any(kw in q for kw in kws)]
    return intents


def _capability_satisfied(extraction: dict[str, Any], intent: str) -> tuple[bool, str]:
    """Return (matched, evidence_phrase) for a single intent against an extraction record.

    "Matched" means the facility has the capability either confirmed or claimed
    (i.e. NOT explicitly absent and NOT uncertain). Evidence is a short
    human-readable phrase fit for the reasoning text.
    """

    def _is_present(profile: dict[str, Any] | None, key: str = "present") -> bool:
        v = (profile or {}).get(key)
        return isinstance(v, str) and v.lower() in {"confirmed", "claimed"}

    icu = extraction.get("icu") or {}
    vent = extraction.get("ventilator") or {}
    dial = extraction.get("dialysis") or {}
    emer = extraction.get("emergency") or {}
    surg = extraction.get("surgery") or {}
    staff = extraction.get("staff") or {}
    specs = [s.lower() for s in (extraction.get("specialties_extracted") or []) if isinstance(s, str)]

    if intent == "icu":
        if _is_present(icu):
            beds = icu.get("bed_count")
            return True, f"ICU available{f' ({beds} beds)' if beds else ''}"
        return False, ""
    if intent == "ventilator":
        if _is_present(vent):
            n = vent.get("count")
            return True, f"Ventilator{'s' if (n or 0) != 1 else ''} on site{f' (n={n})' if n else ''}"
        return False, ""
    if intent == "dialysis":
        if _is_present(dial):
            n = dial.get("machine_count")
            return True, f"Dialysis service{f' ({n} machines)' if n else ''}"
        return False, ""
    if intent == "emergency":
        ec = (emer.get("emergency_care") or "").lower()
        if ec in {"confirmed", "claimed"}:
            return True, "24/7 emergency care" if emer.get("is_24_7") else "Emergency care available"
        return False, ""
    if intent == "trauma":
        if (emer.get("trauma_capability") or "").lower() in {"confirmed", "claimed"}:
            return True, "Trauma capability"
        return False, ""
    if intent == "surgery":
        s_keys = ("general_surgery", "appendectomy", "caesarean", "orthopedic", "cardiac")
        for k in s_keys:
            if (surg.get(k) or "").lower() in {"confirmed", "claimed"}:
                return True, f"Surgery: {k.replace('_', ' ')}"
        return False, ""
    if intent == "ambulance":
        if (emer.get("ambulance") or "").lower() in {"confirmed", "claimed"}:
            return True, "Ambulance service"
        return False, ""
    if intent in {"maternity", "cardiac", "orthopedic", "neonatal", "dentist", "pharmacy"}:
        # match against specialty list
        if any(intent in s for s in specs):
            return True, f"Listed specialty: {intent}"
        if intent == "neonatal" and (icu.get("neonatal_icu") or "").lower() in {"confirmed", "claimed"}:
            return True, "Neonatal ICU"
        if intent == "cardiac" and (surg.get("cardiac") or "").lower() in {"confirmed", "claimed"}:
            return True, "Cardiac surgery available"
        return False, ""
    return False, ""


def _fallback_rank(
    hits: list[VectorHit],
    summaries: list[dict[str, Any]],
    user_query: str,
    top_k_f: int,
) -> list[dict[str, Any]]:
    """Build ranked_results without an LLM, using vector similarity + capability matching.

    Each result's suitability_score is a blend of:
      0.55 * vector similarity   (semantic relevance to the query)
      0.30 * trust_score          (data quality / reliability)
      0.15 * intent_coverage      (fraction of asked-for capabilities present)

    The reasoning text is auto-generated from the matched capabilities; this is
    obviously less nuanced than an LLM but it's *truthful* (every claim cites
    real fields from the gold extraction) and good enough to keep the public
    demo functional when the LLM endpoint is unavailable.
    """
    intents = _query_intents(user_query)
    ranked: list[tuple[float, dict[str, Any]]] = []

    summary_by_id = {s["facility_id"]: s for s in summaries}

    for hit in hits:
        summary = summary_by_id.get(hit.facility_id)
        if summary is None:
            continue
        extraction = json.loads(hit.metadata.get("extraction_json") or "{}")
        trust = float(summary.get("trust_score") or 0.0)
        sim = max(0.0, min(1.0, float(hit.score)))

        matched: list[str] = []
        warnings: list[str] = []
        evidence: list[str] = []
        for intent in intents:
            ok, phrase = _capability_satisfied(extraction, intent)
            if ok:
                matched.append(intent)
                if phrase:
                    evidence.append(phrase)
            else:
                warnings.append(f"No clear evidence of {intent}")

        intent_coverage = (len(matched) / len(intents)) if intents else 0.5
        score = 0.55 * sim + 0.30 * trust + 0.15 * intent_coverage

        # Auto-reasoning: lead with what matched, then a trust note.
        if evidence:
            reasoning = (
                f"Top match for your query — {summary.get('name') or 'facility'} reports: "
                f"{'; '.join(evidence[:4])}. "
                f"Trust score {trust:.2f} based on data completeness and consistency checks."
            )
        elif intents:
            reasoning = (
                f"Closest semantic match by description (similarity {sim:.2f}); however, the "
                f"facility's structured records do not explicitly confirm "
                f"{', '.join(intents)}. Verify by phone before travel. Trust score {trust:.2f}."
            )
        else:
            reasoning = (
                f"Closest semantic match (similarity {sim:.2f}). Trust score {trust:.2f}."
            )

        ranked.append((
            score,
            {
                "rank": 0,  # set after sort
                "facility_id": hit.facility_id,
                "facility_name": summary.get("name") or "Unknown facility",
                "suitability_score": round(min(1.0, max(0.0, score)), 4),
                "reasoning": reasoning,
                "matched_capabilities": matched,
                "warnings": warnings if not matched else [],
                "citations": [],
            },
        ))

    ranked.sort(key=lambda x: x[0], reverse=True)
    out: list[dict[str, Any]] = []
    for i, (_, item) in enumerate(ranked[:top_k_f], start=1):
        item["rank"] = i
        out.append(item)
    return out


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
        llm_failed = False
        llm_error: str | None = None
        resp = None
        with span("llm_reasoning", candidates=len(summaries), top_k_final=top_k_f):
            try:
                data, resp = client.complete_json(messages, temperature=0.1, max_tokens=2000)
            except LLMError as e:
                llm_failed = True
                llm_error = str(e)
                LOGGER.warning("LLM ranker unavailable; falling back to heuristic ranker: %s", e)
                data = None

        if llm_failed or data is None:
            ranked = _fallback_rank(hits, summaries, user_query, top_k_f)
            return {
                "query": user_query,
                "query_interpretation": (
                    f"Heuristic ranker (LLM unavailable). Detected intents: "
                    f"{', '.join(_query_intents(user_query)) or 'general search'}."
                ),
                "ranked_results": ranked,
                "recommendation_summary": (
                    f"Top {len(ranked)} of {len(summaries)} candidates ranked by vector "
                    f"similarity, structured trust score, and capability match. The LLM "
                    f"reasoning service is currently unreachable; results below are "
                    f"deterministic and grounded in the gold extraction records."
                ),
                "uncertainty_note": (
                    f"LLM error: {llm_error}. Reasoning fields are auto-generated from "
                    f"structured capabilities; please verify by phone before travel."
                ),
                "candidates_retrieved": len(summaries),
                "trust_threshold": min_trust,
                "filters": {
                    "state": state_filter,
                    "city": city_filter,
                    "facility_type": facility_type_filter,
                },
                "fallback_mode": True,
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
                    "prompt": resp.prompt_tokens if resp else 0,
                    "completion": resp.completion_tokens if resp else 0,
                },
                "fallback_mode": False,
            }
        )
        return parsed


__all__ = ["query_facilities"]
