"""Notebook 04 equivalent: Extract -> Validate -> Correct loop.

Fixes vs the original guide:

* JSON-mode for both validator and corrector calls.
* On a successful correction we **re-compute trust score AND confidence**
  before merging into Gold (the original guide skipped confidence).
* Only runs on rows below ``CORRECTION_TRIGGER_TRUST`` to save tokens.
* All artefacts persisted via parquet upsert (idempotent).
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd
from pydantic import ValidationError
from rich.console import Console
from rich.progress import Progress, BarColumn, MofNCompleteColumn, TextColumn, TimeElapsedColumn

from ..config import Settings, get_settings
from ..llm import LLMClient, LLMError
from ..prompts import CORRECTOR_SYSTEM_PROMPT, VALIDATOR_SYSTEM_PROMPT
from ..schemas import FacilityExtraction, FacilityType
from ..storage import parquet_exists, read_parquet, upsert_parquet, write_parquet
from ..tracing import init_tracing, run, span
from .trust_score import apply_trust_rules, build_embedding_text, compute_confidence

LOGGER = logging.getLogger(__name__)
console = Console()


@dataclass
class _CorrectionOutcome:
    facility_id: str
    iterations: int
    final_extraction: dict[str, Any]
    validator_reports: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Validator + corrector
# ---------------------------------------------------------------------------


def _validate(client: LLMClient, extraction: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    context = {
        "facility_name": meta.get("name", "Unknown"),
        "facility_type": meta.get("facility_type", "unknown"),
        "city": meta.get("address_city", ""),
        "state": meta.get("address_state", ""),
        "number_doctors": meta.get("number_doctors", "unknown"),
        "extraction_summary": {
            k: extraction.get(k, {}) for k in ("icu", "ventilator", "staff", "emergency", "surgery", "dialysis")
        },
    }
    messages = [
        {"role": "system", "content": VALIDATOR_SYSTEM_PROMPT},
        {"role": "user", "content": f"Review this facility extraction:\n\n{json.dumps(context, indent=2, default=str)}"},
    ]
    try:
        data, _ = client.complete_json(messages, max_tokens=512)
        return data
    except (LLMError, Exception) as e:  # pragma: no cover - LLM dependent
        LOGGER.warning("Validator failed for %s: %s", meta.get("facility_id", "?"), e)
        return {
            "has_contradictions": False,
            "contradiction_flags": [],
            "validator_notes": f"validator_error: {e}",
            "recommend_reextraction": False,
        }


def _correct(
    client: LLMClient,
    facility_id: str,
    composite_text: str,
    extraction: dict[str, Any],
    validator_report: dict[str, Any],
) -> dict[str, Any] | None:
    user_msg = f"""Original facility text:
{composite_text[:2000]}

Validator report:
{json.dumps(validator_report, indent=2)}

Current extraction (to be corrected):
{json.dumps(extraction, indent=2)}

Produce the corrected extraction JSON:"""

    messages = [
        {"role": "system", "content": CORRECTOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    try:
        data, _ = client.complete_json(messages, max_tokens=2048)
    except (LLMError, Exception) as e:  # pragma: no cover - LLM dependent
        LOGGER.warning("Corrector LLM failed for %s: %s", facility_id, e)
        return None

    try:
        validated = FacilityExtraction.parse_relaxed(data, facility_id=facility_id)
    except ValidationError as e:
        LOGGER.warning("Corrector output failed validation for %s: %s", facility_id, e)
        return None
    return validated.model_dump(mode="json")


def run_loop_for_record(
    client: LLMClient,
    facility_id: str,
    composite_text: str,
    initial_extraction: dict[str, Any],
    meta: dict[str, Any],
    *,
    max_iterations: int,
) -> _CorrectionOutcome:
    current = copy.deepcopy(initial_extraction)
    reports: list[dict[str, Any]] = []
    iters = 0

    for i in range(max_iterations):
        with span(f"validate_{i}", facility_id=facility_id, iteration=i):
            report = _validate(client, current, meta | {"facility_id": facility_id})
        reports.append(report)
        if not report.get("recommend_reextraction"):
            break
        with span(f"correct_{i}", facility_id=facility_id, iteration=i):
            corrected = _correct(client, facility_id, composite_text, current, report)
        if corrected is None:
            break
        current = corrected
        iters = i + 1

    return _CorrectionOutcome(
        facility_id=facility_id,
        iterations=iters,
        final_extraction=current,
        validator_reports=reports,
    )


# ---------------------------------------------------------------------------
# Pipeline driver
# ---------------------------------------------------------------------------


def run_self_correction(settings: Settings | None = None) -> pd.DataFrame:
    s = settings or get_settings()
    init_tracing()

    if not parquet_exists(s.gold_path):
        raise FileNotFoundError("Gold parquet missing; run `sehat trust` first.")
    if not parquet_exists(s.bronze_path):
        raise FileNotFoundError("Bronze parquet missing; run `sehat ingest` first.")

    gold = read_parquet(s.gold_path)
    bronze = read_parquet(s.bronze_path)[["facility_id", "composite_text"]]

    pending = gold[gold["trust_score"] < s.correction_trigger_trust]
    pending = pending.merge(bronze, on="facility_id", how="left")
    if s.correction_sample_limit > 0:
        pending = pending.head(s.correction_sample_limit)

    console.log(
        f"Self-correction candidates: {len(pending):,} "
        f"(trust < {s.correction_trigger_trust}, capped at {s.correction_sample_limit})"
    )

    if pending.empty:
        return gold

    client = LLMClient(s)
    updates: list[dict[str, Any]] = []

    with run("self_correction", candidates=len(pending), max_iterations=s.correction_max_iterations):
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("corrected={task.fields[corrected]}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Self-correcting", total=len(pending), corrected=0)

            corrected_n = 0
            for _, row in pending.iterrows():
                try:
                    extraction = json.loads(row["extraction_json"])
                except json.JSONDecodeError:
                    progress.update(task, advance=1)
                    continue

                meta = {
                    "name": row.get("name"),
                    "facility_type": row.get("facility_type"),
                    "address_city": row.get("address_city"),
                    "address_state": row.get("address_state"),
                    "number_doctors": row.get("operator_type"),
                }
                outcome = run_loop_for_record(
                    client,
                    facility_id=row["facility_id"],
                    composite_text=row.get("composite_text") or "",
                    initial_extraction=extraction,
                    meta=meta,
                    max_iterations=s.correction_max_iterations,
                )

                if outcome.iterations == 0:
                    progress.update(task, advance=1)
                    continue

                facility_type = FacilityType.normalise(row.get("facility_type"))
                composite_len = len(row.get("composite_text") or "")
                trust_score, flags = apply_trust_rules(
                    outcome.final_extraction,
                    facility_type=facility_type,
                    number_doctors=None,
                    composite_text_length=composite_len,
                )
                confidence = compute_confidence(
                    outcome.final_extraction,
                    flags=flags,
                    composite_text_length=composite_len,
                )
                embedding_text = build_embedding_text(
                    name=str(row.get("name") or ""),
                    city=row.get("address_city"),
                    state=row.get("address_state"),
                    pin_code=row.get("address_zip"),
                    facility_type=facility_type,
                    extraction=outcome.final_extraction,
                    trust_score=trust_score,
                )

                updates.append(
                    {
                        "facility_id": row["facility_id"],
                        "extraction_json": json.dumps(outcome.final_extraction, ensure_ascii=False),
                        "trust_score": trust_score,
                        "trust_flags_json": json.dumps(
                            [f.model_dump(mode="json") for f in flags], ensure_ascii=False
                        ),
                        "confidence_json": confidence.model_dump_json(),
                        "correction_iterations": outcome.iterations,
                        "embedding_text": embedding_text,
                    }
                )
                corrected_n += 1
                progress.update(task, advance=1, corrected=corrected_n)

        # Apply updates by merging into Gold
        if updates:
            updates_df = pd.DataFrame(updates)
            updated_ids = set(updates_df["facility_id"])
            gold_kept = gold[~gold["facility_id"].isin(updated_ids)].copy()
            for col in (
                "extraction_json",
                "trust_score",
                "trust_flags_json",
                "confidence_json",
                "correction_iterations",
                "embedding_text",
            ):
                if col not in updates_df.columns:
                    continue
            merged_rows: list[dict[str, Any]] = []
            for _, row in gold[gold["facility_id"].isin(updated_ids)].iterrows():
                upd = updates_df[updates_df["facility_id"] == row["facility_id"]].iloc[0]
                new_row = row.to_dict()
                for col in (
                    "extraction_json",
                    "trust_score",
                    "trust_flags_json",
                    "confidence_json",
                    "correction_iterations",
                    "embedding_text",
                ):
                    new_row[col] = upd[col]
                merged_rows.append(new_row)

            new_gold = pd.concat(
                [gold_kept, pd.DataFrame(merged_rows)], ignore_index=True
            ).sort_values("facility_id")
            write_parquet(new_gold, s.gold_path, overwrite=True)
            console.log(f":white_check_mark: Gold updated with {len(updates):,} corrected records")
        else:
            console.log("No corrections applied.")

    return read_parquet(s.gold_path)


__all__ = ["run_self_correction", "run_loop_for_record"]


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_self_correction()
