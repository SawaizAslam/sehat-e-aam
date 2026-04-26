"""Notebook 02 equivalent: LLM extraction of structured capabilities.

Fixes vs the original guide:

* Uses JSON-mode LLM calls (no markdown-fence parsing hack).
* Validates LLM output with the ``FacilityExtraction`` Pydantic model.
* Resumable: rows already present in Silver are skipped.
* Failed extractions are persisted to a sibling parquet (``silver_failures``)
  so a retry pass can target only them.
* Concurrency via ``ThreadPoolExecutor`` with a configurable worker count.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import ValidationError
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from ..config import Settings, get_settings
from ..llm import LLMClient, LLMError
from ..prompts import EXTRACTION_SYSTEM_PROMPT, EXTRACTION_USER_TEMPLATE
from ..schemas import FacilityExtraction
from ..storage import parquet_exists, read_parquet, upsert_parquet
from ..tracing import init_tracing, log_metrics, run

LOGGER = logging.getLogger(__name__)
console = Console()


# ---------------------------------------------------------------------------
# Internal types
# ---------------------------------------------------------------------------


@dataclass
class _ExtractionInput:
    facility_id: str
    composite_text: str


@dataclass
class _ExtractionResult:
    facility_id: str
    ok: bool
    extraction: dict[str, Any] | None = None
    error: str | None = None
    raw_response: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0


def _failures_path(silver_path: Path) -> Path:
    return silver_path.with_name(silver_path.stem + "_failures.parquet")


def _build_messages(item: _ExtractionInput) -> list[dict[str, str]]:
    truncated = item.composite_text[:3000]
    return [
        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
        {"role": "user", "content": EXTRACTION_USER_TEMPLATE.format(composite_text=truncated)},
    ]


def _extract_one(item: _ExtractionInput, client: LLMClient) -> _ExtractionResult:
    try:
        data, resp = client.complete_json(_build_messages(item))
    except LLMError as e:
        return _ExtractionResult(facility_id=item.facility_id, ok=False, error=str(e))
    except Exception as e:  # network errors, timeouts, etc.
        return _ExtractionResult(facility_id=item.facility_id, ok=False, error=f"transport: {e}")

    try:
        validated = FacilityExtraction.parse_relaxed(data, facility_id=item.facility_id)
    except ValidationError as e:
        return _ExtractionResult(
            facility_id=item.facility_id,
            ok=False,
            error=f"validation: {e}",
            raw_response=resp.content[:1000],
        )

    return _ExtractionResult(
        facility_id=item.facility_id,
        ok=True,
        extraction=validated.model_dump(mode="json"),
        prompt_tokens=resp.prompt_tokens,
        completion_tokens=resp.completion_tokens,
        latency_ms=resp.latency_ms,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_extract(
    settings: Settings | None = None,
    *,
    only_missing: bool = True,
) -> pd.DataFrame:
    """Run extraction on Bronze and write Silver. Resumable when ``only_missing`` is True."""

    s = settings or get_settings()
    init_tracing()

    if not parquet_exists(s.bronze_path):
        raise FileNotFoundError(
            f"Bronze parquet not found at {s.bronze_path}. Run `sehat ingest` first."
        )

    bronze = read_parquet(s.bronze_path)
    bronze = bronze[bronze["composite_text"].astype(str).str.len() > 0].copy()

    already: set[str] = set()
    if only_missing and parquet_exists(s.silver_path):
        already = set(read_parquet(s.silver_path)["facility_id"].astype(str).unique())
        console.log(f"Resume: {len(already):,} already in Silver, will be skipped.")

    pending = bronze[~bronze["facility_id"].isin(already)]

    if s.extract_sample_limit > 0:
        pending = pending.head(s.extract_sample_limit)
        console.log(f"EXTRACT_SAMPLE_LIMIT={s.extract_sample_limit}; truncated to {len(pending):,} rows.")

    if pending.empty:
        console.log(":white_check_mark: Nothing to extract.")
        return read_parquet(s.silver_path) if parquet_exists(s.silver_path) else pd.DataFrame()

    items = [
        _ExtractionInput(
            facility_id=row.facility_id,
            composite_text=row.composite_text or "",
        )
        for row in pending.itertuples(index=False)
    ]

    client = LLMClient(s)

    successes: list[_ExtractionResult] = []
    failures: list[_ExtractionResult] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    flush_every = max(1, min(s.extract_batch_size, 200))

    with run(
        "extract",
        total_records=len(items),
        batch_size=s.extract_batch_size,
        max_workers=s.extract_max_workers,
        model=s.llm_model,
        backend=s.llm_backend,
    ):
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("ok={task.fields[ok]}  fail={task.fields[fail]}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Extracting", total=len(items), ok=0, fail=0)

            with ThreadPoolExecutor(max_workers=s.extract_max_workers) as pool:
                futures = {pool.submit(_extract_one, item, client): item for item in items}
                buffered: list[_ExtractionResult] = []
                for fut in as_completed(futures):
                    result = fut.result()
                    buffered.append(result)
                    if result.ok:
                        successes.append(result)
                        total_prompt_tokens += result.prompt_tokens
                        total_completion_tokens += result.completion_tokens
                    else:
                        failures.append(result)
                        LOGGER.warning(
                            "Extraction failed for %s: %s", result.facility_id[:12], result.error
                        )

                    progress.update(
                        task,
                        advance=1,
                        ok=len(successes),
                        fail=len(failures),
                    )

                    if len(buffered) >= flush_every:
                        _flush(buffered, s)
                        buffered = []

                if buffered:
                    _flush(buffered, s)

        log_metrics(
            successful_extractions=len(successes),
            failed_extractions=len(failures),
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
        )

    console.log(
        f":white_check_mark: Extraction complete. ok={len(successes):,} fail={len(failures):,}"
    )
    return read_parquet(s.silver_path)


def _flush(results: list[_ExtractionResult], s: Settings) -> None:
    """Write a batch of results to Silver / failures parquets."""

    ok_rows = [
        {
            "facility_id": r.facility_id,
            "extraction_json": json.dumps(r.extraction, ensure_ascii=False),
            "prompt_tokens": r.prompt_tokens,
            "completion_tokens": r.completion_tokens,
            "latency_ms": r.latency_ms,
        }
        for r in results
        if r.ok
    ]
    fail_rows = [
        {
            "facility_id": r.facility_id,
            "error": r.error or "",
            "raw_response": r.raw_response or "",
        }
        for r in results
        if not r.ok
    ]
    if ok_rows:
        upsert_parquet(pd.DataFrame(ok_rows), s.silver_path, key="facility_id")
    if fail_rows:
        upsert_parquet(pd.DataFrame(fail_rows), _failures_path(s.silver_path), key="facility_id")


__all__ = ["run_extract"]


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_extract()
