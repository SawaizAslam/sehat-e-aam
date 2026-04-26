"""Command-line entry point: ``sehat <command>``."""

from __future__ import annotations

import json
import logging

import typer
from rich.console import Console

from .config import get_settings
from .pipeline.deserts import run_deserts
from .pipeline.extract import run_extract
from .pipeline.ingest import run_ingest
from .pipeline.reasoning import query_facilities
from .pipeline.self_correct import run_self_correction
from .pipeline.trust_score import run_trust_scoring
from .pipeline.vector_search import run_index

app = typer.Typer(help="Sehat-e-Aam pipeline CLI", no_args_is_help=True)
console = Console()


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@app.callback()
def _main(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    _setup_logging(verbose)


@app.command()
def info() -> None:
    """Show current configuration."""

    s = get_settings()
    payload = {
        "llm_backend": s.llm_backend,
        "llm_model": s.llm_model,
        "embedding_backend": s.embedding_backend,
        "embedding_model": s.embedding_model,
        "raw_dataset_path": str(s.raw_dataset_path),
        "lakehouse_dir": str(s.lakehouse_dir),
        "vector_index_dir": str(s.vector_index_dir),
        "extract_sample_limit": s.extract_sample_limit,
        "correction_trigger_trust": s.correction_trigger_trust,
    }
    console.print_json(json.dumps(payload))


@app.command()
def ingest() -> None:
    """Read the raw CSV/XLSX and write the Bronze parquet table."""

    run_ingest()


@app.command()
def extract(
    only_missing: bool = typer.Option(True, help="Skip facility_ids already in Silver."),
) -> None:
    """LLM-extract structured capabilities into Silver."""

    run_extract(only_missing=only_missing)


@app.command()
def trust() -> None:
    """Apply trust rules + confidence and write the Gold parquet table."""

    run_trust_scoring()


@app.command(name="self-correct")
def self_correct() -> None:
    """Run the validate -> correct loop on low-trust records."""

    run_self_correction()


@app.command()
def index() -> None:
    """Build the FAISS vector index over Gold."""

    run_index()


@app.command()
def deserts() -> None:
    """Aggregate medical desert risk by PIN code."""

    run_deserts()


@app.command()
def pipeline(
    skip_extract: bool = typer.Option(False, help="Skip the LLM extraction step."),
    skip_self_correct: bool = typer.Option(False, help="Skip the self-correction step."),
) -> None:
    """Run the full pipeline end-to-end."""

    run_ingest()
    if not skip_extract:
        run_extract()
    run_trust_scoring()
    if not skip_self_correct:
        run_self_correction()
    run_index()
    run_deserts()
    console.log(":sparkles: Pipeline complete.")


@app.command()
def query(
    text: str = typer.Argument(..., help="Free-text medical query."),
    state: str | None = typer.Option(None),
    city: str | None = typer.Option(None),
    min_trust: float | None = typer.Option(None, "--min-trust"),
    top_k: int = typer.Option(5, "--top-k"),
) -> None:
    """Run the reasoning engine from the command line."""

    result = query_facilities(
        user_query=text,
        state_filter=state,
        city_filter=city,
        min_trust_score=min_trust,
        top_k_final=top_k,
    )
    console.print_json(json.dumps(result, default=str))


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8000),
    reload: bool = typer.Option(False, help="Enable hot reload (dev only)."),
) -> None:
    """Run the FastAPI server."""

    import uvicorn

    uvicorn.run("sehat.api.server:app", host=host, port=port, reload=reload)


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
