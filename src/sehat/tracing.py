"""MLflow tracing helpers.

Local-friendly: defaults to file-backed tracking under ``./mlruns``. If a
remote tracking URI is supplied via ``MLFLOW_TRACKING_URI`` it is used
unchanged.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager, nullcontext
from typing import Any, Iterator

import mlflow

from .config import get_settings

LOGGER = logging.getLogger(__name__)
_INITIALISED = False


def init_tracing(experiment: str = "sehat_e_aam") -> None:
    global _INITIALISED
    if _INITIALISED:
        return
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(experiment)
    _INITIALISED = True


@contextmanager
def run(name: str, **params: Any) -> Iterator[Any]:
    """Start an MLflow run with optional params, yield the active run."""

    init_tracing()
    with mlflow.start_run(run_name=name) as active:
        for k, v in params.items():
            try:
                mlflow.log_param(k, v)
            except Exception:  # pragma: no cover - mlflow non-fatal
                LOGGER.debug("Failed to log param %s=%s", k, v)
        yield active


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[Any]:
    """Best-effort span. Falls back to a no-op if MLflow tracing is unavailable."""

    init_tracing()
    try:
        with mlflow.start_span(name=name, attributes=attributes) as s:
            yield s
    except Exception:  # pragma: no cover
        with nullcontext() as s:
            yield s


def log_metrics(**metrics: float) -> None:
    init_tracing()
    for k, v in metrics.items():
        try:
            mlflow.log_metric(k, float(v))
        except Exception:
            LOGGER.debug("Failed to log metric %s=%s", k, v)


def log_text(content: str, artifact_file: str) -> None:
    init_tracing()
    try:
        mlflow.log_text(content, artifact_file)
    except Exception:
        LOGGER.debug("Failed to log text artifact %s", artifact_file)


__all__ = ["init_tracing", "run", "span", "log_metrics", "log_text"]
