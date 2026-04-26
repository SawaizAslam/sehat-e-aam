"""MLflow tracing helpers.

Local-friendly: defaults to file-backed tracking under ``./mlruns``. If a
remote tracking URI is supplied via ``MLFLOW_TRACKING_URI`` it is used
unchanged.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager, nullcontext
from typing import Any, Iterator

import mlflow

from .config import get_settings

LOGGER = logging.getLogger(__name__)
_INITIALISED = False
_TRACING_OK = False


def _resolve_experiment(name: str) -> str:
    """Make ``name`` valid in the active MLflow backend.

    On Databricks the tracking server requires experiment names to be absolute
    workspace paths (e.g. ``/Users/<me>/mlflow-experiments/sehat_e_aam``).
    Locally MLflow accepts bare strings. ``MLFLOW_EXPERIMENT_NAME`` overrides
    everything.

    We deliberately put the experiment in a sub-folder (``mlflow-experiments``)
    so it never collides with a Git folder / Repo named the same as the
    project at ``/Users/<me>/<project>``.
    """

    override = os.environ.get("MLFLOW_EXPERIMENT_NAME_OVERRIDE")
    if override:
        return override

    if name.startswith("/"):
        return name

    in_databricks = bool(
        os.environ.get("DATABRICKS_RUNTIME_VERSION")
        or os.environ.get("DATABRICKS_HOST")
        or os.environ.get("DB_HOME")
    )
    if not in_databricks:
        return name

    user = (
        os.environ.get("DATABRICKS_USER_NAME")
        or os.environ.get("USER_NAME")
        or os.environ.get("USER")
    )
    if user:
        return f"/Users/{user}/mlflow-experiments/{name}"

    return f"/Shared/mlflow-experiments/{name}"


def init_tracing(experiment: str = "sehat_e_aam") -> None:
    """Best-effort MLflow init.

    On Databricks the workspace runtime auto-sets ``MLFLOW_EXPERIMENT_NAME``
    to the executing notebook's workspace path, so MLflow falls back to that
    when ``set_experiment`` fails — which itself fails when the notebook lives
    inside a Git folder (REPO node), because MLflow can't create child
    experiment nodes under a REPO. To stay robust we strip that env var and
    rely solely on our explicit experiment path; if anything still fails we
    fully disable tracing for the rest of the process.
    """

    global _INITIALISED, _TRACING_OK
    if _INITIALISED:
        return
    _INITIALISED = True

    os.environ.pop("MLFLOW_EXPERIMENT_NAME", None)
    os.environ.pop("MLFLOW_EXPERIMENT_ID", None)

    settings = get_settings()
    try:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        resolved = _resolve_experiment(experiment)
        mlflow.set_experiment(resolved)
        _TRACING_OK = True
    except Exception as e:  # pragma: no cover - depends on backend
        LOGGER.warning("MLflow init failed (%s); tracing disabled.", e)
        _TRACING_OK = False


@contextmanager
def run(name: str, **params: Any) -> Iterator[Any]:
    """Start an MLflow run if tracing is healthy, else yield a no-op."""

    init_tracing()
    if not _TRACING_OK:
        with nullcontext(None) as s:
            yield s
        return
    try:
        with mlflow.start_run(run_name=name) as active:
            for k, v in params.items():
                try:
                    mlflow.log_param(k, v)
                except Exception:  # pragma: no cover - mlflow non-fatal
                    LOGGER.debug("Failed to log param %s=%s", k, v)
            yield active
    except Exception as e:  # pragma: no cover - server-side failures
        LOGGER.warning("mlflow.start_run failed (%s); proceeding without it.", e)
        with nullcontext(None) as s:
            yield s


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[Any]:
    """Best-effort span. Falls back to a no-op if MLflow tracing is unavailable."""

    init_tracing()
    if not _TRACING_OK:
        with nullcontext(None) as s:
            yield s
        return
    try:
        with mlflow.start_span(name=name, attributes=attributes) as s:
            yield s
    except Exception:  # pragma: no cover
        with nullcontext(None) as s:
            yield s


def log_metrics(**metrics: float) -> None:
    init_tracing()
    if not _TRACING_OK:
        return
    for k, v in metrics.items():
        try:
            mlflow.log_metric(k, float(v))
        except Exception:
            LOGGER.debug("Failed to log metric %s=%s", k, v)


def log_text(content: str, artifact_file: str) -> None:
    init_tracing()
    if not _TRACING_OK:
        return
    try:
        mlflow.log_text(content, artifact_file)
    except Exception:
        LOGGER.debug("Failed to log text artifact %s", artifact_file)


__all__ = ["init_tracing", "run", "span", "log_metrics", "log_text"]
