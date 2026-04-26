"""Hugging Face Space entry point for the public Sehat-e-Aam FastAPI mirror.

The Databricks App at ``*.databricksapps.com`` is workspace-authenticated, so a
public Lovable / external React frontend cannot call it directly. This Space
re-runs the same FastAPI code (``src/sehat/api/server.py``) on a public URL
with permissive CORS, while still calling the **Databricks Foundation Model
API** for the LLM (via ``DATABRICKS_TOKEN`` set as a Space secret).

What this script does on cold start:
  1. Stages the Silver / Gold parquets shipped in the Docker image into a
     writable lakehouse directory (``/tmp/lakehouse``).
  2. Rebuilds the FAISS vector index from the Gold parquet (we don't ship a
     prebuilt ``.faiss`` because the file is small and the embedder pulls a
     130 MB sentence-transformers model on first import anyway).
  3. Rebuilds the medical-deserts aggregate from Gold so ``/api/deserts``
     answers immediately.
  4. Boots uvicorn on the port HF Spaces injects via ``$PORT`` (default 7860).
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
from pathlib import Path

LOGGER = logging.getLogger("sehat.space")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
SRC = REPO_ROOT / "src"
DEMO = REPO_ROOT / "demo"

# Use writable scratch space inside the Space container; HF Spaces home dir
# (/home/user) is writable by the runtime user but `/data` is not on the free
# tier. ``/tmp`` is always writable and survives the lifetime of the process.
SCRATCH = Path(os.environ.get("SEHAT_SCRATCH", "/tmp/sehat"))
LAKEHOUSE = SCRATCH / "lakehouse"
VECTOR_INDEX = SCRATCH / "vector_index"
MLRUNS = SCRATCH / "mlruns"


def _stage_artifacts() -> None:
    """Copy the parquets we ship with the image into the writable lakehouse."""
    LAKEHOUSE.mkdir(parents=True, exist_ok=True)
    VECTOR_INDEX.mkdir(parents=True, exist_ok=True)
    MLRUNS.mkdir(parents=True, exist_ok=True)

    src_silver = DEMO / "facilities_silver.parquet"
    src_gold = DEMO / "facilities_gold.parquet"
    dst_silver = LAKEHOUSE / "facilities_silver.parquet"
    dst_gold = LAKEHOUSE / "facilities_gold.parquet"

    for src, dst in ((src_silver, dst_silver), (src_gold, dst_gold)):
        if not src.exists():
            raise FileNotFoundError(
                f"Expected shipped artifact missing: {src}. "
                f"Did the Docker build copy demo/*.parquet?"
            )
        if not dst.exists():
            shutil.copy2(src, dst)
            LOGGER.info("Staged %s -> %s (%d bytes)", src.name, dst, dst.stat().st_size)


def _configure_env() -> None:
    """Set every env var the FastAPI server reads before we import it."""
    sys.path.insert(0, str(SRC))

    os.environ.setdefault("LAKEHOUSE_DIR", str(LAKEHOUSE))
    os.environ.setdefault("VECTOR_INDEX_DIR", str(VECTOR_INDEX))
    os.environ.setdefault("DATA_DIR", str(LAKEHOUSE))
    os.environ.setdefault("MLFLOW_TRACKING_URI", str(MLRUNS))

    os.environ.setdefault("LLM_BACKEND", "databricks")
    os.environ.setdefault("LLM_MODEL", "databricks-meta-llama-3-3-70b-instruct")
    os.environ.setdefault("EMBEDDING_BACKEND", "local")
    os.environ.setdefault("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

    # Where sentence-transformers / huggingface_hub cache models. /tmp is
    # writable; /home/user/.cache also works on HF Spaces.
    os.environ.setdefault("HF_HOME", "/tmp/hf_home")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/hf_home/transformers")
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", "/tmp/hf_home/sentence_transformers")

    if not os.environ.get("DATABRICKS_TOKEN"):
        LOGGER.warning(
            "DATABRICKS_TOKEN is not set. /api/query will fail when it tries "
            "to call the Databricks Foundation Model API. Add it as a Space "
            "secret in Settings -> Repository secrets."
        )
    if not os.environ.get("DATABRICKS_HOST"):
        LOGGER.warning(
            "DATABRICKS_HOST is not set. Add it as a Space variable, e.g. "
            "https://<your-workspace>.cloud.databricks.com"
        )


def _build_indexes() -> None:
    """Rebuild FAISS + deserts from the Gold parquet (~30 s cold start)."""
    from sehat.config import get_settings  # noqa: WPS433 (late import on purpose)
    from sehat.pipeline.deserts import run_deserts
    from sehat.pipeline.vector_search import FacilityVectorIndex

    settings = get_settings()
    LOGGER.info("Settings resolved: lakehouse=%s vector=%s", settings.lakehouse_dir, settings.vector_index_dir)

    if not settings.vector_index_path.exists():
        LOGGER.info("Building FAISS index from Gold (one-time, ~30 s)...")
        FacilityVectorIndex(settings).build()
    else:
        LOGGER.info("FAISS index already present, skipping rebuild.")

    if not settings.deserts_path.exists():
        LOGGER.info("Building medical-deserts aggregate from Gold...")
        run_deserts(settings)
    else:
        LOGGER.info("Deserts parquet already present, skipping rebuild.")


def main() -> None:
    _stage_artifacts()
    _configure_env()
    _build_indexes()

    import uvicorn
    from sehat.api.server import app  # noqa: WPS433 (must be after env setup)

    port = int(os.environ.get("PORT", "7860"))
    LOGGER.info("Starting uvicorn on 0.0.0.0:%d", port)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
