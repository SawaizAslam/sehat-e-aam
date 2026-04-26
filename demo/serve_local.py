"""Serve the Sehat-e-Aam FastAPI app locally against the artifacts pulled
from the Databricks Volume in `demo/`.

Why: Databricks Apps require workspace-user OAuth, which is awkward to inject
from a Lovable / external React frontend during a demo. Running the same
FastAPI code locally with permissive CORS gives Lovable a frictionless target.

Prereqs (already present in the .venv used for this project):
  pip install fastapi uvicorn pandas duckdb pyarrow faiss-cpu \
              sentence-transformers databricks-sdk openai mlflow rich pydantic-settings

Usage:
  .venv\\Scripts\\python.exe demo\\serve_local.py
  # then: http://localhost:8000/docs
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
SRC = PROJECT / "src"
sys.path.insert(0, str(SRC))

DEMO_LAKEHOUSE = ROOT  # all parquets we pulled live here flat
DEMO_VECTOR = ROOT / "vector_index"
DEMO_VECTOR.mkdir(exist_ok=True)

os.environ.setdefault("LAKEHOUSE_DIR", str(DEMO_LAKEHOUSE))
os.environ.setdefault("VECTOR_INDEX_DIR", str(DEMO_VECTOR))
os.environ.setdefault("DATA_DIR", str(DEMO_LAKEHOUSE))
os.environ.setdefault("LLM_BACKEND", "databricks")
os.environ.setdefault("LLM_MODEL", "databricks-meta-llama-3-3-70b-instruct")
os.environ.setdefault("EMBEDDING_BACKEND", "local")
os.environ.setdefault("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
os.environ.setdefault("MLFLOW_TRACKING_URI", str(ROOT / "mlruns"))

DOTENV = PROJECT / ".env"
if DOTENV.exists():
    for raw in DOTENV.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

if __name__ == "__main__":
    import uvicorn

    print("Sehat-e-Aam local API")
    print(f"  LAKEHOUSE_DIR    = {os.environ['LAKEHOUSE_DIR']}")
    print(f"  VECTOR_INDEX_DIR = {os.environ['VECTOR_INDEX_DIR']}")
    print(f"  LLM_BACKEND      = {os.environ['LLM_BACKEND']}")
    print(f"  LLM_MODEL        = {os.environ['LLM_MODEL']}")
    print()
    print("Open: http://localhost:8000/docs")
    print()

    from sehat.api.server import app  # noqa: E402

    uvicorn.run(app, host="0.0.0.0", port=8000)
