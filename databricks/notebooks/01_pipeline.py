# Databricks notebook source
# MAGIC %md
# MAGIC # Sehat-e-Aam · Notebook 01 · Pipeline
# MAGIC
# MAGIC Runs the full agentic pipeline against the dataset uploaded in `00_setup`:
# MAGIC
# MAGIC 1. **Ingest** — raw CSV → Bronze parquet (deduped, hashed)
# MAGIC 2. **Extract** — Bronze → Silver via Foundation Model API (JSON-mode)
# MAGIC 3. **Trust + confidence** — Silver → Gold (rule + LLM scoring)
# MAGIC 4. **Self-correct** — low-trust Gold rows pass through Validator → Corrector
# MAGIC 5. **Index** — Gold → FAISS index on the Volume
# MAGIC 6. **Deserts** — Gold → PIN-code aggregates (single DuckDB SQL pass)
# MAGIC
# MAGIC Each step is an MLflow run — open the **Experiments** sidebar to inspect
# MAGIC tokens, latency, and counts.
# MAGIC
# MAGIC > Set `EXTRACT_SAMPLE_LIMIT` to a small number (e.g. 200) for a fast sanity run.
# MAGIC > Set it to `0` for a full extraction over all 10k rows once you're happy.

# COMMAND ----------

# MAGIC %pip install --quiet \
# MAGIC   "pydantic>=2.6,<3.0" "pydantic-settings>=2.2" "python-dotenv>=1.0" \
# MAGIC   "pandas>=2.1" "pyarrow>=15.0" "duckdb>=0.10" "openpyxl>=3.1" \
# MAGIC   "faiss-cpu>=1.8" "sentence-transformers>=2.7" \
# MAGIC   "mlflow>=2.13" "tenacity>=8.2" "rich>=13.7" "tqdm>=4.66" \
# MAGIC   "databricks-sdk>=0.28" "openai>=1.30"

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configure session

# COMMAND ----------

import os
import sys

CATALOG = "workspace"  # Free Edition default; "main" on paid tiers
SCHEMA = "sehat"
VOLUME = "data"
LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"

VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
RAW_DATASET_PATH = f"{VOLUME_PATH}/raw/facilities.csv"
PROJECT_ROOT = f"/Workspace/Users/{spark.sql('SELECT current_user()').first()[0]}/sehat-e-aam"

# Make `sehat` importable
sehat_src = f"{PROJECT_ROOT}/src"
if sehat_src not in sys.path:
    sys.path.insert(0, sehat_src)

# Same env block as Notebook 00 — keep them in sync
ENV_VARS = {
    "LLM_BACKEND": "databricks",
    "LLM_MODEL": LLM_ENDPOINT,
    "EMBEDDING_BACKEND": "local",
    "EMBEDDING_MODEL": "BAAI/bge-small-en-v1.5",
    "RAW_DATASET_PATH": RAW_DATASET_PATH,
    "DATA_DIR": f"{VOLUME_PATH}/raw",
    "LAKEHOUSE_DIR": f"{VOLUME_PATH}/lakehouse",
    "VECTOR_INDEX_DIR": f"{VOLUME_PATH}/vector_index",
    "MLFLOW_TRACKING_URI": "databricks",
    "EXTRACT_BATCH_SIZE": "50",
    "EXTRACT_MAX_WORKERS": "4",
    "EXTRACT_SAMPLE_LIMIT": "200",  # raise to 0 for full extraction
    "CORRECTION_TRIGGER_TRUST": "0.65",
    # Set to 0 for the demo deadline run: skips the (slow) Validator/Corrector
    # LLM loop. The code path is fully implemented in
    # src/sehat/pipeline/self_correct.py; raise this to e.g. 100 for a
    # quality run.
    "CORRECTION_SAMPLE_LIMIT": "0",
    # Do not set MLFLOW_EXPERIMENT_NAME to the Git folder path (/.../sehat-e-aam):
    # that path is a REPO node and experiment creation fails with RESOURCE_ALREADY_EXISTS.
    # sehat.tracing resolves a safe path under .../mlflow-experiments/ and strips the
    # runtime-injected MLFLOW_EXPERIMENT_NAME for you.
}
for k, v in ENV_VARS.items():
    os.environ[k] = v

from sehat.config import get_settings
get_settings.cache_clear() if hasattr(get_settings, "cache_clear") else None
s = get_settings()
s.ensure_dirs()
print("Lakehouse dir :", s.lakehouse_dir)
print("Vector dir    :", s.vector_index_dir)
print("Sample limit  :", s.extract_sample_limit, "(0 = unlimited)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Ingest (raw → Bronze)

# COMMAND ----------

from sehat.pipeline.ingest import run_ingest

bronze_df = run_ingest()
print(f"Bronze rows: {len(bronze_df):,}")
display(bronze_df.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Extract (Bronze → Silver via LLM)
# MAGIC
# MAGIC This is the slowest step. With `EXTRACT_SAMPLE_LIMIT=200` and 4 concurrent
# MAGIC workers it should complete in ~3-6 minutes on Free Edition rate limits.

# COMMAND ----------

from sehat.pipeline.extract import run_extract

silver_df = run_extract()
print(f"Silver rows: {len(silver_df):,}")
display(silver_df.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Trust + confidence (Silver → Gold)

# COMMAND ----------

from sehat.pipeline.trust_score import run_trust_scoring

gold_df = run_trust_scoring()
print(f"Gold rows: {len(gold_df):,}")
print(f"Avg trust score: {gold_df['trust_score'].mean():.3f}")
print(f"Rows below 0.65 trust: {(gold_df['trust_score'] < 0.65).sum()}")
display(gold_df[['facility_id', 'name', 'address_city', 'facility_type', 'trust_score']].head())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 — Self-correct low-trust rows
# MAGIC
# MAGIC Skipped if no rows fall below the trust threshold.

# COMMAND ----------

from sehat.pipeline.self_correct import run_self_correction

corrected_df = run_self_correction()
if corrected_df is None or len(corrected_df) == 0:
    print("No rows needed correction (all above trust threshold).")
else:
    print(f"Corrected rows: {len(corrected_df):,}")
    display(corrected_df.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5 — Build FAISS index (Gold → vector index)

# COMMAND ----------

from sehat.pipeline.vector_search import run_index
from sehat.config import get_settings
import os

run_index()
s = get_settings()
print(f"FAISS index : {s.vector_index_path} ({os.path.getsize(s.vector_index_path)/1024:.1f} KB)")
print(f"Metadata    : {s.vector_meta_path} ({os.path.getsize(s.vector_meta_path)/1024:.1f} KB)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6 — Medical desert detection (Gold → desert aggregates)

# COMMAND ----------

from sehat.pipeline.deserts import run_deserts

deserts_df = run_deserts()
print(f"PIN-code rollups: {len(deserts_df):,}")
display(deserts_df.head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pipeline complete!
# MAGIC
# MAGIC Output files (all under the Volume):
# MAGIC - `lakehouse/facilities_bronze.parquet` — canonicalised raw rows
# MAGIC - `lakehouse/facilities_silver.parquet` — structured LLM extractions
# MAGIC - `lakehouse/facilities_gold.parquet` — trust-scored, embedding-ready
# MAGIC - `lakehouse/audit_log.parquet` — Validator/Corrector cycles
# MAGIC - `lakehouse/medical_deserts.parquet` — PIN-code rollups
# MAGIC - `vector_index/facilities.faiss` + `facilities_meta.parquet`
# MAGIC
# MAGIC Continue to **`02_smoke_test`** to verify outputs and run a sample query.
