# Databricks notebook source
# MAGIC %md
# MAGIC # Sehat-e-Aam · Notebook 00 · Setup
# MAGIC
# MAGIC One-time setup for the Sehat-e-Aam pipeline on Databricks Free Edition.
# MAGIC
# MAGIC This notebook will:
# MAGIC 1. Install the project as an editable package on serverless compute.
# MAGIC 2. Create a Unity Catalog catalog, schema, and Volume to hold the dataset and Parquet artefacts.
# MAGIC 3. Upload the raw CSV (you must drag-and-drop it into the Volume first — see step 2 below).
# MAGIC 4. Verify the project imports cleanly and the LLM endpoint is reachable.
# MAGIC
# MAGIC **Run this notebook on a Serverless cluster.** No GPU is required.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Configure these values

# COMMAND ----------

# Unity Catalog target.
# On Databricks Free Edition the default catalog is "workspace" (you cannot
# create new catalogs without account-admin perms). On paid tiers you may
# prefer "main".
CATALOG = "workspace"
SCHEMA = "sehat"
VOLUME = "data"

# Foundation Model API endpoint to use for extraction.
# Free Edition exposes Llama-family chat models; pick whichever your workspace lists.
# Common names (verify in Compute > Serving > Endpoints):
#   databricks-meta-llama-3-3-70b-instruct
#   databricks-meta-llama-3-1-70b-instruct
#   databricks-meta-llama-3-1-8b-instruct
LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"
EMBEDDING_ENDPOINT = "databricks-gte-large-en"

# Project root inside the workspace (where you uploaded the repo / connected the Git folder)
# If you imported the project as a Git folder under your home, this is the default.
import os
PROJECT_ROOT = f"/Workspace/Users/{os.environ.get('DATABRICKS_USER_NAME', spark.sql('SELECT current_user()').first()[0])}/sehat-e-aam"

print(f"CATALOG.SCHEMA.VOLUME = {CATALOG}.{SCHEMA}.{VOLUME}")
print(f"LLM_ENDPOINT          = {LLM_ENDPOINT}")
print(f"PROJECT_ROOT          = {PROJECT_ROOT}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Create the catalog, schema, and Volume

# COMMAND ----------

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")

VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
RAW_DIR = f"{VOLUME_PATH}/raw"
LAKEHOUSE_DIR = f"{VOLUME_PATH}/lakehouse"
VECTOR_DIR = f"{VOLUME_PATH}/vector_index"
MLFLOW_DIR = f"{VOLUME_PATH}/mlruns"

for d in (RAW_DIR, LAKEHOUSE_DIR, VECTOR_DIR, MLFLOW_DIR):
    dbutils.fs.mkdirs(d)

print(f"Volume path: {VOLUME_PATH}")
print("Subdirs created: raw/, lakehouse/, vector_index/, mlruns/")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Verify the dataset CSV is in the Volume
# MAGIC
# MAGIC The pipeline expects your CSV at:
# MAGIC
# MAGIC ```
# MAGIC /Volumes/workspace/sehat/data/raw/facilities.csv
# MAGIC ```
# MAGIC
# MAGIC If you uploaded via the **Databricks CLI** (recommended for files >1 MB
# MAGIC because the UI uploader is flaky on Free Edition):
# MAGIC
# MAGIC ```bash
# MAGIC databricks fs cp "<local-csv>" \
# MAGIC   dbfs:/Volumes/workspace/sehat/data/raw/facilities.csv --overwrite
# MAGIC ```
# MAGIC
# MAGIC If you uploaded via the **UI**: Catalog sidebar → `workspace` → `sehat`
# MAGIC → `data` → `raw` → **Upload to this volume** → drop your CSV → rename
# MAGIC to `facilities.csv`.
# MAGIC
# MAGIC The next cell just confirms the file is there.

# COMMAND ----------

RAW_FILENAME = "facilities.csv"
RAW_DATASET_PATH = f"{RAW_DIR}/{RAW_FILENAME}"  # /Volumes/workspace/sehat/data/raw/facilities.csv

if not os.path.exists(RAW_DATASET_PATH):
    raise FileNotFoundError(
        f"Dataset not found at {RAW_DATASET_PATH}.\n\n"
        f"Upload it with the Databricks CLI:\n"
        f"  databricks fs cp <local-csv> "
        f"dbfs:/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/raw/{RAW_FILENAME} --overwrite\n\n"
        f"Or via UI: Catalog -> {CATALOG} -> {SCHEMA} -> {VOLUME} -> raw -> "
        f"Upload, then rename to {RAW_FILENAME}.\n\n"
        f"List what's currently in the raw dir to debug:\n"
        f"  %fs ls /Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/raw"
    )
size_mb = os.path.getsize(RAW_DATASET_PATH) / 1_048_576
print(f"Found dataset: {RAW_DATASET_PATH} ({size_mb:.1f} MB)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 — Install project dependencies and the `sehat` package

# COMMAND ----------

# MAGIC %pip install --quiet \
# MAGIC   "pydantic>=2.6,<3.0" "pydantic-settings>=2.2" "python-dotenv>=1.0" \
# MAGIC   "pandas>=2.1" "pyarrow>=15.0" "duckdb>=0.10" "openpyxl>=3.1" \
# MAGIC   "faiss-cpu>=1.8" "sentence-transformers>=2.7" \
# MAGIC   "mlflow>=2.13" "tenacity>=8.2" "rich>=13.7" "tqdm>=4.66" \
# MAGIC   "databricks-sdk>=0.28" "openai>=1.30" "fastapi>=0.111" "typer>=0.12"

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# Re-declare config after kernel restart
import os
CATALOG = "workspace"
SCHEMA = "sehat"
VOLUME = "data"
LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"
EMBEDDING_ENDPOINT = "databricks-gte-large-en"
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
RAW_DATASET_PATH = f"{VOLUME_PATH}/raw/facilities.csv"

PROJECT_ROOT = f"/Workspace/Users/{spark.sql('SELECT current_user()').first()[0]}/sehat-e-aam"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5 — Add the project source to `sys.path`
# MAGIC
# MAGIC The `sehat` Python package lives in `src/sehat`. We add it to `sys.path`
# MAGIC so the pipeline notebooks can `import sehat` without `pip install -e .`.

# COMMAND ----------

import sys
sehat_src = f"{PROJECT_ROOT}/src"
if sehat_src not in sys.path:
    sys.path.insert(0, sehat_src)

# Also add the parent directory so `tests/` etc. are reachable if needed
project_root = PROJECT_ROOT
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print("sys.path updated:")
for p in sys.path[:5]:
    print(f"  {p}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6 — Configure environment variables for the `sehat` package
# MAGIC
# MAGIC The package reads its config from environment variables. We set them
# MAGIC for this Spark session and also write them to `/Volumes/.../config.env`
# MAGIC so the deployed Databricks App can pick them up.

# COMMAND ----------

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
    "EXTRACT_SAMPLE_LIMIT": "200",  # smoke run; raise to 0 (unlimited) for full extraction
    "CORRECTION_TRIGGER_TRUST": "0.65",
    "CORRECTION_SAMPLE_LIMIT": "100",
}

for k, v in ENV_VARS.items():
    os.environ[k] = v

# Persist to a .env file in the Volume for the app deployment
env_text = "\n".join(f"{k}={v}" for k, v in ENV_VARS.items())
env_path = f"{VOLUME_PATH}/sehat.env"
with open(env_path, "w") as f:
    f.write(env_text)
print(f"Wrote {env_path}")
print()
print(env_text)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7 — Smoke-import the `sehat` package

# COMMAND ----------

import sehat
from sehat.config import get_settings
from sehat.schemas import FacilityType, AvailabilityStatus

s = get_settings()
print("sehat version :", sehat.__version__)
print("LLM backend   :", s.llm_backend)
print("LLM model     :", s.llm_model)
print("Lakehouse dir :", s.lakehouse_dir)
print("Vector dir    :", s.vector_index_dir)
print("Raw dataset   :", s.raw_dataset_path)
print("FacilityType.normalise('farmacy') =", FacilityType.normalise("farmacy"))
print("AvailabilityStatus.NOT_PRESENT =", AvailabilityStatus.NOT_PRESENT.value)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8 — Verify the Foundation Model endpoint
# MAGIC
# MAGIC One quick call to confirm we can reach the LLM and parse JSON output.

# COMMAND ----------

from sehat.llm import LLMClient

client = LLMClient()
data, resp = client.complete_json(
    [
        {"role": "system", "content": 'Return a JSON object: {"ok": true, "model_alive": true}.'},
        {"role": "user", "content": "ping"},
    ],
    max_tokens=64,
)
print("Endpoint:", resp.model)
print("Latency :", f"{resp.latency_ms:.0f} ms")
print("Tokens  :", resp.prompt_tokens, "+", resp.completion_tokens)
print("Output  :", data)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 9 — Quick dataset peek

# COMMAND ----------

import pandas as pd
df = pd.read_csv(RAW_DATASET_PATH, low_memory=False, nrows=5)
print(f"Total columns: {len(df.columns)}")
print("First 5 rows × selected columns:")
display(df[["name", "address_city", "address_stateOrRegion", "facilityTypeId", "description"]].head())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Done!
# MAGIC
# MAGIC If every cell above ran clean, you are ready to run **`01_pipeline`** which
# MAGIC executes Bronze → Silver → Gold → FAISS index → Deserts.
