# Databricks notebook source
# MAGIC %md
# MAGIC # Sehat-e-Aam · Notebook 02 · Smoke test
# MAGIC
# MAGIC Sanity-checks the pipeline outputs and runs the reasoning engine end-to-end
# MAGIC against the real data — no FastAPI required.

# COMMAND ----------

# MAGIC %pip install --quiet \
# MAGIC   "pydantic>=2.6,<3.0" "pydantic-settings>=2.2" "python-dotenv>=1.0" \
# MAGIC   "pandas>=2.1" "pyarrow>=15.0" "duckdb>=0.10" \
# MAGIC   "faiss-cpu>=1.8" "sentence-transformers>=2.7" \
# MAGIC   "mlflow>=2.13" "tenacity>=8.2" "rich>=13.7" "tqdm>=4.66" \
# MAGIC   "databricks-sdk>=0.28" "openai>=1.30"

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import os
import sys

CATALOG = "workspace"  # Free Edition default; "main" on paid tiers
SCHEMA = "sehat"
VOLUME = "data"
LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"

VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
PROJECT_ROOT = f"/Workspace/Users/{spark.sql('SELECT current_user()').first()[0]}/sehat-e-aam"

sehat_src = f"{PROJECT_ROOT}/src"
if sehat_src not in sys.path:
    sys.path.insert(0, sehat_src)

ENV_VARS = {
    "LLM_BACKEND": "databricks",
    "LLM_MODEL": LLM_ENDPOINT,
    "EMBEDDING_BACKEND": "local",
    "EMBEDDING_MODEL": "BAAI/bge-small-en-v1.5",
    "RAW_DATASET_PATH": f"{VOLUME_PATH}/raw/facilities.csv",
    "DATA_DIR": f"{VOLUME_PATH}/raw",
    "LAKEHOUSE_DIR": f"{VOLUME_PATH}/lakehouse",
    "VECTOR_INDEX_DIR": f"{VOLUME_PATH}/vector_index",
    "MLFLOW_TRACKING_URI": "databricks",
}
for k, v in ENV_VARS.items():
    os.environ[k] = v

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Inspect the lakehouse via DuckDB

# COMMAND ----------

from sehat.storage import duck

with duck() as con:
    summary = con.execute("""
        SELECT
            count(*)                                                 AS rows,
            round(avg(trust_score), 3)                               AS avg_trust,
            round(avg(json_extract_string(confidence_json, '$.overall')::DOUBLE), 3) AS avg_conf,
            sum(case when trust_score < 0.65 then 1 else 0 end)      AS low_trust
        FROM gold
    """).fetchone()
    print(f"Gold rows: {summary[0]:,}")
    print(f"Avg trust: {summary[1]}")
    print(f"Avg confidence (overall): {summary[2]}")
    print(f"Below 0.65 trust: {summary[3]}")

    sample = con.execute("""
        SELECT
            facility_id,
            name,
            address_city,
            facility_type,
            json_extract_string(extraction_json, '$.staff.total_doctor_count') AS doctors,
            trust_score,
            json_extract_string(confidence_json, '$.overall')::DOUBLE AS confidence_overall
        FROM gold
        ORDER BY trust_score DESC
        LIMIT 10
    """).df()
display(sample)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Run the reasoning engine

# COMMAND ----------

from sehat.pipeline.reasoning import query_facilities

result = query_facilities(
    user_query="I need 24/7 emergency care with cardiac specialists in Mumbai",
    top_k_final=5,
)

print("Recommendation summary:")
print(result.get("recommendation_summary", "(none)"))
print()
print(f"Candidates retrieved: {result.get('candidates_retrieved')}")
print(f"Trust threshold     : {result.get('trust_threshold')}")
print()
print("Ranked results:")
for r in result.get("ranked_results", []):
    print(f"  - {r.get('name')} ({r.get('location')})  "
          f"trust={r.get('trust_score'):.2f}  rank_score={r.get('rank_score')}")
    for w in r.get("warnings", []) or []:
        print(f"      ! {w}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Inspect medical deserts

# COMMAND ----------

with duck() as con:
    deserts = con.execute("""
        SELECT pin_code, state, facility_count, avg_trust_score,
               icu_coverage, emergency_coverage, dialysis_coverage, surgery_coverage,
               desert_risk_score, is_high_risk, desert_categories
        FROM deserts
        ORDER BY desert_risk_score DESC
        LIMIT 15
    """).df()
print("Top 15 most underserved PIN codes:")
display(deserts)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Trust report for the worst-scoring facility

# COMMAND ----------

import json

worst = sample.tail(1).iloc[0]
print(f"Inspecting facility_id = {worst.facility_id} ({worst['name']})")

with duck() as con:
    row = con.execute("""
        SELECT trust_flags_json, confidence_json, embedding_text
        FROM gold
        WHERE facility_id = ?
    """, [worst.facility_id]).fetchone()

print("Trust flags:")
print(json.dumps(json.loads(row[0]), indent=2))
print()
print("Confidence breakdown:")
print(json.dumps(json.loads(row[1]), indent=2))
print()
print("Embedding text:")
print(row[2])

# COMMAND ----------

# MAGIC %md
# MAGIC If the cells above produce sensible results, the pipeline is working.
# MAGIC Continue to the **DEPLOY.md** guide (in this folder) to deploy the FastAPI
# MAGIC server as a Databricks App.
