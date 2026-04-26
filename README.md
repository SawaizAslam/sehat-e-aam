# Sehat-e-Aam

**Agentic Healthcare Intelligence System for Indian healthcare facilities.**

This project ingests a 10k-row dataset of Indian healthcare facilities, uses an
LLM to extract structured medical capabilities, applies a trust-scoring engine,
runs a self-correction loop on low-trust records, indexes the results in a
local FAISS vector store, and exposes a reasoning + medical-desert API via
FastAPI.

It is a Python re-implementation of the original
`AGENTIC_HEALTHCARE_BACKEND_GUIDE.md`, with all known bugs fixed and the
Databricks-only dependencies (Mosaic AI Vector Search, Delta Lake) replaced
with portable equivalents (FAISS, Parquet + DuckDB) so it runs locally as
well as on Databricks.

> **Want a deployed URL on Databricks Free Edition (no local install)?**
> Skip the local quick-start and jump to **[`databricks/DEPLOY.md`](databricks/DEPLOY.md)**.
> Three notebooks + one Apps deploy = a working FastAPI URL.

---

## Architecture (one screen)

```
data/facilities.csv
        |
        v
[ingest]  -->  lakehouse/facilities_bronze.parquet     (raw + composite_text + facility_id)
        |
        v
[extract] -->  lakehouse/facilities_silver.parquet     (LLM-extracted capabilities, validated by Pydantic)
        |
        v
[trust]   -->  lakehouse/facilities_gold.parquet       (trust score + confidence + embedding text)
        |
        +-- [self-correct] -->  Gold updated in place  (validate -> correct loop, max 2 iterations)
        |
        +-- [index]        -->  vector_index/facilities.faiss  (FAISS inner-product index)
        |
        +-- [deserts]      -->  lakehouse/medical_deserts.parquet
        |
        v
FastAPI (sehat serve)
   /api/query              full reasoning pipeline
   /api/facility/{id}      profile
   /api/facility/{id}/trust  trust report
   /api/deserts            geo-tagged desert risk regions
```

Cross-cutting: every LLM call is wrapped in MLflow spans (`./mlruns`).

---

## Quick start

### 1. Install

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### 2. Configure environment

```powershell
copy .env.example .env
```

Then edit `.env`. The default points at a local Ollama instance (`http://localhost:11434/v1`)
running `llama3.1:8b`. To use a hosted provider, set:

```ini
OPENAI_BASE_URL=https://api.together.xyz/v1
OPENAI_API_KEY=tgp_...
LLM_MODEL=meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo
```

To use Databricks Foundation Models:

```ini
LLM_BACKEND=databricks
DATABRICKS_HOST=https://<workspace>.cloud.databricks.com
DATABRICKS_TOKEN=dapi...
LLM_MODEL=databricks-meta-llama-3-1-70b-instruct
```

### 3. Place the dataset

Drop the raw file at `data/facilities.csv` (or change `RAW_DATASET_PATH` in
`.env` to point at the existing
`VF_Hackathon_Dataset_India_Large.xlsx - VF_Hackathon_Dataset_India_Larg (1).csv`).

### 4. Run the pipeline

For a smoke run on 200 rows first, set in `.env`:

```ini
EXTRACT_SAMPLE_LIMIT=200
```

Then:

```powershell
sehat info                 # show config
sehat ingest               # Bronze
sehat extract              # Silver  (LLM-bound; expensive)
sehat trust                # Gold
sehat self-correct         # iterate on low-trust rows
sehat index                # FAISS over Gold
sehat deserts              # PIN-level desert risk

# or do it all in one go (after smoke testing):
sehat pipeline
```

### 5. Query

```powershell
# CLI
sehat query "emergency surgery for appendicitis with full-time anesthesiologist" --state Bihar --top-k 3

# HTTP
sehat serve
# -> http://127.0.0.1:8000/docs
```

---

## CLI reference

| Command              | What it does                                                |
|----------------------|-------------------------------------------------------------|
| `sehat info`         | Print resolved configuration (which LLM, which paths, ...). |
| `sehat ingest`       | Build the Bronze parquet from the raw CSV / XLSX.          |
| `sehat extract`      | Resumable LLM extraction into Silver.                      |
| `sehat trust`        | Compute trust + confidence + embedding text into Gold.     |
| `sehat self-correct` | Run validate -> correct loop on rows below the threshold.  |
| `sehat index`        | Build the FAISS vector index over Gold.                    |
| `sehat deserts`      | Compute per-PIN desert risk into a parquet table.          |
| `sehat pipeline`     | Run all of the above in order.                             |
| `sehat query "..."`  | One-shot reasoning query.                                  |
| `sehat serve`        | Run the FastAPI server.                                    |

---

## What changed vs the original guide

The original `AGENTIC_HEALTHCARE_BACKEND_GUIDE.md` contained several bugs that
would have blocked execution on Databricks Free Edition. This implementation
fixes all of them. Highlights:

- **Vector Search**: replaced Mosaic AI Vector Search (a paid Databricks
  product, despite the guide's claim) with FAISS + sentence-transformers.
- **Auth**: replaced the fragile `dbutils.notebook.entry_point` token grab
  with `databricks-sdk` (and an OpenAI-compatible alternative).
- **`facility_id` collision**: now hashes `name + city + zip` so chains do
  not merge into a single record.
- **`composite_text_length`** is persisted to Bronze (the guide referenced
  but never wrote it).
- **CSV path**: handled correctly for both `dbfs:/` and local paths;
  auto-detects `.xlsx`.
- **JSON-mode LLM calls**: `response_format={"type": "json_object"}` plus
  Pydantic validation, so we no longer rely on stripping markdown fences.
- **`AvailabilityStatus.NOT_PRESENT`** now means *explicitly absent*; absence
  of evidence is `UNCERTAIN`. This fixes the cascading false-positive flags
  the original guide produced.
- **Multiplicative trust dampening** so a single moderate flag cannot crush
  a record to the 0.05 floor.
- **Self-correction recomputes confidence** as well as trust before merging
  back into Gold.
- **Resumable extraction**: rows already in Silver are skipped.
- **Pydantic v2** throughout, with explicit coercion for stringified ints.
- **DuckDB JSON extraction** in the deserts pipeline (no per-row Python UDFs).
- **`farmacy` typo** normalised to `pharmacy`.

---

## Project layout

```
src/sehat/
  schemas.py            Pydantic v2 models (FacilityExtraction, ConfidenceScore, ...)
  config.py             Settings loaded from .env via pydantic-settings
  llm.py                LLMClient (databricks | openai-compatible) with JSON mode + retries
  prompts.py            All LLM prompts (system + user templates)
  storage.py            Parquet + DuckDB lakehouse helpers
  tracing.py            MLflow run / span helpers
  pipeline/
    ingest.py           Bronze
    extract.py          Silver (resumable, validated)
    trust_score.py      Gold (rules + confidence)
    self_correct.py     Validator -> Corrector loop
    vector_search.py    FAISS index + search
    reasoning.py        End-to-end retrieve -> filter -> rank
    deserts.py          PIN-level desert aggregation
  api/server.py         FastAPI service
  cli.py                Typer-based CLI

tests/                  pytest smoke tests
lakehouse/              Generated Parquet tables (gitignored)
vector_index/           Generated FAISS index (gitignored)
mlruns/                 MLflow tracking (gitignored)
data/                   Raw dataset (gitignored)
```

---

## Testing

```powershell
pip install -e ".[dev]"
pytest -q
```

The default test suite is offline (no LLM calls).

---

## Cost & rate-limit guidance

A full run extracts 10,000 facilities × 1 LLM call each. With the suggested
prompt + 3,000-char context, that is ~1,500 input tokens / call.

| Backend                         | Approx tokens     | Notes                                           |
|---------------------------------|-------------------|-------------------------------------------------|
| Local Ollama (`llama3.1:8b`)    | Free (CPU/GPU)    | Slowest; useful for smoke runs.                 |
| Together.ai 70B Turbo           | ~$0.50 per 1M in  | Good free-tier credits; rate ~600 RPM.          |
| Groq (`llama-3.1-70b-versatile`) | ~$0.59 per 1M in | Fastest; tighter token-per-minute caps.         |
| Databricks 70B Llama            | Varies by tier    | Free Edition: limited daily token budget.       |

**Always start with `EXTRACT_SAMPLE_LIMIT=200`** to verify the pipeline before
scaling to the full 10k.

---

## License

MIT.
