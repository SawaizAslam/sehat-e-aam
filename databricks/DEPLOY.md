# Sehat-e-Aam · Databricks deployment

This guide walks you through deploying the full Sehat-e-Aam pipeline + API on
**Databricks Free Edition**. You will end up with a public-ish URL
(workspace-scoped) that serves the four FastAPI endpoints.

> **Two deployment paths.**
> 1. **No-CLI / UI only** — works entirely in the Databricks browser, no
>    install required on your laptop. Recommended for first-time users.
> 2. **Asset Bundle (CLI)** — one command deploys notebooks + app. Requires
>    the Databricks CLI on any machine with internet (a colleague's laptop, a
>    free GitHub Codespace, or the Databricks Web Terminal itself).

The two paths produce identical results.

---

## Free Edition limits to keep in mind

| Resource                | Free Edition limit                                     |
|-------------------------|--------------------------------------------------------|
| Compute                 | Serverless only (no GPU)                               |
| Foundation Model API    | Daily token quota; only chat/embed endpoints           |
| Mosaic AI Vector Search | 1 endpoint, 1 unit (we use FAISS to avoid this slot)   |
| Databricks Apps         | **1 app per account**, auto-stops after 24h running    |
| Account console / API   | Not available                                          |

The deployment below uses **0 Vector Search endpoints** (we use FAISS on a UC
Volume) and **1 Databricks App** so you stay inside Free Edition limits.

---

## Path 1 — Browser-only (recommended)

### Step 1 — Get the project into the workspace

In the Databricks workspace sidebar:

1. **Workspace** → your home → **Create** → **Git folder**
2. Repository URL: paste your fork's HTTPS URL
   (or push this codebase to your own GitHub first).
3. Folder name: `sehat-e-aam`
4. Click **Create Git folder**.

You should now see `/Workspace/Users/<you>/sehat-e-aam` populated with this
project, including `databricks/notebooks/`, `databricks/app/`, and `src/sehat/`.

> **No GitHub?** Use **Workspace → Import** and drag the project as a `.zip`.
> Then unzip it manually inside the workspace using the file browser.

### Step 2 — Upload the dataset

You have two options.

#### Option A — UI upload

1. **Catalog** sidebar → `workspace` → **Create schema** → name it `sehat`.
2. Inside `workspace.sehat` → **Create volume** → managed volume → name `data`.
3. Inside `workspace.sehat.data` → **Create directory** → `raw`.
4. Inside `raw` → **Upload to this volume** → drop your CSV.
5. Rename the uploaded file to `facilities.csv`.

> If the **Upload to this volume** button is missing or the upload silently
> fails (a known Free Edition quirk for files >1MB), use Option B instead.

#### Option B — CLI upload (works for the Free Edition 10MB+ case)

```powershell
# 1. Install the CLI once
winget install Databricks.DatabricksCLI

# 2. Authenticate (opens a browser, click Allow)
databricks auth login `
  --host https://<your-workspace>.cloud.databricks.com `
  --profile sehat

# 3. Create the UC schema + volume
databricks --profile sehat schemas create sehat workspace
databricks --profile sehat volumes create workspace sehat data MANAGED

# 4. Upload the CSV (use --overwrite if you re-upload)
databricks --profile sehat fs cp `
  "<path-to-your-csv>" `
  "dbfs:/Volumes/workspace/sehat/data/raw/facilities.csv" `
  --overwrite

# 5. Verify
databricks --profile sehat fs ls dbfs:/Volumes/workspace/sehat/data/raw
```

(Notebook `00_setup` will skip schema/volume creation if they already exist,
so option B is non-destructive.)

> Free Edition uses the `workspace` catalog by default (you cannot create new
> catalogs without account-admin rights). On paid tiers, edit the `CATALOG`
> constant in the notebooks and `app.yaml` to use `main`.

### Step 3 — Run the setup notebook

1. Open `databricks/notebooks/00_setup.py`.
2. Top-right: **Connect** → pick the **Serverless** compute.
3. Edit the constants in the first cell only if you used different
   catalog/schema/volume names.
4. **Run all**.

The notebook will:
- Create catalog/schema/volume if missing.
- Install pip dependencies on the serverless kernel.
- Set environment variables and write a sidecar `sehat.env` to the Volume.
- Smoke-test the Foundation Model API with a tiny ping.

If the LLM ping cell fails with "endpoint not found", open
**Compute → Serving → Endpoints** and pick whichever Llama-family endpoint your
workspace lists, then update `LLM_ENDPOINT` in the notebook constants.

### Step 4 — Run the pipeline notebook

1. Open `databricks/notebooks/01_pipeline.py`.
2. **Run all**.

For the first run, leave `EXTRACT_SAMPLE_LIMIT=200` so you finish in ~3-6 min.
Once you're happy, edit Notebook 00's `EXTRACT_SAMPLE_LIMIT` to `0` (unlimited)
and re-run only the extract / trust / index / deserts steps.

You will end up with these files inside the Volume:

```
/Volumes/workspace/sehat/data/
├── raw/facilities.csv
├── lakehouse/
│   ├── facilities_bronze.parquet
│   ├── facilities_silver.parquet
│   ├── facilities_gold.parquet
│   ├── medical_deserts.parquet
│   └── audit_log.parquet
├── vector_index/
│   ├── facilities.faiss
│   └── facilities_meta.parquet
└── sehat.env              # written by 00_setup, consumed by the App
```

(MLflow runs are tracked in the workspace at `/Users/<you>/sehat-e-aam` — open
the **Experiments** sidebar inside Databricks to see latency, tokens, and
counts per step.)

### Step 5 — Smoke test

Open `databricks/notebooks/02_smoke_test.py` and **Run all**. You should see:
- Gold-table row count + average trust score.
- A live LLM-generated ranking for "I need 24/7 emergency care with cardiac
  specialists in Mumbai".
- Top 15 most underserved PIN codes.
- Trust flags + confidence breakdown for the worst-scoring facility.

### Step 6 — Deploy the FastAPI App

1. Sidebar → **Compute** → **Apps** → **Create app**.
2. Pick **Custom** (not a template).
3. **App name**: `sehat-e-aam`.
4. Click **Next** → **Source code** → **Browse Workspace files** and pick
   `/Workspace/Users/<you>/sehat-e-aam/databricks/app`.
5. Click **Create**.

Databricks will:
- Read `app.yaml` and install everything from `requirements.txt` (~3-4 min).
- Start uvicorn on port 8000.
- Show you a public-ish URL like
  `https://sehat-e-aam-<hash>.cloud.databricks.com`.

Wait for the status to flip to **Running**.

### Step 7 — Hit the endpoints

Each endpoint requires Databricks workspace authentication; the simplest way
to test is the **Apps URL → Open** button which opens the FastAPI Swagger UI
at `/docs` automatically (FastAPI default).

Once on the Swagger page you can:
- Try `GET /health` — should return `gold_ready: true`, `vector_ready: true`.
- Try `POST /api/query` with body
  ```json
  { "query": "ICU and dialysis in Lucknow", "top_k": 5 }
  ```
- Try `GET /api/facility/{facility_id}/trust` for any facility ID returned.
- Try `GET /api/deserts?high_risk_only=true&limit=20`.

**App URL format**: `https://<app-name>-<workspace-id>.<region>.databricksapps.com`.

### Stopping the App
Free Edition apps auto-stop after 24h. You can also stop manually from the
**Apps** screen → **Stop**. Re-deploys do not consume a fresh app slot.

---

## Path 2 — Asset Bundle (CLI)

If you can run a single CLI command anywhere with internet:

```bash
# Install the CLI (one-time, requires internet but no Databricks workspace).
# Linux / macOS:
curl -fsSL https://raw.githubusercontent.com/databricks/setup-cli/main/install.sh | sh

# Windows (PowerShell):
# winget install Databricks.DatabricksCLI

# Configure auth (one-time)
databricks configure   # paste your workspace URL + a personal access token

# From the repo root
databricks bundle validate
databricks bundle deploy --target dev
databricks bundle run pipeline_job
```

The bundle deploys the three notebooks **and** the app in one go. After
`bundle deploy` you'll see the App URL in the output.

You still need to upload the dataset CSV manually (Step 2 above) — bundles
deliberately don't push large data files.

> **Web Terminal alternative.** From the workspace UI: **Compute → Apps →
> ...→ Web terminal**, then run `databricks bundle …` directly inside the
> workspace. No local install needed.

---

## Path 3 — Mosaic AI Vector Search (optional upgrade)

Free Edition gives you **1 Vector Search endpoint with 1 unit**. If you'd
rather use it instead of FAISS:

1. Create the endpoint: **Compute → Vector search → Create endpoint**, name it
   `sehat-vs`.
2. After Notebook 01 finishes, run this snippet:
   ```python
   from databricks.vector_search.client import VectorSearchClient
   import pandas as pd
   df = pd.read_parquet("/Volumes/workspace/sehat/data/lakehouse/facilities_gold.parquet")
   # write to a Delta table the index can sync against
   spark.createDataFrame(df).write.mode("overwrite").saveAsTable("workspace.sehat.gold_facilities")
   client = VectorSearchClient()
   client.create_delta_sync_index(
       endpoint_name="sehat-vs",
       index_name="workspace.sehat.facility_index",
       source_table_name="workspace.sehat.gold_facilities",
       primary_key="facility_id",
       embedding_source_column="embedding_text",
       embedding_model_endpoint_name="databricks-gte-large-en",
       pipeline_type="TRIGGERED",
   )
   ```
3. Set `EMBEDDING_BACKEND=databricks` and `VECTOR_BACKEND=databricks-vs` in
   `app.yaml`. Then redeploy. *(The current code path is FAISS-only; the
   Databricks VS adapter is on the roadmap — open an issue if you'd like it
   prioritised.)*

---

## Troubleshooting

| Symptom                                      | Fix                                                                                                                |
|---------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| `endpoint not found` from the LLM ping cell | Open **Compute → Serving → Endpoints**; pick a real Llama endpoint name; update `LLM_ENDPOINT` constant.           |
| `quota exceeded` or `rate limited`          | Free Edition has daily token caps. Lower `EXTRACT_SAMPLE_LIMIT`, raise `EXTRACT_BATCH_SIZE`, or wait until tomorrow. |
| `import sehat` fails inside the App         | Confirm the repo is at `/Workspace/Users/<you>/sehat-e-aam`. Edit `SEHAT_PROJECT_ROOT` in `app.yaml` if you moved it. |
| App stuck in **Compute starting**           | First start can take 5+ min on Free Edition. Check **Logs** tab for errors.                                        |
| Pipeline notebook crashes on `faiss-cpu`    | Restart the kernel and re-run the `%pip install` cell. faiss wheels need a clean import.                            |
| `gold_ready: false` from `/health`          | The App is reading from the wrong Volume path. Match `LAKEHOUSE_DIR` in `app.yaml` to your `00_setup.py` constants. |
| App auto-stopped after 24h                  | Free Edition behaviour. Click **Start** in the Apps screen — the index/data persist on the Volume so it just resumes. |
