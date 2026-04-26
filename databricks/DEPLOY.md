# Sehat-e-Aam · Databricks deployment

This guide walks you through deploying the full Sehat-e-Aam pipeline + API on
**Databricks Free Edition**, plus an optional public mirror so an external
React frontend (Lovable, Vercel, …) can call the API without bouncing users
through workspace login.

> **Three deployment paths.**
> 1. **No-CLI / UI only** — works entirely in the Databricks browser, no
>    install required on your laptop. Recommended for first-time users.
> 2. **Asset Bundle (CLI)** — one command deploys notebooks + Databricks App.
>    Requires the Databricks CLI on any machine with internet (a colleague's
>    laptop, a free GitHub Codespace, or the Databricks Web Terminal itself).
> 3. **Public mirror on Hugging Face Spaces** — runs the same FastAPI code on
>    a truly public URL with permissive CORS. Add this if a public-internet
>    frontend needs to call the API.

Paths 1 and 2 produce identical results. Path 3 is a separate, parallel deploy
and is documented in **[Path 4](#path-4--public-mirror-on-hugging-face-spaces)**
below (numbered 4 to keep the original Mosaic AI VS section as Path 3).

> ### About authentication
> Databricks Apps on Free Edition force workspace OAuth on every request.
> A `*.databricksapps.com` URL is "published", but a Lovable-style React
> frontend on the open internet **cannot call it directly** — every request
> bounces to a Databricks login screen. If your frontend lives outside the
> workspace, deploy **both** the Databricks App (Path 1 or 2) **and** the
> public mirror (Path 4). They share the same `src/sehat/api/server.py`.

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

The bundle (`databricks.yml`) deploys the three notebooks **and** the
Databricks App in one go.

### Step 1 — Install the CLI (one-time)

```powershell
# Windows (PowerShell)
winget install Databricks.DatabricksCLI

# Linux / macOS
# curl -fsSL https://raw.githubusercontent.com/databricks/setup-cli/main/install.sh | sh
```

Verify with `databricks --version` (must be >=0.221, the version that
supports `apps:` resources in bundles).

### Step 2 — Authenticate

```powershell
databricks auth login `
  --host https://<your-workspace>.cloud.databricks.com `
  --profile sehat
```

A browser opens — click **Allow**. The profile is saved to
`~/.databrickscfg` so future commands just need `--profile sehat`.

### Step 3 — Stage the self-contained app folder

The bundle points at `databricks/_app_deploy/`, which is `databricks/app/`
plus a copy of `src/sehat/` next to it. Recreate it with:

```powershell
# From repo root
Remove-Item -Recurse -Force databricks/_app_deploy -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path databricks/_app_deploy | Out-Null
Copy-Item -Recurse databricks/app/* databricks/_app_deploy/
Copy-Item -Recurse src databricks/_app_deploy/
```

(The folder is gitignored on purpose — it's a build artefact, not source.)

### Step 4 — Validate, deploy, run

```powershell
databricks bundle validate --profile sehat
databricks bundle deploy   --profile sehat --target dev
databricks bundle run pipeline_job --profile sehat
```

What this does:
- Uploads notebooks to `/Workspace/Users/<you>/.bundle/sehat-e-aam/dev/files/`.
- Creates / updates the `sehat-e-aam` Databricks App from
  `databricks/_app_deploy/`.
- Submits the three-task pipeline job (setup → pipeline → smoke).

The App URL prints in the `bundle deploy` output — typically
`https://sehat-e-aam-<hash>.cloud.databricks.com`. Workspace login required.

You still need to upload the dataset CSV manually (Step 2 of Path 1) — bundles
deliberately don't push large data files.

> **Web Terminal alternative.** From the workspace UI: **Compute → Apps →
> ...→ Web terminal**, then run `databricks bundle …` directly inside the
> workspace. No local install needed.

> **Free Edition gotcha.** `databricks.yml` defaults `catalog` to `workspace`
> (the only catalog you can write to without account-admin rights). On paid
> tiers, override with `--var catalog=main`.

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

## Path 4 — Public mirror on Hugging Face Spaces

Required when an **external** React frontend (Lovable, Vercel, anything
outside the Databricks workspace) needs to call the API. Runs the same
FastAPI code (`src/sehat/api/server.py`) on a truly public URL.

End state:
- `https://<your-hf-username>-sehat-e-aam.hf.space` — open, CORS `*`
- LLM still calls Databricks Foundation Model API
  (`databricks-meta-llama-3-3-70b-instruct`) via a `DATABRICKS_TOKEN` Space
  secret, so the "powered by Databricks" story is end-to-end.

What ships in the Docker image:
- `src/sehat/` (FastAPI server + pipeline)
- `demo/facilities_silver.parquet` + `demo/facilities_gold.parquet`
  (~600 KB total, generated by the Databricks pipeline run)
- `huggingface/space_app.py` — bootstraps env vars, rebuilds the FAISS index
  and the medical-deserts aggregate, then runs uvicorn.

### Step 1 — Generate a Databricks personal access token

1. Workspace **Settings → Developer → Access tokens → Generate new token**.
2. Lifetime: 90 days (or whatever your demo window is).
3. Copy the token now (you cannot view it again).

### Step 2 — Create the Space

1. https://huggingface.co/new-space
2. Owner: your username. Name: `sehat-e-aam`. License: MIT.
3. **SDK: Docker → Blank Docker template**.
4. Hardware: `cpu-basic` (free, 16 GB RAM, 2 vCPU). No upgrade needed.
5. Visibility: **Public**.
6. Click **Create Space**. You'll get a Git URL like
   `https://huggingface.co/spaces/<you>/sehat-e-aam`.

### Step 3 — Add Space secrets / variables

In the new Space → **Settings → Variables and secrets**:

| Type     | Key                | Value                                          |
| -------- | ------------------ | ---------------------------------------------- |
| Variable | `DATABRICKS_HOST`  | `https://<your-workspace>.cloud.databricks.com` |
| Secret   | `DATABRICKS_TOKEN` | the token from Step 1                          |

(Optional: also set `LLM_MODEL` if your workspace has a different endpoint
name; defaults to `databricks-meta-llama-3-3-70b-instruct`.)

### Step 4 — Push the repo to the Space

Pick whichever of A/B suits you:

#### A — Push the whole repo as the Space (simplest)

```powershell
# From repo root
git remote add hf https://huggingface.co/spaces/<your-hf-username>/sehat-e-aam
git push hf main
```

The repo's root `Dockerfile` and the HF frontmatter at the top of `README.md`
make HF Spaces auto-detect everything. Build takes ~4-6 min the first time
(faiss-cpu wheel + sentence-transformers).

#### B — Push only the deploy artefacts (cleaner if your repo is private)

```powershell
git clone https://huggingface.co/spaces/<your-hf-username>/sehat-e-aam space-repo
cd space-repo

# Copy only what the image needs
Copy-Item -Recurse ../src .
Copy-Item -Recurse ../huggingface .
Copy-Item ../Dockerfile .
Copy-Item ../README.md .
New-Item -ItemType Directory -Force -Path demo | Out-Null
Copy-Item ../demo/facilities_silver.parquet demo/
Copy-Item ../demo/facilities_gold.parquet   demo/

git add -A
git commit -m "Initial Sehat-e-Aam Space"
git push origin main
```

### Step 5 — Watch the build, hit the endpoints

In the Space → **Logs** tab. You'll see:
```
Stage: Building → Running
Staged facilities_silver.parquet -> ...
Staged facilities_gold.parquet -> ...
Embedding 565 facilities (local backend)...
:white_check_mark: FAISS index built ...
Building medical-deserts aggregate from Gold...
Starting uvicorn on 0.0.0.0:7860
```

When status flips to **Running**, open:
- `https://<you>-sehat-e-aam.hf.space/docs` — Swagger UI
- `https://<you>-sehat-e-aam.hf.space/health` — should be fully `true`s

Point your Lovable / external frontend at that base URL. CORS is already
`*` in `src/sehat/api/server.py`.

### What it costs

| | Free tier limit | Realistic usage |
| --- | --- | --- |
| HF Spaces (cpu-basic) | unlimited build time, sleeps after 48h idle | $0 |
| Databricks Foundation Model API | daily token cap on Free Edition | ~1500 prompt + 500 completion tokens per `/api/query` call |

If you hit the daily token cap, the Space's `/health` keeps working but
`/api/query` will return a `LLMError`. Wait until the next UTC day or upgrade
to a paid Databricks tier.

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
| HF Space build fails on `faiss-cpu`         | The image already installs `build-essential` and `libgomp1`. Re-run the build. If still failing, pin `faiss-cpu==1.8.0` in `huggingface/requirements.txt`. |
| HF Space `/api/query` returns 5xx           | Check Space **Logs** for `LLMError`. Most often `DATABRICKS_TOKEN` was not set, or the workspace's daily token cap is exhausted. |
| HF Space takes 30+ s on the first request   | Cold start is normal. The container caches the embedding model + FAISS index after the first boot, so subsequent restarts take ~5 s. |
