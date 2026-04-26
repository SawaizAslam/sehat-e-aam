# Hugging Face Space — public mirror

The public-internet mirror of the Sehat-e-Aam FastAPI server, lives at:

```
https://<your-hf-username>-sehat-e-aam.hf.space
```

## Why it exists

Databricks Apps on Free Edition force workspace OAuth on every request. A
public Lovable / external React frontend cannot call those URLs directly. This
Space runs the **same FastAPI code** with permissive CORS so the frontend has
something to point at, while the LLM call still goes to the **Databricks
Foundation Model API** under the hood.

End-to-end:

```
Lovable React UI
    │  fetch()
    ▼
https://<you>-sehat-e-aam.hf.space   <-- this Space (public, CORS *)
    │  src/sehat/api/server.py
    │  src/sehat/llm.py (LLM_BACKEND=databricks)
    ▼
Databricks Foundation Model API   (databricks-meta-llama-3-3-70b-instruct)
    │  authenticated via DATABRICKS_TOKEN secret on the Space
    ▼
LLM response → ranked facilities → JSON to Lovable
```

## Layout

| Path | Role |
| --- | --- |
| `../Dockerfile`                        | Image used by HF Spaces (Docker SDK) |
| `huggingface/space_app.py`             | Runtime bootstrap → FAISS build → uvicorn |
| `huggingface/requirements.txt`         | Python deps installed at build time |
| `demo/facilities_silver.parquet`       | LLM-extracted Silver rows (~230 KB) |
| `demo/facilities_gold.parquet`         | Trust-scored Gold rows (~350 KB) |
| `src/sehat/`                           | FastAPI server + pipeline source |

## Required secrets / variables

Set these in **Space Settings → Variables and secrets**.

| Key | Type | Value |
| --- | --- | --- |
| `DATABRICKS_HOST`  | Variable | `https://<your-workspace>.cloud.databricks.com` |
| `DATABRICKS_TOKEN` | Secret   | A Databricks personal access token with `serving.databricks-meta-llama-3-3-70b-instruct` permission |

If `DATABRICKS_TOKEN` is missing, `/health` and `/api/deserts` still work but
`/api/query` will 5xx on its first LLM call.

## Cold start

~30 seconds the very first time the container boots, then ~3 s after that:

1. Stage `demo/*.parquet` → `/tmp/sehat/lakehouse/`
2. Download `BAAI/bge-small-en-v1.5` (~130 MB) into `/tmp/hf_home/`
3. Embed all Gold rows + write FAISS index to `/tmp/sehat/vector_index/`
4. Run the deserts SQL aggregation
5. `uvicorn 0.0.0.0:$PORT`

Subsequent restarts reuse the cached model + indexes.

## Deploy

See `databricks/DEPLOY.md` → "Path 4 — Public mirror on Hugging Face Spaces".
