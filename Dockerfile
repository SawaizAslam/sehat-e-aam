# Hugging Face Space (Docker SDK) image for the public Sehat-e-Aam FastAPI
# mirror. See huggingface/space_app.py for the runtime bootstrap and
# databricks/DEPLOY.md for how this fits with the Databricks App deploy.
#
# Build context = repo root, so we can COPY src/sehat, demo/*.parquet, and
# huggingface/* in a single image. HF Spaces builds with the Dockerfile at
# the root of the Space repo automatically.

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# HF Spaces runs the container as uid 1000. Create the user up-front and
# pre-create the cache dirs the runtime will write to. /tmp is also writable.
RUN useradd -m -u 1000 -s /bin/bash user \
 && mkdir -p /home/user/app /tmp/sehat /tmp/hf_home \
 && chown -R user:user /home/user /tmp/sehat /tmp/hf_home

# System deps for faiss-cpu + sentence-transformers + duckdb
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        ca-certificates \
        curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /home/user/app

COPY --chown=user:user huggingface/requirements.txt /home/user/app/requirements.txt
RUN pip install --no-cache-dir -r /home/user/app/requirements.txt

# Application code + the small (~600KB) parquet artifacts we ship with the
# image so the API can answer the moment uvicorn starts.
COPY --chown=user:user src /home/user/app/src
COPY --chown=user:user huggingface /home/user/app/huggingface
COPY --chown=user:user demo/facilities_silver.parquet /home/user/app/demo/facilities_silver.parquet
COPY --chown=user:user demo/facilities_gold.parquet /home/user/app/demo/facilities_gold.parquet

USER user

ENV PORT=7860 \
    SEHAT_SCRATCH=/tmp/sehat \
    HF_HOME=/tmp/hf_home \
    TRANSFORMERS_CACHE=/tmp/hf_home/transformers \
    SENTENCE_TRANSFORMERS_HOME=/tmp/hf_home/sentence_transformers \
    PYTHONPATH=/home/user/app/src

EXPOSE 7860

# `space_app.py` stages the parquets, builds FAISS + deserts, then runs uvicorn
# on $PORT. ~30s cold start.
CMD ["python", "huggingface/space_app.py"]
