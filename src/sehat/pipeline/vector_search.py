"""Notebook 05 equivalent: build a FAISS-backed vector search over Gold.

We replace Mosaic AI Vector Search (a paid Databricks product, despite the
guide's claim that it ships with the Free Edition) with local FAISS +
``sentence-transformers``. Embeddings are stored on disk with their facility
metadata so queries return enriched rows immediately.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from rich.console import Console

from ..config import Settings, get_settings
from ..storage import parquet_exists, read_parquet

LOGGER = logging.getLogger(__name__)
console = Console()


# ---------------------------------------------------------------------------
# Embedder protocol
# ---------------------------------------------------------------------------


class _BaseEmbedder:
    dim: int = 0

    def embed(self, texts: list[str]) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError


class _LocalEmbedder(_BaseEmbedder):
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        # ``get_sentence_embedding_dimension`` may be None in old versions
        self.dim = int(self._model.get_sentence_embedding_dimension() or 384)

    def embed(self, texts: list[str]) -> np.ndarray:
        embeddings = self._model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype(np.float32)


class _OpenAIEmbedder(_BaseEmbedder):
    def __init__(self, settings: Settings) -> None:
        from openai import OpenAI

        self._client = OpenAI(
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
        )
        self._model = settings.embedding_model
        # Probe dimension lazily on first call
        self.dim = 0

    def embed(self, texts: list[str]) -> np.ndarray:
        resp = self._client.embeddings.create(model=self._model, input=texts)
        vectors = np.array([d.embedding for d in resp.data], dtype=np.float32)
        # L2-normalise so we can use inner-product search
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
        vectors = vectors / norms
        if self.dim == 0:
            self.dim = vectors.shape[1]
        return vectors


def _make_embedder(settings: Settings) -> _BaseEmbedder:
    if settings.embedding_backend == "openai":
        return _OpenAIEmbedder(settings)
    return _LocalEmbedder(settings.embedding_model)


# ---------------------------------------------------------------------------
# Index API
# ---------------------------------------------------------------------------


@dataclass
class VectorHit:
    facility_id: str
    score: float
    metadata: dict[str, Any]


class FacilityVectorIndex:
    """A wrapper around a FAISS inner-product index + parallel metadata frame."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._index: Any = None
        self._meta: pd.DataFrame | None = None
        self._embedder: _BaseEmbedder | None = None

    # -- build ---------------------------------------------------------------

    def build(self, gold_df: pd.DataFrame | None = None) -> None:
        import faiss

        s = self.settings
        if gold_df is None:
            if not parquet_exists(s.gold_path):
                raise FileNotFoundError("Gold parquet missing; run `sehat trust` first.")
            gold_df = read_parquet(s.gold_path)

        if gold_df.empty:
            raise ValueError("Gold table is empty; nothing to index.")

        embedder = _make_embedder(s)
        texts = gold_df["embedding_text"].astype(str).tolist()
        console.log(f"Embedding {len(texts):,} facilities ({s.embedding_backend} backend)...")
        vectors = embedder.embed(texts)
        if vectors.shape[0] != len(texts):
            raise RuntimeError("Embedder returned wrong number of vectors")

        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)

        s.vector_index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(s.vector_index_path))
        meta = gold_df[
            [
                "facility_id",
                "name",
                "address_city",
                "address_state",
                "address_zip",
                "latitude",
                "longitude",
                "facility_type",
                "trust_score",
                "confidence_json",
                "extraction_json",
                "embedding_text",
            ]
        ].reset_index(drop=True)
        meta.to_parquet(s.vector_meta_path, index=False)

        # cache embedder config so we can refuse mismatched queries
        config_path = s.vector_index_dir / "embedder.pkl"
        with config_path.open("wb") as f:
            pickle.dump(
                {
                    "backend": s.embedding_backend,
                    "model": s.embedding_model,
                    "dim": int(vectors.shape[1]),
                },
                f,
            )

        self._index = index
        self._meta = meta
        self._embedder = embedder
        console.log(
            f":white_check_mark: FAISS index built: {s.vector_index_path} "
            f"({vectors.shape[0]:,} x {vectors.shape[1]}d)"
        )

    # -- load ----------------------------------------------------------------

    def load(self) -> None:
        import faiss

        s = self.settings
        if not s.vector_index_path.exists() or not s.vector_meta_path.exists():
            raise FileNotFoundError("Vector index not built yet. Run `sehat index`.")
        self._index = faiss.read_index(str(s.vector_index_path))
        self._meta = read_parquet(s.vector_meta_path)
        self._embedder = _make_embedder(s)

    # -- query ---------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        top_k: int = 20,
        state: str | None = None,
        city: str | None = None,
        facility_type: str | None = None,
        min_trust: float = 0.0,
    ) -> list[VectorHit]:
        if self._index is None or self._meta is None or self._embedder is None:
            self.load()

        assert self._index is not None
        assert self._meta is not None
        assert self._embedder is not None

        embedding = self._embedder.embed([query])
        # Over-fetch then filter
        fetch_k = min(len(self._meta), top_k * 4)
        scores, indices = self._index.search(embedding, fetch_k)

        hits: list[VectorHit] = []
        for score, idx in zip(scores[0].tolist(), indices[0].tolist(), strict=False):
            if idx < 0 or idx >= len(self._meta):
                continue
            row = self._meta.iloc[idx]
            if state and (row.get("address_state") or "").lower() != state.lower():
                continue
            if city and (row.get("address_city") or "").lower() != city.lower():
                continue
            if facility_type and (row.get("facility_type") or "").lower() != facility_type.lower():
                continue
            if float(row.get("trust_score") or 0.0) < min_trust:
                continue
            hits.append(
                VectorHit(
                    facility_id=str(row["facility_id"]),
                    score=float(score),
                    metadata=row.to_dict(),
                )
            )
            if len(hits) >= top_k:
                break
        return hits


def run_index(settings: Settings | None = None) -> None:
    FacilityVectorIndex(settings).build()


__all__ = ["FacilityVectorIndex", "VectorHit", "run_index"]


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_index()
