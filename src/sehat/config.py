"""Central configuration loaded from environment variables (.env supported)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- LLM backend ---------------------------------------------------------
    llm_backend: Literal["databricks", "openai"] = "openai"
    llm_model: str = "llama3.1:8b"
    extract_temperature: float = 0.0
    extract_max_tokens: int = 2048

    # Databricks
    databricks_host: str | None = None
    databricks_token: str | None = None

    # OpenAI-compatible
    openai_base_url: str = "http://localhost:11434/v1"
    openai_api_key: str = "ollama"

    # --- Embeddings ---------------------------------------------------------
    embedding_backend: Literal["local", "databricks", "openai"] = "local"
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # --- Pipeline knobs -----------------------------------------------------
    extract_batch_size: int = 50
    extract_max_workers: int = 4
    extract_sample_limit: int = 0

    correction_trigger_trust: float = 0.65
    correction_max_iterations: int = 2
    correction_sample_limit: int = 500

    vector_top_k: int = 20
    reasoning_top_k: int = 5
    min_trust_for_reasoning: float = 0.4
    desert_high_risk_threshold: float = 0.70

    # --- Paths --------------------------------------------------------------
    data_dir: Path = Path("./data")
    lakehouse_dir: Path = Path("./lakehouse")
    vector_index_dir: Path = Path("./vector_index")
    mlflow_tracking_uri: str = "./mlruns"
    raw_dataset_path: Path = Path("./data/facilities.csv")

    # --- Derived helpers ----------------------------------------------------

    @property
    def bronze_path(self) -> Path:
        return self.lakehouse_dir / "facilities_bronze.parquet"

    @property
    def silver_path(self) -> Path:
        return self.lakehouse_dir / "facilities_silver.parquet"

    @property
    def gold_path(self) -> Path:
        return self.lakehouse_dir / "facilities_gold.parquet"

    @property
    def deserts_path(self) -> Path:
        return self.lakehouse_dir / "medical_deserts.parquet"

    @property
    def audit_path(self) -> Path:
        return self.lakehouse_dir / "audit_log.parquet"

    @property
    def vector_index_path(self) -> Path:
        return self.vector_index_dir / "facilities.faiss"

    @property
    def vector_meta_path(self) -> Path:
        return self.vector_index_dir / "facilities_meta.parquet"

    def ensure_dirs(self) -> None:
        for p in (
            self.data_dir,
            self.lakehouse_dir,
            self.vector_index_dir,
            Path(self.mlflow_tracking_uri),
        ):
            p.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings


__all__ = ["Settings", "get_settings"]
