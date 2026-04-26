"""Lightweight lakehouse abstraction.

We use Parquet files on local disk + DuckDB for SQL. This gives us a
Bronze/Silver/Gold pattern equivalent in spirit to Delta Lake without any
Databricks dependency. All paths come from :class:`Settings`.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import duckdb
import pandas as pd

from .config import Settings, get_settings


def _normalise_nas(df: pd.DataFrame) -> pd.DataFrame:
    """Replace ``pd.NA`` (and other "missing" sentinels) in object/string columns
    with ``None``.

    Why: ``pd.read_parquet`` (especially via the pyarrow engine on Databricks)
    returns object/string columns whose missing cells are ``pd.NA``. Python's
    ``or`` operator (and many builtin functions) raise
    ``TypeError: boolean value of NA is ambiguous`` on those values. Coercing
    to ``None`` once at the I/O boundary lets downstream code use ordinary
    ``value or default`` patterns safely.

    Numeric columns are left untouched (NaN is fine there).
    """

    if df.empty:
        return df
    for col in df.columns:
        s = df[col]
        if s.dtype == "object" or pd.api.types.is_string_dtype(s):
            df[col] = s.astype(object).where(s.notna(), None)
    return df


def write_parquet(df: pd.DataFrame, path: Path, *, overwrite: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite and path.exists():
        raise FileExistsError(f"{path} exists and overwrite=False")
    df.to_parquet(path, index=False)


def append_parquet(df: pd.DataFrame, path: Path) -> None:
    """Append rows to a parquet 'table' (read-modify-write; fine at 10k scale)."""

    if path.exists():
        existing = _normalise_nas(pd.read_parquet(path))
        combined = pd.concat([existing, df], ignore_index=True)
    else:
        combined = df
    write_parquet(combined, path, overwrite=True)


def read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Parquet not found: {path}")
    return _normalise_nas(pd.read_parquet(path))


def parquet_exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def upsert_parquet(df: pd.DataFrame, path: Path, key: str) -> None:
    """MERGE-style upsert on a parquet file using a primary key."""

    if not parquet_exists(path):
        write_parquet(df, path, overwrite=True)
        return
    existing = _normalise_nas(pd.read_parquet(path))
    merged = (
        pd.concat([existing[~existing[key].isin(df[key])], df], ignore_index=True)
        .drop_duplicates(subset=[key], keep="last")
        .reset_index(drop=True)
    )
    write_parquet(merged, path, overwrite=True)


@contextmanager
def duck(settings: Settings | None = None) -> Iterator[duckdb.DuckDBPyConnection]:
    """Context-managed DuckDB connection with parquet views attached.

    Usage::

        with duck() as con:
            con.sql("SELECT * FROM gold WHERE trust_score > 0.5").df()
    """

    s = settings or get_settings()
    con = duckdb.connect()
    try:
        for name, p in [
            ("bronze", s.bronze_path),
            ("silver", s.silver_path),
            ("gold", s.gold_path),
            ("deserts", s.deserts_path),
        ]:
            if parquet_exists(p):
                con.execute(
                    f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet('{p.as_posix()}')"
                )
        yield con
    finally:
        con.close()


__all__ = [
    "write_parquet",
    "append_parquet",
    "read_parquet",
    "parquet_exists",
    "upsert_parquet",
    "duck",
    "_normalise_nas",
]
