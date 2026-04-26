"""Smoke tests for the storage layer."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sehat.storage import upsert_parquet, write_parquet


def test_upsert_parquet_replaces_keys(tmp_path: Path) -> None:
    p = tmp_path / "table.parquet"

    initial = pd.DataFrame({"id": [1, 2, 3], "value": ["a", "b", "c"]})
    write_parquet(initial, p)

    update = pd.DataFrame({"id": [2, 4], "value": ["B2", "d"]})
    upsert_parquet(update, p, key="id")

    result = pd.read_parquet(p).sort_values("id").reset_index(drop=True)
    assert result.shape == (4, 2)
    assert result.set_index("id").loc[2, "value"] == "B2"
    assert result.set_index("id").loc[4, "value"] == "d"
    assert result.set_index("id").loc[1, "value"] == "a"
