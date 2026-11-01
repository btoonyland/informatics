"""Reading and writing files.

Parquet is the canonical format: it keeps types (arXiv IDs stay strings, lists stay lists).
CSV is only produced for viewing in a spreadsheet, with list columns encoded as JSON.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def _atomic_target(path: Path) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    return tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")


def write_parquet(df: pd.DataFrame, path: str | Path, schema: pa.Schema | None = None) -> Path:
    """Write ``df`` atomically (a crash never leaves a half-written file behind)."""
    path = Path(path)
    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
    fd, tmp = _atomic_target(path)
    os.close(fd)
    try:
        pq.write_table(table, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return path


def read_parquet(path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Read a Parquet file; list columns come back as Python lists (not numpy arrays)."""
    df = pq.read_table(path, columns=columns).to_pandas()
    return lists_to_python(df)


def iter_parquet(
    path: str | Path, columns: list[str] | None = None, batch_size: int = 100_000
) -> Iterator[pd.DataFrame]:
    """Yield a large Parquet file in chunks to keep memory bounded."""
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
        yield lists_to_python(batch.to_pandas())


def lists_to_python(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == object and df[col].map(lambda v: isinstance(v, np.ndarray)).any():
            df[col] = df[col].map(lambda v: v.tolist() if isinstance(v, np.ndarray) else v)
    return df


def _is_listlike(value: Any) -> bool:
    return isinstance(value, (list, tuple, np.ndarray, dict))


def _json_cell(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    return json.dumps(value, ensure_ascii=False) if _is_listlike(value) else value


def write_csv(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a spreadsheet-friendly CSV: list/dict columns are JSON-encoded so they parse back."""
    path = Path(path)
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object and out[col].map(_is_listlike).any():
            out[col] = out[col].map(_json_cell)
    fd, tmp = _atomic_target(path)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            out.to_csv(fh, index=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return path


def write_json(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    fd, tmp = _atomic_target(path)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, ensure_ascii=False, default=_json_default)
            fh.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return path


def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_text(text: str, path: str | Path) -> Path:
    path = Path(path)
    fd, tmp = _atomic_target(path)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return path


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")
