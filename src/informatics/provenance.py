"""Fingerprints of the files a result was made from, so every result can be traced to its data.

The filter summary records the raw file it read, and ``run_info.json`` records every file the
analysis read (candidates, relevance decisions, comparison sample, OpenAlex data, language-model
answers, filter summaries): size, row count and SHA-256 hash. Two runs with the same hashes,
settings, package versions and git commit used the same inputs.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq


def sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def file_info(path: str | Path | None) -> dict | None:
    """Path, size, rows (Parquet files), last change and SHA-256 of a file; None if absent."""
    if path is None or not Path(path).exists():
        return None
    path = Path(path)
    stat = path.stat()
    info = {
        "path": str(path),
        "bytes": stat.st_size,
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(
            timespec="seconds"
        ),
        "sha256": sha256(path),
    }
    if path.suffix == ".parquet":
        info["rows"] = pq.ParquetFile(path).metadata.num_rows
    return info
