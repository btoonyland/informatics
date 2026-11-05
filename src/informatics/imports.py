"""Alternative ways to build ``data/raw/<set>.parquet`` without a multi-hour OAI-PMH harvest."""

from __future__ import annotations

import contextlib
import email.utils
import json
import logging
from collections.abc import Iterable, Iterator
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from informatics.records import (
    RAW_SCHEMA,
    as_str_list,
    normalize_records,
    parse_list_repr,
    repair_mojibake,
)

log = logging.getLogger(__name__)


def in_set(categories: Iterable[str], set_name: str) -> bool:
    """True if any category belongs to the archive (``cs`` matches ``cs.LG``; cross-lists count,
    as they do in arXiv's OAI-PMH sets)."""
    prefix = set_name.split(":")[-1]
    return any(c == prefix or c.startswith(prefix + ".") for c in categories)


def _rfc2822_date(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return email.utils.parsedate_to_datetime(value).date().isoformat()
    except (TypeError, ValueError):
        return None


def kaggle_record(rec: dict) -> dict:
    """One line of Cornell's Kaggle snapshot (arxiv-metadata-oai-snapshot.json) -> raw record."""
    versions = rec.get("versions") or []
    authors = []
    for entry in rec.get("authors_parsed") or []:
        keyname, forenames, suffix = (list(entry) + ["", "", ""])[:3]
        name = " ".join(p.strip() for p in (forenames, keyname, suffix) if p and p.strip())
        if name:
            authors.append(name)
    return {
        "id": rec.get("id"),
        "title": rec.get("title"),
        "abstract": rec.get("abstract"),
        "authors": authors,
        "categories": as_str_list(rec.get("categories"), split=True),
        "created": _rfc2822_date(versions[0].get("created")) if versions else None,
        "updated": _rfc2822_date(versions[-1].get("created")) if len(versions) > 1 else None,
        "doi": rec.get("doi"),
        "journal_ref": rec.get("journal-ref"),
        "comments": rec.get("comments"),
        "license": rec.get("license"),
        "datestamp": rec.get("update_date"),
    }


class _BatchWriter:
    """Write normalised records to Parquet in batches, dropping repeated IDs.

    Records go to a temporary file that replaces ``dest`` only when :meth:`close` succeeds. Used
    as a context manager, an import that fails or is interrupted (no internet, Ctrl+C) removes
    the temporary file and leaves an existing ``dest`` as it was."""

    def __init__(self, dest: Path, source: str):
        self.dest, self.source = Path(dest), source
        self.tmp = self.dest.with_name(f".{self.dest.name}.tmp")
        self.writer: pq.ParquetWriter | None = None
        self.seen: set[str] = set()
        self.count = 0

    def write(self, rows: list[dict] | pd.DataFrame) -> None:
        df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame.from_records(rows)
        if df.empty:
            return
        df = normalize_records(df, source=self.source)
        df = df[~df["id"].isin(self.seen)].drop_duplicates("id")
        self.seen.update(df["id"])
        if self.writer is None:
            self.dest.parent.mkdir(parents=True, exist_ok=True)
            self.writer = pq.ParquetWriter(self.tmp, RAW_SCHEMA)
        self.writer.write_table(pa.Table.from_pandas(df, schema=RAW_SCHEMA, preserve_index=False))
        self.count += len(df)

    def close(self) -> int:
        if self.writer is None:  # nothing matched: still produce an (empty) file
            pq.write_table(RAW_SCHEMA.empty_table(), self.tmp)
        else:
            self.writer.close()
            self.writer = None
        self.tmp.replace(self.dest)
        return self.count

    def abort(self) -> None:
        """Drop everything written so far."""
        if self.writer is not None:
            with contextlib.suppress(Exception):
                self.writer.close()
            self.writer = None
        self.tmp.unlink(missing_ok=True)

    def __enter__(self) -> _BatchWriter:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if exc_type is not None:
            self.abort()


def _iter_json_lines(path: Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def import_kaggle(
    snapshot: str | Path, set_name: str, dest: str | Path, batch: int = 50_000
) -> int:
    """Extract one archive from the Kaggle snapshot (streamed; the file is several GB)."""
    rows: list[dict] = []
    with _BatchWriter(Path(dest), "kaggle") as writer:
        for rec in _iter_json_lines(Path(snapshot)):
            if in_set((rec.get("categories") or "").split(), set_name):
                rows.append(kaggle_record(rec))
                if len(rows) >= batch:
                    writer.write(rows)
                    rows = []
        writer.write(rows)
        count = writer.close()
    log.info("imported %d %s records from %s into %s", count, set_name, snapshot, dest)
    return count


LEGACY_RENAMES = {"summary": "abstract", "published": "created", "update": "updated"}


def legacy_frame(chunk: pd.DataFrame) -> pd.DataFrame:
    """Repair one chunk of a CSV written by the old ``scripts/arXivAPI.py``.

    That script decoded arXiv's UTF-8 as Latin-1 (fixed with :func:`repair_mojibake`), wrote
    author lists as Python reprs, and kept only surnames (which cannot be recovered).
    """
    chunk = chunk.rename(columns=LEGACY_RENAMES)
    for col in ("title", "abstract", "doi"):
        if col in chunk:
            chunk[col] = chunk[col].map(repair_mojibake)
    if "authors" in chunk:
        chunk["authors"] = chunk["authors"].map(
            lambda v: [repair_mojibake(name) for name in parse_list_repr(v)]
        )
    return chunk


def import_legacy_csv(path: str | Path, dest: str | Path, chunksize: int = 100_000) -> int:
    """Convert a CSV from the old harvester (e.g. All_CS.csv) to the canonical Parquet file."""
    # dtype=str keeps IDs such as 0708.2931 intact; only empty cells count as missing.
    reader = pd.read_csv(
        path, dtype=str, keep_default_na=False, na_values=[""], chunksize=chunksize
    )
    with _BatchWriter(Path(dest), "legacy-csv") as writer:
        for chunk in reader:
            writer.write(legacy_frame(chunk))
        count = writer.close()
    log.info("imported %d records from %s into %s", count, path, dest)
    return count
