"""Cache of prepared texts (normalisation steps 1-3), so they are computed once per paper.

Entries are keyed by paper ID and a hash of the raw text, and tagged with the normalisation
version, so a changed abstract or changed rules are recomputed automatically.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from informatics.normalize import NORMALIZATION_VERSION, prepare_text
from informatics.storage import read_parquet, write_parquet

log = logging.getLogger(__name__)


def _digest(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


class TextCache:
    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        self.entries: dict[str, tuple[str, str]] = {}  # id -> (digest of raw text, prepared text)
        self.changed = False
        if self.path is not None and self.path.exists():
            df = read_parquet(self.path)
            df = df[df["version"] == NORMALIZATION_VERSION]
            self.entries = dict(
                zip(df["id"], zip(df["digest"], df["text"], strict=True), strict=True)
            )

    def prepare(self, ids: Sequence[str], raw_texts: Sequence[str]) -> list[str]:
        out, computed = [], 0
        for paper_id, raw in zip(ids, raw_texts, strict=True):
            digest = _digest(raw)
            hit = self.entries.get(paper_id)
            if hit is not None and hit[0] == digest:
                out.append(hit[1])
                continue
            text = prepare_text(raw)
            self.entries[paper_id] = (digest, text)
            out.append(text)
            computed += 1
        if computed:
            self.changed = True
            log.info("prepared %d texts (%d from cache)", computed, len(out) - computed)
        return out

    def save(self) -> None:
        if self.path is None or not self.changed:
            return
        ids = list(self.entries)
        df = pd.DataFrame(
            {
                "id": ids,
                "digest": [self.entries[i][0] for i in ids],
                "version": NORMALIZATION_VERSION,
                "text": [self.entries[i][1] for i in ids],
            }
        )
        write_parquet(df, self.path)
        self.changed = False
