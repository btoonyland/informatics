"""Sentence embeddings for clustering (optional: ``pip install -e ".[embeddings]"``).

``informatics embed`` encodes each selected paper's title + abstract once and caches the vectors
in ``data/interim/embeddings/<model>.parquet``, keyed by paper ID and a hash of the text, so
re-running only encodes new or changed papers. ``informatics analyze --embeddings <model>`` then
clusters on these vectors instead of LSA; cluster labels still come from TF-IDF terms.

Embeddings capture meaning rather than spelling, so word forms and synonyms matter much less.
The default model is small and fast. ``allenai/specter2_base`` was trained on scientific papers
(sentence-transformers loads it with mean pooling; AllenAI's adapters give the full SPECTER2).
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from informatics.normalize import clean_characters
from informatics.storage import read_parquet, write_parquet

log = logging.getLogger(__name__)

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
Encoder = Callable[[list[str]], np.ndarray]


def model_slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model)


def _digest(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def embedding_text(title: object, abstract: object) -> str:
    parts = [str(v) for v in (title, abstract) if isinstance(v, str) and v]
    return clean_characters(". ".join(parts))


def load_encoder(model: str, batch_size: int = 64) -> Encoder:
    """An encoder for ``model``. The model itself (a download the first time) is only loaded
    when there is something to encode, so a fully cached run never touches it."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            'embeddings need sentence-transformers: pip install -e ".[embeddings]"'
        ) from exc
    loaded: list = []

    def encode(texts: list[str]) -> np.ndarray:
        if not loaded:
            loaded.append(SentenceTransformer(model))
        return loaded[0].encode(
            texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True
        )

    return encode


def _read_cache(path: Path) -> dict[str, tuple[str, np.ndarray]]:
    if not path.exists():
        return {}
    df = read_parquet(path)
    return {
        pid: (digest, np.asarray(vec, dtype=np.float32))
        for pid, digest, vec in zip(df["id"], df["digest"], df["vector"], strict=True)
    }


def embed(
    ids: Sequence[str], texts: Sequence[str], cache_path: str | Path, encoder: Encoder
) -> np.ndarray:
    """Vectors for ``texts`` (one row each), encoding only papers not already cached."""
    path = Path(cache_path)
    cache = _read_cache(path)
    digests = [_digest(t) for t in texts]
    missing = [i for i, (pid, d) in enumerate(zip(ids, digests, strict=True))
               if pid not in cache or cache[pid][0] != d]  # fmt: skip
    if missing:
        log.info("encoding %d papers (%d cached)", len(missing), len(ids) - len(missing))
        vectors = np.asarray(encoder([texts[i] for i in missing]), dtype=np.float32)
        for i, vector in zip(missing, vectors, strict=True):
            cache[ids[i]] = (digests[i], vector)
        write_parquet(
            pd.DataFrame(
                {
                    "id": list(cache),
                    "digest": [d for d, _ in cache.values()],
                    "vector": [v.tolist() for _, v in cache.values()],
                }
            ),
            path,
        )
    return np.vstack([cache[pid][1] for pid in ids])


def load_vectors(ids: Sequence[str], texts: Sequence[str], cache_path: str | Path) -> np.ndarray:
    """Cached vectors for exactly these papers; raises if any is missing or out of date."""
    cache = _read_cache(Path(cache_path))
    stale = [pid for pid, t in zip(ids, texts, strict=True)
             if pid not in cache or cache[pid][0] != _digest(t)]  # fmt: skip
    if stale:
        raise ValueError(
            f"{len(stale)} papers have no up-to-date embedding in {cache_path}; "
            "run `informatics embed` with the same --set options first"
        )
    return np.vstack([cache[pid][1] for pid in ids])
