"""Data cleaning before analysis: withdrawn papers, non-English abstracts and near-duplicates."""

from __future__ import annotations

import math
import re
import zlib
from collections import defaultdict
from collections.abc import Sequence

import numpy as np
import simplemma

# arXiv replaces a withdrawn paper's comments (and sometimes its abstract) with a notice.
_WITHDRAWN_COMMENT = re.compile(r"\bwithdrawn\b", re.I)
_WITHDRAWN_ABSTRACT = re.compile(
    r"^\s*(?:this|the) (?:paper|article|manuscript|submission|preprint) "
    r"(?:has been|was|is) withdrawn",
    re.I,
)
_WORDS = re.compile(r"[^\W\d_]+")


def _text(value: object) -> str:
    return "" if value is None or (isinstance(value, float) and math.isnan(value)) else str(value)


def is_withdrawn(comments: object, abstract: object) -> bool:
    return bool(
        _WITHDRAWN_COMMENT.search(_text(comments)) or _WITHDRAWN_ABSTRACT.match(_text(abstract))
    )


def is_non_english(text: str, threshold: float = 0.5, min_words: int = 20) -> bool:
    """True when fewer than half of the words are known English words (simplemma's dictionary).

    English abstracts score above 0.95 even when full of technical terms; French, German, Spanish
    and Portuguese ones score 0.1-0.35. Bilingual abstracts (about 0.7) count as English. Short
    texts are never flagged.
    """
    if len(_WORDS.findall(text)) < min_words:
        return False
    return simplemma.in_target_language(text, lang="en") < threshold


def _shingles(text: str, size: int = 5) -> np.ndarray:
    words = re.findall(r"\w+", text.lower())
    grams = (
        [" ".join(words[i : i + size]) for i in range(len(words) - size + 1)]
        if len(words) >= size
        else [" ".join(words)]
    )
    return np.unique(np.array([zlib.crc32(g.encode()) for g in grams if g], dtype=np.uint64))


def near_duplicates(
    ids: Sequence[str],
    texts: Sequence[str],
    threshold: float = 0.8,
    num_perm: int = 64,
    bands: int = 16,
    seed: int = 42,
) -> dict[str, str]:
    """Map each near-duplicate paper's ID to the ID that is kept (the smallest in its group).

    MinHash signatures of word 5-gram shingles, banded locality-sensitive hashing to find
    candidate pairs, then the exact Jaccard similarity of the shingle sets (>= ``threshold``).
    Papers re-posted under a second ID are the usual case.
    """
    prime = np.uint64(4294967311)  # > 2**32; a * h stays below 2**63, so uint64 cannot overflow
    rng = np.random.default_rng(seed)
    a = rng.integers(1, 2**31, size=(num_perm, 1), dtype=np.uint64)
    b = rng.integers(0, 2**31, size=(num_perm, 1), dtype=np.uint64)
    shingles = [_shingles(t) for t in texts]
    rows = num_perm // bands
    buckets: dict[tuple, list[int]] = defaultdict(list)
    for i, sh in enumerate(shingles):
        if sh.size == 0:
            continue
        signature = ((a * sh[None, :] + b) % prime).min(axis=1)
        for band in range(bands):
            buckets[(band, signature[band * rows : (band + 1) * rows].tobytes())].append(i)

    parent = list(range(len(texts)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    checked: set[tuple[int, int]] = set()
    for members in buckets.values():
        if len(members) < 2:
            continue
        pairs = (
            [(x, y) for n, x in enumerate(members) for y in members[n + 1 :]]
            if len(members) <= 50
            else list(zip(members, members[1:], strict=False))
        )
        for x, y in pairs:
            if (x, y) in checked or root(x) == root(y):
                continue
            checked.add((x, y))
            overlap = np.intersect1d(shingles[x], shingles[y], assume_unique=True).size
            union = shingles[x].size + shingles[y].size - overlap
            if union and overlap / union >= threshold:
                parent[root(x)] = root(y)

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(texts)):
        groups[root(i)].append(i)
    duplicates = {}
    for members in groups.values():
        if len(members) > 1:
            keep = min(ids[i] for i in members)
            duplicates.update({ids[i]: keep for i in members if ids[i] != keep})
    return duplicates
