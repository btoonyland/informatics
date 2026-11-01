"""Phrase counts: how many papers use each phrase, and which phrases set a group of papers apart.

Counts are numbers of papers (document frequencies), not mentions, so they can be reported as
"x% of papers" and compared between groups of different sizes. Phrases are counted by their
normalised key (see normalize.py), so a paper that writes "fine-tuning" and "fine-tuned" counts
once, and tables show each key's most common written form.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import partial

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

from informatics import keywords
from informatics.normalize import Normalizer


@dataclass
class PhraseIndex:
    """Which papers contain which phrase keys: a binary papers x phrases matrix."""

    matrix: sparse.csr_matrix
    keys: np.ndarray
    vectorizer: CountVectorizer | None

    @classmethod
    def build(
        cls,
        texts: Sequence[str],
        normalizer: Normalizer,
        n_min: int = 2,
        n_max: int = 3,
        min_df: int = 5,
    ) -> PhraseIndex:
        analyzer = partial(normalizer.phrase_keys, n_min=n_min, n_max=n_max)
        vectorizer = CountVectorizer(analyzer=analyzer, binary=True, min_df=min_df)
        try:
            matrix = vectorizer.fit_transform(texts).tocsr()
        except ValueError:  # no phrase occurs in enough papers
            return cls(sparse.csr_matrix((len(texts), 0), dtype=np.int64), np.array([]), None)
        return cls(matrix, vectorizer.get_feature_names_out(), vectorizer)

    @property
    def n_docs(self) -> int:
        return self.matrix.shape[0]

    def column_sums(self, rows: np.ndarray | None = None) -> np.ndarray:
        matrix = self.matrix if rows is None else self.matrix[rows]
        return np.asarray(matrix.sum(axis=0)).ravel()

    def counts(self, rows: np.ndarray | None = None) -> dict[str, int]:
        return dict(zip(self.keys, self.column_sums(rows).astype(int).tolist(), strict=True))

    def counts_in(self, texts: Sequence[str]) -> dict[str, int]:
        """Counts of this index's phrases in other papers (e.g. the comparison set)."""
        if self.vectorizer is None or not len(texts):
            return {}
        sums = np.asarray(self.vectorizer.transform(texts).sum(axis=0)).ravel()
        return dict(zip(self.keys, sums.astype(int).tolist(), strict=True))


def surface_forms(
    texts: Iterable[str], keys: set[str], normalizer: Normalizer, n_min: int = 1, n_max: int = 3
) -> dict[str, str]:
    """The most common written form of each key ("neural networks" rather than "neural network")."""
    seen: dict[str, Counter[str]] = defaultdict(Counter)
    for text in texts:
        for key, written in normalizer.phrase_pairs(text, n_min, n_max):
            if key in keys:
                seen[key][written] += 1
    return {k: seen[k].most_common(1)[0][0] if seen[k] else k for k in keys}


def keep_maximal(counts: dict[str, int], ratio: float = 0.9) -> dict[str, int]:
    """Drop a phrase when a longer phrase containing it accounts for >= ``ratio`` of its papers.

    E.g. "resonance imaging" disappears when "magnetic resonance imaging" covers 90% of its uses,
    while "magnetic resonance" stays if it is also used in other ways.
    """
    drop = set()
    for phrase, count in counts.items():
        words = phrase.split(" ")
        if len(words) < 2:
            continue
        for sub in (" ".join(words[:-1]), " ".join(words[1:])):
            if sub in counts and count >= ratio * counts[sub]:
                drop.add(sub)
    return {p: c for p, c in counts.items() if p not in drop}


def contains_selection_term(phrase: str) -> bool:
    """True if the phrase includes one of the keywords papers were selected on."""
    return bool(keywords.AI.regex.search(phrase) or keywords.MEDICINE.regex.search(phrase))


def frequency_table(
    counts: dict[str, int], n_docs: int, surfaces: dict[str, str], top: int
) -> pd.DataFrame:
    rows = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top]
    table = pd.DataFrame(rows, columns=["key", "papers"])
    table.insert(0, "phrase", table["key"].map(lambda k: surfaces.get(k, k)))
    table["share"] = table["papers"] / max(n_docs, 1)
    table["contains_selection_term"] = table["phrase"].map(contains_selection_term)
    return table.drop(columns="key")


def log_odds(
    target: dict[str, int],
    n_target: int,
    background: dict[str, int],
    n_background: int,
    prior: float = 10.0,
) -> pd.DataFrame:
    """Which phrases are over-represented in the target papers compared with the background.

    Log-odds ratio of "paper contains the phrase" between the two groups, smoothed with an
    informative Dirichlet prior (pseudo-counts proportional to the pooled rate; Monroe, Colaresi
    & Quinn 2008, applied to document frequencies), and its z-score. Ranking by z balances how
    much more common a phrase is against how much evidence there is for it. Negative z means the
    phrase is more common in the background.
    """
    keys = list(target)
    y_t = np.array([target[k] for k in keys], dtype=float)
    y_b = np.array([background.get(k, 0) for k in keys], dtype=float)
    pooled = (y_t + y_b) / max(n_target + n_background, 1)
    a1 = prior * pooled + 0.01
    a0 = prior * (1 - pooled) + 0.01
    lo = np.log((y_t + a1) / (n_target - y_t + a0)) - np.log((y_b + a1) / (n_background - y_b + a0))
    var = (
        1 / (y_t + a1) + 1 / (n_target - y_t + a0) + 1 / (y_b + a1) + 1 / (n_background - y_b + a0)
    )
    table = pd.DataFrame(
        {
            "key": keys,
            "papers": y_t.astype(int),
            "share": y_t / max(n_target, 1),
            "background_papers": y_b.astype(int),
            "background_share": y_b / max(n_background, 1),
            "log_odds": lo,
            "z": lo / np.sqrt(var),
        }
    )
    return table.sort_values(["z", "key"], ascending=[False, True]).reset_index(drop=True)


def compare_rows(
    index: PhraseIndex,
    rows_a: np.ndarray,
    rows_b: np.ndarray,
    keep: set[str] | None = None,
    prior: float = 10.0,
) -> pd.DataFrame:
    """Log-odds of each phrase between two groups of papers in the same index (e.g. two periods).

    Every phrase of the index takes part, including those absent from one group, so the table
    shows both rising (z > 0, more common in group a) and falling (z < 0) phrases.
    """
    sums_a, sums_b = index.column_sums(rows_a), index.column_sums(rows_b)
    target = {
        k: int(v) for k, v in zip(index.keys, sums_a, strict=True) if keep is None or k in keep
    }
    background = {k: int(v) for k, v in zip(index.keys, sums_b, strict=True)}
    return log_odds(target, int(rows_a.sum()), background, int(rows_b.sum()), prior)
