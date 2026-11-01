"""Phrase counts: how many papers use each phrase, and which phrases set the selection apart.

Counts are numbers of papers (document frequencies), not numbers of mentions, so they can be
reported as "x% of papers" and compared between corpora of different sizes.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable

import numpy as np
import pandas as pd

from informatics import keywords
from informatics.text import fold, phrases


def document_frequencies(
    texts: Iterable[str],
    n_min: int = 2,
    n_max: int = 3,
    vocabulary: set[str] | None = None,
) -> tuple[Counter[str], int]:
    """(number of papers containing each phrase key, number of papers)."""
    counts: Counter[str] = Counter()
    n_docs = 0
    for text in texts:
        keys = {fold(p) for p in phrases(text, n_min, n_max)}
        if vocabulary is not None:
            keys &= vocabulary
        counts.update(keys)
        n_docs += 1
    return counts, n_docs


def surface_forms(
    texts: Iterable[str], keys: set[str], n_min: int = 1, n_max: int = 3
) -> dict[str, str]:
    """The most common way each key is written ("neural networks" rather than "neural network")."""
    seen: dict[str, Counter[str]] = defaultdict(Counter)
    for text in texts:
        for phrase in phrases(text, n_min, n_max):
            key = fold(phrase)
            if key in keys:
                seen[key][phrase] += 1
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
        for sub in (fold(" ".join(words[:-1])), fold(" ".join(words[1:]))):
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
    much more common a phrase is against how much evidence there is for it.
    """
    keys = list(target)
    y_t = np.array([target[k] for k in keys], dtype=float)
    y_b = np.array([background.get(k, 0) for k in keys], dtype=float)
    pooled = (y_t + y_b) / (n_target + n_background)
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
