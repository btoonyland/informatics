"""Topic clusters: TF-IDF of title + abstract -> LSA -> k-means, with k chosen by silhouette.

Changes from the first version:

* the selection keywords are removed from the vocabulary (every paper contains them, so they
  only blur the clusters), and words in more than ``max_df`` of the papers are dropped;
* features are words and two-word phrases built with the same rules as the phrase counts;
* k is chosen from several candidates by silhouette score, with run-to-run stability reported;
* each cluster is labelled by the terms that are much more common in it than in the other
  clusters, with its size and the titles closest to its centre.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import Normalizer

from informatics import keywords
from informatics.phrases import surface_forms
from informatics.text import fold, phrases

log = logging.getLogger(__name__)

DEFAULT_K_CANDIDATES = (5, 8, 10, 12, 15, 20, 25, 30)


class TermAnalyzer:
    """scikit-learn analyzer: words and two-word phrases, minus the selection keywords."""

    def __init__(self, drop: set[str]):
        self.drop = drop

    def __call__(self, text: str) -> list[str]:
        return [k for k in (fold(p) for p in phrases(text, 1, 2)) if k not in self.drop]


def selection_keys() -> set[str]:
    words = keywords.selection_phrases()
    return words | {fold(w) for w in words}


@dataclass
class ClusterResult:
    labels: np.ndarray  # cluster of each paper; 0 is the largest cluster
    summary: pd.DataFrame  # one row per cluster
    k_selection: pd.DataFrame  # silhouette and stability per candidate k (empty if k was fixed)
    k: int


def cluster_papers(
    texts: Sequence[str],
    titles: Sequence[str],
    ids: Sequence[str],
    *,
    k: int | str = "auto",
    k_candidates: Sequence[int] = DEFAULT_K_CANDIDATES,
    min_df: int = 5,
    max_df: float = 0.5,
    n_components: int = 100,
    seed: int = 42,
    n_terms: int = 10,
    n_examples: int = 3,
    silhouette_sample: int = 10_000,
) -> ClusterResult:
    vectorizer = TfidfVectorizer(
        analyzer=TermAnalyzer(selection_keys()), min_df=min_df, max_df=max_df, sublinear_tf=True
    )
    try:
        tfidf = vectorizer.fit_transform(texts)
    except ValueError as exc:  # e.g. no term is frequent enough
        raise ValueError(
            f"cannot cluster these papers ({exc}); try a lower --cluster-min-df"
        ) from exc
    n_docs, n_features = tfidf.shape
    if n_features < 3 or n_docs < 3:
        raise ValueError(f"too little data to cluster ({n_docs} papers, {n_features} terms)")
    components = max(2, min(n_components, n_features - 1, n_docs - 1))
    reduced = TruncatedSVD(n_components=components, random_state=seed).fit_transform(tfidf)
    reduced = Normalizer(copy=False).fit_transform(reduced)

    def fit(n_clusters: int, random_state: int) -> np.ndarray:
        return KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state).fit_predict(
            reduced
        )

    if k == "auto":
        candidates = sorted({c for c in k_candidates if 2 <= c < n_docs})
        if not candidates:
            raise ValueError(f"no candidate k is smaller than the number of papers ({n_docs})")
        rows, fitted = [], {}
        for n_clusters in candidates:
            labels = fit(n_clusters, seed)
            score = silhouette_score(
                reduced,
                labels,
                metric="cosine",
                sample_size=min(silhouette_sample, n_docs),
                random_state=seed,
            )
            stability = adjusted_rand_score(labels, fit(n_clusters, seed + 1))
            rows.append({"k": n_clusters, "silhouette": score, "stability_ari": stability})
            fitted[n_clusters] = labels
            log.info("k=%d: silhouette %.3f, stability %.2f", n_clusters, score, stability)
        k_selection = pd.DataFrame(rows)
        best = int(
            k_selection.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0]["k"]
        )
        raw_labels = fitted[best]
    else:
        best = int(k)
        if not 2 <= best < n_docs:
            raise ValueError(f"k must be between 2 and {n_docs - 1}")
        k_selection = pd.DataFrame(columns=["k", "silhouette", "stability_ari"])
        raw_labels = fit(best, seed)

    # Renumber clusters by size so cluster 0 is the largest.
    sizes = np.bincount(raw_labels, minlength=best)
    order = np.argsort(-sizes, kind="stable")
    remap = np.empty(best, dtype=int)
    remap[order] = np.arange(best)
    labels = remap[raw_labels]

    terms = vectorizer.get_feature_names_out()
    per_cluster = []
    for c in range(best):
        inside = labels == c
        mean_in = np.asarray(tfidf[inside].mean(axis=0)).ravel()
        mean_out = np.asarray(tfidf[~inside].mean(axis=0)).ravel() if (~inside).any() else 0.0
        score = mean_in - mean_out
        top = [terms[i] for i in np.argsort(-score, kind="stable")[:n_terms] if score[i] > 0]
        centre = reduced[inside].mean(axis=0)
        centre /= np.linalg.norm(centre) or 1.0
        members = np.flatnonzero(inside)
        closest = members[np.argsort(-(reduced[inside] @ centre), kind="stable")[:n_examples]]
        per_cluster.append((c, int(inside.sum()), top, closest))

    surfaces = surface_forms(texts, {t for _, _, top, _ in per_cluster for t in top}, 1, 2)
    summary = pd.DataFrame(
        {
            "cluster": [c for c, *_ in per_cluster],
            "papers": [n for _, n, _, _ in per_cluster],
            "share": [n / n_docs for _, n, _, _ in per_cluster],
            "top_terms": [[surfaces.get(t, t) for t in top] for _, _, top, _ in per_cluster],
            "example_ids": [[ids[i] for i in closest] for *_, closest in per_cluster],
            "example_titles": [[titles[i] for i in closest] for *_, closest in per_cluster],
        }
    )
    return ClusterResult(labels=labels, summary=summary, k_selection=k_selection, k=best)
