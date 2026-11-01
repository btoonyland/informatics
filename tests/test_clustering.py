import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from informatics.analysis import paper_texts
from informatics.clustering import TermAnalyzer, cluster_papers
from informatics.normalize import Normalizer


@pytest.fixture(scope="module")
def ai_medicine(corpus):
    return corpus[corpus["kind"] == "ai_medicine"].reset_index(drop=True)


def test_auto_k_recovers_the_planted_topics(ai_medicine):
    result = cluster_papers(
        paper_texts(ai_medicine),
        ai_medicine["title"].tolist(),
        ai_medicine["id"].tolist(),
        k_candidates=(2, 3, 4, 5, 6),
        min_df=3,
    )
    assert result.k == 4
    assert adjusted_rand_score(ai_medicine["topic"], result.labels) == pytest.approx(1.0)
    assert list(result.k_selection["k"]) == [2, 3, 4, 5, 6]
    sizes = result.summary["papers"].tolist()
    assert sizes == sorted(sizes, reverse=True)  # cluster 0 is the largest
    assert result.summary["papers"].sum() == len(ai_medicine)
    assert all(len(titles) == 3 for titles in result.summary["example_titles"])


def test_labels_exclude_selection_keywords(ai_medicine):
    result = cluster_papers(
        paper_texts(ai_medicine),
        ai_medicine["title"].tolist(),
        ai_medicine["id"].tolist(),
        k=4,
        min_df=3,
    )
    banned = {"health", "patient", "patients", "clinical", "medical", "deep learning", "ai"}
    for terms in result.summary["top_terms"]:
        assert not banned & set(terms)
    assert result.k_selection.empty


def test_analyzer_keeps_phrases_containing_selection_words():
    text = "Mental health of patients; deep learning for health; deep-learning models."
    normalizer = Normalizer.from_texts([text])
    terms = TermAnalyzer(normalizer)(text)
    assert "mental health" in terms  # a phrase that merely contains "health" stays
    for keyword in ("health", "patients", "deep learning", "deep-learning"):
        assert normalizer.key_of(keyword) not in terms


def test_too_little_data_raises():
    with pytest.raises(ValueError):
        cluster_papers(["a b", "c d"], ["t1", "t2"], ["1", "2"], min_df=1)


def test_fixed_k_must_be_valid(ai_medicine):
    texts = paper_texts(ai_medicine)
    with pytest.raises(ValueError, match="k must be"):
        cluster_papers(texts, list(texts), list(np.arange(len(texts)).astype(str)), k=1, min_df=3)
