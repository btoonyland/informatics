from collections import Counter

import numpy as np
import pytest

from informatics.normalize import Normalizer, prepare_text
from informatics.phrases import (
    PhraseIndex,
    compare_rows,
    contains_selection_term,
    frequency_table,
    keep_maximal,
    log_odds,
    surface_forms,
)
from informatics.text import singular


def written_phrases(text: str) -> list[str]:
    return [w for _, w in Normalizer.from_texts([text]).phrase_pairs(text)]


# Each row: text in an abstract, the phrase that must now be counted, and the artefact the
# first version produced instead (see the review of the old analysis.py).
@pytest.mark.parametrize(
    ("text", "expected", "old_artefact"),
    [
        ("low back pain", "low back pain", "low pain"),
        ("adverse side effects", "adverse side effects", "adverse effects"),
        ("region of interest", "region of interest", None),
        ("serious adverse events", "serious adverse events", None),  # was cut to "adverse events"
        ("large language models (LLMs)", "large language models", "large language llms"),
        ("medical image analysis", "medical image analysis", None),
        ("chest X-ray images", "chest x-ray images", "chest ray images"),
        ("U-Net segmentation", "u-net segmentation", "net segmentation"),
        ("naïve Bayes classifier", "naïve bayes classifier", "na ve bayes"),
        ("SARS-CoV-2 infection", "sars-cov-2 infection", "sars cov"),
        ("data privacy and data sharing", "data privacy", "privacy sharing"),
        ("end-to-end training", "end-to-end training", "end end"),
    ],
)
def test_phrases_are_real_text(text, expected, old_artefact):
    found = written_phrases(text)
    assert expected in found
    if old_artefact:
        assert old_artefact not in found


def test_brackets_end_a_phrase():
    found = written_phrases("artificial intelligence (AI) in healthcare")
    assert "artificial intelligence" in found
    assert not any(p.endswith(" ai") for p in found)


def test_possessives_and_urls():
    assert "alzheimer disease" in written_phrases("Alzheimer's disease")
    assert written_phrases("see https://github.com/x/y for code") == []


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("networks", "network"), ("studies", "study"), ("classes", "class"),
        ("x-rays", "x-ray"), ("llms", "llm"), ("diseases", "disease"), ("analysis", "analysis"),
        ("diabetes", "diabetes"), ("genomics", "genomics"), ("bias", "bias"), ("loss", "loss"),
        ("gas", "gas"),
    ],
)  # fmt: skip
def test_singular_fallback(word, expected):
    assert singular(word) == expected


def test_index_counts_papers_not_mentions_and_merges_variants():
    texts = [
        "Neural networks, neural networks and a neural network.",
        "One neural network was fine-tuned.",
        "We study fine tuning; fine-tuning matters.",
    ]
    prepared = [prepare_text(t) for t in texts]
    normalizer = Normalizer.from_texts(prepared, min_hyphenated=1)
    index = PhraseIndex.build(prepared, normalizer, n_min=1, n_max=2, min_df=1)
    counts = index.counts()
    assert index.n_docs == 3
    assert counts["neural network"] == 2  # singular and plural merged, once per paper
    assert counts["finetune"] == 2  # "fine-tuned", "fine tuning" and "fine-tuning" are one term
    rows = np.array([True, False, False])
    assert index.counts(rows)["neural network"] == 1
    assert index.counts_in(["a neural network", "nothing here"])["neural network"] == 1


def test_index_with_no_frequent_phrase_is_empty():
    index = PhraseIndex.build(["one text"], Normalizer.from_texts(["one text"]), min_df=5)
    assert index.counts() == {} and index.counts_in(["x"]) == {}


def test_surface_form_is_the_most_common_spelling():
    texts = ["neural networks"] * 3 + ["neural network"]
    normalizer = Normalizer.from_texts(texts)
    assert surface_forms(texts, {"neural network"}, normalizer, 2, 2) == {
        "neural network": "neural networks"
    }


def test_keep_maximal_removes_fragments_only():
    counts = {
        "magnetic resonance": 940,
        "resonance imaging": 650,
        "magnetic resonance imaging": 647,
        "electronic health": 982,
        "health record": 793,
        "electronic health record": 725,
    }
    kept = keep_maximal(counts)
    assert "resonance imaging" not in kept and "health record" not in kept
    assert {"magnetic resonance", "magnetic resonance imaging", "electronic health"} <= set(kept)


def test_log_odds_ranks_enriched_phrases_first():
    target = {"clinical notes": 40, "proposed method": 50, "graph theory": 1}
    background = {"clinical notes": 2, "proposed method": 500, "graph theory": 300}
    table = log_odds(target, 100, background, 1000)
    assert table["key"].tolist()[0] == "clinical notes"
    z = table.set_index("key")["z"]
    assert z["clinical notes"] > 0 > z["graph theory"]
    assert abs(z["proposed method"]) < z["clinical notes"]  # equally common in both groups


def test_compare_rows_finds_rising_and_falling_phrases():
    texts = ["convolutional networks rule"] * 6 + ["large language models rule"] * 6
    normalizer = Normalizer.from_texts(texts)
    index = PhraseIndex.build(texts, normalizer, min_df=2)
    recent = np.array([False] * 6 + [True] * 6)
    table = compare_rows(index, recent, ~recent).set_index("key")
    assert table.loc["large language model", "z"] > 0 > table.loc["convolutional network", "z"]
    assert table.loc["convolutional network", "papers"] == 0  # absent recently, still listed


def test_frequency_table_flags_selection_terms():
    table = frequency_table(Counter({"deep learning": 5, "low back pain": 3}), 10, {}, top=10)
    assert table["phrase"].tolist() == ["deep learning", "low back pain"]
    assert table["share"].tolist() == [0.5, 0.3]
    assert table["contains_selection_term"].tolist() == [True, False]
    assert contains_selection_term("mental health")
