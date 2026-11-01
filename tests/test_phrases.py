from collections import Counter

import pytest

from informatics.phrases import (
    contains_selection_term,
    document_frequencies,
    frequency_table,
    keep_maximal,
    log_odds,
    surface_forms,
)
from informatics.text import fold, phrases, singular


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
    found = phrases(text)
    assert expected in found
    if old_artefact:
        assert old_artefact not in found


def test_brackets_end_a_phrase():
    found = phrases("artificial intelligence (AI) in healthcare")
    assert "artificial intelligence" in found
    assert not any(p.endswith(" ai") for p in found)


def test_possessives_and_urls():
    assert "alzheimer disease" in phrases("Alzheimer's disease")
    assert phrases("see https://github.com/x/y for code") == []


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("networks", "network"), ("studies", "study"), ("classes", "class"),
        ("x-rays", "x-ray"), ("llms", "llm"), ("diseases", "disease"), ("analysis", "analysis"),
        ("diabetes", "diabetes"), ("genomics", "genomics"), ("bias", "bias"), ("loss", "loss"),
        ("gas", "gas"),
    ],
)  # fmt: skip
def test_singular(word, expected):
    assert singular(word) == expected


def test_fold_only_changes_the_last_word():
    assert fold("clinical trials") == "clinical trial"
    assert fold("patients records") == "patients record"


def test_document_frequency_counts_papers_not_mentions():
    counts, n = document_frequencies(
        ["Neural networks, neural networks and a neural network.", "One neural network."]
    )
    assert n == 2
    assert counts["neural network"] == 2  # singular and plural merged, once per paper


def test_surface_form_is_the_most_common_spelling():
    texts = ["neural networks"] * 3 + ["neural network"]
    assert surface_forms(texts, {"neural network"}, 2, 2) == {"neural network": "neural networks"}


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


def test_frequency_table_flags_selection_terms():
    table = frequency_table(Counter({"deep learning": 5, "low back pain": 3}), 10, {}, top=10)
    assert table["phrase"].tolist() == ["deep learning", "low back pain"]
    assert table["share"].tolist() == [0.5, 0.3]
    assert table["contains_selection_term"].tolist() == [True, False]
    assert contains_selection_term("mental health")
