"""Normalisation rules: which written forms must share a key, and which must not."""

import pytest

from informatics.cache import TextCache
from informatics.normalize import (
    Normalizer,
    american,
    apply_synonyms,
    clean_characters,
    expand_acronyms,
    lemma,
    prepare_text,
)

# A corpus in which the American forms are common, so corpus-checked spelling rules can fire,
# and in which each hyphenated compound appears at least twice.
CORPUS = [
    "multi-modal fusion; multi-modal data; multimodal models",
    "fine-tuning and fine-tuning; we fine-tuned bert",
    "pre-trained encoders and pre-trained models; pre-training",
    "chest x-ray images and x-ray scans",
    "tumor detection, tumor growth, tumor size; organized data, organized care",
    "visualization, visualization; organizational, organizational; analyzed, analyzed",
    "hemorrhage, hemorrhage; pediatric, pediatric",
    "exercise, exercise, exercise and one exercize typo",
]


@pytest.fixture(scope="module")
def normalizer():
    return Normalizer.from_texts(CORPUS)


MUST_MERGE = [
    ("neural networks", "neural network"),
    ("fine-tuning", "fine-tuned"),
    ("fine-tuning", "finetuning"),
    ("fine-tuning", "fine tuning"),
    ("pre-trained", "pretrained"),
    ("pre-training", "pretrained"),
    ("pre trained", "pretrained"),
    ("multi-modal", "multimodal"),
    ("multi modal", "multimodal"),
    ("x-rays", "x-ray"),
    ("x ray", "x-ray"),
    ("tumour", "tumor"),
    ("tumours", "tumor"),
    ("visualisation", "visualization"),
    ("organised", "organized"),
    ("organisational", "organizational"),
    ("modelling", "modeling"),
    ("labelled", "labeled"),
    ("unlabelled", "unlabeled"),
    ("haemorrhage", "hemorrhage"),
    ("paediatric", "pediatric"),
    ("centre", "center"),
    ("analysed", "analyzed"),
    ("studies", "study"),
    ("analyses", "analysis"),
    ("diagnoses", "diagnosis"),
    ("llms", "llm"),
    ("cnns", "cnn"),
    ("electronic health records", "electronic health record"),
]

MUST_NOT_MERGE = [
    # A stemmer would merge these; they mean different things in this field.
    ("specific", "specificity"),
    ("medical", "medication"),
    ("organ", "organization"),
    ("experiment", "experience"),
    ("computer", "computation"),
    ("positive", "positivity"),
    ("university", "universe"),
    ("image", "imaging"),
    # Known mistakes of dictionary lemmatizers.
    ("media", "medium"),
    ("data", "datum"),
    ("aids", "aid"),
    # Spelling rules must not create words or typos.
    ("exercise", "exercize"),
    ("four", "for"),
    ("filled", "filed"),
    ("analyses", "analyzes"),
    ("present", "presend"),
]


@pytest.mark.parametrize(("a", "b"), MUST_MERGE)
def test_variants_share_a_key(normalizer, a, b):
    assert normalizer.key_of(a) == normalizer.key_of(b)


@pytest.mark.parametrize(("a", "b"), MUST_NOT_MERGE)
def test_different_terms_keep_different_keys(normalizer, a, b):
    assert normalizer.key_of(a) != normalizer.key_of(b)


def test_open_compounds_are_only_joined_when_the_corpus_hyphenates_them():
    rare = Normalizer.from_texts(["a multi-modal model", "a multi modal model"])  # once only
    assert rare.key_of("multi modal") != rare.key_of("multimodal")
    assert Normalizer.from_texts(["in vivo and in vitro"]).key_of("in vivo") == "in vivo"


def test_spelling_rules_need_the_american_form_in_the_corpus():
    assert american("optimised", {"optimized": 5, "optimised": 2}) == "optimized"
    assert american("optimised", {"optimised": 2}) == "optimised"  # no evidence: unchanged
    assert american("supervised", {"supervised": 900, "supervized": 2}) == "supervised"
    assert american("modelling", {}) == "modeling"  # curated stems need no corpus evidence
    assert american("paediatric", {}) == "pediatric"


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("studies", "study"), ("fine-tuned", "fine-tune"), ("pretrained", "pretrain"),
        ("llms", "llm"), ("x-rays", "x-ray"), ("media", "media"), ("data", "data"),
        ("genomics", "genomics"), ("statistics", "statistics"), ("covid-19", "covid-19"),
        ("present", "present"), ("diagnoses", "diagnosis"),
    ],
)  # fmt: skip
def test_lemma(word, expected):
    assert lemma(word) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (r"Gr\"{o}bner bases", "Gröbner bases"),
        (r"Zuki\'c and \v{C}ech", "Zukić and Čech"),
        (r"\'{\i}ndice", "índice"),
        (r"$\alpha$-synuclein \emph{in vivo}\cite{smith}", "alpha -synuclein in vivo"),
        (r"accuracy of 95\% and costs of \$5", "accuracy of 95% and costs of $5"),
        ("ﬁne-tuning", "fine-tuning"),  # ligature
        ("COVID‑19", "COVID-19"),  # non-breaking hyphen
        ("cost–effectiveness", "cost-effectiveness"),  # en dash inside a word
        ("AI—a tool", "AI a tool"),  # em dash between words
        ("plain text stays plain", "plain text stays plain"),
    ],
)
def test_clean_characters(raw, expected):
    assert clean_characters(raw) == expected


def test_acronyms_defined_in_the_abstract_are_expanded():
    # The pattern from a real abstract in results/legacy (arXiv 1009.4566).
    text = (
        "Artificial neural networks (ANNs) solve classification problems. "
        "ANNs predict well, but an ANN is a black box."
    )
    expanded, definitions = expand_acronyms(text)
    assert definitions == [("ANNs", "Artificial neural networks")]
    assert "ANN" not in expanded and expanded.count("neural networks") == 3


def test_acronyms_in_the_title_before_the_definition_are_expanded_too():
    expanded, _ = expand_acronyms("LLMs for notes. We test large language models (LLMs).")
    assert expanded == "large language models for notes. We test large language models."


@pytest.mark.parametrize(
    "text",
    ["results (see Fig. 2) hold", "a cohort (n = 40)", "step (i) and (ii)", "graph theory (ABC)"],
)
def test_non_definitions_are_left_alone(text):
    assert expand_acronyms(text) == (text, [])


def test_curated_synonyms():
    assert apply_synonyms("EHR data and electronic medical records") == (
        "electronic health records data and electronic health records"
    )
    assert apply_synonyms("5 mL of saline; ML models; XAI") == (
        "5 mL of saline; machine learning models; explainable artificial intelligence"
    )


def test_prepare_text_uses_the_abstracts_own_definition_first():
    text = prepare_text("Maximum likelihood (ML) estimation. ML is fast.")
    assert "machine learning" not in text and text.count("Maximum likelihood") == 2


def test_phrases_keep_written_forms_for_display(normalizer):
    pairs = normalizer.phrase_pairs("Multi modal fusion of chest X-rays (CXR) and low back pain")
    assert ("multimodal fusion", "multi modal fusion") in pairs
    assert ("chest xray", "chest x-rays") in pairs
    assert ("low back pain", "low back pain") in pairs
    assert not any(written.endswith("cxr") for _, written in pairs)  # brackets end phrases


def test_text_cache_reuses_and_invalidates(tmp_path):
    path = tmp_path / "cache.parquet"
    cache = TextCache(path)
    first = cache.prepare(["1", "2"], ["Brain tumour (BT) study.", "EHR data"])
    cache.save()
    reloaded = TextCache(path)
    assert reloaded.entries["1"][1] == first[0] and not reloaded.changed
    again = reloaded.prepare(["1", "2"], ["Brain tumour (BT) study.", "EMR data"])
    assert reloaded.changed and again[1] == "electronic health records data"
