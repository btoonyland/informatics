import json

import pandas as pd
import pytest

from informatics import keywords
from informatics.filtering import FilterConfig, annotate, run_filter
from informatics.storage import read_parquet


# Word forms the first version's patterns missed (see the review): all must match now.
@pytest.mark.parametrize(
    ("text", "matcher", "term"),
    [
        ("patients were enrolled", keywords.MEDICINE, "patient"),
        ("hospitals in europe", keywords.MEDICINE, "hospital"),
        ("a doctor reviewed", keywords.MEDICINE, "doctors"),
        ("a physician", keywords.MEDICINE, "physicians"),
        ("health-care costs", keywords.MEDICINE, "healthcare"),
        ("a neural network model", keywords.AI, "neural networks"),
        ("llms are used", keywords.AI, "llm"),
        ("large language models", keywords.AI, "llm"),
        ("deep-learning based", keywords.AI, "deep learning"),
    ],
)
def test_keyword_variants_match(text, matcher, term):
    assert matcher.hits(text) == [term]


@pytest.mark.parametrize("text", ["outpatient clinic", "first aid workers", "a healthy brain"])
def test_whole_words_only(text):
    assert keywords.MEDICINE.hits(text) == []
    assert keywords.AI.hits(text) == []


def test_health_care_counts_once():
    assert keywords.MEDICINE.hits("health-care and health care") == ["healthcare", "healthcare"]


@pytest.mark.parametrize(
    ("text", "themes"),
    [
        ("gene regulation and transcriptional regulators", []),
        ("regulatory networks of t-cells", []),
        ("the eu ai act and fda regulatory approval", ["regulation"]),
        ("mri bias field correction", []),
        ("inductive bias of transformers", []),
        ("algorithmic bias against minorities", ["bias"]),
        ("we adopt a u-net", []),
        ("clinical adoption of ai", ["adoption"]),
        ("lung compliance in ventilated patients", []),
        ("gdpr compliance", ["compliance"]),
        ("personalised medicine and personalization", ["personalize"]),
        ("maturity-onset diabetes", []),
        ("digital maturity", ["maturity"]),
        ("ethics and ethical review", ["ethical"]),
    ],
)
def test_theme_matching_and_exclusions(text, themes):
    assert keywords.THEME_MATCHER.find(text)[0] == themes


def test_theme_terms_are_distinct():
    assert keywords.THEME_MATCHER.find("accuracy, accuracy and accuracy")[1] == ["accuracy"]


def test_min_hits_creates_below_threshold_stratum():
    df = pd.DataFrame(
        {
            "title": ["XOR at a single vertex"],
            "abstract": ["Implications for AI. Medical knowledge aside."],
        }
    )
    assert annotate(df, FilterConfig())["stratum"].tolist() == ["selected"]
    strict = annotate(df, FilterConfig(min_medicine_hits=2))
    assert strict["stratum"].tolist() == ["below_threshold"]


def test_run_filter_on_synthetic_corpus(layout, corpus):
    summary = run_filter(
        layout.raw("cs"),
        "cs",
        layout.selected("cs"),
        layout.background("cs"),
        layout.filter_summary("cs"),
        FilterConfig(background_size=50, batch_size=100),
    )
    kinds = corpus.set_index("id")["kind"]
    selected = read_parquet(layout.selected("cs"))
    background = read_parquet(layout.background("cs"))
    assert set(kinds[selected["id"]]) == {"ai_medicine"}
    assert len(selected) == (kinds == "ai_medicine").sum()
    assert set(kinds[background["id"]]) == {"ai_only"} and len(background) == 50

    counts = summary["counts"]
    assert counts["scanned"] == len(corpus)
    assert sum(
        counts[s] for s in ("selected", "ai_only", "medicine_only", "neither", "below_threshold")
    ) == len(corpus)
    assert sum(y["scanned"] for y in summary["by_year"].values()) == len(corpus)
    assert summary["keywords"]["ai"]["llm"] == keywords.AI_TERMS["llm"]  # patterns recorded

    row = selected.iloc[0]
    assert isinstance(row["ai_terms"], list) and isinstance(row["themes"], list)
    assert row["corpus"] == "cs"
    assert json.loads(layout.filter_summary("cs").read_text())["set"] == "cs"


def test_background_sample_is_reproducible(layout):
    paths = [layout.background("cs"), layout.data_dir / "bg2.parquet"]
    for path in paths:
        run_filter(layout.raw("cs"), "cs", layout.selected("cs"), path, layout.filter_summary("cs"),
                   FilterConfig(background_size=20, batch_size=30))  # fmt: skip
    first, second = (read_parquet(p)["id"].tolist() for p in paths)
    assert first == second
