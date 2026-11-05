import json

import pandas as pd
import pytest

from informatics import keywords
from informatics.filtering import STRATA, FilterConfig, annotate, run_filter
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


@pytest.mark.parametrize(
    ("text", "ai", "medicine"),
    [
        ("convolutional networks for tumour segmentation in mri", True, True),
        ("a u-net for retinal vessel segmentation", True, True),
        ("gpt-4 answers usmle questions", True, True),
        ("random forests predict sepsis in the icu", True, True),
        ("we fine-tuned bert on discharge summaries", True, True),
        # Words with a common unrelated sense only count in phrases that fix the sense.
        ("pen strokes in handwriting", False, False),
        ("college admissions and partial discharge", False, False),
        ("this aids the analysis", False, False),
        ("recurrent stroke after ischemic stroke", False, True),
    ],
)
def test_wide_net_terms(text, ai, medicine):
    assert bool(keywords.AI.hits(text) + keywords.AI_CONTEXT.hits(text)) is ai
    assert bool(keywords.MEDICINE.hits(text) + keywords.MEDICINE_CONTEXT.hits(text)) is medicine


def test_wide_net_strata():
    df = pd.DataFrame(
        {
            "title": ["XOR at a single vertex", "Convolutional networks", "Gradient boosting"],
            "abstract": [
                "Implications for AI. Medical knowledge aside.",
                "They segment tumours in MRI scans.",
                "It ranks web pages.",
            ],
        }
    )
    out = annotate(df, FilterConfig())
    assert out["stratum"].tolist() == ["strict", "wide_only", "ai_only"]
    assert out["keyword_rule"].tolist() == [True, False, False]
    assert out.loc[1, "ai_terms"] == ["network architectures"] and out.loc[1, "ai_hits"] == 0
    # Raising the bar of the original rule moves a paper to the wide-only candidates.
    assert annotate(df, FilterConfig(min_medicine_hits=2))["stratum"][0] == "wide_only"
    with pytest.raises(ValueError, match="at least 1"):
        FilterConfig(min_ai_hits=0)


def test_keywords_match_after_tex_cleanup():
    df = pd.DataFrame({"title": [r"\emph{Deep‐learning} for \textit{patients}"],
                       "abstract": [""]})  # fmt: skip
    out = annotate(df, FilterConfig())
    assert out.loc[0, "ai_terms"] == ["deep learning"] and out.loc[0, "keyword_rule"]


def test_run_filter_on_synthetic_corpus(layout, corpus):
    summary = run_filter(
        layout.raw("cs"),
        "cs",
        layout.candidates("cs"),
        layout.background("cs"),
        layout.filter_summary("cs"),
        FilterConfig(background_size=50, batch_size=100),
    )
    kinds = corpus.set_index("id")["kind"]
    candidates = read_parquet(layout.candidates("cs"))
    background = read_parquet(layout.background("cs"))
    # Re-posted papers are candidates but flagged; withdrawn and French papers are excluded.
    strict = candidates[candidates["keyword_rule"]]
    assert set(kinds[strict["id"]]) == {"ai_medicine", "duplicate"}
    assert len(strict) == kinds.isin(["ai_medicine", "duplicate"]).sum()
    wide = candidates[~candidates["keyword_rule"]]
    assert set(kinds[wide["id"]]) == {"wide_relevant", "wide_irrelevant"}
    assert candidates["duplicate_of"].notna().sum() == (kinds == "duplicate").sum() == 3
    assert set(kinds[background["id"]]) == {"ai_only"} and len(background) == 50

    counts = summary["counts"]
    assert counts["scanned"] == len(corpus)
    assert sum(counts[s] for s in STRATA) == len(corpus)
    assert counts["candidates"] == len(candidates) == counts["strict"] + counts["wide_only"]
    assert counts["wide_only"] == kinds.str.startswith("wide_").sum()
    assert counts["excluded_withdrawn"] == 3 and counts["excluded_non_english"] == 2
    assert counts["excluded"] == 5 and counts["near_duplicates"] == 3
    assert sum(y["scanned"] for y in summary["by_year"].values()) == len(corpus)
    assert sum(y.get("candidates", 0) for y in summary["by_year"].values()) == len(candidates)
    assert summary["keywords"] == keywords.keyword_lists()  # patterns recorded

    row = candidates.iloc[0]
    assert isinstance(row["ai_terms"], list) and isinstance(row["themes"], list)
    assert row["corpus"] == "cs"
    assert json.loads(layout.filter_summary("cs").read_text())["set"] == "cs"


def test_background_sample_is_reproducible(layout):
    paths = [layout.background("cs"), layout.data_dir / "bg2.parquet"]
    for path in paths:
        config = FilterConfig(background_size=20, batch_size=30)
        run_filter(
            layout.raw("cs"),
            "cs",
            layout.candidates("cs"),
            path,
            layout.filter_summary("cs"),
            config,
        )
    first, second = (read_parquet(p)["id"].tolist() for p in paths)
    assert first == second


def test_year_window_drops_other_years_first(layout, corpus):
    config = FilterConfig(background_size=30, batch_size=100, years=(2015, 2020))
    summary = run_filter(layout.raw("cs"), "cs", layout.candidates("cs"), layout.background("cs"),
                         layout.filter_summary("cs"), config)  # fmt: skip
    years = corpus.set_index("id")["year"]
    inside = years.between(2015, 2020)
    counts = summary["counts"]
    assert counts["scanned"] == inside.sum() and counts["outside_years"] == (~inside).sum()
    assert sum(counts[s] for s in STRATA) == counts["scanned"]
    assert {int(y) for y in summary["by_year"]} <= set(range(2015, 2021))
    for path in (layout.candidates("cs"), layout.background("cs")):
        assert read_parquet(path)["year"].between(2015, 2020).all()
    assert list(summary["config"]["years"]) == [2015, 2020]
    with pytest.raises(ValueError, match="end is before the start"):
        FilterConfig(years=(2021, 2019))
