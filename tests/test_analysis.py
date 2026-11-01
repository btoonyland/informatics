import json

import numpy as np
import pandas as pd
import pytest
from synthetic import make_corpus

from informatics.analysis import (
    AnalysisConfig,
    combine,
    default_periods,
    papers_by_year,
    run_analysis,
)
from informatics.filtering import FilterConfig, run_filter
from informatics.records import RAW_SCHEMA, normalize_records
from informatics.storage import read_json, read_parquet, write_parquet

CONFIG = AnalysisConfig(
    min_df=3,
    cluster_min_df=3,
    k_candidates=(2, 3, 4, 5, 6),
    trend_min_papers=5,
    trend_min_year_papers=1,
)


def _filter(layout, set_name="cs"):
    run_filter(
        layout.raw(set_name),
        set_name,
        layout.candidates(set_name),
        layout.background(set_name),
        layout.filter_summary(set_name),
        FilterConfig(background_size=100),
    )


def test_end_to_end_outputs(layout, corpus):
    _filter(layout)
    out = run_analysis(["cs"], layout, config=CONFIG)
    assert out == layout.results(["cs"])
    expected = {
        "papers.parquet", "papers.csv", "top_phrases.csv", "distinctive_phrases.csv",
        "rising_phrases.csv", "falling_phrases.csv", "clusters.csv", "k_selection.csv",
        "themes.csv", "themes_by_year.csv", "themes_by_cluster.csv", "concepts.csv",
        "concepts_by_year.csv", "concepts_by_cluster.csv", "papers_by_year.csv",
        "publication_by_year.csv", "run_info.json", "report.md", "theme_trends.csv",
        "concept_trends.csv",
    }  # fmt: skip
    assert {p.name for p in out.iterdir()} == expected

    papers = read_parquet(out / "papers.parquet")
    # The 3 re-posted papers are dropped; withdrawn and French papers never got selected.
    assert len(papers) == (corpus["kind"] == "ai_medicine").sum()
    assert (
        papers["id"].isin(corpus.loc[corpus["kind"].isin(["withdrawn", "non_english"]), "id"]).sum()
        == 0
    )
    assert "clean_summary" not in papers  # no lossy copy of every abstract

    csv = pd.read_csv(out / "papers.csv", dtype=str)
    assert json.loads(csv.loc[0, "authors"]) == ["Ana Kovač", "Jörg Müller"]
    assert json.loads(csv.loc[0, "corpora"]) == ["cs"]

    distinctive = pd.read_csv(out / "distinctive_phrases.csv")
    assert {"phrase", "share", "background_share", "z"} <= set(distinctive.columns)
    assert distinctive["z"].is_monotonic_decreasing
    # Boilerplate shared with the AI-only papers is common but not distinctive.
    top = pd.read_csv(out / "top_phrases.csv")
    assert "results are reported" in top["phrase"].tolist()
    assert "results are reported" not in distinctive["phrase"].head(20).tolist()

    rising = pd.read_csv(out / "rising_phrases.csv")
    falling = pd.read_csv(out / "falling_phrases.csv")
    assert {"phrase", "recent_share", "earlier_share", "z"} <= set(rising.columns)
    assert (rising["z"] > 0).all() and (falling["z"] < 0).all()

    themes_by_year = pd.read_csv(out / "themes_by_year.csv")
    columns = ["year", "papers_in_group", "tag", "papers", "share", "ci_low", "ci_high"]
    assert list(themes_by_year.columns) == columns
    assert (themes_by_year["ci_low"] <= themes_by_year["share"]).all()
    assert (themes_by_year["share"] <= themes_by_year["ci_high"]).all()

    concepts = pd.read_csv(out / "concepts.csv").set_index(["dimension", "tag"])
    assert concepts.loc[("data type", "medical images"), "papers"] >= 30  # the imaging topic
    # 3 of the language topic's 12 words are clinical text, so ~75% of its 40 papers qualify.
    assert concepts.loc[("data type", "clinical text"), "papers"] >= 20
    assert concepts.loc[("clinical task", "segmentation"), "papers"] > 0

    publication = pd.read_csv(out / "publication_by_year.csv")
    with_ref = (publication["share_journal_ref"] * publication["papers"]).round().sum()
    assert with_ref == papers["journal_ref"].notna().sum() > 0
    assert set(papers["publication_status"]) == {
        "journal reference", "accepted (comments)", "no evidence",
    }  # fmt: skip

    theme_trends = pd.read_csv(out / "theme_trends.csv").set_index("tag")
    # Theme sentences were added at random, so no theme has a real trend.
    assert (
        theme_trends.loc["privacy", "adjusted_ci_low"]
        < 1
        < theme_trends.loc["privacy", "adjusted_ci_high"]
    )
    concept_trends = pd.read_csv(out / "concept_trends.csv")
    assert set(concept_trends["dimension"]) == {
        "disease area", "data type", "clinical task", "AI method",
    }  # fmt: skip

    info = read_json(out / "run_info.json")
    assert info["counts"]["papers"] == len(papers) and info["counts"]["k"] == 4
    assert info["counts"]["near_duplicates_dropped"] == 3
    assert info["periods"]["recent"][1] == papers["year"].max()
    assert {"pandas", "scikit-learn"} <= set(info["versions"])
    report = (out / "report.md").read_text(encoding="utf-8")
    assert f"**{len(papers):,} papers**" in report and "## Topic clusters (k = 4" in report
    for section in (
        "## Concepts",
        "## Publication",
        "### Rising phrases",
        "3 withdrawn, 2 not in English",
        "selected by the keyword rule",
        "### Trends (",
    ):
        assert section in report
    assert info["selection"]["cs"]["method"] == "keywords"
    assert info["selection"]["cs"]["duplicates_dropped"] == 3


def test_text_cache_is_reused(layout):
    _filter(layout)
    run_analysis(["cs"], layout, config=CONFIG)
    cache = read_parquet(layout.text_cache())
    assert len(cache) > 0 and set(cache["version"]) == {"1"}
    modified = layout.text_cache().stat().st_mtime
    run_analysis(["cs"], layout, config=CONFIG)
    assert layout.text_cache().stat().st_mtime == modified  # nothing new to prepare


def test_explicit_periods(layout):
    _filter(layout)
    config = AnalysisConfig(
        min_df=3, cluster_min_df=3, k=4, recent=(2020, 2024), earlier=(2010, 2014)
    )
    out = run_analysis(["cs"], layout, config=config)
    assert read_json(out / "run_info.json")["periods"] == {
        "earlier": [2010, 2014],
        "recent": [2020, 2024],
    }


def test_rerun_without_background_removes_stale_files(layout):
    _filter(layout)
    out = run_analysis(["cs"], layout, config=CONFIG)
    assert (out / "distinctive_phrases.csv").exists()
    config = AnalysisConfig(min_df=3, cluster_min_df=3, use_background=False, k=4)
    run_analysis(["cs"], layout, config=config)
    assert not (out / "distinctive_phrases.csv").exists()
    assert not (out / "k_selection.csv").exists()


def _extra_papers(n_per_topic: int, seed: int) -> pd.DataFrame:
    """Synthetic papers whose IDs cannot collide with the main corpus."""
    extra = make_corpus(n_per_topic, seed=seed).drop(columns=["kind", "topic"])
    extra["id"] = [f"{i.split('.')[0]}.{9000 + n:04d}" for n, i in enumerate(extra["id"])]
    return normalize_records(extra, source="synthetic")


def test_two_sets_counted_once_and_compared(layout, corpus):
    # q-bio shares half of cs's papers and has some of its own.
    shared = corpus.iloc[::2].drop(columns=["kind", "topic"])
    write_parquet(
        pd.concat([shared, _extra_papers(5, seed=9)]), layout.raw("q-bio"), schema=RAW_SCHEMA
    )
    _filter(layout, "cs")
    _filter(layout, "q-bio")
    out = run_analysis(["cs", "q-bio"], layout, config=CONFIG)
    assert out.name == "cs+q-bio"
    papers = read_parquet(out / "papers.parquet")
    assert papers["id"].is_unique
    in_both = papers["corpora"].map(len) == 2
    assert in_both.tolist() == papers["id"].isin(shared["id"]).tolist()
    for name in ("themes_by_set.csv", "concepts_by_set.csv", "phrases_between_sets.csv"):
        assert (out / name).exists()
    between = pd.read_csv(out / "phrases_between_sets.csv")
    assert {"phrase", "cs_share", "q-bio_share", "z"} <= set(between.columns)
    assert set(pd.read_csv(out / "themes_by_set.csv")["corpus_group"]) == {
        "cs",
        "cs+q-bio",
        "q-bio",
    }


def test_combine_merges_corpora():
    frames = [
        pd.DataFrame({"id": ["1", "2"], "title": ["a", "b"], "corpus": "cs"}),
        pd.DataFrame({"id": ["2", "3"], "title": ["b", "c"], "corpus": "q-bio"}),
    ]
    combined = combine(frames).set_index("id")
    assert combined["corpora"].to_dict() == {"1": ["cs"], "2": ["cs", "q-bio"], "3": ["q-bio"]}


def test_default_periods():
    assert default_periods(np.array([2015.0, 2024.0, np.nan])) == ((2018, 2021), (2022, 2024))
    assert default_periods(np.array([np.nan])) is None


def test_papers_by_year_shares():
    papers = pd.DataFrame(
        {
            "year": pd.array([2020, 2020, 2021, 2021], dtype="Int16"),
            "corpora": [["cs"], ["cs", "q-bio"], ["cs"], ["q-bio"]],
        }
    )
    by_year = {
        "2019": {"scanned": 5, "candidates": 0},
        "2020": {"scanned": 10, "candidates": 4},
        "2021": {"scanned": 4, "candidates": 2},
    }
    summaries = {"cs": {"by_year": by_year}, "q-bio": None}
    table = papers_by_year(papers, summaries).set_index("year")
    assert table.loc[2020, "selected"] == 2 and table.loc[2019, "selected"] == 0
    assert table.loc[2021, "selected"] == 2 and table.loc[2021, "cs_selected"] == 1
    assert table.loc[2020, "cs_selected_share"] == pytest.approx(0.2)
    assert table.loc[2021, "cs_selected_share"] == pytest.approx(0.25)
    assert table.loc[2020, "cs_candidates"] == 4


def test_missing_input_has_helpful_error(layout):
    with pytest.raises(FileNotFoundError, match="informatics filter --set cs"):
        run_analysis(["cs"], layout, config=CONFIG)
