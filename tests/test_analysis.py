import json

import pandas as pd
import pytest

from informatics.analysis import AnalysisConfig, combine, papers_by_year, run_analysis
from informatics.filtering import FilterConfig, run_filter
from informatics.records import RAW_SCHEMA
from informatics.storage import read_json, read_parquet, write_parquet

CONFIG = AnalysisConfig(min_df=3, cluster_min_df=3, k_candidates=(2, 3, 4, 5, 6))


def _filter(layout, set_name="cs"):
    run_filter(
        layout.raw(set_name),
        set_name,
        layout.selected(set_name),
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
        "clusters.csv", "k_selection.csv", "themes.csv", "themes_by_year.csv",
        "themes_by_cluster.csv", "papers_by_year.csv", "run_info.json", "report.md",
    }  # fmt: skip
    assert {p.name for p in out.iterdir()} == expected

    papers = read_parquet(out / "papers.parquet")
    assert len(papers) == (corpus["kind"] == "ai_medicine").sum()
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

    info = read_json(out / "run_info.json")
    assert info["counts"]["papers"] == len(papers) and info["counts"]["k"] == 4
    assert {"pandas", "scikit-learn"} <= set(info["versions"])
    report = (out / "report.md").read_text(encoding="utf-8")
    assert f"**{len(papers):,} papers**" in report and "## Topic clusters (k = 4)" in report


def test_rerun_without_background_removes_stale_files(layout):
    _filter(layout)
    out = run_analysis(["cs"], layout, config=CONFIG)
    assert (out / "distinctive_phrases.csv").exists()
    config = AnalysisConfig(min_df=3, cluster_min_df=3, use_background=False, k=4)
    run_analysis(["cs"], layout, config=config)
    assert not (out / "distinctive_phrases.csv").exists()
    assert not (out / "k_selection.csv").exists()


def test_cross_listed_papers_are_counted_once(layout, corpus):
    # Build a second set that shares half of its papers with the first.
    shared = corpus.iloc[::2].drop(columns=["kind", "topic"])
    write_parquet(shared, layout.raw("q-bio"), schema=RAW_SCHEMA)
    _filter(layout, "cs")
    _filter(layout, "q-bio")
    out = run_analysis(["cs", "q-bio"], layout, config=CONFIG)
    assert out.name == "cs+q-bio"
    papers = read_parquet(out / "papers.parquet")
    assert papers["id"].is_unique
    assert len(papers) == (corpus["kind"] == "ai_medicine").sum()
    both = papers["corpora"].map(len) == 2
    assert both.sum() == (shared.merge(corpus[["id", "kind"]])["kind"] == "ai_medicine").sum()


def test_combine_merges_corpora():
    frames = [
        pd.DataFrame({"id": ["1", "2"], "title": ["a", "b"], "corpus": "cs"}),
        pd.DataFrame({"id": ["2", "3"], "title": ["b", "c"], "corpus": "q-bio"}),
    ]
    combined = combine(frames).set_index("id")
    assert combined["corpora"].to_dict() == {"1": ["cs"], "2": ["cs", "q-bio"], "3": ["q-bio"]}


def test_papers_by_year_shares():
    papers = pd.DataFrame({"year": pd.array([2020, 2020, 2021], dtype="Int16")})
    by_year = {
        "2019": {"scanned": 5, "ai": 0},
        "2020": {"scanned": 10, "ai": 4, "selected": 2},
        "2021": {"scanned": 4, "ai": 2, "selected": 1},
    }
    summaries = {"cs": {"by_year": by_year}}
    table = papers_by_year(papers, summaries).set_index("year")
    assert table.loc[2020, "selected"] == 2 and table.loc[2019, "selected"] == 0
    assert table.loc[2020, "cs_selected_share"] == pytest.approx(0.2)
    assert table.loc[2021, "cs_selected_share"] == pytest.approx(0.25)


def test_missing_input_has_helpful_error(layout):
    with pytest.raises(FileNotFoundError, match="informatics filter --set cs"):
        run_analysis(["cs"], layout, config=CONFIG)
