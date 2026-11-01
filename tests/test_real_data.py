"""Tests on 17 real arXiv records: the papers the first version of the pipeline selected,
kept in their original, damaged CSV form (float-mangled IDs, surnames only, list reprs)."""

import json

import pandas as pd
import pytest
from helpers import FIXTURES, check_golden

from informatics import concepts
from informatics.analysis import AnalysisConfig, run_analysis
from informatics.filtering import FilterConfig, annotate, run_filter, search_text, tag_themes
from informatics.imports import import_legacy_csv
from informatics.normalize import Normalizer, prepare_text
from informatics.paths import Layout
from informatics.phrases import PhraseIndex, keep_maximal, surface_forms
from informatics.records import title_and_abstract
from informatics.storage import read_parquet


@pytest.fixture(scope="module")
def real(tmp_path_factory):
    path = tmp_path_factory.mktemp("real") / "raw.parquet"
    assert import_legacy_csv(FIXTURES / "real_arxiv_sample.csv", path) == 17
    return read_parquet(path)


def test_legacy_import_repairs_real_records(real):
    ids = set(real["id"])
    assert {"0810.3671", "1004.1230", "1004.2280", "0708.2931"} <= ids  # leading/trailing zeros
    row = real.set_index("id").loc["0806.2925"]
    # The year comes from the identifier (June 2008), not the CSV's later "published" date.
    assert (row["year"], row["month"]) == (2008, 6) and row["created"] == "2009-06-12"
    assert row["authors"] == ["Zukić", "Elsner", "Avdagić", "Domik"]
    assert real.set_index("id").loc["1206.3777", "categories"] == ["cs.NE", "q-bio.TO"]


def test_how_the_filter_treats_real_records(real):
    tags = concepts.tag_concepts(search_text(real))
    table = pd.concat([real[["id"]], annotate(real, FilterConfig())], axis=1)
    table["themes"] = tag_themes(real)["themes"]
    for dimension in concepts.CONCEPTS:
        table[dimension] = tags[concepts.column(dimension)]
    columns = ["id", "stratum", "ai_terms", "medicine_terms", "themes", *concepts.CONCEPTS]
    table = table[columns].sort_values("id")
    for column in columns[2:]:
        table[column] = table[column].map(json.dumps)
    check_golden("real_records.csv", table)


def test_phrases_of_real_abstracts(real):
    texts = [prepare_text(t) for t in title_and_abstract(real)]
    normalizer = Normalizer.from_texts(texts)
    index = PhraseIndex.build(texts, normalizer, 2, 3, min_df=2)
    frequent = keep_maximal(index.counts(), 0.9)
    surfaces = surface_forms(texts, set(frequent), normalizer, 2, 3)
    table = pd.DataFrame(
        {"key": list(frequent), "phrase": [surfaces.get(k, k) for k in frequent],
         "papers": list(frequent.values())}
    ).sort_values(["papers", "key"], ascending=[False, True])  # fmt: skip
    assert table.iloc[0]["papers"] >= 3
    check_golden("real_phrases.csv", table)


def test_whole_pipeline_runs_on_real_records(real, tmp_path):
    from informatics.records import RAW_SCHEMA
    from informatics.storage import write_parquet

    layout = Layout(tmp_path / "data", tmp_path / "results")
    write_parquet(real, layout.raw("real"), schema=RAW_SCHEMA)
    run_filter(layout.raw("real"), "real", layout.candidates("real"), layout.background("real"),
               layout.filter_summary("real"), FilterConfig())  # fmt: skip
    config = AnalysisConfig(min_df=2, cluster_min_df=2, k_candidates=(2, 3), trend_min_papers=2,
                            trend_min_year_papers=1)  # fmt: skip
    out = run_analysis(["real"], layout, config=config)
    papers = pd.read_csv(out / "papers.csv", dtype=str)
    assert len(papers) == 17 and (out / "report.html").exists()
