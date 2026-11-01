import math

import pandas as pd
import pytest

from informatics import concepts, tags
from informatics.stats import wilson_interval


@pytest.mark.parametrize(
    ("text", "dimension", "expected"),
    [
        ("deep learning for glioma segmentation in mri scans", "disease area", ["cancer"]),
        ("deep learning for glioma segmentation in mri scans", "data type", ["medical images"]),
        ("deep learning for glioma segmentation in mri scans", "clinical task", ["segmentation"]),
        ("deep learning for glioma segmentation in mri scans", "AI method", ["deep learning"]),
        ("summarising discharge summaries with large language models", "data type",
         ["clinical text"]),
        ("summarising discharge summaries with large language models", "AI method",
         ["language & foundation models"]),
        ("predicting icu mortality from electronic health records", "clinical task",
         ["prognosis & risk"]),
        ("radiology reports from clinical notes", "data type", ["clinical text"]),  # not images
        ("set cardinality estimation for databases", "disease area", []),  # not cardiology
    ],
)  # fmt: skip
def test_concept_tagging(text, dimension, expected):
    tagged = concepts.tag_concepts(pd.Series([text]))
    assert tagged.loc[0, concepts.column(dimension)] == expected


def test_concept_columns():
    assert concepts.column("data type") == "concepts_data_type"
    assert concepts.column("AI method") == "concepts_ai_method"


@pytest.fixture
def papers():
    return pd.DataFrame(
        {
            "year": [2020, 2020, 2020, 2021],
            "themes": [["privacy"], ["privacy", "bias"], [], ["bias"]],
        }
    )


def test_overview_with_intervals(papers):
    table = tags.overview(papers, "themes", ["privacy", "bias", "legal"]).set_index("tag")
    assert table.loc["privacy", "papers"] == 2 and table.loc["privacy", "share"] == 0.5
    low, high = wilson_interval(2, 4)
    assert (table.loc["privacy", "ci_low"], table.loc["privacy", "ci_high"]) == (low, high)
    assert table.loc["legal", "papers"] == 0 and table.loc["legal", "ci_low"] == 0.0


def test_shares_by_group(papers):
    table = tags.shares_by(papers, "themes", ["privacy", "bias"], "year")
    row = table[(table["year"] == 2020) & (table["tag"] == "privacy")].iloc[0]
    assert (row["papers_in_group"], row["papers"]) == (3, 2)
    assert row["share"] == pytest.approx(2 / 3)
    assert len(table) == 4  # 2 years x 2 tags


def test_pivot_shares_formats_cells(papers):
    table = tags.shares_by(papers, "themes", ["privacy", "bias"], "year")
    wide = tags.pivot_shares(table, "year", ["bias", "privacy"]).set_index("year")
    assert wide.loc[2020, "papers"] == 3
    assert wide.loc[2021, "bias"].startswith("100.0% (")
    plain = tags.pivot_shares(table, "year", ["privacy"], with_ci=False).set_index("year")
    assert plain.loc[2020, "privacy"] == "66.7%"


def test_wilson_interval_contains_the_share_at_the_edges():
    for successes, n in ((0, 7), (7, 7), (3, 7)):
        low, high = wilson_interval(successes, n)
        assert low <= successes / n <= high
    assert all(math.isnan(v) for v in wilson_interval(0, 0))
