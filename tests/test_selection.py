import pandas as pd
import pytest

from informatics.paths import Layout
from informatics.selection import describe_relevance, select, selected_papers
from informatics.storage import write_json, write_parquet


@pytest.fixture
def candidates():
    return pd.DataFrame(
        {
            "id": ["1", "2", "3", "4"],
            "keyword_rule": [True, True, False, True],
            "duplicate_of": [None, None, None, "1"],
        }
    )


def test_keyword_rule_drops_duplicates(tmp_path, candidates):
    chosen, selection = select(candidates, "cs", Layout(tmp_path), "auto")
    assert chosen["id"].tolist() == ["1", "2"]
    assert selection.method == "keywords" and selection.candidates == 3
    assert selection.duplicates_dropped == 1  # paper 4 re-posts paper 1


def test_relevance_decisions_are_used_when_present(tmp_path, candidates):
    layout = Layout(tmp_path)
    decisions = pd.DataFrame({"id": ["1", "2", "3"], "relevant": [False, True, True]})
    write_parquet(decisions, layout.relevance("cs"))
    write_json({"method": "llm", "model": "some-model"}, layout.relevance_info("cs"))
    chosen, selection = select(candidates, "cs", layout, "auto")
    assert chosen["id"].tolist() == ["2", "3"] and selection.method == "relevance"
    assert selection.duplicates_dropped == 0
    assert "language model (some-model)" in selection.description
    keyword_ids = select(candidates, "cs", layout, "keywords")[0]["id"].tolist()
    assert keyword_ids == ["1", "2"]


def test_stale_relevance_file_is_an_error(tmp_path, candidates):
    layout = Layout(tmp_path)
    write_parquet(pd.DataFrame({"id": ["1"], "relevant": [True]}), layout.relevance("cs"))
    with pytest.raises(ValueError, match="run it again"):
        select(candidates, "cs", layout, "relevance")
    with pytest.raises(ValueError, match="unknown selection"):
        select(candidates, "cs", layout, "everything")


def test_missing_files_have_hints(tmp_path, candidates):
    with pytest.raises(FileNotFoundError, match="informatics classify --set cs"):
        select(candidates, "cs", Layout(tmp_path), "relevance")
    with pytest.raises(FileNotFoundError, match="informatics filter --set cs"):
        selected_papers(Layout(tmp_path), "cs")


def test_describe_relevance():
    info = {"method": "model", "labels": 812, "labels_used": 812,
            "cv": {"precision": 0.91, "recall": 0.876}}  # fmt: skip
    text = describe_relevance(info)
    assert text.startswith("a classifier trained on 812 hand-labelled candidates")
    assert "precision 91%, recall 88%" in text and "hand labels used for 812 papers" in text
