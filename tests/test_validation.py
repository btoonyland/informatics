import math

import pandas as pd
import pytest

from informatics.filtering import FilterConfig, run_filter
from informatics.storage import read_json
from informatics.validation import evaluate_labels, make_label_sample, parse_label, wilson_interval


def test_wilson_interval():
    assert wilson_interval(0, 10)[0] == 0.0
    assert wilson_interval(0, 10)[1] == pytest.approx(0.2775, abs=1e-3)
    low, high = wilson_interval(8, 10)
    assert low == pytest.approx(0.4902, abs=1e-3) and high == pytest.approx(0.9433, abs=1e-3)
    assert all(math.isnan(v) for v in wilson_interval(0, 0))


@pytest.mark.parametrize(
    ("value", "expected"),
    [("1", True), ("yes", True), ("0", False), ("N", False), ("", None), (None, None)],
)
def test_parse_label(value, expected):
    assert parse_label(value) is expected


def test_parse_label_rejects_unknown_values():
    with pytest.raises(ValueError, match="unrecognised"):
        parse_label("maybe")


def _labelled_sample():
    """20 labelled candidates, 10 AI-only papers and 10 "neither" papers (one unlabelled)."""
    strata = ["strict"] * 10 + ["wide_only"] * 10 + ["ai_only"] * 10 + ["neither"] * 10
    relevant = (
        ["1"] * 8 + ["0"] * 2  # strict: 8 of 10 relevant
        + ["1"] * 5 + ["0"] * 5  # wide_only: 5 of 10
        + ["1"] + ["0"] * 9  # ai_only: 1 of 10
        + ["0"] * 9 + [""]  # neither: none; one row not labelled yet
    )  # fmt: skip
    key = pd.DataFrame({"id": [str(i) for i in range(40)], "stratum": strata})
    sheet = pd.DataFrame({"id": key["id"], "relevant": relevant})
    summary = {
        "counts": {"strict": 500, "wide_only": 200, "ai_only": 1000, "medicine_only": 0,
                   "neither": 5000}
    }  # fmt: skip
    return sheet, key, summary


def test_evaluate_labels_scales_strata_to_population():
    sheet, key, summary = _labelled_sample()
    table, result = evaluate_labels(sheet, key, summary)
    rule, wide = result["keyword_rule"], result["wide_net"]
    # Estimated relevant papers: 400 strict + 100 wide-only + 100 AI-only (10% of 1000) = 600.
    assert result["estimated_relevant_in_set"] == pytest.approx(600)
    assert rule["precision"] == pytest.approx(0.8)
    assert rule["recall"] == pytest.approx(400 / 600)
    assert rule["recall_ci"][0] < rule["recall"] < rule["recall_ci"][1]
    assert rule["kappa"] == pytest.approx(0.3)
    assert wide["recall"] == pytest.approx(500 / 600)
    assert wide["precision"] == pytest.approx(500 / 700) and wide["candidates"] == 700
    assert result["classifiers"] == {} and result["labellers"] is None
    assert result["unlabelled_rows"] == 1 and result["unlabelled_strata"] == []
    assert table.set_index("stratum").loc["neither", "labelled"] == 9


def test_evaluate_labels_measures_the_classifier():
    sheet, key, summary = _labelled_sample()
    predicted = (
        [True] * 8 + [True, False]  # strict: all relevant found, one false positive
        + [True] * 4 + [False] + [False] * 5  # wide_only: one relevant paper missed
    )  # fmt: skip
    decisions = pd.DataFrame({"id": [str(i) for i in range(20)], "predicted": predicted})
    _, result = evaluate_labels(sheet, key, summary, {"model": decisions})
    c = result["classifiers"]["model"]
    # TP = 8*50 + 4*20 = 480, FP = 1*50, FN = 1*20 among candidates, 100 outside the net.
    assert c["precision"] == pytest.approx(480 / 530)
    assert c["recall"] == pytest.approx(480 / 600)
    assert c["recall_within_candidates"] == pytest.approx(480 / 500)
    assert c["kappa"] == pytest.approx(0.355 / 0.455)
    assert c["compared"] == 20 and c["strata"] == ["strict", "wide_only"]
    assert c["precision_ci"][0] <= c["precision"] <= c["precision_ci"][1]
    # A method that only decided some papers is measured on those.
    partial = decisions[decisions["id"].astype(int) < 10]  # the strict stratum only
    c = evaluate_labels(sheet, key, summary, {"llm": partial})[1]["classifiers"]["llm"]
    assert c["compared"] == 10 and c["strata"] == ["strict"]
    assert c["precision"] == pytest.approx(400 / 450)


def test_agreement_between_labellers():
    sheet, key, summary = _labelled_sample()
    sheet["relevant_2"] = ""
    sheet.loc[:9, "relevant_2"] = ["1"] * 7 + ["0"] * 3  # disagrees on one paper
    _, result = evaluate_labels(sheet, key, summary)
    assert result["labellers"]["papers"] == 10
    assert result["labellers"]["agreement"] == pytest.approx(0.9)


def _filtered(layout):
    run_filter(layout.raw("cs"), "cs", layout.candidates("cs"), layout.background("cs"),
               layout.filter_summary("cs"), FilterConfig())  # fmt: skip
    return read_json(layout.filter_summary("cs"))


def test_label_sample_is_blind_and_stratified(layout):
    summary = _filtered(layout)
    args = (layout.raw("cs"), layout.candidates("cs"), summary)
    sheet, key = make_label_sample(*args, n_strict=20, n_wide=10, n_per_stratum=5, seed=1)
    assert list(sheet.columns) == ["id", "title", "abstract", "relevant", "notes"]  # no stratum
    assert key["stratum"].value_counts().to_dict() == {
        "strict": 20, "wide_only": 10, "ai_only": 5, "medicine_only": 5, "neither": 5,
    }  # fmt: skip
    assert sheet["id"].is_unique and set(sheet["id"]) == set(key["id"])
    again, _ = make_label_sample(*args, n_strict=20, n_wide=10, n_per_stratum=5, seed=1)
    assert again["id"].tolist() == sheet["id"].tolist()
    # Extending a sheet draws papers that are not in it yet.
    more, more_key = make_label_sample(
        *args, n_strict=20, n_wide=10, n_per_stratum=5, seed=1, exclude_ids=set(sheet["id"])
    )
    assert not set(more["id"]) & set(sheet["id"])
    assert more_key["stratum"].value_counts()["wide_only"] == 10  # 40 wide-only candidates


def test_label_sample_needs_the_current_keywords(layout):
    summary = _filtered(layout)
    summary["keywords"]["ai"] = {"ai": "ai"}
    with pytest.raises(ValueError, match="run the filter again"):
        make_label_sample(layout.raw("cs"), layout.candidates("cs"), summary)


def test_theme_sample_and_precision(layout):
    from informatics.selection import selected_papers
    from informatics.validation import evaluate_theme_labels, make_theme_sample

    _filtered(layout)
    papers, _ = selected_papers(layout, "cs", "keywords")
    sheet = make_theme_sample(papers, per_theme=5, seed=1)
    assert list(sheet.columns) == [
        "theme",
        "id",
        "matched",
        "title",
        "abstract",
        "relevant",
        "notes",
    ]
    assert sheet.groupby("theme").size().to_dict() == {
        "bias": 5,
        "ethical": 5,
        "privacy": 5,
        "regulation": 5,
    }
    assert set(sheet.loc[sheet["theme"] == "regulation", "matched"]) == {"regulatory"}
    sheet["relevant"] = "1"
    sheet.loc[sheet.index[sheet["theme"] == "bias"][:2], "relevant"] = "0"
    table = evaluate_theme_labels(sheet).set_index("theme")
    assert table.loc["bias", "precision"] == pytest.approx(0.6)
    assert table.loc["privacy", "precision"] == 1.0 and table.loc["legal", "labelled"] == 0


def test_random_theme_audit_measures_recall():
    from informatics.validation import evaluate_theme_audit, make_random_theme_sample

    papers = pd.DataFrame(
        {"id": ["a", "b", "c", "d"], "title": "t", "abstract": "x", "themes": [[]] * 4}
    )
    sheet = make_random_theme_sample(papers, n=4, seed=0).set_index("id")
    assert list(sheet.columns) == ["title", "abstract", "themes", "done", "notes"]
    sheet.loc[["a", "b", "c"], "done"] = "1"  # d not read yet
    sheet.loc["a", "themes"] = "privacy; bias"
    sheet.loc["b", "themes"] = "privacy"
    sheet = sheet.reset_index()
    tagged = {
        "keywords": pd.Series({"a": ["privacy"], "b": [], "c": ["bias"], "d": ["privacy"]}),
        "llm": pd.Series({"a": ["privacy", "bias"], "b": ["privacy"]}),  # c not classified
    }
    table = evaluate_theme_audit(sheet, tagged).set_index(["method", "theme"])
    privacy = table.loc[("keywords", "privacy")]
    assert (privacy["papers"], privacy["discussing"], privacy["tagged"]) == (3, 2, 1)
    assert privacy["precision"] == 1.0 and privacy["recall"] == pytest.approx(0.5)
    assert table.loc[("keywords", "bias"), "precision"] == pytest.approx(0.0)
    assert table.loc[("llm", "privacy"), "recall"] == 1.0
    assert table.loc[("llm", "bias"), "papers"] == 2
    with pytest.raises(ValueError, match="unknown theme"):
        evaluate_theme_audit(sheet.assign(themes="privcy"), tagged)


def test_label_sample_stays_inside_the_year_window(layout, corpus):
    config = FilterConfig(years=(2012, 2018))
    summary = run_filter(layout.raw("cs"), "cs", layout.candidates("cs"), layout.background("cs"),
                         layout.filter_summary("cs"), config)  # fmt: skip
    summary = read_json(layout.filter_summary("cs"))  # as label-sample reads it, from JSON
    _, key = make_label_sample(layout.raw("cs"), layout.candidates("cs"), summary, n_strict=10,
                               n_wide=5, n_per_stratum=5, seed=1)  # fmt: skip
    years = corpus.set_index("id").loc[key["id"], "year"]
    assert years.between(2012, 2018).all() and len(key) > 20
