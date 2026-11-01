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


def test_evaluate_labels_scales_strata_to_population():
    key = pd.DataFrame(
        {
            "id": [str(i) for i in range(30)],
            "stratum": ["selected"] * 10 + ["ai_only"] * 10 + ["neither"] * 10,
        }
    )
    relevant = ["1"] * 8 + ["0"] * 2 + ["1"] + ["0"] * 9 + ["0"] * 9 + [""]  # one row unlabelled
    sheet = pd.DataFrame({"id": key["id"], "relevant": relevant})
    summary = {
        "counts": {
            "selected": 500,
            "ai_only": 1000,
            "medicine_only": 0,
            "neither": 5000,
            "below_threshold": 0,
        }
    }
    table, result = evaluate_labels(sheet, key, summary)
    assert result["precision"] == pytest.approx(0.8)
    assert result["estimated_relevant_selected"] == pytest.approx(400)
    assert result["estimated_relevant_missed"] == pytest.approx(100)  # 10% of 1000 AI-only papers
    assert result["recall"] == pytest.approx(0.8)
    assert result["unlabelled_rows"] == 1 and result["unlabelled_strata"] == []
    assert table.set_index("stratum").loc["neither", "labelled"] == 9


def test_label_sample_is_blind_and_stratified(layout):
    run_filter(layout.raw("cs"), "cs", layout.selected("cs"), layout.background("cs"),
               layout.filter_summary("cs"), FilterConfig())  # fmt: skip
    summary = read_json(layout.filter_summary("cs"))
    sheet, key = make_label_sample(
        layout.raw("cs"), layout.selected("cs"), summary, n_selected=20, n_per_stratum=5, seed=1
    )
    assert list(sheet.columns) == ["id", "title", "abstract", "relevant", "notes"]  # no stratum
    assert key["stratum"].value_counts().to_dict() == {
        "selected": 20, "ai_only": 5, "medicine_only": 5, "neither": 5,
    }  # fmt: skip
    assert sheet["id"].is_unique and set(sheet["id"]) == set(key["id"])
    again, _ = make_label_sample(
        layout.raw("cs"), layout.selected("cs"), summary, n_selected=20, n_per_stratum=5, seed=1
    )
    assert again["id"].tolist() == sheet["id"].tolist()
