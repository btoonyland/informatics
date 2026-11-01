import math

import numpy as np
import pandas as pd
import pytest

from informatics.records import (
    RAW_COLUMNS,
    RAW_SCHEMA,
    clean_text,
    normalize_records,
    parse_list_repr,
    repair_mojibake,
)
from informatics.storage import read_parquet, write_csv, write_parquet


@pytest.mark.parametrize(
    ("cell", "expected"),
    [
        ("['Sato', 'Braga', 'Santos']", ["Sato", "Braga", "Santos"]),  # Python list repr
        ("['accuracy' 'accuracy']", ["accuracy", "accuracy"]),  # numpy repr from the old CSVs
        ("[\"O'Brien\", 'Smith']", ["O'Brien", "Smith"]),
        ("['diagnostics']", ["diagnostics"]),
        ("[]", []),
        ("", []),
        (math.nan, []),
        (np.array(["a", "b"]), ["a", "b"]),
    ],
)
def test_parse_list_repr(cell, expected):
    assert parse_list_repr(cell) == expected


def test_repair_mojibake_fixes_latin1_decoded_utf8():
    assert repair_mojibake("ZukiÄ\x87") == "Zukić"
    assert repair_mojibake("AvdagiÄ\x87") == "Avdagić"


@pytest.mark.parametrize("text", ["Zukić", "café", "plain ascii", "Gödel"])
def test_repair_mojibake_leaves_correct_text_alone(text):
    assert repair_mojibake(text) == text


def test_clean_text_collapses_whitespace():
    assert (
        clean_text("  Neural networks in 3D\n   medical scans  ")
        == "Neural networks in 3D medical scans"
    )
    assert clean_text("   ") is None
    assert clean_text(None) is None


def test_normalize_records_coerces_schema():
    df = pd.DataFrame(
        [
            {
                "id": 708.2931,
                "title": " A\n title ",
                "authors": ["Ana Kovač"],
                "categories": "cs.AI cs.LG",
            },
            {"id": None, "title": "no id is dropped"},
        ]
    )
    out = normalize_records(df, source="test")
    assert list(out.columns) == RAW_COLUMNS
    row = out.iloc[0]
    assert len(out) == 1
    assert row["id"] == "0708.2931"
    assert row["title"] == "A title"
    assert row["categories"] == ["cs.AI", "cs.LG"]
    assert (row["year"], row["month"], row["source"]) == (2007, 8, "test")


def test_parquet_round_trip_keeps_ids_and_lists(tmp_path):
    df = normalize_records(pd.DataFrame([{"id": "0704.0010", "authors": ["A B", "C D"]}]))
    write_parquet(df, tmp_path / "x.parquet", schema=RAW_SCHEMA)
    back = read_parquet(tmp_path / "x.parquet")
    assert back.loc[0, "id"] == "0704.0010"  # not 704.001
    assert back.loc[0, "authors"] == ["A B", "C D"]


def test_csv_export_json_encodes_lists(tmp_path):
    df = pd.DataFrame({"id": ["0704.0010"], "kw": [["accuracy", "bias"]]})
    write_csv(df, tmp_path / "x.csv")
    back = pd.read_csv(tmp_path / "x.csv", dtype=str)
    assert back.loc[0, "id"] == "0704.0010"
    assert back.loc[0, "kw"] == '["accuracy", "bias"]'
