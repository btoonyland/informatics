import math

import pytest

from informatics.arxiv_ids import normalize_id, submission_year_month

# Every damaged ID in the legacy results (read back from CSV as numbers) and its real value.
LEGACY_DAMAGED = {
    "708.2931": "0708.2931",
    "812.4648": "0812.4648",
    "1004.228": "1004.2280",
    "810.3671": "0810.3671",
    "904.3808": "0904.3808",
    "806.2925": "0806.2925",
    "910.2029": "0910.2029",
    "910.3348": "0910.3348",
    "912.3926": "0912.3926",
    "1004.123": "1004.1230",
}


@pytest.mark.parametrize(("damaged", "expected"), LEGACY_DAMAGED.items())
def test_repairs_ids_damaged_by_float_parsing(damaged, expected):
    assert normalize_id(damaged) == expected
    assert normalize_id(float(damaged)) == expected


@pytest.mark.parametrize(
    "valid", ["0704.0001", "1412.9999", "1501.00001", "2312.10000", "hep-th/9901001", "cs/0601001"]
)
def test_valid_ids_are_unchanged(valid):
    assert normalize_id(valid) == valid


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1501.0001, "1501.00010"),  # five-digit numbering from 2015 on
        ("0704.0001v2", "0704.0001"),
        ("hep-th/9901001v3", "hep-th/9901001"),
        ("arXiv:2101.00001", "2101.00001"),
        ("oai:arXiv.org:0901.0001", "0901.0001"),
        (" 0901.0001 ", "0901.0001"),
        (None, None),
        (math.nan, None),
        ("", None),
    ],
)
def test_normalize_id_variants(value, expected):
    assert normalize_id(value) == expected


@pytest.mark.parametrize(
    ("arxiv_id", "expected"),
    [
        ("0708.2931", (2007, 8)),
        ("2312.10000", (2023, 12)),
        ("hep-th/9901001", (1999, 1)),
        ("cs/0601001", (2006, 1)),
        ("math.GT/0309136", (2003, 9)),
        ("not-an-id", None),
        ("0713.0001", None),  # month 13
        (None, None),
    ],
)
def test_submission_year_month(arxiv_id, expected):
    assert submission_year_month(arxiv_id) == expected
