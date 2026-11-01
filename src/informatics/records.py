"""The canonical schema for raw paper records, shared by every data source."""

from __future__ import annotations

import ast
import math
import re
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa

from informatics.arxiv_ids import normalize_id, submission_year_month

RAW_SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("title", pa.string()),
        ("abstract", pa.string()),
        ("authors", pa.list_(pa.string())),  # "Forenames Surname" display names
        ("categories", pa.list_(pa.string())),
        ("created", pa.string()),  # as given by the source; see arxiv_ids.submission_year_month
        ("updated", pa.string()),
        ("doi", pa.string()),
        ("journal_ref", pa.string()),
        ("comments", pa.string()),
        ("license", pa.string()),
        ("datestamp", pa.string()),  # OAI-PMH record datestamp / Kaggle update_date
        ("source", pa.string()),  # oai-pmh | kaggle | legacy-csv
        ("year", pa.int16()),  # from the arXiv ID
        ("month", pa.int8()),  # from the arXiv ID
    ]
)
RAW_COLUMNS = RAW_SCHEMA.names
TEXT_COLUMNS = [
    "id", "title", "abstract", "created", "updated", "doi",
    "journal_ref", "comments", "license", "datestamp", "source",
]  # fmt: skip
LIST_COLUMNS = ["authors", "categories"]


def clean_text(value: Any) -> str | None:
    """Collapse all whitespace (arXiv wraps long fields over several indented lines)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = " ".join(str(value).split())
    return text or None


def as_str_list(value: Any, split: bool = False) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if v is not None and str(v).strip()]
    text = str(value).strip()
    if not text:
        return []
    return text.split() if split else [text]


_ADJACENT_QUOTED = re.compile(r"""['"]\s+['"]""")
_QUOTED = re.compile(r"""(['"])(.*?)\1""")


def parse_list_repr(value: Any) -> list[str]:
    """Parse a list that was written to CSV as its Python repr.

    Handles both ``"['Sato', 'Braga']"`` (Python list) and ``"['a' 'b']"`` (numpy array).
    ``ast.literal_eval`` alone silently turns the numpy form into ``['ab']``.
    """
    if isinstance(value, (list, tuple, np.ndarray)):
        return as_str_list(value)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    text = str(value).strip()
    if not text or text == "[]":
        return []
    if not text.startswith("["):
        return [text]
    if "," not in text and _ADJACENT_QUOTED.search(text):  # numpy repr: no commas
        return [m.group(2) for m in _QUOTED.finditer(text)]
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return [m.group(2) for m in _QUOTED.finditer(text)]
    return as_str_list(parsed)


def repair_mojibake(value: Any) -> Any:
    """Undo UTF-8 text that was decoded as Latin-1 ("ZukiÄ\\x87" -> "Zukić").

    Safe on correct text: genuine non-Latin-1 characters cannot be encoded as Latin-1, and
    genuine Latin-1 text is almost never valid UTF-8 once encoded, so both are left unchanged.
    """
    if not isinstance(value, str) or value.isascii():
        return value
    try:
        return value.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return value


def normalize_records(df: pd.DataFrame, source: str | None = None) -> pd.DataFrame:
    """Coerce any source's records to :data:`RAW_SCHEMA` (missing columns become empty)."""
    df = df.copy()
    for col in RAW_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df["id"] = df["id"].map(normalize_id)
    df = df[df["id"].notna()].copy()
    for col in TEXT_COLUMNS:
        df[col] = df[col].map(clean_text).astype(object)
    df["authors"] = df["authors"].map(as_str_list)
    df["categories"] = df["categories"].map(lambda v: as_str_list(v, split=True))
    if source is not None:
        df["source"] = source
    # The arXiv ID gives the month of first submission; other sources (PubMed) use their date.
    year_month = [
        submission_year_month(i) or date_year_month(c)
        for i, c in zip(df["id"], df["created"], strict=True)
    ]
    df["year"] = pd.array([ym[0] if ym else None for ym in year_month], dtype="Int16")
    df["month"] = pd.array([ym[1] if ym else None for ym in year_month], dtype="Int8")
    return df[RAW_COLUMNS].reset_index(drop=True)


def date_year_month(value: Any) -> tuple[int, int | None] | None:
    """(year, month) from "YYYY", "YYYY-MM" or "YYYY-MM-DD"; None if not a date."""
    match = re.match(r"^(\d{4})(?:-(\d{2}))?", value) if isinstance(value, str) else None
    if not match:
        return None
    month = int(match.group(2)) if match.group(2) else None
    return int(match.group(1)), month if month and 1 <= month <= 12 else None
