"""arXiv identifier helpers.

New-style IDs look like ``YYMM.NNNN`` (2007-2014) or ``YYMM.NNNNN`` (2015 onwards); old-style IDs
look like ``hep-th/9901001``. When a CSV of IDs is read without ``dtype=str`` they are parsed as
floats, which drops the leading zero and trailing zeros (``0708.2931`` -> ``708.2931``,
``1004.2280`` -> ``1004.228``). :func:`normalize_id` reverses that damage; it is idempotent on
correct IDs.
"""

from __future__ import annotations

import math
import re

_VERSION = re.compile(r"v\d+$")
_NUMERIC = re.compile(r"(\d{1,4})\.(\d{1,5})")
_NEW_STYLE = re.compile(r"(\d{2})(\d{2})\.\d{4,5}")
_OLD_STYLE = re.compile(r"[A-Za-z-]+(?:\.[A-Za-z-]+)?/(\d{2})(\d{2})\d{3}")


def normalize_id(value: object) -> str | None:
    """Return the canonical ID (no version suffix), repairing float-mangled new-style IDs."""
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value):
            return None
        text = repr(value)
    else:
        text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    for prefix in ("oai:arXiv.org:", "arXiv:", "arxiv:"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
    text = _VERSION.sub("", text)
    if "/" in text:  # old-style IDs never went through float conversion intact
        return text
    match = _NUMERIC.fullmatch(text)
    if not match:
        return text
    yymm, number = match.groups()
    yymm = yymm.zfill(4)
    width = 5 if int(yymm) >= 1501 else 4
    if len(number) > width:
        return text
    return f"{yymm}.{number.ljust(width, '0')}"


def submission_year_month(arxiv_id: str | None) -> tuple[int, int] | None:
    """Year and month encoded in the ID (when it was first announced).

    This is more reliable than the ``created`` field of the OAI ``arXiv`` metadata format,
    which for some records lags the first submission by months.
    """
    if not arxiv_id:
        return None
    match = _NEW_STYLE.fullmatch(arxiv_id) or _OLD_STYLE.fullmatch(arxiv_id)
    if not match:
        return None
    yy, mm = (int(part) for part in match.groups())
    if not 1 <= mm <= 12:
        return None
    return (1900 + yy if yy >= 91 else 2000 + yy), mm
