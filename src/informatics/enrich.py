"""Publication data: was the paper published, where, and by what kind of institution?

Two sources:

* :func:`publication_status`: offline, from arXiv's own ``journal_ref`` and ``comments``.
* OpenAlex (``informatics enrich``): venue and its type (journal, conference, repository), the
  paper's field, the types of the authors' institutions (healthcare, company, education, ...),
  countries and citations. Papers are looked up by DOI: the journal DOI from arXiv's metadata if
  there is one, otherwise arXiv's own DataCite DOI (10.48550/arXiv.<id>). OpenAlex requires a free
  API key (openalex.org/settings/api): pass ``--api-key`` or set ``OPENALEX_API_KEY``.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Iterable, Iterator, Sequence
from typing import Any

import pandas as pd

from informatics.arxiv_ids import submission_year_month
from informatics.http_client import HttpClient, HttpError

log = logging.getLogger(__name__)

OPENALEX_WORKS = "https://api.openalex.org/works"
OPENALEX_FIELDS = (
    "id,doi,publication_year,type,cited_by_count,primary_location,authorships,primary_topic"
)
_ACCEPTED = re.compile(r"\b(?:accepted|to appear|in press|published)\b", re.I)
_DOI = re.compile(r"10\.\d{4,9}/\S+", re.I)


def _present(value: object) -> bool:
    return (
        value is not None
        and not (isinstance(value, float) and math.isnan(value))
        and bool(str(value).strip())
    )


def publication_status(journal_ref: object, comments: object) -> str:
    """ "journal reference", "accepted (comments)" or "no evidence" (offline proxy)."""
    if _present(journal_ref):
        return "journal reference"
    if _present(comments) and _ACCEPTED.search(str(comments)):
        return "accepted (comments)"
    return "no evidence"


def clean_doi(value: object) -> str | None:
    """Bare lower-case DOI ("https://doi.org/10.1/ABC" -> "10.1/abc"); first one if several."""
    if not _present(value):
        return None
    match = _DOI.search(str(value))
    return match.group(0).rstrip(".,;").lower() if match else None


def candidate_dois(paper_id: str, doi: object) -> list[str]:
    """DOIs to try for one paper, best first."""
    candidates = []
    if journal := clean_doi(doi):
        candidates.append(journal)
    if submission_year_month(paper_id):  # an arXiv identifier
        candidates.append(f"10.48550/arxiv.{paper_id}".lower())
    return candidates


def parse_work(work: dict[str, Any]) -> dict[str, Any]:
    location = work.get("primary_location") or {}
    source = location.get("source") or {}
    topic = work.get("primary_topic") or {}
    authorships = work.get("authorships") or []
    institutions = [i for a in authorships for i in (a.get("institutions") or [])]
    countries = {c for a in authorships for c in (a.get("countries") or [])}
    countries |= {i["country_code"] for i in institutions if i.get("country_code")}
    return {
        "openalex_id": work.get("id"),
        "venue": source.get("display_name"),
        "venue_type": source.get("type"),
        "field": (topic.get("field") or {}).get("display_name"),
        "institution_types": sorted({i["type"] for i in institutions if i.get("type")}),
        "countries": sorted(countries),
        "cited_by_count": work.get("cited_by_count"),
        "openalex_year": work.get("publication_year"),
    }


def _batches(items: Sequence[str], size: int) -> Iterator[Sequence[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def fetch_works(
    dois: Iterable[str],
    client: HttpClient,
    api_key: str | None = None,
    mailto: str | None = None,
    batch_size: int = 50,
    delay: float = 0.2,
) -> dict[str, dict]:
    """OpenAlex works by DOI, ``batch_size`` DOIs per request (the filter accepts OR-ed values).

    DOIs containing "," or "|" would break the filter syntax; they are looked up one by one.
    """
    unique = sorted(set(dois))
    plain = [d for d in unique if "," not in d and "|" not in d]
    awkward = [d for d in unique if d not in set(plain)]
    common = {"select": OPENALEX_FIELDS}
    if api_key:
        common["api_key"] = api_key
    if mailto:
        common["mailto"] = mailto
    found: dict[str, dict] = {}
    for n, batch in enumerate(_batches(plain, batch_size), start=1):
        params = {**common, "filter": "doi:" + "|".join(batch), "per-page": batch_size}
        for work in client.get_json(OPENALEX_WORKS, params).get("results", []):
            if doi := clean_doi(work.get("doi")):
                found[doi] = work
        if n % 20 == 0:
            log.info(
                "OpenAlex: %d of %d DOIs looked up, %d found",
                n * batch_size,
                len(plain),
                len(found),
            )
        client.sleep(delay)
    for doi in awkward:
        try:
            work = client.get_json(f"{OPENALEX_WORKS}/doi:{doi}", common)
        except HttpError:  # not in OpenAlex
            continue
        found[doi] = work
        client.sleep(delay)
    return found


def enrich(
    papers: pd.DataFrame,
    client: HttpClient | None = None,
    api_key: str | None = None,
    mailto: str | None = None,
    batch_size: int = 50,
) -> pd.DataFrame:
    """One row per paper: the DOI that matched (if any) and the OpenAlex fields."""
    client = client or HttpClient()
    candidates = {
        pid: candidate_dois(pid, doi) for pid, doi in zip(papers["id"], papers["doi"], strict=True)
    }
    works = fetch_works(
        [d for c in candidates.values() for d in c], client, api_key, mailto, batch_size
    )
    rows = []
    for pid, dois in candidates.items():
        doi = next((d for d in dois if d in works), None)
        row = {"id": pid, "matched_doi": doi, "matched_by": None}
        if doi is not None:
            row["matched_by"] = "arxiv doi" if doi.startswith("10.48550/") else "journal doi"
            row.update(parse_work(works[doi]))
        rows.append(row)
    table = pd.DataFrame(rows)
    for col in ("institution_types", "countries"):
        if col not in table:
            table[col] = [[] for _ in range(len(table))]
        table[col] = table[col].map(lambda v: v if isinstance(v, list) else [])
    log.info("OpenAlex matched %d of %d papers", table["matched_doi"].notna().sum(), len(table))
    return table
