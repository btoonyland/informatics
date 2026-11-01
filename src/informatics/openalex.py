"""Import a corpus from OpenAlex: journals, conferences and preprint servers in one place.

``informatics import-openalex --set openalex`` writes ``data/raw/<set>.parquet`` from the works
that match an OpenAlex filter. The default keeps works whose topics include the artificial
intelligence subfield (1702) and the medicine field (27), which spans the clinical journals that
PubMed covers and the computer-science venues it does not. Pass ``--filter`` to define the
population differently (any filter of https://api.openalex.org/works, e.g.
``primary_topic.field.id:27,type:article``).

Results are paged with OpenAlex's cursor; abstracts come as an inverted index (word ->
positions) and are rebuilt in order. Dates are publication dates. OpenAlex requires a free API
key (``--api-key`` or ``OPENALEX_API_KEY``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from informatics.enrich import OPENALEX_WORKS, clean_doi
from informatics.http_client import HttpClient
from informatics.imports import _BatchWriter

log = logging.getLogger(__name__)

SOURCE = "openalex"
DEFAULT_FILTER = "topics.subfield.id:1702,topics.field.id:27"
WORK_FIELDS = (
    "id,doi,title,publication_date,publication_year,abstract_inverted_index,authorships,"
    "primary_location,topics,type"
)


def abstract_from_index(index: dict[str, list[int]] | None) -> str | None:
    """Rebuild an abstract from OpenAlex's ``abstract_inverted_index``."""
    if not index:
        return None
    placed = sorted((position, word) for word, positions in index.items() for position in positions)
    return " ".join(word for _, word in placed) or None


def work_record(work: dict[str, Any]) -> dict:
    """One OpenAlex work -> a raw record (ID ``openalex:W...``)."""
    work_id = (work.get("id") or "").rsplit("/", 1)[-1]
    location = work.get("primary_location") or {}
    source = location.get("source") or {}
    return {
        "id": f"openalex:{work_id}" if work_id else None,
        "title": work.get("title"),
        "abstract": abstract_from_index(work.get("abstract_inverted_index")),
        "authors": [
            (a.get("author") or {}).get("display_name")
            for a in work.get("authorships") or []
            if (a.get("author") or {}).get("display_name")
        ],
        # Topics play the role of arXiv categories.
        "categories": [
            t["display_name"] for t in work.get("topics") or [] if t.get("display_name")
        ],
        "created": work.get("publication_date") or str(work.get("publication_year") or "") or None,
        "doi": clean_doi(work.get("doi")),
        "journal_ref": source.get("display_name"),
    }


def import_openalex(
    dest: str | Path,
    filter_: str = DEFAULT_FILTER,
    from_year: int | None = None,
    to_year: int | None = None,
    api_key: str | None = None,
    mailto: str | None = None,
    client: HttpClient | None = None,
    per_page: int = 200,
    max_records: int | None = None,
    delay: float = 0.1,
) -> int:
    """Page through every work matching ``filter_`` (and the publication years) and write
    them to ``dest``; returns the number of records written."""
    client = client or HttpClient()
    filters = [f for f in filter_.split(",") if f]
    if from_year:
        filters.append(f"from_publication_date:{from_year}-01-01")
    if to_year:
        filters.append(f"to_publication_date:{to_year}-12-31")
    params: dict[str, Any] = {
        "filter": ",".join(filters),
        "per-page": per_page,
        "select": WORK_FIELDS,
    }
    if api_key:
        params["api_key"] = api_key
    if mailto:
        params["mailto"] = mailto
    writer = _BatchWriter(Path(dest), SOURCE)
    cursor: str | None = "*"
    seen = 0
    while cursor:
        page = client.get_json(OPENALEX_WORKS, {**params, "cursor": cursor})
        results = page.get("results") or []
        if not results:
            break
        if max_records is not None:
            results = results[: max(0, max_records - seen)]
        writer.write([r for r in map(work_record, results) if r["id"]])
        seen += len(results)
        total = (page.get("meta") or {}).get("count")
        log.info("OpenAlex: %d of %s works", seen, total if total is not None else "?")
        if max_records is not None and seen >= max_records:
            break
        cursor = (page.get("meta") or {}).get("next_cursor")
        client.sleep(delay)
    return writer.close()
