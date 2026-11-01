"""Import a PubMed comparison corpus through NCBI's E-utilities.

arXiv shows the computer-science side of AI in medicine; PubMed shows the clinical literature.
``informatics import-pubmed --set pubmed`` writes ``data/raw/pubmed.parquet`` in the same schema
as the arXiv sets, so the same filter, themes and concepts apply, and
``informatics analyze --set cs --set pubmed`` compares them side by side.

ESearch returns at most 10,000 records per query, so the search runs one publication year at a
time and keeps up to ``per_year`` records of each (the first ones in PubMed's default order: a
sample, not the whole literature). NCBI allows 3 requests per second without an API key and 10
with one (``--api-key`` or ``NCBI_API_KEY``).
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from pathlib import Path

from informatics.http_client import HttpClient
from informatics.imports import _BatchWriter

log = logging.getLogger(__name__)

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
DEFAULT_QUERY = '"artificial intelligence"[MeSH Terms]'
MAX_PER_QUERY = 9_999
_MONTH_NAMES = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
_MONTHS = {name: number for number, name in enumerate(_MONTH_NAMES, start=1)}


def _text(node: ET.Element | None) -> str | None:
    if node is None:
        return None
    text = " ".join("".join(node.itertext()).split())  # inline markup such as <i> is flattened
    return text or None


def _month(value: str | None) -> int | None:
    if not value:
        return None
    value = value.strip().lower()
    if value.isdigit():
        return int(value) if 1 <= int(value) <= 12 else None
    return _MONTHS.get(value[:3])


def parse_date(node: ET.Element | None) -> str | None:
    """ "YYYY-MM" (or "YYYY") from an ArticleDate or PubDate element, including MedlineDate."""
    if node is None:
        return None
    year, month = node.findtext("Year"), node.findtext("Month")
    if not year:
        match = re.match(r"\s*(\d{4})(?:\s+([A-Za-z]{3}))?", node.findtext("MedlineDate") or "")
        if not match:
            return None
        year, month = match.groups()
    number = _month(month)
    return f"{year}-{number:02d}" if number else year


def parse_article(article: ET.Element) -> dict:
    citation = article.find("MedlineCitation")
    art = citation.find("Article") if citation is not None else None
    if citation is None or art is None:
        return {}
    parts = []
    for node in art.iterfind("Abstract/AbstractText"):
        text, label = _text(node), node.get("Label")
        if text:
            parts.append(f"{label.capitalize()}: {text}" if label else text)
    authors = []
    for author in art.iterfind("AuthorList/Author"):
        name = " ".join(p for p in (author.findtext("ForeName"), author.findtext("LastName")) if p)
        name = name or author.findtext("CollectiveName")
        if name:
            authors.append(name)
    ids = [
        e
        for e in article.iterfind("PubmedData/ArticleIdList/ArticleId")
        if e.get("IdType") == "doi"
    ]
    locations = [e for e in art.iterfind("ELocationID") if e.get("EIdType") == "doi"]
    doi = next((e.text for e in ids + locations if e.text), None)
    return {
        "id": f"pmid:{citation.findtext('PMID')}",
        "title": _text(art.find("ArticleTitle")),
        "abstract": " ".join(parts) or None,
        "authors": authors,
        "categories": [  # MeSH descriptors play the role of arXiv categories
            d.text
            for d in citation.iterfind("MeshHeadingList/MeshHeading/DescriptorName")
            if d.text
        ],
        "created": parse_date(art.find("ArticleDate"))
        or parse_date(art.find("Journal/JournalIssue/PubDate")),
        "doi": doi,
        "journal_ref": art.findtext("Journal/Title"),
    }


def _params(api_key: str | None, **params: object) -> dict:
    params = {"db": "pubmed", "tool": "informatics", **params}
    if api_key:
        params["api_key"] = api_key
    return params


def search_year(
    client: HttpClient, query: str, year: int, limit: int, api_key: str | None = None
) -> tuple[int, list[str]]:
    """(number of matching records, the first ``limit`` PMIDs) for one publication year."""
    params = _params(
        api_key, term=query, datetype="pdat", mindate=str(year), maxdate=str(year),
        retmax=min(limit, MAX_PER_QUERY),
    )  # fmt: skip
    root = client.get_xml(f"{EUTILS}/esearch.fcgi", params)
    return int(root.findtext("Count") or 0), [e.text for e in root.iterfind("IdList/Id") if e.text]


def fetch_articles(client: HttpClient, pmids: list[str], api_key: str | None = None) -> list[dict]:
    root = client.get_xml(
        f"{EUTILS}/efetch.fcgi", _params(api_key, id=",".join(pmids), retmode="xml")
    )
    return [r for r in map(parse_article, root.iterfind("PubmedArticle")) if r]


def import_pubmed(
    dest: str | Path,
    years: Iterable[int],
    query: str = DEFAULT_QUERY,
    per_year: int = 2000,
    api_key: str | None = None,
    client: HttpClient | None = None,
    batch_size: int = 200,
) -> int:
    client = client or HttpClient()
    delay = 0.11 if api_key else 0.34  # stay under NCBI's 10 / 3 requests per second
    writer = _BatchWriter(Path(dest), "pubmed")
    for year in years:
        total, pmids = search_year(client, query, year, per_year, api_key)
        log.info("%d: %d records match; importing %d", year, total, len(pmids))
        client.sleep(delay)
        for start in range(0, len(pmids), batch_size):
            writer.write(fetch_articles(client, pmids[start : start + batch_size], api_key))
            client.sleep(delay)
    return writer.close()
