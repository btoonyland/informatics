"""Import PubMed records: the clinical side of AI in medicine.

arXiv shows the computer-science side of AI in medicine; PubMed shows the clinical literature.
Both imports write ``data/raw/<set>.parquet`` in the same schema as the arXiv sets, so the same
filter, themes and concepts apply, and ``informatics analyze --set cs --set pubmed`` compares
them side by side. There are two ways in:

* ``import-pubmed`` (source ``pubmed-sample``) searches NCBI's E-utilities. ESearch returns at
  most 10,000 records per query, so the search runs one publication year at a time and keeps up
  to ``per_year`` records of each: a sample, not the whole literature. NCBI allows 3 requests
  per second without an API key and 10 with one (``--api-key`` or ``NCBI_API_KEY``).
* ``import-pubmed-files`` (source ``pubmed-baseline``) reads the annual baseline files
  (https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/, plus the daily update files if wanted) and
  keeps every record with an AI MeSH heading or an AI keyword in its title or abstract: the
  whole indexed literature for that definition. MeSH indexing lags by months, which is why
  keywords count too; otherwise the most recent year would look smaller than it is.

Dates: a paper's year should mean the same thing in every set. For arXiv it is the month of
first submission. For PubMed it is the earliest date the record has (received by the journal,
published online, or published in print), which is the closest equivalent.
"""

from __future__ import annotations

import gzip
import logging
import re
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

from informatics import keywords
from informatics.http_client import HttpClient
from informatics.imports import _BatchWriter
from informatics.normalize import clean_characters

log = logging.getLogger(__name__)

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
DEFAULT_QUERY = '"artificial intelligence"[MeSH Terms]'
SAMPLE_SOURCE, BASELINE_SOURCE = "pubmed-sample", "pubmed-baseline"
# MeSH descriptors for AI (edit with --mesh). "Artificial Intelligence" has the others below it,
# but files list the descriptors a record was indexed with, not their parents.
AI_MESH = (
    "Artificial Intelligence", "Machine Learning", "Deep Learning",
    "Supervised Machine Learning", "Unsupervised Machine Learning", "Neural Networks, Computer",
    "Natural Language Processing", "Support Vector Machine", "Expert Systems",
    "Generative Artificial Intelligence", "Large Language Models",
)  # fmt: skip
# History dates that mark when a paper became (or was about to become) public; "revised" and
# "medline" (indexing) are left out.
_HISTORY = {"received", "accepted", "epublish", "aheadofprint", "ppublish", "pubmed", "entrez"}
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


def _sort_key(date: str) -> tuple[int, int]:
    """ "2019" sorts after "2019-03": a year without a month is less precise, not earlier."""
    year, _, month = date.partition("-")
    return int(year), int(month) if month else 12


def earliest_date(article: ET.Element) -> str | None:
    """The earliest date in a PubmedArticle: received, published online or in print."""
    art = article.find("MedlineCitation/Article")
    candidates = []
    if art is not None:
        candidates += [parse_date(d) for d in art.iterfind("ArticleDate")]
        candidates.append(parse_date(art.find("Journal/JournalIssue/PubDate")))
    for node in article.iterfind("PubmedData/History/PubMedPubDate"):
        if node.get("PubStatus") in _HISTORY:
            candidates.append(parse_date(node))
    dates = [d for d in candidates if d]
    return min(dates, key=_sort_key) if dates else None


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
        "created": earliest_date(article),
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
    with _BatchWriter(Path(dest), SAMPLE_SOURCE) as writer:
        for year in years:
            total, pmids = search_year(client, query, year, per_year, api_key)
            log.info("%d: %d records match; importing %d", year, total, len(pmids))
            client.sleep(delay)
            for start in range(0, len(pmids), batch_size):
                writer.write(fetch_articles(client, pmids[start : start + batch_size], api_key))
                client.sleep(delay)
        return writer.close()


def iter_pubmed_file(path: str | Path) -> Iterator[tuple[str, ET.Element | str]]:
    """Stream a PubMed XML file (.xml or .xml.gz): ("article", element) for each record and
    ("delete", pmid) for each deleted citation, keeping memory flat however big the file."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as fh:
        events = ET.iterparse(fh, events=("start", "end"))
        _, root = next(events)
        for event, element in events:
            if event != "end":
                continue
            if element.tag == "PubmedArticle":
                yield "article", element
                root.clear()
            elif element.tag == "DeleteCitation":
                for pmid in element.iterfind("PMID"):
                    if pmid.text:
                        yield "delete", pmid.text.strip()
                root.clear()


def matches_ai(record: dict, mesh: Sequence[str] = AI_MESH, terms: str = "strict") -> bool:
    """True if the record has one of the ``mesh`` descriptors, or (``terms`` "strict" or
    "wide") an AI keyword in its title or abstract (keywords.AI, plus AI_CONTEXT if wide)."""
    if set(record.get("categories") or []) & set(mesh):
        return True
    if terms == "none":
        return False
    text = clean_characters(f"{record.get('title') or ''}. {record.get('abstract') or ''}").lower()
    if keywords.AI.hits(text):
        return True
    return terms == "wide" and bool(keywords.AI_CONTEXT.hits(text))


def import_pubmed_files(
    files: Iterable[str | Path],
    dest: str | Path,
    mesh: Sequence[str] = AI_MESH,
    terms: str = "strict",
    everything: bool = False,
    batch_size: int = 5_000,
) -> dict:
    """Import PubMed baseline and update files, keeping AI records (see :func:`matches_ai`)
    or, with ``everything``, all of them. Files are read newest first (by name), so a record
    revised in an update file replaces its baseline version and deleted citations are dropped.
    Returns counts: files, records read, kept, deleted."""
    if terms not in ("none", "strict", "wide"):
        raise ValueError(f"unknown keyword option {terms!r}; use none, strict or wide")
    paths = sorted((Path(f) for f in files), key=lambda p: p.name, reverse=True)
    if not paths:
        raise ValueError("no PubMed files given")
    deleted: set[str] = set()
    counts: Counter[str] = Counter(files=len(paths))
    rows: list[dict] = []
    with _BatchWriter(Path(dest), BASELINE_SOURCE) as writer:
        for path in paths:
            for kind, item in iter_pubmed_file(path):
                if kind == "delete":
                    deleted.add(f"pmid:{item}")
                    counts["deleted"] += 1
                    continue
                counts["read"] += 1
                record = parse_article(item)
                # A newer version of the record (kept already) or its deletion wins.
                if not record or record["id"] in deleted or record["id"] in writer.seen:
                    continue
                if everything or matches_ai(record, mesh, terms):
                    rows.append(record)
                    if len(rows) >= batch_size:
                        writer.write(rows)
                        rows = []
            log.info("%s: %d records read, %d kept so far", path.name, counts["read"], writer.count)
        writer.write(rows)
        counts["kept"] = writer.close()
    return dict(counts)
