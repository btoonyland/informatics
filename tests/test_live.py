"""Tests against the real services, so changes on their side (formats, endpoints, limits) show up.

Skipped unless INFORMATICS_LIVE=1, because they need network access and are slower:

    INFORMATICS_LIVE=1 pytest -m live

Each test makes only a few requests. With INFORMATICS_RECORD_DIR set, the records fetched are
also saved there as Parquet files, ready to be reviewed and added to tests/fixtures/. The
OpenAlex tests also need OPENALEX_API_KEY, and NCBI_API_KEY is used when set.
"""

import os
from pathlib import Path

import pandas as pd
import pytest

from informatics.count import count_articles
from informatics.enrich import fetch_works
from informatics.harvest import harvest, list_sets
from informatics.http_client import HttpClient
from informatics.openalex import import_openalex
from informatics.pubmed import DEFAULT_QUERY, fetch_articles, search_year
from informatics.records import normalize_records
from informatics.storage import read_parquet, write_parquet

pytestmark = pytest.mark.live
if not os.environ.get("INFORMATICS_LIVE"):
    pytest.skip("live tests run with INFORMATICS_LIVE=1", allow_module_level=True)

RECORD_DIR = os.environ.get("INFORMATICS_RECORD_DIR")
OPENALEX_KEY = os.environ.get("OPENALEX_API_KEY")


def client() -> HttpClient:
    """Few, short retries: an unreachable service should fail the test, not stall it."""
    return HttpClient(timeout=60, max_retries=2, backoff=5)


def _record(name: str, df: pd.DataFrame) -> None:
    if RECORD_DIR:
        write_parquet(df, Path(RECORD_DIR) / name)


def test_arxiv_lists_sets():
    specs = {spec for spec, _ in list_sets(client=client())}
    assert {"cs", "q-bio"} <= specs


def test_arxiv_harvests_one_day(tmp_path):
    state = harvest("q-bio", tmp_path, date_from="2024-03-05", date_until="2024-03-05",
                    delay=1, max_pages=1, client=client())  # fmt: skip
    assert state["records"] > 0
    page = read_parquet(next((tmp_path / "pages").glob("page-*.parquet")))
    assert page["id"].notna().all() and page["abstract"].notna().mean() > 0.95
    _record("arxiv_q-bio_2024-03-05.parquet", normalize_records(page, "oai-pmh"))


def test_arxiv_search_counts():
    assert count_articles("cat:q-bio.QM", client=client()) > 5_000


def test_pubmed_search_and_fetch():
    key = os.environ.get("NCBI_API_KEY")
    total, pmids = search_year(client(), DEFAULT_QUERY, 2020, 20, key)
    assert total > 5_000 and len(pmids) == 20
    records = fetch_articles(client(), pmids, key)
    assert len(records) == 20 and all(r["created"] for r in records)
    _record("pubmed_2020.parquet", normalize_records(pd.DataFrame(records), "pubmed-sample"))


@pytest.mark.skipif(not OPENALEX_KEY, reason="needs OPENALEX_API_KEY")
def test_openalex_finds_an_arxiv_doi():
    works = fetch_works(["10.48550/arxiv.1706.03762"], client(), OPENALEX_KEY)
    assert "10.48550/arxiv.1706.03762" in works


@pytest.mark.skipif(not OPENALEX_KEY, reason="needs OPENALEX_API_KEY")
def test_openalex_import(tmp_path):
    path = tmp_path / "openalex.parquet"
    assert import_openalex(path, from_year=2024, to_year=2024, api_key=OPENALEX_KEY,
                           client=client(), max_records=25) == 25  # fmt: skip
    df = read_parquet(path)
    assert df["abstract"].notna().mean() > 0.5 and (df["year"] == 2024).all()
    _record("openalex_2024.parquet", df)
