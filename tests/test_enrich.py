import json

import pandas as pd
import pytest
from helpers import FakeResponse, client_for, ok

from informatics.enrich import (
    candidate_dois,
    clean_doi,
    enrich,
    fetch_works,
    parse_work,
    publication_status,
)

WORKS = {
    "results": [
        {
            "id": "https://openalex.org/W1",
            "doi": "https://doi.org/10.1000/journal.1",
            "publication_year": 2021,
            "cited_by_count": 12,
            "primary_location": {
                "source": {"display_name": "Journal of Testing", "type": "journal"}
            },
            "primary_topic": {"field": {"display_name": "Medicine"}},
            "authorships": [
                {
                    "institutions": [
                        {"type": "healthcare", "country_code": "SI"},
                        {"type": "education", "country_code": "DE"},
                    ],
                    "countries": ["SI", "DE"],
                }
            ],
        },
        {
            "id": "https://openalex.org/W2",
            "doi": "https://doi.org/10.48550/arXiv.2101.00002",
            "cited_by_count": 0,
            "primary_location": {"source": {"display_name": "arXiv", "type": "repository"}},
            "authorships": [{"institutions": [{"type": "company", "country_code": "US"}]}],
        },
    ]
}


@pytest.mark.parametrize(
    ("journal_ref", "comments", "expected"),
    [
        ("J. Test 1 (2020) 1-10", None, "journal reference"),
        (None, "12 pages; accepted at MICCAI 2023", "accepted (comments)"),
        (float("nan"), "To appear in Radiology", "accepted (comments)"),
        (None, "12 pages, 3 figures", "no evidence"),
        ("", None, "no evidence"),
    ],
)
def test_publication_status(journal_ref, comments, expected):
    assert publication_status(journal_ref, comments) == expected


def test_clean_doi_and_candidates():
    assert clean_doi("https://doi.org/10.1000/ABC.1") == "10.1000/abc.1"
    assert clean_doi("doi:10.1000/x 10.2000/y") == "10.1000/x"
    assert clean_doi("10.1/too-short-prefix") is None
    assert clean_doi(None) is None and clean_doi("not a doi") is None
    assert candidate_dois("2101.00001", "10.1000/J.1") == [
        "10.1000/j.1",
        "10.48550/arxiv.2101.00001",
    ]
    assert candidate_dois("pmid:123", None) == []


def test_parse_work():
    row = parse_work(WORKS["results"][0])
    assert row["venue_type"] == "journal" and row["field"] == "Medicine"
    assert row["institution_types"] == ["education", "healthcare"]
    assert row["countries"] == ["DE", "SI"] and row["cited_by_count"] == 12


def test_fetch_works_batches_and_sends_key():
    client = client_for([ok(json.dumps(WORKS).encode())])
    found = fetch_works(["10.1000/journal.1", "10.48550/arxiv.2101.00002"], client, api_key="KEY")
    assert set(found) == {"10.1000/journal.1", "10.48550/arxiv.2101.00002"}
    params = client.session.calls[0]
    assert params["filter"] == "doi:10.1000/journal.1|10.48550/arxiv.2101.00002"
    assert params["api_key"] == "KEY" and "authorships" in params["select"]
    assert "mailto" not in params  # no e-mail is sent unless one is given


def test_enrich_prefers_the_journal_doi_and_falls_back_to_arxiv():
    papers = pd.DataFrame(
        {
            "id": ["2101.00001", "2101.00002", "pmid:9", "2101.00004"],
            "doi": ["10.1000/JOURNAL.1", None, None, "10.1000/a,b"],
        }
    )
    # One batched request, then a single lookup for the DOI with a comma (not found).
    client = client_for([ok(json.dumps(WORKS).encode()), FakeResponse(404, b"not found")])
    table = enrich(papers, client=client).set_index("id")
    assert table.loc["2101.00001", "matched_by"] == "journal doi"
    assert table.loc["2101.00002", "matched_by"] == "arxiv doi"
    assert table.loc["2101.00002", "institution_types"] == ["company"]
    assert pd.isna(table.loc["pmid:9", "matched_doi"]) and table.loc["pmid:9", "countries"] == []
    assert pd.isna(table.loc["2101.00004", "matched_doi"])
    assert client.session.calls[1] == {"select": client.session.calls[0]["select"]}


def test_analysis_reports_openalex_data(layout):
    from informatics.analysis import AnalysisConfig, run_analysis
    from informatics.filtering import FilterConfig, run_filter
    from informatics.storage import read_parquet, write_parquet

    run_filter(layout.raw("cs"), "cs", layout.selected("cs"), layout.background("cs"),
               layout.filter_summary("cs"), FilterConfig())  # fmt: skip
    ids = read_parquet(layout.selected("cs"))["id"].tolist()
    n = len(ids)
    enriched = pd.DataFrame(
        {
            "id": ids,
            "matched_doi": [f"10.48550/arxiv.{i}" if k % 2 else None for k, i in enumerate(ids)],
            "matched_by": ["arxiv doi" if k % 2 else None for k in range(n)],
            "openalex_id": [f"W{k}" if k % 2 else None for k in range(n)],
            "venue_type": ["journal" if k % 4 == 1 else "repository" for k in range(n)],
            "institution_types": [
                ["healthcare", "education"] if k % 3 == 0 else ["company"] for k in range(n)
            ],
            "countries": [["SI", "DE"] if k % 3 == 0 else ["US"] for k in range(n)],
            "cited_by_count": list(range(n)),
        }
    )
    write_parquet(enriched, layout.openalex("cs"))
    out = run_analysis(["cs"], layout, config=AnalysisConfig(min_df=3, cluster_min_df=3, k=4))
    publication = pd.read_csv(out / "publication_by_year.csv")
    assert {"share_in_openalex", "share_journal_or_conference", "share_healthcare_author",
            "share_company_author", "median_citations"} <= set(publication.columns)  # fmt: skip
    countries = pd.read_csv(out / "countries.csv")
    assert set(countries["country"]) == {"SI", "DE", "US"}
    assert "### From OpenAlex" in (out / "report.md").read_text()
