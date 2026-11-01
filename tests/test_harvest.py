import json
import xml.etree.ElementTree as ET

import pytest
import requests
from helpers import FakeResponse, client_for, fixture_bytes, oai_error, ok

from informatics.harvest import HarvestError, consolidate, harvest, list_sets, parse_page
from informatics.http_client import HttpError
from informatics.storage import read_parquet


def test_parse_page_extracts_full_records():
    records, token, info = parse_page(ET.fromstring(fixture_bytes("oai_page1.xml")))
    assert token == "token-page-2"
    assert info["completeListSize"] == "4"
    assert [r["id"] for r in records] == ["0901.0001", "1501.00010"]  # deleted record skipped
    first = records[0]
    assert first["authors"] == ["Ana Kovač", "Jörg Müller", "John Smith Jr"]  # forenames kept
    assert first["title"] == "Neural networks for 3D medical scan visualization"
    assert (
        first["abstract"].startswith("We train neural networks") and "\n" not in first["abstract"]
    )
    assert first["categories"] == ["cs.AI", "cs.GR"]
    assert first["datestamp"] == "2009-12-01"
    assert records[1]["doi"] == "10.0000/example.1"


def test_last_page_has_no_token():
    _, token, _ = parse_page(ET.fromstring(fixture_bytes("oai_page2.xml")))
    assert token is None


def test_harvest_resumes_after_failure_and_consolidates(tmp_path):
    out = tmp_path / "cs_harvest"
    error = requests.ConnectionError("connection reset")
    # First run: page 1 arrives, then the connection keeps failing.
    first = client_for([ok("oai_page1.xml"), error, error], max_retries=1)
    with pytest.raises(HttpError, match="after 1 retries"):
        harvest("cs", out, client=first, delay=0)
    state = json.loads((out / "state.json").read_text())
    assert state["token"] == "token-page-2" and state["next_page"] == 2 and not state["complete"]

    # Second run continues from the saved token instead of starting over.
    second = client_for([FakeResponse(503, headers={"Retry-After": "1"}), ok("oai_page2.xml")])
    state = harvest("cs", out, client=second, delay=0)
    assert state["complete"] and state["records"] == 4
    assert second.session.calls[0] == {"verb": "ListRecords", "resumptionToken": "token-page-2"}

    df = consolidate(out, tmp_path / "cs.parquet")
    assert sorted(df["id"]) == ["0901.0001", "1501.00010", "cs/0601001"]
    revised = df.set_index("id").loc["0901.0001"]
    assert revised["datestamp"] == "2011-03-04"  # the newer copy wins
    assert revised["title"].endswith("(revised)")
    assert (revised["year"], revised["month"]) == (2009, 1)
    assert read_parquet(tmp_path / "cs.parquet")["source"].unique().tolist() == ["oai-pmh"]


def test_first_request_uses_set_and_dates(tmp_path):
    client = client_for([ok("oai_page2.xml")])
    harvest("q-bio", tmp_path / "h", client=client, date_from="2020-01-01", delay=0)
    assert client.session.calls[0] == {
        "verb": "ListRecords",
        "metadataPrefix": "arXiv",
        "set": "q-bio",
        "from": "2020-01-01",
    }


def test_max_pages_pauses_and_a_rerun_continues(tmp_path):
    out = tmp_path / "h"
    state = harvest("cs", out, client=client_for([ok("oai_page1.xml")]), max_pages=1, delay=0)
    assert not state["complete"]
    state = harvest("cs", out, client=client_for([ok("oai_page2.xml")]), delay=0)
    assert state["complete"]
    assert sorted(p.name for p in (out / "pages").iterdir()) == [
        "page-000001.parquet",
        "page-000002.parquet",
    ]


def test_changed_settings_need_restart(tmp_path):
    out = tmp_path / "h"
    harvest("cs", out, client=client_for([ok("oai_page1.xml")]), max_pages=1, delay=0)
    with pytest.raises(HarvestError, match="--restart"):
        harvest("cs", out, client=client_for([]), date_from="2020-01-01", delay=0)
    state = harvest("cs", out, client=client_for([ok("oai_page2.xml")]), restart=True, delay=0)
    assert state["complete"] and state["records"] == 2  # started over: only page 2's records


def test_expired_token_gives_clear_error(tmp_path):
    out = tmp_path / "h"
    harvest("cs", out, client=client_for([ok("oai_page1.xml")]), max_pages=1, delay=0)
    with pytest.raises(HarvestError, match="expired"):
        harvest("cs", out, client=client_for([ok(oai_error("badResumptionToken"))]), delay=0)


def test_no_records_is_an_empty_complete_harvest(tmp_path):
    state = harvest("cs", tmp_path / "h", client=client_for([ok(oai_error("noRecordsMatch"))]))
    assert state["complete"] and state["records"] == 0


def test_list_sets():
    xml = (
        b'<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"><ListSets>'
        b"<set><setSpec>cs</setSpec><setName>Computer Science</setName></set>"
        b"<set><setSpec>q-bio</setSpec><setName>Quantitative Biology</setName></set>"
        b"</ListSets></OAI-PMH>"
    )
    assert list_sets(client=client_for([ok(xml)])) == [
        ("cs", "Computer Science"),
        ("q-bio", "Quantitative Biology"),
    ]
