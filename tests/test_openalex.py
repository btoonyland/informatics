import json

from helpers import client_for, ok

from informatics.openalex import abstract_from_index, import_openalex, work_record
from informatics.storage import read_parquet

WORK = {
    "id": "https://openalex.org/W100",
    "doi": "https://doi.org/10.1000/ABC",
    "title": "Deep learning for retinal screening",
    "publication_date": "2021-05-17",
    "publication_year": 2021,
    "abstract_inverted_index": {"We": [0], "screen": [1, 4], "retinas.": [2], "and": [3]},
    "authorships": [{"author": {"display_name": "Ana Kovač"}}, {"author": {}}],
    "primary_location": {"source": {"display_name": "Synthetic Journal"}},
    "topics": [{"display_name": "Retinal Imaging"}, {"display_name": "AI in Healthcare"}],
    "type": "article",
}


def test_abstract_is_rebuilt_in_order():
    assert abstract_from_index(WORK["abstract_inverted_index"]) == "We screen retinas. and screen"
    assert abstract_from_index(None) is None and abstract_from_index({}) is None


def test_work_record():
    record = work_record(WORK)
    assert record["id"] == "openalex:W100" and record["doi"] == "10.1000/abc"
    assert record["authors"] == ["Ana Kovač"] and record["created"] == "2021-05-17"
    assert record["categories"] == ["Retinal Imaging", "AI in Healthcare"]
    assert record["journal_ref"] == "Synthetic Journal"
    assert (
        work_record({"id": "https://openalex.org/W1", "publication_year": 2019})["created"]
        == "2019"
    )


def _page(works, next_cursor, count=3):
    return ok(json.dumps({"meta": {"count": count, "next_cursor": next_cursor},
                          "results": works}).encode())  # fmt: skip


def test_import_pages_through_the_cursor(tmp_path):
    second = {**WORK, "id": "https://openalex.org/W200", "publication_date": "2022-01-02"}
    third = {**WORK, "id": "https://openalex.org/W300"}
    client = client_for([_page([WORK, second], "abc"), _page([third], "def"), _page([], None)])
    n = import_openalex(tmp_path / "oa.parquet", from_year=2020, to_year=2024, api_key="K",
                        client=client)  # fmt: skip
    assert n == 3
    first, following = client.session.calls[0], client.session.calls[1]
    assert first["cursor"] == "*" and following["cursor"] == "abc" and first["api_key"] == "K"
    assert first["filter"] == (
        "topics.subfield.id:1702,topics.field.id:27,"
        "from_publication_date:2020-01-01,to_publication_date:2024-12-31"
    )
    df = read_parquet(tmp_path / "oa.parquet").set_index("id")
    assert df.loc["openalex:W200", "year"] == 2022 and df.loc["openalex:W200", "month"] == 1
    assert (df["source"] == "openalex").all()


def test_import_stops_at_max_records(tmp_path):
    client = client_for([_page([WORK, {**WORK, "id": "https://openalex.org/W2"}], "abc")])
    assert import_openalex(tmp_path / "oa.parquet", client=client, max_records=1) == 1
    assert len(client.session.calls) == 1


def test_a_failed_import_keeps_the_previous_file(tmp_path):
    import pandas as pd
    import pytest
    import requests

    from informatics.http_client import HttpError
    from informatics.storage import write_parquet

    dest = tmp_path / "oa.parquet"
    write_parquet(pd.DataFrame({"id": ["openalex:W1"]}), dest)
    before = dest.read_bytes()
    down = requests.ConnectionError("network is unreachable")
    client = client_for([_page([WORK], "abc"), down, down], max_retries=1)
    with pytest.raises(HttpError):
        import_openalex(dest, client=client)
    assert dest.read_bytes() == before  # untouched: written only when the import completes
    assert [p.name for p in tmp_path.iterdir()] == ["oa.parquet"]  # no partial temporary file
