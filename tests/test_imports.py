import json

import pandas as pd
from helpers import client_for, ok

from informatics.count import count_articles, default_query
from informatics.imports import import_kaggle, import_legacy_csv, in_set, kaggle_record
from informatics.storage import read_parquet

KAGGLE_RECORD = {
    "id": "0704.0001",
    "title": "Deep learning for\n  hospital triage",
    "abstract": "Patients and machine learning.",
    "authors": "A. Kovač, J. Smith Jr",
    "categories": "cs.LG q-bio.QM",
    "versions": [
        {"version": "v1", "created": "Mon, 2 Apr 2007 19:18:42 GMT"},
        {"version": "v2", "created": "Tue, 3 Jul 2007 10:00:00 GMT"},
    ],
    "update_date": "2008-11-24",
    "authors_parsed": [["Kovač", "Ana", ""], ["Smith", "John", "Jr"]],
    "journal-ref": "J. Test 1 (2008)",
}


def test_in_set_matches_archive_prefix_and_cross_lists():
    assert in_set(["stat.ML", "cs.LG"], "cs")
    assert in_set(["q-bio.NC"], "q-bio")
    assert not in_set(["physics.bio-ph"], "q-bio")
    assert not in_set(["csx.AB"], "cs")


def test_kaggle_record_mapping():
    rec = kaggle_record(KAGGLE_RECORD)
    assert rec["authors"] == ["Ana Kovač", "John Smith Jr"]
    assert rec["created"] == "2007-04-02" and rec["updated"] == "2007-07-03"
    assert rec["journal_ref"] == "J. Test 1 (2008)"
    assert rec["categories"] == ["cs.LG", "q-bio.QM"]


def test_import_kaggle_streams_and_filters(tmp_path):
    other = {"id": "0704.0002", "title": "Graphs", "abstract": "x", "categories": "math.CO"}
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in (KAGGLE_RECORD, other)) + "\n",
        encoding="utf-8",
    )
    assert import_kaggle(snapshot, "cs", tmp_path / "cs.parquet", batch=1) == 1
    df = read_parquet(tmp_path / "cs.parquet")
    assert df["id"].tolist() == ["0704.0001"]
    assert df.loc[0, "title"] == "Deep learning for hospital triage"
    assert (df.loc[0, "year"], df.loc[0, "month"], df.loc[0, "source"]) == (2007, 4, "kaggle")


def test_import_kaggle_with_no_matches_writes_empty_file(tmp_path):
    snapshot = tmp_path / "s.json"
    snapshot.write_text(json.dumps({"id": "1", "categories": "math.CO"}) + "\n")
    assert import_kaggle(snapshot, "cs", tmp_path / "cs.parquet") == 0
    assert read_parquet(tmp_path / "cs.parquet").empty


def test_import_legacy_csv_repairs_old_harvester_output(tmp_path):
    """A CSV as written by the old arXivAPI.py: Latin-1-decoded UTF-8, list reprs, surnames."""
    legacy = pd.DataFrame(
        {
            "id": ["0806.2925", "0806.2925", "1004.2280"],
            "title": ["Neural networks in 3D medical scan visualization"] * 2 + ["XOR"],
            "summary": ["For medical volume visualization ...", "dup", "New to neuroscience"],
            "published": ["2009-06-12", "2009-06-12", "2010-09-23"],
            "update": ["2009-12-01", "2009-12-01", "2010-09-24"],
            "authors": [
                str(["Zukić".encode().decode("latin-1"), "Elsner"]),
                "['x']",
                "['Burger']",
            ],
            "doi": ["", "", ""],
            "categories": ["cs.AI cs.GR", "cs.AI", "cs.NE q-bio.NC"],
        }
    )
    path = tmp_path / "All_CS.csv"
    legacy.to_csv(path, index=False)
    assert import_legacy_csv(path, tmp_path / "cs.parquet", chunksize=2) == 2
    df = read_parquet(tmp_path / "cs.parquet").set_index("id")
    assert df.index.tolist() == ["0806.2925", "1004.2280"]  # string IDs, duplicate dropped
    assert df.loc["0806.2925", "authors"] == ["Zukić", "Elsner"]
    assert df.loc["0806.2925", "abstract"] == "For medical volume visualization ..."
    assert df.loc["1004.2280", "categories"] == ["cs.NE", "q-bio.NC"]
    assert pd.isna(df.loc["1004.2280", "doi"])


def test_count_articles():
    xml = (
        b'<feed xmlns="http://www.w3.org/2005/Atom" '
        b'xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">'
        b"<opensearch:totalResults>12345</opensearch:totalResults></feed>"
    )
    client = client_for([ok(b"not xml"), ok(xml)])  # a bad response is retried
    assert count_articles(default_query("q-bio"), client=client) == 12345
    assert client.session.calls[-1]["search_query"] == "cat:q-bio*"
