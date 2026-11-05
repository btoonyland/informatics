from informatics.filtering import FilterConfig, run_filter
from informatics.sources import describe
from informatics.storage import write_json


def test_describe_states_population_and_dates():
    summary = {
        "sources": {"pubmed-sample": 1200},
        "import": {"source": "pubmed-sample", "query": "ai[mh]", "per_year": 100,
                   "years": "2015-2026"},
    }  # fmt: skip
    (line,) = describe("pubmed", summary)
    assert line.startswith("pubmed: 1,200 papers, a sample of PubMed")
    assert "query ai[mh]; at most 100 records per year; publication years 2015-2026" in line
    assert line.endswith(
        "Dates: earliest date in the record: received, published online or in print."
    )
    assert "population is unknown" in describe("x", None)[0]
    windowed = {**summary, "config": {"years": [2021, 2025]}}
    assert "; only papers from 2021-2025)" in describe("pubmed", windowed)[0]


def test_filter_summary_records_the_import(layout):
    write_json({"source": "oai-pmh", "set_spec": "cs"}, layout.raw_info("cs"))
    summary = run_filter(layout.raw("cs"), "cs", layout.candidates("cs"), layout.background("cs"),
                         layout.filter_summary("cs"), FilterConfig())  # fmt: skip
    assert summary["import"]["set_spec"] == "cs"
    assert summary["sources"] == {"synthetic": summary["counts"]["scanned"]}
    assert "Dates: the test corpus's own dates." in describe("cs", summary)[0]
