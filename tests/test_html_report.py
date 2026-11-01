import json
import re
import shutil
import subprocess
from importlib import resources

import pandas as pd
import pytest

from informatics.analysis import AnalysisConfig, run_analysis
from informatics.filtering import FilterConfig, run_filter
from informatics.html_report import paper_url
from informatics.records import RAW_SCHEMA
from informatics.storage import write_parquet

CONFIG = AnalysisConfig(min_df=3, cluster_min_df=3, k=4, trend_min_papers=5,
                        trend_min_year_papers=1)  # fmt: skip


def _embedded(page: str) -> dict:
    match = re.search(
        r'<script type="application/json" id="report-data">(.*?)</script>', page, re.S
    )
    assert "<" not in match.group(1)
    return json.loads(match.group(1))


def _report(layout, corpus=None):
    if corpus is not None:
        write_parquet(corpus, layout.raw("cs"), schema=RAW_SCHEMA)
    run_filter(layout.raw("cs"), "cs", layout.candidates("cs"), layout.background("cs"),
               layout.filter_summary("cs"), FilterConfig(background_size=100))  # fmt: skip
    out = run_analysis(["cs"], layout, config=CONFIG)
    return out, (out / "report.html").read_text(encoding="utf-8")


def test_report_page_embeds_every_chart(layout, corpus):
    out, page = _report(layout)
    assert page.startswith("<!doctype html>") and "{{" not in page and "/*JS*/" not in page
    data = _embedded(page)
    papers = pd.read_csv(out / "papers.csv")
    assert data["kpis"]["papers"] == len(papers) == data["papersTotal"] == len(data["papers"])
    assert sum(data["years"]["selected"]) == len(papers)
    assert {t["tag"] for t in data["themes"]} == {"bias", "ethical", "privacy", "regulation"}
    assert all(len(t["years"]) == len(t["low"]) == len(t["high"]) for t in data["themes"])
    assert len(data["trends"]) == 4 and data["clusters"][0]["cluster"] == 0
    assert len(data["map"]["x"]) == len(papers)  # small sets are mapped in full
    assert set(data["map"]["cluster"]) == {0, 1, 2, 3}
    first = dict(zip(data["paperColumns"], data["papers"][0], strict=True))
    assert first["url"] == f"https://arxiv.org/abs/{first['id']}"
    assert (out / "cluster_map.csv").exists()


def test_titles_cannot_break_out_of_the_page(layout, corpus):
    corpus = corpus.drop(columns=["kind", "topic"]).copy()
    corpus.loc[0, "title"] = "</script><script>alert(1)</script> Deep learning for patients"
    _, page = _report(layout, corpus)
    assert "<script>alert(1)" not in page
    assert any("alert(1)" in row[1] for row in _embedded(page)["papers"])


def test_no_html_option(layout):
    run_filter(layout.raw("cs"), "cs", layout.candidates("cs"), layout.background("cs"),
               layout.filter_summary("cs"), FilterConfig())  # fmt: skip
    config = AnalysisConfig(min_df=3, cluster_min_df=3, k=4, html=False)
    out = run_analysis(["cs"], layout, config=config)
    assert not (out / "report.html").exists() and not (out / "cluster_map.csv").exists()


@pytest.mark.parametrize(
    ("paper_id", "doi", "url"),
    [
        ("2101.00001", None, "https://arxiv.org/abs/2101.00001"),
        ("hep-th/9901001", None, "https://arxiv.org/abs/hep-th/9901001"),
        ("pmid:31000001", "10.1/x", "https://pubmed.ncbi.nlm.nih.gov/31000001/"),
        ("openalex:W123", None, "https://openalex.org/W123"),
        ("other-1", "10.1000/a b", "https://doi.org/10.1000/a%20b"),
        ("other-2", None, None),
    ],
)
def test_paper_url(paper_id, doi, url):
    assert paper_url(paper_id, doi) == url


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")
def test_report_script_is_valid_javascript(tmp_path):
    script = resources.files("informatics").joinpath("report_assets", "report.js").read_text()
    path = tmp_path / "report.js"
    path.write_text(script, encoding="utf-8")
    result = subprocess.run(["node", "--check", str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
