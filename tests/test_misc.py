"""Light tests for the stand-alone scripts in misc/ (loaded by path; misc/ is not a package)."""

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

MISC = Path(__file__).resolve().parents[1] / "misc"


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, MISC / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_greentech_keywords_match_whole_words():
    news = load("greentech_news")
    pattern = news.build_pattern(news.DEFAULT_KEYWORDS)
    items = [
        {"title": "Open the window", "description": "", "link": "a", "pubDate": ""},
        {
            "title": "The solar system and dark energy",
            "description": "",
            "link": "b",
            "pubDate": "",
        },
        {
            "title": "Cheaper wind turbines",
            "description": "<p>Renewable energy</p>",
            "link": "c",
            "pubDate": "",
        },
        {"title": "Climate models", "description": "", "link": "d", "pubDate": ""},
    ]
    matches = news.matching_items(items, pattern, limit=5)
    assert [m["link"] for m in matches] == ["c", "d"]
    assert matches[0]["matched"] == ["renewable energy", "wind turbines"]
    assert len(news.matching_items(items, pattern, limit=1)) == 1


def test_explore_dataset(tmp_path, capsys):
    explore = load("explore_dataset")
    src = tmp_path / "data.csv"
    pd.DataFrame({"type": ["CONSUMER_GOOD", "OTHER", "CONSUMER_GOOD"], "x": [1, 2, 3]}).to_csv(
        src, index=False
    )
    out = tmp_path / "out.csv"
    assert explore.main([str(src), "--value", "CONSUMER_GOOD", "--out", str(out)]) == 0
    assert len(pd.read_csv(out)) == 2
    assert explore.main([str(src), "--column", "missing"]) == 1


def test_pdf_extract_writes_one_record_per_pdf(tmp_path):
    pypdf = pytest.importorskip("pypdf")
    pytest.importorskip("pdfminer")
    pdfs = tmp_path / "pdfs"
    pdfs.mkdir()
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.add_metadata({"/Title": "A test document"})
    with open(pdfs / "good.pdf", "wb") as fh:
        writer.write(fh)
    (pdfs / "broken.pdf").write_bytes(b"this is not a pdf")
    (pdfs / "notes.txt").write_text("ignored")

    out = tmp_path / "out.jsonl"
    extract = load("pdf_extract")
    for _ in range(2):  # running twice must not duplicate the output
        assert extract.main([str(pdfs), "--out", str(out)]) == 0
    records = {r["file"]: r for r in map(json.loads, out.read_text().splitlines())}
    assert set(records) == {"good.pdf", "broken.pdf"}
    assert records["good.pdf"]["pages"] == 1
    assert records["good.pdf"]["metadata"]["Title"] == "A test document"
    assert "error" in records["broken.pdf"]
