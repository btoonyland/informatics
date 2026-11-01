import hashlib

import pandas as pd

from informatics.provenance import file_info, sha256
from informatics.storage import write_parquet


def test_file_info(tmp_path):
    path = write_parquet(pd.DataFrame({"id": ["a", "b", "c"]}), tmp_path / "x.parquet")
    info = file_info(path)
    assert info["rows"] == 3 and info["bytes"] == path.stat().st_size
    assert info["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest() == sha256(path)
    text = tmp_path / "notes.json"
    text.write_text("{}")
    assert "rows" not in file_info(text)
    assert file_info(tmp_path / "missing.parquet") is None and file_info(None) is None


def test_run_info_fingerprints_the_inputs(layout):
    from informatics.analysis import AnalysisConfig, run_analysis
    from informatics.filtering import FilterConfig, run_filter
    from informatics.storage import read_json

    summary = run_filter(layout.raw("cs"), "cs", layout.candidates("cs"), layout.background("cs"),
                         layout.filter_summary("cs"), FilterConfig())  # fmt: skip
    assert summary["raw_file"]["sha256"] == sha256(layout.raw("cs"))
    out = run_analysis(["cs"], layout, config=AnalysisConfig(min_df=3, cluster_min_df=3, k=4))
    inputs = read_json(out / "run_info.json")["inputs"]["cs"]
    assert inputs["candidates"]["sha256"] == sha256(layout.candidates("cs"))
    assert inputs["background"]["rows"] == summary["counts"]["background_sample"]
    assert inputs["relevance"] is None and inputs["filter_summary"]["bytes"] > 0
