from pathlib import Path

import pandas as pd
import pytest

from informatics.cli import main, parse_args


@pytest.mark.parametrize(
    ("argv", "data_dir", "verbose"),
    [
        (["filter", "--set", "cs"], "data", False),
        (["--data-dir", "X", "filter", "--set", "cs"], "X", False),  # before the command
        (["filter", "--set", "cs", "--data-dir", "Y", "-v"], "Y", True),  # after the command
    ],
)
def test_global_options_work_on_either_side_of_the_command(argv, data_dir, verbose):
    args = parse_args(argv)
    assert args.data_dir == Path(data_dir) and args.verbose is verbose
    assert args.results_dir == Path("results")


def test_analyze_accepts_several_sets_and_k_options():
    args = parse_args(
        ["analyze", "--set", "cs", "--set", "q-bio", "--k", "12", "--k-candidates", "5,10"]
    )
    assert args.sets == ["cs", "q-bio"] and args.k == 12 and args.k_candidates == (5, 10)
    assert parse_args(["analyze", "--set", "cs"]).k == "auto"


def test_pipeline_through_the_cli(layout, monkeypatch, capsys):
    monkeypatch.chdir(layout.data_dir.parent)
    assert main(["filter", "--set", "cs", "--background-size", "60"]) == 0
    assert main(["analyze", "--set", "cs", "--min-df", "3", "--cluster-min-df", "3",
                 "--k-candidates", "2,3,4,5"]) == 0  # fmt: skip
    assert Path("results/cs/report.md").exists()

    assert main(["label-sample", "--set", "cs", "--n-strict", "10", "--n-wide", "4",
                 "--n-per-stratum", "3"]) == 0  # fmt: skip
    sheet_path = Path("labels/cs_to_label.csv")
    sheet = pd.read_csv(sheet_path, dtype=str, keep_default_na=False)
    sheet["relevant"] = "1"
    sheet.to_csv(sheet_path, index=False)
    # Hand-labelled work is never overwritten without --force.
    assert main(["label-sample", "--set", "cs"]) == 1
    assert "use --extend" in capsys.readouterr().err
    assert pd.read_csv(sheet_path, dtype=str)["relevant"].eq("1").all()
    # --extend adds papers and keeps the labels.
    assert main(["label-sample", "--set", "cs", "--extend", "--n-strict", "2", "--n-wide", "2",
                 "--n-per-stratum", "0"]) == 0  # fmt: skip
    extended = pd.read_csv(sheet_path, dtype=str, keep_default_na=False)
    assert len(extended) == len(sheet) + 4 and extended["id"].is_unique
    assert (extended["relevant"] == "1").sum() == len(sheet)

    assert main(["label-evaluate", "--set", "cs"]) == 0
    assert "Keyword rule: precision 100.0%" in capsys.readouterr().out
    assert Path("labels/cs_evaluation.json").exists()

    assert main(["to-csv", "data/interim/cs_candidates.parquet"]) == 0
    assert Path("data/interim/cs_candidates.csv").exists()


def test_classifier_through_the_cli(layout, corpus, monkeypatch, capsys):
    monkeypatch.chdir(layout.data_dir.parent)
    assert main(["filter", "--set", "cs"]) == 0
    assert main(["classify", "--set", "cs"]) == 1  # nothing labelled yet
    assert "label-sample" in capsys.readouterr().err
    assert main(["label-sample", "--set", "cs", "--n-strict", "40", "--n-wide", "30",
                 "--n-per-stratum", "5"]) == 0  # fmt: skip
    sheet_path = Path("labels/cs_to_label.csv")
    sheet = pd.read_csv(sheet_path, dtype=str, keep_default_na=False)
    kinds = corpus.set_index("id")["kind"]
    relevant = kinds[sheet["id"]].isin(["ai_medicine", "duplicate", "wide_relevant"])
    sheet["relevant"] = relevant.map({True: "1", False: "0"}).to_numpy()
    sheet.to_csv(sheet_path, index=False)

    assert main(["classify", "--set", "cs", "--folds", "3"]) == 0
    assert "Cross-validated precision" in capsys.readouterr().out
    decisions = pd.read_parquet("data/interim/cs_relevance.parquet").set_index("id")
    selected = decisions.index[decisions["relevant"]]
    # The kept copy of a re-posted paper can be either one, hence "duplicate".
    assert {"ai_medicine", "wide_relevant"} <= set(kinds[selected])
    assert set(kinds[selected]) <= {"ai_medicine", "duplicate", "wide_relevant"}

    assert main(["label-evaluate", "--set", "cs"]) == 0
    assert "Classify (model): precision" in capsys.readouterr().out
    assert main(["analyze", "--set", "cs", "--min-df", "3", "--cluster-min-df", "3",
                 "--k", "4"]) == 0  # fmt: skip
    report = Path("results/cs/report.md").read_text(encoding="utf-8")
    assert "selected by a classifier trained on" in report
    assert main(["analyze", "--set", "cs", "--selection", "keywords", "--min-df", "3",
                 "--cluster-min-df", "3", "--k", "4"]) == 0  # fmt: skip
    assert "selected by the keyword rule" in Path("results/cs/report.md").read_text()


def test_missing_input_returns_error_code(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["filter", "--set", "cs"]) == 1
    assert "informatics harvest --set cs" in capsys.readouterr().err


def test_analyze_period_and_embedding_options():
    args = parse_args(["analyze", "--set", "cs", "--recent", "2023-2025", "--earlier", "2019-2022",
                       "--embeddings", "some/model", "--no-cache"])  # fmt: skip
    assert (args.recent, args.earlier) == ((2023, 2025), (2019, 2022))
    assert args.embeddings == "some/model" and args.no_cache
    with pytest.raises(SystemExit):
        parse_args(["analyze", "--set", "cs", "--recent", "2025-2020"])


def test_period_options_must_come_together(layout, monkeypatch, capsys):
    monkeypatch.chdir(layout.data_dir.parent)
    assert main(["filter", "--set", "cs"]) == 0
    assert main(["analyze", "--set", "cs", "--recent", "2023-2024"]) == 1
    assert "both --recent and --earlier" in capsys.readouterr().err


def test_theme_labelling_through_the_cli(layout, monkeypatch, capsys):
    monkeypatch.chdir(layout.data_dir.parent)
    assert main(["filter", "--set", "cs"]) == 0
    assert main(["label-themes-sample", "--set", "cs", "--per-theme", "4"]) == 0
    path = Path("labels/cs_themes_to_label.csv")
    sheet = pd.read_csv(path, dtype=str, keep_default_na=False)
    sheet["relevant"] = "1"
    sheet.to_csv(path, index=False)
    assert main(["label-themes-sample", "--set", "cs"]) == 1  # labels are never overwritten
    assert main(["label-themes-evaluate", "--set", "cs"]) == 0
    assert Path("labels/cs_themes_evaluation.csv").exists()
    assert "privacy" in capsys.readouterr().out


def test_pubmed_files_through_the_cli(tmp_path, monkeypatch, capsys):
    from test_pubmed import _article, _xml_gz

    from informatics.storage import read_json

    monkeypatch.chdir(tmp_path)
    Path("downloads").mkdir()
    _xml_gz(Path("downloads/pubmed26n0001.xml.gz"), _article(1, "Deep learning for sepsis"))
    assert main(["import-pubmed-files", "downloads/*.xml.gz", "--set", "pubmed"]) == 0
    assert "Kept 1 of 1 PubMed records from 1 files" in capsys.readouterr().out
    info = read_json("data/raw/pubmed_import.json")
    assert info["source"] == "pubmed-baseline" and info["keywords"] == "strict"
    assert "Artificial Intelligence" in info["mesh"] and info["counts"]["kept"] == 1
    assert main(["import-pubmed-files", "nothing/*.xml.gz", "--set", "pubmed"]) == 1
    assert "not found" in capsys.readouterr().err


def test_sets_can_be_comma_separated():
    args = parse_args(["analyze", "--set", "cs,q-bio", "--set", "cs", "--set", "pubmed"])
    assert args.sets == ["cs", "q-bio", "pubmed"]


def test_dvc_pipeline_commands_are_valid():
    """Every command in dvc.yaml, with params.yaml filled in, parses with this CLI."""
    import re
    import shlex

    yaml = pytest.importorskip("yaml")
    root = Path(__file__).resolve().parent.parent
    params = yaml.safe_load((root / "params.yaml").read_text())
    stages = yaml.safe_load((root / "dvc.yaml").read_text())["stages"]

    def fill(text: str, item: str) -> str:
        def value(match: re.Match) -> str:
            if match.group(1) == "item":
                return item
            node = params
            for part in match.group(1).split("."):
                node = node[part]
            return str(node)

        return re.sub(r"\$\{([^}]+)\}", value, text)

    commands = []
    for stage in stages.values():
        body = stage.get("do", stage)
        for item in params["sets"] if "foreach" in stage else [""]:
            commands.append(fill(body["cmd"], item))
    assert len(commands) == 2 * len(params["sets"]) + 1
    for command in commands:
        argv = shlex.split(command)
        assert argv[0] == "informatics"
        parse_args(argv[1:])  # exits on an unknown option or value


def test_filter_years_through_the_cli(layout, monkeypatch, capsys):
    monkeypatch.chdir(layout.data_dir.parent)
    assert parse_args(["filter", "--set", "cs", "--years", "2021-2025"]).years == (2021, 2025)
    assert main(["filter", "--set", "cs", "--years", "2015-2020"]) == 0
    assert "Kept papers from 2015-2020; left out" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        parse_args(["filter", "--set", "cs", "--years", "2025-2021"])


def test_vscode_tasks_are_valid_commands():
    """Every informatics task in .vscode/tasks.json, with its default inputs, parses."""
    import json
    import re

    root = Path(__file__).resolve().parent.parent
    config = json.loads((root / ".vscode" / "tasks.json").read_text())
    defaults = {i["id"]: i["default"] for i in config["inputs"]}
    checked = 0
    for task in config["tasks"]:
        if task["command"].endswith("/informatics"):
            args = [re.sub(r"\$\{input:(\w+)\}", lambda m: defaults[m.group(1)], a)
                    for a in task["args"]]  # fmt: skip
            parse_args(args)  # exits on an unknown option or value
            checked += 1
    assert checked >= 7


def test_network_failures_say_how_to_continue(tmp_path, monkeypatch, capsys):
    import requests
    from helpers import client_for

    assert parse_args(["harvest", "--set", "cs"]).retries == 8
    assert parse_args(["update", "--set", "cs", "--retries", "40"]).retries == 40
    monkeypatch.chdir(tmp_path)
    down = requests.ConnectionError("temporary failure in name resolution")
    offline = client_for([down, down], max_retries=1)
    monkeypatch.setattr("informatics.count.HttpClient", lambda **settings: offline)
    assert main(["count", "--set", "cs"]) == 1
    err = capsys.readouterr().err
    assert "network error" in err and "run the same command again once it is back" in err
