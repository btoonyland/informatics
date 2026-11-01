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
    assert "Classifier (model): precision" in capsys.readouterr().out
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
