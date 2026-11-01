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

    assert main(["label-sample", "--set", "cs", "--n-selected", "10", "--n-per-stratum", "3"]) == 0
    sheet_path = Path("labels/cs_to_label.csv")
    sheet = pd.read_csv(sheet_path, dtype=str, keep_default_na=False)
    sheet["relevant"] = "1"
    sheet.to_csv(sheet_path, index=False)
    # Hand-labelled work is never overwritten without --force.
    assert main(["label-sample", "--set", "cs"]) == 1
    assert "use --force" in capsys.readouterr().err
    assert pd.read_csv(sheet_path, dtype=str)["relevant"].eq("1").all()

    assert main(["label-evaluate", "--set", "cs"]) == 0
    assert "Precision: 100.0%" in capsys.readouterr().out
    assert Path("labels/cs_evaluation.json").exists()

    assert main(["to-csv", "data/interim/cs_ai_medicine.parquet"]) == 0
    assert Path("data/interim/cs_ai_medicine.csv").exists()


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
