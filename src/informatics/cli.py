"""Command-line interface: the options of each command (commands.py does the work).

Run ``informatics <command> --help`` for each command's options. Typical use, from the
repository root:

    informatics harvest --set cs          # or: import-kaggle / import-legacy-csv
    informatics filter --set cs           # wide keyword net -> candidates
    informatics label-sample --set cs     # then label, and: informatics classify --set cs
    informatics analyze --set cs          # several --set flags analyse sets together

Optional extras: ``enrich`` (OpenAlex publication data), ``embed`` (sentence embeddings for
clustering), ``import-pubmed`` (a PubMed comparison corpus), and the labelling commands that
measure the selection and the themes.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date
from pathlib import Path

from informatics.clustering import DEFAULT_K_CANDIDATES
from informatics.commands import COMMANDS
from informatics.embeddings import DEFAULT_MODEL
from informatics.harvest import DEFAULT_BASE_URL, HarvestError
from informatics.http_client import HttpError
from informatics.llm import DEFAULT_MODEL as LLM_MODEL
from informatics.paths import Layout
from informatics.pubmed import DEFAULT_QUERY as PUBMED_QUERY
from informatics.selection import SELECTIONS


def _k(value: str) -> int | str:
    return value if value == "auto" else int(value)


def _int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(v) for v in value.split(",") if v.strip())


def _years(value: str) -> tuple[int, int]:
    """ "2019-2022" -> (2019, 2022); "2024" -> (2024, 2024)."""
    first, _, last = value.partition("-")
    start, end = int(first), int(last or first)
    if end < start:
        raise argparse.ArgumentTypeError(f"{value}: the second year is before the first")
    return start, end


GLOBAL_DEFAULTS = {"data_dir": Path("data"), "results_dir": Path("results"), "verbose": False}


def _global_options() -> argparse.ArgumentParser:
    """Options accepted before or after the command name. A fresh parser per use: argparse
    shares option objects with its parents, so one copy would leak defaults between them."""
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-dir", type=Path, default=argparse.SUPPRESS, help="default: data")
    common.add_argument(
        "--results-dir", type=Path, default=argparse.SUPPRESS, help="default: results"
    )
    common.add_argument("-v", "--verbose", action="store_true", default=argparse.SUPPRESS)
    return common


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="informatics",
        description="Harvest arXiv metadata, select AI-in-medicine papers and analyse them.",
        parents=[_global_options()],
    )
    sub = parser.add_subparsers(dest="command", required=True)
    _add_parser = sub.add_parser

    def add_parser(name: str, **kwargs: object) -> argparse.ArgumentParser:
        return _add_parser(name, parents=[_global_options()], **kwargs)

    sub.add_parser = add_parser  # type: ignore[method-assign]

    def add_set(p: argparse.ArgumentParser, help_text: str = "arXiv set, e.g. cs or q-bio") -> None:
        p.add_argument("--set", dest="set_name", metavar="SET", required=True, help=help_text)

    def add_labels_dir(p: argparse.ArgumentParser) -> None:
        p.add_argument("--labels-dir", type=Path, default=Path("labels"))

    def add_selection(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--selection",
            choices=SELECTIONS,
            default="auto",
            help="which candidates count as selected: the keyword rule, the decisions of "
            "`informatics classify`, or auto (classify's if there are any; default)",
        )

    p = sub.add_parser("harvest", help="download a set over OAI-PMH into data/raw (resumable)")
    add_set(p, "OAI-PMH set, e.g. cs or q-bio (see list-sets)")
    p.add_argument("--from", dest="date_from", help="only records changed on/after YYYY-MM-DD")
    p.add_argument("--until", dest="date_until", help="only records changed on/before YYYY-MM-DD")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--delay", type=float, default=5.0, help="seconds between requests")
    p.add_argument("--max-pages", type=int, help="stop after N pages (run again to continue)")
    p.add_argument("--restart", action="store_true", help="discard a previous partial harvest")

    p = sub.add_parser("list-sets", help="list the sets the OAI-PMH endpoint offers")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)

    p = sub.add_parser("count", help="count a set's articles with the arXiv search API")
    add_set(p)
    p.add_argument("--query", help="arXiv API query (default: cat:<set>*)")

    p = sub.add_parser("import-kaggle", help="build data/raw from Cornell's Kaggle snapshot")
    p.add_argument("snapshot", type=Path, help="arxiv-metadata-oai-snapshot.json")
    add_set(p)

    p = sub.add_parser(
        "import-legacy-csv", help="convert a CSV from the old arXivAPI.py (e.g. All_CS.csv)"
    )
    p.add_argument("csv", type=Path)
    add_set(p)

    p = sub.add_parser("import-pubmed", help="build a PubMed comparison set (NCBI E-utilities)")
    add_set(p, "name for the set, e.g. pubmed")
    p.add_argument("--query", default=PUBMED_QUERY, help=f"PubMed query (default: {PUBMED_QUERY})")
    p.add_argument("--from-year", type=int, default=2015)
    p.add_argument("--to-year", type=int, default=date.today().year)
    p.add_argument("--per-year", type=int, default=2000, help="records per year (max 9999)")
    p.add_argument("--api-key", default=os.environ.get("NCBI_API_KEY"), help="or set NCBI_API_KEY")

    p = sub.add_parser("filter", help="find candidate papers with the wide keyword net")
    add_set(p)
    p.add_argument("--min-ai-hits", type=int, default=1, help="for the original keyword rule")
    p.add_argument("--min-medicine-hits", type=int, default=1, help="likewise")
    p.add_argument(
        "--background-size", type=int, default=50_000, help="AI-only papers kept for comparison"
    )
    p.add_argument("--seed", type=int, default=42)

    p = sub.add_parser("analyze", help="phrases, clusters and themes into results/<set>")
    p.add_argument("--set", dest="sets", action="append", required=True, help="repeatable")
    p.add_argument("--out", type=Path, help="default: results/<set>[+<set>...]")
    p.add_argument("--top", type=int, default=100, help="rows in the phrase tables")
    p.add_argument("--min-df", type=int, default=5, help="minimum papers per phrase")
    p.add_argument("--no-background", action="store_true", help="skip the AI-only comparison")
    p.add_argument("--k", type=_k, default="auto", help="number of clusters, or 'auto'")
    p.add_argument(
        "--k-candidates",
        type=_int_tuple,
        default=DEFAULT_K_CANDIDATES,
        help="comma-separated k values tried when --k auto",
    )
    p.add_argument("--cluster-min-df", type=int, default=5)
    p.add_argument("--cluster-max-df", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--recent", type=_years, help="period for rising phrases, e.g. 2023-2025")
    p.add_argument("--earlier", type=_years, help="period to compare with, e.g. 2019-2022")
    p.add_argument("--embeddings", metavar="MODEL", help="cluster on vectors from `embed`")
    p.add_argument("--no-cache", action="store_true", help="re-prepare every text")
    p.add_argument(
        "--trend-min-papers", type=int, default=20, help="trends: papers with and without a tag"
    )
    p.add_argument("--trend-min-year-papers", type=int, default=20, help="trends: papers in a year")
    p.add_argument(
        "--themes",
        choices=["keywords", "llm"],
        default="keywords",
        help="tag themes by keyword patterns, or use the language model's (classify --method llm)",
    )
    add_selection(p)

    p = sub.add_parser("enrich", help="add OpenAlex publication data for the selected papers")
    add_set(p)
    p.add_argument("--api-key", default=os.environ.get("OPENALEX_API_KEY"),
                   help="OpenAlex API key (free; or set OPENALEX_API_KEY)")  # fmt: skip
    p.add_argument("--mailto", help="contact e-mail sent to OpenAlex (optional)")
    add_selection(p)

    p = sub.add_parser("embed", help="cache sentence embeddings of the selected papers")
    p.add_argument("--set", dest="sets", action="append", required=True, help="repeatable")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--batch-size", type=int, default=64)
    add_selection(p)

    p = sub.add_parser("label-sample", help="draw papers to hand-label for measuring selection")
    add_set(p)
    p.add_argument("--n-strict", type=int, default=150, help="candidates meeting the keyword rule")
    p.add_argument("--n-wide", type=int, default=150, help="other candidates")
    p.add_argument(
        "--n-per-stratum", type=int, default=50, help="papers outside the wide net, per stratum"
    )
    p.add_argument("--seed", type=int, default=42)
    add_labels_dir(p)
    p.add_argument("--extend", action="store_true", help="add papers to an existing sheet")
    p.add_argument("--force", action="store_true", help="overwrite an existing labelling sheet")

    p = sub.add_parser("label-evaluate", help="precision and recall from a labelled sample")
    add_set(p)
    add_labels_dir(p)

    p = sub.add_parser("classify", help="decide which candidates are relevant (second stage)")
    add_set(p)
    p.add_argument(
        "--method",
        choices=["model", "llm"],
        default="model",
        help="model: train a classifier on the hand labels; llm: ask a language model, "
        "through the Message Batches API (needs ANTHROPIC_API_KEY)",
    )
    p.add_argument("--folds", type=int, default=5, help="model: cross-validation folds")
    p.add_argument("--seed", type=int, default=42)
    add_labels_dir(p)
    p.add_argument("--llm-model", default=LLM_MODEL, help=f"llm: default {LLM_MODEL}")
    p.add_argument("--rubric", type=Path, help="llm: a text file replacing the built-in rubric")
    p.add_argument(
        "--labelled-only",
        action="store_true",
        help="llm: only send hand-labelled candidates, to check the rubric first",
    )
    p.add_argument("--max-papers", type=int, help="llm: send a random sample of this size")
    p.add_argument("--batch-size", type=int, default=5_000, help="llm: requests per batch")
    p.add_argument("--wait", action="store_true", help="llm: wait until every batch has ended")
    p.add_argument(
        "--retry-failed", action="store_true", help="llm: send refused and invalid requests again"
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="llm: show what would be sent, and one request, without sending",
    )

    p = sub.add_parser("label-themes-sample", help="draw papers to check the themes")
    add_set(p)
    p.add_argument("--per-theme", type=int, default=30)
    p.add_argument(
        "--random",
        type=int,
        metavar="N",
        help="instead draw N papers at random, to tag with all their themes (measures recall)",
    )
    p.add_argument("--seed", type=int, default=42)
    add_labels_dir(p)
    p.add_argument("--force", action="store_true", help="overwrite an existing labelling sheet")
    add_selection(p)

    p = sub.add_parser("label-themes-evaluate", help="precision (and recall) of the themes")
    add_set(p)
    add_labels_dir(p)

    p = sub.add_parser("to-csv", help="convert any Parquet file to a spreadsheet-friendly CSV")
    p.add_argument("parquet", type=Path)
    p.add_argument("csv", type=Path, nargs="?", help="default: same name with .csv")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    for name, default in GLOBAL_DEFAULTS.items():
        if not hasattr(args, name):
            setattr(args, name, default)
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        COMMANDS[args.command](args, Layout(args.data_dir, args.results_dir))
    except (FileNotFoundError, ValueError, RuntimeError, HarvestError, HttpError) as exc:
        print(f"informatics: error: {exc}", file=sys.stderr)
        return 1
    return 0
