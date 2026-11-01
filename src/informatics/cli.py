"""Command-line interface. Run ``informatics <command> --help`` for each command's options.

Typical use, from the repository root:

    informatics harvest --set cs          # or: import-kaggle / import-legacy-csv
    informatics filter --set cs
    informatics analyze --set cs          # several --set flags analyse sets together

Optional extras: ``enrich`` (OpenAlex publication data), ``embed`` (sentence embeddings for
clustering), ``import-pubmed`` (a PubMed comparison corpus), and the labelling commands that
measure the keyword filter and the themes.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from informatics.analysis import AnalysisConfig, run_analysis
from informatics.clustering import DEFAULT_K_CANDIDATES
from informatics.count import count_articles, default_query
from informatics.embeddings import DEFAULT_MODEL, embed, embedding_text, load_encoder, model_slug
from informatics.enrich import enrich
from informatics.filtering import FilterConfig, run_filter
from informatics.harvest import DEFAULT_BASE_URL, HarvestError, consolidate, harvest, list_sets
from informatics.http_client import HttpError
from informatics.imports import import_kaggle, import_legacy_csv
from informatics.paths import Layout, slug
from informatics.pubmed import DEFAULT_QUERY as PUBMED_QUERY
from informatics.pubmed import import_pubmed
from informatics.storage import read_json, read_parquet, write_csv, write_json, write_parquet
from informatics.validation import (
    evaluate_labels,
    evaluate_theme_labels,
    make_label_sample,
    make_theme_sample,
)


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

    p = sub.add_parser("filter", help="select AI-in-medicine papers into data/interim")
    add_set(p)
    p.add_argument("--min-ai-hits", type=int, default=1)
    p.add_argument("--min-medicine-hits", type=int, default=1)
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

    p = sub.add_parser("enrich", help="add OpenAlex publication data for the selected papers")
    add_set(p)
    p.add_argument("--api-key", default=os.environ.get("OPENALEX_API_KEY"),
                   help="OpenAlex API key (free; or set OPENALEX_API_KEY)")  # fmt: skip
    p.add_argument("--mailto", help="contact e-mail sent to OpenAlex (optional)")

    p = sub.add_parser("embed", help="cache sentence embeddings of the selected papers")
    p.add_argument("--set", dest="sets", action="append", required=True, help="repeatable")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--batch-size", type=int, default=64)

    p = sub.add_parser("label-sample", help="draw papers to hand-label for measuring the filter")
    add_set(p)
    p.add_argument("--n-selected", type=int, default=100)
    p.add_argument("--n-per-stratum", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--labels-dir", type=Path, default=Path("labels"))
    p.add_argument("--force", action="store_true", help="overwrite an existing labelling sheet")

    p = sub.add_parser("label-evaluate", help="precision and recall from a labelled sample")
    add_set(p)
    p.add_argument("--labels-dir", type=Path, default=Path("labels"))

    p = sub.add_parser("label-themes-sample", help="draw tagged papers to check each theme")
    add_set(p)
    p.add_argument("--per-theme", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--labels-dir", type=Path, default=Path("labels"))
    p.add_argument("--force", action="store_true", help="overwrite an existing labelling sheet")

    p = sub.add_parser("label-themes-evaluate", help="precision of each theme's keywords")
    add_set(p)
    p.add_argument("--labels-dir", type=Path, default=Path("labels"))

    p = sub.add_parser("to-csv", help="convert any Parquet file to a spreadsheet-friendly CSV")
    p.add_argument("parquet", type=Path)
    p.add_argument("csv", type=Path, nargs="?", help="default: same name with .csv")
    return parser


def _require(path: Path, hint: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; {hint}")
    return path


def cmd_harvest(args: argparse.Namespace, layout: Layout) -> None:
    target = layout.harvest_dir(args.set_name)
    state = harvest(
        args.set_name,
        target,
        base_url=args.base_url,
        date_from=args.date_from,
        date_until=args.date_until,
        delay=args.delay,
        restart=args.restart,
        max_pages=args.max_pages,
    )
    if state["complete"]:
        consolidate(target, layout.raw(args.set_name))
        print(f"Harvest complete: {layout.raw(args.set_name)}")
    else:
        print(f"Paused after {state['records']} records; run the same command to continue.")


def cmd_list_sets(args: argparse.Namespace, layout: Layout) -> None:
    for spec, name in list_sets(args.base_url):
        print(f"{spec}\t{name}")


def cmd_count(args: argparse.Namespace, layout: Layout) -> None:
    query = args.query or default_query(args.set_name)
    print(f"{count_articles(query):,} articles match {query!r}")


def cmd_import_kaggle(args: argparse.Namespace, layout: Layout) -> None:
    n = import_kaggle(args.snapshot, args.set_name, layout.raw(args.set_name))
    print(f"Imported {n:,} records into {layout.raw(args.set_name)}")


def cmd_import_legacy(args: argparse.Namespace, layout: Layout) -> None:
    n = import_legacy_csv(args.csv, layout.raw(args.set_name))
    print(f"Imported {n:,} records into {layout.raw(args.set_name)}")


def cmd_filter(args: argparse.Namespace, layout: Layout) -> None:
    raw = _require(
        layout.raw(args.set_name), f"run `informatics harvest --set {args.set_name}` first"
    )
    config = FilterConfig(
        min_ai_hits=args.min_ai_hits,
        min_medicine_hits=args.min_medicine_hits,
        background_size=args.background_size,
        seed=args.seed,
    )
    summary = run_filter(
        raw,
        args.set_name,
        layout.selected(args.set_name),
        layout.background(args.set_name),
        layout.filter_summary(args.set_name),
        config,
    )
    counts = summary["counts"]
    print(
        f"Selected {counts['selected']:,} of {counts['scanned']:,} papers "
        f"-> {layout.selected(args.set_name)}"
    )


def cmd_analyze(args: argparse.Namespace, layout: Layout) -> None:
    config = AnalysisConfig(
        top=args.top,
        min_df=args.min_df,
        use_background=not args.no_background,
        k=args.k,
        k_candidates=args.k_candidates,
        cluster_min_df=args.cluster_min_df,
        cluster_max_df=args.cluster_max_df,
        seed=args.seed,
        recent=args.recent,
        earlier=args.earlier,
        embeddings=args.embeddings,
        use_cache=not args.no_cache,
    )
    if bool(args.recent) != bool(args.earlier):
        raise ValueError("give both --recent and --earlier, or neither")
    out = run_analysis(args.sets, layout, args.out, config)
    print(f"Results written to {out} (start with {out / 'report.md'})")


def cmd_label_sample(args: argparse.Namespace, layout: Layout) -> None:
    sheet_path = args.labels_dir / f"{slug(args.set_name)}_to_label.csv"
    key_path = args.labels_dir / f"{slug(args.set_name)}_key.csv"
    if sheet_path.exists() and not args.force:
        raise ValueError(f"{sheet_path} exists and may hold your labels; use --force to replace it")
    summary = read_json(
        _require(layout.filter_summary(args.set_name), "run `informatics filter` first")
    )
    sheet, key = make_label_sample(
        layout.raw(args.set_name),
        layout.selected(args.set_name),
        summary,
        n_selected=args.n_selected,
        n_per_stratum=args.n_per_stratum,
        seed=args.seed,
    )
    write_csv(sheet, sheet_path)
    write_csv(key, key_path)
    print(
        f"Wrote {len(sheet)} papers to {sheet_path}. Fill in 'relevant' with 1 or 0, then "
        f"run `informatics label-evaluate --set {args.set_name}`. "
        f"Keep {key_path} closed while labelling."
    )


def cmd_label_evaluate(args: argparse.Namespace, layout: Layout) -> None:
    name = slug(args.set_name)
    read = {"dtype": str, "keep_default_na": False}
    sheet = pd.read_csv(
        _require(args.labels_dir / f"{name}_to_label.csv", "run label-sample"), **read
    )
    key = pd.read_csv(_require(args.labels_dir / f"{name}_key.csv", "run label-sample"), **read)
    summary = read_json(_require(layout.filter_summary(args.set_name), "run filter first"))
    table, result = evaluate_labels(sheet, key, summary)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    def show(value: float) -> str:
        return "n/a" if value is None or math.isnan(value) else f"{100 * value:.1f}%"

    low, high = result["precision_ci"]
    print(f"\nPrecision: {show(result['precision'])} (95% CI {show(low)}-{show(high)})")
    print(f"Estimated recall: {show(result['recall'])}")
    if result["unlabelled_strata"]:
        print(
            "Note: no labels yet for " + ", ".join(result["unlabelled_strata"]) + "; "
            "recall is an upper bound until they are labelled."
        )
    write_json(
        {"strata": table.to_dict("records"), **result}, args.labels_dir / f"{name}_evaluation.json"
    )


def cmd_import_pubmed(args: argparse.Namespace, layout: Layout) -> None:
    years = range(args.from_year, args.to_year + 1)
    n = import_pubmed(layout.raw(args.set_name), years, args.query, args.per_year, args.api_key)
    print(f"Imported {n:,} PubMed records into {layout.raw(args.set_name)}")


def cmd_enrich(args: argparse.Namespace, layout: Layout) -> None:
    selected = read_parquet(
        _require(layout.selected(args.set_name), f"run `informatics filter --set {args.set_name}`"),
        columns=["id", "doi", "duplicate_of"],
    )
    selected = selected[selected["duplicate_of"].isna()]
    if not args.api_key:
        print(
            "Note: OpenAlex has required an API key since February 2026 (free at "
            "openalex.org/settings/api); trying without one.",
            file=sys.stderr,
        )
    table = enrich(selected, api_key=args.api_key, mailto=args.mailto)
    write_parquet(table, layout.openalex(args.set_name))
    found = int(table["matched_doi"].notna().sum())
    print(
        f"Found {found:,} of {len(table):,} papers in OpenAlex -> {layout.openalex(args.set_name)}"
    )


def cmd_embed(args: argparse.Namespace, layout: Layout) -> None:
    frames = []
    for name in args.sets:
        path = _require(layout.selected(name), f"run `informatics filter --set {name}`")
        frames.append(read_parquet(path, columns=["id", "title", "abstract", "duplicate_of"]))
    papers = pd.concat(frames).drop_duplicates("id")
    papers = papers[papers["duplicate_of"].isna()].sort_values("id")
    texts = [embedding_text(t, a) for t, a in zip(papers["title"], papers["abstract"], strict=True)]
    cache = layout.embeddings(model_slug(args.model))
    vectors = embed(papers["id"].tolist(), texts, cache, load_encoder(args.model, args.batch_size))
    print(f"{len(vectors):,} embeddings ({vectors.shape[1]} dimensions) in {cache}")


def cmd_label_themes_sample(args: argparse.Namespace, layout: Layout) -> None:
    sheet_path = args.labels_dir / f"{slug(args.set_name)}_themes_to_label.csv"
    if sheet_path.exists() and not args.force:
        raise ValueError(f"{sheet_path} exists and may hold your labels; use --force to replace it")
    selected = _require(layout.selected(args.set_name), "run `informatics filter` first")
    sheet = make_theme_sample(selected, args.per_theme, args.seed)
    write_csv(sheet, sheet_path)
    print(
        f"Wrote {len(sheet)} papers to {sheet_path}. For each row, set 'relevant' to 1 if the "
        f"paper really discusses the theme, else 0; then run "
        f"`informatics label-themes-evaluate --set {args.set_name}`."
    )


def cmd_label_themes_evaluate(args: argparse.Namespace, layout: Layout) -> None:
    path = _require(
        args.labels_dir / f"{slug(args.set_name)}_themes_to_label.csv", "run label-themes-sample"
    )
    table = evaluate_theme_labels(pd.read_csv(path, dtype=str, keep_default_na=False))
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    write_csv(table, args.labels_dir / f"{slug(args.set_name)}_themes_evaluation.csv")


def cmd_to_csv(args: argparse.Namespace, layout: Layout) -> None:
    out = args.csv or args.parquet.with_suffix(".csv")
    write_csv(read_parquet(args.parquet), out)
    print(f"Wrote {out}")


COMMANDS = {
    "harvest": cmd_harvest,
    "list-sets": cmd_list_sets,
    "count": cmd_count,
    "import-kaggle": cmd_import_kaggle,
    "import-legacy-csv": cmd_import_legacy,
    "import-pubmed": cmd_import_pubmed,
    "filter": cmd_filter,
    "analyze": cmd_analyze,
    "enrich": cmd_enrich,
    "embed": cmd_embed,
    "label-sample": cmd_label_sample,
    "label-evaluate": cmd_label_evaluate,
    "label-themes-sample": cmd_label_themes_sample,
    "label-themes-evaluate": cmd_label_themes_evaluate,
    "to-csv": cmd_to_csv,
}


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
