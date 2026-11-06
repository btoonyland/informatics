"""What each ``informatics`` command does. The options are defined in cli.py."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from informatics.analysis import AnalysisConfig, run_analysis
from informatics.cache import TextCache
from informatics.count import count_articles, default_query
from informatics.embeddings import embed, embedding_text, load_encoder, model_slug
from informatics.enrich import enrich
from informatics.filtering import FilterConfig, run_filter
from informatics.harvest import (
    consolidate,
    harvest,
    latest_datestamp,
    list_sets,
    merge_update,
)
from informatics.imports import import_kaggle, import_legacy_csv
from informatics.llm import RUBRIC, LlmClassifier, LlmConfig, decisions_from_answers, paper_message
from informatics.openalex import import_openalex
from informatics.paths import Layout, slug
from informatics.provenance import file_info
from informatics.pubmed import (
    AI_MESH,
    BASELINE_SOURCE,
    SAMPLE_SOURCE,
    import_pubmed,
    import_pubmed_files,
)
from informatics.records import title_and_abstract
from informatics.relevance import train_and_classify
from informatics.selection import read_candidates, selected_papers
from informatics.storage import read_json, read_parquet, write_csv, write_json, write_parquet
from informatics.validation import (
    evaluate_labels,
    evaluate_theme_audit,
    evaluate_theme_labels,
    make_label_sample,
    make_random_theme_sample,
    make_theme_sample,
    read_labels,
)


def _record_import(layout: Layout, set_name: str, **info: object) -> None:
    """Write data/raw/<set>_import.json: how the raw file was made (see sources.py)."""
    info = {k: (str(v) if isinstance(v, Path) else v) for k, v in info.items() if v is not None}
    info["created_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    write_json(info, layout.raw_info(set_name))


def _require(path: Path, hint: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; {hint}")
    return path


def _read_sheet(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _show(value: float | None) -> str:
    return "n/a" if value is None or math.isnan(value) else f"{100 * value:.1f}%"


def _show_ci(interval: list[float]) -> str:
    low, high = interval
    return f"95% CI {_show(low)}-{_show(high)}"


def _label_paths(labels_dir: Path, set_name: str) -> tuple[Path, Path]:
    name = slug(set_name)
    return labels_dir / f"{name}_to_label.csv", labels_dir / f"{name}_key.csv"


def hand_labels(labels_dir: Path, set_name: str) -> pd.Series:
    """Hand labels of a set's labelling sheet: paper ID -> True (relevant) / False."""
    sheet_path, key_path = _label_paths(labels_dir, set_name)
    hint = f"run `informatics label-sample --set {set_name}` and label some papers first"
    labels = read_labels(
        _read_sheet(_require(sheet_path, hint)), _read_sheet(_require(key_path, hint))
    )
    labelled = labels[labels["relevant"].notna()]
    return labelled.set_index("id")["relevant"].astype(bool)


# ---------------------------------------------------------------------------------------------
# Getting the data
# ---------------------------------------------------------------------------------------------
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
        max_retries=args.retries,
    )
    if state["complete"]:
        consolidate(target, layout.raw(args.set_name))
        _record_import(
            layout, args.set_name, source="oai-pmh", set_spec=args.set_name,
            base_url=args.base_url, date_from=args.date_from, date_until=args.date_until,
        )  # fmt: skip
        print(f"Harvest complete: {state['records']:,} records in {layout.raw(args.set_name)}")
        if state["records"] == 0:  # arXiv answers an unknown set name with "no records match"
            print(
                f"Warning: no records for set {args.set_name!r}"
                + (f" changed since {args.date_from}" if args.date_from else "")
                + ". Check the set name with `informatics list-sets`.",
                file=sys.stderr,
            )
    else:
        print(f"Paused after {state['records']} records; run the same command to continue.")


def cmd_update(args: argparse.Namespace, layout: Layout) -> None:
    """Harvest only the records changed since the newest one in the raw file, and merge them."""
    raw = _require(layout.raw(args.set_name), f"harvest or import {args.set_name} first")
    since = args.date_from or latest_datestamp(raw)
    if not since:
        raise ValueError(f"{raw} has no record dates; give --from YYYY-MM-DD")
    target = layout.data_dir / "raw" / f"{slug(args.set_name)}_updates" / since
    state = harvest(
        args.set_name,
        target,
        base_url=args.base_url,
        date_from=since,
        date_until=args.date_until,
        delay=args.delay,
        max_pages=args.max_pages,
        max_retries=args.retries,
    )
    if not state["complete"]:
        print(f"Paused after {state['records']} records; run the same command to continue.")
        return
    update = consolidate(target, target / "update.parquet")
    counts = merge_update(raw, update)
    info_path = layout.raw_info(args.set_name)
    info = read_json(info_path) if info_path.exists() else {"source": "oai-pmh"}
    info.setdefault("updates", []).append(
        {"from": since, "until": args.date_until, **counts,
         "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    )  # fmt: skip
    write_json(info, info_path)
    print(
        f"{counts['added']:,} new and {counts['updated']:,} changed records since {since}; "
        f"{raw} now holds {counts['total']:,}. Run filter, classify and analyze again."
    )


def cmd_list_sets(args: argparse.Namespace, layout: Layout) -> None:
    for spec, name in list_sets(args.base_url):
        print(f"{spec}\t{name}")


def cmd_count(args: argparse.Namespace, layout: Layout) -> None:
    query = args.query or default_query(args.set_name)
    print(f"{count_articles(query):,} articles match {query!r}")


def cmd_import_kaggle(args: argparse.Namespace, layout: Layout) -> None:
    n = import_kaggle(args.snapshot, args.set_name, layout.raw(args.set_name))
    _record_import(
        layout, args.set_name, source="kaggle", set_spec=args.set_name, file=args.snapshot.name
    )
    print(f"Imported {n:,} records into {layout.raw(args.set_name)}")


def cmd_import_legacy(args: argparse.Namespace, layout: Layout) -> None:
    n = import_legacy_csv(args.csv, layout.raw(args.set_name))
    _record_import(layout, args.set_name, source="legacy-csv", file=args.csv.name)
    print(f"Imported {n:,} records into {layout.raw(args.set_name)}")


def cmd_import_pubmed(args: argparse.Namespace, layout: Layout) -> None:
    years = range(args.from_year, args.to_year + 1)
    n = import_pubmed(layout.raw(args.set_name), years, args.query, args.per_year, args.api_key)
    _record_import(
        layout, args.set_name, source=SAMPLE_SOURCE, query=args.query, per_year=args.per_year,
        years=f"{args.from_year}-{args.to_year}",
    )  # fmt: skip
    print(f"Imported {n:,} PubMed records into {layout.raw(args.set_name)}")


def cmd_import_pubmed_files(args: argparse.Namespace, layout: Layout) -> None:
    files = [f for pattern in args.files for f in sorted(Path().glob(pattern))] or [
        Path(f) for f in args.files
    ]
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(f"PubMed files not found: {', '.join(missing[:5])}")
    mesh = args.mesh or list(AI_MESH)
    counts = import_pubmed_files(
        files, layout.raw(args.set_name), mesh, args.keywords, everything=args.all
    )
    _record_import(
        layout, args.set_name, source=BASELINE_SOURCE, files=counts["files"],
        mesh=None if args.all else mesh, keywords=None if args.all else args.keywords,
        everything=args.all, counts=counts,
    )  # fmt: skip
    print(
        f"Kept {counts['kept']:,} of {counts.get('read', 0):,} PubMed records from "
        f"{counts['files']} files -> {layout.raw(args.set_name)}"
    )


def cmd_import_openalex(args: argparse.Namespace, layout: Layout) -> None:
    if not args.api_key:
        print(
            "Note: OpenAlex has required an API key since February 2026 (free at "
            "openalex.org/settings/api); trying without one.",
            file=sys.stderr,
        )
    n = import_openalex(
        layout.raw(args.set_name), args.filter, args.from_year, args.to_year, args.api_key,
        args.mailto, max_records=args.max_records,
    )  # fmt: skip
    years = f"{args.from_year or ''}-{args.to_year or ''}".strip("-") or None
    _record_import(layout, args.set_name, source="openalex", filter=args.filter, years=years)
    print(f"Imported {n:,} OpenAlex works into {layout.raw(args.set_name)}")


# ---------------------------------------------------------------------------------------------
# Selecting papers
# ---------------------------------------------------------------------------------------------
def cmd_filter(args: argparse.Namespace, layout: Layout) -> None:
    raw = _require(
        layout.raw(args.set_name), f"run `informatics harvest --set {args.set_name}` first"
    )
    config = FilterConfig(
        min_ai_hits=args.min_ai_hits,
        min_medicine_hits=args.min_medicine_hits,
        background_size=args.background_size,
        seed=args.seed,
        years=args.years,
    )
    summary = run_filter(
        raw,
        args.set_name,
        layout.candidates(args.set_name),
        layout.background(args.set_name),
        layout.filter_summary(args.set_name),
        config,
    )
    counts = summary["counts"]
    if args.years:
        print(f"Kept papers from {args.years[0]}-{args.years[1]}; left out "
              f"{counts['outside_years']:,} from other years.")  # fmt: skip
    print(
        f"{counts['candidates']:,} of {counts['scanned']:,} papers are candidates "
        f"({counts['strict']:,} meet the keyword rule) -> {layout.candidates(args.set_name)}"
    )
    if layout.relevance(args.set_name).exists():
        print(
            f"Note: run `informatics classify --set {args.set_name}` again; its decisions are "
            "for the previous candidates.",
            file=sys.stderr,
        )


def cmd_classify(args: argparse.Namespace, layout: Layout) -> None:
    candidates = read_candidates(
        layout, args.set_name, columns=["id", "title", "abstract", "keyword_rule", "duplicate_of"]
    )
    unique = candidates[candidates["duplicate_of"].isna()].reset_index(drop=True)
    if args.method == "llm":
        _classify_llm(args, layout, unique)
        return
    labels = hand_labels(args.labels_dir, args.set_name)
    cache = TextCache(layout.text_cache())
    texts = cache.prepare(unique["id"].tolist(), title_and_abstract(unique))
    cache.save()
    result = train_and_classify(unique, texts, labels, folds=args.folds, seed=args.seed)
    result.info["labels_file"] = file_info(_label_paths(args.labels_dir, args.set_name)[0])
    write_parquet(result.decisions, layout.relevance(args.set_name))
    write_json(result.info, layout.relevance_info(args.set_name))
    cv = result.info["cv"]
    print(
        f"Trained on {result.info['labels']:,} labelled candidates. Cross-validated precision "
        f"{_show(cv['precision'])}, recall {_show(cv['recall'])} (weighted to the candidates); "
        f"{result.info['selected']:,} of {result.info['candidates']:,} candidates relevant "
        f"-> {layout.relevance(args.set_name)}"
    )


def _classify_llm(args: argparse.Namespace, layout: Layout, candidates: pd.DataFrame) -> None:
    rubric = args.rubric.read_text(encoding="utf-8") if args.rubric else RUBRIC
    config = LlmConfig(model=args.llm_model, rubric=rubric, batch_size=args.batch_size)
    papers = candidates
    labels_path = _label_paths(args.labels_dir, args.set_name)[0]
    labels = hand_labels(args.labels_dir, args.set_name) if labels_path.exists() else pd.Series()
    if args.labelled_only:
        papers = candidates[candidates["id"].isin(labels.index)]
        if papers.empty:
            raise ValueError("no labelled candidates yet; run label-sample and label some papers")
    if args.max_papers is not None:
        papers = papers.sample(n=min(args.max_papers, len(papers)), random_state=args.seed)
    classifier = LlmClassifier(layout.llm(args.set_name), layout.llm_batches(args.set_name), config)
    if args.dry_run:
        todo = classifier.to_send(
            {i: paper_message(t, a) for i, t, a in
             zip(papers["id"], papers["title"], papers["abstract"], strict=True)},
            args.retry_failed,
        )  # fmt: skip
        print(f"{len(todo):,} of {len(papers):,} papers would be sent to {config.model}.")
        if todo:
            first = papers.set_index("id").loc[todo[0]]
            request = config.request(todo[0], paper_message(first["title"], first["abstract"]))
            print(json.dumps(request, indent=2, ensure_ascii=False))
        return
    try:
        counts = classifier.run(papers, wait=args.wait, retry_failed=args.retry_failed)
    except Exception as exc:
        if not _is_api_error(exc):
            raise
        raise RuntimeError(
            f"the Anthropic API failed ({type(exc).__name__}: {exc}). Nothing is lost: submitted "
            f"batches keep running and are listed in {layout.llm_batches(args.set_name)}, and "
            f"answers already collected are in {layout.llm(args.set_name)}. Run the same "
            "command again to continue."
        ) from exc
    done = ", ".join(f"{n:,} {status}" for status, n in sorted(counts.items()) if n)
    print(f"Language-model answers for {len(papers):,} papers: {done or 'none yet'}.")
    if counts["waiting"] or counts["unanswered"]:
        print("Run the same command again later (or add --wait) to collect the rest.")
        return
    if len(papers) < len(candidates):
        print(
            "Only some candidates were sent, so the selection is unchanged; check them with "
            f"`informatics label-evaluate --set {args.set_name}`."
        )
        return
    decisions, info = decisions_from_answers(candidates, classifier.answers(), labels, config.model)
    info["rubric_sha1"] = hashlib.sha1(config.system_prompt.encode("utf-8")).hexdigest()
    info["labels_file"] = file_info(labels_path)
    write_parquet(decisions, layout.relevance(args.set_name))
    write_json(info, layout.relevance_info(args.set_name))
    print(
        f"{info['selected']:,} of {info['candidates']:,} candidates relevant "
        f"({info['keyword_rule_fallback']:,} decided by the keyword rule because the model gave "
        f"no answer) -> {layout.relevance(args.set_name)}"
    )


def _is_api_error(exc: Exception) -> bool:
    """An error from the Anthropic SDK (connection, rate limit, server or request errors)."""
    try:
        import anthropic
    except ImportError:
        return False
    return isinstance(exc, anthropic.APIError)


def cmd_label_sample(args: argparse.Namespace, layout: Layout) -> None:
    sheet_path, key_path = _label_paths(args.labels_dir, args.set_name)
    existing = None
    if sheet_path.exists():
        if args.extend:
            existing = (_read_sheet(sheet_path), _read_sheet(_require(key_path, "it is missing")))
        elif not args.force:
            raise ValueError(
                f"{sheet_path} exists and may hold your labels; use --extend to add papers to "
                "it or --force to replace it"
            )
    summary = read_json(
        _require(layout.filter_summary(args.set_name), "run `informatics filter` first")
    )
    sheet, key = make_label_sample(
        layout.raw(args.set_name),
        _require(layout.candidates(args.set_name), "run `informatics filter` first"),
        summary,
        n_strict=args.n_strict,
        n_wide=args.n_wide,
        n_per_stratum=args.n_per_stratum,
        seed=args.seed,
        exclude_ids=set(existing[0]["id"]) if existing else (),
    )
    added = len(sheet)
    if existing:
        sheet = pd.concat([existing[0], sheet], ignore_index=True).fillna("")
        key = pd.concat([existing[1], key], ignore_index=True)
    write_csv(sheet, sheet_path)
    write_csv(key, key_path)
    print(
        f"Wrote {added} papers to {sheet_path}. Fill in 'relevant' with 1 or 0, then "
        f"run `informatics label-evaluate --set {args.set_name}`. "
        f"Keep {key_path} closed while labelling."
    )


def cmd_label_evaluate(args: argparse.Namespace, layout: Layout) -> None:
    sheet_path, key_path = _label_paths(args.labels_dir, args.set_name)
    sheet = _read_sheet(_require(sheet_path, "run label-sample"))
    key = _read_sheet(_require(key_path, "run label-sample"))
    summary = read_json(_require(layout.filter_summary(args.set_name), "run filter first"))
    if "strict" not in summary["counts"]:
        raise ValueError("the filter summary is from an older version; run `informatics filter`")
    decisions, method = {}, None
    if layout.relevance(args.set_name).exists():
        info_path = layout.relevance_info(args.set_name)
        method = read_json(info_path).get("method") if info_path.exists() else None
        decisions[f"classify ({method})"] = read_parquet(
            layout.relevance(args.set_name), columns=["id", "predicted"]
        )
    if layout.llm(args.set_name).exists() and method != "llm":
        answers = read_parquet(layout.llm(args.set_name), columns=["id", "status", "relevant"])
        answers = answers[answers["status"] == "ok"]
        decisions["language model"] = answers.rename(columns={"relevant": "predicted"})
    table, result = evaluate_labels(sheet, key, summary, decisions)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    rule, wide = result["keyword_rule"], result["wide_net"]
    print(
        f"\nKeyword rule: precision {_show(rule['precision'])} ({_show_ci(rule['precision_ci'])}),"
        f" recall {_show(rule['recall'])} ({_show_ci(rule['recall_ci'])}), "
        f"kappa {rule['kappa']:.2f}"
    )
    print(
        f"Wide net: {wide['candidates']:,} candidates, recall {_show(wide['recall'])} "
        f"({_show_ci(wide['recall_ci'])}), precision {_show(wide['precision'])}"
    )
    for name, c in result["classifiers"].items():
        print(
            f"{name[0].upper()}{name[1:]}: precision {_show(c['precision'])} "
            f"({_show_ci(c['precision_ci'])}), recall {_show(c['recall'])} "
            f"({_show_ci(c['recall_ci'])}), kappa {c['kappa']:.2f} on {c['compared']} labelled "
            "candidates"
        )
    if result["labellers"]:
        agree = result["labellers"]
        print(
            f"Labellers agree on {_show(agree['agreement'])} of {agree['papers']} papers "
            f"(kappa {agree['kappa']:.2f})"
        )
    if result["unlabelled_strata"]:
        print(
            "Note: no labels yet for " + ", ".join(result["unlabelled_strata"]) + "; "
            "recall is an upper bound until they are labelled."
        )
    write_json(
        {"strata": table.to_dict("records"), **result},
        args.labels_dir / f"{slug(args.set_name)}_evaluation.json",
    )


def cmd_label_themes_sample(args: argparse.Namespace, layout: Layout) -> None:
    name = slug(args.set_name)
    random_sample = args.random is not None
    sheet_path = args.labels_dir / (
        f"{name}_themes_random.csv" if random_sample else f"{name}_themes_to_label.csv"
    )
    if sheet_path.exists() and not args.force:
        raise ValueError(f"{sheet_path} exists and may hold your labels; use --force to replace it")
    papers, _ = selected_papers(layout, args.set_name, args.selection)
    if random_sample:
        sheet = make_random_theme_sample(papers, args.random, args.seed)
        instructions = (
            "For each paper, list every theme it discusses in 'themes' (separated by ';'), "
            "and set 'done' to 1"
        )
    else:
        sheet = make_theme_sample(papers, args.per_theme, args.seed)
        instructions = (
            "For each row, set 'relevant' to 1 if the paper really discusses the theme, else 0"
        )
    write_csv(sheet, sheet_path)
    print(
        f"Wrote {len(sheet)} papers to {sheet_path}. {instructions}; then run "
        f"`informatics label-themes-evaluate --set {args.set_name}`."
    )


def cmd_label_themes_evaluate(args: argparse.Namespace, layout: Layout) -> None:
    name = slug(args.set_name)
    precision_path = args.labels_dir / f"{name}_themes_to_label.csv"
    random_path = args.labels_dir / f"{name}_themes_random.csv"
    if not precision_path.exists() and not random_path.exists():
        raise FileNotFoundError(
            f"neither {precision_path} nor {random_path} exists; run label-themes-sample"
        )
    if precision_path.exists():
        table = evaluate_theme_labels(_read_sheet(precision_path))
        print("Precision of the keyword themes (papers drawn per theme):")
        print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        write_csv(table, args.labels_dir / f"{name}_themes_evaluation.csv")
    if random_path.exists():
        candidates = read_candidates(layout, args.set_name, columns=["id", "themes"])
        tagged = {"keywords": candidates.set_index("id")["themes"]}
        if layout.llm(args.set_name).exists():
            answers = read_parquet(layout.llm(args.set_name), columns=["id", "themes", "status"])
            answers = answers[answers["status"] == "ok"]
            tagged["llm"] = answers.set_index("id")["themes"]
        table = evaluate_theme_audit(_read_sheet(random_path), tagged)
        print("\nPrecision and recall of each tagging method (random papers):")
        print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        write_csv(table, args.labels_dir / f"{name}_themes_audit.csv")


# ---------------------------------------------------------------------------------------------
# Enrichment and analysis
# ---------------------------------------------------------------------------------------------
def cmd_enrich(args: argparse.Namespace, layout: Layout) -> None:
    papers, _ = selected_papers(layout, args.set_name, args.selection, columns=["doi"])
    if not args.api_key:
        print(
            "Note: OpenAlex has required an API key since February 2026 (free at "
            "openalex.org/settings/api); trying without one.",
            file=sys.stderr,
        )
    table = enrich(papers, api_key=args.api_key, mailto=args.mailto)
    write_parquet(table, layout.openalex(args.set_name))
    found = int(table["matched_doi"].notna().sum())
    print(
        f"Found {found:,} of {len(table):,} papers in OpenAlex -> {layout.openalex(args.set_name)}"
    )


def cmd_embed(args: argparse.Namespace, layout: Layout) -> None:
    frames = [
        selected_papers(layout, name, args.selection, columns=["title", "abstract"])[0]
        for name in args.sets
    ]
    papers = pd.concat(frames).drop_duplicates("id").sort_values("id")
    texts = [embedding_text(t, a) for t, a in zip(papers["title"], papers["abstract"], strict=True)]
    cache = layout.embeddings(model_slug(args.model))
    vectors = embed(papers["id"].tolist(), texts, cache, load_encoder(args.model, args.batch_size))
    print(f"{len(vectors):,} embeddings ({vectors.shape[1]} dimensions) in {cache}")


def cmd_analyze(args: argparse.Namespace, layout: Layout) -> None:
    if bool(args.recent) != bool(args.earlier):
        raise ValueError("give both --recent and --earlier, or neither")
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
        selection=args.selection,
        themes=args.themes,
        trend_min_papers=args.trend_min_papers,
        trend_min_year_papers=args.trend_min_year_papers,
    )
    out = run_analysis(args.sets, layout, args.out, config)
    print(f"Results written to {out} (start with {out / 'report.md'})")


def cmd_to_csv(args: argparse.Namespace, layout: Layout) -> None:
    out = args.csv or args.parquet.with_suffix(".csv")
    write_csv(read_parquet(args.parquet), out)
    print(f"Wrote {out}")


COMMANDS = {
    "harvest": cmd_harvest,
    "update": cmd_update,
    "list-sets": cmd_list_sets,
    "count": cmd_count,
    "import-kaggle": cmd_import_kaggle,
    "import-legacy-csv": cmd_import_legacy,
    "import-pubmed": cmd_import_pubmed,
    "import-pubmed-files": cmd_import_pubmed_files,
    "import-openalex": cmd_import_openalex,
    "filter": cmd_filter,
    "classify": cmd_classify,
    "label-sample": cmd_label_sample,
    "label-evaluate": cmd_label_evaluate,
    "label-themes-sample": cmd_label_themes_sample,
    "label-themes-evaluate": cmd_label_themes_evaluate,
    "enrich": cmd_enrich,
    "embed": cmd_embed,
    "analyze": cmd_analyze,
    "to-csv": cmd_to_csv,
}
