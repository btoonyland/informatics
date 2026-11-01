"""Analyse the selected papers of one or more sets and write tables plus a Markdown report.

Steps: combine the sets (cross-listed papers once) -> drop near-duplicates -> prepare and
normalise the texts (normalize.py, cached) -> phrases (most common, distinctive compared with
AI-only papers, rising and falling between periods, different between sets) -> topic clusters
(LSA or sentence embeddings) -> themes and concepts (shares with 95% intervals, by year, cluster
and set) -> publication data -> CSV/Parquet tables and report.md.
"""

from __future__ import annotations

import logging
import platform
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import sklearn

from informatics import __version__, concepts, tags
from informatics.cache import TextCache
from informatics.clustering import DEFAULT_K_CANDIDATES, cluster_papers
from informatics.embeddings import embedding_text, load_vectors, model_slug
from informatics.enrich import publication_status
from informatics.filtering import search_text
from informatics.keywords import THEMES
from informatics.normalize import NORMALIZATION_VERSION, Normalizer
from informatics.paths import Layout
from informatics.phrases import (
    PhraseIndex,
    compare_rows,
    contains_selection_term,
    frequency_table,
    keep_maximal,
    log_odds,
    surface_forms,
)
from informatics.records import title_and_abstract
from informatics.report import render_report
from informatics.selection import selected_papers
from informatics.stats import wilson_interval
from informatics.storage import (
    read_json,
    read_parquet,
    write_csv,
    write_json,
    write_parquet,
    write_text,
)

log = logging.getLogger(__name__)

OUTPUT_FILES = (
    "papers.parquet", "papers.csv", "top_phrases.csv", "distinctive_phrases.csv",
    "rising_phrases.csv", "falling_phrases.csv", "phrases_between_sets.csv", "clusters.csv",
    "k_selection.csv", "themes.csv", "themes_by_year.csv", "themes_by_cluster.csv",
    "themes_by_set.csv", "concepts.csv", "concepts_by_year.csv", "concepts_by_cluster.csv",
    "concepts_by_set.csv", "papers_by_year.csv", "publication_by_year.csv", "countries.csv",
    "run_info.json", "report.md",
)  # fmt: skip
PAPER_COLUMNS = [
    "id", "title", "authors", "categories", "year", "month", "corpora", "cluster", "ai_terms",
    "medicine_terms", "themes", "theme_terms", *[concepts.column(d) for d in concepts.CONCEPTS],
    "publication_status", "doi", "journal_ref", "comments", "venue", "venue_type", "field",
    "institution_types", "countries", "cited_by_count", "abstract",
]  # fmt: skip
OPENALEX_COLUMNS = [
    "openalex_id", "matched_by", "venue", "venue_type", "field", "institution_types",
    "countries", "cited_by_count",
]  # fmt: skip


@dataclass(frozen=True)
class AnalysisConfig:
    top: int = 100  # rows in the phrase tables
    min_df: int = 5  # a phrase must occur in at least this many papers
    maximal_ratio: float = 0.9  # see phrases.keep_maximal
    prior: float = 10.0  # strength of the log-odds prior, in papers
    use_background: bool = True
    k: int | str = "auto"
    k_candidates: tuple[int, ...] = DEFAULT_K_CANDIDATES
    cluster_min_df: int = 5
    cluster_max_df: float = 0.5
    n_components: int = 100
    seed: int = 42
    # Periods for rising/falling phrases; default: the last 3 years against the 4 before.
    recent: tuple[int, int] | None = None
    earlier: tuple[int, int] | None = None
    embeddings: str | None = None  # model whose vectors `informatics embed` cached
    use_cache: bool = True
    selection: str = "auto"  # see selection.py: auto, keywords or relevance
    themes: str = "keywords"  # or "llm": the themes a language model found (llm.py)


def combine(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate per-set tables; a paper cross-listed in several sets appears once, with all
    of its sets in ``corpora`` (so overlapping sets are never double-counted)."""
    df = pd.concat(frames, ignore_index=True)
    corpora = df.groupby("id")["corpus"].agg(lambda c: sorted(set(c)))
    df = df.drop_duplicates("id").drop(columns="corpus")
    df["corpora"] = df["id"].map(corpora)
    return df.sort_values("id").reset_index(drop=True)


paper_texts = title_and_abstract


def papers_by_year(papers: pd.DataFrame, summaries: dict[str, dict | None]) -> pd.DataFrame:
    """Selected papers per year and, per set, how many papers the set had that year, how many
    were candidates, and the share of all its papers that were selected."""
    years = papers["year"].astype("Float64")
    columns = [years.dropna().astype(int).value_counts().rename("selected")]
    names = [n for n, s in summaries.items() if s]
    for name in names:
        by_year = pd.DataFrame.from_dict(summaries[name]["by_year"], orient="index")
        by_year.index = by_year.index.astype(int)
        by_year = by_year.reindex(columns=["scanned", "candidates"]).fillna(0).astype(int)
        in_set = papers["corpora"].map(lambda c, n=name: n in c)
        selected = years[in_set].dropna().astype(int).value_counts().rename("selected")
        columns.append(pd.concat([by_year, selected], axis=1).add_prefix(f"{name}_"))
    table = pd.concat(columns, axis=1).fillna(0).astype(int).sort_index()
    for name in names:
        scanned = table[f"{name}_scanned"].replace(0, np.nan)
        table[f"{name}_selected_share"] = table[f"{name}_selected"] / scanned
    table.index.name = "year"
    return table.reset_index()


def default_periods(years: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """(earlier, recent): the last three years with papers against the four years before."""
    valid = years[~np.isnan(years)]
    if not valid.size:
        return None
    last = int(valid.max())
    return (last - 6, last - 3), (last - 2, last)


def _share(row: dict, name: str, hits: int, n: int) -> None:
    low, high = wilson_interval(hits, n)
    row[f"share_{name}"] = hits / n if n else np.nan
    row[f"{name}_ci_low"], row[f"{name}_ci_high"] = low, high


def publication_by_year(papers: pd.DataFrame, with_openalex: bool) -> pd.DataFrame:
    """Per year: evidence of publication from arXiv's fields and, if enriched, from OpenAlex
    (OpenAlex shares are among the papers OpenAlex matched)."""
    rows = []
    for year, group in papers.dropna(subset=["year"]).groupby("year"):
        n = len(group)
        status = group["publication_status"]
        row = {
            "year": int(year),
            "papers": n,
            "share_journal_ref": float((status == "journal reference").mean()),
            "share_accepted_comment": float((status == "accepted (comments)").mean()),
        }
        _share(row, "any_evidence", int((status != "no evidence").sum()), n)
        if with_openalex:
            matched = group[group["openalex_id"].notna()]
            row["share_in_openalex"] = len(matched) / n
            m = len(matched)
            published = int(matched["venue_type"].isin(["journal", "conference"]).sum())
            _share(row, "journal_or_conference", published, m)
            for kind in ("healthcare", "company"):
                hits = int(matched["institution_types"].map(lambda t, k=kind: k in t).sum())
                _share(row, f"{kind}_author", hits, m)
            row["median_citations"] = matched["cited_by_count"].median() if m else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def countries_table(papers: pd.DataFrame) -> pd.DataFrame:
    matched = papers[papers["openalex_id"].notna()]
    counts = matched["countries"].explode().dropna().value_counts()
    table = counts.rename_axis("country").reset_index(name="papers")
    table["share"] = table["papers"] / max(len(matched), 1)
    return table


def with_llm_themes(papers: pd.DataFrame, sets: list[str], layout: Layout) -> pd.DataFrame:
    """Replace the keyword themes by the themes the language model found (llm.py)."""
    answers = []
    for name in sets:
        if layout.llm(name).exists():
            found = read_parquet(layout.llm(name), columns=["id", "status", "themes"])
            answers.append(found[found["status"] == "ok"])
    themes = (
        pd.concat(answers).drop_duplicates("id").set_index("id")["themes"]
        if answers
        else pd.Series(dtype=object)
    )
    missing = int((~papers["id"].isin(themes.index)).sum())
    if missing:
        raise ValueError(
            f"{missing:,} of the {len(papers):,} papers have no themes from the language model; "
            f"run `informatics classify --method llm` for {', '.join(sets)} first"
        )
    papers = papers.copy()
    papers["themes"] = papers["id"].map(themes)
    papers["theme_terms"] = [[] for _ in range(len(papers))]  # no matched words to show
    return papers


def _git_commit() -> str | None:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return commit + ("-dirty" if dirty else "")


def _renamed(a: str, b: str) -> dict[str, str]:
    """Column names for a log-odds table comparing group ``a`` with group ``b``."""
    return {
        "papers": f"{a}_papers",
        "share": f"{a}_share",
        "background_papers": f"{b}_papers",
        "background_share": f"{b}_share",
    }


def _with_phrases(table: pd.DataFrame | None, surfaces: dict[str, str]) -> pd.DataFrame | None:
    if table is None:
        return None
    table = table.copy()
    table.insert(0, "phrase", table["key"].map(lambda k: surfaces.get(k, k)))
    table["contains_selection_term"] = table["phrase"].map(contains_selection_term)
    return table.drop(columns="key")


def run_analysis(
    sets: list[str],
    layout: Layout | None = None,
    out_dir: str | Path | None = None,
    config: AnalysisConfig | None = None,
) -> Path:
    layout = layout or Layout()
    config = config or AnalysisConfig()
    out_dir = Path(out_dir) if out_dir else layout.results(sets)

    frames, backgrounds, enriched, summaries, inputs, selections = [], [], [], {}, {}, {}
    for name in sets:
        chosen, selection = selected_papers(layout, name, config.selection)
        frames.append(chosen)
        selections[name] = selection.as_dict()
        summary_path = layout.filter_summary(name)
        summaries[name] = read_json(summary_path) if summary_path.exists() else None
        background_path = layout.background(name)
        use_bg = config.use_background and background_path.exists()
        if use_bg:
            backgrounds.append(read_parquet(background_path))
        if layout.openalex(name).exists():
            enriched.append(read_parquet(layout.openalex(name)))
        inputs[name] = {
            "candidates": str(layout.candidates(name)),
            "relevance": (str(layout.relevance(name)) if selection.method == "relevance" else None),
            "background": str(background_path) if use_bg else None,
            "openalex": str(layout.openalex(name)) if layout.openalex(name).exists() else None,
        }

    papers = combine(frames)
    n_duplicates = sum(s["duplicates_dropped"] for s in selections.values())
    if papers.empty:
        raise ValueError("there are no selected papers to analyse")
    papers["corpus_group"] = papers["corpora"].map("+".join)
    background = combine(backgrounds) if backgrounds else None
    if background is not None:
        background = background[~background["id"].isin(papers["id"])].reset_index(drop=True)
    n_background = 0 if background is None else len(background)
    log.info("analysing %d papers against %d AI-only papers", len(papers), n_background)

    # Prepared and normalised text -------------------------------------------------------------
    cache = TextCache(layout.text_cache() if config.use_cache else None)
    texts = cache.prepare(papers["id"].tolist(), paper_texts(papers))
    bg_texts = (
        cache.prepare(background["id"].tolist(), paper_texts(background)) if n_background else []
    )
    cache.save()
    normalizer = Normalizer.from_texts(texts + bg_texts)

    # Phrases ----------------------------------------------------------------------------------
    index = PhraseIndex.build(texts, normalizer, 2, 3, config.min_df)
    frequent = keep_maximal(index.counts(), config.maximal_ratio)
    top_keys = sorted(frequent, key=lambda k: (-frequent[k], k))[: config.top]
    distinctive = None
    if n_background and frequent:
        distinctive = log_odds(
            frequent, index.n_docs, index.counts_in(bg_texts), n_background, config.prior
        ).head(config.top)

    years = papers["year"].astype("Float64").to_numpy(dtype=float, na_value=np.nan)
    periods = (
        (config.earlier, config.recent)
        if config.recent and config.earlier
        else default_periods(years)
    )
    rising = falling = None
    if periods and frequent:
        (e0, e1), (r0, r1) = periods
        recent_rows = (years >= r0) & (years <= r1)
        earlier_rows = (years >= e0) & (years <= e1)
        if recent_rows.any() and earlier_rows.any():
            trend = compare_rows(index, recent_rows, earlier_rows, set(frequent), config.prior)
            trend = trend.rename(columns=_renamed("recent", "earlier"))
            rising = trend[trend["z"] > 0].head(config.top)
            falling = trend[trend["z"] < 0].sort_values("z").head(config.top)
        else:
            periods = None

    between_sets = None
    if len(sets) >= 2 and frequent:
        only_a = papers["corpora"].map(lambda c: c == [sets[0]]).to_numpy()
        only_b = papers["corpora"].map(lambda c: c == [sets[1]]).to_numpy()
        if only_a.any() and only_b.any():
            table = compare_rows(index, only_a, only_b, set(frequent), config.prior)
            table = table.rename(columns=_renamed(sets[0], sets[1]))
            between_sets = pd.concat([table.head(config.top), table.tail(config.top)])
            between_sets = between_sets[~between_sets.index.duplicated()]

    shown = set(top_keys)
    for table in (distinctive, rising, falling, between_sets):
        if table is not None:
            shown |= set(table["key"])
    surfaces = surface_forms(texts, shown, normalizer, 2, 3)
    top_phrases = frequency_table(frequent, index.n_docs, surfaces, config.top)
    distinctive, rising, falling, between_sets = (
        _with_phrases(t, surfaces) for t in (distinctive, rising, falling, between_sets)
    )
    log.info("counted %d phrases used in at least %d papers", len(frequent), config.min_df)

    # Clusters ---------------------------------------------------------------------------------
    vectors = None
    if config.embeddings:
        vectors = load_vectors(
            papers["id"].tolist(),
            [
                embedding_text(t, a)
                for t, a in zip(papers["title"], papers["abstract"], strict=True)
            ],
            layout.embeddings(model_slug(config.embeddings)),
        )
    clusters = cluster_papers(
        texts,
        papers["title"].fillna("").tolist(),
        papers["id"].tolist(),
        normalizer=normalizer,
        vectors=vectors,
        k=config.k,
        k_candidates=config.k_candidates,
        min_df=config.cluster_min_df,
        max_df=config.cluster_max_df,
        n_components=config.n_components,
        seed=config.seed,
    )
    papers["cluster"] = clusters.labels

    # Themes and concepts ----------------------------------------------------------------------
    if config.themes == "llm":
        papers = with_llm_themes(papers, sets, layout)
    elif config.themes != "keywords":
        raise ValueError(f"unknown theme source {config.themes!r}; use keywords or llm")
    papers = papers.join(concepts.tag_concepts(search_text(papers)))
    groups = ["year", "cluster"] + (["corpus_group"] if len(sets) > 1 else [])
    theme_names = list(THEMES)
    theme_tables = {"overall": tags.overview(papers, "themes", theme_names)}
    for by in groups:
        theme_tables[by] = tags.shares_by(papers, "themes", theme_names, by)
    concept_tables: dict[str, list[pd.DataFrame]] = {"overall": [], **{by: [] for by in groups}}
    for dimension, names in concepts.CONCEPTS.items():
        col = concepts.column(dimension)
        concept_tables["overall"].append(
            tags.overview(papers, col, list(names)).assign(dimension=dimension)
        )
        for by in groups:
            table = tags.shares_by(papers, col, list(names), by)
            concept_tables[by].append(table.assign(dimension=dimension))
    concept_tables = {k: pd.concat(v, ignore_index=True) for k, v in concept_tables.items()}

    # Publication data -------------------------------------------------------------------------
    comments = papers["comments"] if "comments" in papers else pd.Series(None, index=papers.index)
    papers["publication_status"] = [
        publication_status(j, c) for j, c in zip(papers["journal_ref"], comments, strict=True)
    ]
    with_openalex = bool(enriched)
    if with_openalex:
        oa = pd.concat(enriched, ignore_index=True).drop_duplicates("id")
        oa = oa[["id", *[c for c in OPENALEX_COLUMNS if c in oa]]]
        papers = papers.merge(oa, on="id", how="left")
        for col in ("institution_types", "countries"):
            papers[col] = papers[col].map(lambda v: v if isinstance(v, list) else [])
    publication = publication_by_year(papers, with_openalex)
    countries = countries_table(papers) if with_openalex else None
    year_table = papers_by_year(papers, summaries)

    run_info = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "informatics_version": __version__,
        "normalization_version": NORMALIZATION_VERSION,
        "git_commit": _git_commit(),
        "sets": sets,
        "selection": selections,
        "inputs": inputs,
        "config": asdict(config),
        "periods": {"earlier": periods[0], "recent": periods[1]} if periods else None,
        "counts": {
            "papers": len(papers),
            "near_duplicates_dropped": n_duplicates,
            "cross_listed": int((papers["corpora"].map(len) > 1).sum()),
            "background": n_background,
            "phrases_kept": len(frequent),
            "k": clusters.k,
            "openalex_matched": int(papers["openalex_id"].notna().sum()) if with_openalex else None,
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pa.__version__,
            "scikit-learn": sklearn.__version__,
        },
    }

    # Files ------------------------------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in OUTPUT_FILES:  # never leave files from an earlier run that this run did not make
        (out_dir / name).unlink(missing_ok=True)
    columns = [c for c in PAPER_COLUMNS if c in papers.columns]
    write_parquet(papers[columns], out_dir / "papers.parquet")
    write_csv(papers[columns], out_dir / "papers.csv")
    optional = {
        "distinctive_phrases.csv": distinctive,
        "rising_phrases.csv": rising,
        "falling_phrases.csv": falling,
        "phrases_between_sets.csv": between_sets,
        "countries.csv": countries,
        "k_selection.csv": None if clusters.k_selection.empty else clusters.k_selection,
        "themes_by_set.csv": theme_tables.get("corpus_group"),
        "concepts_by_set.csv": concept_tables.get("corpus_group"),
    }
    always = {
        "top_phrases.csv": top_phrases,
        "clusters.csv": clusters.summary,
        "themes.csv": theme_tables["overall"],
        "themes_by_year.csv": theme_tables["year"],
        "themes_by_cluster.csv": theme_tables["cluster"],
        "concepts.csv": concept_tables["overall"],
        "concepts_by_year.csv": concept_tables["year"],
        "concepts_by_cluster.csv": concept_tables["cluster"],
        "papers_by_year.csv": year_table,
        "publication_by_year.csv": publication,
    }
    for name, table in {**always, **optional}.items():
        if table is not None:
            write_csv(table, out_dir / name)
    write_json(run_info, out_dir / "run_info.json")
    report = render_report(
        {
            "sets": sets,
            "papers": papers,
            "summaries": summaries,
            "selections": selections,
            "run_info": run_info,
            "n_background": n_background,
            "year_table": year_table,
            "top_phrases": top_phrases,
            "distinctive_phrases": distinctive,
            "rising_phrases": rising,
            "falling_phrases": falling,
            "phrases_between_sets": between_sets,
            "periods": periods,
            "clusters": clusters,
            "theme_tables": theme_tables,
            "concept_tables": concept_tables,
            "publication": publication,
            "countries": countries,
        }
    )
    write_text(report, out_dir / "report.md")
    log.info("wrote results to %s", out_dir)
    return out_dir
