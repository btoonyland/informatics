"""Analyse the selected papers of one or more sets and write tables plus a Markdown report."""

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

from informatics import __version__, themes
from informatics.clustering import DEFAULT_K_CANDIDATES, cluster_papers
from informatics.paths import Layout
from informatics.phrases import (
    contains_selection_term,
    document_frequencies,
    frequency_table,
    keep_maximal,
    log_odds,
    surface_forms,
)
from informatics.report import render_report
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
    "papers.parquet", "papers.csv", "top_phrases.csv", "distinctive_phrases.csv", "clusters.csv",
    "k_selection.csv", "themes.csv", "themes_by_year.csv", "themes_by_cluster.csv",
    "papers_by_year.csv", "run_info.json", "report.md",
)  # fmt: skip
PAPER_COLUMNS = [
    "id", "title", "authors", "categories", "year", "month", "corpora", "cluster", "ai_terms",
    "medicine_terms", "themes", "theme_terms", "doi", "journal_ref", "abstract",
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


def combine(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate per-set tables; a paper cross-listed in several sets appears once, with all
    of its sets in ``corpora`` (so overlapping sets are never double-counted)."""
    df = pd.concat(frames, ignore_index=True)
    corpora = df.groupby("id")["corpus"].agg(lambda c: sorted(set(c)))
    df = df.drop_duplicates("id").drop(columns="corpus")
    df["corpora"] = df["id"].map(corpora)
    return df.sort_values("id").reset_index(drop=True)


def paper_texts(df: pd.DataFrame) -> list[str]:
    return (df["title"].fillna("") + ". " + df["abstract"].fillna("")).tolist()


def papers_by_year(papers: pd.DataFrame, summaries: dict[str, dict | None]) -> pd.DataFrame:
    """Selected papers per year and, per set, the share of all scanned papers they represent."""
    columns = [papers["year"].dropna().astype(int).value_counts().rename("selected")]
    names = [n for n, s in summaries.items() if s]
    for name in names:
        by_year = pd.DataFrame.from_dict(summaries[name]["by_year"], orient="index")
        by_year.index = by_year.index.astype(int)
        by_year = by_year.reindex(columns=["scanned", "ai", "selected"]).fillna(0).astype(int)
        columns.append(by_year.add_prefix(f"{name}_"))
    table = pd.concat(columns, axis=1).fillna(0).astype(int).sort_index()
    for name in names:
        scanned = table[f"{name}_scanned"].replace(0, np.nan)
        table[f"{name}_selected_share"] = table[f"{name}_selected"] / scanned
    table.index.name = "year"
    return table.reset_index()


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


def run_analysis(
    sets: list[str],
    layout: Layout | None = None,
    out_dir: str | Path | None = None,
    config: AnalysisConfig | None = None,
) -> Path:
    layout = layout or Layout()
    config = config or AnalysisConfig()
    out_dir = Path(out_dir) if out_dir else layout.results(sets)

    frames, backgrounds, summaries, inputs = [], [], {}, {}
    for name in sets:
        selected_path = layout.selected(name)
        if not selected_path.exists():
            raise FileNotFoundError(
                f"{selected_path} not found; run `informatics filter --set {name}` first"
            )
        frames.append(read_parquet(selected_path))
        summary_path = layout.filter_summary(name)
        summaries[name] = read_json(summary_path) if summary_path.exists() else None
        background_path = layout.background(name)
        use_bg = config.use_background and background_path.exists()
        if use_bg:
            backgrounds.append(read_parquet(background_path))
        inputs[name] = {
            "selected": str(selected_path),
            "background": str(background_path) if use_bg else None,
        }

    papers = combine(frames)
    if papers.empty:
        raise ValueError("there are no selected papers to analyse")
    background = combine(backgrounds) if backgrounds else None
    if background is not None:
        background = background[~background["id"].isin(papers["id"])]
    n_background = 0 if background is None else len(background)
    texts = paper_texts(papers)
    log.info("analysing %d papers against %d AI-only papers", len(papers), n_background)

    # Phrases --------------------------------------------------------------------------------
    counts, n_docs = document_frequencies(texts)
    frequent = keep_maximal(
        {k: c for k, c in counts.items() if c >= config.min_df}, config.maximal_ratio
    )
    del counts
    top_keys = sorted(frequent, key=lambda k: (-frequent[k], k))[: config.top]
    distinctive = None
    if n_background:
        bg_counts, n_bg = document_frequencies(paper_texts(background), vocabulary=set(frequent))
        distinctive = log_odds(frequent, n_docs, bg_counts, n_bg, config.prior).head(config.top)
    needed = set(top_keys) | (set(distinctive["key"]) if distinctive is not None else set())
    surfaces = surface_forms(texts, needed, 2, 3)
    top_phrases = frequency_table(frequent, n_docs, surfaces, config.top)
    if distinctive is not None:
        distinctive.insert(0, "phrase", distinctive["key"].map(lambda k: surfaces.get(k, k)))
        distinctive["contains_selection_term"] = distinctive["phrase"].map(contains_selection_term)
        distinctive = distinctive.drop(columns="key")
    log.info("counted %d phrases used in at least %d papers", len(frequent), config.min_df)

    # Clusters -------------------------------------------------------------------------------
    clusters = cluster_papers(
        texts,
        papers["title"].fillna("").tolist(),
        papers["id"].tolist(),
        k=config.k,
        k_candidates=config.k_candidates,
        min_df=config.cluster_min_df,
        max_df=config.cluster_max_df,
        n_components=config.n_components,
        seed=config.seed,
    )
    papers["cluster"] = clusters.labels

    # Themes and trends ----------------------------------------------------------------------
    theme_table = themes.overview(papers)
    themes_by_year = themes.share_by(papers, "year")
    themes_by_cluster = themes.share_by(papers, "cluster")
    year_table = papers_by_year(papers, summaries)

    run_info = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "informatics_version": __version__,
        "git_commit": _git_commit(),
        "sets": sets,
        "inputs": inputs,
        "config": asdict(config),
        "counts": {
            "papers": len(papers),
            "cross_listed": int((papers["corpora"].map(len) > 1).sum()),
            "background": n_background,
            "phrases_kept": len(frequent),
            "k": clusters.k,
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pa.__version__,
            "scikit-learn": sklearn.__version__,
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    for name in OUTPUT_FILES:  # never leave files from an earlier run that this run did not make
        (out_dir / name).unlink(missing_ok=True)
    columns = [c for c in PAPER_COLUMNS if c in papers.columns]
    write_parquet(papers[columns], out_dir / "papers.parquet")
    write_csv(papers[columns], out_dir / "papers.csv")
    write_csv(top_phrases, out_dir / "top_phrases.csv")
    if distinctive is not None:
        write_csv(distinctive, out_dir / "distinctive_phrases.csv")
    write_csv(clusters.summary, out_dir / "clusters.csv")
    if not clusters.k_selection.empty:
        write_csv(clusters.k_selection, out_dir / "k_selection.csv")
    write_csv(theme_table, out_dir / "themes.csv")
    write_csv(themes_by_year, out_dir / "themes_by_year.csv")
    write_csv(themes_by_cluster, out_dir / "themes_by_cluster.csv")
    write_csv(year_table, out_dir / "papers_by_year.csv")
    write_json(run_info, out_dir / "run_info.json")
    report = render_report(
        {
            "sets": sets,
            "papers": papers,
            "summaries": summaries,
            "run_info": run_info,
            "n_background": n_background,
            "year_table": year_table,
            "top_phrases": top_phrases,
            "distinctive_phrases": distinctive,
            "clusters": clusters,
            "themes": theme_table,
            "themes_by_year": themes_by_year,
            "themes_by_cluster": themes_by_cluster,
        }
    )
    write_text(report, out_dir / "report.md")
    log.info("wrote results to %s", out_dir)
    return out_dir
