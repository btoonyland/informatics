"""The interactive report: ``results/<set>/report.html``, one self-contained file.

The page draws its charts in the browser from data embedded in it (report_assets/report.js:
plain SVG, no external libraries or fonts), so it opens offline and can be sent on as an
attachment. Charts: papers per year, the share of each set's papers that were selected, each
theme's share by year with 95% intervals, the trend odds ratios (all papers and within
topics), concepts, rising and falling phrases, and a 2-D map of the topic clusters. Every chart
has a table view, and a table of all papers can be searched and filtered.
"""

from __future__ import annotations

import html
import json
import math
import re
from importlib import resources
from typing import Any
from urllib.parse import quote

import numpy as np
import pandas as pd

from informatics import concepts
from informatics.arxiv_ids import _NEW_STYLE, _OLD_STYLE

MAX_TABLE_PAPERS = 25_000  # papers listed in the page; papers.csv has all of them
TOP_CONCEPTS = 8
TOP_PHRASES = 12


def paper_url(paper_id: str, doi: object = None) -> str | None:
    """Where a paper can be read: arXiv, PubMed or OpenAlex by its ID, else its DOI."""
    if paper_id.startswith("pmid:"):
        return f"https://pubmed.ncbi.nlm.nih.gov/{quote(paper_id[5:])}/"
    if paper_id.startswith("openalex:"):
        return f"https://openalex.org/{quote(paper_id[9:])}"
    if _NEW_STYLE.fullmatch(paper_id) or _OLD_STYLE.fullmatch(paper_id):
        return f"https://arxiv.org/abs/{paper_id}"
    if isinstance(doi, str) and doi:
        return f"https://doi.org/{quote(doi, safe='/')}"
    return None


def _number(value: Any) -> float | int | None:
    """JSON-safe number: NaN and missing values become null."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) or math.isinf(number) else round(number, 6)


def _records(table: pd.DataFrame | None, columns: dict[str, str]) -> list[dict]:
    """Rows of ``table`` as dicts with the given renamed columns (numbers made JSON-safe)."""
    if table is None or table.empty:
        return []
    out = []
    for row in table.to_dict("records"):
        record = {}
        for source, name in columns.items():
            value = row.get(source)
            record[name] = value if isinstance(value, str) else _number(value)
        out.append(record)
    return out


def _years_shown(by_year: pd.DataFrame, min_papers: int) -> list[int]:
    sizes = by_year.drop_duplicates("year").set_index("year")["papers_in_group"]
    years = [int(y) for y, n in sizes.items() if n >= min_papers]
    return years if len(years) >= 3 else [int(y) for y in sizes.index]


def build_payload(ctx: dict, min_year_papers: int = 20) -> dict:
    """Everything the page draws, from the tables :func:`analysis.run_analysis` made."""
    info, papers, sets = ctx["run_info"], ctx["papers"], ctx["sets"]
    counts = info["counts"]
    summaries = ctx["summaries"]

    year_table = ctx["year_table"]
    shares = {
        name: [_number(v) for v in year_table[f"{name}_selected_share"]]
        for name in sets
        if f"{name}_selected_share" in year_table
    }

    theme_years = ctx["theme_tables"]["year"]
    overall = ctx["theme_tables"]["overall"]
    shown_years = _years_shown(theme_years, min_year_papers) if len(theme_years) else []
    theme_panels = []
    for tag in overall.loc[overall["papers"] > 0, "tag"]:
        part = theme_years[(theme_years["tag"] == tag) & theme_years["year"].isin(shown_years)]
        part = part.sort_values("year")
        theme_panels.append(
            {
                "tag": tag,
                "years": [int(y) for y in part["year"]],
                "share": [_number(v) for v in part["share"]],
                "low": [_number(v) for v in part["ci_low"]],
                "high": [_number(v) for v in part["ci_high"]],
                "papers": [int(v) for v in part["papers"]],
                "n": [int(v) for v in part["papers_in_group"]],
            }
        )

    trend_columns = {
        "tag": "tag", "papers": "papers", "or_per_year": "or", "ci_low": "low",
        "ci_high": "high", "adjusted_or_per_year": "adj", "adjusted_ci_low": "adj_low",
        "adjusted_ci_high": "adj_high", "adjusted_pp_per_year": "pp",
        "adjusted_q_value": "q",
    }  # fmt: skip
    trends = ctx["theme_trends"]
    trends = trends[trends["adjusted_or_per_year"].notna()].sort_values("adjusted_or_per_year")

    concept_overall = ctx["concept_tables"]["overall"]
    concept_panels = []
    for dimension in concepts.CONCEPTS:
        part = concept_overall[
            (concept_overall["dimension"] == dimension) & (concept_overall["papers"] > 0)
        ]
        part = part.sort_values("share", ascending=False).head(TOP_CONCEPTS)
        concept_panels.append(
            {
                "dimension": dimension,
                "rows": _records(
                    part,
                    {
                        "tag": "tag",
                        "papers": "papers",
                        "share": "share",
                        "ci_low": "low",
                        "ci_high": "high",
                    },
                ),  # fmt: skip
            }
        )

    phrase_columns = {
        "phrase": "phrase", "recent_share": "recent", "earlier_share": "earlier", "z": "z",
    }  # fmt: skip
    periods = ctx.get("periods")
    phrases = None
    if periods and ctx.get("rising_phrases") is not None:
        rising = ctx["rising_phrases"].head(TOP_PHRASES)
        falling = ctx["falling_phrases"].head(TOP_PHRASES)
        phrases = {
            "earlier": list(periods[0]),
            "recent": list(periods[1]),
            "rows": _records(rising, phrase_columns) + _records(falling, phrase_columns),
        }

    clusters = ctx["clusters"]
    cluster_rows = [
        {
            "cluster": int(row.cluster),
            "papers": int(row.papers),
            "share": _number(row.share),
            "terms": list(row.top_terms)[:6],
        }
        for row in clusters.summary.itertuples(index=False)
    ]
    cluster_map = ctx.get("cluster_map")

    # The paper table (theme names as indices to keep the page small).
    theme_names = list(overall["tag"])
    theme_index = {name: i for i, name in enumerate(theme_names)}
    listed = papers
    if len(listed) > MAX_TABLE_PAPERS:
        listed = listed.sample(n=MAX_TABLE_PAPERS, random_state=0)
    listed = listed.sort_values(["year", "id"], ascending=[False, True], na_position="last")
    status = listed.get("publication_status")
    doi = listed["doi"] if "doi" in listed else pd.Series(None, index=listed.index)
    rows = []
    for i, (row_index, row) in enumerate(listed.iterrows()):
        rows.append(
            [
                row["id"],
                row["title"] if isinstance(row["title"], str) else "",
                _number(row["year"]),
                int(row["cluster"]),
                [theme_index[t] for t in row["themes"] if t in theme_index],
                status.iloc[i] if status is not None else None,
                paper_url(row["id"], doi.loc[row_index]),
                "+".join(row["corpora"]),
            ]
        )
    row_of = {paper_id: n for n, paper_id in enumerate(listed["id"])}
    map_points = None
    if cluster_map is not None and len(cluster_map):
        map_points = {
            "x": [round(float(v), 3) for v in cluster_map["x"]],
            "y": [round(float(v), 3) for v in cluster_map["y"]],
            "cluster": [int(v) for v in cluster_map["cluster"]],
            "row": [row_of.get(i) for i in cluster_map["id"]],
        }

    selections = ctx.get("selections", {})
    return {
        "title": f"AI in medicine: {' + '.join(sets)}",
        "generated": info["created_utc"],
        "version": info["informatics_version"],
        "commit": (info.get("git_commit") or "unknown")[:10],
        "sets": sets,
        "kpis": {
            "papers": counts["papers"],
            "candidates": sum(
                (summaries.get(n) or {}).get("counts", {}).get("candidates", 0) for n in sets
            ),
            "clusters": counts["k"],
            "background": counts["background"],
        },
        "selection": [
            {
                "set": name,
                "description": s["description"],
                "selected": s["selected"],
                "candidates": s["candidates"],
            }
            for name, s in selections.items()
        ],  # fmt: skip
        "themesSource": info["config"].get("themes", "keywords"),
        "years": {
            "year": [int(y) for y in year_table["year"]],
            "selected": [int(v) for v in year_table["selected"]],
            "share": shares,
        },
        "themes": theme_panels,
        "trends": _records(trends, trend_columns),
        "adjustedFor": "topic cluster" + (" and set" if len(sets) > 1 else ""),
        "concepts": concept_panels,
        "phrases": phrases,
        "clusters": cluster_rows,
        "map": map_points,
        "paperColumns": ["id", "title", "year", "cluster", "themes", "status", "url", "sets"],
        "papers": rows,
        "papersTotal": len(papers),
        "themeNames": theme_names,
    }


def _asset(name: str) -> str:
    return resources.files("informatics").joinpath("report_assets", name).read_text("utf-8")


def render_html(ctx: dict, min_year_papers: int = 20) -> str:
    payload = build_payload(ctx, min_year_papers)
    # No "<" at all inside the <script> element: "</script>" would end it early and "<!--" or
    # "<script" would change how the browser reads it. JSON allows the escaped form \u003c.
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    page = _asset("report.html")
    page = page.replace("/*CSS*/", _asset("report.css")).replace("/*JS*/", _asset("report.js"))
    page = re.sub(r"\{\{TITLE\}\}", lambda _: html.escape(payload["title"]), page)
    return page.replace("{{DATA}}", data, 1)
