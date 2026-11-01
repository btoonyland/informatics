"""Render the analysis results as a Markdown report (results/<set>/report.md)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from informatics.themes import THEME_NAMES


def pct(value: float) -> str:
    return "" if pd.isna(value) else f"{100 * value:.1f}%"


def _cell(value: Any) -> str:
    if isinstance(value, float):
        text = f"{value:.2f}"
    elif isinstance(value, (list, tuple)):
        text = ", ".join(map(str, value))
    else:
        text = (
            "" if value is None or (not isinstance(value, str) and pd.isna(value)) else str(value)
        )
    return text.replace("|", "\\|").replace("\n", " ")


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(none)_\n"
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(_cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def render_report(ctx: dict) -> str:
    """``ctx`` holds the tables produced by :func:`informatics.analysis.run_analysis`."""
    papers: pd.DataFrame = ctx["papers"]
    sets: list[str] = ctx["sets"]
    summaries: dict = ctx["summaries"]
    info: dict = ctx["run_info"]
    out = [f"# AI in medicine on arXiv: {' + '.join(sets)}\n"]
    commit = (info.get("git_commit") or "unknown")[:10]
    out.append(
        f"Generated {info['created_utc']} by `informatics analyze` "
        f"(version {info['informatics_version']}, commit {commit}). "
        "All tables are also saved as CSV next to this file.\n"
    )

    out.append("## Papers\n")
    for name in sets:
        summary = summaries.get(name)
        if summary:
            c, cfg = summary["counts"], summary["config"]
            out.append(
                f"- **{name}**: {c['selected']:,} of {c['scanned']:,} papers selected "
                f"({pct(c['selected'] / max(c['scanned'], 1))}), requiring at least "
                f"{cfg['min_ai_hits']} AI and {cfg['min_medicine_hits']} medicine keyword "
                "hit(s) in title + abstract."
            )
    cross = int((papers["corpora"].map(len) > 1).sum())
    out.append(
        f"- Analysed together: **{len(papers):,} papers**; {cross:,} are cross-listed in more "
        "than one set and counted once."
    )
    out.append(
        f"- Comparison set: {ctx['n_background']:,} randomly sampled papers that match the AI "
        "keywords but no medicine keyword."
    )
    out.append(
        "- The keyword rule's precision and recall are unknown until a labelled sample is "
        "evaluated (`informatics label-sample` / `label-evaluate`).\n"
    )  # fmt: skip

    out.append("### Papers per year\n")
    out.append("Years come from the arXiv identifier (month of first submission).\n")
    years = ctx["year_table"].copy()
    keep = ["year", "selected"] + [c for c in years.columns if c.endswith("_selected_share")]
    years = years[keep]
    for col in keep[2:]:
        years[col] = years[col].map(pct)
    years.columns = ["year", "selected"] + [
        f"{c.removesuffix('_selected_share')}: share of all papers" for c in keep[2:]
    ]
    out.append(md_table(years))

    out.append("## Phrases\n")
    out.append(
        "Counts are numbers of papers containing the phrase. Phrases marked * contain a keyword "
        "the papers were selected on, so they are common by construction.\n"
    )
    out.append("### Most common phrases\n")
    top = ctx["top_phrases"].head(30)
    marked = [
        p + (" *" if s else "")
        for p, s in zip(top["phrase"], top["contains_selection_term"], strict=True)
    ]
    out.append(md_table(pd.DataFrame({
        "phrase": marked,
        "papers": top["papers"],
        "share of papers": top["share"].map(pct),
    })))  # fmt: skip
    distinctive = ctx["distinctive_phrases"]
    if distinctive is not None:
        out.append("### Phrases that set these papers apart from AI-only papers\n")
        out.append(
            "Ranked by the z-score of a smoothed log-odds ratio (see `phrases.log_odds`): "
            "phrases much more common here than in AI papers without medicine keywords.\n"
        )
        d = distinctive.head(30)
        out.append(md_table(pd.DataFrame({
            "phrase": d["phrase"],
            "share of these papers": d["share"].map(pct),
            "share of AI-only papers": d["background_share"].map(pct),
            "z": d["z"],
        })))  # fmt: skip

    clusters = ctx["clusters"]
    out.append(f"## Topic clusters (k = {clusters.k})\n")
    if not clusters.k_selection.empty:
        out.append(
            "k was chosen by the highest silhouette score; stability is the adjusted Rand index "
            "between two runs with different seeds (1 = identical clusters).\n"
        )
        out.append(md_table(clusters.k_selection))
    for row in clusters.summary.itertuples(index=False):
        out.append(f"### Cluster {row.cluster}: {row.papers:,} papers ({pct(row.share)})\n")
        out.append(f"Distinctive terms: {', '.join(row.top_terms)}\n")
        out.append("Typical papers:\n")
        out += [f"- {t} ({i})" for t, i in zip(row.example_titles, row.example_ids, strict=True)]
        out.append("")

    out.append("## Themes\n")
    out.append(
        "Share of papers whose title or abstract mentions each theme (keyword match, see "
        "`keywords.THEMES`). `accuracy` appears in most machine-learning papers, so treat it as "
        "a baseline.\n"
    )
    themes = ctx["themes"].copy()
    themes["share"] = themes["share"].map(pct)
    out.append(md_table(themes))
    for title, key, column, min_papers in (
        ("By year (years with at least 20 papers)", "themes_by_year", "year", 20),
        ("By cluster", "themes_by_cluster", "cluster", 0),
    ):
        table = ctx[key]
        table = table[table["papers"] >= min_papers].copy()
        mentioned = [t for t in THEME_NAMES if t in table and table[t].sum() > 0]
        for t in mentioned:
            table[t] = table[t].map(pct)
        out.append(f"### {title}\n")
        out.append(md_table(table[[column, "papers"] + mentioned]))
    return "\n".join(out)
