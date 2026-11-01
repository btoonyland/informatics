"""Render the analysis results as a Markdown report (results/<set>/report.md)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from informatics import sources, tags


def pct(value: float) -> str:
    return "" if pd.isna(value) else f"{100 * value:.1f}%"


def pct_ci(share: float, low: float, high: float) -> str:
    if pd.isna(share):
        return ""
    return (
        f"{100 * share:.1f}% ({100 * low:.0f}-{100 * high:.0f})" if not pd.isna(low) else pct(share)
    )


def _cell(value: Any) -> str:
    if isinstance(value, float):
        text = "" if pd.isna(value) else f"{value:.2f}"
    elif isinstance(value, (list, tuple)):
        text = ", ".join(map(str, value))
    else:
        text = (
            "" if value is None or (not isinstance(value, str) and pd.isna(value)) else str(value)
        )
    return text.replace("|", "\\|").replace("\n", " ")


def md_table(df: pd.DataFrame | None) -> str:
    if df is None or df.empty:
        return "_(none)_\n"
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(_cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def _phrase_table(table: pd.DataFrame, columns: dict[str, str], n: int = 25) -> str:
    """Selected, renamed columns; shares as percentages; selection-keyword phrases marked *."""
    t = table.head(n)
    marked = zip(t["phrase"], t["contains_selection_term"], strict=True)
    out = pd.DataFrame({"phrase": [p + (" *" if s else "") for p, s in marked]})
    for source, label in columns.items():
        out[label] = t[source].map(pct).to_numpy() if "share" in source else t[source].to_numpy()
    return md_table(out)


def _tag_overview(table: pd.DataFrame, label: str) -> str:
    shown = table[table["papers"] > 0]
    rows = zip(shown["share"], shown["ci_low"], shown["ci_high"], strict=True)
    return md_table(
        pd.DataFrame(
            {
                label: shown["tag"],
                "papers": shown["papers"],
                "share (95% interval)": [pct_ci(s, lo, hi) for s, lo, hi in rows],
            }
        )
    )


def _top_tags(table: pd.DataFrame, n: int) -> list[str]:
    return table[table["papers"] > 0].head(n)["tag"].tolist()


def _or_ci(value: float, low: float, high: float) -> str:
    return "" if pd.isna(value) else f"{value:.2f} ({low:.2f}-{high:.2f})"


def _trend_table(table: pd.DataFrame, label: str, with_dimension: bool = False) -> str:
    shown = table[table["adjusted_or_per_year"].notna()]
    out = pd.DataFrame(
        {
            label: shown["tag"],
            "papers": shown["papers"],
            "odds ratio per year": [
                _or_ci(v, lo, hi)
                for v, lo, hi in zip(
                    shown["or_per_year"], shown["ci_low"], shown["ci_high"], strict=True
                )
            ],
            "within topics": [
                _or_ci(v, lo, hi)
                for v, lo, hi in zip(
                    shown["adjusted_or_per_year"],
                    shown["adjusted_ci_low"],
                    shown["adjusted_ci_high"],
                    strict=True,
                )
            ],
            "points per year": shown["adjusted_pp_per_year"].map(
                lambda v: "" if pd.isna(v) else f"{v:+.2f}"
            ),
            "q": shown["adjusted_q_value"].map(lambda v: "" if pd.isna(v) else f"{v:.3f}"),
        }
    )
    if with_dimension:
        out.insert(0, "dimension", shown["dimension"].to_numpy())
    return md_table(out)


def _trend_section(table: pd.DataFrame, label: str, sets: list[str]) -> str:
    estimated = table[table["adjusted_or_per_year"].notna()]
    if estimated.empty:
        return (
            "### Trends\n\nToo few papers per year for trend estimates (see "
            "`--trend-min-papers` / `--trend-min-year-papers`).\n"
        )
    adjusted = "topic cluster" + (" and set" if len(sets) > 1 else "")
    years = estimated["years"].iloc[0]
    intro = (
        f"### Trends ({years})\n\nLogistic regressions of whether a paper has the {label} on "
        "its publication date (`trends.py`). An odds ratio above 1 means the share grows each "
        f"year. *Within topics* adjusts for the {adjusted}: when it is much closer to 1 than "
        "the plain odds ratio, the change comes from some topics growing faster than others, "
        "not from papers within a topic changing. *Points per year* is the within-topic change "
        "in percentage points; *q* is the false discovery rate across all rows.\n"
    )
    return intro + "\n" + _trend_table(estimated.sort_values("adjusted_or_per_year"), label)


def render_report(ctx: dict) -> str:
    """``ctx`` holds the tables produced by :func:`informatics.analysis.run_analysis`."""
    sets: list[str] = ctx["sets"]
    summaries: dict = ctx["summaries"]
    info: dict = ctx["run_info"]
    counts = info["counts"]
    out = [f"# AI in medicine on arXiv: {' + '.join(sets)}\n"]
    commit = (info.get("git_commit") or "unknown")[:10]
    out.append(
        f"Generated {info['created_utc']} by `informatics analyze` (version "
        f"{info['informatics_version']}, commit {commit}). Every table is also saved as CSV "
        "next to this file; shares come with 95% intervals where noted.\n"
    )

    # Papers ----------------------------------------------------------------------------------
    out.append("## Papers\n")
    selections = ctx.get("selections", {})
    for name in sets:
        summary, selection = summaries.get(name), selections.get(name)
        if summary:
            c = summary["counts"]
            out.append(
                f"- **{name}**: {c['candidates']:,} of {c['scanned']:,} papers are candidates: "
                "their title or abstract has an AI term and a medicine term from the wide "
                f"keyword lists ({c['strict']:,} of them meet the original keyword rule). "
                f"Excluded first: {c.get('excluded_withdrawn', 0):,} withdrawn, "
                f"{c.get('excluded_non_english', 0):,} not in English."
            )
        if selection:
            out.append(
                f"  {selection['selected']:,} of the {selection['candidates']:,} distinct "
                f"candidates were selected by {selection['description']}."
            )
    out.append(
        f"- Analysed together: **{counts['papers']:,} papers** after dropping "
        f"{counts['near_duplicates_dropped']:,} near-duplicates (the same paper under another ID); "
        f"{counts['cross_listed']:,} are cross-listed in more than one set and counted once."
    )
    out.append(
        f"- Comparison set: {ctx['n_background']:,} randomly sampled papers that match the AI "
        "keywords but no medicine keyword."
    )
    out.append(
        "- How many selected papers are really about AI in medicine, and how many relevant "
        "papers were missed, is only known from a hand-labelled sample: `informatics "
        "label-sample` / `label-evaluate` measure the keyword rule, the wide net and the "
        "classifier; `label-themes-sample` / `label-themes-evaluate` measure the themes.\n"
    )
    out.append("### Population and sources\n")
    out.append(
        "What each set is a sample of, and what its dates mean. Compare sets with this in view: "
        "a capped sample per year cannot show growth in the literature.\n"
    )
    for name in sets:
        out += [f"- {line}" for line in sources.describe(name, summaries.get(name))]
    out.append("")
    out.append("### Papers per year\n")
    years = ctx["year_table"].copy()
    keep = ["year", "selected"] + [c for c in years.columns if c.endswith("_selected_share")]
    years = years[keep]
    for col in keep[2:]:
        years[col] = years[col].map(pct)
    years.columns = ["year", "selected"] + [
        f"{c.removesuffix('_selected_share')}: share of all papers" for c in keep[2:]
    ]
    out.append(md_table(years))

    # Phrases ---------------------------------------------------------------------------------
    out.append("## Phrases\n")
    out.append(
        "Counts are numbers of papers. Before counting, texts are normalised (normalize.py): TeX "
        "and special characters are cleaned, acronyms defined in an abstract are replaced by "
        "their long form, British spellings become American, words are lemmatised and compounds "
        'unified, so "fine-tuned", "fine tuning" and "finetuning" count as one phrase. Each '
        "phrase is shown in its most common written form. Phrases marked * contain a keyword the "
        "papers were selected on, so they are common by construction.\n"
    )
    out.append("### Most common phrases\n")
    out.append(_phrase_table(ctx["top_phrases"], {"papers": "papers", "share": "share of papers"}))
    if ctx["distinctive_phrases"] is not None:
        out.append("### Phrases that set these papers apart from AI-only papers\n")
        out.append(
            "Ranked by the z-score of a smoothed log-odds ratio (`phrases.log_odds`): phrases "
            "much more common here than in AI papers without medicine keywords.\n"
        )
        labels = {
            "share": "share of these papers",
            "background_share": "share of AI-only papers",
            "z": "z",
        }
        out.append(_phrase_table(ctx["distinctive_phrases"], labels))
    if ctx["periods"] and ctx["rising_phrases"] is not None:
        (e0, e1), (r0, r1) = ctx["periods"]
        columns = {"recent_share": f"{r0}-{r1}", "earlier_share": f"{e0}-{e1}", "z": "z"}
        out.append(f"### Rising phrases ({r0}-{r1} compared with {e0}-{e1})\n")
        out.append(_phrase_table(ctx["rising_phrases"], columns))
        out.append(f"### Falling phrases ({r0}-{r1} compared with {e0}-{e1})\n")
        out.append(_phrase_table(ctx["falling_phrases"], columns))
    if ctx["phrases_between_sets"] is not None:
        a, b = sets[:2]
        out.append(f"### Phrases that differ between {a} and {b}\n")
        out.append(
            f"Papers only in {a} compared with papers only in {b}; positive z: more common in {a}."
            " The full list is in phrases_between_sets.csv.\n"
        )
        table = ctx["phrases_between_sets"]
        cols = {f"{a}_share": a, f"{b}_share": b, "z": "z"}
        out.append(f"More common in {a}:\n")
        out.append(_phrase_table(table[table["z"] > 0], cols, 15))
        out.append(f"More common in {b}:\n")
        out.append(_phrase_table(table[table["z"] < 0].sort_values("z"), cols, 15))

    # Clusters --------------------------------------------------------------------------------
    clusters = ctx["clusters"]
    model = info["config"].get("embeddings")
    method = f"sentence embeddings ({model})" if model else "LSA of TF-IDF terms"
    out.append(f"## Topic clusters (k = {clusters.k}, clustered on {method})\n")
    if not clusters.k_selection.empty:
        out.append(
            "k was chosen by the highest silhouette score; stability is the adjusted Rand index "
            "between two runs with different seeds (1 = identical clusters).\n"
        )
        out.append(md_table(clusters.k_selection))
    by_cluster = ctx["concept_tables"]["cluster"]
    for row in clusters.summary.itertuples(index=False):
        out.append(f"### Cluster {row.cluster}: {row.papers:,} papers ({pct(row.share)})\n")
        out.append(f"Distinctive terms: {', '.join(row.top_terms)}\n")
        mine = by_cluster[(by_cluster["cluster"] == row.cluster) & (by_cluster["papers"] > 0)]
        leading = [
            f"{part.iloc[0]['tag']} ({pct(part.iloc[0]['share'])})"
            for _, part in mine.sort_values("share", ascending=False).groupby(
                "dimension", sort=False
            )
        ]
        if leading:
            out.append(f"Most common concepts: {', '.join(leading)}\n")
        out.append("Typical papers:\n")
        out += [f"- {t} ({i})" for t, i in zip(row.example_titles, row.example_ids, strict=True)]
        out.append("")

    # Themes and concepts --------------------------------------------------------------------
    themes = ctx["theme_tables"]
    out.append("## Themes\n")
    if info["config"].get("themes") == "llm":
        out.append(
            "Share of papers that discuss each theme, as judged by a language model reading the "
            "title and abstract against written definitions (`llm.THEME_DEFINITIONS`). "
            "`accuracy` applies to most papers that evaluate a model, so treat it as a "
            "baseline.\n"
        )
    else:
        out.append(
            "Share of papers whose title or abstract mentions each theme (keyword match with "
            "exclusions, `keywords.THEMES`). `accuracy` appears in most machine-learning papers, "
            "so treat it as a baseline.\n"
        )
    out.append(_tag_overview(themes["overall"], "theme"))
    top_themes = _top_tags(themes["overall"], 6)
    by_year = themes["year"][themes["year"]["papers_in_group"] >= 20]
    out.append("### Themes by year (years with at least 20 papers; 95% intervals)\n")
    out.append(md_table(tags.pivot_shares(by_year, "year", top_themes) if len(by_year) else None))
    out.append(_trend_section(ctx["theme_trends"], "theme", info["sets"]))
    out.append("### Themes by cluster\n")
    out.append(md_table(tags.pivot_shares(themes["cluster"], "cluster", top_themes, with_ci=False)))
    if "corpus_group" in themes:
        out.append("### Themes by set\n")
        out.append(md_table(tags.pivot_shares(themes["corpus_group"], "corpus_group", top_themes)))

    concepts = ctx["concept_tables"]
    out.append("## Concepts\n")
    moving = ctx["concept_trends"]
    moving = moving[moving["adjusted_q_value"] < 0.05].sort_values("adjusted_or_per_year")
    if len(moving):
        out.append(
            "Concepts whose share changed within topics (false discovery rate below 5%; full "
            "table in concept_trends.csv):\n"
        )
        out.append(_trend_table(moving, "concept", with_dimension=True))
    out.append(
        "What the papers are about, along four dimensions (curated patterns in concepts.py; a "
        "paper can have several concepts or none).\n"
    )
    for dimension, part in concepts["overall"].groupby("dimension", sort=False):
        out.append(f"### {dimension[0].upper()}{dimension[1:]}\n")
        out.append(_tag_overview(part, "concept"))
        leading = _top_tags(part, 5)
        part_years = concepts["year"][(concepts["year"]["dimension"] == dimension)
                                      & (concepts["year"]["papers_in_group"] >= 20)]  # fmt: skip
        if len(part_years) and leading:
            out.append("By year (years with at least 20 papers):\n")
            out.append(md_table(tags.pivot_shares(part_years, "year", leading, with_ci=False)))
        if "corpus_group" in concepts:
            part_sets = concepts["corpus_group"][concepts["corpus_group"]["dimension"] == dimension]
            out.append("By set:\n")
            out.append(md_table(tags.pivot_shares(part_sets, "corpus_group", leading)))

    # Publication -----------------------------------------------------------------------------
    publication = ctx["publication"]
    out.append("## Publication\n")
    out.append(
        "Evidence that a paper was published, from arXiv's own fields: a journal reference, or "
        "comments saying accepted / to appear / published. Absence of evidence is not evidence "
        "that a paper was never published.\n"
    )
    if len(publication):
        table = pd.DataFrame({
            "year": publication["year"],
            "papers": publication["papers"],
            "journal reference": publication["share_journal_ref"].map(pct),
            "accepted (comments)": publication["share_accepted_comment"].map(pct),
            "any evidence": [pct_ci(s, lo, hi) for s, lo, hi in zip(
                publication["share_any_evidence"], publication["any_evidence_ci_low"],
                publication["any_evidence_ci_high"], strict=True)],
        })  # fmt: skip
        out.append(md_table(table))
    if "share_in_openalex" in publication:
        out.append("### From OpenAlex\n")
        out.append(
            f"{counts['openalex_matched']:,} papers were found in OpenAlex. Shares below are "
            "among those papers: published in a journal or conference, with at least one author "
            "at a healthcare institution or at a company.\n"
        )
        oa = pd.DataFrame(
            {"year": publication["year"], "in OpenAlex": publication["share_in_openalex"].map(pct)}
        )
        for name, label in (("journal_or_conference", "journal or conference"),
                            ("healthcare_author", "healthcare author"),
                            ("company_author", "company author")):  # fmt: skip
            oa[label] = [pct_ci(s, lo, hi) for s, lo, hi in zip(
                publication[f"share_{name}"], publication[f"{name}_ci_low"],
                publication[f"{name}_ci_high"], strict=True)]  # fmt: skip
        oa["median citations"] = publication["median_citations"]
        out.append(md_table(oa))
        if ctx["countries"] is not None:
            out.append("Countries of the authors' institutions (top 15):\n")
            top = ctx["countries"].head(15)
            out.append(md_table(top.assign(share=top["share"].map(pct))))
    return "\n".join(out)
