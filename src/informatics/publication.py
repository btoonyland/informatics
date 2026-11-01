"""Publication data for the analysed papers: was each paper published, where, and by whom.

Evidence of publication comes from arXiv's own fields (a journal reference, or comments saying
accepted / to appear / published, see enrich.publication_status) and, after
``informatics enrich``, from OpenAlex: venue type, the types and countries of the authors'
institutions, and citations. OpenAlex shares are among the papers OpenAlex matched.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from informatics.enrich import publication_status
from informatics.stats import wilson_interval

OPENALEX_COLUMNS = [
    "openalex_id", "matched_by", "venue", "venue_type", "field", "institution_types",
    "countries", "cited_by_count",
]  # fmt: skip


def add_publication_data(papers: pd.DataFrame, enriched: list[pd.DataFrame]) -> pd.DataFrame:
    """``publication_status`` from arXiv's fields, plus the OpenAlex columns if enriched."""
    papers = papers.copy()
    comments = papers["comments"] if "comments" in papers else pd.Series(None, index=papers.index)
    papers["publication_status"] = [
        publication_status(j, c) for j, c in zip(papers["journal_ref"], comments, strict=True)
    ]
    if enriched:
        oa = pd.concat(enriched, ignore_index=True).drop_duplicates("id")
        oa = oa[["id", *[c for c in OPENALEX_COLUMNS if c in oa]]]
        papers = papers.merge(oa, on="id", how="left")
        for col in ("institution_types", "countries"):
            papers[col] = papers[col].map(lambda v: v if isinstance(v, list) else [])
    return papers


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
