"""Shares of papers carrying each tag (themes, concepts), overall and per group, with 95% intervals.

Tables are in long format (one row per group and tag), ready for plotting or filtering; the
report pivots them for display. Intervals are Wilson score intervals, which stay sensible for the
small groups that early years and small clusters produce.
"""

from __future__ import annotations

import math
from collections import Counter

import pandas as pd

from informatics.stats import wilson_interval


def _counts(tag_lists: pd.Series) -> Counter[str]:
    return Counter(tag for tags in tag_lists for tag in set(tags))


def _row(tag: str, mentioning: int, n: int) -> dict:
    low, high = wilson_interval(mentioning, n)
    return {
        "tag": tag,
        "papers": mentioning,
        "share": mentioning / n if n else math.nan,
        "ci_low": low,
        "ci_high": high,
    }


def overview(papers: pd.DataFrame, column: str, names: list[str]) -> pd.DataFrame:
    """Number and share of papers carrying each tag, most common first."""
    counts, n = _counts(papers[column]), len(papers)
    table = pd.DataFrame([_row(tag, counts.get(tag, 0), n) for tag in names])
    return table.sort_values(["papers", "tag"], ascending=[False, True]).reset_index(drop=True)


def shares_by(papers: pd.DataFrame, column: str, names: list[str], by: str) -> pd.DataFrame:
    """For each value of ``by`` (year, cluster, corpus ...): its size and each tag's share."""
    rows = []
    data = papers[[by, column]].dropna(subset=[by])
    for value, group in data.groupby(by, sort=True):
        counts, n = _counts(group[column]), len(group)
        for tag in names:
            rows.append({by: value, "papers_in_group": n, **_row(tag, counts.get(tag, 0), n)})
    columns = [by, "papers_in_group", "tag", "papers", "share", "ci_low", "ci_high"]
    return pd.DataFrame(rows, columns=columns)


def pivot_shares(
    table: pd.DataFrame, by: str, tags: list[str], with_ci: bool = True
) -> pd.DataFrame:
    """Wide display table: one row per group, one "share (low-high)" cell per tag."""

    def cell(row: pd.Series) -> str:
        share = f"{100 * row['share']:.1f}%"
        if not with_ci or math.isnan(row["ci_low"]):
            return share
        return f"{share} ({100 * row['ci_low']:.0f}-{100 * row['ci_high']:.0f})"

    subset = table[table["tag"].isin(tags)].assign(cell=lambda t: t.apply(cell, axis=1))
    wide = subset.pivot(index=by, columns="tag", values="cell").reindex(columns=tags)
    sizes = table.drop_duplicates(by).set_index(by)["papers_in_group"]
    wide.insert(0, "papers", sizes)
    return wide.reset_index()
