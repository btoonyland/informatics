"""How often the secondary themes (privacy, bias, regulation, ...) are mentioned."""

from __future__ import annotations

import pandas as pd

from informatics.keywords import THEMES

THEME_NAMES = list(THEMES)


def overview(papers: pd.DataFrame) -> pd.DataFrame:
    """Number and share of papers mentioning each theme."""
    counts = papers["themes"].explode().dropna().value_counts()
    table = pd.DataFrame({"theme": THEME_NAMES})
    table["papers"] = table["theme"].map(counts).fillna(0).astype(int)
    table["share"] = table["papers"] / max(len(papers), 1)
    return table.sort_values(["papers", "theme"], ascending=[False, True]).reset_index(drop=True)


def share_by(papers: pd.DataFrame, column: str) -> pd.DataFrame:
    """One row per value of ``column`` (e.g. year or cluster): its paper count, and the share of
    those papers that mention each theme."""
    data = papers[[column, "themes"]].dropna(subset=[column])
    sizes = data.groupby(column).size().rename("papers")
    exploded = data.explode("themes").dropna(subset=["themes"])
    counts = pd.crosstab(exploded[column], exploded["themes"]) if len(exploded) else None
    table = sizes.to_frame()
    for theme in THEME_NAMES:
        if counts is not None and theme in counts.columns:
            mentioned = counts[theme].reindex(table.index, fill_value=0)
        else:
            mentioned = 0
        table[theme] = mentioned / table["papers"]
    return table.reset_index()
