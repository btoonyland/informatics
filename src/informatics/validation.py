"""Measure the keyword filter with a hand-labelled sample.

1. ``informatics label-sample --set cs`` writes two files:
   ``labels/cs_to_label.csv``: papers in random order with empty ``relevant`` and ``notes``
   columns, and ``labels/cs_key.csv``: which stratum (see ``filtering.py``) each paper came from.
   Labellers only open the first file, so they cannot tell which papers the filter selected.
2. Fill in ``relevant`` with 1 (the paper is about AI in medicine) or 0.
3. ``informatics label-evaluate --set cs`` estimates precision (share of selected papers that are
   relevant) and recall (share of all relevant papers that were selected). Each stratum's
   labelled share is scaled up to that stratum's size in the whole set. Excluded papers
   (withdrawn, not English) are left out on purpose and do not count as misses.

The themes are checked the same way: ``label-themes-sample`` draws papers tagged with each theme,
you mark whether the paper really discusses it, and ``label-themes-evaluate`` reports each
theme's precision. Ambiguous words (bias, regulation, compliance) are the reason to do this.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from informatics import keywords
from informatics.filtering import MISSED_STRATA, STRATA, FilterConfig, annotate, search_text
from informatics.stats import wilson_interval
from informatics.storage import read_parquet

__all__ = [
    "evaluate_labels", "evaluate_theme_labels", "make_label_sample", "make_theme_sample",
    "parse_label", "wilson_interval",
]  # fmt: skip

TRUE = {"1", "y", "yes", "true", "t"}
FALSE = {"0", "n", "no", "false", "f"}


def make_label_sample(
    raw_path: str | Path,
    selected_path: str | Path,
    summary: dict,
    n_selected: int = 100,
    n_per_stratum: int = 50,
    seed: int = 42,
    batch_size: int = 20_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (sheet to label, key) for a random sample stratified by filter outcome."""
    cfg = summary["config"]
    config = FilterConfig(
        min_ai_hits=cfg["min_ai_hits"], min_medicine_hits=cfg["min_medicine_hits"]
    )
    selected = read_parquet(selected_path, columns=["id", "title", "abstract", "duplicate_of"])
    selected_ids = set(selected["id"])
    selected = selected[selected["duplicate_of"].isna()]
    picked = [selected.sample(n=min(n_selected, len(selected)), random_state=seed)]
    picked[0] = picked[0].assign(stratum="selected")

    quotas = {s: n_per_stratum for s in MISSED_STRATA}
    if config.min_ai_hits == 1 and config.min_medicine_hits == 1:
        quotas.pop("below_threshold")  # cannot occur with the default rule
    table = pq.read_table(raw_path, columns=["id", "title", "abstract", "comments"])
    order = np.random.default_rng(seed).permutation(table.num_rows)
    for start in range(0, len(order), batch_size):
        if not any(quotas.values()):
            break
        chunk = table.take(order[start : start + batch_size]).to_pandas()
        chunk = chunk[~chunk["id"].isin(selected_ids)]
        strata = annotate(chunk, config)["stratum"]
        for stratum, remaining in quotas.items():
            if remaining:
                take = chunk[strata == stratum].head(remaining)
                picked.append(take.assign(stratum=stratum))
                quotas[stratum] -= len(take)

    sample = pd.concat(picked, ignore_index=True).sample(frac=1, random_state=seed)
    sheet = sample[["id", "title", "abstract"]].assign(relevant="", notes="")
    key = sample[["id", "stratum"]]
    return sheet.reset_index(drop=True), key.reset_index(drop=True)


def parse_label(value: object) -> bool | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if text in TRUE:
        return True
    if text in FALSE:
        return False
    raise ValueError(f"unrecognised label {value!r}: use 1/0, yes/no or true/false")


def evaluate_labels(
    sheet: pd.DataFrame, key: pd.DataFrame, summary: dict
) -> tuple[pd.DataFrame, dict]:
    """Per-stratum table and overall estimates (precision, recall)."""
    df = key.merge(sheet[["id", "relevant"]], on="id", how="left")
    df["relevant"] = df["relevant"].map(parse_label)
    labelled = df[df["relevant"].notna()]
    sizes = summary["counts"]
    rows = []
    for stratum in STRATA:
        part = labelled[labelled["stratum"] == stratum]
        n, relevant = len(part), int(part["relevant"].astype(bool).sum())
        share = relevant / n if n else math.nan
        low, high = wilson_interval(relevant, n)
        size = int(sizes.get(stratum, 0))
        rows.append(
            {
                "stratum": stratum,
                "papers_in_set": size,
                "labelled": n,
                "relevant": relevant,
                "relevant_share": share,
                "ci_low": low,
                "ci_high": high,
                "estimated_relevant_in_set": share * size if n else math.nan,
            }
        )
    table = pd.DataFrame(rows)
    by_stratum = table.set_index("stratum")
    hits = by_stratum.loc["selected", "estimated_relevant_in_set"]
    others = table[table["stratum"].isin(MISSED_STRATA)]
    misses = others["estimated_relevant_in_set"].sum(skipna=True)
    unlabelled = others[(others["labelled"] == 0) & (others["papers_in_set"] > 0)]["stratum"]
    total = hits + misses
    result = {
        "precision": by_stratum.loc["selected", "relevant_share"],
        "precision_ci": [
            by_stratum.loc["selected", "ci_low"],
            by_stratum.loc["selected", "ci_high"],
        ],
        "recall": hits / total if total > 0 else math.nan,
        "estimated_relevant_selected": hits,
        "estimated_relevant_missed": misses,
        "unlabelled_strata": unlabelled.tolist(),  # recall is an upper bound if any are listed
        "labelled": len(labelled),
        "unlabelled_rows": int(df["relevant"].isna().sum()),
    }
    return table, result


def make_theme_sample(
    selected_path: str | Path, per_theme: int = 30, seed: int = 42
) -> pd.DataFrame:
    """Up to ``per_theme`` papers tagged with each theme, with the words that matched."""
    df = read_parquet(selected_path, columns=["id", "title", "abstract", "themes", "duplicate_of"])
    df = df[df["duplicate_of"].isna()].reset_index(drop=True)
    text = search_text(df)
    parts = []
    for theme in keywords.THEMES:
        tagged = df[df["themes"].map(lambda found, t=theme: t in found)]
        sample = tagged.sample(n=min(per_theme, len(tagged)), random_state=seed)
        matched = text.loc[sample.index].map(
            lambda t, name=theme: ", ".join(keywords.THEME_MATCHER.words(name, t))
        )
        parts.append(sample.assign(theme=theme, matched=matched))
    if not parts or not sum(len(p) for p in parts):
        return pd.DataFrame(
            columns=["theme", "id", "matched", "title", "abstract", "relevant", "notes"]
        )
    sheet = pd.concat(parts, ignore_index=True)
    return sheet[["theme", "id", "matched", "title", "abstract"]].assign(relevant="", notes="")


def evaluate_theme_labels(sheet: pd.DataFrame) -> pd.DataFrame:
    """Precision of each theme's keywords: share of tagged papers that really discuss the theme."""
    df = sheet.assign(relevant=sheet["relevant"].map(parse_label))
    rows = []
    for theme in keywords.THEMES:
        part = df[(df["theme"] == theme) & df["relevant"].notna()]
        n, relevant = len(part), int(part["relevant"].astype(bool).sum())
        low, high = wilson_interval(relevant, n)
        rows.append(
            {
                "theme": theme,
                "labelled": n,
                "relevant": relevant,
                "precision": relevant / n if n else math.nan,
                "ci_low": low,
                "ci_high": high,
            }
        )
    return pd.DataFrame(rows)
