"""Measure each selection stage, and the themes, with hand-labelled samples.

1. ``informatics label-sample --set cs`` writes ``labels/cs_to_label.csv``, papers in random
   order with empty ``relevant`` and ``notes`` columns, and ``labels/cs_key.csv``, the stratum
   (see filtering.py) each paper was drawn from: candidates that meet the keyword rule
   (``strict``), the other candidates (``wide_only``), and papers outside the wide net
   (``ai_only``, ``medicine_only``, ``neither``). Labellers only open the first file, so they
   cannot tell how the filter treated a paper. ``--extend`` adds papers to an existing sheet
   without touching the labels already in it.
2. Fill in ``relevant`` with 1 (the paper is about AI in medicine) or 0. A second labeller can
   fill in ``relevant_2`` for some of the papers, to measure how well labellers agree.
3. ``informatics label-evaluate --set cs`` estimates precision and recall of the keyword rule,
   of the wide net, and of the classifier or language model once ``informatics classify`` has
   run. Each stratum's labelled papers are scaled up to the stratum's size in the whole set;
   95% intervals for the scaled estimates come from a bootstrap within strata. Excluded papers
   (withdrawn, not English) are left out on purpose and do not count as misses.

Themes are checked in two ways. ``label-themes-sample`` draws papers tagged with each theme and
``label-themes-evaluate`` reports each theme's precision (ambiguous words such as bias,
regulation and compliance are the reason). ``label-themes-sample --random N`` instead draws N
selected papers at random; listing every theme each one discusses measures recall as well as
precision, for the keyword themes and for the language model's themes alike.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from informatics import keywords
from informatics.filtering import (
    CANDIDATE_STRATA,
    OUTSIDE_STRATA,
    FilterConfig,
    annotate,
    search_text,
)
from informatics.stats import cohen_kappa, wilson_interval
from informatics.storage import read_parquet

__all__ = [
    "evaluate_labels", "evaluate_theme_audit", "evaluate_theme_labels", "make_label_sample",
    "make_random_theme_sample", "make_theme_sample", "parse_label", "parse_themes",
    "read_labels", "wilson_interval",
]  # fmt: skip

TRUE = {"1", "y", "yes", "true", "t"}
FALSE = {"0", "n", "no", "false", "f"}
LABELLED_STRATA = CANDIDATE_STRATA + OUTSIDE_STRATA


def _check_keywords(summary: dict) -> None:
    if summary.get("keywords") != keywords.keyword_lists():
        raise ValueError(
            "keywords.py has changed since `informatics filter` ran; run the filter again so "
            "the strata match the current keywords"
        )


def make_label_sample(
    raw_path: str | Path,
    candidates_path: str | Path,
    summary: dict,
    n_strict: int = 150,
    n_wide: int = 150,
    n_per_stratum: int = 50,
    seed: int = 42,
    exclude_ids: Collection[str] = (),
    batch_size: int = 20_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (sheet to label, key) for a random sample stratified by filter outcome.

    ``exclude_ids`` are papers already in a labelling sheet (``label-sample --extend``)."""
    _check_keywords(summary)
    cfg = summary["config"]
    config = FilterConfig(
        min_ai_hits=cfg["min_ai_hits"], min_medicine_hits=cfg["min_medicine_hits"]
    )
    exclude = set(exclude_ids)
    candidates = read_parquet(
        candidates_path, columns=["id", "title", "abstract", "keyword_rule", "duplicate_of"]
    )
    candidate_ids = set(candidates["id"])
    pool = candidates[candidates["duplicate_of"].isna() & ~candidates["id"].isin(exclude)]
    picked = []
    for stratum, rule, n in (("strict", True, n_strict), ("wide_only", False, n_wide)):
        part = pool[pool["keyword_rule"] == rule]
        picked.append(part.sample(n=min(n, len(part)), random_state=seed).assign(stratum=stratum))

    quotas = {s: n_per_stratum for s in OUTSIDE_STRATA}
    table = pq.read_table(raw_path, columns=["id", "title", "abstract", "comments"])
    order = np.random.default_rng(seed).permutation(table.num_rows)
    for start in range(0, len(order), batch_size):
        if not any(quotas.values()):
            break
        chunk = table.take(order[start : start + batch_size]).to_pandas()
        chunk = chunk[~chunk["id"].isin(candidate_ids) & ~chunk["id"].isin(exclude)]
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


def read_labels(sheet: pd.DataFrame, key: pd.DataFrame) -> pd.DataFrame:
    """One row per sampled paper: id, stratum, relevant (True/False/None) and, if a second
    labeller filled it in, relevant_2."""
    columns = ["id", "relevant"] + (["relevant_2"] if "relevant_2" in sheet else [])
    df = key[["id", "stratum"]].merge(sheet[columns], on="id", how="left")
    for column in columns[1:]:
        df[column] = df[column].map(parse_label).astype(object)
    return df


def _bootstrap_rows(n: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, n, size=(n_boot, n)) if n else np.zeros((n_boot, 0), dtype=int)


def _ci(values: np.ndarray) -> list[float]:
    values = values[np.isfinite(values)]
    if not values.size:
        return [math.nan, math.nan]
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def _ratio(numerator: np.ndarray | float, denominator: np.ndarray | float) -> np.ndarray:
    num, den = np.asarray(numerator, dtype=float), np.asarray(denominator, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


def evaluate_labels(
    sheet: pd.DataFrame,
    key: pd.DataFrame,
    summary: dict,
    decisions: Mapping[str, pd.DataFrame] | None = None,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict]:
    """Per-stratum table and estimates for the keyword rule, the wide net and each method in
    ``decisions`` (name -> table with ``id`` and ``predicted``; e.g. the trained classifier
    and the language model). A method is measured on the labelled candidates it decided.

    Every estimate is a stratified one: a labelled paper stands for ``size / labelled`` papers
    of its stratum. Intervals for the scaled estimates come from resampling the labelled papers
    within strata.
    """
    df = read_labels(sheet, key)
    labelled = df[df["relevant"].notna()]
    sizes = summary["counts"]
    rows = []
    for stratum in LABELLED_STRATA:
        part = labelled[labelled["stratum"] == stratum]
        n, relevant = len(part), int(part["relevant"].astype(bool).sum())
        low, high = wilson_interval(relevant, n)
        size = int(sizes.get(stratum, 0))
        rows.append(
            {
                "stratum": stratum,
                "papers_in_set": size,
                "labelled": n,
                "relevant": relevant,
                "relevant_share": relevant / n if n else math.nan,
                "ci_low": low,
                "ci_high": high,
                "estimated_relevant_in_set": relevant / n * size if n else math.nan,
            }
        )
    table = pd.DataFrame(rows)

    predictions = {
        name: labelled["id"].map(table.set_index("id")["predicted"])
        for name, table in (decisions or {}).items()
    }
    rng = np.random.default_rng(seed)
    # Per stratum: bootstrap draws of the estimated relevant papers and, for candidates, of each
    # method's true positives, false positives and false negatives.
    estimated: dict[str, np.ndarray] = {}
    point: dict[str, float] = {}
    counts: dict[str, dict[str, dict[str, np.ndarray]]] = {name: {} for name in predictions}
    for stratum in LABELLED_STRATA:
        part = (labelled["stratum"] == stratum).to_numpy()
        rel = labelled.loc[part, "relevant"].astype(bool).to_numpy()
        size, n = int(sizes.get(stratum, 0)), len(rel)
        draws = _bootstrap_rows(n, n_boot, rng)
        weight = size / n if n else 0.0
        estimated[stratum] = rel[draws].sum(axis=1) * weight
        point[stratum] = rel.sum() * weight
        if stratum not in CANDIDATE_STRATA:
            continue
        for name, predicted in predictions.items():
            pred = predicted[part]
            known = pred.notna().to_numpy()
            if not known.any():
                continue
            pred_b = pred.fillna(False).astype(bool).to_numpy()
            outcome = {
                "tp": pred_b & rel & known,
                "fp": pred_b & ~rel & known,
                "fn": ~pred_b & rel & known,
            }
            # Only labelled papers the method decided are compared, scaled to the stratum.
            known_draws = known[draws].sum(axis=1)
            scale = np.append(size / known.sum(), size / np.maximum(known_draws, 1))
            counts[name][stratum] = {
                k: np.append(v.sum(), v[draws].sum(axis=1)) * scale for k, v in outcome.items()
            }

    def both(values: dict[str, np.ndarray], points: dict[str, float], strata) -> np.ndarray:
        """Point estimate followed by the bootstrap draws, summed over ``strata``."""
        return sum(np.append(points[s], values[s]) for s in strata)

    total = both(estimated, point, LABELLED_STRATA)
    strict = both(estimated, point, ("strict",))
    wide = both(estimated, point, CANDIDATE_STRATA)
    keyword_recall = _ratio(strict, total)
    wide_recall = _ratio(wide, total)
    strict_row = table.set_index("stratum").loc["strict"]
    candidates = labelled[labelled["stratum"].isin(CANDIDATE_STRATA)]
    n_candidates = int(sizes.get("strict", 0)) + int(sizes.get("wide_only", 0))
    unlabelled = table[(table["labelled"] == 0) & (table["papers_in_set"] > 0)]["stratum"]
    result = {
        "keyword_rule": {
            "precision": strict_row["relevant_share"],
            "precision_ci": [strict_row["ci_low"], strict_row["ci_high"]],
            "recall": float(keyword_recall[0]),
            "recall_ci": _ci(keyword_recall[1:]),
            "kappa": cohen_kappa(
                candidates["stratum"] == "strict", candidates["relevant"].astype(bool)
            ),
        },
        "wide_net": {
            "candidates": n_candidates,
            "precision": float(_ratio(wide[0], n_candidates)),
            "recall": float(wide_recall[0]),
            "recall_ci": _ci(wide_recall[1:]),
        },
        "classifiers": {},
        "labellers": _labeller_agreement(labelled),
        "estimated_relevant_in_set": float(total[0]),
        "labelled": len(labelled),
        "unlabelled_rows": int(df["relevant"].isna().sum()),
        "unlabelled_strata": unlabelled.tolist(),  # recall is an upper bound if any are listed
    }
    outside = both(estimated, point, OUTSIDE_STRATA)
    for name, by_stratum in counts.items():
        if not by_stratum:
            continue
        tp = sum(c["tp"] for c in by_stratum.values())
        fp = sum(c["fp"] for c in by_stratum.values())
        fn = sum(c["fn"] for c in by_stratum.values())
        # A candidate stratum in which the method decided no labelled paper is left out.
        precision, recall = _ratio(tp, tp + fp), _ratio(tp, tp + fn + outside)
        predicted = predictions[name]
        compared = candidates[predicted[candidates.index].notna().to_numpy()]
        result["classifiers"][name] = {
            "precision": float(precision[0]),
            "precision_ci": _ci(precision[1:]),
            "recall": float(recall[0]),
            "recall_ci": _ci(recall[1:]),
            "recall_within_candidates": float(_ratio(tp[0], tp[0] + fn[0])),
            "kappa": cohen_kappa(
                predicted[compared.index].astype(bool), compared["relevant"].astype(bool)
            ),
            "compared": len(compared),
            "strata": sorted(by_stratum),
        }
    return table, result


def _labeller_agreement(labelled: pd.DataFrame) -> dict | None:
    if "relevant_2" not in labelled:
        return None
    both = labelled[labelled["relevant_2"].notna()]
    if both.empty:
        return None
    first, second = both["relevant"].astype(bool), both["relevant_2"].astype(bool)
    return {
        "papers": len(both),
        "agreement": float((first == second).mean()),
        "kappa": cohen_kappa(first, second),
    }


# ---------------------------------------------------------------------------------------------
# Themes
# ---------------------------------------------------------------------------------------------
def make_theme_sample(papers: pd.DataFrame, per_theme: int = 30, seed: int = 42) -> pd.DataFrame:
    """Up to ``per_theme`` papers tagged with each theme, with the words that matched.
    ``papers`` are the selected papers (selection.selected_papers)."""
    df = papers.reset_index(drop=True)
    text = search_text(df)
    parts = []
    for theme in keywords.THEMES:
        tagged = df[df["themes"].map(lambda found, t=theme: t in found)]
        sample = tagged.sample(n=min(per_theme, len(tagged)), random_state=seed)
        matched = text.loc[sample.index].map(
            lambda t, name=theme: ", ".join(keywords.THEME_MATCHER.words(name, t))
        )
        parts.append(sample.assign(theme=theme, matched=matched))
    columns = ["theme", "id", "matched", "title", "abstract"]
    if not parts or not sum(len(p) for p in parts):
        return pd.DataFrame(columns=[*columns, "relevant", "notes"])
    return pd.concat(parts, ignore_index=True)[columns].assign(relevant="", notes="")


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


def make_random_theme_sample(papers: pd.DataFrame, n: int = 200, seed: int = 42) -> pd.DataFrame:
    """``n`` selected papers drawn at random, to be tagged with every theme they discuss.

    Fill in ``themes`` with the theme names, separated by ";" (leave it empty if there are
    none), and set ``done`` to 1 once a paper has been read, so that papers without themes
    are told apart from papers nobody has read yet."""
    sample = papers.sample(n=min(n, len(papers)), random_state=seed)
    return sample[["id", "title", "abstract"]].assign(themes="", done="", notes="")


def parse_themes(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    names = [t.strip().lower() for t in str(value).replace(",", ";").split(";") if t.strip()]
    unknown = sorted(set(names) - set(keywords.THEMES))
    if unknown:
        raise ValueError(
            f"unknown theme(s) {', '.join(unknown)}; use names from: {', '.join(keywords.THEMES)}"
        )
    return names


def evaluate_theme_audit(sheet: pd.DataFrame, tagged: Mapping[str, pd.Series]) -> pd.DataFrame:
    """Precision, recall and F1 of each theme, for each tagging method.

    ``sheet`` is a filled-in :func:`make_random_theme_sample`; ``tagged`` maps a method name
    ("keywords", "llm") to a Series of theme lists indexed by paper ID. Only rows marked done
    and tagged by the method are compared."""
    done = sheet[sheet["done"].map(parse_label).fillna(False).astype(bool)]
    truth = dict(zip(done["id"], done["themes"].map(parse_themes), strict=True))
    rows = []
    for method, series in tagged.items():
        ids = [i for i in truth if i in series.index]
        for theme in keywords.THEMES:
            actual = np.array([theme in truth[i] for i in ids], dtype=bool)
            found = np.array([theme in series[i] for i in ids], dtype=bool)
            tp, fp, fn = (
                int((found & actual).sum()),
                int((found & ~actual).sum()),
                int((~found & actual).sum()),
            )
            precision = tp / (tp + fp) if tp + fp else math.nan
            recall = tp / (tp + fn) if tp + fn else math.nan
            f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else math.nan
            rows.append(
                {
                    "method": method,
                    "theme": theme,
                    "papers": len(ids),
                    "discussing": int(actual.sum()),
                    "tagged": int(found.sum()),
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "kappa": cohen_kappa(found, actual) if len(ids) else math.nan,
                }
            )
    return pd.DataFrame(rows)
