"""Select AI-in-medicine papers, and keep a sample of AI-only papers to compare them against.

Runs on the CPU with pandas, reading the raw file in chunks. Every paper is assigned to exactly
one stratum (used both for the summary and for measuring the filter, see ``validation.py``):

* ``selected``        enough AI hits and enough medicine hits
* ``ai_only``         AI terms but no medicine terms (the comparison set is sampled from these)
* ``medicine_only``   medicine terms but no AI terms
* ``neither``         no terms at all
* ``below_threshold`` both kinds of terms, but fewer than ``--min-*-hits`` requires
* ``excluded``        withdrawn, or (among papers with AI terms) not written in English

Selected papers are also checked for near-duplicates (the same paper under two IDs); the extra
copies stay in the file with ``duplicate_of`` set and are left out of the analysis.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

from informatics import keywords
from informatics.cleaning import is_non_english, is_withdrawn, near_duplicates
from informatics.records import RAW_SCHEMA
from informatics.storage import iter_parquet, write_json, write_parquet

log = logging.getLogger(__name__)

STRATA = ("selected", "ai_only", "medicine_only", "neither", "below_threshold", "excluded")
# Strata whose relevant papers count as missed by the keyword rule (see validation.py).
MISSED_STRATA = ("ai_only", "medicine_only", "neither", "below_threshold")
_STR_LIST = pa.list_(pa.string())
SELECTED_SCHEMA = pa.schema(
    list(RAW_SCHEMA)
    + [
        pa.field("ai_terms", _STR_LIST),
        pa.field("ai_hits", pa.int32()),
        pa.field("medicine_terms", _STR_LIST),
        pa.field("medicine_hits", pa.int32()),
        pa.field("themes", _STR_LIST),
        pa.field("theme_terms", _STR_LIST),
        pa.field("duplicate_of", pa.string()),
        pa.field("corpus", pa.string()),
    ]
)
BACKGROUND_SCHEMA = pa.schema(
    list(RAW_SCHEMA)
    + [
        pa.field("ai_terms", _STR_LIST),
        pa.field("ai_hits", pa.int32()),
        pa.field("corpus", pa.string()),
    ]
)


@dataclass(frozen=True)
class FilterConfig:
    min_ai_hits: int = 1
    min_medicine_hits: int = 1
    background_size: int = 50_000
    seed: int = 42
    batch_size: int = 100_000


def search_text(df: pd.DataFrame) -> pd.Series:
    """Title and abstract, lower-cased: the text every keyword is matched against."""
    return (df["title"].fillna("") + ". " + df["abstract"].fillna("")).str.lower()


def _distinct(hits: list[str]) -> list[str]:
    return list(dict.fromkeys(hits))


def assign_strata(ai_hits: pd.Series, medicine_hits: pd.Series, config: FilterConfig) -> pd.Series:
    conditions = [
        (ai_hits >= config.min_ai_hits) & (medicine_hits >= config.min_medicine_hits),
        (ai_hits >= 1) & (medicine_hits == 0),
        (medicine_hits >= 1) & (ai_hits == 0),
        (ai_hits == 0) & (medicine_hits == 0),
    ]
    labels = np.select(conditions, STRATA[:4], default="below_threshold")
    return pd.Series(labels, index=ai_hits.index, dtype=object)


def exclusion_reason(df: pd.DataFrame, ai_hits: pd.Series) -> pd.Series:
    """ "withdrawn", "non-english" or None. Language is only checked for papers with AI terms (the
    only ones that can be selected or sampled), which keeps the filter fast."""
    comments = df["comments"] if "comments" in df else pd.Series(None, index=df.index)
    reason = pd.Series(
        [
            "withdrawn" if is_withdrawn(c, a) else None
            for c, a in zip(comments, df["abstract"], strict=True)
        ],
        index=df.index,
        dtype=object,
    )
    check = reason.isna() & (ai_hits >= 1)
    if check.any():
        raw = df.loc[check, "title"].fillna("") + ". " + df.loc[check, "abstract"].fillna("")
        reason.loc[check] = raw.map(lambda t: "non-english" if is_non_english(t) else None)
    return reason


def annotate(df: pd.DataFrame, config: FilterConfig) -> pd.DataFrame:
    """Matched AI/medicine terms, hit counts, exclusion reason and stratum for each paper."""
    text = search_text(df)
    ai = text.map(keywords.AI.hits)
    medicine = text.map(keywords.MEDICINE.hits)
    out = pd.DataFrame(index=df.index)
    out["ai_terms"] = ai.map(_distinct)
    out["ai_hits"] = ai.map(len).astype("int32")
    out["medicine_terms"] = medicine.map(_distinct)
    out["medicine_hits"] = medicine.map(len).astype("int32")
    out["exclusion"] = exclusion_reason(df, out["ai_hits"])
    strata = assign_strata(out["ai_hits"], out["medicine_hits"], config)
    out["stratum"] = strata.where(out["exclusion"].isna(), "excluded")
    return out


def tag_themes(df: pd.DataFrame) -> pd.DataFrame:
    found = search_text(df).map(keywords.THEME_MATCHER.find)
    df = df.copy()
    df["themes"] = found.map(lambda f: f[0])
    df["theme_terms"] = found.map(lambda f: f[1])  # distinct words, so no duplicates
    return df


def _add_year_counts(by_year: dict, column: str, years: pd.Series) -> None:
    for year, n in years.dropna().value_counts().items():
        by_year[int(year)][column] += int(n)


def run_filter(
    raw_path: str | Path,
    set_name: str,
    selected_path: str | Path,
    background_path: str | Path,
    summary_path: str | Path,
    config: FilterConfig | None = None,
) -> dict:
    config = config or FilterConfig()
    rng = np.random.default_rng(config.seed)
    strata: Counter[str] = Counter()
    exclusions: Counter[str] = Counter()
    by_year: dict[int, Counter[str]] = defaultdict(Counter)
    selected_parts: list[pd.DataFrame] = []
    reservoir: pd.DataFrame | None = None

    for chunk in iter_parquet(raw_path, batch_size=config.batch_size):
        chunk = pd.concat([chunk, annotate(chunk, config)], axis=1)
        strata.update(chunk["stratum"].value_counts().to_dict())
        exclusions.update(chunk["exclusion"].dropna().value_counts().to_dict())
        is_ai = chunk["ai_hits"] >= config.min_ai_hits
        is_selected = chunk["stratum"] == "selected"
        _add_year_counts(by_year, "scanned", chunk["year"])
        _add_year_counts(by_year, "ai", chunk.loc[is_ai, "year"])
        _add_year_counts(by_year, "selected", chunk.loc[is_selected, "year"])
        if is_selected.any():
            selected_parts.append(tag_themes(chunk[is_selected]))
        candidates = chunk[(chunk["stratum"] == "ai_only") & is_ai]
        if config.background_size > 0 and len(candidates):
            # Reservoir sample: keep the rows with the smallest random keys across all chunks.
            candidates = candidates.assign(_key=rng.random(len(candidates)))
            pool = candidates if reservoir is None else pd.concat([reservoir, candidates])
            reservoir = pool.sort_values("_key").head(config.background_size)
        log.info("scanned %d papers: %d selected so far", sum(strata.values()), strata["selected"])

    selected = _finish(selected_parts, SELECTED_SCHEMA, set_name)
    duplicates = near_duplicates(selected["id"].tolist(), search_text(selected).tolist())
    selected["duplicate_of"] = selected["id"].map(duplicates).astype(object)
    background = _finish(
        [] if reservoir is None else [reservoir.drop(columns="_key")], BACKGROUND_SCHEMA, set_name
    )
    write_parquet(selected[SELECTED_SCHEMA.names], selected_path, schema=SELECTED_SCHEMA)
    write_parquet(background[BACKGROUND_SCHEMA.names], background_path, schema=BACKGROUND_SCHEMA)

    theme_counts = selected.loc[selected["duplicate_of"].isna(), "themes"].explode()
    theme_counts = theme_counts.dropna().value_counts()
    summary = {
        "set": set_name,
        "raw": str(raw_path),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": asdict(config),
        "counts": {
            "scanned": sum(strata.values()),
            **{s: strata.get(s, 0) for s in STRATA},
            "excluded_withdrawn": exclusions.get("withdrawn", 0),
            "excluded_non_english": exclusions.get("non-english", 0),
            "near_duplicates": len(duplicates),
            "background_sample": len(background),
        },
        "by_year": {str(y): dict(by_year[y]) for y in sorted(by_year)},
        "themes": {str(k): int(v) for k, v in theme_counts.items()},
        "keywords": {
            "ai": keywords.AI_TERMS,
            "medicine": keywords.MEDICINE_TERMS,
            "themes": {name: asdict(theme) for name, theme in keywords.THEMES.items()},
        },
    }
    write_json(summary, summary_path)
    log.info(
        "%s: %d of %d papers selected; %d AI-only papers kept for comparison",
        set_name,
        len(selected),
        summary["counts"]["scanned"],
        len(background),
    )
    return summary


def _finish(parts: list[pd.DataFrame], schema: pa.Schema, set_name: str) -> pd.DataFrame:
    df = pd.concat(parts, ignore_index=True) if parts else schema.empty_table().to_pandas()
    df["corpus"] = set_name
    return df.sort_values("id").reset_index(drop=True)
