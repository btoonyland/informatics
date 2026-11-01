"""Find candidate AI-in-medicine papers with a wide keyword net, and keep a sample of AI-only
papers to compare them against.

Selection has two stages. This module is the first: a deliberately wide keyword net
(``keywords.py``) that should miss few relevant papers. The second stage (selection.py) decides
which candidates are really about AI in medicine: the original keyword rule, a classifier
trained on hand labels (relevance.py) or a language model given a written rubric (llm.py).

Runs on the CPU with pandas, reading the raw file in chunks. Every paper is assigned to exactly
one stratum, used in the summary and for measuring each stage (validation.py):

* ``strict``         meets the original keyword rule (enough AI and medicine keyword hits)
* ``wide_only``      has an AI term and a medicine term from the wider lists, but does not meet
                     the original rule
* ``ai_only``        AI terms but no medicine terms (the comparison sample comes from these)
* ``medicine_only``  medicine terms but no AI terms
* ``neither``        no terms at all
* ``excluded``       withdrawn, or not written in English (language is only checked for
                     candidates and for papers that could join the comparison sample)

``strict`` and ``wide_only`` papers are the candidates. They are written to
``data/interim/<set>_candidates.parquet``, where ``keyword_rule`` tells the two apart.
Candidates are also checked for near-duplicates (the same paper under two IDs). The extra copies
stay in the file with ``duplicate_of`` set and are left out of the analysis.
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
from informatics.normalize import clean_characters
from informatics.provenance import file_info
from informatics.records import RAW_SCHEMA
from informatics.storage import iter_parquet, read_json, write_json, write_parquet

log = logging.getLogger(__name__)

STRATA = ("strict", "wide_only", "ai_only", "medicine_only", "neither", "excluded")
CANDIDATE_STRATA = ("strict", "wide_only")
# Relevant papers in these strata are missed by the wide net (see validation.py).
OUTSIDE_STRATA = ("ai_only", "medicine_only", "neither")
_STR_LIST = pa.list_(pa.string())
CANDIDATE_SCHEMA = pa.schema(
    list(RAW_SCHEMA)
    + [
        pa.field("ai_terms", _STR_LIST),  # every AI term found, from both lists
        pa.field("ai_hits", pa.int32()),  # occurrences of the original AI keywords
        pa.field("medicine_terms", _STR_LIST),
        pa.field("medicine_hits", pa.int32()),
        pa.field("keyword_rule", pa.bool_()),  # meets the original rule (stratum "strict")
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
    min_ai_hits: int = 1  # the original keyword rule
    min_medicine_hits: int = 1
    background_size: int = 50_000
    seed: int = 42
    batch_size: int = 100_000

    def __post_init__(self) -> None:
        if self.min_ai_hits < 1 or self.min_medicine_hits < 1:
            raise ValueError("--min-ai-hits and --min-medicine-hits must be at least 1")


def search_text(df: pd.DataFrame) -> pd.Series:
    """Title and abstract with TeX and odd characters cleaned (normalize.clean_characters),
    lower-cased: the text every keyword, theme and concept is matched against."""
    raw = df["title"].fillna("") + ". " + df["abstract"].fillna("")
    return raw.map(clean_characters).str.lower()


def _distinct(hits: list[str]) -> list[str]:
    return list(dict.fromkeys(hits))


def exclusion_reason(df: pd.DataFrame, check_language: pd.Series) -> pd.Series:
    """ "withdrawn", "non-english" or None. Language is only checked where ``check_language``
    is true (candidates and possible comparison papers), which keeps the filter fast."""
    comments = df["comments"] if "comments" in df else pd.Series(None, index=df.index)
    reason = pd.Series(
        [
            "withdrawn" if is_withdrawn(c, a) else None
            for c, a in zip(comments, df["abstract"], strict=True)
        ],
        index=df.index,
        dtype=object,
    )
    check = reason.isna() & check_language
    if check.any():
        raw = df.loc[check, "title"].fillna("") + ". " + df.loc[check, "abstract"].fillna("")
        reason.loc[check] = raw.map(lambda t: "non-english" if is_non_english(t) else None)
    return reason


def annotate(df: pd.DataFrame, config: FilterConfig) -> pd.DataFrame:
    """Matched terms, hit counts, keyword rule, exclusion reason and stratum for each paper."""
    text = search_text(df)
    strict_ai, strict_med = text.map(keywords.AI.hits), text.map(keywords.MEDICINE.hits)
    wide_ai = text.map(keywords.AI_CONTEXT.hits)
    wide_med = text.map(keywords.MEDICINE_CONTEXT.hits)
    out = pd.DataFrame(index=df.index)
    out["ai_terms"] = [_distinct(a + b) for a, b in zip(strict_ai, wide_ai, strict=True)]
    out["ai_hits"] = strict_ai.map(len).astype("int32")
    out["medicine_terms"] = [_distinct(a + b) for a, b in zip(strict_med, wide_med, strict=True)]
    out["medicine_hits"] = strict_med.map(len).astype("int32")
    rule = (out["ai_hits"] >= config.min_ai_hits) & (
        out["medicine_hits"] >= config.min_medicine_hits
    )
    any_ai, any_med = out["ai_terms"].map(bool), out["medicine_terms"].map(bool)
    stratum = pd.Series(
        np.select(
            [rule, any_ai & any_med, any_ai, any_med],
            ["strict", "wide_only", "ai_only", "medicine_only"],
            default="neither",
        ),
        index=df.index,
        dtype=object,
    )
    out["keyword_rule"] = rule
    background_eligible = (stratum == "ai_only") & (out["ai_hits"] >= config.min_ai_hits)
    out["exclusion"] = exclusion_reason(df, stratum.isin(CANDIDATE_STRATA) | background_eligible)
    out["stratum"] = stratum.where(out["exclusion"].isna(), "excluded")
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
    candidates_path: str | Path,
    background_path: str | Path,
    summary_path: str | Path,
    config: FilterConfig | None = None,
) -> dict:
    config = config or FilterConfig()
    rng = np.random.default_rng(config.seed)
    strata: Counter[str] = Counter()
    exclusions: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    by_year: dict[int, Counter[str]] = defaultdict(Counter)
    candidate_parts: list[pd.DataFrame] = []
    reservoir: pd.DataFrame | None = None

    for chunk in iter_parquet(raw_path, batch_size=config.batch_size):
        chunk = pd.concat([chunk, annotate(chunk, config)], axis=1)
        strata.update(chunk["stratum"].value_counts().to_dict())
        sources.update(chunk["source"].fillna("unknown").value_counts().to_dict())
        exclusions.update(chunk["exclusion"].dropna().value_counts().to_dict())
        kept = chunk["exclusion"].isna()
        is_candidate = chunk["stratum"].isin(CANDIDATE_STRATA)
        _add_year_counts(by_year, "scanned", chunk["year"])
        _add_year_counts(by_year, "ai", chunk.loc[kept & (chunk["ai_hits"] > 0), "year"])
        _add_year_counts(by_year, "candidates", chunk.loc[is_candidate, "year"])
        _add_year_counts(by_year, "keyword_rule", chunk.loc[chunk["stratum"] == "strict", "year"])
        if is_candidate.any():
            candidate_parts.append(tag_themes(chunk[is_candidate]))
        pool = chunk[(chunk["stratum"] == "ai_only") & (chunk["ai_hits"] >= config.min_ai_hits)]
        if config.background_size > 0 and len(pool):
            # Reservoir sample: keep the rows with the smallest random keys across all chunks.
            pool = pool.assign(_key=rng.random(len(pool)))
            pool = pool if reservoir is None else pd.concat([reservoir, pool])
            reservoir = pool.sort_values("_key").head(config.background_size)
        log.info(
            "scanned %d papers: %d candidates so far",
            sum(strata.values()),
            strata["strict"] + strata["wide_only"],
        )

    candidates = _finish(candidate_parts, CANDIDATE_SCHEMA, set_name)
    duplicates = near_duplicates(candidates["id"].tolist(), search_text(candidates).tolist())
    candidates["duplicate_of"] = candidates["id"].map(duplicates).astype(object)
    background = _finish(
        [] if reservoir is None else [reservoir.drop(columns="_key")], BACKGROUND_SCHEMA, set_name
    )
    write_parquet(candidates[CANDIDATE_SCHEMA.names], candidates_path, schema=CANDIDATE_SCHEMA)
    write_parquet(background[BACKGROUND_SCHEMA.names], background_path, schema=BACKGROUND_SCHEMA)

    unique = candidates[candidates["duplicate_of"].isna()]
    theme_counts = unique.loc[unique["keyword_rule"], "themes"].explode().dropna().value_counts()
    import_path = Path(raw_path).with_name(f"{Path(raw_path).stem}_import.json")
    summary = {
        "set": set_name,
        "raw": str(raw_path),
        "raw_file": file_info(raw_path),
        # How the raw file was made (see sources.py), and which sources its papers came from.
        "import": read_json(import_path) if import_path.exists() else None,
        "sources": {str(k): int(v) for k, v in sources.items()},
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": asdict(config),
        "counts": {
            "scanned": sum(strata.values()),
            **{s: strata.get(s, 0) for s in STRATA},
            "candidates": strata.get("strict", 0) + strata.get("wide_only", 0),
            "excluded_withdrawn": exclusions.get("withdrawn", 0),
            "excluded_non_english": exclusions.get("non-english", 0),
            "near_duplicates": len(duplicates),
            "background_sample": len(background),
        },
        "by_year": {str(y): dict(by_year[y]) for y in sorted(by_year)},
        "themes_keyword_rule": {str(k): int(v) for k, v in theme_counts.items()},
        "keywords": keywords.keyword_lists(),
    }
    write_json(summary, summary_path)
    counts = summary["counts"]
    log.info(
        "%s: %d candidates (%d meet the keyword rule) of %d papers; %d AI-only papers kept for "
        "comparison",
        set_name,
        counts["candidates"],
        counts["strict"],
        counts["scanned"],
        len(background),
    )
    return summary


def _finish(parts: list[pd.DataFrame], schema: pa.Schema, set_name: str) -> pd.DataFrame:
    df = pd.concat(parts, ignore_index=True) if parts else schema.empty_table().to_pandas()
    df["corpus"] = set_name
    return df.sort_values("id").reset_index(drop=True)
