"""The second selection stage: which candidates count as AI-in-medicine papers.

``informatics filter`` keeps every candidate from the wide keyword net. The analysis, the
OpenAlex and embedding steps and theme labelling then use the candidates that one of these
marks as relevant:

* ``keywords``: the original keyword rule (the ``keyword_rule`` column of the candidates file);
* ``relevance``: the decisions in ``data/interim/<set>_relevance.parquet``, which
  ``informatics classify`` writes from a classifier trained on hand labels or from a language
  model. Hand labels, where there are any, override both.

``auto`` (the default) uses the relevance file when one exists and the keyword rule otherwise.
Near-duplicates are always dropped.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from informatics.paths import Layout
from informatics.storage import read_json, read_parquet

SELECTIONS = ("auto", "keywords", "relevance")


@dataclass(frozen=True)
class Selection:
    set_name: str
    method: str  # "keywords" or "relevance"
    description: str  # how the papers were chosen, for the report
    candidates: int  # candidates that are not near-duplicates
    selected: int
    duplicates_dropped: int  # near-duplicates of selected papers

    def as_dict(self) -> dict:
        return asdict(self)


def read_candidates(
    layout: Layout, set_name: str, columns: list[str] | None = None
) -> pd.DataFrame:
    path = layout.candidates(set_name)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run `informatics filter --set {set_name}` first"
        )
    return read_parquet(path, columns=columns)


def describe_relevance(info: dict) -> str:
    """One phrase saying how ``informatics classify`` made its decisions."""
    method = info.get("method")
    if method == "model":
        cv = info.get("cv", {})
        quality = ""
        if cv.get("precision") is not None and cv.get("recall") is not None:
            quality = (
                f"; cross-validated precision {100 * cv['precision']:.0f}%, "
                f"recall {100 * cv['recall']:.0f}%"
            )
        text = f"a classifier trained on {info.get('labels', 0):,} hand-labelled candidates"
        text += quality
    elif method == "llm":
        text = f"a language model ({info.get('model', 'unknown model')}) given a written rubric"
    else:
        text = f"`informatics classify --method {method}`"
    overridden = info.get("labels_used", 0)
    if overridden:
        text += f"; hand labels used for {overridden:,} papers"
    return text


def select(
    candidates: pd.DataFrame, set_name: str, layout: Layout, how: str = "auto"
) -> tuple[pd.DataFrame, Selection]:
    """The candidates chosen by ``how`` (see the module docstring), without near-duplicates."""
    if how not in SELECTIONS:
        raise ValueError(f"unknown selection {how!r}; choose from {', '.join(SELECTIONS)}")
    path = layout.relevance(set_name)
    method = how if how != "auto" else ("relevance" if path.exists() else "keywords")
    unique = candidates[candidates["duplicate_of"].isna()]
    if method == "keywords":
        keep = unique["keyword_rule"].astype(bool).to_numpy()
        description = "the keyword rule (at least one AI and one medicine keyword)"
    else:
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found; run `informatics classify --set {set_name}` first"
            )
        decisions = read_parquet(path, columns=["id", "relevant"]).set_index("id")["relevant"]
        decided = unique["id"].map(decisions)
        if decided.isna().any():
            raise ValueError(
                f"{int(decided.isna().sum()):,} candidates of {set_name} have no decision in "
                f"{path}; the candidates changed since `informatics classify` ran, so run it "
                "again"
            )
        keep = decided.astype(bool).to_numpy()
        info_path = layout.relevance_info(set_name)
        description = describe_relevance(read_json(info_path) if info_path.exists() else {})
    chosen = unique[keep].reset_index(drop=True)
    duplicates = candidates["duplicate_of"].isin(chosen["id"])
    return chosen, Selection(
        set_name=set_name,
        method=method,
        description=description,
        candidates=len(unique),
        selected=len(chosen),
        duplicates_dropped=int(duplicates.sum()),
    )


def selected_papers(
    layout: Layout, set_name: str, how: str = "auto", columns: list[str] | None = None
) -> tuple[pd.DataFrame, Selection]:
    """Read a set's candidates and apply the selection in one step."""
    if columns is not None:
        columns = list(dict.fromkeys([*columns, "id", "keyword_rule", "duplicate_of"]))
    return select(read_candidates(layout, set_name, columns), set_name, layout, how)
