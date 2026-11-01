"""Keyword definitions: what counts as an AI paper, a medicine paper, and each secondary theme.

Edit this file (not the filtering code) to change what is selected. Each key is the original
term from the first version of the filter; the value is a regular expression that also accepts
its plural, hyphenated and spelling variants. Matching is case-insensitive on title + abstract,
and word boundaries are added automatically, so ``patients?`` matches "patient" and "patients"
but not "outpatient".

Variants were only added where they keep the meaning: for example ``adoption`` does not accept
"adopt", because "we adopt a U-Net" is about methods, not adoption.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import cached_property

AI_TERMS: dict[str, str] = {
    "artificial intelligence": r"artificial[- ]intelligence",
    "ai": r"ai",
    "machine learning": r"machine[- ]learning",
    "deep learning": r"deep[- ]learning",
    "neural networks": r"neural[- ]net(?:work)?s?",
    "nlp": r"nlp",
    "natural language processing": r"natural[- ]language[- ]processing",
    "computer vision": r"computer[- ]vision",
    "llm": r"llms?|large[- ]language[- ]models?",
}

MEDICINE_TERMS: dict[str, str] = {
    "healthcare": r"health[- ]?care",
    "medicine": r"medicines?",
    "health": r"health",
    "hospital": r"hospitals?",
    "doctors": r"doctors?",
    "clinical": r"clinical",
    "patient": r"patients?",
    "medical": r"medical",
    "physicians": r"physicians?",
    "nursing": r"nursing",
}


@dataclass(frozen=True)
class Theme:
    """A secondary theme. ``exclude`` removes phrases that use the word in an unrelated sense
    before ``include`` is searched (e.g. "gene regulation" is not about regulation of AI)."""

    include: list[str]
    exclude: list[str] = field(default_factory=list)


THEMES: dict[str, Theme] = {
    "adoption": Theme([r"adoptions?"]),
    "diagnostics": Theme([r"diagnostics?"]),
    "personalize": Theme([r"personali[sz](?:e|es|ed|ing|ation|ations)"]),
    "operational efficiency": Theme([r"operational[- ]efficienc(?:y|ies)"]),
    "ethical": Theme([r"ethic(?:s|al|ally)"]),
    "legal": Theme([r"legal(?:ly)?"]),
    "privacy": Theme([r"privacy"]),
    "cybersecurity": Theme([r"cyber[- ]?security"]),
    "transparency": Theme([r"transparen(?:cy|t)"]),
    # Almost every machine-learning abstract reports accuracy: treat this as a baseline.
    "accuracy": Theme([r"accurac(?:y|ies)"]),
    "strategic implementation": Theme([r"strategic[- ]implementations?"]),
    "maturity": Theme(
        [r"maturity"],
        exclude=[r"maturity[- ]onset", r"(?:skeletal|sexual|lung|fetal|bone|cell) maturity"],
    ),
    "readiness": Theme([r"readiness"]),
    "bias": Theme(
        [r"bias(?:es|ed)?"],
        exclude=[r"bias[- ]fields?", r"inductive[- ]bias(?:es)?", r"bias[- ]variance"],
    ),
    "compliance": Theme(
        [r"compliance"],
        exclude=[r"(?:lung|arterial|vascular|respiratory|ventricular|chest[- ]wall) compliance"],
    ),
    "regulation": Theme(
        [r"regulations?", r"regulatory", r"regulators?"],
        exclude=[
            r"(?:gene|genes|transcriptional|post[- ]transcriptional|epigenetic|translational"
            r"|metabolic|immune|allosteric|hormonal|homeostatic|feedback|self|up|down|co)"
            r"[- ]?regulation",
            r"regulation of (?:gene|genes|expression|transcription|proteins?|cells?|cellular"
            r"|metabolism)",
            r"regulatory (?:networks?|elements?|regions?|sequences?|genes?|proteins?"
            r"|mechanisms?|pathways?|motifs?|variants?|t[- ]cells?|rnas?|circuits?"
            r"|interactions?|modules?|programs?)",
            r"(?:transcriptional|master|gene|key) regulators?",
        ],
    ),
}


def _alternation(patterns: Iterable[str]) -> str:
    # Longer patterns first so "health-care" is one hit for "healthcare", not also one for "health"
    return "|".join(f"(?:{p})" for p in sorted(patterns, key=len, reverse=True))


def bounded(patterns: Iterable[str]) -> re.Pattern[str]:
    return re.compile(rf"\b(?:{_alternation(patterns)})\b", re.IGNORECASE)


class TermMatcher:
    """Find which named terms occur in a text, and how often."""

    def __init__(self, terms: dict[str, str]):
        self.names = list(terms)
        ordered = sorted(enumerate(terms.values()), key=lambda item: len(item[1]), reverse=True)
        groups = "|".join(f"(?P<t{i}>{pattern})" for i, pattern in ordered)
        self.regex = re.compile(rf"\b(?:{groups})\b", re.IGNORECASE)

    def hits(self, text: str) -> list[str]:
        """Term names in order of occurrence, one per occurrence."""
        return [self.names[int(m.lastgroup[1:])] for m in self.regex.finditer(text)]


class ThemeMatcher:
    def __init__(self, themes: dict[str, Theme]):
        self.themes = themes

    @cached_property
    def _compiled(self) -> list[tuple[str, re.Pattern[str], re.Pattern[str] | None]]:
        return [
            (name, bounded(t.include), bounded(t.exclude) if t.exclude else None)
            for name, t in self.themes.items()
        ]

    def find(self, text: str) -> tuple[list[str], list[str]]:
        """(themes present, distinct matched words) for one text."""
        found, words = [], set()
        for name, include, exclude in self._compiled:
            searchable = exclude.sub(" ", text) if exclude is not None else text
            matched = {m.group(0).lower() for m in include.finditer(searchable)}
            if matched:
                found.append(name)
                words |= matched
        return found, sorted(words)


AI = TermMatcher(AI_TERMS)
MEDICINE = TermMatcher(MEDICINE_TERMS)
THEME_MATCHER = ThemeMatcher(THEMES)


def selection_phrases() -> set[str]:
    """Lower-case surface forms of the selection terms (used to exclude them from clustering)."""
    words = {
        "artificial intelligence", "ai", "machine learning", "deep learning", "neural network",
        "neural networks", "neural net", "neural nets", "nlp", "natural language processing",
        "computer vision", "llm", "llms", "large language model", "large language models",
        "healthcare", "health care", "medicine", "medicines", "health", "hospital", "hospitals",
        "doctor", "doctors", "clinical", "patient", "patients", "medical", "physician",
        "physicians", "nursing",
    }  # fmt: skip
    return words | {w.replace(" ", "-") for w in words if " " in w}
