"""Tokenisation and phrase extraction, shared by the phrase counts and the clustering.

Two rules fix the artefacts of the first version (``"low back pain"`` counted as ``"low pain"``,
``"large language models (LLMs)"`` as ``"large language llms"``):

* stop words are only checked at a phrase's first and last word, never deleted from the middle,
  so words on either side of a stop word are never glued together;
* punctuation and brackets end a phrase, so ``"(AI)"`` cannot attach to the words before it.

Tokens keep digits, hyphens and accents: ``x-ray``, ``covid-19``, ``u-net``, ``3d``, ``naïve``.
"""

from __future__ import annotations

import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# scikit-learn's list also contains content words that matter in medicine ("side effects",
# "low back pain", "region of interest", "serious adverse events"): keep those.
_KEEP = {"side", "back", "interest", "system", "serious", "top", "bottom", "front", "full", "part"}
EDGE_STOP_WORDS = frozenset((ENGLISH_STOP_WORDS - _KEEP) | {"using", "based", "via"})

_URL = re.compile(r"https?://\S+|www\.\S+")
_POSSESSIVE = re.compile(r"'s\b")
_SEGMENT = re.compile(r"[^\w\s'-]+")
_TOKEN = re.compile(r"\w+(?:[-']\w+)*")

# Words ending in "s" that are not plurals (only matters for merging singular/plural phrases)
_NOT_PLURAL = {"diabetes", "series", "species", "news", "bias", "aids", "lens", "herpes", "sars"}


def segments(text: str) -> list[list[str]]:
    """Lower-cased tokens, split into runs that no phrase may cross."""
    text = _URL.sub(" ", str(text).lower().replace("’", "'"))
    text = _POSSESSIVE.sub("", text)
    return [tokens for part in _SEGMENT.split(text) if (tokens := _TOKEN.findall(part))]


def _edge_ok(token: str) -> bool:
    return len(token) > 1 and token not in EDGE_STOP_WORDS and not token.isdigit()


def phrases(text: str, n_min: int = 2, n_max: int = 3) -> list[str]:
    """All n-grams (n_min..n_max) whose first and last words are not stop words or numbers."""
    out = []
    for tokens in segments(text):
        for n in range(n_min, n_max + 1):
            for i in range(len(tokens) - n + 1):
                if _edge_ok(tokens[i]) and _edge_ok(tokens[i + n - 1]):
                    out.append(" ".join(tokens[i : i + n]))
    return out


def singular(word: str) -> str:
    """Rough singular form, used only as a key to merge "neural network(s)"-style pairs."""
    if (
        len(word) <= 3
        or word in _NOT_PLURAL
        or not word.endswith("s")
        or word.endswith(("ss", "us", "is", "ics"))
    ):
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("sses"):
        return word[:-2]
    return word[:-1]


def fold(phrase: str) -> str:
    """Merge key for a phrase: last word made singular ("clinical trials" -> "clinical trial")."""
    words = phrase.split(" ")
    words[-1] = singular(words[-1])
    return " ".join(words)
