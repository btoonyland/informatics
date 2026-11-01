"""Text normalisation: one key per term, however the term is written.

Every count merges variants *before* counting, so a paper that writes a term two ways is counted
once; reports show the most common spelling. The steps, in order:

Per paper (:func:`prepare_text`; cached in ``data/interim/text_cache.parquet``):

1. :func:`clean_characters`: Unicode NFKC, dash variants, TeX markup (``$\\alpha$``,
   ``\\emph{..}``, ``Gr\\"obner``), which is common in arXiv abstracts.
2. :func:`expand_acronyms`: the Schwartz & Hearst (2003) algorithm, per abstract. After
   "artificial neural networks (ANNs)", later "ANNs" become the long form. Acronyms are expanded
   per abstract because they are ambiguous across papers (AD: Alzheimer's disease or anomaly
   detection).
3. :func:`apply_synonyms`: a short curated list of variants with one meaning in this field
   (EHR, EMR, "electronic medical records" -> "electronic health records"; undefined "LLMs").

Per word (:class:`Normalizer`, built from the words of the corpus being analysed):

4. Spelling: British -> American, from a word list plus suffix rules (-ise, -our, -tre, haem-,
   ...). A rule only applies when the American form is at least as common in the corpus, so
   "exercise" never becomes "exercize".
5. Lemma: simplemma dictionary lemmas ("studies" -> "study", "fine-tuned" -> "fine-tune"). Words
   are protected where the lemmatizer is wrong for this field ("media" is not "medium", "data"
   is not "datum", "aids" is not "aid"). Unknown words fall back to a plural rule ("llms" ->
   "llm") or to their parts ("pretrained" -> "pre" + "train").
6. Compounds: keys ignore hyphens ("multi-modal" = "multimodal"), and two words are joined when the
   corpus also writes them hyphenated ("multi modal" -> "multimodal").

Deliberately *not* merged: words that a stemmer would conflate but that mean different things
here (specific/specificity, medical/medication, organ/organization, image/imaging).
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable

import simplemma

from informatics.text import edge_ok, segments, singular

# Bump when the rules change: it invalidates cached prepared texts.
NORMALIZATION_VERSION = "1"

# ---------------------------------------------------------------------------------------------
# 1. Characters and TeX
# ---------------------------------------------------------------------------------------------
_HYPHENS = str.maketrans({"\u2010": "-", "\u2011": "-", "\u2212": "-", "\u2043": "-", "\u00ad": ""})
_EN_DASH_IN_WORD = re.compile(r"(?<=\w)[\u2012\u2013](?=\w)")
_DASHES = re.compile(r"[\u2012\u2013\u2014\u2015\u2e3a\u2e3b]")
_DOLLAR = "\x00"
_GREEK = {
    "alpha", "beta", "gamma", "delta", "epsilon", "varepsilon", "zeta", "eta", "theta", "iota",
    "kappa", "lambda", "mu", "nu", "xi", "pi", "rho", "sigma", "tau", "upsilon", "phi", "varphi",
    "chi", "psi", "omega",
}  # fmt: skip
_MATH = re.compile(r"\$\$(.+?)\$\$|\$(.+?)\$", re.S)
_ACCENT_MARKS = {
    '"': "\u0308", "'": "\u0301", "`": "\u0300", "^": "\u0302", "~": "\u0303", "=": "\u0304",
    ".": "\u0307", "u": "\u0306", "v": "\u030c", "H": "\u030b", "c": "\u0327", "k": "\u0328",
    "r": "\u030a", "b": "\u0331", "d": "\u0323",
}  # fmt: skip
_ACCENT_SYMBOL = re.compile(r"""\\(["'`^~=.])\s*(?:\{\s*(\\?[A-Za-z])\s*\}|(\\?[A-Za-z]))""")
_ACCENT_LETTER = re.compile(r"\\([uvHckrbd])(?:\s*\{\s*(\\?[A-Za-z])\s*\}|\s+([A-Za-z]))")
_SPECIAL_LETTERS = {
    "\\i": "i", "\\j": "j", "\\ss": "ß", "\\o": "ø", "\\O": "Ø", "\\l": "ł", "\\L": "Ł",
    "\\aa": "å", "\\AA": "Å", "\\ae": "æ", "\\AE": "Æ", "\\oe": "œ",
}  # fmt: skip
_SPECIAL_LETTER = re.compile(r"\\(?:ss|aa|AA|ae|AE|oe|[ijoOlL])(?![A-Za-z])")
_KEEP_ARGUMENT = re.compile(
    r"\\(?:emph|textit|textbf|textrm|textsf|texttt|textsc|textup|mathrm|mathbf|mathit|mathsf"
    r"|mathtt|mathcal|mathbb|mathfrak|boldsymbol|text|mbox|underline|operatorname)\s*\{([^{}]*)\}"
)
_DROP_ARGUMENT = re.compile(r"\\(?:cite[tp]?|ref|eqref|label|url|footnote)\s*\{[^{}]*\}")
_HREF = re.compile(r"\\href\s*\{[^{}]*\}\s*\{([^{}]*)\}")
_ESCAPED = re.compile(r"\\([%&_#{}])")
_COMMAND = re.compile(r"\\[A-Za-z]+\*?")


def _accent(match: re.Match[str]) -> str:
    letter = match.group(2) or match.group(3)
    letter = _SPECIAL_LETTERS.get(letter, letter)
    return unicodedata.normalize("NFC", letter + _ACCENT_MARKS[match.group(1)])


def _math_as_text(match: re.Match[str]) -> str:
    body = match.group(1) or match.group(2)
    body = re.sub(
        r"\\([A-Za-z]+)",
        lambda c: f" {c.group(1).lower()} " if c.group(1).lower() in _GREEK else " ",
        body,
    )
    return " " + re.sub(r"[^A-Za-z0-9.]+", " ", body).strip() + " "


def _tex_to_text(text: str) -> str:
    text = text.replace("\\$", _DOLLAR)
    text = _MATH.sub(_math_as_text, text)
    text = _ACCENT_SYMBOL.sub(_accent, text)
    text = _ACCENT_LETTER.sub(_accent, text)
    text = _SPECIAL_LETTER.sub(lambda m: _SPECIAL_LETTERS[m.group(0)], text)
    text = _HREF.sub(r"\1", text)
    text = _DROP_ARGUMENT.sub(" ", text)
    for _ in range(3):  # nested formatting such as \textbf{\emph{x}}
        text = _KEEP_ARGUMENT.sub(r"\1", text)
    text = _ESCAPED.sub(r"\1", text)
    text = _COMMAND.sub(" ", text)
    text = text.replace("{", "").replace("}", "").replace("~", " ")
    return text.replace(_DOLLAR, "$")


def clean_characters(text: str) -> str:
    """Unicode NFKC (ligatures, full-width letters), one hyphen character, and TeX to plain text."""
    text = unicodedata.normalize("NFKC", str(text)).translate(_HYPHENS)
    text = _EN_DASH_IN_WORD.sub("-", text)
    text = _DASHES.sub(" ", text)
    if "\\" in text or "$" in text:
        text = _tex_to_text(text)
    return " ".join(text.split())


# ---------------------------------------------------------------------------------------------
# 2. Acronyms defined in the abstract (Schwartz & Hearst 2003)
# ---------------------------------------------------------------------------------------------
_PARENTHESES = re.compile(r"\(\s*([^()]{2,40}?)\s*\)")
_CLAUSE_END = re.compile(r"[.;:!?\[\]()]")


def _is_short_form(candidate: str) -> bool:
    return (
        2 <= len(candidate) <= 10
        and candidate[0].isalnum()
        and len(candidate.split()) <= 2
        and sum(c.isupper() for c in candidate) >= 2  # "ANNs", "EHR", "fMRI"; not "(i)", "(see)"
    )


def _best_long_form(short: str, words: list[str]) -> str | None:
    """Match the short form's characters right to left in the preceding words (S&H, Fig. 2)."""
    text = " ".join(words)
    s, i = len(short) - 1, len(text) - 1
    while s >= 0:
        char = short[s].lower()
        if not char.isalnum():
            s -= 1
            continue
        # The first character of the short form must start a word of the long form.
        while i >= 0 and (text[i].lower() != char or (s == 0 and i > 0 and text[i - 1].isalnum())):
            i -= 1
        if i < 0:
            return None
        s -= 1
        i -= 1
    long_form = text[text.rfind(" ", 0, i + 1) + 1 :]
    if len(long_form) <= len(short) or short.lower() in long_form.lower().split():
        return None
    return long_form


def _replace_short_form(text: str, short: str, long_form: str) -> str:
    variants = {short, short[:-1] if short.endswith("s") and len(short) > 2 else short + "s"}
    alternatives = "|".join(sorted(map(re.escape, variants), key=len, reverse=True))
    return re.sub(rf"(?<![\w-])(?:{alternatives})(?![\w-])", lambda _m: long_form, text)


def expand_acronyms(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Replace acronyms defined in this text by their long form; returns (text, definitions)."""
    definitions: list[tuple[str, str]] = []
    position = 0
    while match := _PARENTHESES.search(text, position):
        short = re.split(r"[,;]", match.group(1))[0].strip()
        if not _is_short_form(short):
            position = match.end()
            continue
        clause = _CLAUSE_END.split(text[: match.start()])[-1]
        words = clause.split()[-min(len(short) + 5, 2 * len(short)) :]
        long_form = _best_long_form(short, words)
        if not long_form:
            position = match.end()
            continue
        definitions.append((short, long_form))
        head = _replace_short_form(text[: match.start()].rstrip(), short, long_form)
        tail = _replace_short_form(text[match.end() :], short, long_form)
        text = head + tail
        position = len(head)
    return text, definitions


# ---------------------------------------------------------------------------------------------
# 3. Curated synonyms (case-sensitive for acronyms: "mL" millilitre is not "ML")
# ---------------------------------------------------------------------------------------------
SYNONYMS: list[tuple[str, str, int]] = [
    (r"\belectronic (?:medical|patient|health)[- ]records?\b", "electronic health records", re.I),
    (r"\b(?:EHR|EMR|EPR)s?\b", "electronic health records", 0),
    (r"\bheart attacks?\b", "myocardial infarction", re.I),
    (r"\bXAI\b", "explainable artificial intelligence", 0),
    (r"\bAI\b", "artificial intelligence", 0),
    (r"\bML\b", "machine learning", 0),
    (r"\bDL\b", "deep learning", 0),
    (r"\bNLP\b", "natural language processing", 0),
    (r"\bLLMs?\b", "large language models", 0),
    (r"\b(?:CNNs?|ConvNets?)\b", "convolutional neural networks", 0),
    (r"\bRNNs?\b", "recurrent neural networks", 0),
    (r"\bDNNs?\b", "deep neural networks", 0),
    (r"\bGANs?\b", "generative adversarial networks", 0),
    (r"\bLSTMs?\b", "long short-term memory", 0),
    (r"\bRL\b", "reinforcement learning", 0),
    (r"\bMRI\b", "magnetic resonance imaging", 0),
    (r"\bCT\b", "computed tomography", 0),
    (r"\b(?:ECG|EKG)s?\b", "electrocardiogram", 0),
    (r"\bEEGs?\b", "electroencephalography", 0),
    (r"\bICUs?\b", "intensive care unit", 0),
]
_SYNONYMS = [(re.compile(pattern, flags), replacement) for pattern, replacement, flags in SYNONYMS]


def apply_synonyms(text: str) -> str:
    for pattern, replacement in _SYNONYMS:
        text = pattern.sub(replacement, text)
    return text


def prepare_text(text: str) -> str:
    """Steps 1-3 for one paper's title + abstract."""
    expanded, _ = expand_acronyms(clean_characters(text))
    return apply_synonyms(expanded)


# ---------------------------------------------------------------------------------------------
# 4. Spelling
# ---------------------------------------------------------------------------------------------
UK_US_WORDS = {
    "programme": "program", "programmes": "programs", "grey": "gray", "ageing": "aging",
    "artefact": "artifact", "artefacts": "artifacts", "catalogue": "catalog",
    "analogue": "analog", "analogues": "analogs", "licence": "license", "defence": "defense",
    "offence": "offense", "judgement": "judgment", "judgements": "judgments",
    "acknowledgement": "acknowledgment", "enrolment": "enrollment", "enrol": "enroll",
    "fulfil": "fulfill", "skilful": "skillful", "aluminium": "aluminum", "sulphur": "sulfur",
    "manoeuvre": "maneuver", "mould": "mold", "practise": "practice", "practised": "practiced",
    "foetus": "fetus", "foetal": "fetal", "anaesthesia": "anesthesia", "oedema": "edema",
    "orthopaedic": "orthopedic", "paediatric": "pediatric", "paediatrics": "pediatrics",
    "haemoglobin": "hemoglobin", "oesophagus": "esophagus", "oestrogen": "estrogen",
    "diarrhoea": "diarrhea", "coeliac": "celiac", "tumour": "tumor", "tumours": "tumors",
    "fibre": "fiber", "fibres": "fibers", "centre": "center", "centres": "centers",
    "metre": "meter", "metres": "meters", "litre": "liter", "litres": "liters", "odour": "odor",
}  # fmt: skip
_LL_STEMS = (
    "model", "label", "signal", "travel", "cancel", "channel", "fuel", "level", "counsel",
    "tunnel", "total", "dial", "equal", "marshal", "unravel", "panel",
)  # fmt: skip
_DOUBLE_L = re.compile(rf"^(\w*?)({'|'.join(_LL_STEMS)})l(ed|ing|er|ers)$")
# Applied only when the American result is at least as common in the corpus as the input.
_SPELLING_RULES = [
    (re.compile(r"isation(s|al|ally)?$"), r"ization\1"),
    (re.compile(r"(?<=\w\w)is(e|ed|es|ing|er|ers)$"), r"iz\1"),
    (re.compile(r"(?<=\w\w)ys(e|ed|ing)$"), r"yz\1"),  # not "-yses": "analyses" is a noun
    (re.compile(r"(?<=\w\w)our(s|al|ally|ed|ing|ite|ites|able|ism)?$"), r"or\1"),
    (re.compile(r"(?<=\w\w)tre(s?)$"), r"ter\1"),
    (re.compile(r"haem"), "hem"),
    (re.compile(r"(?<=\w)aem"), "em"),  # anaemia, leukaemia, ischaemia, glycaemia
    (re.compile(r"anaes"), "anes"),
    (re.compile(r"paed"), "ped"),
    (re.compile(r"oesoph"), "esoph"),
    (re.compile(r"oedem"), "edem"),
    (re.compile(r"foet"), "fet"),
    (re.compile(r"gynaec"), "gynec"),
    (re.compile(r"oestr"), "estr"),
    (re.compile(r"rrhoea"), "rrhea"),
    (re.compile(r"caesar"), "cesar"),
]


def american(word: str, lexicon: Counter[str] | dict[str, int]) -> str:
    """American spelling of one word (no hyphens), using ``lexicon`` to check suffix rules."""
    if word in UK_US_WORDS:
        return UK_US_WORDS[word]
    if match := _DOUBLE_L.match(word):
        return match.group(1) + match.group(2) + match.group(3)
    if len(word) < 6:
        return word
    for pattern, replacement in _SPELLING_RULES:
        candidate = pattern.sub(replacement, word, count=1)
        if candidate != word and lexicon.get(candidate, 0) >= max(1, lexicon.get(word, 0)):
            return candidate
    return word


# ---------------------------------------------------------------------------------------------
# 5. Lemmas
# ---------------------------------------------------------------------------------------------
# Where the dictionary lemmatizer is wrong for this field, or a plural form is a different term.
LEMMA_OVERRIDES = {
    "data": "data", "media": "media", "aids": "aids", "imaging": "imaging",
    "diabetes": "diabetes", "series": "series", "species": "species", "news": "news",
    "lens": "lens", "covid": "covid", "sars": "sars", "measles": "measles", "mumps": "mumps",
    "rabies": "rabies", "herpes": "herpes", "bias": "bias", "biases": "bias",
    "diagnoses": "diagnosis", "prognoses": "prognosis", "metastases": "metastasis",
    "analyses": "analysis", "hypotheses": "hypothesis", "criteria": "criterion",
}  # fmt: skip
# Prefixes of closed compounds the lemmatizer does not know: "pretrained" -> "pre" + "train".
_PREFIXES = (
    "pre", "post", "fine", "multi", "self", "semi", "non", "sub", "inter", "intra", "over",
    "under", "cross", "auto", "re", "co", "de", "un",
)  # fmt: skip


def lemma(word: str) -> str:
    """Dictionary lemma of one lower-case word (for hyphenated words, of the last part)."""
    if word in LEMMA_OVERRIDES:
        return LEMMA_OVERRIDES[word]
    if "-" in word:
        head, _, last = word.rpartition("-")
        return f"{head}-{lemma(last)}" if last else word
    if not word.isalpha() or word.endswith("ics"):  # numbers, "covid19"; genomics, statistics
        return word
    if simplemma.is_known(word, lang="en"):
        return simplemma.lemmatize(word, lang="en").lower()
    for prefix in _PREFIXES:
        rest = word[len(prefix) :]
        if word.startswith(prefix) and len(rest) >= 4 and simplemma.is_known(rest, lang="en"):
            rest_lemma = simplemma.lemmatize(rest, lang="en").lower()
            if rest_lemma != rest:
                return prefix + rest_lemma
    return singular(word)


# ---------------------------------------------------------------------------------------------
# 4-6 together, with the corpus's words
# ---------------------------------------------------------------------------------------------
class Normalizer:
    """Maps words and phrases to keys, using the corpus's own vocabulary for spelling checks and
    for deciding which open compounds ("multi modal") to join."""

    def __init__(self, lexicon: Counter[str] | dict[str, int], min_hyphenated: int = 2):
        self.lexicon = lexicon
        self._keys: dict[str, str] = {}
        self.joinable = {
            self.key(token)
            for token, count in lexicon.items()
            if "-" in token and count >= min_hyphenated
        }

    @classmethod
    def from_texts(cls, texts: Iterable[str], min_hyphenated: int = 2) -> Normalizer:
        """Build from prepared texts (see :func:`prepare_text`)."""
        lexicon: Counter[str] = Counter()
        for text in texts:
            for segment in segments(text):
                lexicon.update(segment)
        return cls(lexicon, min_hyphenated)

    def key(self, token: str) -> str:
        """Key of one token: American spelling -> lemma -> hyphens removed."""
        cached = self._keys.get(token)
        if cached is None:
            spelled = "-".join(american(part, self.lexicon) for part in token.split("-"))
            cached = lemma(spelled).replace("-", "").replace("'", "")
            self._keys[token] = cached
        return cached

    def tokens(self, text: str) -> list[list[tuple[str, str]]]:
        """(key, written form) per token, per segment, with open compounds joined."""
        out = []
        for segment in segments(text):
            pairs = [(self.key(t), t) for t in segment]
            joined, i = [], 0
            while i < len(pairs):
                if i + 1 < len(pairs) and pairs[i][0] + pairs[i + 1][0] in self.joinable:
                    joined.append(
                        (pairs[i][0] + pairs[i + 1][0], f"{pairs[i][1]} {pairs[i + 1][1]}")
                    )
                    i += 2
                else:
                    joined.append(pairs[i])
                    i += 1
            out.append(joined)
        return out

    def phrase_pairs(self, text: str, n_min: int = 2, n_max: int = 3) -> list[tuple[str, str]]:
        """(key, written form) of every n-gram whose first and last words may edge a phrase."""
        out = []
        for segment in self.tokens(text):
            for n in range(n_min, n_max + 1):
                for i in range(len(segment) - n + 1):
                    gram = segment[i : i + n]
                    if edge_ok(gram[0][1]) and edge_ok(gram[-1][1]):
                        out.append((" ".join(k for k, _ in gram), " ".join(w for _, w in gram)))
        return out

    def phrase_keys(self, text: str, n_min: int = 2, n_max: int = 3) -> list[str]:
        return [key for key, _ in self.phrase_pairs(text, n_min, n_max)]

    def key_of(self, phrase: str) -> str:
        """Key of a phrase written in plain text (e.g. a keyword)."""
        return " ".join(k for segment in self.tokens(phrase) for k, _ in segment)
