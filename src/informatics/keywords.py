"""Keyword definitions: what counts as an AI paper, a medicine paper, and each secondary theme.

Edit this file (not the filtering code) to change what is selected. ``AI_TERMS`` and
``MEDICINE_TERMS`` are the original keyword rule: each key is a term from the first version of
the filter, and the value is a regular expression that also accepts its plural, hyphenated and
spelling variants. Matching is case-insensitive on title + abstract, and word boundaries are
added automatically, so ``patients?`` matches "patient" and "patients" but not "outpatient".

``AI_CONTEXT_TERMS`` and ``MEDICINE_CONTEXT_TERMS`` widen the net for the two-stage selection
(see filtering.py): a paper with any AI term and any medicine term, from either list, is a
*candidate*, and a classifier or a language model then decides whether it is about AI in
medicine. These lists favour recall. A wrong match only costs some classification time, while a
missing term loses relevant papers for good.

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

# The wide net. Words with a common unrelated sense are only accepted in phrases that fix the
# sense: "recurrent" alone is often medical ("recurrent stroke"), "stroke" often is not ("pen
# strokes"), "admissions" can be university admissions and "discharge" electrical.
AI_CONTEXT_TERMS: dict[str, str] = {
    "learning methods": (
        r"(?:reinforcement|representation|transfer|federated|contrastive|metric|multi[- ]?task"
        r"|meta|ensemble|active|statistical|feature|continual|multiple[- ]instance|imitation"
        r"|graph|end[- ]to[- ]end|(?:semi|self|weakly|un)?[- ]?supervised) learning"
        r"|(?:semi|self|weakly|un)[- ]?supervised|(?:few|zero|one)[- ]shot|learning[- ]based"
        r"|machine[- ]learned|pre-?train(?:ed|ing)|fine-?tun(?:e|ed|es|ing)"
        r"|training (?:data|sets?|datasets?)"
    ),
    "network architectures": (
        r"cnns?|rnns?|lstms?|grus?|gnns?|convolutional|recurrent networks?"
        r"|long short[- ]term memory|u-?nets?|resnets?|autoencoders?|(?:vision )?transformers?"
        r"|transformer[- ]based|attention mechanisms?|self[- ]attention"
        r"|multi[- ]?layer perceptrons?"
        r"|deep (?:models?|networks?|architectures?|features?|reinforcement)"
    ),
    "language models": (
        r"(?:pre-?trained |masked |medical |clinical |biomedical )?language models?|chatbots?"
        r"|conversational agents?|chat-?gpt|gpt(?:-?\d(?:\.\d)?o?)?|bert|biobert|clinicalbert"
        r"|pubmedbert|llama|generative (?:ai|pre-?trained)|genai"
    ),
    "generative models": (
        r"generative (?:adversarial networks?|models?)|gans?|diffusion models?"
        r"|variational autoencoders?|vaes?|normali[sz]ing flows?"
    ),
    "classical machine learning": (
        r"random forests?|support vector (?:machines?|regression)|svms?"
        r"|gradient[- ]boost(?:ing|ed)?"
        r"|xgboost|lightgbm|catboost|k[- ]?nearest neighbou?rs?|k-?nn|naive bayes|decision trees?"
        r"|bayesian networks?|hidden markov models?|classifiers?|(?:prediction|predictive) models?"
    ),
    "intelligent systems": (
        r"intelligent (?:systems?|agents?|assistants?)|expert systems?|knowledge[- ]based systems?"
        r"|decision[- ]support systems?|cdss|computer[- ]aided (?:diagnosis|detection)"
        r"|(?:automated|automatic) (?:diagnosis|detection|screening|segmentation|classification)"
        r"|machine intelligence|computational intelligence|soft computing|fuzzy (?:logic|systems?)"
    ),
    "data mining and recognition": (
        r"data[- ]mining|text[- ]mining|pattern recognition|predictive analytics"
        r"|(?:image|speech|object|activity|gesture|emotion|face|facial expression) recognition"
        r"|object detection|(?:semantic|instance) segmentation|image classification"
        r"|named entity recognition|information extraction|question answering"
        r"|(?:word|sentence|graph|node|patient) embeddings?"
    ),
}

MEDICINE_CONTEXT_TERMS: dict[str, str] = {
    "clinicians and carers": (
        r"clinics?|clinicians?|clinically|nurses?|surgeons?|radiologists?|pathologists?"
        r"|dermatologists?|ophthalmologists?|cardiologists?|psychiatrists?|caregivers?"
        r"|care[- ]givers?|general practi(?:ce|tioners?)"
    ),
    "care settings": (
        r"intensive care(?: units?)?|icus?|emergency (?:departments?|rooms?|medicine)"
        r"|primary care|inpatients?|outpatients?|hospitali[sz](?:ed|ation|ations)|telemedicine"
        r"|telehealth|ambulances?|nursing homes?|care homes?|readmissions?"
        r"|discharge (?:summar(?:y|ies)|notes?)|bedside"
    ),
    "specialties": (
        r"radiolog(?:y|ical)|histopatholog\w*|(?:digital|computational|clinical|surgical) pathology"
        r"|oncolog\w*|cardiolog\w*|dermatolog\w*|ophthalmolog\w*|neurolog(?:y|ical)"
        r"|psychiatr(?:y|ic)|surg(?:ery|eries|ical)|ana?esthe(?:sia|siology|tics?)"
        r"|pa?ediatrics?|obstetric\w*|gyna?ecolog\w*|urolog\w*|nephrolog\w*|gastroenterolog\w*"
        r"|endocrinolog\w*|geriatrics?|dentistry|dentists?|dental|epidemiolog\w*"
        r"|pharmacolog\w*|pharmac(?:y|ies|ists?|eutical)"
    ),
    "imaging and signals": (
        r"mri|fmri|magnetic resonance|ct (?:scans?|images?|imaging|volumes?)|computed tomography"
        r"|radiographs?|radiography|x-?ray (?:images?|scans?)|chest x-?rays?"
        r"|mammogra(?:ms?|phy|phic)|ultrasound|ultrasonograph\w*|sonograph\w*|echocardiogra\w*"
        r"|electrocardiogra\w*|ecgs?|ekgs?|eegs?|electroencephalogra\w*|emgs?|electromyogra\w*"
        r"|pet (?:scans?|imaging|images?)|positron emission|fundus|retinal|optical coherence"
        r" tomography|whole[- ]slide images?|histolog(?:y|ical)|biops(?:y|ies)|dermoscop\w*"
        r"|endoscop\w*|colonoscop\w*"
    ),
    "conditions": (
        r"diseases?|disorders?|syndromes?|cancers?|tumou?rs?|carcinomas?|malignan(?:t|cy|cies)"
        r"|lesions?|metasta\w*|diabet(?:es|ic)|alzheimer'?s?|parkinson'?s?|dementia"
        r"|(?:ischa?emic|acute|ha?emorrhagic) strokes?|stroke (?:patients?|survivors?|risk)"
        r"|sepsis|septic|covid(?:-?19)?|sars-cov-2|coronavirus|pneumonia|tuberculosis|hiv"
        r"|malaria|influenza|epilep(?:sy|tic)|seizures?|autis(?:m|tic)|adhd|schizophreni\w*"
        r"|bipolar|depress(?:ion|ive)|anxiety|suicid\w*|mental illness(?:es)?|cardiovascular"
        r"|heart (?:disease|failure|attack|rate)|arrhythmi\w*|atrial fibrillation|hypertension"
        r"|obesity|asthma|copd|kidney|renal|liver|hepat(?:ic|itis|ocellular)|glaucoma"
        r"|retinopathy|macular|cataracts?|melanoma|leuka?emia|lymphoma|infections?|infectious"
        r"|pandemics?|epidemics?|injur(?:y|ies)|wounds?|pain|frailty|comorbid\w*"
    ),
    "care and treatment": (
        r"therap(?:y|ies|eutic|ists?)|drugs?|medications?|prescri(?:be|bed|bing|ptions?)"
        r"|dosage|dosing|vaccin\w*|rehabilitation|symptoms?|diagnos(?:is|es|e|tics?|ed|ing)"
        r"|prognos(?:is|es|tic)|screening|mortality|morbidity|survival analysis"
    ),
    "biomedical": (
        r"biomedic(?:al|ine)|biomarkers?|physiolog(?:y|ical)|pathogen(?:s|esis)?|anatom(?:y|ical)"
        r"|tissues?|blood (?:pressure|glucose|tests?|samples?|cells?)|glucose|insulin|vital signs"
        r"|wearables?|health records|ehrs?|emrs?|cohorts?|clinical trials?"
        r"|randomi[sz]ed (?:controlled )?trials?|healthy (?:controls?|subjects?|volunteers?)"
        r"|elderly|older adults|neonat\w*|newborns?|infants?|pregnan(?:t|cy|cies)|maternal"
        r"|well-?being|lungs?|pulmonary|cardiac|breast|prostate|colorectal|thyroid|pancrea\w*"
        r"|brain (?:tumou?rs?|mri|lesions?|scans?|images?|disorders?|diseases?|injur(?:y|ies))"
        r"|skin (?:lesions?|cancers?|diseases?)|chest"
    ),
    "medical datasets and exams": (
        r"usmle|medqa|pubmedqa|mimic-?(?:iii|iv|cxr)|chexpert|physionet|brats|isic"
    ),
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

    def words(self, theme: str, text: str) -> list[str]:
        """Distinct words that tagged ``text`` with ``theme`` (after its exclusions)."""
        for name, include, exclude in self._compiled:
            if name == theme:
                searchable = exclude.sub(" ", text) if exclude is not None else text
                return sorted({m.group(0).lower() for m in include.finditer(searchable)})
        raise KeyError(theme)

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
AI_CONTEXT = TermMatcher(AI_CONTEXT_TERMS)
MEDICINE_CONTEXT = TermMatcher(MEDICINE_CONTEXT_TERMS)
THEME_MATCHER = ThemeMatcher(THEMES)


def keyword_lists() -> dict:
    """Every pattern used to select papers and tag themes, as recorded with each result."""
    return {
        "ai": AI_TERMS,
        "medicine": MEDICINE_TERMS,
        "ai_context": AI_CONTEXT_TERMS,
        "medicine_context": MEDICINE_CONTEXT_TERMS,
        "themes": {
            name: {"include": t.include, "exclude": t.exclude} for name, t in THEMES.items()
        },
    }


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
