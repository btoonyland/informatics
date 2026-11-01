"""Concept tags: what each paper is about, along four dimensions.

Like ``keywords.THEMES``, each concept is a list of regular expressions matched
case-insensitively, as whole words, against title + abstract, with optional exclusions for
unrelated senses. A paper can have several concepts in a dimension, or none. These lists are a
starting point: check them against a sample of papers (as for the themes) and edit them here.
"""

from __future__ import annotations

import pandas as pd

from informatics.keywords import Theme, ThemeMatcher

CONCEPTS: dict[str, dict[str, Theme]] = {
    "disease area": {
        "cancer": Theme(
            [
                r"cancers?", r"tumou?rs?", r"oncolog\w*", r"carcinomas?", r"melanomas?",
                r"lymphomas?", r"leuka?emias?", r"gliomas?", r"glioblastomas?", r"metasta\w+",
                r"neoplasms?", r"malignan\w+",
            ]
        ),
        "cardiovascular": Theme(
            [
                r"cardiac", r"cardio\w+", r"heart (?:disease|failure|rate|attack)s?", r"coronary",
                r"arrhythmi\w*", r"atrial fibrillation", r"myocardial", r"hypertensi\w*",
            ]
        ),
        "neurological": Theme(
            [
                r"alzheimer\w*", r"parkinson\w*", r"dementia", r"epilep\w*", r"seizures?",
                r"strokes?", r"multiple sclerosis", r"neurodegenerat\w*",
                r"cognitive (?:impairment|decline)",
            ]
        ),
        "mental health": Theme(
            [
                r"mental (?:health|disorders?|illness\w*)", r"depress\w*", r"anxiety",
                r"psychiatr\w*", r"schizophreni\w*", r"bipolar", r"suicid\w*", r"ptsd",
                r"post-?traumatic stress", r"autism", r"autistic", r"adhd",
                r"substance (?:use|abuse)", r"addiction",
            ]
        ),
        "infectious disease": Theme(
            [
                r"covid(?:-19)?", r"sars-cov-2", r"coronavirus\w*", r"infecti\w*", r"sepsis",
                r"septic", r"tuberculosis", r"hiv", r"malaria", r"influenza", r"pneumonia",
                r"pandemics?", r"epidemics?", r"antimicrobial", r"antibiotics?",
            ]
        ),
        "eye": Theme(
            [
                r"retina\w*", r"retinopath\w*", r"glaucoma", r"ophthalm\w*", r"fundus",
                r"macular", r"cataracts?",
            ]
        ),
        "respiratory": Theme(
            [r"lungs?", r"pulmonary", r"copd", r"asthma", r"respiratory", r"chest x-?rays?"]
        ),
        "metabolic": Theme(
            [r"diabet\w*", r"obes\w+", r"insulin", r"glucose", r"glyca?emi\w*", r"thyroid"]
        ),
        "skin": Theme([r"skin (?:lesions?|cancers?|diseases?)", r"dermat\w*", r"dermoscop\w*"]),
        "musculoskeletal": Theme(
            [r"fractures?", r"osteo\w+", r"arthritis", r"orthop(?:a)?edic\w*", r"spin(?:e|al)"]
        ),
    },
    "data type": {
        "medical images": Theme(
            [
                r"imaging", r"mri", r"magnetic resonance", r"computed tomography",
                r"ct (?:scans?|images?|volumes?)", r"x-?rays?", r"radiograph\w*", r"ultrasound",
                r"mammogra\w*", r"histopatholog\w*", r"whole[- ]slide", r"endoscop\w*",
                r"radiolog\w* (?:images?|scans?|exams?)", r"medical images?", r"fundus",
                r"dermoscop\w*",
            ]
        ),
        "health records": Theme(
            [
                r"electronic (?:health|medical|patient) records?", r"ehrs?", r"emrs?",
                r"(?:medical|clinical|patient|hospital) records?", r"claims data",
                r"administrative data", r"billing codes?", r"icd(?:-?\d+)? codes?",
                r"intensive care units?", r"icus?",
            ]
        ),
        "clinical text": Theme(
            [
                r"clinical (?:notes?|text|narratives?|reports?)", r"discharge summar\w*",
                r"radiology reports?", r"medical (?:text|notes?|literature)", r"free[- ]text",
                r"natural language processing", r"nlp",
            ]
        ),
        "omics & molecules": Theme(
            [
                r"genom\w*", r"transcriptom\w*", r"proteom\w*", r"metabolom\w*", r"multi-?omics?",
                r"omics", r"gene expression", r"single[- ]cell", r"rna-?seq\w*", r"sequencing",
                r"dna", r"rna", r"mutations?", r"molecul\w*", r"proteins?",
            ]
        ),
        "signals & wearables": Theme(
            [
                r"wearables?", r"sensors?", r"accelerometer\w*", r"smartwatch\w*",
                r"smartphones?", r"photoplethysmogra\w*", r"ppg", r"electrocardiogra\w*",
                r"ecgs?", r"ekgs?", r"electroencephalogra\w*", r"eegs?", r"emg",
                r"vital signs?", r"physiological signals?",
            ]
        ),
        "speech & audio": Theme([r"speech", r"voice", r"audio", r"cough sounds?", r"acoustic"]),
        "video": Theme([r"videos?"]),
    },
    "clinical task": {
        "diagnosis & detection": Theme([r"diagnos\w*", r"detect\w*", r"screening"]),
        "segmentation": Theme([r"segment\w*"], exclude=[r"market segments?"]),
        "prognosis & risk": Theme(
            [
                r"prognos\w*", r"risk (?:prediction|stratification|scores?|assessment)",
                r"mortality", r"survival", r"readmissions?", r"outcome prediction",
                r"early warning", r"deterioration", r"length of stay",
            ]
        ),
        "treatment & drugs": Theme(
            [
                r"treatment (?:planning|recommendations?|selection|response|effects?)",
                r"therapeutic\w*", r"therap(?:y|ies)",
                r"drug (?:discovery|design|development|repurposing|response|dosing)", r"dosing",
                r"personali[sz]ed (?:medicine|treatments?)", r"precision medicine",
                r"clinical trials?",
            ]
        ),
        "workflow & operations": Theme(
            [
                r"triage", r"workflows?", r"scheduling", r"resource allocation",
                r"hospital operations?", r"documentation", r"administrative burden",
            ]
        ),
        "reports & dialogue": Theme(
            [
                r"report generation", r"question[- ]answering", r"summari[sz]\w*", r"chatbots?",
                r"conversational agents?", r"dialogue systems?",
            ]
        ),
    },
    "AI method": {
        "deep learning": Theme(
            [
                r"deep[- ]learning", r"neural networks?", r"convolutional", r"cnns?", r"u-?nets?",
                r"resnets?", r"transformers?", r"recurrent", r"lstms?", r"rnns?",
                r"autoencoders?",
            ]
        ),
        "classical machine learning": Theme(
            [
                r"random forests?", r"support vector\w*", r"svms?", r"logistic regression",
                r"gradient boosting", r"xgboost", r"lightgbm", r"decision trees?",
                r"k-?nearest neighbou?rs?", r"na[iï]ve bayes",
            ]
        ),
        "language & foundation models": Theme(
            [
                r"large language models?", r"llms?", r"gpt-?\w*", r"chatgpt", r"foundation models?",
                r"bert", r"biobert", r"clinicalbert", r"vision[- ]language models?",
                r"in-?context learning", r"prompt\w*",
            ]
        ),
        "generative models": Theme(
            [
                r"generative adversarial networks?", r"gans?", r"diffusion models?",
                r"variational autoencoders?", r"vaes?", r"synthetic data",
            ]
        ),
        "reinforcement learning": Theme(
            [r"reinforcement learning", r"markov decision process\w*", r"policy gradients?"]
        ),
        "federated & privacy-preserving": Theme(
            [
                r"federated", r"differential privacy", r"privacy[- ]preserving",
                r"homomorphic encryption", r"secure multi-?party",
            ]
        ),
        "explainability": Theme(
            [
                r"explainab\w*", r"interpretab\w*", r"xai", r"shap", r"saliency", r"grad-?cam",
                r"counterfactual explanations?",
            ]
        ),
        "graph methods": Theme(
            [r"graph neural networks?", r"gnns?", r"knowledge graphs?", r"graph convolution\w*"]
        ),
    },
}  # fmt: skip

MATCHERS = {dimension: ThemeMatcher(concepts) for dimension, concepts in CONCEPTS.items()}


def column(dimension: str) -> str:
    """Name of the papers-table column holding a dimension's concepts."""
    return "concepts_" + dimension.lower().replace(" ", "_").replace("&", "and")


def tag_concepts(search_texts: pd.Series) -> pd.DataFrame:
    """One list-valued column per dimension (see :func:`column`)."""
    return pd.DataFrame(
        {
            column(dimension): search_texts.map(lambda t, m=matcher: m.find(t)[0])
            for dimension, matcher in MATCHERS.items()
        },
        index=search_texts.index,
    )
