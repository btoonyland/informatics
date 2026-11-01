"""A small synthetic arXiv-like corpus with known structure, used by the tests.

AI-in-medicine papers come from four topics. Each abstract draws a few words at random from its
topic's vocabulary, so the topics form four clean clusters without hidden sub-groups. The other
kinds of paper (AI only, medicine only, neither) exercise the filter. IDs follow the real arXiv
format, so the year can be derived from them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from informatics.records import normalize_records

TOPIC_WORDS = {
    "imaging": [
        "tumor segmentation", "lesion detection", "mri scans", "ct volumes", "radiology images",
        "chest x-ray", "u-net decoder", "voxel labels", "contrast agents", "image slices",
        "ultrasound frames", "histology slides",
    ],
    "records": [
        "electronic health records", "mortality risk", "readmission rates", "icu stays",
        "billing codes", "admission notes", "discharge timing", "cohort selection",
        "risk scores", "length of stay", "vital signs", "laboratory tests",
    ],
    "language": [
        "clinical notes", "question answering", "text summarisation", "prompt design",
        "discharge summaries", "entity recognition", "consultation transcripts",
        "chatbot dialogue", "hallucination rates", "radiology reports", "benchmark questions",
        "instruction tuning",
    ],
    "mental": [
        "depression screening", "anxiety symptoms", "social media posts", "smartphone sensors",
        "relapse prediction", "mood tracking", "suicide risk", "therapy sessions",
        "wellbeing surveys", "sleep patterns", "stress levels", "peer support",
    ],
}  # fmt: skip
THEME_SENTENCES = {
    "imaging": "Bias across scanner vendors is analysed.",
    "records": "Data privacy limits sharing between hospitals.",
    "language": "Regulatory approval of such tools is discussed.",
    "mental": "Ethical questions about consent are raised.",
}
AI_TERMS = ["deep learning", "neural networks", "machine learning", "a large language model"]
MEDICINE_TERMS = ["patients", "clinical care", "hospital wards", "medical practice"]
AI_ONLY_WORDS = [
    "robot grasping", "click logs", "compiler autotuning", "code completion", "warehouse scenes",
    "game playing", "route planning", "speech synthesis", "chip placement", "video retrieval",
]  # fmt: skip
MEDICINE_ONLY = [
    "A randomised clinical trial of a new statin in patients with diabetes.",
    "Hospital staffing levels and nursing workload during influenza season.",
]
NEITHER = [
    "We prove tight bounds for graph coloring on planar graphs.",
    "A consensus protocol tolerates Byzantine faults in distributed ledgers.",
]
BOILERPLATE = "Results are reported for setting {n}."
NON_ENGLISH = (
    "Nous utilisons le deep learning pour la segmentation des tumeurs chez les patients "
    "hospitalisés. Les résultats montrent une amélioration nette de la précision diagnostique "
    "sur trois cohortes indépendantes de plusieurs hôpitaux."
)


def _arxiv_id(rng: np.random.Generator, serial: int) -> str:
    year = int(rng.integers(2008, 2025))
    yymm = f"{year % 100:02d}{int(rng.integers(1, 13)):02d}"
    width = 5 if int(yymm) >= 1501 else 4
    return f"{yymm}.{serial:0{width}d}"


def make_corpus(n_per_topic: int = 40, seed: int = 0) -> pd.DataFrame:
    """Raw-schema DataFrame plus ``kind`` and ``topic`` columns holding the ground truth.

    Besides the four kinds, there are 3 withdrawn papers and 2 French abstracts (both must be
    excluded) and 3 re-posts of AI-medicine papers under new IDs (kind "duplicate"). Some
    AI-medicine papers have a journal reference or an "accepted" comment.
    """
    rng = np.random.default_rng(seed)
    rows = []

    def add(kind: str, topic: str, title: str, abstract: str, **extra: str) -> None:
        serial = len(rows) + 1
        rows.append(
            {
                "id": _arxiv_id(rng, serial),
                "title": extra.pop("exact_title", f"{title} ({serial})"),
                "abstract": extra.pop(
                    "exact_abstract", f"{abstract} {BOILERPLATE.format(n=serial)}"
                ),
                "authors": ["Ana Kovač", "Jörg Müller"],
                "categories": ["cs.LG", "q-bio.QM"] if kind == "ai_medicine" else ["cs.LG"],
                "kind": kind,
                "topic": topic,
                **extra,
            }
        )

    for topic, vocabulary in TOPIC_WORDS.items():
        for _ in range(n_per_topic):
            w = rng.choice(vocabulary, size=4, replace=False)
            ai, med = rng.choice(AI_TERMS), rng.choice(MEDICINE_TERMS)
            abstract = (
                f"We study {w[0]} and {w[1]} with {ai} for {med}. We improve {w[2]} and {w[3]}."
            )
            if rng.random() < 0.6:
                abstract += " " + THEME_SENTENCES[topic]
            extra = {}
            if len(rows) % 5 == 0:
                extra["journal_ref"] = "J. Synthetic Med. 1 (2020) 1-10"
            elif len(rows) % 7 == 0:
                extra["comments"] = "12 pages; accepted at a test conference"
            add("ai_medicine", topic, f"{w[0].capitalize()} and {w[1]}", abstract, **extra)
    for _ in range(n_per_topic * 3):
        w = rng.choice(AI_ONLY_WORDS, size=3, replace=False)
        add("ai_only", "ai", f"{w[0].capitalize()} with {rng.choice(AI_TERMS)}",
            f"We apply {rng.choice(AI_TERMS)} to {w[1]} and {w[2]}.")  # fmt: skip
    for _ in range(n_per_topic):
        text = str(rng.choice(MEDICINE_ONLY))
        add("medicine_only", "medicine", text.split(" in ")[0], text)
        text = str(rng.choice(NEITHER))
        add("neither", "other", text.split(" for ")[0], text)

    for original in rows[:3]:  # re-posted under a new ID with a short addition
        add("duplicate", original["topic"], "", "", exact_title=original["title"],
            exact_abstract=original["abstract"] + " This version corrects a typo.")  # fmt: skip
    for _ in range(3):
        add("withdrawn", "imaging", "Tumor segmentation with deep learning for patients",
            "We study tumor segmentation with deep learning for patients.",
            comments="This paper has been withdrawn by the authors.")  # fmt: skip
    for _ in range(2):
        add("non_english", "imaging", "Segmentation des tumeurs", NON_ENGLISH)

    df = pd.DataFrame(rows)
    truth = df[["id", "kind", "topic"]]
    return normalize_records(df.drop(columns=["kind", "topic"]), source="synthetic").merge(
        truth, on="id", how="left"
    )
