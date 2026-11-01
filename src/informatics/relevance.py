"""Second selection stage, option 1: a classifier trained on hand-labelled candidates.

``informatics classify --set cs --method model``:

1. represents each candidate by TF-IDF weights of its normalised words and two-word phrases
   (normalize.py; selection keywords included, since here they carry signal) and by whether it
   meets the original keyword rule;
2. trains a logistic regression with balanced class weights on the labelled candidates. Its
   regularisation strength is chosen by an inner cross-validation when there are enough labels;
3. estimates the classifier's quality by stratified k-fold cross-validation, so that every
   labelled paper gets a score from a model that never saw its label. The decision threshold
   maximises F1 on those scores, with each labelled paper weighted by the number of candidates
   of its stratum it stands for (candidates that meet the keyword rule and the others are
   sampled at different rates);
4. scores every candidate and writes ``data/interim/<set>_relevance.parquet`` with ``score``,
   ``predicted`` (the score is at or above the threshold; for labelled papers, the
   cross-validated prediction, which ``label-evaluate`` uses) and ``relevant``, the decision
   the analysis uses: the hand label where there is one, else ``predicted``.

About 1,000 labels give a dependable classifier. With fewer, read the cross-validated recall
before relying on it.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from informatics.clustering import TermAnalyzer
from informatics.normalize import Normalizer

log = logging.getLogger(__name__)

C_GRID = (0.25, 1.0, 4.0, 16.0)
MIN_PER_CLASS = 5  # labelled candidates needed in each class
RELEVANCE_COLUMNS = ["id", "relevant", "predicted", "score", "label", "method"]


@dataclass
class RelevanceResult:
    decisions: pd.DataFrame  # RELEVANCE_COLUMNS, one row per candidate
    info: dict  # method, labels, threshold, cross-validated quality


def decisions_table(
    ids: Sequence[str],
    predicted: Sequence[bool],
    scores: Sequence[float] | None,
    labels: pd.Series,
    method: str,
) -> pd.DataFrame:
    """The relevance file's rows: hand labels (``labels``: id -> bool) override predictions."""
    label = pd.Series(list(ids)).map(labels).astype("boolean")
    predicted = pd.Series(np.asarray(predicted, dtype=bool))
    return pd.DataFrame(
        {
            "id": list(ids),
            "relevant": label.fillna(predicted).astype(bool),
            "predicted": predicted,
            "score": pd.Series(scores, dtype="float32") if scores is not None else np.nan,
            "label": label,
            "method": method,
        }
    )


def weighted_threshold(
    scores: np.ndarray, truth: np.ndarray, weights: np.ndarray
) -> tuple[float, dict]:
    """The score threshold (predict relevant if score >= threshold) with the highest weighted
    F1, and the weighted precision, recall and F1 there."""
    order = np.argsort(-scores, kind="stable")
    s, t, w = scores[order], truth[order].astype(bool), weights[order]
    tp, fp = np.cumsum(w * t), np.cumsum(w * ~t)
    last = np.r_[s[1:] != s[:-1], True]  # one candidate threshold per distinct score
    tp, fp, s = tp[last], fp[last], s[last]
    precision = tp / (tp + fp)
    recall = tp / max(float((w * t).sum()), 1e-12)
    with np.errstate(invalid="ignore"):
        f1 = np.nan_to_num(2 * precision * recall / (precision + recall))
    best = int(np.argmax(f1))
    return float(s[best]), {
        "precision": float(precision[best]),
        "recall": float(recall[best]),
        "f1": float(f1[best]),
    }


def _choose_c(x: sparse.csr_matrix, y: np.ndarray, seed: int) -> float:
    """Regularisation strength with the best log loss in a 3-fold inner cross-validation."""
    if min(int(y.sum()), int((~y).sum())) < 10:
        return 1.0
    losses = []
    for c in C_GRID:
        folds = StratifiedKFold(3, shuffle=True, random_state=seed).split(x, y)
        loss = 0.0
        for train, test in folds:
            model = _logistic(c).fit(x[train], y[train])
            loss += log_loss(y[test], model.predict_proba(x[test])[:, 1], labels=[False, True])
        losses.append(loss)
    return C_GRID[int(np.argmin(losses))]


def _logistic(c: float) -> LogisticRegression:
    return LogisticRegression(C=c, class_weight="balanced", max_iter=5000)


def _fit(x: sparse.csr_matrix, y: np.ndarray, seed: int) -> tuple[LogisticRegression, float]:
    c = _choose_c(x, y, seed)
    return _logistic(c).fit(x, y), c


def train_and_classify(
    candidates: pd.DataFrame,
    texts: Sequence[str],
    labels: pd.Series,
    *,
    folds: int = 5,
    seed: int = 42,
) -> RelevanceResult:
    """``candidates`` (with ``id`` and ``keyword_rule``) are the unique candidates of a set and
    ``texts`` their prepared texts (normalize.prepare_text), in the same order. ``labels`` maps
    paper IDs to hand labels (True = relevant); labels of papers that are not candidates are
    ignored."""
    ids = candidates["id"].tolist()
    rule = candidates["keyword_rule"].astype(bool).to_numpy()
    labelled = candidates["id"].isin(labels.dropna().index).to_numpy()
    y = candidates.loc[labelled, "id"].map(labels).astype(bool).to_numpy()
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if min(n_pos, n_neg) < MIN_PER_CLASS:
        raise ValueError(
            f"training needs at least {MIN_PER_CLASS} relevant and {MIN_PER_CLASS} not relevant "
            f"labelled candidates (there are {n_pos} and {n_neg}); label more with "
            "`informatics label-sample --extend`"
        )
    normalizer = Normalizer.from_texts(texts)
    vectorizer = TfidfVectorizer(
        analyzer=TermAnalyzer(normalizer, drop_selection_terms=False),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    x = sparse.hstack(
        [vectorizer.fit_transform(texts), sparse.csr_matrix(rule.astype(float)[:, None])]
    ).tocsr()
    x_labelled = x[labelled]

    # Each labelled paper stands for (candidates in its stratum) / (labelled in its stratum).
    weights = np.ones(len(y))
    for value in (True, False):
        in_stratum = rule[labelled] == value
        if in_stratum.any():
            weights[in_stratum] = (rule == value).sum() / in_stratum.sum()

    k = min(folds, n_pos, n_neg)
    out_of_fold = np.empty(len(y))
    for train, test in StratifiedKFold(k, shuffle=True, random_state=seed).split(x_labelled, y):
        model, _ = _fit(x_labelled[train], y[train], seed)
        out_of_fold[test] = model.predict_proba(x_labelled[test])[:, 1]
    threshold, at_threshold = weighted_threshold(out_of_fold, y, weights)
    final, c = _fit(x_labelled, y, seed)
    scores = final.predict_proba(x)[:, 1]
    scores[labelled] = out_of_fold  # labelled papers keep their cross-validated score
    predicted = scores >= threshold

    decisions = decisions_table(ids, predicted, scores, labels, "model")
    info = {
        "method": "model",
        "model": "logistic regression on TF-IDF of normalised words and phrases",
        "labels": int(labelled.sum()),
        "relevant_labels": n_pos,
        "labels_used": int(labelled.sum()),
        "candidates": len(ids),
        "selected": int(decisions["relevant"].sum()),
        "threshold": threshold,
        "C": c,
        "features": int(x.shape[1]),
        "cv": {
            "folds": k,
            **at_threshold,
            "roc_auc": float(roc_auc_score(y, out_of_fold, sample_weight=weights)),
            "average_precision": float(
                average_precision_score(y, out_of_fold, sample_weight=weights)
            ),
            "weighted_by_stratum": True,
        },
    }
    log.info(
        "classifier: cross-validated precision %.2f, recall %.2f at threshold %.2f; "
        "%d of %d candidates relevant",
        at_threshold["precision"],
        at_threshold["recall"],
        threshold,
        info["selected"],
        len(ids),
    )
    return RelevanceResult(decisions=decisions, info=info)
