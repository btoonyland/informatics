import numpy as np
import pandas as pd
import pytest

from informatics.normalize import prepare_text
from informatics.records import title_and_abstract
from informatics.relevance import decisions_table, train_and_classify, weighted_threshold

RELEVANT_KINDS = {"ai_medicine", "duplicate", "wide_relevant"}


def test_weighted_threshold_maximises_f1():
    scores = np.array([0.9, 0.8, 0.7, 0.4, 0.2])
    truth = np.array([True, True, False, True, False])
    threshold, metrics = weighted_threshold(scores, truth, np.ones(5))
    assert threshold == pytest.approx(0.4)  # all 3 relevant found, 1 false positive
    assert metrics["recall"] == 1.0 and metrics["precision"] == pytest.approx(0.75)
    # If the third paper stands for many candidates, the false positive costs more.
    threshold, _ = weighted_threshold(scores, truth, np.array([1, 1, 10, 1, 1.0]))
    assert threshold == pytest.approx(0.8)


def test_hand_labels_override_predictions():
    labels = pd.Series({"b": False, "c": True})
    table = decisions_table(["a", "b", "c"], [True, True, False], [0.9, 0.8, 0.1], labels, "x")
    assert table["relevant"].tolist() == [True, False, True]
    assert table["predicted"].tolist() == [True, True, False]
    assert table["label"].isna().tolist() == [True, False, False]


def _candidates(corpus):
    """The synthetic corpus's candidates (strict and wide-only papers), as the filter finds them."""
    kinds = corpus["kind"]
    candidates = corpus[kinds.isin(RELEVANT_KINDS | {"wide_irrelevant"})].reset_index(drop=True)
    return candidates.assign(keyword_rule=~candidates["kind"].str.startswith("wide_"))


def test_classifier_learns_from_labels(corpus):
    candidates = _candidates(corpus)
    truth = candidates["kind"].isin(RELEVANT_KINDS)
    labelled = candidates.groupby("kind", group_keys=False).apply(
        lambda g: g.sample(frac=0.5, random_state=0)
    )
    labels = pd.Series(truth[labelled.index].to_numpy(), index=labelled["id"])
    texts = [prepare_text(t) for t in title_and_abstract(candidates)]
    result = train_and_classify(candidates, texts, labels, folds=3, seed=0)
    decisions = result.decisions.set_index("id")
    unlabelled = ~candidates["id"].isin(labels.index)
    predicted = decisions.loc[candidates.loc[unlabelled, "id"], "predicted"].to_numpy()
    assert (predicted == truth[unlabelled].to_numpy()).mean() >= 0.95
    # Labelled papers keep their label as the decision.
    assert (decisions.loc[labels.index, "relevant"] == labels).all()
    info = result.info
    assert info["labels"] == len(labels) and info["candidates"] == len(candidates)
    assert info["cv"]["precision"] >= 0.9 and info["cv"]["recall"] >= 0.9
    assert info["cv"]["folds"] == 3 and 0 < info["threshold"] < 1


def test_too_few_labels(corpus):
    candidates = _candidates(corpus)
    texts = [prepare_text(t) for t in title_and_abstract(candidates)]
    labels = pd.Series([True] * 10 + [False] * 2, index=candidates["id"][:12])
    with pytest.raises(ValueError, match="at least 5 relevant and 5 not relevant"):
        train_and_classify(candidates, texts, labels)
