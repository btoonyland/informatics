import math

import numpy as np
import pandas as pd
import pytest

from informatics.trends import benjamini_hochberg, decimal_year, fit_logit, tag_trends


def test_logit_matches_the_closed_form_odds_ratio():
    # With one binary covariate, the fitted log odds ratio is log(ad / bc), with Woolf's SE.
    a, b, c, d = 30, 70, 10, 90
    x = np.column_stack([np.ones(200), np.r_[np.ones(100), np.zeros(100)]])
    y = np.r_[np.ones(a), np.zeros(b), np.ones(c), np.zeros(d)]
    fit = fit_logit(x, y)
    assert fit.converged
    assert fit.coef[1] == pytest.approx(math.log(a * d / (b * c)), abs=1e-6)
    assert fit.se(1) == pytest.approx(math.sqrt(1 / a + 1 / b + 1 / c + 1 / d), abs=1e-6)


def _papers(share, years=range(2010, 2025), per_year=300, seed=0):
    """``share(year, cluster)`` is the chance that a paper has the tag."""
    rng = np.random.default_rng(seed)
    rows = []
    for year in years:
        for _ in range(per_year):
            cluster = int(rng.integers(0, 2)) if share.__name__ != "mix" else None
            if cluster is None:  # topic 1 grows from 10% to 90% of papers
                cluster = int(rng.random() < 0.1 + 0.8 * (year - 2010) / 14)
            tagged = rng.random() < share(year, cluster)
            rows.append({"year": year, "month": int(rng.integers(1, 13)), "cluster": cluster,
                         "themes": ["privacy"] if tagged else []})  # fmt: skip
    return pd.DataFrame(rows)


def test_composition_effect_is_separated_from_change_within_topics():
    def mix(year, cluster):  # constant within each topic; only the topic mix changes
        return 0.6 if cluster else 0.1

    row = tag_trends(_papers(mix), "themes", ["privacy"]).iloc[0]
    assert row["or_per_year"] > 1.1 and row["p_value"] < 1e-6
    assert row["adjusted_ci_low"] < 1 < row["adjusted_ci_high"]  # no change within topics

    def within(year, cluster):
        return 0.1 + 0.02 * (year - 2010)

    row = tag_trends(_papers(within), "themes", ["privacy"]).iloc[0]
    assert row["adjusted_or_per_year"] == pytest.approx(row["or_per_year"], rel=0.01)
    assert row["adjusted_ci_low"] > 1 and row["adjusted_pp_per_year"] == pytest.approx(2, abs=0.3)


def test_a_topic_without_the_tag_does_not_break_the_fit():
    def never_in_topic_1(year, cluster):
        return 0 if cluster else 0.1 + 0.02 * (year - 2010)

    row = tag_trends(_papers(never_in_topic_1), "themes", ["privacy"]).iloc[0]
    assert np.isfinite(row["adjusted_or_per_year"]) and row["adjusted_ci_low"] > 1
    assert row["note"] == ""


def test_rare_tags_and_thin_years_are_skipped():
    papers = _papers(lambda year, cluster: 0.002)
    row = tag_trends(papers, "themes", ["privacy", "bias"]).set_index("tag")
    assert np.isnan(row.loc["bias", "or_per_year"]) and "fewer than 20" in row.loc["bias", "note"]
    sparse = pd.concat([papers, pd.DataFrame({"year": [2030], "month": [1], "cluster": [0],
                                              "themes": [["privacy"]]})])  # fmt: skip
    assert tag_trends(sparse, "themes", ["privacy"])["years"].iloc[0] == "2010-2024"


def test_decimal_year_and_q_values():
    papers = pd.DataFrame({"year": pd.array([2020, 2021, None], dtype="Int16"),
                           "month": pd.array([1, None, 5], dtype="Int8")})  # fmt: skip
    assert decimal_year(papers)[:2] == pytest.approx([2020 + 0.5 / 12, 2021.5])
    assert np.isnan(decimal_year(papers)[2])
    q = benjamini_hochberg([0.01, 0.04, np.nan, 0.03])
    assert q[0] == pytest.approx(0.03) and q[1] == q[3] == pytest.approx(0.04)
    assert np.isnan(q[2])
