"""How fast each theme or concept rises or falls, and how much of that is a shift in topics.

A theme's share of papers can grow for two reasons: papers within each topic mention it more
often, or topics that always mentioned it more grow faster (a composition effect). For each tag,
two logistic regressions of "the paper has the tag" on time are fitted:

* unadjusted: ``tag ~ time``, the plain trend;
* adjusted: ``tag ~ time + topic cluster (+ set)``, the trend within topics.

Time is the decimal year of the paper (year + month), so the coefficient is the change in
log-odds per year; tables report it as an odds ratio per year with a 95% interval and p-value,
plus the adjusted change in percentage points per year (the average marginal effect). If the
adjusted odds ratio is much closer to 1 than the unadjusted one, the change mostly reflects a
shift between topics rather than within them.

Cluster and set effects get a weakly informative normal prior (sd 2.5 on the log-odds scale, as
recommended by Gelman et al. 2008), so a cluster in which a tag never occurs does not make the
estimate diverge. Time and the intercept are not penalised.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

Z = 1.959964  # two-sided 95% normal quantile
PRIOR_SD = 2.5


@dataclass
class LogitFit:
    coef: np.ndarray
    cov: np.ndarray
    probabilities: np.ndarray
    converged: bool
    iterations: int

    def se(self, j: int) -> float:
        return float(math.sqrt(max(self.cov[j, j], 0.0)))


def fit_logit(
    x: np.ndarray,
    y: np.ndarray,
    penalized: Sequence[bool] | None = None,
    prior_sd: float = PRIOR_SD,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> LogitFit:
    """Logistic regression by Newton-Raphson; columns flagged in ``penalized`` get a normal
    prior with standard deviation ``prior_sd``. The covariance is the inverse Hessian."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    n, k = x.shape
    penalty = np.zeros(k)
    if penalized is not None:
        penalty[np.asarray(penalized, dtype=bool)] = 1.0 / prior_sd**2
    beta = np.zeros(k)
    mean = y.mean()
    if 0 < mean < 1:
        beta[0] = math.log(mean / (1 - mean))  # start at the overall share (column 0: intercept)
    converged, iterations = False, 0
    while not converged and iterations < max_iter:
        iterations += 1
        p = 1.0 / (1.0 + np.exp(-(x @ beta)))
        w = p * (1.0 - p)
        gradient = x.T @ (y - p) - penalty * beta
        hessian = (x * w[:, None]).T @ x + np.diag(penalty)
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        # Halve the step while it lowers the penalised log-likelihood (keeps Newton stable).
        current = _penalised_loglik(x, y, beta, penalty)
        scale = 1.0
        while scale > 1e-4 and _penalised_loglik(x, y, beta + scale * step, penalty) < current:
            scale /= 2
        beta = beta + scale * step
        converged = bool(np.max(np.abs(scale * step)) < tol)
    p = 1.0 / (1.0 + np.exp(-(x @ beta)))
    w = p * (1.0 - p)
    hessian = (x * w[:, None]).T @ x + np.diag(penalty)
    cov = np.linalg.pinv(hessian)
    return LogitFit(beta, cov, p, converged, iterations)


def _penalised_loglik(x: np.ndarray, y: np.ndarray, beta: np.ndarray, penalty: np.ndarray) -> float:
    eta = x @ beta
    loglik = float(np.sum(y * eta - np.logaddexp(0.0, eta)))
    return loglik - 0.5 * float(np.sum(penalty * beta**2))


def decimal_year(papers: pd.DataFrame) -> np.ndarray:
    """Year plus the middle of the month (or of the year, if the month is unknown)."""
    year = papers["year"].astype("Float64").to_numpy(dtype=float, na_value=np.nan)
    if "month" in papers:
        month = papers["month"].astype("Float64").to_numpy(dtype=float, na_value=np.nan)
    else:
        month = np.full(len(papers), np.nan)
    return year + np.where(np.isnan(month), 0.5, (month - 0.5) / 12)


def _dummies(values: pd.Series) -> tuple[np.ndarray, list[str]]:
    """Indicator columns for every level but the most common one (the reference)."""
    levels = values.astype(str)
    counts = levels.value_counts()
    others = sorted(counts.index[1:])
    if not others:
        return np.zeros((len(values), 0)), []
    return np.column_stack([(levels == v).to_numpy(float) for v in others]), others


def _p_value(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2))


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    """False-discovery-rate q-values (NaN p-values stay NaN): with many tags tested, some
    p-values fall below 0.05 by chance alone."""
    p = np.asarray(p_values, dtype=float)
    q = np.full(p.shape, np.nan)
    valid = np.flatnonzero(~np.isnan(p))
    if valid.size:
        order = valid[np.argsort(p[valid])]
        ranked = p[order] * valid.size / np.arange(1, valid.size + 1)
        q[order] = np.minimum(1.0, np.minimum.accumulate(ranked[::-1])[::-1])
    return q


def _effect(fit: LogitFit, j: int, prefix: str) -> dict:
    beta, se = float(fit.coef[j]), fit.se(j)
    return {
        f"{prefix}or_per_year": math.exp(beta),
        f"{prefix}ci_low": math.exp(beta - Z * se),
        f"{prefix}ci_high": math.exp(beta + Z * se),
        f"{prefix}p_value": _p_value(beta / se) if se > 0 else math.nan,
    }


def tag_trends(
    papers: pd.DataFrame,
    column: str,
    names: Sequence[str],
    adjust_for: Sequence[str] = ("cluster",),
    min_papers: int = 20,
    min_year_papers: int = 20,
) -> pd.DataFrame:
    """One row per tag: unadjusted and adjusted odds ratios per year, and the adjusted change
    in percentage points per year. Tags with fewer than ``min_papers`` papers with (or
    without) them are listed without estimates; years with fewer than ``min_year_papers``
    papers are left out of the models."""
    time = decimal_year(papers)
    years = papers["year"].astype("Float64")
    counts = years.value_counts()
    well_covered = years.isin(counts.index[counts >= min_year_papers])
    keep = ~np.isnan(time) & well_covered.to_numpy(dtype=bool, na_value=False)
    data = papers[keep].reset_index(drop=True)
    t = time[keep]
    centre = float(np.median(t)) if len(t) else 0.0
    base = np.column_stack([np.ones(len(data)), t - centre])
    adjust = [np.zeros((len(data), 0))]
    for name in adjust_for:
        if name in data and data[name].nunique() > 1:
            adjust.append(_dummies(data[name])[0])
    extra = np.hstack(adjust)
    adjusted_x = np.hstack([base, extra])
    penalized = [False, False] + [True] * extra.shape[1]
    first, last = (int(years[keep].min()), int(years[keep].max())) if keep.any() else (0, 0)

    rows = []
    for tag in names:
        y = data[column].map(lambda tags, t=tag: t in tags).to_numpy(dtype=float)
        n_tag = int(y.sum())
        row = {
            "tag": tag,
            "papers": n_tag,
            "share": n_tag / len(y) if len(y) else math.nan,
            "years": f"{first}-{last}" if len(y) else "",
            "note": "",
        }
        if n_tag < min_papers or len(y) - n_tag < min_papers:
            row["note"] = f"fewer than {min_papers} papers with or without the tag"
            rows.append(row)
            continue
        plain = fit_logit(base, y)
        within = fit_logit(adjusted_x, y, penalized)
        row.update(_effect(plain, 1, ""))
        row.update(_effect(within, 1, "adjusted_"))
        slope = float(np.mean(within.probabilities * (1 - within.probabilities)))
        beta, se = float(within.coef[1]), within.se(1)
        row["adjusted_pp_per_year"] = 100 * slope * beta
        row["adjusted_pp_ci_low"] = 100 * slope * (beta - Z * se)
        row["adjusted_pp_ci_high"] = 100 * slope * (beta + Z * se)
        if not (plain.converged and within.converged):
            row["note"] = "model did not converge; treat with caution"
        rows.append(row)
    columns = [
        "tag", "papers", "share", "years", "or_per_year", "ci_low", "ci_high", "p_value",
        "adjusted_or_per_year", "adjusted_ci_low", "adjusted_ci_high", "adjusted_p_value",
        "adjusted_q_value", "adjusted_pp_per_year", "adjusted_pp_ci_low", "adjusted_pp_ci_high",
        "note",
    ]  # fmt: skip
    table = pd.DataFrame(rows).reindex(columns=columns)
    table["adjusted_q_value"] = benjamini_hochberg(table["adjusted_p_value"])
    return table
