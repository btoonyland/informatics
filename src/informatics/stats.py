"""Small statistics helpers."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion (well-behaved for small n and shares near 0/1)."""
    if n == 0:
        return math.nan, math.nan
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    # The interval always contains p; clamping removes rounding error at p = 0 or 1.
    return min(max(0.0, centre - half), p), max(min(1.0, centre + half), p)


def cohen_kappa(a: Sequence[bool], b: Sequence[bool]) -> float:
    """Cohen's kappa: agreement between two sets of yes/no decisions beyond what chance gives
    (1 = perfect, 0 = chance level). NaN when there is nothing to compare."""
    x, y = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    if x.size == 0:
        return math.nan
    observed = float(np.mean(x == y))
    expected = float(x.mean() * y.mean() + (1 - x.mean()) * (1 - y.mean()))
    if expected == 1.0:  # both always say the same single answer
        return 1.0 if observed == 1.0 else math.nan
    return (observed - expected) / (1 - expected)
