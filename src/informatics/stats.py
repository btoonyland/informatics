"""Small statistics helpers."""

from __future__ import annotations

import math


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
