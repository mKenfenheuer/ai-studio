"""Small statistics both halves of the studio need to agree about.

Kept here rather than in either half because a number computed one way on the
runner and another way in the browser is the kind of disagreement nobody
notices until two pages show different confidence intervals for the same
score.
"""
from __future__ import annotations


def wilson(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """A 95% interval on an accuracy, the Wilson way.

    Reported with every benchmark result, because a sampled benchmark has a
    margin wider than most of the differences people read off these tables:
    four hundred questions put about four points either side of any score near
    the middle, so two models three points apart have not been separated by
    it. The normal approximation everybody uses is badly wrong near 0 and 1 --
    it happily reports a negative lower bound -- and near 0 is exactly where a
    small model's GSM8K score lands.
    """
    if total <= 0:
        return 0.0, 1.0
    p = correct / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * ((p * (1 - p) / total + z * z / (4 * total * total)) ** 0.5) / denom
    return max(0.0, centre - half), min(1.0, centre + half)
