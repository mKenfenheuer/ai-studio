"""What "use at most 40% of the card" means, in one place.

Shared because both ends have to agree about it and they agree about nothing
else: the controller validates what somebody typed and plans a batch against
the result, the runner enforces it. Two copies of "what counts as a valid
percentage" is two answers to "will this run fit", and the disagreement would
show up as an out-of-memory crash an hour into a download.

No I/O here, and nothing torch-shaped -- the controller installs four
pure-Python packages on purpose. Enforcement lives in `runner/limits.py`,
which is also where the difference between the two limits is explained.
"""
from __future__ import annotations

# Below this there is no run, only a pause with a progress bar: a 4% duty
# cycle turns a twenty-minute fine-tune into eight hours. Anything lower is
# much more likely to be a typo or a misread slider than an intention.
MIN_COMPUTE_PCT = 5

# Torch accepts a memory fraction near zero and then fails to allocate the
# model, with an error that blames the model. A card is not useful for
# training at a tenth of itself, and the refusal is clearer here than there.
MIN_MEMORY_PCT = 10

UNLIMITED = {"memory_pct": 100, "compute_pct": 100}


def _pct(value: object, floor: int) -> int:
    """A percentage, or 100 for anything that is not a usable one."""
    try:
        n = int(round(float(value)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 100
    if n >= 100:
        return 100
    return max(floor, n)


def clean(limits: dict | None) -> dict:
    """Normalise whatever arrived into the two percentages that mean something."""
    limits = limits or {}
    return {
        "memory_pct": _pct(limits.get("memory_pct"), MIN_MEMORY_PCT),
        "compute_pct": _pct(limits.get("compute_pct"), MIN_COMPUTE_PCT),
    }


def is_unlimited(limits: dict | None) -> bool:
    return clean(limits) == UNLIMITED


def describe(limits: dict | None) -> str:
    """One line of English, or "" when there is nothing to say."""
    lim = clean(limits)
    parts = []
    if lim["memory_pct"] < 100:
        parts.append("at most %d%% of its memory" % lim["memory_pct"])
    if lim["compute_pct"] < 100:
        parts.append("about %d%% of its time" % lim["compute_pct"])
    return " and ".join(parts)


def budget_gb(vram_gb: float | None, limits: dict | None) -> float | None:
    """The memory a run may actually be planned against.

    Separate from the card's size on purpose. Both numbers are shown -- "12 GB
    of 16 GB" -- because a machine that reports 12 GB with no explanation, on a
    card somebody knows is 16, reads as a bug in the probe.
    """
    if vram_gb is None:
        return None
    pct = clean(limits)["memory_pct"]
    return round(float(vram_gb) * pct / 100.0, 1) if pct < 100 else float(vram_gb)
