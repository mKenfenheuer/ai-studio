"""How much of this machine's card the studio may use.

Two limits, two entirely different mechanisms, and the difference is the most
important thing on this page because only one of them is a real cap.

**Memory is enforced.** torch is told the fraction of the card it may allocate
and refuses to go past it, so a run that would have taken the whole card fails
with an out-of-memory error instead of evicting whatever else was on it. The
controller plans against the same fraction, so it sizes the batch to fit
inside the limit rather than inside the card -- which means the usual outcome
of a memory limit is a smaller batch, not a failure.

**Compute is a duty cycle, not a partition.** Neither CUDA nor ROCm will sell
you 40% of a consumer card: there is no hardware mechanism, MPS is NVIDIA-only
and needs a daemon, and nothing equivalent exists on RDNA. What can be done
honestly is to work in bursts -- after each optimiser step the trainer sleeps
for a proportion of the time that step took, so the card averages the share
asked for. 50% means the card is busy about half the time and the run takes
about twice as long.

That is worth having for the reasons people actually ask for it: the machine
is somebody's desktop and the display should stay smooth, the room or the PSU
cannot take a card at 100% for six hours, or a second workload on the same
card has to stay responsive. It is not worth pretending it is isolation, so
the UI says "about" and this module does not round the number up.

Throttling applies to training, not to answering. Stuttering a chat reply to
save power is a worse thing to do to somebody than letting a model that is
already resident finish the sentence.
"""
from __future__ import annotations

import os
import time

from common import gpulimits

# A single sleep is capped, because the gap between two optimiser steps is not
# always an optimiser step. An evaluation pass, a checkpoint write or a sample
# generation lands between them, and a duty cycle computed from "time since I
# was last here" would answer a 90-second checkpoint with a 90-second sleep --
# parking the card for a minute and a half to pay for work that was not
# training. The cap keeps the throttle honest about steps and quietly wrong
# about the occasional long interval, which is the right way round.
MAX_SLEEP_S = 10.0

# What a valid percentage is, and what it reads as, are the controller's
# business too -- it validates the same numbers and plans against them. So
# they live in common/ and are re-exported here rather than written twice.
clean = gpulimits.clean
describe = gpulimits.describe


def from_env() -> dict:
    """What this machine's own configuration says it may give.

    Read here rather than on the controller because it belongs to whoever runs
    the machine: a box lent to the studio on the condition that it stays usable
    says so in its own compose file, and does not depend on anybody remembering
    to set it again in a web page.
    """
    return clean({
        "memory_pct": os.environ.get("AI_STUDIO_GPU_MEMORY_PCT"),
        "compute_pct": os.environ.get("AI_STUDIO_GPU_COMPUTE_PCT"),
    })


def apply_memory(limits: dict | None) -> str:
    """Hold this process to its share of the card. Returns what it did.

    Process-wide and idempotent, which is why it is called at the start of
    every job rather than once at boot: the limit can change between one run
    and the next, and a job is the only moment at which a new one has to hold.
    """
    pct = clean(limits)["memory_pct"]
    if pct >= 100:
        return ""
    try:
        import torch
        if not torch.cuda.is_available():
            return ""
        torch.cuda.set_per_process_memory_fraction(pct / 100.0, 0)
    except Exception as e:  # noqa: BLE001 -- a limit must never fail a run
        # Deliberately not fatal. The controller has already planned the run
        # against the smaller figure, so the run still fits; what is lost is
        # the guarantee, and saying so is more useful than refusing to train.
        return "could not hold this run to %d%% of the card (%s); it is sized " \
               "for that share but not confined to it" % (pct, e.__class__.__name__)
    return "holding this run to %d%% of the card's memory" % pct


class Throttle:
    """Sleeps between optimiser steps to hold a compute duty cycle.

    Constructed even when there is no limit, so the call site in the training
    loop is one unconditional line rather than an `if` around every use.
    """

    def __init__(self, limits: dict | None = None) -> None:
        self.pct = clean(limits)["compute_pct"]
        self._last: float | None = None
        self.slept_s = 0.0
        self.steps = 0

    @property
    def active(self) -> bool:
        return self.pct < 100

    def step(self) -> float:
        """Call once per optimiser step. Returns the seconds slept."""
        now = time.time()
        if not self.active:
            self._last = now
            return 0.0
        if self._last is None:          # first step: nothing to measure yet
            self._last = now
            return 0.0
        busy = now - self._last
        idle = min(busy * (100.0 / self.pct - 1.0), MAX_SLEEP_S)
        if idle > 0:
            time.sleep(idle)
            self.slept_s += idle
            self.steps += 1
        self._last = time.time()
        return idle

    def summary(self) -> str:
        if not self.active or not self.steps:
            return ""
        return ("held to about %d%% of the card: %d steps paused, %.0fs idle"
                % (self.pct, self.steps, self.slept_s))
