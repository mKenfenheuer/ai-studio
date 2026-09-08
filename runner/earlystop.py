"""Knowing when to stop, and which model to keep.

Two separate ideas that are usually conflated, and both are now possible
because every run finally measures a held-out loss:

* **Stopping.** Past the point where held-out loss stops improving, further
  training makes the model worse while the training loss keeps falling. The
  hours after that point are not merely wasted, they are harmful.

* **Keeping the best one.** Even a run that finishes on schedule usually has
  its best model somewhere before the end. Saving the *last* weights when a
  better set existed at step 400 throws away the better model for no reason.
  This is the more valuable half, and it applies to runs that were never going
  to stop early at all.

Patience is counted in evaluations rather than in steps, because `eval_every`
is derived from the length of the run: "stop after 200 steps without
improvement" means something different in a 300-step run and a 30,000-step
one, while "stop after four checks without improvement" does not.

Improvement is relative, not absolute. A gain of 0.001 on a loss of 5 is
noise; the same gain on a loss of 0.05 is a 2% improvement. A single absolute
threshold cannot be right for both, and this app trains models whose losses
span that whole range.
"""
from __future__ import annotations

from typing import Any

# Below this, a "better" number is not better in any way that survives being
# measured again. 0.1% of the current best.
REL_DELTA = 0.001

# Defaults differ by what the two kinds of run actually do. A fine-tune on a
# few hundred examples starts memorising within a couple of passes, so it
# needs a short leash. Pretraining on a large corpus improves for as long as
# there is text left, so a long one -- there it is a safety net for the case
# where the corpus is small enough to be memorised, not an expected outcome.
PATIENCE_FINETUNE = 4
PATIENCE_PRETRAIN = 8


class Stopper:
    """Watches the held-out loss and says when to stop and what to keep."""

    def __init__(self, ctx: Any, patience: int, enabled: bool = True,
                 rel_delta: float = REL_DELTA, kind: str = "model"):
        self.ctx = ctx
        self.patience = max(1, int(patience))
        self.enabled = bool(enabled)
        self.rel_delta = float(rel_delta)
        self.kind = kind
        self.best: float | None = None
        self.best_step = 0
        self.waited = 0
        self.stopped = False
        self.announced = False

    def announce(self, eval_every: int) -> None:
        if not self.enabled:
            return
        self.ctx.log(
            "Watching the held-out loss. If it has not improved after %d "
            "checks in a row -- that is %d steps -- training stops there, and "
            "whichever %s scored best is the one kept."
            % (self.patience, self.patience * max(eval_every, 1), self.kind))

    def update(self, step: int, val_loss: float | None) -> str:
        """Record a held-out reading. Returns "improved", "stop", or "".

        The caller saves a snapshot on "improved" and breaks out of its loop
        on "stop"; deciding *which* of those it is belongs here, so the two
        trainers cannot drift apart on it.
        """
        if val_loss is None:
            return ""
        if self.best is None or val_loss < self.best * (1 - self.rel_delta):
            self.best = val_loss
            self.best_step = step
            self.waited = 0
            return "improved"

        # Still the best number seen, just not by enough to count as progress.
        if self.best is not None and val_loss < self.best:
            self.best = val_loss

        self.waited += 1
        if not self.enabled or self.waited < self.patience:
            return ""
        self.stopped = True
        self.ctx.log(
            "Stopping: the held-out loss has not improved for %d checks. Its "
            "best was %.4f at step %d, and it has been %.4f since. More "
            "training from here makes the %s worse at everything except the "
            "examples it has already seen."
            % (self.patience, self.best, self.best_step, val_loss, self.kind),
            "warn")
        return "stop"

    def should_restore(self, final_val: float | None, step: int) -> bool:
        """Is the snapshot worth going back for?

        Only when there is a distinctly better one. Restoring a model that is
        better by a hundredth of a percent trades a real cost -- the weights
        with a fully decayed learning rate -- for a difference nobody can
        measure.
        """
        if self.best is None or not self.best_step or self.best_step >= step:
            return False
        if final_val is None:
            # Nothing to compare the snapshot against, so keep what training
            # ended with. Guessing in the dark here is how a run silently
            # ships a model from a third of the way through.
            return False
        return final_val > self.best * (1 + self.rel_delta)

    def note_kept(self, final_val: float, step: int) -> None:
        self.ctx.log(
            "Keeping the model from step %d rather than step %d: it scored "
            "%.4f on held-out data against %.4f at the end. The last weights "
            "are not automatically the best ones."
            % (self.best_step, step, self.best, final_val))
