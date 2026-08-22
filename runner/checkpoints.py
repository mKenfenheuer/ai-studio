"""Saving a run's progress, so that losing the machine does not lose the work.

The case this exists for is mundane and expensive: the runner is restarted --
a deploy, a crash, a power cut, a `docker compose up -d` typed by somebody who
forgot to look -- and a run that was four hours in starts again from random
noise. Nothing about that is a hardware limit. The state needed to carry on is
a few hundred megabytes and takes seconds to write.

Three decisions shape this module:

* **Checkpoints live outside the job's working directory.** The agent creates
  a fresh temporary directory per job and deletes it in a `finally`, which is
  correct for scratch space and exactly wrong for the one thing that has to
  outlive the process. They go on the runner's data volume instead, next to
  the model cache.

* **The current checkpoint is named by a single file that is replaced
  atomically.** Each save writes a whole new generation directory and only
  then swaps `state.json` over it with `os.replace`. A checkpoint is therefore
  either fully written and current, or not current at all -- there is no state
  in which a resume reads half a model. Writing in place would create exactly
  that state, and it would be found by a power cut rather than by a test.

* **The expensive-but-unchanging parts are kept once, not per generation.**
  A from-scratch run spends real time training a tokenizer and tokenizing a
  corpus before a single step happens, and neither changes as training
  proceeds. They are stored beside the generations and reused, so resuming
  skips the preparation as well as the training.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

CHECKPOINT_DIR = Path(os.environ.get("AI_STUDIO_CHECKPOINTS", "/data/checkpoints"))

# A checkpoint for a run nobody ever came back to is dead weight on the disk.
# Long enough to survive a weekend, short enough not to fill a volume.
MAX_AGE_S = 14 * 24 * 3600


def dir_for(job_id: str) -> Path:
    return CHECKPOINT_DIR / job_id


def _state_file(job_id: str) -> Path:
    return dir_for(job_id) / "state.json"


def peek(job_id: str) -> dict | None:
    """What a resume would start from, or None if there is nothing to resume.

    Cheap on purpose -- it reads one small file. The agent calls it before
    starting a job, so it can tell the controller which step this attempt is
    beginning at before any of the heavy machinery is imported.
    """
    try:
        state = json.loads(_state_file(job_id).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    payload = dir_for(job_id) / str(state.get("dir") or "")
    if not state.get("dir") or not payload.is_dir():
        return None
    state["path"] = str(payload)
    return state


def payload_dir(job_id: str) -> Path | None:
    state = peek(job_id)
    return Path(state["path"]) if state else None


def prepared_dir(job_id: str) -> Path:
    """Where the parts that do not change as training proceeds are kept.

    The tokenizer and the tokenized corpus, for a from-scratch run. Created on
    demand; the caller decides what goes in it and whether it is complete.
    """
    d = dir_for(job_id) / "prepared"
    d.mkdir(parents=True, exist_ok=True)
    return d


def new_generation(job_id: str, step: int) -> Path:
    """A fresh directory to write a checkpoint into. Not current until committed."""
    d = dir_for(job_id) / ("ckpt-%08d" % int(step))
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)
    return d


def commit(job_id: str, generation: Path, state: dict) -> None:
    """Make this generation the one a resume will find, then drop the old one.

    The replace is the whole point: until this line runs, an interrupted save
    leaves the previous checkpoint intact and current.
    """
    previous = peek(job_id)
    body = dict(state)
    body["dir"] = generation.name
    body["saved_at"] = time.time()

    tmp = _state_file(job_id).with_suffix(".json.tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(body, indent=2), encoding="utf-8")
    os.replace(tmp, _state_file(job_id))

    if previous and previous.get("dir") != generation.name:
        shutil.rmtree(dir_for(job_id) / str(previous["dir"]), ignore_errors=True)


def discard(job_id: str) -> None:
    shutil.rmtree(dir_for(job_id), ignore_errors=True)


def list_ids() -> list[str]:
    """Runs this machine could resume.

    Reported to the controller so that work goes back to the machine holding
    its progress rather than to whichever one happens to be free first.
    """
    try:
        return sorted(d.name for d in CHECKPOINT_DIR.iterdir()
                      if d.is_dir() and (d / "state.json").exists())
    except OSError:
        return []


def size_bytes(job_id: str, path: Path | None = None) -> int:
    """Bytes on disk, for the whole job or for one generation.

    The distinction is not pedantic: the prepared corpus dwarfs the checkpoint
    itself, so reporting the total as "the checkpoint" tells the user a save
    costs three times what it does.
    """
    total = 0
    for root, _dirs, files in os.walk(path or dir_for(job_id)):
        for f in files:
            try:
                total += (Path(root) / f).stat().st_size
            except OSError:
                pass
    return total


def prune(max_age_s: float = MAX_AGE_S) -> list[str]:
    """Forget checkpoints nobody came back for. Returns what was removed."""
    dropped = []
    cutoff = time.time() - max_age_s
    for job_id in list_ids():
        state = peek(job_id)
        if state and float(state.get("saved_at") or 0) < cutoff:
            discard(job_id)
            dropped.append(job_id)
    return dropped


class Saver:
    """Decides when to write, and writes it. One per training run.

    Time-based rather than step-based because a step is not a fixed amount of
    work: the same "every 50 steps" is a minute on one model and forty on
    another. What the user actually cares about is how much time an
    interruption can cost them, and that is what this is expressed in.
    """

    def __init__(self, job_id: str, ctx: Any, every_s: float = 600.0,
                 enabled: bool = True):
        self.job_id = job_id
        self.ctx = ctx
        self.every_s = max(30.0, float(every_s))
        self.enabled = enabled
        self.last = time.time()
        self.saved_step = 0
        self.count = 0

    def due(self, step: int) -> bool:
        return self.enabled and step > self.saved_step \
            and (time.time() - self.last) >= self.every_s

    def write(self, step: int, total: int, writer, extra: dict | None = None) -> bool:
        """`writer(path)` puts the state in `path`; everything else is here."""
        if not self.enabled:
            return False
        t0 = time.time()
        try:
            gen = new_generation(self.job_id, step)
            writer(gen)
            written = size_bytes(self.job_id, gen)
            commit(self.job_id, gen, {"step": step, "total": total,
                                      **(extra or {})})
        except Exception as e:  # noqa: BLE001 - a failed save must not kill a run
            # Deliberately not fatal, and deliberately not silent. Losing the
            # ability to resume is worth a warning; losing four hours of
            # training because the disk filled up while writing a safety net
            # is not.
            self.ctx.log("Could not write a checkpoint (%s). Training carries "
                         "on, but an interruption would now restart this run "
                         "from the beginning." % e, "warn")
            self.enabled = False
            return False
        self.last = time.time()
        self.saved_step = step
        self.count += 1
        # Announced once. After that it is routine, and a line every ten
        # minutes for eight hours is a log nobody reads.
        if self.count == 1:
            mins = max(1, round(self.every_s / 60))
            self.ctx.log("Checkpoint saved at step %d: %.0f MB in %.2fs, and "
                         "%s from here. An interruption now resumes from the "
                         "last checkpoint instead of starting over."
                         % (step, written / 1048576, time.time() - t0,
                            "every minute" if mins == 1
                            else "every %d minutes" % mins))
        return True
