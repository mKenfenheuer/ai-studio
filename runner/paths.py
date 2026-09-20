"""Where this runner keeps the things that must outlive a job.

Checkpoints and the model cache, which are the two directories a run cannot
recreate cheaply: a checkpoint is the only reason a four-hour run survives a
restart, and the cache is the reason the same base model is not downloaded
again for every job.

The default used to be `/data`, which is not a default so much as one
deployment's answer written down. It is where the container images mount their
volume, and it is correct there and nowhere else. A runner installed natively
-- by `scripts/install-runner.sh`, which is the *required* path on macOS
because Metal has no container story -- inherited it and tried to write to a
directory it could not create. On macOS that fails hardest of all: the root
volume is read-only and sealed, so `/data` cannot be brought into existence
even as root.

Nothing crashed, because every caller treats a failed save as survivable, and
that is exactly what made it worth fixing rather than leaving. A run trained
for five hours with no checkpoint behind it, so a restart would have lost all
of it, and if early stopping had wanted to roll back to the best step there
was no snapshot to roll back to -- it would have kept the more overfitted
weights and said nothing. Silent, and only visible as a slightly worse model.

So the location is resolved instead of assumed, and `/data` keeps winning
wherever it is real: the containers are unchanged. `AI_STUDIO_DATA` is the
same name the controller already uses for the same idea.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

# The container mount. Preferred when it is genuinely there and writable, so
# an image that has always used it carries on doing so.
_CONTAINER_DATA = Path("/data")


def _usable(path: Path) -> bool:
    """Whether a directory exists and this process may actually write to it.

    `os.access` rather than a try/write: this runs at import, and the answer
    is wanted for a directory that may hold gigabytes somebody else put there.
    """
    try:
        return path.is_dir() and os.access(path, os.W_OK)
    except OSError:
        return False


def data_root() -> Path:
    """The base directory for this runner's persistent state.

    In order: an explicit `AI_STUDIO_DATA`, then `/data` when it is real and
    writable, then a per-user directory. The last of those is the native
    install's answer and needs no configuration, which is the point -- the
    previous default required an environment variable nobody was told to set.
    """
    if explicit := os.environ.get("AI_STUDIO_DATA"):
        return Path(explicit).expanduser()
    if _usable(_CONTAINER_DATA):
        return _CONTAINER_DATA
    return Path.home() / ".ai-studio" / "data"


def ensure(path: Path) -> Path:
    """Create a directory, falling back to scratch space rather than raising.

    Called at import time by the modules that own these directories, so it
    must not be able to stop a runner from starting. A runner that cannot
    write its cache anywhere still trains; it just re-downloads, which is the
    behaviour that was already being survived before any of this was resolved.
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
        return path
    except OSError:
        fallback = Path(tempfile.gettempdir()) / "ai-studio" / path.name
        try:
            fallback.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        return fallback
