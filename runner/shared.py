"""A directory this runner and its controller both mount.

Without one, a finished model travels three times: packed into the job's
working directory, sent over HTTP, and written again on the controller. For a
6 GB merge that is 12 GB of disk at the worst moment of the run, which on one
host with a nearly full volume is exactly where a run failed with ENOSPC after
training had finished. Fetching a model back costs the same again, as a zip
beside the directory it unpacks into.

With `AI_STUDIO_SHARED_DIR` set on both sides -- one host with a shared volume,
or NFS/SMB across several -- the runner packs its result straight into the
controller's storage and reads models out of it without a download. The
Hugging Face cache moves there too, so a base model is on the disk once for
every runner instead of once each.

A path is not proof. A runner whose mount failed sees an empty directory of the
right name and would write results nobody can see. So the controller writes a
random id into the directory and sends it at registration, and the share is
used only when the runner reads the same id. Anything else, and everything
falls back to HTTP as before.
"""
from __future__ import annotations

import os
from pathlib import Path

MARKER = ".ai-studio-share"

SHARED_DIR: Path | None = (Path(os.environ["AI_STUDIO_SHARED_DIR"]).expanduser()
                           if os.environ.get("AI_STUDIO_SHARED_DIR") else None)

_active = False


def configure_env() -> None:
    """Point the Hugging Face cache into the share, before anything imports it.

    Only when nobody chose a cache explicitly. Called at startup, before the
    marker can be checked: the Hub cache is safe for several processes at once
    and works the same whether or not the controller can see it.
    """
    if SHARED_DIR and not os.environ.get("HF_HUB_CACHE"):
        os.environ["HF_HUB_CACHE"] = str(SHARED_DIR / "hf")


def confirm(share_id: str | None) -> str | None:
    """Check the controller's id against the marker. Returns a line to print."""
    global _active
    _active = False
    if not SHARED_DIR:
        return None
    if not share_id:
        return ("[runner] AI_STUDIO_SHARED_DIR is set, but the controller has "
                "no shared directory; models travel over HTTP.")
    try:
        found = (SHARED_DIR / MARKER).read_text(encoding="utf-8").strip()
    except OSError:
        found = ""
    if found != share_id:
        return ("[runner] %s is not the controller's shared directory (its "
                "marker is %s); models travel over HTTP."
                % (SHARED_DIR, "missing" if not found else "different"))
    try:
        incoming_dir().mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return ("[runner] the shared directory %s is not writable (%s); "
                "models travel over HTTP." % (SHARED_DIR, e))
    _active = True
    return "[runner] using the shared directory %s" % SHARED_DIR


def active() -> bool:
    return _active


def incoming_dir() -> Path:
    assert SHARED_DIR is not None
    return SHARED_DIR / "incoming"


def contains(path: Path | str) -> bool:
    """Whether a path lies inside the share, which other runners use too."""
    if not SHARED_DIR:
        return False
    try:
        Path(path).resolve().relative_to(SHARED_DIR.resolve())
        return True
    except (OSError, ValueError):
        return False


def result_path(ctx, filename: str) -> Path:
    """Where a job packs its result: the share when there is one, else workdir.

    In the share it goes to `incoming/`, named for the job, and becomes an
    artifact only when the controller moves it -- after the zip is complete.
    """
    if _active:
        return incoming_dir() / ("%s-%s" % (ctx.job_id, filename))
    return Path(ctx.workdir) / filename


def is_incoming(path: Path | str) -> bool:
    return bool(_active and SHARED_DIR) and Path(path).parent == incoming_dir()


def artifact_zip(job_id: str, kind: str | None = None) -> Path | None:
    """A run's stored zip, readable in place, or None to download it instead."""
    if not _active:
        return None
    name = ("%s.zip" % job_id if kind in (None, "", "model")
            else "%s-%s.zip" % (job_id, kind))
    path = SHARED_DIR / "artifacts" / name
    return path if path.is_file() else None
