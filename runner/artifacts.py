"""Fetching a finished result from the controller onto this machine.

Models are stored on the controller, not on the runner that made them. That is
what makes a model in this studio portable: the machine that trained it may be
switched off, reinstalled, or a laptop somebody took home, and the result is
still there and still usable by whichever runner is free.

The playground has always worked this way. Training did not -- a run could
only start from a Hugging Face id, so the models you built here were the one
kind of model you could not build on. This module is that path, shared by both
so there is a single cache and a single set of rules about it.
"""
from __future__ import annotations

import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Callable

import httpx

CACHE_DIR = Path(os.environ.get("AI_STUDIO_MODEL_CACHE", "/data/models"))


def cached_dir(job_id: str) -> Path:
    return CACHE_DIR / job_id


def is_present(job_id: str) -> bool:
    d = cached_dir(job_id)
    return (d / "config.json").exists() or (d / "adapter_config.json").exists()


def fetch(controller_url: str, token: str, job_id: str,
          log: Callable[[str], None] = lambda _s: None) -> Path:
    """This run's result as a directory on local disk, downloading if needed.

    Unpacked into a staging directory and moved into place, so an interrupted
    download cannot leave a half-extracted model that `is_present` then
    reports as ready -- the failure that would produce is a missing-weights
    error hours later, blamed on the model rather than on the download.
    """
    dest = cached_dir(job_id)
    if is_present(job_id):
        return dest

    log("Fetching the trained model from the studio...")
    staging = dest.with_name(dest.name + ".partial")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)
    zip_path = staging / "artifact.zip"

    url = "%s/api/jobs/%s/download" % (controller_url.rstrip("/"), job_id)
    size = 0
    with httpx.stream("GET", url, timeout=1800, follow_redirects=True,
                      headers={"X-Runner-Token": token}) as r:
        if r.status_code in (401, 403):
            raise ValueError(
                "The controller would not hand over that model. The runner's "
                "join token was refused, which usually means it was rotated "
                "since this machine last joined.")
        if r.status_code == 404:
            raise ValueError(
                "That run has no saved model on the controller any more. It "
                "may have been deleted.")
        r.raise_for_status()
        with open(zip_path, "wb") as fh:
            for chunk in r.iter_bytes(1 << 20):
                fh.write(chunk)
                size += len(chunk)

    with zipfile.ZipFile(zip_path) as z:
        z.extractall(staging)
    zip_path.unlink(missing_ok=True)

    shutil.rmtree(dest, ignore_errors=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(dest)
    log("Got %.0f MB." % (size / 1048576))
    return dest


def summary(job_id: str) -> dict:
    """What the run that produced this recorded about itself, if anything."""
    try:
        return json.loads(
            (cached_dir(job_id) / "ai_studio_summary.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def clear(job_id: str | None = None) -> None:
    if job_id:
        shutil.rmtree(cached_dir(job_id), ignore_errors=True)
        shutil.rmtree(cached_dir(job_id).with_name(job_id + ".partial"),
                      ignore_errors=True)
    else:
        shutil.rmtree(CACHE_DIR, ignore_errors=True)
