"""Getting at the data a job was pointed at.

A dataset in the studio's own library is private: reaching it needs the join
token, and `load_dataset` has no way to send one. So it is fetched here first,
with the credential, and handed to `datasets` as an ordinary local file.

That indirection is what lets a private dataset be trained on without being
published anywhere. The alternative -- making the file readable without
authentication so the library can fetch it -- would mean every dataset in the
studio was one guessed URL away from being public.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx


def is_studio_dataset(cfg: dict) -> bool:
    name = str(cfg.get("dataset") or "")
    return bool(cfg.get("dataset_is_local")) and name.startswith(("http://", "https://"))


def _reachable(url: str, controller_url: str) -> str:
    """The same dataset URL, but at an address this runner can actually reach.

    The controller builds the URL from the address the *browser* used, which is
    the one address guaranteed to mean nothing here: a job started from
    http://127.0.0.1 sends the runner to its own loopback, and one started
    through a public hostname sends it out and back through a proxy that may
    not even resolve on this network. The runner already knows where its
    controller is -- it is connected to it -- so the path is kept and the host
    is replaced.
    """
    if not controller_url:
        return url
    marker = "/api/datasets/"
    if marker not in url:
        return url
    return controller_url.rstrip("/") + url[url.index(marker):]


def local_copy(cfg: dict, ctx: Any) -> str:
    """The dataset as a path on this machine, downloading it if it is remote.

    Returns whatever `cfg["dataset"]` already was when there is nothing to
    fetch, so callers can use this unconditionally.
    """
    name = str(cfg.get("dataset") or "")
    if not is_studio_dataset(cfg):
        return name

    label = cfg.get("dataset_label") or "the dataset"
    dest = Path(ctx.workdir) / "dataset.jsonl"
    ctx.log("Fetching %s from the studio." % label)

    url = _reachable(name, getattr(ctx, "controller_url", ""))
    token = getattr(ctx, "runner_token", "") or os.environ.get("AI_STUDIO_TOKEN", "")
    with httpx.stream("GET", url, timeout=600, follow_redirects=True,
                      headers={"X-Runner-Token": token}) as r:
        if r.status_code == 403:
            raise ValueError(
                "The controller would not hand over this dataset. The runner's "
                "join token was refused, which usually means it was rotated "
                "since this machine last joined.")
        r.raise_for_status()
        size = 0
        with open(dest, "wb") as fh:
            for chunk in r.iter_bytes(1 << 20):
                fh.write(chunk)
                size += len(chunk)

    ctx.log("Got %.1f MB." % (size / 1048576))
    return str(dest)
