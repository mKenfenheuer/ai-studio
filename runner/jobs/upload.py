"""Putting a finished model or dataset on Hugging Face.

This used to happen inside the HTTP request that asked for it, and that was
the bug. A merged 7B model is fourteen gigabytes; creating the repository
takes a moment and sending the weights takes twenty minutes. Long before the
Hub was finished the browser had given up, the request was cancelled, and the
unpacked folder was deleted out from under the thread still reading it. What
was left on the Hub was the repository the first line had created and nothing
else -- an empty repo, no error anywhere, and every appearance of having
worked.

An upload is a long, interruptible piece of work that wants a progress bar, a
log and a stop button. The studio already has something that is all of those,
so this is a job like the others. It happens to need no GPU at all, which is
what makes it a good candidate for a machine that has none.

Two things it does that the old code did not:

* **Reports where it has got to.** Files are sent one at a time and committed
  once at the end, so the page can name the file in flight and count the
  megabytes. `upload_folder` does both halves in one call and says nothing for
  the duration, which is indistinguishable from being stuck.
* **Can replace what is already there.** Publishing an adapter over a merged
  model otherwise leaves the old shards behind, and a repository holding two
  models loads whichever its config happens to name.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from runner import artifacts

from . import source
from .lora_llm import Cancelled

# How often the progress bar moves during one large file. The Hub client
# uploads a file in one call, so this is the granularity available without
# reaching inside it: per file, plus a heartbeat while a big one is in flight.
_HEARTBEAT_S = 20


def run(cfg: dict, ctx: Any) -> dict:
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

    repo_id = (cfg.get("repo_id") or "").strip()
    if not repo_id:
        raise ValueError("No repository was given to upload to.")
    token = cfg.get("hf_token") or getattr(ctx, "hf_token", None)
    if not token:
        raise ValueError(
            "There is no Hugging Face token to upload with. Connect an "
            "account with write access on the account page and try again.")

    kind = "dataset" if cfg.get("target") == "dataset" else "model"
    replace = bool(cfg.get("replace"))
    message = cfg.get("message") or (
        "Prepared with AI Studio" if kind == "dataset"
        else "Trained with AI Studio")

    api = HfApi(token=token)
    t0 = time.time()

    # ---- what to send ----------------------------------------------------
    ctx.progress(0, 1, stage="collecting")
    if kind == "dataset":
        local = Path(source.local_copy(cfg, ctx))
        files = [(local, "data.jsonl")]
    else:
        if not cfg.get("source_job"):
            raise ValueError("No run was given to upload.")
        # Which of the run's artifacts is being published. A fine-tune keeps
        # both the merged model and the adapter it was merged from, and either
        # is a legitimate thing to send -- to different repositories.
        wanted = cfg.get("artifact_kind") or "model"
        folder = artifacts.fetch(ctx.controller_url, ctx.runner_token,
                                 cfg["source_job"], ctx.log,
                                 kind=None if wanted == "model" else wanted)
        # The last place a mismatch can be caught. Publishing an adapter as an
        # adapter is fine and the card says so. Publishing one *as a model* is
        # not: 50 MB of low-rank matrices load into nothing without the exact
        # base they were fitted to, and a repository containing only those
        # looks like a model right up to the moment somebody tries to use it.
        is_adapter = (folder / "adapter_config.json").exists()
        if is_adapter != (cfg.get("expect", wanted) == "adapter"):
            raise ValueError(
                "That run's %s artifact is %s. Publishing an adapter as a "
                "model produces a repository that cannot be loaded, so this "
                "one has been stopped before anything was created."
                % (wanted, "an adapter" if is_adapter else "a whole model"))
        # Uploaded straight out of the cache rather than copied somewhere
        # first: a copy of a fourteen-gigabyte model, to add one text file
        # beside it, is fourteen gigabytes of disk and several minutes.
        files = [(p, p.relative_to(folder).as_posix())
                 for p in sorted(folder.rglob("*")) if p.is_file()]

    card = cfg.get("card") or ""
    # The card the controller rendered wins over whatever README the training
    # run happened to leave in the archive.
    if card:
        files = [(p, name) for p, name in files if name != "README.md"]
    if not files and not card:
        raise ValueError("There is nothing to upload.")

    total = sum(p.stat().st_size for p, _ in files)
    ctx.log("Uploading %d file%s (%s) to %s."
            % (len(files), "" if len(files) == 1 else "s", _size(total),
               repo_id))

    # ---- the repository --------------------------------------------------
    # exist_ok, so publishing again to a repository that already exists is
    # ordinary rather than an error. Visibility is only applied when the
    # repository is created -- the Hub does not let this flip an existing one,
    # and pretending otherwise would be worse than leaving it where its owner
    # set it.
    existed = _exists(api, repo_id, kind)
    api.create_repo(repo_id=repo_id, repo_type=kind,
                    private=bool(cfg.get("private", True)), exist_ok=True)
    if existed:
        ctx.log("%s already exists, so this adds to it. Its visibility is "
                "left as its owner set it." % repo_id)
    else:
        ctx.log("Created %s as a %s repository."
                % (repo_id, "private" if cfg.get("private", True)
                   else "public"))

    adds = [CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(p))
            for p, name in files]
    if card:
        adds.append(CommitOperationAdd(path_in_repo="README.md",
                                       path_or_fileobj=card.encode("utf-8")))

    deletes: list = []
    if replace and existed:
        keeping = {op.path_in_repo for op in adds} | {".gitattributes"}
        # .gitattributes is never deleted: it is what tells the Hub which
        # patterns are stored in LFS, it is written by the Hub rather than by
        # us, and taking it away makes the next large upload land as a plain
        # file.
        doomed = [p for p in api.list_repo_files(repo_id=repo_id,
                                                 repo_type=kind)
                  if p not in keeping]
        deletes = [CommitOperationDelete(path_in_repo=p) for p in doomed]
        if doomed:
            ctx.log("Replacing everything already there: %d file%s will be "
                    "removed in the same commit (%s)."
                    % (len(doomed), "" if len(doomed) == 1 else "s",
                       ", ".join(doomed[:6])
                       + (", …" if len(doomed) > 6 else "")))
        else:
            ctx.log("Nothing there needed removing.")

    # ---- send ------------------------------------------------------------
    # In megabytes, because the alternative units are "one file of four" for a
    # model that is one file, and a percentage of a number nobody was shown.
    mb_total = max(1, round(total / 1048576))
    sent = 0
    ctx.progress(0, mb_total, stage="uploading")
    for op, (path, name) in zip(adds, files):
        if ctx.should_cancel():
            # Nothing has been committed, so nothing on the Hub has changed.
            # The blobs already sent are unreferenced and the Hub collects
            # them; saying so is better than leaving somebody wondering what
            # a half-finished upload left behind.
            ctx.log("Stopped before anything was committed. The repository is "
                    "as it was.", "warn")
            raise Cancelled()
        size = path.stat().st_size
        ctx.log("Sending %s (%s)…" % (name, _size(size)))
        started = time.time()
        api.preupload_lfs_files(repo_id, additions=[op], repo_type=kind)
        sent += size
        ctx.progress(round(sent / 1048576), mb_total, stage="uploading")
        took = time.time() - started
        if took > _HEARTBEAT_S:
            ctx.log("  %s in %s (%s/s)"
                    % (_size(size), _clock(took),
                       _size(size / max(took, 0.001))))

    ctx.log("Committing." if not deletes
            else "Committing, and removing what it replaces.")
    api.create_commit(repo_id=repo_id, repo_type=kind,
                      operations=deletes + adds, commit_message=message)

    url = "https://huggingface.co/%s%s" % (
        "" if kind == "model" else "datasets/", repo_id)
    ctx.progress(mb_total, mb_total, stage="uploading")
    ctx.log("Done: %s" % url)
    return {
        "kind": "upload", "target": kind, "repo_id": repo_id, "url": url,
        "files": len(adds), "bytes": total, "replaced": len(deletes),
        "private": bool(cfg.get("private", True)),
        "created_repo": not existed,
        "duration_s": round(time.time() - t0, 1),
    }


def _exists(api: Any, repo_id: str, kind: str) -> bool:
    try:
        return bool(api.repo_exists(repo_id=repo_id, repo_type=kind))
    except Exception:  # noqa: BLE001 - a network hiccup is not an answer
        return False


def _size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return "%.1f %s" % (n, unit) if unit != "B" else "%d B" % n
        n /= 1024
    return "%.1f GB" % n


def _clock(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return "%dm %02ds" % (m, s) if m else "%ds" % s
