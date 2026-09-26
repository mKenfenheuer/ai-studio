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
import threading
import time
import zipfile
from pathlib import Path
from typing import Callable, Iterable

import httpx

from . import paths

CACHE_DIR = paths.ensure(Path(
    os.environ.get("AI_STUDIO_MODEL_CACHE") or paths.data_root() / "models"))
HF_CACHE_DIR = Path(
    os.environ.get("HF_HUB_CACHE")
    or os.environ.get("HUGGINGFACE_HUB_CACHE")
    or (Path(os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface"))) / "hub"))

#: Written into a cached directory every time it is used, so eviction can pick
#: the least recently *used* model rather than the least recently downloaded.
USED_MARKER = ".ai_studio_used"

GB = float(1 << 30)


def _suffix(kind: str | None) -> str:
    """A run can leave two artifacts, and they cache side by side.

    The merged model is the primary one and keeps the bare job id, so
    everything that cached a model before this existed still finds it.
    """
    return "" if kind in (None, "", "model") else "-" + kind


def cached_dir(job_id: str, kind: str | None = None) -> Path:
    return CACHE_DIR / (job_id + _suffix(kind))


# What makes a directory a model. A transformers model has config.json, an
# adapter adapter_config.json, a diffusers pipeline model_index.json -- and
# some things are a bare safetensors file with nothing beside it. Named in
# one place so the two functions that ask cannot disagree.
def _looks_like_model(d) -> bool:
    if any((d / name).exists() for name in
           ("config.json", "adapter_config.json", "model_index.json")):
        return True
    try:
        return any(p.suffix == ".safetensors" for p in d.iterdir())
    except OSError:
        return False


def is_present(job_id: str, kind: str | None = None) -> bool:
    return _looks_like_model(cached_dir(job_id, kind))


def touch(job_id: str, kind: str | None = None) -> None:
    """Mark a cached model as used now. Never raises: this is bookkeeping."""
    try:
        (cached_dir(job_id, kind) / USED_MARKER).write_text(
            str(int(time.time())), encoding="utf-8")
    except OSError:
        pass


def cached_ids() -> list[str]:
    """Names of the models this machine has on disk, most recently used first.

    Names, not ids: an adapter kept beside its merged model is `<id>-adapter`,
    which is what a caller has to ask for to get it back.
    """
    try:
        dirs = [d for d in CACHE_DIR.iterdir()
                if d.is_dir() and not d.name.endswith(".partial")]
    except OSError:
        return []
    dirs = [d for d in dirs if _looks_like_model(d)]
    return [d.name for d in sorted(dirs, key=_used_at, reverse=True)]


def _used_at(d: Path) -> float:
    for candidate in (d / USED_MARKER, d):
        try:
            return candidate.stat().st_mtime
        except OSError:
            continue
    return 0.0


def _dir_bytes(root: Path) -> int:
    total = 0
    for dirpath, _dirnames, filenames in os.walk(root, onerror=lambda _e: None):
        for name in filenames:
            try:
                total += os.stat(os.path.join(dirpath, name)).st_size
            except OSError:
                pass
    return total


def min_free_bytes() -> int:
    """How much room to leave for the next model and the next dataset.

    A cache that fills the volume is worse than no cache: the fetch that fails
    leaves a `.partial`, the model looks absent, and the next request downloads
    it again into the same wall. The default is a fifth of a large model or a
    tenth of the volume, whichever is more.
    """
    override = os.environ.get("AI_STUDIO_MIN_FREE_GB")
    if override:
        try:
            return int(float(override) * GB)
        except ValueError:
            pass
    try:
        total = shutil.disk_usage(_usable_cache_dir()).total
    except OSError:
        total = 0
    return max(int(20 * GB), int(total * 0.10))


def _usable_cache_dir() -> Path:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        return CACHE_DIR
    except OSError:
        return Path("/")


_usage_cache: tuple[float, dict] = (0.0, {})


def usage(max_age_s: float = 60.0) -> dict:
    """Free space and what the two caches are using, in GB.

    Walking the caches costs a few thousand stat calls, and the heartbeat asks
    every fifteen seconds, so the answer is remembered for a minute.
    """
    global _usage_cache
    now = time.time()
    if now - _usage_cache[0] < max_age_s and _usage_cache[1]:
        return _usage_cache[1]
    root = _usable_cache_dir()
    try:
        du = shutil.disk_usage(root)
        free_gb, total_gb = du.free / GB, du.total / GB
    except OSError:
        free_gb = total_gb = 0.0
    out = {
        "free_gb": round(free_gb, 1),
        "total_gb": round(total_gb, 1),
        "models_gb": round(_dir_bytes(CACHE_DIR) / GB, 1),
        "hf_gb": round(_dir_bytes(HF_CACHE_DIR) / GB, 1),
    }
    _usage_cache = (now, out)
    return out


def _protected(keep: Iterable[str]) -> set[str]:
    """Directories eviction must not touch, for the models named in `keep`.

    The staging directory of a download in progress is one of them. It ends
    in `.partial`, and leftover partials are the first thing eviction reaches
    for -- which, on a disk near its floor, meant a fetch made room for
    itself by deleting the directory it was about to write into, and failed
    with "no such file" on a path it had just created.
    """
    names = set()
    for job_id in keep or ():
        for name in (job_id, job_id + "-adapter"):
            names.add(name)
            names.add(name + ".partial")
    return names


def _shortfall(need_bytes: int) -> int:
    try:
        free = shutil.disk_usage(_usable_cache_dir()).free
    except OSError:
        return 0
    return int(need_bytes) + min_free_bytes() - free


def _oldest_model(protected: set[str]) -> Path | None:
    try:
        candidates = [d for d in CACHE_DIR.iterdir()
                      if d.is_dir() and d.name not in protected]
    except OSError:
        return None
    if not candidates:
        return None
    # Leftovers from an interrupted download are worth nothing: go first.
    partials = [d for d in candidates if d.name.endswith(".partial")]
    return min(partials or candidates, key=_used_at)


def _evict_hf(log: Callable[[str], None]) -> bool:
    """Drop the least recently used revision from the Hugging Face cache.

    Second in line after our own cache, because a base model here was pulled
    from the network once and may be shared by several of our models, whereas
    a studio artifact can always be fetched back from the controller.

    Wrapped whole: a `huggingface_hub` too old for `scan_cache_dir`, or a cache
    mid-write, must slow a fetch down rather than fail it.
    """
    try:
        if not HF_CACHE_DIR.exists():
            return False
        if os.stat(HF_CACHE_DIR).st_dev != os.stat(_usable_cache_dir()).st_dev:
            return False  # a different volume; deleting there frees nothing here
        from huggingface_hub import scan_cache_dir

        info = scan_cache_dir(HF_CACHE_DIR)
        revisions = [(rev, repo) for repo in info.repos for rev in repo.revisions]
        if not revisions:
            return False
        rev, repo = min(revisions, key=lambda pair: pair[0].last_modified or 0)
        freed = rev.size_on_disk / GB
        info.delete_revisions(rev.commit_hash).execute()
        log("Cache full: dropped %s from the Hugging Face cache (%.1f GB)."
            % (repo.repo_id, freed))
        return True
    except Exception:
        return False


def ensure_room(need_bytes: int, keep: Iterable[str] = (),
                log: Callable[[str], None] = lambda _s: None) -> bool:
    """Free space until `need_bytes` fits above the watermark. True if it does.

    Evicts our own cache first, least recently used, then the Hugging Face
    cache. Models named in `keep` -- loaded on the card, or being fetched right
    now -- are never candidates. Returning False rather than raising is
    deliberate: the fetch is still worth attempting, and the disk's own error
    is a better one than a guess made in advance.
    """
    protected = _protected(keep)
    while _shortfall(need_bytes) > 0:
        victim = _oldest_model(protected)
        if victim is not None:
            freed = _dir_bytes(victim) / GB
            shutil.rmtree(victim, ignore_errors=True)
            log("Cache full: evicted %s (%.1f GB)." % (victim.name, freed))
            continue
        if _evict_hf(log):
            continue
        log("Cache full and nothing left to evict; the download may fail.")
        return False
    return True


# One lock per cached artifact, so two callers asking for the same model at the
# same time share one download instead of destroying each other's.
#
# Before this, every fetch began by deleting the staging directory -- which is
# the right thing to do when the directory is debris from a crash, and exactly
# the wrong thing when it is another thread's download in progress. A runner
# serving a deployment and answering a chat about the same model at the same
# moment runs two fetches, and on a 14.5 GB Mistral-7B that did three things at
# once, all of them observed on one machine in one morning:
#
#   * the second fetch deleted the first's half-written zip, so the first
#     finished writing into a file that no longer had a name and then failed
#     with "No such file or directory: .../job_559a7255aa06.partial/
#     artifact.zip";
#   * the first fetch, part-way through unpacking, had its files deleted under
#     it, went on to unpack the few members still to come -- the tokenizer,
#     last in the archive alphabetically -- and moved that into place as the
#     finished model: a directory of 3.6 MB with no weights and no config;
#   * neither ever completed, so every later request started another 14.5 GB
#     download. Two runners doing that against one disk is the iowait that
#     stalled the host.
#
# A threading lock is enough: one runner is one process, and two runners on
# the same host each have their own data volume.
_FETCH_LOCKS: dict[str, threading.Lock] = {}
_FETCH_LOCKS_GUARD = threading.Lock()


def _fetch_lock(name: str) -> threading.Lock:
    with _FETCH_LOCKS_GUARD:
        return _FETCH_LOCKS.setdefault(name, threading.Lock())


# Kept beside a partial download: the version of the file being downloaded, as
# the controller named it. A resumed download sends it back, and the controller
# answers with the rest of THAT file or, if the file has changed since, with
# the whole of the new one. Without it, the tail of one artifact could be
# spliced onto the head of another.
_ETAG_FILE = "artifact.etag"


def fetch(controller_url: str, token: str, job_id: str,
          log: Callable[[str], None] = lambda _s: None,
          kind: str | None = None,
          keep: Iterable[str] = ()) -> Path:
    """This run's result as a directory on local disk, downloading if needed.

    Unpacked into a staging directory and moved into place, so an interrupted
    download cannot leave a half-extracted model that `is_present` then
    reports as ready -- the failure that would produce is a missing-weights
    error hours later, blamed on the model rather than on the download.

    Three guarantees, each of which was missing once:

    * one download per artifact at a time, however many callers ask -- the
      second waits for the first and then finds the model already here;
    * an interrupted download carries on from where it stopped rather than
      starting again, which for a 14.5 GB model is the difference between
      finishing and not;
    * nothing is moved into place that is not a model. A directory that
      unpacks to anything else is thrown away and said to be wrong, rather
      than handed to a loader that will report "unrecognized model" with no
      hint that the download is what failed.

    `kind="adapter"` asks for the LoRA a fine-tune kept beside its merged
    model. `keep` names models that must survive the eviction this may need.
    """
    dest = cached_dir(job_id, kind)
    with _fetch_lock(dest.name):
        # Checked again inside the lock, and this is the check that matters:
        # a caller that waited here while another fetched the same model now
        # finds it complete, and returns without downloading a byte.
        if is_present(job_id, kind):
            touch(job_id, kind)
            return dest
        return _download(controller_url, token, job_id, kind, dest, keep, log)


def _download(controller_url: str, token: str, job_id: str,
              kind: str | None, dest: Path, keep: Iterable[str],
              log: Callable[[str], None]) -> Path:
    staging = dest.with_name(dest.name + ".partial")
    zip_path = staging / "artifact.zip"
    tag_path = staging / _ETAG_FILE

    # What survives from an earlier attempt: the zip and the version it is
    # of, and nothing else. Anything unpacked beside them is from an unpack
    # that did not finish, and is removed rather than trusted. A zip with no
    # version beside it cannot be resumed safely -- there is nothing to prove
    # the rest of it belongs to the same file -- so that starts again too.
    offset, etag = 0, None
    if zip_path.exists() and tag_path.exists():
        offset = zip_path.stat().st_size
        etag = tag_path.read_text(encoding="utf-8").strip() or None
        for leftover in staging.iterdir():
            if leftover.name not in (zip_path.name, tag_path.name):
                if leftover.is_dir():
                    shutil.rmtree(leftover, ignore_errors=True)
                else:
                    leftover.unlink(missing_ok=True)
    if not etag:
        offset = 0
        shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)

    url = "%s/api/jobs/%s/download" % (controller_url.rstrip("/"), job_id)
    params = {"kind": kind} if _suffix(kind) else None
    headers = {"X-Runner-Token": token}
    if offset and etag:
        headers["Range"] = "bytes=%d-" % offset
        headers["If-Range"] = etag
        log("Carrying on with the download from %.1f GB rather than starting "
            "again." % (offset / 1024 ** 3))
    else:
        log("Fetching the trained model from the studio...")

    total = 0
    with httpx.stream("GET", url, params=params, timeout=1800,
                      follow_redirects=True, headers=headers) as r:
        if r.status_code in (401, 403):
            raise ValueError(
                "The controller would not hand over that model. The runner's "
                "join token was refused, which usually means it was rotated "
                "since this machine last joined.")
        if r.status_code == 404:
            raise ValueError(
                "That run has no saved model on the controller any more. It "
                "may have been deleted.")
        if r.status_code == 416:
            # Asked for bytes past the end: the file was already whole, and
            # only the unpack had not happened. Nothing to download.
            total = offset
        else:
            r.raise_for_status()
            if r.status_code == 206:
                total = _range_total(r.headers.get("Content-Range")) or 0
                mode = "ab"
            else:
                # A 200 to a ranged request means the controller has a
                # different file from the one this partial was of, and is
                # sending the new one whole. Start over, and remember which.
                offset = 0
                total = int(r.headers.get("Content-Length") or 0)
                mode = "wb"
                if r.headers.get("ETag"):
                    tag_path.write_text(r.headers["ETag"], encoding="utf-8")
                else:
                    tag_path.unlink(missing_ok=True)
            # The zip and the unpacked copy live side by side in staging until
            # the extract finishes, so room is made for both -- less whatever
            # of the zip is already here.
            ensure_room(max(total * 2 - offset, 0), list(keep) + [job_id], log)
            with open(zip_path, mode) as fh:
                for chunk in r.iter_bytes(1 << 20):
                    fh.write(chunk)

    got = zip_path.stat().st_size if zip_path.exists() else 0
    if total and got != total:
        # Short, not wrong. Kept, so the next attempt asks for the rest.
        raise RuntimeError(
            "The download of this model stopped at %.1f of %.1f GB. What "
            "arrived is kept and the next attempt will carry on from there."
            % (got / 1024 ** 3, total / 1024 ** 3))

    try:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(staging)
    except (zipfile.BadZipFile, OSError) as e:
        # A zip that is the right length and still does not open is not one
        # a retry can finish. Removed whole, so the next attempt starts
        # clean instead of resuming onto the same damage.
        shutil.rmtree(staging, ignore_errors=True)
        raise RuntimeError(
            "The model downloaded but would not unpack (%s). It has been "
            "discarded and will be fetched again from the start." % e) from e
    zip_path.unlink(missing_ok=True)
    tag_path.unlink(missing_ok=True)

    if not _looks_like_model(staging):
        found = sorted(p.name for p in staging.iterdir())[:8]
        shutil.rmtree(staging, ignore_errors=True)
        raise RuntimeError(
            "The saved result of %s unpacked to %s, which is not a model: "
            "there is no config.json, adapter_config.json or weights file in "
            "it. The artifact on the controller is incomplete; the run that "
            "produced it needs looking at, not this machine."
            % (job_id, ", ".join(found) or "nothing"))

    shutil.rmtree(dest, ignore_errors=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(dest)
    touch(job_id, kind)
    log("Got %.0f MB." % (got / 1048576))
    return dest


def _range_total(content_range: str | None) -> int | None:
    """The whole file's size, from `Content-Range: bytes 0-15/14499765169`."""
    if not content_range or "/" not in content_range:
        return None
    tail = content_range.rsplit("/", 1)[1].strip()
    return int(tail) if tail.isdigit() else None


def pack(root_dir: Path | str, dest: Path | str) -> Path:
    """Zip a finished model for the journey to the controller, without deflate.

    `shutil.make_archive` compresses, and for weights that is work done for
    nothing. A safetensors file is dense float data with no structure deflate
    can find: it comes back about one percent smaller, having read and
    recompressed every byte on a single core. On the machine that prompted this
    -- a 7B merged on the CPU while the GPU runner trained beside it -- that was
    half an hour of the box's remaining capacity spent to save a hundred
    megabytes out of fourteen thousand, with the run looking hung throughout.

    Stored, the same archive is written at disk speed. It stays a zip, so
    nothing that reads one has to know this happened.

    Datasets are a different case and keep their compression: JSONL is text and
    text does compress, often to a fifth.
    """
    dest = Path(dest)
    root = Path(root_dir)
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                z.write(path, path.relative_to(root))
    return dest


def summary(job_id: str, kind: str | None = None) -> dict:
    """What the run that produced this recorded about itself, if anything."""
    try:
        return json.loads((cached_dir(job_id, kind) / "ai_studio_summary.json")
                          .read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def clear(job_id: str | None = None) -> None:
    """Forget a run's cached artifacts -- both of them -- or the lot."""
    if job_id:
        for name in (job_id, job_id + "-adapter"):
            shutil.rmtree(CACHE_DIR / name, ignore_errors=True)
            shutil.rmtree(CACHE_DIR / (name + ".partial"), ignore_errors=True)
    else:
        shutil.rmtree(CACHE_DIR, ignore_errors=True)
