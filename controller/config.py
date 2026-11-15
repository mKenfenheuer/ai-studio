"""Controller configuration. Everything is env-overridable so the same image
runs on a laptop, a server, or inside docker-compose without edits."""
from __future__ import annotations

import os
import secrets
from pathlib import Path


def _env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser().resolve()


# Where the studio keeps its state. Mount this as a volume in Docker.
DATA_DIR: Path = _env_path("AI_STUDIO_DATA", "./data")
# A directory the controller and its runners both mount, on one host or over
# NFS/SMB. Optional. With it, a runner writes its finished model straight into
# the place the controller keeps it and reads a model back without downloading
# it, so a 6 GB result is on the disk once instead of in the runner's workdir,
# in transit and on the controller. Without it, everything travels over HTTP
# as it always has.
SHARED_DIR: Path | None = (_env_path("AI_STUDIO_SHARED_DIR", "")
                           if os.environ.get("AI_STUDIO_SHARED_DIR") else None)
# Where artifacts lived before there was a shared directory. Still read, so an
# install that turns sharing on keeps every model it already has.
LEGACY_ARTIFACT_DIR: Path = DATA_DIR / "artifacts"
ARTIFACT_DIR: Path = (SHARED_DIR / "artifacts" if SHARED_DIR
                      else LEGACY_ARTIFACT_DIR)
# Runners write here first; the controller moves a file into ARTIFACT_DIR when
# the runner says it is complete, so a half-written zip is never an artifact.
INCOMING_DIR: Path | None = SHARED_DIR / "incoming" if SHARED_DIR else None
# Proves to a runner that the directory it mounts is this controller's and not
# an empty folder of the same name. Its content is sent at registration.
SHARE_MARKER = ".ai-studio-share"
DB_PATH: Path = DATA_DIR / "studio.db"

HOST: str = os.environ.get("AI_STUDIO_HOST", "0.0.0.0")
PORT: int = int(os.environ.get("AI_STUDIO_PORT", "8420"))

# Shared secret a runner must present to join the fleet. Generated and
# persisted on first boot so a fresh install is secure by default rather than
# wide open; surfaced in the UI so the user can copy the join command.
_TOKEN_FILE = DATA_DIR / "join_token"


def join_token() -> str:
    if tok := os.environ.get("AI_STUDIO_JOIN_TOKEN"):
        return tok.strip()
    if _TOKEN_FILE.exists():
        return _TOKEN_FILE.read_text(encoding="utf-8").strip()
    tok = secrets.token_urlsafe(24)
    _TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    _TOKEN_FILE.write_text(tok, encoding="utf-8")
    _TOKEN_FILE.chmod(0o600)
    return tok


# The address people actually type. Only needed behind a reverse proxy or a
# tunnel, and only by single sign-on -- a provider matches the redirect URI it
# was given character for character, and behind a proxy the URL the
# application sees is the internal one, which matches nothing anybody
# registered. Left unset, the forwarded headers are used, then the request.
PUBLIC_URL: str | None = (os.environ.get("AI_STUDIO_PUBLIC_URL") or "").strip() or None

# Optional Hugging Face token, forwarded to runners so they can pull gated
# models (Llama, Gemma) and get better rate limits.
HF_TOKEN: str | None = os.environ.get("HF_TOKEN") or None

# A runner that has not sent a heartbeat within this window is shown as lost.
HEARTBEAT_TIMEOUT_S: int = int(os.environ.get("AI_STUDIO_HEARTBEAT_TIMEOUT", "45"))

# ...and how long it must stay silent before the work it was given is taken
# off it and put back on the queue. Deliberately much longer than the window
# above, because the two decisions are not the same size. Greying a machine out
# in the UI is cosmetic and undoes itself the moment it speaks again; requeuing
# is destructive, and on a fleet with a second free machine it starts the same
# work twice.
#
# The gap has to cover a runner that is *doing the work asked of it* rather than
# one that has gone. Merging a 7B on the same box that is training on the GPU
# oversubscribes four cores, and an event loop that gets no time slice for
# twenty seconds fails its websocket keepalive and reconnects -- repeatedly,
# through a merge that takes fifteen minutes. Under the old window each of
# those blips looked like a dead machine, and the log filled with a run being
# requeued and re-assigned to the machine that had never stopped running it.
ORPHAN_TIMEOUT_S: int = int(os.environ.get("AI_STUDIO_ORPHAN_TIMEOUT", "300"))

WEB_DIR: Path = Path(__file__).resolve().parent.parent / "web"

# The longest reply anybody may ask for, in tokens. A ceiling on what a caller
# can request rather than a target: generation here is one forward pass per
# token, so a long limit is a long wait, and the Stop button is what a reader
# uses when they have seen enough. Memory is not the binding constraint -- the
# key/value cache of a 7B costs about an eighth of a megabyte per token, so
# even the whole of this is a fraction of what serving the weights already
# takes.
MAX_NEW_TOKENS: int = int(os.environ.get("AI_STUDIO_MAX_NEW_TOKENS") or 4096)

# How long one reply may take before the runner stops and hands back what it
# has. A runner serves one message at a time, so this bounds how long everybody
# else is refused -- not how good the reply is. Kept under the API's own wait
# below, so the runner gives up first and returns a real (if short) answer
# rather than the caller timing out on a machine still working.
GENERATION_DEADLINE_S: float = float(
    os.environ.get("AI_STUDIO_GENERATION_DEADLINE_S") or 300)


def generation_deadline(caps: dict | None) -> float:
    """The deadline for a reply on this machine: the studio's, or longer if
    the machine says it needs it.

    A processor serving through llama.cpp reads a prompt at about 21 tokens a
    second, so a 6,000-token tagging prompt alone is five minutes. It says so
    in `min_deadline_s`, and every place that sets a deadline or waits on one
    asks here -- the runner raising its own was not enough: the controller
    freed the "stuck" slot at 420 seconds while the runner was still reading.
    """
    floor = float((caps or {}).get("min_deadline_s") or 0)
    return max(GENERATION_DEADLINE_S, floor)

# How many requests may be WAITING for one machine, on top of the one it is
# answering. A runner serves one message at a time because it holds one model
# on one card, and that is not going to change -- so the only question is
# whether the ones that arrive meanwhile are queued or refused.
#
# They were refused, and it was the wrong answer twice over: an evaluation
# sending sixty requests got fifty-nine failures, and it got them as 502s,
# which tells a client library that the upstream is broken rather than that it
# should slow down. Queued, sixty requests take sixty times one reply and all
# sixty arrive.
#
# The queue is bounded because an unbounded one is not a queue, it is a way of
# converting a busy machine into a slow one and then into a timeout. Past this
# depth the honest answer is 429 with a Retry-After, which is the status every
# client library already knows how to back off from.
SERVING_QUEUE_MAX: int = int(os.environ.get("AI_STUDIO_SERVING_QUEUE_MAX") or 32)

# What one place in that queue is worth in seconds, for the Retry-After a
# refusal carries. The deadline above is the worst case for a single reply and
# most are far quicker, so this is deliberately the optimistic end: a client
# that comes back too early is refused again cheaply, and one that waits the
# full worst case for every request ahead of it would sleep for an hour.
SERVING_QUEUE_RETRY_S: float = float(
    os.environ.get("AI_STUDIO_SERVING_QUEUE_RETRY_S") or 20)


def artifact_path(filename: str) -> Path:
    """Where an artifact file is: the shared directory, or where it used to be."""
    path = ARTIFACT_DIR / filename
    if SHARED_DIR and not path.exists():
        old = LEGACY_ARTIFACT_DIR / filename
        if old.exists():
            return old
    return path


def share_id() -> str | None:
    """The shared directory's identity, created on first use. None without one."""
    if not SHARED_DIR:
        return None
    marker = SHARED_DIR / SHARE_MARKER
    try:
        if marker.exists():
            return marker.read_text(encoding="utf-8").strip() or None
        SHARED_DIR.mkdir(parents=True, exist_ok=True)
        tok = secrets.token_hex(16)
        tmp = marker.with_name(marker.name + ".tmp")
        tmp.write_text(tok, encoding="utf-8")
        os.replace(tmp, marker)
        return tok
    except OSError:
        return None


def migrate_artifacts(log=print) -> None:
    """Move artifacts from the data volume into the shared directory.

    Only by rename, which is instant on one filesystem. A file on another
    filesystem stays where it is and `artifact_path` keeps finding it there;
    copying hundreds of gigabytes at startup would hold the controller down
    for as long as that takes.
    """
    if not SHARED_DIR or not LEGACY_ARTIFACT_DIR.is_dir():
        return
    moved = stayed = 0
    for old in LEGACY_ARTIFACT_DIR.iterdir():
        if not old.is_file() or (ARTIFACT_DIR / old.name).exists():
            continue
        try:
            os.rename(old, ARTIFACT_DIR / old.name)
            moved += 1
        except OSError:
            stayed += 1
    if moved or stayed:
        log("[controller] shared directory: moved %d artifacts into %s%s"
            % (moved, ARTIFACT_DIR, ", %d on another filesystem stay in %s"
               % (stayed, LEGACY_ARTIFACT_DIR) if stayed else ""))


def clean_incoming(max_age_s: float = 86400.0) -> None:
    """Drop zips a runner began writing and never handed over, after a day."""
    if not INCOMING_DIR or not INCOMING_DIR.is_dir():
        return
    import time
    for f in INCOMING_DIR.iterdir():
        try:
            if f.is_file() and time.time() - f.stat().st_mtime > max_age_s:
                f.unlink()
        except OSError:
            pass


def ensure_dirs() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    if INCOMING_DIR:
        INCOMING_DIR.mkdir(parents=True, exist_ok=True)
