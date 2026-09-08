"""A copy of the studio that can be put back.

There was none. The database is SQLite in WAL mode -- copying the file while
the controller runs tears it, because the last minutes of writes are in the
`-wal` beside it -- and the models, datasets and pictures are directories on
the same disk with no second copy anywhere. A studio's whole history of runs
lived one disk failure from gone.

A backup is a directory: the database written whole and consistent by
SQLite itself (`VACUUM INTO`, which works while the studio is live), the join
token, the datasets, the stored pictures and clips, and -- when asked, because
they are the size of the disk -- the models. A manifest says what is there
and when; a RESTORE file beside it says how to put it back, in the words
somebody will need at the moment they need them.

Models are off by default. A backup that includes forty gigabytes of merged
7Bs is a backup nobody runs nightly; one that includes everything else is a
few hundred megabytes and can run every hour. The models can be trained
again from the datasets; the datasets cannot be recovered from the models.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from . import config, db

SETTING_KEY = "backup"
DEFAULTS = {
    "dir": str(config.DATA_DIR / "backups"),
    "every_hours": 0,          # 0: only when asked
    "keep": 7,                 # how many to hold on to
    "include_models": False,
}

RESTORE_TEXT = """HOW TO RESTORE THIS BACKUP
==========================

This directory is a complete copy of an AI Studio taken at the time in
MANIFEST.json{models_note}.

1. Stop the studio:
       docker compose -f /opt/ai-studio-build/docker/docker-compose.yml stop controller
   (or stop `python -m controller` if it runs by hand).

2. Put the files back into the studio's data directory -- the one mounted at
   /data in the container, or AI_STUDIO_DATA:
{steps}
   Remove any studio.db-wal and studio.db-shm beside the database first; they
   belong to the copy you are replacing.

3. Start the studio again. Every run, dataset, prompt set, score, account and
   key is as it was at the time of the backup. Models that were not included
   show on their run pages as removed; "Run again" trains them afresh.

   scripts/restore.sh does steps 1-3 for a docker-compose studio:
       bash scripts/restore.sh /path/to/this/backup
"""


def settings() -> dict:
    try:
        got = json.loads(db.get_setting(SETTING_KEY) or "{}")
    except (TypeError, ValueError):
        got = {}
    out = dict(DEFAULTS)
    out["dir"] = str(got.get("dir") or DEFAULTS["dir"]).strip() or DEFAULTS["dir"]
    for k in ("every_hours", "keep"):
        try:
            out[k] = max(0, int(got.get(k, DEFAULTS[k]) or 0))
        except (TypeError, ValueError):
            pass
    out["include_models"] = bool(got.get("include_models", False))
    return out


def save_settings(values: dict, user_id: str | None = None) -> dict:
    current = settings()
    if "dir" in values:
        d = str(values["dir"] or "").strip()
        if not d:
            raise ValueError("Say where backups should go.")
        current["dir"] = d
    for k in ("every_hours", "keep"):
        if k in values:
            try:
                current[k] = max(0, int(values[k] or 0))
            except (TypeError, ValueError):
                raise ValueError("%s must be a whole number." % k.replace("_", " "))
    if "include_models" in values:
        current["include_models"] = values["include_models"] in (True, 1, "1", "true", "on")
    db.set_setting(SETTING_KEY, json.dumps(current), user_id)
    return current


def _dir_bytes(path: Path) -> int:
    total = 0
    try:
        for p in path.rglob("*"):
            if p.is_file():
                total += p.stat().st_size
    except OSError:
        pass
    return total


def create(include_models: bool | None = None, dest_root: str | None = None) -> dict:
    """Take a backup now. Returns its manifest."""
    s = settings()
    include_models = s["include_models"] if include_models is None else include_models
    root = Path(dest_root or s["dir"]).expanduser()
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        _record_failure("The backup folder could not be made: %s" % e)
        raise
    stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    dest = root / stamp
    staging = root / (stamp + ".partial")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    t0 = time.time()
    try:
        return _copy_everything(staging, dest, root, stamp, include_models, s, t0)
    except Exception as e:  # noqa: BLE001 - recorded, then re-raised
        shutil.rmtree(staging, ignore_errors=True)
        _record_failure(str(e))
        raise


def _record_failure(why: str) -> None:
    """A backup that failed is the one thing worth saying out loud.

    The failure goes where the last success goes, so the admin page reads
    "last backup: failed, three days ago" rather than showing a green line
    from the last time it worked and nothing since.
    """
    db.set_setting(SETTING_KEY + "_last",
                   json.dumps({"taken_at": time.time(), "error": why}))


def _copy_everything(staging: Path, dest: Path, root: Path, stamp: str,
                     include_models: bool, s: dict, t0: float) -> dict:

    # The database, written whole and consistent by SQLite itself. This is
    # the one step that cannot be a file copy: in WAL mode the newest writes
    # live in a side file, and copying the main file alone loses them.
    db.connect().execute("VACUUM INTO ?", (str(staging / "studio.db"),))

    parts: dict[str, int] = {"studio.db": (staging / "studio.db").stat().st_size}
    token = config.DATA_DIR / "join_token"
    if token.exists():
        shutil.copy2(token, staging / "join_token")
        parts["join_token"] = token.stat().st_size
    for name in ("datasets", "assets") + (("artifacts",) if include_models else ()):
        src = config.DATA_DIR / name
        if src.exists():
            shutil.copytree(src, staging / name, dirs_exist_ok=True)
            parts[name] = _dir_bytes(staging / name)

    manifest = {
        "taken_at": time.time(), "stamp": stamp, "version": "0.1.0",
        "include_models": include_models,
        "parts": parts, "bytes": sum(parts.values()),
        "seconds": round(time.time() - t0, 1),
        "runs": (db.q1("SELECT COUNT(*) AS n FROM jobs") or {}).get("n"),
        "datasets": (db.q1("SELECT COUNT(*) AS n FROM datasets") or {}).get("n"),
        "users": (db.q1("SELECT COUNT(*) AS n FROM users") or {}).get("n"),
    }
    (staging / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    # The steps name only what is actually in this directory. Instructions
    # that tell you to copy a file that was never here are how somebody
    # restoring at three in the morning decides the whole page is unreliable.
    steps = ["       cp studio.db          <data>/studio.db"]
    if "join_token" in parts:
        steps.append("       cp join_token         <data>/join_token")
    for name in ("datasets", "assets", "artifacts"):
        if name in parts:
            steps.append("       rm -rf <data>/%s && cp -r %s <data>/%s"
                         % (name, name, name))
    if "artifacts" not in parts:
        steps.append("       (no artifacts/: the models were not in this backup)")
    (staging / "RESTORE.md").write_text(RESTORE_TEXT.format(
        models_note="" if include_models else
        " -- without the models, which were left out to keep it small",
        steps="\n".join(steps)), encoding="utf-8")
    # Moved into place whole, so a half-written backup never looks like one.
    staging.rename(dest)
    db.set_setting(SETTING_KEY + "_last", json.dumps({**manifest, "path": str(dest)}))
    prune(s["keep"], root)
    return {**manifest, "path": str(dest)}


def list_backups(root: str | None = None) -> list[dict]:
    root = Path(root or settings()["dir"]).expanduser()
    out = []
    if not root.exists():
        return out
    for d in sorted(root.iterdir(), reverse=True):
        m = d / "MANIFEST.json"
        if d.is_dir() and m.exists():
            try:
                manifest = json.loads(m.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            out.append({**manifest, "path": str(d)})
    return out


def prune(keep: int, root: Path | None = None) -> int:
    """Drop the oldest beyond `keep`. Zero keeps everything."""
    if not keep:
        return 0
    root = root or Path(settings()["dir"]).expanduser()
    backups = list_backups(str(root))
    gone = 0
    for b in backups[keep:]:
        shutil.rmtree(b["path"], ignore_errors=True)
        gone += 1
    return gone


def delete(path: str) -> bool:
    root = Path(settings()["dir"]).expanduser().resolve()
    target = Path(path).expanduser().resolve()
    # Only something under the backup directory, and only a backup.
    if root not in target.parents or not (target / "MANIFEST.json").exists():
        return False
    shutil.rmtree(target, ignore_errors=True)
    return True


def last() -> dict | None:
    try:
        return json.loads(db.get_setting(SETTING_KEY + "_last") or "null")
    except (TypeError, ValueError):
        return None


def due() -> bool:
    """Whether the schedule says it is time."""
    s = settings()
    if not s["every_hours"]:
        return False
    prev = last()
    return not prev or time.time() - float(prev.get("taken_at") or 0) >= s["every_hours"] * 3600
