"""Bytes that are not text: images, audio, anything a dataset points at.

Nothing below the UI could hold one. A dataset is one JSONL file, a row is a
JSON object, and JSON has no way to carry a photograph -- so every modality
past text was blocked on this, and a picture in a dataset was at best a URL to
somebody else's server that would rot.

Three decisions, and each one is the reason for a whole class of bug that does
not happen:

**The file is addressed by its content.** Its path is its SHA-256, so the same
bytes uploaded twice occupy the disk once. That is not a space optimisation
first -- it is what makes a derived dataset free. Filtering ten thousand
images down to nine hundred, or splitting a set in two, copies rows that point
at the same assets, and neither operation has to know that assets exist.

**The row is not the file.** One `assets` row is one reference, owned by
somebody, belonging to a dataset or a run; several rows can name the same
bytes. Deleting a row is therefore always safe: the file goes when the last
row naming it goes, and never before. A store where deleting your copy of an
image tore it out of a colleague's dataset would be worse than no store.

**A dataset points at assets by id, in an ordinary column.** No new tables, no
join, no second file: `{"image": "asset:ast_3f2a...", "label": "cat"}` is a
row like any other. Everything that already works -- splits, filters, the
transform pipeline, the JSONL download, publishing to the Hub -- keeps working
without being taught anything, because as far as it is concerned that is a
string.
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
from pathlib import Path
from typing import Iterable, Iterator

from . import config, db

ASSET_DIR: Path = config.DATA_DIR / "assets"

# How a row names an asset. A prefix rather than a bare id so that a column of
# them is recognisable on sight, in a JSONL file somebody is reading in a
# terminal, without a schema to consult.
PREFIX = "asset:"
REF_RE = re.compile(r"^asset:(ast_[0-9a-f]{12})$")

# What a single file may be. Generous enough for a minute of uncompressed
# audio or a raw photograph, small enough that one bad drag-and-drop cannot
# fill a disk. Whole-collection limits are the quota below.
MAX_BYTES = int(os.environ.get("AI_STUDIO_ASSET_MAX_MB", "128")) * 1024 * 1024

# What the studio will hold in total, per account, unless configured
# otherwise. A number that exists at all is the point: the failure this
# prevents is a training box whose disk fills at three in the morning. Set it
# below what the disk actually has: this is a ceiling per account, and there
# is more than one account.
QUOTA_BYTES = int(os.environ.get("AI_STUDIO_ASSET_QUOTA_GB", "20")) * 1024 ** 3

# Read in pieces so a hundred-megabyte upload is never a hundred megabytes of
# process memory, and hashed on the way past so it is not read twice.
CHUNK = 1024 * 1024

# What may be stored, and what to call it. An allowlist rather than "whatever
# the browser said": the content type is echoed back on download, and a store
# that will serve arbitrary text/html from its own origin is a cross-site
# scripting hole with a database behind it.
KINDS: dict[str, str] = {
    "image/png": "image", "image/jpeg": "image", "image/webp": "image",
    "image/gif": "image", "image/bmp": "image", "image/tiff": "image",
    "audio/wav": "audio", "audio/x-wav": "audio", "audio/mpeg": "audio",
    "audio/flac": "audio", "audio/ogg": "audio", "audio/mp4": "audio",
    "audio/webm": "audio",
    "video/mp4": "video", "video/webm": "video",
    "application/pdf": "document",
}

_BY_SUFFIX = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".webp": "image/webp", ".gif": "image/gif", ".bmp": "image/bmp",
    ".tif": "image/tiff", ".tiff": "image/tiff",
    ".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac",
    ".ogg": "audio/ogg", ".oga": "audio/ogg", ".m4a": "audio/mp4",
    ".opus": "audio/ogg", ".webm": "audio/webm",
    ".mp4": "video/mp4", ".pdf": "application/pdf",
}


def is_media(filename: str) -> bool:
    """Whether a file's name says it is something the store holds."""
    return Path(filename or "").suffix.lower() in _BY_SUFFIX


def guess_mime(filename: str, given: str = "") -> str:
    """What this file is, believing the extension over the browser.

    A browser reports `application/octet-stream` for anything it does not
    recognise and, on some platforms, for WAV files it does. The extension is
    the more reliable of the two here because these files nearly always come
    out of a folder somebody assembled deliberately.
    """
    suffix = Path(filename or "").suffix.lower()
    if by_ext := _BY_SUFFIX.get(suffix):
        return by_ext
    given = (given or "").split(";")[0].strip().lower()
    return given if given in KINDS else ""


def kind_of(mime: str) -> str:
    return KINDS.get((mime or "").lower(), "")


def path_for(sha: str) -> Path:
    """Where the bytes live. Two levels of fan-out, because a hundred thousand
    files in one directory is slow on every filesystem worth naming."""
    return ASSET_DIR / sha[:2] / sha[2:4] / sha


def ref(asset_id: str) -> str:
    return PREFIX + asset_id


def id_in(value: object) -> str:
    """The asset id a cell holds, or "" if it is an ordinary value."""
    if isinstance(value, str) and (m := REF_RE.match(value)):
        return m.group(1)
    return ""


def ids_in_row(row: dict) -> list[str]:
    out = []
    for value in (row or {}).values():
        if got := id_in(value):
            out.append(got)
        elif isinstance(value, list):
            out += [g for g in (id_in(v) for v in value) if g]
    return out


# ------------------------------------------------------------------ writing

def store(chunks: Iterable[bytes], filename: str, mime: str,
          owner_id: str | None, dataset_id: str | None = None,
          job_id: str | None = None) -> dict:
    """Write one file into the store and return its row.

    The bytes are hashed as they are written to a temporary file, which is
    then moved into place under that hash. A crash halfway leaves a temporary
    file and no row, which is the failure worth having: the alternative is a
    row pointing at half a photograph.
    """
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    tmp = ASSET_DIR / ("incoming-%s" % db.new_id("tmp"))
    digest = hashlib.sha256()
    size = 0
    try:
        with tmp.open("wb") as fh:
            for chunk in chunks:
                if not chunk:
                    continue
                size += len(chunk)
                if size > MAX_BYTES:
                    raise ValueError(
                        "%s is larger than %d MB, which is the most one file "
                        "may be." % (filename or "That file",
                                     MAX_BYTES // (1024 * 1024)))
                digest.update(chunk)
                fh.write(chunk)
        if not size:
            raise ValueError("%s is empty." % (filename or "That file"))
        sha = digest.hexdigest()
        final = path_for(sha)
        if final.exists():
            # These exact bytes are already here, put there by somebody or by
            # some earlier row. Nothing to write; the new row simply points at
            # the same file.
            tmp.unlink(missing_ok=True)
        else:
            final.parent.mkdir(parents=True, exist_ok=True)
            tmp.replace(final)
    finally:
        tmp.unlink(missing_ok=True)

    # One row per (owner, bytes, home). Uploading the same picture into the
    # same dataset twice is one row, which is what makes an ingest that is
    # interrupted and restarted harmless.
    if existing := db.find_asset(sha, owner_id, dataset_id, job_id):
        return existing
    aid = db.create_asset(sha=sha, mime=mime, size=size,
                          filename=Path(filename or sha).name,
                          owner_id=owner_id, dataset_id=dataset_id,
                          job_id=job_id, kind=kind_of(mime))
    return db.get_asset(aid)


def usage(owner_id: str | None = None) -> dict:
    """How much room the store is taking, counting shared bytes once."""
    return db.asset_usage(owner_id)


# How much of the disk must stay free, whatever the quota says. The quota is
# per account and set without looking at the machine; this is the machine's
# own limit, and it is the one that matters when the disk is nearly full --
# the lab box this was written against had thirteen gigabytes left and a
# twenty-gigabyte quota.
DISK_FLOOR_BYTES = int(os.environ.get("AI_STUDIO_DISK_FLOOR_GB", "2")) * 1024 ** 3


def free_bytes() -> int | None:
    try:
        ASSET_DIR.mkdir(parents=True, exist_ok=True)
        return shutil.disk_usage(ASSET_DIR).free
    except OSError:
        return None


def check_quota(owner_id: str | None, adding: int = 0) -> None:
    used = usage(owner_id)["bytes"]
    if used + adding > QUOTA_BYTES:
        raise ValueError(
            "That would put this account over its %d GB of stored files. "
            "Delete a dataset of images you no longer need, or raise the "
            "quota." % (QUOTA_BYTES // (1024 ** 3)))
    free = free_bytes()
    if free is not None and free - adding < DISK_FLOOR_BYTES:
        raise ValueError(
            "The disk this studio runs on has %.1f GB left, and storing this "
            "would take it below the %.0f GB it keeps free for running jobs. "
            "Free some space first -- old artifacts and datasets are the "
            "usual answer." % (free / 1024 ** 3, DISK_FLOOR_BYTES / 1024 ** 3))


# ------------------------------------------------------------------ deleting

def release(asset_ids: Iterable[str]) -> dict:
    """Drop these rows, and the files that nothing else points at.

    Reference counting is done over rows naming the same hash rather than over
    a counter column, because a counter is a number that can be wrong and a
    query cannot be. The cost is one COUNT per distinct hash, which happens
    when a dataset is deleted and never in a request somebody is waiting on.
    """
    ids = [a for a in dict.fromkeys(asset_ids) if a]
    if not ids:
        return {"rows": 0, "files": 0, "bytes": 0}
    rows = [r for r in (db.get_asset(a) for a in ids) if r]
    db.delete_assets([r["id"] for r in rows])

    files = freed = 0
    for sha in dict.fromkeys(r["sha256"] for r in rows):
        if db.assets_with_sha(sha):
            continue            # somebody else's dataset still points at it
        path = path_for(sha)
        if path.exists():
            freed += path.stat().st_size
            path.unlink()
            files += 1
            # Tidy the fan-out directories behind it, quietly.
            for parent in (path.parent, path.parent.parent):
                try:
                    parent.rmdir()
                except OSError:
                    break
    return {"rows": len(rows), "files": files, "bytes": freed}


def release_job(job_id: str) -> dict:
    """Everything a run drew while it ran, when the run goes."""
    return release([r["id"] for r in db.q(
        "SELECT id FROM assets WHERE job_id=?", (job_id,))])


def release_dataset(dataset_id: str) -> dict:
    """Everything a dataset brought with it, when the dataset goes."""
    return release([r["id"] for r in db.assets_of_dataset(dataset_id)])


def orphans() -> Iterator[Path]:
    """Files on disk that no row names. Should be empty; worth being able to
    ask, because a crash between the write and the insert leaves one."""
    known = {r["sha256"] for r in db.all_asset_shas()}
    if not ASSET_DIR.exists():
        return
    for path in ASSET_DIR.rglob("*"):
        if path.is_file() and path.name not in known \
                and not path.name.startswith("incoming-"):
            yield path


def sweep() -> dict:
    """Remove orphaned files. Safe to run at any time."""
    freed = count = 0
    for path in list(orphans()):
        freed += path.stat().st_size
        path.unlink(missing_ok=True)
        count += 1
    return {"files": count, "bytes": freed}


def clear_all() -> None:
    """Only for tests and a full reset."""
    shutil.rmtree(ASSET_DIR, ignore_errors=True)


# ------------------------------------------------------------------ sharing

def adopt(dataset_id: str, owner_id: str | None,
          asset_ids: Iterable[str]) -> dict[str, str]:
    """Give this dataset its own reference to somebody else's assets.

    A derived dataset -- a filter, a split, a merge -- copies rows that name
    assets belonging to the dataset they came from. Left alone, deleting the
    parent would take the files out from under the child.

    So the child gets its own rows, naming the same bytes. Nothing is copied
    on disk: the file is addressed by its hash and both rows point at it, and
    it goes when the last of them does. Returns old id -> new id, which the
    caller uses to rewrite the cells.

    An asset that already belongs to this dataset is left exactly as it is,
    so writing a dataset back over itself does not multiply its assets.
    """
    mapping: dict[str, str] = {}
    for row in db.get_assets(list(asset_ids)):
        if row.get("dataset_id") == dataset_id:
            continue
        if not row.get("dataset_id") and not row.get("job_id"):
            # Uploaded loose, before the dataset it was for existed -- which
            # is every file that arrives as part of an upload, since the
            # dataset is only created once its rows are known. Claimed in
            # place rather than duplicated: a second row would leave the
            # first one holding the file open for nothing, forever.
            db.claim_asset(row["id"], dataset_id)
            continue
        existing = db.find_asset(row["sha256"], owner_id, dataset_id, None)
        mapping[row["id"]] = existing["id"] if existing else db.create_asset(
            sha=row["sha256"], mime=row["mime"], size=row["size_bytes"],
            filename=row["filename"], owner_id=owner_id,
            dataset_id=dataset_id, kind=row["kind"])
    return mapping


def rewrite(value: object, mapping: dict[str, str]) -> object:
    """One cell, with any asset reference in it re-pointed."""
    if got := id_in(value):
        return ref(mapping[got]) if got in mapping else value
    if isinstance(value, list):
        return [rewrite(v, mapping) for v in value]
    return value


class Adopter:
    """Re-points the assets in rows as they stream past, once each.

    Rows are written a line at a time and a dataset may be millions of them,
    so this cannot collect them first. It keeps the map it has built so far,
    which is at most one entry per distinct asset -- and for every dataset
    that has no assets at all it is one dictionary lookup per row and nothing
    else.
    """

    __slots__ = ("dataset_id", "owner_id", "mapping", "checked")

    def __init__(self, dataset_id: str, owner_id: str | None = None) -> None:
        self.dataset_id = dataset_id
        self.owner_id = owner_id
        self.mapping: dict[str, str] = {}
        self.checked: set[str] = set()

    def __call__(self, row: dict) -> dict:
        ids = ids_in_row(row)
        if not ids:
            return row
        fresh = [i for i in ids if i not in self.checked]
        if fresh:
            self.checked.update(fresh)
            self.mapping.update(adopt(self.dataset_id, self.owner_id, fresh))
        if not self.mapping:
            return row
        return {k: rewrite(v, self.mapping) for k, v in row.items()}
