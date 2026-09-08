"""Serving and storing the bytes a dataset points at.

The store itself is `controller/assets.py`; this is the door onto it.

Two things here are security decisions rather than conveniences, and both are
about the fact that these bytes are served from the studio's own origin:

* **The type is an allowlist**, checked at upload, and echoed back exactly.
  A store that will serve `text/html` a stranger uploaded, from the same
  origin as the session cookie, is a cross-site scripting hole with a database
  behind it. SVG is excluded for the same reason -- it is a document that can
  run script, wearing an image's extension.
* **Every response says so again anyway**: `nosniff`, so a browser does not
  decide for itself that a JPEG is markup, and a content policy that forbids
  the file from loading anything at all.

Reading is by id and the id is unguessable, but that is not the check --
access is decided by what the asset belongs to. An asset in a dataset is
readable by whoever may read the dataset, which is what makes sharing a
dataset of images work at all.
"""
from __future__ import annotations

from fastapi import (APIRouter, File, Header, HTTPException, Query, Request,
                     UploadFile)
from fastapi.responses import FileResponse

from .. import assets, config, db
from .security import current_user

router = APIRouter(prefix="/api/assets")

# Content-addressed, so the bytes at a URL can never change. A year is what
# "immutable" is conventionally paired with.
CACHE = "private, max-age=31536000, immutable"

SAFE_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Content-Security-Policy": "default-src 'none'; sandbox",
}


def _may_read(request: Request, asset: dict) -> bool:
    user = current_user(request)
    if asset.get("owner_id") and asset["owner_id"] == user["id"]:
        return True
    if did := asset.get("dataset_id"):
        d = db.get_dataset(did)
        if d and db.access_level("dataset", did, d.get("owner_id"), user):
            return True
    if jid := asset.get("job_id"):
        j = db.get_job(jid)
        if j and db.access_level("job", jid, j.get("owner_id"), user):
            return True
    return False


def _asset_or_404(request: Request, asset_id: str) -> dict:
    asset = db.get_asset(asset_id)
    # The same 404-not-403 rule as everywhere else: a file you may not read
    # must not be distinguishable from one that is not there.
    if not asset or not _may_read(request, asset):
        raise HTTPException(404, "No such file.")
    return asset


@router.get("")
async def list_assets(request: Request, dataset_id: str = Query(default="")) -> dict:
    """What is stored, and how much room it is taking."""
    user = current_user(request)
    out: dict = {"usage": assets.usage(user["id"]),
                 "studio": assets.usage() if user.get("role") == "admin" else None,
                 "quota_bytes": assets.QUOTA_BYTES,
                 "max_bytes": assets.MAX_BYTES}
    if dataset_id:
        d = db.get_dataset(dataset_id)
        if not d or not db.access_level("dataset", dataset_id,
                                        d.get("owner_id"), user):
            raise HTTPException(404, "No such dataset.")
        out["assets"] = [_public(a) for a in db.assets_of_dataset(dataset_id)]
    return out


def _public(a: dict) -> dict:
    return {"id": a["id"], "mime": a["mime"], "kind": a["kind"],
            "filename": a["filename"], "bytes": a["size_bytes"],
            "created_at": a["created_at"],
            "url": "/api/assets/%s" % a["id"], "ref": assets.ref(a["id"])}


@router.post("")
async def upload_assets(request: Request,
                        file: list[UploadFile] = File(...),
                        dataset_id: str = Query(default="")) -> dict:
    """Put files in the store, optionally as part of a dataset."""
    user = current_user(request)
    if dataset_id:
        d = db.get_dataset(dataset_id)
        if not d:
            raise HTTPException(404, "No such dataset.")
        if not db.access_level("dataset", dataset_id, d.get("owner_id"),
                               user) in ("edit", "own"):
            raise HTTPException(404, "No such dataset.")

    stored, refused = [], []
    for f in file:
        mime = assets.guess_mime(f.filename or "", f.content_type or "")
        if not mime:
            refused.append({
                "filename": f.filename or "(unnamed)",
                "why": "The studio does not store files of that kind. Images, "
                       "audio, video and PDFs, by their extension."})
            continue
        try:
            assets.check_quota(user["id"])
            row = assets.store(_chunks(f), f.filename or "", mime,
                               owner_id=user["id"],
                               dataset_id=dataset_id or None)
        except ValueError as e:
            refused.append({"filename": f.filename or "(unnamed)",
                            "why": str(e)})
            continue
        stored.append(_public(row))
    if not stored and refused:
        raise HTTPException(400, refused[0]["why"])
    return {"stored": stored, "refused": refused,
            "usage": assets.usage(user["id"])}


def _chunks(f: UploadFile):
    """The upload, a megabyte at a time.

    `UploadFile` spools to disk past a threshold, so reading it whole would
    mean the file is on the disk twice and in memory once, to be written to
    the disk a third time.
    """
    while True:
        chunk = f.file.read(assets.CHUNK)
        if not chunk:
            return
        yield chunk


@router.get("/{asset_id}")
async def get_asset(request: Request, asset_id: str,
                    download: bool = Query(default=False)) -> FileResponse:
    asset = _asset_or_404(request, asset_id)
    path = assets.path_for(asset["sha256"])
    if not path.exists():
        # A row whose file is gone. Says which, because "not found" on
        # something the database swears exists sends people looking in the
        # wrong place for an hour.
        raise HTTPException(
            410, "The record of this file is here but its contents are not. "
                 "It was removed from disk without going through the studio.")
    return FileResponse(
        path, media_type=asset["mime"],
        filename=asset["filename"] if download else None,
        headers={**SAFE_HEADERS, "Cache-Control": CACHE,
                 "ETag": '"%s"' % asset["sha256"],
                 **({} if download else
                    {"Content-Disposition": "inline"})})


@router.delete("/{asset_id}")
async def delete_asset(request: Request, asset_id: str) -> dict:
    asset = _asset_or_404(request, asset_id)
    user = current_user(request)
    if asset.get("owner_id") and asset["owner_id"] != user["id"] \
            and user.get("role") != "admin":
        raise HTTPException(403, "That file belongs to somebody else.")
    freed = assets.release([asset_id])
    return {"ok": True, **freed,
            "note": "The reference is gone. The file itself stays as long as "
                    "anything else points at it." if not freed["files"]
                    else "Removed, and the last reference to it went with it."}


# Ends in a fixed word because the runner-token check matches paths by their
# last segment, and a path that ends in a job id has no last segment to match.
@router.post("/from-runner/{job_id}/sample")
async def upload_from_runner(job_id: str, file: UploadFile = File(...),
                             x_runner_token: str = Header(default="")) -> dict:
    """A file a running job produced -- a sample grid, a clip -- into the store.

    The runner's door, not a person's: authenticated by the join token, as
    artifact uploads are, and the asset belongs to the job rather than to an
    account, so whoever may see the run may see what it drew. It is released
    when the run is deleted, with everything else the run left.
    """
    if x_runner_token != config.join_token():
        raise HTTPException(403, "Invalid runner token.")
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(404, "No such run.")
    mime = assets.guess_mime(file.filename or "", file.content_type or "")
    if not mime:
        raise HTTPException(400, "Not a kind of file the studio stores.")
    try:
        row = assets.store(_chunks(file), file.filename or "", mime,
                           owner_id=job.get("owner_id"), job_id=job_id)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return _public(row)
