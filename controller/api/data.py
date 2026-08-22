"""Datasets: bringing them in, looking at them, fixing them, sharing them."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import (APIRouter, Body, HTTPException, Query, Request, Response,
                     UploadFile)
from fastapi.responses import FileResponse

from common import formatting

from .. import datasets as ds
from .. import db, hfaccount
from .security import current_user, require_edit, require_owner, require_view

router = APIRouter(prefix="/api/datasets")

# An upload has to fit in memory to be parsed, and the controller is not the
# machine with 64 GB in it. Large corpora belong on the Hub, where they can be
# streamed; this is for the file on somebody's laptop.
MAX_UPLOAD_BYTES = 256 * 1024 * 1024


def _get(request: Request, dataset_id: str, need: str = "view") -> dict:
    d = db.get_dataset(dataset_id)
    if not d:
        raise HTTPException(404, "No such dataset.")
    if need == "edit":
        require_edit(request, "dataset", d)
    elif need == "own":
        require_owner(request, "dataset", d)
    else:
        require_view(request, "dataset", d)
    return d


@router.get("")
async def list_datasets(request: Request) -> list[dict]:
    return db.visible_datasets(current_user(request))


@router.get("/{dataset_id}")
async def get_dataset(request: Request, dataset_id: str) -> dict:
    d = _get(request, dataset_id)
    user = current_user(request)
    d["access"] = db.access_level("dataset", d["id"], d.get("owner_id"), user)
    d["mine"] = d.get("owner_id") == user["id"]
    d["shares"] = db.list_shares("dataset", dataset_id)
    d["preview"] = list(ds.iter_rows(dataset_id, ds.PREVIEW_ROWS))
    if parent := d.get("parent_id"):
        if p := db.get_dataset(parent):
            d["parent_name"] = p["name"]
    return d


@router.get("/{dataset_id}/inspect")
async def inspect_dataset(request: Request, dataset_id: str,
                          sample: int = 2000) -> dict:
    return ds.inspect(_get(request, dataset_id), min(int(sample), 20000))


@router.get("/{dataset_id}/rows")
async def dataset_rows(request: Request, dataset_id: str, offset: int = 0,
                       limit: int = 25) -> dict:
    d = _get(request, dataset_id)
    limit = min(max(int(limit), 1), 200)
    rows = []
    fmt = formatting.resolve_format(d.get("format") or {})
    for i, row in enumerate(ds.iter_rows(dataset_id, offset + limit)):
        if i < offset:
            continue
        rows.append({"index": i, "row": row,
                     "rendered": formatting.format_example(row, fmt) or ""})
    return {"rows": rows, "offset": offset, "total": d.get("rows") or 0}


@router.get("/{dataset_id}/dataset-file")
async def download_dataset(request: Request, dataset_id: str):
    """The raw JSONL.

    Also how a runner fetches a studio dataset to train on -- it presents the
    join token instead of a session, which the middleware allows for this path.
    """
    if not getattr(request.state, "runner", False):
        _get(request, dataset_id)
    p = ds.path_for(dataset_id)
    if not p.exists():
        raise HTTPException(404, "This dataset has no file.")
    name = (db.get_dataset(dataset_id) or {}).get("name") or dataset_id
    safe = "".join(c for c in name if c.isalnum() or c in "-_ ").strip() or dataset_id
    return FileResponse(p, media_type="application/x-ndjson",
                        filename="%s.jsonl" % safe.replace(" ", "-"))


# ---------------------------------------------------------------------------
# Getting data in
# ---------------------------------------------------------------------------

@router.post("/upload")
async def upload_dataset(request: Request, file: UploadFile,
                         name: str = Query(default="")) -> dict:
    user = current_user(request)
    blob = await file.read()
    if len(blob) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            413, "That file is larger than %d MB. Put a corpus this size on "
                 "the Hub and import it by name instead."
                 % (MAX_UPLOAD_BYTES // (1024 * 1024)))
    try:
        rows = list(ds.rows_from_upload(file.filename or "", blob))
    except ValueError as e:
        raise HTTPException(
            400, "That file could not be read as JSONL, JSON, CSV or plain "
                 "text (%s)." % e) from e
    if not rows:
        raise HTTPException(400, "That file contained no rows.")

    created = ds.register(user["id"], name or (file.filename or "Uploaded data"),
                          "upload", iter(rows), origin=file.filename)
    return created


@router.post("/import")
async def import_dataset(request: Request, payload: dict = Body(...)) -> dict:
    user = current_user(request)
    hub_id = (payload.get("dataset") or "").strip()
    if not hub_id:
        raise HTTPException(400, "Which dataset on the Hub?")
    try:
        rows = await ds.rows_from_hub(
            hub_id, payload.get("config"), payload.get("split") or "train",
            int(payload.get("limit") or 5000), hfaccount.token_for(user))
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    if not rows:
        raise HTTPException(400, "That dataset returned no rows.")

    created = ds.register(
        user["id"], payload.get("name") or hub_id.split("/")[-1], "hub",
        iter(rows), origin=hub_id,
        notes="Imported %s rows from %s (%s / %s)."
              % (f"{len(rows):,}", hub_id, payload.get("config") or "default",
                 payload.get("split") or "train"))
    return created


# ---------------------------------------------------------------------------
# Changing it
# ---------------------------------------------------------------------------

@router.patch("/{dataset_id}")
async def rename_dataset(request: Request, dataset_id: str,
                         payload: dict = Body(...)) -> dict:
    d = _get(request, dataset_id, "edit")
    fields = {}
    if "name" in payload:
        fields["name"] = (payload["name"] or "").strip()[:120] or d["name"]
    if "notes" in payload:
        fields["notes"] = (payload["notes"] or "")[:4000]
    if "format" in payload:
        fields["format"] = payload["format"] or {}
    db.update_dataset(dataset_id, **fields)
    return db.get_dataset(dataset_id)


@router.post("/{dataset_id}/transform")
async def transform_dataset(request: Request, dataset_id: str,
                            payload: dict = Body(...)) -> dict:
    d = _get(request, dataset_id)
    user = current_user(request)
    try:
        return ds.transform(d, payload.get("ops") or {}, user["id"],
                            payload.get("name"))
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@router.post("/{dataset_id}/split")
async def split_dataset(request: Request, dataset_id: str,
                        payload: dict = Body(...)) -> list[dict]:
    d = _get(request, dataset_id)
    user = current_user(request)
    try:
        return ds.split(d, float(payload.get("fraction") or 0.1), user["id"],
                        int(payload.get("seed") or 1234))
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@router.post("/merge")
async def merge_datasets(request: Request, payload: dict = Body(...)) -> dict:
    user = current_user(request)
    ids = payload.get("datasets") or []
    if len(ids) < 2:
        raise HTTPException(400, "Choose at least two datasets to merge.")
    parts = []
    for i in ids:
        d = db.get_dataset(i)
        if not d:
            raise HTTPException(404, "No such dataset: %s" % i)
        require_view(request, "dataset", d)
        parts.append(d)
    try:
        return ds.merge(parts, user["id"],
                        payload.get("name") or "Merged dataset",
                        bool(payload.get("shuffle", True)))
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@router.delete("/{dataset_id}")
async def delete_dataset(request: Request, dataset_id: str) -> dict:
    d = _get(request, dataset_id, "own")
    children = db.q("SELECT id, name FROM datasets WHERE parent_id=?", (dataset_id,))
    ds.delete_files(dataset_id)
    db.clear_shares("dataset", dataset_id)
    db.delete_dataset(dataset_id)
    return {"ok": True,
            "note": ("%d dataset(s) made from this one were kept."
                     % len(children)) if children else None}


@router.post("/{dataset_id}/publish")
async def publish_dataset(request: Request, dataset_id: str,
                          payload: dict = Body(...)) -> dict:
    d = _get(request, dataset_id)
    user = current_user(request)
    try:
        return await hfaccount.publish_dataset(
            user, d, ds.path_for(dataset_id), (payload.get("repo_id") or "").strip(),
            bool(payload.get("private", True)), payload.get("message") or "")
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:  # noqa: BLE001 - hub errors are not ours to classify
        raise HTTPException(502, "Hugging Face refused the upload: %s" % e) from e
