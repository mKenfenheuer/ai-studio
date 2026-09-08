"""The model library: models that were published, here or to the Hub.

Not every run -- most runs are attempts -- but the ones somebody decided were
finished enough to have a name and a version and be used by other people. A
library entry points at a run and says what it was published as; the run
keeps the weights, the chart and the log. Publishing to Hugging Face files
an entry automatically when the upload finishes; publishing locally is a
button that asks for a name and a version and nothing else.
"""
from __future__ import annotations

from fastapi import APIRouter, Body, HTTPException, Query, Request

from .. import db, serving
from .security import current_user, require_view

router = APIRouter(prefix="/api/library")


def _public(r: dict) -> dict:
    summary = r.get("run_summary") or {}
    cfg = r.get("run_config") or {}
    pm = summary.get("primary_metric") or (
        {"label": "Held-out loss", "value": summary.get("best_val_loss"),
         "lower_better": True} if summary.get("best_val_loss") is not None else None)
    return {
        "id": r["id"], "name": r["name"], "version": r.get("version") or "",
        "notes": r.get("notes") or "", "location": r["location"],
        "repo_id": r.get("repo_id"), "url": r.get("url"),
        "published_at": r["published_at"], "mine": r.get("mine", False),
        "project_id": r.get("project_id"), "project_name": r.get("project_name"),
        "job_id": r["job_id"], "run_name": r.get("run_name"),
        "run_kind": r.get("run_kind"), "has_model": r.get("has_model", False),
        "base_model": cfg.get("base_model_label") or cfg.get("base_model"),
        "dataset": cfg.get("dataset_label") or cfg.get("dataset"),
        "primary_metric": pm,
        "served_as": db.aliases_for_job(r["job_id"]),
    }


@router.get("")
async def list_library(request: Request, project_id: str = Query(default="")) -> list[dict]:
    user = current_user(request)
    return [_public(r) for r in db.library_rows(user, project_id or None)]


@router.post("")
async def publish_locally(request: Request, payload: dict = Body(...)) -> dict:
    """Put a run's model in the library, under a name and a version."""
    user = current_user(request)
    job = db.get_job(payload.get("job_id") or "")
    if not job:
        raise HTTPException(404, "No such run.")
    require_view(request, "job", job)
    if job["kind"] not in serving.MODEL_KINDS and job["kind"] != "finetune_vision_cls":
        raise HTTPException(400, "That run did not produce a model.")
    if not db.list_artifacts(job["id"]):
        raise HTTPException(400, "That run has no saved model to publish.")
    name = (payload.get("name") or job["name"]).strip()[:120]
    version = (payload.get("version") or "").strip()[:40]
    project_id = payload.get("project_id") or job.get("project_id")
    lid = db.publish_to_library(job["id"], user["id"], name, version,
                                (payload.get("notes") or "")[:2000],
                                location="local", project_id=project_id)
    db.add_log(job["id"], "Published to the model library as %s%s."
               % (name, " " + version if version else ""))
    rows = [r for r in db.library_rows(user) if r["id"] == lid]
    return _public(rows[0]) if rows else {"id": lid}


@router.delete("/{library_id}")
async def unpublish(request: Request, library_id: str) -> dict:
    """Take an entry out of the library. The run and its model stay."""
    user = current_user(request)
    row = db.get_library_row(library_id)
    if not row:
        raise HTTPException(404, "No such entry.")
    if row.get("owner_id") != user["id"] and user.get("role") != "admin":
        raise HTTPException(403, "That entry belongs to somebody else.")
    db.delete_library_row(library_id)
    return {"ok": True, "note": "Removed from the library. The run and its model "
                                "are untouched%s." % (
                                    "; the Hub repository is not touched either"
                                    if row["location"] == "hf" else "")}
