"""Letting a colleague in on a run or a dataset.

Two subjects only: a named person, or everyone with an account here. There is
deliberately no "anyone with the link" -- this studio has no notion of an
anonymous visitor, and inventing one would mean inventing a way to reach data
without signing in, which is the thing accounts were added to prevent.

Two levels, because "can see it" and "can change it" are genuinely different
questions and collapsing them makes sharing something people avoid doing.
Re-sharing and deleting stay with the owner in both cases.
"""
from __future__ import annotations

from fastapi import APIRouter, Body, HTTPException, Request

from .. import db
from .security import current_user, require_owner

router = APIRouter(prefix="/api")

KINDS = {"job": "run", "dataset": "dataset", "eval": "prompt set", "project": "project"}


def _resource(kind: str, resource_id: str) -> dict:
    if kind not in KINDS:
        raise HTTPException(404, "Nothing of that kind can be shared.")
    row = {"job": db.get_job, "dataset": db.get_dataset,
           "eval": db.get_eval, "project": db.get_project}[kind](resource_id)
    if not row:
        raise HTTPException(404, "No such %s." % KINDS[kind])
    return row


@router.get("/{kind}s/{resource_id}/shares")
async def get_shares(request: Request, kind: str, resource_id: str) -> dict:
    row = _resource(kind, resource_id)
    user = current_user(request)
    level = db.access_level(kind, resource_id, row.get("owner_id"), user)
    if not level:
        raise HTTPException(404, "No such %s." % KINDS[kind])
    owner = db.get_user(row["owner_id"]) if row.get("owner_id") else None
    return {
        "owner": db.public_user(owner) if owner else None,
        "mine": row.get("owner_id") == user["id"] or user["role"] == "admin",
        "your_access": level,
        "shares": db.list_shares(kind, resource_id),
    }


@router.post("/{kind}s/{resource_id}/shares")
async def add_share(request: Request, kind: str, resource_id: str,
                    payload: dict = Body(...)) -> dict:
    row = _resource(kind, resource_id)
    user = require_owner(request, kind, row)

    level = payload.get("level") or "view"
    if level not in db.LEVELS:
        raise HTTPException(400, "Share as 'view' or as 'edit'.")

    subject_type = payload.get("subject_type") or "user"
    subject_id = payload.get("subject_id")
    if subject_type == "everyone":
        subject_id = None
    elif subject_type == "user":
        target = db.get_user(subject_id or "")
        if not target:
            raise HTTPException(404, "No such person.")
        if target["id"] == row.get("owner_id"):
            raise HTTPException(400, "That is already the owner.")
    else:
        raise HTTPException(400, "Share with a person, or with everyone.")

    db.share(kind, resource_id, subject_type, subject_id, level, user["id"])
    return {"ok": True, "shares": db.list_shares(kind, resource_id)}


@router.delete("/{kind}s/{resource_id}/shares")
async def remove_share(request: Request, kind: str, resource_id: str,
                       subject_type: str = "user",
                       subject_id: str | None = None) -> dict:
    row = _resource(kind, resource_id)
    user = current_user(request)
    # Someone can always remove their own access, without owning anything --
    # otherwise leaving a project you were added to requires asking the person
    # who added you.
    if not (subject_type == "user" and subject_id == user["id"]):
        require_owner(request, kind, row)
    db.unshare(kind, resource_id, subject_type, subject_id)
    return {"ok": True, "shares": db.list_shares(kind, resource_id)}


@router.post("/{kind}s/{resource_id}/transfer")
async def transfer(request: Request, kind: str, resource_id: str,
                   payload: dict = Body(...)) -> dict:
    """Hand ownership to somebody else, keeping yourself on as an editor."""
    row = _resource(kind, resource_id)
    user = require_owner(request, kind, row)
    target = db.get_user(payload.get("user_id") or "")
    if not target:
        raise HTTPException(404, "No such person.")
    if not target["active"]:
        raise HTTPException(400, "That account is disabled.")

    table = {"job": "jobs", "dataset": "datasets", "eval": "evals"}[kind]
    db.ex("UPDATE %s SET owner_id=? WHERE id=?" % table, (target["id"], resource_id))
    db.unshare(kind, resource_id, "user", target["id"])
    if row.get("owner_id") and row["owner_id"] != target["id"]:
        db.share(kind, resource_id, "user", row["owner_id"], "edit", user["id"])
    return {"ok": True, "owner": db.public_user(target)}
