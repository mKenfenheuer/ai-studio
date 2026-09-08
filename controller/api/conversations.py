"""Conversations somebody wanted to keep.

The playground threw its conversations away on navigation, so the exchange
that showed a model at its best -- or its worst -- was gone the moment
somebody went to look at the run that produced it. A saved conversation is a
named message list, private to the person who saved it, that can be reopened
against the run it was had with or against a later one, which is the
comparison the playground could not make: the same questions to last
month's model and this month's.
"""
from __future__ import annotations

from fastapi import APIRouter, Body, HTTPException, Query, Request

from .. import db
from .security import current_user

router = APIRouter(prefix="/api/conversations")


@router.get("")
async def list_saved(request: Request, job_id: str = Query(default="")) -> list[dict]:
    return db.list_conversations(current_user(request)["id"], job_id or None)


@router.get("/{conversation_id}")
async def get_saved(request: Request, conversation_id: str) -> dict:
    user = current_user(request)
    row = db.get_conversation(conversation_id)
    if not row or row.get("owner_id") != user["id"]:
        raise HTTPException(404, "No such conversation.")
    return row


@router.post("")
async def save(request: Request, payload: dict = Body(...)) -> dict:
    user = current_user(request)
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise HTTPException(400, "There is nothing to save yet.")
    if len(messages) > 400:
        raise HTTPException(400, "That is more than four hundred turns.")
    title = (payload.get("title") or "").strip()[:120]
    if not title:
        first = next((m.get("content") for m in messages
                      if m.get("role") == "user" and m.get("content")), "")
        title = (first or "A conversation")[:60]
    cid = payload.get("id")
    if cid:
        existing = db.get_conversation(cid)
        if not existing or existing.get("owner_id") != user["id"]:
            raise HTTPException(404, "No such conversation.")
    cid = db.save_conversation(user["id"], payload.get("job_id") or None, title,
                               messages, payload.get("tools") or [], cid)
    return db.get_conversation(cid)


@router.delete("/{conversation_id}")
async def delete_saved(request: Request, conversation_id: str) -> dict:
    if not db.delete_conversation(current_user(request)["id"], conversation_id):
        raise HTTPException(404, "No such conversation.")
    return {"ok": True}
