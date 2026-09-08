"""Connections to hosted models: OpenAI, Azure OpenAI, Anthropic, and the many
services that copied the OpenAI shape.

Two scopes, and the difference matters.

**Account.** An API key is usually a billable credential belonging to one
person, and "whose key paid for this dataset" should have an answer -- the same
rule the Hugging Face token already follows here.

**Studio.** But often it is not personal at all: one Azure resource, set up
once, that everybody is meant to use. Requiring each person to paste the same
key is how half a team ends up unable to generate data, and how that key ends
up in a chat message. A studio connection is configured by an admin and usable
by everyone, and it is honest about what that means -- anyone who can start a
job can spend it.

A personal connection wins over a studio one for the same provider, so
somebody who wants their own quota simply connects their own.

Either way the key is stored encrypted, never returned to the browser, and
injected into a job's config only at the moment the job is created, the way
`hf_token` is.
"""
from __future__ import annotations

import json

import httpx
from fastapi import APIRouter, Body, HTTPException, Request

from common import apimodels

from .. import auth, db
from .security import current_user

router = APIRouter(prefix="/api/providers")

# Long enough for a slow first token on a large model, short enough that a
# wrong base URL fails while somebody is still looking at the screen.
TIMEOUT = 30.0


# Where the studio-wide connections live in the settings table.
STUDIO_KEY = "model_providers"


def _decode(blob: str | None) -> dict[str, dict]:
    if not blob:
        return {}
    raw = auth.decrypt_secret(blob)
    if not raw:
        return {}
    try:
        stored = json.loads(raw)
    except ValueError:
        return {}
    return {k: v for k, v in stored.items() if isinstance(v, dict)}


def studio_connections() -> dict[str, dict]:
    """Connections configured for the whole studio."""
    return _decode(db.get_setting(STUDIO_KEY))


def account_connections(user: dict | None) -> dict[str, dict]:
    """Connections belonging to one person."""
    return _decode((user or {}).get("model_providers_enc"))


def connections(user: dict | None) -> dict[str, dict]:
    """Everything this user can reach, decrypted. Never leaves the server.

    Studio first, then the account on top: somebody who connects their own key
    for a provider the studio also has gets their own, which is the only
    sensible way round -- the personal one is the deliberate act.
    """
    merged: dict[str, dict] = {}
    for pid, conn in studio_connections().items():
        merged[pid] = {**conn, "scope": "studio"}
    for pid, conn in account_connections(user).items():
        merged[pid] = {**conn, "scope": "account"}
    return merged


def connection(user: dict | None, provider_id: str) -> dict | None:
    """One connection, with its provider id in it, ready to be used."""
    conn = connections(user).get(provider_id)
    if not conn:
        return None
    return {**conn, "provider": provider_id}


def _save(user: dict, all_conns: dict) -> None:
    db.update_user(user["id"],
                   model_providers_enc=auth.encrypt_secret(json.dumps(all_conns)))


def _save_studio(user: dict, all_conns: dict) -> None:
    db.set_setting(STUDIO_KEY, auth.encrypt_secret(json.dumps(all_conns)),
                   user.get("id"))


def _require_admin(user: dict) -> None:
    if (user or {}).get("role") != "admin":
        raise HTTPException(
            403, "A connection for the whole studio is an administrator's to "
                 "set up. You can connect your own key for this provider "
                 "instead.")


def _scope_of(payload: dict, request: Request) -> str:
    wanted = (payload.get("scope")
              or request.query_params.get("scope") or "account").strip()
    if wanted not in ("account", "studio"):
        raise HTTPException(400, "Scope is either account or studio.")
    return wanted


def _public(provider_id: str, conn: dict) -> dict:
    """What the browser may see: everything except the key itself."""
    return {
        "provider": provider_id,
        "connected": True,
        "endpoint": conn.get("endpoint") or "",
        "base_url": conn.get("base_url") or "",
        "deployment": conn.get("deployment") or "",
        "api_version": conn.get("api_version") or "",
        "model": conn.get("model") or "",
        # Not the key. Enough of it to tell two of your own keys apart, which
        # is the only thing anybody needs to see it for.
        "key_hint": ("…" + conn["api_key"][-4:]) if conn.get("api_key") else "",
        "checked_at": conn.get("checked_at"),
        "note": conn.get("note") or "",
        "api_style": conn.get("api_style") or "",
        # Whose credential this is. The UI says so plainly, because "shared
        # with everyone in this studio" is not a detail somebody should have
        # to infer from a key working that they never pasted.
        "scope": conn.get("scope") or "account",
    }


@router.get("")
async def list_providers(request: Request) -> dict:
    """The catalogue, and which of them this account has connected."""
    user = current_user(request)
    mine = connections(user)
    return {
        "providers": apimodels.public_providers(),
        "connected": [_public(pid, conn) for pid, conn in sorted(mine.items())],
        # Whether this account may connect something on everyone's behalf.
        "can_manage_studio": (user or {}).get("role") == "admin",
    }


@router.put("/{provider_id}")
async def save_provider(request: Request, provider_id: str,
                        payload: dict = Body(...)) -> dict:
    user = current_user(request)
    spec = apimodels.provider(provider_id)
    if not spec:
        raise HTTPException(404, "No such provider.")

    scope = _scope_of(payload, request)
    if scope == "studio":
        _require_admin(user)
        mine = studio_connections()
    else:
        mine = account_connections(user)
    conn = dict(mine.get(provider_id) or {})
    for field in spec["required"] + spec["optional"] + ["model", "api_style"]:
        if field not in payload:
            continue
        value = (payload.get(field) or "").strip()
        # A blank key means "leave the one you have", not "delete it" -- the
        # form cannot show the stored key, so it cannot send it back either.
        if field == "api_key" and not value:
            continue
        conn[field] = value
    conn["note"] = (payload.get("note") or "").strip()

    if problem := apimodels.problems({**conn, "provider": provider_id}):
        raise HTTPException(400, problem)
    mine[provider_id] = conn
    if scope == "studio":
        _save_studio(user, mine)
    else:
        _save(user, mine)
    return _public(provider_id, {**conn, "scope": scope})


@router.delete("/{provider_id}")
async def delete_provider(request: Request, provider_id: str) -> dict:
    user = current_user(request)
    scope = _scope_of({}, request)
    if scope == "studio":
        _require_admin(user)
        mine = studio_connections()
        if mine.pop(provider_id, None) is None:
            raise HTTPException(404, "That provider is not connected for the "
                                     "studio.")
        _save_studio(user, mine)
        return {"ok": True, "note": "Disconnected for everyone. Runs already "
                                    "queued keep going."}
    mine = account_connections(user)
    if mine.pop(provider_id, None) is None:
        raise HTTPException(404, "That provider is not connected.")
    _save(user, mine)
    # Jobs already queued keep the credential they were given: they were
    # authorised when they were created, and failing them now would be a
    # surprise rather than a security improvement.
    return {"ok": True, "note": "Disconnected. Runs already queued keep going."}


@router.get("/{provider_id}/models")
async def provider_models(request: Request, provider_id: str) -> dict:
    """What this connection can reach, asked of the provider itself.

    Fetched rather than hardcoded. A list of model names baked into this app
    is wrong the week after it ships, and wrong in the direction that hides
    the model somebody is paying for.
    """
    user = current_user(request)
    conn = connection(user, provider_id)
    if not conn:
        raise HTTPException(400, "That provider is not connected.")
    req = apimodels.models_request(conn)
    if not req:
        return {"models": [apimodels.model_name(conn, None)],
                "note": "Azure lists deployments through its management API, "
                        "not this one. The deployment you named is the model."}
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            r = await client.get(req["url"], headers=req["headers"])
    except httpx.HTTPError as e:
        raise HTTPException(502, "Could not reach %s: %s"
                            % (apimodels.describe(conn), e)) from e
    if r.status_code >= 400:
        raise HTTPException(400, apimodels.error_message(
            conn, r.status_code, _body(r)))
    return {"models": apimodels.models_from(r.json())}


@router.post("/{provider_id}/test")
async def test_provider(request: Request, provider_id: str,
                        payload: dict = Body(default={})) -> dict:
    """Say hello to the model, and report exactly what came back.

    A real completion rather than a model listing: listing works with a key
    that has no access to any model, and Azure cannot list at all. The point
    of this button is to find out now instead of on row 1 of 5,000.
    """
    user = current_user(request)
    conn = connection(user, provider_id)
    if not conn:
        raise HTTPException(400, "That provider is not connected.")
    model = apimodels.model_name(conn, payload.get("model"))
    if not model:
        raise HTTPException(400, "Which model should it try?")

    req = apimodels.chat_request(
        conn, model,
        [{"role": "user", "content": "Reply with the single word: ready"}],
        {"max_new_tokens": 16, "temperature": 0})
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            r = await client.post(req["url"], headers=req["headers"],
                                  json=req["json"])
            if r.status_code >= 400:
                if fixed := apimodels.retry_body(conn, req["json"], r.text):
                    r = await client.post(req["url"], headers=req["headers"],
                                          json=fixed)
    except httpx.HTTPError as e:
        raise HTTPException(502, "Could not reach %s: %s"
                            % (apimodels.describe(conn), e)) from e
    if r.status_code >= 400:
        raise HTTPException(400, apimodels.error_message(
            conn, r.status_code, _body(r)))

    data = r.json()
    # Stamped on whichever record was actually used, not on a merged copy that
    # belongs to nobody -- writing the studio's connection into the caller's
    # account would silently give them a private copy of a shared key.
    scope = conn.get("scope") or "account"
    store = studio_connections() if scope == "studio"         else account_connections(user)
    if provider_id in store:
        store[provider_id]["checked_at"] = __import__("time").time()
        if model and not store[provider_id].get("model"):
            store[provider_id]["model"] = model
        if scope == "studio":
            if user.get("role") == "admin":
                _save_studio(user, store)
        else:
            _save(user, store)
    answer = apimodels.reply(conn, data)
    return {"ok": True, "model": model, "scope": scope,
            "reply": (answer["content"] or answer["reasoning"])[:200],
            "usage": apimodels.chat_usage(conn, data)}


def _body(r: httpx.Response):
    try:
        return r.json()
    except ValueError:
        return r.text
