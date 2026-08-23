"""The endpoints behind a "Sign in with…" button, and the screen that sets it up.

Three of these are reachable without a session, which is unavoidable -- they
are how a session is obtained -- so each one is written to be safe for a
stranger to call:

* `/api/auth/providers` returns a name and a button and nothing else. Not the
  client id, not the issuer, not whether anybody has ever signed in with it.
* `/api/auth/sso/{id}/start` only redirects. It creates a row that expires in
  ten minutes and reveals nothing about the studio.
* `/api/auth/sso/{id}/callback` refuses anything whose `state` it did not
  itself write, which is the whole of CSRF protection for this flow.

Everything else here needs an administrator.
"""
from __future__ import annotations

import json
import time
from urllib.parse import quote

from fastapi import APIRouter, Body, HTTPException, Query, Request
from fastapi.responses import RedirectResponse

from .. import auth, config, db, directory, oidc
from .accounts import _set_cookie
from .security import require_admin

router = APIRouter(prefix="/api")


def public_base(request: Request) -> str:
    """The address a browser reaches this studio at.

    Not `request.base_url`, at least not first. A redirect URI has to match the
    one registered with the provider *exactly*, and behind any reverse proxy
    the URL the application sees is the internal one -- http, port 8420,
    container hostname -- which matches nothing anybody registered.
    """
    if fixed := config.PUBLIC_URL:
        return fixed.rstrip("/")
    proto = request.headers.get("x-forwarded-proto", "").split(",")[0].strip()
    host = request.headers.get("x-forwarded-host", "").split(",")[0].strip()
    if proto and host:
        return "%s://%s" % (proto, host)
    return str(request.base_url).rstrip("/")


def redirect_uri(request: Request, idp_id: str) -> str:
    return "%s/api/auth/sso/%s/callback" % (public_base(request), idp_id)


# ---------------------------------------------------------------------------
# Signing in
# ---------------------------------------------------------------------------

@router.get("/auth/providers")
async def providers() -> dict:
    idps = [oidc.public_idp(i) for i in db.list_idps(enabled_only=True)]
    return {
        "providers": idps,
        # Password sign-in is never taken away by configuring SSO. A studio
        # whose provider is unreachable at 8am must not be a studio nobody can
        # get into, and somebody always has to be able to fix it.
        "password_login": True,
    }


@router.get("/auth/sso/{idp_id}/start")
async def start(request: Request, idp_id: str, next: str | None = None):
    idp = db.get_idp(idp_id)
    if not idp or not idp["enabled"]:
        return _back(request, "That sign-in method is not available here.")
    try:
        url = oidc.begin(idp, redirect_uri(request, idp_id), _safe_next(next))
    except oidc.SsoError as e:
        return _back(request, str(e))
    return RedirectResponse(url, status_code=302)


@router.get("/auth/sso/{idp_id}/callback")
async def callback(request: Request, idp_id: str,
                   code: str | None = None, state: str | None = None,
                   error: str | None = None,
                   error_description: str | None = None):
    if error:
        # The provider itself refused -- consent declined, app disabled, user
        # not assigned. Its words are more useful than any of ours.
        return _back(request, error_description or error)
    if not code or not state:
        return _back(request, "That sign-in did not complete.")
    try:
        user, next_url = await oidc.complete(state, code)
    except oidc.SsoError as e:
        return _back(request, str(e))

    response = RedirectResponse(
        public_base(request) + "/" + (next_url or ""), status_code=302)
    _set_cookie(response, request,
                auth.start_session(user["id"],
                                   request.headers.get("user-agent", "")))
    auth.purge_expired_sessions()
    return response


def _safe_next(value: str | None) -> str | None:
    """Only ever a hash route on this studio.

    An open redirect is the classic way a sign-in flow becomes a phishing
    tool: the link looks like ours, and the destination is not.
    """
    value = (value or "").strip()
    if value.startswith("#/") and "\\" not in value and "//" not in value[1:]:
        return value
    return None


def _back(request: Request, message: str) -> RedirectResponse:
    return RedirectResponse(
        "%s/?sso_error=%s" % (public_base(request), quote(message[:400])),
        status_code=302)


# ---------------------------------------------------------------------------
# Setting it up
# ---------------------------------------------------------------------------

def free_slug(kind: str) -> str:
    """The id a provider of this kind would get.

    Providers are named after their kind rather than given a random id, and it
    matters: the redirect URI contains the id, and it has to be registered at
    the provider *before* this studio has one to give. A predictable slug is
    what lets the setup screen show the exact address to paste while the form
    is still empty.
    """
    taken = {i["id"] for i in db.list_idps()}
    if kind not in taken:
        return kind
    for n in range(2, 50):
        if "%s%d" % (kind, n) not in taken:
            return "%s%d" % (kind, n)
    raise HTTPException(400, "That is a lot of sign-in methods.")


@router.get("/idp/presets")
async def presets(request: Request) -> dict:
    require_admin(request)
    base = public_base(request)
    return {
        "presets": oidc.PRESETS,
        "base": base,
        # What each kind would be called, and therefore where its provider
        # should be told to send people back to.
        "redirect_uris": {kind: "%s/api/auth/sso/%s/callback"
                                % (base, free_slug(kind))
                          for kind in oidc.PRESETS},
    }


@router.get("/idp")
async def list_idps(request: Request) -> list[dict]:
    require_admin(request)
    out = []
    for idp in db.list_idps():
        row = oidc.admin_idp(idp)
        row["redirect_uri"] = redirect_uri(request, idp["id"])
        out.append(row)
    return out


def _settings(payload: dict, existing: dict | None = None) -> dict:
    """Validate what an administrator typed, once, for create and for edit."""
    fields: dict = {}
    if "name" in payload:
        fields["name"] = (payload.get("name") or "").strip()[:60]
    for flag in ("enabled", "auto_create", "link_by_email", "sync_enabled"):
        if flag in payload:
            fields[flag] = 1 if payload[flag] else 0
    for text in ("allowed_domains", "admin_groups", "sync_group",
                 "sync_subject", "scopes"):
        if text in payload:
            fields[text] = (payload.get(text) or "").strip()[:400] or None
    if "client_id" in payload:
        client_id = (payload.get("client_id") or "").strip()
        if not client_id:
            raise HTTPException(400, "The application (client) ID is required.")
        fields["client_id"] = client_id
    if payload.get("client_secret"):
        fields["client_secret_enc"] = auth.encrypt_secret(payload["client_secret"])
    elif payload.get("client_secret") == "":
        fields["client_secret_enc"] = None
    if payload.get("sync_secret"):
        fields["sync_secret_enc"] = auth.encrypt_secret(payload["sync_secret"])
    elif payload.get("sync_secret") == "":
        fields["sync_secret_enc"] = None

    # Turning on sync without a way to read the directory is the mistake worth
    # catching here rather than at 2am when the timer first fires.
    kind = (existing or {}).get("kind") or payload.get("kind")
    if fields.get("sync_enabled") and not directory.supports({"kind": kind}):
        raise HTTPException(
            400, "There is no standard way to read a list of people out of an "
                 "OpenID Connect provider. Accounts here are created as people "
                 "sign in instead.")
    return fields


@router.post("/idp")
async def create_idp(request: Request, payload: dict = Body(...)) -> dict:
    admin = require_admin(request)
    kind = (payload.get("kind") or "oidc").strip()
    if kind not in oidc.PRESETS:
        raise HTTPException(400, "Unknown kind of provider.")
    try:
        issuer = oidc.issuer_for(kind, payload.get("tenant") or "")
        doc = await oidc.discover(issuer)
    except oidc.SsoError as e:
        raise HTTPException(400, str(e)) from e

    fields = _settings(payload, {"kind": kind})
    if not fields.get("client_id"):
        raise HTTPException(400, "The application (client) ID is required.")
    fields["name"] = fields.get("name") or oidc.PRESETS[kind]["label"]
    idp_id = db.create_idp(
        free_slug(kind), kind=kind, issuer=issuer,
        tenant=(payload.get("tenant") or "").strip() or None,
        discovery=json.dumps(doc), discovered_at=time.time(),
        created_by=admin["id"], **fields)
    if not db.get_idp(idp_id)["scopes"]:
        db.update_idp(idp_id, scopes=oidc.PRESETS[kind]["scopes"])
    row = oidc.admin_idp(db.get_idp(idp_id))
    row["redirect_uri"] = redirect_uri(request, idp_id)
    return row


@router.patch("/idp/{idp_id}")
async def modify_idp(request: Request, idp_id: str,
                     payload: dict = Body(...)) -> dict:
    require_admin(request)
    idp = db.get_idp(idp_id)
    if not idp:
        raise HTTPException(404, "No such sign-in method.")
    fields = _settings(payload, idp)
    if "tenant" in payload:
        try:
            issuer = oidc.issuer_for(idp["kind"], payload["tenant"])
            fields["issuer"] = issuer
            fields["tenant"] = (payload["tenant"] or "").strip() or None
        except oidc.SsoError as e:
            raise HTTPException(400, str(e)) from e
    db.update_idp(idp_id, **fields)
    if fields.get("issuer"):
        await _rediscover(idp_id)
    row = oidc.admin_idp(db.get_idp(idp_id))
    row["redirect_uri"] = redirect_uri(request, idp_id)
    return row


async def _rediscover(idp_id: str) -> dict:
    idp = db.get_idp(idp_id)
    doc = await oidc.discover(idp["issuer"])
    db.update_idp(idp_id, discovery=json.dumps(doc), discovered_at=time.time())
    return doc


@router.post("/idp/{idp_id}/rediscover")
async def rediscover(request: Request, idp_id: str) -> dict:
    require_admin(request)
    if not db.get_idp(idp_id):
        raise HTTPException(404, "No such sign-in method.")
    try:
        await _rediscover(idp_id)
    except oidc.SsoError as e:
        raise HTTPException(400, str(e)) from e
    return oidc.admin_idp(db.get_idp(idp_id))


@router.delete("/idp/{idp_id}")
async def remove_idp(request: Request, idp_id: str) -> dict:
    require_admin(request)
    if not db.get_idp(idp_id):
        raise HTTPException(404, "No such sign-in method.")
    detached = db.delete_idp(idp_id)
    return {"ok": True, "detached": detached,
            "note": "%d account(s) kept their runs but can no longer sign in "
                    "this way. Give them a password to let them back in."
                    % detached}


@router.post("/idp/{idp_id}/sync")
async def run_sync(request: Request, idp_id: str,
                   dry_run: bool = Query(False)) -> dict:
    require_admin(request)
    idp = db.get_idp(idp_id)
    if not idp:
        raise HTTPException(404, "No such sign-in method.")
    try:
        return await directory.sync(idp, dry_run=dry_run)
    except directory.SyncError as e:
        raise HTTPException(400, str(e)) from e
