"""Signing in, your own account, and administering other people's."""
from __future__ import annotations

import json
import time

from fastapi import APIRouter, Body, HTTPException, Request, Response

from .. import auth, db, hfaccount, notify
from .security import current_user, require_admin

router = APIRouter(prefix="/api")


def _set_cookie(response: Response, request: Request, raw: str) -> None:
    response.set_cookie(
        auth.SESSION_COOKIE, raw,
        max_age=auth.SESSION_TTL_S,
        httponly=True,            # script on the page can never read it
        samesite="lax",           # a form on another site cannot post with it
        # Only over TLS when there is TLS. Setting this unconditionally would
        # silently break every plain-HTTP install on a home network, which is
        # most of them, and the failure looks like "login does nothing".
        secure=request.url.scheme == "https",
        path="/")


def _email_or_none(value, exclude_id: str | None = None) -> str | None:
    """A usable address, or a refusal saying why.

    Checked for uniqueness rather than merely stored. An address is what a
    single sign-on provider matches an existing account by, so two accounts
    sharing one is not an untidiness -- it is an ambiguity in the middle of
    deciding who somebody is.
    """
    email = (value or "").strip().lower()
    if not email:
        return None
    if "@" not in email or email.startswith("@") or email.endswith("@"):
        raise HTTPException(400, "That does not look like an email address.")
    clash = db.get_user_by_email(email)
    if clash and clash["id"] != exclude_id:
        raise HTTPException(409, "Another account here already uses %s." % email)
    return email[:200]


@router.get("/auth/state")
async def auth_state(request: Request) -> dict:
    """What the login screen needs to know before anyone types anything."""
    user = getattr(request.state, "user", None)
    if user is None:
        user = auth.session_user(request.cookies.get(auth.SESSION_COOKIE))
    return {
        "setup_required": db.count_users() == 0,
        "authenticated": bool(user),
        "user": db.public_user(user),
        "must_change": bool(user and user["must_change"]),
    }


@router.post("/auth/setup")
async def first_account(request: Request, response: Response,
                        payload: dict = Body(...)) -> dict:
    """Create the first administrator, once, on a studio that has none.

    Open without authentication because there is nothing yet to authenticate
    against, and closed forever the moment one account exists. That window is
    the same one every self-hosted install has; the honest mitigation is to
    make it as short as possible and to say so on the screen.
    """
    if db.count_users():
        raise HTTPException(409, "This studio already has an account. Sign in.")

    username = (payload.get("username") or "").strip().lower()
    password = payload.get("password") or ""
    if problem := auth.username_problem(username):
        raise HTTPException(400, problem)
    if problem := auth.password_problem(password):
        raise HTTPException(400, problem)

    uid = db.create_user(username, (payload.get("display_name") or "").strip()
                         or username, auth.hash_password(password), "admin")
    adopted = db.adopt_ownerless(uid)
    db.update_user(uid, last_login=time.time())
    _set_cookie(response, request, auth.start_session(
        uid, request.headers.get("user-agent", "")))
    return {"ok": True, "user": db.public_user(db.get_user(uid)),
            "adopted_runs": adopted}


@router.post("/auth/login")
async def login(request: Request, response: Response,
                payload: dict = Body(...)) -> dict:
    username = (payload.get("username") or "").strip().lower()
    password = payload.get("password") or ""
    client = request.client.host if request.client else "?"
    key = "%s|%s" % (username, client)

    if wait := auth.locked_out(key):
        raise HTTPException(
            429, "Too many failed attempts. Try again in %d seconds." % wait)

    user = db.get_user_by_name(username)
    # Verified even when the user does not exist, against a throwaway hash, so
    # that a missing account and a wrong password take the same time. Without
    # it, response timing lists who has an account here.
    stored = user["password_hash"] if user else auth.hash_password("x" * 24)
    ok = auth.verify_password(password, stored)

    if user and not user["password_hash"] and user.get("provider"):
        # No hash and a provider means this account signs in elsewhere. Telling
        # them so is not a leak worth worrying about -- the sign-in button is
        # on the same screen, in front of them.
        idp = db.get_idp(user["provider"])
        raise HTTPException(
            400, "This account signs in with %s. Use the button above."
                 % ((idp or {}).get("name") or "single sign-on"))

    if not user or not ok or not user["active"]:
        auth.note_failure(key)
        if user and ok and not user["active"]:
            raise HTTPException(403, "This account has been disabled.")
        raise HTTPException(401, "That username and password do not match.")

    auth.clear_failures(key)
    db.update_user(user["id"], last_login=time.time())
    _set_cookie(response, request, auth.start_session(
        user["id"], request.headers.get("user-agent", "")))
    auth.purge_expired_sessions()
    return {"ok": True, "user": db.public_user(db.get_user(user["id"]))}


@router.post("/auth/logout")
async def logout(request: Request, response: Response) -> dict:
    auth.end_session(request.cookies.get(auth.SESSION_COOKIE))
    response.delete_cookie(auth.SESSION_COOKIE, path="/")
    return {"ok": True}


# ---------------------------------------------------------------------------
# Your own account
# ---------------------------------------------------------------------------

@router.get("/me")
async def me(request: Request) -> dict:
    user = current_user(request)
    out = db.public_user(user)
    out["sessions"] = db.list_sessions(user["id"])
    return out


@router.patch("/me")
async def update_me(request: Request, payload: dict = Body(...)) -> dict:
    user = current_user(request)
    fields: dict = {}
    if "display_name" in payload:
        name = (payload.get("display_name") or "").strip()
        if not name:
            raise HTTPException(400, "A display name cannot be empty.")
        fields["display_name"] = name[:80]
    if "email" in payload:
        # Not for signing in -- for being found. A colleague looking for you
        # in the share box is far more likely to type your address than your
        # username, and an account with neither is one nobody can share with.
        if user.get("provider"):
            raise HTTPException(
                400, "Your address comes from the account you sign in with, "
                     "and changing it here would only make the two disagree.")
        fields["email"] = _email_or_none(payload["email"], exclude_id=user["id"])
    if not fields:
        raise HTTPException(400, "Nothing to change.")
    db.update_user(user["id"], **fields)
    return db.public_user(db.get_user(user["id"]))


@router.post("/me/password")
async def change_password(request: Request, response: Response,
                          payload: dict = Body(...)) -> dict:
    user = current_user(request)
    current = payload.get("current_password") or ""
    new = payload.get("new_password") or ""
    if user.get("provider") and not user["password_hash"]:
        idp = db.get_idp(user["provider"])
        raise HTTPException(
            400, "This account signs in with %s, so there is no password here "
                 "to change. Change it where that account lives."
                 % ((idp or {}).get("name") or "single sign-on"))

    # Skipped only for an account an administrator has just reset, which by
    # definition has no password its owner knows.
    if not user["must_change"] and not auth.verify_password(current, user["password_hash"]):
        raise HTTPException(403, "That is not your current password.")
    if problem := auth.password_problem(new):
        raise HTTPException(400, problem)
    if auth.verify_password(new, user["password_hash"]):
        raise HTTPException(400, "That is the password you already have.")

    db.update_user(user["id"], password_hash=auth.hash_password(new),
                   must_change=0)
    # Every other session is ended: changing a password is what someone does
    # when they think somebody else has it, and leaving the other sessions
    # alive would make the act pointless.
    keep = request.cookies.get(auth.SESSION_COOKIE)
    ended = auth.end_all_sessions(user["id"], keep=keep)
    return {"ok": True, "other_sessions_ended": ended}


@router.post("/me/sessions/revoke")
async def revoke_sessions(request: Request) -> dict:
    user = current_user(request)
    keep = request.cookies.get(auth.SESSION_COOKIE)
    return {"ok": True, "ended": auth.end_all_sessions(user["id"], keep=keep)}


# ---------------------------------------------------------------------------
# Hugging Face
# ---------------------------------------------------------------------------

@router.get("/me/api-keys")
async def api_keys(request: Request) -> list[dict]:
    return db.list_api_keys(current_user(request)["id"])


@router.post("/me/api-keys")
async def api_key_create(request: Request, payload: dict = Body(default=None)) -> dict:
    """Mint a key. Shown once, here, and never again.

    Stored as a SHA-256, so there is no version of this endpoint that could
    show it a second time -- which is the property that makes a leaked
    database not a set of working credentials.
    """
    user = current_user(request)
    name = ((payload or {}).get("name") or "").strip() or "API key"
    if len(db.list_api_keys(user["id"])) >= 20:
        raise HTTPException(400, "That is twenty keys. Delete one you no "
                                 "longer use before making another.")
    raw, key_hash, prefix = auth.new_api_key()
    kid = db.create_api_key(user["id"], name[:60], key_hash, prefix)
    return {"id": kid, "name": name[:60], "prefix": prefix, "key": raw,
            "note": "Copy this now. It is stored hashed and cannot be shown "
                    "again -- if you lose it, delete it and make another."}


@router.delete("/me/api-keys/{key_id}")
async def api_key_delete(request: Request, key_id: str) -> dict:
    if not db.delete_api_key(current_user(request)["id"], key_id):
        raise HTTPException(404, "No such key.")
    return {"ok": True}


@router.get("/me/notifications")
async def notify_state(request: Request) -> dict:
    return notify.settings_for(current_user(request))


@router.post("/me/notifications")
async def notify_set(request: Request, payload: dict = Body(...)) -> dict:
    """Set the webhook, the events, or both."""
    user = current_user(request)
    fields: dict = {}
    if "url" in payload:
        url = (payload.get("url") or "").strip()
        if url and not url.startswith(("http://", "https://")):
            raise HTTPException(400, "That does not look like a web address. "
                                     "It should start with http:// or https://.")
        fields["notify_url_enc"] = auth.encrypt_secret(url) if url else None
    if "events" in payload:
        wanted = [e for e in (payload.get("events") or []) if e in notify.EVENTS]
        fields["notify_events"] = json.dumps(wanted)
    if fields:
        db.update_user(user["id"], **fields)
    return notify.settings_for(db.get_user(user["id"]))


@router.post("/me/notifications/test")
async def notify_test(request: Request) -> dict:
    """Prove the webhook works now, rather than at 3am when a run ends."""
    return await notify.send_test(db.get_user(current_user(request)["id"]))


@router.delete("/me/notifications")
async def notify_clear(request: Request) -> dict:
    db.update_user(current_user(request)["id"], notify_url_enc=None)
    return notify.settings_for(db.get_user(current_user(request)["id"]))


@router.get("/me/huggingface")
async def hf_state(request: Request) -> dict:
    return db.public_user(current_user(request))["hf"]


@router.post("/me/huggingface")
async def hf_connect(request: Request, payload: dict = Body(...)) -> dict:
    user = current_user(request)
    try:
        await hfaccount.connect(user, payload.get("token") or "")
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return db.public_user(db.get_user(user["id"]))["hf"]


@router.delete("/me/huggingface")
async def hf_disconnect(request: Request) -> dict:
    user = current_user(request)
    hfaccount.disconnect(user)
    return {"ok": True}


@router.get("/me/huggingface/repos")
async def hf_repos(request: Request, kind: str = "models",
                   author: str | None = None) -> list[dict]:
    user = current_user(request)
    try:
        return await hfaccount.my_repos(user, kind, author)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@router.delete("/me/huggingface/repos")
async def hf_delete_repo(request: Request, repo_id: str, kind: str = "models") -> dict:
    user = current_user(request)
    try:
        await hfaccount.delete_repo(user, repo_id, kind)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return {"ok": True}


# ---------------------------------------------------------------------------
# Other people's accounts
# ---------------------------------------------------------------------------

@router.get("/users")
async def list_users(request: Request, q: str = "", limit: int = 200,
                     pending: bool = False) -> dict:
    """Accounts on this studio, filtered and capped.

    Not admin-only: sharing a run with a colleague requires knowing that the
    colleague exists. Members see names and roles; nothing else about an
    account is exposed, and the Hugging Face block is stripped below.

    Capped, and paged by search rather than by page number, because a studio
    connected to a company directory has thousands of accounts and an
    administration screen that tries to draw all of them is one that never
    finishes. People imported from a directory who have never signed in are
    left out unless asked for.
    """
    user = current_user(request)
    rows, total = db.list_users_page(q, max(1, min(limit, 500)), pending)
    out = []
    for u in rows:
        pub = db.public_user(u)
        if user["role"] != "admin":
            pub = {k: pub[k] for k in ("id", "username", "display_name", "role",
                                       "active", "pending")}
        out.append(pub)
    return {"users": out, "total": total, "shown": len(out),
            "pending_total": db.count_pending()}


@router.get("/users/search")
async def find_users(request: Request, q: str = "", limit: int = 12,
                     exclude: str = "") -> list[dict]:
    """People matching what has been typed, for the share box.

    Separate from `/api/users` rather than a parameter on it, because the two
    answer different questions. That one lists everybody, which is what an
    administrator wants and what a directory of four thousand people makes
    useless. This one is a lookup, capped, ranked, and safe to call on every
    keystroke.

    What comes back is deliberately thin -- a name, a handle, an address and
    whether they have ever been here. Sharing needs exactly that and a search
    box should not be a way to read the staff list.
    """
    current_user(request)
    rows = db.search_users(q, limit=max(1, min(limit, 50)),
                           exclude=[x for x in exclude.split(",") if x])
    return [{"id": u["id"], "username": u["username"],
             "display_name": u["display_name"], "email": u.get("email"),
             "avatar_url": u.get("avatar_url"),
             "job_title": u.get("job_title"),
             "department": u.get("department"),
             "role": u["role"], "pending": bool(u.get("pending"))}
            for u in rows]


@router.post("/users")
async def create_user(request: Request, payload: dict = Body(...)) -> dict:
    require_admin(request)
    username = (payload.get("username") or "").strip().lower()
    password = payload.get("password") or ""
    role = payload.get("role") or "member"
    if role not in ("admin", "member"):
        raise HTTPException(400, "A role is either admin or member.")
    if problem := auth.username_problem(username):
        raise HTTPException(400, problem)
    if db.get_user_by_name(username):
        raise HTTPException(409, "That username is taken.")
    if problem := auth.password_problem(password):
        raise HTTPException(400, problem)

    email = _email_or_none(payload.get("email"))
    uid = db.create_user(username,
                         (payload.get("display_name") or "").strip() or username,
                         auth.hash_password(password), role,
                         must_change=bool(payload.get("must_change", True)))
    if email:
        db.update_user(uid, email=email)
    return db.public_user(db.get_user(uid))


@router.patch("/users/{user_id}")
async def modify_user(request: Request, user_id: str,
                      payload: dict = Body(...)) -> dict:
    admin = require_admin(request)
    target = db.get_user(user_id)
    if not target:
        raise HTTPException(404, "No such account.")

    fields: dict = {}
    if "display_name" in payload:
        fields["display_name"] = (payload["display_name"] or "").strip()[:80] \
            or target["display_name"]
    if "role" in payload:
        if payload["role"] not in ("admin", "member"):
            raise HTTPException(400, "A role is either admin or member.")
        fields["role"] = payload["role"]
    if "active" in payload:
        fields["active"] = 1 if payload["active"] else 0
    if "email" in payload:
        fields["email"] = _email_or_none(payload["email"], exclude_id=user_id)

    # The two ways to lock everybody out of their own studio, refused rather
    # than explained afterwards.
    demoting = fields.get("role") == "member" and target["role"] == "admin"
    disabling = fields.get("active") == 0 and target["active"]
    if (demoting or disabling) and target["role"] == "admin" \
            and db.count_admins() <= 1:
        raise HTTPException(
            400, "This is the only administrator left. Promote somebody else "
                 "first, or nobody will be able to manage the studio.")
    if disabling and target["id"] == admin["id"]:
        raise HTTPException(400, "You cannot disable your own account.")

    db.update_user(user_id, **fields)
    if fields.get("active") == 0:
        auth.end_all_sessions(user_id)
    return db.public_user(db.get_user(user_id))


@router.post("/users/{user_id}/password")
async def reset_password(request: Request, user_id: str,
                         payload: dict = Body(...)) -> dict:
    """Set a temporary password the account must replace at next sign-in."""
    require_admin(request)
    if not db.get_user(user_id):
        raise HTTPException(404, "No such account.")
    password = payload.get("password") or ""
    if problem := auth.password_problem(password):
        raise HTTPException(400, problem)
    db.update_user(user_id, password_hash=auth.hash_password(password),
                   must_change=1)
    auth.end_all_sessions(user_id)
    # And the API keys. An administrator resetting somebody's password means
    # either they were locked out or the account is suspect, and in the second
    # case a key left working is a session that survived the reset -- the very
    # thing ending the sessions was for.
    revoked = db.revoke_api_keys(user_id)
    return {"ok": True, "keys_revoked": revoked}


@router.delete("/users/{user_id}")
async def remove_user(request: Request, user_id: str) -> dict:
    admin = require_admin(request)
    target = db.get_user(user_id)
    if not target:
        raise HTTPException(404, "No such account.")
    if target["id"] == admin["id"]:
        raise HTTPException(400, "You cannot delete your own account.")
    if target["role"] == "admin" and db.count_admins() <= 1:
        raise HTTPException(400, "This is the only administrator left.")
    db.delete_user(user_id)
    return {"ok": True, "note": "Their runs and datasets were kept and are now "
                                "unowned."}
