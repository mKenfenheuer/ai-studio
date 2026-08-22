"""Who is allowed to call what.

The important decision here is that the gate is a **middleware over every
`/api/` path**, with a short list of exceptions, rather than a dependency
added to each route. Thirty-odd routes exist today and more arrive every time
a feature lands; a dependency that has to be remembered is a dependency that
will eventually be forgotten, and the failure mode of forgetting is an
endpoint that quietly serves anybody who asks.

Written this way round, a new route is private until somebody deliberately
lists it as public, and the list of public things is short enough to read.
"""
from __future__ import annotations

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from .. import auth, config, db

# Reachable without a session, and each for a stated reason.
PUBLIC_PATHS = {
    # A runner authenticates with the join token over the socket itself.
    "/api/runner/ws",
    # Liveness only, and it carries nothing. A healthcheck that needs a
    # session is a healthcheck that reports every healthy controller as sick,
    # which is exactly what happened here until this line existed.
    "/api/health",
    # The login screen has to be able to ask what it is looking at: is this a
    # fresh install that needs its first account, or a studio to log in to?
    "/api/auth/state",
    "/api/auth/login",
    "/api/auth/setup",
}

# Runners call these with the join token instead of a cookie, because a runner
# has no session and no user.
RUNNER_TOKEN_PATHS = (
    "/artifact",
    "/download",
    "/dataset-file",
    # Read by the deploy script on the host, which has the join token in the
    # same .env that starts the containers and has no browser session.
    "/in-flight",
)


def _wants_json(request: Request) -> bool:
    return request.url.path.startswith("/api/")


async def authenticate(request: Request, call_next):
    """Attach the caller's account to the request, or refuse it."""
    path = request.url.path
    request.state.user = None

    if not path.startswith("/api/"):
        # The single-page app itself is served to anyone; it shows a login
        # screen and can do nothing without a session. Serving the shell
        # publicly is what lets the login screen exist at all.
        return await call_next(request)

    if path in PUBLIC_PATHS:
        return await call_next(request)

    token = request.headers.get("X-Runner-Token")
    if token and any(path.endswith(s) for s in RUNNER_TOKEN_PATHS):
        if token == config.join_token():
            request.state.runner = True
            return await call_next(request)
        return JSONResponse({"detail": "Invalid runner token."}, status_code=403)

    user = auth.session_user(request.cookies.get(auth.SESSION_COOKIE))
    if not user:
        # 401 rather than 403: the browser app treats it as "show the login
        # screen", and the distinction is the difference between "log in" and
        # "you are logged in and still may not".
        if db.count_users() == 0:
            return JSONResponse(
                {"detail": "This studio has no accounts yet.",
                 "setup_required": True}, status_code=401)
        return JSONResponse({"detail": "Please sign in."}, status_code=401)

    # An account flagged to change its password can do exactly two things:
    # read who it is, and change its password. Anything else would let an
    # administrator's temporary password stay in use indefinitely.
    if user["must_change"] and path not in (
            "/api/auth/state", "/api/auth/logout", "/api/me/password", "/api/me"):
        return JSONResponse(
            {"detail": "Set a new password before continuing.",
             "must_change": True}, status_code=403)

    request.state.user = user
    # Everything this request does against the Hub now runs as this person:
    # their rate limits, their gated-model access, their private repositories.
    from .. import hfaccount, hub
    hub.CURRENT_TOKEN.set(hfaccount.token_for(user))
    return await call_next(request)


def current_user(request: Request) -> dict:
    user = getattr(request.state, "user", None)
    if not user:
        raise HTTPException(401, "Please sign in.")
    return user


def require_admin(request: Request) -> dict:
    user = current_user(request)
    if user["role"] != "admin":
        raise HTTPException(403, "This needs an administrator account.")
    return user


def access(request: Request, kind: str, resource: dict) -> str:
    """"edit", "view", or a 404 -- what this caller may do with this thing.

    404 rather than 403 for something they cannot see at all. A 403 confirms
    that a run with that id exists and belongs to somebody, which is a small
    leak but a real one: run ids appear in URLs people paste to each other.
    """
    user = current_user(request)
    level = db.access_level(kind, resource["id"], resource.get("owner_id"), user)
    if not level:
        raise HTTPException(404, "No such %s." % ("run" if kind == "job" else kind))
    return level


def require_view(request: Request, kind: str, resource: dict) -> dict:
    access(request, kind, resource)
    return current_user(request)


def require_edit(request: Request, kind: str, resource: dict) -> dict:
    if access(request, kind, resource) != "edit":
        raise HTTPException(
            403, "This was shared with you to look at, not to change. Ask its "
                 "owner for edit access.")
    return current_user(request)


def require_owner(request: Request, kind: str, resource: dict) -> dict:
    """Stricter than edit: only the owner (or an admin) may re-share or delete.

    Someone given edit access can run and change a thing. Passing it on to
    other people, or destroying it, stays with whoever it belongs to.
    """
    user = current_user(request)
    owner = resource.get("owner_id")
    if user["role"] == "admin" or owner in (None, user["id"]):
        return user
    raise HTTPException(
        403, "Only the owner of this can share or delete it.")
