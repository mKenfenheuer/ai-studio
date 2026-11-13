"""Signing another program in to the studio: OAuth 2 in the IndieAuth way.

What this is for: an MCP client -- Claude on a phone, a desktop app, an agent
on another machine -- that should act as you on this studio without being
handed your password or a key pasted out of the account page.

The IndieAuth idea is that a client is not registered here first. Its
`client_id` is a URL, and the studio fetches that URL to learn the client's
name and the addresses it may be sent back to. Two shapes of document are
read: the JSON "client ID metadata document" that MCP clients publish, and
the HTML page with `<link rel="redirect_uri">` of classic IndieAuth. A
`redirect_uri` on the same host as the client id is always allowed, as the
IndieAuth spec says; anything else has to be listed by the client itself.

The flow is the authorization code flow with PKCE (S256, required) and no
client secret. The token is an ordinary API key of this account with studio
access, so it shows on the account page beside the others, labelled with the
client that holds it, and deleting it there signs that client out. It lives
for a week and comes with a refresh token that lives for half a year; each
refresh replaces both.

Endpoints:
    /.well-known/oauth-authorization-server   server metadata (RFC 8414)
    /.well-known/oauth-protected-resource     what /mcp accepts (RFC 9728)
    /oauth/authorize                          sign in and say yes or no
    /oauth/token                              code or refresh token -> token
    /people/{username}                        the `me` URL of a token
"""
from __future__ import annotations

import base64
import hashlib
import html
import ipaddress
import json
import re
import secrets
import time
from urllib.parse import urlencode, urljoin, urlsplit

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from .. import auth, config, db

router = APIRouter()

ACCESS_TTL_S = 7 * 24 * 3600
REFRESH_TTL_S = 180 * 24 * 3600
SCOPE = "studio"

# The consent form in flight: nonce -> what was asked, by whom, until when.
# In memory on purpose -- one controller process, a form that lives for ten
# minutes, and a restart that forgets it costs a click on "Allow" again.
_PENDING: dict[str, dict] = {}


def base_url(request: Request) -> str:
    """This studio's own address, as the client reaches it.

    AI_STUDIO_PUBLIC_URL when it is set -- behind a reverse proxy the request
    says http://controller:8420, which is not where anybody can send a browser.
    """
    if config.PUBLIC_URL:
        return config.PUBLIC_URL.rstrip("/")
    proto = request.headers.get("x-forwarded-proto") or request.url.scheme
    host = request.headers.get("x-forwarded-host") or request.headers.get("host") \
        or request.url.netloc
    return "%s://%s" % (proto.split(",")[0].strip(), host.split(",")[0].strip())


def _hash(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


# --------------------------------------------------------------- metadata ---

def server_metadata(base: str) -> dict:
    return {
        "issuer": base,
        "authorization_endpoint": base + "/oauth/authorize",
        "token_endpoint": base + "/oauth/token",
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code", "refresh_token"],
        "code_challenge_methods_supported": ["S256"],
        "token_endpoint_auth_methods_supported": ["none"],
        "scopes_supported": [SCOPE],
        "client_id_metadata_document_supported": True,
        "authorization_response_iss_parameter_supported": True,
    }


def resource_metadata(base: str) -> dict:
    return {
        "resource": base + "/mcp",
        "authorization_servers": [base],
        "scopes_supported": [SCOPE],
        "bearer_methods_supported": ["header"],
        "resource_name": "AI Studio",
    }


@router.get("/.well-known/oauth-authorization-server")
@router.get("/.well-known/oauth-authorization-server/{rest:path}")
@router.get("/.well-known/openid-configuration")
async def well_known_server(request: Request, rest: str = "") -> dict:
    return server_metadata(base_url(request))


@router.get("/.well-known/oauth-protected-resource")
@router.get("/.well-known/oauth-protected-resource/{rest:path}")
async def well_known_resource(request: Request, rest: str = "") -> dict:
    return resource_metadata(base_url(request))


@router.get("/people/{username}", response_class=HTMLResponse)
async def profile(request: Request, username: str) -> HTMLResponse:
    """The `me` of a token: a page that names its own authorization server.

    Nothing about the account is shown beyond the name already in the URL.
    """
    base = base_url(request)
    meta = base + "/.well-known/oauth-authorization-server"
    return HTMLResponse(
        "<!doctype html><meta charset=utf-8><title>%s</title>"
        '<link rel="indieauth-metadata" href="%s">'
        '<link rel="authorization_endpoint" href="%s/oauth/authorize">'
        '<link rel="token_endpoint" href="%s/oauth/token">'
        '<div class="h-card"><a class="u-url p-name" href="%s/people/%s">%s</a>'
        " on AI Studio</div>"
        % (html.escape(username), meta, base, base, base,
           html.escape(username), html.escape(username)),
        headers={"Link": '<%s>; rel="indieauth-metadata"' % meta})


# ---------------------------------------------------------- the client id ---

def _loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def check_client_id(client_id: str) -> str:
    """A client id as IndieAuth allows it, or the reason it is not one."""
    parts = urlsplit(client_id)
    if parts.scheme not in ("https", "http") or not parts.hostname:
        return "The client_id has to be the URL of the client."
    if parts.scheme == "http" and not _loopback(parts.hostname):
        return "A client_id over plain http is only accepted for this computer."
    if parts.username or parts.password or parts.fragment:
        return "The client_id may not carry a user name, password or fragment."
    if "/./" in parts.path or "/../" in parts.path:
        return "The client_id may not contain dot segments."
    return ""


def _same_site(a: str, b: str) -> bool:
    pa, pb = urlsplit(a), urlsplit(b)
    return (pa.scheme, pa.hostname, pa.port) == (pb.scheme, pb.hostname, pb.port)


async def fetch_client(client_id: str) -> dict:
    """What the client says about itself: a name and where it may return to.

    Fetched every time a person is asked, not cached: a client that changes
    its redirect addresses has to be believed about the new ones and not about
    the old. A client whose page cannot be read still works for a redirect on
    its own host, under its URL as its name -- IndieAuth's rule.
    """
    info = {"client_id": client_id, "name": urlsplit(client_id).hostname or client_id,
            "redirect_uris": [], "logo": None, "fetched": False}
    try:
        async with httpx.AsyncClient(timeout=5, follow_redirects=True,
                                     max_redirects=3) as c:
            r = await c.get(client_id, headers={
                "Accept": "application/json, text/html;q=0.9"})
    except httpx.HTTPError:
        return info
    if r.status_code != 200:
        return info
    body = r.content[:65536].decode("utf-8", "replace")
    ctype = r.headers.get("content-type", "")
    if "json" in ctype or body.lstrip().startswith("{"):
        try:
            doc = json.loads(body)
        except ValueError:
            return info
        if not isinstance(doc, dict) or doc.get("client_id") not in (None, client_id):
            return info
        info.update(fetched=True,
                    name=str(doc.get("client_name") or info["name"])[:80],
                    logo=doc.get("logo_uri"),
                    redirect_uris=[str(u) for u in doc.get("redirect_uris") or []])
        return info
    links = re.findall(r'<link\b[^>]*>', body, re.IGNORECASE)
    for tag in links:
        if re.search(r'rel=["\']?[^"\'>]*\bredirect_uri\b', tag, re.IGNORECASE):
            if m := re.search(r'href=["\']([^"\']+)', tag, re.IGNORECASE):
                info["redirect_uris"].append(urljoin(str(r.url), html.unescape(m.group(1))))
    for part in r.headers.get_list("link"):
        if 'rel="redirect_uri"' in part or "rel=redirect_uri" in part:
            if m := re.search(r"<([^>]+)>", part):
                info["redirect_uris"].append(urljoin(str(r.url), m.group(1)))
    if m := re.search(r'class=["\'][^"\']*\bp-name\b[^"\']*["\'][^>]*>([^<]{1,80})<', body):
        info["name"] = html.unescape(m.group(1)).strip()
    elif m := re.search(r"<title>([^<]{1,80})</title>", body, re.IGNORECASE):
        info["name"] = html.unescape(m.group(1)).strip()
    info["fetched"] = True
    return info


def redirect_allowed(client: dict, redirect_uri: str) -> bool:
    """Whether the browser may be sent back to this address with a code.

    https, or http to this computer only -- the same rule as for a client id.
    Then either the client listed it, or it is on the client's own host.
    """
    parts = urlsplit(redirect_uri)
    if not parts.hostname or parts.fragment or parts.username:
        return False
    if parts.scheme != "https" and not (parts.scheme == "http"
                                        and _loopback(parts.hostname)):
        return False
    return redirect_uri in client["redirect_uris"] or \
        _same_site(client["client_id"], redirect_uri)


# --------------------------------------------------------- the authorize ---

_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sign in to AI Studio</title>
<style>
:root {{ --bg:#f6f6f4; --card:#fff; --ink:#1d1d1b; --muted:#6b6b66; --line:#deded8;
        --accent:#2f5bd3; --err:#b3261e; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#161615; --card:#1f1f1d;
        --ink:#ecece7; --muted:#a2a29b; --line:#33332f; --accent:#7b9cff; --err:#ff8a80; }} }}
body {{ margin:0; background:var(--bg); color:var(--ink);
       font:16px/1.5 system-ui, -apple-system, Segoe UI, sans-serif; }}
main {{ max-width:420px; margin:8vh auto; padding:0 16px; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:12px;
        padding:24px; }}
h1 {{ font-size:20px; margin:0 0 12px; }}
p {{ margin:0 0 12px; }} .muted {{ color:var(--muted); font-size:14px; }}
code {{ font-size:13px; word-break:break-all; }}
label {{ display:block; font-size:14px; margin:12px 0 4px; }}
input {{ width:100%; box-sizing:border-box; padding:10px; font:inherit; color:inherit;
        background:transparent; border:1px solid var(--line); border-radius:8px; }}
.row {{ display:flex; gap:8px; margin-top:20px; }}
button {{ flex:1; padding:10px; font:inherit; border-radius:8px; cursor:pointer;
         border:1px solid var(--line); background:transparent; color:inherit; }}
button.primary {{ background:var(--accent); border-color:var(--accent); color:#fff; }}
.err {{ color:var(--err); }}
</style></head><body><main><div class="card">{body}</div></main></body></html>"""


def _page(body: str, status: int = 200) -> HTMLResponse:
    return HTMLResponse(_PAGE.format(body=body), status_code=status,
                        headers={"Cache-Control": "no-store",
                                 "X-Frame-Options": "DENY",
                                 "Content-Security-Policy": "frame-ancestors 'none'"})


def _error_page(text: str) -> HTMLResponse:
    return _page("<h1>This sign-in cannot go ahead</h1><p class=err>%s</p>"
                 % html.escape(text), 400)


def _back(redirect_uri: str, params: dict) -> RedirectResponse:
    sep = "&" if "?" in redirect_uri else "?"
    return RedirectResponse(redirect_uri + sep + urlencode(params), status_code=302)


async def _ask(request: Request) -> tuple[dict | None, HTMLResponse | None]:
    """Read and check an authorization request, from the query or the form."""
    src = dict(request.query_params)
    if request.method == "POST":
        form = await request.form()
        src.update({k: str(v) for k, v in form.items()})
    want = {k: (src.get(k) or "").strip() for k in (
        "response_type", "client_id", "redirect_uri", "state", "code_challenge",
        "code_challenge_method", "scope", "me", "resource")}
    if problem := check_client_id(want["client_id"]):
        return None, _error_page(problem)
    client = await fetch_client(want["client_id"])
    if not redirect_allowed(client, want["redirect_uri"]):
        # Never redirect to an address the client has not vouched for: that
        # is the open redirect this whole check exists to prevent.
        return None, _error_page(
            "%s asked to send you back to an address it does not list as its "
            "own." % client["name"])
    if want["response_type"] != "code":
        return None, _back(want["redirect_uri"], {
            "error": "unsupported_response_type", "state": want["state"]})
    if not want["code_challenge"] or (want["code_challenge_method"] or "S256") != "S256":
        return None, _back(want["redirect_uri"], {
            "error": "invalid_request", "state": want["state"],
            "error_description": "PKCE with S256 is required."})
    want["client"] = client
    want["src"] = src
    return want, None


def _consent(request: Request, want: dict, user: dict) -> HTMLResponse:
    nonce = secrets.token_urlsafe(24)
    for k in [k for k, v in _PENDING.items() if v["until"] < time.time()]:
        _PENDING.pop(k, None)
    _PENDING[nonce] = {"user_id": user["id"], "until": time.time() + 600,
                       **{k: want[k] for k in want if k not in ("client", "src")},
                       "client_name": want["client"]["name"]}
    c = want["client"]
    host = urlsplit(want["redirect_uri"]).hostname or ""
    return _page(
        "<h1>Allow %s to use AI Studio?</h1>"
        "<p>It will act as <strong>%s</strong>: read your projects, datasets "
        "and runs, upload data and start or stop training — everything your "
        "account can do here.</p>"
        "<p class=muted>Client: <code>%s</code><br>Returns to: <code>%s</code>%s</p>"
        "<p class=muted>The sign-in shows on your account page as an API key "
        "and ends when you delete it there.</p>"
        '<form method="post" action="/oauth/authorize">'
        '<input type="hidden" name="nonce" value="%s">'
        '<div class="row"><button name="decision" value="deny">Cancel</button>'
        '<button class="primary" name="decision" value="allow">Allow</button></div>'
        "</form>" % (
            html.escape(c["name"]), html.escape(user.get("display_name")
                                                or user["username"]),
            html.escape(c["client_id"]), html.escape(host),
            "" if c["fetched"] else "<br>Its page could not be read; only its "
                                    "own host is accepted as a return address.",
            nonce))


def _login(want: dict, error: str = "") -> HTMLResponse:
    hidden = "".join('<input type="hidden" name="%s" value="%s">'
                     % (html.escape(k), html.escape(v)) for k, v in want["src"].items()
                     if k in ("response_type", "client_id", "redirect_uri", "state",
                              "code_challenge", "code_challenge_method", "scope",
                              "me", "resource"))
    return _page(
        "<h1>Sign in to AI Studio</h1>"
        "<p><strong>%s</strong> wants to use the studio as you.</p>%s"
        '<form method="post" action="/oauth/authorize">%s'
        '<input type="hidden" name="decision" value="login">'
        '<label for=u>Username</label><input id=u name="username" autocomplete="username" required>'
        '<label for=p>Password</label><input id=p name="password" type="password" '
        'autocomplete="current-password" required>'
        '<div class="row"><button class="primary">Sign in</button></div></form>'
        % (html.escape(want["client"]["name"]),
           "<p class=err>%s</p>" % html.escape(error) if error else "", hidden))


def _session_user(request: Request) -> dict | None:
    user = auth.session_user(request.cookies.get(auth.SESSION_COOKIE))
    if user and (user.get("must_change") or user.get("role") == "chat"):
        return None
    return user


@router.get("/oauth/authorize")
async def authorize(request: Request):
    want, refusal = await _ask(request)
    if refusal:
        return refusal
    user = _session_user(request)
    return _consent(request, want, user) if user else _login(want)


@router.post("/oauth/authorize")
async def authorize_post(request: Request):
    form = {k: str(v) for k, v in (await request.form()).items()}
    decision = form.get("decision")
    if decision in ("allow", "deny"):
        pending = _PENDING.pop(form.get("nonce") or "", None)
        user = _session_user(request)
        if not pending or pending["until"] < time.time() or not user \
                or user["id"] != pending["user_id"]:
            return _error_page("This page has expired. Start the sign-in again "
                               "from the app.")
        if decision == "deny":
            return _back(pending["redirect_uri"], {
                "error": "access_denied", "state": pending["state"],
                "iss": base_url(request)})
        code = secrets.token_urlsafe(32)
        db.save_oauth_code(_hash(code), user["id"], pending["client_id"],
                           pending["client_name"], pending["redirect_uri"],
                           pending["code_challenge"], SCOPE, pending["resource"])
        return _back(pending["redirect_uri"], {
            "code": code, "state": pending["state"], "iss": base_url(request)})

    # Signing in on the way. The same checks as the studio's own login form,
    # including the lockout after repeated failures.
    want, refusal = await _ask(request)
    if refusal:
        return refusal
    username = (form.get("username") or "").strip().lower()
    client_ip = request.client.host if request.client else "?"
    key = "%s|%s" % (username, client_ip)
    if wait := auth.locked_out(key):
        return _login(want, "Too many failed attempts. Try again in %d seconds." % wait)
    user = db.get_user_by_name(username)
    stored = user["password_hash"] if user and user.get("password_hash") \
        else auth.hash_password("x" * 24)
    if not user or not auth.verify_password(form.get("password") or "", stored) \
            or not user["active"]:
        auth.note_failure(key)
        return _login(want, "That username and password do not match.")
    auth.clear_failures(key)
    db.update_user(user["id"], last_login=time.time())
    raw = auth.start_session(user["id"], request.headers.get("user-agent", ""))
    user = db.get_user(user["id"])
    if user.get("must_change"):
        resp = _error_page("Set a new password in the studio first, then sign in again.")
    else:
        resp = _consent(request, want, user)
    resp.set_cookie(auth.SESSION_COOKIE, raw, max_age=auth.SESSION_TTL_S,
                    httponly=True, samesite="lax",
                    secure=request.url.scheme == "https", path="/")
    return resp


# ----------------------------------------------------------------- token ---

def _token_error(code: str, text: str, status: int = 400) -> JSONResponse:
    return JSONResponse({"error": code, "error_description": text},
                        status_code=status, headers={"Cache-Control": "no-store"})


def _issue(request: Request, user: dict, client_id: str, client_name: str) -> JSONResponse:
    raw, key_hash, prefix = auth.new_api_key()
    refresh = "rt-ais-" + secrets.token_urlsafe(32)
    db.create_api_key(user["id"], ("MCP: %s" % client_name)[:60], key_hash, prefix,
                      expires_at=time.time() + ACCESS_TTL_S, access="studio",
                      client_id=client_id, refresh_hash=_hash(refresh))
    base = base_url(request)
    return JSONResponse({
        "access_token": raw, "token_type": "Bearer", "expires_in": ACCESS_TTL_S,
        "refresh_token": refresh, "scope": SCOPE,
        "me": "%s/people/%s" % (base, user["username"]),
    }, headers={"Cache-Control": "no-store", "Pragma": "no-cache"})


@router.post("/oauth/token")
async def token(request: Request):
    if "json" in request.headers.get("content-type", ""):
        try:
            form = {k: str(v) for k, v in (await request.json()).items()}
        except (ValueError, AttributeError):
            return _token_error("invalid_request", "The body is not JSON.")
    else:
        form = {k: str(v) for k, v in (await request.form()).items()}
    grant = form.get("grant_type")
    client_id = (form.get("client_id") or "").strip()

    if grant == "authorization_code":
        row = db.take_oauth_code(_hash(form.get("code") or ""))
        if not row:
            return _token_error("invalid_grant", "The code is unknown, used or expired.")
        if client_id and client_id != row["client_id"]:
            return _token_error("invalid_grant", "The code was issued to another client.")
        if (form.get("redirect_uri") or row["redirect_uri"]) != row["redirect_uri"]:
            return _token_error("invalid_grant", "redirect_uri does not match.")
        verifier = form.get("code_verifier") or ""
        challenge = base64.urlsafe_b64encode(
            hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
        if not verifier or not secrets.compare_digest(challenge, row["code_challenge"]):
            return _token_error("invalid_grant", "The PKCE code_verifier does not match.")
        user = db.get_user(row["user_id"])
        if not user or not user["active"]:
            return _token_error("invalid_grant", "That account is no longer active.")
        return _issue(request, user, row["client_id"], row.get("client_name")
                      or row["client_id"])

    if grant == "refresh_token":
        old = db.api_key_by_refresh(_hash(form.get("refresh_token") or ""))
        if not old or time.time() - float(old["created_at"]) > REFRESH_TTL_S:
            return _token_error("invalid_grant", "The refresh token is unknown or expired.")
        if client_id and client_id != old.get("client_id"):
            return _token_error("invalid_grant", "The refresh token belongs to another client.")
        user = db.get_user(old["user_id"])
        if not user or not user["active"]:
            return _token_error("invalid_grant", "That account is no longer active.")
        # Rotated: the old pair stops working the moment the new one exists.
        db.delete_api_key_by_id(old["id"])
        return _issue(request, user, old.get("client_id") or "",
                      (old.get("name") or "").removeprefix("MCP: "))

    return _token_error("unsupported_grant_type",
                        "Use authorization_code or refresh_token.")


def challenge_header(request: Request) -> str:
    """What a 401 from /mcp says, so a client knows where to sign in."""
    return 'Bearer resource_metadata="%s/.well-known/oauth-protected-resource/mcp"' \
        % base_url(request)


def require_token(request: Request) -> dict:
    raw = ""
    header = request.headers.get("authorization") or ""
    if header.lower().startswith("bearer "):
        raw = header[7:].strip()
    user = db.studio_key_identity(auth.api_key_hash(raw))[0] if raw else None
    if not user or user.get("must_change") or user.get("role") == "chat":
        raise HTTPException(401, "Sign in first.",
                            headers={"WWW-Authenticate": challenge_header(request)})
    return user
