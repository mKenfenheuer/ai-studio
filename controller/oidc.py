"""Signing in with an account that lives somewhere else.

The studio has had exactly one way in: a username and a password it stores
itself. That is fine for one person on a home network and wrong for a team --
it means a colleague who leaves still has an account here, that nobody's
password policy applies, and that there is a second list of people to keep in
step with the real one.

So this adds OpenID Connect: Entra ID, Google, Okta, Keycloak, Authentik, or
anything else that publishes a discovery document. There is one implementation
and the named providers are *presets over it* -- a label, an issuer URL and a
default set of scopes -- rather than five code paths that drift apart. What
Microsoft and a self-hosted Keycloak actually do here is identical.

The flow is authorization code with PKCE, which is the current recommendation
even for a server-side application that can keep a secret, because it also
closes the case where the code is intercepted before the exchange. State,
nonce and the PKCE verifier all live in the database for the ten minutes a
sign-in takes, never in the browser: a callback whose state has no row is a
callback nobody here started.

Two decisions worth stating plainly, because both are places where an
identity integration can quietly become a way in for strangers:

* **Who is allowed.** A provider like Google will happily authenticate the
  entire internet. `allowed_domains` is therefore checked against the verified
  address before anything is created, and `auto_create` can be turned off
  entirely so that only people already known here may sign in.

* **Matching an existing account by address.** Convenient, and the standard
  way an SSO rollout avoids duplicating everybody -- but it means whoever can
  make the provider assert an address can take over the matching account. It
  is done only when the provider says the address is verified, only when the
  administrator left it on, and never onto an account that already belongs to
  a different provider.
"""
from __future__ import annotations

import base64
import hashlib
import json
import secrets
import time
from urllib.parse import urlencode
from typing import Any

import httpx

from . import auth, db, idtoken

# A preset is a label and the shape of an issuer URL. Nothing else about a
# provider is special-cased anywhere in this file.
PRESETS: dict[str, dict[str, Any]] = {
    "entra": {
        "label": "Microsoft",
        "blurb": "Entra ID (Azure AD), Microsoft 365 work or school accounts.",
        "issuer_template": "https://login.microsoftonline.com/{tenant}/v2.0",
        "tenant_label": "Directory (tenant) ID",
        "tenant_hint": "The GUID from Entra admin centre → Overview. Not "
                       "'common': that means every Microsoft tenant in the "
                       "world, and it is refused.",
        "scopes": "openid profile email",
        "directory": "Microsoft Graph",
        "docs": "https://entra.microsoft.com → App registrations → New",
    },
    "google": {
        "label": "Google",
        "blurb": "Google Workspace, or any Google account.",
        "issuer_template": "https://accounts.google.com",
        "tenant_label": "Workspace domain",
        "tenant_hint": "Used for directory sync. Restrict who may sign in "
                       "with the allowed domains field.",
        "scopes": "openid profile email",
        "directory": "Google Workspace Admin SDK",
        "docs": "https://console.cloud.google.com → APIs & Services → "
                "Credentials → OAuth client ID",
    },
    "okta": {
        "label": "Okta",
        "blurb": "An Okta org, or Auth0.",
        "issuer_template": "{tenant}",
        "tenant_label": "Issuer URL",
        "tenant_hint": "For example https://example.okta.com/oauth2/default",
        "scopes": "openid profile email groups",
        "directory": None,
        "docs": "Okta admin → Applications → Create → OIDC Web Application",
    },
    "oidc": {
        "label": "Single sign-on",
        "blurb": "Keycloak, Authentik, Zitadel, Gitea, or anything else that "
                 "publishes an OpenID Connect discovery document.",
        "issuer_template": "{tenant}",
        "tenant_label": "Issuer URL",
        "tenant_hint": "The address whose /.well-known/openid-configuration "
                       "describes the provider.",
        "scopes": "openid profile email",
        "directory": None,
        "docs": "",
    },
}

# Entra publishes one set of signing keys for every tenant, so a token from a
# stranger's tenant verifies perfectly well against them. What separates "our
# company" from "anybody with a Microsoft account" is the tenant in the issuer
# -- which is why these three values, all of which mean "any tenant", are
# refused rather than accepted with a warning nobody reads.
_ANY_TENANT = {"common", "organizations", "consumers"}


class SsoError(Exception):
    """Something a person needs to read, on the sign-in screen."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def issuer_for(kind: str, tenant: str) -> str:
    preset = PRESETS.get(kind) or PRESETS["oidc"]
    tenant = (tenant or "").strip().rstrip("/")
    if kind == "entra" and tenant.lower() in _ANY_TENANT:
        raise SsoError(
            "'%s' means every Microsoft tenant in the world, and this studio "
            "cannot tell one of them from another once it does. Use your own "
            "directory (tenant) ID." % tenant)
    issuer = preset["issuer_template"].format(tenant=tenant)
    if "{tenant}" in preset["issuer_template"] and not tenant:
        raise SsoError("%s is required." % preset["tenant_label"])
    if not issuer.startswith("https://") and not _loopback(issuer):
        # http is refused everywhere except a loopback address. Over plain
        # http the authorization code and the ID token are both readable in
        # transit, which is the whole of the sign-in. On localhost there is no
        # transit, and a provider running on the same box -- a test one, or a
        # Keycloak somebody is still setting up -- is a real case.
        raise SsoError("The issuer has to be an https:// address. Plain http "
                       "is only accepted on localhost.")
    return issuer


def _loopback(url: str) -> bool:
    from urllib.parse import urlsplit
    host = urlsplit(url).hostname or ""
    return host in ("localhost", "127.0.0.1", "::1")


async def discover(issuer: str) -> dict:
    """The provider's own description of itself."""
    url = issuer.rstrip("/") + "/.well-known/openid-configuration"
    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as c:
            r = await c.get(url)
            r.raise_for_status()
            doc = r.json()
    except httpx.HTTPError as e:
        raise SsoError("Could not read %s -- %s" % (url, e)) from e
    for required in ("authorization_endpoint", "token_endpoint", "jwks_uri"):
        if not doc.get(required):
            raise SsoError("That address answered, but it is not an OpenID "
                           "Connect provider: no %s." % required)
    return doc


def config_of(idp: dict) -> dict:
    """The cached discovery document, as a dict."""
    return json.loads(idp.get("discovery") or "{}")


def public_idp(idp: dict) -> dict:
    """What the sign-in screen may know: a name and a button, nothing else."""
    preset = PRESETS.get(idp["kind"]) or PRESETS["oidc"]
    return {"id": idp["id"], "name": idp["name"] or preset["label"],
            "kind": idp["kind"]}


def admin_idp(idp: dict) -> dict:
    """Everything except the secrets, which no endpoint ever returns."""
    doc = config_of(idp)
    out = {k: idp[k] for k in
           ("id", "name", "kind", "issuer", "client_id", "tenant",
            "allowed_domains", "admin_groups", "scopes", "sync_group",
            "sync_subject",
            "last_sync_at", "last_sync_note", "discovered_at")}
    out.update({
        "enabled": bool(idp["enabled"]),
        "auto_create": bool(idp["auto_create"]),
        "link_by_email": bool(idp["link_by_email"]),
        "sync_enabled": bool(idp["sync_enabled"]),
        "has_secret": bool(idp["client_secret_enc"]),
        "has_sync_secret": bool(idp["sync_secret_enc"]),
        "endpoints": {k: doc.get(k) for k in
                      ("authorization_endpoint", "token_endpoint",
                       "userinfo_endpoint", "jwks_uri", "end_session_endpoint")},
        "people": len(db.directory_users(idp["id"])),
        "pending": db.count_pending(idp["id"]),
        "directory_kind": (PRESETS.get(idp["kind"]) or {}).get("directory"),
    })
    return out


# ---------------------------------------------------------------------------
# The sign-in itself
# ---------------------------------------------------------------------------

def _pkce() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(64)[:96]
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode("ascii")).digest()).decode().rstrip("=")
    return verifier, challenge


def begin(idp: dict, redirect_uri: str, next_url: str | None) -> str:
    """Record a pending sign-in and return where to send the browser."""
    doc = config_of(idp)
    if not doc.get("authorization_endpoint"):
        raise SsoError("This provider has not been set up completely: its "
                       "discovery document was never read.")
    state = secrets.token_urlsafe(24)
    nonce = secrets.token_urlsafe(24)
    verifier, challenge = _pkce()
    db.put_oauth_state(state, idp["id"], nonce, verifier, redirect_uri, next_url)

    params = {
        "response_type": "code",
        "client_id": idp["client_id"],
        "redirect_uri": redirect_uri,
        "scope": idp["scopes"] or "openid profile email",
        "state": state,
        "nonce": nonce,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    }
    if idp["kind"] == "google":
        # Without this Google returns no email on a repeat sign-in for some
        # account types, and the studio then cannot tell who arrived.
        params["prompt"] = "select_account"
    return doc["authorization_endpoint"] + (
        "&" if "?" in doc["authorization_endpoint"] else "?") + urlencode(params)


async def _exchange(idp: dict, code: str, redirect_uri: str,
                    verifier: str) -> dict:
    doc = config_of(idp)
    data = {"grant_type": "authorization_code", "code": code,
            "redirect_uri": redirect_uri, "client_id": idp["client_id"],
            "code_verifier": verifier}
    secret = auth.decrypt_secret(idp.get("client_secret_enc"))
    if secret:
        data["client_secret"] = secret
    try:
        async with httpx.AsyncClient(timeout=20.0) as c:
            r = await c.post(doc["token_endpoint"], data=data,
                             headers={"Accept": "application/json"})
    except httpx.HTTPError as e:
        raise SsoError("Could not reach the provider to finish signing in "
                       "(%s)." % e) from e
    if r.status_code >= 400:
        body = r.json() if r.headers.get("content-type", "").startswith(
            "application/json") else {}
        # The provider's own words, which are usually the actionable ones:
        # a wrong redirect URI or an expired secret both say so here.
        raise SsoError("The provider refused the sign-in: %s"
                       % (body.get("error_description") or body.get("error")
                          or r.text[:200]))
    return r.json()


async def _userinfo(idp: dict, access_token: str) -> dict:
    doc = config_of(idp)
    url = doc.get("userinfo_endpoint")
    if not url or not access_token:
        return {}
    try:
        async with httpx.AsyncClient(timeout=15.0) as c:
            r = await c.get(url, headers={"Authorization": "Bearer " + access_token})
            if r.status_code >= 400:
                return {}
            return r.json()
    except (httpx.HTTPError, ValueError):
        # Optional by design: everything needed is normally in the ID token,
        # and a provider whose userinfo is down should still let people in.
        return {}


def _claim_groups(claims: dict) -> list[str]:
    for key in ("groups", "roles", "wids"):
        value = claims.get(key)
        if isinstance(value, list):
            return [str(v) for v in value]
        if isinstance(value, str) and value:
            return [v.strip() for v in value.split(",") if v.strip()]
    return []


async def complete(state: str, code: str) -> tuple[dict, str | None]:
    """Finish a sign-in. Returns the account and where to send them next."""
    pending = db.take_oauth_state(state)
    if not pending:
        raise SsoError("That sign-in has expired or was already used. Start "
                       "again from the sign-in page.")
    idp = db.get_idp(pending["idp_id"])
    if not idp or not idp["enabled"]:
        raise SsoError("That sign-in method is no longer available here.")

    tokens = await _exchange(idp, code, pending["redirect_uri"],
                             pending["verifier"])
    raw_id = tokens.get("id_token")
    if not raw_id:
        raise SsoError("The provider did not return an ID token. Check that "
                       "the 'openid' scope is allowed for this application.")
    try:
        claims = await idtoken.verify(
            raw_id, jwks_uri=config_of(idp)["jwks_uri"], issuer=idp["issuer"],
            audience=idp["client_id"], nonce=pending["nonce"])
    except idtoken.TokenError as e:
        raise SsoError(str(e)) from e
    except httpx.HTTPError as e:
        raise SsoError("Could not fetch the provider's signing keys (%s)." % e) from e

    profile = dict(claims)
    if not profile.get("email"):
        profile.update({k: v for k, v in
                        (await _userinfo(idp, tokens.get("access_token") or "")).items()
                        if v is not None})
    return sign_in(idp, profile), pending["next_url"]


def _address(claims: dict) -> tuple[str | None, bool]:
    """The person's address, and whether the provider vouches for it."""
    email = (claims.get("email") or "").strip()
    if not email:
        # Entra puts the work address here when the `email` claim is not
        # configured, and it is an address rather than a login name whenever
        # it contains an @.
        upn = (claims.get("preferred_username") or claims.get("upn") or "").strip()
        if "@" in upn:
            email = upn
    verified = claims.get("email_verified")
    if verified is None:
        # Absent, not false. Entra and most enterprise providers omit it
        # because every address they hold is one they issued.
        verified = True
    return (email.lower() or None), bool(verified)


def sign_in(idp: dict, claims: dict) -> dict:
    """Turn verified claims into the account they belong to.

    Never returns an account it is not certain of: the alternative to raising
    here is signing somebody in as somebody else.
    """
    subject = claims.get("sub")
    if not subject:
        raise SsoError("The provider did not say who signed in.")
    email, verified = _address(claims)
    name = (claims.get("name") or claims.get("given_name") or "").strip()

    allowed = [d.strip().lower().lstrip("@")
               for d in (idp["allowed_domains"] or "").split(",") if d.strip()]
    if allowed:
        domain = (email or "").rsplit("@", 1)[-1]
        if not email or domain not in allowed:
            raise SsoError(
                "%s is not one of the addresses allowed to sign in here (%s)."
                % (email or "That account", ", ".join(allowed)))

    user = db.get_user_by_external(idp["id"], subject)

    if user is None and email and verified and idp["link_by_email"]:
        candidate = db.get_user_by_email(email)
        if candidate and candidate.get("provider") in (None, "", idp["id"]):
            user = candidate
        elif candidate:
            raise SsoError("An account here already uses %s through a "
                           "different sign-in method. An administrator has to "
                           "sort that out." % email)

    if user is None and email and verified and idp["link_by_email"]:
        # The username case: an SSO rollout over a studio where people already
        # have password accounts named after them.
        local = db.get_user_by_name(email.split("@")[0])
        if local and not local.get("provider") and not local.get("email"):
            user = local

    if user is None:
        if not idp["auto_create"]:
            raise SsoError(
                "%s has no account on this studio, and this sign-in method is "
                "set to admit only people who already do. Ask an "
                "administrator to add you." % (email or "That account"))
        username = db.unique_username(
            (email or "").split("@")[0] or name.replace(" ", ".") or subject)
        uid = db.create_user(username, name or username, "", "member")
        user = db.get_user(uid)

    if not user["active"]:
        raise SsoError("That account has been disabled on this studio.")

    fields: dict[str, Any] = {
        "provider": idp["id"], "external_id": subject, "pending": 0,
        "last_login": time.time(), "synced_at": time.time(),
        # A password account that becomes an SSO account keeps nothing to
        # verify against, so the old hash goes. Leaving it would mean the
        # password screen still works for an account the directory now owns.
        "password_hash": "", "must_change": 0,
    }
    if email:
        fields["email"] = email
    if name:
        fields["display_name"] = name[:80]
    if picture := (claims.get("picture") or "").strip():
        fields["avatar_url"] = picture[:500]

    groups = _claim_groups(claims)
    if groups:
        fields["groups"] = json.dumps(groups[:200])
    if role := role_for(idp, groups):
        fields["role"] = role

    db.update_user(user["id"], **fields)
    return db.get_user(user["id"])


def role_for(idp: dict, groups: list[str]) -> str | None:
    """Administrator, if one of their groups says so. Never a demotion.

    Promotion is automatic and demotion is not, deliberately: a group claim
    that is briefly missing -- which happens, because Entra omits `groups`
    entirely once somebody is in more than about two hundred of them -- would
    otherwise quietly strip an administrator of their own studio.
    """
    wanted = [g.strip().lower()
              for g in (idp["admin_groups"] or "").split(",") if g.strip()]
    if not wanted:
        return None
    if any(g.strip().lower() in wanted for g in groups):
        return "admin"
    return None
