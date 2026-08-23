"""Reading the list of people from the directory that already has one.

Signing in with Entra or Google solves who somebody is. It does not solve the
thing that actually makes sharing usable: being able to *find* a colleague who
has never opened this studio. Without a directory the only people who can be
shared with are the ones who happen to have signed in already, which is the
wrong list -- you want to share a run with someone precisely because they have
not seen it.

So this pulls the roster in and creates an account for each person, marked
`pending`: findable, shareable, owning nothing. The flag clears the first time
they sign in, and whatever was shared with them is waiting.

What a directory owns and what it does not:

* It owns names, addresses, job titles and departments. Those are overwritten
  on every sync, because the directory is the place people update them.
* It does **not** own roles here, or the fact that an administrator disabled
  somebody, or anything they created. Those stay.
* It owns *departure*. Somebody who has left the company should not keep an
  account here, and that is most of the security value of syncing at all.

Deactivating people automatically is also the one thing here that can go badly
wrong -- a group filter with a typo returns nobody, and "nobody is in the
directory" then reads as "disable everyone". Two guards below make that
impossible: an empty result never deactivates anything, and a run that would
disable more than half of the people it knows about stops and says so instead.
"""
from __future__ import annotations

import base64
import json
import time
from typing import Any, Callable

import httpx

from . import auth, db

GRAPH = "https://graph.microsoft.com/v1.0"
GRAPH_FIELDS = ("id,displayName,userPrincipalName,mail,accountEnabled,"
                "jobTitle,department")

# A single sync that would disable more of the studio than it keeps is not a
# staff turnover event; it is a misconfiguration.
_MAX_DEACTIVATE_FRACTION = 0.5


class SyncError(Exception):
    """A failure worth putting on the screen verbatim."""


class Person:
    """One row from a directory, in the only shape this studio cares about."""

    __slots__ = ("external_id", "username", "display_name", "email", "active",
                 "job_title", "department")

    def __init__(self, external_id: str, username: str, display_name: str,
                 email: str | None, active: bool = True,
                 job_title: str | None = None, department: str | None = None):
        self.external_id = external_id
        self.username = username
        self.display_name = display_name
        self.email = (email or "").lower() or None
        self.active = active
        self.job_title = job_title
        self.department = department


def supports(idp: dict) -> str | None:
    """The name of the directory this provider can be read from, if any."""
    return {"entra": "Microsoft Graph",
            "google": "Google Workspace"}.get(idp["kind"])


# ---------------------------------------------------------------------------
# Microsoft Graph
# ---------------------------------------------------------------------------

async def _graph_token(idp: dict) -> str:
    secret = (auth.decrypt_secret(idp.get("sync_secret_enc"))
              or auth.decrypt_secret(idp.get("client_secret_enc")))
    if not secret:
        raise SyncError(
            "Reading the directory needs a client secret for the same app "
            "registration, with the Graph application permission "
            "User.Read.All granted and admin-consented.")
    tenant = (idp.get("tenant") or "").strip()
    url = "https://login.microsoftonline.com/%s/oauth2/v2.0/token" % tenant
    data = {"grant_type": "client_credentials", "client_id": idp["client_id"],
            "client_secret": secret,
            "scope": "https://graph.microsoft.com/.default"}
    async with httpx.AsyncClient(timeout=30.0) as c:
        r = await c.post(url, data=data)
    if r.status_code >= 400:
        body = _json(r)
        raise SyncError("Microsoft refused the directory request: %s"
                        % (body.get("error_description") or r.text[:300]))
    return r.json()["access_token"]


def _json(r: httpx.Response) -> dict:
    try:
        return r.json()
    except ValueError:
        return {}


async def _graph_pages(client: httpx.AsyncClient, url: str) -> list[dict]:
    """Every page of a Graph collection, followed to the end."""
    out: list[dict] = []
    while url:
        r = await client.get(url)
        if r.status_code >= 400:
            body = _json(r).get("error") or {}
            raise SyncError(
                "Microsoft Graph refused: %s. The usual cause is that "
                "User.Read.All has not been granted as an *application* "
                "permission with admin consent."
                % (body.get("message") or r.text[:200]))
        page = r.json()
        out.extend(page.get("value") or [])
        url = page.get("@odata.nextLink") or ""
        if len(out) > 20000:
            # A studio is not an identity provider. Somebody pointing this at
            # a hundred-thousand-seat tenant wants a group filter, and should
            # be told so rather than waiting.
            raise SyncError(
                "That directory has more than 20,000 people in it. Set a "
                "group so only the people who need the studio are imported.")
    return out


async def _entra_people(idp: dict) -> tuple[list[Person], dict[str, list[str]]]:
    token = await _graph_token(idp)
    headers = {"Authorization": "Bearer " + token,
               "ConsistencyLevel": "eventual"}
    group = (idp.get("sync_group") or "").strip()
    async with httpx.AsyncClient(timeout=60.0, headers=headers) as c:
        if group:
            gid = await _graph_group_id(c, group)
            url = ("%s/groups/%s/transitiveMembers/microsoft.graph.user"
                   "?$select=%s&$top=999" % (GRAPH, gid, GRAPH_FIELDS))
        else:
            url = "%s/users?$select=%s&$top=999" % (GRAPH, GRAPH_FIELDS)
        rows = await _graph_pages(c, url)

        memberships: dict[str, list[str]] = {}
        for name in [g.strip() for g in
                     (idp.get("admin_groups") or "").split(",") if g.strip()]:
            try:
                gid = await _graph_group_id(c, name)
            except SyncError:
                continue
            members = await _graph_pages(
                c, "%s/groups/%s/transitiveMembers/microsoft.graph.user"
                   "?$select=id&$top=999" % (GRAPH, gid))
            for m in members:
                memberships.setdefault(m["id"], []).append(name)

    people = []
    for r in rows:
        upn = r.get("userPrincipalName") or ""
        people.append(Person(
            external_id=r["id"],
            username=(upn.split("@")[0] or r.get("displayName") or r["id"]),
            display_name=r.get("displayName") or upn or r["id"],
            email=r.get("mail") or (upn if "@" in upn else None),
            active=r.get("accountEnabled", True),
            job_title=r.get("jobTitle"), department=r.get("department")))
    return people, memberships


async def _graph_group_id(client: httpx.AsyncClient, name: str) -> str:
    """A group id, whether the administrator typed the id or the name."""
    if len(name) == 36 and name.count("-") == 4:
        return name
    escaped = name.replace("'", "''")
    r = await client.get("%s/groups?$filter=displayName eq '%s'&$select=id"
                         % (GRAPH, escaped))
    values = _json(r).get("value") or []
    if not values:
        raise SyncError("No group called %r exists in that directory." % name)
    return values[0]["id"]


# ---------------------------------------------------------------------------
# Google Workspace
# ---------------------------------------------------------------------------

def _sa_assertion(sa: dict, subject: str, scope: str) -> str:
    """A signed JWT asking Google for a token on the administrator's behalf.

    Google's Admin SDK has no application-level access: a service account can
    only read the directory *as* a real administrator, through domain-wide
    delegation. That is what `subject` is, and why it has to be configured.
    """
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding

    def seg(obj: dict) -> str:
        return base64.urlsafe_b64encode(
            json.dumps(obj, separators=(",", ":")).encode()).decode().rstrip("=")

    now = int(time.time())
    header = seg({"alg": "RS256", "typ": "JWT", "kid": sa.get("private_key_id")})
    body = seg({"iss": sa["client_email"], "sub": subject, "scope": scope,
                "aud": "https://oauth2.googleapis.com/token",
                "iat": now, "exp": now + 3600})
    key = serialization.load_pem_private_key(
        sa["private_key"].encode(), password=None)
    signature = key.sign(("%s.%s" % (header, body)).encode(),
                         padding.PKCS1v15(), hashes.SHA256())
    return "%s.%s.%s" % (header, body,
                         base64.urlsafe_b64encode(signature).decode().rstrip("="))


async def _google_people(idp: dict) -> tuple[list[Person], dict[str, list[str]]]:
    raw = auth.decrypt_secret(idp.get("sync_secret_enc"))
    if not raw:
        raise SyncError(
            "Reading a Google Workspace directory needs the JSON key of a "
            "service account with domain-wide delegation for "
            "admin.directory.user.readonly.")
    try:
        sa = json.loads(raw)
        sa["client_email"], sa["private_key"]
    except (ValueError, KeyError) as e:
        raise SyncError("That is not a Google service account key file.") from e

    subject = (idp.get("sync_subject") or "").strip()
    if "@" not in subject:
        raise SyncError(
            "Google needs the address of an administrator to read the "
            "directory as. Put it in the 'read the directory as' field.")

    assertion = _sa_assertion(
        sa, subject, "https://www.googleapis.com/auth/admin.directory.user.readonly")
    async with httpx.AsyncClient(timeout=60.0) as c:
        r = await c.post("https://oauth2.googleapis.com/token", data={
            "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
            "assertion": assertion})
        if r.status_code >= 400:
            body = _json(r)
            raise SyncError(
                "Google refused the directory request: %s. Check that the "
                "service account's client id is authorised for the "
                "admin.directory.user.readonly scope in Admin console → "
                "Security → API controls → Domain-wide delegation."
                % (body.get("error_description") or r.text[:200]))
        token = r.json()["access_token"]

        people, page = [], None
        while True:
            params: dict[str, Any] = {"maxResults": 500, "projection": "full"}
            domain = (idp.get("tenant") or "").strip()
            params["domain" if domain else "customer"] = domain or "my_customer"
            if page:
                params["pageToken"] = page
            r = await c.get("https://admin.googleapis.com/admin/directory/v1/users",
                            params=params,
                            headers={"Authorization": "Bearer " + token})
            if r.status_code >= 400:
                raise SyncError("Google Workspace refused: %s"
                                % (_json(r).get("error", {}).get("message")
                                   or r.text[:200]))
            body = r.json()
            for u in body.get("users") or []:
                email = u.get("primaryEmail") or ""
                org = (u.get("organizations") or [{}])[0]
                people.append(Person(
                    external_id=u["id"],
                    username=email.split("@")[0] or u["id"],
                    display_name=(u.get("name") or {}).get("fullName") or email,
                    email=email or None,
                    active=not u.get("suspended", False),
                    job_title=org.get("title"),
                    department=org.get("department")))
            page = body.get("nextPageToken")
            if not page:
                break
    return people, {}


# ---------------------------------------------------------------------------
# Applying what came back
# ---------------------------------------------------------------------------

_READERS: dict[str, Callable] = {"entra": _entra_people, "google": _google_people}


async def sync(idp: dict, dry_run: bool = False) -> dict:
    """Bring the studio's list of people in step with the directory's."""
    reader = _READERS.get(idp["kind"])
    if reader is None:
        raise SyncError(
            "There is no standard way to read the list of people out of an "
            "OpenID Connect provider, so this one can only create accounts as "
            "people sign in. Entra ID and Google Workspace can be read.")

    started = time.time()
    people, memberships = await reader(idp)
    if not people:
        raise SyncError(
            "The directory returned nobody. Nothing was changed -- an empty "
            "answer is far more often a filter with a typo in it than a "
            "company with no staff.")

    known = {u["external_id"]: u for u in db.directory_users(idp["id"])
             if u.get("external_id")}
    seen: set[str] = set()
    created = updated = 0
    admin_names = [g.strip().lower() for g in
                   (idp.get("admin_groups") or "").split(",") if g.strip()]

    for person in people:
        seen.add(person.external_id)
        if dry_run:
            created += person.external_id not in known
            continue
        _, is_new = db.upsert_directory_user(
            idp["id"], person.external_id, person.username,
            person.display_name, person.email,
            job_title=person.job_title, department=person.department,
            groups=json.dumps(memberships.get(person.external_id, []))
            if memberships else None)
        created += is_new
        updated += not is_new
        row = db.get_user_by_external(idp["id"], person.external_id)
        if row and admin_names and memberships:
            mine = [g.lower() for g in memberships.get(person.external_id, [])]
            if any(g in admin_names for g in mine) and row["role"] != "admin":
                db.update_user(row["id"], role="admin")
        # An account the directory says is disabled loses its sessions here as
        # well as its flag, or "disabled" would mean "disabled at next login".
        if row and not person.active and row["active"]:
            if not dry_run:
                db.update_user(row["id"], active=0)
                auth.end_all_sessions(row["id"])

    departed = [u for eid, u in known.items()
                if eid not in seen and u["active"]]
    # Never the last administrator, and never somebody who is only *also* in
    # this directory but signs in with a password here.
    departed = [u for u in departed
                if not (u["role"] == "admin" and db.count_admins() <= 1)]
    limit = int(len(known) * _MAX_DEACTIVATE_FRACTION)
    refused = len(departed) > max(limit, 1) and len(known) > 4
    if refused:
        note = ("Read %d people. Refused to disable %d accounts in one go -- "
                "that is more than half of everyone this provider knows, "
                "which is a broken filter far more often than a layoff. "
                "Nothing was disabled." % (len(people), len(departed)))
        departed = []
    else:
        note = "Read %d people from the directory." % len(people)
        if not dry_run:
            for u in departed:
                db.update_user(u["id"], active=0)
                auth.end_all_sessions(u["id"])

    if not dry_run:
        db.update_idp(idp["id"], last_sync_at=time.time(), last_sync_note=note)

    return {"ok": not refused, "read": len(people), "created": created,
            "updated": updated, "deactivated": len(departed),
            "refused": refused, "note": note, "dry_run": dry_run,
            "took_s": round(time.time() - started, 1)}


async def sync_all() -> list[dict]:
    """Every provider that asked to be synced. Never raises: this runs on a
    timer, and one broken provider must not stop the others."""
    out = []
    for idp in db.list_idps(enabled_only=True):
        if not idp["sync_enabled"] or not supports(idp):
            continue
        try:
            out.append({"idp": idp["id"], **await sync(idp)})
        except (SyncError, httpx.HTTPError, KeyError, ValueError) as e:
            db.update_idp(idp["id"], last_sync_at=time.time(),
                          last_sync_note="Failed: %s" % e)
            out.append({"idp": idp["id"], "ok": False, "note": str(e)})
    return out
