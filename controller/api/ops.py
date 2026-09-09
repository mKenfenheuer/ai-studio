"""Running the models: what is deployed where, what it costs, and who may ask.

The studio could train a model, register a name for it and serve it over an
OpenAI-compatible API, and then had nothing to say about any of it. Whether a
served name was answering, how fast, how often it failed, which key was doing
all the work, and which machine was carrying it were all facts the system had
and no page asked for. The numbers existed in `usage`; the residency existed in
the fleet's heartbeats; neither was ever put in front of anybody.

Three groups of endpoints, and the split matters:

**Deployments** are the only ones that change what the fleet is doing. A
deployment pins one model to one machine's card and keeps it there -- see the
`deployments` table for why that is a table rather than a request. Creating one
does not load anything itself: it writes the row and lets the scheduler's
reconciler do the work, so a machine that is away, busy, or restarting reaches
the same end state as one that is idle, by the same path.

**Metrics** are read-only and derived entirely from the `usage` ledger. Nothing
here is sampled or estimated -- the token counts are the runner's own, and the
seconds are measured around the generation. The one number worth explaining is
tokens per second, which is a sum divided by a sum rather than an average of
per-reply rates: averaging those weights a two-token reply the same as a
two-thousand-token one and reports a fleet far faster than it has ever been.

**Keys** here are the whole studio's, which is the difference between this and
the same list on somebody's account page. An administrator turning off a key
that is hammering a model needs to see keys that are not theirs.
"""
from __future__ import annotations

import time

from fastapi import APIRouter, Body, HTTPException, Query, Request

from .. import db, serving as spec_for
from .security import current_user, require_admin, require_view

router = APIRouter(prefix="/api/ops")

# Set by the application once the fleet exists. Same arrangement as the
# serving router: the fleet is live state owned by the app, not by a module.
FLEET = None

# How far back the pages default to looking. A day is the operational window
# -- "is it healthy right now" -- and the longer views are for the usage page
# that already exists.
DEFAULT_HOURS = 24


def _rate(tokens: float | None, seconds: float | None) -> float | None:
    """Tokens per second, or None when there is nothing to divide.

    Returned as None rather than 0.0 for "no data": a fleet that has served
    nothing is not a fleet serving at zero tokens a second, and a dash in the
    interface is honest where a number is not.
    """
    if not seconds or seconds <= 0 or not tokens:
        return None
    return round(float(tokens) / float(seconds), 1)


# ------------------------------------------------------------- deployments

def _job_or_400(job_id: str) -> dict:
    job = db.get_job(job_id or "")
    if not job:
        raise HTTPException(404, "No such run.")
    if job["kind"] not in spec_for.MODEL_KINDS:
        raise HTTPException(400, "That run did not produce a model to serve.")
    return job


def _describe(row: dict, runners: dict, user: dict) -> dict:
    """One deployment, with enough around it to be read without a second call."""
    job = db.get_job(row["job_id"])
    runner = runners.get(row["runner_id"])
    visible = bool(job) and bool(
        db.access_level("job", row["job_id"], job.get("owner_id"), user))
    live = (FLEET.pinned.get(row["runner_id"]) or []) if FLEET else []
    return {
        "id": row["id"],
        "job_id": row["job_id"] if visible else "",
        "job_name": (job or {}).get("name") if visible else "a run you cannot see",
        "job_gone": job is None,
        "runner_id": row["runner_id"],
        "runner_name": (runner or {}).get("name") or "a machine that is gone",
        "runner_online": bool(runner) and bool(
            FLEET and row["runner_id"] in FLEET.connections),
        "state": row["state"],
        "detail": row.get("detail") or "",
        # What the machine says right now, as opposed to what the row last
        # recorded. They disagree for the few seconds around a restart, and
        # the honest thing is to show the machine's answer.
        "resident": row["job_id"] in live,
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "ready_at": row.get("ready_at"),
        "load_s": row.get("load_s"),
        # The names clients are pointed at that resolve to this model. This is
        # the join that makes a deployment mean something operationally: it is
        # not "a model on a card", it is "the thing home-assistant talks to".
        "aliases": db.aliases_for_job(row["job_id"]) if visible else [],
    }


@router.get("/deployments")
async def list_deployments(request: Request) -> list[dict]:
    user = current_user(request)
    runners = {r["id"]: r for r in db.list_runners()}
    return [_describe(row, runners, user) for row in db.list_deployments()]


@router.post("/deployments")
async def create_deployment(request: Request, payload: dict = Body(...)) -> dict:
    """Hold a model on a machine, from now until somebody says otherwise.

    Deliberately allowed to anyone who can see the run rather than restricted
    to administrators. Deploying is how you make a model you trained answer
    quickly, and making that an administrator's job means it does not happen.
    What it costs is a machine's memory, which the reconciler and the eviction
    rules already arbitrate; what it cannot do is take a card away from a run.
    """
    user = current_user(request)
    job = _job_or_400((payload.get("job_id") or "").strip())
    require_view(request, "job", job)

    runner_id = (payload.get("runner_id") or "").strip()
    runner = db.get_runner(runner_id)
    if not runner:
        raise HTTPException(404, "No such machine.")
    caps = runner.get("capabilities") or {}
    if caps.get("backend") not in ("cuda", "rocm", "mps"):
        raise HTTPException(
            400, "That machine has no graphics card. It would load the model "
                 "and answer at a word every few seconds, which is not worth "
                 "the memory it would hold.")
    if db.runner_role(runner) == "training":
        raise HTTPException(
            400, "That machine is reserved for training. Set it to serving, "
                 "or to both, before deploying a model to it.")

    row = db.create_deployment(job["id"], runner_id, owner_id=user["id"])
    db.add_log(job["id"], "Deployed to '%s': the model is held on that "
                          "machine's card so it answers without loading first."
                          % runner["name"])
    # Woken rather than sent directly: the reconciler is the one path that
    # loads a deployment, so a deployment made while the machine is training
    # behaves exactly like one made while it is idle.
    if FLEET:
        FLEET.wake()
        await FLEET.broadcast_ui({"type": "deployments_changed",
                                  "deployment_id": row["id"]})
    return _describe(row, {runner_id: runner}, user)


def _deployment_or_404(deployment_id: str) -> dict:
    row = db.get_deployment(deployment_id or "")
    if not row:
        raise HTTPException(404, "No such deployment.")
    return row


@router.delete("/deployments/{deployment_id}")
async def remove_deployment(request: Request, deployment_id: str) -> dict:
    """Stop holding a model, and let go of the memory now rather than later."""
    user = current_user(request)
    row = _deployment_or_404(deployment_id)
    if row.get("owner_id") and row["owner_id"] != user["id"] \
            and user.get("role") != "admin":
        raise HTTPException(
            403, "Somebody else deployed this. Ask them to take it down, or "
                 "ask an administrator.")
    db.delete_deployment(row["id"])
    # Unpinned *and* unloaded. Leaving it resident would be defensible -- it
    # is a warm cache once it is no longer a deployment -- but somebody taking
    # a deployment down is usually doing it to get the memory back, and a
    # button that frees nothing visible reads as a button that did nothing.
    if FLEET:
        await FLEET.send_to_runner(row["runner_id"],
                                   {"type": "unload_model",
                                    "job_id": row["job_id"]})
        await FLEET.broadcast_ui({"type": "deployments_changed",
                                  "deployment_id": row["id"]})
    return {"ok": True}


@router.post("/deployments/{deployment_id}/retry")
async def retry_deployment(request: Request, deployment_id: str) -> dict:
    """Ask again after a failure.

    A failed deployment is left alone by the reconciler -- retrying a load
    that ran out of memory every five seconds forever is not helping anybody
    -- so it takes a person to say "again", usually after making room.
    """
    current_user(request)
    row = _deployment_or_404(deployment_id)
    db.set_deployment_state(row["id"], "pending", "Asked again.")
    if FLEET:
        FLEET.wake()
        await FLEET.broadcast_ui({"type": "deployments_changed",
                                  "deployment_id": row["id"]})
    return {"ok": True}


# ------------------------------------------------------------ reservations

@router.put("/runners/{runner_id}/role")
async def set_runner_role(request: Request, runner_id: str,
                          payload: dict = Body(...)) -> dict:
    """Set a machine aside for serving, for training, or for neither.

    An administrator's decision, unlike deploying: it changes where everybody
    else's runs are dispatched, and a member reserving the only card for
    serving would stop the studio training at all.
    """
    require_admin(request)
    if not db.get_runner(runner_id):
        raise HTTPException(404, "No such machine.")
    role = (payload.get("role") or "").strip().lower()
    if role not in db.RUNNER_ROLES:
        raise HTTPException(400, "A machine is set to one of: %s."
                                 % ", ".join(db.RUNNER_ROLES))
    db.set_runner_role(runner_id, role, payload.get("note"))
    if FLEET:
        # A machine that has just been freed for training may have work
        # waiting for it this second.
        FLEET.wake()
        await FLEET.broadcast_ui({"type": "runners_changed"})
    return {"ok": True, "role": role}


# ----------------------------------------------------------------- metrics

@router.get("/metrics")
async def metrics(request: Request, hours: int = Query(DEFAULT_HOURS)) -> dict:
    """How the served models are behaving, over the last few hours.

    Scoped to the caller unless they are an administrator, for the same reason
    the usage page is: a per-user report nobody can see the total of is not a
    report, and one person's traffic is not another's business.
    """
    user = current_user(request)
    hours = max(1, min(int(hours or DEFAULT_HOURS), db.USAGE_DAYS * 24))
    since = time.time() - hours * 3600
    whose = None if user.get("role") == "admin" else user["id"]

    totals = db.usage_totals(whose, since)
    runners = {r["id"]: r for r in db.list_runners()}
    keys = {k["id"]: k for k in db.all_api_keys()} if whose is None else \
        {k["id"]: k for k in db.list_api_keys(user["id"])}

    def by(column: str, name):
        out = []
        for row in db.usage_by(column, whose, since):
            out.append({**row, "rate": _rate(row.get("completion_tokens"),
                                             row.get("seconds")),
                        **name(row)})
        return out

    def model_name(row):
        job = db.get_job(row["key"] or "")
        return {"name": (job or {}).get("name") or row["key"] or "unknown",
                "job_id": row["key"], "gone": job is None}

    def key_name(row):
        key = keys.get(row["key"])
        if not row["key"]:
            # Usage with no key behind it came in on a session cookie: the
            # studio's own playground, or somebody testing from the browser.
            return {"name": "the studio itself", "prefix": ""}
        return {"name": (key or {}).get("name") or "a key that has been deleted",
                "prefix": (key or {}).get("prefix") or ""}

    def runner_name(row):
        r = runners.get(row["key"] or "")
        return {"name": (r or {}).get("name")
                or ("unrecorded" if not row["key"] else "a machine that is gone")}

    return {
        "hours": hours,
        "scope": "studio" if whose is None else "you",
        "totals": {
            **totals,
            "rate": _rate(totals.get("completion_tokens"),
                          totals.get("seconds")),
            # As a share rather than only a count: six failures matters very
            # differently at sixty calls and at sixty thousand.
            "error_rate": round(100.0 * (totals.get("errors") or 0)
                                / totals["calls"], 1) if totals.get("calls")
            else None,
        },
        "series": db.usage_series(whose, hours),
        "by_model": by("job_id", model_name),
        "by_key": by("api_key_id", key_name),
        "by_runner": by("runner_id", runner_name),
        "by_alias": [r for r in by("alias", lambda row: {"name": row["key"]})
                     if r["key"]],
        # The failures themselves, not just how many there were. "It is
        # erroring" is where somebody starts; what they need next is the
        # sentence the runner produced.
        "recent_errors": db.usage_recent(whose, limit=25, only_errors=True),
        "kept_days": db.USAGE_DAYS,
    }


# -------------------------------------------------------------------- keys

@router.get("/keys")
async def studio_keys(request: Request) -> list[dict]:
    """Every key in the studio, with whose it is. Administrators only.

    The same shape as the account page's list, and never the key itself --
    only its prefix. Nothing anywhere can show a key again after it is made,
    which is a property of storing the hash rather than a policy.
    """
    require_admin(request)
    out = []
    for row in db.all_api_keys():
        owner = db.get_user(row.get("user_id") or "")
        out.append({**row,
                    "owner": db.public_user(owner) if owner else None})
    return out


@router.delete("/keys/{key_id}")
async def revoke_key(request: Request, key_id: str) -> dict:
    require_admin(request)
    if not db.delete_any_api_key(key_id):
        raise HTTPException(404, "No such key.")
    return {"ok": True, "note": "Anything using that key is now getting a 401."}
