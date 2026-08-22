"""Telling somebody a run has ended.

A training run here takes hours. Until now the only way to learn that yours
had finished -- or failed in the first two minutes, which is the case that
actually costs you the evening -- was to have the page open and be looking at
it. That is a poor deal for a tool whose whole premise is that you start
something and go away.

Two channels, chosen because between them they cover where people actually
are:

* **A webhook.** One HTTP POST with a small JSON body. It works with ntfy,
  Slack, Discord, Home Assistant, Gotify, or twelve lines of your own -- which
  matters more than picking one of them, because this app has no business
  holding your Slack workspace or your mail server credentials.

* **A browser notification**, for the tab you left open on the machine you
  are sitting at. Free: the event stream already carries the news, and it only
  fires when the page is not the one you are looking at.

Deliberately not e-mail. Sending it properly needs an SMTP server, credentials
to store, a bounce story and a deliverability story, and the result would be
worse than a webhook into whatever you already read.
"""
from __future__ import annotations

import asyncio
import json
from urllib.parse import urlparse

import httpx

from . import auth, db

TIMEOUT_S = 8.0

# What a notification may be about. Stored per user as a list, so somebody who
# only wants to hear about failures does not have to mute everything.
EVENTS = {
    "succeeded": "a run finished",
    "failed": "a run failed",
    "cancelled": "a run was stopped",
}
DEFAULT_EVENTS = ["succeeded", "failed"]


def masked(url: str | None) -> str | None:
    """Enough of the URL to recognise it, never enough to use it.

    A webhook URL is a credential -- anyone holding a Slack or ntfy one can
    post as you -- so it is encrypted at rest and never returned in full, the
    same rule the Hugging Face token follows.
    """
    if not url:
        return None
    try:
        p = urlparse(url)
        tail = (p.path or "/").rstrip("/").rsplit("/", 1)[-1]
    except ValueError:
        return "(unreadable)"
    # The last few characters only when there are enough of them for four to
    # be a hint rather than most of the answer. A Slack webhook path is long
    # and its tail identifies it; an ntfy topic can be seven characters, where
    # showing four would be showing it.
    if len(tail) >= 12:
        return "%s://%s/…%s" % (p.scheme, p.netloc, tail[-4:])
    return "%s://%s/…" % (p.scheme, p.netloc)


def settings_for(user: dict) -> dict:
    raw = user.get("notify_events")
    try:
        events = json.loads(raw) if raw else DEFAULT_EVENTS
    except (TypeError, ValueError):
        events = DEFAULT_EVENTS
    return {
        "configured": bool(user.get("notify_url_enc")),
        "url_hint": masked(auth.decrypt_secret(user.get("notify_url_enc"))),
        "events": events,
        "available_events": EVENTS,
    }


def payload_for(job: dict, status: str) -> dict:
    """The body posted to the webhook.

    Flat, small, and self-describing. `text` exists because half the services
    people point this at render one field and ignore the rest; the structured
    fields exist because the other half do something useful with them.
    """
    summary = job.get("summary") or {}
    kind = {"pretrain_llm": "from-scratch model", "finetune_llm": "fine-tune",
            "generate_dataset": "dataset generation",
            "evaluate": "evaluation"}.get(job.get("kind"), job.get("kind"))
    lines = ["%s: %s" % (EVENTS.get(status, status).capitalize(), job.get("name"))]
    if status == "failed" and job.get("error"):
        lines.append(str(job["error"])[:300])
    else:
        bits = []
        if summary.get("best_val_loss") is not None:
            bits.append("held-out loss %.4f" % summary["best_val_loss"])
        elif summary.get("final_loss") is not None:
            bits.append("loss %.4f" % summary["final_loss"])
        if summary.get("steps"):
            bits.append("%s steps" % f"{summary['steps']:,}")
        if summary.get("early_stopped"):
            bits.append("stopped early, at its best point")
        if summary.get("kept_from_step"):
            bits.append("kept the model from step %d" % summary["kept_from_step"])
        if bits:
            lines.append(", ".join(bits))
    return {
        "event": "job_%s" % status,
        "status": status,
        "job_id": job.get("id"),
        "name": job.get("name"),
        "kind": kind,
        "error": job.get("error"),
        "steps": summary.get("steps"),
        "held_out_loss": summary.get("best_val_loss"),
        "duration_s": summary.get("duration_s"),
        "text": "\n".join(lines),
    }


async def _post(url: str, body: dict) -> tuple[bool, str]:
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_S) as c:
            r = await c.post(url, json=body,
                             headers={"User-Agent": "ai-studio/0.1"})
        if r.status_code >= 400:
            return False, "%d %s" % (r.status_code, r.text[:120])
        return True, str(r.status_code)
    except Exception as e:  # noqa: BLE001 - somebody else's server, any failure
        return False, type(e).__name__ + ": " + str(e)[:120]


async def send_test(user: dict) -> dict:
    url = auth.decrypt_secret(user.get("notify_url_enc"))
    if not url:
        return {"ok": False, "detail": "No webhook is set up."}
    ok, detail = await _post(url, {
        "event": "test", "status": "test", "name": "Test notification",
        "text": "AI Studio can reach this webhook. Real notifications look "
                "like this one, with the run's name and how it ended."})
    return {"ok": ok, "detail": detail}


async def job_ended(job: dict, status: str) -> None:
    """Notify the run's owner, if they asked to hear about this outcome.

    Only the owner. People a run was shared with did not start it and did not
    ask to be woken by it; sharing a run is not subscribing to it.

    Every failure here is swallowed. The webhook points at somebody else's
    server, and a run must not be recorded as failed because a Slack endpoint
    was down.
    """
    owner_id = job.get("owner_id")
    if not owner_id:
        return
    user = db.get_user(owner_id)
    if not user:
        return
    conf = settings_for(user)
    if not conf["configured"] or status not in conf["events"]:
        return
    url = auth.decrypt_secret(user.get("notify_url_enc"))
    if not url:
        return
    ok, detail = await _post(url, payload_for(job, status))
    if not ok:
        # Recorded on the run rather than swallowed silently: a webhook that
        # has been failing for a week is worth being able to discover.
        db.add_log(job["id"], "Could not deliver the notification for this "
                   "run (%s). Check the webhook in your account settings."
                   % detail, "warn")


def fire(job: dict, status: str) -> None:
    """Send in the background. Never blocks the scheduler on a slow endpoint."""
    try:
        asyncio.get_running_loop().create_task(job_ended(job, status))
    except RuntimeError:
        pass
