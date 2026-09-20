#!/usr/bin/env python
"""Check the operations side: reservations, deployments, and what gets counted.

Run it with `python scripts/check-ops.py` from the repository root.

Three things here are worth a check that runs, because all three are invisible
until somebody is depending on them.

**A reservation has to be honoured in two places.** The scheduler must not
dispatch a run to a machine set aside for serving, and the "why is this run
waiting" explanation must say that is why. Those are different functions and
they disagreed by construction once already: the dispatcher skipped the
machine and the explanation cheerfully reported it as able to take the work,
so a run sat in a queue in front of a card that the page said was free.

**A deployment is a promise about a card.** Asking twice must not make two of
them, a failure must be retryable by asking again, and the state must be what
the machine last said rather than what was intended.

**The ledger has to count failures.** Every row in `usage` used to be a
success, because a failure returned before anything was written -- so the one
question operations asks first, "is it erroring", could not be answered at
all, and every rate computed from the table was flattered by the calls that
never happened.

And a fourth, smaller: an account with the `chat` role must be able to reach
the handful of endpoints its page calls and nothing else. That list is an
allowlist on purpose (see api/security.py), so the check that matters is that
a path nobody thought about is refused rather than allowed.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_TMP = Path(tempfile.mkdtemp(prefix="ai-studio-ops-"))
os.environ["AI_STUDIO_DATA"] = str(_TMP)

from controller import db  # noqa: E402
from controller.api import security  # noqa: E402
from controller.scheduler import Fleet  # noqa: E402

FAILED: list[str] = []

GPU = {"backend": "cuda", "vram_gb": 24.0, "quantization": {"4bit": True}}


def ok(label: str, condition: bool, note: str = "") -> None:
    print("  %s  %s %s" % ("ok  " if condition else "FAIL", label, note))
    if not condition:
        FAILED.append(label)


def runner(rid: str, name: str, role: str | None = None) -> str:
    db.upsert_runner(rid, name, GPU)
    if role:
        db.set_runner_role(rid, role)
    return rid


# --------------------------------------------------------------- reserving
print("A machine set aside for serving")

a = runner("run_a", "workstation")
b = runner("run_b", "the serving box", "serving")

ok("defaults to taking anything",
   db.runner_role(db.get_runner(a)) == "both",
   "-- a machine that has never been given a role")
ok("remembers what it was set to",
   db.runner_role(db.get_runner(b)) == "serving")

fleet = Fleet()
# `connections` is what "reachable" means to the dispatcher; the sockets
# themselves are irrelevant to which machine it picks.
fleet.connections = {a: object(), b: object()}

idle = [rid for rid in fleet.connections if rid not in fleet.busy
        and db.runner_role(db.get_runner(rid)) != "serving"]
ok("is not offered training work", idle == [a],
   "-- only %s is eligible" % ", ".join(idle))

# The other half of the same fact, which is the half that drifted before.
job = db.create_job("a fine-tune", "finetune_llm",
                    {"base_model": "HuggingFaceTB/SmolLM2-135M", "params_b": 0.135})
job = db.get_job(job)
waiting = {w["runner_id"]: w for w in fleet.why_waiting(job)}
ok("and says so when asked why a run is waiting",
   waiting[b]["can"] is False and "reserved for serving" in waiting[b]["reason"],
   "-- %r" % waiting[b]["reason"])
ok("while the general-purpose machine still reads as able",
   waiting[a]["can"] is True)

# A machine reserved for training is the mirror image: it takes runs and is
# never picked to answer a message. That rule lives in app._serves_models,
# which needs the whole application, so the column it reads is checked here.
c = runner("run_c", "the trainer", "training")
ok("and a machine reserved for training records that too",
   db.runner_role(db.get_runner(c)) == "training")

# ------------------------------------------------------------- deployments
print("\nHolding a model on a card")

d1 = db.create_deployment(job["id"], a, owner_id="usr_1")
ok("starts out waiting for the machine", d1["state"] == "pending")

d2 = db.create_deployment(job["id"], a, owner_id="usr_1")
ok("asking twice is the same deployment", d1["id"] == d2["id"],
   "-- not two rows for one card")
ok("and there is exactly one of them",
   len(db.list_deployments(a)) == 1)

db.set_deployment_state(d1["id"], "loading", "Fetching and loading the model.")
db.set_deployment_state(d1["id"], "ready", "On the card, and held there.",
                        load_s=74.0)
row = db.get_deployment(d1["id"])
ok("becomes ready once the machine says so", row["state"] == "ready")
ok("and keeps how long that took", row["load_s"] == 74.0,
   "-- %ss, which is the number that justifies deploying at all" % row["load_s"])
ok("and when it became ready", bool(row["ready_at"]))

db.set_deployment_state(d1["id"], "failed", "The GPU ran out of memory.")
again = db.create_deployment(job["id"], a)
ok("a failed one is retried by asking again",
   again["state"] == "pending" and again["id"] == d1["id"],
   "-- rather than refusing about something visibly not deployed")
ok("and the old reason is cleared with it", not again["detail"])

db.set_deployment_state(d1["id"], "ready", "On the card.")
ok("deployments are found by the run behind them",
   [r["id"] for r in db.deployments_of_job(job["id"])] == [d1["id"]])
ok("taking it down removes it", db.delete_deployment(d1["id"])
   and not db.list_deployments(a))

# A deployment that kills the machine it is sent to, which is the one failure
# the state machine could not see: the runner dies mid-load, so it never
# reports "failed", so the deployment stays pending and is sent again the
# moment the machine reconnects. One model that faulted the GPU became
# twenty-four restarts of a production runner before this existed.
print("")
print("A model the machine does not survive loading")

import asyncio  # noqa: E402

fleet2 = Fleet()
fleet2.attach(a, object())
silent = db.create_deployment(job["id"], a, owner_id="usr_1")
sent = []

async def _never_answers():
    """Drive the reconciler the way a crash loop does: the preload is sent,
    the machine dies without a word, and it reconnects clean."""
    async def fake_send(row):
        sent.append(row["id"])
    fleet2.send_deployment = fake_send
    for _ in range(6):
        fleet2.deploying.clear()       # what a restart looks like from here
        await fleet2.reconcile_deployments()

asyncio.run(_never_answers())
ok("it is sent a few times, not endlessly",
   len(sent) == Fleet.PRELOAD_TRIES,
   "-- %d attempts, then it stops" % len(sent))
ok("and is written off rather than left pending",
   db.get_deployment(silent["id"])["state"] == "failed")
ok("saying what actually happened",
   "did not survive" in (db.get_deployment(silent["id"])["detail"] or ""))
db.delete_deployment(silent["id"])

bad = False
try:
    db.set_deployment_state("dep_nope", "sideways")
except ValueError:
    bad = True
ok("and a state nobody defined is refused", bad,
   "-- a typo must not become a state the page then has to render")

# ------------------------------------------------------------------ ledger
print("\nWhat the numbers are made of")

db.record_usage(job["id"], 100, 50, user_id="usr_1", seconds=2.0,
                source="api", runner_id=a)
db.record_usage(job["id"], 300, 150, user_id="usr_1", seconds=3.0,
                source="playground", runner_id=a)
db.record_usage(job["id"], 0, 0, user_id="usr_1", seconds=0.4,
                status="error", error="No machine was free.",
                source="api", runner_id=a)

totals = db.usage_totals("usr_1")
ok("a failure is a row like any other", totals["calls"] == 3,
   "-- three calls, not the two that worked")
ok("counted as a failure", totals["errors"] == 1)
ok("carrying no tokens", totals["completion_tokens"] == 200,
   "-- 50 + 150, and nothing from the one that failed")
ok("and no seconds", totals["seconds"] == 5.0,
   "-- the 0.4s spent failing is not time spent generating, and counting it "
   "would report the fleet as slower than it is")

rate = totals["completion_tokens"] / totals["seconds"]
ok("so tokens per second is the honest 40.0", rate == 40.0,
   "-- got %.1f" % rate)

by_runner = {r["key"]: r for r in db.usage_by("runner_id", "usr_1")}
ok("and the ledger can be grouped by machine",
   by_runner[a]["calls"] == 3 and by_runner[a]["errors"] == 1)
by_source = {r["key"]: r for r in db.usage_by("source", "usr_1")}
ok("and by which door the request came in at",
   by_source["api"]["calls"] == 2 and by_source["playground"]["calls"] == 1,
   "-- the playground is the same cards and the same seconds")

errs = db.usage_recent("usr_1", only_errors=True)
ok("and the failures themselves are readable",
   len(errs) == 1 and errs[0]["error"] == "No machine was free.",
   "-- \"it is erroring\" is where somebody starts, not where they finish")

# -------------------------------------------------------------- chat role
print("\nAn account that may only talk to the models")

allowed = ["/api/me", "/api/models", "/api/conversations",
           "/api/conversations/cnv_1", "/api/assets", "/api/assets/ast_1/file"]
for path in allowed:
    ok("may reach %s" % path, security._chat_role_allows(path))

refused = ["/api/jobs", "/api/jobs/job_1", "/api/datasets", "/api/runners",
           "/api/users", "/api/ops/deployments", "/api/ops/keys",
           "/api/storage", "/api/backups", "/api/idp",
           # The one that matters most: a path added next month that nobody
           # thought about here. An allowlist refuses it by construction.
           "/api/something-invented-later"]
for path in refused:
    ok("may not reach %s" % path, not security._chat_role_allows(path))

# A near-miss, because prefix matching is where an allowlist usually leaks.
ok("and a path that merely starts the same way is refused",
   not security._chat_role_allows("/api/models-secret")
   and not security._chat_role_allows("/api/conversations-of-everyone"),
   "-- \"/api/models\" is a path, not a prefix")

# ---------------------------------------------------------------------------
shutil.rmtree(_TMP, ignore_errors=True)
print("")
if FAILED:
    print("%d check(s) failed: %s" % (len(FAILED), ", ".join(FAILED)))
    sys.exit(1)
print("All checks passed.")
