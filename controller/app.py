"""FastAPI application: runner fleet, job queue, Hub proxy, and the web UI."""
from __future__ import annotations

import asyncio
import contextlib
import json
import time
from pathlib import Path
from typing import Any

from fastapi import (Body, FastAPI, Header, HTTPException, Query, Request,
                     UploadFile, WebSocket, WebSocketDisconnect)
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from common import apimodels, formatting

from . import architectures as arch
from . import config, datasets as dsets, db, diagnose, hfaccount, hub, serving
from .api import (accounts, data, evals, providers, security,
                  serving as serving_api, sharing, sso)
from .scheduler import Fleet

fleet = Fleet()


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    config.ensure_dirs()
    dsets.ensure_dirs()
    db.connect()
    # Any runner marked online in a previous process is stale until it dials in.
    for r in db.q("SELECT id FROM runners WHERE status != 'offline'"):
        db.mark_runner_offline(r["id"])
    task = asyncio.create_task(fleet.scheduler_loop())
    sync = asyncio.create_task(_directory_loop())
    yield
    for t in (task, sync):
        t.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await t


async def _directory_loop() -> None:
    """Keep the list of people in step with whatever directory owns it.

    Hourly, and only for providers that asked. Deliberately not on startup
    with everything else: a directory that is slow or unreachable must not be
    able to hold up a controller that has training to schedule, so the first
    pass waits a minute and every failure is recorded rather than raised.
    """
    from . import directory
    await asyncio.sleep(60)
    while True:
        try:
            for result in await directory.sync_all():
                print("[directory] %s" % result.get("note", result))
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 -- a timer must not die
            print("[directory] sync failed: %s" % e)
        await asyncio.sleep(3600)


app = FastAPI(title="AI Studio", version="0.1.0", lifespan=lifespan)

# Registered before anything else so that no route can be reached without
# passing it. Order matters here: middleware added later runs first, and the
# authentication gate must run before any handler.
app.middleware("http")(security.authenticate)

app.include_router(accounts.router)
app.include_router(data.router)
app.include_router(evals.router)
app.include_router(providers.router)
app.include_router(serving_api.router)
app.include_router(sharing.router)
app.include_router(sso.router)

# The evaluation routes queue jobs, which means they need the live fleet. Set
# here rather than imported the other way round, because the fleet is created
# in this module and an import back into it would be a cycle.
evals.FLEET = fleet
serving_api.FLEET = fleet


# ===========================================================================
# Runner websocket
# ===========================================================================

@app.websocket("/api/runner/ws")
async def runner_ws(ws: WebSocket) -> None:
    await ws.accept()
    runner_id: str | None = None
    try:
        first = json.loads(await ws.receive_text())
        if first.get("type") != "register":
            await ws.send_text(json.dumps({"type": "error", "message": "expected register"}))
            await ws.close()
            return
        if first.get("token") != config.join_token():
            await ws.send_text(json.dumps(
                {"type": "error", "message": "invalid join token"}))
            await ws.close()
            return

        runner_id = first["runner_id"]
        db.upsert_runner(runner_id, first.get("name") or runner_id,
                         first.get("capabilities") or {})
        fleet.note_checkpoints(runner_id, first.get("checkpoints") or [])
        fleet.attach(runner_id, ws)
        await ws.send_text(json.dumps({"type": "registered", "runner_id": runner_id}))
        # A runner says what it is training as it joins. A machine that has
        # just restarted is training nothing, and anything this controller
        # still has pinned to it needs to go back on the queue now rather than
        # after a timeout that a fast restart never reaches.
        #
        # `busy` absent means "did not say", which is not the same as "no" --
        # an older runner must not have its live work requeued underneath it.
        if first.get("busy") is not None:
            await fleet.reconcile_runner(
                runner_id, first.get("job_id") if first.get("busy") else None)
        await fleet.broadcast_ui({"type": "runners_changed"})
        fleet.wake()

        while True:
            msg = json.loads(await ws.receive_text())
            await fleet.handle_runner_message(runner_id, msg)

    except WebSocketDisconnect:
        pass
    except Exception:  # noqa: BLE001
        pass
    finally:
        if runner_id:
            fleet.detach(runner_id)
            db.mark_runner_offline(runner_id)
            # Deliberately NOT requeuing here. A dropped socket does not stop
            # a training run: the runner keeps going and dials back in. The
            # commonest cause of this branch is the controller itself being
            # restarted for an update, and requeuing on that would abandon
            # every run in progress and start them again from noise.
            #
            # Instead the job is left alone and the scheduler reconciles it
            # once the runner has been silent long enough to be genuinely
            # gone. See Fleet.reconcile_orphans.
            await fleet.broadcast_ui({"type": "runners_changed"})


# ===========================================================================
# Browser event stream
# ===========================================================================

@app.websocket("/api/events")
async def events_ws(ws: WebSocket) -> None:
    """Live updates for a signed-in browser.

    Middleware does not run for websockets, so the cookie is checked here by
    hand. Without this the event stream would be the one door in the building
    with no lock on it -- and it carries job names, dataset names and progress
    for everyone in the studio.
    """
    from . import auth
    if not auth.session_user(ws.cookies.get(auth.SESSION_COOKIE)):
        await ws.close(code=4401)
        return
    await ws.accept()
    fleet.ui_clients.add(ws)
    try:
        while True:
            await ws.receive_text()
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        fleet.ui_clients.discard(ws)


# ===========================================================================
# Status / runners
# ===========================================================================

@app.get("/api/health")
async def health() -> dict:
    """Is the controller up? Nothing more, and deliberately public.

    This exists because /api/status stopped being an honest health check the
    day accounts landed: it answers 401 to anyone without a session, which is
    correct -- it carries the join token and the fleet's shape -- but it meant
    the container healthcheck had reported "unhealthy" ever since, for a
    controller that was working perfectly. A liveness probe must not need a
    credential, so it must not carry anything worth protecting.
    """
    return {"ok": True, "version": "0.1.0"}


@app.get("/api/fleet/in-flight")
async def fleet_in_flight() -> dict:
    """What a restart would interrupt right now.

    Authenticated with the join token rather than a session, because the
    caller is a deploy script on the host, not a browser. It is the same
    credential a runner uses and it is already in the .env file that starts
    the containers -- so the guard needs no new secret to be useful.
    """
    jobs = fleet.in_flight()
    return {
        "busy": bool(jobs),
        "jobs": jobs,
        "queued": len(db.q("SELECT id FROM jobs WHERE status='queued'")),
        # Whether interrupting would actually cost anything. With a checkpoint
        # a restart costs minutes; without one it costs the whole run.
        "resumable": all(j["checkpoint_step"] for j in jobs) if jobs else True,
    }


@app.get("/api/status")
async def status(request: Request) -> dict:
    runners = db.list_runners()
    user = security.current_user(request)
    return {
        "version": "0.1.0",
        # The join token lets a machine attach to the studio and read the work
        # on it. Members can see that machines exist; only an administrator
        # gets the credential that adds one.
        "join_token": config.join_token() if user["role"] == "admin" else None,
        "hf_token_set": bool(hfaccount.token_for(user)),
        "hf_token_is_yours": bool(user.get("hf_token_enc")),
        "runners_online": sum(1 for r in runners if r["status"] != "offline"),
        "runners_total": len(runners),
        "jobs_running": len(db.q("SELECT id FROM jobs WHERE status='running'")),
        "jobs_queued": len(db.q("SELECT id FROM jobs WHERE status='queued'")),
    }


@app.get("/api/runners")
async def get_runners() -> list[dict]:
    out = []
    for r in db.list_runners():
        r["connected"] = r["id"] in fleet.connections
        r["current_job"] = fleet.busy.get(r["id"])
        r["checkpoints"] = sorted(fleet.checkpoints.get(r["id"], set()))
        out.append(r)
    return out


@app.post("/api/runners/{runner_id}/reprobe")
async def reprobe(runner_id: str) -> dict:
    if runner_id not in fleet.connections:
        raise HTTPException(404, "That runner is not connected right now.")
    await fleet.send_to_runner(runner_id, {"type": "reprobe"})
    return {"ok": True}


# ===========================================================================
# Jobs
# ===========================================================================

@app.get("/api/jobs")
async def get_jobs(request: Request, limit: int = 100) -> list[dict]:
    jobs = db.visible_jobs(security.current_user(request), limit)
    positions = fleet.queue_positions()
    waiting = len(positions)
    for j in jobs:
        if j["status"] == "queued":
            j["queue_position"] = positions.get(j["id"])
            j["queue_length"] = waiting
    return jobs


def _job_or_404(request: Request, job_id: str, need: str = "view") -> dict:
    """Fetch a run and check the caller may have it.

    Every route below goes through this. Anything that reaches a job by id
    without it is a bug, and the reason it is a helper rather than a repeated
    two lines is that the repeated two lines are the ones people forget.
    """
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(404, "No such run.")
    if need == "own":
        security.require_owner(request, "job", job)
    elif need == "edit":
        security.require_edit(request, "job", job)
    else:
        security.require_view(request, "job", job)
    return job


def _check_generation_source(request: Request, cfg: dict) -> None:
    """You may only generate data with a model you are allowed to use."""
    if source := (cfg.get("model") or {}).get("job_id"):
        src = db.get_job(source)
        if not src:
            raise HTTPException(404, "That model does not exist.")
        security.require_view(request, "job", src)


def _attach_provider(user: dict | None, cfg: dict) -> None:
    """Resolve a hosted-model choice into a usable connection.

    The key is attached here, at creation, exactly as the Hugging Face token
    is: the runner cannot ask for it later, and the run belongs to the person
    whose key paid for it. It is stripped again from every response that hands
    a job back to a browser -- see `_public_job`.
    """
    model = cfg.get("model") or {}
    conn = providers.connection(user, model.get("provider"))
    if not conn:
        raise HTTPException(
            400, "That provider is not connected to your account. Connect it "
                 "on your account page first.")
    if problem := apimodels.problems(conn):
        raise HTTPException(400, problem)
    name = apimodels.model_name(conn, model.get("model"))
    if not name:
        raise HTTPException(400, "Which model at %s should write it?"
                            % apimodels.describe(conn))
    model["model"] = name
    model["connection"] = conn
    model["label"] = "%s · %s" % (apimodels.describe(conn), name)
    cfg["model"] = model
    # It reaches the model over the network, so any machine will do -- a
    # runner with no GPU at all is a perfectly good place to run it from.
    cfg.setdefault("allow_cpu", True)


@app.post("/api/jobs")
async def create_job(request: Request, payload: dict = Body(...)) -> dict:
    return {"id": await _create_job(request, payload)}


async def _create_job(request: Request, payload: dict) -> str:
    """Validate and queue one run. The single door every run comes through.

    Extracted so that sweeps create their variants through exactly the same
    checks. A second, looser path would be a second set of rules about what a
    valid run is, and the looser one always wins by accident.
    """
    user = security.current_user(request)
    kind = payload.get("kind", "finetune_llm")
    cfg = payload.get("config") or {}
    # A dataset from the studio's own library travels as a URL the runner can
    # fetch with its join token, so a private dataset never has to be public
    # to be trained on.
    if studio_id := cfg.get("studio_dataset"):
        d = db.get_dataset(studio_id)
        if not d:
            raise HTTPException(404, "No such dataset.")
        security.require_view(request, "dataset", d)
        cfg["dataset"] = "%s/api/datasets/%s/dataset-file" % (
            str(request.base_url).rstrip("/"), studio_id)
        cfg["dataset_is_local"] = True
        cfg["dataset_label"] = d["name"]

    # Training on top of something this studio already built. The permission
    # check is the point: without it, any run id pasted into this field would
    # hand out the weights of a model you are not allowed to see.
    if source_id := (cfg.get("continue_from") or cfg.get("base_model_job")
                     or cfg.get("source_job")):
        src = _job_or_404(request, source_id)
        if not (config.ARTIFACT_DIR / ("%s.zip" % source_id)).exists():
            raise HTTPException(400, "That run has no saved model to build on.")
        if src["status"] not in ("succeeded", "cancelled"):
            raise HTTPException(400, "That run has not finished yet.")
        if kind == "pretrain_llm" and src["kind"] != "pretrain_llm":
            raise HTTPException(
                400, "Only a model trained from scratch can be trained further "
                     "this way. A fine-tune produces an adapter, which is "
                     "carried on with a fine-tuning run instead.")
        if src["kind"] == "finetune_llm":
            # An adapter needs the model it was built for. Carried across so
            # the run does not depend on the user retyping it correctly.
            cfg.setdefault("base_model", src["config"].get("base_model"))
        cfg["source_run_name"] = src["name"]

    if kind == "finetune_llm":
        if not cfg.get("base_model") and not cfg.get("base_model_job"):
            raise HTTPException(400, "Missing required setting: base_model")
        if not cfg.get("dataset"):
            raise HTTPException(400, "Missing required setting: dataset")
    elif kind == "pretrain_llm":
        if not cfg.get("dataset"):
            raise HTTPException(400, "Choose some text to learn from.")
        if not cfg.get("arch"):
            raise HTTPException(400, "No model architecture was chosen.")
    elif kind == "merge_adapter":
        src = _job_or_404(request, cfg.get("source_job") or "")
        if src["kind"] != "finetune_llm":
            raise HTTPException(
                400, "Only a fine-tune produces an adapter to merge. A model "
                     "built from scratch is already standalone.")
        if not (config.ARTIFACT_DIR / ("%s.zip" % src["id"])).exists():
            raise HTTPException(400, "That run has no saved adapter.")
        # Carried across so the merge does not depend on the adapter file
        # recording its own base, which older adapters may not.
        cfg.setdefault("base_model", src["config"].get("base_model"))
        cfg.setdefault("base_model_job", src["config"].get("base_model_job"))
        # The merged model answers in the shape the fine-tune was trained in,
        # so the format travels with it or the playground would guess again.
        cfg.setdefault("format", src["config"].get("format"))
        cfg.setdefault("system_prompt", src["config"].get("system_prompt"))
    elif kind == "generate_dataset":
        model = cfg.get("model") or {}
        if not model.get("job_id") and not model.get("base_model") \
                and not model.get("provider"):
            raise HTTPException(400, "Choose a model to write the data with.")
        if not int(cfg.get("count") or 0):
            raise HTTPException(400, "How many rows should it write?")
        if model.get("provider"):
            _attach_provider(user, cfg)
        else:
            _check_generation_source(request, cfg)
    else:
        raise HTTPException(400, "Unknown kind of training run: %s" % kind)

    # Reject work that provably cannot run, at creation time. The scheduler
    # would otherwise skip it silently and the job would sit "queued" forever
    # with nothing telling the user why.
    if pinned := cfg.get("required_runner"):
        runner = db.get_runner(pinned)
        if not runner:
            raise HTTPException(400, "That machine is not known to this studio.")
        caps = dict(runner["capabilities"])
        caps["_id"] = pinned
        ok, why = fleet.can_run({"config": cfg, "kind": kind}, caps)
        if not ok:
            raise HTTPException(400, "This will not run on '%s': %s" % (runner["name"], why))

    name = payload.get("name") or _default_job_name(cfg, kind)
    # The creator's own Hugging Face token, falling back to the studio's. A
    # run downloads gated models as the person who started it, not as a shared
    # identity nobody can attribute.
    if token := hfaccount.token_for(user):
        cfg.setdefault("hf_token", token)
    jid = db.create_job(name, kind, cfg, owner_id=user["id"])
    db.add_log(jid, "Job created and queued.")
    await fleet.broadcast_ui({"type": "jobs_changed"})
    fleet.wake()
    return jid


# How many variants one sweep may launch. Each is a whole training run on the
# same card, so eight is already most of a night; the limit exists to make
# that obvious before the queue does.
MAX_SWEEP_RUNS = 8


def _label(key: str, value) -> str:
    """A short, readable name for one varied setting."""
    short = {"learning_rate": "lr", "lora_r": "rank", "batch_size": "batch",
             "grad_accum": "accum", "epochs": "epochs",
             "weight_decay": "decay", "lora_alpha": "alpha",
             "max_steps": "steps", "warmup_steps": "warmup"}.get(key, key)
    # One format for every value in a sweep. Switching to scientific notation
    # below a threshold made a single sweep read "lr 0.001, lr 6.0e-04,
    # lr 3.0e-04" -- three notations for three neighbouring numbers, in the
    # one place where they are meant to be compared at a glance.
    text = ("%.6g" % value) if isinstance(value, float) else str(value)
    return "%s %s" % (short, text)


@app.post("/api/sweeps")
async def create_sweep(request: Request, payload: dict = Body(...)) -> dict:
    """Launch several variants of one run, differing in named settings.

    The reason to do this in the app rather than by hand is that the hard part
    was never launching the runs -- it was comparing them afterwards, and
    keeping straight which was which a week later. Every variant carries the
    sweep it belongs to and the exact values that make it different, so the
    comparison is a fact about the runs rather than something reconstructed
    from their names.

    Runs are created together and then compete for the card like anything
    else. The queue is dealt round-robin between people, so a sweep of eight
    does not lock a colleague out -- it simply takes its turns.
    """
    base = payload.get("base") or {}
    vary = payload.get("vary") or {}
    if not isinstance(vary, dict) or not vary:
        raise HTTPException(400, "Name at least one setting to vary.")

    combos: list[dict] = [{}]
    for key, values in vary.items():
        if not isinstance(values, list) or not values:
            raise HTTPException(400, "'%s' needs a list of values to try." % key)
        combos = [{**c, key: v} for c in combos for v in values]

    if len(combos) > MAX_SWEEP_RUNS:
        raise HTTPException(
            400, "That is %d runs. Each one is a full training run on the same "
                 "card, so a sweep is limited to %d -- vary one setting at a "
                 "time, or try fewer values."
                 % (len(combos), MAX_SWEEP_RUNS))

    sweep_id = db.new_id("swp")
    sweep_name = (payload.get("name") or "").strip()         or ("Trying %s" % " and ".join(vary))
    base_name = (base.get("name") or "").strip()

    created = []
    for combo in combos:
        cfg = {**(base.get("config") or {}), **combo,
               "sweep_id": sweep_id, "sweep_name": sweep_name,
               "sweep_values": combo}
        label = ", ".join(_label(k, v) for k, v in combo.items())
        name = "%s · %s" % (base_name or sweep_name, label)
        try:
            created.append(await _create_job(request, {
                "kind": base.get("kind", "finetune_llm"),
                "name": name, "config": cfg}))
        except HTTPException as e:
            # One variant that cannot run means the settings being varied are
            # not all viable. Undo the rest rather than leaving half a sweep
            # in the queue: a comparison missing three of its five runs looks
            # exactly like a complete one.
            for jid in created:
                db.delete_job(jid)
            raise HTTPException(
                e.status_code,
                "The variant with %s cannot run: %s" % (label, e.detail)) from e

    return {"sweep_id": sweep_id, "name": sweep_name, "jobs": created}


def _sweep_rows(user: dict) -> dict[str, dict]:
    """Sweeps assembled from the runs in them, rather than from a table.

    There is no sweeps table on purpose. A sweep is exactly its runs: it
    inherits their visibility with no second set of sharing rules, and when
    the last run is deleted the sweep stops existing, which is the right
    answer rather than a dangling row.
    """
    out: dict[str, dict] = {}
    for job in db.visible_jobs(user, 500):
        sid = (job.get("config") or {}).get("sweep_id")
        if not sid:
            continue
        entry = out.setdefault(sid, {
            "id": sid, "name": job["config"].get("sweep_name") or sid,
            "created_at": job["created_at"], "runs": []})
        entry["created_at"] = min(entry["created_at"], job["created_at"])
        entry["runs"].append({
            "id": job["id"], "name": job["name"], "status": job["status"],
            "step": job["step"], "total_steps": job["total_steps"],
            "values": job["config"].get("sweep_values") or {},
            "has_model": job.get("has_model"),
            "summary": job.get("summary"),
            "finished_at": job.get("finished_at"),
        })
    return out


@app.get("/api/sweeps")
async def list_sweeps(request: Request) -> list[dict]:
    rows = list(_sweep_rows(security.current_user(request)).values())
    for r in rows:
        r["runs"].sort(key=lambda x: x["name"])
    return sorted(rows, key=lambda r: r["created_at"], reverse=True)


@app.get("/api/sweeps/{sweep_id}")
async def get_sweep(request: Request, sweep_id: str) -> dict:
    row = _sweep_rows(security.current_user(request)).get(sweep_id)
    if not row:
        raise HTTPException(404, "No such sweep.")
    row["runs"].sort(key=lambda x: x["name"])
    # Which setting actually differs, so the table can put it in a column
    # rather than leaving the reader to diff the names.
    keys: set = set()
    for r in row["runs"]:
        keys.update(r["values"])
    row["varied"] = sorted(keys)
    done = [r for r in row["runs"]
            if r["status"] in ("succeeded", "cancelled", "failed")]
    row["finished"] = len(done)
    scored = [r for r in done if (r.get("summary") or {}).get("best_val_loss")]
    row["best"] = min(scored,
                      key=lambda r: r["summary"]["best_val_loss"])["id"]         if scored else None
    return row


def _default_job_name(cfg: dict, kind: str = "finetune_llm") -> str:
    data = str(cfg.get("dataset", "data")).split("/")[-1]
    if kind == "merge_adapter":
        return "%s, merged" % (cfg.get("source_run_name") or "Fine-tune")
    if kind == "pretrain_llm":
        a = cfg.get("arch") or {}
        label = (arch.preset(a.get("size_id", "")) or {}).get("label", "Model")
        return "%s from scratch on %s" % (label, data)
    model = str(cfg.get("base_model", "model")).split("/")[-1]
    return "%s on %s" % (model, data)


@app.get("/api/jobs/{job_id}")
async def get_job(request: Request, job_id: str) -> dict:
    job = _job_or_404(request, job_id)
    user = security.current_user(request)
    job["access"] = db.access_level("job", job_id, job.get("owner_id"), user)
    job["mine"] = job.get("owner_id") == user["id"]
    owner = db.get_user(job["owner_id"]) if job.get("owner_id") else None
    job["owner"] = db.public_user(owner) if owner else None
    job["shares"] = db.list_shares("job", job_id)
    job["artifacts"] = db.list_artifacts(job_id)
    job["runner"] = db.get_runner(job["runner_id"]) if job["runner_id"] else None
    if job["status"] == "queued":
        positions = fleet.queue_positions()
        job["queue_position"] = positions.get(job_id)
        job["queue_length"] = len(positions)
    job["resumable"] = _resumable(job)
    # Never hand a credential back to the browser: not the runner's copy of
    # the Hugging Face token, and not the API key a hosted model was written
    # with. Both went in at creation and only the runner needs them.
    job["config"].pop("hf_token", None)
    if isinstance(job["config"].get("model"), dict):
        job["config"]["model"].pop("connection", None)
    return job


def _resumable(job: dict) -> dict | None:
    """Whether this run could be picked up where it stopped, and on what.

    A checkpoint is only useful while the machine holding it still exists, so
    the answer includes which machine that is and whether it is reachable --
    "resume" that silently starts from zero would be worse than no button.
    """
    step = int(job.get("checkpoint_step") or 0)
    if not step or job["status"] not in ("failed", "cancelled"):
        return None
    holder = job.get("checkpoint_runner")
    runner = db.get_runner(holder) if holder else None
    return {
        "step": step,
        "total": job.get("total_steps") or 0,
        "runner_id": holder,
        "runner": (runner or {}).get("name"),
        "online": holder in fleet.connections,
    }


@app.post("/api/jobs/{job_id}/resume")
async def resume_job(request: Request, job_id: str) -> dict:
    """Put a stopped or failed run back on the queue, to carry on from its
    checkpoint rather than from the beginning."""
    job = _job_or_404(request, job_id, "edit")
    info = _resumable(job)
    if not info:
        raise HTTPException(
            400, "There is no checkpoint for this run, so starting it again "
                 "would start it from the beginning. Create a new run instead.")
    if not info["online"]:
        raise HTTPException(
            400, "The machine holding this run's progress (%s) is not "
                 "connected. Bring it back and try again."
                 % (info["runner"] or "unknown"))
    db.ex("UPDATE jobs SET status='queued', runner_id=NULL, error=NULL,"
          " finished_at=NULL, step=? WHERE id=?", (info["step"], job_id))
    fleet.declined.discard(job_id)
    fleet.gave_up_waiting.discard(job_id)
    db.add_log(job_id, "Queued again, to carry on from the checkpoint at step "
               "%d on %s." % (info["step"], info["runner"] or "its machine"))
    await fleet.broadcast_ui({"type": "jobs_changed", "job_id": job_id})
    fleet.wake()
    return {"ok": True, "from_step": info["step"]}


@app.get("/api/jobs/{job_id}/metrics")
async def job_metrics(request: Request, job_id: str) -> list[dict]:
    _job_or_404(request, job_id)
    return db.get_metrics(job_id)


@app.get("/api/jobs/{job_id}/report")
async def job_report(request: Request, job_id: str) -> dict:
    """What this run says about itself, in words rather than in a curve.

    Computed on demand rather than stored, so improving the analysis improves
    every past run rather than only the ones trained after the change.
    """
    job = _job_or_404(request, job_id)
    job["runner"] = db.get_runner(job["runner_id"]) if job["runner_id"] else None
    job["artifacts"] = db.list_artifacts(job_id)
    return diagnose.report(job, db.get_metrics(job_id))


@app.get("/api/jobs/{job_id}/logs")
async def job_logs(request: Request, job_id: str, limit: int = 500) -> list[dict]:
    _job_or_404(request, job_id)
    return db.get_logs(job_id, limit)


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(request: Request, job_id: str,
                     payload: dict = Body(default=None)) -> dict:
    """Stop a run, optionally keeping the model it has built so far.

    `save` defaults to true. Stopping is reversible in the sense that the run
    can be started again; deleting hours of GPU time is not, so the default is
    the one that cannot lose anything. Discarding has to be asked for.
    """
    job = _job_or_404(request, job_id, "edit")
    if job["status"] in ("succeeded", "failed", "cancelled"):
        return {"ok": True, "already": job["status"]}

    save = bool((payload or {}).get("save", True))
    if job["runner_id"] and job["runner_id"] in fleet.connections:
        await fleet.send_to_runner(job["runner_id"],
                                   {"type": "job_cancel", "job_id": job_id,
                                    "save": save})
    else:
        db.set_job_status(job_id, "cancelled")
    db.add_log(job_id, "Stopping, and keeping the model trained so far."
               if save else "Stopping, and discarding the partly trained model.",
               "warn")
    await fleet.broadcast_ui({"type": "jobs_changed"})
    return {"ok": True, "saving": save}


@app.delete("/api/jobs/{job_id}")
async def delete_job(request: Request, job_id: str) -> dict:
    """Delete a run, its metrics, its logs and its model file."""
    job = _job_or_404(request, job_id, "own")
    db.clear_shares("job", job_id)
    if job["status"] in ("queued", "assigned", "running"):
        # Deleting the row of a job a runner is still working on would leave
        # the runner training something that no longer exists, and its next
        # message would arrive for an unknown job. Stopping it first is one
        # click, and it is the honest order of operations.
        raise HTTPException(
            400, "This run has not finished. Stop it first, then delete it.")

    for filename in db.delete_job(job_id):
        with contextlib.suppress(OSError):
            (config.ARTIFACT_DIR / filename).unlink()

    # Runners cache a copy of any model they have served. Ask them to drop it,
    # so deleting a run actually reclaims the space rather than only hiding it.
    for runner_id in list(fleet.connections):
        await fleet.send_to_runner(runner_id, {"type": "purge_model",
                                               "job_id": job_id})
    fleet.declined.discard(job_id)
    await fleet.broadcast_ui({"type": "jobs_changed"})
    return {"ok": True, "deleted": job_id}


@app.post("/api/jobs/{job_id}/artifact")
async def upload_artifact(job_id: str, file: UploadFile,
                          x_runner_token: str = Header(default="")) -> dict:
    if x_runner_token != config.join_token():
        raise HTTPException(401, "Invalid runner token.")
    if not db.get_job(job_id):
        raise HTTPException(404, "No such job.")
    config.ensure_dirs()
    dest = config.ARTIFACT_DIR / ("%s.zip" % job_id)
    size = 0
    with open(dest, "wb") as fh:
        while chunk := await file.read(1 << 20):
            fh.write(chunk)
            size += len(chunk)
    # A fine-tune produces an adapter that needs its base model; a from-scratch
    # run produces a standalone model; a generation run produces data. Labelling
    # them apart matters because what you do with each is completely different.
    job = db.get_job(job_id)
    kind = {"pretrain_llm": "model", "generate_dataset": "dataset",
            "merge_adapter": "model"}.get((job or {}).get("kind"), "adapter")
    db.add_artifact(job_id, kind, dest.name, size)

    if kind == "dataset" and job:
        # Rows written by a model are only useful once they are a dataset you
        # can look at, clean and train on. Doing that here, rather than making
        # the user download a zip and upload it again, is the whole point of
        # generation being part of the studio.
        try:
            created = _register_generated(job, dest)
            db.add_log(job_id, "Saved as the dataset \"%s\" (%s rows). It is "
                       "yours, and private until you share it."
                       % (created["name"], f"{created['rows']:,}"))
        except Exception as e:  # noqa: BLE001 - the rows are safe in the zip
            db.add_log(job_id, "The rows were generated but could not be saved "
                       "as a dataset (%s). The zip on this run still has them."
                       % e, "error")
    return {"ok": True, "size": size}


def _register_generated(job: dict, archive: Path) -> dict:
    """Unpack a generation run's JSONL and enter it in the dataset library."""
    import tempfile
    import zipfile

    cfg = job.get("config") or {}
    with tempfile.TemporaryDirectory(prefix="aistudio_gen_") as tmp:
        with zipfile.ZipFile(archive) as z:
            for member in z.namelist():
                if member.endswith(".jsonl") and "/" not in member \
                        and not member.startswith(".."):
                    z.extract(member, tmp)
                    src = Path(tmp) / member
                    break
            else:
                raise ValueError("no rows in the archive")
        rows = list(dsets.rows_from_upload(src.name, src.read_bytes()))

    return dsets.register(
        job.get("owner_id"),
        cfg.get("dataset_name") or ("Generated by %s" % job["name"]),
        "generated", iter(rows), origin=job["id"],
        notes="Written by a model on %s. Read a sample before training on it: "
              "generated data can be fluent and wrong at the same time."
              % time.strftime("%d %b %Y"),
        recipe={"steps": ["Generated by run %s (%s mode)"
                          % (job["id"], cfg.get("mode") or "?")]})


@app.post("/api/jobs/{job_id}/publish")
async def publish_job(request: Request, job_id: str,
                      payload: dict = Body(...)) -> dict:
    """Push a finished model to the publisher's own Hugging Face account.

    Anyone who can see the run may publish it, and it goes to *their* account
    using *their* token -- so what appears on the Hub is attributed to the
    person who put it there, which is the only attribution that is true.
    """
    job = _job_or_404(request, job_id)
    user = security.current_user(request)
    if not (config.ARTIFACT_DIR / ("%s.zip" % job_id)).exists():
        raise HTTPException(400, "This run has no saved model to publish.")
    try:
        return await hfaccount.publish_job(
            user, job, (payload.get("repo_id") or "").strip(),
            bool(payload.get("private", True)), payload.get("message") or "")
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:  # noqa: BLE001 - hub failures are not ours to classify
        raise HTTPException(502, "Hugging Face refused the upload: %s" % e) from e


@app.get("/api/jobs/{job_id}/chat-template")
async def job_chat_template(request: Request, job_id: str) -> dict:
    """The chat template baked into this run's own tokenizer, if it has one.

    Read straight out of the saved zip rather than kept on the job, because
    the tokenizer is the authoritative copy -- the same rule the runner and
    the playground already follow. It exists so that fine-tuning a model this
    studio built can preview the training text in the shape that model was
    actually taught, instead of falling back to a generic rendering and
    showing the user something the run will not produce.
    """
    _job_or_404(request, job_id)
    path = config.ARTIFACT_DIR / ("%s.zip" % job_id)
    if not path.exists():
        return {"available": False, "reason": "This run has no saved model."}
    import zipfile
    try:
        with zipfile.ZipFile(path) as z:
            names = {n.rsplit("/", 1)[-1]: n for n in z.namelist()}
            if "tokenizer_config.json" not in names:
                return {"available": False, "reason": "no tokenizer in the result"}
            conf = json.loads(z.read(names["tokenizer_config.json"]))
    except (OSError, ValueError, zipfile.BadZipFile) as e:
        return {"available": False, "reason": str(e)[:200]}
    template = conf.get("chat_template")
    if isinstance(template, dict):
        template = template.get("default") or next(iter(template.values()), None)
    if isinstance(template, list):        # transformers 5 ships a list of dicts
        template = next((t.get("template") for t in template
                         if isinstance(t, dict)), None)
    return {"available": bool(template), "chat_template": template,
            "eos_token": conf.get("eos_token")}


@app.get("/api/jobs/{job_id}/download")
async def download_artifact(request: Request, job_id: str):
    path = config.ARTIFACT_DIR / ("%s.zip" % job_id)
    if not path.exists():
        raise HTTPException(404, "No result file for this job yet.")
    # A runner fetching a model to serve presents the join token; a person
    # downloading one presents a session and has to be allowed the run.
    job = _job_or_404(request, job_id) \
        if not getattr(request.state, "runner", False) else db.get_job(job_id)
    safe = "".join(c for c in (job["name"] if job else job_id)
                   if c.isalnum() or c in "-_ ").strip().replace(" ", "-")
    suffix = "model" if (job or {}).get("kind") in ("pretrain_llm",
                                                    "merge_adapter") else "adapter"
    return FileResponse(path, media_type="application/zip",
                        filename="%s-%s.zip" % (safe or job_id, suffix))


# ===========================================================================
# Hugging Face proxy
# ===========================================================================

@app.get("/api/hub/starters")
async def starters() -> dict:
    return {"models": hub.STARTER_MODELS, "datasets": hub.STARTER_DATASETS,
            "corpora": hub.STARTER_CORPORA, "vocab_presets": arch.VOCAB_PRESETS,
            "default_vocab": arch.DEFAULT_VOCAB}


@app.get("/api/hub/models")
async def hub_models(q: str = "", limit: int = 30,
                     task: str = "text-generation") -> list[dict]:
    try:
        return await hub.search_models(q, limit, task)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, "Could not reach Hugging Face: %s" % e)


@app.get("/api/hub/datasets")
async def hub_datasets(q: str = "", limit: int = 30) -> list[dict]:
    try:
        return await hub.search_datasets(q, limit)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, "Could not reach Hugging Face: %s" % e)


@app.get("/api/hub/model")
async def hub_model(id: str = Query(...)) -> dict:
    try:
        detail = await hub.model_detail(id)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(404, "Could not find that model: %s" % e)
    detail["fit"] = {
        r["id"]: hub.fit_report(detail.get("params_b"), r["capabilities"])
        for r in db.list_runners() if r["status"] != "offline"
    }
    return detail


@app.get("/api/hub/dataset-preview")
async def hub_dataset_preview(id: str = Query(...), config_name: str | None = None,
                              split: str = "train") -> dict:
    try:
        return await hub.dataset_preview(id, config_name, split)
    except Exception as e:  # noqa: BLE001
        return {"available": False, "reason": str(e), "dataset": id}


@app.get("/api/hub/dataset-configs")
async def hub_dataset_configs(id: str = Query(...)) -> dict:
    """Which configurations a dataset offers, so one can be chosen up front."""
    try:
        return await hub.dataset_configs(id)
    except Exception as e:  # noqa: BLE001
        return {"available": False, "reason": str(e)[:200], "configs": []}


@app.get("/api/hub/model-template")
async def hub_model_template(id: str = Query(...)) -> dict:
    """The chat template this model was trained to expect."""
    try:
        return await hub.model_chat_template(id)
    except Exception as e:  # noqa: BLE001
        return {"available": False, "reason": str(e)[:200], "model": id}


@app.get("/api/chat-formats")
async def chat_formats() -> dict:
    """Message-boundary formats a from-scratch run can be taught."""
    from common import chat_formats as cf
    return {"formats": cf.public_formats(), "default": cf.DEFAULT_FORMAT}


@app.get("/api/selectors")
async def selector_fields() -> dict:
    """The field mapping a dataset can be read through."""
    from common import formatting as fmt
    return {
        "fields": [{"id": k, "hint": h} for k, h in fmt.SELECTOR_FIELDS],
        "role_map_key": fmt.ROLE_MAP_KEY,
        "roles": list(fmt.KNOWN_ROLES),
    }


@app.get("/api/hub/builtin-template")
async def hub_builtin_template() -> dict:
    """The fallback conversation template, as a starting point to edit."""
    return {"template": hub.formatting.BUILTIN_CHAT_TEMPLATE,
            "instruction_template": hub.formatting.DEFAULT_INSTRUCTION_TEMPLATE}


@app.post("/api/hub/training-preview")
async def hub_training_preview(request: Request,
                               payload: dict = Body(...)) -> dict:
    """The exact text the model will be trained on.

    Rendered by the same function the runner uses to build its batches, so
    what is shown here is what the model reads -- not an approximation of it.

    A dataset from this studio's own library is read off the disk here rather
    than fetched from the Hub, which is the difference between the wizard
    showing your own rows and showing "no preview available" for every dataset
    you brought in yourself.
    """
    rows_source = None
    if studio_id := payload.get("studio_dataset"):
        d = db.get_dataset(studio_id)
        if not d:
            return {"available": False, "reason": "That dataset is gone."}
        security.require_view(request, "dataset", d)
        split = (payload.get("split") or "").strip() or None
        rows = list(dsets.iter_rows(studio_id, 40, split))
        rows_source = {
            "available": bool(rows),
            "reason": None if rows else "That split has no rows in it.",
            "rows": rows,
            "columns": d.get("columns") or sorted({k for r in rows for k in r}),
            "split": split or "all",
            "detected_format": d.get("format")
            or formatting.detect_format(d.get("columns") or [], rows),
        }
    try:
        return await hub.training_preview(
            payload.get("dataset") or "", payload.get("config") or None,
            payload.get("split") or "train", payload.get("format"),
            payload.get("text_field"), payload.get("base_model"),
            rows_source)
    except Exception as e:  # noqa: BLE001
        return {"available": False, "reason": str(e)[:300],
                "dataset": payload.get("dataset")}


@app.post("/api/plan")
async def plan(payload: dict = Body(...)) -> dict:
    """Turn a chosen model + runner into concrete, explained settings.

    This is the heart of the low-code promise: the user picks what they want to
    achieve, and the machine's measured capabilities decide the hyperparameters.
    """
    runner = db.get_runner(payload.get("runner_id", "")) if payload.get("runner_id") else None
    caps = (runner or {}).get("capabilities", {})
    params_b = payload.get("params_b")
    goal = payload.get("goal", "instructions")
    rows = int(payload.get("dataset_rows") or 1000)

    fit = hub.fit_report(params_b, caps) if params_b else None
    quant = "4bit" if (fit and fit["verdict"] == "fits_quantized") else "none"
    dtype = caps.get("recommended_dtype", "float32")
    seq = min(1024, caps.get("max_recommended_seq_len", 1024))

    vram = caps.get("vram_gb") or 0
    batch = 4 if vram >= 24 else 2 if vram >= 12 else 1
    accum = max(1, 16 // batch)
    epochs = 3 if rows < 2000 else 2 if rows < 20000 else 1
    steps = max(20, int(rows * epochs / (batch * accum)))

    lora_r = 32 if goal in ("style", "domain") else 16
    return {
        "settings": {
            "dtype": dtype, "quantization": quant, "max_seq_len": seq,
            "batch_size": batch, "grad_accum": accum, "epochs": epochs,
            "learning_rate": 2e-4, "lora_r": lora_r, "lora_alpha": lora_r * 2,
            "lora_dropout": 0.05, "gradient_checkpointing": True,
            "max_steps": min(steps, 2000),
            # On by default here because a fine-tune on a few hundred examples
            # starts memorising within a couple of passes, and the run has no
            # way to know that without being told to watch for it.
            "early_stop": True,
            "early_stop_patience": 4,
        },
        "fit": fit,
        "explanations": _explain(dtype, quant, batch, accum, epochs, caps),
        "estimated_minutes": _estimate_minutes(min(steps, 2000), params_b, caps),
    }


def _explain(dtype: str, quant: str, batch: int, accum: int,
             epochs: int, caps: dict) -> list[dict]:
    out = [
        {"setting": "Number precision", "value": dtype,
         "why": "Chosen from a speed test on this exact GPU, not from its spec sheet."},
        {"setting": "Batch size", "value": "%d x %d accumulation" % (batch, accum),
         "why": "Processes %d examples before each update. Small batches with "
                "accumulation give the same result as one large batch, but fit "
                "in less memory." % (batch * accum)},
        {"setting": "Passes over data", "value": str(epochs),
         "why": "Small datasets need more passes to learn; large ones risk "
                "memorising instead of generalising."},
        {"setting": "Stop when it stops improving", "value": "on",
         "why": "Some of your examples are held back and never trained on. "
                "When the model stops getting better at those, training ends "
                "and the best version is the one kept -- the last one is "
                "usually not the best."},
    ]
    if quant == "4bit":
        out.append({"setting": "Compression", "value": "4-bit",
                    "why": "The model is too large to fit uncompressed, so its "
                           "weights are squeezed to 4 bits. Slight quality cost."})
    if dtype == "float16" and caps.get("backend") == "rocm":
        out.append({"setting": "Why not bfloat16?",
                    "value": "float16 is faster here",
                    "why": "Most guides say bfloat16, but this GPU has no "
                           "bfloat16 hardware and computes it far more slowly."})
    return out


def _estimate_minutes(steps: int, params_b: float | None, caps: dict) -> float | None:
    tflops = (caps.get("dtypes") or {}).get(caps.get("recommended_dtype", "float16"))
    if not tflops or not params_b:
        return None
    # ~6 FLOPs per parameter per token (forward+backward), with a generous
    # efficiency factor -- real training never hits peak matmul throughput.
    tokens_per_step = 1024 * 4
    flops = 6 * params_b * 1e9 * tokens_per_step
    secs = steps * flops / (tflops * 1e12 * 0.25)
    return round(secs / 60, 1)


# ===========================================================================
# Training from scratch
# ===========================================================================

TIME_BUDGETS = [
    {"minutes": 15, "label": "15 minutes",
     "hint": "Long enough to see whether it is working."},
    {"minutes": 60, "label": "1 hour",
     "hint": "Enough for a small model to learn to write."},
    {"minutes": 240, "label": "4 hours",
     "hint": "A properly trained small model."},
    {"minutes": 720, "label": "Overnight",
     "hint": "About the most a single GPU sensibly gives."},
]


def _caps_for(runner_id: str | None) -> dict:
    runner = db.get_runner(runner_id) if runner_id else None
    if not runner:
        raise HTTPException(400, "Choose a machine first.")
    return runner["capabilities"] or {}


@app.get("/api/scratch/sizes")
async def scratch_sizes(runner_id: str = Query(...), minutes: float = 60,
                        vocab_size: int = arch.DEFAULT_VOCAB,
                        experts: int = 0, experts_per_token: int = 0) -> dict:
    """Every model size scored against this machine and this much time.

    Returned together rather than one at a time because the trade-off is the
    whole point. The user needs to see that the small model finishes and the
    large one does not *before* choosing, not eight hours afterwards.
    """
    caps = _caps_for(runner_id)
    moe = {"enabled": True, "num_local_experts": experts,
           "num_experts_per_tok": experts_per_token or arch.MOE_DEFAULT_ACTIVE} \
        if experts > 1 else None
    return {
        "sizes": arch.size_options(caps, minutes, vocab_size, moe),
        "time_budgets": TIME_BUDGETS,
        "vocab_presets": arch.VOCAB_PRESETS,
        "tokens_per_param_target": arch.TOKENS_PER_PARAM_TARGET,
        "max_scratch_params": arch.max_trainable_params(caps),
        "limits": arch.LIMITS,
        "moe_limits": arch.MOE_LIMITS,
        "moe_defaults": {"num_local_experts": arch.MOE_DEFAULT_EXPERTS,
                         "num_experts_per_tok": arch.MOE_DEFAULT_ACTIVE},
        "presets": arch.size_presets(caps, vocab_size),
    }


@app.post("/api/scratch/plan")
async def scratch_plan(payload: dict = Body(...)) -> dict:
    """Turn a size, a corpus and a time budget into a complete recipe."""
    caps = _caps_for(payload.get("runner_id"))
    size_id = payload.get("size") or "tiny"
    minutes = float(payload.get("minutes") or 60)
    vocab_size = int(payload.get("vocab_size") or arch.DEFAULT_VOCAB)

    moe = payload.get("moe") or None
    if size_id == "custom":
        architecture = arch.build_custom_arch(payload.get("custom") or {},
                                              vocab_size, moe)
    else:
        # `caps` is not optional here. The five sizes are a window onto a
        # ladder positioned by the machine's memory, so building one without
        # the machine gives a different model from the one the picker just
        # offered -- on a 16 GB card the two happened to coincide, which is
        # exactly the kind of luck that hides a bug until somebody else runs it.
        architecture = arch.build_arch(size_id, vocab_size, moe=moe, caps=caps)
    if not architecture:
        raise HTTPException(400, "Unknown model size: %s" % size_id)

    counts = arch.count_params(architecture)
    flash = bool((caps.get("attention") or {}).get("flash"))
    optim_8bit = bool((caps.get("quantization") or {}).get("optim_8bit"))

    fit = arch.pick_batch_size(architecture, caps.get("vram_gb"),
                               optim_8bit=optim_8bit, checkpointing=False,
                               flash=flash)
    checkpointing = False
    if not fit["fits"]:
        # Recomputing activations instead of storing them does the forward
        # pass twice -- about a quarter slower -- and can be the difference
        # between running and not. `tokens_per_second` is told, so the time
        # this plan quotes is the time with recomputation on.
        checkpointing = True
        fit = arch.pick_batch_size(architecture, caps.get("vram_gb"),
                                   optim_8bit=optim_8bit, checkpointing=True,
                                   flash=flash)
    # How much text the clock allows. This is a ceiling, not a plan: three
    # separate things can make the right answer smaller, and each says so.
    # `checkpointing` matters here and was ignored. The planner turns it on to
    # make a shape fit and then used to quote a speed that assumed it was off,
    # so the budget it derived was a third too large before anything else went
    # wrong.
    clock_budget = arch.tokens_in_time(
        architecture, caps, minutes, checkpointing) or 0
    budget = clock_budget
    notes: list[str] = []

    # A model of a given size stops learning much at a knowable point, and
    # spending the rest of the evening past it buys almost nothing. This used
    # to fill whatever time it was given: a 29M model was handed 1.9 billion
    # tokens -- 65 per parameter, more than three times what it can use -- and
    # ran for ten and a half hours to reach, in its last seven, a loss it had
    # essentially arrived at in the first three.
    #
    # The compute-optimal move with time left over is a BIGGER model, not more
    # tokens for a small one, which is exactly what the note below says. Real
    # small models are trained well past this point and do improve, so the cap
    # is a default and not a limit -- every setting on this screen can still be
    # overridden by hand.
    enough = int(counts["effective"] * arch.TOKENS_PER_PARAM_TARGET)
    if budget > enough > 0:
        saved = arch.time_for_tokens(architecture, caps, budget - enough,
                                     checkpointing)
        notes.append(
            "Your machine could read %s in the time you allowed, but a model "
            "this size has learned what it can from about %s -- %d tokens for "
            "every parameter it has. Training is planned to stop there, which "
            "gives you about %s back. To use the whole slot, choose a larger "
            "model: with time to spare, size is what buys quality, not more "
            "passes over the same text."
            % (_human(budget), _human(enough), arch.TOKENS_PER_PARAM_TARGET,
               _hours_words(saved)))
        budget = enough

    # A corpus can be smaller than the budget wants to consume. Reading the
    # same text more than about four times teaches the model to recite it
    # rather than to write, so the budget is trimmed and the reason is stated.
    corpus_tokens = payload.get("corpus_tokens")
    if corpus_tokens and budget > corpus_tokens * 4:
        # Phrased against the plan rather than against the clock, because the
        # cap above may already have shortened it -- "this machine could get
        # through X in the time you have" was true only when this was the
        # first thing to trim, and read as nonsense once it was the second.
        notes.append(
            "The plan wants %s tokens and the text you chose holds only about "
            "%s. Training stops after four passes over it, at %s, because past "
            "that the model starts memorising the text instead of learning "
            "from it."
            % (_human(budget), _human(corpus_tokens),
               _human(corpus_tokens * 4)))
        budget = int(corpus_tokens * 4)

    steps = int(max(20, budget // max(fit["tokens_per_step"], 1)))
    # What the run will actually take, which is not what was asked for the
    # moment anything above trimmed the budget. The review screen showed the
    # requested time and was therefore wrong by hours whenever a cap applied.
    planned_minutes = arch.time_for_tokens(
        architecture, caps, steps * fit["tokens_per_step"], checkpointing) \
        or minutes
    lr = arch.recommended_lr(architecture["hidden_size"])
    verdict = arch.training_verdict(architecture, budget)

    settings = {
        "arch": architecture,
        "vocab_size": vocab_size,
        "dtype": caps.get("recommended_dtype", "float32"),
        "fits": fit["fits"],
        "batch_size": fit["batch_size"],
        "grad_accum": fit["grad_accum"],
        "max_steps": steps,
        "token_budget": int(budget),
        "learning_rate": lr,
        "min_lr_ratio": 0.1,
        "weight_decay": 0.1,
        "grad_clip": 1.0,
        "warmup_steps": min(200, max(10, steps // 20)),
        "gradient_checkpointing": checkpointing,
        "optim_8bit": optim_8bit,
        "eval_every": max(20, steps // 25),
        "sample_every": max(40, steps // 10),
        # A safety net rather than an expected outcome. Pretraining on a large
        # corpus improves for as long as there is text left, so this will
        # almost never fire -- it exists for the case where the corpus is
        # small enough to be memorised, which the planner warns about
        # separately and which is common on one graphics card.
        "early_stop": True,
        "early_stop_patience": 8,
        "sample_prompt": payload.get("sample_prompt") or "Once upon a time",
        "text_field": payload.get("text_field") or "text",
    }
    settings.update(payload.get("overrides") or {})

    issues = arch.validate_arch(architecture, caps, minutes=minutes,
                                corpus_tokens=corpus_tokens)
    issues += arch.check_settings(settings, architecture)

    return {
        "settings": settings,
        "params": counts,
        "params_label": arch.fmt_params(counts["total"]),
        "active_params_label": arch.fmt_params(counts["active"]),
        "moe": arch.moe_spec(architecture),
        "verdict": verdict,
        "notes": notes,
        "issues": issues,
        "blocked": any(i["level"] == "error" for i in issues),
        "recommended_lr": arch.recommended_lr(architecture["hidden_size"]),
        "limits": arch.LIMITS,
        # Where each slider clicks, given everything else as it stands now.
        "scales": arch.slider_scales(architecture, caps),
        "memory_gb": fit["memory"]["total_gb"],
        "tokens_per_step": fit["tokens_per_step"],
        "estimated_minutes": round(planned_minutes, 1),
        "requested_minutes": round(minutes, 1),
        "minutes_for_full": arch.time_for_tokens(
            architecture, caps, counts["effective"] * arch.TOKENS_PER_PARAM_TARGET,
            checkpointing),
        "explanations": _explain_scratch(settings, counts, verdict, fit),
    }


def _hours_words(minutes: float | None) -> str:
    if not minutes:
        return "no time"
    if minutes < 90:
        return "%d minutes" % round(minutes)
    return "%.1f hours" % (minutes / 60)


def _human(n: float) -> str:
    if n >= 1e9:
        return "%.1f billion" % (n / 1e9)
    if n >= 1e6:
        return "%.0f million" % (n / 1e6)
    return "%.0f thousand" % (n / 1e3)


def _explain_scratch(s: dict, counts: dict, verdict: dict, fit: dict) -> list[dict]:
    a = s["arch"]
    out = [
        {"setting": "Model size", "value": arch.fmt_params(counts["total"]),
         "why": "%d layers, %d wide. Every one of those numbers starts as noise "
                "and has to be learned from the text you chose."
                % (a["num_hidden_layers"], a["hidden_size"])},
    ]
    if counts.get("experts"):
        out.append({
            "setting": "Experts",
            "value": "%d, %d per token" % (counts["experts"],
                                           counts["experts_per_token"]),
            "why": "Each block holds %d separate feed-forward networks and a "
                   "router that picks %d of them for every token. All %s "
                   "parameters sit in memory and have to be learned; only %s "
                   "of them do the work on any given token."
                   % (counts["experts"], counts["experts_per_token"],
                      arch.fmt_params(counts["total"]),
                      arch.fmt_params(counts["active"]))})
    out += [
        {"setting": "Vocabulary", "value": "%s tokens" % f"{s['vocab_size']:,}",
         "why": "Built from your own text rather than borrowed. Even at this "
                "size the vocabulary is %.0f%% of the model -- a borrowed one "
                "would be most of it."
                % (counts["embedding_share"] * 100)},
        {"setting": "Text per step", "value": "%s tokens" % f"{fit['tokens_per_step']:,}",
         "why": "Batch of %d, %d times over before each update, %d tokens at a "
                "time. Pretraining needs a lot of text per update or the "
                "gradient is too noisy to follow."
                % (s["batch_size"], s["grad_accum"], a["max_position_embeddings"])},
        {"setting": "Learning rate", "value": "%.1e" % s["learning_rate"],
         "why": "Several times higher than fine-tuning uses, because a model "
                "starting from random weights has much further to travel."},
        {"setting": "How well it will learn",
         "value": "%s tokens per parameter" % (verdict.get("ratio") or "?"),
         "why": "%s Twenty per parameter is roughly where a model has learned "
                "everything its size can hold." % verdict["message"]},
    ]
    if s["gradient_checkpointing"]:
        out.append({"setting": "Memory saving", "value": "on",
                    "why": "This size does not fit otherwise, so activations "
                           "are recomputed during the backward pass rather than "
                           "stored. Does the forward pass twice, so about "
                           "a quarter slower."})
    if s["optim_8bit"]:
        out.append({"setting": "8-bit optimiser", "value": "on",
                    "why": "Adam normally keeps two 32-bit numbers per "
                           "parameter. Holding them in 8 bits frees memory for "
                           "a bigger batch, at no measurable quality cost."})
    return out


# ===========================================================================
# Playground -- talking to a finished model
# ===========================================================================
#
# Inference runs on a runner, never here: the controller has no GPU and no
# torch, and that is the property that lets it run on a NAS. So a chat message
# is a request routed over the same websocket the fleet already holds open,
# and the reply streams back through the browser event stream token by token.

chat_spec = serving.chat_spec


def _pick_chat_runner(job: dict) -> tuple[str, dict]:
    """Prefer the machine that trained it; fall back to any idle capable one."""
    online = [r for r in db.list_runners() if r["status"] != "offline"
              and r["id"] in fleet.connections]
    if not online:
        raise HTTPException(400, "No machine is connected to run the model on.")

    def usable(r: dict) -> bool:
        if fleet.busy.get(r["id"]):
            return False
        caps = r["capabilities"] or {}
        params_b = job["config"].get("params_b")
        if job["kind"] != "pretrain_llm" and params_b and caps.get("vram_gb"):
            # Serving needs the base model resident but no optimiser state, so
            # the bar is far lower than training's.
            mem = hub.estimate_memory(params_b) or {}
            if (mem.get("inference_fp16_gb") or 0) > caps["vram_gb"]:
                return False
        return True

    trained_on = job.get("runner_id")
    ordered = sorted(online, key=lambda r: r["id"] != trained_on)
    for r in ordered:
        if usable(r):
            return r["id"], r
    busy = [r["name"] for r in ordered if fleet.busy.get(r["id"])]
    if busy:
        raise HTTPException(400,
            "%s is training right now. Wait for the run to finish, or connect "
            "another machine." % busy[0])
    raise HTTPException(400,
        "No connected machine has enough memory to load this model.")


@app.get("/api/playground")
async def playground(request: Request) -> list[dict]:
    """Finished runs you can talk to."""
    out = []
    for job in db.visible_jobs(security.current_user(request), 200):
        # A run that was stopped early but kept its model belongs here too.
        # The artifact on disk is the real test of whether there is something
        # to talk to; the status only says how it got there.
        if job["status"] not in ("succeeded", "cancelled"):
            continue
        if not (config.ARTIFACT_DIR / ("%s.zip" % job["id"])).exists():
            continue
        cfg = job["config"]
        spec = chat_spec(job)
        entry = {
            "id": job["id"], "name": job["name"], "kind": job["kind"],
            "finished_at": job["finished_at"], "dataset": cfg.get("dataset"),
            "base_model": cfg.get("base_model"),
            "style": spec["style"],
            "mode": spec["style"],          # kept for older cached scripts
            "system_prompt": spec["system_prompt"],
            "reasoning": spec["reasoning"],
            "stopped_early": job["status"] == "cancelled",
        }
        if job["kind"] == "pretrain_llm":
            a = cfg.get("arch") or {}
            entry["size"] = (arch.preset(a.get("size_id", "")) or {}).get("label")
        out.append(entry)
    return out


@app.get("/api/jobs/{job_id}/system-prompt")
async def job_system_prompt(request: Request, job_id: str) -> dict:
    """The system prompt this run was trained with.

    Recorded on the job when it was created, and looked up from the dataset
    when it was not -- runs made before this existed, or through the API, still
    have their prompt sitting in the data they trained on. A model fine-tuned
    with a system prompt behaves noticeably worse without it, so it is worth
    going back for.
    """
    job = _job_or_404(request, job_id)
    cfg = job["config"]
    if cfg.get("system_prompt"):
        return {"system_prompt": cfg["system_prompt"], "source": "recorded"}
    if not cfg.get("dataset"):
        return {"system_prompt": "", "source": "none"}

    try:
        preview = await hub.training_preview(
            cfg["dataset"], cfg.get("dataset_config"),
            cfg.get("dataset_split") or "train", cfg.get("format"),
            None, cfg.get("base_model"))
    except Exception as e:  # noqa: BLE001
        return {"system_prompt": "", "source": "none", "reason": str(e)[:200]}

    found = (preview.get("system_prompts") or [""])[0]
    if found:
        # Remembered now, so the dataset is only read once.
        cfg["system_prompt"] = found
        db.update_job_config(job_id, cfg)
    return {"system_prompt": found, "source": "dataset" if found else "none"}


@app.post("/api/jobs/{job_id}/chat")
async def chat(request: Request, job_id: str, payload: dict = Body(...)) -> dict:
    job = _job_or_404(request, job_id)
    # Not "did it succeed" but "is there a model". A run stopped early that
    # kept its model has one, and refusing to talk to it would make the
    # keeping pointless.
    if job["status"] not in ("succeeded", "cancelled"):
        raise HTTPException(400, "This run has not finished.")
    if not (config.ARTIFACT_DIR / ("%s.zip" % job_id)).exists():
        raise HTTPException(400, "This run did not leave a downloadable model.")
    # A conversation, not a string. The playground keeps the turns so the model
    # sees the same running context it was trained on; a single prompt is still
    # accepted, and becomes a one-turn conversation.
    messages = payload.get("messages")
    if not messages:
        prompt = (payload.get("prompt") or "").strip()
        if not prompt:
            raise HTTPException(400, "Type something first.")
        messages = [{"role": "user", "content": prompt}]
    if not any((m.get("content") or "").strip() for m in messages
               if m.get("role") != "system"):
        raise HTTPException(400, "Type something first.")

    system = (payload.get("system") or "").strip()
    if system:
        messages = [{"role": "system", "content": system}] + [
            m for m in messages if m.get("role") != "system"]

    runner_id, runner = _pick_chat_runner(job)
    spec = chat_spec(job)
    if config.HF_TOKEN:
        spec["hf_token"] = config.HF_TOKEN
    request_id = db.new_id("gen")
    sent = await fleet.send_to_runner(runner_id, {
        "type": "generate", "request_id": request_id, "spec": spec,
        "messages": messages,
        "params": {
            "max_new_tokens": min(int(payload.get("max_new_tokens") or 200), 512),
            "temperature": float(payload.get("temperature") or 0.8),
            "top_k": int(payload.get("top_k") or 50),
            "top_p": float(payload.get("top_p") or 0.95),
            # Ask the model to work through the problem before answering. Only
            # meaningful if it was trained that way; a model that never saw a
            # reasoning block will simply carry on writing prose inside one.
            "reasoning": bool(payload.get("reasoning")),
        },
    })
    if not sent:
        raise HTTPException(503, "That machine dropped off just now. Try again.")
    fleet.generations[request_id] = runner_id
    return {"request_id": request_id, "runner": runner["name"],
            "runner_id": runner_id, "style": spec["style"]}


@app.post("/api/chat/{request_id}/cancel")
async def chat_cancel(request_id: str) -> dict:
    runner_id = fleet.generations.get(request_id)
    if runner_id:
        await fleet.send_to_runner(runner_id, {"type": "generate_cancel",
                                               "request_id": request_id})
    return {"ok": True}


# ===========================================================================
# Static web UI
# ===========================================================================

if config.WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(config.WEB_DIR)), name="static")


@app.middleware("http")
async def revalidate_ui(request: Request, call_next):
    """Make the browser check before reusing any part of the UI.

    Without a Cache-Control header a browser is free to invent one: it caches
    heuristically, for a fraction of the file's age, and reuses the file with
    no request at all. For a page and its scripts that ship as one unit, that
    produces the worst possible failure -- a *mixture* of versions. Exactly
    that happened here: a phone picked up new HTML, which advertised a page
    the cached app.js had no route for, so a real link answered "page not
    found".

    Content-hashed filenames are the usual fix, and they need a build step
    this app deliberately does not have. `no-cache` is the honest alternative:
    it does not mean "do not store", it means "revalidate before reuse". The
    files are already served with ETags, so an unchanged asset costs a 304 and
    a few hundred bytes rather than a re-download.
    """
    response = await call_next(request)
    path = request.url.path
    if path == "/" or path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


@app.get("/", response_class=HTMLResponse)
async def index() -> Any:
    idx = config.WEB_DIR / "index.html"
    if not idx.exists():
        return HTMLResponse("<h1>AI Studio</h1><p>Web UI not found.</p>", status_code=500)
    return FileResponse(idx, headers={"Cache-Control": "no-cache"})
