"""FastAPI application: runner fleet, job queue, Hub proxy, and the web UI."""
from __future__ import annotations

import asyncio
import contextlib
import json
import time
import zipfile
from pathlib import Path
from typing import Any

from fastapi import (Body, FastAPI, File, Header, HTTPException, Query, Request,
                     UploadFile, WebSocket, WebSocketDisconnect)
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from common import apimodels, formatting

from . import assets
from . import architectures as arch
from . import cards, config, datasets as dsets, db, diagnose, hfaccount, hub
from . import preflight
from . import serving
from .api import (accounts, conversations, data, evals, media, providers, security,
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
    if filled := _backfill_dataset_facts():
        print("[studio] worked out what %d existing dataset(s) contain "
              "(reasoning, tool calls, roles)." % filled, flush=True)
    if sized := _backfill_job_sizes():
        print("[studio] recorded how large the model is on %d existing run(s), "
              "so the memory checks apply to them." % sized, flush=True)
    task = asyncio.create_task(fleet.scheduler_loop())
    sync = asyncio.create_task(_directory_loop())
    tidy = asyncio.create_task(_retention_loop())
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


async def _retention_loop() -> None:
    """Once an hour: expire old models, thin old runs, sweep orphans.

    Off the request path, in a thread, because a sweep over a few thousand
    runs reads and writes the database for a while and the studio must keep
    answering meanwhile. Errors are printed and the timer goes on: a tidy
    that dies is a disk that fills.
    """
    from . import retention
    await asyncio.sleep(120)
    while True:
        try:
            result = await asyncio.to_thread(retention.sweep)
            if result["models_removed"] or result["runs_thinned"] or result["orphan_files"]:
                print("[retention] %s" % json.dumps(result))
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 -- a timer must not die
            print("[retention] sweep failed: %s" % e)
        await asyncio.sleep(3600)


app = FastAPI(title="AI Studio", version="0.1.0", lifespan=lifespan)

# Registered before anything else so that no route can be reached without
# passing it. Order matters here: middleware added later runs first, and the
# authentication gate must run before any handler.
app.middleware("http")(security.authenticate)

app.include_router(accounts.router)
app.include_router(data.router)
app.include_router(media.router)
app.include_router(conversations.router)
app.include_router(evals.router)
app.include_router(providers.router)
app.include_router(serving_api.router)
app.include_router(serving_api.registry)
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
    user = auth.session_user(ws.cookies.get(auth.SESSION_COOKIE))
    if not user:
        await ws.close(code=4401)
        return
    await ws.accept()
    fleet.ui_clients[ws] = user
    try:
        while True:
            await ws.receive_text()
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        fleet.ui_clients.pop(ws, None)


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


@app.get("/api/storage")
async def storage_report(request: Request) -> dict:
    """What the disk is holding, and the rules for keeping it. Administrators."""
    security.require_admin(request)
    from . import retention
    return await asyncio.to_thread(retention.report)


@app.put("/api/storage/settings")
async def storage_settings(request: Request, payload: dict = Body(...)) -> dict:
    user = security.require_admin(request)
    from . import retention
    try:
        return retention.save_settings(payload, user["id"])
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@app.post("/api/storage/sweep")
async def storage_sweep(request: Request) -> dict:
    security.require_admin(request)
    from . import retention
    return await asyncio.to_thread(retention.sweep)


@app.delete("/api/jobs/{job_id}/artifacts")
async def drop_job_model(request: Request, job_id: str) -> dict:
    """Remove a run's model and keep the run: the chart, the log, the notes.

    The usual reason to delete a run is the space its model takes, and
    deleting the run threw away the record of what was tried with it.
    """
    job = _job_or_404(request, job_id, "own")
    if job["status"] in ("queued", "assigned", "running"):
        raise HTTPException(400, "This run has not finished.")
    if not db.list_artifacts(job_id):
        raise HTTPException(400, "This run has no model to remove.")
    if db.aliases_for_job(job_id):
        raise HTTPException(400, "This run is served under a name (%s). Point "
                                 "the name elsewhere first."
                            % ", ".join(db.aliases_for_job(job_id)))
    from . import retention
    result = retention.drop_model(job_id, "removed by %s" % (
        security.current_user(request).get("display_name") or "its owner"))
    for runner_id in list(fleet.connections):
        await fleet.send_to_runner(runner_id, {"type": "purge_model", "job_id": job_id})
    return {"ok": True, **result}


@app.get("/api/runners")
async def get_runners() -> list[dict]:
    out = []
    for r in db.list_runners():
        r["connected"] = r["id"] in fleet.connections
        r["current_job"] = fleet.busy.get(r["id"])
        r["checkpoints"] = sorted(fleet.checkpoints.get(r["id"], set()))
        # The heartbeat has always carried this; the page never got it.
        r["disk"] = fleet.disk.get(r["id"])
        r["checkpoint_detail"] = fleet.checkpoint_detail.get(r["id"]) or []
        out.append(r)
    return out


@app.post("/api/runners/{runner_id}/reprobe")
async def reprobe(request: Request, runner_id: str) -> dict:
    # Re-probing pauses a machine's hardware for a benchmark. Members can see
    # the fleet; only an administrator gets to poke it.
    security.require_admin(request)
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
        _hide_credentials(j)
    return jobs


def primary_metric(summary: dict) -> dict | None:
    """What a run says it should be judged by, and which way is up.

    Runs finished before they said carry only best_val_loss; that is read as
    a held-out loss, lower better, which is what it always was.
    """
    if not summary:
        return None
    if isinstance(summary.get("primary_metric"), dict) \
            and summary["primary_metric"].get("value") is not None:
        m = summary["primary_metric"]
        return {"name": m.get("name") or "metric", "label": m.get("label") or "Metric",
                "value": float(m["value"]), "lower_better": bool(m.get("lower_better", True))}
    if summary.get("best_val_loss") is not None:
        return {"name": "held_out_loss", "label": "Held-out loss",
                "value": float(summary["best_val_loss"]), "lower_better": True}
    return None


def _hide_credentials(job: dict) -> None:
    """Never hand a credential back to the browser.

    Not the runner's copy of the Hugging Face token, and not the API key a
    hosted model was written with. Both went in at creation and only the
    runner needs them -- and a run can be shared with someone who should be
    able to read its settings without inheriting the owner's token.
    """
    job["config"].pop("hf_token", None)
    if isinstance(job["config"].get("model"), dict):
        job["config"]["model"].pop("connection", None)
    # An evaluation carries one of these per model it scores, because a
    # scoring can put a hosted model up against a studio run as a baseline.
    for m in job["config"].get("models") or []:
        if isinstance(m, dict):
            m.pop("connection", None)
            if isinstance(m.get("spec"), dict):
                m["spec"].pop("hf_token", None)
    if isinstance(job["config"].get("judge"), dict):
        job["config"]["judge"].pop("connection", None)


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
        # What the data looked like at this moment. A dataset can be edited in
        # place -- curating rows is the work -- so the id alone does not answer
        # "was this trained on the same data", which is the first question
        # asked when two runs of the same settings disagree.
        cfg["dataset_fingerprint"] = dsets.fingerprint(d)

    # Training on top of something this studio already built. The permission
    # check is the point: without it, any run id pasted into this field would
    # hand out the weights of a model you are not allowed to see.
    #
    # An upload names a source run too, but wants none of the rules below: a
    # run that failed halfway and kept its model is a perfectly reasonable
    # thing to publish, and nothing about the base model needs carrying
    # across. It does its own, narrower check.
    if kind != "upload" and (
            source_id := (cfg.get("continue_from") or cfg.get("base_model_job")
                          or cfg.get("source_job"))):
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

    if kind in ("finetune_llm", "pretrain_llm", "finetune_vision_cls"):
        # Over the account's ceiling of stored models? Said now, with the
        # numbers, rather than when the model arrives after four hours.
        from . import retention
        try:
            retention.check_quota(user["id"])
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
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
    elif kind == "finetune_vision_cls":
        # Pictures and labels, from the studio's own library only: the rows
        # point at stored files, and a Hub image dataset would arrive as URLs
        # nobody here can fetch. The columns are checked against the dataset
        # so a typo fails now rather than after the download.
        if not cfg.get("studio_dataset"):
            raise HTTPException(400, "Choose a dataset of labelled pictures.")
        d = db.get_dataset(cfg["studio_dataset"]) or {}
        cols = list(d.get("columns") or [])
        cfg["image_field"] = (cfg.get("image_field") or "image").strip()
        cfg["label_field"] = (cfg.get("label_field") or "label").strip()
        for field, what in ((cfg["image_field"], "pictures"), (cfg["label_field"], "labels")):
            if cols and field not in cols:
                raise HTTPException(
                    400, "This dataset has no \"%s\" column to take the %s from. "
                         "Its columns are: %s." % (field, what, ", ".join(cols)))
        cfg.setdefault("base_model", "google/vit-base-patch16-224")
        cfg["modality"] = "vision"
        cfg["params_b"] = cfg.get("params_b") or 0.09
        cfg.setdefault("hf_token", hfaccount.token_for(user))
    elif kind == "generate_dataset":
        model = cfg.get("model") or {}
        if not model.get("job_id") and not model.get("base_model") \
                and not model.get("provider"):
            raise HTTPException(400, "Choose a model to write the data with.")
        # Answering an existing split needs no count: there is one answer per
        # row, and the number of rows is the size of the split. Everything
        # else is inventing rows, and how many is the whole question.
        if cfg.get("mode") != "from_dataset" and not int(cfg.get("count") or 0):
            raise HTTPException(400, "How many rows should it write?")
        # Extending conversations reads an existing dataset. It travels the
        # same way a training dataset does -- as a URL the runner fetches with
        # its join token -- so a private dataset is never made public in order
        # to be lengthened.
        if cfg.get("mode") == "conversations":
            if not (cfg.get("instruction") or "").strip():
                raise HTTPException(
                    400, "Say what these conversations should be about.")
            if tools := (cfg.get("tools") or "").strip():
                try:
                    parsed = json.loads(tools)
                except ValueError as e:
                    raise HTTPException(
                        400, "The tools are not valid JSON: %s" % e) from None
                names = [t for t in (parsed if isinstance(parsed, list)
                                     else [parsed])
                         if isinstance(t, dict)]
                if not names:
                    raise HTTPException(
                        400, "The tools must be a JSON array of function "
                             "definitions.")
        # Two modes read an existing dataset: lengthening its conversations,
        # and answering the questions already in it. The dataset travels the
        # same way a training one does -- as a URL the runner fetches with its
        # join token -- so a private dataset is never made public in order to
        # be read.
        if cfg.get("mode") in ("extend_conversations", "from_dataset"):
            answering = cfg["mode"] == "from_dataset"
            src_id = (cfg.get("source_dataset_id") or "").strip()
            if not src_id:
                raise HTTPException(
                    400, "Which dataset holds the questions?" if answering
                    else "Which dataset should the extra turns be added to?")
            d = db.get_dataset(src_id)
            if not d:
                raise HTTPException(404, "No such dataset.")
            security.require_view(request, "dataset", d)
            cfg["source_dataset"] = "%s/api/datasets/%s/dataset-file" % (
                str(request.base_url).rstrip("/"), src_id)
            cfg["source_label"] = d["name"]
            # How the runner should read a row of it. Sent rather than guessed
            # again on the far side: the studio already knows this dataset's
            # format, and two guesses that disagree is a run that answers the
            # wrong column.
            cfg["source_format"] = d.get("format") or {}
            split = (cfg.get("source_split") or "").strip()
            splits = d.get("splits") or {}
            if split and splits and split not in splits:
                raise HTTPException(
                    400, "That dataset has no split called %r. It has: %s."
                         % (split, ", ".join(splits) or "none"))
            # One answer per row, so the count is the size of the split rather
            # than a number somebody has to guess. Asking for fewer is a
            # sample of it; asking for more would just run out.
            available = splits.get(split) if split else (d.get("rows") or 0)
            if answering:
                asked = int(cfg.get("count") or 0)
                cfg["count"] = min(asked, available) if asked else available
                if not cfg["count"]:
                    raise HTTPException(
                        400, "That split has no rows to answer.")
            else:
                cfg.setdefault("count", d.get("rows") or 0)
        if model.get("provider"):
            _attach_provider(user, cfg)
        else:
            _check_generation_source(request, cfg)
    elif kind == "export_gguf":
        # Turning a finished model into the one file everything outside this
        # studio wants. No GPU, so it goes to whichever machine is free --
        # which on most fleets is the CPU box beside the controller.
        src_id = (cfg.get("source_job") or "").strip()
        if not src_id:
            raise HTTPException(400, "Which run's model should be exported?")
        src = _job_or_404(request, src_id)
        # Not "is there a file" — a fine-tune that did not merge stores its
        # adapter under the primary name, so the file exists and is the wrong
        # thing. The recorded kind is read off the archive when it arrives and
        # is the honest answer. Checked here so the refusal costs nothing,
        # rather than after a runner has downloaded fourteen gigabytes.
        kinds = {a["kind"] for a in db.list_artifacts(src["id"])}
        if "model" not in kinds:
            raise HTTPException(
                400, "That run left an adapter, not a standalone model, and "
                     "there is no such thing as a GGUF of an adapter — it is a "
                     "few megabytes that mean nothing without the exact weights "
                     "they were fitted to. Run it again with \"also produce a "
                     "standalone model\" turned on, and export that.")
        cfg["allow_cpu"] = True
        cfg.setdefault("name_hint", src["name"])
        cfg["source_run_name"] = src["name"]
    elif kind == "upload":
        # Sending something to Hugging Face. No GPU, no model to load: this
        # one is network and patience, and it comes through the same door as
        # everything else so that it gets the same queue, log, progress bar
        # and stop button. Whether the account may write there is settled
        # here, at creation, rather than twenty minutes into an upload.
        cfg["repo_id"] = (cfg.get("repo_id") or "").strip()
        if problem := hfaccount.repo_problem(cfg["repo_id"]):
            raise HTTPException(400, problem)
        if problem := hfaccount.can_publish_to(user, cfg["repo_id"]):
            raise HTTPException(403, problem)
        if cfg.get("target") == "dataset":
            if not cfg.get("studio_dataset"):
                raise HTTPException(400, "Which dataset should be published?")
        else:
            if not cfg.get("source_job"):
                raise HTTPException(400,
                                    "Which run's model should be published?")
            src = _job_or_404(request, cfg["source_job"])
            wanted = cfg.get("artifact_kind") or "model"
            source_file = artifact_file(src["id"],
                                        "" if wanted == "model" else wanted)
            if not source_file.exists():
                raise HTTPException(400, "That run has no saved %s." % wanted)
            # The backstop, and it is worth keeping now that a run can hold
            # both: an adapter published as a model is a repository that
            # cannot be loaded, and it looks entirely fine until somebody
            # tries. Publishing an adapter *as an adapter* is fine and says so
            # on the card.
            actual = _artifact_kind(source_file, src)
            if wanted == "model" and actual == "adapter":
                raise HTTPException(
                    400, "That run produced an adapter, not a standalone "
                         "model. Publish it as an adapter -- it will name the "
                         "base model it was fitted to -- or merge it first.")
            cfg["expect"] = wanted
            cfg["source_run_name"] = src["name"]
        cfg["allow_cpu"] = True
    else:
        raise HTTPException(400, "Unknown kind of training run: %s" % kind)

    # A size for the fit check to work with. Read off the model's name, which
    # needs no network call and is right for essentially every model on the
    # Hub -- "Mistral-7B-Instruct-v0.3" is not ambiguous. Without it the checks
    # below have nothing to compare against and wave the job through.
    if kind == "finetune_llm" and not cfg.get("params_b"):
        if guessed := hub.params_from_name(cfg.get("base_model") or ""):
            cfg["params_b"] = guessed

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
    scored = [(r, m) for r in done
              if (m := primary_metric(r.get("summary") or {}))]
    # The best run, by whatever the runs said they should be judged on and
    # in the direction they said. A sweep of classifiers ranks by accuracy,
    # highest first; a sweep of language models by loss, lowest first.
    if scored:
        lower = scored[0][1]["lower_better"]
        pick = min if lower else max
        row["best"] = pick(scored, key=lambda rm: rm[1]["value"])[0]["id"]
        row["ranked_by"] = scored[0][1]
    else:
        row["best"] = None
        row["ranked_by"] = None
    return row


def _default_job_name(cfg: dict, kind: str = "finetune_llm") -> str:
    data = str(cfg.get("dataset", "data")).split("/")[-1]
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
        # What each machine says about it. Only on the one job being looked
        # at: it is a per-runner fit check and the list page shows a hundred.
        job["waiting_on"] = fleet.why_waiting(job)
    job["resumable"] = _resumable(job)
    _hide_credentials(job)
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
    detail = next((c for c in fleet.checkpoint_detail.get(holder or "", [])
                   if c.get("job_id") == job["id"]), None)
    return {
        "step": step,
        "total": job.get("total_steps") or 0,
        "runner_id": holder,
        "runner": (runner or {}).get("name"),
        "online": holder in fleet.connections,
        # What is actually on that disk, when the machine is here to say.
        "bytes": (detail or {}).get("bytes"),
        "at": (detail or {}).get("at"),
        "best_step": (detail or {}).get("best_step"),
        "best_val_loss": (detail or {}).get("best_val_loss"),
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
    force = bool((payload or {}).get("force"))

    if force:
        # Cancelling is cooperative -- the trainer checks between steps, which
        # is what lets a stop keep the model it has built. It only works while
        # steps finish. A run wedged inside one never reaches the check, so the
        # ordinary button does nothing however many times it is pressed.
        #
        # So a force stop does not wait to be agreed to. The job is written off
        # here and the machine is freed here; the runner is told separately,
        # and ends its own process if the job will not let go. Nothing is kept,
        # because there is no cooperating run to keep anything from.
        db.set_job_status(job_id, "cancelled",
                          "Force-stopped: the run was not responding to an "
                          "ordinary stop.")
        db.clear_checkpoint(job_id)
        if runner_id := job.get("runner_id"):
            fleet.busy.pop(runner_id, None)
            fleet.dispatched_at.pop(runner_id, None)
            fleet.checkpoints.get(runner_id, set()).discard(job_id)
            await fleet.send_to_runner(runner_id, {"type": "job_kill",
                                                   "job_id": job_id})
        db.add_log(job_id, "Force-stopped. The machine was told to end it, and "
                           "will restart itself if the run will not let go. "
                           "Nothing was kept.", "warn")
        await fleet.broadcast_ui({"type": "jobs_changed"})
        fleet.wake()
        return {"ok": True, "forced": True}

    if job["runner_id"] and job["runner_id"] in fleet.connections:
        # Recorded, so the page can tell "asked to stop" from "not asked". A
        # run that was asked and carried on is stuck inside a step rather than
        # between two, and that is the only situation where forcing is the
        # right answer -- so it is the only situation where it is offered
        # without being gone looking for.
        db.set_job_summary(job_id, {**(job.get("summary") or {}),
                                    "stop_requested": db.now()})
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

    # What the run drew while it ran -- sample grids, clips -- goes with it.
    assets.release_job(job_id)
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


def artifact_file(job_id: str, kind: str = "") -> Path:
    """Where a run's result lives on disk.

    A run can leave two things: a fine-tune keeps the model it merged *and*
    the adapter it merged from. The model is the primary one and keeps the
    bare name it always had, so every reader written before there was a second
    one -- download, publish, the fit checks, an older runner -- still finds
    the thing it expects.
    """
    return config.ARTIFACT_DIR / (
        "%s.zip" % job_id if kind in ("", "model", None)
        else "%s-%s.zip" % (job_id, kind))


@app.put("/api/jobs/{job_id}/artifact")
async def put_artifact(job_id: str, request: Request, kind: str = "",
                       x_runner_token: str = Header(default="")) -> dict:
    """Receive a finished model as a raw body, straight to its final place.

    The same thing as the POST below, minus a detour that cost a real run. A
    multipart upload arrives as Starlette's UploadFile, which buffers the whole
    body to a spooled temp file *before* the handler is called; the handler
    then copies it to the artifact directory. For a fourteen-gigabyte merged
    model that is twenty-eight gigabytes of writes, twice the free space, and
    several minutes of silence on a loaded box while the runner waits for a
    response it has a deadline for. It stopped waiting, and a merge that had
    successfully written every shard was recorded as failed.

    The POST is kept because a runner older than this controller still speaks
    it, and an artifact is the last thing that should break on a version skew.
    """
    if x_runner_token != config.join_token():
        raise HTTPException(401, "Invalid runner token.")
    if not db.get_job(job_id):
        raise HTTPException(404, "No such job.")
    config.ensure_dirs()
    dest = artifact_file(job_id, kind)
    size = 0
    with open(dest, "wb") as fh:
        async for chunk in request.stream():
            fh.write(chunk)
            size += len(chunk)
    return await _artifact_stored(job_id, dest, size)


@app.post("/api/jobs/{job_id}/artifact")
async def upload_artifact(job_id: str, file: UploadFile, kind: str = "",
                          x_runner_token: str = Header(default="")) -> dict:
    if x_runner_token != config.join_token():
        raise HTTPException(401, "Invalid runner token.")
    if not db.get_job(job_id):
        raise HTTPException(404, "No such job.")
    config.ensure_dirs()
    dest = artifact_file(job_id, kind)
    size = 0
    with open(dest, "wb") as fh:
        while chunk := await file.read(1 << 20):
            fh.write(chunk)
            size += len(chunk)
    return await _artifact_stored(job_id, dest, size)


def _artifact_kind(dest: Path, job: dict | None) -> str:
    """What was actually uploaded, read off the archive rather than assumed.

    A fine-tune used to mean "an adapter" and a from-scratch run "a model",
    and the label was worked out from the run's kind. A fine-tune now sends
    both, so the run's kind no longer answers the question -- but the file
    does, and unambiguously: a zip's central directory is read without
    unpacking a byte of it.
    """
    try:
        names = {Path(n).name for n in zipfile.ZipFile(dest).namelist()}
    except (OSError, zipfile.BadZipFile):
        names = set()
    if "adapter_config.json" in names:
        return "adapter"
    # config.json is a transformers model; model_index.json a diffusers
    # pipeline; a bare safetensors with neither is still weights.
    if "config.json" in names or "model_index.json" in names:
        return "model"
    if any(n.endswith(".safetensors") for n in names) \
            and not any(n.endswith(".jsonl") for n in names):
        return "model"
    if any(n.endswith(".jsonl") for n in names):
        return "dataset"
    if any(n.endswith(".gguf") for n in names):
        # Not a model in the sense the rest of this app means: nothing here
        # can load one, and the playground and the fit checks must not be
        # offered it. It is a file to download and take elsewhere.
        return "gguf"
    # An archive that says nothing about itself: fall back to what the run was.
    return {"pretrain_llm": "model", "generate_dataset": "dataset",
            "export_gguf": "gguf",
            "merge_adapter": "model"}.get((job or {}).get("kind"), "adapter")


def _sha256_of(path: Path) -> str | None:
    """The hash of a file, read a megabyte at a time. None if it cannot be."""
    import hashlib
    try:
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            while chunk := fh.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


async def _artifact_stored(job_id: str, dest: Path, size: int) -> dict:
    """Everything that happens once the bytes are on disk, however they came.

    Recording what kind of thing it is, and -- for a generation run -- turning
    the rows it wrote into a dataset. Merging used to be queued from here as a
    second run; it is now the last step of the training run itself, so by the
    time this is called both artifacts are already made.
    """
    job = db.get_job(job_id)
    kind = _artifact_kind(dest, job)
    # What is actually in the archive. Recorded so a download can be checked
    # against what the runner sent -- a truncated upload and a complete one
    # look identical in a directory listing, and a model that will not load
    # is the first anybody hears of it.
    db.add_artifact(job_id, kind, dest.name, size, _sha256_of(dest))

    if kind == "dataset" and job:
        # Rows written by a model are only useful once they are a dataset you
        # can look at, clean and train on. Doing that here, rather than making
        # the user download a zip and upload it again, is the whole point of
        # generation being part of the studio.
        try:
            created = _register_generated(job, dest)
            # Recorded on the run, so its page can link straight to what it
            # made. Without this the only way from a finished generation to
            # its rows was to go and find them by name on the datasets page.
            summary = dict(job.get("summary") or {})
            summary["dataset_id"] = created["id"]
            summary["dataset_name"] = created["name"]
            summary["rows"] = created.get("rows")
            db.set_job_summary(job_id, summary)
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

    This queues the upload rather than performing it. Fourteen gigabytes takes
    twenty minutes, which is longer than any browser will hold a request open;
    doing it here is what used to leave an empty repository behind.

    A fine-tune now finishes with two artifacts, and either may go: the merged
    model, which loads with `from_pretrained` and needs nothing else, or the
    adapter, which is fifty megabytes and is what the Hub understands as a
    fine-tune *of* something. `what` chooses -- "model", "adapter" or "both",
    and "both" means two repositories, because they are two different things
    and a repository holding both loads as neither.

    What is refused is publishing an adapter *as* a model. That is a
    repository that looks complete and loads as nothing.
    """
    job = _job_or_404(request, job_id)
    want = {
        "repo_id": (payload.get("repo_id") or "").strip(),
        "private": bool(payload.get("private", True)),
        "replace": bool(payload.get("replace")),
        "message": payload.get("message") or "",
    }
    have = _artifacts_present(job_id)
    if not have:
        raise HTTPException(400, "This run has no saved model to publish.")

    what = (payload.get("what") or "model").strip().lower()
    if what not in ("model", "adapter", "both"):
        raise HTTPException(400, "Publish what: the model, the adapter, or both.")
    if what in ("model", "both") and "model" not in have:
        # A fine-tune that could not be merged -- too little memory, or the
        # user asked for the adapter alone. There is nothing whole to send.
        raise HTTPException(
            400, "This run kept its adapter but no merged model, so there is "
                 "no standalone model to publish. Publish the adapter -- it "
                 "names the base model it was fitted to, and the Hub shows it "
                 "as a fine-tune of it.")
    if what in ("adapter", "both") and "adapter" not in have:
        raise HTTPException(400, "This run has no adapter saved.")

    targets = []
    if what in ("model", "both"):
        targets.append(("model", want["repo_id"]))
    if what in ("adapter", "both"):
        # Two things cannot share one repository, so "both" needs a second
        # name. `-lora` because that is what the Hub's own ecosystem calls it.
        adapter_repo = (payload.get("adapter_repo_id") or "").strip()
        if not adapter_repo:
            adapter_repo = ("%s-lora" % want["repo_id"]) if what == "both" \
                else want["repo_id"]
        targets.append(("adapter", adapter_repo))

    queued = []
    for artifact_kind, repo_id in targets:
        queued.append(await _queue_upload(request, job,
                                          {**want, "repo_id": repo_id},
                                          artifact_kind))
    first = queued[0]
    return {**first, "queued": queued}


def _has_artifact(job_id: str) -> bool:
    return artifact_file(job_id).exists()


def _artifacts_present(job_id: str) -> dict[str, Path]:
    """This run's saved artifacts, by what they actually are.

    Read from the archives themselves rather than from the run's kind: a
    fine-tune can now have both, one, or -- if the merge would not fit on the
    machine that trained it -- only its adapter.
    """
    job = db.get_job(job_id)
    out: dict[str, Path] = {}
    for kind in ("", "adapter"):
        path = artifact_file(job_id, kind)
        if path.exists():
            out.setdefault(_artifact_kind(path, job), path)
    return out


async def _queue_upload(request: Request, trained: dict, want: dict,
                        artifact_kind: str = "model") -> dict:
    """Queue one upload run: one artifact of one run to one repository."""
    jid = await _create_job(request, {
        "name": "Publishing %s" % (want["repo_id"] or trained["name"]),
        "kind": "upload",
        "config": {
            "target": "model", "source_job": trained["id"],
            "trained_by": trained["id"],
            # Which of the run's artifacts to fetch, and what the runner
            # should refuse to send if it turns out to be the other one.
            "artifact_kind": artifact_kind, "expect": artifact_kind,
            **want,
            "card": cards.for_publish(trained, trained, want["repo_id"],
                                      adapter=(artifact_kind == "adapter")),
        }})
    return {"job_id": jid, "repo_id": want["repo_id"],
            "artifact_kind": artifact_kind,
            "url": "https://huggingface.co/%s" % want["repo_id"]}


@app.get("/api/jobs/{job_id}/publish-info")
async def publish_info(request: Request, job_id: str) -> dict:
    """What this run could publish, and what its base is doing on the Hub.

    The publish dialog asks before it draws itself. The base matters because
    a fine-tune whose base is a local run that has never been published has no
    honest `base_model:` to declare -- so the card drops the line and the Hub
    shows a model with no parentage. Publishing the base first fixes that, and
    saying so is a recommendation, not a rule.
    """
    job = _job_or_404(request, job_id)
    cfg = job.get("config") or {}
    artifacts = [{"kind": kind, "size": path.stat().st_size}
                 for kind, path in sorted(_artifacts_present(job_id).items())]

    base: dict | None = None
    if base_job_id := cfg.get("base_model_job"):
        if base_job := db.get_job(base_job_id):
            base = {"job_id": base_job["id"], "name": base_job["name"],
                    "published": cards.published_repo(base_job)}
    elif cfg.get("base_model"):
        base = {"hub_id": cfg["base_model"]}

    return {"job_id": job_id, "artifacts": artifacts, "base": base,
            "published": db.publications(job_id),
            "merged": bool((job.get("summary") or {}).get("merged"))}


@app.get("/api/jobs/{job_id}/card")
async def get_job_card(request: Request, job_id: str) -> dict:
    """This run's model card, as it would go to the Hub.

    Generated on the spot for a run that has never had one, so a model trained
    before any of this existed still shows a card rather than an empty box.
    Nothing is written by a GET: the card is only stored when a run finishes,
    when an evaluation scores it, or when somebody saves an edit.
    """
    job = _job_or_404(request, job_id)
    if job["kind"] not in ("finetune_llm", "pretrain_llm", "merge_adapter"):
        raise HTTPException(400, "Only a run that produces a model has a "
                                 "model card.")
    return {**cards.card_for(job), "job_id": job_id, "kind": job["kind"]}


@app.put("/api/jobs/{job_id}/card")
async def put_job_card(request: Request, job_id: str,
                       payload: dict = Body(...)) -> dict:
    """Save an edited card, or throw the edits away and go back to generated.

    A saved card is marked `edited` and is never rewritten again -- not when
    the model is evaluated, not when it is merged, not when it is published.
    That is the point of saving it. `reset` is the way back: it deletes the
    stored card, and what you get next is generated from the run as it stands
    today, including anything that has happened since.
    """
    job = _job_or_404(request, job_id, "edit")
    if payload.get("reset"):
        db.clear_job_card(job_id)
        return {**cards.card_for(job), "job_id": job_id, "reset": True}
    text = payload.get("markdown")
    if not isinstance(text, str) or not text.strip():
        raise HTTPException(400, "A card cannot be empty. Use reset to go "
                                 "back to the generated one.")
    if len(text) > 256_000:
        raise HTTPException(400, "That card is too long for a README.")
    db.set_job_card(job_id, text, edited=True)
    db.add_log(job_id, "Model card edited. It will not be regenerated from "
                       "now on; reset it to hand it back to the studio.")
    return {"markdown": text, "edited": True, "saved": True,
            "job_id": job_id}


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
    CF = hub.chat_formats
    try:
        with zipfile.ZipFile(path) as z:
            names = {n.rsplit("/", 1)[-1]: n for n in z.namelist()}
            if CF.TOKENIZER_CONFIG not in names:
                return {"available": False, "reason": "no tokenizer in the result"}
            # Both places, because which one holds it depends on the version of
            # transformers that saved it. Looking only in the tokenizer config
            # found nothing in EVERY model this studio has produced -- so the
            # wizard told people their instruct fine-tune "ships no chat
            # template of its own, which usually means it is a base model".
            files = {n: z.read(names[n]).decode("utf-8")
                     for n in (CF.TEMPLATE_FILE, CF.TOKENIZER_CONFIG)
                     if n in names}
            conf = json.loads(files.get(CF.TOKENIZER_CONFIG) or "{}")
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as e:
        return {"available": False, "reason": str(e)[:200]}
    template = CF.template_in(files)
    return {"available": bool(template), "chat_template": template,
            "eos_token": conf.get("eos_token")}


@app.get("/api/jobs/{job_id}/download")
async def download_artifact(request: Request, job_id: str, kind: str = ""):
    """A run's result. `kind=adapter` asks for the second one, where there is
    one -- a fine-tune keeps the adapter beside the model it merged into."""
    path = artifact_file(job_id, kind)
    if not path.exists():
        raise HTTPException(
            404, "This run has no adapter saved." if kind == "adapter"
            else "No result file for this job yet.")
    # A runner fetching a model to serve presents the join token; a person
    # downloading one presents a session and has to be allowed the run.
    job = _job_or_404(request, job_id) \
        if not getattr(request.state, "runner", False) else db.get_job(job_id)
    safe = "".join(c for c in (job["name"] if job else job_id)
                   if c.isalnum() or c in "-_ ").strip().replace(" ", "-")
    # Named for what is in it, which is now something the file itself knows.
    return FileResponse(path, media_type="application/zip",
                        filename="%s-%s.zip" % (safe or job_id,
                                                _artifact_kind(path, job)))


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


@app.get("/api/hub/recommendations")
async def hub_recommendations(runner_id: str = Query(default="")) -> dict:
    """Which models are worth training on a particular machine.

    Costed against that runner's own measured memory rather than against a
    table of assumptions, and ordered so that the largest one which still
    leaves room is the one being pointed at.
    """
    runner = db.get_runner(runner_id) if runner_id else None
    caps = dict(runner["capabilities"]) if runner else {}
    out = hub.recommend_models(caps)
    out["runner"] = runner["name"] if runner else None
    return out


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


# See formatting.CONTENT_FACTS: what a format records about *reading* a dataset,
# versus what it records about what is *in* it. The first is a decision and is
# kept; the second is an observation and is re-made from the rows.
_CONTENT_FACTS = hub.formatting.CONTENT_FACTS


def _describe_rows(stored: dict | None, columns: list, rows: list) -> dict:
    """A dataset's format, with what is actually in these rows measured afresh.

    A stored format is a decision: which column holds the conversation, which
    chat format to render it in. Those are the user's, and they are kept.

    The content facts are not decisions. Whether a dataset has reasoning in it
    is true or false about the rows, and reading it off a record written when
    the dataset was created is how it goes stale -- or, in the case that
    prompted this, is never written at all. The conversion to conversations
    stores `{"mode": "chat"}`, which is a complete and correct description of
    how to read the result and says nothing about what is in it.
    """
    stored = dict(stored or {})
    if not rows:
        return stored
    observed = formatting.detect_format(columns, rows)
    # Only the facts, and only when the observation actually made them. A row
    # sample that yielded nothing should not overwrite a truth with a silence.
    for key in _CONTENT_FACTS:
        if key in observed:
            stored[key] = observed[key]
    for key in ("mode", "messages_field", "tools_field"):
        stored.setdefault(key, observed.get(key))
    return {k: v for k, v in stored.items() if v is not None}

def _backfill_job_sizes() -> int:
    """Record how large each finished model is, for runs that never said.

    Every memory guard in this app reads "refuse if it does not fit", and an
    unknown size does not fail safe -- it disables the check. A run created
    before the size was recorded therefore has no protection, and neither does
    anything continuing from it: that is exactly how a 7B came to be planned in
    16-bit for a 16 GB card.

    Two sources, best first. The runner counts the parameters of the model it
    actually loaded and reports the number, so a finished run knows its size
    exactly. Failing that the model's name is read, which is unambiguous for
    essentially everything on the Hub and is at least a number.

    Writes only where the answer is missing or wrong, so the second boot is a
    no-op.
    """
    fixed = 0
    for row in db.q("SELECT id FROM jobs WHERE kind IN "
                    "('finetune_llm','merge_adapter')"):
        job = db.get_job(row["id"])
        if not job:
            continue
        cfg = job.get("config") or {}
        summary = job.get("summary") or {}
        counted = summary.get("total_params") or summary.get("params_total")
        measured = round(int(counted) / 1e9, 3) if counted else None
        named = hub.params_from_name(cfg.get("base_model") or "")

        # A measurement can be wrong, and one particular way of being wrong is
        # recoverable here. A run trained in 4-bit counted its packed weights,
        # which hold two parameters per element, and so reported half the
        # model: a Mistral-7B came back as 3.8B. A fine-tune contains its base
        # and cannot be smaller than it, so a size well under what the base
        # model's own name implies was never a measurement of the model -- it
        # is an artefact of how the weights were stored.
        def artefact(size) -> bool:
            return bool(size and named and size < named * 0.9)

        if artefact(measured):
            measured = None
        size = measured or named
        if not size or cfg.get("params_b") == size:
            continue
        # A counted size always wins; a guessed one only fills a blank, so a
        # number somebody set deliberately is never quietly overwritten by an
        # inference from a filename. A stored size that is itself the packing
        # artefact is not such a number -- nobody chose it, and leaving it
        # there is what lets a 7B go on claiming to be a 3.8B.
        if cfg.get("params_b") and not measured \
                and not artefact(cfg.get("params_b")):
            continue
        cfg = dict(cfg)
        cfg["params_b"] = size
        db.update_job_config(job["id"], cfg)
        fixed += 1
    return fixed


def _backfill_dataset_facts() -> int:
    """Fill in what existing datasets never recorded about their own contents.

    A dataset's stored format used to describe only how to *read* it. Whether
    it contains reasoning, or tool calls, and which roles appear are facts the
    wizard needs -- it decides whether to offer "teach it to reason" from
    them -- and a dataset created before those were written down, or by the
    conversion to conversations, simply has none.

    Reading the rows answers it, and the rows are the authority -- so this
    re-measures rather than only filling blanks, and writes only where the
    stored answer and the rows disagree. Filling blanks alone was not enough:
    a dataset imported before the role aliases were broadened still had
    `function_call` recorded as a role of its own, and having *an* answer meant
    it was never looked at again.

    Because it writes only on disagreement, the second boot is a no-op. It
    costs one sample read per dataset, which is what makes it safe to repeat
    rather than needing a version number to remember it has run -- and it means
    any later improvement to detection heals every dataset by itself.
    """
    fixed = 0
    for row in db.q("SELECT id FROM datasets"):
        d = db.get_dataset(row["id"])
        if not d:
            continue
        fmt = d.get("format") or {}
        sample = list(dsets.iter_rows(d["id"], 200))
        if not sample:
            continue
        described = _describe_rows(fmt, d.get("columns") or [], sample)
        if described != fmt:
            db.update_dataset(d["id"], format=described)
            fixed += 1
    return fixed


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
            # A stored format says how to *read* these rows. Whether they
            # contain reasoning, or tool calls, or which roles appear, is a
            # fact about the rows themselves -- so it is measured from the rows
            # every time rather than taken from whatever was recorded when the
            # dataset was made.
            #
            # Trusting the stored copy meant a dataset converted to
            # conversations, whose recorded format is the bare {"mode":
            # "chat"}, was reported as having no reasoning in it. The wizard
            # then greyed out "teach it to reason" over a dataset whose every
            # row has a reasoning block.
            "detected_format": _describe_rows(d.get("format"),
                                              d.get("columns") or [], rows),
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


@app.post("/api/preflight")
async def preflight_check(request: Request, payload: dict = Body(...)) -> dict:
    """What would go wrong with this configuration, before it is queued.

    Everything here used to be found out an hour into a run, on a machine, by
    reading a log: rows cut off at the context length, a split with no rows in
    it, a dataset too small to hold anything back, a corpus read six times
    over. Counting the tokens needs a tokenizer, which the controller does not
    have and should not; a runner answers that part.
    """
    cfg = payload.get("config") or payload
    if ds_id := cfg.get("studio_dataset"):
        if d := db.get_dataset(ds_id):
            security.require_view(request, "dataset", d)
    try:
        return await preflight.check(cfg, fleet)
    except Exception as e:  # noqa: BLE001 - a check that fails is not a block
        return {"issues": [], "facts": {}, "blocked": False,
                "reason": str(e)[:300]}


@app.post("/api/plan")
async def plan(payload: dict = Body(...)) -> dict:
    """Turn a chosen model + runner into concrete, explained settings.

    This is the heart of the low-code promise: the user picks what they want to
    achieve, and the machine's measured capabilities decide the hyperparameters.
    """
    runner = db.get_runner(payload.get("runner_id", "")) if payload.get("runner_id") else None
    caps = (runner or {}).get("capabilities", {})
    # The size, worked out from the model's name when the caller does not have
    # it. Without it there is no fit report, so the plan quietly chooses
    # 16-bit -- which is how a 7B was planned for a 16 GB card that can only
    # hold it in 4-bit, and then refused at dispatch with advice the screen
    # offered no way to follow.
    params_b = payload.get("params_b") \
        or hub.params_from_name(payload.get("base_model") or "")
    goal = payload.get("goal", "instructions")
    rows = int(payload.get("dataset_rows") or 1000)

    fit = hub.fit_report(params_b, caps) if params_b else None
    quant = "4bit" if (fit and fit["verdict"] == "fits_quantized") else "none"
    # Nothing is gained by 16-bit that this machine cannot hold. If 4-bit is
    # the only precision that fits, that is the plan rather than a suggestion
    # printed next to a plan that does not work.
    if fit and fit.get("verdict") == "fits_quantized":
        quant = "4bit"
    dtype = caps.get("recommended_dtype", "float32")
    seq = min(1024, caps.get("max_recommended_seq_len", 1024))

    vram = caps.get("vram_gb") or 0
    batch = 4 if vram >= 24 else 2 if vram >= 12 else 1
    accum = max(1, 16 // batch)
    epochs = 3 if rows < 2000 else 2 if rows < 20000 else 1
    steps = max(20, int(rows * epochs / (batch * accum)))

    lora_r = 32 if goal in ("style", "domain") else 16
    return {
        "params_b": params_b,
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
            # Recorded on the run, so "the same settings" really is the same
            # run: the held-out slice and the batch order both come from here.
            "seed": 1234,
        },
        "fit": fit,
        # What each precision would actually cost, so the choice between them
        # can be shown as two numbers rather than as two words.
        "memory": hub.estimate_memory(params_b) if params_b else None,
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


#: The kinds of work a machine set aside for models is meant to do. A runner's
#: `kinds` is a whitelist of JOB kinds and chat is not a job, so a box
#: configured for `upload,generate_dataset` cannot be tested for "serving" --
#: it has to be read as "not meant for models" instead.
# Read off the one list of kinds that leave a model, plus scoring, rather
# than repeated here: the last time this was its own set it fell behind that
# list and a machine restricted to a new kind of training read as "not for
# models".
_SERVING_KINDS = set(serving.MODEL_KINDS) | {"evaluate"}


def _serves_models(r: dict) -> bool:
    """Whether this machine is one to hand a conversation to at all.

    Two ways to fail. A processor-only runner can load a model and will answer,
    at a word every few seconds -- which is what the playground was doing every
    time a merged model was chatted with, because the merge ran on the CPU box
    and the router sent conversations to whichever machine produced the
    artifact. And a machine restricted to uploads or dataset generation is
    somebody's deliberate arrangement, not a serving box.
    """
    caps = r.get("capabilities") or {}
    if caps.get("backend") not in ("cuda", "rocm", "mps"):
        return False
    kinds = {k.strip() for k in (caps.get("kinds") or []) if k}
    return not kinds or bool(kinds & _SERVING_KINDS)


def _pick_chat_runner(job: dict) -> tuple[str, dict]:
    """The best machine to talk to this model on, of the ones that can.

    In order: one that can actually serve, then one that already has the model
    on its card, then one that has it on disk, and only then the machine that
    trained it. That last rule used to be the only rule, and it is the reason
    the playground felt like it did -- a model whose training runner was busy
    or gone still went there, and a merged model went to the CPU box that
    merged it. Nothing remembered that another machine had loaded the same
    model a minute earlier and could answer immediately.
    """
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
                # Too big at full precision is not the end of it: the runner
                # loads a model that will not fit in 4-bit instead. Refusing
                # here on the 16-bit figure alone turned "slightly worse
                # answers" into "you cannot talk to this model at all".
                #
                # `4bit_decode`, not `4bit`: some cards quantize correctly for
                # training's wide shapes and return noise for the single-token
                # shape a reply is written with. Serving through one of those
                # produces fluent-looking gibberish, which is far worse than
                # saying the model does not fit.
                if not caps.get("quantization", {}).get("4bit_decode"):
                    return False
                if (mem.get("inference_int4_gb") or 0) > caps["vram_gb"]:
                    return False
        return True

    trained_on = job.get("runner_id")

    def preference(r: dict) -> tuple:
        rid = r["id"]
        return (
            # A GPU box meant for models, before anything else. False sorts
            # first, so each of these reads as "not this" costing a place.
            not _serves_models(r),
            # Already on the card: no fetch, no load, an answer now. This is
            # also what makes a conversation stick to one machine.
            job["id"] not in (fleet.loaded.get(rid) or []),
            # On its disk: a load, but no fourteen gigabytes over the network.
            job["id"] not in (fleet.cached.get(rid) or set()),
            # It trained the model, so it probably has the base cached too.
            # The old rule, now the tiebreak it should always have been.
            rid != trained_on,
        )

    ordered = sorted(online, key=preference)
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
        # Runs that produce a MODEL. Not every run does: writing a dataset
        # leaves a zip on disk exactly as training does, and "has an artifact"
        # was the whole test -- so a run that wrote 1,200 rows of JSONL was
        # offered here as something to chat with, and in the training wizard,
        # which reads this same list, as a model to fine-tune from. The
        # OpenAI-compatible endpoint has always filtered by kind; this did not.
        if job["kind"] not in serving.MODEL_KINDS:
            continue
        # A run that was stopped early but kept its model belongs here too.
        # The artifact on disk is the real test of whether there is something
        # to talk to; the status only says how it got there.
        if job["status"] not in ("succeeded", "cancelled"):
            continue
        if not (config.ARTIFACT_DIR / ("%s.zip" % job["id"])).exists():
            continue
        # Resolved, not raw: a merge records how to perform the merge, and
        # borrows from the run it was made from everything about how to talk
        # to the result. See controller/serving.resolved_config.
        cfg = serving.resolved_config(job)
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
        # Which dataset this run learned from, so the playground can offer its
        # held-out rows to try. The held-out split is the one that matters --
        # a row the model trained on proves nothing, because reciting it is
        # exactly what it was rewarded for.
        if studio_id := cfg.get("studio_dataset"):
            if d := db.get_dataset(studio_id):
                splits = d.get("splits") or {}
                entry["dataset_id"] = studio_id
                entry["dataset_name"] = d["name"]
                entry["splits"] = splits
                entry["held_out_split"] = next(
                    (s for s in ("validation", "test", "eval", "holdout")
                     if splits.get(s)), None)
        if job["kind"] == "pretrain_llm":
            a = cfg.get("arch") or {}
            entry["size"] = (arch.preset(a.get("size_id", "")) or {}).get("label")
        out.append(entry)
    return out


@app.patch("/api/jobs/{job_id}")
async def rename_job(request: Request, job_id: str,
                     payload: dict = Body(...)) -> dict:
    """Give a run a name that means something.

    Runs are named when they are created, from the model and the dataset, and
    that name is a decent guess and a poor label: a page of "Mistral-7B on
    support-tickets" tells you nothing about which one had the higher learning
    rate or which one you are actually shipping. Renaming needs edit rights,
    the same as anything else that changes a run.
    """
    job = _job_or_404(request, job_id, "edit")
    out = dict(job)
    if "notes" in payload:
        # Why this run was made, which nothing recorded. Six runs of the same
        # model on the same data were six identical rows and a memory test.
        notes = (payload.get("notes") or "")[:2000]
        db.set_job_notes(job_id, notes)
        out["notes"] = notes
    if "tags" in payload:
        # Labels to find it by -- "baseline", "shipped" -- as distinct from
        # notes, which are for reading.
        tags = db.clean_tags(payload.get("tags"))
        db.set_job_tags(job_id, tags)
        out["tags"] = tags
    if "name" in payload:
        name = (payload.get("name") or "").strip()[:120]
        if not name:
            raise HTTPException(400, "A run needs a name.")
        db.rename_job(job_id, name)
        out["name"] = name
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
    cfg = serving.resolved_config(job)
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


@app.post("/api/jobs/{job_id}/classify")
async def classify_picture(request: Request, job_id: str,
                           file: UploadFile = File(...)) -> dict:
    """One picture to a classifier this studio trained; every label back.

    The playground is built around a conversation and a classifier has none,
    so this is its own small door: the picture goes to a machine that can
    decode images and has, or can fetch, the model, and the answer comes back
    with a probability per category.
    """
    job = _job_or_404(request, job_id)
    if job["kind"] != "finetune_vision_cls":
        raise HTTPException(400, "That run is not a classifier.")
    if not _has_artifact(job_id):
        raise HTTPException(400, "That run has no saved model to ask.")
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "The picture is empty.")
    if len(raw) > 20 * 1024 * 1024:
        raise HTTPException(413, "That picture is over 20 MB.")

    # A machine that can look at pictures. The one that trained it has the
    # model on disk already; failing that, any that reports vision, or any
    # at all if none has said either way.
    online = [r for r in db.list_runners()
              if r["status"] != "offline" and r["id"] in fleet.connections]
    def ok(r):
        mods = (r.get("capabilities") or {}).get("modalities")
        return mods is None or "vision" in mods
    able = [r for r in online if ok(r)]
    if not able:
        raise HTTPException(503, "No connected machine can look at pictures "
                                 "right now.")
    able.sort(key=lambda r: (bool(fleet.busy.get(r["id"])),
                             r["id"] != job.get("runner_id")))
    runner = able[0]

    import base64
    rid = db.new_id("cls")
    queue: asyncio.Queue = asyncio.Queue()
    fleet.waiters[rid] = queue
    try:
        sent = await fleet.send_to_runner(runner["id"], {
            "type": "classify", "request_id": rid, "job_id": job_id,
            "image_b64": base64.b64encode(raw).decode("ascii"), "top": 8})
        if not sent:
            raise HTTPException(503, "That machine dropped off just now.")
        msg = await asyncio.wait_for(queue.get(), 180.0)
    except asyncio.TimeoutError:
        raise HTTPException(504, "The machine did not answer in three "
                                 "minutes.") from None
    finally:
        fleet.waiters.pop(rid, None)
    if msg.get("type") == "classify_error":
        raise HTTPException(502, msg.get("error") or "The classifier failed.")
    return {"labels": msg.get("labels") or [], "seconds": msg.get("seconds"),
            "size": msg.get("size"), "runner": runner["name"]}


@app.post("/api/jobs/{job_id}/chat")
async def chat(request: Request, job_id: str, payload: dict = Body(...)) -> dict:
    job = _job_or_404(request, job_id)
    if job["kind"] not in serving.MODEL_KINDS:
        raise HTTPException(
            400, "That run did not produce a model. It left a file behind -- "
                 "a dataset, or a set of scores -- which is not something to "
                 "talk to.")
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
    # A turn counts as something to answer if it has words OR a tool call OR is
    # a tool result. The last two matter: continuing a conversation past a tool
    # call means sending back a result and nothing else, and requiring text
    # there made the second half of every tool exchange impossible.
    # A picture with no words is a question -- "what is this?" is implied --
    # so a turn that shows something counts as saying something.
    if not any((m.get("content") or "").strip() or m.get("tool_calls")
               or m.get("media") or m.get("role") in ("tool", "function")
               for m in messages if m.get("role") != "system"):
        raise HTTPException(400, "Type something first.")

    system = (payload.get("system") or "").strip()
    if system:
        messages = [{"role": "system", "content": system}] + [
            m for m in messages if m.get("role") != "system"]

    runner_id, runner = _pick_chat_runner(job)
    spec = chat_spec(job)
    if config.HF_TOKEN:
        spec["hf_token"] = config.HF_TOKEN
    # Tools the caller wants declared for this exchange. They travel per
    # request rather than being fixed on the run, because trying a model
    # against a held-out row means declaring *that row's* tools -- which are
    # rarely the same from one row to the next.
    if tools := payload.get("tools"):
        spec["tools"] = [
            {"name": (t.get("function") or t).get("name"),
             "description": (t.get("function") or t).get("description") or "",
             "parameters": (t.get("function") or t).get("parameters") or {}}
            for t in tools if isinstance(t, dict)
            and (t.get("function") or t).get("name")]
    request_id = db.new_id("gen")
    sent = await fleet.send_to_runner(runner_id, {
        "type": "generate", "request_id": request_id, "spec": spec,
        "messages": messages,
        "params": {
            # A ceiling, not a default. 512 was neither: it was low enough
            # that ordinary answers ran into it and were cut off mid-sentence,
            # which reads as a broken model rather than as a setting. The
            # limit that matters is the one the reader chose, and they are
            # told when a reply reaches it.
            "max_new_tokens": min(int(payload.get("max_new_tokens") or 512),
                                  config.MAX_NEW_TOKENS),
            "temperature": float(payload.get("temperature") or 0.8),
            "top_k": int(payload.get("top_k") or 50),
            "top_p": float(payload.get("top_p") or 0.95),
            # Ask the model to work through the problem before answering. Only
            # meaningful if it was trained that way; a model that never saw a
            # reasoning block will simply carry on writing prose inside one.
            "reasoning": bool(payload.get("reasoning")),
            "deadline_s": config.GENERATION_DEADLINE_S,
        },
    })
    if not sent:
        raise HTTPException(503, "That machine dropped off just now. Try again.")
    fleet.generations[request_id] = runner_id
    fleet.generation_owner[request_id] = security.current_user(request)["id"]
    return {"request_id": request_id, "runner": runner["name"],
            "runner_id": runner_id, "style": spec["style"]}


@app.post("/api/chat/{request_id}/cancel")
async def chat_cancel(request: Request, request_id: str) -> dict:
    # Only the person who asked may stop the answer. Ids are random, so this
    # was a small hole, but it was the one route here with no check at all.
    user = security.current_user(request)
    owner = fleet.generation_owner.get(request_id)
    if owner and owner != user["id"] and user["role"] != "admin":
        raise HTTPException(404, "No such request.")
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
