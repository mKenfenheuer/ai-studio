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
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import architectures as arch
from . import config, db, hub
from .scheduler import Fleet

fleet = Fleet()


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    config.ensure_dirs()
    db.connect()
    # Any runner marked online in a previous process is stale until it dials in.
    for r in db.q("SELECT id FROM runners WHERE status != 'offline'"):
        db.mark_runner_offline(r["id"])
    task = asyncio.create_task(fleet.scheduler_loop())
    yield
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


app = FastAPI(title="AI Studio", version="0.1.0", lifespan=lifespan)


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
        fleet.attach(runner_id, ws)
        await ws.send_text(json.dumps({"type": "registered", "runner_id": runner_id}))
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
            requeued = db.requeue_jobs_for_runner(runner_id)
            for jid in requeued:
                db.add_log(jid, "Runner disconnected; job returned to the queue.", "warn")
            await fleet.broadcast_ui({"type": "runners_changed"})
            if requeued:
                await fleet.broadcast_ui({"type": "jobs_changed"})


# ===========================================================================
# Browser event stream
# ===========================================================================

@app.websocket("/api/events")
async def events_ws(ws: WebSocket) -> None:
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

@app.get("/api/status")
async def status() -> dict:
    runners = db.list_runners()
    return {
        "version": "0.1.0",
        "join_token": config.join_token(),
        "hf_token_set": bool(config.HF_TOKEN),
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
async def get_jobs(limit: int = 100) -> list[dict]:
    return db.list_jobs(limit)


@app.post("/api/jobs")
async def create_job(payload: dict = Body(...)) -> dict:
    kind = payload.get("kind", "finetune_llm")
    cfg = payload.get("config") or {}
    if kind == "finetune_llm":
        for field in ("base_model", "dataset"):
            if not cfg.get(field):
                raise HTTPException(400, "Missing required setting: %s" % field)
    elif kind == "pretrain_llm":
        if not cfg.get("dataset"):
            raise HTTPException(400, "Choose some text to learn from.")
        if not cfg.get("arch"):
            raise HTTPException(400, "No model architecture was chosen.")
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
    if config.HF_TOKEN:
        cfg.setdefault("hf_token", config.HF_TOKEN)
    jid = db.create_job(name, kind, cfg)
    db.add_log(jid, "Job created and queued.")
    await fleet.broadcast_ui({"type": "jobs_changed"})
    fleet.wake()
    return {"id": jid}


def _default_job_name(cfg: dict, kind: str = "finetune_llm") -> str:
    data = str(cfg.get("dataset", "data")).split("/")[-1]
    if kind == "pretrain_llm":
        a = cfg.get("arch") or {}
        label = (arch.preset(a.get("size_id", "")) or {}).get("label", "Model")
        return "%s from scratch on %s" % (label, data)
    model = str(cfg.get("base_model", "model")).split("/")[-1]
    return "%s on %s" % (model, data)


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str) -> dict:
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(404, "No such job.")
    job["artifacts"] = db.list_artifacts(job_id)
    job["runner"] = db.get_runner(job["runner_id"]) if job["runner_id"] else None
    # Never hand the runner's copy of the HF token back to the browser.
    job["config"].pop("hf_token", None)
    return job


@app.get("/api/jobs/{job_id}/metrics")
async def job_metrics(job_id: str) -> list[dict]:
    return db.get_metrics(job_id)


@app.get("/api/jobs/{job_id}/logs")
async def job_logs(job_id: str, limit: int = 500) -> list[dict]:
    return db.get_logs(job_id, limit)


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str) -> dict:
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(404, "No such job.")
    if job["status"] in ("succeeded", "failed", "cancelled"):
        return {"ok": True, "already": job["status"]}
    if job["runner_id"] and job["runner_id"] in fleet.connections:
        await fleet.send_to_runner(job["runner_id"],
                                   {"type": "job_cancel", "job_id": job_id})
    else:
        db.set_job_status(job_id, "cancelled")
    db.add_log(job_id, "Cancellation requested.", "warn")
    await fleet.broadcast_ui({"type": "jobs_changed"})
    return {"ok": True}


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
    # run produces a standalone model. Labelling them apart matters because the
    # instructions for using them are completely different.
    job = db.get_job(job_id)
    kind = "model" if (job or {}).get("kind") == "pretrain_llm" else "adapter"
    db.add_artifact(job_id, kind, dest.name, size)
    return {"ok": True, "size": size}


@app.get("/api/jobs/{job_id}/download")
async def download_artifact(job_id: str):
    path = config.ARTIFACT_DIR / ("%s.zip" % job_id)
    if not path.exists():
        raise HTTPException(404, "No result file for this job yet.")
    job = db.get_job(job_id)
    safe = "".join(c for c in (job["name"] if job else job_id)
                   if c.isalnum() or c in "-_ ").strip().replace(" ", "-")
    suffix = "model" if (job or {}).get("kind") == "pretrain_llm" else "adapter"
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

# Peak learning rate by model width. A model starting from noise wants a much
# higher rate than fine-tuning ever uses -- 6e-4 against 2e-4 -- because it has
# far further to travel. It falls off with width because wider layers produce
# larger activations, and the same step size starts to destabilise them.
_SCRATCH_LR = {256: 1.0e-3, 384: 8.0e-4, 512: 6.0e-4, 768: 3.0e-4, 1024: 2.5e-4}

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
                        vocab_size: int = arch.DEFAULT_VOCAB) -> dict:
    """Every model size scored against this machine and this much time.

    Returned together rather than one at a time because the trade-off is the
    whole point. The user needs to see that the small model finishes and the
    large one does not *before* choosing, not eight hours afterwards.
    """
    caps = _caps_for(runner_id)
    return {
        "sizes": arch.size_options(caps, minutes, vocab_size),
        "time_budgets": TIME_BUDGETS,
        "vocab_presets": arch.VOCAB_PRESETS,
        "tokens_per_param_target": arch.TOKENS_PER_PARAM_TARGET,
        "max_scratch_params": arch.max_trainable_params(caps),
    }


@app.post("/api/scratch/plan")
async def scratch_plan(payload: dict = Body(...)) -> dict:
    """Turn a size, a corpus and a time budget into a complete recipe."""
    caps = _caps_for(payload.get("runner_id"))
    size_id = payload.get("size") or "tiny"
    minutes = float(payload.get("minutes") or 60)
    vocab_size = int(payload.get("vocab_size") or arch.DEFAULT_VOCAB)

    architecture = arch.build_arch(size_id, vocab_size)
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
        # Recomputing activations instead of storing them costs about 30% of
        # the speed, and can be the difference between running and not.
        checkpointing = True
        fit = arch.pick_batch_size(architecture, caps.get("vram_gb"),
                                   optim_8bit=optim_8bit, checkpointing=True,
                                   flash=flash)
    if not fit["fits"]:
        raise HTTPException(
            400, "The %s size needs about %.1f GB and this machine has %.1f GB. "
                 "Choose a smaller size."
                 % (size_id, fit["memory"]["total_gb"], caps.get("vram_gb") or 0))

    budget = arch.tokens_in_time(architecture, caps, minutes) or 0

    # A corpus can be smaller than the time budget wants to consume. Reading
    # the same text more than about four times teaches the model to recite it
    # rather than to write, so the budget is trimmed and the reason is stated.
    notes: list[str] = []
    corpus_tokens = payload.get("corpus_tokens")
    if corpus_tokens and budget > corpus_tokens * 4:
        notes.append(
            "This machine could get through %s tokens in the time you have, but "
            "the text you chose only holds about %s. Training stops after four "
            "passes over it, because past that the model starts memorising the "
            "text instead of learning from it."
            % (_human(budget), _human(corpus_tokens)))
        budget = int(corpus_tokens * 4)

    steps = int(max(20, budget // max(fit["tokens_per_step"], 1)))
    lr = _SCRATCH_LR.get(architecture["hidden_size"], 3e-4)
    verdict = arch.training_verdict(architecture, budget)

    settings = {
        "arch": architecture,
        "vocab_size": vocab_size,
        "dtype": caps.get("recommended_dtype", "float32"),
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
        "sample_prompt": payload.get("sample_prompt") or "Once upon a time",
        "text_field": payload.get("text_field") or "text",
    }
    return {
        "settings": settings,
        "params": counts,
        "params_label": arch.fmt_params(counts["total"]),
        "verdict": verdict,
        "notes": notes,
        "memory_gb": fit["memory"]["total_gb"],
        "tokens_per_step": fit["tokens_per_step"],
        "estimated_minutes": round(minutes, 1),
        "minutes_for_full": arch.time_for_tokens(
            architecture, caps, counts["total"] * arch.TOKENS_PER_PARAM_TARGET),
        "explanations": _explain_scratch(settings, counts, verdict, fit),
    }


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
                           "stored. Costs about 30% of the speed."})
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

# The prompt shapes each kind of run was trained on. Handing a fine-tuned model
# a bare question when it learned "### Instruction:" produces rambling
# continuation, and the user reasonably concludes the training failed.
_TEMPLATES = {
    "instruction": {
        "template": "### Instruction:\n{instruction}\n\n### Response:\n",
        # More than just the training template. A fine-tuned instruct model
        # answers correctly and then carries on inventing a conversation,
        # because a LoRA over a few thousand examples learns the shape of a
        # response but never learns to emit an end-of-text token. These are
        # the turn markers the underlying base model falls back on.
        "stop": ["### Instruction:", "\n### ", "Human:", "\nUser:",
                 "Assistant:", "<|im_end|>", "<|endoftext|>"],
    },
    "chat": {
        "template": "user: {instruction}\nassistant:",
        "stop": ["\nuser:", "\nsystem:", "Human:", "<|im_end|>"],
    },
}


def chat_spec(job: dict) -> dict:
    """Everything a runner needs to serve this particular result."""
    cfg = job["config"]
    spec = {"job_id": job["id"], "kind": job["kind"],
            "base_model": cfg.get("base_model")}
    if job["kind"] == "pretrain_llm":
        # A model trained from scratch is a text continuer, not an assistant.
        # It has never seen a question-and-answer shape in its life.
        spec["mode"] = "continue"
        return spec
    mode = (cfg.get("format") or {}).get("mode", "auto")
    shape = _TEMPLATES.get(mode)
    if shape:
        spec["prompt_template"] = shape["template"]
        spec["stop"] = shape["stop"]
        spec["mode"] = "instruct"
    else:
        spec["mode"] = "continue"
    return spec


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
async def playground() -> list[dict]:
    """Finished runs you can talk to."""
    out = []
    for job in db.list_jobs(200):
        if job["status"] != "succeeded":
            continue
        if not (config.ARTIFACT_DIR / ("%s.zip" % job["id"])).exists():
            continue
        cfg = job["config"]
        entry = {
            "id": job["id"], "name": job["name"], "kind": job["kind"],
            "finished_at": job["finished_at"], "dataset": cfg.get("dataset"),
            "base_model": cfg.get("base_model"),
            "mode": chat_spec(job)["mode"],
        }
        if job["kind"] == "pretrain_llm":
            a = cfg.get("arch") or {}
            entry["size"] = (arch.preset(a.get("size_id", "")) or {}).get("label")
        out.append(entry)
    return out


@app.post("/api/jobs/{job_id}/chat")
async def chat(job_id: str, payload: dict = Body(...)) -> dict:
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(404, "No such run.")
    if job["status"] != "succeeded":
        raise HTTPException(400, "This run has not finished successfully.")
    if not (config.ARTIFACT_DIR / ("%s.zip" % job_id)).exists():
        raise HTTPException(400, "This run did not leave a downloadable model.")
    prompt = (payload.get("prompt") or "").strip()
    if not prompt:
        raise HTTPException(400, "Type something first.")

    runner_id, runner = _pick_chat_runner(job)
    spec = chat_spec(job)
    if config.HF_TOKEN:
        spec["hf_token"] = config.HF_TOKEN
    request_id = db.new_id("gen")
    sent = await fleet.send_to_runner(runner_id, {
        "type": "generate", "request_id": request_id, "spec": spec,
        "prompt": prompt,
        "params": {
            "max_new_tokens": min(int(payload.get("max_new_tokens") or 200), 512),
            "temperature": float(payload.get("temperature") or 0.8),
            "top_k": int(payload.get("top_k") or 50),
            "top_p": float(payload.get("top_p") or 0.95),
        },
    })
    if not sent:
        raise HTTPException(503, "That machine dropped off just now. Try again.")
    fleet.generations[request_id] = runner_id
    return {"request_id": request_id, "runner": runner["name"],
            "runner_id": runner_id, "mode": spec["mode"]}


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


@app.get("/", response_class=HTMLResponse)
async def index() -> Any:
    idx = config.WEB_DIR / "index.html"
    if not idx.exists():
        return HTMLResponse("<h1>AI Studio</h1><p>Web UI not found.</p>", status_code=500)
    return FileResponse(idx)
