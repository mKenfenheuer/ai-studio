"""An OpenAI-compatible API, so a trained model can be used by other software.

Until now a model trained here could be talked to in this app's own playground
or downloaded as a zip, and that was all. That is a strange place to stop: the
whole point of fine-tuning a small model on your own data is to put it behind
something -- a home automation system, a script, an editor plugin -- and every
one of those things already speaks one protocol.

So this is that protocol, and deliberately not a new one. `/v1/chat/completions`
with a bearer key is the shape every client library, every self-hosted tool and
every "OpenAI base URL" field already expects. A new and better-designed API
would be worse, because nothing would be pointed at it.

## What is and is not implemented

Implemented: `/v1/models`, `/v1/chat/completions` with and without streaming,
`temperature`, `top_p`, `max_tokens`, `stop`, and multi-turn messages. Errors
are returned in OpenAI's error shape, because a client library handed a bare
`{"detail": ...}` turns a clear refusal into a parse error about missing keys.

Not implemented, and not faked: `n` above 1, `logprobs`, function calling as a
protocol feature, embeddings. A field that is accepted and ignored is worse
than one that is refused -- it produces a client that believes it asked for
something.

Token counts are real counts from the runner, not estimates. `prompt_tokens`
is what the model was actually given after the run's own chat template was
applied, which is usually not what a client would compute from the message
text.
"""
from __future__ import annotations

import asyncio
import json
import re
import time

from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .. import config, db, serving as spec_for
from .security import current_user

router = APIRouter(prefix="/v1")

# How long one reply may take before the request is abandoned. Generous: a
# long answer from a large model on a busy card is slow, and a client that
# gets an error after waiting is worse off than one that waits.
GENERATE_TIMEOUT_S = 600.0

# Set by the application once the fleet exists.
FLEET = None


def _error(status: int, message: str, code: str = "invalid_request_error"):
    return JSONResponse({"error": {"message": message, "type": code,
                                   "code": code}}, status_code=status)


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")


def _servable(user: dict) -> list[dict]:
    """Runs this caller may use, that have a model behind them."""
    out = []
    for job in db.visible_jobs(user, 300):
        if job["status"] not in ("succeeded", "cancelled"):
            continue
        if not (config.ARTIFACT_DIR / ("%s.zip" % job["id"])).exists():
            continue
        if job["kind"] not in ("pretrain_llm", "finetune_llm", "merge_adapter"):
            continue
        out.append(job)
    return out


def _resolve(user: dict, wanted: str) -> dict | None:
    """Find the run a client means by `model`.

    Four ways, in order of how specific they are: the run's id, its exact
    name, its name ignoring case, and a slug of its name. Clients put this in
    a config file and type it by hand, and being strict about a capital letter
    would buy nothing.
    """
    jobs = _servable(user)
    wanted = (wanted or "").strip()
    if not wanted:
        return None
    for match in (lambda j: j["id"] == wanted,
                  lambda j: j["name"] == wanted,
                  lambda j: j["name"].lower() == wanted.lower(),
                  lambda j: _slug(j["name"]) == _slug(wanted)):
        found = [j for j in jobs if match(j)]
        if len(found) == 1:
            return found[0]
        if len(found) > 1:
            # Two runs with the same name is normal here. The id is the only
            # thing guaranteed unique, so say that rather than picking one.
            raise HTTPException(
                400, "More than one run is called %r. Use its id instead -- "
                     "/v1/models lists them." % wanted)
    return None


@router.get("/models")
async def list_models(request: Request) -> dict:
    """Every run this key can serve, in OpenAI's model-list shape."""
    user = current_user(request)
    out = []
    for job in _servable(user):
        summary = job.get("summary") or {}
        out.append({
            "id": job["id"],
            "object": "model",
            "created": int(job.get("finished_at") or job["created_at"]),
            "owned_by": job.get("owner_username") or "ai-studio",
            # Everything below is ours, not OpenAI's. Clients ignore unknown
            # fields, and a person reading this by hand needs to know which
            # run each id is.
            "name": job["name"],
            "alias": _slug(job["name"]),
            "kind": job["kind"],
            "base_model": (job.get("config") or {}).get("base_model"),
            "held_out_loss": summary.get("best_val_loss"),
        })
    return {"object": "list", "data": out}


def _messages(payload: dict) -> list[dict]:
    raw = payload.get("messages")
    if not isinstance(raw, list) or not raw:
        raise HTTPException(400, "`messages` must be a non-empty list.")
    out = []
    for m in raw:
        if not isinstance(m, dict) or "role" not in m:
            raise HTTPException(400, "Each message needs a `role` and `content`.")
        content = m.get("content")
        if isinstance(content, list):
            # The multi-part content form. Only the text parts mean anything
            # to a text model, and silently dropping an image would be worse
            # than joining what is there.
            content = "".join(p.get("text", "") for p in content
                              if isinstance(p, dict))
        out.append({"role": str(m["role"]), "content": str(content or "")})
    if not any(m["content"].strip() for m in out if m["role"] != "system"):
        raise HTTPException(400, "There is nothing to answer.")
    return out


async def _dispatch(job: dict, messages: list[dict], payload: dict) -> tuple:
    """Send the request to a runner and return (request_id, queue)."""
    from ..app import _pick_chat_runner        # local: avoids an import cycle

    runner_id, _runner = _pick_chat_runner(job)
    spec = spec_for.chat_spec(job)
    if config.HF_TOKEN:
        spec["hf_token"] = config.HF_TOKEN
    stops = payload.get("stop")
    if isinstance(stops, str):
        stops = [stops]
    if stops:
        spec["stop"] = [str(x) for x in stops][:4]

    rid = db.new_id("gen")
    queue: asyncio.Queue = asyncio.Queue()
    FLEET.waiters[rid] = queue
    sent = await FLEET.send_to_runner(runner_id, {
        "type": "generate", "request_id": rid, "spec": spec,
        "messages": messages,
        "params": {
            "max_new_tokens": min(int(payload.get("max_tokens")
                                      or payload.get("max_completion_tokens")
                                      or 256), 2048),
            "temperature": float(payload.get("temperature", 0.8)),
            "top_p": float(payload.get("top_p", 0.95)),
            "top_k": int(payload.get("top_k") or 50),
            "reasoning": bool(payload.get("reasoning")),
        },
    })
    if not sent:
        FLEET.waiters.pop(rid, None)
        raise HTTPException(503, "That machine dropped off just now. Try again.")
    FLEET.generations[rid] = runner_id
    return rid, queue


@router.post("/chat/completions")
async def chat_completions(request: Request, payload: dict = Body(...)):
    user = current_user(request)
    for unsupported, why in (
            ("n", "Only one reply per request is produced."),
            ("logprobs", "Token probabilities are not available."),
            ("tools", "Tool calling is a property of how a model was trained "
                      "here, not a request option. Train with tools in the "
                      "data and the model will emit them.")):
        if payload.get(unsupported) not in (None, False, 1):
            return _error(400, "`%s` is not supported. %s" % (unsupported, why))

    wanted = str(payload.get("model") or "")
    try:
        job = _resolve(user, wanted)
    except HTTPException as e:
        return _error(e.status_code, e.detail)
    if not job:
        return _error(404, "No model called %r. GET /v1/models lists what this "
                           "key can use." % wanted, "model_not_found")

    try:
        messages = _messages(payload)
    except HTTPException as e:
        return _error(e.status_code, e.detail)

    try:
        rid, queue = await _dispatch(job, messages, payload)
    except HTTPException as e:
        return _error(e.status_code, e.detail)

    created = int(time.time())
    if payload.get("stream"):
        return StreamingResponse(
            _stream(rid, queue, job, created),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    return await _collect(rid, queue, job, created)


def _shell(job: dict, created: int, rid: str) -> dict:
    return {"id": "chatcmpl-" + rid, "object": "chat.completion",
            "created": created, "model": job["id"]}


async def _collect(rid: str, queue: asyncio.Queue, job: dict, created: int):
    """Wait for the whole reply and answer once."""
    text: list[str] = []
    try:
        while True:
            msg = await asyncio.wait_for(queue.get(), GENERATE_TIMEOUT_S)
            kind = msg.get("type")
            if kind == "generate_delta":
                text.append(msg.get("delta") or "")
            elif kind == "generate_error":
                return _error(502, msg.get("error") or "Generation failed.",
                              "upstream_error")
            elif kind == "generate_done":
                whole = msg.get("text") or "".join(text)
                return {
                    **_shell(job, created, rid),
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": whole,
                                    **({"reasoning": msg["reasoning"]}
                                       if msg.get("reasoning") else {})},
                        "finish_reason": "stop",
                    }],
                    "usage": {
                        "prompt_tokens": msg.get("prompt_tokens") or 0,
                        "completion_tokens": msg.get("tokens") or 0,
                        "total_tokens": (msg.get("prompt_tokens") or 0)
                                        + (msg.get("tokens") or 0),
                    },
                }
    except asyncio.TimeoutError:
        return _error(504, "The model did not finish in %d seconds."
                      % GENERATE_TIMEOUT_S, "timeout")
    finally:
        FLEET.waiters.pop(rid, None)


async def _stream(rid: str, queue: asyncio.Queue, job: dict, created: int):
    """Server-sent events, in the exact chunk shape OpenAI clients parse."""
    def chunk(delta: dict, finish=None) -> str:
        body = {**_shell(job, created, rid), "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": delta,
                             "finish_reason": finish}]}
        return "data: %s\n\n" % json.dumps(body)

    try:
        yield chunk({"role": "assistant", "content": ""})
        while True:
            msg = await asyncio.wait_for(queue.get(), GENERATE_TIMEOUT_S)
            kind = msg.get("type")
            if kind == "generate_delta":
                yield chunk({"content": msg.get("delta") or ""})
            elif kind == "generate_error":
                # There is no error frame in this protocol once the stream has
                # started, so the failure is delivered as the reply -- silence
                # would look like a model with nothing to say.
                yield chunk({"content": "\n\n[error: %s]"
                             % (msg.get("error") or "generation failed")})
                yield chunk({}, "stop")
                yield "data: [DONE]\n\n"
                return
            elif kind == "generate_done":
                yield chunk({}, "stop")
                yield "data: [DONE]\n\n"
                return
    except asyncio.TimeoutError:
        yield chunk({"content": "\n\n[error: timed out]"})
        yield chunk({}, "stop")
        yield "data: [DONE]\n\n"
    finally:
        FLEET.waiters.pop(rid, None)
