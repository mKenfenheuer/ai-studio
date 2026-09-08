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
`temperature`, `top_p`, `max_tokens`, `stop`, multi-turn messages, `tools`,
reasoning, and tool calls in both directions. Errors are returned in OpenAI's
error shape, because a client library handed a bare `{"detail": ...}` turns a
clear refusal into a parse error about missing keys.

**Tool calling** works the way a client expects it to. `tools` on the request
are declared to the model in the idiom of the format it was trained in; a call
the model makes comes back as `message.tool_calls` with `finish_reason:
"tool_calls"`; and a `role: "tool"` message with its `tool_call_id` can be sent
straight back to continue the exchange. That last one is the half that is
usually missing: without it the model is handed a conversation in which it
never made the call it is being given a result for.

What it cannot promise is that a call *will* be made. Nothing here constrains
decoding, so `tool_choice` accepts `"auto"` and `"none"` and refuses the rest
rather than pretending. Whether a model calls tools well at all is a property
of what it was trained on -- declare tools to a model whose data had none and
it will ignore them.

**Reasoning** is returned as `reasoning_content` (what vLLM, SGLang and the
DeepSeek API emit, and what most clients read) and as `reasoning`, kept apart
from `content` rather than left inline for the client to strip.

Not implemented, and not faked: `n` above 1, `logprobs`, embeddings. A field
that is accepted and ignored is worse than one that is refused -- it produces a
client that believes it asked for something.

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

from common import conversation

from .. import config, db, serving as spec_for
from .security import current_user

router = APIRouter(prefix="/v1")

# How long one reply may take before the request is abandoned. Generous: a
# long answer from a large model on a busy card is slow, and a client that
# gets an error after waiting is worse off than one that waits.
GENERATE_TIMEOUT_S = 600.0

# Set by the application once the fleet exists.
FLEET = None


def _finish(msg: dict) -> str:
    """Why the model stopped, in OpenAI's vocabulary.

    "length" is not decoration: a client that asked for 200 tokens and got 200
    tokens needs to know whether that was the whole answer or the first part of
    one, and reporting "stop" for both told it the reply was complete when it
    had been cut mid-sentence.
    """
    # A reply the runner cut at its deadline is exactly as incomplete as one
    # cut at the token ceiling, and was being reported as a clean "stop".
    return "length" if msg.get("stop_reason") in ("length", "timeout") else "stop"


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
        if job["kind"] not in spec_for.MODEL_KINDS:
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
    """The request's conversation, canonical.

    Everything the format carries is kept, not just the text: an assistant turn
    that made a tool call, the result that answered it, and the id linking the
    two. Flattening those to `{role, content}` -- as this did -- meant a client
    could send a tool result back and the model would be given a conversation
    in which it had never made the call, so the second turn of every
    tool-calling exchange was nonsense.
    """
    raw = payload.get("messages")
    if not isinstance(raw, list) or not raw:
        raise HTTPException(400, "`messages` must be a non-empty list.")
    for m in raw:
        if not isinstance(m, dict) or "role" not in m:
            raise HTTPException(400, "Each message needs a `role` and `content`.")

    conv = conversation.from_messages(raw, payload.get("tools"))
    conv, _ = conversation.repair(conv)
    out = conv[conversation.MESSAGES_KEY]
    if not out:
        raise HTTPException(400, "There is nothing to answer.")
    # A conversation is answerable if anything but the system prompt is in it.
    # A bare tool result counts: "here is what the function returned, carry on"
    # is a legitimate request and refusing it would break the very loop tool
    # calling exists for.
    if not any(m.get("content", "").strip() or m.get("tool_calls")
               for m in out if m["role"] not in ("system", "developer")):
        raise HTTPException(400, "There is nothing to answer.")
    return out


def _tools(payload: dict) -> list[dict]:
    """Tool definitions from the request, flattened the way the runner wants."""
    raw = payload.get("tools")
    if not isinstance(raw, list):
        return []
    conv = conversation.from_messages([], raw)
    return conversation.flat_tools(conv)


async def _dispatch(job: dict, messages: list[dict], payload: dict) -> tuple:
    """Send the request to a runner and return (request_id, queue)."""
    from ..app import _pick_chat_runner        # local: avoids an import cycle

    runner_id, _runner = _pick_chat_runner(job)
    spec = spec_for.chat_spec(job)
    if config.HF_TOKEN:
        spec["hf_token"] = config.HF_TOKEN
    # Tools travel on the spec because that is where the runner reads them
    # when it renders the prompt -- the same field the playground fills, so a
    # tool declared over this API is declared to the model in exactly the same
    # words as one declared in the browser.
    if tools := _tools(payload):
        spec["tools"] = tools
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
                                      or 512), config.MAX_NEW_TOKENS),
            "temperature": float(payload.get("temperature", 0.8)),
            "top_p": float(payload.get("top_p", 0.95)),
            "top_k": int(payload.get("top_k") or 50),
            "reasoning": bool(payload.get("reasoning")),
            "deadline_s": config.GENERATION_DEADLINE_S,
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
            ("logprobs", "Token probabilities are not available.")):
        if payload.get(unsupported) not in (None, False, 1):
            return _error(400, "`%s` is not supported. %s" % (unsupported, why))

    # `tool_choice` is honoured only where it can be. Nothing here constrains
    # decoding, so "you must call a tool" cannot be promised -- and a request
    # option that is accepted and quietly ignored produces a client that
    # believes it asked for something.
    choice = payload.get("tool_choice")
    if isinstance(choice, dict) or choice not in (None, "auto", "none", "required"):
        return _error(400, "`tool_choice` may be \"auto\" or \"none\". Naming a "
                           "tool, or requiring one, would need constrained "
                           "decoding, which this server does not do.")
    if choice == "required":
        return _error(400, "`tool_choice: \"required\"` cannot be honoured: "
                           "nothing here constrains what the model emits, so a "
                           "tool call cannot be guaranteed.")
    if choice == "none":
        # Not an error -- the plain meaning is "answer without tools", and the
        # way to do that is not to declare any.
        payload = {**payload, "tools": []}

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
            # A reply that may contain a tool call cannot be streamed as it
            # arrives. The call is recognised by syntax spanning many tokens --
            # `<tool_call>{...}</tool_call>` and its equivalents -- so
            # streaming the text through would deliver that syntax to the
            # client as the assistant's *content*, and then deliver the same
            # call again, parsed, at the end. Where tools are declared the
            # reply is therefore held until it is whole. Where they are not,
            # nothing can appear that needs parsing, and it streams token by
            # token as before.
            _stream(rid, queue, job, created, buffer=bool(payload.get("tools"))),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    return await _collect(rid, queue, job, created)


def _shell(job: dict, created: int, rid: str) -> dict:
    return {"id": "chatcmpl-" + rid, "object": "chat.completion",
            "created": created, "model": job["id"]}


def _tool_calls(msg: dict) -> list[dict]:
    """The runner's parsed calls, in the exact shape OpenAI clients unpack.

    The runner reports whether each call's arguments parse. A malformed call is
    still returned rather than dropped -- a client that gets nothing back
    cannot tell a model that made no call from one whose call was thrown away,
    and the second is the one worth knowing about.
    """
    out = []
    for i, call in enumerate(msg.get("tool_calls") or []):
        fn = call.get("function") or {}
        out.append({"id": call.get("id") or "call_%d" % (i + 1),
                    "type": "function",
                    "function": {"name": fn.get("name") or "",
                                 "arguments": fn.get("arguments") or ""}})
    return out


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
                # The explanation, plus the numbers behind it. A caller
                # holding a 502 cannot see the runner's log or the run's, and
                # "it ran out of memory" without the size of the conversation
                # that did it is not something anybody can act on.
                facts = msg.get("diagnostics") or {}
                where = ", ".join("%s=%s" % kv for kv in sorted(facts.items()))
                return _error(502, "%s%s" % (
                    msg.get("error") or "Generation failed.",
                    (" (%s)" % where) if where else ""), "upstream_error")
            elif kind == "generate_done":
                calls = _tool_calls(msg)
                # The runner's parsed text is authoritative when it sends one,
                # *including when it is empty*. Falling back to the accumulated
                # deltas on a falsy value -- as this did -- undid the parsing
                # completely for the one case that matters: a reply that is
                # nothing but a tool call has no content, and the deltas are
                # the raw call syntax the parser had just finished removing.
                # So a client got the call twice, once as structure and once as
                # a line of JSON above it.
                whole = msg["text"] if "text" in msg else "".join(text)
                message = {"role": "assistant", "content": whole or None}
                if reasoning := msg.get("reasoning"):
                    # Both spellings. `reasoning_content` is what vLLM, SGLang
                    # and the DeepSeek API emit and what most clients read;
                    # `reasoning` is what the Responses API calls it. Sending
                    # both costs a few bytes and saves every client a
                    # translation.
                    message["reasoning_content"] = reasoning
                    message["reasoning"] = reasoning
                if calls:
                    message["tool_calls"] = calls
                return {
                    **_shell(job, created, rid),
                    "choices": [{
                        "index": 0,
                        "message": message,
                        # A reply that ends in a tool call has not finished
                        # answering -- it is waiting for a result. A client
                        # driving a tool loop branches on exactly this, so
                        # reporting "stop" would stall the loop at the first
                        # call.
                        "finish_reason": "tool_calls" if calls
                                         else _finish(msg),
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


async def _stream(rid: str, queue: asyncio.Queue, job: dict, created: int,
                  buffer: bool = False):
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
                if not buffer:
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
                # Reasoning and tool calls are only known once the reply is
                # whole -- both are recognised by syntax that spans many
                # tokens, so neither can honestly be streamed as it arrives.
                # They are delivered in a final delta before the stop, which is
                # a shape every client already handles.
                if buffer and (whole := msg.get("text")):
                    yield chunk({"content": whole})
                if reasoning := msg.get("reasoning"):
                    yield chunk({"reasoning_content": reasoning,
                                 "reasoning": reasoning})
                calls = _tool_calls(msg)
                if calls:
                    yield chunk({"tool_calls": [
                        {**c, "index": i} for i, c in enumerate(calls)]})
                yield chunk({}, "tool_calls" if calls else _finish(msg))
                yield "data: [DONE]\n\n"
                return
    except asyncio.TimeoutError:
        yield chunk({"content": "\n\n[error: timed out]"})
        yield chunk({}, "stop")
        yield "data: [DONE]\n\n"
    finally:
        FLEET.waiters.pop(rid, None)
