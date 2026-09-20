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

**`response_format` is enforced, not requested.** `json_object` and
`json_schema` are kept by constraining the sampler: at every step the tokens
that would break the grammar are scored at negative infinity before anything is
chosen, so the reply could not have been written in any other shape. It is not
checked afterwards and never repaired. The schema is also shown to the model,
which changes nothing about validity and a great deal about whether the fields
are filled with the answer or with a guess. `tools` and `reasoning` are refused
alongside it, because both would need the model to write something the grammar
forbids. A machine whose image has no grammar engine is not sent these requests
at all, and if no machine has one the request is refused saying so.

**Concurrency.** A runner holds one model on one card and answers one message
at a time. Requests that arrive meanwhile are QUEUED, not refused -- an
evaluation sending sixty prompts gets sixty answers, one after another, rather
than one answer and fifty-nine errors. The queue is bounded; past its depth the
answer is `429` with a `Retry-After`, which is the status a client library
knows how to back off from, and not the `502` that tells it to give up.

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
from .security import current_user, require_view

router = APIRouter(prefix="/v1")

# The registry and the usage figures are the studio's own surface, not
# OpenAI's, so they live behind /api with the rest of the app's routes -- but
# in this file, next to the code that reads them, because a name that resolves
# differently from the way it is registered is the bug this exists to prevent.
registry = APIRouter(prefix="/api")

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


def _error(status: int, message: str, code: str = "invalid_request_error",
           retry_after: float | None = None):
    """OpenAI's error shape, and the one header a refusal sometimes carries.

    `Retry-After` is not decoration on a 429. Without it a client library
    backs off on whatever schedule it invented, which for the ones that
    invented "immediately" means a refused request becomes a refused request
    per millisecond.
    """
    headers = ({"Retry-After": str(max(1, int(round(retry_after))))}
               if retry_after else None)
    return JSONResponse({"error": {"message": message, "type": code,
                                   "code": code}}, status_code=status,
                        headers=headers)


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


def _in_scope(request: Request, jobs: list[dict]) -> list[dict]:
    """The runs a scoped key may reach: by run id, or by a name it serves.

    A key made for `assistant-prod` follows the name when it is repointed --
    that is what a name is for -- and reaches nothing else, including the run
    the name used to point at.
    """
    scope = getattr(request.state, "api_key_scope", None)
    if not scope:
        return jobs
    allowed = set(scope)
    out = []
    for j in jobs:
        if j["id"] in allowed or any(a in allowed for a in db.aliases_for_job(j["id"])):
            out.append(j)
    return out


def _resolve(user: dict, wanted: str, request: Request | None = None) -> dict | None:
    """Find the run a client means by `model`.

    Five ways, in order of how specific they are: a registered alias, the
    run's id, its exact name, its name ignoring case, and a slug of its name.
    Clients put this in a config file and type it by hand, and being strict
    about a capital letter would buy nothing.

    The alias comes first on purpose. That is the whole point of registering
    one: `assistant-prod` has to mean whatever it currently points at, even
    when some run in the studio happens to be called the same thing. An alias
    whose run this caller cannot use falls through to the other four rather
    than saying so, because "that name exists but is not yours" is a fact
    about somebody else's work.
    """
    jobs = _servable(user)
    if request is not None:
        jobs = _in_scope(request, jobs)
    wanted = (wanted or "").strip()
    if not wanted:
        return None
    if alias := db.get_alias(wanted):
        if job := next((j for j in jobs if j["id"] == alias["job_id"]), None):
            return job
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


def _model_entry(job: dict, aliases: list[str]) -> dict:
    summary = job.get("summary") or {}
    return {
        "id": job["id"],
        "object": "model",
        "created": int(job.get("finished_at") or job["created_at"]),
        "owned_by": job.get("owner_username") or "ai-studio",
        # Everything below is ours, not OpenAI's. Clients ignore unknown
        # fields, and a person reading this by hand needs to know which
        # run each id is.
        "name": job["name"],
        "alias": _slug(job["name"]),
        "aliases": aliases,
        "kind": job["kind"],
        "base_model": (job.get("config") or {}).get("base_model"),
        "held_out_loss": summary.get("best_val_loss"),
    }


@router.get("/models")
async def list_models(request: Request) -> dict:
    """Every run this key can serve, in OpenAI's model-list shape.

    Registered aliases are listed as models in their own right, ahead of the
    runs, because an alias is the name a client should be configured with:
    it survives a rename and it moves to next month's model without anything
    outside this studio being edited.
    """
    user = current_user(request)
    jobs = _in_scope(request, _servable(user))
    by_job: dict[str, list[str]] = {}
    for row in db.list_aliases():
        by_job.setdefault(row["job_id"], []).append(row["alias"])

    out = []
    for row in db.list_aliases():
        job = next((j for j in jobs if j["id"] == row["job_id"]), None)
        if not job:
            continue
        entry = _model_entry(job, by_job.get(job["id"]) or [])
        out.append({**entry, "id": row["alias"], "run_id": job["id"],
                    "is_alias": True,
                    "stage": row.get("stage") or "",
                    "created": int(row.get("updated_at") or entry["created"])})
    for job in jobs:
        out.append(_model_entry(job, by_job.get(job["id"]) or []))
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
               or m.get("media")
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


class TooManyRequests(HTTPException):
    """A 429 that carries the Retry-After a client should honour.

    Its own class because the header has to survive the trip through the
    handler that converts an HTTPException into OpenAI's error shape, and a
    status code with no header is a client that retries immediately and is
    refused again.
    """

    def __init__(self, detail: str, retry_after: float) -> None:
        super().__init__(429, detail)
        self.retry_after = max(1, int(round(retry_after)))


async def _dispatch(job: dict, messages: list[dict], payload: dict) -> tuple:
    """Send the request to a runner and return (request_id, queue, runner_id)."""
    from ..app import _pick_chat_runner        # local: avoids an import cycle

    fmt = payload.get("response_format") or {}
    constrained = (fmt.get("type") or "text") != "text"
    runner_id, _runner = _pick_chat_runner(job, needs_grammar=constrained)
    spec = spec_for.chat_spec(job)
    if config.HF_TOKEN:
        spec["hf_token"] = config.HF_TOKEN
    if constrained:
        # Travels on the spec, beside `tools` and `stop`, because the runner
        # needs it before it renders the prompt: the schema is shown to the
        # model as well as enforced on the sampler, and the two have to be the
        # same schema.
        spec["response_format"] = fmt
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
    # Registered BEFORE the request is submitted, because a queued request is
    # told where it stands the moment it joins the line -- and a waiter that
    # does not exist yet would miss that frame.
    FLEET.waiters[rid] = queue
    placed = await FLEET.submit_generation(runner_id, rid, {
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
    if placed == "gone":
        FLEET.waiters.pop(rid, None)
        raise HTTPException(503, "That machine dropped off just now. Try again.")
    if placed == "full":
        FLEET.waiters.pop(rid, None)
        # 429, not 502. The difference matters more than it looks: 502 tells a
        # client library the upstream is broken, and the correct response to
        # that is to stop. This is "come back shortly", which every client
        # already knows how to do, and Retry-After says how shortly.
        raise TooManyRequests(
            "This machine already has %d requests waiting. It answers one at "
            "a time; try again shortly."
            % FLEET.queue_depth(runner_id),
            retry_after=config.SERVING_QUEUE_RETRY_S)
    return rid, queue, runner_id


def unsupported_options(payload: dict):
    """The request options this server will not pretend to honour.

    A JSONResponse to return as-is, or None when there is nothing to refuse.
    Pulled out of the route so it can be checked without a running controller,
    a signed-in user and a machine to send the request to -- which is what it
    took before, and is why the one field missing from it stayed missing.
    """
    # Checked one at a time rather than in a loop over a tuple of allowed
    # values, which is how this was written and why `logprobs: true` was never
    # actually refused: in Python `True == 1`, so a boolean sailed through a
    # membership test whose `1` was meant for `n`. The field this file exists
    # to refuse was accepted and ignored by the code that refuses fields.
    try:
        if int(payload.get("n") or 1) != 1:
            return _error(400, "`n` is not supported. Only one reply per "
                               "request is produced.")
    except (TypeError, ValueError):
        return _error(400, "`n` must be a number, and only 1 is supported.")
    if payload.get("logprobs") or payload.get("top_logprobs"):
        return _error(400, "`logprobs` is not supported. Token probabilities "
                           "are not available.")

    # `response_format` carries two promises -- `json_object` promises the
    # reply parses as JSON, `json_schema` promises it matches a schema -- and
    # both are now kept, by constraining what the sampler may pick rather than
    # by asking the model nicely. See runner/grammar.py.
    #
    # What is refused here is the shape of the field, and the two requests
    # that ask for a format and something incompatible with it in the same
    # breath. Whether the MACHINE can enforce it is a separate question with a
    # separate answer, asked once a machine has been chosen.
    fmt = payload.get("response_format")
    if fmt is not None and not isinstance(fmt, dict):
        return _error(400, "`response_format` must be an object, such as "
                           "{\"type\": \"text\"}.")
    kind = (fmt or {}).get("type") or "text"
    if kind not in ("text", "json_object", "json_schema"):
        return _error(400, "`response_format.type` may be \"text\", "
                           "\"json_object\" or \"json_schema\"; this "
                           "request asked for %r." % kind)
    if kind == "json_schema":
        block = (fmt or {}).get("json_schema")
        schema = block.get("schema") if isinstance(block, dict) else None
        if not isinstance(schema, dict):
            return _error(400, "`response_format: \"json_schema\"` needs the "
                               "schema at `response_format.json_schema."
                               "schema`. Use `json_object` to ask only for "
                               "valid JSON of any shape.")
    if kind != "text":
        # Both of these would be honoured by producing something the grammar
        # forbids, so one of the two promises would have to give way. Which
        # one is not ours to choose quietly.
        if payload.get("tools"):
            return _error(400, "`tools` and `response_format` cannot both be "
                               "asked for: a tool call is not a value of the "
                               "format the reply is being held to, so the "
                               "model would be unable to make one.")
        if payload.get("reasoning"):
            return _error(400, "`reasoning` and `response_format` cannot both "
                               "be asked for: the reply is constrained to the "
                               "format from its first token, which leaves no "
                               "room for the model to work through anything "
                               "first.")

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
    # "none" is not refused: it means "answer without tools", which this can
    # do. The route acts on it.
    return None


@router.post("/chat/completions")
async def chat_completions(request: Request, payload: dict = Body(...)):
    user = current_user(request)
    if refusal := unsupported_options(payload):
        return refusal
    if payload.get("tool_choice") == "none":
        # Not an error -- the plain meaning is "answer without tools", and the
        # way to do that is not to declare any. Done here rather than in the
        # check above, because it changes the request instead of refusing it.
        payload = {**payload, "tools": []}

    wanted = str(payload.get("model") or "")
    try:
        job = _resolve(user, wanted, request)
    except HTTPException as e:
        return _error(e.status_code, e.detail)
    if not job:
        return _error(404, "No model called %r. GET /v1/models lists what this "
                           "key can use." % wanted, "model_not_found")

    # Who to bill this reply to, and under which name it was asked for. The
    # alias matters as much as the run: a name serving ten thousand calls a
    # day is a fact about a deployment, and after it has been repointed twice
    # the run ids underneath it tell you nothing.
    named = wanted.strip().lower()
    hit = db.get_alias(named)
    who = {"user_id": user["id"],
           "api_key_id": getattr(request.state, "api_key_id", "") or None,
           "alias": named if hit and hit["job_id"] == job["id"] else None}

    try:
        messages = _messages(payload)
    except HTTPException as e:
        return _error(e.status_code, e.detail)

    started = time.time()
    try:
        rid, queue, who["runner_id"] = await _dispatch(job, messages, payload)
    except HTTPException as e:
        # A request that never reached a machine is still a request that
        # failed, and it is the failure most worth seeing: it means the fleet
        # had nothing free, which no count of successful replies would show.
        _record(job, who, {}, started, stream=bool(payload.get("stream")),
                failed=str(e.detail))
        return _error(e.status_code, e.detail,
                      "rate_limit_exceeded" if e.status_code == 429
                      else "invalid_request_error",
                      retry_after=getattr(e, "retry_after", None))

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
            _stream(rid, queue, job, created, who,
                    buffer=bool(payload.get("tools"))),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    return await _collect(rid, queue, job, created, who)


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


def _record(job: dict, who: dict, msg: dict, started: float,
            stream: bool, failed: str = "") -> None:
    """File what this reply cost, or that it did not happen.

    `failed` carries the reason when there was no reply. A failure is a row
    like any other, with no tokens and a status: a table of successes says a
    studio in which nothing goes wrong, and "is it erroring" is the first
    question anybody asks of something other software depends on.

    Never raises. A ledger entry that fails must not turn a reply the model
    has already produced into a 500 -- the numbers are worth having and they
    are not worth that.
    """
    try:
        db.record_usage(job["id"],
                        0 if failed else (msg.get("prompt_tokens") or 0),
                        0 if failed else (msg.get("tokens") or 0),
                        user_id=who.get("user_id"),
                        api_key_id=who.get("api_key_id"),
                        alias=who.get("alias"),
                        # The runner's own measure of time on the card where it
                        # sent one, so the rate quoted is the model's speed and
                        # not the network's. Wall clock is the fallback.
                        seconds=msg.get("seconds")
                                or round(time.time() - started, 3),
                        stream=stream,
                        status="error" if failed else "ok",
                        error=failed or None,
                        source="api",
                        runner_id=who.get("runner_id"))
    except Exception:  # noqa: BLE001 - see the docstring
        pass


async def _collect(rid: str, queue: asyncio.Queue, job: dict, created: int,
                   who: dict):
    """Wait for the whole reply and answer once."""
    text: list[str] = []
    started = time.time()
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
                why = msg.get("error") or "Generation failed."
                _record(job, who, msg, started, stream=False, failed=why)
                return _error(502, "%s%s" % (
                    why, (" (%s)" % where) if where else ""), "upstream_error")
            elif kind == "generate_done":
                _record(job, who, msg, started, stream=False)
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
        why = "No reply within %d seconds." % GENERATE_TIMEOUT_S
        _record(job, who, {}, started, stream=False, failed=why)
        return _error(504, "The model did not finish in %d seconds."
                      % GENERATE_TIMEOUT_S, "timeout")
    finally:
        FLEET.waiters.pop(rid, None)


async def _stream(rid: str, queue: asyncio.Queue, job: dict, created: int,
                  who: dict, buffer: bool = False):
    """Server-sent events, in the exact chunk shape OpenAI clients parse."""
    started = time.time()
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
                why = msg.get("error") or "generation failed"
                _record(job, who, msg, started, stream=True, failed=why)
                yield chunk({"content": "\n\n[error: %s]" % why})
                yield chunk({}, "stop")
                yield "data: [DONE]\n\n"
                return
            elif kind == "generate_done":
                _record(job, who, msg, started, stream=True)
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
        _record(job, who, {}, started, stream=True,
                failed="No reply within %d seconds." % GENERATE_TIMEOUT_S)
        yield chunk({"content": "\n\n[error: timed out]"})
        yield chunk({}, "stop")
        yield "data: [DONE]\n\n"
    finally:
        FLEET.waiters.pop(rid, None)


# --------------------------------------------------------------- registry

# What an alias may be called. Lowercase because it is typed into config files
# and compared exactly; no leading "job_" because that is what a run id looks
# like and a name that could be either is a name nobody can reason about.
ALIAS_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{1,63}$")

# Labels, not a workflow. Nothing here enforces an order or a promotion path:
# a studio of four people does not need approval gates, and the useful part of
# a stage is that a person reading /v1/models can see which name is the real
# one.
STAGES = ("production", "staging", "experiment")


def _alias_or_404(alias: str) -> dict:
    row = db.get_alias(alias)
    if not row:
        raise HTTPException(404, "No model is registered under that name.")
    return row


def _may_change(request: Request, row: dict | None) -> None:
    """Who may repoint or remove a name.

    Its owner, or an administrator. Not everyone who can see the run behind
    it: an alias is what other software is pointed at, and repointing one is
    changing what a running system answers with.
    """
    user = current_user(request)
    if not row or not row.get("owner_id"):
        return
    if row["owner_id"] == user["id"] or user.get("role") == "admin":
        return
    raise HTTPException(
        403, "That name belongs to somebody else. Ask them to repoint it, or "
             "register one of your own.")


@registry.get("/models")
async def registered_models(request: Request) -> list[dict]:
    """Every registered name, and what it currently points at."""
    user = current_user(request)
    out = []
    for row in db.list_aliases():
        job = db.get_job(row["job_id"])
        visible = bool(job) and bool(
            db.access_level("job", row["job_id"], job.get("owner_id"), user))
        owner = db.get_user(row["owner_id"]) if row.get("owner_id") else None
        out.append({
            "alias": row["alias"],
            "stage": row.get("stage") or "",
            "notes": row.get("notes") or "",
            "updated_at": row["updated_at"],
            "created_at": row["created_at"],
            "mine": row.get("owner_id") == user["id"],
            "owner": db.public_user(owner) if owner else None,
            # A name registered against a run this caller cannot see is still
            # a name that is taken. What it points at is not theirs to know.
            "job_id": row["job_id"] if visible else "",
            "job_name": job["name"] if visible and job else "",
            "job_status": job["status"] if visible and job else "",
            "job_gone": job is None,
            "visible": visible,
            "history": [h for h in row.get("history") or []][:5] if visible else [],
        })
    return out


@registry.put("/models/{alias}")
async def register_model(request: Request, alias: str,
                         payload: dict = Body(...)) -> dict:
    """Point a name at a run, or move it to a different one."""
    user = current_user(request)
    alias = (alias or "").strip().lower()
    if not ALIAS_RE.match(alias):
        raise HTTPException(
            400, "A name is 2 to 64 characters of lowercase letters, digits, "
                 "dot, dash or underscore -- \"assistant-prod\", not \"My "
                 "Model\". It is typed into other software's configuration, "
                 "which is why it is strict.")
    if alias.startswith("job_"):
        raise HTTPException(400, "Names starting with \"job_\" look like run "
                                 "ids. Choose something else.")
    job_id = (payload.get("job_id") or "").strip()
    job = db.get_job(job_id) if job_id else None
    if not job:
        raise HTTPException(404, "No such run.")
    require_view(request, "job", job)
    if job["kind"] not in spec_for.MODEL_KINDS:
        raise HTTPException(400, "That run did not produce a model to serve.")
    if not (config.ARTIFACT_DIR / ("%s.zip" % job_id)).exists():
        raise HTTPException(400, "That run has no saved model, so nothing "
                                 "could be served under this name.")

    existing = db.get_alias(alias)
    _may_change(request, existing)
    stage = (payload.get("stage") or "").strip().lower()
    if stage and stage not in STAGES:
        raise HTTPException(400, "A stage is one of: %s." % ", ".join(STAGES))
    row = db.set_alias(alias, job_id,
                       owner_id=(existing or {}).get("owner_id") or user["id"],
                       stage=stage, notes=payload.get("notes"),
                       by=user["id"])
    moved = bool(existing) and existing["job_id"] != job_id
    if moved:
        # On the run, not only in the alias's own history: the run page is
        # where somebody looks to find out why a model started being served.
        db.add_log(job_id, "Serving as \"%s\" from now on." % alias)
        db.add_log(existing["job_id"],
                   "No longer served as \"%s\"; it now points at %s."
                   % (alias, job["name"]))
    return {**row, "moved": moved}


@registry.delete("/models/{alias}")
async def unregister_model(request: Request, alias: str) -> dict:
    row = _alias_or_404(alias)
    _may_change(request, row)
    db.delete_alias(row["alias"])
    return {"ok": True, "note": "Anything configured with that name will stop "
                                "working; the run itself is untouched."}


# ------------------------------------------------------------------ usage

@registry.get("/usage")
async def usage(request: Request, days: int = 30) -> dict:
    """What has been served, and to whom.

    Scoped to the caller unless they are an administrator, in which case it is
    the whole studio -- there is no third answer that is useful, and a
    per-user report somebody cannot see the total of is not a report.
    """
    user = current_user(request)
    days = max(1, min(int(days or 30), db.USAGE_DAYS))
    since = time.time() - days * 86400
    whose = None if user.get("role") == "admin" else user["id"]

    keys = {k["id"]: k for k in db.list_api_keys(user["id"])}
    by_key = []
    for row in db.usage_by("api_key_id", whose, since):
        key = keys.get(row["key"])
        by_key.append({**row,
                       "name": key["name"] if key else
                               ("the browser" if not row["key"] else "a key"),
                       "prefix": key["prefix"] if key else ""})
    by_model = []
    for row in db.usage_by("job_id", whose, since):
        job = db.get_job(row["key"] or "")
        by_model.append({**row, "name": (job or {}).get("name") or row["key"],
                         "job_id": row["key"],
                         "gone": job is None})
    return {
        "days": days,
        "scope": "studio" if whose is None else "you",
        "totals": db.usage_totals(whose, since),
        "by_key": by_key,
        "by_model": by_model,
        "by_alias": [r for r in db.usage_by("alias", whose, since) if r["key"]],
        "daily": db.usage_daily(whose, days),
        "kept_days": db.USAGE_DAYS,
    }
