"""Models that live behind somebody else's API.

Everything else in this studio runs a model on a machine you own. This module
is for the other case: a hosted model, reached over HTTP, used to *write* data
rather than to be trained. That is the one job where paying per token is
obviously worth it -- the dataset outlives the run, and a small local model
writing a thousand mediocre examples produces a thousand mediocre examples.

Three shapes cover almost every provider anyone actually has an account with:

* **openai** -- `POST {base}/chat/completions`, a bearer token, and the
  `choices[0].message.content` response. Together, Groq, OpenRouter, Mistral,
  Fireworks, vLLM, llama.cpp, Ollama and this studio's own API all speak it,
  which is why "any OpenAI-compatible endpoint" is one entry rather than ten.
* **azure** -- the same body, but the deployment is in the URL, the version is
  a query parameter, and the key is in `api-key` rather than in
  `Authorization`. Different enough to break every client that assumes the
  first shape, similar enough that only the envelope changes.
* **anthropic** -- `POST {base}/v1/messages`, `x-api-key`, an explicit
  `anthropic-version`, the system prompt as its own field rather than a
  message, and a list of content blocks in the reply.

NOTHING HERE PERFORMS I/O. Each function builds a request or reads a
response, so the controller can send it with its async client and the runner
with its sync one, and there is exactly one place that knows what a provider's
URL, headers and body look like. Two copies of that knowledge would be two
copies to get wrong.
"""
from __future__ import annotations

from typing import Any

# The version pin Anthropic's API requires on every request. It is a date, not
# a number, and sending none at all is an error rather than a default.
ANTHROPIC_VERSION = "2023-06-01"
# Azure dates its API the same way, on the query string. This is a recent
# stable version; a connection may name its own.
AZURE_API_VERSION = "2024-10-21"

# Room for a reasoning model's thinking, on top of the answer the caller asked
# for. Only the Responses API needs this: it counts both against one budget,
# where Chat Completions does not.
REASONING_HEADROOM_TOKENS = 2048

# What a connection needs before it can be used. `secret` fields are never
# sent back to the browser once stored.
FIELD = {
    "api_key": {"label": "API key", "secret": True,
                "hint": "Stored encrypted. It is never shown again, and never "
                        "sent to the browser."},
    "base_url": {"label": "Base URL",
                 "hint": "Everything before /chat/completions."},
    "endpoint": {"label": "Resource endpoint",
                 "hint": "https://your-resource.openai.azure.com"},
    "deployment": {"label": "Deployment name",
                   "hint": "What you named the model in Azure — not the "
                           "model id."},
    "api_version": {"label": "API version",
                    "hint": "Leave blank for %s." % AZURE_API_VERSION},
    "model": {"label": "Model",
              "hint": "The model name, as the service spells it — "
                      "gpt-5.4, gpt-oss-120b."},
    "api_style": {"label": "Which API",
                  "choices": [
                      {"value": "chat", "label": "Chat Completions",
                       "hint": "/chat/completions — what almost everything "
                               "speaks."},
                      {"value": "responses", "label": "Responses",
                       "hint": "/responses — OpenAI's newer surface. Needed "
                               "for the models served only there, and the only "
                               "one that returns reasoning as its own item."}],
                  "hint": "Chat Completions unless the model is only served "
                          "on /responses."},
}

PROVIDERS = [
    {
        "id": "openai",
        "label": "OpenAI",
        "flavour": "openai",
        "base_url": "https://api.openai.com/v1",
        "required": ["api_key"],
        "optional": ["base_url"],
        "lists_models": True,
        "blurb": "GPT models, billed to your own OpenAI account.",
    },
    {
        "id": "azure",
        "label": "Azure OpenAI",
        "flavour": "azure",
        "required": ["endpoint", "api_key", "deployment"],
        "optional": ["api_version"],
        "lists_models": False,
        "blurb": "The same models inside your Azure subscription, with your "
                 "own region, quota and data handling. The model is whatever "
                 "you named the deployment.",
    },
    {
        "id": "azure_v1",
        "label": "Azure OpenAI (v1 API)",
        "flavour": "azure_v1",
        "required": ["endpoint", "api_key", "model"],
        "optional": ["api_style"],
        "lists_models": True,
        "blurb": "Azure's newer unified surface: one endpoint, the model named "
                 "in the request like everywhere else, and no deployment in "
                 "the URL or version on the query string. This is the one to "
                 "use for an AI Foundry resource — the older entry above is "
                 "for a classic per-deployment Azure OpenAI resource.",
    },
    {
        "id": "anthropic",
        "label": "Anthropic",
        "flavour": "anthropic",
        "base_url": "https://api.anthropic.com",
        "required": ["api_key"],
        "optional": ["base_url"],
        "lists_models": True,
        "blurb": "Claude models, billed to your own Anthropic account.",
    },
    {
        "id": "compatible",
        "label": "Any OpenAI-compatible API",
        "flavour": "openai",
        "required": ["base_url"],
        "optional": ["api_key"],
        "lists_models": True,
        "blurb": "One entry for the many services that copied the OpenAI "
                 "shape: Together, Groq, OpenRouter, Mistral, Fireworks, a "
                 "vLLM or Ollama server on your own network — or another AI "
                 "Studio. Give the base URL up to and including /v1.",
    },
]


def provider(provider_id: str | None) -> dict | None:
    return next((p for p in PROVIDERS if p["id"] == provider_id), None)


def public_providers() -> list[dict]:
    """The catalogue, with each field described, for the settings screen."""
    out = []
    for p in PROVIDERS:
        fields = [{"name": n, "required": n in p["required"], **FIELD[n]}
                  for n in p["required"] + p["optional"]]
        out.append({k: p[k] for k in ("id", "label", "blurb", "lists_models")}
                   | {"fields": fields, "base_url": p.get("base_url", "")})
    return out


def describe(conn: dict) -> str:
    """A connection in one line, for a log or a job summary."""
    spec = provider(conn.get("provider"))
    label = spec["label"] if spec else (conn.get("provider") or "?")
    if conn.get("deployment"):
        return "%s (%s)" % (label, conn["deployment"])
    if base := conn.get("base_url"):
        if not spec or base != spec.get("base_url"):
            return "%s (%s)" % (label, base)
    return label


def problems(conn: dict) -> str | None:
    """Why this connection cannot be used yet, or None."""
    spec = provider(conn.get("provider"))
    if not spec:
        return "Unknown provider: %s" % conn.get("provider")
    missing = [FIELD[f]["label"] for f in spec["required"]
               if not (conn.get(f) or "").strip()]
    if missing:
        return "%s needs %s." % (spec["label"], ", ".join(missing))
    return None


def _base(conn: dict) -> str:
    spec = provider(conn["provider"]) or {}
    if spec.get("flavour") == "azure_v1":
        # One endpoint for everything. The resource may be given with or
        # without the /openai/v1 suffix, because both are what people copy out
        # of the portal, and appending it twice is the failure that produces a
        # 404 with nothing in it to explain why.
        endpoint = (conn.get("endpoint") or "").strip().rstrip("/")
        if endpoint.endswith("/openai/v1"):
            return endpoint
        if endpoint.endswith("/openai"):
            return endpoint + "/v1"
        return endpoint + "/openai/v1"
    base = (conn.get("base_url") or spec.get("base_url") or "").strip()
    return base.rstrip("/")


def api_style(conn: dict) -> str:
    """Which of the two OpenAI request shapes this connection speaks."""
    return "responses" if (conn.get("api_style") or "").strip() == "responses" \
        else "chat"


def flavour(conn: dict) -> str:
    """How to build a request for this connection.

    The provider decides the envelope -- where the key goes, what the URL looks
    like -- and `api_style` decides the body, because OpenAI and Azure both
    serve two different APIs from the same host and credential.
    """
    spec = provider(conn.get("provider")) or {}
    base = spec.get("flavour") or ""
    if base in ("openai", "azure_v1") and api_style(conn) == "responses":
        return "responses"
    return base


def _auth(conn: dict) -> dict:
    """Headers that authenticate this connection.

    Azure's v1 surface accepts a bearer token, which is what every other
    OpenAI-shaped service uses; the older per-deployment surface accepts only
    `api-key`. Sending both would be harmless and is not done, because a
    request that works for the wrong reason is a request nobody can debug.
    """
    key = (conn.get("api_key") or "").strip()
    if not key:
        return {"content-type": "application/json"}
    if (provider(conn.get("provider")) or {}).get("flavour") == "azure":
        return {"api-key": key, "content-type": "application/json"}
    return {"authorization": "Bearer %s" % key,
            "content-type": "application/json"}


def model_name(conn: dict, model: str | None) -> str:
    """What to call the model in the request.

    Classic Azure has no model field worth sending -- the deployment in the URL
    *is* the model -- so the deployment doubles as its name everywhere the UI
    needs one. The v1 surface named the model again, like everyone else.
    """
    spec = provider(conn["provider"]) or {}
    if spec.get("flavour") == "azure":
        return conn.get("deployment") or ""
    return (model or conn.get("model") or "").strip()


# ---------------------------------------------------------------------------
# One completion
# ---------------------------------------------------------------------------

def chat_request(conn: dict, model: str, messages: list[dict],
                 params: dict | None = None,
                 tools: list[dict] | None = None) -> dict:
    """The HTTP request that asks this provider for one reply.

    Returns a dict of url/headers/json rather than performing the call, so the
    async controller and the sync runner can both send it.

    `tools` are declared in whichever shape this API wants. The two OpenAI
    surfaces disagree about it: Chat Completions nests the definition under
    `function`, Responses puts the name and parameters at the top level of each
    tool. Sending one to the other is a 400 that names no field.
    """
    spec = provider(conn["provider"])
    if not spec:
        raise ValueError("Unknown provider: %s" % conn.get("provider"))
    params = params or {}
    max_tokens = int(params.get("max_new_tokens") or params.get("max_tokens") or 512)
    temperature = params.get("temperature")
    top_p = params.get("top_p")
    shape = flavour(conn)

    if shape == "responses":
        return _responses_request(conn, model, messages, params, tools,
                                  max_tokens, temperature, top_p)

    if shape == "anthropic":
        # The system prompt is a field of its own here, not a message with a
        # role. Sending it as a message is accepted by nothing.
        system = "\n\n".join(m["content"] for m in messages
                             if m.get("role") == "system" and m.get("content"))
        turns = [{"role": m["role"], "content": m.get("content") or ""}
                 for m in messages if m.get("role") in ("user", "assistant")]
        body: dict[str, Any] = {
            "model": model, "max_tokens": max_tokens, "messages": turns,
        }
        if system:
            body["system"] = system
        if tools:
            # A third spelling: flat like Responses, but the schema is called
            # `input_schema` rather than `parameters`.
            body["tools"] = [{"name": t.get("name") or "",
                              "description": t.get("description") or "",
                              "input_schema": t.get("parameters")
                              or {"type": "object", "properties": {}}}
                             for t in tools]
        if temperature is not None:
            body["temperature"] = float(temperature)
        if top_p is not None:
            body["top_p"] = float(top_p)
        return {
            "url": "%s/v1/messages" % _base(conn),
            "headers": {"x-api-key": conn.get("api_key") or "",
                        "anthropic-version": ANTHROPIC_VERSION,
                        "content-type": "application/json"},
            "json": body,
        }

    body = {"messages": _chat_messages(messages), "max_tokens": max_tokens}
    if tools:
        # Nested under `function`, which is the one shape the Responses API
        # does *not* accept. Same tools, three encodings, and no provider
        # tolerates another's.
        body["tools"] = [{"type": "function",
                          "function": {"name": t.get("name") or "",
                                       "description": t.get("description") or "",
                                       "parameters": t.get("parameters")
                                       or {"type": "object", "properties": {}}}}
                         for t in tools]
    if temperature is not None:
        body["temperature"] = float(temperature)
    if top_p is not None:
        body["top_p"] = float(top_p)

    if shape == "azure":
        version = (conn.get("api_version") or "").strip() or AZURE_API_VERSION
        endpoint = (conn.get("endpoint") or "").strip().rstrip("/")
        url = "%s/openai/deployments/%s/chat/completions?api-version=%s" % (
            endpoint, conn.get("deployment") or "", version)
        # The deployment is the model, and Azure ignores the field -- but some
        # gateways in front of it do not, so it is left out rather than
        # guessed.
        return {"url": url,
                "headers": {"api-key": conn.get("api_key") or "",
                            "content-type": "application/json"},
                "json": body}

    body["model"] = model
    headers = {"content-type": "application/json"}
    if key := (conn.get("api_key") or "").strip():
        headers["authorization"] = "Bearer %s" % key
    return {"url": "%s/chat/completions" % _base(conn), "headers": headers,
            "json": body}


def _chat_messages(messages: list[dict]) -> list[dict]:
    """Canonical messages as Chat Completions wants them.

    Mostly a pass-through, and mostly about what to leave out: `reasoning` is
    ours and not a field this API accepts, an assistant turn that only calls a
    tool must send `content: null` rather than an empty string, and a tool
    result needs its `tool_call_id` beside it or the provider rejects the
    whole conversation.
    """
    out = []
    for m in messages:
        role = m.get("role") or "user"
        item: dict[str, Any] = {"role": role}
        content = m.get("content") or ""
        if role == "tool":
            item["content"] = content
            if call_id := m.get("tool_call_id"):
                item["tool_call_id"] = call_id
            out.append(item)
            continue
        if calls := m.get("tool_calls"):
            item["content"] = content or None
            item["tool_calls"] = [
                {"id": c.get("id") or "", "type": "function",
                 "function": {"name": (c.get("function") or {}).get("name") or "",
                              "arguments": (c.get("function") or {}).get("arguments") or "{}"}}
                for c in calls]
        else:
            item["content"] = content
        out.append(item)
    return out


def _responses_request(conn: dict, model: str, messages: list[dict],
                       params: dict, tools: list[dict] | None,
                       max_tokens: int, temperature, top_p) -> dict:
    """A request to the Responses API.

    Not a variant of the chat body -- a different one. The conversation is
    `input` rather than `messages`, the limit is `max_output_tokens`, a tool
    call and its result are *items in the input list* rather than fields on a
    message, and a tool is declared flat rather than nested under `function`.

    Reasoning models are the reason this exists at all: several are served here
    and nowhere else, and this is the only surface that returns the model's
    working as its own item instead of leaving it to be dug out of the text.
    """
    body: dict[str, Any] = {"model": model, "max_output_tokens": max_tokens}

    # The system prompt is `instructions`, its own top-level field. It may also
    # be passed as a message, but the field is unambiguous and every model
    # served here honours it.
    system = "\n\n".join(m.get("content") or "" for m in messages
                         if m.get("role") in ("system", "developer")
                         and m.get("content"))
    if system:
        body["instructions"] = system

    items: list[dict] = []
    for m in messages:
        role = m.get("role")
        if role in ("system", "developer"):
            continue
        if role == "tool":
            # A result is an item of its own, addressed by the call's id.
            items.append({"type": "function_call_output",
                          "call_id": m.get("tool_call_id") or "",
                          "output": m.get("content") or ""})
            continue
        if content := (m.get("content") or ""):
            items.append({"role": role, "content": content})
        for call in m.get("tool_calls") or []:
            fn = call.get("function") or call
            items.append({"type": "function_call",
                          "call_id": call.get("id") or "",
                          "name": fn.get("name") or "",
                          "arguments": fn.get("arguments") or "{}"})
    body["input"] = items

    if tools:
        # Flat, not nested. Chat Completions wants
        # {"type":"function","function":{...}}; this wants the name and
        # parameters at the top level of the tool. Sending the wrong one is a
        # 400 that names no field.
        body["tools"] = [{"type": "function",
                          "name": t.get("name") or "",
                          "description": t.get("description") or "",
                          "parameters": t.get("parameters")
                          or {"type": "object", "properties": {}}}
                         for t in tools]
    # Getting a model's working back off this API takes TWO settings, and
    # each is useless without the other. Measured against gpt-5.4:
    #
    #   summary alone      no reasoning item at all      0 characters
    #   effort alone       a reasoning item, empty       0 characters
    #   effort + summary   a reasoning item with text  322 characters
    #
    # `effort` is what makes it think; `summary` is what makes the thinking
    # visible. Sending one was the same as sending neither, which is how a run
    # that extended a reasoning dataset produced turns with no reasoning on
    # them -- the model reasoned, and nobody asked to see it.
    effort = (params.get("reasoning_effort") or "").strip()
    if effort or params.get("reasoning"):
        body["reasoning"] = {"effort": effort or "medium", "summary": "auto"}
        # Thinking is charged against `max_output_tokens` here, unlike on Chat
        # Completions, so a small budget is spent thinking and returns a
        # fragment -- five characters of answer out of 900 tokens, measured.
        # The caller's number is what they want to READ; the thinking gets
        # room of its own on top of it.
        body["max_output_tokens"] = max_tokens + REASONING_HEADROOM_TOKENS
    if temperature is not None:
        body["temperature"] = float(temperature)
    if top_p is not None:
        body["top_p"] = float(top_p)
    return {"url": "%s/responses" % _base(conn), "headers": _auth(conn),
            "json": body}


def reply(conn: dict, data: dict) -> dict:
    """A provider's answer as {content, reasoning, tool_calls}.

    The same three things every caller wants, out of four different response
    shapes. Tool calls come back in the canonical nested form regardless of
    which surface produced them, so a generated conversation can be written
    straight into a dataset without another translation.
    """
    shape = flavour(conn)
    if shape == "responses":
        text, thinking, calls = [], [], []
        for item in data.get("output") or []:
            kind = item.get("type")
            if kind == "message":
                for part in item.get("content") or []:
                    if part.get("type") in ("output_text", "text"):
                        text.append(part.get("text") or "")
            elif kind == "reasoning":
                # Only a summary is returned for the hosted reasoning models;
                # the raw chain is not exposed. What comes back is what there
                # is, and an empty summary is not an error.
                for part in item.get("summary") or []:
                    thinking.append(part.get("text") or ""
                                    if isinstance(part, dict) else str(part))
            elif kind == "function_call":
                calls.append({
                    "id": item.get("call_id") or item.get("id"),
                    "type": "function",
                    "function": {"name": item.get("name") or "",
                                 "arguments": item.get("arguments") or ""}})
        return {"content": "".join(text).strip(),
                "reasoning": "\n\n".join(t for t in thinking if t).strip(),
                "tool_calls": calls}

    if shape == "anthropic":
        text, thinking, calls = [], [], []
        for block in data.get("content") or []:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                text.append(block.get("text") or "")
            elif block.get("type") == "thinking":
                thinking.append(block.get("thinking") or "")
            elif block.get("type") == "tool_use":
                import json as _json
                calls.append({
                    "id": block.get("id"), "type": "function",
                    "function": {"name": block.get("name") or "",
                                 "arguments": _json.dumps(
                                     block.get("input") or {},
                                     ensure_ascii=False)}})
        return {"content": "".join(text).strip(),
                "reasoning": "\n\n".join(thinking).strip(),
                "tool_calls": calls}

    choices = data.get("choices") or []
    if not choices:
        return {"content": "", "reasoning": "", "tool_calls": []}
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(p.get("text") or "" for p in content
                          if isinstance(p, dict))
    # `reasoning_content` is what vLLM, SGLang, DeepSeek and Azure's own
    # gpt-oss deployments emit. Without reading it, a reasoning model that
    # spends its whole budget thinking looks like a model that returned
    # nothing at all.
    thinking = message.get("reasoning_content") or message.get("reasoning") or ""
    calls = []
    for call in message.get("tool_calls") or []:
        fn = call.get("function") or {}
        calls.append({"id": call.get("id"), "type": "function",
                      "function": {"name": fn.get("name") or "",
                                   "arguments": fn.get("arguments") or ""}})
    return {"content": (content or "").strip(),
            "reasoning": (thinking or "").strip(), "tool_calls": calls}


def retry_body(conn: dict, body: dict, error_text: str) -> dict | None:
    """A second attempt at a request the provider rejected on a technicality.

    The newer OpenAI models renamed `max_tokens` to `max_completion_tokens`
    and refuse a `temperature` other than the default. Both come back as a
    400 naming the parameter, which is enough to fix the request and try once
    more rather than failing a thousand-row generation on its first row.
    """
    shape = flavour(conn)
    if shape not in ("openai", "azure", "azure_v1", "responses"):
        return None
    text = (error_text or "").lower()
    fixed = dict(body)
    changed = False
    if "max_completion_tokens" in text and "max_tokens" in fixed:
        fixed["max_completion_tokens"] = fixed.pop("max_tokens")
        changed = True
    if "max_output_tokens" in text and "max_tokens" in fixed:
        fixed["max_output_tokens"] = fixed.pop("max_tokens")
        changed = True
    if "temperature" in text and "temperature" in fixed:
        fixed.pop("temperature")
        changed = True
    if "top_p" in text and "top_p" in fixed:
        fixed.pop("top_p")
        changed = True
    # A model on this surface that does not reason at all rejects the field.
    # Dropped rather than fatal: asking for a summary is an attempt to get
    # more, never a requirement.
    if "reasoning" in text and "reasoning" in fixed:
        fixed.pop("reasoning")
        changed = True
    return fixed if changed else None


def chat_text(conn: dict, data: dict) -> str:
    """Just the answer. Kept because most callers only want that.

    A reply with nothing but a tool call in it has no text, and that is the
    correct answer rather than a failure -- callers that care about calls use
    `reply` instead.
    """
    return reply(conn, data)["content"]


def chat_usage(conn: dict, data: dict) -> dict:
    """Tokens in and out, under one set of names."""
    usage = data.get("usage") or {}
    shape = flavour(conn)
    if shape in ("anthropic", "responses"):
        # Both already call them this. The Responses API also reports how many
        # of the output tokens went on reasoning, which is the number that
        # explains a bill nobody expected.
        out = {"input_tokens": usage.get("input_tokens") or 0,
               "output_tokens": usage.get("output_tokens") or 0}
        details = usage.get("output_tokens_details") or {}
        if details.get("reasoning_tokens"):
            out["reasoning_tokens"] = details["reasoning_tokens"]
        return out
    return {"input_tokens": usage.get("prompt_tokens") or 0,
            "output_tokens": usage.get("completion_tokens") or 0}


def error_message(conn: dict, status: int, data: Any) -> str:
    """What went wrong, in the words the provider used.

    Providers bury the sentence that matters at three different depths, and
    "HTTP 400" on its own has never once been enough to fix anything.
    """
    detail = ""
    if isinstance(data, dict):
        err = data.get("error")
        if isinstance(err, dict):
            detail = err.get("message") or err.get("code") or ""
        elif isinstance(err, str):
            detail = err
        detail = detail or data.get("message") or ""
    elif isinstance(data, str):
        detail = data[:400]
    hint = ""
    if status in (401, 403):
        hint = " Check the API key, and that it may use this model."
    elif status == 404:
        spec = provider(conn.get("provider")) or {}
        hint = (" Check the deployment name." if spec.get("flavour") == "azure"
                else " Check the model name and the base URL.")
    elif status == 429:
        hint = " That is the provider's rate limit, not this studio's."
    return "%s said %d: %s%s" % (describe(conn), status,
                                 detail or "no explanation given", hint)


# ---------------------------------------------------------------------------
# Which models are there
# ---------------------------------------------------------------------------

def models_request(conn: dict) -> dict | None:
    """The request that lists what this connection can reach, if it can.

    Azure has no such endpoint on the data plane -- deployments are listed by
    the management API, with different credentials -- so it returns None and
    the deployment name stands as the answer.
    """
    spec = provider(conn.get("provider"))
    if not spec or not spec.get("lists_models"):
        return None
    if spec["flavour"] == "anthropic":
        return {"url": "%s/v1/models" % _base(conn),
                "headers": {"x-api-key": conn.get("api_key") or "",
                            "anthropic-version": ANTHROPIC_VERSION}}
    headers = {}
    if key := (conn.get("api_key") or "").strip():
        headers["authorization"] = "Bearer %s" % key
    return {"url": "%s/models" % _base(conn), "headers": headers}


def models_from(data: dict) -> list[str]:
    """Model ids out of a listing, sorted, whatever it wrapped them in."""
    rows = data.get("data") if isinstance(data, dict) else None
    if rows is None and isinstance(data, dict):
        rows = data.get("models")
    out = []
    for row in rows or []:
        if isinstance(row, dict):
            name = row.get("id") or row.get("name") or row.get("model")
        else:
            name = row
        if name:
            out.append(str(name))
    return sorted(set(out))
