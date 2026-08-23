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
    base = (conn.get("base_url") or spec.get("base_url") or "").strip()
    return base.rstrip("/")


def model_name(conn: dict, model: str | None) -> str:
    """What to call the model in the request.

    Azure has no model field worth sending -- the deployment in the URL *is*
    the model -- so the deployment doubles as its name everywhere the UI
    needs one.
    """
    spec = provider(conn["provider"]) or {}
    if spec.get("flavour") == "azure":
        return conn.get("deployment") or ""
    return (model or conn.get("model") or "").strip()


# ---------------------------------------------------------------------------
# One completion
# ---------------------------------------------------------------------------

def chat_request(conn: dict, model: str, messages: list[dict],
                 params: dict | None = None) -> dict:
    """The HTTP request that asks this provider for one reply.

    Returns a dict of url/headers/json rather than performing the call, so the
    async controller and the sync runner can both send it.
    """
    spec = provider(conn["provider"])
    if not spec:
        raise ValueError("Unknown provider: %s" % conn.get("provider"))
    params = params or {}
    max_tokens = int(params.get("max_new_tokens") or params.get("max_tokens") or 512)
    temperature = params.get("temperature")
    top_p = params.get("top_p")
    flavour = spec["flavour"]

    if flavour == "anthropic":
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

    body = {"messages": messages, "max_tokens": max_tokens}
    if temperature is not None:
        body["temperature"] = float(temperature)
    if top_p is not None:
        body["top_p"] = float(top_p)

    if flavour == "azure":
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


def retry_body(conn: dict, body: dict, error_text: str) -> dict | None:
    """A second attempt at a request the provider rejected on a technicality.

    The newer OpenAI models renamed `max_tokens` to `max_completion_tokens`
    and refuse a `temperature` other than the default. Both come back as a
    400 naming the parameter, which is enough to fix the request and try once
    more rather than failing a thousand-row generation on its first row.
    """
    spec = provider(conn.get("provider")) or {}
    if spec.get("flavour") not in ("openai", "azure"):
        return None
    text = (error_text or "").lower()
    fixed = dict(body)
    changed = False
    if "max_completion_tokens" in text and "max_tokens" in fixed:
        fixed["max_completion_tokens"] = fixed.pop("max_tokens")
        changed = True
    if "temperature" in text and "temperature" in fixed:
        fixed.pop("temperature")
        changed = True
    if "top_p" in text and "top_p" in fixed:
        fixed.pop("top_p")
        changed = True
    return fixed if changed else None


def chat_text(conn: dict, data: dict) -> str:
    """The reply itself, whichever shape it arrived in."""
    spec = provider(conn.get("provider")) or {}
    if spec.get("flavour") == "anthropic":
        # A reply is a list of blocks; only the text ones are the answer, and
        # a thinking block at the front is not part of it.
        parts = [b.get("text") or "" for b in (data.get("content") or [])
                 if isinstance(b, dict) and b.get("type") == "text"]
        return "".join(parts).strip()
    choices = data.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        # Some compatible servers return blocks rather than a string.
        return "".join(p.get("text") or "" for p in content
                       if isinstance(p, dict)).strip()
    return (content or "").strip()


def chat_usage(conn: dict, data: dict) -> dict:
    """Tokens in and out, under one set of names."""
    spec = provider(conn.get("provider")) or {}
    usage = data.get("usage") or {}
    if spec.get("flavour") == "anthropic":
        return {"input_tokens": usage.get("input_tokens") or 0,
                "output_tokens": usage.get("output_tokens") or 0}
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
