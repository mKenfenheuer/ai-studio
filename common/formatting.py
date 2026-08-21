"""Turning a dataset row into the exact string a model is trained on.

This lives in `common` for one reason: the web UI promises to show you what
your model will actually read, and a preview built by lookalike code is not
that promise. The controller renders the preview and the runner builds the
training batch by calling the *same* functions, so the two cannot drift.

Four shapes are supported, which between them cover almost everything on the
Hub: plain text, instruction/response pairs, conversations, and -- for anything
those three get wrong -- a Jinja template you write yourself.

Conversations are the awkward one. There is no single "messages" format:
`content` may be a string or a list of typed parts, roles may be under `role`
or `from`, text may be under `content` or `value`, and a tool-calling dataset
carries `tool_calls` on assistant turns plus a `tools` schema alongside. All of
that is normalised to one shape here before any template sees it.
"""
from __future__ import annotations

import json
from typing import Any

DEFAULT_INSTRUCTION_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n### Response:\n{response}")

_INSTRUCTION_FIELDS = ["instruction", "prompt", "question", "input"]
_RESPONSE_FIELDS = ["output", "response", "answer", "completion"]
_TEXT_FIELDS = ["text", "content", "document", "sentence", "raw", "body"]
_MESSAGE_FIELDS = ["messages", "conversations", "conversation", "chat", "turns"]
_TOOL_FIELDS = ["tools", "functions", "tool_schema"]

# Roles a chat template is expected to understand. Anything else is passed
# through untouched rather than dropped -- an unfamiliar role is far more
# likely to be a dataset we have not seen than a mistake.
KNOWN_ROLES = ("system", "user", "assistant", "tool", "function")

# The fallback conversation rendering, used when neither the model nor the user
# supplies a template. Deliberately plain and readable: its job is to be
# obviously right in a preview, not to imitate any particular model's format.
# Written without `{%-` whitespace control on purpose. The environment sets
# trim_blocks and lstrip_blocks, so a tag alone on a line already contributes
# nothing. Adding `-` on top of that also swallows the newline that ends the
# *content* line, and every turn runs into the next one.
BUILTIN_CHAT_TEMPLATE = """\
{% if tools %}
Available tools:
{% for t in tools %}
- {{ t.name }}: {{ t.description }}
{% endfor %}

{% endif %}
{% for m in messages %}
{{ m.role }}: {{ m.content }}
{% for c in m.tool_calls %}
{{ m.role }} calls {{ c.name }}({{ c.arguments }})
{% endfor %}
{% endfor %}"""


def first_present(row: dict, names: list[str]) -> str | None:
    for n in names:
        if n in row and row[n]:
            return n
    return None


# ---------------------------------------------------------------------------
# Normalising conversations
# ---------------------------------------------------------------------------

def _part_text(part: Any) -> str:
    """Text out of one content part.

    Newer datasets carry content as a list of typed parts rather than a
    string -- `[{"type": "text", "text": "..."}]` -- which stringifies into
    Python dict syntax if you treat it as text. That is the difference between
    training on a sentence and training on `[{'type': 'text', ...}]`.
    """
    if isinstance(part, str):
        return part
    if isinstance(part, dict):
        for key in ("text", "content", "value", "data"):
            if isinstance(part.get(key), str):
                return part[key]
        # A non-text part (an image, say). Name it rather than dumping it.
        kind = part.get("type")
        return "[%s]" % kind if kind else ""
    return str(part) if part is not None else ""


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(_part_text(p) for p in content)
    return _part_text(content)


def _normalize_tool_call(call: Any) -> dict | None:
    """One tool call, flattened out of whichever wrapper it arrived in."""
    if not isinstance(call, dict):
        return None
    fn = call.get("function") if isinstance(call.get("function"), dict) else call
    name = fn.get("name")
    if not name:
        return None
    args = fn.get("arguments", fn.get("parameters"))
    if isinstance(args, (dict, list)):
        args = json.dumps(args, ensure_ascii=False)
    return {"id": call.get("id"), "name": name, "arguments": args or ""}


def normalize_messages(value: Any) -> list[dict]:
    """One consistent message shape, whatever the dataset used.

    Returns a list of {role, content, tool_calls, train, name}. Handles the
    OpenAI shape, the ShareGPT `from`/`value` shape, typed content parts, and
    tool calls under either `tool_calls` or a bare `function_call`.
    """
    if not isinstance(value, list):
        return []
    out = []
    for m in value:
        if isinstance(m, str):
            out.append({"role": "user", "content": m, "tool_calls": [],
                        "train": True, "name": None})
            continue
        if not isinstance(m, dict):
            continue

        role = m.get("role") or m.get("from") or "user"
        role = {"human": "user", "gpt": "assistant", "bot": "assistant",
                "system_prompt": "system"}.get(str(role).lower(), str(role).lower())

        content = m.get("content")
        if content is None:
            content = m.get("value")

        raw_calls = m.get("tool_calls")
        if raw_calls is None and m.get("function_call"):
            raw_calls = [m["function_call"]]
        calls = [c for c in (_normalize_tool_call(c) for c in (raw_calls or []))
                 if c]

        out.append({
            "role": role,
            "content": _content_text(content),
            "tool_calls": calls,
            # Some datasets mark which turns are worth learning from. Carried
            # through so a template can act on it, even though the trainer
            # currently learns from the whole conversation.
            "train": bool(m.get("train_on_turn", m.get("train", True))),
            "name": m.get("name"),
        })
    return out


def normalize_tools(value: Any) -> list[dict]:
    """Tool definitions flattened to {name, description, parameters}."""
    if not isinstance(value, list):
        return []
    out = []
    for t in value:
        if not isinstance(t, dict):
            continue
        fn = t.get("function") if isinstance(t.get("function"), dict) else t
        name = fn.get("name")
        if not name:
            continue
        out.append({
            "name": name,
            "description": fn.get("description") or "",
            "parameters": _prune(fn.get("parameters")),
        })
    return out


def _prune(value: Any) -> Any:
    """Drop null-valued keys from a tool schema.

    The Hub's Arrow conversion pads every tool's parameter object with the
    union of every parameter any tool uses, filling the gaps with nulls. A
    19-tool schema becomes mostly nulls, which would be handed to the model as
    if those parameters existed.
    """
    if isinstance(value, dict):
        return {k: _prune(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_prune(v) for v in value if v is not None]
    return value


def find_messages(row: dict, field: str | None = None) -> list[dict]:
    key = field or first_present(row, _MESSAGE_FIELDS)
    return normalize_messages(row.get(key)) if key else []


def find_tools(row: dict, field: str | None = None) -> list[dict]:
    key = field or first_present(row, _TOOL_FIELDS)
    return normalize_tools(row.get(key)) if key else []


# ---------------------------------------------------------------------------
# Jinja
# ---------------------------------------------------------------------------

class TemplateError(ValueError):
    """A template that could not be compiled or rendered."""


def _environment():
    """A sandboxed Jinja environment with the helpers chat templates expect.

    Sandboxed because templates are text the user pastes in, and a plain Jinja
    environment lets a template reach attributes of the objects passed to it.
    The globals mirror what transformers provides, so a chat template copied
    from a model repository renders here unchanged.
    """
    try:
        from jinja2.exceptions import TemplateError as JinjaTemplateError
        from jinja2.sandbox import ImmutableSandboxedEnvironment
    except ImportError as e:  # pragma: no cover - dependency is declared
        raise TemplateError("Jinja2 is not installed: %s" % e) from e

    def raise_exception(message):
        raise JinjaTemplateError(message)

    def tojson(x, indent=None, **_):
        return json.dumps(x, ensure_ascii=False, indent=indent)

    def strftime_now(fmt):
        import datetime
        return datetime.datetime.now().strftime(fmt)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = strftime_now
    return env


def render_template(template: str, *, row: dict | None = None,
                    messages: list[dict] | None = None,
                    tools: list[dict] | None = None,
                    specials: dict | None = None,
                    add_generation_prompt: bool = False) -> str:
    """Render one example with a Jinja template.

    The same call renders the preview in the browser and the training batch on
    the runner, which is the only way the two can be guaranteed to agree.
    """
    env = _environment()
    try:
        compiled = env.from_string(template)
    except Exception as e:  # noqa: BLE001 - jinja raises several types
        raise TemplateError("Template will not compile: %s" % e) from e

    ctx = dict(row or {})
    ctx.update({
        "messages": messages or [],
        "tools": tools or None,
        "row": row or {},
        "add_generation_prompt": add_generation_prompt,
    })
    ctx.update(specials or {})
    try:
        return compiled.render(**ctx)
    except Exception as e:  # noqa: BLE001
        raise TemplateError("Template failed on this row: %s" % e) from e


# ---------------------------------------------------------------------------
# The main entry point
# ---------------------------------------------------------------------------

def format_example(row: dict, fmt: dict) -> str | None:
    """Render one row as the model will see it, or None if it cannot be read."""
    fmt = fmt or {}
    mode = fmt.get("mode", "auto")

    if mode == "jinja":
        template = fmt.get("template")
        if not template:
            return None
        text = render_template(
            template, row=row,
            messages=find_messages(row, fmt.get("messages_field")),
            tools=find_tools(row, fmt.get("tools_field")),
            specials=fmt.get("specials"))
        return text.strip() or None

    text_field = fmt.get("text_field")
    if mode == "text" or (mode == "auto" and text_field and text_field in row):
        val = row.get(text_field or "text")
        return str(val) if val else None

    if mode in ("chat", "auto"):
        messages = find_messages(row, fmt.get("messages_field"))
        if messages:
            tools = find_tools(row, fmt.get("tools_field"))
            template = fmt.get("chat_template") or BUILTIN_CHAT_TEMPLATE
            text = render_template(template, row=row, messages=messages,
                                   tools=tools, specials=fmt.get("specials"))
            return text.strip() or None

    if mode in ("instruction", "auto"):
        instr_f = fmt.get("instruction_field") or first_present(row, _INSTRUCTION_FIELDS)
        resp_f = fmt.get("response_field") or first_present(row, _RESPONSE_FIELDS)
        if instr_f and resp_f:
            instr, resp = row.get(instr_f), row.get(resp_f)
            if instr and resp:
                # A separate "input" column is extra context for the
                # instruction, not a second instruction. Alpaca and its many
                # derivatives are laid out this way.
                ctx = row.get("input") if instr_f != "input" else None
                tmpl = fmt.get("template") or DEFAULT_INSTRUCTION_TEMPLATE
                prompt = str(instr) + (("\n\n" + str(ctx)) if ctx else "")
                return tmpl.format(instruction=prompt, response=str(resp))

    if mode == "auto":
        for f in _TEXT_FIELDS:
            if row.get(f):
                return str(row[f])
    return None


def detect_format(columns: list[str], rows: list[dict] | None = None) -> dict:
    """Guess how a dataset is laid out, so the UI can pre-fill the mapping."""
    cols = {c.lower() for c in columns if c}

    msg_field = next((c for c in _MESSAGE_FIELDS if c in cols), None)
    if msg_field:
        out = {"mode": "chat", "messages_field": msg_field, "confidence": "high"}
        tool_field = next((c for c in _TOOL_FIELDS if c in cols), None)
        if tool_field:
            out["tools_field"] = tool_field
        # Report the roles actually present, so the UI can say what it found
        # rather than making the user open the raw rows to find out.
        if rows:
            roles, has_calls = [], False
            for r in rows:
                for m in normalize_messages(r.get(msg_field)):
                    if m["role"] not in roles:
                        roles.append(m["role"])
                    has_calls = has_calls or bool(m["tool_calls"])
            out["roles"] = roles
            out["has_tool_calls"] = has_calls
        return out

    if {"instruction"} & cols and cols & {"output", "response"}:
        return {"mode": "instruction", "instruction_field": "instruction",
                "response_field": "output" if "output" in cols else "response",
                "confidence": "high"}
    if {"prompt"} & cols and cols & {"completion", "response", "answer"}:
        resp = next(c for c in ("completion", "response", "answer") if c in cols)
        return {"mode": "instruction", "instruction_field": "prompt",
                "response_field": resp, "confidence": "high"}
    for c in _TEXT_FIELDS:
        if c in cols:
            return {"mode": "text", "text_field": c, "confidence": "medium"}
    return {"mode": "auto", "confidence": "low"}


def pick_text_column(columns: list[str], rows: list[dict]) -> str | None:
    """The column holding the body text, for pretraining.

    Chosen by looking at the rows rather than by name. Corpora call this column
    'text', 'content', 'raw' and a dozen other things, but it is reliably the
    one with the most characters in it.
    """
    best, best_len = None, 0
    for c in columns or []:
        lengths = [len(r[c]) for r in (rows or []) if isinstance(r.get(c), str)]
        avg = sum(lengths) / len(lengths) if lengths else 0
        if avg > best_len:
            best_len, best = avg, c
    return best if best_len > 40 else None


# ---------------------------------------------------------------------------
# The other direction: a conversation, ready for the model to continue
# ---------------------------------------------------------------------------
#
# Training renders a *finished* conversation. Talking to the result renders an
# *unfinished* one and asks the model to write the next turn. Both must use the
# same template, or the playground speaks a dialect the model was never taught
# and the run looks like a failure when it is not. So the job's training format
# is carried through to inference and reused here verbatim.

ASSISTANT_CUE = "assistant:"


def system_prompts(rows: list[dict], fmt: dict | None = None,
                   limit: int = 3) -> list[str]:
    """Distinct system messages present in these rows.

    Offered back in the playground, because a model fine-tuned with a system
    prompt behaves quite differently without one -- and the prompt it was
    trained with is rarely something anybody remembers to write down.
    """
    fmt = fmt or {}
    seen: list[str] = []
    for row in rows or []:
        for m in find_messages(row, fmt.get("messages_field")):
            if m["role"] == "system" and m["content"].strip():
                text = m["content"].strip()
                if text not in seen:
                    seen.append(text)
                if len(seen) >= limit:
                    return seen
    return seen


def conversation_style(fmt: dict | None) -> str:
    """How this model expects to be talked to: chat, instruct, or continue."""
    fmt = fmt or {}
    if fmt.get("use_model_template") or fmt.get("chat_template"):
        return "chat"
    mode = fmt.get("mode")
    if mode == "chat":
        return "chat"
    if mode == "jinja":
        # A hand-written template is given the conversation, so it is a chat
        # unless it clearly never looks at one.
        return "chat" if "messages" in (fmt.get("template") or "") else "continue"
    if mode == "instruction":
        return "instruct"
    return "continue"


def render_prompt(messages: list[dict], fmt: dict | None = None,
                  tools: list[dict] | None = None) -> str:
    """Text for the model to continue, given the conversation so far."""
    fmt = fmt or {}
    messages = normalize_messages(messages)
    style = conversation_style(fmt)
    specials = fmt.get("specials")

    template = fmt.get("chat_template") or (
        fmt.get("template") if fmt.get("mode") == "jinja" else None)

    if style == "chat" and template:
        # A model's own template understands add_generation_prompt and emits
        # the opening of the assistant turn itself.
        return render_template(template, messages=messages, tools=tools,
                               specials=specials, add_generation_prompt=True)

    if style == "chat":
        body = render_template(BUILTIN_CHAT_TEMPLATE, messages=messages,
                               tools=tools, specials=specials)
        # The built-in format writes one "role: content" line per turn, so the
        # cue for the next turn is the assistant's label with no newline after
        # it -- the model continues on the same line, exactly as it was taught.
        return body.rstrip("\n") + "\n" + ASSISTANT_CUE + " "

    if style == "instruct":
        tmpl = fmt.get("template") or DEFAULT_INSTRUCTION_TEMPLATE
        instruction = _last_user(messages)
        # System text is prepended rather than dropped: an instruction-tuned
        # model has no system turn, but the words still steer it.
        system = " ".join(m["content"] for m in messages if m["role"] == "system")
        if system:
            instruction = system.strip() + "\n\n" + instruction
        return tmpl.format(instruction=instruction, response="")

    return _last_user(messages)


def _last_user(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m["role"] == "user" and m["content"]:
            return m["content"]
    return messages[-1]["content"] if messages else ""


def stop_sequences(fmt: dict | None, specials: dict | None = None) -> list[str]:
    """Where a reply should be cut, given how this model was trained.

    Derived from the template rather than guessed: whatever marks the start of
    the *next* turn is what must not be generated into.
    """
    fmt = fmt or {}
    style = conversation_style(fmt)
    out: list[str] = []
    for key in ("eos_token", "bos_token"):
        token = (specials or fmt.get("specials") or {}).get(key)
        if token:
            out.append(token)

    template = fmt.get("chat_template") or fmt.get("template") or ""
    for marker in ("<|im_start|>", "<|start_header_id|>", "<|user|>",
                   "[INST]", "<start_of_turn>"):
        if marker in template:
            out.append(marker)

    if style == "instruct":
        out += ["### Instruction:", "\n### "]
    if style == "chat" and not template:
        out += ["\nuser:", "\nsystem:", "\ntool:"]
    # Turn markers a base model falls back on when nothing taught it to stop.
    out += ["Human:", "\nUser:", "<|endoftext|>"]
    return list(dict.fromkeys(out))
