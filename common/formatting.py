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
import re
from typing import Any

from . import chat_formats

DEFAULT_INSTRUCTION_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n### Response:\n{response}")

_INSTRUCTION_FIELDS = ["instruction", "prompt", "question", "input"]
_RESPONSE_FIELDS = ["output", "response", "answer", "completion"]
_TEXT_FIELDS = ["text", "content", "document", "sentence", "raw", "body"]
_MESSAGE_FIELDS = ["messages", "conversations", "conversation", "chat", "turns"]
_TOOL_FIELDS = ["tools", "functions", "tool_schema"]
_SYSTEM_FIELDS = ["system", "system_prompt"]
_REASONING_FIELDS = ["reasoning", "reasoning_content", "thinking",
                     "thought", "analysis", "rationale"]

# A reply that reasons first is written one of two ways: tagged inline in
# the content, or carried in a field of its own. Both are unpacked into the
# same place, so a template never has to know which the dataset used.
_THINK_RE = re.compile(
    r"<(think|thinking|reasoning)>(.*?)</\1>", re.DOTALL | re.IGNORECASE)

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


def _normalize_tool_call(call: Any, selectors: dict | None = None) -> dict | None:
    """One tool call, flattened out of whichever wrapper it arrived in."""
    if not isinstance(call, dict):
        return None
    sel = selectors or {}
    fn = call.get("function") if isinstance(call.get("function"), dict) else call
    name = select(call, sel.get("tool_name")) or select(fn, sel.get("tool_name")) \
        or fn.get("name")
    if not name:
        return None
    args = select(call, sel.get("tool_arguments"))
    if args is None:
        args = select(fn, sel.get("tool_arguments"))
    if args is None:
        args = fn.get("arguments", fn.get("parameters"))
    if isinstance(args, (dict, list)):
        args = json.dumps(args, ensure_ascii=False)
    return {"id": call.get("id"), "name": name, "arguments": args or ""}


def normalize_messages(value: Any, selectors: dict | None = None) -> list[dict]:
    """One consistent message shape, whatever the dataset used.

    Returns a list of {role, content, reasoning, tool_calls, train, name}.
    Handles the OpenAI shape, the ShareGPT `from`/`value` shape, typed content
    parts, and tool calls under either `tool_calls` or a bare `function_call`.

    `selectors` overrides any of that with an explicit path, for the datasets
    that do none of the above.
    """
    if not isinstance(value, list):
        return []
    sel = selectors or {}
    out = []
    for m in value:
        if isinstance(m, str):
            out.append({"role": "user", "content": m, "reasoning": "",
                        "tool_calls": [], "train": True, "name": None})
            continue
        if not isinstance(m, dict):
            continue

        role = select_in_message(m, sel.get("role")) \
            or m.get("role") or m.get("from") or "user"
        role = str(role).lower()
        # Known aliases first, then any mapping supplied alongside the
        # selectors. A dataset that calls its tool turns "tool_out" needs them
        # recognised as tool results, or the format renders them as an unknown
        # speaker and the model never learns what a tool reply looks like.
        role = {"human": "user", "gpt": "assistant", "bot": "assistant",
                "system_prompt": "system"}.get(role, role)
        role = {str(k).lower(): str(v).lower()
                for k, v in (sel.get("role_map") or {}).items()}.get(role, role)

        content = select_in_message(m, sel.get("content"))
        if content is None:
            content = m.get("content")
        if content is None:
            content = m.get("value")

        reasoning = ""
        chosen = select_in_message(m, sel.get("reasoning"))
        if isinstance(chosen, str) and chosen.strip():
            reasoning = chosen.strip()
        for key in [] if reasoning else _REASONING_FIELDS:
            if isinstance(m.get(key), str) and m[key].strip():
                reasoning = m[key].strip()
                break
        # Harmony carries the same thing as a channel on the message.
        if not reasoning and str(m.get("channel", "")).lower() == "analysis":
            reasoning, content = _content_text(content), ""

        text = _content_text(content)
        if not reasoning:
            # Tagged inline. Lifted out of the answer, so the visible reply is
            # the reply and the reasoning can be shown or hidden separately.
            found = _THINK_RE.search(text)
            if found:
                reasoning = found.group(2).strip()
                text = _THINK_RE.sub("", text, count=1).strip()
        content = text

        raw_calls = m.get("tool_calls")
        if raw_calls is None and m.get("function_call"):
            raw_calls = [m["function_call"]]
        calls = [c for c in (_normalize_tool_call(c, sel)
                             for c in (raw_calls or [])) if c]

        # A call described by selectors alone, with no tool_calls list at all.
        if not calls and (sel.get("tool_name") or sel.get("tool_arguments")):
            named = select_in_message(m, sel.get("tool_name"))
            args = select_in_message(m, sel.get("tool_arguments"))
            if named and args is not None and role == "assistant":
                if isinstance(args, (dict, list)):
                    args = json.dumps(args, ensure_ascii=False)
                calls = [{"id": None, "name": str(named), "arguments": str(args)}]

        # A tool result whose payload is a field of a larger object.
        picked = select_in_message(m, sel.get("tool_result"))
        if picked is not None and role in ("tool", "function"):
            content = picked if isinstance(picked, str) else json.dumps(
                picked, ensure_ascii=False)

        out.append({
            "role": role,
            "content": content,
            "reasoning": reasoning,
            "tool_calls": calls,
            # Which call this result answers. Carried through rather than
            # dropped: it is the only thing that pairs a result with its call
            # when two ran in parallel, and rebuilding it afterwards by
            # position gets that case wrong exactly when it matters.
            "tool_call_id": m.get("tool_call_id") or m.get("tool_use_id"),
            # A dataset that marks the weight of a turn keeps it. OpenAI's
            # fine-tuning format spells this `weight`; `train_on_turn` and
            # `train` are what the datasets that predate it use.
            **({"weight": m["weight"]} if "weight" in m else {}),
            # Some datasets mark which turns are worth learning from. Carried
            # through so a template can act on it, even though the trainer
            # currently learns from the whole conversation.
            "train": bool(m.get("train_on_turn", m.get("train", True))),
            "name": m.get("name"),
        })
    return _name_tool_results(out)


def _name_tool_results(messages: list[dict]) -> list[dict]:
    """Move a selector-chosen name onto the message, and nothing else.

    Naming an unnamed tool result -- from its own payload, or from the call it
    answers -- used to happen here. It now happens in
    `conversation.repair`, for two reasons. It belongs with the linking of
    `tool_call_id`, because both are the same inference from the same evidence
    and doing them apart got parallel calls wrong. And doing it here was
    silent: every unnamed result was named `tool` and nobody was told, so a
    dataset whose tool names never survived import looked like a dataset that
    never had any.

    What is left is the one thing that is not a guess: a name the person
    pointed at with a selector.
    """
    for m in messages:
        if m.get("_selected_name"):
            m["name"] = m.pop("_selected_name")
    return messages


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


def find_messages(row: dict, field: str | None = None,
                  selectors: dict | None = None) -> list[dict]:
    """The conversation in this row, wherever it lives.

    A named field wins, but only if this row actually has it. A dataset's
    format is detected from the union of every column across a sample, so a
    file that carries `messages` on most rows and `conversations` on the rest
    gets `messages_field: "messages"` -- and the rest then read as empty and
    were dropped without a word. Falling back keeps them.
    """
    key = field if (field and field in row) else first_present(row, _MESSAGE_FIELDS)
    return normalize_messages(row.get(key), selectors) if key else []


def messages_from_pair(row: dict, fmt: dict | None = None) -> list[dict]:
    """A conversation built out of instruction/response columns.

    Half the instruction datasets on the Hub have no `messages` column at all
    -- they are two or three flat columns, Alpaca-style. That is still a
    conversation: one user turn and one assistant turn. Rendering it as such is
    what lets a chat template be chosen for *any* dataset, rather than only for
    the ones that happen to ship their turns pre-assembled.

    Used only when a chat template is actually in play. Left to itself, an
    instruction dataset still renders through the instruction template, which
    is what a base model without a chat format wants.
    """
    fmt = fmt or {}
    instr_f = fmt.get("instruction_field") or first_present(row, _INSTRUCTION_FIELDS)
    resp_f = fmt.get("response_field") or first_present(row, _RESPONSE_FIELDS)
    if not instr_f or not resp_f:
        return []
    instruction, response = row.get(instr_f), row.get(resp_f)
    if not instruction or not response:
        return []
    # A separate "input" column is extra context for the instruction, not a
    # second instruction -- the same rule the instruction template follows.
    context = row.get("input") if instr_f != "input" else None
    user = str(instruction) + (("\n\n" + str(context)) if context else "")
    turns = []
    system_f = fmt.get("system_field") or first_present(row, _SYSTEM_FIELDS)
    if system_f and row.get(system_f):
        turns.append({"role": "system", "content": str(row[system_f])})
    turns.append({"role": "user", "content": user})
    turns.append({"role": "assistant", "content": str(response)})
    # Through the normaliser like any other conversation, so a reply with
    # <think> tags in it is unpacked into reasoning here too.
    return normalize_messages(turns)


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
                    add_generation_prompt: bool = False,
                    reasoning: bool = False,
                    tools_text: str = "",
                    extra: dict | None = None) -> str:
    """Render one example with a Jinja template.

    The same call renders the preview in the browser and the training batch on
    the runner, which is the only way the two can be guaranteed to agree.

    `extra` carries variables only some templates read -- `more_turns` for a
    format that ends its final message differently, `reasoning_effort` for one
    that writes it into the system message. A template that has never heard of
    them simply does not mention them, which is why they can be passed
    unconditionally.
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
        "tools_text": tools_text or "",
        "row": row or {},
        "add_generation_prompt": add_generation_prompt,
        # Whether the model should be invited to reason before answering. The
        # template decides what that looks like in its own idiom.
        "reasoning": reasoning,
    })
    ctx.update(extra or {})
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
    fmt = resolve_format(fmt)
    mode = fmt.get("mode", "auto")

    if mode == "jinja":
        template = fmt.get("template")
        if not template:
            return None
        text = render_template(
            template, row=row,
            messages=find_messages(row, fmt.get("messages_field"),
                                   fmt.get("selectors")),
            tools=find_tools(row, fmt.get("tools_field")),
            specials=fmt.get("specials"))
        return text.strip() or None

    text_field = fmt.get("text_field")
    if mode == "text" or (mode == "auto" and text_field and text_field in row):
        val = row.get(text_field or "text")
        return str(val) if val else None

    if mode in ("chat", "auto"):
        messages = find_messages(row, fmt.get("messages_field"),
                                 fmt.get("selectors"))
        # An instruction dataset is a conversation that has not been assembled
        # yet. Assemble it -- but only when a chat template was actually asked
        # for, either by name or by the model's own. Without one, an
        # instruction pair still belongs in the instruction template below,
        # which is the shape a base model expects.
        if not messages and (fmt.get("chat_template") or mode == "chat"):
            messages = messages_from_pair(row, fmt)
        if messages:
            # Through the canonical form rather than straight to the template.
            # That is what puts a tool call in front of the template in the
            # shape it reads, gives reasoning all three of its names, and
            # declares the tools in this format's own idiom. Rendering the
            # normalised messages directly -- as this did -- worked for plain
            # conversations and silently dropped every tool call in four of the
            # five formats.
            #
            # Imported here rather than at the top because `conversation` is
            # built on this module. The layering is deliberate: normalising is
            # the lower layer, the canonical record is the upper one, and only
            # this one function needs to reach upwards.
            from . import conversation
            conv, _ = conversation.repair(conversation.from_row(row, fmt))
            # Returned exactly as the template produced it, trailing newline
            # and all. Stripping would leave our training text one token
            # different from what transformers' own apply_chat_template emits
            # for the very template we save onto the tokenizer -- so anyone who
            # downloaded the model would prompt it slightly differently from
            # how it was taught.
            text = conversation.render(conv, fmt)
            return text if text.strip() else None

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
            roles, has_calls, has_reasoning = [], False, False
            for r in rows:
                for m in normalize_messages(r.get(msg_field)):
                    if m["role"] not in roles:
                        roles.append(m["role"])
                    has_calls = has_calls or bool(m["tool_calls"])
                    has_reasoning = has_reasoning or bool(m["reasoning"])
            out["roles"] = roles
            out["has_tool_calls"] = has_calls
            out["has_reasoning"] = has_reasoning
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
        turns = (find_messages(row, fmt.get("messages_field"))
                 or messages_from_pair(row, fmt))
        for m in turns:
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
    if (fmt.get("use_model_template") or fmt.get("chat_template")
            or fmt.get("chat_format")):
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
                  tools: list[dict] | None = None,
                  reasoning: bool = False) -> str:
    """Text for the model to continue, given the conversation so far."""
    fmt = resolve_format(fmt)
    messages = normalize_messages(messages, fmt.get("selectors"))
    style = conversation_style(fmt)
    specials = fmt.get("specials")

    template = fmt.get("chat_template") or (
        fmt.get("template") if fmt.get("mode") == "jinja" else None)

    if style == "chat":
        # The same canonical record training builds, so the prompt the model is
        # given at generation time is the prompt it was taught on -- including
        # the tools, which the playground now sends along so a tool-calling
        # model can actually be asked to call one.
        from . import conversation
        conv, _ = conversation.repair(
            conversation.from_messages(messages, tools))
        if template:
            # A model's own template understands add_generation_prompt and
            # emits the opening of the assistant turn itself.
            return conversation.render(conv, fmt, add_generation_prompt=True,
                                       reasoning=reasoning)
        body = conversation.render(conv, {**fmt, "chat_template":
                                          BUILTIN_CHAT_TEMPLATE})
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
    fmt = resolve_format(fmt)
    # A named format states exactly where a reply ends; nothing needs guessing.
    if fmt.get("stop"):
        return list(fmt["stop"])
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


# ---------------------------------------------------------------------------
# Named message formats
# ---------------------------------------------------------------------------

def resolve_format(fmt: dict | None) -> dict:
    """Fill in the template and tokens implied by a named chat format.

    `{"mode": "chat", "chat_format": "chatml"}` becomes a format carrying the
    ChatML Jinja and its special tokens. Done in one place so the preview, the
    trainer and the playground cannot each interpret the name differently.
    """
    fmt = dict(fmt or {})
    name = fmt.get("chat_format")
    if not name or fmt.get("chat_template"):
        return fmt
    spec = chat_formats.chat_format(name)
    if not spec:
        return fmt
    fmt["chat_template"] = spec["template"]
    # A format may end a reply differently once it is reasoning; Harmony does.
    fmt["stop"] = list(spec.get("reasoning_stop") if fmt.get("reasoning")
                       and spec.get("reasoning_stop") else spec["stop"])
    specials = dict(fmt.get("specials") or {})
    specials.setdefault("eos_token", spec["eos_token"])
    if spec["bos_token"]:
        specials.setdefault("bos_token", spec["bos_token"])
    fmt["specials"] = specials
    # A named format implies a conversation -- unless the run also names the
    # column its text lives in, which is how a from-scratch run says "the
    # corpus is prose, but reserve this format's tokens in the vocabulary
    # anyway". Forcing chat there would make every row of that corpus
    # unreadable.
    if fmt.get("mode") in (None, "auto") and not fmt.get("text_field"):
        fmt["mode"] = "chat"
    return fmt


def split_reasoning(text: str, fmt: dict | None = None) -> tuple[str, str]:
    """Separate a model's reasoning from its answer.

    The mirror of rendering. A reply generated in a reasoning format arrives as
    one string with the working still in it; the playground shows the two apart
    so the answer is readable and the reasoning is there when you want it.
    """
    if not text:
        return "", ""

    # Harmony keeps them in named channels rather than tags.
    if "<|channel|>" in text:
        reasoning, answer = "", text
        analysis = re.search(
            r"<\|channel\|>analysis<\|message\|>(.*?)(?:<\|end\|>|<\|start\|>|$)",
            text, re.DOTALL)
        if analysis:
            reasoning = analysis.group(1).strip()
        final = re.search(
            r"<\|channel\|>final<\|message\|>(.*?)(?:<\|end\|>|<\|return\|>|$)",
            text, re.DOTALL)
        if final:
            answer = final.group(1).strip()
        elif reasoning:
            answer = ""
        return reasoning, answer

    found = _THINK_RE.search(text)
    if found:
        return found.group(2).strip(), _THINK_RE.sub("", text, count=1).strip()

    # An unterminated block: the model was still thinking when it was cut off.
    open_tag = re.search(r"<(think|thinking|reasoning)>(.*)$", text,
                         re.DOTALL | re.IGNORECASE)
    if open_tag:
        return open_tag.group(2).strip(), ""
    return "", text


def tool_declaration(tools: list[dict] | None, format_id: str | None = None) -> str:
    """How this format tells the model which tools exist.

    Each family declares them differently, and the difference is not cosmetic:
    a model only learns to call a tool it was shown, in the shape it was shown
    it. Getting this wrong is how a fine-tune ends up inventing plausible
    function names that do not exist.

        harmony   a TypeScript-ish namespace in a developer message
        chatml    JSON schemas inside a <tools> block, the Hermes/Qwen layout
        llama3    a JSON array, which is what Llama 3.1 is shown
        inst      a JSON array, for Mistral's [AVAILABLE_TOOLS]
        plain     a readable list, because that format reserves nothing

    A schema is what the model needs -- the parameter names and types are the
    part it has to reproduce. The earlier version of this listed only names and
    descriptions for every format except Harmony, so a model in ChatML was
    asked to call `get_order` having never been told it takes an `order_id`.
    """
    tools = tools or []
    if not tools:
        return ""
    if format_id == "harmony":
        return _harmony_namespace(tools)
    if format_id == "chatml":
        return ("# Tools\n\nYou may call one or more of these functions. Their "
                "signatures are given inside <tools></tools>:\n<tools>\n"
                + "\n".join(json.dumps(_schema(t), ensure_ascii=False)
                            for t in tools)
                + "\n</tools>\n\nTo call one, write a <tool_call> block "
                  "holding {\"name\": ..., \"arguments\": ...}.")
    if format_id in ("llama3", "inst"):
        return json.dumps([_schema(t) for t in tools], ensure_ascii=False)
    return "Available tools:\n" + "\n".join(
        "- %s%s: %s" % (t["name"], _signature(t), t["description"])
        for t in tools)


def _schema(tool: dict) -> dict:
    """One tool in the nested shape every published tool encoding writes."""
    return {"type": "function",
            "function": {"name": tool["name"],
                         "description": tool.get("description") or "",
                         "parameters": tool.get("parameters")
                         or {"type": "object", "properties": {}}}}


def _signature(tool: dict) -> str:
    """`(order_id, limit?)` -- the parameter names, for the readable list."""
    params = tool.get("parameters") or {}
    props = params.get("properties") if isinstance(params, dict) else None
    if not isinstance(props, dict) or not props:
        return "()"
    required = set(params.get("required") or [])
    return "(%s)" % ", ".join(
        "%s%s" % (name, "" if name in required else "?") for name in props)


def _ts_type(schema: Any) -> str:
    """A JSON-schema property as the type Harmony's namespace block writes."""
    if not isinstance(schema, dict):
        return "any"
    kind = schema.get("type")
    if isinstance(kind, list):
        kind = next((k for k in kind if k != "null"), "any")
    if schema.get("enum"):
        return " | ".join(json.dumps(v, ensure_ascii=False) for v in schema["enum"])
    if kind == "array":
        return _ts_type(schema.get("items")) + "[]"
    return {"string": "string", "integer": "number", "number": "number",
            "boolean": "boolean", "object": "object"}.get(kind, "any")


def _harmony_namespace(tools: list[dict]) -> str:
    lines = ["# Tools", "", "## functions", "", "namespace functions {", ""]
    for t in tools:
        if t["description"]:
            lines.append("// %s" % t["description"].replace("\n", " "))
        params = t.get("parameters") or {}
        props = params.get("properties") if isinstance(params, dict) else None
        required = set(params.get("required") or []) if isinstance(params, dict) else set()
        if not props:
            lines.append("type %s = () => any;" % t["name"])
            lines.append("")
            continue
        lines.append("type %s = (_: {" % t["name"])
        for name, schema in props.items():
            if isinstance(schema, dict) and schema.get("description"):
                lines.append("// %s" % str(schema["description"]).replace("\n", " "))
            lines.append("%s%s: %s," % (name, "" if name in required else "?",
                                        _ts_type(schema)))
        lines.append("}) => any;")
        lines.append("")
    lines.append("} // namespace functions")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Selectors
# ---------------------------------------------------------------------------
#
# Auto-detection covers the shapes that recur, and there are always datasets it
# does not. A selector says exactly where a value lives instead of guessing:
#
#     tool_name        content.tool_name
#     tool_arguments   function.arguments
#     reasoning        extra.thinking
#     content          value
#
# Dotted paths with numeric indices, evaluated against the message object. A
# path may start with `content.` to look inside the message's content when that
# content is JSON -- which is where a tool result usually hides its own name.
# Deliberately not full JSONPath: a path someone can read at a glance, and
# predict the behaviour of, is worth more here than filters and wildcards.

_INDEX_RE = re.compile(r"^(.*?)\[(-?\d+)\]$")


def select(obj: Any, path: str | None) -> Any:
    """Follow a dotted path, returning None rather than raising."""
    if not path or obj is None:
        return None
    current = obj
    for raw in str(path).split("."):
        part, index = raw, None
        found = _INDEX_RE.match(raw)
        if found:
            part, index = found.group(1), int(found.group(2))
        if part:
            if isinstance(current, dict):
                current = current.get(part)
            else:
                current = getattr(current, part, None)
        if current is None:
            return None
        if index is not None:
            if not isinstance(current, (list, tuple)) or \
                    not (-len(current) <= index < len(current)):
                return None
            current = current[index]
    return current


def select_in_message(message: dict, path: str | None) -> Any:
    """A selector against a message, able to reach inside JSON content.

    `content.tool_name` looks in the parsed content when the message's own
    `content` is a JSON string, which is how most datasets carry a tool result.
    """
    if not path:
        return None
    value = select(message, path)
    if value is not None:
        return value
    if path.startswith("content."):
        raw = message.get("content")
        if isinstance(raw, str):
            try:
                return select(json.loads(raw), path[len("content."):])
            except (ValueError, TypeError):
                return None
    return None


# Applied after the role selector, for datasets whose role *values* are their
# own invention: {"tool_out": "tool", "narrator": "system"}.
ROLE_MAP_KEY = "role_map"

SELECTOR_FIELDS = [
    ("role", "Which speaker the turn belongs to"),
    ("content", "The words of the turn"),
    ("reasoning", "The model's working, if the data records it separately"),
    ("tool_name", "Which tool a call or a result belongs to"),
    ("tool_arguments", "The arguments a tool was called with"),
    ("tool_result", "The payload a tool returned"),
]
