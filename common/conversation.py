"""The canonical form of a training example: one conversation, one shape.

Everything upstream of this module is somebody else's idea of what a dataset
row looks like -- ShareGPT's `from`/`value`, Alpaca's three flat columns, a CSV
of questions and answers, a tool-calling set with the call under `function_call`
and the schema in a sibling column. Everything downstream of it -- the trainer,
the preview, the playground, the evaluator -- wants exactly one thing, and it
is this:

    {
      "messages": [
        {"role": "system",    "content": "..."},
        {"role": "user",      "content": "..."},
        {"role": "assistant", "reasoning": "...", "tool_calls": [
            {"id": "call_001", "type": "function",
             "function": {"name": "get_order",
                          "arguments": "{\\"order_id\\":\\"12345\\"}"}}]},
        {"role": "tool", "tool_call_id": "call_001",
         "name": "get_order", "content": "{...}"},
        {"role": "assistant", "content": "...", "weight": 1}
      ],
      "tools": [{"type": "function", "function": {"name": ..., "description":
                 ..., "parameters": {JSON Schema}}}],
      "meta": {...}
    }

## Why this shape and not a nicer one

It is the OpenAI chat format, and more precisely it is OpenAI's *fine-tuning*
file format -- the same object with `weight` allowed on assistant turns. That
is not a stylistic preference. `apply_chat_template(messages, tools=...)` is
defined over exactly this shape, so every chat template published with every
model on the Hub already knows how to read it. Since the entire point of this
module is that a conversation gets rendered by *whichever template the run
selected* -- frequently a base model's own Jinja, which we do not control and
cannot change -- the intermediate format has to be the one those templates
already speak. A cleaner data model (Anthropic's typed content blocks, the
Responses API's `input` items) would need a lossy conversion in front of every
template, and lossy conversions are where tool calls quietly go missing.

## The four places the ecosystem disagrees with itself

Each of these is a real bug rather than a matter of taste, and each is handled
here once so that nothing downstream has to think about it again.

**1. `arguments`: a JSON string or a JSON object?** OpenAI says a string.
Hugging Face's chat-template convention says an object, and its templates
overwhelmingly write `{{ tool_call.function.arguments | tojson }}`. Hand a
string to that and the model is trained on
`"{\\"order_id\\":\\"12345\\"}"` -- escaped, double-quoted, wrong. Hand an
object to a template that writes `{{ ... }}` bare and it is trained on a Python
dict repr with single quotes, which is also wrong. There is no value that is
correct for both, so `for_template` reads the template and passes the form it
asks for. See `arguments_form`.

**2. `reasoning` is spelled three ways.** Serving APIs and the templates built
for them (vLLM, SGLang, DeepSeek) use `reasoning_content`; several templates
read `thinking`; the Responses API says `reasoning`. Canonical here is
`reasoning`, because it reads best, and `for_template` puts all three aliases
on every message -- a fine-tune on a base whose template reads a different
spelling would otherwise drop every reasoning block and look merely mediocre.

**3. A tool result has to know which call it answers.** OpenAI links them with
`tool_call_id`. Harmony cannot render a tool result at all without the tool's
*name*, and most datasets carry neither. Both are reconstructed at import --
see `repair` -- and, once reconstructed, they are written down rather than
re-guessed on every read.

**4. Not every turn is worth learning from.** `weight: 0` on an assistant
message means "render this, do not train on it", which is what lets a run learn
the final answer without also learning to imitate the user. OpenAI's
fine-tuning format defines this and nothing else standard does, so it is the
one extension worth having.
"""
from __future__ import annotations

import json
import re
from typing import Any

from . import formatting

# The roles a canonical conversation may use. `function` is folded into `tool`
# on the way in -- it is the same role under the name OpenAI used before 2023,
# and carrying both means every template downstream has to check for two.
ROLES = ("system", "developer", "user", "assistant", "tool")

# Keys a canonical message may carry, in the order they are written out. Order
# matters only because a person reads these files, and a message whose `role`
# is the fourth key is harder to skim than one where it is the first.
MESSAGE_KEYS = ("role", "name", "content", "media", "reasoning", "tool_calls",
                "tool_call_id", "weight")

# Where a conversation lives in a row. `messages` is the name every consumer of
# this format already uses; the row may also carry `tools`, `meta`, and the
# studio's own reserved `split` column.
MESSAGES_KEY = "messages"
TOOLS_KEY = "tools"
META_KEY = "meta"


# ---------------------------------------------------------------------------
# Reading a row into canonical form
# ---------------------------------------------------------------------------

def from_row(row: dict, fmt: dict | None = None,
             selectors: dict | None = None) -> dict:
    """One dataset row, in whatever shape it arrived, as one conversation.

    The heavy normalisation -- role aliases, typed content parts, `<think>`
    blocks, tool calls under three different keys, explicit selectors for the
    datasets that match none of the patterns -- already lives in `formatting`
    and is reused rather than reimplemented. What happens here is the step
    after it: turning that normalised list into the canonical *record*, with
    tool calls in their nested OpenAI shape, results linked back to the calls
    they answer, and the tool schema attached.

    A row that is not a conversation at all still becomes one. An
    instruction/response pair is a two-turn conversation; a lone block of prose
    is a single assistant turn. That is what makes it possible to choose a chat
    template for *any* dataset rather than only for the ones that happened to
    ship their turns pre-assembled.
    """
    fmt = formatting.resolve_format(fmt or {})
    sel = selectors or fmt.get("selectors")

    messages = formatting.find_messages(row, fmt.get("messages_field"), sel)
    if not messages:
        messages = formatting.messages_from_pair(row, fmt)
    if not messages:
        # Nothing conversational in this row at all. Rather than dropping it,
        # read it the way the rest of the app would and call the result a
        # single assistant turn. A corpus of prose converted this way trains
        # exactly as it did before.
        #
        # Read as `auto`, never as the format that was passed in: a chat format
        # asked to render a row with no messages in it renders nothing, so
        # converting a plain-text corpus would have silently dropped every row
        # of it. `auto` is the one mode that will look at a `text` column.
        text = (formatting.format_example(
            row, {k: v for k, v in fmt.items()
                  if k not in ("mode", "chat_template")}) or "").strip()
        if not text:
            return {MESSAGES_KEY: [], TOOLS_KEY: [], META_KEY: {}}
        messages = [{"role": "assistant", "content": text, "reasoning": "",
                     "tool_calls": [], "name": None, "train": True}]

    conv = {
        MESSAGES_KEY: [_message(m) for m in messages],
        TOOLS_KEY: _tools(row, fmt),
        META_KEY: _meta(row),
    }
    return conv


def from_messages(messages: list[dict], tools: list[dict] | None = None,
                  meta: dict | None = None) -> dict:
    """A canonical conversation from a bare message list and tool schema.

    The entry point for everything that already has turns in hand rather than
    a dataset row: the playground's running conversation, an evaluation's
    prompt, an OpenAI-shaped request arriving at /v1/chat/completions. Put
    through the same normaliser as a dataset row, so a conversation typed into
    the browser and a conversation read off disk render identically.
    """
    normalised = formatting.normalize_messages(messages or [])
    return {
        MESSAGES_KEY: [_message(m) for m in normalised],
        TOOLS_KEY: [{"type": "function",
                     "function": {"name": t["name"],
                                  "description": t.get("description") or "",
                                  "parameters": t.get("parameters") or {}}}
                    for t in formatting.normalize_tools(
                        [{"function": t} if "function" not in t else t
                         for t in (tools or []) if isinstance(t, dict)])],
        META_KEY: dict(meta or {}),
    }


def _message(m: dict) -> dict:
    """One normalised message as a canonical one."""
    role = str(m.get("role") or "user").lower()
    if role == "function":
        role = "tool"
    if role not in ROLES:
        # An unfamiliar role is far more likely to be a dataset nobody has seen
        # than a mistake, so it is kept as written. `validate` reports it; the
        # person reading the report decides whether it is wrong.
        role = role or "user"

    out: dict[str, Any] = {"role": role}
    if name := m.get("name"):
        out["name"] = str(name)
    content = m.get("content")
    content = "" if content is None else str(content)
    if role == "tool":
        # The wrapper some datasets put around a result. It belongs to the
        # format, not to the data: the chat format adds its own when rendering,
        # so keeping this one would nest a <tool_response> inside a
        # <tool_response> for every result in the file.
        content = _INLINE_RESULT.sub("", content).strip()
    out["content"] = content
    # What the turn shows rather than says: pictures and clips, each
    # `{"kind", "ref"|"url"}`. `content` stays the words -- every consumer of
    # a message, from the validator to the loss mask to the browser, reads it
    # as text, and making it sometimes a list would have broken each of them
    # in a different place. The media rides beside it, and the template puts
    # the model's placeholder where each piece goes.
    media = [{k: v for k, v in mm.items() if k in ("kind", "ref", "url")}
             for mm in (m.get("media") or []) if isinstance(mm, dict)
             and mm.get("kind") and (mm.get("ref") or mm.get("url"))]
    if media:
        out["media"] = media
    if reasoning := (m.get("reasoning") or "").strip():
        out["reasoning"] = reasoning

    calls = [_tool_call(c) for c in (m.get("tool_calls") or [])]
    calls = [c for c in calls if c]
    if not calls:
        # Datasets that keep the call in the message body rather than in a
        # `tool_calls` list. Both of the widely used function-calling sets do
        # this and in different ways: one writes `<tool_call>{...}</tool_call>`
        # blocks inside the assistant's text, the other gives the turn a role
        # of its own and makes the body a JSON array of calls. Left as text,
        # the model is trained to *type* a call rather than to make one, and
        # nothing downstream -- the validator, the playground, the API -- can
        # see that the row contains a call at all.
        lifted, remainder = _calls_in_body(out["content"], role)
        if lifted:
            calls = lifted
            out["content"] = remainder
    if calls:
        out["tool_calls"] = calls
        if role not in ("assistant",):
            # A turn that makes a call is the assistant's, whatever the dataset
            # decided to name the role.
            out["role"] = role = "assistant"
    if tool_call_id := m.get("tool_call_id"):
        out["tool_call_id"] = str(tool_call_id)

    # `train` is what `formatting.normalize_messages` calls the flag when a
    # dataset marks which turns are worth learning from. Recorded as `weight`
    # here, which is the name OpenAI's fine-tuning format gives it -- and only
    # when it is actually False, so an ordinary conversation stays free of a
    # column of 1s that mean nothing.
    if "weight" in m:
        out["weight"] = 1 if m["weight"] in (True, 1, "1") else 0
    elif m.get("train") is False:
        out["weight"] = 0
    return out


# A role whose whole point is that the body is a call, not prose.
_CALL_ROLES = ("function_call", "tool_call", "functioncall", "toolcall",
               "tool_calls")
# `<tool_call>{...}</tool_call>`, the Hermes and Qwen layout, written into the
# assistant's text by the datasets that predate a structured field for it.
_INLINE_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_INLINE_RESULT = re.compile(r"</?tool_response>", re.IGNORECASE)


def _calls_in_body(content: str, role: str) -> tuple[list[dict], str]:
    """Tool calls written into a message's text, and what text is left.

    Returns ([], content) unchanged unless the body really does hold calls --
    a message that merely mentions a function name is prose and stays prose.
    """
    text = (content or "").strip()
    if not text:
        return [], content

    found = _INLINE_CALL.findall(text)
    if found:
        calls = [c for c in (_tool_call(j) for j in
                             (_loads(f) for f in found)) if c]
        if calls:
            return calls, _INLINE_CALL.sub("", text).strip()

    # A body that is nothing but a call, or a list of them. Only trusted where
    # the role says so, or where the whole body parses to objects that have a
    # name and arguments and nothing else -- otherwise a message quoting a JSON
    # payload would be turned into a call the conversation never made.
    if not text.startswith(("[", "{")):
        return [], content
    parsed = _loads(text)
    items = parsed if isinstance(parsed, list) else [parsed]
    if not items or not all(
            isinstance(i, dict) and i.get("name")
            and ("arguments" in i or "parameters" in i) for i in items):
        return [], content
    if role not in _CALL_ROLES and role != "assistant":
        return [], content
    calls = [c for c in (_tool_call(i) for i in items) if c]
    return (calls, "") if calls else ([], content)


def _loads(text: Any) -> Any:
    if isinstance(text, (dict, list)):
        return text
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return None


def _tool_call(call: Any) -> dict | None:
    """One tool call in the nested OpenAI shape, arguments as a JSON string.

    Accepts the flat shape `formatting` produces, the nested shape OpenAI uses,
    and a bare `{"name": ..., "arguments": {...}}`. Arguments are stored as a
    string because that is what the format specifies; `for_template` converts
    to an object for the templates that want one.
    """
    if not isinstance(call, dict):
        return None
    fn = call.get("function") if isinstance(call.get("function"), dict) else call
    name = fn.get("name") or call.get("name")
    if not name:
        return None
    args = fn.get("arguments")
    if args is None:
        args = fn.get("parameters")
    if args is None:
        args = call.get("arguments")
    if isinstance(args, (dict, list)):
        args = json.dumps(args, ensure_ascii=False)
    out = {"type": "function",
           "function": {"name": str(name), "arguments": str(args or "")}}
    if call_id := call.get("id"):
        out["id"] = str(call_id)
    return out


def _tools(row: dict, fmt: dict) -> list[dict]:
    """The tool schema for this row, in OpenAI's nested shape.

    `formatting.normalize_tools` flattens to {name, description, parameters},
    which is what the existing templates read; the canonical record nests it
    the way the format specifies, and `for_template` offers both.
    """
    flat = formatting.find_tools(row, fmt.get("tools_field"))
    return [{"type": "function",
             "function": {"name": t["name"],
                          "description": t.get("description") or "",
                          "parameters": t.get("parameters") or {
                              "type": "object", "properties": {}}}}
            for t in flat]


# Row-level keys worth keeping as provenance. `model` and `reasoning_effort`
# describe where a record came from and how it was generated -- they are not
# part of the conversation, so they do not sit beside `messages` where a
# template would trip over them. `reasoning_effort` is reachable from here
# because Harmony genuinely renders it into the system message.
_META_FIELDS = ("reasoning_effort", "source", "id", "dataset",
                "parallel_tool_calls")


def _meta(row: dict) -> dict:
    out = {k: row[k] for k in _META_FIELDS if row.get(k) not in (None, "")}
    if isinstance(row.get(META_KEY), dict):
        out = {**row[META_KEY], **out}
    return out


def to_row(conv: dict, split: str | None = None,
           extra: dict | None = None) -> dict:
    """The canonical conversation as a JSONL row.

    Keys are written in a fixed order because these files are read by people.
    Empty collections are dropped rather than written as `[]`: a dataset with
    no tools should not have a `tools` column of two thousand empty lists.
    """
    row: dict[str, Any] = {MESSAGES_KEY: [
        {k: m[k] for k in MESSAGE_KEYS if k in m and m[k] not in (None, "")}
        or {"role": m.get("role", "user"), "content": ""}
        for m in conv.get(MESSAGES_KEY) or []]}
    if tools := conv.get(TOOLS_KEY):
        row[TOOLS_KEY] = tools
    if meta := conv.get(META_KEY):
        row[META_KEY] = meta
    if extra:
        row.update(extra)
    if split:
        row["split"] = split
    return row


def is_canonical(row: dict) -> bool:
    """Whether this row is already in canonical form.

    Deliberately strict about `messages` being a list of objects with a role,
    and deliberately silent about everything else -- a row that carries extra
    columns beside its conversation is still a conversation.
    """
    msgs = row.get(MESSAGES_KEY)
    if not isinstance(msgs, list) or not msgs:
        return False
    return all(isinstance(m, dict) and isinstance(m.get("role"), str)
               for m in msgs)


# ---------------------------------------------------------------------------
# Repair
# ---------------------------------------------------------------------------

def _merge_split_replies(conv: dict) -> int:
    """Join an assistant turn that is only working to the answer after it.

    A writer asked for reasoning and an answer sometimes emits them as two
    consecutive assistant turns rather than as one turn with two fields. The
    text is right and the shape is not: rendered, it becomes an assistant turn
    that ends and a second one that starts, which teaches a model to close its
    reply and then open another -- a different spelling of the same habit that
    makes a reply never finish.

    Only the unambiguous case is merged: an assistant turn carrying working and
    nothing else, immediately followed by another assistant turn. A turn with
    words or a tool call in it is a real turn and is left where it is, and
    assistant/tool/assistant is a tool exchange rather than a split reply.
    """
    msgs = conv.get(MESSAGES_KEY) or []
    out: list[dict] = []
    merged = 0
    for m in msgs:
        prev = out[-1] if out else None
        orphan = (prev is not None
                  and prev.get("role") == "assistant"
                  and m.get("role") == "assistant"
                  and (prev.get("reasoning") or "").strip()
                  and not (prev.get("content") or "").strip()
                  and not prev.get("tool_calls"))
        if orphan:
            if not (m.get("reasoning") or "").strip():
                m["reasoning"] = prev["reasoning"]
            else:
                m["reasoning"] = "%s" % (prev["reasoning"] + "\n\n" + m["reasoning"])
            out[-1] = m
            merged += 1
            continue
        out.append(m)
    conv[MESSAGES_KEY] = out
    return merged


def repair(conv: dict) -> tuple[dict, list[str]]:
    """Fill in the links a dataset left implicit, and say what was inferred.

    Two things are nearly always missing from real data and are needed to
    render at all:

    * **`tool_call_id`** on a tool result. Without it there is no way to know
      which of two parallel calls a result answers, and the OpenAI format
      requires it.
    * **`name`** on a tool result. Harmony addresses a result by its author --
      `<|start|>functions.get_order` -- so an unnamed result cannot be written.

    Both are recovered from the assistant turn that preceded, and from the
    result's own payload where the dataset put the name inside it. What is
    returned alongside is the list of what had to be guessed, because an
    inference made silently is one nobody can check.
    """
    notes: list[str] = []
    if joined := _merge_split_replies(conv):
        notes.append("joined %d reply%s that had been split into a turn of "
                     "working and a turn of answer"
                     % (joined, "" if joined == 1 else "s"))
    if unbaked := _unbake_tools(conv):
        notes.append("took the tool schema out of %d system prompt%s, where it "
                     "was written in one format's syntax"
                     % (unbaked, "" if unbaked == 1 else "s"))
    # Calls still waiting for a result, oldest first. A conversation with two
    # parallel calls and two results pairs them in order, which is the only
    # ordering any of the datasets that do this actually use.
    pending: list[dict] = []
    named = linked = 0

    for m in conv.get(MESSAGES_KEY) or []:
        if m["role"] == "assistant" and m.get("tool_calls"):
            for i, c in enumerate(m["tool_calls"]):
                if not c.get("id"):
                    # An id the conversation did not have. Generated rather
                    # than left out: a result cannot point at a call with no
                    # name, and every downstream reader assumes the link.
                    c["id"] = "call_%d" % (len(pending) + i + 1)
                pending.append(c)
            continue
        if m["role"] != "tool":
            continue

        # An id the data already carries is the authority on which call this
        # answers, and it is the only thing that gets parallel calls right:
        # two calls issued together may be answered in either order, so
        # pairing by position would attach each result to the wrong one
        # exactly when a conversation has more than one call in flight.
        stated = m.get("tool_call_id")
        matched = next((c for c in pending if c["id"] == stated), None) \
            if stated else None

        if not m.get("name"):
            found = matched["function"]["name"] if matched \
                else _name_in_payload(m.get("content"))
            # Falling back to "the oldest call still waiting" is only safe
            # when the row named no call at all. A result that points at a
            # call which is not there is left unnamed rather than given
            # somebody else's name -- `validate` reports it, and a wrong name
            # would render as a confident lie.
            if not found and not stated and pending:
                found = pending[0]["function"]["name"]
            if found:
                m["name"] = found
                named += 1
        if not stated:
            match = next((c for c in pending
                          if c["function"]["name"] == m.get("name")), None) \
                or (pending[0] if pending else None)
            if match:
                m["tool_call_id"] = match["id"]
                linked += 1
        pending = [c for c in pending if c["id"] != m.get("tool_call_id")]

    if named:
        notes.append("named %d tool result%s from the call before it"
                     % (named, "" if named == 1 else "s"))
    if linked:
        notes.append("linked %d tool result%s to the call it answers"
                     % (linked, "" if linked == 1 else "s"))
    return conv, notes


# A tool schema written into the system prompt, in the syntax of whichever
# format the dataset was built for. The payload is required: the same prompts
# also *mention* the tags in prose -- "function signatures within <tools>
# </tools> XML tags" -- and matching that leaves the sentence saying "within
# XML tags" with a hole in the middle of it. That is the dataset's own
# instruction to the model and none of our business; only the schema goes.
_BAKED_TOOLS = re.compile(r"<tools>\s*[\[{].*?</tools>", re.DOTALL | re.IGNORECASE)


def _unbake_tools(conv: dict) -> int:
    """Remove a tool schema that a dataset wrote into its own system prompt.

    Several of the widely used function-calling sets ship the declaration
    inline -- `<tools>[...]</tools>` inside the system message -- because they
    were built for one particular format. Once the schema has been read into
    `tools`, leaving the inline copy means the model is shown the same
    functions twice, and shown them in a syntax that may not be the one this
    run is training in: a conversation rendered in Harmony would carry a
    ChatML-style block *and* a namespace block, disagreeing about nothing but
    costing a thousand tokens of context to say so.

    Only ever removed when the schema was actually captured. A system prompt
    that mentions tools nobody parsed keeps every word of it.
    """
    if not conv.get(TOOLS_KEY):
        return 0
    changed = 0
    for m in conv.get(MESSAGES_KEY) or []:
        if m.get("role") not in ("system", "developer"):
            continue
        content = m.get("content") or ""
        if not _BAKED_TOOLS.search(content):
            continue
        stripped = _BAKED_TOOLS.sub("", content)
        # The sentence that introduced the block is left alone -- it is
        # instruction, not schema -- but the hole it leaves is closed up.
        m["content"] = re.sub(r"\n{3,}", "\n\n", stripped).strip()
        changed += 1
    return changed


def _name_in_payload(content: Any) -> str | None:
    """The tool's name, when the dataset put it inside the result's JSON."""
    if not isinstance(content, str) or not content.strip().startswith("{"):
        return None
    try:
        payload = json.loads(content)
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    for key in ("tool_name", "name", "function", "tool"):
        if isinstance(payload.get(key), str) and payload[key].strip():
            return payload[key]
    return None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(conv: dict) -> list[dict]:
    """What is wrong with this conversation, in the order it matters.

    Each problem is {level, code, message}. `error` means the row cannot be
    rendered or would teach the model something false; `warn` means it will
    render but is probably not what anybody intended.

    The point of validating at conversion time rather than at training time is
    that a dataset of forty thousand rows with two hundred broken ones is a
    thing you want to *see*, on the page where you can fix it -- not a slightly
    worse loss curve three hours later.
    """
    out: list[dict] = []
    msgs = conv.get(MESSAGES_KEY) or []
    if not msgs:
        return [{"level": "error", "code": "empty",
                 "message": "This row has no messages in it."}]

    declared = {t["function"]["name"] for t in (conv.get(TOOLS_KEY) or [])
                if isinstance(t.get("function"), dict)
                and t["function"].get("name")}
    call_ids: set[str] = set()
    called: set[str] = set()

    for i, m in enumerate(msgs):
        role = m.get("role")
        if role not in ROLES:
            out.append({"level": "warn", "code": "role",
                        "message": "Message %d has the role %r, which no chat "
                                   "template knows what to do with." % (i + 1, role)})
        if role == "assistant":
            if not (m.get("content") or "").strip() \
                    and not m.get("tool_calls") and not m.get("reasoning"):
                out.append({"level": "error", "code": "empty_assistant",
                            "message": "Message %d is an assistant turn with "
                                       "nothing in it -- no answer, no tool "
                                       "call, no reasoning." % (i + 1)})
            for c in m.get("tool_calls") or []:
                name = c["function"]["name"]
                called.add(name)
                if c.get("id"):
                    call_ids.add(c["id"])
                args = c["function"].get("arguments") or ""
                if args.strip() and not _parses(args):
                    out.append({"level": "error", "code": "bad_arguments",
                                "message": "The call to %s in message %d has "
                                           "arguments that are not valid JSON."
                                           % (name, i + 1)})
        elif role == "tool":
            if not m.get("tool_call_id"):
                out.append({"level": "warn", "code": "unlinked_result",
                            "message": "Message %d is a tool result that does "
                                       "not say which call it answers."
                                       % (i + 1)})
            elif m["tool_call_id"] not in call_ids:
                out.append({"level": "error", "code": "orphan_result",
                            "message": "Message %d answers a call (%s) that "
                                       "never happened in this conversation."
                                       % (i + 1, m["tool_call_id"])})
            if not m.get("name"):
                out.append({"level": "warn", "code": "unnamed_result",
                            "message": "Message %d is a tool result with no "
                                       "tool name. Some formats -- Harmony "
                                       "among them -- cannot render it."
                                       % (i + 1)})

    if undeclared := called - declared:
        out.append({
            "level": "warn" if declared else "error", "code": "undeclared_tool",
            "message": "The assistant calls %s, which %s. A model trained on "
                       "this learns to invent tools it was never shown."
                       % (", ".join(sorted(undeclared)),
                          "is not in this row's tool list" if declared
                          else "has no tool definitions at all")})

    # A conversation whose last turn is the user's has nothing to learn from:
    # every token of it would be masked, or -- worse, if nothing is masked --
    # the model learns to write the user's next question.
    if msgs[-1].get("role") not in ("assistant",):
        out.append({"level": "warn", "code": "no_final_answer",
                    "message": "This conversation does not end with an "
                               "assistant turn, so there is no reply in it to "
                               "learn from."})

    if not any(m.get("role") == "assistant" and _trainable(m) for m in msgs):
        out.append({"level": "error", "code": "nothing_to_train",
                    "message": "Every assistant turn here is marked weight 0, "
                               "so this row would contribute nothing."})
    return out


def _parses(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except ValueError:
        return False


def _trainable(m: dict) -> bool:
    return m.get("role") == "assistant" and m.get("weight", 1) != 0


def validate_many(convs: list[dict]) -> dict:
    """One report over many rows: what is wrong, how often, and where.

    Counted by code rather than listed one per row -- "412 rows have a tool
    result that answers no call" is actionable, and four hundred copies of the
    same sentence is not. A few example row numbers come with each, because the
    first thing anybody does with a count is go and look at one.
    """
    kinds: dict[str, dict] = {}
    bad_rows = 0
    for i, conv in enumerate(convs):
        problems = validate(conv)
        if any(p["level"] == "error" for p in problems):
            bad_rows += 1
        for p in problems:
            slot = kinds.setdefault(p["code"], {
                "code": p["code"], "level": p["level"], "count": 0,
                "message": p["message"], "rows": []})
            slot["count"] += 1
            # The level is the worst seen for this code: the same code is an
            # error in a row with tools declared and a warning in one without.
            if p["level"] == "error":
                slot["level"] = "error"
            if len(slot["rows"]) < 5:
                slot["rows"].append(i)
    order = {"error": 0, "warn": 1, "info": 2}
    problems = sorted(kinds.values(),
                      key=lambda p: (order.get(p["level"], 3), -p["count"]))
    return {"rows": len(convs), "rows_with_errors": bad_rows,
            "problems": problems,
            "ok": not any(p["level"] == "error" for p in problems)}


# ---------------------------------------------------------------------------
# Reading a reply back
# ---------------------------------------------------------------------------
#
# Rendering has an inverse, and until now nothing implemented it. A model
# trained to call tools emitted a call, the playground printed it as prose, and
# there was no way to tell a working tool-caller from one that had learned to
# type angle brackets. Worse, a conversation could not continue past a call:
# nothing knew a call had been made, so nothing could hand back a result.
#
# Each format's calls are recognised by the syntax that format renders. The
# fallbacks are deliberately generous -- a half-trained model produces
# half-formed calls, and showing "it tried to call get_order and the arguments
# would not parse" is far more useful than showing the raw text.

_HARMONY_CALL = re.compile(
    r"<\|start\|>?assistant to=functions\.([\w.-]+)"
    r".*?<\|message\|>(.*?)(?:<\|call\|>|<\|end\|>|$)", re.DOTALL)
# The same, for a reply that begins mid-message: the prompt already emitted
# `<|start|>assistant`, so the model's own output opens at the recipient.
_HARMONY_TAIL = re.compile(
    r"(?:^|<\|start\|>assistant)\s*to=functions\.([\w.-]+)"
    r".*?<\|message\|>(.*?)(?:<\|call\|>|<\|end\|>|$)", re.DOTALL)
_CHATML_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_INST_CALL = re.compile(r"\[TOOL_CALLS\]\s*(\[.*?\]|\{.*?\})", re.DOTALL)
_PLAIN_CALL = re.compile(r"^\s*\w+ calls ([\w.-]+)\((.*)\)\s*$", re.MULTILINE)
# A bare JSON object naming a function, which is what Llama 3 emits and what a
# model that half-learned any of the others falls back to.
_BARE_CALL = re.compile(
    r'\{\s*"name"\s*:\s*"([\w.-]+)"\s*,\s*"(?:parameters|arguments)"\s*:\s*'
    r'(\{.*?\})\s*\}', re.DOTALL)


def split_progressive(text: str, fmt: dict | None = None,
                      reasoning_on: bool = False) -> tuple[str, str]:
    """Working and answer, in a reply that has not finished arriving.

    The front half of `parse_reply`, on its own, so a reply can be shown
    correctly *while* it streams instead of being rearranged once it stops.
    Without it the playground put a model's entire working in the answer bubble
    and then snatched it back at the end -- and for a reasoning model that is
    most of what you watch it do.

    Reading it the same way `parse_reply` will is the whole point: two readings
    means the answer rearranges itself when the stream ends, which looks exactly
    like a bug whichever of them is right.

    Called once per token against the whole reply so far, so it is deliberately
    stateless. What is safe to *show* is the caller's problem: a marker still
    arriving one character at a time reads as ordinary text until its last
    bracket lands. See `_showable` in runner/inference.
    """
    fmt = formatting.resolve_format(fmt or {})
    return formatting.split_reasoning(
        _reopen_reasoning(text or "", fmt, reasoning_on), fmt)


def parse_reply(text: str, fmt: dict | None = None,
                reasoning_on: bool = False) -> dict:
    """A generated reply as {content, reasoning, tool_calls}.

    The mirror of `render`, and the thing that lets the playground show a tool
    call as a call rather than as a line of syntax -- and lets a conversation
    carry on past one, by handing back the result the dataset recorded.

    `reasoning_on` says the prompt invited the model to reason, which changes
    how the reply has to be read: the *opening* of the reasoning block was
    emitted by the prompt, so the model never produces one and the reply begins
    already inside it. Without knowing that, the first thing the splitter meets
    is a closing tag with nothing before it, and the model's entire working is
    returned as its answer.

    Never raises. A model that produces malformed JSON has produced a tool call
    with malformed JSON, which is a fact worth showing, not an exception.
    """
    fmt = formatting.resolve_format(fmt or {})
    text = _reopen_reasoning(text or "", fmt, reasoning_on)
    reasoning, answer = formatting.split_reasoning(text, fmt)
    if reasoning_on and not reasoning and not fmt.get("chat_format") in (
            "harmony",) and "<think>" not in text:
        # A format with no reasoning tokens at all -- the plain one -- marks
        # the block with a phrase rather than a tag, so there is nothing to
        # close. Everything before the model announces its next move is the
        # working.
        reasoning, answer = _split_untagged(answer)
    name = fmt.get("chat_format")
    calls: list[dict] = []
    body = answer

    # Calls are searched for in the WHOLE reply and stripped out of the answer.
    # Those are two different strings and it matters which is which: Harmony
    # puts a call in its own message on the commentary channel, and
    # `split_reasoning` -- whose job is to find the *final* channel -- returns
    # an empty answer for a reply that is nothing but a call. Searching the
    # answer therefore found nothing at all, which is precisely the case that
    # has to work.
    def take(pattern, group_name=1, group_args=2):
        nonlocal body
        found = list(pattern.finditer(text))
        for m in found:
            calls.append({"name": m.group(group_name),
                          "arguments": (m.group(group_args) or "").strip()})
        if found:
            body = pattern.sub("", body).strip()
        return bool(found)

    def take_json(pattern):
        nonlocal body
        found = False
        for m in pattern.finditer(text):
            parsed = _calls_from_json(m.group(1))
            calls.extend(parsed)
            found = found or bool(parsed)
        if found:
            body = pattern.sub("", body).strip()
        return found

    if name == "harmony":
        for pattern in (_HARMONY_CALL, _HARMONY_TAIL):
            if take(pattern):
                break
    elif name == "inst":
        take_json(_INST_CALL)
    elif name == "plain":
        take(_PLAIN_CALL)

    if not calls:
        take_json(_CHATML_CALL)
    if not calls:
        take(_BARE_CALL)
    if not calls:
        # The whole reply IS the call, as bare JSON. This is what a model
        # trained on Mistral's own template produces once the reply has been
        # decoded: the marker in front of it, `[TOOL_CALLS]`, is a special
        # token, and decoding for display strips special tokens -- so what
        # arrives here is an unannounced `[{"name": ..., "arguments": "..."}]`
        # and every pattern above is looking for an announcement.
        #
        # Recognised only when the entire answer parses as a call: a reply
        # that merely contains JSON is a reply, and a model asked to produce
        # JSON must not have its answer turned into a phantom tool call.
        whole = body.strip()
        if whole[:1] in ("{", "["):
            parsed = [c for c in _calls_from_json(whole) if c.get("arguments")]
            if parsed:
                calls.extend(parsed)
                body = ""

    if calls:
        # Once the calls are structure, any call syntax still in the body is
        # residue -- not prose. Stripped here rather than in each branch above,
        # because which branch matched depends on what the model emitted and
        # the leftovers do not: a truncated second copy, the brackets of
        # `[TOOL_CALLS] [ {...} ]`, a comma between two calls. Returning any of
        # it as `content` puts a line of raw JSON above the reply in every
        # client that renders content and tool_calls together.
        for pattern in (_CHATML_CALL, _INST_CALL, _BARE_CALL, _PLAIN_CALL):
            body = pattern.sub("", body)

    # Whatever marker the format ends a message with, left over because
    # generation stopped on it rather than before it.
    for token in ("<|call|>", "<|end|>", "<|return|>", "<|eom_id|>",
                  "<|eot_id|>", "<|im_end|>", "</s>", "[TOOL_CALLS]",
                  "<tool_call>", "</tool_call>"):
        body = body.replace(token, "")
    body = body.strip()

    # What is left after all that is punctuation or nothing.
    if calls and not re.search(r"\w", body):
        body = ""

    return {"content": body, "reasoning": reasoning,
            "tool_calls": [_parsed_call(c, i) for i, c in enumerate(calls)]}


def _reopen_reasoning(text: str, fmt: dict, reasoning_on: bool) -> str:
    """Put back the opening of a reasoning block the prompt already emitted.

    Generation starts *after* the prompt, so when the prompt ends by opening
    the model's working the reply arrives with only the closing half of it.
    Reconstructed here, in one place, rather than in each caller -- the runner
    did this itself and the controller did not, so the same reply was read two
    different ways depending on which asked.
    """
    if not text:
        return text
    if "</think>" in text and "<think>" not in text:
        return "<think>\n" + text
    if not reasoning_on:
        return text
    if fmt.get("chat_format") == "harmony":
        # The prompt ended with `<|channel|>analysis<|message|>`. A reply that
        # opens with a marker of its own decided not to reason after all.
        if not text.lstrip().startswith("<|"):
            return "<|start|>assistant<|channel|>analysis<|message|>" + text
        return text
    if "<think>" not in text and "</think>" not in text \
            and fmt.get("chat_format") not in ("plain", None):
        return "<think>\n" + text
    return text


# Where the working stops and the reply starts, in a format that marks neither
# with a token. Only `plain` is in this position, and only because it reserves
# nothing by design.
_PLAIN_TURN = re.compile(r"^\s*\w+(?: calls | thinks: |: )", re.MULTILINE)


def _split_untagged(text: str) -> tuple[str, str]:
    found = _PLAIN_TURN.search(text)
    if not found or not found.start():
        return text.strip(), ""
    return text[:found.start()].strip(), text[found.start():].strip()


def _calls_from_json(blob: str) -> list[dict]:
    """`{"name":..,"arguments":..}`, or a list of them."""
    try:
        parsed = json.loads(blob)
    except ValueError:
        return []
    items = parsed if isinstance(parsed, list) else [parsed]
    out = []
    for item in items:
        if not isinstance(item, dict):
            continue
        called = item.get("name") or item.get("function")
        if isinstance(called, dict):
            called = called.get("name")
        if not called:
            continue
        args = item.get("arguments")
        if args is None:
            args = item.get("parameters")
        out.append({"name": str(called),
                    "arguments": json.dumps(args, ensure_ascii=False)
                    if isinstance(args, (dict, list)) else str(args or "")})
    return out


def _parsed_call(call: dict, index: int) -> dict:
    """One parsed call, in canonical shape, saying whether it is well formed."""
    args = call.get("arguments") or ""
    return {
        "id": "call_%d" % (index + 1), "type": "function",
        "function": {"name": call["name"], "arguments": args},
        # Not part of the stored format -- this is a fact about what the model
        # just produced, and the playground shows it. A call whose arguments
        # will not parse is the single most common thing a half-trained
        # tool-caller emits, and "it called get_order but the JSON is broken"
        # is a much more useful thing to read than the raw text.
        "valid": _parses(args) if args.strip() else False,
    }


# ---------------------------------------------------------------------------
# Handing it to a template
# ---------------------------------------------------------------------------

# A template that writes `arguments | tojson` wants an object; one that writes
# `{{ arguments }}` wants the string. There is no third possibility worth
# supporting and no value that satisfies both, so the template is read.
_TOJSON_ARGS = re.compile(r"arguments\s*\|\s*tojson")


def arguments_form(template: str | None) -> str:
    """Whether this template wants tool arguments as an object or a string.

    Hugging Face's convention is an object and its templates say so out loud,
    by piping through `tojson`. OpenAI's is a string, and a template written
    against that interpolates it directly. Guessing wrong is not subtle: one
    way trains the model on escaped JSON inside a quoted string, the other on
    a Python dict repr with single quotes.
    """
    return "object" if template and _TOJSON_ARGS.search(template) else "string"


# Where a picture goes in the text the model reads. Every vision-language
# model has a token for "an image is here" and they all spell it differently;
# a format names its own, and this is the fallback that at least marks the
# place rather than dropping it. Audio the same.
DEFAULT_PLACEHOLDERS = {"image": "<image>", "audio": "<audio>", "video": "<video>"}


def placeholders_for(fmt: dict | None) -> dict:
    fmt = fmt or {}
    return {"image": fmt.get("image_token") or DEFAULT_PLACEHOLDERS["image"],
            "audio": fmt.get("audio_token") or DEFAULT_PLACEHOLDERS["audio"],
            "video": fmt.get("video_token") or DEFAULT_PLACEHOLDERS["video"]}


def with_placeholders(content: str, media: list | None, tokens: dict) -> str:
    """The text of a turn with a token in front for each thing it shows.

    In front rather than behind, which is the convention the published
    vision templates follow (LLaVA, Qwen-VL, Idefics all put the image before
    the question). One token per item, in order, separated by newlines from
    the words so a placeholder is never glued to a word.
    """
    if not media:
        return content
    marks = [tokens.get(mm.get("kind"), "") for mm in media]
    marks = [t for t in marks if t]
    if not marks:
        return content
    return "\n".join(marks) + ("\n" + content if content else "")


def for_template(conv: dict, template: str | None = None,
                 placeholders: dict | None = None) -> list[dict]:
    """The messages as a template should see them, aliases and all.

    Three things happen here that do not belong in the stored record:

    * `reasoning` is repeated as `reasoning_content` and `thinking`, so a
      template written against any of the three spellings finds it.
    * tool calls are given both the flat `name`/`arguments` this studio's own
      formats read and the nested `function.*` every published template reads,
      with arguments in whichever form the template asked for.
    * `train` is offered beside `weight`, because that is the name the older
      templates in this repo use.

    Duplication in a render context costs nothing and is invisible in the
    output. Duplication in a stored file is a second copy that can disagree
    with the first, which is why it is done here and not in `to_row`.
    """
    form = arguments_form(template)
    ids = _template_ids(conv)
    out = []
    for m in conv.get(MESSAGES_KEY) or []:
        reasoning = m.get("reasoning") or ""
        called = m.get("tool_call_id")
        item = {
            "role": m.get("role", "user"),
            "content": with_placeholders(m.get("content") or "", m.get("media"),
                                         placeholders or DEFAULT_PLACEHOLDERS),
            # Offered to a template that wants to place them itself.
            "media": list(m.get("media") or []),
            "name": m.get("name"),
            "reasoning": reasoning,
            "reasoning_content": reasoning,
            "thinking": reasoning,
            "tool_call_id": ids.get(called, called),
            "weight": m.get("weight", 1),
            "train": m.get("weight", 1) != 0,
            "tool_calls": [_call_for_template(c, form, ids)
                           for c in (m.get("tool_calls") or [])],
        }
        out.append(item)
    return out


# Mistral's published template does not merely read a tool call's id, it
# *validates* it: `Tool call IDs should be alphanumeric strings with length 9`,
# raised from inside the Jinja. Nothing else in the format says so, OpenAI's own
# ids do not satisfy it, and neither does anything a dataset is likely to carry.
_ID_LEN = 9
_ID_ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def _template_ids(conv: dict) -> dict:
    """A rendering-safe id for every tool call, mapped from its real one.

    The stored record keeps whatever id the data used -- that id is the link
    between a call and its result, and rewriting it on disk would be rewriting
    somebody's data to suit one vendor's template. So the substitution happens
    here, where the conversation is being handed to Jinja, and both halves of
    the link are substituted together so they still point at each other.

    Derived from the original id rather than counted, so the same conversation
    renders identically every time -- which matters because a run's training
    text and its prompts have to agree token for token.
    """
    import hashlib
    out: dict[str, str] = {}
    for m in conv.get(MESSAGES_KEY) or []:
        for c in m.get("tool_calls") or []:
            real = c.get("id")
            if not real or real in out:
                continue
            digest = hashlib.sha1(str(real).encode("utf-8")).digest()
            out[real] = "".join(
                _ID_ALPHABET[b % len(_ID_ALPHABET)] for b in digest[:_ID_LEN])
    return out


def _call_for_template(call: dict, form: str, ids: dict | None = None) -> dict:
    name = call["function"]["name"]
    raw = call["function"].get("arguments") or ""
    args: Any = raw
    if form == "object":
        try:
            parsed = json.loads(raw) if raw.strip() else {}
            args = parsed if isinstance(parsed, (dict, list)) else raw
        except ValueError:
            # Arguments that will not parse are passed through as written.
            # `validate` has already reported them; mangling them here would
            # hide the row that needs fixing.
            args = raw
    real = call.get("id")
    return {
        "id": (ids or {}).get(real, real), "type": "function",
        # Flat, for this studio's own formats in chat_formats.py.
        "name": name, "arguments": args,
        # Nested, for every template published with a model on the Hub.
        "function": {"name": name, "arguments": args},
    }


def tools_for_template(conv: dict) -> list[dict]:
    """Tool definitions in both the nested and the flattened shape.

    The nested one is what published templates iterate; the flat `name` and
    `description` are what `formatting.BUILTIN_CHAT_TEMPLATE` and the Harmony
    namespace writer read. Same objects, two ways in.
    """
    out = []
    for t in conv.get(TOOLS_KEY) or []:
        fn = t.get("function") or {}
        out.append({"type": "function", "function": fn,
                    "name": fn.get("name"), "description": fn.get("description") or "",
                    "parameters": fn.get("parameters") or {}})
    return out


def split_for_trial(conv: dict) -> tuple[list[dict], list[dict]]:
    """Cut a conversation where the model would have to take over.

    Everything up to and including the last user turn is the *prompt*: what a
    model would be given. Everything after it is what the data says should
    happen next -- which may be several messages, because an answer that calls
    a tool is a call, a result, and then a reply.

    This is what makes a held-out row usable in the playground. Sending the
    whole conversation would be asking the model to continue past an answer it
    was already given; sending only the last user message would throw away the
    context the reply depends on.
    """
    msgs = conv.get(MESSAGES_KEY) or []
    cut = next((i + 1 for i in range(len(msgs) - 1, -1, -1)
                if msgs[i].get("role") == "user"), None)
    if cut is None:
        # No user turn at all -- a corpus row, or a conversation that opens
        # with the assistant. Everything the assistant says is what it should
        # have produced, and whatever came before it is the prompt.
        cut = next((i for i, m in enumerate(msgs)
                    if m.get("role") == "assistant"), len(msgs))
    return msgs[:cut], msgs[cut:]


def flat_tools(conv: dict) -> list[dict]:
    """Tools as `formatting.tool_declaration` wants them."""
    return [{"name": t["name"], "description": t["description"],
             "parameters": t["parameters"]} for t in tools_for_template(conv)]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render(conv: dict, fmt: dict | None = None, *,
           add_generation_prompt: bool = False, reasoning: bool = False,
           more_turns: bool = False) -> str:
    """The exact text a model is trained on, or prompted with.

    One function, used by the dataset preview, by both trainers, by the
    playground and by the evaluator. That is the whole reason this lives in
    `common`: a preview built by lookalike code is not a preview.

    `more_turns` says that what is being rendered is a *prefix* of a longer
    conversation. Only Harmony cares -- it ends the model's last message with
    `<|return|>` and every earlier one with `<|end|>` -- but it has to be
    passed through, or measuring where each turn starts (see `segments`) would
    do it by rendering prefixes that each claim to be the end.
    """
    fmt = formatting.resolve_format(fmt or {})
    template = fmt.get("chat_template") or (
        fmt.get("template") if fmt.get("mode") == "jinja" else None) \
        or formatting.BUILTIN_CHAT_TEMPLATE
    tools = tools_for_template(conv)
    text = formatting.render_template(
        template,
        row=conv.get(META_KEY) or {},
        messages=for_template(conv, template, placeholders_for(fmt)),
        tools=tools or None,
        specials=fmt.get("specials"),
        add_generation_prompt=add_generation_prompt,
        reasoning=reasoning,
        tools_text=formatting.tool_declaration(flat_tools(conv),
                                               fmt.get("chat_format")),
        extra={"more_turns": more_turns,
               "reasoning_effort": (conv.get(META_KEY) or {}).get(
                   "reasoning_effort") or "medium"})
    return text


def segments(conv: dict, fmt: dict | None = None) -> list[dict]:
    """The rendered text cut into one piece per message, each marked trainable.

    This is what makes `weight` mean something. To train on the assistant's
    replies but not on the user's questions you need to know which characters
    of the finished string belong to which turn -- and the finished string is
    produced by a Jinja template that may be the base model's own, may fold the
    system prompt into the first user turn, and is in no sense a concatenation
    of independently renderable pieces.

    So the boundaries are *measured* rather than assumed: render the first
    message, then the first two, then the first three. Each render must extend
    the one before it, and the difference is that message's text. Where a
    template breaks that -- Qwen3's strips reasoning from all but the final
    assistant turn, so an earlier prefix is not a prefix of a later one -- this
    returns a single un-maskable segment instead of a wrong answer, and the
    caller trains on the whole conversation and says so.

    Returns [{role, text, trainable}], or one segment with `exact: False` when
    the boundaries could not be established.
    """
    msgs = conv.get(MESSAGES_KEY) or []
    whole = render(conv, fmt)
    if not msgs:
        return []
    if len(msgs) == 1:
        return [{"role": msgs[0].get("role"), "text": whole,
                 "trainable": _trainable(msgs[0]), "exact": True}]

    out: list[dict] = []
    previous = ""
    for i in range(len(msgs) - 1):
        # Rendered as a prefix, so a format that marks its final turn
        # differently does not mark every intermediate one that way too.
        prefix = render({**conv, MESSAGES_KEY: msgs[:i + 1]}, fmt,
                        more_turns=True)
        if not prefix.startswith(previous) or not whole.startswith(prefix):
            return [{"role": None, "text": whole, "trainable": True,
                     "exact": False}]
        out.append({"role": msgs[i].get("role"), "text": prefix[len(previous):],
                    "trainable": _trainable(msgs[i]), "exact": True})
        previous = prefix
    out.append({"role": msgs[-1].get("role"), "text": whole[len(previous):],
                "trainable": _trainable(msgs[-1]), "exact": True})
    return out


# What a run learns from, when the data is a conversation.
#
#   assistant   only the model's own turns -- the standard for supervised
#               fine-tuning, and what almost everybody means
#   all         every token, including the user's questions
#
# The difference is not small. Training on the user's turns teaches the model
# to write the *next question*, which is why a model trained that way tends to
# answer and then carry on inventing a conversation with itself.
TRAIN_ON = {
    "assistant": "The assistant's replies only",
    "all": "Every token, including the questions",
}
DEFAULT_TRAIN_ON = "assistant"


def trainable_spans(conv: dict, fmt: dict | None = None,
                    train_on: str = DEFAULT_TRAIN_ON
                    ) -> tuple[str, list[tuple[int, int]], bool]:
    """The rendered text, the character ranges to learn from, and whether the
    ranges could be established at all.

    A thin wrapper over `segments` for the trainers, which need the text once
    and the ranges as offsets into it. An empty list of ranges means "learn
    from all of it": that is what `train_on="all"` asks for, and it is also
    the honest answer when a template turned out not to be prefix-stable, in
    which case the third value is False so the caller can say so rather than
    quietly training on a different thing than it was asked to.
    """
    parts = segments(conv, fmt)
    text = "".join(p["text"] for p in parts)
    if train_on == "all":
        return text, [], True
    if not parts or not parts[0].get("exact", True):
        return text, [], False

    spans: list[tuple[int, int]] = []
    at = 0
    for p in parts:
        end = at + len(p["text"])
        if p["trainable"]:
            spans.append((at, end))
        at = end
    return text, spans, True
