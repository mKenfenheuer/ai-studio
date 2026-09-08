"""Message-boundary formats for models trained from scratch.

A conversation rendered as plain text -- "user: hello" -- teaches a model
nothing about where a turn begins or ends. The word "assistant" is three or
four ordinary BPE pieces, indistinguishable from the same word inside a
sentence, so the model has no reliable signal for "a turn ended here" and
generation has nothing dependable to stop on.

Real chat models solve this with **special tokens**: single, atomic ids that
appear nowhere in ordinary text and mean exactly one thing. A from-scratch run
trains its own tokenizer, so it can reserve those ids properly -- something a
fine-tune cannot do without resizing the base model's embedding table.

Each format below declares the tokens, the Jinja that lays a conversation out
with them, and where generation must stop. The template is written onto the
finished tokenizer as its `chat_template`, so the resulting model is
self-describing: the playground reads it back exactly as it reads Qwen's.

## Tool calling

Every format here renders tool calls and tool results, each in the encoding
its own family published:

    plain     readable words, no reserved tokens
    chatml    <tool_call>/<tool_response> JSON blocks, the Hermes and Qwen
              layout that most ChatML-speaking models are trained on
    llama3    a bare JSON object ended by <|eom_id|>, with the result coming
              back from the `ipython` role -- Llama 3.1's own convention
    harmony   a `commentary` channel addressed with `to=functions.name`,
              ended by <|call|>
    inst      Mistral's [TOOL_CALLS] / [TOOL_RESULTS] tokens

This used to be Harmony only, and the omission was not visible from anywhere:
the other four rendered an assistant turn that *had* a tool call as an
assistant turn with nothing in it. A tool-calling dataset trained in ChatML
taught the model to think about calling a tool and then fall silent, and the
only symptom was a model that did not work.

The rule when a format's published encoding is elaborate -- Qwen's system
preamble runs to a hundred and fifty tokens of instructions -- is to keep the
*tokens and the structure* and drop the prose. A 20M-parameter model trained
from scratch here cannot spare a sixth of its context on an explanation it will
never generalise from, and a fine-tune renders through its base model's own
template anyway.

WHY EVERY TEMPLATE IS ONE LONG LINE: Jinja's trim_blocks removes the newline
after a `{% %}` tag but never after a `{{ }}` expression. A template laid out
across source lines therefore emits the source's own newlines on top of the
ones it means -- a blank line between every turn, and a stray one after the
generation prompt. Chat templates in the wild are written this way for the
same reason.

WHAT THE TEMPLATES ARE GIVEN: `common.conversation.for_template` builds the
message list, so every message reliably has `content`, `reasoning`,
`tool_calls` (each with a flat `name` and `arguments`, and a nested
`function.*`), `name` and `tool_call_id` -- present and empty rather than
missing. `tools_text` is the tool declaration already written in this format's
own idiom by `formatting.tool_declaration`. `more_turns` is true when what is
being rendered is a prefix of a longer conversation, which is how the trainer
finds the boundary of each turn in order to mask the ones it must not learn
from; only Harmony needs it, because only Harmony ends its final message with
a different token.
"""
from __future__ import annotations

import json

CHAT_FORMATS = [
    {
        "id": "plain",
        "label": "Plain text",
        "blurb": "Roles written out as ordinary words, with no special tokens. "
                 "Easiest to read while checking your data, and the weakest "
                 "signal for the model — it has to infer where a turn ends "
                 "from punctuation, like everything else.",
        "specials": ["<|endoftext|>"],
        "eos_token": "<|endoftext|>",
        "bos_token": None,
        "stop": ["\nuser:", "\nsystem:", "\ntool:", "<|endoftext|>"],
        "template": (
            "{% if tools_text %}{{ tools_text + '\\n\\n' }}{% endif %}"
            "{% for m in messages %}"
            "{% if m.reasoning %}{{ m.role + ' thinks: ' + m.reasoning + '\\n' }}{% endif %}"
            "{% if m.role == 'tool' %}"
            "{{ 'tool ' + (m.name or 'result') + ': ' + m.content + '\\n' }}"
            "{% else %}"
            "{% if m.content %}{{ m.role + ': ' + m.content + '\\n' }}{% endif %}"
            "{% for c in m.tool_calls %}"
            "{{ m.role + ' calls ' + c.name + '(' + c.arguments + ')\\n' }}"
            "{% endfor %}"
            "{% if not m.content and not m.tool_calls and not m.reasoning %}"
            "{{ m.role + ': \\n' }}{% endif %}"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{% if reasoning %}{{ 'assistant thinks: ' }}"
            "{% else %}{{ 'assistant: ' }}{% endif %}"
            "{% endif %}"),
        "reasoning_specials": [],
        "reasoning_note": 'Written as an extra line. No reserved token, so the model has to learn the phrase like any other words.',
        "tool_note": "Written out as words: `assistant calls get_order({...})`. "
                     "Nothing is reserved, so the model has to learn the shape "
                     "of a call the way it learns any other sentence.",
        "sample_prompt": "user: hello\nassistant: ",
    },
    {
        "id": "chatml",
        "label": "ChatML",
        "blurb": "Two tokens wrap every turn: one opens it with a role, one "
                 "closes it. The most widely used open format — Qwen and many "
                 "others speak it — and the right default when you have no "
                 "particular reason to prefer another.",
        "specials": ["<|endoftext|>", "<|im_start|>", "<|im_end|>"],
        "eos_token": "<|im_end|>",
        "bos_token": None,
        "stop": ["<|im_end|>", "<|im_start|>", "<|endoftext|>"],
        # Tools are declared in their own leading system turn rather than
        # folded into the user's system prompt. Qwen folds them; keeping them
        # separate means a run whose data has no system prompt does not
        # suddenly acquire one, and the model sees the declaration in the same
        # place every time.
        #
        # <tool_call> and <tool_response> are ordinary text, not reserved
        # tokens -- which is what Hermes and Qwen do, so a model fine-tuned
        # from one of those already knows them. `c.arguments` is spliced in as
        # raw JSON text rather than quoted, so the result is a JSON *object*:
        # {"name": "get_order", "arguments": {"order_id": "12345"}}.
        "template": (
            "{% if tools_text %}"
            "{{ '<|im_start|>system\\n' + tools_text + '<|im_end|>\\n' }}"
            "{% endif %}"
            "{% for m in messages %}"
            "{% if m.role == 'tool' %}"
            "{{ '<|im_start|>tool\\n<tool_response>\\n' }}"
            "{% if m.name %}{{ '{\"name\": \"' + m.name + '\", \"content\": ' }}"
            "{{ m.content + '}' }}{% else %}{{ m.content }}{% endif %}"
            "{{ '\\n</tool_response><|im_end|>\\n' }}"
            "{% else %}"
            "{{ '<|im_start|>' + m.role + '\\n' }}"
            "{% if m.reasoning %}{{ '<think>\\n' + m.reasoning + '\\n</think>\\n\\n' }}{% endif %}"
            "{{ m.content }}"
            "{% for c in m.tool_calls %}"
            # The separating newline belongs before a call only when something
            # was already written on that line. After a </think> block, which
            # already ends in a blank line, one more would train the model on a
            # stray empty line before every tool call it ever makes.
            "{{ ('\\n' if m.content or not loop.first else '')"
            " + '<tool_call>\\n{\"name\": \"' + c.name + '\", \"arguments\": '"
            " + c.arguments + '}\\n</tool_call>' }}"
            "{% endfor %}"
            "{{ '<|im_end|>\\n' }}"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|im_start|>assistant\\n' }}"
            "{% if reasoning %}{{ '<think>\\n' }}{% endif %}"
            "{% endif %}"),
        "reasoning_specials": ['<think>', '</think>'],
        "reasoning_note": "A reserved <think> block inside the assistant's turn — the layout DeepSeek-R1 and the Qwen reasoning models use.",
        "tool_note": "A <tool_call> block holding {\"name\", \"arguments\"} "
                     "JSON, answered by a <tool_response> block in a turn of "
                     "its own. The Hermes and Qwen layout, so a fine-tune from "
                     "either already speaks it.",
        "sample_prompt": "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n",
    },
    {
        "id": "llama3",
        "label": "Llama 3",
        "blurb": "Each role sits in its own header between two tokens, and a "
                 "turn ends with an end-of-turn token distinct from "
                 "end-of-text. Verbose, and unusually unambiguous about where "
                 "a turn stops.",
        # <|eom_id|> -- "end of message" rather than "end of turn" -- is what
        # Llama 3.1 added for exactly this: an assistant message that is a tool
        # call is not the end of the assistant's turn, because a result is
        # coming back and the assistant will speak again.
        "specials": ["<|begin_of_text|>", "<|end_of_text|>",
                     "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>",
                     "<|eom_id|>"],
        "eos_token": "<|eot_id|>",
        "bos_token": "<|begin_of_text|>",
        "stop": ["<|eot_id|>", "<|eom_id|>", "<|start_header_id|>",
                 "<|end_of_text|>"],
        # A tool result comes back from the `ipython` role, which is the name
        # Llama 3 gives the tool-executing environment. Calls are a bare JSON
        # object with `parameters` -- not `arguments` -- which is the spelling
        # Llama's own tool encoding uses and differs from every other format
        # here on purpose.
        "template": (
            "{{ '<|begin_of_text|>' }}"
            "{% if tools_text %}"
            "{{ '<|start_header_id|>system<|end_header_id|>\\n\\n'"
            " + tools_text + '<|eot_id|>' }}"
            "{% endif %}"
            "{% for m in messages %}"
            "{% if m.role == 'tool' %}"
            "{{ '<|start_header_id|>ipython<|end_header_id|>\\n\\n'"
            " + m.content + '<|eot_id|>' }}"
            "{% else %}"
            "{{ '<|start_header_id|>' + m.role + '<|end_header_id|>\\n\\n' }}"
            "{% if m.reasoning %}{{ '<think>\\n' + m.reasoning + '\\n</think>\\n\\n' }}{% endif %}"
            "{{ m.content }}"
            "{% if m.tool_calls %}"
            "{% for c in m.tool_calls %}"
            "{{ '{\"name\": \"' + c.name + '\", \"parameters\": '"
            " + c.arguments + '}' }}"
            "{% endfor %}"
            "{{ '<|eom_id|>' }}"
            "{% else %}{{ '<|eot_id|>' }}{% endif %}"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}"
            "{% if reasoning %}{{ '<think>\\n' }}{% endif %}"
            "{% endif %}"),
        "reasoning_specials": ['<think>', '</think>'],
        "reasoning_note": "A reserved <think> block inside the assistant's turn.",
        "tool_note": "A bare JSON object with \"name\" and \"parameters\", "
                     "closed by <|eom_id|> rather than <|eot_id|> because the "
                     "turn is not over — the result comes back from the "
                     "`ipython` role. Llama 3.1's own convention.",
        "sample_prompt": ("<|begin_of_text|><|start_header_id|>user"
                          "<|end_header_id|>\n\nhello<|eot_id|>"
                          "<|start_header_id|>assistant<|end_header_id|>\n\n"),
    },
    {
        "id": "harmony",
        "label": "Harmony",
        "blurb": "OpenAI's format for gpt-oss. A separate token divides the "
                 "role from the message body, and the assistant writes into a "
                 "named channel — which is how a model keeps its reasoning "
                 "apart from its answer. Reasoning goes in the 'analysis' "
                 "channel and the reply in 'final', so the two never have to "
                 "be untangled from one another afterwards.",
        "specials": ["<|start|>", "<|end|>", "<|message|>", "<|channel|>",
                     "<|return|>", "<|call|>", "<|constrain|>"],
        "eos_token": "<|end|>",
        "bos_token": None,
        # Only two tokens end a *generated* reply: <|return|> when the model
        # has finished answering, <|call|> when it has finished specifying a
        # tool call. <|end|> merely closes one message of several -- an
        # analysis turn before a final one, or a preamble before a call -- so
        # stopping there would cut the reply off partway through.
        "stop": ["<|return|>", "<|call|>"],
        # Follows the published Harmony layout, which routes each kind of
        # message to a different channel:
        #
        #   analysis    the model's private reasoning
        #   commentary  preambles AND tool calls, addressed with `to=`
        #   final       what the user sees
        #
        # A tool call is NOT a final message. The recipient goes on the AUTHOR,
        # before the channel -- `<|start|>assistant to=functions.{name}` --
        # then `<|channel|>commentary <|constrain|>json`, ending in <|call|>
        # rather than <|end|>. The tool's answer comes back as a message
        # authored by the tool and nothing else: `<|start|>functions.{name}`,
        # with no recipient and no channel.
        #
        # Both of those were wrong here until they were checked against the
        # implementation's own fixtures rather than against prose about it
        # (test-data/test_does_not_drop_if_ongoing_analysis.txt). Putting the
        # recipient after the channel, or addressing the reply back `to=
        # assistant`, produces text no gpt-oss model has ever seen.
        #
        # Tools are declared in a developer message, after the system message,
        # as a namespace block. See formatting.tool_declaration.
        "template": (
            "{% for m in messages if m.role in ['system', 'developer'] %}"
            "{{ '<|start|>' + m.role + '<|message|>' + m.content + '<|end|>' }}"
            "{% endfor %}"
            "{% if tools_text %}"
            "{{ '<|start|>developer<|message|>' + tools_text + '<|end|>' }}"
            "{% endif %}"
            "{% for m in messages if m.role not in ['system', 'developer'] %}"
            "{% if m.role == 'assistant' %}"
            "{% if m.reasoning %}"
            "{{ '<|start|>assistant<|channel|>analysis<|message|>'"
            " + m.reasoning + '<|end|>' }}"
            "{% endif %}"
            "{% if m.tool_calls %}"
            "{% if m.content %}"
            "{{ '<|start|>assistant<|channel|>commentary<|message|>'"
            " + m.content + '<|end|>' }}"
            "{% endif %}"
            "{% for c in m.tool_calls %}"
            "{{ '<|start|>assistant to=functions.' + c.name"
            " + '<|channel|>commentary <|constrain|>json<|message|>'"
            " + c.arguments + '<|call|>' }}"
            "{% endfor %}"
            "{% elif m.content %}"
            "{{ '<|start|>assistant<|channel|>final<|message|>' + m.content }}"
            # `more_turns` says this render is a *prefix* of a longer
            # conversation, which is how the trainer measures where each turn
            # begins in order to mask the ones it must not learn from. Without
            # it every prefix would claim to be the end and close with
            # <|return|>, so no prefix would be a prefix of the next and the
            # measurement would silently give up. Undefined -- which is what it
            # is everywhere except that measurement -- is falsy, so this reads
            # exactly as `loop.last` did for every other caller.
            "{{ '<|return|>' if loop.last and not more_turns else '<|end|>' }}"
            "{% endif %}"
            "{% elif m.role == 'tool' %}"
            "{{ '<|start|>functions.' + (m.name or 'tool')"
            " + '<|message|>' + m.content + '<|end|>' }}"
            "{% else %}"
            "{{ '<|start|>' + m.role + '<|message|>' + m.content + '<|end|>' }}"
            "{% endif %}"
            "{% endfor %}"
            # The generation prompt is a bare `<|start|>assistant`, with no
            # channel. That is the point of the format: the model chooses
            # whether to reason, call a tool, or answer. Forcing `final` here
            # -- as this did -- makes a tool call impossible to produce, which
            # would have left a model trained to call tools unable to.
            "{% if add_generation_prompt %}"
            "{{ '<|start|>assistant' }}"
            "{% if reasoning %}{{ '<|channel|>analysis<|message|>' }}{% endif %}"
            "{% endif %}"
        ),
        "reasoning_specials": [],
        "reasoning_note": 'Its own analysis channel — the mechanism the format was designed around. Nothing extra to reserve; the channel tokens are already there.',
        "tool_note": "A commentary-channel message addressed to the tool by "
                     "name and closed with <|call|>; the answer comes back "
                     "authored by `functions.{name}`. The most explicit of the "
                     "five, and the only one where a call is structurally "
                     "distinct from an answer.",
        "sample_prompt": ("<|start|>user<|message|>hello<|end|>"
                          "<|start|>assistant"),
    },
    {
        "id": "inst",
        "label": "Llama 2 / Mistral",
        "blurb": "The bracket style: the user's turn is wrapped in [INST] and "
                 "the reply simply follows it, ended by the end-of-sequence "
                 "token. Compact, and it reuses tokens the model needs anyway. "
                 "A system prompt is folded into the first user turn, which is "
                 "how the format defines it.",
        # [TOOL_CALLS], [AVAILABLE_TOOLS] and [TOOL_RESULTS] are real tokens in
        # Mistral's v3 tokenizer, reserved here for the same reason [INST] is.
        "specials": ["<s>", "</s>", "[AVAILABLE_TOOLS]", "[/AVAILABLE_TOOLS]",
                     "[TOOL_CALLS]", "[TOOL_RESULTS]", "[/TOOL_RESULTS]"],
        "eos_token": "</s>",
        "bos_token": "<s>",
        "stop": ["</s>", "[INST]"],
        # Written across several source lines because it is now too long to
        # read on one; the concatenation still produces a single line, which is
        # what matters (see the note at the top of this file).
        "template": (
            "{% set sys = messages | selectattr('role', 'equalto', 'system')"
            " | map(attribute='content') | join('\\n') %}"
            "{{ '<s>' }}"
            "{% if tools_text %}"
            "{{ '[AVAILABLE_TOOLS] ' + tools_text + '[/AVAILABLE_TOOLS]' }}"
            "{% endif %}"
            "{% for m in messages if m.role != 'system' %}"
            "{% if m.role == 'user' %}"
            "{{ '[INST] ' + (sys + '\\n\\n' if loop.first and sys else '')"
            " + m.content + ' [/INST]' }}"
            "{% elif m.role == 'tool' %}"
            "{{ '[TOOL_RESULTS] ' + m.content + '[/TOOL_RESULTS]' }}"
            "{% else %}"
            "{% if m.reasoning %}{{ ' <think>\\n' + m.reasoning + '\\n</think>\\n\\n' }}"
            "{% elif m.content or not m.tool_calls %}{{ ' ' }}{% endif %}"
            "{{ m.content }}"
            "{% if m.tool_calls %}"
            "{{ '[TOOL_CALLS] [' }}"
            "{% for c in m.tool_calls %}"
            "{{ '{\"name\": \"' + c.name + '\", \"arguments\": '"
            " + c.arguments + '}' + (', ' if not loop.last else '') }}"
            "{% endfor %}"
            "{{ ']</s>' }}"
            "{% else %}{{ '</s>' }}{% endif %}"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ ' <think>\\n' if reasoning else ' ' }}"
            "{% endif %}"),
        "reasoning_specials": ['<think>', '</think>'],
        "reasoning_note": 'A reserved <think> block before the reply.',
        "tool_note": "Mistral's own [TOOL_CALLS] and [TOOL_RESULTS] tokens, "
                     "with the tool list declared once inside "
                     "[AVAILABLE_TOOLS].",
        "sample_prompt": "<s>[INST] hello [/INST]",
    },
]

DEFAULT_FORMAT = "chatml"


def chat_format(format_id: str | None) -> dict | None:
    return next((f for f in CHAT_FORMATS if f["id"] == format_id), None)


def format_or_default(format_id: str | None) -> dict:
    return chat_format(format_id) or chat_format(DEFAULT_FORMAT)


def special_tokens(format_id: str | None, reasoning: bool = False,
                   tools: bool = False) -> list[str]:
    """Tokens the tokenizer must reserve, in order, before it is trained.

    Reserved before, never added after. A token added to an already-trained
    tokenizer grows the vocabulary past the size the model was built for, and
    the embedding table stops matching.

    Teaching the model to reason adds a few more, unless the format already
    carries the machinery -- Harmony reasons in a channel it already has. The
    same is true of tools: `tools` is accepted so a caller can be explicit,
    but the tool tokens are part of `specials` in every format that has them,
    because a vocabulary is sized once and a run that discovers tools in its
    data on the second epoch cannot go back and reserve them.
    """
    spec = format_or_default(format_id)
    out = list(spec["specials"])
    if reasoning:
        out += [t for t in spec.get("reasoning_specials", []) if t not in out]
    return out


# A default system prompt, injected by the saved template when a conversation
# arrives without one. Written as Jinja rather than baked into the text so the
# model stays usable both ways: give it a system message and yours is used,
# give it none and it gets the one it was trained under.
#
# `{% set %}` inside `{% if %}` is deliberate and is safe here -- Jinja scopes
# assignments inside `for`, not inside `if`. Tested, because a template that
# quietly failed to apply the default would be indistinguishable from one that
# had no default.
# Built by substitution rather than %-formatting or .format(): the text is full
# of Jinja's own `{%-` and `{{`, and both of those mechanisms read those as
# their own syntax.
_DEFAULT_SYSTEM = (
    "{%- if not messages or messages[0]['role'] != 'system' -%}"
    "{%- set messages = [{'role': 'system', 'content': __PROMPT__}] + messages -%}"
    "{%- endif -%}"
)


def with_default_system(template: str, prompt: str | None) -> str:
    """A template that supplies this system prompt when a caller gives none.

    A fine-tune trained with a system prompt behaves noticeably differently
    without it, and the prompt is the one part of its training nobody writes
    down. Carrying it in the template means the model behaves as it was taught
    by default, and still honours a system message when one is sent.
    """
    if not (prompt or "").strip():
        return template
    import json as _json
    return _DEFAULT_SYSTEM.replace("__PROMPT__",
                                   _json.dumps(prompt)) + template


def template_for_model(fmt: dict | None, system_prompt: str | None = None
                       ) -> tuple[str | None, str]:
    """The chat template a finished model should carry, and why.

    Returns (template, reason). A template of None means "leave whatever the
    tokenizer already has", which is right in exactly one case: the run trained
    with the base model's own template, so the base model's own template is the
    correct thing for it to keep.

    Everything else has to be written down. A run that trained in ChatML on a
    Mistral base and then shipped Mistral's template is a model that answers in
    a format it was never taught -- and nothing about the artifact says so.
    """
    fmt = dict(fmt or {})
    if fmt.get("use_model_template"):
        return None, "the base model's own template, which is what it trained with"

    if name := fmt.get("chat_format"):
        if spec := chat_format(name):
            return (with_default_system(spec["template"], system_prompt),
                    "the %s format this run trained with" % spec["label"])

    template = fmt.get("chat_template")
    if not template and fmt.get("mode") == "jinja":
        template = fmt.get("template")
    if template and "messages" in template:
        return (with_default_system(template, system_prompt),
                "the template this run trained with")

    # A Jinja template written against the dataset's own columns. It renders a
    # finished row, has no notion of a conversation to continue, and cannot be
    # a chat template -- see the same problem in runner/inference.
    if template:
        return None, ("a template written against the dataset's columns, which "
                      "cannot be used as a chat template")
    return None, "no chat format was chosen for this run"


def public_formats() -> list[dict]:
    """What the UI needs in order to describe the choice, without the Jinja."""
    return [{
        "id": f["id"],
        "label": f["label"],
        "blurb": f["blurb"],
        "specials": f["specials"],
        "eos_token": f["eos_token"],
        "sample": f["sample_prompt"],
        "token_count": len(f["specials"]),
        "reasoning_specials": f.get("reasoning_specials", []),
        "reasoning_note": f.get("reasoning_note", ""),
        "tool_note": f.get("tool_note", ""),
    } for f in CHAT_FORMATS]


# Where a chat template has to end up, and every place a reader might look.
#
# Transformers 5 saves it to `chat_template.jinja` and leaves `chat_template`
# out of `tokenizer_config.json` entirely. Plenty of readers only ever look in
# the config -- older transformers, several serving stacks, and this studio's
# own artifact endpoint until recently, which is why the wizard told people
# their instruct fine-tune "ships no chat template of its own, which usually
# means it is a base model" about a model whose template it had written itself.
#
# A model that carries its template in one of the two places is a model that
# half the ecosystem reads as a base model. So it goes in both.
TEMPLATE_FILE = "chat_template.jinja"
TOKENIZER_CONFIG = "tokenizer_config.json"


def template_in(files: dict) -> str | None:
    """The chat template out of a saved tokenizer, from wherever it is.

    `files` maps a file name to its already-read contents, so the same reading
    serves a directory on disk and a zip nobody wants to unpack. The dedicated
    file wins: where both exist it is the one transformers wrote last.
    """
    if jinja := (files.get(TEMPLATE_FILE) or "").strip():
        return jinja
    try:
        conf = json.loads(files.get(TOKENIZER_CONFIG) or "{}")
    except ValueError:
        return None
    template = conf.get("chat_template")
    if isinstance(template, dict):
        template = template.get("default") or next(iter(template.values()), None)
    if isinstance(template, list):          # some saves ship a list of dicts
        template = next((t.get("template") for t in template
                         if isinstance(t, dict)), None)
    return template or None


def stamp_into(model_dir, template: str | None) -> list[str]:
    """Put this template in every place a reader looks. Returns what changed.

    Called after the tokenizer has been saved, because `save_pretrained` is
    what decides where transformers puts it and that answer has changed
    between versions. Writing both afterwards is version-proof in a way that
    trusting the library is not.
    """
    from pathlib import Path
    model_dir = Path(model_dir)
    if not template or not model_dir.is_dir():
        return []
    written = []
    jinja = model_dir / TEMPLATE_FILE
    if not jinja.exists() or jinja.read_text(encoding="utf-8") != template:
        jinja.write_text(template, encoding="utf-8")
        written.append(TEMPLATE_FILE)
    conf_path = model_dir / TOKENIZER_CONFIG
    if conf_path.exists():
        try:
            conf = json.loads(conf_path.read_text(encoding="utf-8"))
        except ValueError:
            return written
        if conf.get("chat_template") != template:
            conf["chat_template"] = template
            conf_path.write_text(json.dumps(conf, indent=2, ensure_ascii=False),
                                 encoding="utf-8")
            written.append(TOKENIZER_CONFIG)
    return written
