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

The templates follow each format's published layout for roles and message
boundaries, which is what a small model can actually learn. They do not
attempt the full tool-calling encodings the larger formats also define.

WHY EVERY TEMPLATE IS ONE LONG LINE: Jinja's trim_blocks removes the newline
after a `{% %}` tag but never after a `{{ }}` expression. A template laid out
across source lines therefore emits the source's own newlines on top of the
ones it means -- a blank line between every turn, and a stray one after the
generation prompt. Chat templates in the wild are written this way for the
same reason.
"""
from __future__ import annotations

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
        "template": "{% for m in messages %}{% if m.reasoning %}{{ m.role + ' thinks: ' + m.reasoning + '\\n' }}{% endif %}{{ m.role + ': ' + m.content + '\\n' }}{% endfor %}{% if add_generation_prompt %}{% if reasoning %}{{ 'assistant thinks: ' }}{% else %}{{ 'assistant: ' }}{% endif %}{% endif %}",
        "reasoning_specials": [],
        "reasoning_note": 'Written as an extra line. No reserved token, so the model has to learn the phrase like any other words.',
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
        "template": "{% for m in messages %}{{ '<|im_start|>' + m.role + '\\n' }}{% if m.reasoning %}{{ '<think>\\n' + m.reasoning + '\\n</think>\\n\\n' }}{% endif %}{{ m.content + '<|im_end|>\\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% if reasoning %}{{ '<think>\\n' }}{% endif %}{% endif %}",
        "reasoning_specials": ['<think>', '</think>'],
        "reasoning_note": "A reserved <think> block inside the assistant's turn — the layout DeepSeek-R1 and the Qwen reasoning models use.",
        "sample_prompt": "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n",
    },
    {
        "id": "llama3",
        "label": "Llama 3",
        "blurb": "Each role sits in its own header between two tokens, and a "
                 "turn ends with an end-of-turn token distinct from "
                 "end-of-text. Verbose, and unusually unambiguous about where "
                 "a turn stops.",
        "specials": ["<|begin_of_text|>", "<|end_of_text|>",
                     "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"],
        "eos_token": "<|eot_id|>",
        "bos_token": "<|begin_of_text|>",
        "stop": ["<|eot_id|>", "<|start_header_id|>", "<|end_of_text|>"],
        "template": "{{ '<|begin_of_text|>' }}{% for m in messages %}{{ '<|start_header_id|>' + m.role + '<|end_header_id|>\\n\\n' }}{% if m.reasoning %}{{ '<think>\\n' + m.reasoning + '\\n</think>\\n\\n' }}{% endif %}{{ m.content + '<|eot_id|>' }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}{% if reasoning %}{{ '<think>\\n' }}{% endif %}{% endif %}",
        "reasoning_specials": ['<think>', '</think>'],
        "reasoning_note": "A reserved <think> block inside the assistant's turn.",
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
        "stop": ["<|end|>", "<|return|>", "<|start|>", "<|call|>"],
        # With reasoning on, `<|end|>` closes the analysis channel partway
        # through the reply, so stopping there would throw the answer away
        # before it was written. Only `<|return|>` means finished -- which is
        # exactly the distinction Harmony draws between a message already in
        # the context and one the model has just produced.
        "reasoning_stop": ["<|return|>", "<|call|>"],
        # Follows the published Harmony layout, which routes each kind of
        # message to a different channel:
        #
        #   analysis    the model's private reasoning
        #   commentary  preambles AND tool calls, addressed with `to=`
        #   final       what the user sees
        #
        # A tool call is NOT a final message. It goes to commentary, names its
        # recipient as `functions.{name}`, declares its argument type with
        # <|constrain|>, and ends with <|call|> rather than <|end|>. The tool's
        # answer comes back authored by `functions.{name} to=assistant`, also
        # on commentary. Getting this wrong -- putting calls and results in
        # `final` -- teaches the model to announce tool output as if it were
        # speaking to the user.
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
            "{{ '<|start|>assistant<|channel|>commentary to=functions.' + c.name"
            " + ' <|constrain|>json<|message|>' + c.arguments + '<|call|>' }}"
            "{% endfor %}"
            "{% elif m.content %}"
            "{{ '<|start|>assistant<|channel|>final<|message|>' + m.content }}"
            "{{ '<|return|>' if loop.last else '<|end|>' }}"
            "{% endif %}"
            "{% elif m.role in ['tool', 'function'] %}"
            "{{ '<|start|>functions.' + (m.name or 'tool') + ' to=assistant"
            "<|channel|>commentary<|message|>' + m.content + '<|end|>' }}"
            "{% else %}"
            "{{ '<|start|>' + m.role + '<|message|>' + m.content + '<|end|>' }}"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{% if reasoning %}"
            "{{ '<|start|>assistant<|channel|>analysis<|message|>' }}"
            "{% else %}"
            "{{ '<|start|>assistant<|channel|>final<|message|>' }}"
            "{% endif %}"
            "{% endif %}"
        ),
        "reasoning_specials": [],
        "reasoning_note": 'Its own analysis channel — the mechanism the format was designed around. Nothing extra to reserve; the channel tokens are already there.',
        "sample_prompt": ("<|start|>user<|message|>hello<|end|>"
                          "<|start|>assistant<|channel|>final<|message|>"),
    },
    {
        "id": "inst",
        "label": "Llama 2 / Mistral",
        "blurb": "The bracket style: the user's turn is wrapped in [INST] and "
                 "the reply simply follows it, ended by the end-of-sequence "
                 "token. Compact, and it reuses tokens the model needs anyway. "
                 "A system prompt is folded into the first user turn, which is "
                 "how the format defines it.",
        "specials": ["<s>", "</s>"],
        "eos_token": "</s>",
        "bos_token": "<s>",
        "stop": ["</s>", "[INST]"],
        "template": "{% set sys = messages | selectattr('role', 'equalto', 'system') | map(attribute='content') | join('\\n') %}{{ '<s>' }}{% for m in messages if m.role != 'system' %}{% if m.role == 'user' %}{{ '[INST] ' + (sys + '\\n\\n' if loop.first and sys else '') + m.content + ' [/INST]' }}{% else %}{% if m.reasoning %}{{ ' <think>\\n' + m.reasoning + '\\n</think>\\n\\n' + m.content + '</s>' }}{% else %}{{ ' ' + m.content + '</s>' }}{% endif %}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ ' <think>\\n' if reasoning else ' ' }}{% endif %}",
        "reasoning_specials": ['<think>', '</think>'],
        "reasoning_note": 'A reserved <think> block before the reply.',
        "sample_prompt": "<s>[INST] hello [/INST]",
    },
]

DEFAULT_FORMAT = "chatml"


def chat_format(format_id: str | None) -> dict | None:
    return next((f for f in CHAT_FORMATS if f["id"] == format_id), None)


def format_or_default(format_id: str | None) -> dict:
    return chat_format(format_id) or chat_format(DEFAULT_FORMAT)


def special_tokens(format_id: str | None, reasoning: bool = False) -> list[str]:
    """Tokens the tokenizer must reserve, in order, before it is trained.

    Reserved before, never added after. A token added to an already-trained
    tokenizer grows the vocabulary past the size the model was built for, and
    the embedding table stops matching.

    Teaching the model to reason adds a few more, unless the format already
    carries the machinery -- Harmony reasons in a channel it already has.
    """
    spec = format_or_default(format_id)
    out = list(spec["specials"])
    if reasoning:
        out += [t for t in spec.get("reasoning_specials", []) if t not in out]
    return out


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
    } for f in CHAT_FORMATS]
