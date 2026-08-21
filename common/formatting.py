"""Turning a dataset row into the exact string a model is trained on.

This lives in `common` for one reason: the web UI promises to show you what
your model will actually read, and a preview built by lookalike code is not
that promise. The controller renders the preview and the runner builds the
training batch by calling the *same* function, so the two cannot drift.

Supports the three shapes that cover almost everything on the Hub: plain
text, instruction/response pairs, and chat message lists.
"""
from __future__ import annotations

DEFAULT_INSTRUCTION_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n### Response:\n{response}")

_INSTRUCTION_FIELDS = ["instruction", "prompt", "question", "input"]
_RESPONSE_FIELDS = ["output", "response", "answer", "completion"]
_TEXT_FIELDS = ["text", "content", "document", "sentence", "raw", "body"]


def first_present(row: dict, names: list[str]) -> str | None:
    for n in names:
        if n in row and row[n]:
            return n
    return None


def format_example(row: dict, fmt: dict) -> str | None:
    """Render one row as the model will see it, or None if it cannot be read."""
    mode = (fmt or {}).get("mode", "auto")
    text_field = (fmt or {}).get("text_field")

    if mode == "text" or (mode == "auto" and text_field and text_field in row):
        val = row.get(text_field or "text")
        return str(val) if val else None

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

    if mode in ("chat", "auto"):
        msgs = row.get(fmt.get("messages_field") or "messages")
        if isinstance(msgs, list) and msgs:
            parts = []
            for m in msgs:
                if isinstance(m, dict) and "content" in m:
                    parts.append("%s: %s" % (m.get("role", "user"), m["content"]))
            if parts:
                return "\n".join(parts)

    if mode == "auto":
        for f in _TEXT_FIELDS:
            if row.get(f):
                return str(row[f])
    return None


def detect_format(columns: list[str]) -> dict:
    """Guess how a dataset is laid out, so the UI can pre-fill the mapping."""
    cols = {c.lower() for c in columns if c}
    if {"instruction"} & cols and cols & {"output", "response"}:
        return {"mode": "instruction", "instruction_field": "instruction",
                "response_field": "output" if "output" in cols else "response",
                "confidence": "high"}
    if "messages" in cols or "conversations" in cols:
        return {"mode": "chat",
                "messages_field": "messages" if "messages" in cols else "conversations",
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
        lengths = [len(r[c]) for r in (rows or [])
                   if isinstance(r.get(c), str)]
        avg = sum(lengths) / len(lengths) if lengths else 0
        if avg > best_len:
            best_len, best = avg, c
    return best if best_len > 40 else None
