"""What would go wrong, said before the run starts rather than after.

Every check here corresponds to a way a run currently fails, or quietly
disappoints, after it has been queued, dispatched, given a GPU and left for an
hour:

* rows longer than the context length are cut, and nothing says so. For an
  instruction dataset the part that gets cut is the answer, so the model reads
  the question and learns to stop.
* a dataset too small to hold anything back trains with no held-out loss at
  all, which is discovered when the chart has one line on it.
* a chosen split with no rows in it raises on the runner, minutes in.
* a from-scratch corpus too small for the token budget silently repeats itself
  many times over, which is memorisation wearing a training curve.
* a chat template whose turn boundaries cannot be measured trains on the
  questions as well as the answers, which is a different run from the one
  everybody thinks they started.

The shape of a finding is deliberately the same as `architectures.validate_arch`
uses -- level, field, message, fix -- because the wizard already knows how to
draw those and how to refuse to continue past an error.
"""
from __future__ import annotations

import asyncio
from typing import Any

from common import conversation, formatting

from . import datasets as dsets, db, hub

# How many rows to render and count. Enough to be believable about a
# percentage, few enough that a person waits a moment rather than a minute.
SAMPLE = 200

# Below this a held-out slice is too small to mean anything, and the trainer
# will not cut one.
MIN_ROWS_FOR_HOLDOUT = 32


def _issue(level: str, field: str, message: str, fix: str = "") -> dict:
    return {"level": level, "field": field, "message": message, "fix": fix}


async def _token_lengths(fleet, tokenizer: str, texts: list[str],
                         timeout: float = 90.0) -> dict:
    """Ask a runner how long these texts are, in tokens.

    The controller has no tokenizer and is not going to grow one: it installs
    four pure-Python packages so it can run on a NAS. Every machine that can
    train already has the exact tokenizer the run will use, and answering this
    needs no GPU, so a CPU-only runner will do.
    """
    online = [r for r in db.list_runners()
              if r["status"] != "offline" and r["id"] in fleet.connections]
    # Prefer a machine that is not busy: this is a question somebody is
    # waiting on, and a runner mid-training answers it slowly.
    online.sort(key=lambda r: bool(fleet.busy.get(r["id"])))
    if not online:
        return {"available": False,
                "reason": "No machine is connected to count them on."}

    rid = db.new_id("tok")
    queue: asyncio.Queue = asyncio.Queue()
    fleet.waiters[rid] = queue
    try:
        sent = await fleet.send_to_runner(online[0]["id"], {
            "type": "tokenize", "request_id": rid,
            "tokenizer": tokenizer, "texts": texts})
        if not sent:
            return {"available": False,
                    "reason": "That machine dropped off just now."}
        msg = await asyncio.wait_for(queue.get(), timeout)
    except asyncio.TimeoutError:
        return {"available": False, "reason": "Counting them took too long."}
    finally:
        fleet.waiters.pop(rid, None)

    if msg.get("type") == "tokenize_error":
        return {"available": False, "reason": msg.get("error") or "unknown"}
    return {"available": True, "lengths": msg.get("lengths") or [],
            "model_max_length": msg.get("model_max_length"),
            "runner": online[0]["name"]}


def _percentile(values: list[int], p: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * p))]


async def check(cfg: dict, fleet: Any) -> dict:
    """Everything worth knowing before this configuration is queued."""
    issues: list[dict] = []
    facts: dict = {}

    # An `arch` block, even an empty one, means a model being built rather
    # than one being adapted. Tested for presence, not for truth: `{}` is a
    # from-scratch run whose architecture has not been filled in yet.
    kind = cfg.get("kind") or ("pretrain_llm" if cfg.get("arch") is not None
                               else "finetune_llm")
    dataset_id = cfg.get("studio_dataset")
    split = (cfg.get("dataset_split") or "train").strip()

    # ---- the data ---------------------------------------------------------
    rows: list[dict] = []
    d = db.get_dataset(dataset_id) if dataset_id else None
    if d:
        splits = d.get("splits") or {}
        in_split = splits.get(split)
        if split and splits and not in_split:
            issues.append(_issue(
                "error", "dataset_split",
                "This dataset has no rows in a split called \"%s\". It has: %s."
                % (split, ", ".join(splits) or "none"),
                "Choose one of the splits it does have."))
        facts["rows"] = in_split or d.get("rows") or 0
        rows = list(dsets.iter_rows(dataset_id, SAMPLE, split or None))

        # Something to measure on. The trainer holds back 5% capped at 256
        # rows, and refuses below a floor -- so a small dataset trains with no
        # held-out loss at all, which is found out when the chart has one line.
        held = next((n for n in ("validation", "test", "eval", "dev")
                     if n in splits), None)
        facts["held_out_split"] = held
        if held:
            facts["held_out_rows"] = splits.get(held)
        elif (facts["rows"] or 0) < MIN_ROWS_FOR_HOLDOUT:
            issues.append(_issue(
                "warn", "dataset",
                "Only %d rows to train on, which is too few for any of them to "
                "be held back. There will be no held-out loss, so nothing will "
                "tell you whether the model is learning or memorising."
                % facts["rows"],
                "Add more examples, or accept a training loss only."))
        else:
            issues.append(_issue(
                "info", "dataset",
                "No held-out split, so the run will take 5% of the training "
                "data at random to measure on. A split you hold back yourself "
                "is a better test, and the same rows every time.",
                "Hold back a split on the dataset page."))

    # ---- what the rows render to -----------------------------------------
    fmt = formatting.resolve_format(cfg.get("format") or (d or {}).get("format") or {})
    texts: list[str] = []
    unreadable = 0
    if rows:
        for r in rows:
            try:
                text = formatting.format_example(r, fmt)
            except formatting.TemplateError as e:
                issues.append(_issue(
                    "error", "format",
                    "The template does not render these rows: %s" % e,
                    "Fix the template on the format step."))
                text = None
                break
            if text:
                texts.append(text)
            elif not hub._row_is_blank(r):
                # Blank in the source is fine -- a line-oriented corpus is full
                # of empty lines and the trainer skips them. A row with real
                # content the mapping cannot reach is the mistake.
                unreadable += 1
        share = unreadable / max(len(rows), 1)
        if share > 0.5:
            issues.append(_issue(
                "error", "format",
                "%.0f%% of the rows have content the chosen columns cannot "
                "reach, so almost nothing here would be trained on."
                % (share * 100),
                "Change the mapping on the dataset's Training tab."))
        elif unreadable:
            issues.append(_issue(
                "warn", "format",
                "%d of %d sampled rows have content the chosen columns cannot "
                "reach. Those rows are skipped." % (unreadable, len(rows)),
                "Change the mapping on the dataset's Training tab."))

    # ---- how long they are, in tokens -------------------------------------
    tokenizer = cfg.get("base_model") or cfg.get("tokenizer_from") or ""
    max_len = int(cfg.get("max_seq_len") or 0)
    if texts and tokenizer and max_len:
        counted = await _token_lengths(fleet, tokenizer, texts[:SAMPLE])
        facts["tokens"] = counted
        if counted.get("available"):
            lengths = counted["lengths"]
            over = [n for n in lengths if n > max_len]
            facts["token_p50"] = _percentile(lengths, 0.5)
            facts["token_p90"] = _percentile(lengths, 0.9)
            facts["token_max"] = max(lengths) if lengths else 0
            facts["over_limit"] = len(over)
            share = len(over) / max(len(lengths), 1)
            if share > 0.25:
                issues.append(_issue(
                    "error", "max_seq_len",
                    "%.0f%% of these rows are longer than %d tokens and would "
                    "be cut. On an instruction dataset the part that gets cut "
                    "is the answer, so the model reads the question and learns "
                    "to stop." % (share * 100, max_len),
                    "Raise the context length, or filter the long rows out."))
            elif share > 0.05:
                issues.append(_issue(
                    "warn", "max_seq_len",
                    "%.0f%% of these rows are longer than %d tokens and would "
                    "be cut at that point." % (share * 100, max_len),
                    "Raise the context length, or filter the long rows out."))
            elif over:
                # A handful is not worth stopping for and is worth saying:
                # claiming "every row fits" while reporting two that do not is
                # the kind of small lie that makes a reader stop believing the
                # rest of the page.
                issues.append(_issue(
                    "ok", "max_seq_len",
                    "%d of %d rows measured are longer than %d tokens and "
                    "would be cut. The rest fit; the longest is %d."
                    % (len(over), len(lengths), max_len, facts["token_max"])))
            elif lengths:
                issues.append(_issue(
                    "ok", "max_seq_len",
                    "Every row measured fits in %d tokens; the longest is %d."
                    % (max_len, facts["token_max"])))
            # A context length the model itself will not accept.
            limit = counted.get("model_max_length")
            if limit and max_len > limit:
                issues.append(_issue(
                    "error", "max_seq_len",
                    "This model's context is %d tokens and the run asks for "
                    "%d." % (limit, max_len),
                    "Lower it to %d or less." % limit))

    # ---- can the turns be told apart -------------------------------------
    if kind == "finetune_llm" and fmt.get("mode") == "chat" and rows:
        want = (cfg.get("train_on") or "assistant")
        if want != "all":
            inexact = 0
            for r in rows[:40]:
                try:
                    conv, _ = conversation.repair(conversation.from_row(r, fmt))
                    _, _, exact = conversation.trainable_spans(conv, fmt, want)
                    if not exact:
                        inexact += 1
                except Exception:  # noqa: BLE001 - a row that will not parse
                    inexact += 1
            if inexact > len(rows[:40]) * 0.5:
                issues.append(_issue(
                    "warn", "format",
                    "The turn boundaries of this template cannot be measured on "
                    "most of these rows, so they would train on the questions "
                    "as well as the answers. Some templates rewrite earlier "
                    "turns when a later one arrives.",
                    "Choose a different chat format, or accept it."))

    # ---- from scratch: is there enough text ------------------------------
    if kind == "pretrain_llm":
        budget = int(cfg.get("token_budget") or 0)
        # The corpus is measured in characters here and in tokens by the run.
        # Four characters a token understates a small purpose-built vocabulary
        # rather than overstating it, which is the safe direction for a
        # warning about not having enough.
        chars = sum(len(t) for t in texts)
        if texts and budget:
            per_row = chars / max(len(texts), 1)
            total_rows = facts.get("rows") or len(texts)
            corpus = int(per_row * total_rows / 4)
            facts["corpus_tokens"] = corpus
            passes = budget / max(corpus, 1)
            facts["passes"] = round(passes, 1)
            if passes > 4:
                issues.append(_issue(
                    "warn", "token_budget",
                    "This corpus is about %s tokens and the run asks for %s, "
                    "so it would read the same text %.0f times over. Past "
                    "three or four passes a model memorises rather than learns."
                    % (f"{corpus:,}", f"{budget:,}", passes),
                    "Shorten the run, or give it more text."))

    return {"issues": issues, "facts": facts,
            "blocked": any(i["level"] == "error" for i in issues)}
