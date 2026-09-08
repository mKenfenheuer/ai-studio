"""Writing a dataset with a model you already have.

The hardest part of fine-tuning is not the training; it is that most people
have no dataset and no realistic way to write ten thousand examples by hand.
A model can write them, and this makes that a first-class job rather than a
script somebody runs on the side.

It is a training job in every structural sense -- it takes a GPU, it takes
hours, it reports progress per row, it can be stopped and what it made so far
is kept -- so it *is* a job, and inherits all of that rather than reimplementing
it. The output is a JSONL artifact the controller registers as a dataset.

Three ways to use it, because they solve different problems:

* **from_prompts** -- you have the questions and want the answers. Point it at
  a column of prompts (or paste a list) and it answers each one. This is
  distillation: a capable model teaching a small one.
* **from_seeds** -- you have a handful of examples and want a thousand. It
  shows the model a few at random and asks for a new one in the same spirit.
  Self-instruct, roughly.
* **from_topics** -- you have subject matter and want coverage. It works
  through a list of topics, generating examples about each.
* **conversations** -- you have a brief and want whole exchanges, tool calls
  and all. The reply is constrained to a JSON schema by the provider, so what
  comes back is a conversation rather than prose that describes one. This is
  the mode for a dataset that has to teach *calling something*: the model
  writes the user's turn, the assistant's working, the call, what the function
  returned and the answer, in one request per row.
* **extend_conversations** -- you have conversations and they are too short.
  It carries each one further, a turn at a time, with the row's own tools
  declared so the model can call one. This is the mode most tool-calling
  datasets need: they are almost entirely single exchanges, and a model
  trained only on those never learns the fourth turn.

The honest caveats are printed into the run's log rather than buried in docs,
because generated data has failure modes that look like success: a model that
repeats itself, that agrees with everything, or that is confidently wrong
produces a dataset that trains beautifully and teaches a small model to be
confidently wrong too.
"""
from __future__ import annotations

import collections
import itertools
import json
import random
import re
import shutil
import time
from pathlib import Path
from typing import Any, Iterator

from .lora_llm import Cancelled

# Cut a generation off if the model will not stop. Generated rows are training
# data; a runaway 4000-token ramble is not a training example, it is noise
# with a token budget.
DEFAULT_MAX_TOKENS = 512
# A whole conversation is not one reply. Cut off at 512 tokens the JSON ends
# mid-string, the row cannot be read, and the run drops what it paid for.
CONVERSATION_MAX_TOKENS = 1600
# Warmer than a chat assistant would be set to, on purpose. Every row here is
# an independent request, so a cautious temperature makes the writer walk the
# same well-worn path a few hundred times -- which reads as a fine dataset
# until you count the distinct examples in it. Variety is the product.
DEFAULT_TEMPERATURE = 1.05
# Rows written over the network are IO, not computation: the machine waits.
# Enough in flight to keep a long run to an hour, few enough that a shared
# endpoint is not being hammered by one studio.
_HOSTED_WORKERS = 6
_MAX_WORKERS = 16
# Rows already written, shown back to the writer so it stops reinventing the
# same house. Every request is independent, so without this the model returns
# to its favourite rooms and names hundreds of times in one run.
_RECENT_SHOWN = 16


def run(cfg: dict, ctx: Any) -> dict:
    from .. import inference

    mode = cfg.get("mode") or "from_prompts"
    target = int(cfg.get("count") or 100)
    out_path = Path(ctx.workdir) / "generated.jsonl"

    spec = dict(cfg.get("model") or {})
    spec.setdefault("hf_token", ctx.hf_token)
    # Two kinds of writer, one interface. A hosted model is reached over the
    # network and a local one is loaded onto the GPU, and the loop below cares
    # about neither -- it asks for a reply and gets one.
    hosted = bool(spec.get("connection"))
    if hosted:
        host: Any = HostedModel(spec, ctx)
    elif spec.get("job_id") or spec.get("base_model"):
        host = inference.ModelHost(ctx.controller_url, ctx.runner_token,
                                   ctx.capabilities)
    else:
        raise ValueError("No model was chosen to generate the data with.")

    # Named for what is actually about to happen. A hosted writer is reached
    # over the network and there is no model to download; saying "downloading
    # and loading the model" while opening an HTTPS connection describes a
    # different job entirely, and this one is not training either.
    ctx.progress(0, target, stage="connecting" if hosted else "loading_model")
    host.ensure_loaded(spec, lambda line: ctx.log(line))

    _preamble(ctx, cfg, mode, target, spec)

    params = {
        "max_new_tokens": int(cfg.get("max_new_tokens")
                              or (CONVERSATION_MAX_TOKENS
                                  if mode == "conversations"
                                  else DEFAULT_MAX_TOKENS)),
        "temperature": float(cfg.get("temperature") or DEFAULT_TEMPERATURE),
        "top_p": float(cfg.get("top_p") or 0.95),
    }
    if effort := (cfg.get("reasoning_effort") or "").strip():
        params["reasoning_effort"] = effort

    if mode == "conversations":
        # Constrained decoding, where the provider can do it. The prompt says
        # what to write; the schema is what makes it arrive as a conversation
        # instead of a description of one.
        from common import apimodels

        params["schema"] = _conversation_schema(cfg)
        if hosted and not apimodels.supports_schema(spec["connection"]):
            ctx.log("This provider cannot constrain a reply to a schema, so "
                    "the shape is asked for in the prompt instead. Rows that "
                    "come back malformed are counted and dropped.", "warn")
        elif not hosted:
            ctx.log("A model on this machine is asked for the shape in the "
                    "prompt -- there is no constrained decoding here. A small "
                    "model will get it wrong often; watch the dropped count.",
                    "warn")

    if mode == "extend_conversations":
        # A different shape of work: many calls per row rather than one, and
        # the row already exists. It gets its own loop rather than being bent
        # into the single-shot one below.
        return _extend(cfg, ctx, host, spec, params, out_path, target)

    recent: collections.deque = collections.deque(maxlen=_RECENT_SHOWN)
    sources = _sources(cfg, mode, ctx, recent)

    seen: set[str] = set()
    written = 0
    duplicates = 0
    empty = 0
    rejected = 0
    stopped_early = False
    t0 = time.time()

    workers = _workers(cfg, hosted, target)
    if workers > 1:
        ctx.log("Writing %d rows at a time, over the network." % workers)

    with out_path.open("w", encoding="utf-8") as fh:
        for _i, meta, result in _attempts(host, spec, params, sources, target,
                                          workers, ctx):
            if ctx.should_cancel():
                # Nothing here is unfinished work: every row already written is
                # a complete row. Stopping always keeps them, whichever way the
                # user answered, because there is no half-written state to
                # discard and throwing away an hour of generation to honour a
                # flag would be perverse.
                stopped_early = True
                ctx.log("Stopping after %d rows, and keeping them." % written,
                        "warn")
                break

            text = (result.get("text") or "").strip()
            if not text:
                empty += 1
                continue

            row = _row(text, meta, cfg)
            if row is None:
                # Not an empty reply: something arrived and was refused. On a
                # conversation that means a call to a function that does not
                # exist, arguments that are not JSON, a call nothing answered,
                # or an exchange that stops before the assistant replies.
                rejected += 1
                if rejected in (25, 100, 400):
                    ctx.log("%d replies so far were dropped for not being "
                            "usable rows. If this keeps climbing, the brief "
                            "and the tools are asking for something the model "
                            "cannot produce consistently." % rejected, "warn")
                continue

            key = _dedupe_key(row)
            if key in seen:
                duplicates += 1
                if duplicates in (25, 100, 400):
                    # Two different things wear the same counter. Inventing
                    # the same example twice is a model repeating itself;
                    # answering the same question twice is a split that holds
                    # it twice, and telling somebody to raise the temperature
                    # would be advice about a problem they do not have.
                    ctx.log(
                        ("%d prompts so far were already in this run -- the "
                         "split holds them more than once, and only the first "
                         "answer is kept." if mode == "from_dataset" else
                         "%d generated rows so far were exact repeats of "
                         "earlier ones. A model asked the same thing twice "
                         "tends to answer it the same way -- raise the "
                         "temperature, or give it more varied prompts.")
                        % duplicates, "warn")
                continue
            seen.add(key)

            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            written += 1
            if note := _recent_note(row):
                recent.append(note)

            rate = written / max(time.time() - t0, 1e-6)
            ctx.metric(written, {
                "rows": written,
                "duplicates": duplicates,
                "empty": empty,
                "rejected": rejected,
                "rows_per_sec": round(rate, 3),
                "tokens_per_sec": result.get("tokens_per_sec"),
                "eta_s": round((target - written) / max(rate, 1e-6)),
            })
            ctx.progress(written, target, stage="writing")
            if written <= 2 or written % max(1, target // 10) == 0:
                ctx.log("Row %d of %d: %s"
                        % (written, target, _describe(row, text)))

    if not written:
        raise ValueError(
            "The model produced nothing usable. Check the instructions, and "
            "try the model in the Playground first to see what it says.")

    return _finish(cfg, ctx, out_path, {
        "written": written, "duplicates": duplicates, "empty": empty,
        "rejected": rejected, "stopped_early": stopped_early, "target": target,
        "duration_s": time.time() - t0, "mode": mode,
    })


# ---------------------------------------------------------------------------
# A model that is not on this machine
# ---------------------------------------------------------------------------

class ProviderRefused(RuntimeError):
    """The provider said no in a way that will not change on the next row."""


# Attempts per row before giving up on it, and the wait between them. Rate
# limits are a fact of every hosted API, and a thousand-row generation that
# fails the moment one arrives is not a feature.
_ATTEMPTS = 4
_BACKOFF = (2, 8, 20)


class HostedModel:
    """A model behind an API, wearing the two methods ModelHost has.

    The generation loop asks for a reply and gets one; whether that took a
    GPU or a POST is not its business. Everything about *which* API and what
    its request looks like lives in common/apimodels.py, so the preview in the
    browser and the run on the machine cannot disagree about it.

    Also used by an evaluation, which is why the name is not private: scoring
    a studio model against a hosted one is the same problem -- ask a model a
    question, take what comes back -- and a second implementation of retries,
    rate limits and per-provider parameter spellings would have drifted from
    this one within a month.
    """

    def __init__(self, spec: dict, ctx: Any) -> None:
        import httpx
        from common import apimodels

        self.api = apimodels
        self.conn = dict(spec.get("connection") or {})
        self.model = apimodels.model_name(self.conn, spec.get("model"))
        self.ctx = ctx
        self.warned_limit = False
        # Corrections this model turned out to need, remembered for the whole
        # run. Learned once and reapplied, never re-discovered: see `generate`.
        self.fixes: dict = {}
        if problem := apimodels.problems(self.conn):
            raise ValueError(problem)
        if not self.model:
            raise ValueError("No model was named at %s."
                             % apimodels.describe(self.conn))
        # One client for the whole run: a new TLS handshake per row is most of
        # the time a small row takes.
        self.client = httpx.Client(timeout=120.0)

    def ensure_loaded(self, _spec: dict, log) -> None:
        log("Writing with %s, over the network. This machine's GPU is not "
            "used -- the rows are billed to the account you connected."
            % self.api.describe(self.conn))

    def generate(self, _spec: dict, messages: list[dict], params: dict,
                 _stop, _log) -> dict:
        import httpx

        messages = self._inline_media(messages)
        req = self.api.chat_request(self.conn, self.model, messages, params,
                                    tools=params.get("tools"),
                                    schema=params.get("schema"))
        # Whatever this model objected to last time, it will object to again.
        # Applying the correction up front is the difference between one
        # request per row and two: without it every call sent the rejected
        # parameter, took a 400, fixed it, and sent the whole thing again --
        # doubling the requests, doubling the latency, and filling the log with
        # the same sentence thirty-four times in a three-row run.
        body = self._corrected(req["json"])
        started = time.time()
        last = ""
        for attempt in range(_ATTEMPTS):
            try:
                r = self.client.post(req["url"], headers=req["headers"],
                                     json=body)
            except httpx.HTTPError as e:
                last = str(e)
                self._wait(attempt, None, "the network")
                continue

            if r.status_code < 400:
                data = r.json()
                usage = self.api.chat_usage(self.conn, data)
                elapsed = max(time.time() - started, 1e-6)
                answer = self.api.reply(self.conn, data)
                return {
                    "text": answer["content"],
                    # Kept apart rather than folded into the text. A reasoning
                    # model's working is not part of the answer, and writing it
                    # into a dataset as though it were teaches a small model to
                    # narrate its own thinking out loud to the user.
                    "reasoning": answer["reasoning"],
                    "tool_calls": answer["tool_calls"],
                    "tokens_per_sec": round(usage["output_tokens"] / elapsed, 2),
                    "usage": usage,
                }

            last = self.api.error_message(self.conn, r.status_code, _json(r))
            # A parameter this model spells differently. Fix it once and the
            # rest of the run uses the corrected body.
            if fixed := self.api.retry_body(self.conn, body, r.text):
                self._remember(body, fixed)
                body = fixed
                continue
            if r.status_code in (401, 403, 404):
                raise ProviderRefused(last)
            if r.status_code == 429 or r.status_code >= 500:
                self._wait(attempt, r.headers.get("retry-after"), "the provider")
                continue
            raise ProviderRefused(last)
        raise RuntimeError(last or "no reply")

    def _inline_media(self, messages: list[dict]) -> list[dict]:
        """Pictures in the studio's store, as data URLs the provider can open.

        A provider cannot reach this studio, so a reference to a stored file
        is fetched here, through the runner's own door, and sent as bytes.
        Fetched once per file per run and remembered. A picture that cannot
        be fetched is left out, with a note, rather than sent as a string.
        """
        import base64
        cache = getattr(self, "_media_cache", None)
        if cache is None:
            cache = self._media_cache = {}
        out = []
        for m in messages:
            media = m.get("media")
            if not media:
                out.append(m)
                continue
            fixed = []
            for mm in media:
                ref = (mm or {}).get("ref") or ""
                if not ref.startswith("asset:"):
                    fixed.append(mm)
                    continue
                aid = ref[6:]
                if aid not in cache:
                    try:
                        import httpx as _hx
                        r = _hx.get("%s/api/assets/%s/file" % (self.ctx.controller_url, aid),
                                    headers={"X-Runner-Token": self.ctx.runner_token},
                                    timeout=60.0)
                        r.raise_for_status()
                        mime = r.headers.get("content-type", "image/png").split(";")[0]
                        cache[aid] = "data:%s;base64,%s" % (
                            mime, base64.b64encode(r.content).decode("ascii"))
                    except Exception as e:  # noqa: BLE001 - said, then skipped
                        self.ctx.log("A picture could not be fetched for the "
                                     "hosted model (%s); it was left out." % e, "warn")
                        cache[aid] = None
                if cache[aid]:
                    fixed.append({**mm, "url": cache[aid]})
            out.append({**m, "media": fixed})
        return out

    def _corrected(self, body: dict) -> dict:
        """This request with the corrections this model already asked for."""
        if not self.fixes:
            return body
        out = dict(body)
        for name, replacement in self.fixes.items():
            if name not in out:
                continue
            value = out.pop(name)
            if replacement:
                out[replacement] = value
        return out

    def _remember(self, before: dict, after: dict) -> None:
        """Record what had to change, and say so once rather than per row."""
        added = set(after) - set(before)
        for name in set(before) - set(after):
            self.fixes[name] = next(iter(added), None)
        self.ctx.log(
            "%s does not accept %s. Adjusted, and every later request in this "
            "run is sent that way -- this is not repeated per row."
            % (self.api.describe(self.conn),
               ", ".join(sorted(set(before) - set(after))) or "one of the settings"))

    def _wait(self, attempt: int, retry_after: str | None, who: str) -> None:
        delay = _BACKOFF[min(attempt, len(_BACKOFF) - 1)]
        if retry_after:
            try:
                delay = max(delay, min(float(retry_after), 60))
            except ValueError:
                pass
        if not self.warned_limit:
            self.warned_limit = True
            self.ctx.log("%s asked this run to slow down. Waiting %ds and "
                         "carrying on -- this is their rate limit, not the "
                         "studio's." % (who.capitalize(), delay), "warn")
        time.sleep(delay)


def _attempt(host, spec: dict, messages: list[dict], params: dict,
             i: int, ctx: Any) -> dict | None:
    """One request, or None if it failed in a way the run can walk past."""
    try:
        return host.generate(spec, messages, params, None, lambda _l: None)
    except ProviderRefused:
        # A rejected key or a model that does not exist is not a bad row: it
        # is every row. Carrying on would spend five thousand attempts
        # discovering the same thing.
        raise
    except Exception as e:  # noqa: BLE001 - one bad row must not end the run
        ctx.log("Row %d failed (%s); carrying on." % (i + 1, type(e).__name__),
                "debug")
        return None


def _attempts(host, spec: dict, params: dict, sources: Iterator, target: int,
              workers: int, ctx: Any) -> Iterator[tuple[int, dict, dict]]:
    """Finished attempts as (index, metadata, result), as they come back.

    One request at a time is the only honest thing to do to a GPU that can
    hold one model; against a hosted API it is just slow, and a thousand-row
    run spends its afternoon waiting on the network. So hosted runs keep
    several requests in flight and take the answers in whatever order they
    arrive -- rows are independent, and nothing downstream cares about order.
    """
    if workers <= 1:
        for i in range(target):
            messages, meta = next(sources)
            if result := _attempt(host, spec, messages, params, i, ctx):
                yield i, meta, result
        return

    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="gen")
    pending: dict = {}
    queued = 0
    try:
        while queued < target or pending:
            while queued < target and len(pending) < workers:
                messages, meta = next(sources)
                pending[pool.submit(_attempt, host, spec, messages, params,
                                    queued, ctx)] = (queued, meta)
                queued += 1
            done, _ = wait(list(pending), return_when=FIRST_COMPLETED)
            for fut in done:
                i, meta = pending.pop(fut)
                if result := fut.result():
                    yield i, meta, result
    finally:
        # Reached on cancellation too, when the consumer closes this generator.
        pool.shutdown(wait=False, cancel_futures=True)


def _workers(cfg: dict, hosted: bool, target: int) -> int:
    """How many requests to keep in flight."""
    if not hosted:
        return 1
    try:
        asked = int(cfg.get("workers") or _HOSTED_WORKERS)
    except (TypeError, ValueError):
        asked = _HOSTED_WORKERS
    return max(1, min(asked, _MAX_WORKERS, target))


def _json(r: Any):
    try:
        return r.json()
    except ValueError:
        return r.text


# ---------------------------------------------------------------------------
# What to ask for
# ---------------------------------------------------------------------------

def _sources(cfg: dict, mode: str, ctx: Any,
             recent: Any = None) -> Iterator[tuple[list[dict], dict]]:
    """An endless stream of (messages, metadata) to send to the model.

    `recent` is the last few rows this run has kept, filled in by the writer
    loop as they are written. A model asked the same question a thousand times
    answers it much the same way a thousand times; shown what it has already
    written, it goes somewhere else.
    """
    system = (cfg.get("system_prompt") or "").strip()
    instruction = (cfg.get("instruction") or "").strip()
    rng = random.Random(int(cfg.get("seed") or 1234))

    if mode == "from_prompts":
        prompts = _lines(cfg.get("prompts"))
        if not prompts:
            raise ValueError("No prompts were given to answer.")
        ctx.log("Answering %d prompts%s." % (
            len(prompts), ", cycling through them" if cfg.get("cycle") else ""))
        i = 0
        while True:
            p = prompts[i % len(prompts)]
            i += 1
            msgs = ([{"role": "system", "content": system}] if system else [])
            msgs.append({"role": "user", "content": p})
            yield msgs, {"prompt": p}

    elif mode == "from_dataset":
        # Batch inference: answer prompts that already exist rather than
        # inventing them. This is the mode for distilling a capable model's
        # answers onto questions you already have, and for producing a model's
        # own answers to a split so they can be read, corrected and trained
        # on -- both of which meant exporting the prompts, running a script,
        # and importing the result.
        #
        # Finite, unlike every other source here: it stops when the split runs
        # out, and the run reports fewer rows than it was asked for rather
        # than looping over the same prompts again.
        field = (cfg.get("prompt_field") or "").strip()
        from common import formatting
        fmt = formatting.resolve_format(cfg.get("source_format") or {})
        seen_any = False
        for row in _source_rows(cfg, ctx):
            prompt = _prompt_of(row, field, fmt)
            if not prompt:
                continue
            seen_any = True
            msgs = ([{"role": "system", "content": system}] if system else [])
            if instruction:
                # An instruction wraps the prompt rather than replacing it:
                # "answer this as a support agent would" over the question the
                # data already holds.
                msgs.append({"role": "user",
                             "content": instruction.replace("{prompt}", prompt)
                             if "{prompt}" in instruction
                             else "%s\n\n%s" % (instruction, prompt)})
            else:
                msgs.append({"role": "user", "content": prompt})
            yield msgs, {"prompt": prompt}
        if not seen_any:
            raise ValueError(
                "No prompts could be read from that split. Name the column "
                "the question is in, or choose a dataset of conversations.")
        return

    elif mode == "from_topics":
        topics = _lines(cfg.get("topics"))
        if not topics:
            raise ValueError("No topics were given to write about.")
        template = instruction or (
            "Write one realistic example about: {topic}\n\n"
            "Reply with the example only.")
        ctx.log("Generating across %d topics." % len(topics))
        i = 0
        while True:
            topic = topics[i % len(topics)]
            i += 1
            msgs = ([{"role": "system", "content": system}] if system else [])
            msgs.append({"role": "user",
                         "content": template.replace("{topic}", topic)})
            yield msgs, {"topic": topic}

    elif mode == "conversations":
        brief = instruction
        if not brief.strip():
            raise ValueError(
                "Say what the conversations should be about. This mode writes "
                "whole exchanges from a brief, and an empty brief describes "
                "every conversation equally.")
        tools = _tool_specs(cfg)
        topics = _lines(cfg.get("topics"))
        languages = _lines(cfg.get("languages"))
        writer = _conversation_prompt(cfg, tools)
        ctx.log("Writing whole conversations%s%s%s."
                % (" with %d tool%s declared" % (len(tools),
                                                 "" if len(tools) == 1 else "s")
                   if tools else "",
                   ", across %d situations" % len(topics) if topics else "",
                   ", in %s" % ", ".join(languages) if languages else ""))
        # Situation and language advance at different rates, so ten of each
        # give a hundred combinations rather than ten. Two lists stepped in
        # lockstep would ask for the same situation in the same language every
        # time they came round, and half a dataset would be one language.
        for i in itertools.count():
            topic = topics[i % len(topics)] if topics else ""
            language = ""
            if languages:
                # Shifted by one on every pass through the topics, so the
                # second time round a situation it is asked for in a different
                # language.
                nth = i + (i // len(topics) if topics else 0)
                language = languages[nth % len(languages)]
            ask = brief.replace("{topic}", topic).replace("{language}",
                                                          language)
            if topic and "{topic}" not in brief:
                ask += "\n\nThis one is about: %s" % topic
            if language and "{language}" not in brief:
                ask += "\n\nWrite this one in %s. Everything the person and "  \
                       "the assistant say is in %s." % (language, language)
            if already := _already_written(recent):
                ask += already
            msgs = [{"role": "system", "content": writer}]
            if system:
                msgs[0]["content"] += "\n\n" + system
            msgs.append({"role": "user", "content":
                         ask + "\n\nWrite one conversation."})
            yield msgs, {"conversation": True, "topic": topic,
                         "language": language, "tools": tools}

    elif mode == "from_seeds":
        seeds = _lines(cfg.get("seeds"))
        if len(seeds) < 2:
            raise ValueError(
                "Give at least two example rows to work from -- one is not "
                "enough for the model to see the pattern.")
        shown = int(cfg.get("seeds_shown") or 3)
        template = instruction or (
            "Here are some examples:\n\n{examples}\n\n"
            "Write one more in the same style, on a different subject. "
            "Reply with the new example only.")
        ctx.log("Expanding %d seed examples, showing %d at a time."
                % (len(seeds), shown))
        while True:
            picked = rng.sample(seeds, min(shown, len(seeds)))
            body = "\n\n".join("- %s" % s for s in picked)
            msgs = ([{"role": "system", "content": system}] if system else [])
            msgs.append({"role": "user",
                         "content": template.replace("{examples}", body)})
            yield msgs, {"seeds": picked}

    else:
        raise ValueError("Unknown generation mode %r." % mode)


# ---------------------------------------------------------------------------
# Whole conversations, in a shape the provider enforces
# ---------------------------------------------------------------------------
#
# Every other mode asks for text and hopes. This one states the shape of the
# answer as a JSON schema and lets the provider constrain decoding to it, which
# is the difference between a dataset of conversations and a dataset of prose
# describing conversations. It matters most for tool calls: a model asked in
# prose to "include a function call" writes one into its answer as text, and
# the row then teaches a model to TYPE a call rather than to make one.
#
# Arguments are a string in the schema rather than an object, for two reasons.
# Strict mode has no way to say "any object", and a string is exactly what the
# canonical row stores -- so nothing is re-encoded on the way in, and whatever
# the model wrote is what a reader sees.


def _conversation_schema(cfg: dict) -> dict:
    """The JSON shape one generated conversation must arrive in."""
    message: dict[str, Any] = {
        "role": {"type": "string", "enum": ["user", "assistant", "tool"]},
        "content": {"type": "string"},
    }
    if _wants_reasoning(cfg):
        message["reasoning"] = {"type": "string"}
    if _tool_specs(cfg):
        message["tool_calls"] = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "arguments": {"type": "string"},
                },
                "required": ["name", "arguments"],
                "additionalProperties": False,
            },
        }
    props: dict[str, Any] = {}
    if cfg.get("system_from_model"):
        props["system"] = {"type": "string"}
    props["messages"] = {
        "type": "array",
        "items": {"type": "object", "properties": message,
                  # Strict decoding requires every property to be required.
                  # "None of these" is an empty string or an empty list, which
                  # is why nothing here is nullable.
                  "required": list(message), "additionalProperties": False},
    }
    return {"type": "object", "properties": props,
            "required": list(props), "additionalProperties": False}


def _wants_reasoning(cfg: dict) -> bool:
    return bool(cfg.get("with_reasoning"))


def _tool_specs(cfg: dict) -> list[dict]:
    """The tools these conversations may call, flat and validated.

    Accepts what people actually have in front of them: a bare list of
    functions, OpenAI's `{"type": "function", "function": {...}}` wrapper, or
    a single function on its own.
    """
    raw = cfg.get("tools")
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return []
        try:
            raw = json.loads(raw)
        except ValueError as e:
            raise ValueError(
                "The tools could not be read as JSON (%s). Paste the function "
                "definitions as a JSON array." % e) from None
    if isinstance(raw, dict):
        raw = [raw]
    out = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        fn = item.get("function") if isinstance(item.get("function"), dict) \
            else item
        if not fn.get("name"):
            raise ValueError("A tool was given with no name.")
        out.append({"name": str(fn["name"]),
                    "description": str(fn.get("description") or ""),
                    "parameters": fn.get("parameters")
                    or fn.get("input_schema")
                    or {"type": "object", "properties": {}}})
    return out


_CONVERSATION_RULES = (
    "You write single training conversations for a dataset. Reply with one "
    "JSON object in the required shape and nothing else.\n"
    "\n"
    "Rules that make a row usable:\n"
    "* The person speaks first, and speaks like a person: short, direct, "
    "sometimes assuming context. Not a well-formed request every time.\n"
    "* Everything said is in the conversation's own language, including the "
    "assistant's answer. Do not translate, do not add an English gloss.\n"
    "* The assistant's answer is one or two sentences. It states what was "
    "done or what is true, and never restates the question back.\n"
    "* Write the whole exchange: the last message is always the assistant's "
    "answer, and it is never empty."
)

_TOOL_RULES = (
    "\n"
    "These functions exist. Only these, spelled exactly like this:\n"
    "%s\n"
    "* Call one only to DO something or to fetch something you were not told. "
    "A question about a state you already know is answered without a call.\n"
    "* `arguments` is a JSON object encoded as a string, and every argument in "
    "it must appear in that function's schema.\n"
    "* Every call is answered: the next message has role \"tool\" and its "
    "content is the JSON that function would have returned. Invent a "
    "realistic result, consistent with the arguments.\n"
    "* After the result, the assistant says what happened, in the "
    "conversation's language."
)

_REASONING_RULES = (
    "\n"
    "* Every assistant message carries `reasoning`: one or two sentences of "
    "first-person working that stops the moment the decision is made. It "
    "names the entity or the value it settled on. The answer stands alone and "
    "never refers back to it. The person's messages and tool results carry an "
    "empty `reasoning`."
)

# Without this the writer reasons about the request and leaves the call itself
# unexplained, which is the half the model has to learn to produce.
_REASONING_CALL_RULES = (
    "\n"
    "* A message that calls a function reasons about the call: which function, "
    "which entity or value it is passing and why that one. One short sentence, "
    "in the conversation's language, not the JSON written out in words."
)


def _conversation_prompt(cfg: dict, tools: list[dict]) -> str:
    """The standing instructions the writer gets for every row."""
    out = _CONVERSATION_RULES
    if tools:
        out += _TOOL_RULES % json.dumps(tools, ensure_ascii=False, indent=None)
    if _wants_reasoning(cfg):
        out += _REASONING_RULES
        if tools:
            out += _REASONING_CALL_RULES
    if cfg.get("system_from_model"):
        out += ("\n"
                "* `system` is the system prompt this conversation was held "
                "under. Write it in full, exactly as the brief describes it, "
                "including any data it is supposed to contain. Everything the "
                "assistant says must be consistent with it. It belongs there "
                "and nowhere else: never repeat it inside the conversation.")
    return out


def _recent_note(row: dict) -> str:
    """One line describing a written row, for the next request to avoid.

    The opening and whatever names the model invented for the call: those are
    what it repeats. Short on purpose -- this is prepended to every later
    request, so it is paid for on every row of the run.
    """
    opening = ""
    call = ""
    for m in row.get("messages") or []:
        if m.get("role") == "user" and not opening:
            opening = _clean(m.get("content") or "")[:90]
        for c in m.get("tool_calls") or []:
            call = call or (c.get("function") or {}).get("arguments") or ""
    if not opening:
        return ""
    return "%s%s" % (opening, "  ->  %s" % call[:110] if call else "")


def _already_written(recent: Any) -> str:
    """The avoid-list, as the writer sees it."""
    lines = list(recent or [])
    if not lines:
        return ""
    return ("\n\nAlready written in this dataset. Write none of these again, "
            "and invent different names, rooms, values and people from the "
            "ones showing here:\n"
            + "\n".join("- %s" % line for line in lines))


def _echoes(content: str, system: str) -> bool:
    """Whether a user turn is really the system prompt copied back.

    A writer that was asked for the system prompt sometimes writes it twice,
    once where it belongs and once as the thing the person said out loud.
    """
    head = content[:120].strip()
    return bool(system) and len(head) > 60 and head in system


def _conversation_row(text: str, meta: dict, cfg: dict) -> dict | None:
    """One generated JSON object as one canonical conversation row.

    Returns None for anything that would be a bad training example rather than
    repairing it into one: a call to a function that does not exist, arguments
    that are not JSON, a call nothing answered or unreasoned, a user turn that
    is the system prompt read back, an exchange that stops before the assistant
    replies. All
    of them are cheap to detect and expensive to find later, in a dataset that
    trained without complaint.
    """
    from common import conversation as C

    body = text.strip()
    if m := _FENCE.match(body):
        body = m.group(1).strip()
    try:
        obj = json.loads(body)
    except ValueError:
        return None
    if not isinstance(obj, dict) or not isinstance(obj.get("messages"), list):
        return None

    tools = meta.get("tools") or []
    known = {t["name"] for t in tools}
    messages: list[dict] = []

    system = (obj.get("system") or "").strip() \
        or (cfg.get("dataset_system_prompt") or "").strip()
    if system:
        messages.append({"role": "system", "content": _clean(system)})

    waiting: list[dict] = []      # calls made and not yet answered
    calls_made = 0
    for item in obj["messages"]:
        if not isinstance(item, dict):
            return None
        role = str(item.get("role") or "").lower()
        content = _clean(str(item.get("content") or ""))

        if role == "tool":
            if not waiting:
                return None       # a result for a call nobody made
            call = waiting.pop(0)
            messages.append({"role": "tool", "name": call["function"]["name"],
                             "tool_call_id": call["id"], "content": content})
            continue

        if role == "user":
            if waiting or not content:
                return None       # the person spoke over an unanswered call
            if _echoes(content, system):
                return None       # the system prompt copied into a user turn
            messages.append({"role": "user", "content": content})
            continue

        if role != "assistant":
            return None

        turn: dict[str, Any] = {"role": "assistant", "content": content}
        if reasoning := _clean(str(item.get("reasoning") or "")):
            turn["reasoning"] = reasoning
        calls = []
        for raw in item.get("tool_calls") or []:
            if not isinstance(raw, dict):
                return None
            name = str(raw.get("name") or "")
            if name not in known:
                return None
            args = raw.get("arguments")
            if isinstance(args, (dict, list)):
                args = json.dumps(args, ensure_ascii=False)
            try:
                json.loads(args or "")
            except (ValueError, TypeError):
                return None       # arguments a runtime could not parse
            calls_made += 1
            call = {"id": "call_%d" % calls_made, "type": "function",
                    "function": {"name": name, "arguments": str(args)}}
            calls.append(call)
            waiting.append(call)
        if calls:
            turn["tool_calls"] = calls
            if _wants_reasoning(cfg) and not turn.get("reasoning"):
                return None       # a call with nothing said about why
        elif not content:
            return None           # an assistant turn that says nothing
        messages.append(turn)

    if waiting:
        return None               # a call the conversation never answered
    roles = [m["role"] for m in messages]
    if "user" not in roles or messages[-1]["role"] != "assistant" \
            or not messages[-1].get("content"):
        return None
    if cfg.get("require_tool_call") and not calls_made:
        return None

    conv, _ = C.repair(C.from_messages(messages, tools))
    row = C.to_row(conv)
    for key in ("topic", "language"):
        if meta.get(key):
            row[key] = meta[key]
    return row


def _lines(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [ln.strip() for ln in str(value or "").splitlines() if ln.strip()]


# ---------------------------------------------------------------------------
# What comes back
# ---------------------------------------------------------------------------

_FENCE = re.compile(r"^```[a-zA-Z]*\n(.*?)\n?```$", re.DOTALL)


def _row(text: str, meta: dict, cfg: dict) -> dict | None:
    """Turn one generation into one dataset row.

    Chat shape by default, because that is what both trainers here read
    without any further configuration -- a generated dataset should be usable
    from the wizard immediately, not after a conversion step.
    """
    if meta.get("conversation"):
        return _conversation_row(text, meta, cfg)

    text = text.strip()
    if m := _FENCE.match(text):
        text = m.group(1).strip()

    if cfg.get("output") == "json":
        # The model was asked for JSON. Accept it if it really is; otherwise
        # keep the text rather than dropping the row, and let the dataset
        # inspector show what happened.
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except ValueError:
            pass

    system = (cfg.get("dataset_system_prompt") or "").strip()
    prompt = meta.get("prompt")

    if prompt is None and cfg.get("output") == "text":
        return {"text": text}

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    if prompt is not None:
        messages.append({"role": "user", "content": prompt})
        messages.append({"role": "assistant", "content": text})
    else:
        # No prompt: the generation is the whole example. Split it into a turn
        # pair if the model produced an obvious one, otherwise keep it whole.
        pair = _split_pair(text)
        if pair:
            messages.append({"role": "user", "content": pair[0]})
            messages.append({"role": "assistant", "content": pair[1]})
        else:
            return {"text": text, **({"topic": meta["topic"]} if meta.get("topic") else {})}

    row: dict = {"messages": messages}
    if meta.get("topic"):
        row["topic"] = meta["topic"]
    return row


_PAIR = re.compile(
    r"^\s*(?:Q|Question|Instruction|User|Prompt)\s*[:.\-]\s*(.+?)\n+"
    r"\s*(?:A|Answer|Response|Assistant|Output)\s*[:.\-]\s*(.+)$",
    re.IGNORECASE | re.DOTALL)


def _split_pair(text: str) -> tuple[str, str] | None:
    if m := _PAIR.match(text):
        q, a = m.group(1).strip(), m.group(2).strip()
        if q and a:
            return q, a
    return None


def _clip(text: str, n: int = 110) -> str:
    one = " ".join(text.split())
    return one if len(one) <= n else one[:n] + "…"


def _prompt_of(row: dict, field: str, fmt: dict) -> str:
    """The question in one row of the source split.

    A named column wins. Failing that, a conversation's last user turn, and
    failing that the first column that looks like a question -- the same order
    the prompt-set builder uses, so a dataset that can become a prompt set can
    also be answered in bulk.
    """
    if field:
        return str(row.get(field) or "").strip()
    if isinstance(row.get("messages"), list):
        from common import conversation as C
        conv, _ = C.repair(C.from_row(row, fmt))
        for m in reversed(conv[C.MESSAGES_KEY]):
            if m.get("role") == "user" and (m.get("content") or "").strip():
                return m["content"].strip()
        return ""
    for name in ("instruction", "prompt", "question", "input", "text"):
        if value := str(row.get(name) or "").strip():
            return value
    return ""


def _dedupe_key(row: dict) -> str:
    """What makes two generated rows the same row.

    For a conversation it is what the person said: two rows that open with the
    same question are the same example even when the invented sensor readings
    in them differ, and comparing whole rows would have kept both.
    """
    for m in row.get("messages") or []:
        if m.get("role") == "user" and (m.get("content") or "").strip():
            return " ".join(m["content"].lower().split())[:2000]
    return json.dumps(row, sort_keys=True)[:2000]


def _describe(row: dict, text: str) -> str:
    """One line about a written row, for the log.

    A conversation's raw JSON says nothing worth reading at 110 characters --
    the question and whether it called anything is the whole of what someone
    watching wants to know.
    """
    messages = row.get("messages") or []
    if not messages:
        return _clip(text)
    asked = next((m.get("content") for m in messages
                  if m.get("role") == "user"), "")
    called = [c["function"]["name"] for m in messages
              for c in (m.get("tool_calls") or [])]
    if not asked:
        return _clip(text)
    return "%s%s" % (_clip(asked, 80),
                     " → %s" % ", ".join(called) if called else "")


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Carrying an existing conversation further
# ---------------------------------------------------------------------------
#
# The other three modes write conversations from nothing. This one takes the
# ones you already have and makes them longer, which is a different problem and
# the one most tool-calling datasets actually have: they are full of single
# exchanges -- a question, a call, an answer -- and a model trained only on
# those never learns what to do on the fourth turn, when the context is long
# and half of it is its own earlier work.
#
# Each extra turn is two calls, not one, and deliberately so. A model asked to
# write "the next exchange" writes both halves in its own voice and produces a
# user who talks like an assistant: complete sentences, no typos, no shifting
# subject, never impatient. Asking separately -- once in the character of the
# person, once as the assistant -- gets a user turn that reads like a user.

# Asked of the writer when the conversation being extended shows its working.
#
# The hosted APIs will hand back a reasoning summary, but only when the model
# happened to reason enough to have one: measured over six short follow-up
# turns, four came back with a summary and two -- "Perfect, thanks." among them
# -- came back with none, identically whether the summary was asked for as
# "auto" or as "detailed". That is reasonable behaviour and useless to rely on,
# because a row where some assistant turns show their working and others do not
# teaches a model to reason once and then stop.
#
# So it is asked for in the open, as a block this studio already knows how to
# read back -- the same `<think>` split the trainer and the playground use.
_WRITE_WORKING = (
    "You are writing the assistant's next turn in a dataset where every turn "
    "shows its working.\n"
    "Reply with one JSON object and nothing else, no code fence:\n"
    '{"reasoning": "<your working>", "content": "<the answer>"}\n'
    "The working is first-person deliberation and it stops the moment you have "
    "decided. The answer stands on its own and never refers back to it. Keep "
    "the working in proportion: a turn that needs little thought gets a "
    "sentence of it. Always write both fields."
)

_USER_PROMPT = (
    "You are simulating the PERSON in this conversation, not the assistant.\n"
    "Write only their next message. Nothing else: no preamble, no quotation "
    "marks, no explanation of what you are doing.\n"
    "It should follow naturally from what has been said, and be the kind of "
    "thing a real person types -- short, direct, and sometimes assuming "
    "context rather than restating it. Vary what you ask for: a follow-up, a "
    "correction, a change of mind, a related but different request.\n"
    "%s"
)


def _has_reasoning(rows: list[dict]) -> bool:
    """Whether the conversations being lengthened show their working.

    Read off the rows themselves rather than off the dataset's recorded
    format: a dataset reached through a URL arrives without one, and guessing
    "no" silently drops the very thing being extended.
    """
    from common import conversation as C
    for row in rows:
        conv = C.from_row(row if isinstance(row, dict) else {})
        for m in conv[C.MESSAGES_KEY]:
            if (m.get("reasoning") or "").strip():
                return True
    return False


def _extend(cfg: dict, ctx: Any, host: Any, spec: dict, params: dict,
            out_path: Path, target: int) -> dict:
    """Make the conversations in a dataset longer, a turn at a time."""
    from common import conversation as C

    turns = max(1, int(cfg.get("extra_turns") or 2))
    persona = (cfg.get("persona") or "").strip()
    invent_results = bool(cfg.get("invent_tool_results", True))
    rows = _source_rows(cfg, ctx)

    # A conversation whose existing turns carry working needs its new ones to
    # carry working too, or the row teaches a model to reason on the first
    # answer and stop reasoning on every one after it -- which is a stranger
    # lesson than either reasoning or not. The writing model is asked for it
    # explicitly, since a hosted model does not volunteer its thinking.
    #
    # The rows arrive as a stream, so the ones read to decide are put back
    # rather than consumed -- reading a dataset twice means downloading it
    # twice, and dropping the first fifty rows to answer a question about
    # them would be worse.
    head = list(itertools.islice(rows, 50))
    wants_reasoning = _has_reasoning(head)
    rows = itertools.chain(head, rows)
    if wants_reasoning:
        params = {**params, "reasoning": True}
        ctx.log("These conversations show the model's working, so the added "
                "turns are written with theirs as well.")
    else:
        ctx.log("These conversations carry no working, so the added turns "
                "carry none either.")

    written = extended = failed = 0
    invented = 0
    stopped_early = False
    t0 = time.time()

    with out_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            if written >= target:
                break
            if ctx.should_cancel():
                stopped_early = True
                ctx.log("Stopping after %d rows, and keeping them." % written,
                        "warn")
                break

            conv, _ = C.repair(C.from_row(row))
            if not conv[C.MESSAGES_KEY]:
                continue
            tools = C.flat_tools(conv)
            before = len(conv[C.MESSAGES_KEY])

            try:
                added, made_up = _extend_one(conv, host, spec, params, turns,
                                             persona, tools, invent_results,
                                             wants_reasoning)
                invented += made_up
            except ProviderRefused:
                raise
            except Exception as e:  # noqa: BLE001 - one bad row is not the run
                failed += 1
                ctx.log("Row %d could not be extended (%s); it is kept as it "
                        "was." % (written + 1, type(e).__name__), "debug")
                added = 0

            if added:
                extended += 1
            # Written whether or not it grew. A row that could not be extended
            # is still a row, and dropping it would quietly shrink the dataset
            # every time the provider hiccuped.
            fh.write(json.dumps(
                C.to_row(conv, split=row.get("split")), ensure_ascii=False) + "\n")
            fh.flush()
            written += 1

            rate = written / max(time.time() - t0, 1e-6)
            ctx.metric(written, {
                "rows": written, "extended": extended, "failed": failed,
                "invented_results": invented,
                "rows_per_sec": round(rate, 3),
                "eta_s": round((target - written) / max(rate, 1e-6)),
            })
            ctx.progress(written, target, stage="writing")
            if written <= 2 or written % max(1, target // 10) == 0:
                grew = len(conv[C.MESSAGES_KEY]) - before
                ctx.log("Row %d of %d: %d turns -> %d (+%d)"
                        % (written, target, before, before + grew, grew))

    if not written:
        raise ValueError(
            "No rows in that dataset could be read as conversations. Convert "
            "it to the standard conversation format first -- the dataset page "
            "has the button.")
    if invented:
        ctx.log("%d tool results in this dataset were INVENTED by the model "
                "because no real tool was called. They are plausible and they "
                "are not true. That is usually what you want for teaching the "
                "*shape* of a tool conversation, and it is never what you want "
                "for teaching facts." % invented, "warn")

    return _finish(cfg, ctx, out_path, {
        "written": written, "duplicates": 0, "empty": failed,
        "stopped_early": stopped_early, "target": target,
        "duration_s": time.time() - t0, "mode": "extend_conversations",
    })


def _extend_one(conv: dict, host: Any, spec: dict, params: dict, turns: int,
                persona: str, tools: list[dict], invent_results: bool,
                show_working: bool = False) -> tuple[int, int]:
    """Add `turns` exchanges to one conversation, in place."""
    from common import conversation as C

    msgs = conv[C.MESSAGES_KEY]
    added = invented = 0

    # A conversation that ends on a call nobody answered cannot be carried
    # further, and a great many of them do: the best-known tool-calling
    # datasets are question-then-call and stop there, with no result and no
    # reply. That is also not a conversation any provider will accept as
    # input -- the Responses API refuses outright, "no tool output found for
    # function call" -- so the loop is closed before anything is added to it.
    if invent_results:
        for call in _unanswered(msgs):
            msgs.append({"role": "tool", "tool_call_id": call.get("id"),
                         "name": call["function"]["name"],
                         "content": _invent_result(host, spec, params, call, tools)})
            invented += 1
            added += 1
        if added:
            # The assistant never got to say what the result meant. Without
            # this the row still ends mid-exchange, one step further along.
            text, thinking, _calls = _assistant_turn(
                conv, host, spec, params, tools, show_working)
            if text:
                turn: dict[str, Any] = {"role": "assistant", "content": text}
                if thinking:
                    turn["reasoning"] = thinking
                msgs.append(turn)
                added += 1

    for _ in range(turns):
        # 1. The person's next message, written in their character.
        history = _as_plain(msgs)
        ask = [{"role": "system", "content": _USER_PROMPT % (
            ("They are: " + persona) if persona else
            "Stay consistent with how they have written so far.")},
            {"role": "user", "content":
             "The conversation so far:\n\n%s\n\nWrite their next message."
             % history}]
        said = (host.generate(spec, ask, {**params, "tools": None},
                              None, lambda _l: None).get("text") or "").strip()
        said = said.strip('"').strip()
        if not said:
            break
        msgs.append({"role": "user", "content": said})
        added += 1

        # 2. The assistant's reply, with this row's own tools declared so it
        #    can call one -- which is the whole point on a tool dataset.
        for _step in range(4):
            text, thinking, calls = _assistant_turn(
                conv, host, spec, params, tools, show_working)
            turn: dict[str, Any] = {"role": "assistant"}
            if text:
                turn["content"] = text
            if thinking:
                turn["reasoning"] = thinking
            if calls:
                turn["tool_calls"] = calls
            if not turn.get("content") and not calls:
                break
            msgs.append(turn)
            added += 1
            if not calls:
                break
            if not invent_results:
                break
            # A call with nothing to answer it leaves the conversation
            # unfinished, and an unfinished conversation is a row that ends on
            # the model asking a question of a tool that never replied.
            for call in calls:
                msgs.append({
                    "role": "tool",
                    "tool_call_id": call.get("id"),
                    "name": call["function"]["name"],
                    "content": _invent_result(host, spec, params, call, tools),
                })
                invented += 1
                added += 1
    return added, invented


def _invent_result(host: Any, spec: dict, params: dict, call: dict,
                   tools: list[dict]) -> str:
    """A plausible return value for a call nothing actually executed."""
    schema = next((t for t in tools
                   if t.get("name") == call["function"]["name"]), {})
    ask = [{"role": "system", "content":
            "You are standing in for a piece of software. Reply with the JSON "
            "that this function would return, and nothing else -- no prose, no "
            "code fence, no explanation. Make it realistic and internally "
            "consistent with the arguments."},
           {"role": "user", "content":
            "Function: %s\nWhat it does: %s\nIts schema: %s\n"
            "It was called with: %s\n\nReturn the JSON it would produce."
            % (call["function"]["name"], schema.get("description") or "unknown",
               json.dumps(schema.get("parameters") or {}, ensure_ascii=False),
               call["function"]["arguments"])}]
    text = (host.generate(spec, ask, {**params, "tools": None}, None,
                          lambda _l: None).get("text") or "").strip()
    # A model told not to use a code fence sometimes uses a code fence.
    text = re.sub(r"^```[a-z]*\s*|\s*```$", "", text).strip()
    try:
        json.loads(text)
        return text
    except ValueError:
        # Not JSON. Wrapped rather than discarded, so the conversation still
        # has a result in the shape a result goes in.
        return json.dumps({"result": text[:2000]}, ensure_ascii=False)


def _unanswered(msgs: list[dict]) -> list[dict]:
    """Tool calls in this conversation that no result ever answered."""
    answered = {m.get("tool_call_id") for m in msgs if m.get("role") == "tool"}
    return [c for m in msgs if m.get("role") == "assistant"
            for c in (m.get("tool_calls") or [])
            if c.get("id") not in answered]


def _assistant_turn(conv: dict, host: Any, spec: dict, params: dict,
                    tools: list[dict], show_working: bool) -> tuple:
    """One reply from the writer, as (answer, working, tool calls).

    Two ways of asking, and the reliable one cannot call a tool.

    A hosted API hands back its own reasoning only when the model happened to
    reason enough to have a summary worth giving. Measured over six short
    follow-up turns: four came back with one and two did not, identically
    whether the summary was requested as "auto" or "detailed", and asking the
    model in the prompt to write a <think> block instead did worse -- three of
    six, because a reasoning model does its thinking internally and answers.

    Asking for a JSON object with the two fields in it got six of six. So that
    is what is asked for whenever a row has no tools to declare -- which is
    every row of a plain reasoning dataset.

    A row WITH tools is asked the ordinary way, because a tool call has to come
    back through the API's own machinery and cannot be a field in an object the
    model typed. Those turns take the API's working when there is one.
    """
    if show_working and not tools:
        msgs = [{"role": "system", "content": _WRITE_WORKING}] + _for_provider(conv)
        reply = host.generate(spec, msgs, {**params, "tools": None},
                              None, lambda _l: None)
        text, thinking = _read_written(reply.get("text") or "")
        if text or thinking:
            return _clean(text), _clean(thinking), []
        # The model did not produce the object. Falling through loses the
        # working rather than the turn, which is the better of the two.
    reply = host.generate(spec, _for_provider(conv),
                          {**params, "tools": tools or None},
                          None, lambda _l: None)
    return (_clean(reply.get("text") or ""),
            _clean(reply.get("reasoning") or ""),
            reply.get("tool_calls") or [])


def _read_written(text: str) -> tuple[str, str]:
    """The answer and the working out of the object the writer was asked for."""
    body = (text or "").strip()
    if m := _FENCE.match(body):
        body = m.group(1).strip()
    try:
        obj = json.loads(body)
    except ValueError:
        return "", ""
    if not isinstance(obj, dict):
        return "", ""
    return str(obj.get("content") or ""), str(obj.get("reasoning") or "")


def _clean(text: str) -> str:
    """No control token belonging to a chat template reaches a dataset.

    Stripped on the way IN as well as on the way out. Reading a dataset strips
    these anyway, but a row is also downloaded, published and read by people,
    and a file that is only clean once something else has been through it is
    not a clean file.
    """
    from common import formatting as F
    return F.strip_special(text or "").strip()


def _for_provider(conv: dict) -> list[dict]:
    """The conversation as a provider's chat API wants it.

    Two things are stripped. `reasoning` is ours and not a field any of them
    accept. And a tool call with no result is dropped rather than sent: the
    Responses API rejects the whole request over one, and there is no useful
    version of "carry on from here" that includes a question the model asked
    and nothing answered.
    """
    from common import conversation as C
    msgs = conv[C.MESSAGES_KEY]
    orphans = {c.get("id") for c in _unanswered(msgs)}
    out = []
    for m in msgs:
        item = {k: v for k, v in m.items() if k != "reasoning"}
        if calls := item.get("tool_calls"):
            kept = [c for c in calls if c.get("id") not in orphans]
            if kept:
                item["tool_calls"] = kept
            else:
                item.pop("tool_calls", None)
                if not (item.get("content") or "").strip():
                    continue
        out.append(item)
    return out


def _as_plain(messages: list[dict]) -> str:
    """The conversation as readable text, for the user-simulator prompt.

    Deliberately not the training format. What is wanted here is for a model to
    *read* the conversation and understand who wants what, and turn markers are
    noise for that -- a tool call is far more legible as one line naming the
    function than as the syntax of whichever format it will eventually render
    in.
    """
    lines = []
    for m in messages[-12:]:
        role = m.get("role")
        if role == "tool":
            lines.append("[%s returned: %s]"
                         % (m.get("name") or "tool", _clip(m.get("content") or "", 200)))
            continue
        who = {"user": "Person", "assistant": "Assistant"}.get(role, role)
        if content := (m.get("content") or "").strip():
            lines.append("%s: %s" % (who, content))
        for call in m.get("tool_calls") or []:
            lines.append("[%s called %s(%s)]"
                         % (who, call["function"]["name"],
                            _clip(call["function"]["arguments"], 200)))
    return "\n".join(lines)


def _source_rows(cfg: dict, ctx: Any) -> Iterator[dict]:
    """Rows of the dataset being extended."""
    from . import source
    path = source.local_copy({**cfg, "dataset": cfg.get("source_dataset"),
                              "dataset_is_local": True,
                              "dataset_label": cfg.get("source_label")
                              or "the dataset"}, ctx)
    want = (cfg.get("source_split") or "").strip()
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if not isinstance(row, dict):
                continue
            if want and (row.get("split") or "train") != want:
                continue
            yield row


def _preamble(ctx: Any, cfg: dict, mode: str, target: int, spec: dict) -> None:
    who = spec.get("label") or spec.get("base_model") or spec.get("model")
    if not who and (conn := spec.get("connection")):
        from common import apimodels
        who = apimodels.describe(conn)
    who = who or "the model you trained"
    if mode == "extend_conversations":
        ctx.log("Adding turns to %s conversations from %s, using %s."
                % (f"{target:,}", cfg.get("source_label") or "the dataset", who))
    else:
        ctx.log("Writing %s rows with %s." % (f"{target:,}", who))
    ctx.log("Generated data is not free data. Three things go wrong with it, "
            "and none of them look like failure while it runs:")
    ctx.log("  1. The model repeats itself. Watch the duplicate count below; "
            "if it climbs, the prompts are not varied enough.")
    ctx.log("  2. The model is confidently wrong, and a model trained on this "
            "learns to be wrong the same way. Read a sample before training "
            "on it -- the dataset page shows you rows.")
    ctx.log("  3. The result cannot be better than the model that wrote it. "
            "This is for teaching a small model what a larger one knows, not "
            "for creating knowledge neither has.")
    if mode == "from_seeds":
        ctx.log("Everything it writes will resemble the seeds you gave it. "
                "That is the point, and it is also the ceiling.")
    if mode == "conversations" and _tool_specs(cfg):
        ctx.log("The tool results in these conversations are INVENTED. No "
                "function runs: the model writes what it thinks one would have "
                "returned. That teaches the shape of a tool exchange -- when to "
                "call, with which arguments, how to answer afterwards -- and it "
                "teaches nothing true about your systems.", "warn")
    if mode == "extend_conversations":
        ctx.log("Each extra turn is two requests: one asking the model to be "
                "the person, one asking it to be the assistant. Asked for both "
                "at once it writes a user who talks like an assistant, and a "
                "model trained on that learns to answer questions nobody "
                "phrases that way.")


def _finish(cfg: dict, ctx: Any, path: Path, stats: dict) -> dict:
    archive = Path(ctx.workdir) / "dataset.zip"
    shutil.make_archive(str(archive.with_suffix("")), "zip",
                        root_dir=str(path.parent), base_dir=path.name)

    kept = stats["written"]
    asked = stats["target"]
    rejected = stats.get("rejected") or 0
    ctx.log("%s %s rows in %.0fs (%d exact repeats and %d empty replies were "
            "dropped%s)." % ("Stopped early with" if stats["stopped_early"] else "Wrote",
                             f"{kept:,}", stats["duration_s"], stats["duplicates"],
                             stats["empty"],
                             ", and %d replies that were not usable rows"
                             % rejected if rejected else ""))
    if stats["duplicates"] > kept * 0.2:
        ctx.log("More than a fifth of what the model produced was a repeat of "
                "something it had already written. The dataset is smaller and "
                "less varied than the number of rows suggests.", "warn")
    if kept < asked and not stats["stopped_early"]:
        ctx.log("Asked for %s rows and kept %s: the difference was dropped as "
                "repeats, empty replies%s." % (f"{asked:,}", f"{kept:,}",
                                               " or unusable rows" if rejected
                                               else ""), "warn")

    return {
        "kind": "generate_dataset",
        "rows": kept,
        "requested": asked,
        "duplicates_dropped": stats["duplicates"],
        "empty_dropped": stats["empty"],
        "mode": stats["mode"],
        "duration_s": round(stats["duration_s"], 1),
        "stopped_early": stats["stopped_early"],
        "dataset_name": cfg.get("dataset_name") or "Generated dataset",
        "artifact_path": str(archive),
        "artifact_size": archive.stat().st_size,
    }
