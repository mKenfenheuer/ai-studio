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
    if spec.get("connection"):
        host: Any = _HostedModel(spec, ctx)
    elif spec.get("job_id") or spec.get("base_model"):
        host = inference.ModelHost(ctx.controller_url, ctx.runner_token,
                                   ctx.capabilities)
    else:
        raise ValueError("No model was chosen to generate the data with.")

    ctx.progress(0, target, stage="loading_model")
    host.ensure_loaded(spec, lambda line: ctx.log(line))

    _preamble(ctx, cfg, mode, target, spec)

    params = {
        "max_new_tokens": int(cfg.get("max_new_tokens") or DEFAULT_MAX_TOKENS),
        "temperature": float(cfg.get("temperature") or 0.9),
        "top_p": float(cfg.get("top_p") or 0.95),
    }
    if effort := (cfg.get("reasoning_effort") or "").strip():
        params["reasoning_effort"] = effort

    if mode == "extend_conversations":
        # A different shape of work: many calls per row rather than one, and
        # the row already exists. It gets its own loop rather than being bent
        # into the single-shot one below.
        return _extend(cfg, ctx, host, spec, params, out_path, target)

    sources = _sources(cfg, mode, ctx)

    seen: set[str] = set()
    written = 0
    duplicates = 0
    empty = 0
    stopped_early = False
    t0 = time.time()

    with out_path.open("w", encoding="utf-8") as fh:
        for i in range(target):
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

            messages, meta = next(sources)
            try:
                result = host.generate(spec, messages, params, None,
                                       lambda _l: None)
            except ProviderRefused:
                # A rejected key or a model that does not exist is not a bad
                # row: it is every row. Carrying on would spend five thousand
                # attempts discovering the same thing.
                raise
            except Exception as e:  # noqa: BLE001 - one bad row must not end the run
                ctx.log("Row %d failed (%s); carrying on." % (i + 1, type(e).__name__),
                        "debug")
                continue

            text = (result.get("text") or "").strip()
            if not text:
                empty += 1
                continue

            row = _row(text, meta, cfg)
            if row is None:
                empty += 1
                continue

            key = json.dumps(row, sort_keys=True)[:2000]
            if key in seen:
                duplicates += 1
                if duplicates in (25, 100, 400):
                    ctx.log("%d generated rows so far were exact repeats of "
                            "earlier ones. A model asked the same thing twice "
                            "tends to answer it the same way -- raise the "
                            "temperature, or give it more varied prompts."
                            % duplicates, "warn")
                continue
            seen.add(key)

            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            written += 1

            rate = written / max(time.time() - t0, 1e-6)
            ctx.metric(written, {
                "rows": written,
                "duplicates": duplicates,
                "empty": empty,
                "rows_per_sec": round(rate, 3),
                "tokens_per_sec": result.get("tokens_per_sec"),
                "eta_s": round((target - written) / max(rate, 1e-6)),
            })
            ctx.progress(written, target, stage="training")
            if written <= 2 or written % max(1, target // 10) == 0:
                ctx.log("Row %d of %d: %s" % (written, target, _clip(text)))

    if not written:
        raise ValueError(
            "The model produced nothing usable. Check the instructions, and "
            "try the model in the Playground first to see what it says.")

    return _finish(cfg, ctx, out_path, {
        "written": written, "duplicates": duplicates, "empty": empty,
        "stopped_early": stopped_early, "target": target,
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


class _HostedModel:
    """A model behind an API, wearing the two methods ModelHost has.

    The generation loop asks for a reply and gets one; whether that took a
    GPU or a POST is not its business. Everything about *which* API and what
    its request looks like lives in common/apimodels.py, so the preview in the
    browser and the run on the machine cannot disagree about it.
    """

    def __init__(self, spec: dict, ctx: Any) -> None:
        import httpx
        from common import apimodels

        self.api = apimodels
        self.conn = dict(spec.get("connection") or {})
        self.model = apimodels.model_name(self.conn, spec.get("model"))
        self.ctx = ctx
        self.warned_limit = False
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

        req = self.api.chat_request(self.conn, self.model, messages, params,
                                    tools=params.get("tools"))
        body = req["json"]
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
                self.ctx.log("Adjusting the request for this model: %s"
                             % ", ".join(sorted(set(fixed) - set(body))
                                         or ["dropped an unsupported setting"]))
                body = fixed
                continue
            if r.status_code in (401, 403, 404):
                raise ProviderRefused(last)
            if r.status_code == 429 or r.status_code >= 500:
                self._wait(attempt, r.headers.get("retry-after"), "the provider")
                continue
            raise ProviderRefused(last)
        raise RuntimeError(last or "no reply")

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


def _json(r: Any):
    try:
        return r.json()
    except ValueError:
        return r.text


# ---------------------------------------------------------------------------
# What to ask for
# ---------------------------------------------------------------------------

def _sources(cfg: dict, mode: str, ctx: Any) -> Iterator[tuple[list[dict], dict]]:
    """An endless stream of (messages, metadata) to send to the model."""
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


def _extend(cfg: dict, ctx: Any, host: Any, spec: dict, params: dict,
            out_path: Path, target: int) -> dict:
    """Make the conversations in a dataset longer, a turn at a time."""
    from common import conversation as C

    turns = max(1, int(cfg.get("extra_turns") or 2))
    persona = (cfg.get("persona") or "").strip()
    invent_results = bool(cfg.get("invent_tool_results", True))
    rows = _source_rows(cfg, ctx)

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
                                             persona, tools, invent_results)
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
            ctx.progress(written, target, stage="training")
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
                persona: str, tools: list[dict], invent_results: bool) -> tuple[int, int]:
    """Add `turns` exchanges to one conversation, in place."""
    from common import conversation as C

    msgs = conv[C.MESSAGES_KEY]
    added = invented = 0

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
            reply = host.generate(spec, _for_provider(conv),
                                  {**params, "tools": tools or None},
                                  None, lambda _l: None)
            calls = reply.get("tool_calls") or []
            turn: dict[str, Any] = {"role": "assistant"}
            if text := (reply.get("text") or "").strip():
                turn["content"] = text
            if thinking := (reply.get("reasoning") or "").strip():
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


def _for_provider(conv: dict) -> list[dict]:
    """The conversation as a provider's chat API wants it."""
    from common import conversation as C
    return [{k: v for k, v in m.items() if k != "reasoning"}
            for m in conv[C.MESSAGES_KEY]]


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
    who = spec.get("label") or spec.get("base_model") or "the model you trained"
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
    ctx.log("%s %s rows in %.0fs (%d exact repeats and %d empty replies were "
            "dropped)." % ("Stopped early with" if stats["stopped_early"] else "Wrote",
                           f"{kept:,}", stats["duration_s"], stats["duplicates"],
                           stats["empty"]))
    if stats["duplicates"] > kept * 0.2:
        ctx.log("More than a fifth of what the model produced was a repeat of "
                "something it had already written. The dataset is smaller and "
                "less varied than the number of rows suggests.", "warn")
    if kept < asked and not stats["stopped_early"]:
        ctx.log("Asked for %s rows and kept %s: the difference was dropped as "
                "repeats or empty replies." % (f"{asked:,}", f"{kept:,}"), "warn")

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
