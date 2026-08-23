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

    sources = _sources(cfg, mode, ctx)
    params = {
        "max_new_tokens": int(cfg.get("max_new_tokens") or DEFAULT_MAX_TOKENS),
        "temperature": float(cfg.get("temperature") or 0.9),
        "top_p": float(cfg.get("top_p") or 0.95),
    }

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

        req = self.api.chat_request(self.conn, self.model, messages, params)
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
                return {
                    "text": self.api.chat_text(self.conn, data),
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
