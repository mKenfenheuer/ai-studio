"""Scoring models against a fixed set of prompts.

The question this answers is the one a studio with more than one week of
history cannot otherwise answer: **is the model I trained today better than
the one I trained last week?** Nothing already in the app could say. Training
loss is not comparable between runs on different data. A conversation in the
playground compares two models on whatever you happened to type, in whatever
order you happened to type it, with the previous one no longer loaded.

So an evaluation is a job like any other -- queued, logged, stoppable -- and
it does the one thing that makes comparison possible: it puts the *same*
prompts to every model, and writes down what came back.

## What is scored, and what each number is worth

Four measures, because no single one of them is trustworthy on its own and
saying so is more useful than picking a favourite:

* **Loss on the expected answer.** Teacher-forced: how surprised the model is
  by the answer you said was right, given the prompt. This is the honest one.
  It does not care about wording, it varies smoothly, and it can tell two
  models apart that both got zero exact matches. It is also the only one that
  needs no judge and cannot be gamed by verbosity.

* **Exact match** and **contains**. Blunt instruments, useful for a
  classification-shaped task ("reply with one of: on, off, toggle") and close
  to meaningless for free text. Reported because when they *are* meaningful
  they are the easiest number in the world to explain.

* **Token overlap (F1).** A middle ground: it credits a right answer worded
  differently, and it happily credits a wrong answer that reuses the right
  words. Read alongside the loss, not instead of it.

* **Character overlap (chrF).** Token F1 with the tokens replaced by
  character n-grams. It is the one that behaves on morphology, on languages
  that do not put spaces between words, and on an answer that is right but
  inflected differently -- all cases where token overlap reports zero and a
  reader concludes the model failed.

* **JSON validity.** Only when the expected answer is itself JSON. A model
  asked for structured output either produces something that parses or does
  not, and that is a different question from whether the contents are right;
  both are reported, because a model that is always valid and usually wrong
  and a model that is usually right and sometimes unparseable need different
  fixes.

## What it is being compared against

A scoring can include models that are not runs of this studio at all:

* **A model off the Hub** -- most usefully the base a fine-tune was built
  from. Downloaded and loaded like any other, so every measure works on it.
  This is the comparison that says whether the training helped, and until it
  existed the studio could only ever compare two of your own models.
* **A model behind an API.** Asked over the network and scored on what comes
  back. There is no loss on the expected answer for one of these: that needs
  the model's own probabilities, and no hosted provider gives them out. The
  row says so rather than leaving the column blank.

Each model is asked with **its own** recorded system prompt unless the
scoring overrides it for everybody. A run trained with a system prompt and
scored without one is being asked to do a job nobody told it about.

Every generation is greedy by default. Sampling makes a model score
differently on two identical runs, and a comparison whose noise is larger than
its signal is worse than no comparison.
"""
from __future__ import annotations

import json
import math
import re
import time
from typing import Any

from runner import inference

from .lora_llm import Cancelled

# A hard ceiling on how much text one answer contributes to the stored record.
# The results are kept in the database so they can be compared months later;
# a runaway model that writes 8000 tokens of nothing should not be the reason
# that stops working.
MAX_STORED_CHARS = 1500


def _normalise(text: str) -> str:
    """Lower case, collapse whitespace, drop trailing punctuation.

    Deliberately mild. Aggressive normalisation (stripping articles, stemming)
    inflates exact-match scores without the model having got anything more
    right, which is the opposite of what a score is for.
    """
    text = (text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text.rstrip(" .!?,;:")


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", _normalise(text))


def _f1(answer: str, expected: str) -> float:
    """Token overlap, as the harmonic mean of precision and recall."""
    got, want = _tokens(answer), _tokens(expected)
    if not got or not want:
        return 1.0 if got == want else 0.0
    common: dict[str, int] = {}
    for t in want:
        common[t] = common.get(t, 0) + 1
    hits = 0
    for t in got:
        if common.get(t, 0) > 0:
            common[t] -= 1
            hits += 1
    if not hits:
        return 0.0
    precision = hits / len(got)
    recall = hits / len(want)
    return 2 * precision * recall / (precision + recall)


def _ngrams(text: str, n: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    for i in range(len(text) - n + 1):
        g = text[i:i + n]
        counts[g] = counts.get(g, 0) + 1
    return counts


def _chrf(answer: str, expected: str, order: int = 6, beta: float = 2.0) -> float:
    """Character n-gram F-score: token overlap that survives morphology.

    Token F1 scores "walked" against "walking" as a complete miss, and scores
    any language that does not separate words with spaces as a complete miss
    every time. chrF sees five sixths of the same characters. Recall is
    weighted more heavily than precision (beta=2), which is the standard
    choice and the right one here: an answer that omits half of what was
    asked for is a worse failure than one that adds something.
    """
    a, b = _normalise(answer), _normalise(expected)
    if not a or not b:
        return 1.0 if a == b else 0.0
    precisions, recalls = [], []
    for n in range(1, order + 1):
        got, want = _ngrams(a, n), _ngrams(b, n)
        if not got or not want:
            continue
        hits = sum(min(c, want.get(g, 0)) for g, c in got.items())
        precisions.append(hits / sum(got.values()))
        recalls.append(hits / sum(want.values()))
    if not precisions:
        return 0.0
    p = sum(precisions) / len(precisions)
    r = sum(recalls) / len(recalls)
    if p + r == 0:
        return 0.0
    return (1 + beta ** 2) * p * r / (beta ** 2 * p + r)


def _json_check(answer: str, expected: str) -> dict | None:
    """Does the answer parse, and does it say the same thing?

    Only asked when the expected answer is itself JSON -- otherwise every
    model would be marked invalid for writing prose, which is what it was
    asked for. Fenced code blocks are unwrapped first: a model told to emit
    JSON that wraps it in ```json has produced the right thing and been let
    down by its own politeness, and marking that invalid teaches nothing.
    """
    try:
        want = json.loads(expected)
    except (TypeError, ValueError):
        return None
    # A bare number or quoted word parses as JSON and is not what anybody
    # means by structured output. Without this, an expected answer of "42"
    # would mark every model that wrote "forty-two" as producing invalid JSON.
    if not isinstance(want, (dict, list)):
        return None
    text = (answer or "").strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.S)
    if fenced:
        text = fenced.group(1)
    try:
        got = json.loads(text)
    except (TypeError, ValueError):
        return {"json_valid": False, "json_match": False}
    return {"json_valid": True, "json_match": got == want}


def _expected_loss(host, prompt_text: str, expected: str, torch) -> float | None:
    """How surprised this model is by the answer you called correct.

    Scored on the answer only -- the prompt tokens are masked out with -100 --
    because including them would mostly measure how predictable the *question*
    was, which is a property of the prompt set and identical for every model
    being compared.
    """
    if not expected:
        return None
    tok, model = host.tok, host.model
    prompt_ids = tok(prompt_text, return_tensors="pt").input_ids
    full_ids = tok(prompt_text + expected, return_tensors="pt").input_ids
    if full_ids.shape[1] <= prompt_ids.shape[1]:
        return None
    full_ids = full_ids.to(host.device)
    labels = full_ids.clone()
    labels[:, :prompt_ids.shape[1]] = -100
    with torch.no_grad():
        return float(model(input_ids=full_ids, labels=labels).loss)


def _host_for(entry: dict, shared, ctx: Any):
    """The thing that will answer these prompts, and whether it is local.

    Local means the weights are on this machine, which is the only condition
    under which the loss on the expected answer can be measured at all.
    """
    if entry.get("source") == "api":
        from .generate_data import HostedModel
        return HostedModel({"connection": entry.get("connection"),
                            "model": entry.get("model")}, ctx), False
    return shared, True


def run(cfg: dict, ctx: Any) -> dict:
    items = list(cfg.get("items") or [])
    models = list(cfg.get("models") or [])
    if not items:
        raise ValueError("This prompt set has no prompts in it.")
    if not models:
        raise ValueError("No models were chosen to score.")

    params = {
        "max_new_tokens": int(cfg.get("max_new_tokens") or 200),
        # Greedy by default: see the module docstring. A comparison that moves
        # when nothing changed is not a comparison.
        "temperature": float(cfg.get("temperature") or 0.0),
        "top_k": int(cfg.get("top_k") or 50),
        "top_p": float(cfg.get("top_p") or 1.0),
    }
    # One prompt for everybody, when the scoring said so. Otherwise each model
    # is asked with the system prompt it was trained under, which the
    # controller has already put on each entry.
    override = (cfg.get("system_prompt") or "").strip()

    # Only built if something local is being scored: importing torch and
    # standing up a model host on a machine that is about to make four HTTPS
    # requests is a minute of nothing.
    shared = None
    if any(m.get("source", "run") != "api" for m in models):
        shared = inference.ModelHost(ctx.controller_url, ctx.runner_token,
                                     ctx.capabilities)

    total_work = len(items) * len(models)
    done = 0
    scores = []

    ctx.log("Scoring %d model%s on %d prompt%s. Every model gets the same "
            "prompts in the same order, which is the only thing that makes "
            "the results comparable."
            % (len(models), "" if len(models) == 1 else "s",
               len(items), "" if len(items) == 1 else "s"))
    if override:
        ctx.log("Every model is given the same system prompt, which overrides "
                "the one each was trained with.")
    ctx.progress(0, total_work, stage="evaluating")

    for entry in models:
        spec = dict(entry.get("spec") or {})
        job_id = spec.get("job_id") or entry.get("job_id") or ""
        source = entry.get("source") or "run"
        label = entry.get("name") or job_id
        system = (override or entry.get("system_prompt") or "").strip()
        ref = entry.get("ref") or ("job:" + job_id if job_id else label)
        ctx.log("--- %s%s" % (label, {"hub": "  (baseline, from the Hub)",
                                      "api": "  (baseline, over the network)"}
                              .get(source, "")))

        def record_failure(reason: str) -> None:
            scores.append({"model_job_id": job_id if source == "run" else "",
                           "ref": ref, "source": source, "name": label,
                           "system_prompt": system,
                           "metrics": {"error": reason[:300], "items": 0},
                           "items": []})

        try:
            host, local = _host_for(entry, shared, ctx)
            host.ensure_loaded(spec, lambda line: ctx.log("  %s" % line))
        except Exception as e:  # noqa: BLE001 - one bad model must not sink the rest
            # A model that cannot be loaded is recorded as such and the other
            # models are still scored. Failing the whole run would mean one
            # deleted artifact costs you the comparison of everything else.
            ctx.log("Could not load %s (%s). Skipping it; the other models are "
                    "still being scored." % (label, e), "error")
            record_failure(str(e))
            done += len(items)
            ctx.progress(done, total_work, stage="evaluating")
            continue

        results = []
        t_model = time.time()
        for item in items:
            if ctx.should_cancel():
                # Nothing half-scored is worth keeping: a comparison missing
                # two thirds of its prompts would sit in the table looking
                # exactly like a complete one.
                raise Cancelled()

            prompt = str(item.get("prompt") or "")
            expected = str(item.get("expected") or "")
            messages = ([{"role": "system", "content": system}] if system else []) \
                + [{"role": "user", "content": prompt}]

            t0 = time.time()
            out = host.generate(spec, messages, params, lambda *_: None,
                                lambda _l: None)
            answer = (out.get("text") or "").strip()

            row = {
                "prompt": prompt[:MAX_STORED_CHARS],
                "expected": expected[:MAX_STORED_CHARS],
                "answer": answer[:MAX_STORED_CHARS],
                "tokens": out.get("tokens"),
                "seconds": round(time.time() - t0, 2),
            }
            if expected:
                row["exact"] = _normalise(answer) == _normalise(expected)
                row["contains"] = _normalise(expected) in _normalise(answer)
                row["f1"] = round(_f1(answer, expected), 4)
                row["chrf"] = round(_chrf(answer, expected), 4)
                if js := _json_check(answer, expected):
                    row.update(js)
                if local:
                    _fmt, rendered = host.render(spec, messages)
                    loss = _expected_loss(host, rendered, expected, _torch())
                    row["expected_loss"] = round(loss, 5) if loss is not None else None
            results.append(row)

            done += 1
            ctx.progress(done, total_work, stage="evaluating")
            ctx.metric(done, {"prompts_scored": done,
                              "seconds_per_prompt": round(row["seconds"], 3)})

        metrics = _aggregate(results, time.time() - t_model)
        if not local:
            # Said in the metrics rather than left as an empty column. A
            # missing number in the column everything is ranked by reads as a
            # failure, and this one is a property of hosted APIs.
            metrics["loss_unavailable"] = (
                "%s is reached over the network, and the loss on the expected "
                "answer needs the model's own probabilities. It is scored on "
                "what it wrote." % label)
        scores.append({"model_job_id": job_id if source == "run" else "",
                       "ref": ref, "source": source, "name": label,
                       "system_prompt": system,
                       "metrics": metrics, "items": results})
        ctx.log("  %s" % _describe(metrics))

    ranked = _rank(scores)
    verdict = _verdict(scores, ranked)
    ctx.log(verdict)
    return {"kind": "evaluate", "eval_id": cfg.get("eval_id"),
            "verdict": verdict,
            # Whether the ranking this scoring produced is worth drawing as a
            # ranking. Sent as a fact rather than left for the UI to re-derive
            # from the verdict text, so the table and the log cannot end up
            # disagreeing about whether there was a winner.
            "decisive": _decisive(ranked),
            "ranked_by": ranked["key"] if ranked else None,
            "eval_name": cfg.get("eval_name"),
            "prompts": len(items), "scores": scores}


def _torch():
    import torch
    return torch


def _mean(rows: list[dict], key: str) -> float | None:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def _rate(rows: list[dict], key: str) -> float | None:
    vals = [bool(r[key]) for r in rows if r.get(key) is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def _aggregate(rows: list[dict], seconds: float) -> dict:
    scored = [r for r in rows if r.get("expected")]
    losses = [r["expected_loss"] for r in scored if r.get("expected_loss") is not None]
    return {
        "items": len(rows),
        "scored": len(scored),
        "exact": _rate(scored, "exact"),
        "contains": _rate(scored, "contains"),
        "f1": _mean(scored, "f1"),
        "chrf": _mean(scored, "chrf"),
        # Only present when the expected answers were JSON at all, so an
        # ordinary prompt set does not grow two empty columns.
        "json_valid": _rate(scored, "json_valid"),
        "json_match": _rate(scored, "json_match"),
        "expected_loss": round(sum(losses) / len(losses), 5) if losses else None,
        # Perplexity of the expected answer, which is the same number in the
        # units people actually have intuitions about: "1 in N" surprise.
        "expected_perplexity": round(math.exp(min(sum(losses) / len(losses), 20)), 2)
        if losses else None,
        "tokens_per_sec": round(
            sum(r.get("tokens") or 0 for r in rows) / max(seconds, 1e-6), 1),
        "seconds": round(seconds, 1),
    }


def _describe(m: dict) -> str:
    parts = []
    if m.get("expected_loss") is not None:
        parts.append("loss on the expected answer %.4f" % m["expected_loss"])
    if m.get("chrf") is not None:
        parts.append("character overlap %.0f%%" % (m["chrf"] * 100))
    if m.get("f1") is not None:
        parts.append("token overlap %.0f%%" % (m["f1"] * 100))
    if m.get("exact") is not None:
        parts.append("exact %.0f%%" % (m["exact"] * 100))
    if m.get("json_valid") is not None:
        parts.append("valid JSON %.0f%%" % (m["json_valid"] * 100))
    parts.append("%.0f tokens/s" % (m.get("tokens_per_sec") or 0))
    return ", ".join(parts)


# What a comparison is ranked by, in the order it is preferred. The loss on
# the expected answer is the measure to trust and is used whenever two models
# have one -- but a scoring that includes a hosted baseline may not have two,
# because no provider hands out probabilities. Falling back to what every
# model does have is better than declining to say anything at all, as long as
# the verdict says which measure it used.
RANKING = [
    ("expected_loss", "loss on the expected answers", True),
    ("chrf", "character overlap with the expected answers", False),
    ("f1", "token overlap with the expected answers", False),
]


def _separation(best: dict, worst: dict, key: str,
                lower_better: bool) -> tuple[float, float, int] | None:
    """How large the gap between two models is, next to the noise in it.

    Compared prompt by prompt rather than average against average, because
    every model answered the *same* prompts: some of them are simply harder
    than others, and pairing cancels that out instead of letting it swamp the
    difference being measured.

    Returns (mean difference, standard error of that mean, prompts compared),
    signed so that a positive mean means the better model really is ahead.
    """
    a = {i["prompt"]: i.get(key) for i in best.get("items") or []}
    b = {i["prompt"]: i.get(key) for i in worst.get("items") or []}
    diffs = [(b[k] - a[k]) if lower_better else (a[k] - b[k])
             for k in a if a.get(k) is not None and b.get(k) is not None]
    if len(diffs) < 3:
        return None
    n = len(diffs)
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    return mean, (var / n) ** 0.5, n


def _rank(scores: list[dict]) -> dict | None:
    """The measure that can rank these models, and what it says.

    Whichever measure covers *everybody* is preferred, even when a better one
    covers only some. Scoring a fine-tune against a hosted model would
    otherwise rank on the loss, which the hosted model cannot have, and
    announce a winner chosen from two of the three models on the page while
    the third sat above it in the table looking as though it had lost. When
    nothing covers everybody, the best available measure is used and the
    verdict says who is missing from it.

    None when nothing can rank them: fewer than two models with any measure at
    all, which happens when the prompts have no expected answers.
    """
    scored = [s for s in scores if (s["metrics"].get("scored") or 0)]
    for require_all in (True, False):
        for key, label, lower in RANKING:
            usable = [s for s in scored if s["metrics"].get(key) is not None]
            if len(usable) < 2 or (require_all and len(usable) != len(scored)):
                continue
            pick, anti = (min, max) if lower else (max, min)
            best = pick(usable, key=lambda s: s["metrics"][key])
            worst = anti(usable, key=lambda s: s["metrics"][key])
            return {
                "key": key, "label": label, "lower": lower,
                "best": best, "worst": worst,
                "sep": _separation(best, worst, key, lower),
                "excluded": [s["name"] for s in scored if s not in usable],
            }
    return None


def _decisive(ranked: dict | None) -> bool:
    """Did this scoring actually separate the models it compared?

    False whenever the difference between best and worst is smaller than the
    spread between prompts -- which is the usual case for two models that
    differ by a little more training, and exactly when a table drawing a
    winner would be inventing one.
    """
    if not ranked or not ranked["sep"]:
        return False
    mean, se, _n = ranked["sep"]
    return mean >= 2 * se


def _verdict(scores: list[dict], ranked: dict | None) -> str:
    """Say which one won and by how much -- or that the prompts cannot tell.

    The second half of that is the part worth writing carefully. A difference
    of a tenth of a nat, averaged over four prompts, is not a result; it is
    the spread between four prompts. Reporting it as a winner would be the
    single easiest way for this feature to mislead somebody, so the size of
    the difference is always judged against the noise in it rather than
    against a threshold picked in advance.
    """
    if not ranked:
        if len(scores) == 1:
            return ("Scored. Run the same prompt set against another model to "
                    "get a comparison -- a single set of numbers has nothing "
                    "to be better or worse than. The model this one was "
                    "trained from is the usual answer.")
        return ("Scored. Without expected answers there is no measure that can "
                "rank these, only the text each one produced. Add expected "
                "answers to the prompt set to get a number.")

    key, lower = ranked["key"], ranked["lower"]
    best, worst = ranked["best"], ranked["worst"]
    fmt = (lambda v: "%.4f" % v) if lower else (lambda v: "%.0f%%" % (v * 100))
    head = "%s %s: %s (%s, against %s for %s)." % (
        "Lowest" if lower else "Highest", ranked["label"], best["name"],
        fmt(best["metrics"][key]), fmt(worst["metrics"][key]), worst["name"])
    if key != "expected_loss":
        head += (" Ranked on what the models wrote rather than on the loss, "
                 "which a model reached over the network cannot be measured "
                 "on -- no provider exposes the probabilities it needs.")
    if ranked["excluded"]:
        # Never silently. A model on the page that took no part in the
        # ranking, with a winner announced above it, is the single most
        # misleading thing this table could do.
        head += (" %s %s not in this ranking: there is no %s for %s."
                 % (", ".join(ranked["excluded"]),
                    "is" if len(ranked["excluded"]) == 1 else "are",
                    ranked["label"],
                    "it" if len(ranked["excluded"]) == 1 else "them"))

    if not ranked["sep"]:
        return (head + " With so few prompts that is a difference between two "
                "numbers, not a result. Add prompts before believing it.")
    mean, se, n = ranked["sep"]
    # Two standard errors is roughly the 95% mark. Below it, the difference
    # between these models is smaller than the difference between prompts.
    if mean < 2 * se:
        return (head + " But prompt by prompt the gap is %.3f give or take "
                "%.3f across %d prompts, which is smaller than the spread "
                "between the prompts themselves. This set cannot tell these "
                "two apart -- more prompts, or harder ones, would."
                % (mean, se, n))
    tail = (" Lower is better: it means the model found the answer you called "
            "correct less surprising." if lower else
            " Higher is better: more of the expected answer turned up in what "
            "the model wrote.")
    return (head + " Prompt by prompt the gap is %.3f give or take %.3f "
            "across %d prompts, so it is a real difference and not noise."
            % (mean, se, n)) + tail
