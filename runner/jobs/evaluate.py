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

Every generation is greedy by default. Sampling makes a model score
differently on two identical runs, and a comparison whose noise is larger than
its signal is worse than no comparison.
"""
from __future__ import annotations

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


def run(cfg: dict, ctx: Any) -> dict:
    import torch

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
    system = (cfg.get("system_prompt") or "").strip()

    host = inference.ModelHost(ctx.controller_url, ctx.runner_token,
                               ctx.capabilities)
    total_work = len(items) * len(models)
    done = 0
    scores = []

    ctx.log("Scoring %d model%s on %d prompt%s. Every model gets the same "
            "prompts in the same order, which is the only thing that makes "
            "the results comparable."
            % (len(models), "" if len(models) == 1 else "s",
               len(items), "" if len(items) == 1 else "s"))
    ctx.progress(0, total_work, stage="evaluating")

    for model_spec in models:
        spec = dict(model_spec.get("spec") or {})
        job_id = spec.get("job_id") or model_spec.get("job_id")
        label = model_spec.get("name") or job_id
        ctx.log("--- %s" % label)

        try:
            host.ensure_loaded(spec, lambda line: ctx.log("  %s" % line))
        except Exception as e:  # noqa: BLE001 - one bad model must not sink the rest
            # A model that cannot be loaded is recorded as such and the other
            # models are still scored. Failing the whole run would mean one
            # deleted artifact costs you the comparison of everything else.
            ctx.log("Could not load %s (%s). Skipping it; the other models are "
                    "still being scored." % (label, e), "error")
            scores.append({"model_job_id": job_id, "name": label,
                           "metrics": {"error": str(e)[:300], "items": 0},
                           "items": []})
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
            out = host.generate(spec, messages, params, lambda _d: None,
                                lambda _l: None)
            answer = (out.get("text") or "").strip()

            _fmt, rendered = host.render(spec, messages)
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
                loss = _expected_loss(host, rendered, expected, torch)
                row["expected_loss"] = round(loss, 5) if loss is not None else None
            results.append(row)

            done += 1
            ctx.progress(done, total_work, stage="evaluating")
            ctx.metric(done, {"prompts_scored": done,
                              "seconds_per_prompt": round(row["seconds"], 3)})

        metrics = _aggregate(results, time.time() - t_model)
        scores.append({"model_job_id": job_id, "name": label,
                       "metrics": metrics, "items": results})
        ctx.log("  %s" % _describe(metrics))

    verdict = _verdict(scores)
    ctx.log(verdict)
    return {"kind": "evaluate", "eval_id": cfg.get("eval_id"),
            "verdict": verdict,
            # Whether the ranking this scoring produced is worth drawing as a
            # ranking. Sent as a fact rather than left for the UI to re-derive
            # from the verdict text, so the table and the log cannot end up
            # disagreeing about whether there was a winner.
            "decisive": _decisive(scores),
            "eval_name": cfg.get("eval_name"),
            "prompts": len(items), "scores": scores}


def _aggregate(rows: list[dict], seconds: float) -> dict:
    scored = [r for r in rows if r.get("expected")]
    losses = [r["expected_loss"] for r in scored if r.get("expected_loss") is not None]
    return {
        "items": len(rows),
        "scored": len(scored),
        "exact": round(sum(1 for r in scored if r.get("exact")) / len(scored), 4)
        if scored else None,
        "contains": round(sum(1 for r in scored if r.get("contains")) / len(scored), 4)
        if scored else None,
        "f1": round(sum(r.get("f1") or 0.0 for r in scored) / len(scored), 4)
        if scored else None,
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
    if m.get("f1") is not None:
        parts.append("token overlap %.0f%%" % (m["f1"] * 100))
    if m.get("exact") is not None:
        parts.append("exact %.0f%%" % (m["exact"] * 100))
    parts.append("%.0f tokens/s" % (m.get("tokens_per_sec") or 0))
    return ", ".join(parts)


def _separation(best: dict, worst: dict) -> tuple[float, float, int] | None:
    """How large the gap between two models is, next to the noise in it.

    Compared prompt by prompt rather than average against average, because
    every model answered the *same* prompts: some of them are simply harder
    than others, and pairing cancels that out instead of letting it swamp the
    difference being measured.

    Returns (mean difference, standard error of that mean, prompts compared).
    """
    a = {i["prompt"]: i.get("expected_loss") for i in best.get("items") or []}
    b = {i["prompt"]: i.get("expected_loss") for i in worst.get("items") or []}
    diffs = [b[k] - a[k] for k in a
             if a.get(k) is not None and b.get(k) is not None]
    if len(diffs) < 3:
        return None
    n = len(diffs)
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    return mean, (var / n) ** 0.5, n


def _decisive(scores: list[dict]) -> bool:
    """Did this scoring actually separate the models it compared?

    False whenever the difference between best and worst is smaller than the
    spread between prompts -- which is the usual case for two models that
    differ by a little more training, and exactly when a table drawing a
    winner would be inventing one.
    """
    usable = [s for s in scores if s["metrics"].get("expected_loss") is not None]
    if len(usable) < 2:
        return False
    best = min(usable, key=lambda s: s["metrics"]["expected_loss"])
    worst = max(usable, key=lambda s: s["metrics"]["expected_loss"])
    sep = _separation(best, worst)
    return bool(sep and sep[0] >= 2 * sep[1])


def _verdict(scores: list[dict]) -> str:
    """Say which one won and by how much -- or that the prompts cannot tell.

    The second half of that is the part worth writing carefully. A difference
    of a tenth of a nat, averaged over four prompts, is not a result; it is
    the spread between four prompts. Reporting it as a winner would be the
    single easiest way for this feature to mislead somebody, so the size of
    the difference is always judged against the noise in it rather than
    against a threshold picked in advance.
    """
    usable = [s for s in scores if s["metrics"].get("expected_loss") is not None]
    if len(usable) < 2:
        if len(scores) == 1:
            return ("Scored. Run the same prompt set against another model to "
                    "get a comparison -- a single set of numbers has nothing "
                    "to be better or worse than.")
        return ("Scored. Without expected answers there is no measure that can "
                "rank these, only the text each one produced. Add expected "
                "answers to the prompt set to get a number.")

    best = min(usable, key=lambda s: s["metrics"]["expected_loss"])
    worst = max(usable, key=lambda s: s["metrics"]["expected_loss"])
    gap = worst["metrics"]["expected_loss"] - best["metrics"]["expected_loss"]
    head = ("Lowest loss on the expected answers: %s (%.4f, against %.4f for "
            "%s)." % (best["name"], best["metrics"]["expected_loss"],
                      worst["metrics"]["expected_loss"], worst["name"]))

    sep = _separation(best, worst)
    if sep is None:
        return (head + " With so few prompts that is a difference between two "
                "numbers, not a result. Add prompts before believing it.")
    mean, se, n = sep
    # Two standard errors is roughly the 95% mark. Below it, the difference
    # between these models is smaller than the difference between prompts.
    if mean < 2 * se:
        return (head + " But prompt by prompt the gap is %.3f give or take "
                "%.3f across %d prompts, which is smaller than the spread "
                "between the prompts themselves. This set cannot tell these "
                "two apart -- more prompts, or harder ones, would."
                % (mean, se, n))
    return (head + " Prompt by prompt the gap is %.3f give or take %.3f "
            "across %d prompts, so it is a real difference and not noise. "
            "Lower is better: it means the model found the answer you called "
            "correct less surprising." % (mean, se, n))
