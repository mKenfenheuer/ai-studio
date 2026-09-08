"""Reading a finished run, so the user does not have to.

The app has always shown the curve and left the interpretation to whoever was
looking at it. That is exactly backwards for the audience it is built for: the
shapes that matter -- a held-out loss turning up while the training loss keeps
falling, a loss that never moved, a spike the run never recovered from, a
mixture of experts that quietly collapsed onto one -- are obvious once you
know them and invisible until you do.

Everything here is computed from measurements the run already recorded. There
is no model of quality and no score out of ten, because neither would be
honest: a run can be textbook-healthy and still produce something useless for
your task, and the only thing that answers *that* is a prompt set.

Each finding says what was seen, what it means, and what to change. The last
of those is the part that makes it worth writing at all.
"""
from __future__ import annotations

import statistics

# A held-out loss this much above its best is a rise, not a wobble.
OVERFIT_RATIO = 1.05
# Tokens per parameter. Twenty is a full Chinchilla-style budget; below five a
# model is heavily undertrained whatever its loss curve looks like.
FULL_BUDGET = 20.0
UNDERTRAINED = 5.0


def _series(metrics: list[dict], key: str) -> list[tuple[int, float]]:
    return [(m["step"], m[key]) for m in metrics
            if m.get(key) is not None and isinstance(m[key], (int, float))]


def _finding(level: str, title: str, saw: str, means: str, do: str,
             change: dict | None = None) -> dict:
    """One thing this run says about itself.

    `change` is the advice in `do`, as settings rather than as prose. It is
    what lets the run page offer to start a corrected copy instead of leaving
    somebody to read the paragraph, work out which field it means, find the
    rerun form and type the number in -- which is the step at which most
    people give up and run the same thing again.
    """
    out = {"level": level, "title": title, "saw": saw, "means": means, "do": do}
    if change:
        out["change"] = change
    return out


def report(job: dict, metrics: list[dict]) -> dict:
    """What this run says about itself."""
    summary = job.get("summary") or {}
    cfg = job.get("config") or {}
    scratch = job.get("kind") == "pretrain_llm"
    loss = _series(metrics, "loss")
    val = _series(metrics, "val_loss")
    findings: list[dict] = []

    if job.get("status") == "failed":
        return {"status": "failed",
                "findings": _failure(str(job.get("error") or ""), cfg),
                "facts": _facts(job, metrics)}

    if len(loss) < 3:
        return {"status": job.get("status"), "findings": [], "facts": _facts(job, metrics)}

    findings += _learning(loss, val, summary, cfg)
    findings += _overfitting(val, loss, summary, scratch)
    findings += _stability(loss, metrics)
    findings += _budget(summary, scratch, cfg)
    findings += _experts(summary)
    findings += _hardware(metrics, job)
    findings += _how_it_ended(summary, job)

    # Nothing wrong is itself worth saying, and saying it plainly stops the
    # absence of warnings reading as the absence of a check.
    if not any(f["level"] in ("warn", "error") for f in findings):
        findings.insert(0, _finding(
            "ok", "Nothing looks wrong with this run",
            "The loss fell and stayed down, and nothing in the measurements "
            "suggests it was fighting its settings.",
            "A healthy run is not the same as a useful model. It means the "
            "training worked, not that the result does what you want.",
            "Put it to a prompt set to find out whether it is any good at "
            "your actual task."))
    return {"status": job.get("status"), "findings": findings,
            "facts": _facts(job, metrics)}


def _scaled_lr(cfg: dict | None, factor: float) -> dict | None:
    """The same learning rate, times something. None when there is none to scale."""
    try:
        lr = float((cfg or {}).get("learning_rate") or 0)
    except (TypeError, ValueError):
        return None
    return {"learning_rate": float("%.2g" % (lr * factor))} if lr else None


def _failure(error: str, cfg: dict) -> list[dict]:
    """Why it stopped, and what to change so it does not stop again.

    The log has the traceback. What was missing was the step after reading it:
    the advice was a paragraph, and turning a paragraph into a corrected run
    meant working out which field it meant, finding the rerun form, and typing
    a number. Most people ran the same thing again instead.
    """
    low = error.lower()
    said = error or "It stopped with an error."

    # Out of memory, which is far and away the most common failure and the one
    # with the most mechanical answer.
    if "out of memory" in low or "cuda error" in low or "hip out of" in low \
            or "alloc" in low and "fail" in low:
        change: dict = {}
        why = []
        if cfg.get("quantization") != "4bit":
            change["quantization"] = "4bit"
            why.append("loading the frozen base in 4-bit")
        seq = int(cfg.get("max_seq_len") or 0)
        if seq > 512:
            change["max_seq_len"] = max(512, seq // 2)
            why.append("halving the context length")
        batch = int(cfg.get("batch_size") or 1)
        if batch > 1:
            change["batch_size"] = max(1, batch // 2)
            change["grad_accum"] = int(cfg.get("grad_accum") or 1) * 2
            why.append("halving the batch and doubling the accumulation, "
                       "which keeps the effective batch the same")
        if not cfg.get("gradient_checkpointing"):
            change["gradient_checkpointing"] = True
            why.append("recomputing activations instead of storing them")
        return [_finding(
            "error", "It ran out of memory on the card", said,
            "The model, its optimiser state and the activations for one batch "
            "did not fit together. Nothing about the data or the settings was "
            "wrong in itself -- there was simply not room.",
            "Try " + ("; ".join(why) if why else "a smaller model") + ".",
            change or None)]

    if "no rows" in low or "has no rows in a split" in low:
        return [_finding(
            "error", "It could not find the rows to train on", said,
            "The split named on the run does not exist in that dataset, or is "
            "empty.",
            "Choose a split the dataset actually has.")]

    if "trust_remote_code" in low:
        return [_finding(
            "error", "That model needs code this studio will not run", said,
            "The model ships custom Python that transformers would execute to "
            "load it. This studio does not run model code it did not ship.",
            "Choose a model with a standard architecture.")]

    if "401" in low or "403" in low or "gated" in low or "authoriz" in low:
        return [_finding(
            "error", "It was not allowed to download that", said,
            "Gated models -- Llama and Gemma among them -- need a Hugging Face "
            "token belonging to an account that has accepted their licence.",
            "Connect your Hugging Face account, and accept the model's terms "
            "on its page.")]

    return [_finding(
        "error", "This run failed", said,
        "Nothing was produced, so there is nothing to diagnose in the numbers.",
        "The log below has the full error. If there is a checkpoint, fixing "
        "the cause and starting it again carries on from there rather than "
        "from the beginning.")]


def _learning(loss, val, summary, cfg=None) -> list[dict]:
    first = summary.get("initial_loss") or loss[0][1]
    last = summary.get("final_loss") or loss[-1][1]
    if first <= 0:
        return []
    drop = 1 - (last / first)

    # Worse than it started is its own failure, and a specific one. Reporting
    # it as "barely learned" would send the reader looking for a learning rate
    # that is too low when the cause is almost always the opposite.
    if last > first:
        return [_finding(
            "error", "It ended worse than it started",
            "The loss went from %.4f up to %.4f." % (first, last),
            "The model was not merely failing to learn -- it was being pushed "
            "away from anything useful. Nearly always a learning rate high "
            "enough to destroy the weights faster than training repairs them.",
            "Cut the learning rate by ten and run a short test. If the loss "
            "still rises, the data is being read wrongly rather than the "
            "settings being wrong.",
            _scaled_lr(cfg, 0.1))]

    if drop < 0.02:
        return [_finding(
            "error", "It barely learned anything",
            "The loss went from %.4f to %.4f -- a %.1f%% change over %d steps."
            % (first, last, drop * 100, loss[-1][0]),
            "Something is stopping the model from learning at all. The usual "
            "causes are a learning rate far too low, a dataset being read "
            "from the wrong column so every example is empty, or a frozen "
            "model with nothing trainable in it.",
            "Check the example training text on the run's settings: if it is "
            "blank or nonsense, the column mapping is wrong. If it looks "
            "right, raise the learning rate tenfold and try a short run.",
            _scaled_lr(cfg, 10))]
    if drop < 0.15:
        return [_finding(
            "warn", "It learned very little",
            "The loss fell only %.0f%%, from %.4f to %.4f." % (drop * 100, first, last),
            "The model moved in the right direction but nowhere near as far "
            "as a working run does.",
            "A higher learning rate or a longer run is the usual fix. If the "
            "loss was still falling steadily at the end, it simply needed "
            "more steps.")]

    # Still falling steeply at the end means it was cut off, not finished --
    # but judged on the held-out loss where there is one. A training loss that
    # is still dropping while the held-out loss climbs is the model
    # memorising, and telling somebody to train that for longer would be the
    # worst advice this report could give.
    series = val if len(val) >= 6 else loss
    tail = [v for _s, v in series[-max(6, len(series) // 10):]]
    if len(tail) >= 6:
        early, late = statistics.fmean(tail[:len(tail) // 2]), statistics.fmean(tail[len(tail) // 2:])
        if early > 0 and (early - late) / early > 0.03:
            return [_finding(
                "warn", "It was still improving when it ran out of steps",
                "Over the last %d readings the loss was still falling by "
                "%.1f%% per half." % (len(tail), (early - late) / early * 100),
                "The schedule ended before the model stopped getting better. "
                "Whatever this run produced, a longer one would produce more.",
                "Train it further -- the follow-on run starts from these "
                "weights rather than from noise, so the extra steps are the "
                "only cost.")]
    return []


def _overfitting(val, loss, summary, scratch) -> list[dict]:
    if len(val) < 3:
        return [_finding(
            "warn", "Nothing was held back to measure on",
            "This run has no held-out loss.",
            "The only number available is the training loss, which falls just "
            "as happily when a model memorises its examples as when it learns "
            "from them. There is no way to tell those apart here.",
            "Use a larger dataset, or split the one you have, so some of it "
            "can be kept back. Without that, no run can be compared with "
            "another.")] if not scratch else []

    best = min(v for _s, v in val)
    best_step = next(s for s, v in val if v == best)
    final = val[-1][1]
    findings = []
    if final > best * OVERFIT_RATIO:
        findings.append(_finding(
            "warn", "It started memorising",
            "The held-out loss reached %.4f at step %d and finished at %.4f, "
            "%.0f%% worse, while the training loss kept falling."
            % (best, best_step, final, (final / best - 1) * 100),
            "Past step %d the model was learning the specific examples in "
            "front of it rather than the pattern behind them. Everything "
            "after that point made it worse at anything it had not already "
            "seen." % best_step,
            "%s More data helps most; fewer passes over the same data helps "
            "immediately."
            % ("The model kept is the one from step %d, so this run is not "
               "damaged by it." % summary["kept_from_step"]
               if summary.get("kept_from_step") else
               "Turn on 'stop when it stops improving' so the best version is "
               "the one kept.")))
    return findings


def _spikes(loss, window: int = 12) -> list[tuple[int, float, float]]:
    """Points where the loss jumped well above where it had just been.

    Compared against a trailing window rather than against the run's own
    median, which was the first thing tried and was wrong: a loss decaying
    normally from 3.0 to 0.05 spends its first third above twice its median,
    so every healthy run was reported as spiking. A spike is a departure from
    the local level, and the local level is what has to be measured.
    """
    out = []
    for i in range(window, len(loss)):
        recent = [v for _s, v in loss[i - window:i]]
        level = statistics.median(recent)
        step, value = loss[i]
        if level > 0 and value > level * 2.0 and value - level > 0.5:
            out.append((step, value, level))
    return out


def _stability(loss, metrics) -> list[dict]:
    spikes = _spikes(loss)
    findings = []
    if spikes:
        step, value, level = spikes[0]
        # Recovered means it came back to where it had been *before* the
        # spike, not merely below the whole run's median -- a run that spikes
        # and then flatlines high still sits under a median dominated by its
        # own early values.
        recovered = loss[-1][1] <= level * 1.1
        findings.append(_finding(
            "warn" if recovered else "error",
            "The loss spiked" + ("" if recovered else " and never recovered"),
            "It jumped to %.4f at step %d, from %.4f just before."
            % (value, step, level),
            "A spike this size is almost always a learning rate too high for "
            "this model's width, or one batch of unusual data. "
            + ("It came back down to where it had been, so the run "
               "survived it."
               if recovered else
               "It settled at %.4f rather than the %.4f it was at before, "
               "which means the weights were damaged and everything after "
               "the spike was spent recovering rather than learning."
               % (loss[-1][1], level)),
            "Halve the learning rate. If it spikes at the same step every "
            "time, the cause is in the data at that point, not the rate."))

    grads = [v for _s, v in _series(metrics, "grad_norm")]
    if grads and max(grads) > 100 * (statistics.median(grads) or 1):
        findings.append(_finding(
            "warn", "The gradients blew up at least once",
            "The largest gradient was %.1f against a typical %.2f."
            % (max(grads), statistics.median(grads)),
            "Gradient clipping caught it, which is what clipping is for, but "
            "an update that large means the model was briefly far outside "
            "the range its settings assume.",
            "A lower learning rate, or a longer warmup, makes this stop "
            "happening rather than being caught."))
    return findings


def _budget(summary, scratch, cfg) -> list[dict]:
    if not scratch:
        rows = summary.get("held_out_rows")
        if rows is not None and 0 < rows < 20:
            return [_finding(
                "warn", "Very few examples were held back",
                "Only %d examples were kept back to measure on." % rows,
                "A held-out loss computed on a handful of examples moves "
                "around a lot on its own, which makes it a poor guide to "
                "whether the model improved.",
                "A few hundred held-out examples is where the number starts "
                "being steady enough to compare two runs by.")]
        return []

    ratio = summary.get("tokens_per_param")
    if ratio is None:
        return []
    if ratio < UNDERTRAINED:
        return [_finding(
            "warn", "This model is heavily undertrained",
            "It saw %.1f tokens for every parameter it has, against about %d "
            "for a full budget." % (ratio, FULL_BUDGET),
            "A model of this size has far more capacity than this much text "
            "can fill. It will produce words and grammar but not much sense, "
            "and that is a property of the budget rather than of the run.",
            "Either give it much more text, or build a smaller model -- a "
            "small model trained fully beats a large one trained briefly, "
            "every time.")]
    if ratio < FULL_BUDGET:
        return [_finding(
            "ok", "Undertrained, but respectably so",
            "It saw %.1f tokens per parameter; a full budget is about %d."
            % (ratio, FULL_BUDGET),
            "There is real capacity left unfilled, but enough text went "
            "through it to be coherent.",
            "Training it further on more text is the cheapest improvement "
            "available -- it starts from these weights, not from noise.")]
    return []


def _experts(summary) -> list[dict]:
    experts = summary.get("experts")
    share = summary.get("worst_expert_share")
    if not experts or not share:
        return []
    even = 1.0 / experts
    if share > even * 2.5:
        return [_finding(
            "error", "The router collapsed onto a few experts",
            "At its worst, one expert of %d took %.0f%% of the tokens, where "
            "an even split is %.0f%%." % (experts, share * 100, even * 100),
            "The experts that were starved never learned anything. The model "
            "holds the memory cost of %d experts and the ability of far "
            "fewer -- the loss curve will not show this, because a collapsed "
            "mixture still learns." % experts,
            "Raise the load-balancing coefficient, or use fewer experts. A "
            "dense model of the same active size would be smaller, faster "
            "and no worse.")]
    if share > even * 1.5:
        return [_finding(
            "warn", "The router had favourites",
            "The busiest of %d experts took %.0f%% of the tokens against "
            "%.0f%% for an even split." % (experts, share * 100, even * 100),
            "Some imbalance is normal and expected. This much means part of "
            "the model is doing less work than it costs to keep.",
            "Worth watching rather than fixing. If it grows on a longer run, "
            "raise the load-balancing coefficient.")]
    return []


def _hardware(metrics, job) -> list[dict]:
    vram = [v for _s, v in _series(metrics, "vram_gb")]
    runner = job.get("runner") or {}
    total = (runner.get("capabilities") or {}).get("vram_gb")
    if not vram or not total:
        return []
    peak = max(vram)
    if peak < total * 0.45:
        return [_finding(
            "ok", "There was memory to spare",
            "It peaked at %.1f GB of the %.1f GB on this card." % (peak, total),
            "The run was sized conservatively. Memory left unused is capacity "
            "that could have gone into a larger batch or a larger model.",
            "A bigger batch makes each step do more work for the same "
            "overhead, which usually means more tokens per second.")]
    return []


def _how_it_ended(summary, job) -> list[dict]:
    out = []
    if summary.get("early_stopped"):
        out.append(_finding(
            "ok", "It stopped when it stopped improving",
            "The run ended at step %s of %s planned."
            % (summary.get("steps"), summary.get("planned_steps")),
            "The held-out loss had gone several checks without getting "
            "better, so the remaining steps would have cost time and made "
            "the model worse.",
            "Nothing to do. If you want it to run longer before giving up, "
            "raise the patience setting."))
    if summary.get("kept_from_step"):
        out.append(_finding(
            "ok", "The model kept is not the last one",
            "The weights saved are from step %d, not step %s."
            % (summary["kept_from_step"], summary.get("steps")),
            "That step scored better on held-out data than the end of the "
            "run did.",
            "Nothing to do -- this is the better model."))
    if job.get("status") == "cancelled" and job.get("artifacts"):
        out.append(_finding(
            "warn", "This run was stopped by hand",
            "It completed %s of %s planned steps."
            % (summary.get("steps"), summary.get("planned_steps")),
            "The model is real but had less practice than the plan called "
            "for, and its learning rate never finished decaying -- so it is "
            "a little rougher than the same run taken to the end.",
            "Train it further to finish the job; the follow-on run starts "
            "from these weights."))
    return out


def _facts(job: dict, metrics: list[dict]) -> list[dict]:
    """The handful of numbers worth putting next to the findings."""
    s = job.get("summary") or {}
    out = []

    def add(label, value, note=""):
        if value not in (None, ""):
            out.append({"label": label, "value": value, "note": note})

    add("Held-out loss", "%.4f" % s["best_val_loss"] if s.get("best_val_loss")
        else None, "the number to compare against other runs")
    add("Training loss", "%.4f → %.4f" % (s["initial_loss"], s["final_loss"])
        if s.get("initial_loss") and s.get("final_loss") else None)
    add("Steps", "%s of %s" % (f"{s['steps']:,}", f"{s.get('planned_steps') or s['steps']:,}")
        if s.get("steps") else None)
    add("Text seen", "%s tokens" % f"{s['tokens_seen']:,}" if s.get("tokens_seen") else None,
        "%.1f per parameter" % s["tokens_per_param"] if s.get("tokens_per_param") else "")
    add("Took", _duration(s.get("duration_s")))
    speeds = [v for _s, v in _series(metrics, "tokens_per_sec")]
    add("Speed", "%s tokens/s" % f"{int(statistics.fmean(speeds)):,}" if speeds else None,
        "average over the run")
    vram = [v for _s, v in _series(metrics, "vram_gb")]
    add("Peak GPU memory", "%.1f GB" % max(vram) if vram else None)
    return out


def _duration(seconds) -> str | None:
    if not seconds:
        return None
    seconds = int(seconds)
    if seconds < 60:
        return "%ds" % seconds
    if seconds < 3600:
        return "%dm %ds" % (seconds // 60, seconds % 60)
    return "%dh %dm" % (seconds // 3600, (seconds % 3600) // 60)
