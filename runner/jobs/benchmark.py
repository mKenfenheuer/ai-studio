"""Scoring a model on a published benchmark, the way the publishers score it.

The catalogue and the reasoning behind it are in controller/benchmarks.py.
This is the half that runs on the machine with the weights.

Two things happen here that do not happen anywhere else in the studio:

**The dataset is fetched by the runner.** Every other job is handed its data
by the controller. A benchmark is a public dataset of a known shape, and
`load_dataset` is how the tools that produce the published numbers get it --
so it is how this gets it too, rather than through a paging import that would
have to sample it and would sample it badly (MMLU is stored subject by
subject, so the first four hundred rows are four hundred questions about
abstract algebra).

**The model is not asked to write anything**, for four of the five. A
multiple-choice benchmark is decided by which answer the model finds most
likely, scored directly off its probabilities. That is what makes it work on
a model far too small to follow an instruction, and it is the reason a hosted
model cannot be scored on one at all: no provider hands out the numbers it
would need.
"""
from __future__ import annotations

import random
import re
import time
from typing import Any, Callable

from .lora_llm import Cancelled

# Answers are stored so they can be read afterwards, and a benchmark run is
# thousands of rows. Only a sample of them is kept -- enough to see what the
# model actually did wrong, not enough to put ten megabytes of JSON in the
# database for every scoring.
KEEP_ROWS = 60
MAX_STORED_CHARS = 800

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


# --------------------------------------------------------------- loading

def _load_split(bench: dict, split: str, ctx: Any):
    try:
        from datasets import load_dataset
    except ImportError as e:
        # Said in the words of the thing that is missing rather than as a bare
        # ImportError. This machine can hold a model and score a prompt set;
        # it cannot fetch a Hub dataset, which is a property of how it was
        # built and not of the benchmark being asked for.
        raise ValueError(
            "This machine has no `datasets` library, so it cannot fetch a "
            "benchmark from the Hub. The GPU runner images ship one; a "
            "machine set up by hand needs `pip install datasets`.") from e
    config = bench.get("config") or None
    try:
        return load_dataset(bench["dataset"], config, split=split)
    except Exception as e:  # noqa: BLE001 - turned into something readable
        raise ValueError(
            "Could not load %s (%s, split %s): %s. A gated dataset needs a "
            "Hugging Face token on this machine; a renamed one needs the "
            "benchmark's entry updating."
            % (bench["label"], bench["dataset"], split, e)) from e


def _hellaswag(row: dict) -> tuple[str, list[str]]:
    """The preprocessing every published HellaSwag number is computed after.

    Not decoration. The dataset's context arrives with markup in it -- "[header]
    Then, the man" -- and scoring the raw string moves the result by a couple
    of points against every number anybody has published. Copied in behaviour
    from the harness for that reason alone.
    """
    def clean(text: str) -> str:
        text = (text or "").strip()
        text = text.replace(" [title]", ". ")
        text = re.sub(r"\[.*?\]", "", text)
        return text.replace("  ", " ")

    if row.get("ctx_a") is not None:
        ctx_text = "%s: %s %s" % (row.get("activity_label") or "",
                                  row.get("ctx_a") or "",
                                  (row.get("ctx_b") or "").capitalize())
    else:
        ctx_text = "%s: %s" % (row.get("activity_label") or "",
                               row.get("ctx") or "")
    return clean(ctx_text), [clean(e) for e in (row.get("endings") or [])]


def _answer_index(value: Any, choices: list[str], labels: list | None) -> int:
    """Which choice is the right one, whichever way the dataset says so.

    Three conventions among five datasets: a list of labels to look the answer
    up in (ARC, whose keys are "A".."D" in some rows and "1".."4" in others),
    a zero-based index as a string (HellaSwag), and a plain integer (MMLU).
    Guessing wrong here does not fail -- it silently scores every model
    against the wrong answers -- so each is handled rather than coerced.
    """
    if isinstance(value, bool):
        raise ValueError("the answer is a boolean")
    if isinstance(value, int):
        return value
    text = str(value if value is not None else "").strip()
    if labels:
        flat = [str(x).strip() for x in labels]
        if text in flat:
            return flat.index(text)
    if text.isdigit():
        return int(text)
    if len(text) == 1 and text.upper() in LETTERS:
        return LETTERS.index(text.upper())
    raise ValueError("cannot tell which answer is correct from %r" % value)


def _items(bench: dict, rows, ctx: Any) -> list[dict]:
    """One benchmark row, normalised: a question, its choices, the right one."""
    f = bench["fields"]
    out = []
    bad = 0
    for row in rows:
        try:
            if bench["protocol"] == "generate_extract":
                # No choices at all: the model writes the answer. Handled
                # before the multiple-choice mapping rather than after it,
                # because that mapping asks for a `choices` column this kind
                # of benchmark does not have and never did.
                out.append({"question": str(row.get(f["question"]) or "").strip(),
                            "answer": str(row.get(f["answer"]) or "")})
                continue
            if bench.get("preprocess") == "hellaswag":
                question, choices = _hellaswag(row)
                labels = None
            else:
                question = str(row.get(f["question"]) or "").strip()
                raw = row.get(f["choices"])
                labels = None
                if isinstance(raw, dict):
                    # ARC stores {"text": [...], "label": [...]}.
                    labels = list(raw.get("label") or [])
                    raw = raw.get("text")
                choices = [str(c) for c in (raw or [])]
            if not question or len(choices) < 2:
                bad += 1
                continue
            index = _answer_index(row.get(f["answer"]), choices, labels)
            if not 0 <= index < len(choices):
                bad += 1
                continue
            out.append({"question": question, "choices": choices,
                        "answer_index": index,
                        "group": str(row.get(f.get("group") or "") or "")})
        except (ValueError, TypeError, KeyError):
            bad += 1
    if bad:
        ctx.log("%d row%s of this dataset could not be read and were skipped."
                % (bad, "" if bad == 1 else "s"), "warn")
    if not out:
        raise ValueError(
            "None of the rows of %s could be read. Its columns may have been "
            "renamed since this benchmark was described." % bench["label"])
    return out


def prepare(bench: dict, recipe: dict, ctx: Any) -> tuple[list[dict], list[dict]]:
    """The questions to ask, and the worked examples to put in front of them."""
    ctx.log("Fetching %s from %s. This is downloaded and cached on this "
            "machine, not sent from the controller."
            % (bench["label"], bench["dataset"]))
    rows = _load_split(bench, bench["split"], ctx)
    items = _items(bench, rows, ctx)

    sample = int(recipe.get("sample") or 0)
    if sample and sample < len(items):
        # Shuffled with the recorded seed, so re-running the same recipe next
        # month asks the same questions. A benchmark whose sample moved
        # between two runs has measured two different things.
        rng = random.Random(int(recipe.get("seed") or 1234))
        items = rng.sample(items, sample)
        ctx.log("Asking %s of the %s questions, chosen with seed %s -- the "
                "same ones every time this recipe is run."
                % (f"{len(items):,}", f"{len(list(rows)):,}",
                   recipe.get("seed")))
    else:
        ctx.log("Asking all %s questions." % f"{len(items):,}")

    shots: list[dict] = []
    if int(recipe.get("shots") or 0) > 0:
        split = bench.get("fewshot_split") or bench["split"]
        try:
            shot_rows = _load_split(bench, split, ctx)
            shots = _items(bench, shot_rows, ctx)
        except ValueError as e:
            ctx.log("No worked examples: %s. Running it 0-shot instead, which "
                    "is a different number from the published one." % e, "warn")
    return items, shots


# --------------------------------------------------------------- scoring

def _pick_shots(shots: list[dict], item: dict, n: int, rng) -> list[dict]:
    """Worked examples for one question.

    Matched by subject where the dataset has subjects, because that is what
    MMLU's own few-shot split is for: five examples from the same subject,
    which is the recipe every published MMLU number uses.
    """
    if not shots or n <= 0:
        return []
    group = item.get("group")
    pool = [s for s in shots if s.get("group") == group] if group else []
    if len(pool) < n:
        pool = shots
    return pool[:n] if len(pool) <= n else rng.sample(pool, n)


def _mc_block(item: dict, answer: int | None) -> str:
    """One multiple-choice question as the harness writes it."""
    lines = [item["question"]]
    for i, choice in enumerate(item["choices"]):
        lines.append("%s. %s" % (LETTERS[i], choice))
    lines.append("Answer:" if answer is None
                 else "Answer: %s" % LETTERS[answer])
    return "\n".join(lines)


def _context(bench: dict, recipe: dict, item: dict, shots: list[dict]) -> str:
    """Everything the model sees before the answer it is being scored on."""
    style = recipe.get("style") or bench.get("style") or "cloze"
    parts = []
    if preamble := bench.get("preamble"):
        # "professional psychology", not "professional_psychology". The
        # harness every published MMLU number comes from writes it with
        # spaces, and the model is reading it.
        group = (item.get("group") or "many subjects").replace("_", " ")
        parts.append(preamble.replace("{group}", group))
    if style == "letter":
        parts += [_mc_block(s, s["answer_index"]) for s in shots]
        parts.append(_mc_block(item, None))
    else:
        parts += ["%s %s" % (s["question"], s["choices"][s["answer_index"]])
                  for s in shots]
        parts.append(item["question"])
    return "\n\n".join(p for p in parts if p)


def _continuations(item: dict, style: str) -> list[str]:
    if style == "letter":
        return [" %s" % LETTERS[i] for i in range(len(item["choices"]))]
    return [" %s" % c for c in item["choices"]]


def _logprobs(host: Any, context: str, continuations: list[str],
              torch) -> list[float]:
    """How likely this model finds each ending, given the same beginning.

    The whole multiple-choice protocol rests on this one number. Every ending
    is scored in a single batch against one shared context: separate forward
    passes would give the same answer and take four times as long, and on a
    fourteen-thousand-question benchmark that is the difference between twenty
    minutes and an hour and a half.
    """
    tok, model = host.tok, host.model
    device = host.device
    ctx_ids = tok(context, add_special_tokens=True).input_ids
    ctx_len = len(ctx_ids)

    encoded = [tok(context + c, add_special_tokens=True).input_ids
               for c in continuations]

    # The letter protocol's endings are " A", " B", " C", " D" -- one token
    # each, over an identical two-thousand-token context. Encoding that
    # context once per letter and running it four times is four times the
    # work for an answer that is already sitting in the last position's
    # logits. On MMLU this is the difference between twenty seconds a
    # question and five, and it is exact rather than an approximation.
    singles = [e[ctx_len:] for e in encoded]
    if all(len(t) == 1 for t in singles) and all(
            e[:ctx_len] == ctx_ids for e in encoded):
        ids = torch.tensor([ctx_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(input_ids=ids).logits
        last = torch.log_softmax(logits[0, -1].float(), dim=-1)
        return [float(last[t[0]]) for t in singles]
    width = max(len(e) for e in encoded)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    ids = torch.full((len(encoded), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(encoded), width), dtype=torch.long)
    for i, e in enumerate(encoded):
        ids[i, :len(e)] = torch.tensor(e, dtype=torch.long)
        mask[i, :len(e)] = 1
    ids, mask = ids.to(device), mask.to(device)

    with torch.no_grad():
        logits = model(input_ids=ids, attention_mask=mask).logits
    logprobs = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    targets = ids[:, 1:]
    taken = logprobs.gather(2, targets.unsqueeze(-1)).squeeze(-1)

    out = []
    for i, e in enumerate(encoded):
        # Position j is predicted by the logits at j-1, so the continuation
        # tokens (j from ctx_len to the end) start at index ctx_len - 1.
        start = max(ctx_len - 1, 0)
        end = len(e) - 1
        out.append(float(taken[i, start:end].sum()) if end > start else -1e9)
    return out


def _numbers(text: str) -> list[str]:
    return re.findall(r"-?\$?\d[\d,]*\.?\d*", text or "")


def _extract(text: str) -> str | None:
    """The answer a model's working ends on.

    The last number in the reply, which is the "flexible extract" the
    published GSM8K numbers use. A model that reasons out loud and finishes
    with the answer is scored on the answer; one that states it first and then
    explains is not, and that is a known limitation of the measure rather
    than of the model.
    """
    found = _numbers(text)
    if not found:
        return None
    return found[-1].replace(",", "").replace("$", "").rstrip(".")


def _gold(answer: str) -> str | None:
    if "####" in answer:
        return answer.split("####")[-1].strip().replace(",", "")
    found = _numbers(answer)
    return found[-1].replace(",", "") if found else None


def _same_number(a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return a.strip() == b.strip()


def score(bench: dict, recipe: dict, host: Any, spec: dict, local: bool,
          params: dict, ctx: Any, tick: Callable[[], None]) -> tuple[list, dict]:
    """Put one benchmark to one model. Returns (kept rows, metrics)."""
    import torch

    protocol = bench["protocol"]
    if protocol == "multiple_choice" and not local:
        raise ValueError(
            "%s is decided by the model's own probabilities for each answer, "
            "and a model reached over the network does not expose them. It "
            "can be scored on %s, which is answered by writing."
            % (bench["label"], "a generated benchmark like GSM8K"))

    items, shots = prepare(bench, recipe, ctx)
    n_shots = int(recipe.get("shots") or 0)
    rng = random.Random(int(recipe.get("seed") or 1234))
    style = recipe.get("style") or bench.get("style") or "cloze"

    correct = correct_norm = 0
    asked = 0
    kept: list[dict] = []
    by_group: dict[str, list[int]] = {}
    t0 = time.time()

    for item in items:
        if ctx.should_cancel():
            raise Cancelled()
        picked = _pick_shots(shots, item, n_shots, rng)

        if protocol == "multiple_choice":
            context = _context(bench, recipe, item, picked)
            scores = _logprobs(host, context, _continuations(item, style), torch)
            # Length-normalised as well as raw. `acc_norm` is what ARC and
            # HellaSwag report, and on a cloze benchmark it is the more
            # sensible of the two: without it the model is largely being asked
            # which answer is shortest.
            lengths = [max(len(c), 1) for c in item["choices"]]
            normed = [s / lengths[i] for i, s in enumerate(scores)]
            got = max(range(len(scores)), key=lambda i: scores[i])
            got_norm = max(range(len(normed)), key=lambda i: normed[i])
            hit = got == item["answer_index"]
            hit_norm = got_norm == item["answer_index"]
            correct += hit
            correct_norm += hit_norm
            if len(kept) < KEEP_ROWS:
                kept.append({
                    "prompt": context[-MAX_STORED_CHARS:],
                    "expected": item["choices"][item["answer_index"]][:300],
                    "answer": item["choices"][got][:300],
                    "exact": hit,
                })
        else:
            prompt = "\n\n".join(
                ["Question: %s\nAnswer: %s" % (s["question"], s["answer"])
                 for s in picked]
                + ["Question: %s\nAnswer:" % item["question"]])
            # Asked as one message in the model's own format rather than as
            # raw text. A chat fine-tune given a bare completion prompt
            # answers badly for a reason that has nothing to do with
            # arithmetic -- and the published numbers for instruct models are
            # measured with their chat template applied too.
            out = host.generate(
                {**spec, "stop": ["\nQuestion:", "Question:"]},
                [{"role": "user", "content": prompt}],
                {**params, "max_new_tokens": bench.get("max_new_tokens", 256),
                 "temperature": 0.0},
                lambda *_: None, lambda _l: None)
            text = (out.get("text") or "")
            gold = _gold(item["answer"])
            hit = _same_number(_extract(text), gold)
            hit_norm = hit
            correct += hit
            correct_norm += hit
            if len(kept) < KEEP_ROWS:
                kept.append({
                    "prompt": item["question"][:MAX_STORED_CHARS],
                    "expected": gold or item["answer"][:200],
                    "answer": text[:MAX_STORED_CHARS],
                    "exact": hit,
                })

        asked += 1
        if group := item.get("group"):
            by_group.setdefault(group, [0, 0])
            by_group[group][0] += hit_norm if style != "letter" else hit
            by_group[group][1] += 1
        tick()
        if asked in (1, 25) or asked % 250 == 0:
            ctx.log("  %d of %d · %.1f%% correct so far"
                    % (asked, len(items), 100 * correct / max(asked, 1)))

    from common.stats import wilson

    headline = correct_norm if style == "cloze" else correct
    low, high = wilson(headline, asked)
    return kept, {
        "items": asked, "scored": asked,
        "accuracy": round(headline / max(asked, 1), 4),
        "accuracy_raw": round(correct / max(asked, 1), 4),
        "accuracy_norm": round(correct_norm / max(asked, 1), 4),
        "correct": headline,
        # The margin, always. A benchmark result quoted without one invites
        # exactly the comparison it cannot support.
        "accuracy_low": round(low, 4),
        "accuracy_high": round(high, 4),
        "seconds": round(time.time() - t0, 1),
        "by_group": {g: round(v[0] / max(v[1], 1), 4)
                     for g, v in sorted(by_group.items())} if by_group else None,
    }
