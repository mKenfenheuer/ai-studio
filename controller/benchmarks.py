"""The benchmarks a model card quotes, as recipes this studio can run.

Every published model is announced with a table: MMLU 74.2, GSM8K 88.1,
ARC-C 65.9. Nothing here could produce a number on that scale. A prompt set
measures a model against questions *you* wrote, which is the more useful
measurement for a model you are actually shipping -- and no help at all when
the question is "is this small fine-tune anywhere near the model I could have
paid for instead".

So: a benchmark is a prompt set whose questions come from a public dataset
instead of from you, scored the way the people who publish those numbers
score them.

## Comparable to a model card, on purpose

The number on a card is a *recipe*, not a measurement: MMLU at five worked
examples over all 14,042 questions, scored on the letter; ARC-Challenge at
twenty-five, length-normalised; HellaSwag at ten; GSM8K at five with the
answer read out of "#### 42". Change any one of those and the number moves by
more than the gap between most models -- which is why reproducing a
leaderboard is famously annoying, and why this module implements those
recipes exactly rather than something near them.

So each benchmark here carries the published recipe as its default: the whole
set, the conventional number of worked examples, the metric that gets quoted.
Run it that way and the result belongs beside a model card.

Departing is still allowed -- four hundred questions is two minutes and
answers "has this fine-tune fallen off a cliff" -- but a run that departs
says how, in `deviations()`, on its result and in its title. The two states
are kept apart deliberately: a sampled score sitting unlabelled next to a
published one is the single most misleading thing this could produce.

One benchmark cannot be run as published, and says so rather than pretending:
MMLU-Pro is quoted with chain-of-thought, and this scores the letter
directly.

## How each one is scored

Two protocols, because that is genuinely how these datasets work.

**multiple_choice.** The model is never asked to write anything. Each answer
is scored by how likely the model finds it, and the highest wins -- which is
what "MMLU accuracy" means, and why a model too small to follow an
instruction can still score on it. Two styles:

* `letter` -- the choices are printed in the prompt and the model is scored on
  " A" / " B" / " C" / " D". MMLU and MMLU-Pro are scored this way.
* `cloze` -- the choices are not printed; each one is scored as a
  continuation of the question, normalised by its length in characters so a
  long answer is not penalised for being long. This is `acc_norm`, and it is
  what ARC and HellaSwag report.

**generate_extract.** The model writes an answer and a number is pulled out of
it. GSM8K works this way: the last number in the reply is taken as the answer,
which is the "flexible extract" the harness reports, and is why a model that
reasons out loud is not punished for it.

Code benchmarks -- HumanEval, MBPP, LiveCodeBench -- are missing on purpose
and are listed as such rather than quietly omitted: scoring them means
executing whatever the model wrote, and a sandbox that is safe to do that in
is its own piece of work, not a line in a table here.
"""
from __future__ import annotations

from common.stats import wilson          # re-exported: the UI reads it here

# Where the few-shot examples come from, and how many of them each benchmark
# is conventionally run with. The counts are the published ones: changing them
# changes the number, which is why they are recorded on every result.
BENCHMARKS: dict[str, dict] = {
    "mmlu": {
        "label": "MMLU",
        "what": "Fifty-seven school and professional subjects, four choices "
                "each. The most quoted number there is, and the one most "
                "sensitive to how it is asked.",
        "dataset": "cais/mmlu", "config": "all", "split": "test",
        "fewshot_split": "dev",
        "protocol": "multiple_choice", "style": "letter",
        "shots": 5, "size": 14042,
        # What the number on a card is: MMLU quotes plain accuracy at 5-shot,
        # over all 14,042 questions, scored on the letter.
        "metric": "acc", "card_shots": 5, "faithful": True,
        "fields": {"question": "question", "choices": "choices",
                   "answer": "answer", "group": "subject"},
        "preamble": "The following are multiple choice questions (with "
                    "answers) about {group}.",
        "published": "Quoted 5-shot. Reasoning models often quote a much "
                     "higher chain-of-thought score instead; that is a "
                     "different measurement from this one.",
    },
    "mmlu_pro": {
        "label": "MMLU-Pro",
        "what": "MMLU made harder: ten choices instead of four, and the "
                "easiest questions removed. Scores land far below MMLU and "
                "that is the point.",
        "dataset": "TIGER-Lab/MMLU-Pro", "config": "default", "split": "test",
        "fewshot_split": "validation",
        "protocol": "multiple_choice", "style": "letter",
        "shots": 5, "size": 12032,
        "metric": "acc", "card_shots": 5, "faithful": False,
        # The one benchmark here that cannot be run the way it is published.
        "deviation": "Published MMLU-Pro numbers are produced by asking the "
                     "model to reason and then extracting a letter from what "
                     "it wrote. This scores the letter directly, which is a "
                     "different measurement and lands several points lower on "
                     "every model.",
        "fields": {"question": "question", "choices": "options",
                   "answer": "answer_index", "group": "category"},
        "preamble": "The following are multiple choice questions (with "
                    "answers) about {group}.",
        "published": "Usually quoted with chain-of-thought, which this does "
                     "not do. Expect a lower number here than on a model "
                     "card, on every model, by a similar amount.",
    },
    "arc_challenge": {
        "label": "ARC-Challenge",
        "what": "Grade-school science questions chosen because simple "
                "retrieval systems got them wrong.",
        "dataset": "allenai/ai2_arc", "config": "ARC-Challenge", "split": "test",
        "fewshot_split": "train",
        "protocol": "multiple_choice", "style": "cloze",
        # 25-shot, length-normalised accuracy: the recipe the numbers on model
        # cards come from. The harness's own default is 0-shot, which is a
        # different number and several points lower.
        "shots": 25, "size": 1172,
        "metric": "acc_norm", "card_shots": 25, "faithful": True,
        "fields": {"question": "question", "choices": "choices",
                   "answer": "answerKey"},
        "published": "Quoted at 25-shot, length-normalised. Run at 0-shot it "
                     "is several points lower, and that is a different number "
                     "rather than a worse model.",
    },
    "hellaswag": {
        "label": "HellaSwag",
        "what": "Which of four endings actually continues this sentence. "
                "Easy for a person, historically hard for a model.",
        "dataset": "Rowan/hellaswag", "config": "default", "split": "validation",
        "fewshot_split": "train",
        "protocol": "multiple_choice", "style": "cloze",
        "shots": 10, "size": 10042,
        "metric": "acc_norm", "card_shots": 10, "faithful": True,
        "fields": {"question": "ctx", "choices": "endings", "answer": "label",
                   "group": "activity_label"},
        "preprocess": "hellaswag",
        "published": "Quoted at 10-shot on the older leaderboards. The test "
                     "split has no answers in it, so this is the validation "
                     "split -- which is what everybody reports.",
    },
    "gsm8k": {
        "label": "GSM8K",
        "what": "Grade-school word problems that take several steps of "
                "arithmetic. The model writes its working and the last number "
                "in it is taken as the answer.",
        "dataset": "openai/gsm8k", "config": "main", "split": "test",
        "fewshot_split": "train",
        "protocol": "generate_extract",
        "shots": 5, "size": 1319,
        # Strict match: the answer has to arrive as "#### 42", which is what
        # the five worked examples demonstrate and what the published number
        # measures. The looser "last number in the reply" is kept beside it.
        "metric": "acc", "card_shots": 5, "faithful": True,
        "extract": "strict",
        "fields": {"question": "question", "answer": "answer"},
        "max_new_tokens": 256,
        "published": "Quoted 5-shot or 8-shot. A model that cannot follow the "
                     "worked-example format scores near zero here for reasons "
                     "that have nothing to do with arithmetic.",
    },
}

# Named so the list is not silently missing the benchmark everybody asks
# about. Each says what it would take, because "not supported" without a
# reason reads as "not thought about".
UNAVAILABLE: dict[str, str] = {
    "HumanEval": "Scoring it means running code the model wrote. That needs a "
                 "sandbox that is safe to execute a stranger's Python in, "
                 "which is a piece of work in its own right and not something "
                 "to do casually on a machine that also holds your data.",
    "MBPP": "Same as HumanEval: the score is whether the code passes tests, "
            "so the code has to run somewhere isolated.",
    "IFEval": "Its checks are a library of little verifiers -- \"is this "
              "under 300 words\", \"does it avoid the letter e\" -- and the "
              "score is only meaningful with exactly that library.",
    "TruthfulQA": "Its MC2 score sums probability across several correct "
                  "answers at once, which is a third scoring protocol; its "
                  "generative score needs two judge models that were fine-"
                  "tuned for it and are no longer published.",
}

# The whole benchmark, unless somebody asks for less. A published number is
# every question in the set; a sample of them is a different measurement --
# 400 questions cannot tell 61% from 63%, which is what the confidence
# interval on every result is there to say -- and quoting one beside a model
# card is exactly the mistake this module exists to prevent.
#
# Sampling is still offered, because "has this fine-tune fallen off a cliff"
# is a real question worth two minutes rather than two hours. A sampled run
# says so on its face, in its title and in its result.
FULL = 0                    # what `sample` means when it is not a sample
DEFAULT_SAMPLE = FULL
MAX_SAMPLE = 20000


def sample_size(b: dict, asked: int | None) -> int:
    """How many questions to ask: all of them unless a smaller number is."""
    if not asked or asked <= 0 or asked >= b["size"]:
        return b["size"]
    return max(20, asked)


def deviations(b: dict, recipe: dict) -> list[str]:
    """Every way this run departs from the number a model card quotes.

    Written out rather than summarised, because each of these moves the
    result by more than the gap between two models, and a reader comparing
    against a card needs to know which of them applies. An empty list is the
    interesting case: it means the number can be quoted beside the card's.
    """
    out = []
    asked = int(recipe.get("sample") or 0)
    if asked and asked < b["size"]:
        out.append("%s of the %s questions, chosen at random with seed %s -- "
                   "a published score is the whole set."
                   % (f"{asked:,}", f"{b['size']:,}", recipe.get("seed")))
    shots = recipe.get("shots")
    if shots is not None and b.get("card_shots") is not None \
            and int(shots) != int(b["card_shots"]):
        out.append("%d worked examples rather than the %d this benchmark is "
                   "published with." % (int(shots), int(b["card_shots"])))
    if recipe.get("chat_template"):
        out.append("Asked through the model's chat template. Published "
                   "numbers are measured on raw completions, and for a "
                   "chat model the two differ.")
    if b.get("deviation"):
        out.append(b["deviation"])
    return out


def get(name: str) -> dict | None:
    b = BENCHMARKS.get((name or "").strip().lower())
    return {**b, "id": (name or "").strip().lower()} if b else None


def public() -> list[dict]:
    """The catalogue, for the browser."""
    out = []
    for bid, b in BENCHMARKS.items():
        out.append({
            "id": bid, "label": b["label"], "what": b["what"],
            "dataset": b["dataset"], "split": b["split"],
            "protocol": b["protocol"], "style": b.get("style") or "",
            "shots": b["shots"], "size": b["size"],
            "published": b.get("published") or "",
            # A model reached over the network cannot be scored on the
            # multiple-choice benchmarks at all: they are decided by the
            # model's own probabilities, and no provider gives those out.
            "needs_local": b["protocol"] == "multiple_choice",
        })
    return out


def recipe(b: dict, shots: int, sample: int, seed: int) -> dict:
    """Exactly what was run, in a form worth storing next to the result.

    Every field here changes the number. A result without them is a number
    somebody will compare against a different number next year and draw a
    conclusion from the difference between two recipes.
    """
    return {
        "benchmark": b["id"], "label": b["label"],
        "dataset": b["dataset"], "config": b.get("config"),
        "split": b["split"], "protocol": b["protocol"],
        "style": b.get("style") or "", "shots": shots,
        "sample": sample, "seed": seed,
        "fewshot_split": b.get("fewshot_split"),
        # Which of the two accuracies is *the* number for this benchmark, and
        # how a written answer is read out of the reply. Recorded because a
        # result compared against a card has to have been measured the same
        # way, and "accuracy" alone does not say that.
        "metric": b.get("metric") or ("acc_norm" if b.get("style") == "cloze"
                                      else "acc"),
        "extract": b.get("extract") or "flexible",
    }


def title(b: dict, shots: int, sample: int) -> str:
    whole = not sample or sample >= b["size"]
    return "%s · %d-shot · %s" % (
        b["label"], shots,
        "all %s questions" % f"{b['size']:,}" if whole
        else "%s of %s questions" % (f"{sample:,}", f"{b['size']:,}"))
