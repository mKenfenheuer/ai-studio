#!/usr/bin/env python
"""Check that a benchmark is asked the way its published number was asked.

Run it with `python scripts/check-benchmarks.py` from the repository root.

A benchmark score is a recipe, not a measurement. MMLU at five worked
examples over all 14,042 questions, scored on the letter; ARC-Challenge at
twenty-five, length-normalised; GSM8K at five with the answer read out of
"#### 42". Change one of those and the number moves by more than the gap
between most models -- so the prompts here are compared against the exact
strings the harness builds, character for character, rather than against a
description of them.

The strings below are not "what the code currently does". They are what the
published recipe is, written out by hand from the harness's own templates,
which is the only way this check is worth anything.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from controller import benchmarks as bm  # noqa: E402
from runner.jobs import benchmark as run  # noqa: E402

FAILED: list[str] = []


def check(name: str, got: object, want: object = True) -> None:
    ok = got == want
    print("  %s  %s%s" % ("ok  " if ok else "FAIL", name,
                          "" if ok else "\n        -> %r, wanted %r" % (got, want)))
    if not ok:
        FAILED.append(name)


def same(name: str, got: str, want: str) -> None:
    ok = got == want
    print("  %s  %s%s" % ("ok  " if ok else "FAIL", name,
                          "" if ok else "\n      got:\n%s\n      wanted:\n%s"
                          % (_show(got), _show(want))))
    if not ok:
        FAILED.append(name)


def _show(text: str) -> str:
    return "\n".join("        | " + line for line in str(text).split("\n"))


def main() -> int:
    print("\nThe recipes are the published ones")
    for bid, want_shots, want_metric in (("mmlu", 5, "acc"),
                                         ("arc_challenge", 25, "acc_norm"),
                                         ("hellaswag", 10, "acc_norm"),
                                         ("gsm8k", 5, "acc")):
        b = bm.get(bid)
        check("%s is %d-shot" % (bid, want_shots), b["shots"], want_shots)
        check("  and quotes %s" % want_metric, b["metric"], want_metric)
    check("a run asks the whole set by default",
          bm.sample_size(bm.get("mmlu"), None), 14042)
    check("and says nothing is off about it",
          bm.deviations(bm.get("mmlu"),
                        bm.recipe(bm.get("mmlu"), 5, 14042, 1234)), [])
    off = bm.deviations(bm.get("mmlu"), bm.recipe(bm.get("mmlu"), 0, 400, 1234))
    check("a sampled, 0-shot run says both", len(off), 2)
    check("naming the sample", "400 of the 14,042" in off[0])
    check("and the missing worked examples", "5 this benchmark" in off[1])

    print("\nMMLU, as the harness writes it")
    mmlu = bm.get("mmlu")
    item = {"question": "What is 2 + 2?", "choices": ["3", "4", "5", "6"],
            "answer_index": 1, "group": "elementary_mathematics"}
    shot = {"question": "What is 1 + 1?", "choices": ["1", "2", "3", "4"],
            "answer_index": 1, "group": "elementary_mathematics"}
    same("the prompt matches, subject and all",
         run._context(mmlu, bm.recipe(mmlu, 1, 14042, 1234), item, [shot]),
         "The following are multiple choice questions (with answers) about "
         "elementary mathematics.\n"
         "\n"
         "What is 1 + 1?\n"
         "A. 1\n"
         "B. 2\n"
         "C. 3\n"
         "D. 4\n"
         "Answer: B\n"
         "\n"
         "What is 2 + 2?\n"
         "A. 3\n"
         "B. 4\n"
         "C. 5\n"
         "D. 6\n"
         "Answer:")
    check("and it is scored on the letter",
          run._continuations(item, "letter"), [" A", " B", " C", " D"])

    print("\nARC and HellaSwag, which are scored on the answer text")
    arc = bm.get("arc_challenge")
    a_item = {"question": "Which is a mammal?",
              "choices": ["A snake", "A whale"], "answer_index": 1}
    a_shot = {"question": "Which is a bird?",
              "choices": ["A cat", "A robin"], "answer_index": 1}
    same("the question and the example run together",
         run._context(arc, bm.recipe(arc, 1, 1172, 1234), a_item, [a_shot]),
         "Which is a bird? A robin\n\nWhich is a mammal?")
    check("the endings are the continuations",
          run._continuations(a_item, "cloze"), [" A snake", " A whale"])
    got_ctx, got_ends = run._hellaswag(
        {"activity_label": "Roof shingle removal",
         "ctx_a": "A man is on a roof.",
         "ctx_b": "he starts to [title] remove shingles",
         "endings": ["He falls.", "He [header] waves."]})
    same("HellaSwag's markup is stripped the way the harness strips it",
         got_ctx, "Roof shingle removal: A man is on a roof. He starts to. "
                  "remove shingles")
    check("  and out of the endings too", got_ends, ["He falls.", "He waves."])

    print("\nGSM8K, which is read out of what the model wrote")
    check("the published reading is the strict one",
          bm.get("gsm8k")["extract"], "strict")
    reply = ("Janet has 16 eggs. She eats 3 and bakes 4, so 16 - 7 = 9 are "
             "left. At $2 each that is 9 * 2 = 18.\n#### 18")
    check("a reply in the demonstrated form is read",
          run._extract_strict(reply), "18")
    check("and so is the last number, separately",
          run._extract(reply), "18")
    loose = "The answer is 18 dollars, because 9 eggs at $2."
    check("a reply that never uses the form fails the strict reading",
          run._extract_strict(loose), None)
    check("but the loose reading still finds a number",
          run._extract(loose), "2")
    check("the gold answer is the number after ####",
          run._gold("Some working here.\n#### 1,024"), "1024")
    check("thousands separators do not break the comparison",
          run._same_number("1024", run._gold("x\n#### 1,024")), True)

    print("\nWhat is recorded beside the result")
    r = bm.recipe(bm.get("gsm8k"), 5, 1319, 1234)
    for key in ("benchmark", "dataset", "split", "shots", "sample", "seed",
                "metric", "extract", "fewshot_split"):
        check("the recipe records %s" % key, key in r)
    check("MMLU-Pro is marked as not the published protocol",
          bm.get("mmlu_pro")["faithful"], False)
    check("and says what it does instead",
          "reason" in (bm.get("mmlu_pro").get("deviation") or ""))
    check("which travels with every result of it",
          bm.get("mmlu_pro")["deviation"]
          in bm.deviations(bm.get("mmlu_pro"),
                           bm.recipe(bm.get("mmlu_pro"), 5, 12032, 1234)))

    print()
    if FAILED:
        print("%d check(s) failed:" % len(FAILED))
        for name in FAILED:
            print("  - %s" % name)
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
