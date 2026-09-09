#!/usr/bin/env python
"""Check that a model card says true things, and no Python in it.

Run it with `python scripts/check-cards.py` from the repository root.

A card is the one artefact of this studio that leaves it: it is the README of
a public repository, read by people who will never see the run behind it. That
makes its failures unusually expensive and unusually quiet -- a table printing
"None" under three columns, or claiming a score was measured on held-out rows
when it was measured on the training set, is not something the app will ever
complain about.

So: build a run with real-looking facts, generate its card, and read it.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_TMP = Path(tempfile.mkdtemp(prefix="ai-studio-check-"))
os.environ["AI_STUDIO_DATA"] = str(_TMP)

from controller import cards, db  # noqa: E402

FAILED: list[str] = []


def check(name: str, ok: bool, detail: object = "") -> None:
    print("  %s  %s%s" % ("ok  " if ok else "FAIL", name,
                          "" if ok else "   -> %s" % (detail,)))
    if not ok:
        FAILED.append(name)


def score(job_id: str, eval_id: str, name: str, metrics: dict,
          source: dict | None, when: float) -> None:
    db.ex("INSERT INTO evals (id,owner_id,name,notes,items,source,created_at,"
          "updated_at) VALUES (?,?,?,?,?,?,?,?)",
          (eval_id, None, name, "", "[]",
           json.dumps(source) if source else None, when, when))
    db.ex("INSERT INTO eval_scores (id,eval_id,model_job_id,run_job_id,metrics,"
          "items,created_at) VALUES (?,?,?,?,?,?,?)",
          ("scr_" + eval_id, eval_id, job_id, "job_run", json.dumps(metrics),
           "[]", when))


def main() -> int:
    try:
        now = time.time()
        ds = db.create_dataset(None, "house rows", "upload", rows=9504,
                               splits={"train": 9504})
        job = db.create_job("Mistral on house rows", "finetune_llm", {
            "base_model": "mistralai/Mistral-7B-Instruct-v0.3",
            "params_b": 7.0, "studio_dataset": ds, "dataset_split": "train",
            "quantization": "4bit", "epochs": 1, "learning_rate": 2e-4,
            "batch_size": 2, "grad_accum": 4, "max_seq_len": 1024,
            "lora_r": 16, "dtype": "bfloat16",
        })
        db.ex("UPDATE jobs SET status='succeeded', finished_at=? WHERE id=?",
              (now, job))
        db.set_job_summary(job, {
            "final_loss": 0.135, "initial_loss": 2.08, "best_val_loss": 0.1398,
            "steps": 40, "duration_s": 491, "trainable_params": 13631488,
            "held_out_rows": 256, "peak_vram_gb": 5.29,
            "primary_metric": {"name": "held_out_loss", "label": "Held-out loss",
                               "value": 0.1398, "lower_better": True},
        })
        db.add_artifact(job, "adapter", "%s.zip" % job, 58_249_247)

        # A benchmark, a tool-calling set scored twice, and a set that errored.
        score(job, "ev_arc", "ARC-Challenge · 0-shot · 60 questions",
              {"accuracy": 0.4333, "accuracy_low": 0.3157, "items": 60},
              {"benchmark": "arc_challenge",
               "recipe": {"benchmark": "arc_challenge", "sample": 60}}, now - 300)
        score(job, "ev_tools", "House questions",
              {"tool_name_ok": 0.25, "tool_args_ok": 0.33, "items": 12},
              {"dataset_id": ds, "split": "train"}, now - 200)
        score(job, "ev_tools_again", "House questions",
              {"tool_name_ok": 0.20, "items": 12},
              {"dataset_id": ds, "split": "train"}, now - 400)
        score(job, "ev_broken", "A scoring that failed",
              {"error": "the runner went away", "items": 0}, None, now - 100)

        card = cards.generate(db.get_job(job))
        print("\nThe card as a whole")
        check("nothing Python leaked into it",
              "None" not in card and "{'" not in card,
              [line for line in card.split("\n") if "None" in line][:2])
        check("it opens with front matter", card.startswith("---\n"))
        check("and names the base model",
              "base_model: mistralai/Mistral-7B-Instruct-v0.3" in card)

        print("\nThe results table")
        table = card[card.find("## How it scored"):]
        check("a benchmark's accuracy is in it", "43%" in table, table[:400])
        check("so is the tool-calling measure", "25%" in table)
        check("the columns nothing measured are left out",
              "Token overlap" not in table and "Exact" not in table.split("\n")[5])
        check("a benchmark is named, not called hand-written",
              "ARC-CHALLENGE, 60 asked" in table and "hand-written" not in table,
              table[:300])
        check("a set scored twice appears once",
              table.count("| House questions |"), 1)
        check("and the warning about the training split survives",
              "**from the split this model trained on**" in table)

        print("\nThe front matter the Hub indexes")
        head = card.split("\n---")[0]
        check("model-index is there", "model-index:" in head)
        check("with the accuracy in it", "value: 0.4333" in head, head[-400:])
        check("a failed scoring contributes nothing",
              "A scoring that failed" not in head)

        print("\nA generated card is never stale")
        db.set_job_card(job, "# From an older generator\n", edited=False)
        served = cards.card_for(db.get_job(job))
        check("reading it rewrites it from what is known now",
              "How it scored" in served["markdown"], served["markdown"][:80])
        check("and stores what it served",
              db.get_job_card(job)["markdown"], served["markdown"])

        print("\nAn edit is the author's")
        db.set_job_card(job, "# Mine\n\nHands off.", edited=True)
        kept = cards.for_publish(db.get_job(job), db.get_job(job), "me/model")
        check("publishing sends what they wrote", kept, "# Mine\n\nHands off.")
        cards.refresh(job)
        check("and regeneration leaves it alone",
              db.get_job_card(job)["markdown"], "# Mine\n\nHands off.")
    finally:
        shutil.rmtree(_TMP, ignore_errors=True)

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
