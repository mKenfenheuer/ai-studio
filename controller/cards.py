"""Model cards: what a run produced, written where the Hub can read it.

Everything on a card is something this studio already knew -- what the model
was trained from, on what data, with which hyperparameters, how far the loss
came down, and what it scored when it was evaluated. Until now that knowledge
lived in five different places on the run's page and was assembled, once, at
the moment somebody pressed Publish. A model that was never published had no
card at all, and a model that was published before it was evaluated kept a
card that pre-dated its own results.

So a card is a thing a run *has*, from the moment it produces a model:

* **Generated when the facts change.** When training finishes, when the merge
  lands, when an evaluation scores it. Each of those is a moment the card
  would have become wrong, which is the only sensible time to rewrite it.
* **Editable, and then left alone.** A generated card is a floor, not a
  ceiling: nobody but the person who trained the model knows what it is for,
  where the data came from, or who should not use it. Once they have written
  that down, `edited` is set and nothing here overwrites it again.
* **The same file the Hub gets.** Not a studio-flavoured summary that is
  translated into a card at upload time -- the card *is* the README, front
  matter and all, so what you read here is what the Hub will render.

The front matter follows the Hub's model card spec: `library_name`,
`base_model` with `base_model_relation`, `datasets`, `pipeline_tag`, `tags`,
and `model-index` for evaluation results, which is the only form the Hub
understands well enough to index and display as metrics.
"""
from __future__ import annotations

import time

from . import db, serving

# Hugging Face reads these from `model-index` and renders them as a metrics
# table. The keys on the left are what evaluate.py measures; the values are
# what the Hub calls them. Anything not in here is left out of the front
# matter rather than invented, and still appears in the prose table below it.
_METRIC_NAMES = {
    "exact": ("exact_match", "Exact match"),
    "contains": ("accuracy", "Answer contained"),
    "f1": ("f1", "Token overlap (F1)"),
    "expected_loss": ("loss", "Loss on the expected answer"),
    "expected_perplexity": ("perplexity", "Perplexity of the expected answer"),
}


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def card_for(job: dict, *, artifact_job: dict | None = None,
             repo_id: str | None = None) -> dict:
    """This run's card, generated on the spot if it has never had one.

    Returned rather than stored, so opening the page of a run that finished
    before any of this existed shows a real card instead of an empty box and
    an invitation to write one from nothing.
    """
    if row := db.get_job_card(job["id"]):
        return {"markdown": row["markdown"], "edited": bool(row["edited"]),
                "updated_at": row["updated_at"], "saved": True}
    return {"markdown": generate(job, artifact_job=artifact_job,
                                 repo_id=repo_id),
            "edited": False, "updated_at": None, "saved": False}


def refresh(job_id: str, *, artifact_job: dict | None = None,
            reason: str = "") -> str | None:
    """Rewrite this run's card from what is now known about it.

    Does nothing to a card somebody has edited. That rule is the whole reason
    the flag exists: these are called from the scheduler, hours after the
    person who wrote the card has gone home, and silently replacing their
    paragraph about what the model is for with a generated one would make the
    editor pointless.
    """
    job = db.get_job(job_id)
    if not job:
        return None
    row = db.get_job_card(job_id)
    if row and row["edited"]:
        return None
    try:
        text = generate(job, artifact_job=artifact_job)
    except Exception as e:      # noqa: BLE001 - a card is never worth a failure
        db.add_log(job_id, "Could not write the model card (%s). Everything "
                           "else about this run is unaffected." % e, "warn")
        return None
    if row and row["markdown"] == text:
        return None             # nothing changed; do not touch updated_at
    db.set_job_card(job_id, text, edited=False)
    if reason:
        db.add_log(job_id, "Model card updated: %s" % reason)
    return text


def for_publish(trained: dict, artifact: dict, repo_id: str) -> str:
    """The card to send with an upload.

    An edited card goes as it was written -- it is somebody's words about
    their own model, and rewriting them on the way out would be worse than
    having no editor at all. A generated one is regenerated here instead of
    reused, because publishing is the first moment two facts are known that
    the stored card could not have: which repository this is, and which run's
    files are actually being sent.
    """
    row = db.get_job_card(trained["id"])
    if row and row["edited"]:
        return row["markdown"]
    return generate(trained, artifact_job=artifact, repo_id=repo_id)


# ---------------------------------------------------------------------------
# Writing one
# ---------------------------------------------------------------------------

def generate(job: dict, *, artifact_job: dict | None = None,
             repo_id: str | None = None) -> str:
    """The whole card: front matter, then prose.

    `job` is the run being described and `artifact_job` the run whose files
    the card will sit beside. They are the same thing except for a fine-tune,
    where what goes to the Hub is the merged model and what is worth writing
    down -- the data, the hyperparameters, where the loss got to -- belongs to
    the training run.
    """
    artifact_job = artifact_job or job
    facts = _facts(job, artifact_job, repo_id)
    return "\n".join(_front_matter(facts) + _body(facts)).rstrip() + "\n"


def _facts(job: dict, artifact_job: dict, repo_id: str | None) -> dict:
    cfg = serving.resolved_config(job) if job.get("kind") == "merge_adapter" \
        else (job.get("config") or {})
    summary = job.get("summary") or {}
    scratch = job.get("kind") == "pretrain_llm"
    merged = artifact_job.get("kind") == "merge_adapter"
    # What is in the repository, which is not always what the run produced: a
    # fine-tune's own artifact is an adapter, and its merged run's is a model.
    adapter = not scratch and not merged

    base = cfg.get("base_model") \
        or (artifact_job.get("config") or {}).get("base_model") \
        or summary.get("base_model")
    # A base that is another run in this studio is named, not linked: its id
    # means nothing on the Hub, and `base_model:` there must be a Hub id or
    # the card fails to render.
    base_run = db.get_job(cfg.get("base_model_job") or "") \
        if cfg.get("base_model_job") else None
    if base_run and not _is_hub_id(base):
        base = None

    return {
        "job": job, "artifact_job": artifact_job, "cfg": cfg,
        "summary": summary, "scratch": scratch, "merged": merged,
        "adapter": adapter, "base": base, "base_run": base_run,
        "repo_id": repo_id,
        "name": (repo_id or "").split("/")[-1] or job.get("name") or "model",
        "dataset": cfg.get("dataset"),
        "dataset_local": bool(cfg.get("dataset_is_local")),
        "dataset_label": cfg.get("dataset_label") or cfg.get("dataset"),
        "scores": db.scores_for_model(job["id"]),
        "load_id": repo_id or ".",
    }


def _front_matter(f: dict) -> list[str]:
    tags = ["ai-studio"]
    tags.append("pretrained-from-scratch" if f["scratch"]
                # `lora` on the Hub means a repository holding an adapter, and
                # a merged model is not one -- it is a complete set of weights
                # that happens to have been trained with LoRA. Tagging it
                # `lora` sends people looking for an adapter_config.json that
                # is not there.
                else "lora" if f["adapter"] else "merged-lora")
    if not f["scratch"]:
        tags.append("fine-tuned")

    out = ["---",
           # An adapter repository is a PEFT repository: `transformers` alone
           # tells the Hub to offer a widget that cannot load it.
           "library_name: %s" % ("peft" if f["adapter"] else "transformers"),
           "pipeline_tag: text-generation", "tags:"]
    out += ["  - %s" % t for t in tags]
    if f["base"]:
        out.append("base_model: %s" % f["base"])
        # The Hub uses this to decide what to say on the base model's page
        # about the things built from it. Guessing wrong there is how a merged
        # model ends up listed as somebody's adapter.
        out.append("base_model_relation: %s"
                   % ("adapter" if f["adapter"] else "merge"))
    if f["dataset"] and not f["dataset_local"]:
        out += ["datasets:", "  - %s" % f["dataset"]]
    out += _model_index(f)
    out += ["---", ""]
    return out


def _model_index(f: dict) -> list[str]:
    """Evaluation results, in the one shape the Hub indexes.

    Only scores with numbers in them: an evaluation that errored, or one that
    ran with no expected answers to compare against, has nothing to say here
    and an empty `metrics:` list makes the whole block invalid.
    """
    results: list[list[str]] = []
    for score in f["scores"]:
        metrics = score.get("metrics") or {}
        rows = [(hub, label, metrics[key])
                for key, (hub, label) in _METRIC_NAMES.items()
                if isinstance(metrics.get(key), (int, float))]
        if not rows:
            continue
        block = ["    - task:", "        type: text-generation",
                 "        name: Text Generation", "      dataset:",
                 "        name: %s" % _yaml(score.get("eval_name")
                                            or "A saved prompt set"),
                 # Not a Hub dataset. The type is required and is a free
                 # string, so it says where the prompts actually came from.
                 "        type: ai-studio-eval", "      metrics:"]
        for hub, label, value in rows:
            block += ["        - type: %s" % hub,
                      "          name: %s" % _yaml(label),
                      "          value: %s" % _number(value)]
        results.append(block)
        if len(results) >= 4:   # the four most recent; the table below has all
            break
    if not results:
        return []
    out = ["model-index:", "  - name: %s" % _yaml(f["name"]), "    results:"]
    for block in results:
        out += block
    return out


def _body(f: dict) -> list[str]:
    job, cfg, summary = f["job"], f["cfg"], f["summary"]
    out = ["# %s" % f["name"], ""]

    made = "trained from scratch with" if f["scratch"] else "fine-tuned with"
    on = _dataset_phrase(f)
    out.append("%s, %s [AI Studio](https://github.com/)%s."
               % (_lead(f), made, on))
    if f["merged"]:
        out += ["", "The LoRA adapter has been folded into the base weights, "
                "so this is a complete model: it loads with `from_pretrained` "
                "and needs nothing downloaded alongside it."]
    if f["adapter"]:
        out += ["", "This repository holds the **adapter only** — a few "
                "megabytes of low-rank matrices, not a model. It is applied "
                "to its base at load time, so the base is downloaded too and "
                "must be the one named above."]
    if job.get("status") == "cancelled":
        out += ["", "> **Stopped before the end of its schedule.** It "
                "completed %s of %s planned steps, so its learning rate never "
                "finished decaying and it is rougher than the same run taken "
                "to completion." % (job.get("step"), job.get("total_steps"))]

    out += _table("What it is", _identity_rows(f))
    out += ["", "## Use it", "", "```python", *_usage(f), "```"]
    out += _table("Architecture", _arch_rows(f))
    out += _table("How it was trained", _training_rows(f))
    out += _table("Where it got to", _result_rows(f))
    out += _evaluations(f)
    out += _talking_to_it(f)

    out += ["", "## What it is not", "",
            "It learned the shape of the examples it was shown, and nothing "
            "beyond them. Its answers are as accurate, as current and as "
            "biased as %s, and it will produce a plausible answer where it "
            "has no grounds for one. Check anything that matters."
            % ("the text it was trained on" if f["scratch"]
               else "the data it was fine-tuned on"), ""]
    if f["base"]:
        out += ["It inherits the licence and the terms of use of "
                "[%s](https://huggingface.co/%s). Nothing here relaxes them."
                % (f["base"], f["base"]), ""]
    out += ["---", "",
            "<sub>Card generated by [AI Studio](https://github.com/) from the "
            "run that produced this model%s. Edit it freely; it is not "
            "regenerated once you have.</sub>"
            % (" on %s" % _date(summary.get("finished_at")
                                or job.get("finished_at")) if
               (summary.get("finished_at") or job.get("finished_at")) else "")]
    return out


def _lead(f: dict) -> str:
    if f["scratch"]:
        arch = (f["cfg"].get("arch") or {})
        size = _params(f["summary"].get("params_total")
                       or f["summary"].get("params"))
        return "A %s language model%s" % (
            arch.get("model_type") or "small", ", %s," % size if size else "")
    base = f["base"] or (f["base_run"] or {}).get("name") or "a base model"
    return "[%s](https://huggingface.co/%s)" % (base, base) \
        if _is_hub_id(base) else base


def _dataset_phrase(f: dict) -> str:
    label = f["dataset_label"]
    if not label:
        return ""
    if f["dataset"] and not f["dataset_local"]:
        return " on [%s](https://huggingface.co/datasets/%s)" % (label, label)
    return " on %s, a dataset held in the studio" % label


def _identity_rows(f: dict) -> list[tuple[str, str]]:
    cfg, summary = f["cfg"], f["summary"]
    base = f["base"] or (f["base_run"] or {}).get("name")
    rows = [
        ("Base model", "[%s](https://huggingface.co/%s)" % (base, base)
         if _is_hub_id(base) else base),
        ("Method", "trained from scratch" if f["scratch"]
         else "LoRA, merged into the base" if f["merged"]
         else "LoRA adapter"),
        ("Parameters", _params(summary.get("params_total")
                               or (cfg.get("params_b") or 0) * 1e9)),
        ("Precision", summary.get("dtype") or cfg.get("dtype")),
        ("Trained on", f["dataset_label"]),
        ("Rows held out", _num(summary.get("held_out_rows"))),
        ("Finished", _date(f["job"].get("finished_at"))),
    ]
    if f["adapter"]:
        rows.insert(3, ("Trainable parameters",
                        _params(summary.get("trainable_params"))))
    return rows


def _arch_rows(f: dict) -> list[tuple[str, str | None]]:
    """The shape of a model built here from nothing.

    Only for a from-scratch run. For a fine-tune all of this belongs to the
    base model and is already on the base model's own card, which is linked
    above -- repeating it here would be describing somebody else's work.
    """
    arch = f["cfg"].get("arch") or {}
    if not f["scratch"] or not arch:
        return []
    return [
        ("Type", arch.get("model_type")),
        ("Layers", _num(arch.get("num_hidden_layers"))),
        ("Hidden size", _num(arch.get("hidden_size"))),
        ("Attention heads", _num(arch.get("num_attention_heads"))),
        ("Key/value heads", _num(arch.get("num_key_value_heads"))),
        ("Context length", _num(arch.get("max_position_embeddings"))),
        ("Vocabulary", _num(arch.get("vocab_size"))),
        ("Experts", _num(arch.get("num_local_experts"))),
    ]


def _training_rows(f: dict) -> list[tuple[str, str]]:
    cfg, summary = f["cfg"], f["summary"]
    rows = [
        ("Epochs", _num(cfg.get("epochs"))),
        ("Learning rate", "%.2e" % float(cfg["learning_rate"])
         if cfg.get("learning_rate") else None),
        ("Batch size", "%s × %s accumulation"
         % (cfg.get("batch_size"), cfg.get("grad_accum"))
         if cfg.get("batch_size") else None),
        ("Sequence length", _num(cfg.get("max_seq_len"))),
        ("Steps", "%s%s" % (_num(summary.get("steps")),
                            " of %s planned" % _num(summary["planned_steps"])
                            if summary.get("planned_steps") else "")
         if summary.get("steps") else None),
    ]
    if not f["scratch"]:
        rows += [
            ("LoRA rank", _num(cfg.get("lora_r"))),
            ("LoRA alpha", _num(cfg.get("lora_alpha"))),
            ("LoRA dropout", _num(cfg.get("lora_dropout"))),
            ("Adapted modules", ", ".join(cfg["target_modules"])
             if isinstance(cfg.get("target_modules"), list) else None),
            ("Base loaded in", "4-bit" if cfg.get("quantization") == "4bit"
             else "8-bit" if cfg.get("quantization") == "8bit" else None),
        ]
    if summary.get("early_stopped"):
        rows.append(("Stopped early", "yes — the held-out loss stopped "
                                      "improving"))
    return rows


def _result_rows(f: dict) -> list[tuple[str, str]]:
    s = f["summary"]
    return [
        ("Training loss", "%s → %s" % (_float(s.get("initial_loss")),
                                       _float(s.get("final_loss")))
         if s.get("final_loss") is not None else None),
        # The number that answers "is this better than last week's". Training
        # loss cannot: it falls just as happily when the model is memorising.
        ("Best held-out loss", _float(s.get("best_val_loss"))),
        ("Took", _duration(s.get("duration_s"))),
    ]


def _evaluations(f: dict) -> list[str]:
    if not f["scores"]:
        return []
    out = ["", "## How it scored", "",
           "Measured in AI Studio against saved prompt sets. Every number is "
           "on prompts the model did not train on.", "",
           "| Prompt set | When | Loss | Token overlap | Exact |",
           "|---|---|---|---|---|"]
    for s in f["scores"]:
        m = s.get("metrics") or {}
        out.append("| %s | %s | %s | %s | %s |" % (
            s.get("eval_name") or "—", _date(s.get("created_at")),
            _float(m.get("expected_loss")), _pct(m.get("f1")),
            _pct(m.get("exact"))))
    return out


def _talking_to_it(f: dict) -> list[str]:
    """The chat format and system prompt, which are part of the model.

    A fine-tune answers the shape it was trained in and nothing else. Leaving
    this out is how a perfectly good model looks broken to the next person:
    prompted with a bare string it has never seen, it says nothing at all.
    """
    cfg = f["cfg"]
    fmt = cfg.get("format") or {}
    mode = fmt.get("mode")
    if not mode and not cfg.get("system_prompt"):
        return []
    out = ["", "## Talking to it", ""]
    if mode in ("chat", "conversations", "messages"):
        out.append("It was trained on conversations, so it expects the chat "
                   "template rather than a bare prompt — "
                   "`tokenizer.apply_chat_template(messages)`.")
    elif mode:
        out.append("It was trained on %s. Prompt it the same way." % mode)
    if prompt := cfg.get("system_prompt"):
        out += ["", "The system prompt it was trained with:", "",
                "```text", prompt.strip(), "```"]
    return out


def _usage(f: dict) -> list[str]:
    load = f["load_id"]
    if f["adapter"]:
        base = f["base"] or "the base model"
        return ["from peft import PeftModel",
                "from transformers import AutoModelForCausalLM, AutoTokenizer",
                "",
                'base = AutoModelForCausalLM.from_pretrained("%s")' % base,
                'model = PeftModel.from_pretrained(base, "%s")' % load,
                'tok = AutoTokenizer.from_pretrained("%s")' % load]
    return ["from transformers import AutoModelForCausalLM, AutoTokenizer",
            "",
            'tok = AutoTokenizer.from_pretrained("%s")' % load,
            'model = AutoModelForCausalLM.from_pretrained("%s")' % load]


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _table(title: str, rows: list[tuple[str, str | None]]) -> list[str]:
    """A two-column table, minus the rows nothing is known for.

    A card full of "| Learning rate | — |" says less than one that leaves the
    line out: an empty row still claims the question was worth asking of this
    run, which for a merge or a from-scratch model it often was not.
    """
    kept = [(k, v) for k, v in rows if v not in (None, "", "—")]
    if not kept:
        return []
    return ["", "## %s" % title, "", "| | |", "|---|---|"] + \
        ["| %s | %s |" % (k, v) for k, v in kept]


def _is_hub_id(name: str | None) -> bool:
    """Whether this names something on the Hub, as opposed to a local path.

    A base that is another of this studio's runs arrives as a runner cache
    directory, and linking to huggingface.co/data/models/job_abc is worse than
    saying nothing.
    """
    return bool(name) and "/" in name and not name.startswith(("/", ".")) \
        and len(name.split("/")) == 2 and " " not in name


def _yaml(s: str) -> str:
    """A YAML scalar that survives colons, quotes and leading dashes."""
    s = str(s).replace("\\", "\\\\").replace('"', '\\"')
    return '"%s"' % s.replace("\n", " ")


def _number(v: float) -> str:
    return ("%d" % v) if float(v) == int(v) else ("%.4f" % v)


def _num(v: object) -> str | None:
    return "{:,}".format(v) if isinstance(v, (int, float)) and v else None


def _float(v: object) -> str | None:
    return "%.4f" % v if isinstance(v, (int, float)) else None


def _pct(v: object) -> str | None:
    return "%.0f%%" % (v * 100) if isinstance(v, (int, float)) else None


def _params(n: object) -> str | None:
    if not isinstance(n, (int, float)) or not n:
        return None
    for unit, size in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= size:
            return "%.2f%s" % (n / size, unit)
    return "%d" % n


def _duration(seconds: object) -> str | None:
    if not isinstance(seconds, (int, float)) or not seconds:
        return None
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return "%dh %dm" % (h, m) if h else "%dm %ds" % (m, s) if m else "%ds" % s


def _date(ts: object) -> str | None:
    if not isinstance(ts, (int, float)) or not ts:
        return None
    return time.strftime("%d %B %Y", time.gmtime(ts))


def dataset_card(dataset: dict, repo_id: str) -> str:
    """The dataset equivalent, unchanged in shape from where it used to live."""
    fmt = dataset.get("format") or {}
    lines = [
        "---", "tags:", "  - ai-studio", "---", "",
        "# %s" % (dataset.get("name") or repo_id.split("/")[-1]), "",
        "%s rows, prepared with AI Studio." % f"{dataset.get('rows') or 0:,}", "",
        "| | |", "|---|---|",
        "| Rows | %s |" % f"{dataset.get('rows') or 0:,}",
        "| Columns | %s |" % ", ".join(dataset.get("columns") or []),
        "| Source | %s |" % (dataset.get("origin") or dataset.get("source")),
    ]
    if fmt.get("mode"):
        lines.append("| Read as | %s |" % fmt["mode"])
    if dataset.get("notes"):
        lines += ["", dataset["notes"]]
    recipe = dataset.get("recipe") or {}
    if recipe.get("steps"):
        lines += ["", "## How it was made", ""]
        lines += ["- %s" % s for s in recipe["steps"]]
    lines += ["", "```python", "from datasets import load_dataset", "",
              'ds = load_dataset("%s", split="train")' % repo_id, "```", ""]
    return "\n".join(lines)
