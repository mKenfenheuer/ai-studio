"""Folding an adapter back into its base, to get a model that can leave.

A fine-tune here produces an *adapter*: a few megabytes of low-rank matrices
that mean nothing without the multi-gigabyte model they were trained against.
That is exactly the right thing to produce -- it is what makes fine-tuning
cheap, and this studio can serve it by loading both halves.

It is also the wrong thing to hand somebody. "Here is my model" followed by
"you also need to download this other model and apply mine to it with the
right library version" is not a deliverable. Merging computes
`W + BA` once and writes the result, giving one directory that loads with
`from_pretrained` and nothing else.

What it costs, said plainly because the numbers surprise people:

* **Size.** The merged model is the size of the *base*, not of the adapter. A
  12 MB adapter on a 7B model produces 14 GB. The adapter is not made bigger
  by this; the base is simply now included.

* **Precision.** Merging happens in float32 and is saved in the dtype asked
  for. Merging into a 4-bit quantised base is refused rather than approximated
  -- the arithmetic would be done against weights that have already lost the
  precision the adapter was fitted to, and the result is quietly worse than
  serving the two halves separately.

* **Reversibility.** None. The output is a new run; the adapter it came from
  is untouched and still there.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

from common import chat_formats
from runner import artifacts
from runner.capabilities import expert_kernel


def run(cfg: dict, ctx: Any) -> dict:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    source = cfg.get("source_job")
    if not source:
        raise ValueError("No run was given to merge.")
    base_model = cfg.get("base_model")
    dtype_name = cfg.get("dtype") or "float16"
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}.get(dtype_name, torch.float16)

    t0 = time.time()
    ctx.progress(0, 4, stage="loading_model")
    adapter = artifacts.fetch(ctx.controller_url, ctx.runner_token, source,
                              ctx.log)
    if not (adapter / "adapter_config.json").exists():
        raise ValueError(
            "That run did not produce an adapter, so there is nothing to "
            "merge. A model built from scratch is already standalone -- "
            "download it directly.")

    conf = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    if not base_model:
        base_model = conf.get("base_model_name_or_path")
    if not base_model:
        raise ValueError(
            "The adapter does not record which model it was trained on, and "
            "none was given. Merging needs the base it was fitted to.")

    # A base that is itself one of this studio's runs, fetched the same way.
    if base_job := cfg.get("base_model_job"):
        fetched = artifacts.fetch(ctx.controller_url, ctx.runner_token,
                                  base_job, ctx.log)
        if not (fetched / "adapter_config.json").exists():
            base_model = str(fetched)

    ctx.log("Merging into %s. The result is a complete model of that size, "
            "not the size of the adapter -- the base is now part of it."
            % base_model)

    extra = {}
    if (kernel := expert_kernel(ctx.capabilities)):
        extra["experts_implementation"] = kernel

    def load(as_dtype):
        try:
            return AutoModelForCausalLM.from_pretrained(
                base_model, dtype=as_dtype, token=ctx.hf_token, **extra)
        except (TypeError, ValueError) as e:
            if not extra or "experts_implementation" not in str(e):
                raise
            return AutoModelForCausalLM.from_pretrained(
                base_model, dtype=as_dtype, token=ctx.hf_token)

    # This runs on the processor, not the card: merging is one addition per
    # weight and needs no GPU, but it does need the whole model in system
    # memory. In float32 that is four bytes a parameter -- 28 GB for a 7B
    # model -- which many machines running this do not have.
    try:
        model = load(torch.float32)
    except (MemoryError, RuntimeError, OSError) as e:
        if "memory" not in str(e).lower() and not isinstance(e, MemoryError):
            raise
        ctx.log("Not enough memory to merge in full precision (%s), so the "
                "arithmetic is being done in %s instead. Small adapter "
                "updates can round away at that precision, which makes the "
                "merged model slightly weaker than serving the adapter and "
                "its base separately." % (type(e).__name__, dtype_name), "warn")
        model = load(dtype)

    # float32 for the arithmetic itself. `W + BA` in float16 rounds every
    # update the adapter makes to the nearest representable value at the
    # *base* weight's magnitude, which for small adapter deltas is often zero
    # -- a merge that quietly does nothing. It is cast down once, afterwards.
    ctx.progress(1, 4, stage="loading_model")
    ctx.log("Applying the adapter…")
    model = PeftModel.from_pretrained(model, str(adapter))

    ctx.progress(2, 4, stage="saving")
    merged = model.merge_and_unload()
    merged = merged.to(dtype)

    out_dir = Path(ctx.workdir) / "model"
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.config.use_cache = True
    merged.save_pretrained(str(out_dir), safe_serialization=True)

    # The tokenizer from the fine-tune where there is one -- it may carry
    # tokens the base does not, and shipping the base's would produce a model
    # that cannot read its own chat template.
    tok_src = str(adapter) if (adapter / "tokenizer_config.json").exists() \
        else base_model
    tok = AutoTokenizer.from_pretrained(tok_src, token=ctx.hf_token)
    tok.save_pretrained(str(out_dir))
    # In both places a reader looks, whatever this version of transformers
    # decided to write. See chat_formats.stamp_into.
    if put := chat_formats.stamp_into(out_dir, getattr(tok, "chat_template", None)):
        ctx.log("Chat template written into %s, so every tool that reads a "
                "model finds it." % " and ".join(put))
    ctx.log("Tokenizer taken from %s."
            % ("the fine-tune" if tok_src == str(adapter) else "the base model"))

    params = sum(p.numel() for p in merged.parameters())
    summary = {
        "kind": "merge_adapter",
        "merged_from": source,
        "base_model": base_model,
        "params_total": params,
        "dtype": dtype_name,
        "duration_s": round(time.time() - t0, 1),
    }
    (out_dir / "ai_studio_summary.json").write_text(json.dumps(summary, indent=2))
    _write_readme(out_dir, summary, cfg)

    ctx.progress(3, 4, stage="saving")
    archive = Path(ctx.workdir) / "model.zip"
    shutil.make_archive(str(archive.with_suffix("")), "zip", root_dir=out_dir)
    summary["artifact_path"] = str(archive)
    summary["artifact_size"] = archive.stat().st_size
    ctx.progress(4, 4, stage="saving")
    ctx.log("Done. %s parameters, %.1f GB on disk. This is a complete model: "
            "it loads with from_pretrained and needs nothing else."
            % (f"{params:,}", archive.stat().st_size / 1024 ** 3))
    return summary


def _write_readme(out_dir: Path, summary: dict, cfg: dict) -> None:
    (out_dir / "README.md").write_text(
        "# Merged model from AI Studio\n\n"
        "`%s` with a fine-tuned adapter folded in. Nothing else is needed to "
        "run it.\n\n"
        "| | |\n|---|---|\n| Parameters | %s |\n| Precision | %s |\n\n"
        "```python\n"
        "from transformers import AutoModelForCausalLM, AutoTokenizer\n\n"
        "tok = AutoTokenizer.from_pretrained(\".\")\n"
        "model = AutoModelForCausalLM.from_pretrained(\".\")\n"
        "```\n\n"
        "The adapter this was made from is still separate in the studio, and "
        "is a few megabytes rather than the size of this. Prefer it if you "
        "are staying inside the studio; prefer this if the model has to go "
        "somewhere else.\n"
        % (summary.get("base_model", "a base model"),
           f"{summary.get('params_total', 0):,}", summary.get("dtype")),
        encoding="utf-8")
