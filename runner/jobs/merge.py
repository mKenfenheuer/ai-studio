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

    # What the base is CALLED, which is not always where it is. A base that is
    # another run in this studio arrives as a cache directory, and neither a
    # log line nor a model card on the Hub should be naming a path on some
    # runner's disk.
    label = cfg.get("base_model_label") or base_model
    ctx.log("Merging into %s. The result is a complete model of that size, "
            "not the size of the adapter -- the base is now part of it."
            % label)

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
    #
    # Whether this one does is worked out before loading rather than caught
    # afterwards, because on Linux it cannot be caught. Asking for 28 GB on a
    # machine with 24 does not raise MemoryError: the kernel hands over the
    # address space, the process touches it, and the OOM killer sends SIGKILL.
    # There is no exception, no log line and no failed run -- the container
    # goes away mid-merge and the studio reports that the runner vanished.
    # This is the difference between merging on a spare CPU and merging on a
    # training box, so it stopped being a theoretical concern the moment the
    # controller started taking merges.
    plan = _memory_plan(cfg.get("params_b"), dtype_name)
    if plan.get("note"):
        ctx.log(plan["note"], plan.get("level", "info"))
    try:
        model = load(torch.float32 if plan["float32"] else dtype)
    except (MemoryError, RuntimeError, OSError) as e:
        # Still worth keeping: an allocator that does refuse politely, a
        # machine whose limit could not be read, and a base model bigger than
        # the size the run recorded all end up here.
        if "memory" not in str(e).lower() and not isinstance(e, MemoryError):
            raise
        if not plan["float32"]:
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
        "base_model": label,
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


# What merging costs on top of simply holding the weights: PEFT builds `BA`
# for each target module it folds in, and `.to(dtype)` keeps the tensor it is
# converting alongside the one it produces. A tenth, plus a fixed couple of
# gigabytes for the Python process, the tokenizer and the safetensors writer.
_OVERHEAD = 1.12
_SLACK = 2 * 1024 ** 3


def _memory_plan(params_b: float | None, dtype_name: str) -> dict:
    """Whether to merge in float32, decided against memory this machine has.

    Raises when not even the requested precision fits, which is a far better
    outcome than starting: an hour of downloading followed by a process that
    disappears without writing anything tells nobody anything.
    """
    have = _memory_available()
    params = float(params_b or 0) * 1e9
    if not params or not have:
        # Nothing to compare -- the run does not record its size, or the
        # machine will not say what it has. Ask for full precision and let the
        # caller's fallback deal with it, as this did before it could measure.
        return {"float32": True}

    need32 = params * 4 * _OVERHEAD + _SLACK
    if need32 <= have:
        return {"float32": True,
                "note": "Merging in float32. It needs about %s and this "
                        "machine has %s." % (_gb(need32), _gb(have))}

    width = {"float32": 4}.get(dtype_name, 2)
    need = params * width * _OVERHEAD + _SLACK
    if need > have:
        raise ValueError(
            "There is not enough memory here to merge a model this size: "
            "holding %s parameters in %s takes about %s and this machine has "
            "%s. Nothing has been changed -- the adapter and its base are both "
            "still here -- so this can be merged on a larger machine."
            % (f"{int(params):,}", dtype_name, _gb(need), _gb(have)))
    return {
        "float32": False, "level": "warn",
        "note": "Merging in %s rather than float32: full precision would need "
                "about %s and this machine has %s. Small adapter updates can "
                "round away at that precision, which makes the merged model "
                "slightly weaker than serving the adapter and its base "
                "separately." % (dtype_name, _gb(need32), _gb(have)),
    }


def _memory_available() -> int | None:
    """Bytes this process could actually get, or None if it cannot be told.

    Both the host's free memory and the container's own limit can be the thing
    that kills the run, and neither implies the other -- /proc/meminfo is not
    namespaced, so inside a container it describes the machine and says
    nothing about the cap this process is under. The smaller wins.
    """
    limits: list[int] = []
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                limits.append(int(line.split()[1]) * 1024)
                break
    except (OSError, ValueError, IndexError):
        pass
    for cap_file, used_file in (
            ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
            ("/sys/fs/cgroup/memory/memory.limit_in_bytes",
             "/sys/fs/cgroup/memory/memory.usage_in_bytes")):
        try:
            cap = int(Path(cap_file).read_text().strip())
            used = int(Path(used_file).read_text().strip())
        except (OSError, ValueError):
            continue        # "max" for an unlimited cgroup v2, or no cgroup
        if cap < (1 << 62):  # cgroup v1 writes a vast number to mean no limit
            limits.append(max(cap - used, 0))
    return min(limits) if limits else None


def _gb(n: float) -> str:
    return "%.1f GB" % (n / 1024 ** 3)


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
