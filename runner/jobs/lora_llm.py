"""LoRA / QLoRA fine-tuning for causal language models.

A hand-written training loop rather than transformers' Trainer. Three reasons:
per-step metric streaming to the controller, cancellation that takes effect
within one step, and immunity to Trainer's API churn across releases.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

from common import chat_formats, conversation
from common.formatting import (conversation_style, detect_format,
                               format_example)
from runner import artifacts, checkpoints, earlystop
from runner.capabilities import expert_kernel

from . import merge, source

# How much of the training set is held back to measure honestly. A fine-tune
# had no held-out set at all until now, which meant the only number on screen
# was the one that goes down when a model memorises as readily as when it
# learns. Capped as well as proportioned: 5% of a 200k-row dataset is 10k rows
# nobody needs to evaluate on, and every one of them is an example not trained.
VAL_FRACTION = 0.05
VAL_ROWS_MAX = 256
VAL_ROWS_MIN = 8

# Split names that mean "nothing trains on this". Checked in this order, so a
# dataset with both a validation and a test split measures on the validation
# one and keeps the test set for later.
HELD_OUT_SPLITS = ("validation", "test", "eval", "dev", "val", "holdout")

# LoRA adapts attention (and often MLP) projections. Names differ per
# architecture, so we match against what the model actually contains rather
# than hardcoding one family's naming.
_TARGET_CANDIDATES = [
    ["q_proj", "k_proj", "v_proj", "o_proj"],        # Llama, Mistral, Qwen, SmolLM
    ["c_attn", "c_proj"],                            # GPT-2, GPT-J
    ["query_key_value", "dense"],                    # Falcon, Bloom
    ["Wqkv", "out_proj"],                            # MPT, Phi
    ["qkv_proj", "o_proj"],                          # Phi-3
]


# The router. Never adapted, whatever else is: it decides which expert each
# token reaches, it is a single tiny matrix, and moving it during a small
# fine-tune re-routes tokens to experts that were never trained on them.
_ROUTER_NAMES = {"gate", "router", "wg", "gate_proj_router"}

_MOE_CONFIG_KEYS = ("num_local_experts", "num_experts", "n_routed_experts",
                    "moe_num_experts")


class Cancelled(Exception):
    """Raised when the controller asks for the job to stop."""


def _fmt(n: int) -> str:
    if n >= 1e9:
        return "%.1fB" % (n / 1e9)
    if n >= 1e6:
        return "%.0fM" % (n / 1e6)
    return "%.0fK" % (n / 1e3)


def param_count(params) -> int:
    """How many parameters these tensors hold, not how many elements they use.

    A 4-bit weight is stored packed, two values to a byte, so `numel()` reports
    half of what the model actually has -- and bitsandbytes keeps the real
    shape on the tensor's quant_state precisely because the tensor no longer
    shows it. Counting the stored elements recorded a Mistral-7B fine-tuned in
    4-bit as a 3.8B model.

    That number is not decoration. It is what every memory guard in this studio
    compares against, and a size that is wrong by half is worse than a size
    that is missing: a missing one is visibly unknown, while this one sails
    through the check and fails on the card.
    """
    total = 0
    for p in params:
        shape = getattr(getattr(p, "quant_state", None), "shape", None)
        total += math.prod(tuple(shape)) if shape else p.numel()
    return int(total)


def moe_report(model) -> dict | None:
    """Whether this base model is a mixture of experts, and how large a one."""
    conf = getattr(model, "config", None)
    if conf is None:
        return None
    experts = next((int(getattr(conf, k)) for k in _MOE_CONFIG_KEYS
                    if getattr(conf, k, None)), 0)
    if experts < 2:
        return None
    active = int(getattr(conf, "num_experts_per_tok", 0)
                 or getattr(conf, "moe_top_k", 0) or 0)
    expert_params = param_count(p for n, p in model.named_parameters()
                                if ".experts." in n)
    total = param_count(model.parameters())
    return {
        "experts": experts,
        "experts_per_token": active or None,
        "expert_params": expert_params,
        "total_params": total,
        "expert_share": round(expert_params / max(total, 1), 3),
    }


def expert_targets(model) -> list[str]:
    """Names of expert weights LoRA is actually able to wrap, if any.

    Often none, and that is a fact about the model's layout rather than a
    policy. Transformers 5 stores Mixtral's experts as fused 3-D tensors --
    `experts.gate_up_proj` of shape (experts, 2*ffn, width) inside a single
    `MixtralExperts` module -- so there is no `nn.Linear` to attach an adapter
    to. Older releases, and some other families, keep one Linear per expert
    and can be adapted.

    Detected by looking for Linear layers under an `.experts.` path rather
    than by matching known names, because the answer changes with the library
    version. Asserting a name list here would have produced a setting that
    reported success and adapted nothing.
    """
    import torch.nn as nn
    return sorted({n.split(".")[-1] for n, m in model.named_modules()
                   if isinstance(m, nn.Linear) and ".experts." in n
                   and n.split(".")[-1] not in _ROUTER_NAMES})


def _pick_target_modules(model, moe: dict | None = None,
                         adapt_experts: bool = False) -> list[str]:
    present = {name.split(".")[-1] for name, _ in model.named_modules()}

    attention = []
    for group in _TARGET_CANDIDATES:
        hit = [m for m in group if m in present]
        if len(hit) >= 2:
            attention = hit
            break

    if moe:
        # Attention only, unless asked otherwise. In a MoE model the attention
        # projections are shared by every token regardless of routing, so an
        # adapter there is trained by all of the data. An adapter on the
        # experts is trained by only the fraction of tokens routed to that
        # expert -- on a small fine-tuning set, several experts may see almost
        # nothing, and those adapters are then fitted to a handful of examples.
        # It also multiplies the adapter's size by the number of experts.
        targets = list(attention)
        if adapt_experts:
            targets += expert_targets(model)
        return [t for t in targets if t not in _ROUTER_NAMES] or sorted(present)[:4]

    if attention:
        return attention

    # Last resort: every Linear that is not the output head or a router.
    import torch.nn as nn
    names = {n.split(".")[-1] for n, m in model.named_modules()
             if isinstance(m, nn.Linear) and "head" not in n and "lm_head" not in n}
    return sorted(names - _ROUTER_NAMES)[:6]


# How much of the card the weights alone may occupy before training is
# hopeless. Training needs room on top of the weights for activations,
# gradients and optimiser state; past this there is not enough left and the
# allocator spends its time evicting rather than computing.
#
# 0.82 rather than something tighter because LoRA's own extra is small and
# gradient checkpointing keeps activations modest -- the successful runs on a
# 16 GB card sit around 0.35 with 4-bit weights. Anything above 0.82 has never
# been a run that finished.
_WEIGHTS_CEILING = 0.82


def _stamp_chat_template(tok, fmt: dict, cfg: dict, ctx: Any) -> dict:
    """Write the format this run trained with onto the tokenizer, before saving.

    Without this a fine-tune ships whatever template its base model had. That
    is not a cosmetic mismatch: a run trained in ChatML on a Mistral base was
    saving Mistral's own 4,000-character template, so anyone who downloaded the
    model -- or served it with vLLM, or opened it in any tool that reads
    `chat_template` -- prompted it in a format it had never been taught, and
    nothing in the artifact said so. Only this studio's own playground got it
    right, because only the playground had the run's config to read.

    The template is written to the tokenizer, so it lands in
    `tokenizer_config.json` (and `chat_template.jinja`) inside the saved model
    and travels with it everywhere.

    The system prompt goes in too, as a default the template applies when a
    conversation arrives without one. A fine-tune trained under a system prompt
    behaves noticeably differently without it, and that prompt is the part of a
    run nobody writes down.
    """
    template, why = chat_formats.template_for_model(fmt, cfg.get("system_prompt"))
    if not template:
        ctx.log("Saving with %s." % why)
        return {"chat_template": why, "chat_template_written": False}

    tok.chat_template = template
    said = "Writing %s into the model, so it carries the format it was trained "
    said += "in wherever it goes."
    ctx.log(said % why)
    if (cfg.get("system_prompt") or "").strip():
        ctx.log("Its system prompt is written in as a default: send your own "
                "and yours is used, send none and it gets the one it was "
                "trained under.")
    return {"chat_template": why, "chat_template_written": True,
            "default_system_prompt": bool((cfg.get("system_prompt") or "").strip())}


def _check_room_to_train(model, device: str, use_4bit: bool, ctx: Any) -> None:
    """Refuse a run that cannot fit, while refusing is still cheap.

    A model too large for the card does not reliably raise out-of-memory. It
    loads, training starts, and then every backward pass fights the allocator
    for room that is not there -- on ROCm it spills over PCIe and the GPU sits
    at 99% "busy" doing almost no useful memory traffic. What that looks like
    from outside is a run that has been going for thirteen hours and completed
    no steps, which is exactly what it was: a 7B in 16-bit needs about 19.6 GB
    and the card has 16.

    The controller checks this too, from the model's name, and that check is
    the one that stops the job being dispatched at all. This one exists because
    that check can only work when the size is known, and here the weights are
    already on the card and there is nothing left to guess.
    """
    import torch
    if device != "cuda":
        return
    try:
        free, total = torch.cuda.mem_get_info()
    except Exception:  # noqa: BLE001 - a backend that cannot say is not a failure
        return
    if not total:
        return
    used = total - free
    share = used / total
    gb = 1024 ** 3
    ctx.log("The weights take %.1f GB of this card's %.1f GB, leaving %.1f GB "
            "for the training itself." % (used / gb, total / gb, free / gb))
    if share <= _WEIGHTS_CEILING:
        return

    advice = ("Switch this run to 4-bit, which is what the same model needs to "
              "fit here." if not use_4bit else
              "It is already in 4-bit, so the remaining levers are a shorter "
              "sequence length, a smaller batch, or a smaller model.")
    raise ValueError(
        "This model does not leave enough room on this card to train. Its "
        "weights alone occupy %.1f GB of %.1f GB (%.0f%%), and training needs "
        "room on top of that for activations, gradients and the optimiser. %s "
        "Stopping now rather than starting a run that would spend hours "
        "fighting the allocator without completing a single step."
        % (used / gb, total / gb, share * 100, advice))


def _last_layers(model, n: int) -> list[int] | None:
    """Indexes of the last `n` decoder layers, or None for all of them."""
    if not n or n <= 0:
        return None
    total = None
    for attr in ("num_hidden_layers", "n_layer", "num_layers"):
        total = getattr(getattr(model, "config", None), attr, None)
        if total:
            break
    if not total or n >= total:
        return None
    return list(range(total - n, total))


def _in_layers(param_name: str, layers: list[int]) -> bool:
    """Whether a parameter belongs to one of these decoder layers."""
    import re as _re
    m = _re.search(r"\.(?:layers|h|blocks)\.(\d+)\.", param_name)
    return bool(m) and int(m.group(1)) in layers


def _versions() -> dict:
    """What this run was trained with.

    "The same settings" is not the same run if the libraries underneath moved:
    a transformers release that changes a chat template, or a torch release
    that changes an attention kernel, changes the result while every number on
    the page stays identical. Recorded so a comparison two months apart can be
    told apart from a comparison of two models.
    """
    out: dict = {}
    for name in ("torch", "transformers", "peft", "datasets", "accelerate"):
        try:
            out[name] = __import__(name).__version__
        except Exception:  # noqa: BLE001 - absent is an answer
            pass
    return out


def run(cfg: dict, ctx: Any) -> dict:
    """Execute a fine-tune. `ctx` supplies log/metric/progress/cancel hooks."""
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Not cfg["base_model"]. A run built on a model this studio trained from
    # scratch has no Hub id at all -- the base is the artifact fetched below,
    # and the controller deliberately records only `base_model_job` for it.
    # Reading the key outright raised KeyError('base_model') here, thirteen
    # lines before the code that would have filled it in, and the run died in
    # a tenth of a second with nothing but the word "base_model" to show.
    base_model = cfg.get("base_model")
    out_dir = Path(ctx.workdir) / "adapter"
    caps = ctx.capabilities

    # A model this studio produced, used as the base for a new run. Two
    # different things arrive here wearing the same setting, and they are told
    # apart by what is actually in the directory rather than by what the
    # controller claimed: a complete model becomes the base, and an adapter
    # becomes an adapter to carry on training on top of the base it was built
    # for. Guessing from the job kind would be one more thing to keep in step.
    resume = checkpoints.peek(ctx.job_id) \
        if cfg.get("checkpointing_enabled", True) else None
    load_adapter = Path(resume["path"]) / "adapter" if resume else None
    if base_job := cfg.get("base_model_job"):
        fetched = artifacts.fetch(ctx.controller_url, ctx.runner_token,
                                  base_job, ctx.log)
        if (fetched / "adapter_config.json").exists():
            if load_adapter is None:
                load_adapter = fetched
                ctx.log("Carrying on from the adapter that run produced, "
                        "rather than starting a new one. It keeps everything "
                        "it already learned and adds this data on top.")
        else:
            base_model = str(fetched)
            ctx.log("Fine-tuning a model this studio built, not one from "
                    "Hugging Face.")

    # Said here rather than left to `from_pretrained(None)`, which reports it
    # as a type error about a path it was never given.
    if not base_model:
        raise ValueError(
            "This run has no base model to train on. It is an adapter, which "
            "needs the model it was built for, and that model is not recorded "
            "on the run it continues.")

    # ---- resolve settings against what this machine can actually do ----
    dtype_name = cfg.get("dtype") or caps.get("recommended_dtype", "float32")
    use_4bit = bool(cfg.get("quantization") == "4bit")
    if use_4bit and not caps.get("quantization", {}).get("4bit"):
        ctx.log("4-bit was requested but this runner has no working 4-bit "
                "support; continuing in %s instead." % dtype_name, "warn")
        use_4bit = False

    max_seq = int(cfg.get("max_seq_len") or 512)
    cap_seq = caps.get("max_recommended_seq_len")
    if cap_seq and max_seq > cap_seq:
        ctx.log("Sequence length %d exceeds what this GPU handles comfortably "
                "without flash attention; capping to %d." % (max_seq, cap_seq), "warn")
        max_seq = cap_seq

    device = "cuda" if caps["backend"] in ("cuda", "rocm") else (
        "mps" if caps["backend"] == "mps" else "cpu")
    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                   "float32": torch.float32}.get(dtype_name, torch.float32)
    if device == "cpu":
        torch_dtype = torch.float32

    ctx.log("Loading tokenizer and base model: %s" % base_model)
    ctx.progress(0, 0, stage="loading_model")

    tok = AutoTokenizer.from_pretrained(base_model, token=ctx.hf_token, trust_remote_code=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    load_kwargs: dict = {"token": ctx.hf_token}
    if use_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        load_kwargs["device_map"] = {"": 0}
    else:
        load_kwargs["dtype"] = torch_dtype

    # A mixture-of-experts base model needs the expert dispatch this backend
    # can actually run -- see capabilities.expert_kernel for what goes wrong
    # otherwise. Passed only when it is needed, and retried without it if the
    # installed transformers is too old to know the argument, so that a
    # dense model on an old library is never affected by any of this.
    kernel = expert_kernel(caps)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, **({"experts_implementation": kernel} if kernel else {}),
            **load_kwargs)
    except (TypeError, ValueError) as e:
        if not kernel or "experts_implementation" not in str(e):
            raise
        model = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)
    if not use_4bit:
        model = model.to(device)
    model.config.use_cache = False

    if cfg.get("gradient_checkpointing", True) and device != "cpu":
        # Trades ~30% speed for a large activation-memory saving. Essential on
        # cards without flash attention, where activations dominate.
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    moe = moe_report(model)
    adapt_experts = bool(cfg.get("adapt_experts"))
    if moe:
        ctx.log("This is a mixture-of-experts model: %d experts per block%s, "
                "and %.0f%% of its weight sits in them."
                % (moe["experts"],
                   ", %d used per token" % moe["experts_per_token"]
                   if moe["experts_per_token"] else "",
                   moe["expert_share"] * 100))
        ctx.log("Every expert has to be held in memory even though each token "
                "only passes through a few, so this needs the memory of its "
                "full %s parameters, not of the fraction that does the work."
                % _fmt(moe["total_params"]))
        if adapt_experts and not expert_targets(model):
            # Said out loud rather than quietly ignored. The setting was asked
            # for, cannot be honoured on this model, and a run that reports
            # "training the experts" while training only attention would be a
            # lie that nothing downstream could catch.
            adapt_experts = False
            ctx.log("You asked to adapt the experts, and this model does not "
                    "allow it: its experts are stored as one fused block of "
                    "weights per layer rather than as separate layers, and "
                    "there is nothing for an adapter to attach to. Training "
                    "attention only instead.", "warn")
        if adapt_experts:
            ctx.log("Adapting the experts as well as attention, as asked. Each "
                    "expert's adapter only learns from the tokens routed to "
                    "that expert, so this needs noticeably more data than the "
                    "same fine-tune on a dense model.", "warn")
        else:
            ctx.log("Fine-tuning attention only. Attention is shared by every "
                    "token whatever the router decides, so all of your data "
                    "trains all of the adapter. The router itself is left "
                    "alone -- moving it sends tokens to experts that were "
                    "never trained on them.")

    # How the weights change: an adapter (LoRA, the default), an adapter that
    # also learns magnitudes (DoRA), or every weight (full). And how much of
    # the model: everything, or only the last N layers -- the layers nearest
    # the output are where a task lives, and freezing the rest is how a
    # large model is fine-tuned on a small card.
    method = str(cfg.get("method") or "lora").lower()
    if method not in ("lora", "dora", "full"):
        raise ValueError("The method is lora, dora or full, not %r." % method)
    train_layers = int(cfg.get("train_layers") or 0)   # 0: all of them
    if method == "full" and cfg.get("checkpointing_enabled", True):
        # A checkpoint here is an adapter directory and a resume loads it as
        # one; a full fine-tune has no adapter, and a checkpoint of the
        # whole model is a second copy of it every few hundred steps. Off,
        # and said, rather than a resume that fails an hour in.
        cfg["checkpointing_enabled"] = False
        ctx.log("Checkpoints are not kept for a full fine-tune yet: a stop "
                "keeps what was trained, but the run cannot be resumed from "
                "partway.")

    if load_adapter is not None and (load_adapter / "adapter_config.json").exists():
        from peft import PeftModel
        # is_trainable is the whole difference between continuing a fine-tune
        # and looking at one. Without it PEFT loads the adapter frozen, the
        # optimiser is handed an empty parameter list, and the run completes
        # having changed nothing at all -- successfully, and to no effect.
        model = PeftModel.from_pretrained(model, str(load_adapter),
                                          is_trainable=True)
        targets = list(getattr(model.peft_config.get("default"),
                               "target_modules", []) or [])
        ctx.log("Continuing an existing adapter on: %s"
                % (", ".join(sorted(targets)) or "its recorded layers"))
    elif method == "full":
        # Every weight trains. Sixteen bytes a parameter with AdamW in mixed
        # precision -- the weights, their gradients, and two optimiser states
        # in fp32 -- against two for a LoRA run, which is why the fit check
        # asks about the method. Only worth it with a lot of data and a card
        # to match; the wizard says so.
        if use_4bit:
            raise ValueError(
                "Full fine-tuning cannot train a 4-bit base: the quantised "
                "weights have no gradient. Choose LoRA, or 16-bit.")
        targets = ["every weight"]
        last_n = _last_layers(model, train_layers)
        frozen = 0
        for name, p in model.named_parameters():
            p.requires_grad = last_n is None or _in_layers(name, last_n) \
                or "lm_head" in name or name.endswith("norm.weight")
            frozen += 0 if p.requires_grad else 1
        ctx.log("Full fine-tuning%s." % (
            " of the last %d layers; the rest is frozen" % train_layers
            if last_n is not None else ""))
    else:
        targets = cfg.get("target_modules") or _pick_target_modules(
            model, moe, adapt_experts)
        last_n = _last_layers(model, train_layers)
        ctx.log("Applying %s to: %s%s" % (
            "DoRA" if method == "dora" else "LoRA", ", ".join(targets),
            " in the last %d layers only" % train_layers if last_n is not None else ""))
        lconf = LoraConfig(
            r=int(cfg.get("lora_r", 16)),
            lora_alpha=int(cfg.get("lora_alpha", 32)),
            lora_dropout=float(cfg.get("lora_dropout", 0.05)),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=targets,
            # DoRA: the adapter learns a direction and the magnitude is
            # learned separately, which closes most of the gap to full
            # fine-tuning at the same rank. Costs a little more per step.
            use_dora=(method == "dora"),
            layers_to_transform=last_n,
        )
        model = get_peft_model(model, lconf)
    trainable = param_count(p for p in model.parameters() if p.requires_grad)
    total = param_count(model.parameters())
    ctx.log("Training %s of %s parameters (%.2f%%)"
            % (f"{trainable:,}", f"{total:,}", 100 * trainable / max(total, 1)))
    ctx.emit_meta({"trainable_params": trainable, "total_params": total,
                   "target_modules": targets, "dtype": dtype_name,
                   "quantized": use_4bit, "max_seq_len": max_seq,
                   "moe": moe, "adapt_experts": adapt_experts if moe else None})
    _check_room_to_train(model, device, use_4bit, ctx)

    # ---- dataset -------------------------------------------------------
    # One seed, used by everything that draws a random number, so a run can be
    # repeated. Fine-tuning had none at all: the held-out slice was cut with a
    # hardcoded 1234 and the batch order was unseeded, so "the same run twice"
    # was two different runs and comparing them measured the noise.
    seed = int(cfg.get("seed") or 1234)
    import random as _random
    _random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as _np
        _np.random.seed(seed)
    except Exception:  # noqa: BLE001 - numpy is present in practice
        pass

    ctx.progress(0, 0, stage="loading_dataset")
    ds_name = cfg["dataset"]
    ctx.log("Loading dataset: %s" % (cfg.get("dataset_label") or ds_name))
    held_raw = None
    held_split = ""
    if cfg.get("dataset_is_local"):
        ds = load_dataset("json", data_files=source.local_copy(cfg, ctx),
                          split="train")
        # A studio dataset is one file holding every split, with the split
        # named on each row. "train" is the whole file for a dataset that has
        # no splits, so the filter only bites where splits actually exist.
        want = (cfg.get("dataset_split") or "").strip()
        if want and "split" in (ds.column_names or []):
            names = {(r or "train") for r in ds["split"]}
            # The held-out split, if the dataset has one. Somebody who cuts a
            # validation set in the dataset editor has said exactly what they
            # want measured on; this run used to ignore that entirely, train on
            # the training split, and then carve a *second* validation set out
            # of it -- so the rows deliberately held back were neither trained
            # on nor measured on, which is the worst of both.
            #
            # An explicit empty string means "carve your own"; an absent key
            # means "work it out", so runs created before this existed get the
            # right behaviour too.
            evs = cfg.get("dataset_eval_split")
            if evs is None:
                evs = next((n for n in HELD_OUT_SPLITS
                            if n in names and n != want), "")
            evs = (evs or "").strip()
            before = len(ds)
            if evs and evs in names:
                held_split = evs
                held_raw = ds.filter(lambda r: (r.get("split") or "train") == evs)
                ctx.log("Measuring on the %s split of your dataset: %s rows "
                        "the model never sees." % (evs, f"{len(held_raw):,}"))
            ds = ds.filter(lambda r: (r.get("split") or "train") == want)
            ctx.log("Training on the %s split: %s of %s rows."
                    % (want, f"{len(ds):,}", f"{before:,}"))
            if not len(ds):
                raise ValueError(
                    "That dataset has no rows in a split called %r. Check the "
                    "split chosen for this run." % want)
    else:
        ds = load_dataset(ds_name, cfg.get("dataset_config") or None,
                          split=cfg.get("dataset_split") or "train",
                          token=ctx.hf_token)

    if (limit := cfg.get("max_samples")):
        ds = ds.select(range(min(int(limit), len(ds))))

    fmt = dict(cfg.get("format") or {"mode": "auto"})

    # The model's own chat template, when the plan asked for it. Pulled off the
    # tokenizer here and rendered through the shared Jinja code rather than
    # through apply_chat_template, so that what trains is byte-for-byte what
    # the preview showed. Two renderers would be two chances to differ.
    if fmt.get("use_model_template") and not fmt.get("chat_template"):
        template = getattr(tok, "chat_template", None)
        if isinstance(template, dict):
            template = template.get("default") or next(iter(template.values()), None)
        if template:
            fmt["chat_template"] = template
            fmt["specials"] = {
                k: v for k, v in (
                    ("bos_token", tok.bos_token), ("eos_token", tok.eos_token),
                    ("pad_token", tok.pad_token), ("unk_token", tok.unk_token))
                if v}
            ctx.log("Using %s's own chat template." % base_model)
        else:
            ctx.log("%s ships no chat template, so the conversations are "
                    "rendered in a plain readable form instead." % base_model,
                    "warn")

    sample_row = ds[0] if len(ds) else {}

    # "auto" is an instruction to work it out, not a description of anything.
    # Left unresolved on the job, it is also what the playground and the API
    # are handed later -- and they have no dataset to work it out from, so
    # they render a bare question to a model that was trained on
    # "### Instruction: ... ### Response:" and get an empty reply back.
    #
    # The trainer is the one place that can see both the data and the answer,
    # so it resolves the shape here and reports it. What was trained and what
    # is served then come from the same decision rather than from two guesses
    # made with different information.
    if fmt.get("mode", "auto") == "auto" and sample_row:
        detected = detect_format(list(sample_row.keys()), [sample_row])
        detected.pop("confidence", None)
        fmt = {**detected, **{k: v for k, v in fmt.items() if k != "mode"}}
        ctx.log("The shape of this data was worked out rather than given: "
                "reading it as %s. That is recorded on the run, so the "
                "playground speaks to the model the way it was taught."
                % (fmt.get("mode") or "plain text"))

    preview = format_example(sample_row, fmt)
    if not preview:
        raise ValueError(
            "Could not work out how to read this dataset. Its columns are: %s. "
            "Pick which column holds the text on the previous step."
            % ", ".join(map(str, sample_row.keys())))
    ctx.log("Example training text:\n%s" % preview[:600])
    ctx.emit_meta({"dataset_rows": len(ds), "dataset_columns": list(sample_row.keys()),
                   "example_text": preview[:1000],
                   # Sent back so the controller can put it on the job. The
                   # template itself is dropped: it can be tens of kilobytes,
                   # and where it came from -- the model's own tokenizer -- is
                   # already recorded by `use_model_template`.
                   "resolved_format": {k: v for k, v in fmt.items()
                                       if k not in ("chat_template", "specials")}})

    # ---- what the loss is computed over -------------------------------
    #
    # A conversation is not uniformly worth learning from. The user's questions
    # and the tool's answers are *context*: the model is never asked to produce
    # them, and training on them teaches it to write the next question itself,
    # which is why a model trained that way answers and then carries on holding
    # both sides of the conversation.
    #
    # So for conversational data the loss covers the assistant's turns and
    # nothing else, unless the run says otherwise. The boundaries come from
    # `conversation.segments`, which measures them by rendering prefixes rather
    # than by matching on turn markers -- see that function for what happens
    # when a template makes that impossible.
    is_chat = conversation_style(fmt) == "chat"
    train_on = (cfg.get("train_on")
                or (conversation.DEFAULT_TRAIN_ON if is_chat else "all"))
    # Offsets need a fast tokenizer. Nearly every model on the Hub ships one;
    # the few that do not fall back to learning from the whole conversation,
    # which is what this did for everything until now.
    can_mask = train_on != "all" and getattr(tok, "is_fast", False)
    if train_on != "all" and not can_mask:
        ctx.log("This tokenizer cannot report character offsets, so the loss "
                "has to cover the whole conversation rather than only the "
                "assistant's turns.", "warn")
    if can_mask:
        ctx.log("The loss covers the assistant's replies only. The questions "
                "and tool results are still rendered -- the model reads them "
                "-- but it is not asked to learn to write them.")
    inexact = 0

    def tokenize(batch_rows: dict) -> dict:
        nonlocal inexact
        keys = list(batch_rows.keys())
        n = len(batch_rows[keys[0]])
        texts: list[str] = []
        keep: list[list[tuple[int, int]]] = []
        for i in range(n):
            row = {k: batch_rows[k][i] for k in keys}
            spans: list[tuple[int, int]] = []
            if can_mask:
                conv, _ = conversation.repair(conversation.from_row(row, fmt))
                t, spans, exact = conversation.trainable_spans(conv, fmt, train_on)
                if not exact:
                    inexact += 1
            else:
                t = format_example(row, fmt)
            # Every example ends with the end-of-text token. Without it the
            # model learns what a response looks like but never learns that one
            # has *finished*, so at generation time it answers correctly and
            # then keeps going, inventing a follow-up conversation.
            #
            # Unless the format already ended it, which every chat format does
            # -- ChatML closes the last turn with <|im_end|>. Appending
            # unconditionally, as this did, put two terminators on every
            # conversational example and taught the model to emit the stop
            # token twice.
            if t:
                if not t.rstrip().endswith(tok.eos_token):
                    # The terminator is part of the last thing the model has to
                    # produce, so it belongs *inside* the final trained span.
                    # Leaving it outside is the same failure in a subtler form:
                    # the model would read an ending it is never scored on.
                    if spans and spans[-1][1] == len(t):
                        spans[-1] = (spans[-1][0], len(t) + len(tok.eos_token))
                    t += tok.eos_token
                texts.append(t)
            else:
                texts.append("")
            keep.append(spans)

        # No padding here. Every example used to be padded to the full
        # context, so a short instruction trained on a batch that was seven
        # eighths padding -- paid for in memory and arithmetic, then discarded
        # by the attention mask. Each batch is padded to its own longest row
        # in `collate` below, at load time.
        enc = tok(texts, truncation=True, max_length=max_seq,
                  padding=False, return_tensors=None,
                  return_offsets_mapping=can_mask)
        offsets = enc.pop("offset_mapping", None)

        labels = []
        for i, ids in enumerate(enc["input_ids"]):
            mask = enc["attention_mask"][i]
            spans = keep[i]
            row_labels = []
            for j, token_id in enumerate(ids):
                # Padding is not data. Scoring it taught the model that the
                # most likely thing after an answer is another end-of-text,
                # forty times over -- and on a batch padded to the full context
                # it was most of the loss.
                if not mask[j]:
                    row_labels.append(-100)
                    continue
                if not spans:
                    row_labels.append(token_id)
                    continue
                start, end = offsets[i][j]
                # A zero-width offset is a special token the tokenizer added
                # itself. It belongs to whichever span contains it, and a
                # bare `start < end` test would drop the very tokens that mark
                # the end of an assistant turn.
                inside = any(s <= start and (end or start + 1) <= e
                             for s, e in spans)
                row_labels.append(token_id if inside else -100)
            labels.append(row_labels)
        enc["labels"] = labels
        return enc

    held_ds = None
    if held_raw is not None and len(held_raw):
        held_ds = held_raw.map(tokenize, batched=True, batch_size=64,
                               remove_columns=held_raw.column_names,
                               desc="Tokenizing the held-out split")
        held_ds.set_format(type=None,
                           columns=["input_ids", "attention_mask", "labels"])
    ds = ds.map(tokenize, batched=True, batch_size=64,
                remove_columns=ds.column_names, desc="Tokenizing")
    if inexact:
        # Not a failure, but not something to pass over either: those rows
        # trained on every token including the questions, which is a different
        # thing from what the rest of the run did.
        ctx.log("%d row(s) had turn boundaries this chat template does not "
                "allow to be measured -- some templates rewrite earlier turns "
                "when a later one arrives. Those rows learned from the whole "
                "conversation." % inexact, "warn")
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    val_ds = None
    if held_ds is None:
        held_split = ""
    if held_ds is not None:
        # The split the dataset already holds back. Nothing is carved out of
        # the training data, because the person who made this dataset already
        # said which rows are the honest ones to measure on.
        val_ds = held_ds
        if len(val_ds) > VAL_ROWS_MAX:
            # Every evaluation costs a forward pass over all of these, several
            # times a run. Beyond a couple of hundred rows the number stops
            # moving and the run just gets slower.
            val_ds = val_ds.select(range(VAL_ROWS_MAX))
            ctx.log("Measuring on the first %d rows of that split; the rest "
                    "are left for a proper scoring afterwards." % VAL_ROWS_MAX)
    want_val = min(VAL_ROWS_MAX,
                   int(len(ds) * float(cfg.get("val_fraction", VAL_FRACTION))))
    if val_ds is not None:
        pass
    elif want_val >= VAL_ROWS_MIN and len(ds) - want_val >= 16:
        parts = ds.train_test_split(test_size=want_val, seed=seed)
        ds, val_ds = parts["train"], parts["test"]
        ctx.log("Holding back %d of the %d examples to measure on. The model "
                "never trains on these, so their loss is the one that tells "
                "you whether it is learning your task or memorising your "
                "examples." % (want_val, want_val + len(ds)))
    elif len(ds) >= 16:
        ctx.log("This dataset is too small to hold any of it back for "
                "measurement, so there is only a training loss to go on. A "
                "falling training loss on %d examples can mean memorisation "
                "rather than learning." % len(ds), "warn")

    # ---- training ------------------------------------------------------
    from torch.utils.data import DataLoader

    bs = int(cfg.get("batch_size", 1))
    accum = int(cfg.get("grad_accum", 8))
    epochs = float(cfg.get("epochs", 1))
    lr = float(cfg.get("learning_rate", 2e-4))

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)

    def collate(rows: list) -> dict:
        """One batch, padded to its own longest row and no further.

        Padding is not data: the mask hides it from attention and -100
        hides it from the loss, so the only thing a padded position ever
        costs is the arithmetic spent on it -- which, padded to the full
        context, was most of every step.
        """
        def as_list(v):
            return v.tolist() if hasattr(v, "tolist") else list(v)
        width = max(len(as_list(r["input_ids"])) for r in rows)
        out = {"input_ids": [], "attention_mask": [], "labels": []}
        for r in rows:
            ids = as_list(r["input_ids"])
            mask = as_list(r["attention_mask"])
            labels = as_list(r["labels"])
            gap = width - len(ids)
            out["input_ids"].append(ids + [pad_id] * gap)
            out["attention_mask"].append(mask + [0] * gap)
            out["labels"].append(labels + [-100] * gap)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in out.items()}

    # Seeded, so the batches come in the same order on a second run.
    _gen = torch.Generator()
    _gen.manual_seed(seed)
    loader = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=False,
                        generator=_gen, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=bs,
                            collate_fn=collate) if val_ds is not None else None
    steps_per_epoch = max(1, math.ceil(len(loader) / accum))
    total_steps = int(cfg.get("max_steps") or max(1, int(steps_per_epoch * epochs)))
    eval_every = int(cfg.get("eval_every") or max(5, total_steps // 20))

    params = [p for p in model.parameters() if p.requires_grad]
    if caps.get("quantization", {}).get("optim_8bit") and cfg.get("optim_8bit", True):
        import bitsandbytes.optim as bopt
        opt = bopt.AdamW8bit(params, lr=lr)
        ctx.log("Using 8-bit AdamW to save memory.")
    else:
        opt = torch.optim.AdamW(params, lr=lr)

    warmup = max(1, int(total_steps * 0.03))

    def lr_at(s: int) -> float:
        if s < warmup:
            return lr * s / warmup
        prog = (s - warmup) / max(1, total_steps - warmup)
        return lr * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))

    # fp16 needs loss scaling to stop small gradients flushing to zero.
    # bf16 and fp32 do not, and enabling it there is pure overhead.
    use_scaler = (torch_dtype == torch.float16 and device == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    if device == "cuda":
        # Per-process high-water mark, and the agent runs many jobs in one
        # process. Reset it, or this run reports the previous run's peak.
        torch.cuda.reset_peak_memory_stats()

    prior = _resume_state(resume, ctx, opt, scaler, use_scaler, torch) if resume else {}
    start_step = int(prior.get("step") or 0)
    if start_step:
        ctx.log("Carrying on from the checkpoint at step %d of %d."
                % (start_step, total_steps))
        # Said plainly rather than left for someone to deduce from a chart.
        # The weights and the optimiser come back exactly; the order the
        # examples arrive in does not, because the loader reshuffles every
        # epoch anyway. It changes nothing about the result and it would be
        # dishonest to imply the resume is bit-for-bit.
        ctx.log("The examples will be shuffled fresh rather than continuing "
                "the exact order of the interrupted attempt. The adapter and "
                "the optimiser resume exactly; only the order differs.")

    ctx.log("Starting training: %d steps, batch %d x %d accumulation, lr %.2e"
            % (total_steps, bs, accum, lr))
    ctx.progress(start_step, total_steps, stage="training")

    @torch.no_grad()
    def evaluate() -> float | None:
        """Loss on the examples the model never trains on."""
        if val_loader is None:
            return None
        model.eval()
        total, n = 0.0, 0
        for vb in val_loader:
            vb = {k: v.to(device) for k, v in vb.items()}
            with torch.amp.autocast("cuda", dtype=torch_dtype,
                                    enabled=device == "cuda"
                                    and torch_dtype != torch.float32):
                total += float(model(**vb).loss)
            n += 1
        model.train()
        return total / max(n, 1)

    model.train()
    step = start_step
    micro = 0
    running: list[float] = []
    first_loss = prior.get("first_loss")
    best_val = prior.get("best_val")
    last_val = None
    prior_seconds = float(prior.get("duration_s") or 0.0)
    t_start = time.time()
    stop = False
    stopped_early = False
    # One-element lists rather than plain names: these are written from inside
    # the loop body and read after it, and a bare assignment there would shadow
    # rather than update if this ever moves into a closure.
    warned_overfit = [False]
    final_val = [None]
    stopper = earlystop.Stopper(
        ctx, int(cfg.get("early_stop_patience") or earlystop.PATIENCE_FINETUNE),
        enabled=bool(cfg.get("early_stop", True)) and val_loader is not None,
        kind="adapter")
    stopper.best = prior.get("best_val")
    stopper.best_step = int(prior.get("best_step") or 0)
    keep_best = bool(cfg.get("checkpointing_enabled", True))
    saver = checkpoints.Saver(
        ctx.job_id, ctx, every_s=float(cfg.get("checkpoint_every_s") or 600),
        enabled=bool(cfg.get("checkpointing_enabled", True)))

    def write_checkpoint(path):
        model.save_pretrained(str(path / "adapter"))
        torch.save({"optimizer": opt.state_dict(),
                    "scaler": scaler.state_dict() if use_scaler else None},
                   str(path / "trainer.pt"))
        (path / "train.json").write_text(json.dumps({
            "first_loss": first_loss, "best_val": stopper.best,
            "best_step": stopper.best_step,
            "duration_s": prior_seconds + (time.time() - t_start),
        }), encoding="utf-8")

    stopper.announce(eval_every)

    while not stop:
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            autocast_on = device == "cuda" and torch_dtype != torch.float32
            with torch.amp.autocast("cuda", dtype=torch_dtype, enabled=autocast_on):
                loss = model(**batch).loss / accum
            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            running.append(loss.item() * accum)
            micro += 1

            if micro % accum:
                continue

            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            if use_scaler:
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            if use_scaler:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1

            window = running[-accum:]
            avg = sum(window) / len(window)
            if first_loss is None:
                first_loss = avg
            done_now = step - start_step
            elapsed = time.time() - t_start

            if step % eval_every == 0 or step == total_steps:
                last_val = evaluate()
                if last_val is not None:
                    best_val = last_val if best_val is None else min(best_val, last_val)
                    final_val[0] = last_val
                    if last_val > best_val * 1.05 and not warned_overfit[0]:
                        warned_overfit[0] = True
                        ctx.log("The held-out loss has started rising while the "
                                "training loss falls. That is the model "
                                "memorising your examples rather than learning "
                                "from them -- the best result was at the low "
                                "point, around step %d. Fewer passes, or more "
                                "data, would help." % stopper.best_step, "warn")
                    verdict = stopper.update(step, last_val)
                    if verdict == "improved" and keep_best and step < total_steps:
                        checkpoints.save_best(
                            ctx.job_id, step,
                            lambda pth: model.save_pretrained(str(pth / "adapter")),
                            {"val_loss": last_val})
                    elif verdict == "stop":
                        stop = True

            ctx.metric(step, {
                "loss": round(avg, 5),
                "val_loss": round(last_val, 5) if last_val is not None else None,
                "perplexity": round(min(math.exp(min(avg, 20)), 1e6), 3),
                "learning_rate": lr_at(step),
                # Rates describe this attempt, not the resumed step number
                # divided by the time since this process started.
                "steps_per_sec": round(done_now / max(elapsed, 1e-6), 3),
                "vram_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)
                if device == "cuda" else None,
                "eta_s": round((total_steps - step) * elapsed / max(done_now, 1)),
            })
            last_val = None
            ctx.progress(step, total_steps, stage="training")

            if saver.due(step):
                saver.write(step, total_steps, write_checkpoint,
                            {"kind": "finetune_llm", "loss": round(avg, 5)})
                ctx.emit_meta({"checkpoint": {"step": step, "total": total_steps}})

            if stop:
                # Set by the stopper: the held-out loss stopped improving.
                # Not `stopped_early`, which means a person pressed Stop and
                # turns the run into a cancelled one -- this run succeeded,
                # sooner than planned.
                break
            if ctx.should_cancel():
                # An adapter stopped partway is still a usable adapter -- less
                # trained than planned, and a great deal better than nothing
                # when the alternative is discarding the run. The user chose
                # which of those they wanted when they pressed Stop.
                if not ctx.should_save():
                    raise Cancelled()
                stopped_early = True
                ctx.log("Stopping at step %d of %d, and keeping the adapter as "
                        "it stands." % (step, total_steps), "warn")
                stop = True
                break
            if step >= total_steps:
                stop = True
                break
        else:
            continue
        break

    # ---- keep the best adapter, which is not always the last one --------
    kept_step = None
    if stopper.should_restore(final_val[0], step):
        best = checkpoints.peek(ctx.job_id, "best") if keep_best else None
        if best and (Path(best["path"]) / "adapter" / "adapter_model.safetensors").exists():
            try:
                from safetensors.torch import load_file
                weights = load_file(str(Path(best["path"]) / "adapter"
                                        / "adapter_model.safetensors"))
                # PEFT names its saved tensors without the wrapper prefix that
                # the live module tree uses, so they are matched by suffix
                # rather than assumed to line up.
                live = dict(model.named_parameters())
                loaded = 0
                for name, tensor in weights.items():
                    key = next((k for k in live
                                if k.endswith(name.replace("base_model.model.", ""))
                                or k == name), None)
                    if key is not None:
                        with torch.no_grad():
                            live[key].copy_(tensor.to(live[key].device,
                                                      live[key].dtype))
                        loaded += 1
                if loaded < len(weights):
                    raise ValueError("only %d of %d adapter tensors matched"
                                     % (loaded, len(weights)))
                stopper.note_kept(final_val[0], step)
                kept_step = stopper.best_step
            except Exception as e:  # noqa: BLE001 - the trained adapter is fine
                ctx.log("The better adapter from step %d could not be read "
                        "(%s), so this run keeps the one it ended with."
                        % (stopper.best_step, e), "warn")

    # ---- save ----------------------------------------------------------
    ctx.progress(step, total_steps, stage="saving")
    ctx.log("Saving the adapter as it stands." if stopped_early
            else "Saving adapter to %s" % out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamped = _stamp_chat_template(tok, fmt, cfg, ctx)
    model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    # In BOTH places a reader looks, whatever this version of transformers
    # decided to write. See chat_formats.stamp_into.
    if put := chat_formats.stamp_into(out_dir, getattr(tok, "chat_template", None)):
        ctx.log("Chat template written into %s, so every tool that reads a "
                "model finds it." % " and ".join(put))


    summary = {
        "kind": "finetune_llm",
        **stamped,
        "final_loss": round(running[-1], 5) if running else None,
        "initial_loss": round(first_loss, 5) if first_loss is not None else None,
        # The number that answers "is this one better than last week's". A
        # training loss cannot, because it falls just as happily when the model
        # is memorising the examples it is being scored on.
        "best_val_loss": round(best_val, 5) if best_val is not None else None,
        # The same number, named. Every page that ranks runs -- compare, a
        # sweep, the champion of a prompt set -- used to reach for
        # best_val_loss by name and assume smaller was better, which is true
        # of a loss and false of an accuracy, a WER-inverse or an F1. A run
        # says what it should be judged by and which way is up, and a
        # classifier's accuracy ranks beside a language model's loss without
        # either page knowing the difference.
        "primary_metric": {"name": "held_out_loss", "label": "Held-out loss",
                           "value": round(best_val, 5), "lower_better": True}
                          if best_val is not None else None,
        "held_out_rows": len(val_ds) if val_ds is not None else 0,
        # Which rows the held-out loss was measured on, because "on unseen
        # examples" means two quite different things: a split somebody
        # deliberately held back, or 5% of the training data taken at random.
        # Only the first is a fair test of anything.
        # What it would take to run this again and get this back.
        "seed": seed,
        "versions": _versions(),
        # Which tokens the loss covered. "On unseen examples" is not the only
        # phrase on this page that meant two quite different things.
        "trained_on": ("the assistant's replies only" if can_mask
                       else "every token, including the questions"),
        "held_out_from": ("the %s split of the dataset" % held_split
                          if held_split else
                          "a slice taken from the training data"
                          if val_ds is not None else None),
        "steps": step,
        "planned_steps": total_steps,
        "stopped_early": stopped_early,
        "resumed_from_step": start_step or None,
        "continued_from": cfg.get("base_model_job"),
        "kept_from_step": kept_step,
        "early_stopped": stopper.stopped,
        "duration_s": round(prior_seconds + (time.time() - t_start), 1),
        "trainable_params": trainable,
        "base_model": base_model,
        "dtype": dtype_name,
        "quantized": use_4bit,
        "moe": moe,
        "method": method,
        "train_layers": train_layers or None,
        # The most memory the card held at once, so the estimate that put
        # this run on this card can be checked against what it actually
        # took. The fit check learns from the pair.
        "peak_vram_gb": (round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)
                         if torch.cuda.is_available() else None),
    }
    (out_dir / "ai_studio_summary.json").write_text(json.dumps(summary, indent=2))
    adapter_zip = artifacts.pack(out_dir, Path(ctx.workdir) / "adapter.zip")

    ctx.log("%s Loss %.4f -> %.4f over %d steps%s."
            % ("Stopped early." if stopped_early else "Training done.",
               summary["initial_loss"] or 0, summary["final_loss"] or 0, step,
               " of the %d planned" % total_steps if stopped_early else ""))
    if summary["best_val_loss"] is not None:
        ctx.log("Best held-out loss %.4f, on %d examples it never trained on. "
                "That is the number to compare against another run."
                % (summary["best_val_loss"], summary["held_out_rows"]))

    # ---- merge, as the last step of this run ---------------------------
    # Not a second run queued afterwards. The base is loaded, the adapter is
    # in memory, and the machine is already this one; a follow-on run threw
    # all three away and downloaded fourteen gigabytes to get them back.
    # Handed over in a box, and this frame lets go of it: the 4-bit path has
    # to give the whole card back before it can read the base again, and a
    # `del` inside the callee would leave this frame's reference holding every
    # weight it was trying to free. Nothing below uses the model.
    model_box = [model]
    del model
    # A full fine-tune saved the whole model above; there is no adapter
    # to fold into anything, and the saved directory is the model.
    merged_zip = None if method == "full" else _merge_here(model_box, tok, cfg, ctx, summary, out_dir,
                             base_model, dtype_name, use_4bit, torch)

    # The merged model is the primary artifact when there is one -- it is what
    # somebody means by "the model" -- and the adapter travels beside it.
    summary["artifact_paths"] = ({"model": str(merged_zip),
                                  "adapter": str(adapter_zip)}
                                 if merged_zip else {"model": str(adapter_zip)})
    summary["artifact_size"] = (merged_zip or adapter_zip).stat().st_size
    return summary


def _merge_here(model_box: list, tok, cfg: dict, ctx: Any, summary: dict,
                adapter_dir: Path, base_model: str, dtype_name: str,
                use_4bit: bool, torch) -> Path | None:
    """Fold this run's adapter into its base and pack the result. Or don't.

    Both artifacts are kept, and each answers a different question. The adapter
    is a few megabytes, is what the studio prefers to carry on training from,
    and is what says "this is a fine-tune of *that*" on the Hub. The merged
    model is the size of the base and loads with `from_pretrained` alone, which
    is what anybody outside this studio needs.

    Never fatal. The adapter is already saved and packed by the time this runs,
    so a merge that will not fit or will not load costs a warning and a
    `merged: False`, not the run.
    """
    summary["merged"] = False
    if not cfg.get("merge_after", True):
        ctx.log("Not merging: this run was asked for the adapter only. It can "
                "still be served here, and needs %s to run anywhere else."
                % (cfg.get("base_model_label") or base_model))
        return None

    model_dir = Path(ctx.workdir) / "model"
    label = cfg.get("base_model_label") or cfg.get("base_model") or base_model
    try:
        # Whether the arithmetic fits in system memory, worked out before
        # anything is moved: on Linux, asking for more than there is does not
        # raise, it gets the process killed. See merge.memory_plan.
        plan = merge.memory_plan(cfg.get("params_b"), dtype_name)
        if plan.get("note"):
            ctx.log(plan["note"], plan.get("level", "info"))
        target = torch.float32 if plan["float32"] else \
            {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}.get(dtype_name, torch.float16)
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                 "float32": torch.float32}.get(dtype_name, torch.float16)

        ctx.log("Merging the adapter into %s, so this run also produces a "
                "model that loads on its own." % label)
        if use_4bit:
            # A 4-bit base cannot be merged into: the weights the adapter was
            # fitted against have already lost the precision it was fitted to.
            # Free the training copy, then re-read the base at full precision
            # -- from the local Hugging Face cache, which this run just filled.
            info = _merge_by_reload(model_box, ctx, adapter_dir, base_model,
                                    dtype_name, model_dir, label, cfg, torch)
        else:
            # The fast path, and the reason this is part of the run: the
            # weights are already here. Moved to the processor first, because
            # holding the merged copy beside the training copy on the card is
            # how a run that fitted ends in an out-of-memory error at the very
            # last step -- and because the addition is done in float32 where
            # there is room for it, small adapter deltas rounding to nothing
            # being the classic quietly-useless merge.
            merged = model_box[0].to("cpu", dtype=target).merge_and_unload()
            model_box.clear()
            merged = merged.to(dtype)
            model_dir.mkdir(parents=True, exist_ok=True)
            merged.config.use_cache = True
            merged.save_pretrained(str(model_dir), safe_serialization=True)
            merge.save_tokenizer(tok, model_dir, ctx, "the fine-tune")
            info = {"params_total": sum(p.numel() for p in merged.parameters()),
                    "dtype": dtype_name, "base_model": label}
            del merged
    except Exception as e:  # noqa: BLE001 - the adapter is safe either way
        ctx.log("The merged copy could not be made (%s: %s), so this run keeps "
                "the adapter alone. It still serves here, and it can be merged "
                "on a larger machine later." % (type(e).__name__, e), "warn")
        return None

    # The merged directory describes the run that made it -- a fine-tune, of
    # that base, in that chat format -- and not a merge of unknown parentage.
    # It is what the serving side reads back out of the cache.
    merged_summary = {**{k: v for k, v in summary.items()
                         if k not in ("artifact_paths", "artifact_size")},
                      "merged": True,
                      "params_total": info.get("params_total"),
                      "base_model": summary.get("base_model")}
    (model_dir / "ai_studio_summary.json").write_text(
        json.dumps(merged_summary, indent=2))
    merge.write_readme(model_dir, {**merged_summary, "base_model": label})

    archive = artifacts.pack(model_dir, Path(ctx.workdir) / "model.zip")
    summary["merged"] = True
    summary["params_total"] = info.get("params_total")
    ctx.log("Done. The merged model is %.1f GB and loads with "
            "from_pretrained; the adapter is kept beside it at %.0f MB."
            % (archive.stat().st_size / 1024 ** 3,
               (Path(ctx.workdir) / "adapter.zip").stat().st_size / 1048576))
    return archive


def _merge_by_reload(model_box: list, ctx: Any, adapter_dir: Path,
                     base_model: str, dtype_name: str, model_dir: Path,
                     label: str, cfg: dict, torch) -> dict:
    """Give the card back, then merge from disk. The 4-bit path."""
    import gc

    model_box.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    ctx.log("This run trained against a 4-bit copy of the base, which cannot "
            "be merged into. Reading the base again at full precision -- from "
            "this machine's cache, not over the network.")
    return merge.merge_into(adapter_dir, base_model, dtype_name, ctx,
                            model_dir, label=label,
                            params_b=cfg.get("params_b"))


def _resume_state(resume: dict, ctx: Any, opt, scaler, use_scaler: bool,
                  torch) -> dict:
    """Read back an interrupted fine-tune. A broken one starts over, loudly."""
    path = Path(resume["path"])
    try:
        out = json.loads((path / "train.json").read_text(encoding="utf-8"))
        # weights_only=False: this holds an optimiser state, not just tensors.
        # Written by this runner into its own data volume minutes ago.
        blob = torch.load(str(path / "trainer.pt"), map_location="cpu",
                          weights_only=False)
        if blob.get("optimizer"):
            opt.load_state_dict(blob["optimizer"])
        if use_scaler and blob.get("scaler"):
            scaler.load_state_dict(blob["scaler"])
    except Exception as e:  # noqa: BLE001 - any failure means "start over"
        ctx.log("The checkpoint for this run could not be read (%s), so it "
                "starts again from the beginning." % e, "warn")
        return {}
    out["step"] = int(resume.get("step") or 0)
    return out
