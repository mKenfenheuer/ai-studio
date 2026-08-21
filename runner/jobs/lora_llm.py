"""LoRA / QLoRA fine-tuning for causal language models.

A hand-written training loop rather than transformers' Trainer. Three reasons:
per-step metric streaming to the controller, cancellation that takes effect
within one step, and immunity to Trainer's API churn across releases.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any, Callable

from common.formatting import format_example

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


class Cancelled(Exception):
    """Raised when the controller asks for the job to stop."""


def _pick_target_modules(model) -> list[str]:
    present = {name.split(".")[-1] for name, _ in model.named_modules()}
    for group in _TARGET_CANDIDATES:
        hit = [m for m in group if m in present]
        if len(hit) >= 2:
            return hit
    # Last resort: every Linear that is not the output head.
    import torch.nn as nn
    names = {n.split(".")[-1] for n, m in model.named_modules()
             if isinstance(m, nn.Linear) and "head" not in n and "lm_head" not in n}
    return sorted(names)[:6]


def run(cfg: dict, ctx: Any) -> dict:
    """Execute a fine-tune. `ctx` supplies log/metric/progress/cancel hooks."""
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_model = cfg["base_model"]
    out_dir = Path(ctx.workdir) / "adapter"
    caps = ctx.capabilities

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

    model = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)
    if not use_4bit:
        model = model.to(device)
    model.config.use_cache = False

    if cfg.get("gradient_checkpointing", True) and device != "cpu":
        # Trades ~30% speed for a large activation-memory saving. Essential on
        # cards without flash attention, where activations dominate.
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    targets = cfg.get("target_modules") or _pick_target_modules(model)
    ctx.log("Applying LoRA to: %s" % ", ".join(targets))
    lconf = LoraConfig(
        r=int(cfg.get("lora_r", 16)),
        lora_alpha=int(cfg.get("lora_alpha", 32)),
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
    )
    model = get_peft_model(model, lconf)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    ctx.log("Training %s of %s parameters (%.2f%%)"
            % (f"{trainable:,}", f"{total:,}", 100 * trainable / max(total, 1)))
    ctx.emit_meta({"trainable_params": trainable, "total_params": total,
                   "target_modules": targets, "dtype": dtype_name,
                   "quantized": use_4bit, "max_seq_len": max_seq})

    # ---- dataset -------------------------------------------------------
    ctx.progress(0, 0, stage="loading_dataset")
    ds_name = cfg["dataset"]
    ctx.log("Loading dataset: %s" % ds_name)
    if cfg.get("dataset_is_local"):
        ds = load_dataset("json", data_files=ds_name, split="train")
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
    preview = format_example(sample_row, fmt)
    if not preview:
        raise ValueError(
            "Could not work out how to read this dataset. Its columns are: %s. "
            "Pick which column holds the text on the previous step."
            % ", ".join(map(str, sample_row.keys())))
    ctx.log("Example training text:\n%s" % preview[:600])
    ctx.emit_meta({"dataset_rows": len(ds), "dataset_columns": list(sample_row.keys()),
                   "example_text": preview[:1000]})

    def tokenize(batch_rows: dict) -> dict:
        keys = list(batch_rows.keys())
        n = len(batch_rows[keys[0]])
        texts = []
        for i in range(n):
            row = {k: batch_rows[k][i] for k in keys}
            t = format_example(row, fmt)
            # Every example ends with the end-of-text token. Without it the
            # model learns what a response looks like but never learns that one
            # has *finished*, so at generation time it answers correctly and
            # then keeps going, inventing a follow-up conversation.
            texts.append((t + tok.eos_token) if t else "")
        enc = tok(texts, truncation=True, max_length=max_seq,
                  padding="max_length", return_tensors=None)
        enc["labels"] = [list(ids) for ids in enc["input_ids"]]
        return enc

    ds = ds.map(tokenize, batched=True, batch_size=64,
                remove_columns=ds.column_names, desc="Tokenizing")
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # ---- training ------------------------------------------------------
    from torch.utils.data import DataLoader

    bs = int(cfg.get("batch_size", 1))
    accum = int(cfg.get("grad_accum", 8))
    epochs = float(cfg.get("epochs", 1))
    lr = float(cfg.get("learning_rate", 2e-4))

    loader = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=False)
    steps_per_epoch = max(1, math.ceil(len(loader) / accum))
    total_steps = int(cfg.get("max_steps") or max(1, int(steps_per_epoch * epochs)))

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

    ctx.log("Starting training: %d steps, batch %d x %d accumulation, lr %.2e"
            % (total_steps, bs, accum, lr))
    ctx.progress(0, total_steps, stage="training")

    model.train()
    step = 0
    micro = 0
    running: list[float] = []
    t_start = time.time()
    stop = False

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
            elapsed = time.time() - t_start
            ctx.metric(step, {
                "loss": round(avg, 5),
                "perplexity": round(min(math.exp(min(avg, 20)), 1e6), 3),
                "learning_rate": lr_at(step),
                "steps_per_sec": round(step / max(elapsed, 1e-6), 3),
                "vram_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)
                if device == "cuda" else None,
                "eta_s": round((total_steps - step) * elapsed / max(step, 1)),
            })
            ctx.progress(step, total_steps, stage="training")

            if ctx.should_cancel():
                raise Cancelled()
            if step >= total_steps:
                stop = True
                break
        else:
            continue
        break

    # ---- save ----------------------------------------------------------
    ctx.progress(total_steps, total_steps, stage="saving")
    ctx.log("Saving adapter to %s" % out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))

    summary = {
        "final_loss": round(running[-1], 5) if running else None,
        "initial_loss": round(running[0], 5) if running else None,
        "steps": step,
        "duration_s": round(time.time() - t_start, 1),
        "trainable_params": trainable,
        "base_model": base_model,
        "dtype": dtype_name,
        "quantized": use_4bit,
    }
    (out_dir / "ai_studio_summary.json").write_text(json.dumps(summary, indent=2))

    archive = Path(ctx.workdir) / "adapter.zip"
    shutil.make_archive(str(archive.with_suffix("")), "zip", root_dir=out_dir)
    summary["artifact_path"] = str(archive)
    summary["artifact_size"] = archive.stat().st_size
    ctx.log("Done. Loss %.4f -> %.4f over %d steps."
            % (summary["initial_loss"] or 0, summary["final_loss"] or 0, step))
    return summary
