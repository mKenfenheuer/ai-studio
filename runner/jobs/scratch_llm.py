"""Pretraining a language model from scratch.

Not a variation on fine-tuning. Every assumption from `lora_llm` is inverted
here, and each inversion is a place where copying the fine-tuning recipe
produces a run that appears to work and learns nothing:

* **Master weights stay float32.** Fine-tuning loads a frozen base in fp16
  because it is never updated. A from-scratch model *is* the thing being
  updated, and fp16 weights lose every update smaller than one ulp -- the loss
  drops to ~7 and then flatlines forever. Speed comes from autocasting the
  matmuls, not from storing the weights narrow.

* **Text is packed, not padded.** Fine-tuning pads each example to a fixed
  length so it can keep example boundaries. Pretraining has no examples, only
  text: documents are concatenated and sliced into equal blocks so that every
  position in every batch carries a real token. Padding here would spend most
  of the compute budget on nothing.

* **The tokenizer is trained too.** Borrowing a modern pretrained vocabulary
  (Qwen's is 151k tokens) onto a small model spends more parameters on the
  embedding table than on the entire network. See `controller/architectures`.

* **Held-out loss decides.** A run this long needs an answer to "is it
  actually learning" that training loss cannot give, so a slice of the corpus
  is never trained on and is measured periodically.
"""
from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path
from typing import Any, Iterator

from .lora_llm import Cancelled

EOS = "<|endoftext|>"

# How much of the corpus is held back to measure honestly. Small in relative
# terms because tokens spent on evaluation are tokens not spent learning.
VAL_FRACTION = 0.005
VAL_BLOCKS_MAX = 200
VAL_BLOCKS_MIN = 8


def run(cfg: dict, ctx: Any) -> dict:
    import numpy as np
    import torch

    arch = cfg.get("arch")
    if not arch:
        raise ValueError("No architecture was supplied for this run.")

    caps = ctx.capabilities
    device = "cuda" if caps["backend"] in ("cuda", "rocm") else (
        "mps" if caps["backend"] == "mps" else "cpu")
    dtype_name = cfg.get("dtype") or caps.get("recommended_dtype", "float32")
    if device == "cpu":
        dtype_name = "float32"
    autocast_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                      "float32": torch.float32}.get(dtype_name, torch.float32)

    seq_len = int(arch["max_position_embeddings"])
    batch = int(cfg.get("batch_size", 8))
    accum = int(cfg.get("grad_accum", 1))
    total_steps = int(cfg.get("max_steps") or 1000)
    tokens_per_step = batch * accum * seq_len
    token_budget = int(cfg.get("token_budget") or total_steps * tokens_per_step)

    out_dir = Path(ctx.workdir) / "model"

    # ---- 1. tokenizer ---------------------------------------------------
    ctx.progress(0, 0, stage="training_tokenizer")
    tok = _build_tokenizer(cfg, ctx)
    vocab_size = len(tok)
    if vocab_size != arch["vocab_size"]:
        # BPE can finish below the requested size on a small or repetitive
        # corpus. The embedding table must match what the tokenizer emits.
        ctx.log("Tokenizer settled at %d tokens rather than %d; sizing the "
                "model to match." % (vocab_size, arch["vocab_size"]))
        arch = {**arch, "vocab_size": vocab_size}

    # ---- 2. corpus -> one flat array of token ids -----------------------
    ctx.progress(0, 0, stage="tokenizing")
    tokens = _tokenize_corpus(cfg, ctx, tok, token_budget, np)
    if tokens.size < seq_len * 16:
        raise ValueError(
            "Only %d tokens could be read from this dataset, which is far too "
            "little to train on. Check that the right text column is selected."
            % tokens.size)

    n_blocks = tokens.size // seq_len
    n_val = int(min(VAL_BLOCKS_MAX, max(VAL_BLOCKS_MIN, n_blocks * VAL_FRACTION)))
    n_val = min(n_val, max(1, n_blocks // 10))
    train_tokens = (n_blocks - n_val) * seq_len

    epochs_over_data = token_budget / max(train_tokens, 1)
    if epochs_over_data > 4:
        ctx.log("The plan needs %s tokens but this corpus only yields %s. The "
                "model will see the same text %.1f times, and past about four "
                "passes it starts memorising rather than learning. Consider a "
                "larger dataset or a shorter run."
                % (f"{token_budget:,}", f"{train_tokens:,}", epochs_over_data), "warn")

    ctx.log("Corpus ready: %s training tokens in %s blocks of %d, plus %d "
            "held-out blocks for measuring."
            % (f"{train_tokens:,}", f"{n_blocks - n_val:,}", seq_len, n_val))

    # ---- 3. the model ---------------------------------------------------
    ctx.progress(0, 0, stage="building_model")
    model, counts = _build_model(arch, ctx, torch)
    model = model.to(device)
    model.config.use_cache = False

    checkpointing = bool(cfg.get("gradient_checkpointing", False))
    if checkpointing and device != "cpu":
        model.gradient_checkpointing_enable()

    ctx.emit_meta({
        "architecture": arch, "params_total": counts["total"],
        "params_body": counts["body"], "params_embedding": counts["embedding"],
        "vocab_size": vocab_size, "train_tokens": train_tokens,
        "token_budget": token_budget, "tokens_per_step": tokens_per_step,
        "dtype": dtype_name, "from_scratch": True,
    })
    ctx.log("Built a %s-parameter model from random weights: %d layers, width "
            "%d, %d heads, context %d."
            % (_fmt(counts["total"]), arch["num_hidden_layers"],
               arch["hidden_size"], arch["num_attention_heads"], seq_len))
    ctx.log("Nothing in it knows anything yet -- every weight is noise. The "
            "loss should start near %.1f, which is what pure guessing costs "
            "with a %d-token vocabulary." % (math.log(vocab_size), vocab_size))

    # ---- 4. train -------------------------------------------------------
    return _train(cfg, ctx, model, tok, tokens, arch, {
        "device": device, "dtype_name": dtype_name, "autocast_dtype": autocast_dtype,
        "seq_len": seq_len, "batch": batch, "accum": accum,
        "total_steps": total_steps, "tokens_per_step": tokens_per_step,
        "n_blocks": n_blocks, "n_val": n_val, "counts": counts,
        "checkpointing": checkpointing, "out_dir": out_dir,
    }, np, torch)


# ===========================================================================
# Tokenizer
# ===========================================================================

def _build_tokenizer(cfg: dict, ctx: Any):
    """Train a fresh byte-level BPE vocabulary on this corpus, or reuse one.

    Byte-level means no token is ever unrepresentable: any input decodes,
    including emoji and broken text, without an <unk> escape hatch.
    """
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    if borrowed := cfg.get("tokenizer_from"):
        ctx.log("Using the existing tokenizer from %s." % borrowed)
        tok = AutoTokenizer.from_pretrained(borrowed, token=ctx.hf_token)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.model_max_length = 10 ** 9
        return tok

    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    vocab_size = int((cfg.get("arch") or {}).get("vocab_size", 8192))
    sample_rows = int(cfg.get("tokenizer_sample_rows", 200_000))
    ctx.log("Training a new %d-token vocabulary on this text. A small "
            "vocabulary keeps the embedding table from swallowing the "
            "parameter budget." % vocab_size)

    tk = Tokenizer(models.BPE(unk_token=None))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[EOS],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )

    seen = 0

    def texts() -> Iterator[str]:
        nonlocal seen
        for text in _iter_texts(cfg, ctx):
            seen += 1
            if seen > sample_rows:
                return
            if seen % 25_000 == 0:
                ctx.progress(seen, sample_rows, stage="training_tokenizer")
            yield text

    tk.train_from_iterator(texts(), trainer=trainer)

    tok = PreTrainedTokenizerFast(
        tokenizer_object=tk, eos_token=EOS, bos_token=EOS,
        unk_token=EOS, pad_token=EOS,
    )
    # Documents are encoded whole and then packed into blocks, so a document
    # longer than the context window is expected and correct. Without this the
    # tokenizer prints an alarming length warning for most of the corpus.
    tok.model_max_length = 10 ** 9
    ctx.log("Vocabulary trained from %s samples: %d tokens."
            % (f"{seen:,}", len(tok)))
    return tok


# ===========================================================================
# Corpus
# ===========================================================================

def _iter_texts(cfg: dict, ctx: Any) -> Iterator[str]:
    """Yield raw strings from the dataset, streaming where possible.

    Streaming matters more here than for fine-tuning: a from-scratch run is
    defined by a token budget, not by a number of examples, and that budget is
    usually a fraction of the corpus. Downloading 2 GB to read the first 400 MB
    of it wastes the user's first ten minutes on a progress bar.
    """
    from datasets import load_dataset

    name = cfg["dataset"]
    field = cfg.get("text_field") or "text"
    kwargs = {"split": cfg.get("dataset_split") or "train", "token": ctx.hf_token}
    if conf := cfg.get("dataset_config"):
        kwargs["name"] = conf

    try:
        ds = load_dataset(name, streaming=True, **kwargs)
    except Exception as e:  # noqa: BLE001 - not every dataset can stream
        ctx.log("Streaming is not available for this dataset (%s); downloading "
                "it in full instead." % type(e).__name__, "warn")
        ds = load_dataset(name, **kwargs)

    for row in ds:
        value = row.get(field)
        if value is None:
            # Wrong column name is the most common setup mistake, and silently
            # yielding nothing would look like a mysteriously tiny corpus.
            raise ValueError(
                "This dataset has no column called %r. Its columns are: %s."
                % (field, ", ".join(map(str, row.keys()))))
        text = str(value).strip()
        if text:
            yield text


def _tokenize_corpus(cfg: dict, ctx: Any, tok, token_budget: int, np):
    """Encode text into one flat array of ids, with document separators.

    uint16 is not a micro-optimisation. A 500M-token budget held as Python
    integers is roughly 14 GB of RAM; the same tokens in uint16 are 1 GB, and
    every vocabulary offered here fits in 16 bits.
    """
    dtype = np.uint16 if len(tok) < 65536 else np.uint32
    # The buffer lives in host RAM for the whole run, so it gets a ceiling
    # independent of the token budget. A long run on a small model can ask for
    # billions of tokens; holding those would need more memory than the
    # machine training the model has. Past the cap the run makes repeated
    # passes instead, and says so.
    cap = int(cfg.get("max_corpus_tokens") or 500_000_000)
    if token_budget > cap:
        ctx.log("The plan asks for %s tokens; holding more than %s at once "
                "would need too much system memory, so training will make "
                "repeated passes over that much text."
                % (f"{token_budget:,}", f"{cap:,}"), "warn")
        token_budget = cap
    # Slack lets the final batch of documents land without a reallocation.
    capacity = int(token_budget * 1.02) + 1_000_000
    buf = np.empty(capacity, dtype=dtype)
    filled = 0
    eos_id = tok.eos_token_id or 0

    ctx.log("Reading and tokenizing text until %s tokens are collected."
            % f"{token_budget:,}")
    pending: list[str] = []
    docs = 0
    t0 = time.time()
    exhausted = True

    def flush() -> bool:
        """Encode the pending batch. Returns False when the buffer is full."""
        nonlocal filled, pending
        if not pending:
            return True
        for ids in tok(pending, add_special_tokens=False)["input_ids"]:
            room = capacity - filled
            if room <= 1:
                return False
            take = min(len(ids), room - 1)
            buf[filled:filled + take] = ids[:take]
            filled += take
            buf[filled] = eos_id
            filled += 1
        pending = []
        return True

    for text in _iter_texts(cfg, ctx):
        pending.append(text)
        docs += 1
        if len(pending) < 1000:
            continue
        if not flush():
            exhausted = False
            break
        if filled >= token_budget:
            exhausted = False
            break
        if docs % 50_000 == 0:
            ctx.progress(min(filled, token_budget), token_budget, stage="tokenizing")
            ctx.log("  %s tokens from %s documents (%.0f k tokens/s)"
                    % (f"{filled:,}", f"{docs:,}", filled / max(time.time() - t0, 1e-6) / 1000))
    else:
        flush()

    if exhausted and filled < token_budget:
        ctx.log("Reached the end of the dataset after %s tokens, short of the "
                "%s the plan asked for. Training will make repeated passes "
                "over what is there." % (f"{filled:,}", f"{token_budget:,}"), "warn")

    ctx.log("Tokenized %s documents into %s tokens in %.0fs."
            % (f"{docs:,}", f"{filled:,}", time.time() - t0))
    return buf[:filled]


# ===========================================================================
# Model
# ===========================================================================

def _build_model(arch: dict, ctx: Any, torch):
    from transformers import AutoModelForCausalLM, LlamaConfig

    conf = LlamaConfig(
        vocab_size=arch["vocab_size"],
        hidden_size=arch["hidden_size"],
        intermediate_size=arch["intermediate_size"],
        num_hidden_layers=arch["num_hidden_layers"],
        num_attention_heads=arch["num_attention_heads"],
        num_key_value_heads=arch["num_key_value_heads"],
        max_position_embeddings=arch["max_position_embeddings"],
        rms_norm_eps=arch.get("rms_norm_eps", 1e-5),
        tie_word_embeddings=arch.get("tie_word_embeddings", True),
        bos_token_id=0, eos_token_id=0, pad_token_id=0,
        attention_dropout=0.0,
        use_cache=False,
    )
    # from_config, not from_pretrained: random initialisation is the point.
    model = AutoModelForCausalLM.from_config(conf)

    embedding = model.get_input_embeddings().weight.numel()
    total = sum(p.numel() for p in model.parameters())
    counts = {"total": total, "embedding": embedding, "body": total - embedding}
    share = embedding / max(total, 1)
    if share > 0.4:
        ctx.log("The vocabulary accounts for %.0f%% of this model's parameters. "
                "That is a lot of capacity spent on the embedding table rather "
                "than on the network -- a smaller vocabulary would leave more "
                "room to learn." % (share * 100), "warn")
    return model, counts


def _param_groups(model, weight_decay: float):
    """Decay matrices, never norms or biases.

    Weight decay on a LayerNorm gain or a bias pulls it toward zero for no
    benefit and measurably hurts. Every serious pretraining recipe splits
    these; most fine-tuning code does not bother because it barely matters
    over a few hundred steps.
    """
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


# ===========================================================================
# Training
# ===========================================================================

def _train(cfg, ctx, model, tok, tokens, arch, S, np, torch) -> dict:
    device, seq_len = S["device"], S["seq_len"]
    batch, accum = S["batch"], S["accum"]
    total_steps = S["total_steps"]
    n_blocks, n_val = S["n_blocks"], S["n_val"]
    autocast_dtype = S["autocast_dtype"]

    lr = float(cfg.get("learning_rate", 6e-4))
    min_lr = lr * float(cfg.get("min_lr_ratio", 0.1))
    warmup = int(cfg.get("warmup_steps") or min(200, max(10, total_steps // 20)))
    weight_decay = float(cfg.get("weight_decay", 0.1))
    eval_every = int(cfg.get("eval_every") or max(20, total_steps // 25))
    sample_every = int(cfg.get("sample_every") or max(40, total_steps // 10))
    sample_prompt = cfg.get("sample_prompt") or "Once upon a time"
    grad_clip = float(cfg.get("grad_clip", 1.0))

    # Blocks are indexed rather than copied: the token array can be a gigabyte
    # and materialising shuffled copies of it would double that for nothing.
    rng = np.random.default_rng(int(cfg.get("seed", 1234)))
    all_blocks = np.arange(n_blocks)
    rng.shuffle(all_blocks)
    val_idx, train_idx = all_blocks[:n_val], all_blocks[n_val:]

    def get_batch(indices, size):
        picked = indices[rng.integers(0, len(indices), size)]
        starts = picked.astype(np.int64) * seq_len
        rows = np.stack([tokens[s:s + seq_len] for s in starts])
        ids = torch.from_numpy(rows.astype(np.int64)).to(device, non_blocking=True)
        return ids

    params = _param_groups(model, weight_decay)
    use_8bit = bool(cfg.get("optim_8bit")) and \
        ctx.capabilities.get("quantization", {}).get("optim_8bit")
    if use_8bit:
        import bitsandbytes.optim as bopt
        opt = bopt.AdamW8bit(params, lr=lr, betas=(0.9, 0.95), eps=1e-8)
        ctx.log("Using 8-bit AdamW, which halves the optimiser's memory.")
    else:
        opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), eps=1e-8)

    # beta2 = 0.95 rather than Adam's default 0.999: pretraining gradients
    # change character quickly early on, and a shorter second-moment memory
    # keeps the optimiser from being anchored to a stale estimate.

    def lr_at(s: int) -> float:
        if s < warmup:
            return lr * (s + 1) / warmup
        prog = (s - warmup) / max(1, total_steps - warmup)
        return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * min(prog, 1.0)))

    use_scaler = (autocast_dtype == torch.float16 and device == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    autocast_on = device == "cuda" and autocast_dtype != torch.float32

    @torch.no_grad()
    def evaluate(iters: int = 12) -> float | None:
        if len(val_idx) == 0:
            return None
        model.eval()
        losses = []
        for _ in range(iters):
            ids = get_batch(val_idx, min(batch, len(val_idx)))
            with torch.amp.autocast("cuda", dtype=autocast_dtype, enabled=autocast_on):
                losses.append(model(input_ids=ids, labels=ids).loss.item())
        model.train()
        return sum(losses) / len(losses)

    ctx.log("Training for %d steps at %s tokens per step (batch %d x %d "
            "accumulation x %d context). Peak learning rate %.2e after %d "
            "warmup steps." % (total_steps, f"{S['tokens_per_step']:,}",
                               batch, accum, seq_len, lr, warmup))
    ctx.progress(0, total_steps, stage="training")

    model.train()
    t_start = time.time()
    tokens_seen = 0
    first_loss = None
    last_loss = None
    best_val = None
    samples: list[dict] = []

    for step in range(1, total_steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step - 1)

        opt.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(accum):
            ids = get_batch(train_idx, batch)
            with torch.amp.autocast("cuda", dtype=autocast_dtype, enabled=autocast_on):
                loss = model(input_ids=ids, labels=ids).loss / accum
            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            accum_loss += loss.item() * accum

        if use_scaler:
            scaler.unscale_(opt)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(
            [p for g in params for p in g["params"]], grad_clip))
        if use_scaler:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()

        avg = accum_loss / accum
        first_loss = first_loss if first_loss is not None else avg
        last_loss = avg
        tokens_seen += S["tokens_per_step"]
        elapsed = time.time() - t_start

        val_loss = None
        if step % eval_every == 0 or step == total_steps:
            val_loss = evaluate()
            if val_loss is not None:
                best_val = val_loss if best_val is None else min(best_val, val_loss)

        ctx.metric(step, {
            "loss": round(avg, 5),
            "val_loss": round(val_loss, 5) if val_loss is not None else None,
            "perplexity": round(min(math.exp(min(avg, 20)), 1e6), 2),
            "learning_rate": lr_at(step - 1),
            "grad_norm": round(grad_norm, 4) if math.isfinite(grad_norm) else None,
            "tokens_seen": tokens_seen,
            "steps_per_sec": round(step / max(elapsed, 1e-6), 3),
            "tokens_per_sec": round(tokens_seen / max(elapsed, 1e-6)),
            "vram_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)
            if device == "cuda" else None,
            "eta_s": round((total_steps - step) * elapsed / max(step, 1)),
        })
        ctx.progress(step, total_steps, stage="training")

        if step % sample_every == 0 or step == total_steps:
            text = _sample(model, tok, sample_prompt, device, S, torch, ctx)
            if text:
                samples.append({"step": step, "text": text})
                # Sent as meta rather than as a log line: the controller files
                # these under their step so the UI can show them as their own
                # panel instead of losing them in a scrolling log.
                ctx.emit_meta({"sample": {"step": step, "text": text,
                                          "prompt": sample_prompt}})

        if ctx.should_cancel():
            raise Cancelled()

    # ---- save -----------------------------------------------------------
    return _save(cfg, ctx, model, tok, arch, S, {
        "first_loss": first_loss, "last_loss": last_loss, "best_val": best_val,
        "tokens_seen": tokens_seen, "steps": total_steps,
        "duration_s": time.time() - t_start, "samples": samples,
    })


def _sample(model, tok, prompt: str, device: str, S: dict, torch, ctx) -> str | None:
    """Generate a short continuation so the user can watch language appear.

    This is the single most motivating thing a from-scratch run can show. The
    loss curve says the number is going down; the samples say what that means,
    turning noise into words into sentences over the course of a run.
    """
    checkpointing = S.get("checkpointing")
    try:
        model.eval()
        if checkpointing:
            # Caching and recomputation are mutually exclusive; generation
            # without a cache is quadratic and pointlessly slow.
            model.gradient_checkpointing_disable()
        model.config.use_cache = True
        ids = tok(prompt, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            out = model.generate(
                ids, max_new_tokens=int(S.get("sample_tokens", 60)),
                do_sample=True, temperature=0.8, top_k=50,
                pad_token_id=tok.eos_token_id,
            )
        return tok.decode(out[0], skip_special_tokens=True)
    except Exception as e:  # noqa: BLE001 - a failed sample must not kill a run
        ctx.log("Could not generate a sample (%s); training continues."
                % type(e).__name__, "debug")
        return None
    finally:
        model.config.use_cache = False
        if checkpointing:
            model.gradient_checkpointing_enable()
        model.train()


def _save(cfg, ctx, model, tok, arch, S, stats) -> dict:
    out_dir = S["out_dir"]
    ctx.progress(S["total_steps"], S["total_steps"], stage="saving")
    ctx.log("Saving the finished model.")
    out_dir.mkdir(parents=True, exist_ok=True)

    model.config.use_cache = True
    model.save_pretrained(str(out_dir), safe_serialization=True)
    # Restore the real limit before saving: the huge value above exists only so
    # packing can encode long documents, and shipping it would tell anyone who
    # loads this model that it has a billion-token context.
    tok.model_max_length = arch["max_position_embeddings"]
    tok.save_pretrained(str(out_dir))

    summary = {
        "kind": "pretrain_llm",
        "architecture": arch,
        "params_total": S["counts"]["total"],
        "params_body": S["counts"]["body"],
        "params_embedding": S["counts"]["embedding"],
        "vocab_size": len(tok),
        "initial_loss": round(stats["first_loss"], 5) if stats["first_loss"] else None,
        "final_loss": round(stats["last_loss"], 5) if stats["last_loss"] else None,
        "best_val_loss": round(stats["best_val"], 5) if stats["best_val"] else None,
        "final_perplexity": round(math.exp(min(stats["last_loss"] or 20, 20)), 2),
        "tokens_seen": stats["tokens_seen"],
        "tokens_per_param": round(stats["tokens_seen"] / max(S["counts"]["total"], 1), 2),
        "steps": stats["steps"],
        "duration_s": round(stats["duration_s"], 1),
        "dataset": cfg.get("dataset"),
        "dtype": S["dtype_name"],
    }
    (out_dir / "ai_studio_summary.json").write_text(json.dumps(summary, indent=2))
    _write_readme(out_dir, summary, stats["samples"], cfg)

    archive = Path(ctx.workdir) / "model.zip"
    shutil.make_archive(str(archive.with_suffix("")), "zip", root_dir=out_dir)
    summary["artifact_path"] = str(archive)
    summary["artifact_size"] = archive.stat().st_size
    summary["samples"] = stats["samples"][-3:]

    ctx.log("Done. Loss %.4f -> %.4f over %s tokens (%.1f tokens per parameter)."
            % (summary["initial_loss"] or 0, summary["final_loss"] or 0,
               f"{stats['tokens_seen']:,}", summary["tokens_per_param"]))
    return summary


def _write_readme(out_dir: Path, summary: dict, samples: list, cfg: dict) -> None:
    """Ship instructions with the model.

    A directory of safetensors is not a usable result for the audience this
    app is for. Three lines of copy-paste that actually produce text is.
    """
    last = samples[-1]["text"] if samples else "(no sample was generated)"
    ratio = summary["tokens_per_param"]
    if ratio >= 20:
        standing = ("This model saw %.0f tokens per parameter, which is a full "
                    "training budget for its size." % ratio)
    elif ratio >= 5:
        standing = ("This model saw %.1f tokens per parameter. Twenty is a full "
                    "budget, so it is undertrained but should still be "
                    "coherent." % ratio)
    else:
        standing = ("This model saw only %.1f tokens per parameter against a "
                    "full budget of twenty, so it is heavily undertrained. "
                    "Expect words rather than sentences." % ratio)

    (out_dir / "README.md").write_text(
        "# Model trained from scratch with AI Studio\n\n"
        "Trained from random initialisation on `%s`.\n\n"
        "| | |\n|---|---|\n"
        "| Parameters | %s |\n| Vocabulary | %s tokens |\n"
        "| Tokens seen | %s |\n| Final loss | %s |\n"
        "| Held-out loss | %s |\n| Perplexity | %s |\n\n"
        "%s\n\n## Using it\n\n"
        "```python\n"
        "from transformers import AutoModelForCausalLM, AutoTokenizer\n\n"
        "tok = AutoTokenizer.from_pretrained(\".\")\n"
        "model = AutoModelForCausalLM.from_pretrained(\".\")\n\n"
        "ids = tok(\"%s\", return_tensors=\"pt\").input_ids\n"
        "out = model.generate(ids, max_new_tokens=80, do_sample=True,\n"
        "                     temperature=0.8, top_k=50)\n"
        "print(tok.decode(out[0], skip_special_tokens=True))\n"
        "```\n\n## Last sample from training\n\n> %s\n"
        % (summary.get("dataset", "a text corpus"),
           f"{summary['params_total']:,}", f"{summary['vocab_size']:,}",
           f"{summary['tokens_seen']:,}", summary["final_loss"],
           summary["best_val_loss"], summary["final_perplexity"],
           standing,
           (cfg.get("sample_prompt") or "Once upon a time").replace('"', "'"),
           last.replace("\n", "\n> ")),
        encoding="utf-8")


def _fmt(n: int) -> str:
    if n >= 1e9:
        return "%.1fB" % (n / 1e9)
    if n >= 1e6:
        return "%.1fM" % (n / 1e6)
    return "%.0fK" % (n / 1e3)
