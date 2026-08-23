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

import contextlib
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any, Iterator

from common import chat_formats
from common.formatting import (conversation_style,
                               detect_format, format_example, resolve_format)
from runner import artifacts, checkpoints, earlystop
from runner.capabilities import expert_kernel

from . import source
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
    keep_checkpoints = bool(cfg.get("checkpointing_enabled", True))
    resume = checkpoints.peek(ctx.job_id) if keep_checkpoints else None
    prepared = checkpoints.prepared_dir(ctx.job_id) if keep_checkpoints else None

    # ---- 0. an earlier model of our own, to carry on training -----------
    #
    # A finished model is a starting point, not only an endpoint. Training it
    # further on more text is the cheapest way to improve one, and until now
    # the only thing this app could not use as a base was a model it had
    # built itself.
    started_from = None
    if source_job := cfg.get("continue_from"):
        started_from = artifacts.fetch(ctx.controller_url, ctx.runner_token,
                                       source_job, ctx.log)
        arch = _arch_from_model(started_from, arch, ctx)
        seq_len = int(arch["max_position_embeddings"])
        tokens_per_step = batch * accum * seq_len
        token_budget = int(cfg.get("token_budget") or total_steps * tokens_per_step)

    # ---- 1. tokenizer ---------------------------------------------------
    ctx.progress(0, 0, stage="training_tokenizer")
    tok = _reuse_tokenizer(started_from, prepared, ctx)
    if tok is None:
        tok = _build_tokenizer(cfg, ctx)
        if prepared is not None:
            # Saved before a single training step runs. Building a vocabulary
            # is minutes of work that produces a *different* vocabulary every
            # time it is done -- so a resume that retrained it would load the
            # checkpoint's embedding table against a tokenizer that no longer
            # agrees with it, and the model would emit fluent nonsense.
            with contextlib.suppress(OSError, ValueError):
                tok.save_pretrained(str(prepared / "tokenizer"))
    vocab_size = len(tok)
    if vocab_size != arch["vocab_size"]:
        # BPE can finish below the requested size on a small or repetitive
        # corpus. The embedding table must match what the tokenizer emits.
        ctx.log("Tokenizer settled at %d tokens rather than %d; sizing the "
                "model to match." % (vocab_size, arch["vocab_size"]))
        arch = {**arch, "vocab_size": vocab_size}

    # ---- 2. corpus -> one flat array of token ids -----------------------
    ctx.progress(0, 0, stage="tokenizing")
    fingerprint = {"dataset": cfg.get("dataset"), "split": cfg.get("dataset_split"),
                   "field": cfg.get("text_field"), "vocab": vocab_size,
                   "budget": token_budget, "format": cfg.get("format")}
    tokens = _cached_corpus(prepared, fingerprint, np, ctx)
    if tokens is None:
        tokens = _tokenize_corpus(cfg, ctx, tok, token_budget, np)
        _cache_corpus(prepared, fingerprint, tokens, np, ctx)
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
    if started_from is not None:
        model, counts = _load_model(started_from, arch, ctx, torch)
    else:
        model, counts = _build_model(arch, ctx, torch, tok)
    if resume:
        _load_weights(model, Path(resume["path"]), ctx, torch)
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
    ctx.log("%s %s-parameter model: %d layers, width %d, %d heads, context %d."
            % ("Loaded a" if started_from else "Built a",
               _fmt(counts["total"]), arch["num_hidden_layers"],
               arch["hidden_size"], arch["num_attention_heads"], seq_len))
    if resume:
        ctx.log("Picking up from the checkpoint at step %d rather than starting "
                "again. Everything before that step is work this run has "
                "already done." % int(resume.get("step") or 0))
    elif started_from:
        ctx.log("This model already knows something -- it starts from what the "
                "earlier run taught it, so expect the loss to begin near where "
                "that run left off rather than near %.1f."
                % math.log(vocab_size))
    else:
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
        "resume": resume, "keep_checkpoints": keep_checkpoints,
        "started_from": str(started_from) if started_from else None,
        "continue_from": cfg.get("continue_from"),
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

    # Message-boundary tokens are reserved BEFORE training, so each becomes a
    # single atomic id inside the vocabulary the model was sized for. Added
    # afterwards they would grow the vocabulary past that size, and the
    # embedding table would no longer match. Reserved here, "<|im_start|>" is
    # one token that means exactly one thing; left to ordinary BPE it is half a
    # dozen pieces the model must learn to recognise in sequence, and which
    # also occur in ordinary text.
    fmt = resolve_format(cfg.get("format"))
    is_chat = conversation_style(fmt) == "chat"
    spec = chat_formats.format_or_default(fmt.get("chat_format")) if is_chat else None
    wants_reasoning = bool(fmt.get("reasoning"))
    specials = (chat_formats.special_tokens(fmt.get("chat_format"), wants_reasoning)
                if spec else [EOS])

    ctx.log("Training a new %d-token vocabulary on this text. A small "
            "vocabulary keeps the embedding table from swallowing the "
            "parameter budget." % vocab_size)
    if spec:
        ctx.log("Reserving %d %s boundary tokens so the model can learn where "
                "a turn starts and stops: %s"
                % (len(specials), spec["label"], " ".join(specials)))
        if wants_reasoning:
            extra = spec.get("reasoning_specials") or []
            ctx.log("Teaching it to reason before answering. %s%s"
                    % (spec.get("reasoning_note", ""),
                       (" Reserved: " + " ".join(extra)) if extra else ""))

    tk = Tokenizer(models.BPE(unk_token=None))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=specials,
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

    eos = spec["eos_token"] if spec else EOS
    bos = (spec.get("bos_token") if spec else None) or eos
    tok = PreTrainedTokenizerFast(
        tokenizer_object=tk, eos_token=eos, bos_token=bos,
        unk_token=eos, pad_token=eos,
        additional_special_tokens=[t for t in specials if t not in (eos, bos)],
    )
    if spec:
        # Written onto the tokenizer so the finished model is self-describing:
        # the playground reads this back the same way it reads a model
        # downloaded from the Hub, and speaks the format it was taught.
        tok.chat_template = spec["template"]
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
    local = source.is_studio_dataset(cfg)
    # A corpus is not always a column of prose. Conversations, instruction
    # pairs and hand-written templates all work here too, rendered by exactly
    # the same code the fine-tuner uses -- a model learning language from
    # scratch can learn a conversation format at the same time, and it can only
    # do that if what it reads matches what the playground will later send it.
    fmt = dict(cfg.get("format") or {})
    kwargs = {"split": cfg.get("dataset_split") or "train", "token": ctx.hf_token}
    if conf := cfg.get("dataset_config"):
        kwargs["name"] = conf

    if local:
        # A studio dataset is one JSONL file, already on disk after the fetch.
        # Nothing to stream from the Hub and no config or split to resolve.
        ds = load_dataset("json", data_files=source.local_copy(cfg, ctx),
                          split="train")
        kwargs = {}
    try:
        ds = ds if local else load_dataset(name, streaming=True, **kwargs)
    except Exception as e:  # noqa: BLE001 - not every dataset can stream
        ctx.log("Streaming is not available for this dataset (%s); downloading "
                "it in full instead." % type(e).__name__, "warn")
        ds = load_dataset(name, **kwargs)

    # A studio dataset holds every split in one file, with the split named on
    # each row. Rows from another split are skipped here rather than filtered
    # up front, because the Hub path streams and cannot be filtered at all.
    want = (cfg.get("dataset_split") or "").strip() if local else ""

    seen = 0
    unreadable = 0
    for row in ds:
        if want and (row.get("split") or "train") != want:
            continue
        seen += 1
        if not fmt:
            # No format recorded -- an older job, or one made through the API.
            # Work it out from the data rather than assuming a column called
            # "text", which fails instantly on any conversation dataset and
            # tells the user about a column they never asked for.
            fmt = detect_format(list(row.keys()), [row])
            if field and field in row and fmt.get("mode") != "chat":
                fmt = {"mode": "text", "text_field": field}
            ctx.log("No format was recorded for this run; reading these rows "
                    "as %s." % fmt.get("mode", "text"))
        text = (format_example(row, fmt) or "").strip()
        if text:
            yield text
            continue
        unreadable += 1
        # A dataset where nothing at all can be read is a setup mistake, not a
        # quiet zero-token corpus. Checked on a sample rather than on the first
        # row, because blank rows are normal in line-oriented text.
        if seen == 200 and unreadable == seen:
            raise ValueError(
                "None of the first %d rows could be read as %s. The columns "
                "are: %s. Check the column or template chosen for this dataset."
                % (seen, fmt.get("mode", "text"), ", ".join(map(str, row.keys()))))


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
    # Documents are separated by the same token that ends a turn, so the model
    # sees one consistent "this is finished" signal everywhere.
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
        if ctx.should_cancel():
            # Nothing exists to keep yet -- there is no model until the corpus
            # is read -- so stopping here always discards, whichever way the
            # user answered. Checked at all because tokenizing a large corpus
            # is minutes of work, and a Stop button that does nothing for four
            # of them reads as a broken Stop button.
            raise Cancelled()
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

def is_moe(arch: dict) -> bool:
    return int(arch.get("num_local_experts") or 0) > 1


def _build_model(arch: dict, ctx: Any, torch, tok=None):
    from transformers import AutoModelForCausalLM, LlamaConfig, MixtralConfig

    shared = dict(
        vocab_size=arch["vocab_size"],
        hidden_size=arch["hidden_size"],
        intermediate_size=arch["intermediate_size"],
        num_hidden_layers=arch["num_hidden_layers"],
        num_attention_heads=arch["num_attention_heads"],
        num_key_value_heads=arch["num_key_value_heads"],
        max_position_embeddings=arch["max_position_embeddings"],
        rms_norm_eps=arch.get("rms_norm_eps", 1e-5),
        tie_word_embeddings=arch.get("tie_word_embeddings", True),
        # Taken from the tokenizer rather than assumed to be zero. With a
        # chat format these are the boundary tokens, and generation stops on
        # the real one instead of on whatever happened to land at id 0.
        bos_token_id=(tok.bos_token_id if tok else 0) or 0,
        eos_token_id=(tok.eos_token_id if tok else 0) or 0,
        pad_token_id=(tok.pad_token_id if tok else 0) or 0,
        attention_dropout=0.0,
        use_cache=False,
    )

    if is_moe(arch):
        conf = MixtralConfig(
            num_local_experts=int(arch["num_local_experts"]),
            num_experts_per_tok=int(arch.get("num_experts_per_tok") or 2),
            router_aux_loss_coef=float(arch.get("router_aux_loss_coef", 0.01)),
            # Not decorative. The load-balancing loss is only computed when the
            # router's logits are returned, and without that term the router
            # collapses onto one expert within a few hundred steps: whichever
            # expert is marginally better early receives more tokens, trains
            # faster, and receives more still. The run does not fail -- it
            # quietly becomes a dense model carrying seven dead copies of a
            # feed-forward network.
            output_router_logits=True,
            sliding_window=None,
            **shared,
        )
        if kernel := expert_kernel(ctx.capabilities):
            # Older transformers has no such setting and simply carries the
            # attribute along harmlessly; there the eager loop is the only
            # implementation anyway.
            conf._experts_implementation = kernel
            ctx.log("Using the %r expert kernel, which is the one that works "
                    "on this backend." % kernel, "debug")
    else:
        conf = LlamaConfig(**shared)

    # from_config, not from_pretrained: random initialisation is the point.
    model = AutoModelForCausalLM.from_config(conf)

    embedding = model.get_input_embeddings().weight.numel()
    total = sum(p.numel() for p in model.parameters())
    counts = {"total": total, "embedding": embedding, "body": total - embedding,
              "active": total, "experts": 0, "experts_per_token": 0}

    if is_moe(arch):
        # Counted off the built model rather than recomputed from the config,
        # so it cannot drift from what was actually constructed.
        expert_params = sum(p.numel() for n, p in model.named_parameters()
                            if ".experts." in n)
        E = int(arch["num_local_experts"])
        k = min(int(arch.get("num_experts_per_tok") or 1), E)
        counts["active"] = int(total - expert_params * (E - k) / E)
        counts["experts"] = E
        counts["experts_per_token"] = k
        ctx.log("Mixture of experts: %d experts per block, %d chosen for each "
                "token. %s parameters in memory, %s of them used per token."
                % (E, k, _fmt(total), _fmt(counts["active"])))
        ctx.log("All %d experts are trained, but each one only learns from the "
                "tokens the router sends it -- roughly %s of your text. "
                "Watch the balance figure below: at %.2f every expert is "
                "getting an equal share, and much above that means the router "
                "has picked favourites."
                % (E, _share_phrase(k, E), 1.0 / E))

    share = embedding / max(total, 1)
    if share > 0.4:
        ctx.log("The vocabulary accounts for %.0f%% of this model's parameters. "
                "That is a lot of capacity spent on the embedding table rather "
                "than on the network -- a smaller vocabulary would leave more "
                "room to learn." % (share * 100), "warn")
    return model, counts


def _arch_from_model(path: Path, arch: dict, ctx: Any) -> dict:
    """The shape of a model we are about to carry on training.

    Read off the model rather than taken from the plan, and not negotiable:
    the weights on disk have one width, one depth and one vocabulary, and a
    request to continue training them at some other size is not a request that
    can be honoured. Saying so here beats a shape-mismatch traceback two
    minutes in.
    """
    conf = json.loads((path / "config.json").read_text(encoding="utf-8"))
    fields = ("vocab_size", "hidden_size", "intermediate_size",
              "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
              "max_position_embeddings", "num_local_experts",
              "num_experts_per_tok", "router_aux_loss_coef",
              "rms_norm_eps", "tie_word_embeddings")
    out = dict(arch or {})
    changed = []
    for f in fields:
        if conf.get(f) is not None:
            if out.get(f) not in (None, conf[f]) and f in (
                    "hidden_size", "num_hidden_layers", "vocab_size"):
                changed.append(f)
            out[f] = conf[f]
    out["model_type"] = conf.get("model_type", out.get("model_type"))
    if changed:
        ctx.log("Continuing an existing model, so its own shape is used rather "
                "than the one that was requested (%s). Width, depth and "
                "vocabulary are fixed once a model has been trained."
                % ", ".join(changed), "warn")
    return out


def _reuse_tokenizer(started_from: Path | None, prepared: Path | None, ctx: Any):
    """The vocabulary this run must use, if one already exists for it."""
    from transformers import AutoTokenizer

    candidates = [(started_from, "the model being continued")]
    if prepared is not None:
        candidates.append((prepared / "tokenizer", "the interrupted attempt"))

    for path, why in candidates:
        if not path or not (Path(path) / "tokenizer_config.json").exists():
            continue
        try:
            tok = AutoTokenizer.from_pretrained(str(path))
        except (OSError, ValueError) as e:
            ctx.log("Could not reuse the vocabulary from %s (%s); building a "
                    "new one." % (why, e), "warn")
            continue
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        # Documents are encoded whole and packed afterwards, so the real
        # context limit must not truncate them here. Needed on every path that
        # produces a tokenizer, including the ones that load one from disk.
        tok.model_max_length = 10 ** 9
        ctx.log("Reusing the %d-token vocabulary from %s." % (len(tok), why))
        return tok
    return None


def _cached_corpus(prepared: Path | None, fingerprint: dict, np, ctx: Any):
    """The tokenized corpus from an interrupted attempt, if it still applies.

    The fingerprint check is what makes this safe rather than merely fast: the
    same job id with a different dataset, vocabulary or budget describes
    different tokens, and silently training on the old ones would be a bug
    with no symptom other than a model that learned the wrong text.
    """
    if prepared is None:
        return None
    meta_file, data_file = prepared / "corpus.json", prepared / "tokens.npy"
    if not (meta_file.exists() and data_file.exists()):
        return None
    try:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if meta.get("fingerprint") != fingerprint:
        return None
    try:
        tokens = np.load(str(data_file), mmap_mode="r")
    except (OSError, ValueError) as e:
        ctx.log("The cached corpus could not be read (%s); reading the text "
                "again." % e, "warn")
        return None
    ctx.log("Reusing the %s tokens already read for this run, so the resume "
            "goes straight to training." % f"{tokens.size:,}")
    return tokens


def _cache_corpus(prepared: Path | None, fingerprint: dict, tokens, np,
                  ctx: Any) -> None:
    """Keep the tokenized corpus so a resume does not read the text twice.

    Bounded, because this is a convenience and not the output of the run: past
    a few gigabytes the disk cost stops being worth the minutes it saves, and
    a full data volume would break the training it was meant to protect.
    """
    if prepared is None or tokens.nbytes > 4 * 1024 ** 3:
        return
    try:
        np.save(str(prepared / "tokens.npy"), tokens)
        (prepared / "corpus.json").write_text(
            json.dumps({"fingerprint": fingerprint, "tokens": int(tokens.size)}),
            encoding="utf-8")
    except OSError as e:
        ctx.log("Could not cache the tokenized text (%s). Training carries on."
                % e, "warn")


def _load_model(path: Path, arch: dict, ctx: Any, torch):
    """Load one of this studio's own finished models, to train it further.

    float32 is not a default carried over by accident. A model being trained
    is the thing being updated, and fp16 master weights silently discard every
    update smaller than one ulp -- the same trap the module docstring opens
    with, and just as easy to fall into on the load path as on the build path.
    """
    from transformers import AutoModelForCausalLM

    extra = {}
    if is_moe(arch) and (kernel := expert_kernel(ctx.capabilities)):
        extra["experts_implementation"] = kernel
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(path), dtype=torch.float32, **extra)
    except (TypeError, ValueError) as e:
        if not extra or "experts_implementation" not in str(e):
            raise
        model = AutoModelForCausalLM.from_pretrained(str(path), dtype=torch.float32)

    embedding = model.get_input_embeddings().weight.numel()
    total = sum(p.numel() for p in model.parameters())
    counts = {"total": total, "embedding": embedding, "body": total - embedding,
              "active": total, "experts": 0, "experts_per_token": 0}
    if is_moe(arch):
        expert_params = sum(p.numel() for n, p in model.named_parameters()
                            if ".experts." in n)
        E = int(arch["num_local_experts"])
        k = min(int(arch.get("num_experts_per_tok") or 1), E)
        counts.update(active=int(total - expert_params * (E - k) / E),
                      experts=E, experts_per_token=k)
    return model, counts


def _load_pretrained_into(model, path: Path, torch) -> None:
    """Copy a saved snapshot's weights into the live model.

    Loaded into the model already on the GPU rather than by constructing a
    second one: the point in the run where this happens is the point of peak
    memory, and building a duplicate model to throw the first one away is how
    a run that trained perfectly fails on the last line.
    """
    from safetensors.torch import load_file

    files = sorted(path.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError("no weights in %s" % path)
    state: dict = {}
    for f in files:
        state.update(load_file(str(f)))
    missing, unexpected = model.load_state_dict(state, strict=False)
    # Tied embeddings legitimately leave `lm_head.weight` out of the file.
    real = [k for k in missing if "lm_head" not in k]
    if real or unexpected:
        raise ValueError("snapshot does not match this model (%d missing, "
                         "%d unexpected)" % (len(real), len(unexpected)))


def _load_weights(model, path: Path, ctx: Any, torch) -> None:
    model.load_state_dict(torch.load(str(path / "model.pt"), map_location="cpu",
                                     weights_only=True))


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
    sample_prompt = cfg.get("sample_prompt")
    if not sample_prompt:
        # A chat model asked to continue "Once upon a time" is being shown a
        # shape it never trained on. Ask it for a turn instead.
        chat_fmt = resolve_format(cfg.get("format"))
        if conversation_style(chat_fmt) == "chat":
            sample_prompt = chat_formats.format_or_default(
                chat_fmt.get("chat_format"))["sample_prompt"]
        else:
            sample_prompt = "Once upon a time"
    grad_clip = float(cfg.get("grad_clip", 1.0))

    # Blocks are indexed rather than copied: the token array can be a gigabyte
    # and materialising shuffled copies of it would double that for nothing.
    rng = np.random.default_rng(int(cfg.get("seed", 1234)))
    all_blocks = np.arange(n_blocks)
    rng.shuffle(all_blocks)
    val_idx, train_idx = all_blocks[:n_val], all_blocks[n_val:]

    # The split is drawn from the freshly seeded generator, and only then is
    # the saved stream position restored. Order matters: restore first and a
    # resumed run gets a *different* held-out set, so its held-out loss stops
    # being comparable with the first half of its own chart -- and some of the
    # text it is now measured on is text it has already trained on.
    resume = S.get("resume")
    prior = _resume_state(resume, ctx) if resume else {}
    if prior.get("rng"):
        rng.bit_generator.state = prior["rng"]

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

    if prior.get("optimizer"):
        # Adam without its moments is not Adam. Dropping the optimiser state
        # and keeping only the weights makes a resumed run take several
        # hundred steps of bad updates to recover the momentum estimates it
        # already had, which shows up as a visible bump in the loss exactly
        # where the checkpoint was.
        opt.load_state_dict(prior["optimizer"])
        if use_scaler and prior.get("scaler"):
            scaler.load_state_dict(prior["scaler"])
        if prior.get("torch_rng") is not None:
            torch.set_rng_state(prior["torch_rng"])

    moe = is_moe(arch)
    aux_coef = float(arch.get("router_aux_loss_coef", 0.01)) if moe else 0.0
    experts_per_tok = int(arch.get("num_experts_per_tok") or 1) if moe else 0

    def split_loss(out):
        """The language-modelling loss on its own, separated from the router's.

        With load balancing switched on, `out.loss` is the sum of the two. Left
        combined, the number on the chart is not comparable with a dense run's
        and the perplexity derived from it is simply wrong -- inflated by a
        term that has nothing to do with predicting text.
        """
        aux = getattr(out, "aux_loss", None)
        if aux is None:
            return out.loss, None
        return out.loss - aux_coef * aux.to(out.loss.device), aux

    @torch.no_grad()
    def evaluate(iters: int = 12) -> tuple[float | None, float | None]:
        """Held-out loss, and how evenly the router is spreading its work."""
        if len(val_idx) == 0:
            return None, None
        model.eval()
        losses = []
        busiest = []
        for _ in range(iters):
            ids = get_batch(val_idx, min(batch, len(val_idx)))
            with torch.amp.autocast("cuda", dtype=autocast_dtype, enabled=autocast_on):
                out = model(input_ids=ids, labels=ids)
            lm_loss, _ = split_loss(out)
            losses.append(lm_loss.item())
            if moe:
                share = _expert_share(getattr(out, "router_logits", None),
                                      experts_per_tok, torch)
                if share is not None:
                    busiest.append(share)
        model.train()
        return (sum(losses) / len(losses),
                sum(busiest) / len(busiest) if busiest else None)

    if device == "cuda":
        # The peak-memory counter is a high-water mark for the whole process,
        # and the agent trains many jobs in one process. Without this reset,
        # every run after the first reports the largest job's peak as its own
        # -- which is how a batch that was reduced to fit still appeared to use
        # exactly the memory of the run that had just failed.
        torch.cuda.reset_peak_memory_stats()

    ctx.log("Training for %d steps at %s tokens per step (batch %d x %d "
            "accumulation x %d context). Peak learning rate %.2e after %d "
            "warmup steps." % (total_steps, f"{S['tokens_per_step']:,}",
                               batch, accum, seq_len, lr, warmup))
    ctx.progress(0, total_steps, stage="training")

    model.train()
    t_start = time.time()
    start_step = int(prior.get("step") or 0)
    tokens_seen = int(prior.get("tokens_seen") or 0)
    first_loss = prior.get("first_loss")
    last_loss = prior.get("last_loss")
    best_val = prior.get("best_val")
    worst_share = prior.get("worst_share")
    warned_router = bool(prior.get("warned_router"))
    # Time spent by the attempts that came before this one, so a resumed run
    # reports how long it has taken in total rather than restarting its own
    # clock and claiming an eight-hour job took twenty minutes.
    prior_seconds = float(prior.get("duration_s") or 0.0)
    stopped_early = False
    last_val = prior.get("best_val")
    samples: list[dict] = list(prior.get("samples") or [])
    step = start_step
    saver = checkpoints.Saver(
        ctx.job_id, ctx, every_s=float(cfg.get("checkpoint_every_s") or 600),
        enabled=bool(S.get("keep_checkpoints", True)))
    stopper = earlystop.Stopper(
        ctx, int(cfg.get("early_stop_patience") or earlystop.PATIENCE_PRETRAIN),
        enabled=bool(cfg.get("early_stop", True)) and n_val > 0,
        kind="model")
    stopper.best = prior.get("best_val")
    stopper.best_step = int(prior.get("best_step") or 0)
    # The best snapshot is only worth going back for if it can still be read.
    # A run resumed on a machine that has the checkpoint but lost the `best`
    # directory should keep training, not promise a model it cannot produce.
    keep_best = bool(S.get("keep_checkpoints", True))

    stopper.announce(eval_every)

    def write_best(path):
        model.config.use_cache = True
        try:
            model.save_pretrained(str(path), safe_serialization=True)
        finally:
            model.config.use_cache = False

    def write_checkpoint(path):
        torch.save(model.state_dict(), str(path / "model.pt"))
        torch.save({"optimizer": opt.state_dict(),
                    "scaler": scaler.state_dict() if use_scaler else None,
                    "rng": rng.bit_generator.state,
                    "torch_rng": torch.get_rng_state()},
                   str(path / "trainer.pt"))
        (path / "train.json").write_text(json.dumps({
            "tokens_seen": tokens_seen, "first_loss": first_loss,
            "last_loss": last_loss, "worst_share": worst_share,
            "warned_router": warned_router,
            "best_val": stopper.best, "best_step": stopper.best_step,
            "duration_s": prior_seconds + (time.time() - t_start),
            "samples": samples[-6:],
        }), encoding="utf-8")

    for step in range(start_step + 1, total_steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step - 1)

        opt.zero_grad(set_to_none=True)
        accum_loss = 0.0
        accum_router = 0.0
        for _ in range(accum):
            ids = get_batch(train_idx, batch)
            with torch.amp.autocast("cuda", dtype=autocast_dtype, enabled=autocast_on):
                out = model(input_ids=ids, labels=ids)
            # The router's balancing term is trained through, but reported
            # apart: `loss` on the chart stays the cost of predicting text.
            lm_loss, aux = split_loss(out)
            loss = out.loss / accum
            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            accum_loss += lm_loss.item()
            if aux is not None:
                accum_router += aux.item()

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
        done_now = step - start_step
        elapsed = time.time() - t_start

        val_loss = None
        expert_share = None
        stop_now = False
        if step % eval_every == 0 or step == total_steps:
            val_loss, expert_share = evaluate()
            last_val = val_loss if val_loss is not None else last_val
            if val_loss is not None:
                best_val = val_loss if best_val is None else min(best_val, val_loss)
                verdict = stopper.update(step, val_loss)
                if verdict == "improved" and keep_best and step < total_steps:
                    # Written on improvement rather than at the end, because
                    # the point of the snapshot is that the run may not get
                    # back to this quality.
                    checkpoints.save_best(ctx.job_id, step, write_best,
                                          {"val_loss": val_loss})
                elif verdict == "stop":
                    stop_now = True
            if expert_share is not None:
                worst_share = max(worst_share or 0.0, expert_share)
                even = 1.0 / max(S["counts"]["experts"], 1)
                if expert_share > even * 2.5 and not warned_router:
                    warned_router = True
                    ctx.log("The router is sending %.0f%% of tokens to a single "
                            "expert, where an even split would be %.0f%%. The "
                            "other experts are being starved of the text they "
                            "need to learn from."
                            % (expert_share * 100, even * 100), "warn")

        ctx.metric(step, {
            "loss": round(avg, 5),
            "val_loss": round(val_loss, 5) if val_loss is not None else None,
            "router_loss": round(accum_router / accum, 5) if moe else None,
            "expert_balance": round(expert_share, 4) if expert_share is not None else None,
            "perplexity": round(min(math.exp(min(avg, 20)), 1e6), 2),
            "learning_rate": lr_at(step - 1),
            "grad_norm": round(grad_norm, 4) if math.isfinite(grad_norm) else None,
            "tokens_seen": tokens_seen,
            # Rates are for *this* attempt. Dividing the resumed step number
            # by the time since this process started would report a machine
            # several times faster than it is, and an ETA to match.
            "steps_per_sec": round(done_now / max(elapsed, 1e-6), 3),
            "tokens_per_sec": round(
                done_now * S["tokens_per_step"] / max(elapsed, 1e-6)),
            "vram_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)
            if device == "cuda" else None,
            "eta_s": round((total_steps - step) * elapsed / max(done_now, 1)),
        })
        ctx.progress(step, total_steps, stage="training")

        if saver.due(step):
            saver.write(step, total_steps, write_checkpoint,
                        {"kind": "pretrain_llm", "loss": last_loss})
            ctx.emit_meta({"checkpoint": {"step": step, "total": total_steps}})

        if step % sample_every == 0 or step == total_steps:
            text = _sample(model, tok, sample_prompt, device, S, torch, ctx)
            if text:
                samples.append({"step": step, "text": text})
                # Sent as meta rather than as a log line: the controller files
                # these under their step so the UI can show them as their own
                # panel instead of losing them in a scrolling log.
                ctx.emit_meta({"sample": {"step": step, "text": text,
                                          "prompt": sample_prompt}})

        if stop_now:
            # Deliberately NOT `stopped_early`. That flag means "a person
            # pressed Stop", and the agent turns it into a cancelled run. A
            # run that stopped because it had finished improving did not get
            # cancelled -- it succeeded, sooner than planned, which is the
            # whole point.
            break

        if ctx.should_cancel():
            # Stopping does not have to mean throwing the work away. A model
            # halfway through its schedule is a real model -- undertrained,
            # and worth keeping when the alternative is four hours of GPU time
            # deleted on the way out. Which of the two happens is the user's
            # choice, made when they press the button, and carried here.
            if not ctx.should_save():
                raise Cancelled()
            stopped_early = True
            ctx.log("Stopping at step %d of %d, and keeping the model as it "
                    "stands. Its learning rate never finished decaying, so it "
                    "is a little rougher than the same model trained to the "
                    "end would be." % (step, total_steps), "warn")
            break

    # ---- keep the best model, which is not always the last one ----------
    kept_step = None
    if stopper.should_restore(last_val, step):
        best = checkpoints.peek(ctx.job_id, "best") if keep_best else None
        if best:
            try:
                _load_pretrained_into(model, Path(best["path"]), torch)
                stopper.note_kept(last_val, step)
                kept_step = stopper.best_step
            except Exception as e:  # noqa: BLE001 - the trained model is still fine
                ctx.log("The better snapshot from step %d could not be read "
                        "(%s), so this run keeps the weights it ended with."
                        % (stopper.best_step, e), "warn")
        else:
            ctx.log("The held-out loss was better at step %d than at the end, "
                    "but no snapshot of it was kept -- checkpointing is off "
                    "for this run. Keeping the final weights."
                    % stopper.best_step, "warn")

    # ---- save -----------------------------------------------------------
    return _save(cfg, ctx, model, tok, arch, S, {
        "kept_from_step": kept_step,
        "early_stopped": stopper.stopped,
        "first_loss": first_loss, "last_loss": last_loss, "best_val": best_val,
        "tokens_seen": tokens_seen, "steps": step,
        "planned_steps": total_steps, "stopped_early": stopped_early,
        "worst_expert_share": worst_share, "resumed_from": start_step or None,
        "duration_s": prior_seconds + (time.time() - t_start), "samples": samples,
    })


def _resume_state(resume: dict, ctx: Any) -> dict:
    """Read back an interrupted attempt. A broken one starts over, loudly.

    Anything unreadable here means the checkpoint cannot be trusted, and the
    honest response is a fresh run rather than a half-restored optimiser
    producing a curve nobody can explain.
    """
    import torch

    path = Path(resume["path"])
    out: dict = {}
    try:
        out = json.loads((path / "train.json").read_text(encoding="utf-8"))
        # weights_only=False because this holds an optimiser state and a
        # random-number generator state, not just tensors. The file was
        # written by this runner into its own data volume a few minutes ago;
        # if that is reachable by someone else, the checkpoint is not the
        # thing to worry about.
        out.update(torch.load(str(path / "trainer.pt"), map_location="cpu",
                              weights_only=False))
    except Exception as e:  # noqa: BLE001 - any failure means "start over"
        ctx.log("The checkpoint for this run could not be read (%s), so it "
                "starts again from the beginning." % e, "warn")
        return {}
    out["step"] = int(resume.get("step") or 0)
    return out


def _share_phrase(k: int, experts: int) -> str:
    """"a half", not "one 2th". The fraction of the corpus one expert sees."""
    words = {1: "all", 2: "a half", 3: "a third", 4: "a quarter", 5: "a fifth",
             6: "a sixth", 8: "an eighth", 10: "a tenth", 16: "a sixteenth"}
    ratio = experts / max(k, 1)
    if ratio in words:
        return words[ratio]
    if ratio.is_integer():
        return "one %dth" % int(ratio)
    return "about %.0f%%" % (100 / ratio)


def _effective_params(counts: dict) -> int:
    total = int(counts.get("total") or 0)
    active = int(counts.get("active") or total)
    return total if active >= total else int(round((total * active) ** 0.5))


def _expert_share(router_logits, top_k: int, torch) -> float | None:
    """The share of tokens going to the single busiest expert.

    One number, and the only one that says whether the mixture is working. An
    even router gives 1/E; a collapsed one gives something close to 1, and a
    collapsed router is invisible in the loss curve -- the model still learns,
    it just learns with one expert doing the work and the rest carried dead.
    """
    if not router_logits:
        return None
    counts = None
    routed = 0
    for logits in router_logits:
        if logits is None or logits.numel() == 0:
            continue
        flat = logits.reshape(-1, logits.shape[-1]).float()
        picked = flat.topk(min(top_k, flat.shape[-1]), dim=-1).indices.reshape(-1)
        hist = torch.bincount(picked, minlength=flat.shape[-1]).float()
        counts = hist if counts is None else counts + hist
        routed += picked.numel()
    if counts is None or not routed:
        return None
    return float(counts.max().item() / routed)


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
    ctx.progress(stats.get("steps") or S["total_steps"], S["total_steps"],
                 stage="saving")
    ctx.log("Saving the model as it stands." if stats.get("stopped_early")
            else "Saving the finished model.")
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
        "params_active": S["counts"].get("active", S["counts"]["total"]),
        "params_body": S["counts"]["body"],
        "params_embedding": S["counts"]["embedding"],
        "experts": S["counts"].get("experts") or None,
        "experts_per_token": S["counts"].get("experts_per_token") or None,
        "worst_expert_share": (round(stats["worst_expert_share"], 4)
                               if stats.get("worst_expert_share") else None),
        "stopped_early": bool(stats.get("stopped_early")),
        "planned_steps": stats.get("planned_steps"),
        # How this run got to where it is. A model that continued an earlier
        # one is not comparable with a model of the same size trained once,
        # and the comparison view needs to be able to say so.
        "resumed_from_step": stats.get("resumed_from"),
        "continued_from": S.get("continue_from"),
        # Which step the weights actually came from, when that is not the last
        # one, and whether the run ended by itself.
        "kept_from_step": stats.get("kept_from_step"),
        "early_stopped": bool(stats.get("early_stopped")),
        "vocab_size": len(tok),
        "initial_loss": round(stats["first_loss"], 5) if stats["first_loss"] else None,
        "final_loss": round(stats["last_loss"], 5) if stats["last_loss"] else None,
        "best_val_loss": round(stats["best_val"], 5) if stats["best_val"] else None,
        "final_perplexity": round(math.exp(min(stats["last_loss"] or 20, 20)), 2),
        "tokens_seen": stats["tokens_seen"],
        # Against effective parameters, so a sparse model is judged by the
        # dense one it is worth rather than by either of its own two counts.
        # Same geometric mean the controller plans with -- see
        # controller/architectures.effective_params, which is where the
        # reasoning for it lives.
        "tokens_per_param": round(
            stats["tokens_seen"] / max(_effective_params(S["counts"]), 1), 2),
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

    if stats.get("resumed_from"):
        ctx.log("This run was interrupted and carried on from its checkpoint at "
                "step %d, so the %s figures below cover the whole run, not "
                "just the part after the interruption."
                % (stats["resumed_from"], "loss and token"))
    ctx.log("%s Loss %.4f -> %.4f over %s tokens (%.1f tokens per parameter)."
            % ("Stopped early." if stats.get("stopped_early") else "Done.",
               summary["initial_loss"] or 0, summary["final_loss"] or 0,
               f"{stats['tokens_seen']:,}", summary["tokens_per_param"]))
    if stats.get("stopped_early"):
        ctx.log("It completed %d of the %d steps that were planned. The model "
                "works and can be talked to; it simply had less practice than "
                "the plan called for."
                % (stats["steps"], stats.get("planned_steps") or 0))
    if summary.get("worst_expert_share"):
        E = summary.get("experts") or 1
        ctx.log("Busiest expert took %.0f%% of the tokens at its worst, against "
                "%.0f%% for a perfectly even split."
                % (summary["worst_expert_share"] * 100, 100.0 / E))
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
