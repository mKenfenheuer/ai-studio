"""Model architectures for training from scratch, and the honest arithmetic
that decides whether a given plan is worth starting.

Fine-tuning and pretraining look similar in a UI and are nothing alike in
cost. Fine-tuning a 3B model with LoRA touches ~0.1% of its parameters for a
few thousand steps. Training from scratch touches 100% of them for as many
tokens as you can afford, and the useful result is bounded by *compute*, not
by VRAM.

So the numbers here exist to answer one question before the user commits
hours of GPU time: given this machine and this much patience, what size of
model can actually finish learning? Getting that answer wrong is the defining
beginner experience of from-scratch training -- a 24-hour run that produces
word-shaped noise -- and it is entirely predictable in advance.

Two rules drive everything below:

1. **Chinchilla.** A model needs roughly 20 training tokens per parameter to
   be worth its size. Below that it is undertrained: it will be outperformed
   by a smaller model given the same compute.

2. **Small models waste big GPUs.** A 256-wide matmul cannot fill an RX 6900
   XT. Peak throughput measured on 2048x2048 matrices is simply not available
   to a tiny transformer, so estimates scale efficiency with model width
   instead of pretending otherwise.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Sizes
# ---------------------------------------------------------------------------
# Deliberately stops at ~200M. Anything larger cannot be trained to a useful
# standard on one consumer GPU, and offering it would be a trap rather than a
# feature. The ceiling is compute, not memory -- see time_for_tokens().

SIZE_PRESETS = [
    {
        "id": "nano", "label": "Nano", "layers": 4, "dim": 256, "heads": 4,
        "seq": 256,
        "blurb": "Learns spelling, punctuation and short sentence shapes. "
                 "Finishes in minutes, so it is the right way to check your "
                 "data and settings before committing to a long run.",
        "expect": "Real words, wobbly grammar.",
    },
    {
        "id": "tiny", "label": "Tiny", "layers": 6, "dim": 384, "heads": 6,
        "seq": 512,
        "blurb": "The smallest size that writes genuinely coherent English "
                 "when trained on simple text. A satisfying first real model.",
        "expect": "Short readable passages that stay on topic.",
    },
    {
        "id": "small", "label": "Small", "layers": 8, "dim": 512, "heads": 8,
        "seq": 512,
        "blurb": "Noticeably better sentence structure and longer memory. "
                 "Wants several hours of training to earn its size.",
        "expect": "Fluent paragraphs, simple reasoning.",
    },
    {
        "id": "base", "label": "Base", "layers": 12, "dim": 768, "heads": 12,
        "seq": 1024,
        "blurb": "GPT-2 small's shape. A serious model that needs serious "
                 "compute -- expect days, not hours, on one consumer card.",
        "expect": "Good text, but only if you train it properly.",
    },
    {
        "id": "large", "label": "Large", "layers": 16, "dim": 1024, "heads": 16,
        "seq": 1024,
        "blurb": "Included for completeness. On a single GPU you will run out "
                 "of patience long before this model runs out of things to "
                 "learn from.",
        "expect": "Only worth it with a week of GPU time.",
    },
]

# Vocabulary size for a freshly trained tokenizer.
#
# This matters far more than it looks. Embeddings cost vocab x width, and that
# is charged against a parameter budget the body of a small model also needs.
# Borrowing a modern pretrained tokenizer (Qwen: 151k tokens) onto a 384-wide
# model would spend 58M parameters on the vocabulary alone -- five times the
# entire rest of the network. A small, purpose-trained vocabulary is not a
# compromise here; it is the correct choice.
VOCAB_PRESETS = [
    {"id": 4096, "label": "4,096", "hint": "Very small. Fine for simple text."},
    {"id": 8192, "label": "8,192", "hint": "A good default for small models."},
    {"id": 16384, "label": "16,384", "hint": "Better for varied or technical text."},
    {"id": 32768, "label": "32,768", "hint": "Only worth it above ~100M parameters."},
]
DEFAULT_VOCAB = 8192

TOKENS_PER_PARAM_TARGET = 20     # Chinchilla-optimal
BYTES_PER_TOKEN = 4.0            # rough English average for a small BPE vocab


def _intermediate(dim: int) -> int:
    """SwiGLU uses three matrices instead of two, so the hidden width is
    scaled by 8/3 rather than 4 to keep the parameter count comparable."""
    return int(round(8 * dim / 3 / 64)) * 64


def preset(size_id: str) -> dict | None:
    return next((p for p in SIZE_PRESETS if p["id"] == size_id), None)


def build_arch(size_id: str, vocab_size: int = DEFAULT_VOCAB,
               seq_len: int | None = None) -> dict | None:
    """Concrete architecture the runner can hand straight to transformers.

    Resolved here rather than on the runner so that the parameter count shown
    in the UI and the model that actually gets built can never disagree.
    """
    p = preset(size_id)
    if not p:
        return None
    seq = int(seq_len or p["seq"])
    return {
        "size_id": size_id,
        "model_type": "llama",
        "vocab_size": int(vocab_size),
        "hidden_size": p["dim"],
        "intermediate_size": _intermediate(p["dim"]),
        "num_hidden_layers": p["layers"],
        "num_attention_heads": p["heads"],
        "num_key_value_heads": p["heads"],
        "max_position_embeddings": seq,
        # Tied input and output embeddings. At these sizes an untied output
        # head would add another vocab x width block for very little gain.
        "tie_word_embeddings": True,
        "rms_norm_eps": 1e-5,
    }


def count_params(arch: dict) -> dict:
    """Exact parameter count for a Llama-shaped model.

    Split into embedding and body because the ratio is the thing worth
    showing: when embeddings dominate, the vocabulary is too big for the
    model, and shrinking it buys real capacity for free.
    """
    d = arch["hidden_size"]
    L = arch["num_hidden_layers"]
    v = arch["vocab_size"]
    i = arch["intermediate_size"]
    kv = (arch["num_key_value_heads"] * d) // arch["num_attention_heads"]

    embedding = v * d
    if not arch.get("tie_word_embeddings", True):
        embedding *= 2
    per_layer = (
        d * d          # q
        + d * kv       # k
        + d * kv       # v
        + d * d        # o
        + 3 * d * i    # gate, up, down
        + 2 * d        # two RMSNorms
    )
    body = L * per_layer + d          # + final norm
    return {"embedding": embedding, "body": body, "total": embedding + body,
            "embedding_share": round(embedding / max(embedding + body, 1), 3)}


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

def training_memory_gb(arch: dict, batch: int, *, optim_8bit: bool = False,
                       checkpointing: bool = False, flash: bool = False) -> dict:
    """VRAM needed to train every parameter of this model.

    Nothing like the LoRA estimate in hub.py. There the base model is frozen
    and only the adapter carries optimiser state; here every parameter needs
    fp32 master weights, fp32 gradients and two Adam moments -- 16 bytes per
    parameter before a single activation is stored.
    """
    n = count_params(arch)["total"]
    d = arch["hidden_size"]
    L = arch["num_hidden_layers"]
    h = arch["num_attention_heads"]
    s = arch["max_position_embeddings"]

    # 4 (weights) + 4 (grads) + 8 (Adam m and v), or 2 bytes of state with an
    # 8-bit optimiser. Plus the half-precision copies autocast keeps around.
    per_param = 10 if optim_8bit else 16
    state = n * (per_param + 2)

    tokens = batch * s
    # 44 bytes per token, per unit of width, per layer. Counting the tensors a
    # Llama block keeps for its backward pass gives ~34; the rest is the parts
    # autocast keeps in fp32 rather than fp16 -- both RMSNorms and the rotary
    # inputs. Calibrated against two measured runs rather than derived, and it
    # is deliberately the conservative end of what they imply.
    if checkpointing:
        # Only layer inputs survive the forward pass; the rest is recomputed.
        acts = tokens * d * L * 2 + tokens * d * 44
    else:
        acts = tokens * d * L * 44

    attn = 0
    if not flash:
        # Without a fused kernel the scores matrix is materialised per layer
        # AND its softmax is kept for the backward pass -- two tensors, not
        # one, which is why this is doubled. Quadratic in sequence length, so
        # it is negligible at 256 tokens and the largest single term at 1024.
        attn = 2 * batch * h * s * s * 2 * L

    # The output logits, and the single most surprising term here.
    #
    # For a small model this is larger than everything else combined: one
    # value per token per vocabulary entry, and the cross-entropy path keeps
    # about five copies of it -- fp16 logits, the fp32 upcast transformers
    # does before the loss, log-softmax, and the gradients of both. Measured
    # on a 5.3M-parameter model at batch 64 x 256 with an 8k vocabulary, this
    # alone was 2.15 GB against 0.72 GB for the whole rest of the run.
    #
    # Omitting it made the estimate 4x optimistic, which is exactly the
    # direction that picks a batch size and then runs out of memory.
    logits = tokens * arch["vocab_size"] * 16

    total = (state + acts + attn + logits) / 1024 ** 3
    return {
        "optimizer_gb": round(state / 1024 ** 3, 2),
        "activations_gb": round((acts + attn) / 1024 ** 3, 2),
        "logits_gb": round(logits / 1024 ** 3, 2),
        "total_gb": round(total, 2),
    }


def pick_batch_size(arch: dict, vram_gb: float | None, *, optim_8bit: bool = False,
                    checkpointing: bool = False, flash: bool = False,
                    target_tokens_per_step: int = 65536) -> dict:
    """Largest batch that fits, then accumulation to reach a sane token count.

    Pretraining wants a large number of tokens per optimiser step -- small
    batches make the gradient noisy and the loss curve unstable. Whatever the
    card cannot hold at once is made up with gradient accumulation, which is
    mathematically equivalent and costs only wall-clock time.
    """
    seq = arch["max_position_embeddings"]
    # 72% of the card, not 100%. The estimate above is good to about 10%, and
    # the allocator's fragmentation is real memory that no formula sees. An
    # over-optimistic batch does not degrade -- it dies at step one, an hour
    # into a download, which is the single worst outcome this app can produce.
    budget = (vram_gb or 8.0) * 0.72

    batch = 1
    for candidate in (64, 48, 32, 24, 16, 12, 8, 6, 4, 2, 1):
        mem = training_memory_gb(arch, candidate, optim_8bit=optim_8bit,
                                 checkpointing=checkpointing, flash=flash)
        if mem["total_gb"] <= budget:
            batch = candidate
            break

    accum = max(1, round(target_tokens_per_step / (batch * seq)))
    mem = training_memory_gb(arch, batch, optim_8bit=optim_8bit,
                             checkpointing=checkpointing, flash=flash)
    return {"batch_size": batch, "grad_accum": accum,
            "tokens_per_step": batch * accum * seq, "memory": mem,
            "fits": mem["total_gb"] <= budget}


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------

def _efficiency(dim: int, flash: bool = False) -> float:
    """Fraction of the GPU's measured peak a model of this width can reach.

    The capability probe multiplies 2048x2048 matrices, which saturates the
    card. A transformer does not reach that, and how far short it falls is the
    difference between a 30-minute estimate and a six-hour run.

    These numbers are MEASURED, on an RX 6900 XT training TinyStories:

        nano   d=256, seq=256, batch 64   222k tokens/s  ->  0.223 of peak
        small  d=512, seq=512, batch 48    50k tokens/s  ->  0.279 of peak

    The direction was right and the magnitude was not: efficiency does climb
    with width, but the first guess here (0.06 at 256 wide, rising to 0.22)
    was nearly four times too pessimistic at the small end. That error would
    have reported the smallest size as unable to finish when in fact it
    reaches a full Chinchilla budget in eight minutes.

    One measurement was itself misleading and worth recording. The same Small
    model first measured 0.203, not 0.279 -- because that run was at batch 64,
    which sat against the memory ceiling and stalled on the allocator. A model
    squeezed into barely enough memory does not fail; it just runs a third
    slower, which is a good reason for the batch planner to leave headroom.

    Sizes above 512 wide are extrapolated, and so is every flash-attention
    figure: no card in this fleet has the kernels.
    """
    if flash:
        return 0.26 if dim <= 256 else 0.30 if dim <= 384 else 0.34
    return 0.22 if dim <= 256 else 0.25 if dim <= 384 else 0.28 if dim <= 512 else 0.30


def tokens_per_second(arch: dict, caps: dict) -> float | None:
    tflops = (caps.get("dtypes") or {}).get(caps.get("recommended_dtype", "float16"))
    if not tflops:
        return None
    n = count_params(arch)["total"]
    flash = bool((caps.get("attention") or {}).get("flash"))
    # 6 FLOPs per parameter per token covers forward and backward.
    return (tflops * 1e12 * _efficiency(arch["hidden_size"], flash)) / (6 * n)


def tokens_in_time(arch: dict, caps: dict, minutes: float) -> int | None:
    tps = tokens_per_second(arch, caps)
    return int(tps * minutes * 60) if tps else None


def time_for_tokens(arch: dict, caps: dict, tokens: float) -> float | None:
    """Minutes needed to train on this many tokens."""
    tps = tokens_per_second(arch, caps)
    return round(tokens / tps / 60, 1) if tps else None


# Verdicts, ordered worst to best. The wording avoids jargon and commits to a
# prediction about the output, because "0.4 tokens per parameter" means
# nothing to someone training their first model.
_VERDICTS = [
    (20.0, "well_trained", "ok",
     "Fully trained for its size. This is the result you want."),
    (8.0, "good", "ok",
     "Slightly undertrained but genuinely useful. A good place to stop."),
    (3.0, "undertrained", "warn",
     "Undertrained. Readable in places, muddled in others -- a smaller model "
     "would do better in the same time."),
    (1.0, "weak", "warn",
     "Badly undertrained. Expect real words in broken sentences."),
    (0.0, "not_viable", "err",
     "Not enough training to produce anything meaningful. Choose a smaller "
     "model or allow much more time."),
]


def training_verdict(arch: dict, tokens: int | None) -> dict:
    n = count_params(arch)["total"]
    if not tokens:
        return {"verdict": "unknown", "tone": "", "ratio": None,
                "message": "Cannot estimate speed on this machine."}
    ratio = tokens / n
    for threshold, verdict, tone, message in _VERDICTS:
        if ratio >= threshold:
            return {"verdict": verdict, "tone": tone, "ratio": round(ratio, 1),
                    "message": message}
    return {"verdict": "not_viable", "tone": "err", "ratio": round(ratio, 1),
            "message": _VERDICTS[-1][3]}


def size_options(caps: dict, minutes: float, vocab_size: int = DEFAULT_VOCAB) -> list[dict]:
    """Every size, scored against this machine and this much patience.

    Returned as one list so the UI can show the trade-off directly: the user
    sees that the small model finishes and the big one does not, rather than
    picking the biggest and finding out overnight.
    """
    vram = caps.get("vram_gb")
    flash = bool((caps.get("attention") or {}).get("flash"))
    optim_8bit = bool((caps.get("quantization") or {}).get("optim_8bit"))

    out = []
    for p in SIZE_PRESETS:
        arch = build_arch(p["id"], vocab_size)
        counts = count_params(arch)
        fit = pick_batch_size(arch, vram, optim_8bit=optim_8bit,
                              checkpointing=False, flash=flash)
        if not fit["fits"]:
            fit = pick_batch_size(arch, vram, optim_8bit=optim_8bit,
                                  checkpointing=True, flash=flash)
            fit["checkpointing"] = True
        achievable = tokens_in_time(arch, caps, minutes)
        recommended = int(counts["total"] * TOKENS_PER_PARAM_TARGET)
        out.append({
            **{k: p[k] for k in ("id", "label", "blurb", "expect")},
            "layers": p["layers"], "dim": p["dim"], "heads": p["heads"],
            "seq": arch["max_position_embeddings"],
            "params": counts["total"],
            "params_label": fmt_params(counts["total"]),
            "embedding_share": counts["embedding_share"],
            "fits": fit["fits"],
            "batch_size": fit["batch_size"],
            "memory_gb": fit["memory"]["total_gb"],
            "tokens_achievable": achievable,
            "tokens_recommended": recommended,
            "coverage": round(achievable / recommended, 3) if achievable else None,
            "minutes_for_full": time_for_tokens(arch, caps, recommended),
            **training_verdict(arch, achievable),
        })

    # Mark the largest size that still finishes properly. Both failure modes
    # are real: too big and it never learns, too small and the leftover time
    # is spent re-reading data the model already knows. The best choice is the
    # boundary between them, so point at it rather than making the user infer
    # it from five sets of numbers.
    viable = [o for o in out if o["fits"]
              and o["verdict"] in ("well_trained", "good")]
    if viable:
        viable[-1]["recommended"] = True
    return out


def fmt_params(n: int) -> str:
    if n >= 1e9:
        return "%.1fB" % (n / 1e9)
    if n >= 1e6:
        return "%.0fM" % (n / 1e6)
    return "%.0fK" % (n / 1e3)


def max_trainable_params(caps: dict) -> int | None:
    """Biggest model this machine could hold for full training.

    Reported for completeness, and almost always the wrong limit to think
    about: compute runs out long before memory does.
    """
    vram = caps.get("vram_gb")
    if not vram:
        return None
    optim_8bit = bool((caps.get("quantization") or {}).get("optim_8bit"))
    per_param = (10 if optim_8bit else 16) + 2
    usable = max(0.0, vram - 3.0) * 1024 ** 3
    return int(usable / per_param)


# ===========================================================================
# Designing your own architecture
# ===========================================================================
#
# The presets above are five points on a very large surface. Someone who knows
# what they want should be able to reach the rest of it -- but a transformer
# has combinations that are silently wrong rather than loudly broken, and a
# from-scratch run is far too slow to discover them by trying. Everything here
# exists to say what is wrong, and why, before anything starts.

LIMITS = {
    "num_hidden_layers":       {"min": 1,  "max": 64,   "label": "Layers"},
    "hidden_size":             {"min": 64, "max": 4096, "label": "Width"},
    "num_attention_heads":     {"min": 1,  "max": 64,   "label": "Attention heads"},
    "max_position_embeddings": {"min": 64, "max": 8192, "label": "Context length"},
    "vocab_size":              {"min": 256, "max": 65536, "label": "Vocabulary"},
    "intermediate_size":       {"min": 64, "max": 16384, "label": "Feed-forward width"},
}

# Head dimensions that attention kernels are actually tuned for. Anything else
# runs, and runs slower.
GOOD_HEAD_DIMS = (32, 48, 64, 80, 96, 128)

# Width divided by depth. Real language models cluster tightly in this range;
# the GPT and Llama families all sit between roughly 60 and 130.
ASPECT_HEALTHY = (32, 320)


def recommended_lr(dim: int) -> float:
    """Peak learning rate for a model of this width.

    Scales as 1/width, which is the rule maximal-update parametrisation
    arrives at and which happens to fit the presets that were tuned by hand:
    1e-3 at 256 wide, 2.5e-4 at 1024. Wider layers produce larger activations,
    and the same step size that trains a narrow model destabilises a wide one.
    """
    return round(min(2e-3, max(1e-4, 0.256 / max(dim, 1))), 6)


def build_custom_arch(spec: dict, vocab_size: int) -> dict:
    """An architecture from raw numbers, with the derivable parts derived."""
    dim = int(spec.get("hidden_size") or 512)
    heads = int(spec.get("num_attention_heads") or max(1, dim // 64))
    kv = int(spec.get("num_key_value_heads") or heads)
    return {
        "size_id": "custom",
        "model_type": "llama",
        "vocab_size": int(vocab_size),
        "hidden_size": dim,
        "intermediate_size": int(spec.get("intermediate_size") or _intermediate(dim)),
        "num_hidden_layers": int(spec.get("num_hidden_layers") or 8),
        "num_attention_heads": heads,
        "num_key_value_heads": kv,
        "max_position_embeddings": int(spec.get("max_position_embeddings") or 512),
        "tie_word_embeddings": bool(spec.get("tie_word_embeddings", True)),
        "rms_norm_eps": 1e-5,
    }


def _issue(level, field, message, fix=None):
    return {"level": level, "field": field, "message": message, "fix": fix}


def validate_arch(arch: dict, caps: dict, *, corpus_tokens: int | None = None,
                  minutes: float | None = None) -> list[dict]:
    """Everything wrong or questionable about this architecture.

    Three levels. `error` cannot be started -- it would crash or refuse to
    build. `warn` will run and will disappoint. `info` is a trade-off worth
    knowing about but not a mistake.
    """
    out: list[dict] = []
    d = arch["hidden_size"]
    layers = arch["num_hidden_layers"]
    heads = arch["num_attention_heads"]
    kv = arch.get("num_key_value_heads", heads)
    seq = arch["max_position_embeddings"]
    vocab = arch["vocab_size"]

    # ---- hard errors ----------------------------------------------------
    for field, bound in LIMITS.items():
        v = arch.get(field)
        if v is None:
            continue
        if v < bound["min"] or v > bound["max"]:
            out.append(_issue("error", field,
                "%s must be between %s and %s." % (bound["label"], bound["min"], bound["max"])))

    if d % heads:
        # Two ways out, and the useful one depends on the numbers. A prime
        # width has no divisors worth suggesting -- "try 1 head" is technically
        # a fix and practically nonsense -- so in that case nudge the width
        # instead, to the nearest multiple of the head count they asked for.
        divisor = _nearest_divisor(d, heads)
        fix = ("Try %d heads." % divisor if divisor > 2 or heads <= 2
               else "Use a width of %d, which divides evenly by %d heads."
                    % (max(heads, round(d / heads) * heads), heads))
        out.append(_issue("error", "num_attention_heads",
            "Width %d cannot be split across %d heads -- attention divides the "
            "width evenly between them, so it has to divide exactly." % (d, heads),
            fix))
    if heads % kv:
        out.append(_issue("error", "num_key_value_heads",
            "Attention heads (%d) must be a multiple of key/value heads (%d)."
            % (heads, kv)))

    # ---- performance warnings -------------------------------------------
    if d % heads == 0:
        head_dim = d // heads
        if head_dim not in GOOD_HEAD_DIMS:
            out.append(_issue("warn", "num_attention_heads",
                "Each head would be %d wide. Attention kernels are written for "
                "%s, and other sizes fall back to a slower path."
                % (head_dim, ", ".join(str(x) for x in GOOD_HEAD_DIMS)),
                "%d heads gives 64 per head." % max(1, d // 64)))

    if d % 64:
        out.append(_issue("warn", "hidden_size",
            "A width that is not a multiple of 64 leaves part of every matrix "
            "tile idle on the GPU. The cost is real and easy to avoid.",
            "Use %d." % (round(d / 64) * 64 or 64)))

    aspect = d / max(layers, 1)
    if aspect < ASPECT_HEALTHY[0]:
        out.append(_issue("warn", "num_hidden_layers",
            "This is very deep for its width (%d layers at %d wide). Deep, "
            "narrow models train slowly and are prone to unstable gradients."
            % (layers, d),
            _layer_advice(d)))
    elif aspect > ASPECT_HEALTHY[1]:
        out.append(_issue("warn", "num_hidden_layers",
            "This is very wide for its depth (%d layers at %d wide). Most of "
            "the parameters end up in a handful of layers, which limits how "
            "much the model can compose." % (layers, d),
            _layer_advice(d)))

    counts = count_params(arch)
    if counts["embedding_share"] > 0.5:
        out.append(_issue("warn", "vocab_size",
            "The vocabulary is %.0f%% of this model. More of it would be spent "
            "on the lookup table than on the network that does the thinking."
            % (counts["embedding_share"] * 100),
            "Shrink the vocabulary, or widen the model."))
    elif counts["embedding_share"] > 0.3:
        out.append(_issue("info", "vocab_size",
            "The vocabulary is %.0f%% of this model's parameters. Workable, "
            "but a smaller one would leave more capacity for the network."
            % (counts["embedding_share"] * 100)))

    flash = bool((caps.get("attention") or {}).get("flash"))
    if seq > 1024 and not flash:
        out.append(_issue("warn", "max_position_embeddings",
            "This machine has no fused attention kernel, so attention memory "
            "grows with the square of context length. At %d tokens that "
            "becomes the largest thing on the card." % seq,
            "1024 or less is comfortable here."))

    if arch.get("intermediate_size") and d:
        ratio = arch["intermediate_size"] / d
        if ratio < 1.5:
            out.append(_issue("warn", "intermediate_size",
                "The feed-forward width is only %.1fx the model width. This is "
                "where most of a transformer's capacity lives, and starving it "
                "wastes the rest of the model." % ratio,
                "%d is the usual choice." % _intermediate(d)))
        elif ratio > 6:
            out.append(_issue("info", "intermediate_size",
                "The feed-forward width is %.1fx the model width, well above "
                "the usual 2.7x. It will work, and most of the parameters will "
                "sit here." % ratio))

    # ---- can it actually run --------------------------------------------
    vram = caps.get("vram_gb")
    if vram:
        optim_8bit = bool((caps.get("quantization") or {}).get("optim_8bit"))
        fit = pick_batch_size(arch, vram, optim_8bit=optim_8bit,
                              checkpointing=False, flash=flash)
        if not fit["fits"]:
            fit = pick_batch_size(arch, vram, optim_8bit=optim_8bit,
                                  checkpointing=True, flash=flash)
            if fit["fits"]:
                out.append(_issue("info", "hidden_size",
                    "This only fits with activation recomputation turned on, "
                    "which costs about 30%% of the speed."))
            else:
                at_one = training_memory_gb(arch, 1, optim_8bit=optim_8bit,
                                            checkpointing=True, flash=flash)
                out.append(_issue("error", "hidden_size",
                    "This will not fit in %.1f GB. Even a single sequence at a "
                    "time needs about %.1f GB." % (vram, at_one["total_gb"]),
                    "Reduce the width, the depth, or the context length."))

    # ---- will it learn anything -----------------------------------------
    if minutes:
        budget = tokens_in_time(arch, caps, minutes)
        verdict = training_verdict(arch, budget)
        if verdict["verdict"] in ("not_viable", "weak"):
            out.append(_issue(
                "error" if verdict["verdict"] == "not_viable" else "warn",
                "budget",
                "In %s this model would see %.1f tokens per parameter. %s"
                % (_hours(minutes), verdict.get("ratio") or 0, verdict["message"]),
                "Choose a smaller model, or allow more time."))
        elif verdict.get("ratio", 0) > TOKENS_PER_PARAM_TARGET * 3:
            out.append(_issue("info", "budget",
                "This model finishes learning well inside your time budget "
                "(%.0f tokens per parameter against a target of %d). A larger "
                "one would use the time better."
                % (verdict["ratio"], TOKENS_PER_PARAM_TARGET)))

    if corpus_tokens and vocab > corpus_tokens / 200:
        out.append(_issue("warn", "vocab_size",
            "A %s-token vocabulary needs far more text than this dataset holds "
            "to be built well. Rare merges will be learned from a handful of "
            "examples." % f"{vocab:,}",
            "Around %s would suit this dataset." % f"{max(256, int(corpus_tokens / 400)):,}"))

    return out


def check_settings(settings: dict, arch: dict) -> list[dict]:
    """Training hyperparameters that will run but should not."""
    out: list[dict] = []
    d = arch["hidden_size"]
    seq = arch["max_position_embeddings"]

    lr = float(settings.get("learning_rate") or 0)
    want = recommended_lr(d)
    if lr <= 0:
        out.append(_issue("error", "learning_rate",
            "The learning rate must be greater than zero."))
    elif lr > want * 3:
        out.append(_issue("warn", "learning_rate",
            "%.1e is %.1fx the rate this width usually tolerates. Too high and "
            "the loss spikes and never recovers." % (lr, lr / want),
            "%.1e is the recommendation for a %d-wide model." % (want, d)))
    elif lr < want / 4:
        out.append(_issue("warn", "learning_rate",
            "%.1e is well below what this width can take. Training will work "
            "and will waste most of your time budget getting nowhere." % lr,
            "%.1e is the recommendation for a %d-wide model." % (want, d)))

    tokens_per_step = (int(settings.get("batch_size") or 1)
                       * int(settings.get("grad_accum") or 1) * seq)
    if tokens_per_step < 16384:
        out.append(_issue("warn", "grad_accum",
            "Only %s tokens per update. Pretraining gradients are noisy, and "
            "below about 16,000 tokens per step the loss curve wanders instead "
            "of descending." % f"{tokens_per_step:,}",
            "Raise gradient accumulation to %d."
            % max(1, round(65536 / max(tokens_per_step, 1)
                           * int(settings.get("grad_accum") or 1)))))

    steps = int(settings.get("max_steps") or 0)
    warmup = int(settings.get("warmup_steps") or 0)
    if steps and warmup > steps * 0.5:
        out.append(_issue("warn", "warmup_steps",
            "Warmup covers more than half the run, so the model spends most of "
            "its time below the learning rate you chose."))
    elif steps and warmup < 5:
        out.append(_issue("warn", "warmup_steps",
            "Almost no warmup. A model starting from random weights takes a "
            "large, badly-aimed first step without it.",
            "%d steps is a reasonable warmup." % max(10, steps // 20)))

    wd = settings.get("weight_decay")
    if wd is not None and not (0 <= float(wd) <= 0.5):
        out.append(_issue("warn", "weight_decay",
            "Weight decay is normally between 0 and 0.2. Outside that range it "
            "either does nothing or pulls the model apart."))

    clip = settings.get("grad_clip")
    if clip is not None and float(clip) <= 0:
        out.append(_issue("error", "grad_clip",
            "Gradient clipping must be greater than zero."))
    elif clip is not None and float(clip) > 5:
        out.append(_issue("info", "grad_clip",
            "Clipping above 5 effectively disables it, which leaves the run "
            "exposed to a single bad batch."))

    return out


def _layer_advice(dim: int) -> str:
    """Depth that suits a given width, phrased for a human."""
    n = max(2, round(dim / 96))
    return "Around %d layer%s suits a width of %d." % (n, "" if n == 1 else "s", dim)


def _nearest_divisor(n: int, near: int) -> int:
    divisors = [i for i in range(1, min(n, 64) + 1) if n % i == 0]
    return min(divisors, key=lambda x: abs(x - near)) if divisors else 1


def _hours(minutes: float) -> str:
    if minutes < 90:
        return "%d minutes" % round(minutes)
    if minutes < 2880:
        return "%.1f hours" % (minutes / 60)
    return "%.1f days" % (minutes / 1440)
