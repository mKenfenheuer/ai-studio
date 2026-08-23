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
# Shapes with proportions that are known to work, from a toy to something that
# needs a serious card. Width, layers, heads, context.
#
# The five sizes offered are a WINDOW onto this ladder rather than a fixed
# list, positioned by how much memory the machine has. A fixed list meant that
# on a small card the top two could never be trained, and on a large one the
# offer stopped well short of what the card could do. In both directions the
# studio was describing some other computer.
SHAPE_LADDER = [
    (256, 4, 4, 256),
    (384, 6, 6, 512),
    (512, 8, 8, 512),
    (640, 10, 10, 1024),
    (768, 12, 12, 1024),
    (1024, 16, 16, 1024),
    (1280, 20, 20, 2048),
    (1536, 24, 24, 2048),
    (2048, 24, 32, 2048),
]

# Five names, smallest to largest. They are relative to the machine: "Large"
# means the largest this card can train at all, not a fixed parameter count.
# Which is why every option also carries its parameter count, and why what is
# said about each comes from that count rather than from the name.
SIZE_TIERS = [
    ("nano", "Nano"), ("tiny", "Tiny"), ("small", "Small"),
    ("base", "Base"), ("large", "Large"),
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


# ---------------------------------------------------------------------------
# Mixture of experts
# ---------------------------------------------------------------------------
#
# A mixture-of-experts layer replaces each block's single feed-forward network
# with several, and a small router that sends every token to just a few of
# them. The appeal is real: parameters grow with the number of experts while
# the arithmetic per token grows only with how many are used.
#
# The appeal is also the trap, and it is worth being blunt about which half of
# it applies to one consumer GPU:
#
#   * Memory follows the TOTAL parameter count. Every expert is resident,
#     every expert carries fp32 master weights and two Adam moments. Eight
#     experts is eight feed-forward networks in VRAM whether a token visits
#     them or not. MoE does not make a model cheaper to hold -- it makes it
#     much more expensive.
#   * Compute follows the ACTIVE parameter count, which is where the win is,
#     and it is a win against a *dense model of the same total size* -- not
#     against the dense model you would otherwise have trained.
#   * Data requirements follow the total count more than the active one. Each
#     expert only learns from the tokens routed to it, so E experts split the
#     corpus E/k ways. This is the failure nobody predicts: a MoE model that
#     fits, trains, and produces worse text than the dense model it replaced,
#     because each expert saw an eighth of the data.
#
# So MoE is offered here with the arithmetic exposed rather than hidden, and
# validate_arch() says plainly when it is the wrong tool.

MOE_DEFAULT_EXPERTS = 8
MOE_DEFAULT_ACTIVE = 2
# Load balancing. Without this term the router collapses within a few hundred
# steps: whichever expert is marginally better early gets more tokens, trains
# faster, and gets more still. 0.01 is the Mixtral and Switch-Transformer
# value and there is no reason to differ.
MOE_AUX_LOSS_COEF = 0.01

MOE_LIMITS = {
    "num_local_experts":  {"min": 2, "max": 64, "label": "Experts"},
    "num_experts_per_tok": {"min": 1, "max": 16, "label": "Experts per token"},
}


def is_moe(arch: dict) -> bool:
    return int(arch.get("num_local_experts") or 0) > 1


def apply_moe(arch: dict, moe: dict | None) -> dict:
    """Turn a dense architecture into a sparse one, or leave it alone.

    MoE is deliberately orthogonal to size: it is something done *to* a chosen
    shape, so that the cost of turning it on is visible as a change against a
    model the user has already understood.
    """
    if not moe or not moe.get("enabled"):
        return arch
    experts = int(moe.get("num_local_experts") or MOE_DEFAULT_EXPERTS)
    active = int(moe.get("num_experts_per_tok") or MOE_DEFAULT_ACTIVE)
    out = {
        **arch,
        # Mixtral's names, because that is the config the runner builds. Using
        # the library's own field names means nothing has to be translated
        # between the plan shown here and the model that gets constructed.
        "model_type": "mixtral",
        "num_local_experts": experts,
        "num_experts_per_tok": active,
        "router_aux_loss_coef": float(
            moe.get("router_aux_loss_coef", MOE_AUX_LOSS_COEF)),
    }
    # Each expert gets its own feed-forward width. Left at the dense default,
    # eight experts multiply the model's largest component by eight; most MoE
    # designs shrink each expert so that the total lands somewhere sane. Only
    # applied when the caller did not say otherwise.
    if moe.get("expert_intermediate_size"):
        out["intermediate_size"] = int(moe["expert_intermediate_size"])
    return out


def moe_spec(arch: dict) -> dict | None:
    """The MoE part of an architecture, as the UI passes it around."""
    if not is_moe(arch):
        return None
    return {
        "enabled": True,
        "num_local_experts": int(arch["num_local_experts"]),
        "num_experts_per_tok": int(arch.get("num_experts_per_tok") or 1),
        "expert_intermediate_size": int(arch["intermediate_size"]),
        "router_aux_loss_coef": float(
            arch.get("router_aux_loss_coef", MOE_AUX_LOSS_COEF)),
    }


def _intermediate(dim: int) -> int:
    """SwiGLU uses three matrices instead of two, so the hidden width is
    scaled by 8/3 rather than 4 to keep the parameter count comparable."""
    return int(round(8 * dim / 3 / 64)) * 64


def _arch_from_shape(shape: tuple, vocab_size: int) -> dict:
    dim, layers, heads, seq = shape
    return {
        "model_type": "llama", "vocab_size": int(vocab_size),
        "hidden_size": dim, "intermediate_size": _intermediate(dim),
        "num_hidden_layers": layers, "num_attention_heads": heads,
        "num_key_value_heads": heads, "max_position_embeddings": seq,
        "tie_word_embeddings": True, "rms_norm_eps": 1e-5,
    }


def _describe(params: int) -> tuple[str, str]:
    """What a model of this size is for, and what to expect from it.

    Keyed on the parameter count and never on the tier, because the tiers
    slide with the machine: "Nano" on a 24 GB card can be bigger than "Small"
    on an 8 GB one, and a description tied to the name would be wrong on both.
    """
    if params < 15e6:
        return ("Learns spelling, punctuation and short sentence shapes. "
                "Finishes in minutes, so it is the right way to check your "
                "data and settings before committing to a long run.",
                "Real words, wobbly grammar.")
    if params < 60e6:
        return ("The smallest size that writes genuinely coherent English "
                "when trained on simple text. A satisfying first real model.",
                "Short readable passages that stay on topic.")
    if params < 200e6:
        return ("Noticeably better sentence structure and longer memory. "
                "Wants several hours of training to earn its size.",
                "Fluent paragraphs, simple reasoning.")
    if params < 600e6:
        return ("A serious model that needs serious compute -- expect a day "
                "or more on one consumer card.",
                "Good text, but only if you train it properly.")
    return ("At the edge of what one card can train at all. It will not reach "
            "a full budget of text; what you get is a glimpse of a larger "
            "model rather than a finished one.",
            "Capable in places, and visibly undertrained.")


def _shape_fits(shape: tuple, vocab_size: int, caps: dict) -> bool:
    """Whether this machine can train this shape at all -- one sequence at a
    time with activation recomputation, which is the most frugal it gets."""
    vram = caps.get("vram_gb")
    if not vram:
        return True
    mem = training_memory_gb(
        _arch_from_shape(shape, vocab_size), 1,
        optim_8bit=bool((caps.get("quantization") or {}).get("optim_8bit")),
        checkpointing=True,
        flash=bool((caps.get("attention") or {}).get("flash")))
    return mem["total_gb"] <= usable_vram_gb(vram)


def size_presets(caps: dict | None = None,
                 vocab_size: int = 8192) -> list[dict]:
    """The five sizes worth offering on this machine, smallest first."""
    top = len(SHAPE_LADDER) - 1
    if caps:
        while top > 0 and not _shape_fits(SHAPE_LADDER[top], vocab_size, caps):
            top -= 1
    # Spread across the ladder from the bottom rung to whatever this card can
    # reach, rather than sliding a fixed-width window up it. The smallest
    # option has a job -- finish in minutes so you can check your data before
    # committing to a long run -- and a sliding window destroyed it: on a
    # 16 GB card "Nano" became a 55M model, which is not a sanity check.
    # So the bottom is pinned and only the top moves.
    n = min(len(SIZE_TIERS), top + 1)
    picks = sorted({round(i * top / max(n - 1, 1)) for i in range(n)})
    window = [SHAPE_LADDER[i] for i in picks]
    tiers = SIZE_TIERS[-len(window):]
    out = []
    for (size_id, label), shape in zip(tiers, window):
        params = count_params(_arch_from_shape(shape, vocab_size))["total"]
        blurb, expect = _describe(params)
        dim, layers, heads, seq = shape
        out.append({"id": size_id, "label": label, "layers": layers,
                    "dim": dim, "heads": heads, "seq": seq,
                    "blurb": blurb, "expect": expect})
    return out


def preset(size_id: str, caps: dict | None = None) -> dict | None:
    return next((p for p in size_presets(caps) if p["id"] == size_id), None)


def build_arch(size_id: str, vocab_size: int = DEFAULT_VOCAB,
               seq_len: int | None = None, moe: dict | None = None,
               caps: dict | None = None) -> dict | None:
    """Concrete architecture the runner can hand straight to transformers.

    Resolved here rather than on the runner so that the parameter count shown
    in the UI and the model that actually gets built can never disagree.
    """
    p = preset(size_id, caps)
    if not p:
        return None
    seq = int(seq_len or p["seq"])
    return apply_moe({
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
    }, moe)


def count_params(arch: dict) -> dict:
    """Exact parameter count for a Llama- or Mixtral-shaped model.

    Split into embedding and body because the ratio is the thing worth
    showing: when embeddings dominate, the vocabulary is too big for the
    model, and shrinking it buys real capacity for free.

    For a mixture of experts there are two counts that both matter and mean
    different things. `total` is what has to be held in memory and what has to
    be learned; `active` is what each token actually passes through, and so
    what sets the speed. Reporting only one of them is how MoE gets sold as
    free capacity.
    """
    d = arch["hidden_size"]
    L = arch["num_hidden_layers"]
    v = arch["vocab_size"]
    i = arch["intermediate_size"]
    kv = (arch["num_key_value_heads"] * d) // arch["num_attention_heads"]

    embedding = v * d
    if not arch.get("tie_word_embeddings", True):
        embedding *= 2

    attn = (
        d * d          # q
        + d * kv       # k
        + d * kv       # v
        + d * d        # o
        + 2 * d        # two RMSNorms
    )
    one_ffn = 3 * d * i                       # gate, up, down

    experts = int(arch.get("num_local_experts") or 0)
    if experts > 1:
        chosen = min(int(arch.get("num_experts_per_tok") or 1), experts)
        router = d * experts                  # one linear, width -> experts
        ffn_total = experts * one_ffn + router
        ffn_active = chosen * one_ffn + router
    else:
        chosen = 0
        ffn_total = ffn_active = one_ffn

    body = L * (attn + ffn_total) + d         # + final norm
    body_active = L * (attn + ffn_active) + d
    total = embedding + body
    active = embedding + body_active

    return {
        "embedding": embedding, "body": body, "total": total,
        "active": active,
        "experts": experts if experts > 1 else 0,
        "experts_per_token": chosen,
        "expert_params": L * experts * one_ffn if experts > 1 else 0,
        "effective": effective_params(total, active),
        "embedding_share": round(embedding / max(total, 1), 3),
    }


def effective_params(total: int, active: int) -> int:
    """The dense model a sparse one is worth, for the purpose of Chinchilla.

    A MoE model does not learn like a dense model of its total size, nor like
    one of its active size -- it sits between the two, and the geometric mean
    is the approximation the scaling-law work on sparse models keeps arriving
    at. It is a rule of thumb and is treated as one: it decides how much text
    this app recommends, not whether anything runs.

    The important consequence, and the reason it is worth computing at all:
    turning on eight experts roughly doubles the amount of text the model
    needs before it is worth its size. Someone who expects MoE to be free
    capacity will otherwise train it on a dense model's budget and get a
    worse result than they started with.
    """
    if active >= total:
        return total
    return int(round((float(total) * float(active)) ** 0.5))


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

# Everything the terms below do not name: cuBLAS workspaces, the quantisation
# buffers the 8-bit optimiser allocates during its step, and the slack inside
# blocks the allocator has already handed out. Not derived -- measured, and
# the same factor fits both runs there is data for, which is the only reason
# it is a single number rather than a term:
#
#   29M,  d=512,  8 layers, 512 tokens,  batch 48, no checkpointing
#         10.97 GB predicted raw   vs   11.89 GB measured   (+8.4%)
#   207M, d=1024, 15 layers, 4096 tokens, batch 1, checkpointing
#          4.10 GB predicted raw   vs    4.56 GB measured   (+11.2%)
#
# Both are `max_memory_allocated`, which is what a run reports. What the
# allocator RESERVES is higher again, and that is what the separate headroom
# in `usable_vram_gb` is for -- these are two different overheads and folding
# them into one number is how an estimate ends up unable to explain itself.
UNNAMED_OVERHEAD = 1.10


def training_memory_gb(arch: dict, batch: int, *, optim_8bit: bool = False,
                       checkpointing: bool = False, flash: bool = False) -> dict:
    """VRAM needed to train every parameter of this model.

    Nothing like the LoRA estimate in hub.py. There the base model is frozen
    and only the adapter carries optimiser state; here every parameter needs
    fp32 master weights, fp32 gradients and two Adam moments -- 16 bytes per
    parameter before a single activation is stored.
    """
    counts = count_params(arch)
    n = counts["total"]
    d = arch["hidden_size"]
    L = arch["num_hidden_layers"]
    h = arch["num_attention_heads"]
    s = arch["max_position_embeddings"]

    # 4 (weights) + 4 (grads) + 8 (Adam m and v), or 2 bytes of state with an
    # 8-bit optimiser. Plus the half-precision copies autocast keeps around.
    #
    # Charged against the TOTAL parameter count, including every expert that
    # no token happens to visit. A sparse model is sparse in arithmetic, never
    # in memory, and this line is where that shows up.
    per_param = 10 if optim_8bit else 16
    state = n * (per_param + 2)

    tokens = batch * s
    # 44 bytes per token, per unit of width, per layer. Counting the tensors a
    # Llama block keeps for its backward pass gives ~34; the rest is the parts
    # autocast keeps in fp32 rather than fp16 -- both RMSNorms and the rotary
    # inputs. Calibrated against two measured runs rather than derived, and it
    # is deliberately the conservative end of what they imply.
    per_layer_bytes = 44.0
    if is_moe(arch):
        # Of those 44 bytes, the feed-forward network's share is the three
        # i-wide tensors it keeps -- gate, up and the activation between them.
        # A token that visits k experts keeps k copies of that, so this is the
        # one term MoE genuinely multiplies. The attention half is untouched.
        ffn_bytes = 6.0 * arch["intermediate_size"] / max(d, 1)
        k = max(1, counts["experts_per_token"])
        per_layer_bytes = (per_layer_bytes - ffn_bytes) + k * ffn_bytes
    if checkpointing:
        # Only layer inputs survive the forward pass; the rest is recomputed.
        acts = tokens * d * L * 2 + tokens * d * per_layer_bytes
        # ...and that is exactly why only ONE layer's attention scores are
        # ever live. See the note on `live_layers` below.
        live_layers = 1
    else:
        acts = tokens * d * L * per_layer_bytes
        live_layers = L

    attn = 0
    if not flash:
        # Without a fused kernel the scores matrix is materialised per layer
        # AND its softmax is kept for the backward pass -- two tensors, not
        # one, which is why this is doubled. Quadratic in sequence length, so
        # it is negligible at 256 tokens and dominant at 4096.
        #
        # Multiplied by the number of layers whose scores are alive AT ONCE,
        # which is not the same as the number of layers. Without checkpointing
        # every layer keeps its own for the backward pass, so it is all of
        # them. With checkpointing nothing is kept: each layer's scores are
        # rebuilt during its own recomputation and freed again, so the peak
        # holds one.
        #
        # This was charged against every layer in both cases, and it is the
        # largest term in the whole estimate at a long context. On a 207M
        # model at 4096 tokens it predicted 7.50 GB of attention scores where
        # 0.50 GB was live, and turned a 4.6 GB run into an 11.1 GB one --
        # which then forced batch 1 to "fit", on a card with 11 GB spare.
        attn = 2 * batch * h * s * s * 2 * live_layers

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

    total = (state + acts + attn + logits) * UNNAMED_OVERHEAD / 1024 ** 3
    return {
        "optimizer_gb": round(state / 1024 ** 3, 2),
        "activations_gb": round((acts + attn) / 1024 ** 3, 2),
        "logits_gb": round(logits / 1024 ** 3, 2),
        "total_gb": round(total, 2),
    }


# 72% of the card, not 100%. The estimate is good to about 10%, the
# allocator's fragmentation is real memory that no formula sees, and the
# desktop compositor is often on the same card. An over-optimistic batch does
# not degrade gracefully -- it dies at step one, an hour into a download,
# which is the single worst outcome this app can produce.
VRAM_USABLE_FRACTION = 0.72


def usable_vram_gb(vram_gb: float | None) -> float:
    """How much of a card can actually be planned against.

    Named, and used by both the decision and the sentence that explains it.
    They were the same number written twice, and a message that says "will not
    fit in 16.0 GB" about a model needing 15.1 GB is what that costs: true,
    and unreadable, because the 16.0 is not the number being compared.
    """
    return (vram_gb or 8.0) * VRAM_USABLE_FRACTION


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
    budget = usable_vram_gb(vram_gb)

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

# How much slower the attention matmuls are than the rest of the model when
# there is no fused kernel. They are memory-bound, not arithmetic-bound: the
# scores matrix is written out and read back rather than staying in registers,
# so the card spends its time moving s x s numbers instead of multiplying.
#
# 0.40, from the three runs below -- and it is the term that makes the
# difference at a long context. Attention is 9% of the arithmetic at 256
# tokens and 38% of it at 4096, which is why an estimate calibrated on short
# runs was 2.8x optimistic on a 4096-token one.
ATTENTION_EFFICIENCY = 0.40

# What gradient checkpointing costs. The forward pass is thrown away and done
# again during the backward pass: 2N + 4N becomes 2N + 2N + 4N, so 4/3.
CHECKPOINT_COST = 4.0 / 3.0


def _efficiency(dim: int, flash: bool = False) -> float:
    """Fraction of the GPU's measured peak a model of this width can reach.

    The capability probe multiplies 2048x2048 matrices, which saturates the
    card. A transformer does not, and how far short it falls is the difference
    between a three-hour estimate and an eleven-hour run.

    MEASURED, on an RX 6900 XT (gfx1030, fp16, 31.6 TFLOPS by the probe):

        nano   d=256,  4 layers, 256 tokens,  batch 64   222,000 tok/s
        small  d=512,  8 layers, 512 tokens,  batch 48    49,809 tok/s
        custom d=1024, 15 layers, 4096 tokens, batch 1     2,752 tok/s
                                              (checkpointing on)

    Solved against the FLOP model in `tokens_per_second` -- which now counts
    attention and recomputation, both of which this function used to absorb --
    those three imply efficiencies of 0.279, 0.374 and 0.363. So it rises
    steeply from 256 to 512 and then flattens, which is what "small matmuls
    are what a GPU is bad at" predicts.

    The previous version of this table absorbed attention into the width
    number, which worked at 512 tokens and failed badly at 4096: the same
    32-fold growth in the attention term had nowhere to go. That is the error
    the third measurement here caught.

    One measurement was itself misleading and worth recording. The 512-wide
    model first measured 0.203 of peak, not 0.374 -- because that run was at
    batch 64, which sat against the memory ceiling and stalled on the
    allocator. A model squeezed into barely enough memory does not fail; it
    runs a third slower, which is a good reason for the batch planner to leave
    headroom.

    Widths above 1024 are extrapolated flat rather than upward. Guessing high
    on speed means promising an overnight run that takes two nights.
    """
    del flash  # a fused kernel changes the attention term, not this one
    return 0.28 if dim <= 256 else 0.33 if dim <= 384 else 0.37


def moe_time_multiplier(experts: int, active: int) -> float:
    """How much longer a sparse block takes than the dense block it replaced.

    The arithmetic predicts the opposite of what happens, so this is measured.
    A sparse block does not do one big matmul; it sorts tokens by which expert
    each was routed to and does one small matmul per expert. Small matmuls are
    exactly what a GPU is bad at, and the sorting costs the same whether one
    expert is chosen or four.

    MEASURED on the RX 6900 XT (gfx1030, fp16), 8 layers at 512 wide, an
    8k vocabulary, batch 8 x 512, twelve optimiser steps after warmup:

        dense                 47,136 tok/s     1.00x time
        4 experts, 2 active   25,303 tok/s     1.86x
        8 experts, 1 active   19,543 tok/s     2.41x
        8 experts, 2 active   17,652 tok/s     2.67x
        16 experts, 2 active   9,324 tok/s     5.05x

    Two things fall out of that, and both contradict the sales pitch:

    * Cost is driven by how many experts EXIST, not by how many are used.
      Going from 1 active expert to 2 -- doubling the arithmetic each token
      does -- cost 11%. Adding experts that most tokens never touch cost far
      more. The routing dominates the maths.

    * There is no configuration here that beat the dense model. A sparse model
      is quicker than a dense model OF THE SAME TOTAL SIZE, which is not the
      choice anyone is actually making on one GPU.

    Fitted as time/dense_time = 0.77 + 0.24*E + 0.26*(k-1), which reproduces
    the five measurements to within 11% and is conservative in four of them.
    The first attempt at this was 1/(1 + c*E) applied to the active parameter
    count; it could not fit the data at all -- the implied c ranged over 3x --
    which is what measuring it was for.
    """
    if experts <= 1:
        return 1.0
    return 0.77 + 0.24 * experts + 0.26 * (max(active, 1) - 1)


def tokens_per_second(arch: dict, caps: dict,
                      checkpointing: bool = False) -> float | None:
    """Training throughput, from the card's measured peak and this shape.

    Three terms, because the studio got each of them wrong in turn:

    * **the model itself**, 6 FLOPs per parameter per token, forward and
      backward. This was the only term for a long time.
    * **attention**, 12 * layers * context * width per token, which does not
      scale with parameter count at all. It is a rounding error at 256 tokens
      and more than a third of the work at 4096, and leaving it out is why a
      4096-token run was predicted at 7,650 tokens per second and delivered
      2,752.
    * **recomputation**, when gradient checkpointing is on. The forward pass
      is simply done twice. The planner has been turning checkpointing on to
      make things fit and then quoting a time that assumed it was off.
    """
    tflops = (caps.get("dtypes") or {}).get(caps.get("recommended_dtype", "float16"))
    if not tflops:
        return None
    counts = count_params(arch)
    flash = bool((caps.get("attention") or {}).get("flash"))
    eff = _efficiency(arch["hidden_size"])
    peak = tflops * 1e12 * eff
    if peak <= 0:
        return None

    layers = arch["num_hidden_layers"]
    seq = arch["max_position_embeddings"]
    width = arch["hidden_size"]

    if counts["experts"] > 1:
        # Measured against the same shape with one feed-forward network per
        # block, not against the active parameter count. That was the first
        # attempt and the data rejected it: routing overhead scales with how
        # many experts exist, and the active count does not know how many
        # exist.
        dense = count_params({k: v for k, v in arch.items()
                              if k != "num_local_experts"})["total"]
        weights_s = (6 * dense / peak) * moe_time_multiplier(
            counts["experts"], counts["experts_per_token"])
    else:
        weights_s = 6 * counts["total"] / peak

    # Attention is untouched by a mixture of experts -- only the feed-forward
    # half is routed -- so it sits outside the multiplier above.
    attention_flops = 12 * layers * seq * width
    attention_s = attention_flops / (peak * (1.0 if flash else ATTENTION_EFFICIENCY))

    seconds_per_token = weights_s + attention_s
    if checkpointing:
        seconds_per_token *= CHECKPOINT_COST
    return 1.0 / seconds_per_token if seconds_per_token > 0 else None


def tokens_in_time(arch: dict, caps: dict, minutes: float,
                   checkpointing: bool = False) -> int | None:
    tps = tokens_per_second(arch, caps, checkpointing)
    return int(tps * minutes * 60) if tps else None


def time_for_tokens(arch: dict, caps: dict, tokens: float,
                    checkpointing: bool = False) -> float | None:
    """Minutes needed to train on this many tokens."""
    tps = tokens_per_second(arch, caps, checkpointing)
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
    # Measured against effective parameters, which for a dense model is just
    # its size. For a sparse one it is the number that decides how much text
    # the model needs -- not the active count, which would flatter it badly.
    n = count_params(arch)["effective"]
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


def size_options(caps: dict, minutes: float, vocab_size: int = DEFAULT_VOCAB,
                 moe: dict | None = None) -> list[dict]:
    """Every size, scored against this machine and this much patience.

    Returned as one list so the UI can show the trade-off directly: the user
    sees that the small model finishes and the big one does not, rather than
    picking the biggest and finding out overnight.
    """
    vram = caps.get("vram_gb")
    flash = bool((caps.get("attention") or {}).get("flash"))
    optim_8bit = bool((caps.get("quantization") or {}).get("optim_8bit"))

    out = []
    for p in size_presets(caps, vocab_size):
        arch = build_arch(p["id"], vocab_size, moe=moe, caps=caps)
        counts = count_params(arch)
        fit = pick_batch_size(arch, vram, optim_8bit=optim_8bit,
                              checkpointing=False, flash=flash)
        if not fit["fits"]:
            fit = pick_batch_size(arch, vram, optim_8bit=optim_8bit,
                                  checkpointing=True, flash=flash)
            fit["checkpointing"] = True
        achievable = tokens_in_time(arch, caps, minutes,
                                    fit.get("checkpointing", False))
        recommended = int(counts["effective"] * TOKENS_PER_PARAM_TARGET)
        out.append({
            **{k: p[k] for k in ("id", "label", "blurb", "expect")},
            "layers": p["layers"], "dim": p["dim"], "heads": p["heads"],
            "seq": arch["max_position_embeddings"],
            "params": counts["total"],
            "params_label": fmt_params(counts["total"]),
            "active_params": counts["active"],
            "active_params_label": fmt_params(counts["active"]),
            "experts": counts["experts"],
            "experts_per_token": counts["experts_per_token"],
            "embedding_share": counts["embedding_share"],
            "fits": fit["fits"],
            "batch_size": fit["batch_size"],
            "memory_gb": fit["memory"]["total_gb"],
            # The weights-and-optimiser half of the memory, which is the only
            # part that does not move when the batch size does. Two cards
            # showing total memory can mislead badly: a sparse model has more
            # weights, so the planner gives it a smaller batch, so its total
            # can land BELOW the dense model's -- which reads as a saving and
            # is the exact opposite of what happened.
            "weights_gb": fit["memory"]["optimizer_gb"],
            "tokens_achievable": achievable,
            "tokens_recommended": recommended,
            "coverage": round(achievable / max(recommended, 1), 3) if achievable else None,
            "minutes_for_full": time_for_tokens(
                arch, caps, recommended, fit.get("checkpointing", False)),
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

# The stops a slider clicks through. Free-typing any number is still allowed --
# there is a box beside each one -- but dragging should never land on a value
# that is merely legal. A width of 517 divides by no sensible head count; a
# context of 1500 wastes the last block of every tile.
WIDTH_STOPS = (128, 192, 256, 320, 384, 448, 512, 640, 768, 896, 1024,
               1152, 1280, 1536, 1792, 2048, 2560, 3072, 4096)
DEPTH_STOPS = (1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 64)
CONTEXT_STOPS = (128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192)

# Width divided by depth. Real language models cluster tightly in this range;
# the GPT and Llama families all sit between roughly 60 and 130.
ASPECT_HEALTHY = (32, 320)


def _head_stops(dim: int) -> list[int]:
    """Head counts that divide this width into a size the kernels like.

    This is the "based on the rest" part: change the width and the heads
    slider changes under it, because the two are not independent. A head
    dimension that is not one of the tuned sizes runs -- and runs slower --
    and one that does not divide the width at all does not run.
    """
    good = [dim // hd for hd in GOOD_HEAD_DIMS
            if dim % hd == 0 and 1 <= dim // hd <= LIMITS["num_attention_heads"]["max"]]
    if good:
        return sorted(set(good))
    # An awkward width still gets something to drag along: every divisor.
    return sorted({h for h in range(1, LIMITS["num_attention_heads"]["max"] + 1)
                   if dim % h == 0}) or [1]


def _ffn_stops(dim: int) -> list[int]:
    """Feed-forward widths from about 1.5x to 5x the model width.

    The canonical value is always in the list, so the slider can land exactly
    on what the studio would have chosen for you.
    """
    step = 128 if dim >= 512 else 64
    lo, hi = int(dim * 1.5), int(dim * 5)
    stops = {int(round(v / step) * step) for v in
             (lo + (hi - lo) * i / 12 for i in range(13))}
    stops.add(_intermediate(dim))
    return sorted(v for v in stops
                  if LIMITS["intermediate_size"]["min"] <= v
                  <= LIMITS["intermediate_size"]["max"])


def slider_scales(arch: dict, caps: dict | None = None) -> dict:
    """For each dimension of a custom model: where the slider clicks, and how
    far along it this machine can still hold the result.

    The stops depend on the rest of the architecture, which is the whole
    point -- heads follow width, feed-forward width follows width -- so this
    is recomputed with every plan rather than fixed once.
    """
    dim = arch["hidden_size"]
    scales = {
        "hidden_size": list(WIDTH_STOPS),
        "num_hidden_layers": [v for v in DEPTH_STOPS
                              if v <= LIMITS["num_hidden_layers"]["max"]],
        "num_attention_heads": _head_stops(dim),
        "max_position_embeddings": [
            v for v in CONTEXT_STOPS
            if v <= LIMITS["max_position_embeddings"]["max"]],
        "intermediate_size": _ffn_stops(dim),
    }

    out = {}
    for field, stops in scales.items():
        out[field] = {
            "stops": stops,
            "label": LIMITS.get(field, {}).get("label", field),
            "fits_up_to": _largest_that_fits(arch, field, stops, caps),
        }
    return out


def _largest_that_fits(arch: dict, field: str, stops: list[int],
                       caps: dict | None) -> int | None:
    """The last stop this card can still train, everything else held still.

    Not a limit -- turning one thing down makes room for another, and the
    slider will happily go past it. It is a mark on the track saying where
    the machine gives out, which is the question people are actually asking
    when they drag one of these.
    """
    if not caps or not caps.get("vram_gb"):
        return None          # no machine in hand: no mark to draw
    budget = usable_vram_gb(caps["vram_gb"])
    optim_8bit = bool((caps.get("quantization") or {}).get("optim_8bit"))
    flash = bool((caps.get("attention") or {}).get("flash"))
    best = None
    for value in stops:
        candidate = dict(arch)
        candidate[field] = value
        if field == "hidden_size":
            # Width drags the parts derived from it along, or the answer is
            # about a model nobody would build.
            candidate["intermediate_size"] = _intermediate(value)
            heads = _head_stops(value)
            candidate["num_attention_heads"] = heads[len(heads) // 2]
            candidate["num_key_value_heads"] = candidate["num_attention_heads"]
        mem = training_memory_gb(candidate, 1, optim_8bit=optim_8bit,
                                 checkpointing=True, flash=flash)
        if mem["total_gb"] <= budget:
            best = value
        else:
            break
    # Zero, not None, when even the smallest stop is too big. The two mean
    # opposite things -- "no machine to check against" and "none of this fits
    # on the one you have" -- and collapsing them into None made the screen
    # draw a full, unmarked track for a model that could not be built at all.
    return best if best is not None else 0


def recommended_lr(dim: int) -> float:
    """Peak learning rate for a model of this width.

    Scales as 1/width, which is the rule maximal-update parametrisation
    arrives at and which happens to fit the presets that were tuned by hand:
    1e-3 at 256 wide, 2.5e-4 at 1024. Wider layers produce larger activations,
    and the same step size that trains a narrow model destabilises a wide one.
    """
    return round(min(2e-3, max(1e-4, 0.256 / max(dim, 1))), 6)


def build_custom_arch(spec: dict, vocab_size: int, moe: dict | None = None) -> dict:
    """An architecture from raw numbers, with the derivable parts derived."""
    dim = int(spec.get("hidden_size") or 512)
    heads = int(spec.get("num_attention_heads") or max(1, dim // 64))
    kv = int(spec.get("num_key_value_heads") or heads)
    return apply_moe({
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
    }, moe)


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

    out += _validate_moe(arch, caps, minutes=minutes)

    # ---- can it actually run --------------------------------------------
    vram = caps.get("vram_gb")
    recomputing = False
    if vram:
        optim_8bit = bool((caps.get("quantization") or {}).get("optim_8bit"))
        fit = pick_batch_size(arch, vram, optim_8bit=optim_8bit,
                              checkpointing=False, flash=flash)
        if not fit["fits"]:
            fit = pick_batch_size(arch, vram, optim_8bit=optim_8bit,
                                  checkpointing=True, flash=flash)
            recomputing = fit["fits"]
            if fit["fits"]:
                out.append(_issue("info", "hidden_size",
                    "This only fits by recomputing activations instead of "
                    "storing them, which does the forward pass twice and "
                    "makes the run about a quarter slower. The time shown "
                    "already accounts for it."))
            else:
                at_one = training_memory_gb(arch, 1, optim_8bit=optim_8bit,
                                            checkpointing=True, flash=flash)
                out.append(_issue("error", "hidden_size",
                    "This will not fit. One sequence at a time needs about "
                    "%.1f GB, and only about %.1f GB of this %.1f GB card can "
                    "be planned against -- the rest goes to memory "
                    "fragmentation, to the display, and to the margin of error "
                    "in the estimate itself."
                    % (at_one["total_gb"], usable_vram_gb(vram), vram),
                    "Reduce the width, the depth, or the context length."))

    # ---- will it learn anything -----------------------------------------
    if minutes:
        budget = tokens_in_time(arch, caps, minutes, recomputing)
        verdict = training_verdict(arch, budget)
        if verdict["verdict"] in ("not_viable", "weak"):
            # A warning, never an error, however bad the number is. Not fitting
            # in memory is a fact -- the run dies at step one. This is a
            # prediction about how good the result will be, and somebody
            # deliberately training a tiny model to watch the loss move, or
            # stopping early on purpose, is entitled to overrule it. Refusing
            # to start was the studio mistaking its own opinion for a limit.
            out.append(_issue("warn", "budget",
                "In %s this model would see %.1f tokens per parameter. %s"
                % (_hours(minutes), verdict.get("ratio") or 0, verdict["message"]),
                "A smaller model, or more time, would give a better result -- "
                "but this will run if you want to see it."))
        # `or 0`, not a default: the key is always present, and it is None
        # whenever the machine's speed could not be estimated -- so a plain
        # `.get("ratio", 0)` returns None and the comparison raises. That is
        # every unprobed runner, and it took the whole planner down with it.
        elif (verdict.get("ratio") or 0) > TOKENS_PER_PARAM_TARGET * 3:
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


def _validate_moe(arch: dict, caps: dict, *, minutes: float | None = None) -> list[dict]:
    """What a mixture of experts costs, said out loud before it is started.

    Every rule here exists because the sales pitch for MoE ("more parameters
    for the same compute") is true at scale and misleading on one GPU, and
    because nothing about the failure is visible until the run has finished.
    """
    if not is_moe(arch):
        return []
    out: list[dict] = []
    counts = count_params(arch)
    E = counts["experts"]
    # What was ASKED for, not what count_params clamped it to. Reading the
    # clamped value here made the k > E check unreachable: the arithmetic
    # quietly capped 8-per-token at 4 experts and the validator then agreed
    # with itself that nothing was wrong -- while transformers would have
    # crashed on the topk at step one.
    k = int(arch.get("num_experts_per_tok") or 1)
    d = arch["hidden_size"]

    for field, bound in MOE_LIMITS.items():
        v = int(arch.get(field) or 0)
        if v < bound["min"] or v > bound["max"]:
            out.append(_issue("error", field, "%s must be between %s and %s."
                              % (bound["label"], bound["min"], bound["max"])))

    if k > E:
        out.append(_issue("error", "num_experts_per_tok",
            "Each token would be sent to %d experts, but there are only %d."
            % (k, E), "Use %d or fewer." % E))
    elif k == E:
        out.append(_issue("warn", "num_experts_per_tok",
            "Every token goes through every expert, so nothing is sparse. This "
            "is an ordinary model that costs %dx as much to run as its shape "
            "suggests." % E,
            "Two experts per token is the usual choice."))

    # The one that actually decides whether this was a good idea.
    if minutes:
        budget = tokens_in_time(arch, caps, minutes) or 0
        per_expert = budget * k / max(E, 1)
        if budget and per_expert < counts["expert_params"] / max(E, 1) * 20:
            out.append(_issue("warn", "num_local_experts",
                "Each expert would see about %s tokens -- the router splits "
                "your text %d ways and each expert only learns from its own "
                "share. Expect %d half-taught feed-forward networks rather "
                "than one well-taught one."
                % (_human_tokens(per_expert), round(E / max(k, 1)), E),
                "Fewer experts, or much more text."))

    dense_active = fmt_params(counts["active"])
    out.append(_issue("info", "num_local_experts",
        "This model holds %s parameters and uses %s of them per token. Memory "
        "and training data follow the first number; speed follows the second. "
        "It is not a %s model that happens to be cheap."
        % (fmt_params(counts["total"]), dense_active, dense_active)))

    # Compared against what the same VRAM would have bought densely. This is
    # the comparison the user is actually making and never gets shown.
    slowdown = moe_time_multiplier(E, k)
    if slowdown > 1.3:
        out.append(_issue("warn", "num_local_experts",
            "Measured on this kind of card, %d experts run about %.1fx slower "
            "per token than one feed-forward network of the same shape -- "
            "sorting tokens by expert costs more than the arithmetic it "
            "saves. Nearly all of that is the number of experts, not how many "
            "of them each token uses." % (E, slowdown),
            "Fewer experts. Two per token costs almost nothing extra."))

    if d < 384:
        out.append(_issue("warn", "hidden_size",
            "A mixture of experts at %d wide is mostly overhead. The experts "
            "are too small for the GPU to run efficiently, and a dense model "
            "of the same total size would train faster and learn more." % d,
            "MoE starts paying off well above this size."))

    if arch.get("router_aux_loss_coef", MOE_AUX_LOSS_COEF) <= 0:
        out.append(_issue("warn", "router_aux_loss_coef",
            "With no load-balancing term the router collapses: one expert wins "
            "early, gets more tokens, trains faster, and the rest are never "
            "used again.", "%.3f is the standard value." % MOE_AUX_LOSS_COEF))

    return out


def _human_tokens(n: float) -> str:
    if n >= 1e9:
        return "%.1f billion" % (n / 1e9)
    if n >= 1e6:
        return "%.0f million" % (n / 1e6)
    return "%s" % f"{int(n):,}"


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


# What a caller with no machine in hand sees. Defined at the end because it
# needs everything above it.
SIZE_PRESETS = size_presets(None)
