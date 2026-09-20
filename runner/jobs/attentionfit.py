"""Making a training run fit the attention kernel the machine actually has.

`common/attention.py` says what the machine's attention is; this says what a
job does about it. Two things, both of which every text training loop needs and
neither of which belongs to one of them:

  * asking a model for the attention path we want, and coping with the models
    and library versions that will not give it;
  * cutting the micro-batch that the scores matrix cannot afford, and putting
    the batch back together with gradient accumulation.

The second is the workaround the studio leans on. Without a fused kernel
attention keeps `batch x heads x seq x seq` scores AND the softmax over them,
which is quadratic in length and linear in batch. Nobody wants to give up the
length -- it is the thing the dataset needs -- so the batch is what moves, and
it moves in a way that changes nothing about the resulting model.
"""
from __future__ import annotations

from typing import Any

from common import attention

describe = attention.describe

# Optional `from_pretrained` arguments, and the words a library that does not
# understand one puts in the error. Matched on the message because transformers
# reports an unsupported attention implementation as a ValueError about the
# function rather than about the argument that asked for it: "does not support
# an attention implementation through torch.nn.functional.scaled_dot_product_
# attention yet".
_OPTIONAL_KWARG_HINTS = {
    "attn_implementation": ("attn_implementation",
                            "scaled_dot_product_attention",
                            # What a refused FlexAttention compile says. It
                            # never mentions the argument that asked for it.
                            "triton", "shared memory", "flex_attention",
                            "InductorError"),
    "experts_implementation": ("experts_implementation",),
}


def load_base_model(cls, name: str, required: dict, optional: dict,
                    ctx: Any, plan: dict | None = None) -> Any:
    """Load a model, dropping the optional arguments this one will not take.

    Both of the optional arguments used here -- which attention path to use,
    which expert dispatch to use -- are refused by some combinations of model
    and library version, and each refusal looks like a different exception.
    Asking for both and retrying without whichever was rejected keeps one code
    path for every model instead of a matrix of version checks that would need
    editing on every transformers release.

    The retry is free in the way that matters: `from_pretrained` validates its
    arguments before it reads any weights, so a rejected argument costs a
    fraction of a second rather than a second download.
    """
    attempt = dict(optional)
    while True:
        try:
            return cls.from_pretrained(name, **attempt, **required)
        except Exception as e:  # noqa: BLE001 - see below
            message = str(e)
            dropped = next((k for k in attempt
                            if any(h in message
                                   for h in _OPTIONAL_KWARG_HINTS.get(k, (k,)))),
                           None)
            # `Exception`, not `(TypeError, ValueError)`, and the widening was
            # paid for. An attention implementation that the library accepts
            # and the COMPILER then refuses raises from inside Inductor --
            # `InductorError: No valid triton configs. out of resource: shared
            # memory, Required: 131072, Hardware limit: 65536` on RDNA2 -- and
            # that is neither of the two types this used to catch. It escaped,
            # killed the load, and the runner was restarted into loading the
            # same model again, twenty-four times.
            #
            # Nothing here is load-bearing enough to fail a run over: every
            # optional argument is a performance choice with a working
            # default behind it. So anything that names one is treated as
            # that argument being refused, whatever type it arrives as, and
            # an exception naming none of them is re-raised untouched.
            if dropped is None:
                raise
            attempt.pop(dropped)
            if dropped == "attn_implementation" and plan and plan.get("fused")                     and not (plan.get("flash") or plan.get("mem_efficient")):
                # The plan said this run would be fused, the memory estimate
                # was written against that, and the only thing making it true
                # was the argument that has just been refused. Said in terms
                # of the consequence rather than the argument, because the
                # consequence is what somebody has to act on: attention is
                # quadratic again, and the sequence length that was going to
                # fit may not.
                ctx.log("This model has no FlexAttention path, so attention "
                        "falls back to the method whose memory grows with the "
                        "square of sequence length. The plan for this run "
                        "assumed otherwise -- if it stops for lack of memory, "
                        "sequence length is the first thing to bring down.",
                        "warn")
            else:
                ctx.log("This model does not accept %s, so it is being loaded "
                        "without it." % dropped, "warn")


# The most of the card's free memory the attention scores may claim. The rest
# is the activations, the gradients, the optimiser and the allocator's own
# slack, none of which this term knows anything about -- so it is a share
# rather than the lot.
SCORES_SHARE = 0.4


def _largest_divisor(n: int, cap: int) -> int:
    """The largest divisor of `n` that is no greater than `cap`.

    A divisor rather than `cap` itself, and that is the whole subtlety of this
    file. Accumulating four sequences in groups of three is nine sequences a
    step, not eight: the effective batch changes, and with it the tokens per
    step, the token budget and the learning rate the schedule was written for.
    Taking a divisor keeps `batch x accumulation` exactly where it was, so the
    only thing this workaround alters is the peak.
    """
    for d in range(min(cap, n), 0, -1):
        if n % d == 0:
            return d
    return 1


def fit_batch(batch: int, accum: int, plan: dict, model, seq_len: int,
              ctx: Any, checkpointing: bool = True) -> tuple[int, int]:
    """Shrink a micro-batch the scores matrix cannot afford, keeping the
    effective batch by accumulating proportionally more.

    Four sequences one at a time and four at once produce the same gradient,
    the same number of optimiser steps and the same learning rate schedule --
    only the peak memory differs, by a factor of four. That makes this the one
    lever that buys memory without changing the run, which is why it is pulled
    automatically here rather than offered as a setting somebody has to find
    after their first out-of-memory.

    A no-op where attention is fused, where the batch is already one, or where
    the card's free memory cannot be read.
    """
    if not plan.get("quadratic") or batch <= 1:
        return batch, accum
    try:
        import torch
    except ImportError:
        return batch, accum
    conf = getattr(model, "config", None)
    heads = int(getattr(conf, "num_attention_heads", 0) or 0)
    if not heads or not torch.cuda.is_available():
        return batch, accum
    try:
        free_bytes = torch.cuda.mem_get_info()[0]
    except (RuntimeError, AttributeError):
        # mem_get_info is CUDA/ROCm only and has been known to raise on a
        # device that is busy. Nothing here is worth failing a run over.
        return batch, accum

    # One live layer when gradient checkpointing is on, because each layer's
    # scores are rebuilt inside its own recomputation and freed again. With it
    # off, every layer keeps its own for the backward pass and they are all
    # live at once -- which is a far larger number, and one no batch size
    # rescues on a long context. The log is the useful part in that case.
    layers_live = 1 if checkpointing else int(
        getattr(conf, "num_hidden_layers", 1) or 1)
    per_sequence = attention.scores_bytes(1, heads, seq_len, layers_live)
    allowed = _largest_divisor(batch, max(1, int(free_bytes * SCORES_SHARE
                                                 // max(per_sequence, 1))))
    if allowed >= batch:
        return batch, accum

    scaled = accum * (batch // allowed)
    ctx.log("This machine has no fused attention kernel, so %d sequences of "
            "%d tokens at once would need about %.1f GB for the attention "
            "scores alone, against %.1f GB free. Training %d at a time and "
            "accumulating over %d steps instead of %d: the same effective "
            "batch of %d and the same result, at a lower peak."
            % (batch, seq_len,
               attention.scores_bytes(batch, heads, seq_len, layers_live) / 1024 ** 3,
               free_bytes / 1024 ** 3, allowed, scaled, accum, batch * accum),
            "warn")
    return allowed, scaled
