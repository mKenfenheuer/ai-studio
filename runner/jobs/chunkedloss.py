"""Computing the training loss without ever holding all of the logits.

The output layer is the most surprising term in a long-context run, and the
arithmetic is worth spelling out because it is not close.

At 4,096 tokens against Qwen2.5's 151,936-token vocabulary, one copy of the
logits is 4096 x 151936 x 2 bytes = 1.24 GB. The cross-entropy path keeps
about five of them alive at once -- the half-precision logits, the float32
upcast the loss is computed in, the softmax, and the gradients of both -- so
roughly 6 GB. On the same run, one layer's attention scores at batch 1 are
about 2 GB. The term everybody talks about is a third of the term nobody
mentions, and the output layer does not care whether the card has a fused
attention kernel.

## Why chunking alone does not help, and what does

Splitting the sequence and computing the loss a slice at a time is the obvious
move and, on its own, saves nothing: autograd keeps each slice's logits for the
backward pass, so the peak is the same logits in more pieces.

What makes it work is recomputation. Each slice is wrapped in a checkpoint, so
the forward keeps only the hidden states going in -- a few megabytes -- and the
logits for a slice are rebuilt during that slice's own backward and freed
again. Peak becomes one slice's worth instead of the whole sequence: at a
512-token slice, 0.16 GB instead of 1.24, and the same for every copy behind
it. The cost is one extra matmul through the output layer, which is a few
percent of a step.

The loss is identical, to the last bit that float arithmetic allows: the slices
are summed by token count and divided once, not averaged as averages, so a
short final slice cannot weigh the same as a full one.

## Why it is not simply always on

It reaches past the model's front door. `model(**batch)` computes the loss
inside the architecture; doing this means calling the body and the output layer
separately, and while `get_decoder()` and `get_output_embeddings()` are stable
transformers APIs, not every model implements them usefully -- and a PEFT
wrapper or a custom architecture may not pass them through. So it is attempted,
verified once against the ordinary path, and abandoned in favour of the
ordinary path if anything is missing. A run that cannot use it is slower to run
out of memory, not wrong.
"""
from __future__ import annotations

from typing import Any

# Tokens per slice. Small enough that one slice's logits are a rounding error
# against a 16 GB card, large enough that the matmuls stay efficient: at 512
# tokens and a 152k vocabulary a slice is 0.16 GB in half precision, and the
# output layer is still being asked for a 512-row matmul rather than a 1-row
# one.
CHUNK_TOKENS = 512

# Below this there is nothing to win. The whole term is `tokens x vocabulary`,
# and on a small vocabulary at a short length it is already smaller than the
# attention it sits beside -- at which point the extra matmul is pure cost.
MIN_LOGIT_BYTES = 512 * 1024 ** 2


def worth_it(seq_len: int, batch: int, vocab_size: int) -> bool:
    """Whether the logits are big enough for this to pay for itself."""
    return batch * seq_len * vocab_size * 2 >= MIN_LOGIT_BYTES


def parts(model) -> tuple[Any, Any] | None:
    """The model's body and its output layer, or None if they cannot be had.

    Both are documented transformers APIs, which is the only reason this is
    attempted at all. `get_decoder` is unwrapped through PEFT by hand when the
    wrapper does not forward it, because a LoRA fine-tune is the commonest
    thing running here and it would otherwise never qualify.
    """
    target = model
    for _ in range(3):
        if hasattr(target, "get_decoder") and hasattr(
                target, "get_output_embeddings"):
            try:
                body, head = target.get_decoder(), target.get_output_embeddings()
            except (AttributeError, NotImplementedError):
                body = head = None
            if body is not None and head is not None:
                return body, head
        # PEFT wraps the model twice: PeftModel -> LoraModel -> the real one.
        nxt = getattr(target, "base_model", None) or getattr(
            target, "model", None)
        if nxt is None or nxt is target:
            return None
        target = nxt
    return None


def enable(model, batch: dict, torch, ctx, seq_len: int, batch_size: int,
           vocab_size: int) -> bool:
    """Decide once whether to use this, by checking it against the real thing.

    Not a capability check. The question is not "does the model expose a body
    and an output layer" -- it is "does calling them separately produce the
    loss the model itself would have produced", and those are different
    questions for any architecture that does anything unusual on the way out:
    a final norm applied in the wrapper, logit softcapping, a scaled head, a
    tied embedding read in a way `get_output_embeddings` does not reflect.

    Getting that wrong would not crash. It would train, with a loss curve that
    looked plausible and a model that learned the wrong thing, which is the
    worst failure available here. So one batch goes through both paths and the
    answers are compared before anything is believed.

    The tolerance is loose on purpose: the two paths sum in a different order
    and in half precision that is genuinely visible. It is set to catch "this
    is a different number", not "this is a different bit".
    """
    if not worth_it(seq_len, batch_size, vocab_size):
        return False
    if parts(model) is None:
        return False
    try:
        with torch.no_grad():
            mine = loss(model, batch, torch)
            theirs = model(**batch).loss
        if mine is None or theirs is None:
            return False
        mine_f, theirs_f = float(mine), float(theirs)
        if not (mine_f == mine_f and theirs_f == theirs_f):   # nan
            return False
        if abs(mine_f - theirs_f) > max(0.01, 0.01 * abs(theirs_f)):
            ctx.log("Not computing the loss in slices: doing so gives %.4f "
                    "where this model's own head gives %.4f, so the two are "
                    "not the same calculation. Using the model's own, which "
                    "costs memory at a long context and is correct."
                    % (mine_f, theirs_f), "warn")
            return False
    except Exception as e:  # noqa: BLE001 - any failure means "use the usual path"
        ctx.log("Not computing the loss in slices on this model (%s). It "
                "costs memory at a long context and changes nothing else."
                % str(e)[:120], "debug")
        return False

    ctx.log("Computing the loss %d tokens at a time and rebuilding each "
            "slice during its own backward pass. The output layer is %.1f GB "
            "of logits at this length and vocabulary, and this holds about "
            "%.2f GB of it at once."
            % (CHUNK_TOKENS,
               batch_size * seq_len * vocab_size * 2 / 1024 ** 3,
               CHUNK_TOKENS * vocab_size * 2 / 1024 ** 3))
    return True


def loss(model, batch: dict, torch, chunk: int = CHUNK_TOKENS):
    """Cross-entropy over this batch, a slice of the sequence at a time.

    Returns None where the model does not expose what this needs, so the
    caller can fall back to `model(**batch).loss` without having to know why.
    """
    got = parts(model)
    if got is None:
        return None
    body, head = got

    labels = batch.get("labels")
    if labels is None:
        return None
    inputs = {k: v for k, v in batch.items() if k != "labels"}
    out = body(**inputs)
    hidden = getattr(out, "last_hidden_state", None)
    if hidden is None:
        return None

    # The usual shift: position i predicts the token at i+1, so the last
    # hidden state has nothing to predict and the first label has nothing to
    # predict it.
    hidden = hidden[:, :-1, :]
    targets = labels[:, 1:]

    flat_hidden = hidden.reshape(-1, hidden.shape[-1])
    flat_targets = targets.reshape(-1)

    total = None
    counted = 0
    for start in range(0, flat_targets.shape[0], chunk):
        piece_h = flat_hidden[start:start + chunk]
        piece_t = flat_targets[start:start + chunk]
        # Every position in this slice is padding or prompt. Skipping it is
        # not an optimisation -- cross-entropy over an all-ignored slice is a
        # division by zero, and returns nan.
        live = int((piece_t != -100).sum())
        if not live:
            continue
        summed = _chunk_loss(piece_h, piece_t, head, torch)
        total = summed if total is None else total + summed
        counted += live

    if not counted:
        # Nothing in this batch was trainable. The caller decides what that
        # means; returning a zero with no graph behind it would look like a
        # perfect step and quietly contribute nothing to the gradient.
        return None
    return total / counted


def _chunk_loss(hidden, targets, head, torch):
    """Summed cross-entropy for one slice, recomputed during backward.

    Summed rather than averaged, because the caller divides once by the total
    number of live tokens. Averaging here and averaging those would weigh a
    forty-token final slice as heavily as a five-hundred-token one.
    """
    def compute(h, t):
        logits = head(h).float()
        return torch.nn.functional.cross_entropy(
            logits, t, ignore_index=-100, reduction="sum")

    if not torch.is_grad_enabled() or not hidden.requires_grad:
        # Evaluation, or a frozen stack: there is no backward pass to save
        # memory for, and checkpointing without one recomputes for nothing.
        return compute(hidden, targets)
    from torch.utils.checkpoint import checkpoint
    return checkpoint(compute, hidden, targets, use_reentrant=False)
