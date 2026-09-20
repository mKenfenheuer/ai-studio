"""What a machine's attention kernels are, and what to do when there are none.

One place, because four callers have to agree about it. The runner picks a
kernel and a batch size against this, the controller's memory estimate charges
for the scores matrix against this, the scheduler decides what a machine may
be given against this, and the web UI greys out a sequence length against it.
Four copies of `caps["attention"]["flash"]` was four answers to "does attention
cost memory with the square of length here", and the disagreement surfaced as
an out-of-memory an hour into a download.

The correction worth spelling out: the question was never "is flash attention
available". It is "is attention FUSED" -- that is, does the kernel compute the
softmax in tiles and never write the scores matrix to memory. Flash attention
does. So does the memory-efficient kernel, which is a different implementation
of the same idea and is available on hardware flash attention is not. Reading
only the flash flag charged a memory-efficient card for a quadratic term it
never pays, and capped its sequence length at a quarter of what it can do.

No I/O and nothing torch-shaped: the controller installs four pure-Python
packages on purpose. The probe that fills the capability dict lives in
`runner/capabilities.py`; the code that acts on this lives in the jobs.
"""
from __future__ import annotations

# How long a sequence is comfortable, by kernel. Not a limit -- every one of
# these is honoured when asked for, and the cost is reported instead. See the
# note in `runner/jobs/lora_llm.py` about saying rather than doing.
#
# A fused kernel's attention memory grows with the LENGTH, so the ceiling is
# the rest of the model rather than attention. Without one the scores matrix
# and its softmax are both materialised, memory grows with the SQUARE, and
# 2048 is where that term stops being a rounding error on a 16 GB card.
COMFORTABLE_SEQ_FUSED = 8192
COMFORTABLE_SEQ_MATH = 2048

# The SDPA backends to allow, per kind, best first. Applied by
# `runner.capabilities.apply_sdpa_backends` so the choice is the probe's rather
# than a heuristic inside torch, which picks from the shapes of the tensors and
# has no way of knowing which kernels are actually sound on this card.
#
# MATH is in every list on purpose: it is the one backend that always runs, and
# a list without it turns "this shape has no fused kernel" (a head dim of 100,
# an unusual mask) into a crash instead of a slow step.
_BACKENDS = {
    "flash": ("FLASH_ATTENTION", "EFFICIENT_ATTENTION", "MATH"),
    "mem_efficient": ("EFFICIENT_ATTENTION", "MATH"),
    "math": ("MATH",),
}


def report(caps: dict | None) -> dict:
    """What this machine's attention does, and the plan for working around it.

    `caps` is a runner capability dict. A missing or empty one reports the
    pessimistic answer rather than raising: the controller costs runs against
    machines that have not finished probing yet, and guessing "fused" there
    would under-estimate the memory of every single one of them.
    """
    attn = ((caps or {}).get("attention") or {})
    flash = bool(attn.get("flash"))
    mem_efficient = bool(attn.get("mem_efficient"))
    kind = "flash" if flash else ("mem_efficient" if mem_efficient else "math")
    fused = kind != "math"
    return {
        "flash": flash,
        "mem_efficient": mem_efficient,
        "fused": fused,
        "kind": kind,
        # Which of transformers' attention paths to ask for. `sdpa` either way:
        # even with no fused backend behind it, torch's math SDPA keeps fewer
        # copies of the scores than transformers' eager path, which upcasts
        # them to fp32 in a separate tensor. The job retries with `eager` if
        # the model has no SDPA implementation, so this is a preference rather
        # than a requirement.
        "implementation": "sdpa",
        "backends": list(_BACKENDS[kind]),
        # Environment the job has to set for the above to be true. Empty on
        # almost every machine; see `runner/capabilities.py` for the one case
        # that fills it in.
        "env": dict(attn.get("env") or {}),
        "quadratic": not fused,
        "comfortable_seq": (COMFORTABLE_SEQ_FUSED if fused
                            else COMFORTABLE_SEQ_MATH),
    }


def is_fused(caps: dict | None) -> bool:
    """Whether attention here avoids materialising the scores matrix.

    The one question the memory estimate actually needs. Named for what it
    means rather than for the library that made it famous, because the answer
    is yes on cards that have never run flash attention.
    """
    return report(caps)["fused"]


def comfortable_seq(caps: dict | None) -> int:
    return report(caps)["comfortable_seq"]


def scores_bytes(batch: int, heads: int, seq: int, layers_live: int = 1) -> int:
    """Memory the attention scores occupy when nothing fuses them away.

    Two tensors per layer, not one: the scores and the softmax over them are
    both kept for the backward pass. Two bytes each in half precision.

    `layers_live` is the number of layers holding theirs AT ONCE, which is not
    the number of layers. With gradient checkpointing nothing survives the
    forward pass -- each layer's scores are rebuilt during its own
    recomputation and freed again -- so the peak holds one. Without it, every
    layer keeps its own and this is all of them. Charging every layer in both
    cases is a mistake this studio has already made once; see the note in
    `controller/architectures.py`.
    """
    return 2 * int(batch) * int(heads) * int(seq) * int(seq) * 2 * int(layers_live)


def describe(caps: dict | None) -> str:
    """One line of English for a log or a capability page."""
    rep = report(caps)
    if rep["kind"] == "flash":
        return "flash attention"
    if rep["kind"] == "mem_efficient":
        return ("the memory-efficient attention kernel, which costs the same "
                "memory as flash attention and rather less speed")
    return ("no fused attention kernel, so attention memory grows with the "
            "square of sequence length")
