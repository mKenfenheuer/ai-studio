#!/usr/bin/env python
"""Check what the studio does about the attention kernel a machine has.

Run it with `python scripts/check-attention.py` from the repository root.

Three things are worth proving without a GPU in the room.

The first is that "fused" and "flash" are not the same word. The
memory-efficient kernel costs the same memory as flash attention and is
available on hardware flash attention is not, so a card that has it must not be
charged for a quadratic term it never pays, nor capped at a quarter of the
sequence length it can train at. Reading only the flash flag did both.

The second is that the workaround preserves the run. A micro-batch cut to fit
the scores matrix has to be made back up exactly by accumulation: four
sequences in groups of three is nine a step, not eight, and an effective batch
that drifts changes the tokens per step, the token budget and the learning rate
the schedule was written for. So the replacement is a DIVISOR, always.

The third is that the plan survives a machine that has not been probed yet. The
controller costs runs against those, and guessing "fused" there would
under-estimate the memory of every one of them.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import attention  # noqa: E402
from controller import architectures as arch  # noqa: E402
from runner.jobs.attentionfit import _largest_divisor  # noqa: E402

FAILED: list[str] = []

FLASH = {"backend": "cuda", "vram_gb": 24.0,
         "attention": {"flash": True, "mem_efficient": True, "math": True}}
MEM_EFF = {"backend": "rocm", "vram_gb": 16.0,
           "attention": {"flash": False, "mem_efficient": True, "math": True}}
MATH = {"backend": "rocm", "vram_gb": 16.0,
        "attention": {"flash": False, "mem_efficient": False, "math": True}}
UNPROBED: dict = {"backend": "rocm", "vram_gb": 16.0}


def check(name: str, got: object, want: object = True) -> None:
    ok = got == want
    print("  %s  %s%s" % ("ok  " if ok else "FAIL", name,
                          "" if ok else "   -> %r, wanted %r" % (got, want)))
    if not ok:
        FAILED.append(name)


def main() -> int:
    print("\nWhat counts as a fused kernel")
    check("flash attention is fused", attention.is_fused(FLASH))
    check("so is the memory-efficient kernel", attention.is_fused(MEM_EFF))
    check("the math fallback is not", attention.is_fused(MATH), False)
    check("an unprobed machine is assumed not to be",
          attention.is_fused(UNPROBED), False)
    check("nor is a machine with no capabilities at all",
          attention.is_fused(None), False)

    print("\nWhat each one is allowed to dispatch to")
    check("flash may use every backend",
          attention.report(FLASH)["backends"],
          ["FLASH_ATTENTION", "EFFICIENT_ATTENTION", "MATH"])
    check("a card without it does not try it",
          "FLASH_ATTENTION" not in attention.report(MEM_EFF)["backends"])
    check("and every plan keeps math as a last resort",
          all("MATH" in attention.report(c)["backends"]
              for c in (FLASH, MEM_EFF, MATH, UNPROBED)))
    check("the environment a probe needed travels with the plan",
          attention.report({"attention": {"flash": True, "env": {"A": "1"}}})["env"],
          {"A": "1"})

    print("\nHow long a sequence is comfortable")
    check("8192 with flash", attention.comfortable_seq(FLASH), 8192)
    check("the same with the memory-efficient kernel",
          attention.comfortable_seq(MEM_EFF), 8192)
    check("and 2048 with neither", attention.comfortable_seq(MATH), 2048)

    print("\nThe scores matrix, when nothing fuses it away")
    # 1 x 32 heads x 4096 x 4096 at two bytes is 1 GiB of scores. The softmax
    # over them is kept for the backward pass too, so the layer costs 2 GiB.
    check("the softmax is charged as well as the scores",
          attention.scores_bytes(1, 32, 4096) / 1024 ** 3, 2.0)
    check("linear in batch",
          attention.scores_bytes(4, 32, 4096)
          == 4 * attention.scores_bytes(1, 32, 4096))
    check("quadratic in length",
          attention.scores_bytes(1, 32, 8192)
          == 4 * attention.scores_bytes(1, 32, 4096))
    check("and counted once per layer that is live at a time",
          attention.scores_bytes(1, 32, 4096, layers_live=8)
          == 8 * attention.scores_bytes(1, 32, 4096))

    print("\nCutting a batch keeps the effective batch exactly")
    check("8 capped at 3 becomes 2, not 3", _largest_divisor(8, 3), 2)
    check("8 capped at 5 becomes 4", _largest_divisor(8, 5), 4)
    check("12 capped at 5 becomes 4", _largest_divisor(12, 5), 4)
    check("a prime batch falls to one", _largest_divisor(7, 3), 1)
    check("a batch that already fits is left alone", _largest_divisor(8, 8), 8)
    check("and a cap below one never returns zero", _largest_divisor(8, 0), 1)
    check("batch x accumulation is unchanged in every case",
          all((b // _largest_divisor(b, c)) * a * _largest_divisor(b, c) == b * a
              for b in range(1, 33) for c in range(1, 33) for a in (1, 8)))

    print("\nThe memory estimate charges the right machines")
    shape = {"model_type": "llama", "vocab_size": 32000, "hidden_size": 2048,
             "intermediate_size": 5632, "num_hidden_layers": 22,
             "num_attention_heads": 32, "num_key_value_heads": 4,
             "max_position_embeddings": 4096}
    fused_gb = arch.training_memory_gb(shape, 1, checkpointing=True,
                                       fused=True)["total_gb"]
    math_gb = arch.training_memory_gb(shape, 1, checkpointing=True,
                                      fused=False)["total_gb"]
    check("a fused kernel costs less at 4096 tokens", fused_gb < math_gb)
    check("the memory-efficient card is costed as fused, not as math",
          arch.training_memory_gb(shape, 1, checkpointing=True,
                                  fused=attention.is_fused(MEM_EFF))["total_gb"],
          fused_gb)

    print()
    if FAILED:
        print("%d check(s) failed:" % len(FAILED))
        for name in FAILED:
            print("  - %s" % name)
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
