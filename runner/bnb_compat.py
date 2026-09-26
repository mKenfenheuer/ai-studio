"""Make bitsandbytes 4-bit decoding correct on cards where it is not.

On some ROCm cards -- gfx1030 among them -- a 4-bit matmul returns noise when
the input has a single row and is accurate when it has several. Nothing
raises. The capability probe measures exactly this, against the same weights
in float16: 8 rows lands at 0.11 relative error, as 4-bit should; 1 row at
0.97, which is the reference's own magnitude, i.e. noise.

One row is the shape of writing the next token. So a model loaded in 4-bit
trains correctly and then answers with rubbish, and a runner that cannot
trust 4-bit decoding has to serve every model at full precision -- which on a
16 GB card leaves a 7B model with room for about 3,500 tokens of context.

The fix here is deliberately indirect. It does not reach into bitsandbytes'
kernels or reimplement its matmul: the runner images build bitsandbytes from
source at whatever `main` was that day, and its internal dispatch has been
restructured across releases -- a patch aimed at one branch of it would
silently stop applying. Instead it relies only on the measured fact: small
row counts are wrong, 8 rows are right. So a small input is padded with zero
rows up to 8, the untouched original function computes it, and the real rows
are taken back out. Rows of a matmul do not interact, so the answer for the
real row is exactly the one the verified 8-row path gives.

The cost is small. Decoding is bound by reading the weights, not by
arithmetic on them, and seven extra rows are seven extra dot products against
weights that are being read anyway.

Installed only where the probe says it is needed and that it works: on a card
whose decoding is already correct it would be pure overhead.
"""
from __future__ import annotations

from typing import Any

# The row count the probe verifies as correct. Anything below it is padded up
# to it; anything at or above it is passed through untouched.
PAD_ROWS = 8

_original: Any = None


def installed() -> bool:
    return _original is not None


def install_decode_padding() -> bool:
    """Route small 4-bit matmuls through the row count that is correct.

    Idempotent. Returns False, and changes nothing, if bitsandbytes cannot be
    imported or does not expose `matmul_4bit` where this expects it -- the
    caller then keeps treating 4-bit decoding as untrustworthy, which is the
    safe answer.
    """
    global _original
    if _original is not None:
        return True
    try:
        import bitsandbytes as bnb
        import torch
    except Exception:  # noqa: BLE001 - no bitsandbytes means nothing to fix
        return False
    original = getattr(bnb, "matmul_4bit", None)
    if original is None:
        return False

    def matmul_4bit(A, B, quant_state, out=None, bias=None):
        features = A.shape[-1]
        rows = A.numel() // features if features else 0
        if rows == 0 or rows >= PAD_ROWS:
            return original(A, B, quant_state, out=out, bias=bias)
        flat = A.reshape(rows, features)
        padded = torch.cat(
            [flat, flat.new_zeros(PAD_ROWS - rows, features)], dim=0)
        result = original(padded, B, quant_state, bias=bias)[:rows]
        result = result.reshape(*A.shape[:-1], result.shape[-1])
        if out is not None:
            out.copy_(result)
            return out
        return result

    # `Linear4bit.forward` calls `bnb.matmul_4bit(...)` through the module, so
    # replacing the attribute is enough for every layer, including ones already
    # built. The autograd module keeps its own name for it on some releases;
    # replaced too, so nothing reaches the original by the side door.
    bnb.matmul_4bit = matmul_4bit
    try:
        from bitsandbytes.autograd import _functions as fn
        if getattr(fn, "matmul_4bit", None) is original:
            fn.matmul_4bit = matmul_4bit
    except Exception:  # noqa: BLE001 - absent on some releases; fine
        pass
    _original = original
    return True
