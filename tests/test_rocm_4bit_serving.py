"""The ROCm 4-bit serving fix: the decode padding shim and context-aware planning.

Runs on a CPU, with no GPU and no real bitsandbytes. The shim is tested against
a stand-in `bitsandbytes` that reproduces the gfx1030 fault exactly as the
capability probe measures it: correct for 8 rows and more, noise below that.
The planner is tested with the real shapes of the models it was written for.

    python tests/test_rocm_4bit_serving.py
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------- the shim --

def _broken_bnb(weight: torch.Tensor):
    """A stand-in bitsandbytes whose 4-bit matmul is wrong below 8 rows.

    `quant_state` and `B` are ignored: the point is only the row-count fault.
    The correct answer is an ordinary linear layer with `weight`.
    """
    calls = []

    def matmul_4bit(A, B, quant_state, out=None, bias=None):
        rows = A.numel() // A.shape[-1]
        calls.append(rows)
        flat = A.reshape(rows, A.shape[-1])
        if rows < 8:
            # The fault: noise of the reference's own magnitude.
            res = torch.randn(rows, weight.shape[0], dtype=A.dtype) * 3
        else:
            res = flat @ weight.t()
            if bias is not None:
                res = res + bias
        res = res.reshape(*A.shape[:-1], weight.shape[0])
        if out is not None:
            out.copy_(res)
            return out
        return res

    bnb = types.ModuleType("bitsandbytes")
    bnb.matmul_4bit = matmul_4bit
    autograd = types.ModuleType("bitsandbytes.autograd")
    fn = types.ModuleType("bitsandbytes.autograd._functions")
    fn.matmul_4bit = matmul_4bit
    autograd._functions = fn
    bnb.autograd = autograd
    sys.modules["bitsandbytes"] = bnb
    sys.modules["bitsandbytes.autograd"] = autograd
    sys.modules["bitsandbytes.autograd._functions"] = fn
    return bnb, fn, calls


def _fresh_shim():
    """bnb_compat with its install state reset, so each test starts clean.

    A reload, not a pop from sys.modules: `from runner import bnb_compat`
    finds the submodule cached as an attribute of the `runner` package and
    hands back the already-installed one, whose idempotency guard then
    (correctly) declines to patch the next test's stand-in.
    """
    import importlib
    from runner import bnb_compat
    return importlib.reload(bnb_compat)


def _rel_error(got: torch.Tensor, want: torch.Tensor) -> float:
    scale = want.abs().mean().clamp(min=1e-6)
    return float((got - want).abs().mean() / scale)


def test_one_row_is_wrong_without_the_shim():
    torch.manual_seed(0)
    weight = torch.randn(64, 32)
    bnb, _, _ = _broken_bnb(weight)
    x = torch.randn(1, 32)
    err = _rel_error(bnb.matmul_4bit(x, None, None), x @ weight.t())
    assert err > 0.35, "the stand-in must reproduce the fault (%.2f)" % err


def test_shim_makes_one_row_correct():
    torch.manual_seed(1)
    weight = torch.randn(64, 32)
    bnb, _, calls = _broken_bnb(weight)
    shim = _fresh_shim()
    assert shim.install_decode_padding()
    x = torch.randn(1, 32)
    err = _rel_error(bnb.matmul_4bit(x, None, None), x @ weight.t())
    assert err < 1e-5, "padded decode must match the reference (%.2e)" % err
    assert calls[-1] == 8, "the kernel must only ever see the padded 8 rows"


def test_shim_keeps_decode_shapes():
    """(batch, seq=1, hidden) is what generation actually passes."""
    torch.manual_seed(2)
    weight = torch.randn(48, 16)
    bnb, _, _ = _broken_bnb(weight)
    _fresh_shim().install_decode_padding()
    x = torch.randn(1, 1, 16)
    got = bnb.matmul_4bit(x, None, None)
    assert got.shape == (1, 1, 48)
    assert _rel_error(got, x @ weight.t()) < 1e-5


def test_shim_handles_bias_and_out():
    torch.manual_seed(3)
    weight = torch.randn(20, 12)
    bias = torch.randn(20)
    bnb, _, _ = _broken_bnb(weight)
    _fresh_shim().install_decode_padding()
    x = torch.randn(3, 12)
    out = torch.empty(3, 20)
    got = bnb.matmul_4bit(x, None, None, out=out, bias=bias)
    assert got is out
    assert _rel_error(out, x @ weight.t() + bias) < 1e-5


def test_wide_inputs_pass_straight_through():
    """Prefill and training shapes must not be padded or altered."""
    torch.manual_seed(4)
    weight = torch.randn(40, 24)
    bnb, _, calls = _broken_bnb(weight)
    _fresh_shim().install_decode_padding()
    x = torch.randn(30, 24)
    got = bnb.matmul_4bit(x, None, None)
    assert calls[-1] == 30, "a wide input must reach the kernel unpadded"
    assert _rel_error(got, x @ weight.t()) < 1e-5


def test_side_door_is_closed_too():
    """Callers reaching it through the autograd module get the shim as well."""
    torch.manual_seed(5)
    weight = torch.randn(16, 8)
    _, fn, _ = _broken_bnb(weight)
    _fresh_shim().install_decode_padding()
    x = torch.randn(1, 8)
    assert _rel_error(fn.matmul_4bit(x, None, None), x @ weight.t()) < 1e-5


def test_install_is_idempotent():
    weight = torch.randn(8, 8)
    bnb, _, _ = _broken_bnb(weight)
    shim = _fresh_shim()
    assert shim.install_decode_padding()
    first = bnb.matmul_4bit
    assert shim.install_decode_padding()
    assert bnb.matmul_4bit is first, "a second install must not wrap the wrapper"


def test_no_bitsandbytes_means_no_install():
    for name in ("bitsandbytes", "bitsandbytes.autograd",
                 "bitsandbytes.autograd._functions"):
        sys.modules[name] = None      # makes the import raise
    try:
        assert _fresh_shim().install_decode_padding() is False
    finally:
        for name in ("bitsandbytes", "bitsandbytes.autograd",
                     "bitsandbytes.autograd._functions"):
            sys.modules.pop(name, None)


# ------------------------------------------------------------- the planner --

def _config(layers, heads, kv_heads, head_dim):
    return types.SimpleNamespace(num_hidden_layers=layers,
                                 num_attention_heads=heads,
                                 num_key_value_heads=kv_heads,
                                 head_dim=head_dim)


# The real shapes, from each model's config.json.
MISTRAL_7B = dict(params_b=7.25, cfg=_config(32, 32, 8, 128))
QWEN25_3B = dict(params_b=3.09, cfg=_config(36, 16, 2, 128))


def _host(free_gb: float, can_quantize: bool, cfg):
    from runner.inference import ModelHost
    host = ModelHost.__new__(ModelHost)      # no controller, no GPU
    host.caps = {}
    host._free_gb = lambda: free_gb
    host._can_quantize = lambda: can_quantize
    host._model_config = lambda spec, path: cfg
    return host


def test_mistral_7b_on_16gb_is_compressed_for_context():
    """The production case: it fits at fp16, with no room to talk to it."""
    logs = []
    host = _host(15.6, True, MISTRAL_7B["cfg"])
    assert host._plan_precision({"params_b": MISTRAL_7B["params_b"]},
                                logs.append, None) is True
    assert any("4-bit" in l for l in logs)


def test_qwen_3b_on_16gb_stays_full_precision():
    """A small model has room to spare and must not be compressed for nothing."""
    host = _host(15.6, True, QWEN25_3B["cfg"])
    assert host._plan_precision({"params_b": QWEN25_3B["params_b"]},
                                lambda _: None, None) is False


def test_untrusted_4bit_is_never_chosen():
    """Without a trustworthy 4-bit decode, loading fp16 beats serving noise."""
    host = _host(15.6, False, MISTRAL_7B["cfg"])
    assert host._plan_precision({"params_b": MISTRAL_7B["params_b"]},
                                lambda _: None, None) is False


def test_unknown_shape_disables_the_check():
    host = _host(15.6, True, None)
    assert host._plan_precision({"params_b": MISTRAL_7B["params_b"]},
                                lambda _: None, None) is False


def test_weights_that_do_not_fit_still_compress():
    """The original rule is untouched: fp16 weights too big means 4-bit."""
    host = _host(10.0, True, MISTRAL_7B["cfg"])
    assert host._plan_precision({"params_b": MISTRAL_7B["params_b"]},
                                lambda _: None, None) is True


def test_room_arithmetic_matches_the_card():
    """fp16 Mistral on this card should leave room of the order it reported.

    The server said "room for about 3,523"; the planner's estimate should land
    in the same neighbourhood, or its threshold means nothing.
    """
    from runner.inference import (ModelHost, PREFILL_CHUNK,
                                  prefill_headroom_gb)
    per_token = ModelHost._kv_bytes_from_config(MISTRAL_7B["cfg"])
    assert per_token == 131072, per_token
    free = 15.6
    room = (free - MISTRAL_7B["params_b"] * 2
            - prefill_headroom_gb(PREFILL_CHUNK)) * 1024 ** 3 // per_token
    assert 1500 < room < 8192, room


# ------------------------------------------------------- serving attention --

def test_serving_drops_the_training_flex_opt_in():
    """Flex decode kernels fault gfx1030; serving must not inherit the opt-in."""
    import os
    from runner.inference import ModelHost
    caps = {"attention": {"flex": True, "flex_opt_in": True, "math": True}}
    os.environ.pop("AI_STUDIO_SERVE_FLEX_ATTENTION", None)
    served = ModelHost._serving_caps(caps)
    assert served["attention"]["flex_opt_in"] is False
    assert caps["attention"]["flex_opt_in"] is True, "training caps untouched"
    os.environ["AI_STUDIO_SERVE_FLEX_ATTENTION"] = "1"
    try:
        assert ModelHost._serving_caps(caps) is caps
    finally:
        os.environ.pop("AI_STUDIO_SERVE_FLEX_ATTENTION", None)


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print("  ok    %s" % name)
        except Exception as e:  # noqa: BLE001
            failed += 1
            print("  FAIL  %s: %s" % (name, e))
    print("\n%d passed, %d failed" % (len(tests) - failed, failed))
    sys.exit(1 if failed else 0)
