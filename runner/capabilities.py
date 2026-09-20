"""Hardware capability probe.

The controller schedules work based on what a runner reports here, and the web
UI greys out options a runner cannot do. So this module has one job: find out
what the machine can *actually* do, never what it claims on paper.

Two hard-won rules, both from real failures on an RX 6900 XT (gfx1030):

1. Risky probes run in a SUBPROCESS. Importing bitsandbytes on unsupported
   ROCm hardware does not raise a Python exception -- it aborts the process at
   the HIP level ("Module not initialized"). An in-process probe would take the
   whole agent down at startup, permanently, with no useful error.

2. Supported != fast. RDNA2 advertises bfloat16 and computes it correctly, but
   has no hardware path for it: measured 11.2 TFLOP/s bf16 against 39.9
   TFLOP/s fp16. Nearly every fine-tuning recipe defaults to bf16, which would
   silently waste 70% of the card. We benchmark and recommend, we do not trust.
"""
from __future__ import annotations

import contextlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from common import attention

# GPU architectures with no usable flash/memory-efficient attention kernels.
# On these, attention falls back to the O(n^2)-memory math path, which caps
# usable sequence length far below what VRAM alone would suggest.
_NO_FLASH_ATTN_ARCHS = ("gfx1030", "gfx1031", "gfx1032", "gfx1010", "gfx1012")

# Environment that has been observed to switch a fused attention kernel ON for
# a card that reports none without it.
#
# ROCm's flash and memory-efficient kernels come from AOTriton, and the list of
# architectures its wheels compile for is shorter than the list PyTorch will
# dispatch to. On the ones in between, torch says no and the kernel exists --
# it is behind a flag because AMD has not finished validating it, not because
# it is absent. A card that gains flash attention here gains about four times
# the sequence length it can train at, which is too large a difference to leave
# on the table because an environment variable has the word EXPERIMENTAL in it.
#
# Measured rather than assumed, and that distinction is the whole point: this
# is applied to a probe, not to a training run. If attention still fails with
# it set, nothing is reported and nothing is exported. If it succeeds, the
# variable travels with the capability report so the job that eventually runs
# sets exactly the environment the measurement was taken in. Reporting "flash:
# true" from a probe whose environment the job does not reproduce would be the
# worst outcome available: an optimistic memory estimate against a pessimistic
# kernel.
_ATTENTION_RETRY_ENV = {"TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "1"}

# Attention alone, for the retry above. Deliberately not the full probe: it
# touches no bitsandbytes, so it is seconds rather than minutes, and it cannot
# abort the interpreter.
_ATTENTION_PROBE = r"""
import json, warnings
warnings.filterwarnings("ignore")
out = {"flash_attn": False, "mem_efficient_attn": False, "flex_attn": False}
try:
    import torch
    if torch.cuda.is_available():
        from torch.nn.functional import scaled_dot_product_attention as sdpa
        from torch.nn.attention import SDPBackend, sdpa_kernel
        qq = torch.randn(1, 4, 128, 64, device="cuda", dtype=torch.float16)
        for key, be in (("flash_attn", SDPBackend.FLASH_ATTENTION),
                        ("mem_efficient_attn", SDPBackend.EFFICIENT_ATTENTION)):
            try:
                with sdpa_kernel(be):
                    sdpa(qq, qq, qq)
                torch.cuda.synchronize()
                out[key] = True
            except Exception:
                out[key] = False
        try:
            from torch.nn.attention.flex_attention import flex_attention
            fq = torch.randn(1, 4, 256, 64, device="cuda", dtype=torch.float16)
            torch.compile(flex_attention, dynamic=False)(fq, fq, fq)
            torch.cuda.synchronize()
            out["flex_attn"] = True
        except Exception:
            out["flex_attn"] = False
except Exception:
    pass
print("RESULT:" + json.dumps(out))
"""

# Probes that may abort the interpreter rather than raise. Run out-of-process.
_SUBPROCESS_PROBE = r"""
import json, sys, warnings
warnings.filterwarnings("ignore")
out = {"bnb_4bit": False, "bnb_4bit_decode": False, "bnb_8bit": False,
       "bnb_optim": False, "bnb_4bit_error": None, "bnb_4bit_decode_error": None,
       "flash_attn": False, "mem_efficient_attn": False, "flex_attn": False,
       "bnb_error": None, "sdpa": False, "sdpa_error": None}
try:
    import torch
    if torch.cuda.is_available():
        dev = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        dev = "mps"
    else:
        dev = None
    if dev:
        def _sync():
            (torch.cuda if dev == "cuda" else torch.mps).synchronize()

        from torch.nn.functional import scaled_dot_product_attention as sdpa
        qq = torch.randn(1, 4, 128, 64, device=dev, dtype=torch.float16)
        if dev == "cuda":
            from torch.nn.attention import SDPBackend, sdpa_kernel
            for key, be in (("flash_attn", SDPBackend.FLASH_ATTENTION),
                            ("mem_efficient_attn", SDPBackend.EFFICIENT_ATTENTION)):
                try:
                    with sdpa_kernel(be):
                        sdpa(qq, qq, qq)
                    _sync()
                    out[key] = True
                except Exception:
                    out[key] = False
            # FlexAttention, which is a different answer to the same question:
            # instead of calling a kernel somebody shipped, it COMPILES one with
            # Triton. That matters here because ROCm's flash and memory-efficient
            # kernels come from AOTriton, whose architecture list is short, while
            # Triton itself supports rather more cards -- so a GPU with neither
            # kernel can still get tiled attention that never writes the scores
            # matrix. Worth minutes of probing for four times the sequence length.
            #
            # Compiled here rather than assumed, because "the import works" and
            # "the kernel runs on this card" have already been shown to be
            # different facts on this hardware.
            #
            # Not attempted on Metal: the reason to pay for a Triton compile is
            # a card whose shipped kernels are missing because AOTriton did not
            # build for it, which is a ROCm predicament. Inductor has no such
            # route to a Metal kernel, so this would buy minutes of compiling
            # for a no.
            try:
                from torch.nn.attention.flex_attention import flex_attention
                fq = torch.randn(1, 4, 256, 64, device=dev, dtype=torch.float16)
                compiled = torch.compile(flex_attention, dynamic=False)
                compiled(fq, fq, fq)
                _sync()
                out["flex_attn"] = True
            except Exception:
                out["flex_attn"] = False
        else:
            # `sdpa_kernel` is a CUDA dispatch hint that MPS ignores, so asking
            # it which backend is in use answers yes to everything and proves
            # nothing. Measuring is no better: the MPS allocator does not count
            # a kernel's scratch memory and Metal buffers do not appear in the
            # process's resident size, so a materialised score matrix and a
            # fused kernel are indistinguishable from in here -- both were
            # tried, both report zero. So flash and memory-efficient stay
            # false, which costs a conservative memory estimate rather than an
            # over-promise, and all that is established is that attention runs.
            try:
                sdpa(qq, qq, qq)
                _sync()
                out["sdpa"] = True
            except Exception as e:
                out["sdpa_error"] = str(e)[:200]
        # Emit attention results before touching bitsandbytes: if bnb aborts
        # the process, the parent still learns what we proved up to here.
        sys.stderr.write("PARTIAL:" + json.dumps(out) + "\n")
        sys.stderr.flush()
        try:
            # Whether 4-bit RUNS is the easy half. Whether it is CORRECT is the
            # half that matters, and they are not the same question: on gfx1030
            # the 4-bit matmul returns noise for a handful of rows -- exactly
            # the shape token-by-token generation uses -- while being perfectly
            # accurate for the wide shapes training uses. Nothing raises. The
            # model just answers with rubbish.
            #
            # So both shapes are measured against the same weights in float16.
            # Correct 4-bit lands near 0.10 relative error; the broken kernel
            # measures 0.94 and up, which is the reference's own magnitude --
            # noise. Anything past a third of the signal is not quantization
            # loss, whatever it is.
            import bitsandbytes as bnb
            from bitsandbytes.nn import Linear4bit, Params4bit
            n_in, n_out = 1024, 2048          # not square: the orientation is
            ref = torch.nn.Linear(n_in, n_out, bias=False)  # then unambiguous
            ref = ref.to(dev, torch.float16).eval()
            lin = Linear4bit(n_in, n_out, bias=False, compute_dtype=torch.float16)
            lin.weight = Params4bit(ref.weight.data.clone().cpu(),
                                    requires_grad=False)
            lin = lin.to(dev).eval()

            def _relative_error(rows):
                x = torch.randn(rows, n_in, device=dev, dtype=torch.float16)
                with torch.no_grad():
                    want, got = ref(x), lin(x)
                scale = want.float().abs().mean().clamp(min=1e-6)
                return float((got.float() - want.float()).abs().mean() / scale)

            # Eight rows is what a training step sees; one row is what writing
            # the next token of a reply sees.
            out["bnb_4bit_error"] = round(_relative_error(8), 3)
            out["bnb_4bit_decode_error"] = round(_relative_error(1), 3)
            out["bnb_4bit"] = out["bnb_4bit_error"] < 0.35
            out["bnb_4bit_decode"] = out["bnb_4bit_decode_error"] < 0.35
            if not out["bnb_4bit"]:
                out["bnb_error"] = (
                    "4-bit ran but returned wrong numbers (%.2f relative error "
                    "against float16). Treating quantization as unavailable."
                    % out["bnb_4bit_error"])
        except Exception as e:
            out["bnb_error"] = str(e)[:200]
        try:
            from bitsandbytes.nn import Linear8bitLt
            l8 = Linear8bitLt(256, 256, has_fp16_weights=False).to(dev).eval()
            with torch.no_grad():
                l8(torch.randn(2, 256, device=dev, dtype=torch.float16))
            out["bnb_8bit"] = True
        except Exception:
            pass
        try:
            import bitsandbytes.optim as bopt
            p = torch.nn.Parameter(torch.randn(64, 64, device=dev))
            o = bopt.AdamW8bit([p])
            (p * p).sum().backward()
            o.step()
            out["bnb_optim"] = True
        except Exception:
            pass
except Exception as e:
    out["bnb_error"] = "probe failed: " + str(e)[:150]
print("RESULT:" + json.dumps(out))
"""


def _run_subprocess_probe(timeout: int = 900) -> dict:
    """Run the dangerous probes in a child process.

    Three outcomes, all handled: clean JSON on stdout; a hard abort (segfault /
    HIP fault) where we fall back to the PARTIAL line the child flushed to
    stderr before dying; or a hang, which we kill.

    The timeout is generous because a cold ROCm container compiles its GPU
    kernels on first use, which can take many minutes with no output.
    """
    fallback = {"bnb_4bit": False, "bnb_4bit_decode": False, "bnb_8bit": False,
                "bnb_optim": False, "bnb_4bit_error": None,
                "bnb_4bit_decode_error": None,
                "flash_attn": False, "mem_efficient_attn": False,
                "flex_attn": False, "bnb_error": None,
                "sdpa": False, "sdpa_error": None}
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _SUBPROCESS_PROBE],
            capture_output=True, text=True, timeout=timeout,
            env={**os.environ, "PYTHONWARNINGS": "ignore"},
        )
    except subprocess.TimeoutExpired:
        fallback["bnb_error"] = "probe timed out"
        return fallback

    # Scan for the marker ANYWHERE, not just at the start of a line. GPU
    # libraries print banners without trailing newlines, which would otherwise
    # glue our marker onto their output and hide a perfectly good result.
    if (found := _extract(proc.stdout, "RESULT:")) is not None:
        return found

    # No result. Recover whatever the child proved before it stopped.
    partial = _extract(proc.stderr, "PARTIAL:")
    if partial is not None:
        partial["bnb_error"] = (
            "bitsandbytes aborted the process (signal/exit %s) -- the installed "
            "build has no kernels for this GPU. Quantization is unavailable."
            % proc.returncode
        ) if proc.returncode != 0 else (
            "bitsandbytes did not report a result, though the probe exited "
            "cleanly. Treating quantization as unavailable."
        )
        return partial
    fallback["bnb_error"] = "probe produced no result (exit %s)" % proc.returncode
    return fallback


def _retry_attention(sub: dict, caps: dict) -> dict:
    """Second opinion on attention, with the experimental kernels switched on.

    Only ever runs when the first probe found no fused kernel at all, because
    it can only turn a no into a yes. Returns the environment that has to be in
    place for the answer it gives -- empty when the retry changed nothing,
    which is the common case and the one that costs a few seconds.
    """
    if sub.get("flash_attn") or sub.get("mem_efficient_attn"):
        return {}
    if caps.get("backend") != "rocm":
        # The flag is a ROCm one. Setting it elsewhere is not harmful, but a
        # probe that cannot succeed is a probe not worth the seconds.
        return {}
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _ATTENTION_PROBE],
            capture_output=True, text=True, timeout=300,
            env={**os.environ, **_ATTENTION_RETRY_ENV, "PYTHONWARNINGS": "ignore"},
        )
    except (subprocess.TimeoutExpired, OSError):
        return {}
    got = _extract(proc.stdout, "RESULT:")
    if not got or not (got.get("flash_attn") or got.get("mem_efficient_attn")):
        return {}
    sub["flash_attn"] = bool(got.get("flash_attn"))
    sub["mem_efficient_attn"] = bool(got.get("mem_efficient_attn"))
    return dict(_ATTENTION_RETRY_ENV)


def _extract(stream: str | None, marker: str) -> dict | None:
    """Pull the last JSON object tagged with `marker` out of a noisy stream."""
    if not stream:
        return None
    idx = stream.rfind(marker)
    if idx < 0:
        return None
    tail = stream[idx + len(marker):]
    line = tail.split("\n", 1)[0].strip()
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def _benchmark_dtypes(torch, device: str) -> dict:
    """Measure real throughput per dtype. See rule 2 in the module docstring."""
    results: dict[str, float | None] = {}
    n = 2048
    for name, dt in (("float32", torch.float32),
                     ("float16", torch.float16),
                     ("bfloat16", torch.bfloat16)):
        try:
            a = torch.randn(n, n, device=device, dtype=dt)
            b = torch.randn(n, n, device=device, dtype=dt)
            torch.matmul(a, b)
            _sync(torch, device)
            t0 = time.time()
            iters = 10
            for _ in range(iters):
                torch.matmul(a, b)
            _sync(torch, device)
            elapsed = time.time() - t0
            results[name] = round((iters * 2 * n ** 3) / elapsed / 1e12, 1)
            del a, b
            if device == "cuda":
                torch.cuda.empty_cache()
            elif device == "mps":
                torch.mps.empty_cache()
        except Exception:
            results[name] = None
    return results


def _sync(torch, device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def _agent_version() -> str:
    """The version in pyproject.toml, read from the installed tree."""
    root = Path(__file__).resolve().parent.parent
    try:
        for line in (root / "pyproject.toml").read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("version"):
                return line.split("=", 1)[1].strip().strip("\"'")
    except OSError:
        pass
    return "unknown"


AGENT_VERSION = _agent_version()


def _env_limits() -> dict:
    """This machine's own cap on what the studio may use of its card."""
    from . import limits as gpu_limits
    return gpu_limits.from_env()


def probe(quick: bool = False) -> dict:
    """Build the capability report this runner advertises to the controller."""
    caps: dict = {
        "hostname": platform.node(),
        "os": platform.system(),
        "python": platform.python_version(),
        # Which build of the agent this is. A runner whose image is a month
        # old reports a capability set a month old -- no `kinds`, no
        # `modalities`, none of the libraries -- and the controller, which
        # deliberately does not refuse a machine for saying nothing, hands it
        # work it cannot do. Saying the version out loud is what lets the
        # Machines page point at the actual problem instead of leaving
        # somebody to infer it from a missing field.
        "agent_version": AGENT_VERSION,
        # What this machine's own configuration allows, before the controller
        # has any say. Reported rather than applied here: the probe describes
        # the machine, and the limit is enforced where the work happens.
        "limits": _env_limits(),
        "probed_at": time.time(),
        "backend": "cpu",
        "device_name": platform.processor() or "CPU",
        "arch": None,
        "vram_gb": None,
        "compute_units": None,
        "torch_version": None,
        "dtypes": {},
        "quantization": {"4bit": False, "8bit": False, "optim_8bit": False},
        "attention": {"flash": False, "mem_efficient": False, "flex": False,
                      "flex_opt_in": False, "math": True, "env": {}},
        "recommended_dtype": "float32",
        "warnings": [],
        "notes": [],
    }

    # A machine can be told to take only certain kinds of work. The reason
    # this exists is the queue: a runner does one job at a time, so an upload
    # or a dataset written by a hosted model -- neither of which touches the
    # GPU -- would otherwise sit behind six hours of training. A second,
    # GPU-less runner set to `upload,generate_dataset` picks those up while
    # the card carries on training.
    if only := os.environ.get("AI_STUDIO_RUNNER_KINDS", "").strip():
        caps["kinds"] = [k.strip() for k in only.split(",") if k.strip()]

    # What else is on this machine besides a card. A vision run needs an
    # image library, a speech run an audio one, and a diffusion run the
    # diffusers package; a machine without them can hold the model and still
    # fail at the first batch. Reported so the scheduler can say "not on this
    # machine" before the download rather than after it. Cores and RAM
    # matter for the same reason on the CPU-bound stages -- decoding a
    # thousand images a step is not GPU work.
    caps["libraries"] = _libraries()
    # Said in its own key as well as in the library list, because it is a
    # promise the serving API makes to callers rather than a package somebody
    # installed. A runner whose image predates this reports False and the
    # controller refuses the request in words instead of ignoring the field.
    caps["constrained_decoding"] = bool(
        caps["libraries"].get("lmformatenforcer"))
    caps["cpu_cores"] = os.cpu_count()
    caps["ram_gb"] = _ram_gb()

    try:
        import torch
    except ImportError:
        caps["warnings"].append(
            "PyTorch is not installed, so this runner cannot train anything.")
        return caps

    caps["torch_version"] = torch.__version__

    # ---- identify the accelerator ------------------------------------
    if torch.cuda.is_available():
        is_rocm = getattr(torch.version, "hip", None) is not None
        caps["backend"] = "rocm" if is_rocm else "cuda"
        props = torch.cuda.get_device_properties(0)
        caps["device_name"] = props.name
        caps["vram_gb"] = round(props.total_memory / 1024 ** 3, 1)
        caps["compute_units"] = props.multi_processor_count
        caps["arch"] = getattr(props, "gcnArchName", None) or (
            "sm_%d%d" % (props.major, props.minor))
        if is_rocm:
            caps["rocm_version"] = torch.version.hip
            # rocminfo gives the real marketing name; torch often reports a
            # generic "AMD Radeon Graphics" placeholder.
            caps["device_name"] = _rocm_marketing_name() or props.name
        else:
            caps["cuda_version"] = torch.version.cuda
        device = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        caps["backend"] = "mps"
        gpu = _apple_gpu()
        caps["device_name"] = ("%s GPU" % gpu["name"]) if gpu.get("name") \
            else "Apple Silicon GPU"
        caps["compute_units"] = gpu.get("cores")
        caps["arch"] = platform.machine()
        caps["unified_memory"] = True
        if gpu.get("metal"):
            caps["metal_family"] = gpu["metal"]
        # Metal's own recommendation for how much this process should hold at
        # once (`recommendedMaxWorkingSetSize`), which is the number to plan
        # against on a unified-memory machine. Total RAM is the wrong one: it
        # is shared with everything else running, and a model sized against it
        # does not fail cleanly, it swaps, and a swapping training run is
        # indistinguishable from a hung one.
        with contextlib.suppress(Exception):
            caps["vram_gb"] = round(
                torch.mps.recommended_max_memory() / 1024 ** 3, 1)
        device = "mps"
        caps["notes"].append(
            "Apple Metal shares system RAM with the GPU. Of this machine's %s, "
            "Metal recommends holding at most %s at once, and that is the "
            "figure the size estimates here are based on."
            % ("%.0f GB" % caps["ram_gb"] if caps.get("ram_gb") else "memory",
               "%.1f GB" % caps["vram_gb"] if caps.get("vram_gb") else "a share"))
    else:
        device = "cpu"
        caps["warnings"].append(
            "No GPU detected. Training will run on the CPU, which is fine for "
            "trying the app out but far too slow for real work.")

    # ---- what can it compute, and how fast ---------------------------
    if not quick and device != "cpu":
        caps["dtypes"] = _benchmark_dtypes(torch, device)
        sub = _run_subprocess_probe()
        caps["quantization"] = {
            "4bit": sub["bnb_4bit"],
            # Separately, because they come apart: a card can quantize
            # correctly for the wide shapes training uses and return noise for
            # the one-row shape that writing a reply uses. Serving reads this
            # one; training reads the other.
            "4bit_decode": sub.get("bnb_4bit_decode", False),
            "4bit_error": sub.get("bnb_4bit_error"),
            "4bit_decode_error": sub.get("bnb_4bit_decode_error"),
            "8bit": sub["bnb_8bit"],
            "optim_8bit": sub["bnb_optim"],
        }
        # A card that reports no fused kernel gets one more chance, with the
        # experimental ROCm kernels enabled -- see _ATTENTION_RETRY_ENV. The
        # environment that produced the answer is reported alongside it, and
        # every job sets it before loading a model, so the kernel a run gets is
        # the kernel the memory estimate was written against.
        attn_env = _retry_attention(sub, caps)
        caps["attention"] = {
            "flash": sub["flash_attn"],
            "mem_efficient": sub["mem_efficient_attn"],
            "flex": sub.get("flex_attn", False),
            # Whether this machine has been told to USE FlexAttention, which
            # is a separate question from whether it compiles here -- see the
            # long note in common/attention.py for the twenty-four restarts
            # that separated the two.
            "flex_opt_in": bool(os.environ.get("AI_STUDIO_FLEX_ATTENTION")),
            "math": True,
            "env": attn_env,
        }
        if sub.get("flex_attn") and not os.environ.get("AI_STUDIO_FLEX_ATTENTION"):
            caps["notes"].append(
                "FlexAttention compiles on this card, which would allow much "
                "longer sequences. It is off because compiling in a probe "
                "does not predict compiling for a real model here -- one that "
                "passed this check then failed on Qwen2.5-3B, and took the "
                "runner down with it. Set AI_STUDIO_FLEX_ATTENTION=1 to try "
                "it; a model it cannot compile for falls back rather than "
                "failing the run.")
        if attn_env:
            caps["notes"].append(
                "Fused attention on this card needs %s, which the studio sets "
                "for every run it starts here. Without it torch reports no "
                "kernel and falls back to attention whose memory grows with "
                "the square of sequence length."
                % ", ".join("%s=%s" % kv for kv in sorted(attn_env.items())))
        if sub.get("bnb_error"):
            caps["quantization"]["error"] = sub["bnb_error"]
        if sub.get("sdpa_error"):
            caps["warnings"].append(
                "This machine's GPU could not run an attention kernel at all "
                "(%s). Training here will fail; the PyTorch build most likely "
                "does not match the hardware." % sub["sdpa_error"])

    _derive_recommendations(caps)
    caps["modalities"] = _modalities(caps)
    return caps


# The libraries each kind of work needs, by import name. Checked for presence
# only -- versions are the job's problem -- and checked without importing
# torch-heavy packages fully where a lighter probe exists.
_LIBRARIES = ("torchvision", "torchaudio", "diffusers", "PIL", "soundfile",
              "librosa", "timm", "transformers", "peft", "datasets",
              # The grammar engine behind `response_format`. Reported because
              # the controller must be able to tell "this machine will hold a
              # reply to your schema" from "this machine will ignore that you
              # asked", and answer differently -- see runner/grammar.py.
              "lmformatenforcer")


def _libraries() -> dict:
    import importlib.util
    import shutil
    found = {name: importlib.util.find_spec(name) is not None
             for name in _LIBRARIES}
    found["ffmpeg"] = shutil.which("ffmpeg") is not None
    return found


def _ram_gb() -> float | None:
    try:
        if hasattr(os, "sysconf"):
            pages = os.sysconf("SC_PHYS_PAGES")
            size = os.sysconf("SC_PAGE_SIZE")
            if pages > 0 and size > 0:
                return round(pages * size / 1024 ** 3, 1)
    except (ValueError, OSError):
        pass
    return None


def _modalities(caps: dict) -> list[str]:
    """What kinds of data this machine can train on and serve, by name.

    Derived from the libraries rather than declared, so a machine that gains
    torchaudio gains audio on its next probe and nobody edits a config.
    """
    libs = caps.get("libraries") or {}
    out = []
    if caps.get("torch_version"):
        out.append("text")
        if libs.get("torchvision") or libs.get("PIL"):
            out.append("vision")
        if libs.get("torchaudio") or libs.get("soundfile") or libs.get("librosa"):
            out.append("audio")
        if libs.get("diffusers"):
            out.append("diffusion")
    return out


def _apple_gpu() -> dict:
    """Chip name, GPU core count and Metal family, asked of macOS itself.

    torch says nothing about an Apple GPU beyond "there is one" -- no name, no
    core count, no memory -- which is why this runner used to report the bare
    string "Apple Silicon GPU" and a dash for every other field. macOS knows
    all of it; it just has to be asked.
    """
    info: dict = {}
    try:
        out = subprocess.run(["system_profiler", "-json", "SPDisplaysDataType"],
                             capture_output=True, text=True, timeout=30).stdout
        for gpu in json.loads(out).get("SPDisplaysDataType", []):
            if gpu.get("sppci_device_type") not in (None, "spdisplays_gpu"):
                continue
            info["name"] = gpu.get("sppci_model") or gpu.get("_name")
            cores = str(gpu.get("sppci_cores") or "").strip()
            if cores.isdigit():
                info["cores"] = int(cores)
            # e.g. "spdisplays_metal4" -> "metal4"
            if fam := (gpu.get("spdisplays_mtlgpufamilysupport") or ""):
                info["metal"] = fam.replace("spdisplays_", "") or None
            break
    except Exception:  # noqa: BLE001 - absent or reshaped output is not fatal
        pass
    if not info.get("name"):
        # system_profiler is slow and occasionally unavailable; the chip name
        # alone is worth a second, cheaper try.
        with contextlib.suppress(Exception):
            info["name"] = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=10).stdout.strip() or None
    return info


def _rocm_marketing_name() -> str | None:
    try:
        out = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return None
    for line in out.splitlines():
        if "Marketing Name" in line:
            name = line.split(":", 1)[1].strip()
            if name and "CPU" not in name.upper():
                return name
    return None


def _derive_recommendations(caps: dict) -> None:
    """Turn raw measurements into the guidance the UI shows to a beginner."""
    dt = caps.get("dtypes") or {}
    fp16, bf16 = dt.get("float16"), dt.get("bfloat16")

    if caps["backend"] == "cpu":
        caps["recommended_dtype"] = "float32"
    elif fp16 and bf16 and fp16 > bf16 * 1.3:
        # The RDNA2 case: bf16 works but is emulated and far slower.
        caps["recommended_dtype"] = "float16"
        caps["warnings"].append(
            "This GPU computes bfloat16 %.1fx slower than float16 (%.1f vs %.1f "
            "TFLOP/s) because it has no native bfloat16 hardware. Use float16 "
            "here, even though most guides recommend bfloat16."
            % (bf16 and fp16 / bf16 or 0, bf16 or 0, fp16 or 0))
    elif bf16:
        # bf16 is the safer default when it is not penalised: its wider
        # exponent range avoids the loss-scaling instability fp16 can hit.
        caps["recommended_dtype"] = "bfloat16"
    elif fp16:
        caps["recommended_dtype"] = "float16"

    arch = (caps.get("arch") or "").lower()
    # Flash attention is not the question -- see common/attention.py. The
    # memory-efficient kernel is a different implementation of the same idea
    # and costs the same memory, so a card that has it is not penalised here
    # for lacking the one with the famous name.
    plan = attention.report(caps)
    caps["max_recommended_seq_len"] = plan["comfortable_seq"]
    if plan["quadratic"] and caps["backend"] != "cpu":
        detail = ""
        if caps["backend"] == "mps":
            detail = (" Metal has no flash-attention kernels, and whether"
                      " PyTorch's own path avoids that cost cannot be measured"
                      " from here, so the safe assumption is made.")
        elif any(a in arch for a in _NO_FLASH_ATTN_ARCHS):
            detail = (" %s has no flash-attention kernels in ROCm." % caps["arch"])
        caps["warnings"].append(
            "No fused attention kernel is available, so attention falls back "
            "to a method whose memory use grows with the square of sequence "
            "length." + detail + " %d is the comfortable length here; longer"
            " runs are allowed and cost memory rising with the square, so the"
            " studio drops them to one sequence at a time and makes the batch"
            " up by accumulation." % plan["comfortable_seq"])
    elif not plan["flash"] and caps["backend"] != "cpu":
        # Worth saying, because the Machines page shows a "Flash attention"
        # row and a bare no there reads as a card that cannot do long
        # sequences. It can: it just gets there by the other kernel.
        caps["notes"].append(
            "Flash attention is unavailable, but the memory-efficient "
            "attention kernel works here. It costs the same memory -- linear "
            "in sequence length, not quadratic -- and a little more time, so "
            "long sequences are fine on this card.")

    if not caps["quantization"].get("4bit") and caps["backend"] == "mps":
        # Apple silicon is NOT a platform without 4-bit: bitsandbytes ships a
        # macOS arm64 wheel and its 4-bit path measures correct here (about
        # 0.11 relative error, the same as a working CUDA card). So when it is
        # missing the cause is the install, which is worth saying, because
        # "unavailable on this machine" reads as a dead end and this one is a
        # pip install away from quadrupling the model size that fits.
        caps["warnings"].append(
            "4-bit quantization is unavailable because bitsandbytes is not "
            "installed or is too old (%s). Apple silicon does support it: "
            "install bitsandbytes 0.50 or newer and re-check the hardware."
            % (caps["quantization"].get("error") or "no reason reported"))
    elif not caps["quantization"].get("4bit") and caps["backend"] != "cpu":
        caps["warnings"].append(
            "4-bit quantization is unavailable on this runner, so large models "
            "cannot be shrunk to fit. That lowers the biggest model you can "
            "fine-tune here.")
    elif not caps["quantization"].get("4bit_decode") and caps["backend"] != "cpu":
        # The half-broken case, and the one worth spelling out: it can train
        # this way and it must not answer this way. Left unsaid, the studio
        # would compress a model to make it fit and then hand back noise.
        caps["warnings"].append(
            "4-bit works here for training but returns wrong numbers when a "
            "model writes one token at a time (%.2f relative error against "
            "float16, where correct is about 0.10). Models are fine-tuned in "
            "4-bit on this machine and served at full precision, so a model "
            "too large to serve uncompressed cannot be talked to here."
            % (caps["quantization"].get("4bit_decode_error") or 0))

    caps["max_finetune_params_b"] = _estimate_max_model(caps)
    caps["max_scratch_params_m"] = _estimate_max_scratch(caps)


def _estimate_max_model(caps: dict) -> float | None:
    """Largest model (in billions of params) this runner can LoRA fine-tune.

    LoRA freezes the base model, so the frozen weights dominate. Activations do
    NOT shrink with quantization, so they are subtracted as a fixed budget
    rather than folded into a percentage -- and that budget roughly doubles
    without a fused attention kernel, where attention memory grows with the
    square of sequence length.
    """
    vram = caps.get("vram_gb")
    if not vram:
        return None
    bytes_per_param = 0.5 if caps["quantization"].get("4bit") else 2.0
    activation_budget = 2.0 if attention.is_fused(caps) else 3.5
    usable = max(0.0, vram - activation_budget) * 0.85
    return round(usable / bytes_per_param, 1) or None


def _estimate_max_scratch(caps: dict) -> float | None:
    """Largest model (in millions of params) this runner could train from
    scratch, where every parameter is updated.

    An order of magnitude below the LoRA ceiling, and for a different reason.
    LoRA freezes the base model, so only the tiny adapter carries optimiser
    state. Here every parameter needs fp32 weights, fp32 gradients and two
    Adam moments -- 16 bytes each before a single activation exists, against
    the 2 bytes a frozen fp16 weight costs.

    Worth reporting, but rarely the binding limit: on a single consumer GPU
    compute runs out long before memory does. See controller/architectures.
    """
    vram = caps.get("vram_gb")
    if not vram:
        return None
    per_param = (10 if caps["quantization"].get("optim_8bit") else 16) + 2
    usable = max(0.0, vram - 3.0) * 1024 ** 3
    return round(usable / per_param / 1e6) or None


def peak_memory_gb(device: str) -> float | None:
    """The most GPU memory this process has held, for a run's progress and
    summary.

    Every caller used to ask `torch.cuda.max_memory_allocated()` and report
    nothing at all when the answer was not CUDA, so a run on Apple silicon
    charted a flat empty line where its memory use should have been -- on the
    one kind of machine where that number matters most, because the memory
    being filled is the same memory the rest of the desktop is using.

    Metal keeps no high-water mark, so the driver's allocation stands in for
    one: it counts what torch has taken from the system and, unlike the live
    figure, does not drop back when a tensor is freed into the allocator's
    cache.
    """
    try:
        import torch
        if device == "cuda" and torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)
        if device == "mps":
            return round(torch.mps.driver_allocated_memory() / 1024 ** 3, 2)
    except Exception:  # noqa: BLE001 - telemetry never fails a run
        return None
    return None


def attention_plan(caps: dict) -> dict:
    """What attention this job should ask for, with the environment applied.

    Applying the environment is the point of doing this here rather than
    reading `common.attention.report` directly: the variable that unlocked a
    fused kernel during the probe has to be set in the process that loads the
    model, or the run gets the slow kernel and the fast estimate. Every job --
    fine-tuning, training from scratch and serving -- calls this once before
    `from_pretrained`.

    Setting it is safe to repeat and safe on machines that never needed it,
    where the dict is empty and this does nothing at all. The same call limits
    torch to the attention backends the probe proved -- see
    `apply_sdpa_backends` -- so one call covers both halves of the plan.
    """
    plan = attention.report(caps)
    for key, value in (plan.get("env") or {}).items():
        os.environ.setdefault(key, value)
    plan["enabled_backends"] = apply_sdpa_backends(plan)
    return plan


# The switch torch offers for each SDPA backend. Process-global rather than
# scoped, which suits a runner exactly: it does one job at a time, and each job
# sets these once against its own plan before it loads a model.
_SDPA_SWITCHES = {
    "FLASH_ATTENTION": "enable_flash_sdp",
    "EFFICIENT_ATTENTION": "enable_mem_efficient_sdp",
    "MATH": "enable_math_sdp",
}


def apply_sdpa_backends(plan: dict) -> list[str]:
    """Allow exactly the attention backends the probe proved, and no others.

    Left alone, torch decides per call which backend to dispatch to, and it
    decides from the shapes rather than from whether the kernel works on this
    card. On ROCm that has been observed to pick a backend that then raises
    inside the backward pass -- a fault an hour into a run, from a heuristic
    that had no way of knowing better. The probe does know, so the answer is
    set here instead of being rediscovered per call.

    MATH is in every plan, so this never leaves torch with nothing to
    dispatch to: an unusual head dimension or mask that no fused kernel
    supports still runs, slowly, rather than raising.

    Returns the backends left enabled, for the log. An older torch that lacks
    one of these switches simply keeps whatever it does by default.
    """
    try:
        import torch
    except ImportError:
        return []
    wanted = set(plan.get("backends") or _SDPA_SWITCHES)
    enabled = []
    for name, switch in _SDPA_SWITCHES.items():
        fn = getattr(torch.backends.cuda, switch, None)
        if fn is None:
            continue
        try:
            fn(name in wanted)
        except (RuntimeError, TypeError):
            continue
        if name in wanted:
            enabled.append(name)
    return enabled


def expert_kernel(caps: dict) -> str | None:
    """Which of transformers' expert dispatch paths this machine can use.

    None means "leave the library to choose". `"eager"` is forced on ROCm, and
    the reason is worth recording because the library's own guard gets it
    wrong: transformers 5 defaults to a grouped GEMM, asks
    `_grouped_mm_can_dispatch()` whether that is available, is told yes on
    ROCm because torch exposes the symbol, and then dies inside the forward
    pass with "grouped gemm is not supported on ROCM". The fallback exists and
    never fires, so it has to be chosen here instead.

    Measured on gfx1030 with transformers 5.15: `grouped_mm` raises,
    `batched_mm` runs every token through every expert and asked for 44 GB on
    a model that fits in 5, and `eager` works and returns the router logits
    the load-balancing loss needs. That leaves one option, not a preference.
    """
    return "eager" if caps.get("backend") == "rocm" else None


if __name__ == "__main__":
    print(json.dumps(probe(), indent=2))
