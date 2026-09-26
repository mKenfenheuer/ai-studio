"""Turning a finished model into the file everything outside this studio wants.

The run page has said for a long time that merging an adapter produces "what
Ollama, llama.cpp and everything else outside this studio want". That was half
true. Those tools want **GGUF**: one file, with the weights quantised down to
four or five bits, that loads on a laptop with no Python at all. A merged
model in Hugging Face format is what you convert *from*.

So the last mile was missing, and it is the mile most people actually walk:
train something here, then run it on the machine under the desk. Doing it by
hand means cloning llama.cpp, finding the right converter for the month, and
knowing which of eleven quantisation types to ask for.

Two steps, both of them on the CPU:

* **Convert.** `convert_hf_to_gguf.py` from llama.cpp reads the safetensors
  and writes one GGUF at full precision. This is the step that knows about
  architectures, and it is the reason the converter is pinned to a known
  commit in the image rather than fetched at run time -- a model that
  converted last week and does not this week is the worst kind of surprise.
* **Quantise.** `llama-quantize` rewrites that file smaller. Q4_K_M is the
  default because it is the one people mean: about a quarter of the size, and
  the quality loss is small enough that it is the standard thing to ship.

Both are subprocesses, and their output is streamed into the run's log. A
conversion that fails says why in the same place as everything else.
"""
from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from runner import artifacts

from .lora_llm import Cancelled

# Where the image puts llama.cpp. Both are overridable so a bare-metal runner
# with its own checkout can point at it.
CONVERTER = os.environ.get(
    "AI_STUDIO_GGUF_CONVERT", "/opt/llama.cpp/convert_hf_to_gguf.py")
QUANTIZE = os.environ.get("AI_STUDIO_GGUF_QUANTIZE", "/opt/llama.cpp/llama-quantize")

# The Python that runs the converter. Its own environment in the image, because
# llama.cpp pins an older torch and transformers than the runner uses -- see the
# note in docker/Dockerfile.runner.cpu. A bare-metal runner with no such
# environment runs it with the runner's own interpreter, as before.
_VENV_PYTHON = "/opt/llama.cpp/venv/bin/python"
CONVERT_PYTHON = os.environ.get("AI_STUDIO_GGUF_PYTHON") or (
    _VENV_PYTHON if os.path.exists(_VENV_PYTHON) else "python")

# What to offer, and what each is for. Ordered by size.
QUANT_TYPES = {
    "Q4_K_M": "About a quarter of the size. The one nearly everybody means.",
    "Q5_K_M": "A little bigger, a little better. Worth it if the model is small.",
    "Q6_K": "Close to the original, at about half the size.",
    "Q8_0": "Barely distinguishable from full precision, at half the size.",
    "F16": "No quantisation at all. The conversion, and nothing else.",
}


def _stream(cmd: list[str], ctx: Any, cwd: str | None = None) -> None:
    """Run a subprocess, putting its output in the run's log as it arrives.

    Line by line rather than at the end: a conversion of a 7B model is several
    minutes of silence otherwise, which is indistinguishable from a hang.
    """
    ctx.log("$ " + " ".join(cmd[:2]) + " …")
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    tail: list[str] = []
    try:
        for line in proc.stdout or ():
            line = line.rstrip()
            if not line:
                continue
            tail.append(line)
            del tail[:-40]
            # The converters are chatty about every tensor. Only the lines
            # that say something are worth a reader's attention.
            if re.search(r"error|warn|traceback|quantiz|writing|model|size",
                         line, re.I):
                ctx.log(line[:300])
            if ctx.should_cancel():
                proc.terminate()
                raise Cancelled()
    finally:
        proc.wait()
    if proc.returncode != 0:
        raise ValueError("%s failed (%d). Last of its output:\n%s"
                         % (Path(cmd[1] if len(cmd) > 1 else cmd[0]).name,
                            proc.returncode, "\n".join(tail[-12:])))


def run(cfg: dict, ctx: Any) -> dict:
    source_job = cfg.get("source_job") or ""
    if not source_job:
        raise ValueError("No run was named to export.")
    quant = (cfg.get("quantize") or "Q4_K_M").upper()
    if quant not in QUANT_TYPES:
        raise ValueError("%s is not a quantisation this can produce. Choose "
                         "one of: %s." % (quant, ", ".join(QUANT_TYPES)))

    if not Path(CONVERTER).exists():
        raise ValueError(
            "This machine has no GGUF converter. The CPU runner image ships "
            "one; a machine set up by hand needs llama.cpp, with "
            "AI_STUDIO_GGUF_CONVERT pointing at its convert_hf_to_gguf.py.")

    ctx.progress(0, 3, stage="loading_model")
    folder = artifacts.fetch(ctx.controller_url, ctx.runner_token,
                             source_job, ctx.log)
    if (folder / "adapter_config.json").exists():
        raise ValueError(
            "That is an adapter, not a model. An adapter is a few megabytes "
            "that mean nothing without the exact weights they were fitted to, "
            "and there is no such thing as a GGUF of one. Export the merged "
            "model instead.")

    out_dir = Path(ctx.workdir) / "gguf"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = re.sub(r"[^A-Za-z0-9._-]+", "-",
                  cfg.get("name_hint") or source_job).strip("-") or "model"
    f16 = out_dir / ("%s.f16.gguf" % name)

    ctx.progress(1, 3, stage="converting")
    ctx.log("Converting to GGUF. This reads every tensor once and writes one "
            "file at full precision; the quantisation comes after.")
    t0 = time.time()
    _stream([CONVERT_PYTHON, CONVERTER, str(folder), "--outfile", str(f16),
             "--outtype", "f16"], ctx)
    ctx.log("Converted in %s. %s"
            % (_took(time.time() - t0), _size(f16)))

    final = f16
    if quant != "F16":
        if not Path(QUANTIZE).exists():
            raise ValueError(
                "This machine can convert but not quantise: llama-quantize is "
                "not on it. Export as F16, or point AI_STUDIO_GGUF_QUANTIZE at "
                "a built copy.")
        ctx.progress(2, 3, stage="quantizing")
        final = out_dir / ("%s.%s.gguf" % (name, quant))
        ctx.log("Quantising to %s. %s" % (quant, QUANT_TYPES[quant]))
        t0 = time.time()
        _stream([QUANTIZE, str(f16), str(final), quant], ctx)
        ctx.log("Quantised in %s. %s" % (_took(time.time() - t0), _size(final)))
        # The intermediate is several times the size of the answer and is of
        # no use to anybody once the answer exists.
        f16.unlink(missing_ok=True)

    # What to do with it, in the file, because a model downloaded in six weeks
    # will not have this page open beside it.
    (out_dir / "Modelfile").write_text(
        "# Load this into Ollama with:\n"
        "#   ollama create %s -f Modelfile\n"
        "#   ollama run %s\n"
        "FROM ./%s\n" % (name, name, final.name), encoding="utf-8")
    (out_dir / "README.txt").write_text(
        "%s\n\nQuantisation: %s -- %s\n\n"
        "llama.cpp:\n  llama-cli -m %s -p \"Hello\"\n\n"
        "Ollama:\n  ollama create %s -f Modelfile\n  ollama run %s\n\n"
        "Made by AI Studio from run %s.\n"
        % (name, quant, QUANT_TYPES[quant], final.name, name, name, source_job),
        encoding="utf-8")

    ctx.progress(3, 3, stage="saving")
    size = final.stat().st_size
    # Stored, not deflated, for the same reason every other artifact is: a
    # quantised GGUF is dense and comes back one percent smaller having read
    # every byte.
    zip_path = artifacts.pack(out_dir, Path(ctx.workdir) / ("%s-gguf.zip" % name))
    return {
        "kind": "export_gguf",
        "source_job": source_job,
        "quantize": quant,
        "filename": final.name,
        "bytes": size,
        "artifact_paths": {"gguf": str(zip_path)},
        "artifact_size": zip_path.stat().st_size,
        "note": "%s, %s" % (final.name, _size(final)),
    }


def _size(path: Path) -> str:
    n = float(path.stat().st_size)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return "%.1f %s" % (n, unit)
        n /= 1024
    return "%.1f GB" % n


def _took(seconds: float) -> str:
    return ("%.0f seconds" % seconds if seconds < 90
            else "%.0f minutes" % (seconds / 60))
