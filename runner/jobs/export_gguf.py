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
import json
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
# The adapter converter sits beside the model converter in every llama.cpp.
LORA_CONVERTER = os.environ.get(
    "AI_STUDIO_GGUF_CONVERT_LORA",
    str(Path(CONVERTER).with_name("convert_lora_to_gguf.py")))
# What an adapter can be written as. Small either way -- a rank-32 adapter on
# a 1.7B model's attention is 26 MB at f16 -- so there is little to gain from
# going lower, and it is applied on top of whatever the base was quantised to.
LORA_TYPES = {"F16": "f16", "Q8_0": "q8_0", "BF16": "bf16"}

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
    if (cfg.get("what") or "model") == "adapter":
        return _export_adapter(cfg, ctx, source_job,
                               (cfg.get("quantize") or "F16").upper())
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
    folder = _readable_by_converter(folder, Path(ctx.workdir) / "convert-src")
    _stream([CONVERT_PYTHON, CONVERTER, str(folder), "--outfile", str(f16),
             "--outtype", "f16"], ctx)
    ctx.log("Converted in %s. %s"
            % (_took(time.time() - t0), _size(f16)))

    # A model that looks is two files in llama.cpp: the language model, and
    # the projector that turns a picture into what the language model reads
    # (`mmproj`). The converter writes the second on its own pass. It is kept
    # at 16 bits -- it is a few hundred megabytes, it is what every runtime
    # expects, and quantising the image path costs accuracy the language
    # model's quantisation does not.
    mmproj = None
    if _is_vision(folder):
        mmproj = out_dir / ("mmproj-%s.f16.gguf" % name)
        ctx.log("This model has an image encoder; writing its projector as a "
                "second file, which llama.cpp loads beside the model "
                "(--mmproj).")
        t0 = time.time()
        _stream([CONVERT_PYTHON, CONVERTER, str(folder), "--mmproj",
                 "--outfile", str(mmproj), "--outtype", "f16"], ctx)
        ctx.log("Projector written in %s. %s" % (_took(time.time() - t0), _size(mmproj)))

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
        "llama.cpp:\n  llama-cli -m %s -p \"Hello\"\n%s\n"
        "Ollama:\n  ollama create %s -f Modelfile\n  ollama run %s\n\n"
        "Made by AI Studio from run %s.\n"
        % (name, quant, QUANT_TYPES[quant], final.name,
           ("  llama-mtmd-cli -m %s --mmproj %s --image photo.jpg -p \"...\"\n"
            % (final.name, mmproj.name)) if mmproj else "",
           name, name, source_job),
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
        "mmproj": mmproj.name if mmproj else None,
        "bytes": size,
        "artifact_paths": {"gguf": str(zip_path)},
        "artifact_size": zip_path.stat().st_size,
        "note": "%s, %s" % (final.name, _size(final)),
    }


def _export_adapter(cfg: dict, ctx: Any, source_job: str, quant: str) -> dict:
    """A run's LoRA as a GGUF adapter, for llama.cpp to apply to its base.

    `convert_lora_to_gguf.py` reads the adapter's weights, and the base's
    config for the shapes and names they belong to; the base's weights are
    never read. The base is the model the adapter was fitted to -- another
    run's merged model, fetched like any other, or a Hub id.
    """
    if not Path(LORA_CONVERTER).exists():
        raise ValueError("This machine's llama.cpp has no convert_lora_to_gguf.py.")
    outtype = LORA_TYPES.get(quant)
    if not outtype:
        raise ValueError("An adapter is written as %s; %s is for models."
                         % (", ".join(LORA_TYPES), quant))
    ctx.progress(0, 3, stage="loading_model")
    # A run that also merged keeps its adapter beside the model; one that did
    # not keeps it under the run's own name.
    try:
        adapter = artifacts.fetch(ctx.controller_url, ctx.runner_token,
                                  source_job, ctx.log, kind="adapter",
                                  keep=[source_job, cfg.get("base_job") or ""])
    except Exception:  # noqa: BLE001 - the other place it can be
        adapter = artifacts.fetch(ctx.controller_url, ctx.runner_token,
                                  source_job, ctx.log)
    if not (adapter / "adapter_config.json").exists():
        raise ValueError("Run %s has no adapter to export." % source_job)

    base_args: list[str]
    if base_job := cfg.get("base_job"):
        base = artifacts.fetch(ctx.controller_url, ctx.runner_token, base_job,
                               ctx.log, keep=[source_job, base_job])
        base = _readable_by_converter(base, Path(ctx.workdir) / "convert-base")
        base_args = ["--base", str(base)]
        ctx.log("The adapter's base is run %s's model." % base_job)
    else:
        base_args = ["--base-model-id", cfg["base_model"]]
        ctx.log("The adapter's base is %s." % cfg["base_model"])

    out_dir = Path(ctx.workdir) / "gguf"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = re.sub(r"[^A-Za-z0-9._-]+", "-",
                  cfg.get("name_hint") or source_job).strip("-") or "adapter"
    final = out_dir / ("%s.lora.%s.gguf" % (name, outtype))
    ctx.progress(1, 3, stage="converting")
    t0 = time.time()
    _stream([CONVERT_PYTHON, LORA_CONVERTER, str(adapter), *base_args,
             "--outfile", str(final), "--outtype", outtype], ctx)
    ctx.log("Adapter written in %s. %s" % (_took(time.time() - t0), _size(final)))
    (out_dir / "README.txt").write_text(
        "%s\n\nA LoRA adapter, %s. It means nothing on its own: load it on top of\n"
        "the model it was fitted to (%s).\n\n"
        "llama.cpp:\n  llama-cli -m <base>.gguf --lora %s -p \"Hello\"\n\n"
        "Made by AI Studio from run %s.\n"
        % (name, outtype, cfg.get("base_job") or cfg.get("base_model"), final.name, source_job),
        encoding="utf-8")
    ctx.progress(3, 3, stage="saving")
    zip_path = artifacts.pack(out_dir, Path(ctx.workdir) / ("%s-gguf.zip" % name))
    return {
        "kind": "export_gguf",
        "what": "adapter",
        "source_job": source_job,
        "base_job": cfg.get("base_job") or None,
        "base_model": cfg.get("base_model") or None,
        "quantize": quant,
        "filename": final.name,
        "mmproj": None,
        "bytes": final.stat().st_size,
        "artifact_paths": {"gguf": str(zip_path)},
        "artifact_size": zip_path.stat().st_size,
        "note": "%s, %s (adapter)" % (final.name, _size(final)),
    }


def _readable_by_converter(folder: Path, shadow: Path) -> Path:
    """The model as the converter's own transformers can read it.

    The converter runs in its own environment on the transformers llama.cpp
    pins (4.x -- see the image), and every model here is saved by 5.x, which
    writes `extra_special_tokens` as a list where 4.x expects a mapping. The
    tokenizer then refuses to open ("'list' object has no attribute 'keys'")
    and no fine-tune of any model converts. The tokens it lists are declared
    in tokenizer.json anyway, so the field is dropped -- from a copy made of
    links, never from the cached model, which serving is reading.
    """
    cfg_path = folder / "tokenizer_config.json"
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return folder
    if not isinstance(cfg.get("extra_special_tokens"), list):
        return folder
    shadow.mkdir(parents=True, exist_ok=True)
    for item in folder.iterdir():
        link = shadow / item.name
        if item.name != "tokenizer_config.json" and not link.exists():
            link.symlink_to(item.resolve())
    cfg.pop("extra_special_tokens")
    (shadow / "tokenizer_config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    return shadow


def _is_vision(folder: Path) -> bool:
    try:
        cfg = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return bool(cfg.get("vision_config"))


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
