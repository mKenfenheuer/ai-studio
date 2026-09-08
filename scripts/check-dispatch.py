#!/usr/bin/env python
"""Check who a run is allowed onto, and that asking does not break anything.

Run it with `python scripts/check-dispatch.py` from the repository root.

`Fleet.can_run` answers two different questions with one function, and that is
deliberate: "will this run on that machine" has to give the same answer when
the scheduler asks it about a queued run and when the wizard asks it about a
run that does not exist yet. The second caller has no job id -- and a line
that wrote the memory estimate back to the database by id turned every
fine-tune started from the wizard into a 500, because the wizard pins a
machine and so always takes this path.

The rest of it is the arithmetic that decides whether somebody's afternoon is
spent training or spent reading an error: a model that only fits compressed
must not be dispatched uncompressed, a machine with no 4-bit support must say
so rather than accept the work, and a machine with no card at all must not be
handed a 20B model because it could not measure itself.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_TMP = Path(tempfile.mkdtemp(prefix="ai-studio-check-"))
os.environ["AI_STUDIO_DATA"] = str(_TMP)

from controller.scheduler import Fleet  # noqa: E402

FAILED: list[str] = []

GPU = {"_id": "run_gpu", "backend": "cuda", "vram_gb": 24.0,
       "quantization": {"4bit": True}, "modalities": ["text", "vision"]}
NO_4BIT = {"_id": "run_old", "backend": "cuda", "vram_gb": 24.0,
           "quantization": {"4bit": False}}
CPU = {"_id": "run_cpu", "backend": "cpu", "quantization": {"4bit": False},
       "modalities": ["text", "vision"]}


def check(name: str, got: object, want: object = True) -> None:
    ok = got == want
    print("  %s  %s%s" % ("ok  " if ok else "FAIL", name,
                          "" if ok else "   -> %r, wanted %r" % (got, want)))
    if not ok:
        FAILED.append(name)


def job(**cfg: object) -> dict:
    """A run as the wizard asks about it: a config, a kind, and no id yet."""
    return {"kind": cfg.pop("kind", "finetune_llm"), "config": dict(cfg)}


def main() -> int:
    fleet = Fleet()
    try:
        print("\nAsking about a run that does not exist yet")
        j = job(base_model="mistralai/Mistral-7B-Instruct-v0.3", params_b=7.0,
                required_runner="run_gpu")
        ok, why = fleet.can_run(j, GPU)
        check("a 7B in 16-bit fits a 24 GB card", (ok, why), (True, ""))
        check("and the estimate is left on the config the caller holds",
              j["config"].get("estimated_vram_gb"), 18.9)

        j4 = job(base_model="openai/gpt-oss-20b", params_b=20.0,
                 quantization="4bit", required_runner="run_gpu")
        fleet.can_run(j4, GPU)
        check("a 4-bit run is estimated in 4-bit, not in 16-bit",
              j4["config"].get("estimated_vram_gb"), 13.5)

        print("\nPrecision")
        ok, why = fleet.can_run(
            job(base_model="openai/gpt-oss-20b", params_b=20.0), GPU)
        check("a 20B in 16-bit is refused on 24 GB", ok, False)
        check("and the refusal says what to change", "4-bit" in why)
        ok, _ = fleet.can_run(
            job(base_model="openai/gpt-oss-20b", params_b=20.0,
                quantization="4bit"), GPU)
        check("the same 20B in 4-bit is accepted", ok, True)
        ok, why = fleet.can_run(
            job(base_model="openai/gpt-oss-20b", params_b=20.0,
                quantization="4bit"), NO_4BIT)
        check("a machine without 4-bit refuses 4-bit work", ok, False)
        check("saying so plainly", "4-bit" in why)

        print("\nMachines that cannot do the work")
        ok, why = fleet.can_run(job(base_model="x", params_b=7.0), CPU)
        check("no card, no training", (ok, why), (False, "runner has no GPU"))
        ok, _ = fleet.can_run(
            job(base_model="x", params_b=7.0, allow_cpu=True), CPU)
        check("unless the run says it does not mind", ok, True)
        ok, why = fleet.can_run(
            job(kind="finetune_vision_cls", base_model="x", allow_cpu=True),
            {**CPU, "modalities": ["text"]})
        check("a vision run needs an image library", ok, False)
        check("and is told which", "image" in why)

        print("\nPinning")
        ok, why = fleet.can_run(
            job(base_model="x", params_b=1.0, required_runner="run_other"), GPU)
        check("a run pinned elsewhere stays there",
              (ok, why), (False, "pinned to a different runner"))
        ok, _ = fleet.can_run(
            job(kind="upload", base_model="x", allow_cpu=True),
            {**CPU, "kinds": ["upload", "evaluate"]})
        check("a machine set aside for uploads takes an upload", ok, True)
        ok, why = fleet.can_run(
            job(kind="finetune_llm", base_model="x", allow_cpu=True),
            {**CPU, "kinds": ["upload", "evaluate"]})
        check("and refuses a fine-tune", ok, False)
        check("naming what it does take", "upload" in why)
    finally:
        shutil.rmtree(_TMP, ignore_errors=True)

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
