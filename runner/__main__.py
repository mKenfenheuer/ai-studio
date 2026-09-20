"""Entry point: ai-studio-runner --controller http://host:8420 --token XXX"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys


# How the caching allocator should grow, set before torch is imported anywhere.
#
# The default allocator carves VRAM into fixed segments and cannot hand a big
# request the free space lying in several small ones. At a long sequence length
# that is the difference between fitting and not: the attention scores of one
# layer are a single enormous allocation, and on a card with no fused kernel
# they arrive and leave every step, leaving the heap in exactly the shape that
# cannot satisfy the next one. `expandable_segments` lets a segment grow
# instead, so the same total free memory is actually usable.
#
# It changes no arithmetic and no result -- only whether an allocation that
# should have fitted does. Both spellings are set because the variable is named
# for the backend, and a runner does not know which one it has until torch is
# imported, which is after this.
_ALLOC_CONF = "expandable_segments:True"


def _tune_allocator() -> None:
    """Ask for a growable heap, unless this machine has already said otherwise.

    `setdefault`, not assignment: somebody who has tuned their own allocator
    for a reason we cannot see from here is not overruled by a default.
    """
    for name in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF"):
        os.environ.setdefault(name, _ALLOC_CONF)


def main() -> None:
    _tune_allocator()
    p = argparse.ArgumentParser(
        prog="ai-studio-runner",
        description="Join this machine's GPU to an AI Studio controller.")
    p.add_argument("--controller", default=os.environ.get("AI_STUDIO_CONTROLLER"),
                   help="Controller URL, e.g. http://studio.example.com:8420")
    p.add_argument("--token", default=os.environ.get("AI_STUDIO_JOIN_TOKEN"),
                   help="Join token shown in the controller UI under Runners.")
    p.add_argument("--name", default=os.environ.get("AI_STUDIO_RUNNER_NAME"),
                   help="Display name for this machine (defaults to hostname).")
    p.add_argument("--probe-only", action="store_true",
                   help="Print detected hardware capabilities and exit.")
    args = p.parse_args()

    if args.probe_only:
        from . import capabilities
        print(json.dumps(capabilities.probe(), indent=2))
        return

    if not args.controller or not args.token:
        p.error("--controller and --token are required "
                "(or set AI_STUDIO_CONTROLLER / AI_STUDIO_JOIN_TOKEN)")

    from .agent import Runner
    runner = Runner(args.controller, args.token, args.name)
    try:
        asyncio.run(runner.start())
    except KeyboardInterrupt:
        print("\n[runner] shutting down")
        sys.exit(0)


if __name__ == "__main__":
    main()
