"""Entry point: ai-studio-runner --controller http://host:8420 --token XXX"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys


def main() -> None:
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
