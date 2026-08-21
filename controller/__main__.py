"""Entry point: ai-studio [--host H] [--port P]"""
from __future__ import annotations

import argparse

import uvicorn

from . import config


def main() -> None:
    p = argparse.ArgumentParser(prog="ai-studio", description="AI Studio controller")
    p.add_argument("--host", default=config.HOST)
    p.add_argument("--port", type=int, default=config.PORT)
    p.add_argument("--reload", action="store_true")
    args = p.parse_args()

    config.ensure_dirs()
    token = config.join_token()
    print("=" * 66)
    print("  AI Studio  ->  http://%s:%s" % (
        "localhost" if args.host in ("0.0.0.0", "::") else args.host, args.port))
    print("  Join token :  %s" % token)
    print("=" * 66)

    uvicorn.run("controller.app:app", host=args.host, port=args.port,
                reload=args.reload, log_level="info")


if __name__ == "__main__":
    main()
