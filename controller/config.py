"""Controller configuration. Everything is env-overridable so the same image
runs on a laptop, a server, or inside docker-compose without edits."""
from __future__ import annotations

import os
import secrets
from pathlib import Path


def _env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser().resolve()


# Where the studio keeps its state. Mount this as a volume in Docker.
DATA_DIR: Path = _env_path("AI_STUDIO_DATA", "./data")
ARTIFACT_DIR: Path = DATA_DIR / "artifacts"
DB_PATH: Path = DATA_DIR / "studio.db"

HOST: str = os.environ.get("AI_STUDIO_HOST", "0.0.0.0")
PORT: int = int(os.environ.get("AI_STUDIO_PORT", "8420"))

# Shared secret a runner must present to join the fleet. Generated and
# persisted on first boot so a fresh install is secure by default rather than
# wide open; surfaced in the UI so the user can copy the join command.
_TOKEN_FILE = DATA_DIR / "join_token"


def join_token() -> str:
    if tok := os.environ.get("AI_STUDIO_JOIN_TOKEN"):
        return tok.strip()
    if _TOKEN_FILE.exists():
        return _TOKEN_FILE.read_text(encoding="utf-8").strip()
    tok = secrets.token_urlsafe(24)
    _TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    _TOKEN_FILE.write_text(tok, encoding="utf-8")
    _TOKEN_FILE.chmod(0o600)
    return tok


# The address people actually type. Only needed behind a reverse proxy or a
# tunnel, and only by single sign-on -- a provider matches the redirect URI it
# was given character for character, and behind a proxy the URL the
# application sees is the internal one, which matches nothing anybody
# registered. Left unset, the forwarded headers are used, then the request.
PUBLIC_URL: str | None = (os.environ.get("AI_STUDIO_PUBLIC_URL") or "").strip() or None

# Optional Hugging Face token, forwarded to runners so they can pull gated
# models (Llama, Gemma) and get better rate limits.
HF_TOKEN: str | None = os.environ.get("HF_TOKEN") or None

# A runner that has not sent a heartbeat within this window is shown as lost.
HEARTBEAT_TIMEOUT_S: int = int(os.environ.get("AI_STUDIO_HEARTBEAT_TIMEOUT", "45"))

WEB_DIR: Path = Path(__file__).resolve().parent.parent / "web"

# The longest reply anybody may ask for, in tokens. A ceiling on what a caller
# can request rather than a target: generation here is one forward pass per
# token, so a long limit is a long wait, and the Stop button is what a reader
# uses when they have seen enough. Memory is not the binding constraint -- the
# key/value cache of a 7B costs about an eighth of a megabyte per token, so
# even the whole of this is a fraction of what serving the weights already
# takes.
MAX_NEW_TOKENS: int = int(os.environ.get("AI_STUDIO_MAX_NEW_TOKENS") or 4096)

# How long one reply may take before the runner stops and hands back what it
# has. A runner serves one message at a time, so this bounds how long everybody
# else is refused -- not how good the reply is. Kept under the API's own wait
# below, so the runner gives up first and returns a real (if short) answer
# rather than the caller timing out on a machine still working.
GENERATION_DEADLINE_S: float = float(
    os.environ.get("AI_STUDIO_GENERATION_DEADLINE_S") or 300)


def ensure_dirs() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
