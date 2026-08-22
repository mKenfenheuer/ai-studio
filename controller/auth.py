"""Accounts, sessions, and the secrets a user entrusts to the studio.

Until now the studio had exactly one credential -- the join token a machine
presents -- and everything reachable from a browser was open to whoever could
reach the port. That was defensible while the only thing behind it was your
own GPU on your own network. It stops being defensible the moment the studio
holds *your Hugging Face token*, because then a browser session can act as you
on someone else's service.

So the rules here are deliberately narrow:

* **Passwords are never stored.** scrypt with a per-user salt, from the
  standard library, so this adds no dependency and no home-made hashing.

* **Session tokens are never stored either.** The cookie holds a random value;
  the database holds its SHA-256. Someone who reads the database cannot use
  what they find to log in.

* **The Hugging Face token is encrypted at rest, and never leaves the
  server.** No endpoint returns it, not even to its owner -- there is nothing
  a UI needs it for. Be clear about what the encryption is worth: the key sits
  in a file next to the database, so it protects a copied database file, a
  backup, or a screenshot of a table. It does not protect against someone who
  already has the host.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import re
import secrets
import time
from typing import Any

from . import config, db

# scrypt cost. n=2^15 with r=8 needs 32 MB and about a tenth of a second per
# check on a modern core -- slow enough to make an offline guessing attack
# expensive, fast enough that nobody notices logging in.
_SCRYPT_N = 1 << 15
_SCRYPT_R = 8
_SCRYPT_P = 1
_DKLEN = 32

SESSION_COOKIE = "ais_session"
SESSION_TTL_S = 30 * 24 * 3600          # a month; refreshed on use
SESSION_IDLE_S = 14 * 24 * 3600         # ...but not if untouched for a fortnight

MIN_PASSWORD = 10
USERNAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{1,31}$")


# ---------------------------------------------------------------------------
# Passwords
# ---------------------------------------------------------------------------

def _maxmem(n: int, r: int) -> int:
    """The memory ceiling to hand OpenSSL for these parameters.

    Not optional. scrypt needs 128 * n * r bytes, which at the settings above
    is exactly 32 MB -- and OpenSSL's default ceiling is also 32 MB, so it
    refuses with "memory limit exceeded" before doing any work. Left at the
    default this function raises on every call, which is a login screen that
    can never succeed.
    """
    return 128 * n * r * 2


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=_SCRYPT_N,
                        r=_SCRYPT_R, p=_SCRYPT_P, dklen=_DKLEN,
                        maxmem=_maxmem(_SCRYPT_N, _SCRYPT_R))
    return "scrypt$%d$%d$%d$%s$%s" % (
        _SCRYPT_N, _SCRYPT_R, _SCRYPT_P,
        base64.b64encode(salt).decode(), base64.b64encode(dk).decode())


def verify_password(password: str, stored: str) -> bool:
    """Constant-time check that also survives a stored value it cannot parse."""
    try:
        scheme, n, r, p, salt_b64, dk_b64 = stored.split("$")
        if scheme != "scrypt":
            return False
        # Parameters come from the stored hash, so an old password still
        # verifies after the cost settings above are raised.
        dk = hashlib.scrypt(password.encode("utf-8"),
                            salt=base64.b64decode(salt_b64),
                            n=int(n), r=int(r), p=int(p),
                            dklen=len(base64.b64decode(dk_b64)),
                            maxmem=_maxmem(int(n), int(r)))
    except (ValueError, TypeError, MemoryError):
        return False
    return hmac.compare_digest(dk, base64.b64decode(dk_b64))


def password_problem(password: str) -> str | None:
    """Why this password is not acceptable, in words a person can act on.

    Deliberately short: a length floor and a check against the handful of
    passwords that actually get tried first. Composition rules ("one capital,
    one symbol") push people toward Password1! and buy nothing.
    """
    if len(password or "") < MIN_PASSWORD:
        return ("Use at least %d characters. Length is what makes a password "
                "hard to guess -- a short one with symbols in it is not."
                % MIN_PASSWORD)
    common = {"password", "password123", "12345678", "123456789", "1234567890",
              "qwertyuiop", "letmein123", "adminadmin", "aistudio", "changeme",
              "iloveyou", "welcome123", "administrator"}
    if password.lower().strip() in common:
        return "That is one of the first passwords anyone would try."
    if len(set(password)) <= 3:
        return "That is too repetitive to be hard to guess."
    return None


def username_problem(username: str) -> str | None:
    if not USERNAME_RE.match(username or ""):
        return ("Usernames are 2-32 characters, lower case, and may contain "
                "letters, numbers, dots, dashes and underscores.")
    return None


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

def _token_hash(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def start_session(user_id: str, user_agent: str = "") -> str:
    raw = secrets.token_urlsafe(32)
    now = time.time()
    db.ex("INSERT INTO sessions (id,user_id,created_at,last_used,expires_at,"
          "user_agent) VALUES (?,?,?,?,?,?)",
          (_token_hash(raw), user_id, now, now, now + SESSION_TTL_S,
           (user_agent or "")[:200]))
    return raw


def session_user(raw: str | None) -> dict | None:
    """The account this cookie belongs to, or None. Also renews it."""
    if not raw:
        return None
    row = db.q1("SELECT * FROM sessions WHERE id=?", (_token_hash(raw),))
    if not row:
        return None
    now = time.time()
    if now > row["expires_at"] or now - row["last_used"] > SESSION_IDLE_S:
        db.ex("DELETE FROM sessions WHERE id=?", (row["id"],))
        return None
    user = db.get_user(row["user_id"])
    if not user or not user["active"]:
        # Deactivating an account has to end its sessions, or "disabled" means
        # "disabled at the next login" -- which is not what anyone reads it as.
        db.ex("DELETE FROM sessions WHERE user_id=?", (row["user_id"],))
        return None
    # Written at most once a minute: every API call would otherwise be a write,
    # and the UI makes several a second while a run streams.
    if now - row["last_used"] > 60:
        db.ex("UPDATE sessions SET last_used=? WHERE id=?", (now, row["id"]))
    return user


def end_session(raw: str | None) -> None:
    if raw:
        db.ex("DELETE FROM sessions WHERE id=?", (_token_hash(raw),))


def end_all_sessions(user_id: str, keep: str | None = None) -> int:
    rows = db.q("SELECT id FROM sessions WHERE user_id=?", (user_id,))
    keep_hash = _token_hash(keep) if keep else None
    n = 0
    for r in rows:
        if r["id"] != keep_hash:
            db.ex("DELETE FROM sessions WHERE id=?", (r["id"],))
            n += 1
    return n


def purge_expired_sessions() -> None:
    db.ex("DELETE FROM sessions WHERE expires_at < ?", (time.time(),))


# ---------------------------------------------------------------------------
# API keys
# ---------------------------------------------------------------------------
#
# Hashed, not encrypted. There is never a reason to read a key back -- it is
# checked against, never displayed -- so the weaker of the two options is also
# the wrong one. A plain SHA-256 rather than scrypt because unlike a password
# this is 32 bytes of randomness: there is no dictionary to try it against,
# and a per-call scrypt would make the API slower than the model.

API_KEY_PREFIX = "sk-ais-"


def new_api_key() -> tuple[str, str, str]:
    """A fresh key, its hash, and the part that may be shown afterwards."""
    raw = API_KEY_PREFIX + secrets.token_urlsafe(32)
    return raw, _token_hash(raw), raw[:len(API_KEY_PREFIX) + 6]


def api_key_hash(raw: str) -> str:
    return _token_hash(raw)


# ---------------------------------------------------------------------------
# Secrets at rest
# ---------------------------------------------------------------------------

_KEY_FILE = config.DATA_DIR / "secret_key"
_fernet: Any = None


def _key() -> bytes:
    if _KEY_FILE.exists():
        return _KEY_FILE.read_bytes().strip()
    key = base64.urlsafe_b64encode(os.urandom(32))
    _KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
    _KEY_FILE.write_bytes(key)
    try:
        _KEY_FILE.chmod(0o600)
    except OSError:
        # Windows and some mounted volumes do not implement this. Worth
        # attempting and not worth failing over.
        pass
    return key


def _cipher():
    global _fernet
    if _fernet is None:
        from cryptography.fernet import Fernet
        _fernet = Fernet(_key())
    return _fernet


def encrypt_secret(value: str) -> str:
    return _cipher().encrypt(value.encode("utf-8")).decode("ascii")


def decrypt_secret(value: str | None) -> str | None:
    """Plaintext, or None if it cannot be read.

    Returning None rather than raising is deliberate: a key file that was lost
    or replaced should mean "this user needs to reconnect their account", not
    "every page 500s".
    """
    if not value:
        return None
    try:
        from cryptography.fernet import InvalidToken
        try:
            return _cipher().decrypt(value.encode("ascii")).decode("utf-8")
        except InvalidToken:
            return None
    except Exception:  # noqa: BLE001 - a missing library must not break login
        return None


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------
#
# In memory, per username+address, and forgotten on restart. A studio on a home
# network does not need a distributed counter; it needs the difference between
# "somebody can try ten passwords" and "somebody can try ten million".

_failures: dict[str, list[float]] = {}
_LOCK_AFTER = 8
_LOCK_WINDOW_S = 300


def note_failure(key: str) -> None:
    now = time.time()
    hits = [t for t in _failures.get(key, []) if now - t < _LOCK_WINDOW_S]
    hits.append(now)
    _failures[key] = hits


def locked_out(key: str) -> int:
    """Seconds left before this key may try again; 0 when it may try now."""
    now = time.time()
    hits = [t for t in _failures.get(key, []) if now - t < _LOCK_WINDOW_S]
    _failures[key] = hits
    if len(hits) < _LOCK_AFTER:
        return 0
    return int(_LOCK_WINDOW_S - (now - hits[0])) + 1


def clear_failures(key: str) -> None:
    _failures.pop(key, None)
