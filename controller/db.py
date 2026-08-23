"""SQLite persistence.

Plain sqlite3 rather than an ORM: the controller must stay installable with no
compiler and no heavyweight dependencies, and the schema here is small enough
that an ORM would cost more than it saves.
"""
from __future__ import annotations

import json
import re
import sqlite3
import time
import uuid
from typing import Any, Iterable

from . import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runners (
    id            TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    capabilities  TEXT NOT NULL,
    status        TEXT NOT NULL,
    last_seen     REAL NOT NULL,
    first_seen    REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    id            TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    kind          TEXT NOT NULL,
    status        TEXT NOT NULL,
    config        TEXT NOT NULL,
    runner_id     TEXT,
    created_at    REAL NOT NULL,
    started_at    REAL,
    finished_at   REAL,
    step          INTEGER NOT NULL DEFAULT 0,
    total_steps   INTEGER NOT NULL DEFAULT 0,
    error         TEXT
);

CREATE TABLE IF NOT EXISTS metrics (
    job_id  TEXT NOT NULL,
    step    INTEGER NOT NULL,
    ts      REAL NOT NULL,
    data    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_metrics_job ON metrics(job_id, step);

CREATE TABLE IF NOT EXISTS logs (
    job_id  TEXT NOT NULL,
    ts      REAL NOT NULL,
    level   TEXT NOT NULL,
    line    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_logs_job ON logs(job_id, ts);

CREATE TABLE IF NOT EXISTS artifacts (
    id          TEXT PRIMARY KEY,
    job_id      TEXT NOT NULL,
    kind        TEXT NOT NULL,
    filename    TEXT NOT NULL,
    size_bytes  INTEGER NOT NULL,
    created_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS users (
    id             TEXT PRIMARY KEY,
    username       TEXT NOT NULL UNIQUE,
    display_name   TEXT NOT NULL,
    password_hash  TEXT NOT NULL,
    role           TEXT NOT NULL DEFAULT 'member',
    active         INTEGER NOT NULL DEFAULT 1,
    created_at     REAL NOT NULL,
    last_login     REAL,
    must_change    INTEGER NOT NULL DEFAULT 0,
    -- Encrypted, and never returned by any endpoint. See controller/auth.
    hf_token_enc   TEXT,
    hf_username    TEXT,
    hf_fullname    TEXT,
    hf_avatar      TEXT,
    hf_orgs        TEXT,
    hf_can_write   INTEGER NOT NULL DEFAULT 0,
    hf_checked_at  REAL
);

-- The cookie's SHA-256, not the cookie. A leaked database is not a set of
-- working logins.
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    created_at  REAL NOT NULL,
    last_used   REAL NOT NULL,
    expires_at  REAL NOT NULL,
    user_agent  TEXT
);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);

CREATE TABLE IF NOT EXISTS datasets (
    id          TEXT PRIMARY KEY,
    owner_id    TEXT,
    name        TEXT NOT NULL,
    source      TEXT NOT NULL,
    origin      TEXT,
    rows        INTEGER NOT NULL DEFAULT 0,
    bytes       INTEGER NOT NULL DEFAULT 0,
    columns     TEXT,
    format      TEXT,
    notes       TEXT,
    parent_id   TEXT,
    recipe      TEXT,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_datasets_owner ON datasets(owner_id);

-- Sharing. A run or a dataset belongs to the person who made it and is
-- invisible to everyone else until they say otherwise; this table is how they
-- say otherwise. `subject_type='everyone'` means everyone with an account on
-- this studio, which is a different and much smaller claim than "public".
CREATE TABLE IF NOT EXISTS shares (
    id             TEXT PRIMARY KEY,
    resource_type  TEXT NOT NULL,
    resource_id    TEXT NOT NULL,
    subject_type   TEXT NOT NULL,
    subject_id     TEXT,
    level          TEXT NOT NULL DEFAULT 'view',
    created_at     REAL NOT NULL,
    created_by     TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_shares_unique
    ON shares(resource_type, resource_id, subject_type, IFNULL(subject_id,''));
CREATE INDEX IF NOT EXISTS idx_shares_subject ON shares(subject_type, subject_id);

-- A saved set of prompts, kept so that "is this week's model better than last
-- week's" has an answer that does not depend on remembering what you typed.
-- The prompts are the fixed part of the experiment; the models change.
CREATE TABLE IF NOT EXISTS evals (
    id          TEXT PRIMARY KEY,
    owner_id    TEXT,
    name        TEXT NOT NULL,
    notes       TEXT,
    items       TEXT NOT NULL,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_evals_owner ON evals(owner_id);

-- One row per (prompt set, model) scoring. Kept rather than derived from the
-- evaluation run's artifact, because the comparison across weeks is the whole
-- point and it has to survive the run that produced it being deleted.
CREATE TABLE IF NOT EXISTS eval_scores (
    id            TEXT PRIMARY KEY,
    eval_id       TEXT NOT NULL,
    model_job_id  TEXT NOT NULL,
    run_job_id    TEXT,
    created_at    REAL NOT NULL,
    metrics       TEXT NOT NULL,
    items         TEXT
);
CREATE INDEX IF NOT EXISTS idx_eval_scores ON eval_scores(eval_id, created_at);

-- Keys for the OpenAI-compatible API. The key itself is never stored, only
-- its SHA-256 -- the same rule sessions follow, for the same reason: reading
-- this table must not hand anybody a working credential. `prefix` is the
-- handful of visible characters that let a person tell two of their own keys
-- apart without either of them being recoverable.
-- An external identity provider: Entra ID, Google, Okta, Keycloak, anything
-- that speaks OpenID Connect. Rows here are configuration, not credentials of
-- this studio's own -- except `client_secret_enc`, which is encrypted at rest
-- for the same reason the Hugging Face token is: whoever holds it can ask the
-- provider for tokens as this application.
CREATE TABLE IF NOT EXISTS idp (
    id             TEXT PRIMARY KEY,
    name           TEXT NOT NULL,
    kind           TEXT NOT NULL,          -- entra | google | okta | oidc ...
    issuer         TEXT NOT NULL,
    client_id      TEXT NOT NULL,
    client_secret_enc TEXT,
    tenant         TEXT,                   -- Entra tenant id, Google domain
    enabled        INTEGER NOT NULL DEFAULT 1,
    auto_create    INTEGER NOT NULL DEFAULT 1,
    link_by_email  INTEGER NOT NULL DEFAULT 1,
    allowed_domains TEXT,                  -- comma separated; empty = any
    admin_groups   TEXT,                   -- comma separated group names/ids
    scopes         TEXT,
    -- The discovery document, cached. Kept so a provider that is briefly
    -- unreachable does not take the login button down with it.
    discovery      TEXT,
    discovered_at  REAL,
    sync_enabled   INTEGER NOT NULL DEFAULT 0,
    sync_group     TEXT,                   -- only import members of this group
    sync_subject   TEXT,                   -- Google: the admin to read as
    sync_secret_enc TEXT,                  -- Graph app secret / Google SA key
    last_sync_at   REAL,
    last_sync_note TEXT,
    created_at     REAL NOT NULL,
    created_by     TEXT
);

-- One in-flight sign-in. Holds the state, the nonce and the PKCE verifier
-- server-side rather than in a cookie, so none of the three can be read or
-- replayed from the browser, and so a callback that arrives without a
-- matching row is refused instead of trusted.
CREATE TABLE IF NOT EXISTS oauth_states (
    state        TEXT PRIMARY KEY,
    idp_id       TEXT NOT NULL,
    nonce        TEXT NOT NULL,
    verifier     TEXT NOT NULL,
    redirect_uri TEXT NOT NULL,
    next_url     TEXT,
    created_at   REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS api_keys (
    id          TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    name        TEXT NOT NULL,
    key_hash    TEXT NOT NULL UNIQUE,
    prefix      TEXT NOT NULL,
    created_at  REAL NOT NULL,
    last_used   REAL,
    calls       INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_api_keys_user ON api_keys(user_id);
"""

# Columns added after the first release. SQLite has no "ADD COLUMN IF NOT
# EXISTS", and an existing studio must not lose its runs to an upgrade, so
# each one is attempted and its "duplicate column" complaint ignored.
_ADDED_COLUMNS = [
    ("jobs", "owner_id", "TEXT"),
    # What the run reported when it ended. It was already written into the log
    # as JSON, which was fine for reading one run and useless for comparing
    # twenty: answering "which of these had the lowest held-out loss" meant
    # parsing prose out of a log table.
    ("jobs", "summary", "TEXT"),
    # The furthest step a checkpoint exists for, and the machine holding it.
    # Persisted rather than kept in memory so that a controller restart does
    # not lose the one fact that decides whether a run resumes or starts over.
    ("jobs", "checkpoint_step", "INTEGER NOT NULL DEFAULT 0"),
    ("jobs", "checkpoint_runner", "TEXT"),
    # A webhook URL is a credential -- whoever holds a Slack or ntfy one can
    # post as you -- so it is encrypted at rest like the Hugging Face token,
    # and no endpoint returns it in full.
    ("users", "notify_url_enc", "TEXT"),
    ("users", "notify_events", "TEXT"),
    # Keys for hosted models -- OpenAI, Azure, Anthropic, anything that speaks
    # their shape. One encrypted JSON object per user rather than a column per
    # provider, because the set of providers is a list in code and adding one
    # should not mean migrating the database. Encrypted like the Hugging Face
    # token, for the same reason: it is somebody's billable credential.
    ("users", "model_providers_enc", "TEXT"),
    # {"train": 9000, "validation": 1000}. A dataset holds its splits in a
    # reserved column on each row; this is the tally, taken once when the file
    # is written so that no page has to count two million rows to draw a
    # badge. NULL means a dataset from before splits, which is all one split.
    ("datasets", "splits", "TEXT"),
    # Where this account comes from. NULL means a password on this studio;
    # anything else is the id of a row in `idp`, and the account signs in
    # through that provider instead.
    ("users", "provider", "TEXT"),
    ("users", "external_id", "TEXT"),
    ("users", "email", "TEXT"),
    ("users", "avatar_url", "TEXT"),
    ("users", "job_title", "TEXT"),
    ("users", "department", "TEXT"),
    ("users", "groups", "TEXT"),
    ("users", "synced_at", "REAL"),
    # Imported from a directory and has never signed in. Such an account can
    # be found, shared with and mentioned -- it simply has nothing of its own
    # yet. The flag clears on first sign-in.
    ("users", "pending", "INTEGER NOT NULL DEFAULT 0"),
]

_ADDED_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_users_external"
    " ON users(provider, external_id)",
    "CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)",
]

_conn: sqlite3.Connection | None = None


def connect() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(config.DB_PATH, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        # WAL lets the UI poll while a job streams metrics in, without blocking.
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute("PRAGMA synchronous=NORMAL")
        _conn.executescript(_SCHEMA)
        for table, column, decl in _ADDED_COLUMNS:
            try:
                _conn.execute("ALTER TABLE %s ADD COLUMN %s %s"
                              % (table, column, decl))
            except sqlite3.OperationalError as e:
                if "duplicate column" not in str(e).lower():
                    raise
        # Indexes last. An index over a column that only exists because of
        # the migration above cannot be created before that migration has run.
        for stmt in _ADDED_INDEXES:
            _conn.execute(stmt)
        _conn.commit()
    return _conn


def q(sql: str, args: Iterable[Any] = ()) -> list[dict[str, Any]]:
    cur = connect().execute(sql, tuple(args))
    return [dict(r) for r in cur.fetchall()]


def q1(sql: str, args: Iterable[Any] = ()) -> dict[str, Any] | None:
    rows = q(sql, args)
    return rows[0] if rows else None


def ex(sql: str, args: Iterable[Any] = ()) -> None:
    c = connect()
    c.execute(sql, tuple(args))
    c.commit()


def new_id(prefix: str) -> str:
    return prefix + "_" + uuid.uuid4().hex[:12]


def now() -> float:
    return time.time()


# ---------------------------------------------------------------- runners

def upsert_runner(runner_id: str, name: str, capabilities: dict) -> None:
    if q1("SELECT id FROM runners WHERE id=?", (runner_id,)):
        ex("UPDATE runners SET name=?, capabilities=?, status='online', last_seen=? WHERE id=?",
           (name, json.dumps(capabilities), now(), runner_id))
    else:
        ex("INSERT INTO runners (id,name,capabilities,status,last_seen,first_seen)"
           " VALUES (?,?,?,'online',?,?)",
           (runner_id, name, json.dumps(capabilities), now(), now()))


def touch_runner(runner_id: str, status: str | None = None) -> None:
    if status:
        ex("UPDATE runners SET last_seen=?, status=? WHERE id=?", (now(), status, runner_id))
    else:
        ex("UPDATE runners SET last_seen=? WHERE id=?", (now(), runner_id))


def mark_runner_offline(runner_id: str) -> None:
    ex("UPDATE runners SET status='offline' WHERE id=?", (runner_id,))


def list_runners() -> list[dict]:
    rows = q("SELECT * FROM runners ORDER BY first_seen")
    cutoff = now() - config.HEARTBEAT_TIMEOUT_S
    for r in rows:
        r["capabilities"] = json.loads(r["capabilities"])
        # Report a silent runner as offline even if it never closed its socket
        # cleanly, which is what a crash, power loss or network drop looks like.
        if r["status"] != "offline" and r["last_seen"] < cutoff:
            r["status"] = "offline"
    return rows


def get_runner(runner_id: str) -> dict | None:
    r = q1("SELECT * FROM runners WHERE id=?", (runner_id,))
    if r:
        r["capabilities"] = json.loads(r["capabilities"])
    return r


# ------------------------------------------------------------------- jobs

def create_job(name: str, kind: str, cfg: dict, owner_id: str | None = None) -> str:
    jid = new_id("job")
    ex("INSERT INTO jobs (id,name,kind,status,config,created_at,owner_id)"
       " VALUES (?,?,?,'queued',?,?,?)",
       (jid, name, kind, json.dumps(cfg), now(), owner_id))
    return jid


def _hydrate(r: dict) -> dict:
    r["config"] = json.loads(r["config"])
    if r.get("summary"):
        try:
            r["summary"] = json.loads(r["summary"])
        except (TypeError, ValueError):
            r["summary"] = None
    return r


def list_jobs(limit: int = 100) -> list[dict]:
    """Recent runs, each flagged with whether a model came out of it.

    `has_model` rather than the status, because a run that was stopped early
    and kept its model has one too. Anything offering to open, play with or
    download a result has to ask that question, and asking it per row against
    the artifacts table would be one query per job.
    """
    rows = q("SELECT j.*, EXISTS(SELECT 1 FROM artifacts a WHERE a.job_id = j.id)"
             " AS has_model, u.display_name AS owner_name, u.username AS owner_username"
             " FROM jobs j LEFT JOIN users u ON u.id = j.owner_id"
             " ORDER BY j.created_at DESC LIMIT ?",
             (limit,))
    for r in rows:
        r["has_model"] = bool(r["has_model"])
    return [_hydrate(r) for r in rows]


def get_job(job_id: str) -> dict | None:
    r = q1("SELECT * FROM jobs WHERE id=?", (job_id,))
    return _hydrate(r) if r else None


def queued_jobs() -> list[dict]:
    return [_hydrate(r) for r in
            q("SELECT * FROM jobs WHERE status='queued' ORDER BY created_at")]


def assign_job(job_id: str, runner_id: str) -> None:
    ex("UPDATE jobs SET status='assigned', runner_id=? WHERE id=?", (runner_id, job_id))


def set_job_status(job_id: str, status: str, error: str | None = None) -> None:
    if status == "running":
        ex("UPDATE jobs SET status=?, started_at=COALESCE(started_at,?) WHERE id=?",
           (status, now(), job_id))
    elif status in ("succeeded", "failed", "cancelled"):
        ex("UPDATE jobs SET status=?, finished_at=?, error=? WHERE id=?",
           (status, now(), error, job_id))
    else:
        ex("UPDATE jobs SET status=? WHERE id=?", (status, job_id))


def set_job_progress(job_id: str, step: int, total: int) -> None:
    ex("UPDATE jobs SET step=?, total_steps=? WHERE id=?", (step, total, job_id))


def assign_job_runner(job_id: str, runner_id: str) -> None:
    ex("UPDATE jobs SET runner_id=? WHERE id=?", (runner_id, job_id))


def orphaned_jobs() -> list[dict]:
    """Jobs whose machine has been silent past the heartbeat deadline."""
    cutoff = now() - config.HEARTBEAT_TIMEOUT_S
    return [_hydrate(r) for r in q(
        "SELECT j.* FROM jobs j JOIN runners r ON r.id = j.runner_id"
        " WHERE j.status IN ('assigned','running') AND r.last_seen < ?",
        (cutoff,))]


def requeue_jobs_for_runner(runner_id: str,
                            except_job: str | None = None) -> tuple[list[str], list[str]]:
    """A runner is no longer running work the controller thinks it is running.

    Returns (requeued, rescued).

    The distinction matters more than it looks. A job that already uploaded an
    artifact has *finished* -- the runner died in the gap between the upload
    landing and the "done" message being processed. Requeuing that job throws
    away a completed model and starts a twenty-minute run again from noise,
    which is the worst possible response to work that already succeeded.

    `except_job` is the run the machine says it *is* working on, which must
    never be touched. Without it, a runner that reconnects mid-training would
    have its own live job requeued underneath it.
    """
    rows = q("SELECT id FROM jobs WHERE runner_id=? AND status IN ('assigned','running')",
             (runner_id,))
    requeued, rescued = [], []
    for r in rows:
        jid = r["id"]
        if except_job and jid == except_job:
            continue
        if q1("SELECT id FROM artifacts WHERE job_id=? LIMIT 1", (jid,)):
            ex("UPDATE jobs SET status='succeeded', finished_at=COALESCE(finished_at,?)"
               " WHERE id=?", (now(), jid))
            rescued.append(jid)
        else:
            # step goes back to wherever a checkpoint exists, not to zero. The
            # runner that holds it keeps its name on the row so the scheduler
            # can send the work back to the one machine that can carry on
            # rather than to whichever is free first.
            ex("UPDATE jobs SET status='queued', runner_id=NULL,"
               " step=COALESCE(checkpoint_step,0) WHERE id=?", (jid,))
            requeued.append(jid)
    return requeued, rescued


def set_job_summary(job_id: str, summary: dict) -> None:
    ex("UPDATE jobs SET summary=? WHERE id=?", (json.dumps(summary), job_id))


def set_checkpoint(job_id: str, step: int, runner_id: str | None) -> None:
    ex("UPDATE jobs SET checkpoint_step=?, checkpoint_runner=? WHERE id=?",
       (int(step), runner_id, job_id))


def clear_checkpoint(job_id: str) -> None:
    ex("UPDATE jobs SET checkpoint_step=0, checkpoint_runner=NULL WHERE id=?",
       (job_id,))


def jobs_with_checkpoint_on(runner_id: str) -> list[str]:
    return [r["id"] for r in q(
        "SELECT id FROM jobs WHERE checkpoint_runner=? AND checkpoint_step>0",
        (runner_id,))]


def gpu_seconds_by_owner(window_s: float = 86400.0) -> dict[str | None, float]:
    """How much machine time each person has had lately.

    Counts only the part of each run that falls inside the window, and counts a
    run still going as running up to now -- otherwise somebody eight hours into
    an overnight job registers as having used nothing at all, which is the
    exact case fair queueing exists for.
    """
    cutoff = now() - window_s
    out: dict[str | None, float] = {}
    rows = q("SELECT owner_id, started_at, finished_at FROM jobs"
             " WHERE started_at IS NOT NULL"
             "   AND COALESCE(finished_at, ?) > ?", (now(), cutoff))
    for r in rows:
        start = max(float(r["started_at"]), cutoff)
        end = float(r["finished_at"] or now())
        out[r["owner_id"]] = out.get(r["owner_id"], 0.0) + max(0.0, end - start)
    return out


def running_by_owner() -> dict[str | None, int]:
    rows = q("SELECT owner_id, COUNT(*) AS n FROM jobs"
             " WHERE status IN ('assigned','running') GROUP BY owner_id")
    return {r["owner_id"]: int(r["n"]) for r in rows}


# ------------------------------------------------- metrics / logs / files

def add_metric(job_id: str, step: int, data: dict) -> None:
    ex("INSERT INTO metrics (job_id,step,ts,data) VALUES (?,?,?,?)",
       (job_id, step, now(), json.dumps(data)))


def count_metrics(job_id: str) -> int:
    return len(q("SELECT 1 FROM metrics WHERE job_id=? LIMIT 1", (job_id,)))


def clear_metrics(job_id: str) -> None:
    """Drop the measurements of an attempt that is being redone."""
    ex("DELETE FROM metrics WHERE job_id=?", (job_id,))


def trim_metrics(job_id: str, after_step: int) -> int:
    """Drop only the measurements past where a resumed run picks up.

    A run that carries on from step 120 has a real curve up to 120 and a set of
    readings from 121 onward belonging to an attempt that no longer exists.
    Clearing everything would throw away the half of the chart that is still
    true; keeping everything would draw the curve doubling back on itself.
    """
    c = connect()
    cur = c.execute("DELETE FROM metrics WHERE job_id=? AND step > ?",
                    (job_id, int(after_step)))
    c.commit()
    return cur.rowcount


def get_metrics(job_id: str) -> list[dict]:
    out = []
    for r in q("SELECT step,ts,data FROM metrics WHERE job_id=? ORDER BY step", (job_id,)):
        out.append({"step": r["step"], "ts": r["ts"], **json.loads(r["data"])})
    return out


def add_log(job_id: str, line: str, level: str = "info") -> None:
    ex("INSERT INTO logs (job_id,ts,level,line) VALUES (?,?,?,?)", (job_id, now(), level, line))


def get_logs(job_id: str, limit: int = 500) -> list[dict]:
    rows = q("SELECT ts,level,line FROM logs WHERE job_id=? ORDER BY ts DESC LIMIT ?",
             (job_id, limit))
    return list(reversed(rows))


def add_artifact(job_id: str, kind: str, filename: str, size: int) -> str:
    aid = new_id("art")
    ex("INSERT INTO artifacts (id,job_id,kind,filename,size_bytes,created_at)"
       " VALUES (?,?,?,?,?,?)", (aid, job_id, kind, filename, size, now()))
    return aid


def update_job_config(job_id: str, cfg: dict) -> None:
    ex("UPDATE jobs SET config=? WHERE id=?", (json.dumps(cfg), job_id))


def delete_job(job_id: str) -> list[str]:
    """Remove a job and everything hanging off it.

    Returns the artifact filenames so the caller can delete the files too --
    the rows are cheap, the zips are not, and leaving a few hundred megabytes
    behind on every delete would make the feature actively harmful.
    """
    files = [r["filename"] for r in
             q("SELECT filename FROM artifacts WHERE job_id=?", (job_id,))]
    c = connect()
    for table in ("metrics", "logs", "artifacts"):
        c.execute("DELETE FROM %s WHERE job_id=?" % table, (job_id,))
    c.execute("DELETE FROM jobs WHERE id=?", (job_id,))
    c.commit()
    return files


def list_artifacts(job_id: str) -> list[dict]:
    return q("SELECT * FROM artifacts WHERE job_id=? ORDER BY created_at", (job_id,))


# ===========================================================================
# Accounts
# ===========================================================================
#
# `password_hash` and `hf_token_enc` never leave this module by accident:
# public_user() is what every endpoint returns, and it is a whitelist rather
# than a blacklist so a column added later is private until someone decides
# otherwise.

_PUBLIC_USER_FIELDS = ("id", "username", "display_name", "role", "active",
                       "created_at", "last_login", "must_change", "email",
                       "avatar_url", "job_title", "department", "provider")


def public_user(user: dict | None) -> dict | None:
    if not user:
        return None
    out = {k: user.get(k) for k in _PUBLIC_USER_FIELDS}
    out["active"] = bool(out.get("active"))
    out["must_change"] = bool(out.get("must_change"))
    # "Pending" is the honest word for somebody the directory knows about who
    # has never been here. They can be found and shared with; they simply own
    # nothing yet, and the flag clears the first time they sign in.
    out["pending"] = bool(user.get("pending"))
    out["groups"] = json.loads(user.get("groups") or "[]")
    out["sso"] = bool(user.get("provider"))
    out["hf"] = {
        "connected": bool(user.get("hf_token_enc")),
        "username": user.get("hf_username"),
        "fullname": user.get("hf_fullname"),
        "avatar": user.get("hf_avatar"),
        "orgs": json.loads(user.get("hf_orgs") or "[]"),
        "can_write": bool(user.get("hf_can_write")),
        "checked_at": user.get("hf_checked_at"),
    }
    return out


def count_users() -> int:
    row = q1("SELECT COUNT(*) AS n FROM users")
    return int(row["n"]) if row else 0


def count_admins(active_only: bool = True) -> int:
    sql = "SELECT COUNT(*) AS n FROM users WHERE role='admin'"
    if active_only:
        sql += " AND active=1"
    row = q1(sql)
    return int(row["n"]) if row else 0


def get_user(user_id: str) -> dict | None:
    return q1("SELECT * FROM users WHERE id=?", (user_id,))


def get_user_by_name(username: str) -> dict | None:
    return q1("SELECT * FROM users WHERE username=?", ((username or "").lower(),))


def list_users() -> list[dict]:
    return q("SELECT * FROM users ORDER BY role, username")


def create_user(username: str, display_name: str, password_hash: str,
                role: str = "member", must_change: bool = False) -> str:
    uid = new_id("usr")
    ex("INSERT INTO users (id,username,display_name,password_hash,role,active,"
       "created_at,must_change) VALUES (?,?,?,?,?,1,?,?)",
       (uid, username.lower(), display_name or username, password_hash,
        role, now(), 1 if must_change else 0))
    return uid


def update_user(user_id: str, **fields: Any) -> None:
    allowed = {"display_name", "role", "active", "password_hash", "last_login",
               "must_change", "hf_token_enc", "hf_username", "hf_fullname",
               "hf_avatar", "hf_orgs", "hf_can_write", "hf_checked_at",
               "notify_url_enc", "notify_events", "model_providers_enc",
               "provider", "external_id",
               "email", "avatar_url", "job_title", "department", "groups",
               "synced_at", "pending", "username"}
    sets, args = [], []
    for k, v in fields.items():
        if k not in allowed:
            raise ValueError("refusing to update unknown column %r" % k)
        sets.append("%s=?" % k)
        args.append(v)
    if not sets:
        return
    args.append(user_id)
    ex("UPDATE users SET %s WHERE id=?" % ",".join(sets), args)


def delete_user(user_id: str) -> None:
    c = connect()
    c.execute("DELETE FROM sessions WHERE user_id=?", (user_id,))
    # Keys do not outlive their owner. A run is somebody else's to inherit; a
    # credential that still acts as a deleted account is not.
    c.execute("DELETE FROM api_keys WHERE user_id=?", (user_id,))
    # Runs and datasets outlive their owner and become unowned rather than
    # being destroyed. Deleting an account should not silently delete a week
    # of somebody else's GPU time.
    c.execute("UPDATE jobs SET owner_id=NULL WHERE owner_id=?", (user_id,))
    c.execute("UPDATE datasets SET owner_id=NULL WHERE owner_id=?", (user_id,))
    c.execute("DELETE FROM users WHERE id=?", (user_id,))
    c.commit()


def adopt_ownerless(user_id: str) -> int:
    """Give everything that predates accounts to the first administrator.

    Without this, upgrading a running studio would hide every existing run
    behind a "not yours" filter and look exactly like data loss.
    """
    c = connect()
    cur = c.execute("UPDATE jobs SET owner_id=? WHERE owner_id IS NULL", (user_id,))
    n = cur.rowcount
    c.execute("UPDATE datasets SET owner_id=? WHERE owner_id IS NULL", (user_id,))
    c.commit()
    return n



def get_user_by_external(provider: str, external_id: str) -> dict | None:
    return q1("SELECT * FROM users WHERE provider=? AND external_id=?",
              (provider, external_id))


def get_user_by_email(email: str) -> dict | None:
    """Case-insensitive, because an address is not case sensitive and every
    directory in existence disagrees with itself about the capitals."""
    if not email:
        return None
    return q1("SELECT * FROM users WHERE lower(email)=lower(?)"
              " ORDER BY pending, created_at LIMIT 1", (email.strip(),))


def search_users(term: str, limit: int = 20,
                 exclude: Iterable[str] = ()) -> list[dict]:
    """People matching what has been typed so far, best match first.

    Ranked rather than merely filtered, because the answer to "ma" should be
    Maria before Norman Hallmark, and a plain LIKE cannot tell those apart.
    The order is: an exact username or address, then anything *starting* with
    the term, then anything containing it. Within each band, people who have
    signed in come before people who are only in the directory -- a colleague
    you work with is a likelier target than a name from the org chart.
    """
    term = (term or "").strip().lower()
    skip = {x for x in exclude if x}
    if not term:
        rows = q("SELECT * FROM users WHERE active=1 ORDER BY pending,"
                 " last_login IS NULL, last_login DESC, display_name"
                 " LIMIT ?", (limit + len(skip),))
    else:
        like = "%" + term.replace("%", "").replace("_", "") + "%"
        rows = q(
            "SELECT *, ("
            "  CASE WHEN lower(username)=? OR lower(IFNULL(email,''))=? THEN 0"
            "       WHEN lower(username) LIKE ? OR lower(display_name) LIKE ?"
            "         OR lower(IFNULL(email,'')) LIKE ? THEN 1"
            "       ELSE 2 END) AS rank"
            " FROM users WHERE active=1 AND ("
            "  lower(username) LIKE ? OR lower(display_name) LIKE ?"
            "  OR lower(IFNULL(email,'')) LIKE ?)"
            " ORDER BY rank, pending, display_name LIMIT ?",
            (term, term, term + "%", term + "%", term + "%",
             like, like, like, limit + len(skip)))
    return [r for r in rows if r["id"] not in skip][:limit]


def unique_username(base: str) -> str:
    """A username nobody else has, derived from what the directory offered."""
    stem = re.sub(r"[^a-z0-9._-]", "", (base or "").strip().lower())
    stem = re.sub(r"^[^a-z0-9]+", "", stem)[:28] or "person"
    if len(stem) < 2:
        stem += "0"
    if not get_user_by_name(stem):
        return stem
    for n in range(2, 500):
        candidate = "%s%d" % (stem[:28], n)
        if not get_user_by_name(candidate):
            return candidate
    return new_id("usr")


def list_users_page(term: str = "", limit: int = 200,
                    include_pending: bool = False) -> tuple[list[dict], int]:
    """A page of accounts for the administration screen, and the total.

    Separate from `search_users` because the two want different things: that
    one is a lookup and only ever offers people who can actually be shared
    with, while this one has to show disabled accounts -- being able to see
    them is the point of the screen.

    Directory imports are hidden by default. A tenant of four thousand people
    would otherwise bury the handful who actually use the studio, which is the
    list an administrator came here to read.
    """
    where, args = [], []
    if not include_pending:
        where.append("pending=0")
    if term := (term or "").strip().lower():
        like = "%" + term.replace("%", "").replace("_", "") + "%"
        where.append("(lower(username) LIKE ? OR lower(display_name) LIKE ?"
                     " OR lower(IFNULL(email,'')) LIKE ?)")
        args += [like, like, like]
    clause = (" WHERE " + " AND ".join(where)) if where else ""
    total = q1("SELECT COUNT(*) AS n FROM users" + clause, args)
    rows = q("SELECT * FROM users" + clause
             + " ORDER BY pending, role, username LIMIT ?", args + [limit])
    return rows, int(total["n"]) if total else 0


def upsert_directory_user(idp_id: str, external_id: str, username: str,
                          display_name: str, email: str | None = None,
                          **extra: Any) -> tuple[str, bool]:
    """Record somebody the directory told us about. Returns (id, created).

    Deliberately conservative about what it overwrites. A directory owns the
    name, the address and the job title; it does not own the role somebody has
    in this studio, their password, or the fact that an administrator disabled
    them -- so those are left exactly as they are.
    """
    fields = {k: v for k, v in extra.items() if v is not None}
    existing = get_user_by_external(idp_id, external_id)
    if not existing and email:
        candidate = get_user_by_email(email)
        # Only adopt an account that is not already tied to a *different*
        # provider. Two directories claiming the same address is a conflict to
        # leave alone, not one to resolve by guessing.
        if candidate and candidate.get("provider") in (None, "", idp_id):
            existing = candidate
    if existing:
        update_user(existing["id"], provider=idp_id, external_id=external_id,
                    display_name=display_name or existing["display_name"],
                    email=email, synced_at=now(), **fields)
        return existing["id"], False

    uid = new_id("usr")
    ex("INSERT INTO users (id,username,display_name,password_hash,role,active,"
       "created_at,must_change,provider,external_id,email,pending,synced_at)"
       " VALUES (?,?,?,'','member',1,?,0,?,?,?,1,?)",
       (uid, unique_username(username), display_name or username,
        now(), idp_id, external_id, email, now()))
    if fields:
        update_user(uid, **fields)
    return uid, True


def directory_users(idp_id: str) -> list[dict]:
    return q("SELECT * FROM users WHERE provider=?", (idp_id,))


def count_pending(idp_id: str | None = None) -> int:
    sql = "SELECT COUNT(*) AS n FROM users WHERE pending=1"
    args: tuple = ()
    if idp_id:
        sql += " AND provider=?"
        args = (idp_id,)
    row = q1(sql, args)
    return int(row["n"]) if row else 0


# --------------------------------------------------------- identity providers

def create_idp(idp_id: str | None = None, **fields: Any) -> str:
    iid = idp_id or new_id("idp")
    cols = ["id", "created_at"]
    vals: list[Any] = [iid, now()]
    for k, v in fields.items():
        cols.append(k)
        vals.append(v)
    ex("INSERT INTO idp (%s) VALUES (%s)"
       % (",".join(cols), ",".join("?" * len(cols))), vals)
    return iid


_IDP_FIELDS = {"name", "kind", "issuer", "client_id", "client_secret_enc",
               "tenant", "enabled", "auto_create", "link_by_email",
               "allowed_domains", "admin_groups", "scopes", "discovery",
               "discovered_at", "sync_enabled", "sync_group",
               "sync_subject", "sync_secret_enc", "last_sync_at", "last_sync_note"}


def update_idp(idp_id: str, **fields: Any) -> None:
    sets, args = [], []
    for k, v in fields.items():
        if k not in _IDP_FIELDS:
            raise ValueError("refusing to update unknown idp column %r" % k)
        sets.append("%s=?" % k)
        args.append(v)
    if not sets:
        return
    args.append(idp_id)
    ex("UPDATE idp SET %s WHERE id=?" % ",".join(sets), args)


def get_idp(idp_id: str) -> dict | None:
    return q1("SELECT * FROM idp WHERE id=?", (idp_id,))


def list_idps(enabled_only: bool = False) -> list[dict]:
    sql = "SELECT * FROM idp"
    if enabled_only:
        sql += " WHERE enabled=1"
    return q(sql + " ORDER BY created_at")


def delete_idp(idp_id: str) -> int:
    """Remove a provider, and detach -- never delete -- the people it brought.

    Deleting those accounts would take their runs' owner with them. An account
    that can no longer sign in is a problem an administrator can fix in a
    minute; a week of somebody GPU time gone unowned is not.
    """
    c = connect()
    n = c.execute("UPDATE users SET provider=NULL, external_id=NULL"
                  " WHERE provider=?", (idp_id,)).rowcount
    c.execute("DELETE FROM oauth_states WHERE idp_id=?", (idp_id,))
    c.execute("DELETE FROM idp WHERE id=?", (idp_id,))
    c.commit()
    return n


# ------------------------------------------------------------- sign-in state

def put_oauth_state(state: str, idp_id: str, nonce: str, verifier: str,
                    redirect_uri: str, next_url: str | None) -> None:
    ex("INSERT INTO oauth_states (state,idp_id,nonce,verifier,redirect_uri,"
       "next_url,created_at) VALUES (?,?,?,?,?,?,?)",
       (state, idp_id, nonce, verifier, redirect_uri, next_url, now()))


def take_oauth_state(state: str, max_age_s: float = 600.0) -> dict | None:
    """Read a pending sign-in and destroy it. Single use, by construction."""
    row = q1("SELECT * FROM oauth_states WHERE state=?", (state,))
    ex("DELETE FROM oauth_states WHERE state=? OR created_at < ?",
       (state, now() - max_age_s))
    if not row or now() - row["created_at"] > max_age_s:
        return None
    return row


# ---------------------------------------------------------------- api keys

def create_api_key(user_id: str, name: str, key_hash: str, prefix: str) -> str:
    kid = new_id("key")
    ex("INSERT INTO api_keys (id,user_id,name,key_hash,prefix,created_at)"
       " VALUES (?,?,?,?,?,?)", (kid, user_id, name, key_hash, prefix, now()))
    return kid


def api_key_owner(key_hash: str) -> dict | None:
    """The account a key belongs to, and a note that it was used.

    The usage counter is written at most once a minute per key. An API meant
    to be called from a script can be called several times a second, and a
    database write per call to record "yes, still being used" would be the
    most expensive part of serving a short reply.
    """
    row = q1("SELECT * FROM api_keys WHERE key_hash=?", (key_hash,))
    if not row:
        return None
    user = get_user(row["user_id"])
    if not user or not user["active"]:
        return None
    if now() - float(row["last_used"] or 0) > 60:
        ex("UPDATE api_keys SET last_used=?, calls=calls+1 WHERE id=?",
           (now(), row["id"]))
    return user


def list_api_keys(user_id: str) -> list[dict]:
    return q("SELECT id,name,prefix,created_at,last_used,calls FROM api_keys"
             " WHERE user_id=? ORDER BY created_at DESC", (user_id,))


def revoke_api_keys(user_id: str) -> int:
    c = connect()
    cur = c.execute("DELETE FROM api_keys WHERE user_id=?", (user_id,))
    c.commit()
    return cur.rowcount


def delete_api_key(user_id: str, key_id: str) -> bool:
    c = connect()
    cur = c.execute("DELETE FROM api_keys WHERE id=? AND user_id=?",
                    (key_id, user_id))
    c.commit()
    return cur.rowcount > 0


def list_sessions(user_id: str) -> list[dict]:
    return q("SELECT created_at,last_used,expires_at,user_agent FROM sessions"
             " WHERE user_id=? ORDER BY last_used DESC", (user_id,))


# ===========================================================================
# Datasets
# ===========================================================================

def create_dataset(owner_id: str | None, name: str, source: str, **fields: Any) -> str:
    did = new_id("ds")
    ts = now()
    ex("INSERT INTO datasets (id,owner_id,name,source,origin,rows,bytes,columns,"
       "format,notes,parent_id,recipe,created_at,updated_at)"
       " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
       (did, owner_id, name, source, fields.get("origin"),
        int(fields.get("rows") or 0), int(fields.get("bytes") or 0),
        json.dumps(fields.get("columns") or []),
        json.dumps(fields.get("format") or {}),
        fields.get("notes"), fields.get("parent_id"),
        json.dumps(fields.get("recipe") or {}), ts, ts))
    return did


def _hydrate_dataset(r: dict) -> dict:
    r["columns"] = json.loads(r.get("columns") or "[]")
    r["format"] = json.loads(r.get("format") or "{}")
    r["recipe"] = json.loads(r.get("recipe") or "{}")
    # {name: rows}. Absent on datasets written before splits existed, which
    # are one unnamed split of everything -- said here rather than at each of
    # the half-dozen places that read it.
    r["splits"] = json.loads(r.get("splits") or "null") or (
        {"train": r.get("rows") or 0} if r.get("rows") else {})
    return r


def get_dataset(dataset_id: str) -> dict | None:
    r = q1("SELECT * FROM datasets WHERE id=?", (dataset_id,))
    return _hydrate_dataset(r) if r else None


def list_datasets(owner_id: str | None = None) -> list[dict]:
    sql = ("SELECT d.*, u.display_name AS owner_name FROM datasets d"
           " LEFT JOIN users u ON u.id = d.owner_id")
    args: tuple = ()
    if owner_id:
        sql += " WHERE d.owner_id = ?"
        args = (owner_id,)
    sql += " ORDER BY d.updated_at DESC"
    return [_hydrate_dataset(r) for r in q(sql, args)]


def update_dataset(dataset_id: str, **fields: Any) -> None:
    allowed = {"name", "notes", "rows", "bytes", "columns", "format", "origin",
               "splits", "recipe"}
    sets, args = [], []
    for k, v in fields.items():
        if k not in allowed:
            raise ValueError("refusing to update unknown column %r" % k)
        sets.append("%s=?" % k)
        args.append(json.dumps(v)
                    if k in ("columns", "format", "splits", "recipe") else v)
    if not sets:
        return
    sets.append("updated_at=?")
    args.append(now())
    args.append(dataset_id)
    ex("UPDATE datasets SET %s WHERE id=?" % ",".join(sets), args)


def delete_dataset(dataset_id: str) -> None:
    ex("DELETE FROM datasets WHERE id=?", (dataset_id,))


# ===========================================================================
# Sharing
# ===========================================================================

LEVELS = {"view": 1, "edit": 2}


def share(resource_type: str, resource_id: str, subject_type: str,
          subject_id: str | None, level: str, by: str | None) -> None:
    ex("INSERT INTO shares (id,resource_type,resource_id,subject_type,subject_id,"
       "level,created_at,created_by) VALUES (?,?,?,?,?,?,?,?)"
       " ON CONFLICT(resource_type,resource_id,subject_type,IFNULL(subject_id,''))"
       " DO UPDATE SET level=excluded.level",
       (new_id("shr"), resource_type, resource_id, subject_type, subject_id,
        level, now(), by))


def unshare(resource_type: str, resource_id: str, subject_type: str,
            subject_id: str | None) -> None:
    ex("DELETE FROM shares WHERE resource_type=? AND resource_id=?"
       " AND subject_type=? AND IFNULL(subject_id,'')=?",
       (resource_type, resource_id, subject_type, subject_id or ""))


def list_shares(resource_type: str, resource_id: str) -> list[dict]:
    return q("SELECT s.*, u.username, u.display_name FROM shares s"
             " LEFT JOIN users u ON u.id = s.subject_id"
             " WHERE s.resource_type=? AND s.resource_id=?"
             " ORDER BY s.subject_type, u.username",
             (resource_type, resource_id))


def clear_shares(resource_type: str, resource_id: str) -> None:
    ex("DELETE FROM shares WHERE resource_type=? AND resource_id=?",
       (resource_type, resource_id))


def access_level(resource_type: str, resource_id: str, owner_id: str | None,
                 user: dict | None) -> str | None:
    """"edit", "view", or None: what this user may do with this thing.

    Owner and administrator get edit. Otherwise the best of whatever has been
    shared with them directly or with everyone. Unowned resources -- the ones
    that predate accounts -- are visible to all, because hiding a studio's own
    history behind an owner that does not exist helps nobody.
    """
    if not user:
        return None
    if owner_id is None:
        return "edit" if user.get("role") == "admin" else "view"
    if owner_id == user["id"] or user.get("role") == "admin":
        return "edit"
    rows = q("SELECT level, subject_type FROM shares WHERE resource_type=?"
             " AND resource_id=? AND (subject_type='everyone'"
             " OR (subject_type='user' AND subject_id=?))",
             (resource_type, resource_id, user["id"]))
    best = None
    for r in rows:
        if best is None or LEVELS[r["level"]] > LEVELS[best]:
            best = r["level"]
    return best


def _visible_clause(user: dict, resource_type: str, alias: str) -> tuple[str, list]:
    """SQL fragment selecting the rows this user is allowed to see."""
    if user.get("role") == "admin":
        return "1=1", []
    return (
        "({a}.owner_id = ? OR {a}.owner_id IS NULL OR EXISTS ("
        "  SELECT 1 FROM shares s WHERE s.resource_type = ?"
        "    AND s.resource_id = {a}.id"
        "    AND (s.subject_type='everyone'"
        "         OR (s.subject_type='user' AND s.subject_id = ?))))".format(a=alias),
        [user["id"], resource_type, user["id"]])


def visible_jobs(user: dict, limit: int = 100) -> list[dict]:
    where, args = _visible_clause(user, "job", "j")
    rows = q("SELECT j.*, EXISTS(SELECT 1 FROM artifacts a WHERE a.job_id = j.id)"
             " AS has_model, u.display_name AS owner_name, u.username AS owner_username,"
             " EXISTS(SELECT 1 FROM shares s WHERE s.resource_type='job'"
             "        AND s.resource_id = j.id) AS is_shared"
             " FROM jobs j LEFT JOIN users u ON u.id = j.owner_id"
             " WHERE " + where + " ORDER BY j.created_at DESC LIMIT ?",
             args + [limit])
    for r in rows:
        r["has_model"] = bool(r["has_model"])
        r["is_shared"] = bool(r["is_shared"])
        r["mine"] = r["owner_id"] == user["id"]
    return [_hydrate(r) for r in rows]


def visible_datasets(user: dict) -> list[dict]:
    where, args = _visible_clause(user, "dataset", "d")
    rows = q("SELECT d.*, u.display_name AS owner_name, u.username AS owner_username,"
             " EXISTS(SELECT 1 FROM shares s WHERE s.resource_type='dataset'"
             "        AND s.resource_id = d.id) AS is_shared"
             " FROM datasets d LEFT JOIN users u ON u.id = d.owner_id"
             " WHERE " + where + " ORDER BY d.updated_at DESC", args)
    out = []
    for r in rows:
        r["is_shared"] = bool(r["is_shared"])
        r["mine"] = r["owner_id"] == user["id"]
        out.append(_hydrate_dataset(r))
    return out


# ===========================================================================
# Evaluations
# ===========================================================================
#
# An "eval" here is a saved set of prompts, not a score. The score belongs to
# the pair (prompt set, model) and lives in eval_scores, because the same
# prompts are meant to be re-run against next week's model -- that is the
# entire reason for saving them rather than typing them into the playground.

def create_eval(owner_id: str | None, name: str, items: list[dict],
                notes: str = "") -> str:
    eid = new_id("ev")
    ts = now()
    ex("INSERT INTO evals (id,owner_id,name,notes,items,created_at,updated_at)"
       " VALUES (?,?,?,?,?,?,?)",
       (eid, owner_id, name, notes, json.dumps(items), ts, ts))
    return eid


def _hydrate_eval(r: dict) -> dict:
    r["items"] = json.loads(r.get("items") or "[]")
    return r


def get_eval(eval_id: str) -> dict | None:
    r = q1("SELECT * FROM evals WHERE id=?", (eval_id,))
    return _hydrate_eval(r) if r else None


def update_eval(eval_id: str, **fields: Any) -> None:
    allowed = {"name", "notes", "items"}
    sets, args = [], []
    for k, v in fields.items():
        if k not in allowed:
            raise ValueError("refusing to update unknown column %r" % k)
        sets.append("%s=?" % k)
        args.append(json.dumps(v) if k == "items" else v)
    if not sets:
        return
    sets.append("updated_at=?")
    args += [now(), eval_id]
    ex("UPDATE evals SET %s WHERE id=?" % ",".join(sets), args)


def delete_eval(eval_id: str) -> None:
    c = connect()
    c.execute("DELETE FROM eval_scores WHERE eval_id=?", (eval_id,))
    c.execute("DELETE FROM evals WHERE id=?", (eval_id,))
    c.commit()


def visible_evals(user: dict) -> list[dict]:
    where, args = _visible_clause(user, "eval", "e")
    rows = q("SELECT e.*, u.display_name AS owner_name, u.username AS owner_username,"
             " EXISTS(SELECT 1 FROM shares s WHERE s.resource_type='eval'"
             "        AND s.resource_id = e.id) AS is_shared,"
             " (SELECT COUNT(*) FROM eval_scores sc WHERE sc.eval_id = e.id)"
             "   AS score_count"
             " FROM evals e LEFT JOIN users u ON u.id = e.owner_id"
             " WHERE " + where + " ORDER BY e.updated_at DESC", args)
    out = []
    for r in rows:
        r["is_shared"] = bool(r["is_shared"])
        r["mine"] = r["owner_id"] == user["id"]
        out.append(_hydrate_eval(r))
    return out


def record_score(eval_id: str, model_job_id: str, run_job_id: str | None,
                 metrics: dict, items: list[dict] | None) -> str:
    sid = new_id("scr")
    ex("INSERT INTO eval_scores (id,eval_id,model_job_id,run_job_id,created_at,"
       "metrics,items) VALUES (?,?,?,?,?,?,?)",
       (sid, eval_id, model_job_id, run_job_id, now(), json.dumps(metrics),
        json.dumps(items or [])))
    return sid


def list_scores(eval_id: str, with_items: bool = False) -> list[dict]:
    """Every scoring of this prompt set, newest first, one row per model run.

    Joined against the job so a score keeps its meaning after the model it
    describes has been renamed -- and so a deleted run shows as a score with
    no run behind it rather than as a dangling id.
    """
    rows = q("SELECT sc.*, j.name AS model_name, j.kind AS model_kind,"
             " j.finished_at AS model_finished_at, j.summary AS model_summary"
             " FROM eval_scores sc LEFT JOIN jobs j ON j.id = sc.model_job_id"
             " WHERE sc.eval_id=? ORDER BY sc.created_at DESC", (eval_id,))
    out = []
    for r in rows:
        r["metrics"] = json.loads(r["metrics"] or "{}")
        r["items"] = json.loads(r["items"] or "[]") if with_items else []
        if r.get("model_summary"):
            try:
                r["model_summary"] = json.loads(r["model_summary"])
            except (TypeError, ValueError):
                r["model_summary"] = None
        out.append(r)
    return out


def get_score(score_id: str) -> dict | None:
    r = q1("SELECT * FROM eval_scores WHERE id=?", (score_id,))
    if not r:
        return None
    r["metrics"] = json.loads(r["metrics"] or "{}")
    r["items"] = json.loads(r["items"] or "[]")
    return r


def delete_score(score_id: str) -> None:
    ex("DELETE FROM eval_scores WHERE id=?", (score_id,))
