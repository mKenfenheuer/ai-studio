"""SQLite persistence.

Plain sqlite3 rather than an ORM: the controller must stay installable with no
compiler and no heavyweight dependencies, and the schema here is small enough
that an ORM would cost more than it saves.
"""
from __future__ import annotations

import json
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


def requeue_jobs_for_runner(runner_id: str) -> tuple[list[str], list[str]]:
    """A runner vanished mid-job. Decide what happens to its work.

    Returns (requeued, rescued).

    The distinction matters more than it looks. A job that already uploaded an
    artifact has *finished* -- the runner died in the gap between the upload
    landing and the "done" message being processed. Requeuing that job throws
    away a completed model and starts a twenty-minute run again from noise,
    which is the worst possible response to work that already succeeded.
    """
    rows = q("SELECT id FROM jobs WHERE runner_id=? AND status IN ('assigned','running')",
             (runner_id,))
    requeued, rescued = [], []
    for r in rows:
        jid = r["id"]
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
                       "created_at", "last_login", "must_change")


def public_user(user: dict | None) -> dict | None:
    if not user:
        return None
    out = {k: user.get(k) for k in _PUBLIC_USER_FIELDS}
    out["active"] = bool(out.get("active"))
    out["must_change"] = bool(out.get("must_change"))
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
               "hf_avatar", "hf_orgs", "hf_can_write", "hf_checked_at"}
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
    allowed = {"name", "notes", "rows", "bytes", "columns", "format", "origin"}
    sets, args = [], []
    for k, v in fields.items():
        if k not in allowed:
            raise ValueError("refusing to update unknown column %r" % k)
        sets.append("%s=?" % k)
        args.append(json.dumps(v) if k in ("columns", "format") else v)
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
