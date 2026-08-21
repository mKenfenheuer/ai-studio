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
"""

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

def create_job(name: str, kind: str, cfg: dict) -> str:
    jid = new_id("job")
    ex("INSERT INTO jobs (id,name,kind,status,config,created_at) VALUES (?,?,?,'queued',?,?)",
       (jid, name, kind, json.dumps(cfg), now()))
    return jid


def _hydrate(r: dict) -> dict:
    r["config"] = json.loads(r["config"])
    return r


def list_jobs(limit: int = 100) -> list[dict]:
    return [_hydrate(r) for r in
            q("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,))]


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


def requeue_jobs_for_runner(runner_id: str) -> list[str]:
    """A runner vanished mid-job. Put its unfinished work back on the queue so
    another capable runner can take it, instead of the job hanging forever."""
    ids = [r["id"] for r in
           q("SELECT id FROM jobs WHERE runner_id=? AND status IN ('assigned','running')",
             (runner_id,))]
    for jid in ids:
        ex("UPDATE jobs SET status='queued', runner_id=NULL, step=0 WHERE id=?", (jid,))
    return ids


# ------------------------------------------------- metrics / logs / files

def add_metric(job_id: str, step: int, data: dict) -> None:
    ex("INSERT INTO metrics (job_id,step,ts,data) VALUES (?,?,?,?)",
       (job_id, step, now(), json.dumps(data)))


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


def list_artifacts(job_id: str) -> list[dict]:
    return q("SELECT * FROM artifacts WHERE job_id=? ORDER BY created_at", (job_id,))
