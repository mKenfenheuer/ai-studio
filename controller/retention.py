"""What is kept, for how long, and what it is taking.

A studio that trains models fills a disk. Every run leaves a model of a few
hundred megabytes to a few tens of gigabytes; every run writes a metric per
step and a log line per event; every dataset and every picture stays. None of
it was ever removed by anything but a person deleting a run, and the lab box
this was written against was at 97% when this file was started.

Three rules, each a setting an administrator can turn off:

* **Models expire.** A finished run's model is removed after N days -- the
  run stays, with its chart, its log and its summary, and a line saying the
  model went and why. A run that is served under a registered name, or that
  carries the `keep` tag, is never touched: those are the ones somebody has
  said matter.
* **Old runs get lighter.** Past M days, a finished run's metrics are thinned
  to a few hundred points that still draw the same curve, and its log is cut
  to its head and tail. The chart of a run from March keeps its shape at a
  resolution nobody would notice.
* **An account holds so many gigabytes of models.** Checked when a training
  run is started, not when its model arrives -- refusing a model after four
  hours of training would be the worst moment to do it.

And one report: what the disk is holding, by kind, with the biggest runs
first, so the answer to "why is the disk full" is a table rather than a
`du`.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from . import assets, config, db

SETTING_KEY = "retention"
KEEP_TAG = "keep"

DEFAULTS = {
    "artifact_days": 0,        # 0: models are kept forever
    "trim_days": 90,           # metrics and logs of runs older than this are thinned
    "artifact_quota_gb": 0,    # 0: no per-account ceiling
}

# What a thinned run keeps. Enough points to draw the curve, and the log's
# beginning (what it was) and end (how it went).
TRIM_METRICS_TO = 400
TRIM_LOG_HEAD = 60
TRIM_LOG_TAIL = 240

# Runs that ended this way are candidates; a queued or running run is not.
FINISHED = ("succeeded", "failed", "cancelled")


def settings() -> dict:
    try:
        got = json.loads(db.get_setting(SETTING_KEY) or "{}")
    except (TypeError, ValueError):
        got = {}
    out = dict(DEFAULTS)
    for k in DEFAULTS:
        try:
            out[k] = max(0, int(got.get(k, DEFAULTS[k]) or 0))
        except (TypeError, ValueError):
            pass
    return out


def save_settings(values: dict, user_id: str | None = None) -> dict:
    current = settings()
    for k in DEFAULTS:
        if k in values:
            try:
                current[k] = max(0, int(values[k] or 0))
            except (TypeError, ValueError):
                raise ValueError("%s must be a whole number of %s."
                                 % (k.replace("_", " "),
                                    "gigabytes" if k.endswith("_gb") else "days"))
    db.set_setting(SETTING_KEY, json.dumps(current), user_id)
    return current


# ------------------------------------------------------------------ report

def _dir_bytes(path: Path) -> int:
    total = 0
    try:
        for p in path.rglob("*"):
            if p.is_file():
                total += p.stat().st_size
    except OSError:
        pass
    return total


def report() -> dict:
    """What the studio's own disk is holding."""
    artifacts = db.q("SELECT a.job_id, a.kind, a.size_bytes, j.name, j.owner_id,"
                     " j.status, j.finished_at, j.tags"
                     " FROM artifacts a LEFT JOIN jobs j ON j.id = a.job_id")
    by_kind: dict[str, int] = {}
    by_job: dict[str, dict] = {}
    for a in artifacts:
        by_kind[a["kind"]] = by_kind.get(a["kind"], 0) + (a["size_bytes"] or 0)
        entry = by_job.setdefault(a["job_id"], {
            "job_id": a["job_id"], "name": a["name"] or a["job_id"],
            "owner_id": a["owner_id"], "status": a["status"],
            "finished_at": a["finished_at"], "bytes": 0, "kinds": [],
            "tags": _tags(a.get("tags"))})
        entry["bytes"] += a["size_bytes"] or 0
        entry["kinds"].append(a["kind"])
    served = {r["job_id"] for r in db.list_aliases()}
    for e in by_job.values():
        e["served"] = e["job_id"] in served
        e["kept"] = KEEP_TAG in e["tags"]
    biggest = sorted(by_job.values(), key=lambda e: -e["bytes"])[:20]
    for e in biggest:
        owner = db.get_user(e["owner_id"]) if e.get("owner_id") else None
        e["owner"] = db.public_user(owner) if owner else None

    try:
        usage = shutil.disk_usage(config.DATA_DIR)
        disk = {"free_gb": round(usage.free / 1024 ** 3, 1),
                "total_gb": round(usage.total / 1024 ** 3, 1),
                "used_pct": round(100 * (usage.total - usage.free) / usage.total, 1)}
    except OSError:
        disk = None
    db_bytes = 0
    for suffix in ("", "-wal", "-shm"):
        try:
            db_bytes += (Path(str(config.DB_PATH) + suffix)).stat().st_size
        except OSError:
            pass
    metrics_rows = (db.q1("SELECT COUNT(*) AS n FROM metrics") or {}).get("n") or 0
    log_rows = (db.q1("SELECT COUNT(*) AS n FROM logs") or {}).get("n") or 0
    return {
        "disk": disk,
        "artifacts": {"bytes": sum(by_kind.values()), "by_kind": by_kind,
                      "runs": len(by_job)},
        "datasets_bytes": _dir_bytes(config.DATA_DIR / "datasets"),
        "assets": assets.usage(),
        "database": {"bytes": db_bytes, "metrics_rows": metrics_rows,
                     "log_rows": log_rows},
        "biggest": biggest,
        "settings": settings(),
        "last_sweep": _last_sweep(),
    }


def _tags(raw) -> list[str]:
    try:
        return list(json.loads(raw or "[]"))
    except (TypeError, ValueError):
        return []


def account_model_bytes(user_id: str) -> int:
    row = db.q1("SELECT SUM(a.size_bytes) AS b FROM artifacts a"
                " JOIN jobs j ON j.id = a.job_id WHERE j.owner_id=?", (user_id,))
    return int((row or {}).get("b") or 0)


def check_quota(user_id: str) -> None:
    """Refuse a new training run when the account is over its ceiling.

    Raised at creation, with the numbers: a model refused after four hours of
    training would be the worst moment to find out.
    """
    quota = settings()["artifact_quota_gb"]
    if not quota:
        return
    held = account_model_bytes(user_id)
    if held >= quota * 1024 ** 3:
        raise ValueError(
            "This account holds %.1f GB of models, over its %d GB ceiling. "
            "Remove a model you no longer need -- on its run page, or from "
            "Settings -- and start again." % (held / 1024 ** 3, quota))


# ------------------------------------------------------------------ sweeping

def drop_model(job_id: str, why: str) -> dict:
    """Remove a run's model files and rows; keep the run.

    What was there is written on the run, in the summary and in the log, so
    a page that used to have a Download button says what happened to it.
    """
    rows = db.list_artifacts(job_id)
    freed = 0
    for a in rows:
        path = config.ARTIFACT_DIR / a["filename"]
        try:
            freed += path.stat().st_size
            path.unlink()
        except OSError:
            pass
    db.ex("DELETE FROM artifacts WHERE job_id=?", (job_id,))
    job = db.get_job(job_id)
    if job:
        summary = dict(job.get("summary") or {})
        summary["artifact_removed"] = {"at": time.time(), "why": why,
                                       "bytes": freed,
                                       "kinds": [a["kind"] for a in rows]}
        db.set_job_summary(job_id, summary)
        db.add_log(job_id, "The model was removed: %s (%.1f GB freed)."
                   % (why, freed / 1024 ** 3), "warn")
    return {"job_id": job_id, "bytes": freed, "files": len(rows)}


def expire_models(days: int, now: float | None = None) -> list[dict]:
    """Models of finished runs older than `days`, unless somebody said keep."""
    if not days:
        return []
    cutoff = (now or time.time()) - days * 86400
    served = {r["job_id"] for r in db.list_aliases()}
    out = []
    for j in db.q("SELECT DISTINCT j.id, j.tags FROM jobs j JOIN artifacts a"
                  " ON a.job_id = j.id WHERE j.status IN (?,?,?)"
                  " AND COALESCE(j.finished_at, j.created_at) < ?",
                  (*FINISHED, cutoff)):
        if j["id"] in served or KEEP_TAG in _tags(j.get("tags")):
            continue
        out.append(drop_model(j["id"], "older than %d days" % days))
    return out


def thin_run(job_id: str) -> dict:
    """Fewer metric points, a shorter log; the same picture."""
    metrics = db.q("SELECT rowid AS id, step FROM metrics WHERE job_id=?"
                   " ORDER BY step, rowid", (job_id,))
    dropped_metrics = 0
    if len(metrics) > TRIM_METRICS_TO:
        # Every k-th point, keeping the first and the last; the curve of a
        # thousand-step run drawn from four hundred of them is the same curve.
        keep_every = max(2, round(len(metrics) / TRIM_METRICS_TO))
        keep = {m["id"] for i, m in enumerate(metrics)
                if i % keep_every == 0 or i == len(metrics) - 1}
        # Points that carry something rare stay regardless: a held-out reading
        # or a sample is not a thing to thin.
        for m in db.q("SELECT rowid AS id FROM metrics WHERE job_id=? AND"
                      " (data LIKE '%val_%' OR data LIKE '%sample_%')", (job_id,)):
            keep.add(m["id"])
        drop = [m["id"] for m in metrics if m["id"] not in keep]
        for i in range(0, len(drop), 500):
            chunk = drop[i:i + 500]
            db.ex("DELETE FROM metrics WHERE rowid IN (%s)" % ",".join("?" * len(chunk)),
                  chunk)
        dropped_metrics = len(drop)
    logs = db.q("SELECT rowid AS rid FROM logs WHERE job_id=? ORDER BY ts, rowid",
                (job_id,))
    dropped_logs = 0
    if len(logs) > TRIM_LOG_HEAD + TRIM_LOG_TAIL:
        middle = [l["rid"] for l in logs[TRIM_LOG_HEAD:-TRIM_LOG_TAIL]]
        for i in range(0, len(middle), 500):
            chunk = middle[i:i + 500]
            db.ex("DELETE FROM logs WHERE rowid IN (%s)" % ",".join("?" * len(chunk)),
                  chunk)
        dropped_logs = len(middle)
        db.add_log(job_id, "This log was shortened to its beginning and end "
                   "after %d lines; the run is old." % len(logs))
    return {"job_id": job_id, "metrics": dropped_metrics, "logs": dropped_logs}


def thin_old_runs(days: int, now: float | None = None) -> list[dict]:
    if not days:
        return []
    cutoff = (now or time.time()) - days * 86400
    out = []
    for j in db.q("SELECT id, summary FROM jobs WHERE status IN (?,?,?)"
                  " AND COALESCE(finished_at, created_at) < ?", (*FINISHED, cutoff)):
        try:
            summary = json.loads(j.get("summary") or "{}") or {}
        except (TypeError, ValueError):
            summary = {}
        if summary.get("thinned_at"):
            continue
        result = thin_run(j["id"])
        summary["thinned_at"] = time.time()
        db.set_job_summary(j["id"], summary)
        if result["metrics"] or result["logs"]:
            out.append(result)
    return out


def _last_sweep() -> dict | None:
    try:
        return json.loads(db.get_setting(SETTING_KEY + "_last") or "null")
    except (TypeError, ValueError):
        return None


def sweep() -> dict:
    """Everything above, once. Called hourly and from a button."""
    s = settings()
    expired = expire_models(s["artifact_days"])
    thinned = thin_old_runs(s["trim_days"])
    orphans = assets.sweep()
    result = {
        "at": time.time(),
        "models_removed": len(expired),
        "bytes_freed": sum(e["bytes"] for e in expired) + orphans["bytes"],
        "runs_thinned": len(thinned),
        "metric_rows_dropped": sum(t["metrics"] for t in thinned),
        "log_rows_dropped": sum(t["logs"] for t in thinned),
        "orphan_files": orphans["files"],
    }
    db.set_setting(SETTING_KEY + "_last", json.dumps(result))
    return result
