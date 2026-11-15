#!/usr/bin/env python
"""Check the shared directory between controller and runners, and resume.

Run it with `python scripts/check-shared.py` from the repository root.

What it guards:

* **A runner trusts the share only when it is the controller's.** An empty
  directory of the right name -- a mount that failed -- must fall back to HTTP,
  not swallow results nobody can see.
* **A result written into the share becomes an artifact by rename**, only
  when the runner hands it over, and only for its own run.
* **A model is read from the share in place**, with no download.
* **Artifacts from before the share are still found.**
* **A resumed run stops where the interrupted one would have.** Early stopping
  counts checks without improvement; a checkpoint that dropped that count gave
  every resume a fresh run of patience.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_TMP = Path(tempfile.mkdtemp(prefix="ai-studio-shared-"))
os.environ["AI_STUDIO_DATA"] = str(_TMP / "data")
os.environ["AI_STUDIO_SHARED_DIR"] = str(_TMP / "shared")
os.environ["AI_STUDIO_JOIN_TOKEN"] = "check-join-token"
os.environ["AI_STUDIO_MODEL_CACHE"] = str(_TMP / "runner-cache")
os.environ.pop("HF_HUB_CACHE", None)

from fastapi.testclient import TestClient  # noqa: E402

from controller import config, db  # noqa: E402
from runner import shared  # noqa: E402

FAILED: list[str] = []


def ok(label: str, condition: bool, note: str = "") -> None:
    print("  %s  %s %s" % ("ok  " if condition else "FAIL", label, note))
    if not condition:
        FAILED.append(label)


class Ctx:
    def __init__(self, job_id: str, workdir: Path):
        self.job_id, self.workdir = job_id, str(workdir)

    def log(self, *_a, **_k):
        pass


def zip_model(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("config.json", json.dumps({"model_type": "qwen2"}))
        z.writestr("model.safetensors", b"\0" * 64)


def check_resume() -> None:
    print("resume keeps early stopping's count")
    from runner import earlystop
    s = earlystop.Stopper(Ctx("x", _TMP), 4)
    for i, v in enumerate([1.0, 0.9, 0.95, 0.96, 0.97]):
        s.update(i * 100, v)
    state = s.state()
    ok("the checkpoint carries the count", state.get("stop_waited") == 3, str(state))
    t = earlystop.Stopper(Ctx("x", _TMP), 4)
    t.restore(state)
    ok("the resumed run stops at the next check without improvement",
       t.update(500, 0.98) == "stop")
    old = earlystop.Stopper(Ctx("x", _TMP), 4)
    old.restore({"best_val": 0.9, "best_step": 100})
    ok("a checkpoint from before this reads as zero", old.waited == 0)


def main() -> int:
    check_resume()
    c = TestClient(__import__("controller.app", fromlist=["app"]).app)
    with c:
        print("the share's identity")
        sid = config.share_id()
        ok("the controller marks the directory", bool(sid)
           and (config.SHARED_DIR / config.SHARE_MARKER).read_text().strip() == sid)
        ok("a runner with the wrong id falls back",
           "HTTP" in (shared.confirm("someone-else") or "") and not shared.active())
        ok("a controller without a share means HTTP",
           "HTTP" in (shared.confirm(None) or "") and not shared.active())
        shared.confirm(sid)
        ok("the right id turns it on", shared.active())
        ok("the Hub cache goes into the share",
           (shared.configure_env() or True)
           and os.environ.get("HF_HUB_CACHE") == str(config.SHARED_DIR / "hf"))

        print("handing a result over")
        jid = db.create_job("t", "finetune_llm", {})
        other = db.create_job("u", "finetune_llm", {})
        ctx = Ctx(jid, _TMP / "work")
        out = shared.result_path(ctx, "model.zip")
        ok("a result is packed into incoming/", shared.is_incoming(out)
           and out.name.startswith(jid), str(out))
        zip_model(out)
        tok = {"X-Runner-Token": "check-join-token"}
        r = c.post("/api/jobs/%s/artifact/shared" % jid, params={"name": out.name})
        ok("without the join token it is refused", r.status_code in (401, 403),
           str(r.status_code))
        r = c.post("/api/jobs/%s/artifact/shared" % other,
                   params={"name": out.name}, headers=tok)
        ok("another run cannot take it", r.status_code == 400, str(r.status_code))
        r = c.post("/api/jobs/%s/artifact/shared" % jid,
                   params={"name": "../" + out.name}, headers=tok)
        ok("a path outside incoming/ is refused", r.status_code == 400,
           str(r.status_code))
        r = c.post("/api/jobs/%s/artifact/shared" % jid,
                   params={"name": out.name}, headers=tok)
        ok("the run's own result is taken", r.status_code == 200,
           "%s %s" % (r.status_code, r.text[:200]))
        final = config.ARTIFACT_DIR / ("%s.zip" % jid)
        ok("it is now the artifact, moved not copied",
           final.is_file() and not out.exists())
        ok("the controller lists it", bool(db.list_artifacts(jid)))
        out2 = shared.result_path(ctx, "adapter.zip")
        zip_model(out2)
        r = c.post("/api/jobs/%s/artifact/shared" % jid,
                   params={"name": out2.name, "kind": "adapter"}, headers=tok)
        ok("an adapter lands under its kind", r.status_code == 200
           and (config.ARTIFACT_DIR / ("%s-adapter.zip" % jid)).is_file())

        print("reading it back")
        from runner import artifacts
        ok("the runner finds it in place", shared.artifact_zip(jid) == final)
        got = artifacts.fetch("http://unreachable.invalid", "x", jid)
        ok("fetch unpacks it without a download",
           (got / "config.json").is_file() and final.is_file())
        ok("the HF cache in the share is never evicted",
           shared.contains(config.SHARED_DIR / "hf"))

        print("artifacts from before the share")
        config.LEGACY_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        old_job = db.create_job("old", "finetune_llm", {})
        zip_model(config.LEGACY_ARTIFACT_DIR / ("%s.zip" % old_job))
        ok("one left in the data volume is still found",
           config.artifact_path("%s.zip" % old_job).parent == config.LEGACY_ARTIFACT_DIR)
        config.migrate_artifacts(log=lambda _s: None)
        ok("and on one filesystem it is moved over",
           (config.ARTIFACT_DIR / ("%s.zip" % old_job)).is_file()
           and config.artifact_path("%s.zip" % old_job).parent == config.ARTIFACT_DIR)

    if FAILED:
        print("\n%d check(s) failed: %s" % (len(FAILED), ", ".join(FAILED)))
        return 1
    print("\nall shared-directory checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
