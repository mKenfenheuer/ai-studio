#!/usr/bin/env python
"""Walk a project through the whole workflow, against a real controller.

Run it with `python scripts/check-workflow.py` from the repository root. It
starts a controller of its own on a throwaway data directory, so it touches
nothing you care about and needs no machine to be connected.

Why this exists, when there are four other check scripts: every defect the
end-to-end run on the lab box turned up (ROADMAP section 14) was invisible to
all of them. `check-formats` proves a function; `check-render` proves a page
draws. Neither can see "the wizard pins a machine, so creating a run returns
500", because that fault lives in the seam between a browser, a route and the
scheduler -- and the seams are where this app breaks.

So this drives the same HTTP endpoints a browser drives, in the order a person
uses them: make a project, put data in it, file things, start a run the way
the wizard starts one, publish it, and check the map says what it should at
each step. It does not train anything -- no GPU, no model download -- because
the training is not what breaks. What breaks is everything around it.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FAILED: list[str] = []
PORT = 8788


def check(name: str, ok: bool, detail: object = "") -> None:
    print("  %s  %s%s" % ("ok  " if ok else "FAIL", name,
                          "" if ok else "   -> %s" % (detail,)))
    if not ok:
        FAILED.append(name)


class Studio:
    """A controller on a throwaway directory, and the calls a browser makes."""

    def __init__(self, port: int = PORT) -> None:
        self.port = port
        self.base = "http://127.0.0.1:%d" % port
        self.dir = Path(tempfile.mkdtemp(prefix="ai-studio-workflow-"))
        self.proc: subprocess.Popen | None = None
        self.cookie = ""

    def start(self) -> None:
        env = {**os.environ, "AI_STUDIO_DATA": str(self.dir)}
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "controller.app:app",
             "--port", str(self.port), "--log-level", "warning"],
            cwd=str(ROOT), env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        for _ in range(120):
            try:
                with socket.create_connection(("127.0.0.1", self.port), 0.2):
                    return
            except OSError:
                if self.proc.poll() is not None:
                    err = (self.proc.stderr.read() or b"").decode()[-600:]
                    raise RuntimeError("the controller would not start:\n" + err)
                time.sleep(0.25)
        raise RuntimeError("the controller did not answer in 30s")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        shutil.rmtree(self.dir, ignore_errors=True)

    # -- the calls -----------------------------------------------------------

    def call(self, method: str, path: str, body: object = None) -> object:
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(
            self.base + path, data=data, method=method,
            headers={"Content-Type": "application/json",
                     **({"Cookie": self.cookie} if self.cookie else {})})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                for header, value in r.getheaders():
                    if header.lower() == "set-cookie":
                        self.cookie = value.split(";")[0]
                text = r.read().decode()
                return json.loads(text) if text else None
        except urllib.error.HTTPError as e:
            raw = e.read().decode()
            try:
                detail = json.loads(raw).get("detail")
            except ValueError:
                detail = raw[:200]
            return {"ERROR": e.code, "detail": detail}

    def add_runner(self, name: str, vram_gb: float) -> str:
        """A machine in the fleet, without one having to be connected."""
        import sqlite3  # noqa: PLC0415 - only the fixtures need the database
        caps = {"backend": "cuda", "device_name": name, "vram_gb": vram_gb,
                "quantization": {"4bit": True, "8bit": True},
                "modalities": ["text", "vision"], "recommended_dtype": "bfloat16"}
        db = sqlite3.connect(self.dir / "studio.db")
        db.execute("INSERT OR REPLACE INTO runners (id,name,capabilities,status,"
                   "last_seen,first_seen) VALUES (?,?,?,?,?,?)",
                   ("run_smoke", name, json.dumps(caps), "online",
                    time.time(), time.time()))
        db.commit()
        db.close()
        return "run_smoke"

    def upload(self, rows: list[dict], name: str, project: str = "") -> dict:
        """The multipart upload the dataset page performs."""
        boundary = uuid.uuid4().hex
        body = ("\n".join(json.dumps(r) for r in rows)).encode()
        payload = (
            ("--%s\r\nContent-Disposition: form-data; name=\"file\"; "
             "filename=\"%s.jsonl\"\r\nContent-Type: application/json\r\n\r\n"
             % (boundary, name)).encode()
            + body + ("\r\n--%s--\r\n" % boundary).encode())
        query = "?name=%s%s" % (name, "&project=" + project if project else "")
        req = urllib.request.Request(
            self.base + "/api/datasets/upload" + query, data=payload,
            method="POST",
            headers={"Content-Type": "multipart/form-data; boundary=" + boundary,
                     "Cookie": self.cookie})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode())


def conversation(i: int) -> dict:
    """One row shaped like the tool-calling data this studio is used for."""
    return {"messages": [
        {"role": "system", "content": "You manage a house. Devices: light.hall"},
        {"role": "user", "content": "Turn the hall light %s, please (%d)."
                                    % ("on" if i % 2 else "off", i)},
        {"role": "assistant", "tool_calls": [
            {"type": "function", "id": "call_%d" % i,
             "function": {"name": "set_light",
                          "arguments": json.dumps({"entity": "light.hall",
                                                   "state": "on" if i % 2 else "off"})}}]},
        {"role": "tool", "name": "set_light", "tool_call_id": "call_%d" % i,
         "content": "done"},
        {"role": "assistant", "content": "The hall light is %s."
                                         % ("on" if i % 2 else "off")},
    ], "tools": [{"type": "function", "function": {
        "name": "set_light", "description": "Switch a light.",
        "parameters": {"type": "object",
                       "properties": {"entity": {"type": "string"},
                                      "state": {"type": "string"}}}}}]}


def stage(project: dict, key: str) -> dict:
    return next(m for m in project["map"] if m["key"] == key)


def main() -> int:
    studio = Studio()
    try:
        studio.start()
        print("\nSigning in")
        setup = studio.call("POST", "/api/auth/setup",
                            {"username": "smoke", "password": "correcthorse1",
                             "display_name": "Smoke"})
        check("the first account is made", bool(setup.get("ok")), setup)

        print("\nA project")
        project = studio.call("POST", "/api/projects",
                              {"name": "Smoke", "goal": "Prove the loop."})
        check("a project can be started", "id" in project, project)
        pid = project["id"]
        listing = studio.call("GET", "/api/projects")
        check("and appears in the list",
              any(p["id"] == pid for p in listing["projects"]))

        print("\nData")
        rows = [conversation(i) for i in range(60)]
        data = studio.upload(rows, "smoke-rows", pid)
        check("an upload is filed into the project", data.get("project_id"), pid)
        got = studio.call("GET", "/api/projects/%s" % pid)
        check("the map counts it", stage(got, "data")["count"], 1)
        check("and warns that nothing is held back",
              stage(got, "data")["state"], "warn")
        check("naming the dataset to split",
              stage(got, "data")["split_target"], data["id"])

        split = studio.call("POST", "/api/datasets/%s/split" % data["id"],
                            {"fraction": 0.25})
        check("a split can be held back", "id" in split, split)
        check("the copy stays in the project", split.get("project_id"), pid)
        check("and has a validation split",
              "validation" in (split.get("splits") or {}), split.get("splits"))
        got = studio.call("GET", "/api/projects/%s" % pid)
        check("the map stops warning once something is held back",
              stage(got, "data")["state"], "done")

        print("\nA run, started the way the wizard starts one")
        # A machine for it to be pinned to. The wizard always pins one, and
        # pinning is what made creating a run raise KeyError and return 500 --
        # the check that answers "will it run there" was writing the memory
        # estimate back to a job that did not exist yet. Without a machine in
        # the fleet that path is never taken, and this script would have
        # watched the bug go past.
        studio.add_runner("smoke-3090", vram_gb=24.0)
        no_project = studio.call("POST", "/api/jobs", {
            "name": "unfiled", "kind": "finetune_llm",
            "config": {"base_model": "sshleifer/tiny-gpt2", "dataset": "x"}})
        check("a run with no project is refused",
              no_project.get("ERROR"), 400)
        check("and says what to do about it",
              "project" in (no_project.get("detail") or "").lower(),
              no_project.get("detail"))

        job = studio.call("POST", "/api/jobs", {
            "name": "smoke run", "kind": "finetune_llm", "project_id": pid,
            "config": {"base_model": "mistralai/Mistral-7B-Instruct-v0.3",
                       "params_b": 7.0, "studio_dataset": split["id"],
                       "dataset_split": "train", "quantization": "4bit",
                       "required_runner": "run_smoke"}})
        check("a run can be created", "id" in job, job)
        if "id" not in job:
            raise SystemExit(1)
        run = studio.call("GET", "/api/jobs/%s" % job["id"])
        check("it knows its project", (run.get("project") or {}).get("id"), pid)
        check("and is queued", run["status"], "queued")
        check("with the memory it will need, in the precision it asked for",
              (run.get("config") or {}).get("estimated_vram_gb"), 4.9)

        too_big = studio.call("POST", "/api/jobs", {
            "name": "far too big", "kind": "finetune_llm", "project_id": pid,
            "config": {"base_model": "meta-llama/Llama-3.1-70B", "params_b": 70.0,
                       "studio_dataset": split["id"], "dataset_split": "train",
                       "required_runner": "run_smoke"}})
        check("a model that cannot fit the pinned machine is refused",
              too_big.get("ERROR"), 400)
        check("and the refusal says how much it needed",
              "GB" in (too_big.get("detail") or ""), too_big.get("detail"))

        inherited = studio.call("POST", "/api/jobs", {
            "name": "inherits", "kind": "finetune_llm",
            "config": {"base_model": "sshleifer/tiny-gpt2",
                       "studio_dataset": split["id"], "dataset_split": "train"}})
        check("a run inherits the project of the data it trains on",
              "id" in inherited, inherited)

        print("\nA prompt set out of the held-out rows")
        ev = studio.call("POST", "/api/evals/from-dataset",
                         {"dataset_id": split["id"], "split": "validation",
                          "limit": 5, "project_id": pid})
        check("a prompt set can be taken from the split", "id" in ev, ev)
        items = ev.get("items") or []
        check("with prompts in it", len(items) > 0, len(items))
        if items:
            check("each carrying the context it was asked in",
                  bool(items[0].get("system")), items[0])
            check("and the tool call it should make",
                  (items[0].get("expected_tool") or {}).get("name"), "set_light")
        check("the tools travel with the set",
              [t["name"] for t in (ev.get("source") or {}).get("tools", [])],
              ["set_light"])
        check("it is filed in the project", ev.get("project_id"), pid)
        check("and is marked as held out",
              (ev.get("source") or {}).get("held_out"), True)

        print("\nPublishing")
        early = studio.call("POST", "/api/library",
                            {"job_id": job["id"], "name": "too soon"})
        check("a run with no model cannot be published",
              early.get("ERROR"), 400)

        # Give the run something to publish, the way a finished run would --
        # the one place this script reaches past the API, because the only
        # other way to get an artifact is to train for an hour on a GPU.
        artifact = studio.dir / "artifacts" / ("%s.zip" % job["id"])
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(b"PK\x05\x06" + b"\0" * 18)
        import sqlite3  # noqa: PLC0415 - only this step needs the database
        db = sqlite3.connect(studio.dir / "studio.db")
        db.execute("UPDATE jobs SET status='succeeded', finished_at=?"
                   " WHERE id=?", (time.time(), job["id"]))
        db.execute("INSERT INTO artifacts (id,job_id,kind,filename,size_bytes,"
                   "created_at) VALUES (?,?,?,?,?,?)",
                   ("art_smoke", job["id"], "model", artifact.name,
                    artifact.stat().st_size, time.time()))
        db.commit()
        db.close()

        published = studio.call("POST", "/api/library", {
            "job_id": job["id"], "name": "Smoke model", "version": "v1",
            "project_id": pid})
        check("a finished run can be published", "id" in published, published)
        library = studio.call("GET", "/api/library")
        check("and is in the library", [r["name"] for r in library],
              ["Smoke model"])
        got = studio.call("GET", "/api/projects/%s" % pid)
        check("the map calls publishing done", stage(got, "publish")["state"],
              "done")
        check("and knows the best run so far",
              stage(got, "train")["count"] >= 1)

        print("\nUnfiling and deleting")
        studio.call("POST", "/api/projects/unfile",
                    {"kind": "dataset", "id": data["id"]})
        counts = studio.call("GET", "/api/projects")["unfiled"]
        check("unfiling puts a thing back on the pile", counts["datasets"], 1)
        gone = studio.call("DELETE", "/api/projects/%s" % pid)
        check("a project can be deleted", bool(gone.get("ok")), gone)
        check("its runs survive it",
              len(studio.call("GET", "/api/jobs")) >= 2)
        check("as does what it published",
              len(studio.call("GET", "/api/library")), 1)
    finally:
        studio.stop()

    print()
    if FAILED:
        print("%d check(s) failed:" % len(FAILED))
        for name in FAILED:
            print("  - %s" % name)
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
