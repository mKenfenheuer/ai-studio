"""Check the rented-GPU lifecycle against a fake RunPod.

Run it with `python scripts/check-cloud.py` from the repository root.

Money is what makes this worth a check of its own: a pod that is started when
it should not be, or not deleted when it should be, costs something every hour
until somebody notices. So this walks the whole life of a pod -- which GPU is
chosen, when one is started, when one is refused, how an idle one is given back
(its checkpoints to the studio first, then deleted, then its runner forgotten)
-- against a fake provider, and checks the brakes: the price cap, the number of
pods, the daily budget, and Secure Cloud for a run that asks for it.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
_TMP = Path(tempfile.mkdtemp(prefix="ai-studio-check-"))
os.environ["AI_STUDIO_DATA"] = str(_TMP)
os.environ["AI_STUDIO_PUBLIC_URL"] = "https://studio.example"

from controller import config, db  # noqa: E402
from controller.cloud import manager  # noqa: E402
from controller.scheduler import Fleet  # noqa: E402

FAILED: list[str] = []


def check(name: str, got: object, want: object = True) -> None:
    ok = got == want
    print("  %s  %s%s" % ("ok  " if ok else "FAIL", name, "" if ok else "   -> %r, wanted %r" % (got, want)))
    if not ok:
        FAILED.append(name)


class FakeSocket:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send_text(self, text: str) -> None:
        self.sent.append(json.loads(text))


CATALOG = {
    "SECURE": [
        {"id": "NVIDIA GeForce RTX 4090", "name": "RTX 4090", "manufacturer": "NVIDIA", "memory": 24,
         "price": {"secure": 0.69, "community": 0.34}, "secure": True, "community": True, "availability": "HIGH"},
        {"id": "NVIDIA RTX A6000", "name": "RTX A6000", "manufacturer": "NVIDIA", "memory": 48,
         "price": {"secure": 0.49, "community": 0.33}, "secure": True, "community": True, "availability": "MEDIUM"},
        {"id": "NVIDIA H100 80GB HBM3", "name": "H100 SXM", "manufacturer": "NVIDIA", "memory": 80,
         "price": {"secure": 2.69, "community": 2.09}, "secure": True, "community": True, "availability": "HIGH"},
        {"id": "NVIDIA L4", "name": "L4", "manufacturer": "NVIDIA", "memory": 24,
         "price": {"secure": 0.43, "community": 0.25}, "secure": True, "community": True, "availability": "NONE"},
    ],
}
CATALOG["COMMUNITY"] = CATALOG["SECURE"]


class FakeRunPod:
    """RunPod as the manager sees it: a catalog and pods that do what they are told."""
    pods: dict[str, dict] = {}
    created: list[dict] = []
    terminated: list[str] = []
    # One made by hand in the console, the way the first one was: an old image
    # and the join token sitting in its env for anyone with console access.
    templates: dict[str, dict] = {"tpl_hand": {
        "id": "tpl_hand", "name": "ai-studio", "image": "ghcr.io/x/ai-studio-runner:cuda-old",
        "disk": 50, "mounts": {"persistent": {"path": "/data", "size": 80}},
        "env": {"AI_STUDIO_CONTROLLER": "https://studio.example/", "AI_STUDIO_JOIN_TOKEN": "leaked"},
        "ports": [], "serverless": False, "startSsh": False, "startJupyter": False}}
    template_calls: list[str] = []

    def __init__(self, api_key: str, base: str | None = None) -> None:
        assert api_key == "rp-secret-key-1234"

    async def gpus(self, cloud: str = "SECURE") -> list[dict]:
        return CATALOG[cloud]

    async def list_templates(self) -> list[dict]:
        return [dict(t) for t in FakeRunPod.templates.values()]

    async def create_template(self, body: dict) -> dict:
        tid = "tpl_%d" % (len(FakeRunPod.templates) + 1)
        FakeRunPod.templates[tid] = {"id": tid, "serverless": False, **body}
        FakeRunPod.template_calls.append("create")
        return FakeRunPod.templates[tid]

    async def update_template(self, template_id: str, body: dict) -> dict:
        FakeRunPod.templates[template_id].update(body)
        FakeRunPod.template_calls.append("update")
        return FakeRunPod.templates[template_id]

    async def create_pod(self, **kw) -> dict:
        pid = "pod%d" % (len(FakeRunPod.created) + 1)
        FakeRunPod.created.append(kw)
        if kw.get("template_id"):
            # What RunPod does with one: its settings, the body's env merged over its own.
            t = FakeRunPod.templates[kw["template_id"]]
            kw = {**kw, "image": t["image"], "env": {**t["env"], **kw["env"]}}
        FakeRunPod.created[-1] = kw
        FakeRunPod.pods[pid] = {"id": pid, "status": "RUNNING",
                                "cost": [g for g in CATALOG[kw["cloud"]] if g["id"] == kw["gpu_id"]][0]["price"][kw["cloud"].lower()]}
        return FakeRunPod.pods[pid]

    async def get_pod(self, pod_id: str) -> dict | None:
        return FakeRunPod.pods.get(pod_id)

    async def terminate(self, pod_id: str) -> None:
        FakeRunPod.terminated.append(pod_id)
        FakeRunPod.pods.pop(pod_id, None)


manager.RunPod = FakeRunPod


def finetune(params_b: float, **cfg) -> str:
    return db.create_job("ft", "finetune_llm", {"base_model": "Qwen/Qwen3-1.7B", "params_b": params_b,
                                                 "method": "lora", "max_steps": 60, "batch_size": 2,
                                                 "grad_accum": 8, "max_seq_len": 4096, **cfg})


def main() -> int:
    config.ensure_dirs()
    db.connect()
    fleet = Fleet()
    m = manager.Manager()

    print("settings: the key is stored encrypted and never shown")
    out = manager.save_settings({"api_key": "rp-secret-key-1234", "enabled": True, "max_price_per_hour": 0.80,
                                 "max_pods": 1, "daily_cap": 5.0, "idle_minutes": 15}, "admin")
    check("key set", out["api_key_set"])
    check("only a hint of it", out["api_key_hint"], "…1234")
    check("not in the stored row", "rp-secret-key-1234" in (db.get_setting(manager.SETTING_KEY) or ""), False)
    check("not in the status", "rp-secret-key-1234" in json.dumps(m.status()), False)

    print("the pod template")
    out = asyncio.run(m.sync_template())
    check("the hand-made one is adopted by name, not duplicated", (out["template_id"], len(FakeRunPod.templates)),
          ("tpl_hand", 1))
    check("and brought in line", out["result"].startswith("updated"))
    t = FakeRunPod.templates["tpl_hand"]
    check("to the image in the settings", t["image"], manager.DEFAULTS["image"])
    check("with the join token taken out of it", "AI_STUDIO_JOIN_TOKEN" in t["env"], False)
    check("and the studio's address", t["env"]["AI_STUDIO_CONTROLLER"], "https://studio.example")
    check("remembered", manager.settings()["template_id"], "tpl_hand")
    FakeRunPod.template_calls.clear()
    out = asyncio.run(m.sync_template())
    check("left alone when it already matches", (out["result"], FakeRunPod.template_calls),
          ("already up to date", []))
    manager.save_settings({"volume_gb": 120}, "admin")
    asyncio.run(m.sync_template())
    check("a changed setting reaches it", FakeRunPod.templates["tpl_hand"]["mounts"]["persistent"]["size"], 120)
    manager.save_settings({"volume_gb": 80}, "admin")
    del FakeRunPod.templates["tpl_hand"]
    out = asyncio.run(m.sync_template())
    check("made when there is none", (out["result"], out["template_id"] in FakeRunPod.templates), ("created", True))

    print("choosing a GPU")
    s = manager.settings()
    offers = asyncio.run(manager.offers("any", refresh=True))
    small = {"id": "x", "kind": "finetune_llm", "config": {"params_b": 1.7, "method": "lora", "cloud": "any"}}
    best, _ = manager.choose(fleet, small, offers, s)
    check("fastest under the price cap", (best["name"], best["cloud"]), ("RTX 4090", "COMMUNITY"))
    secure = {**small, "config": {**small["config"], "cloud": "secure"}}
    best, _ = manager.choose(fleet, secure, offers, s)
    check("Secure Cloud only when the run asks", (best["name"], best["cloud"]), ("RTX 4090", "SECURE"))
    cheap, _ = manager.choose(fleet, small, offers, {**s, "strategy": "cheapest"})
    check("cheapest in stock when asked (the L4 is cheaper but sold out)", (cheap["name"], cheap["cloud"]),
          ("RTX A6000", "COMMUNITY"))
    check("never a GPU out of stock", all(o["name"] != "L4" for o in [best, cheap]))
    big = {"id": "y", "kind": "finetune_llm", "config": {"params_b": 32, "method": "full", "cloud": "any"}}
    none, why = manager.choose(fleet, big, offers, s)
    check("a run too big for anything under the cap gets none", none, None)
    check("and says why", "memory" in why)
    check("the H100 was over the cap all along", all(o["price"] > 0.8 for o in offers if o["name"] == "H100 SXM"))

    print("starting a pod for a run nothing can take")
    j1 = finetune(1.7, cloud="secure")
    asyncio.run(m.reconcile(fleet))
    check("one pod requested", len(FakeRunPod.created), 1)
    req = FakeRunPod.created[0]
    check("on Secure Cloud, as the run asked", req["cloud"], "SECURE")
    check("from the studio's template", req.get("template_id"), manager.settings()["template_id"])
    check("the runner image", req["image"], manager.DEFAULTS["image"])
    check("it dials home with the join token", req["env"]["AI_STUDIO_JOIN_TOKEN"], config.join_token())
    check("to the public address", req["env"]["AI_STUDIO_CONTROLLER"], "https://studio.example")
    pod_row = manager.pods()[0]
    check("under a name that ties the runner to the pod", req["env"]["AI_STUDIO_RUNNER_NAME"], pod_row["name"])
    check("the run is told", any("RunPod" in (l.get("line") or "") for l in db.get_logs(j1)))

    print("the brakes")
    j2 = finetune(1.7, cloud="any")
    asyncio.run(m.reconcile(fleet))
    check("no second pod beyond max_pods", len(FakeRunPod.created), 1)
    check("the second run says why", "1 of 1" in m.note_for(j2))
    j_never = finetune(1.7, cloud="never")
    asyncio.run(m.reconcile(fleet))
    check("a run that says never is never offered a pod", m.note_for(j_never), "")

    print("the pod's runner connects")
    rid = "run_pod1"
    db.upsert_runner(rid, pod_row["name"], {"backend": "cuda", "vram_gb": 48, "quantization": {"4bit": True},
                                            "modalities": ["text", "vision"]})
    fleet.connections[rid] = FakeSocket()
    asyncio.run(m.reconcile(fleet))
    p = manager.pod(pod_row["id"])
    check("linked to its runner", p["runner_id"], rid)
    check("running", p["status"], "running")
    check("kept for training, not serving", db.runner_role(db.get_runner(rid)), "training")

    print("idle with nothing to do: checkpoints to the studio, then deleted")
    for jid in (j1, j2, j_never):
        db.set_job_status(jid, "cancelled")
    unfinished = finetune(1.7, cloud="never")
    db.set_job_status(unfinished, "failed")
    db.ex("UPDATE jobs SET status='queued' WHERE id=?", (unfinished,))   # waiting for its checkpoint's machine
    db.ex("UPDATE jobs SET status='running' WHERE id=?", (unfinished,))
    db.set_checkpoint(unfinished, 36, rid)
    fleet.checkpoints[rid] = {unfinished}
    db.ex("UPDATE cloud_pods SET last_busy_at=? WHERE id=?", (time.time() - 3600, pod_row["id"]))
    asyncio.run(m.reconcile(fleet))
    p = manager.pod(pod_row["id"])
    check("draining", p["status"], "draining")
    check("no new work while draining", rid in fleet.draining)
    sent = [f for f in fleet.connections[rid].sent if f.get("type") == "checkpoint_upload"]
    check("asked to hand over the checkpoint", [f.get("job_id") for f in sent], [unfinished])
    check("not deleted while the upload runs", FakeRunPod.terminated, [])
    # The runner's PUT lands (see put_checkpoint): the studio now holds it.
    db.set_checkpoint(unfinished, 36, manager.STUDIO_HOLDER)
    fleet.checkpoints[rid].discard(unfinished)
    asyncio.run(m.reconcile(fleet))
    check("then deleted", FakeRunPod.terminated, ["pod1"])
    check("terminated in the ledger", manager.pod(pod_row["id"])["status"], "terminated")
    check("its runner forgotten", db.get_runner(rid), None)
    check("the checkpoint stays the studio's", db.get_job(unfinished)["checkpoint_runner"], "studio")

    print("a checkpoint the studio holds goes to any machine, with the checkpoint")
    db.set_job_status(unfinished, "queued")
    job = db.get_job(unfinished)
    check("eligible anywhere", fleet._eligible(job, ["run_other"]), ["run_other"])

    print("spend and the daily cap")
    check("spend was accrued", manager.spend() > 0)
    db.ex("INSERT INTO cloud_spend (day, usd) VALUES (?, 100) ON CONFLICT(day) DO UPDATE SET usd=100",
          (manager._today(),))
    j3 = finetune(1.7, cloud="any")
    FakeRunPod.created.clear()
    asyncio.run(m.reconcile(fleet))
    check("no pod once the budget is spent", FakeRunPod.created, [])
    check("and the run says so", "budget" in m.note_for(j3))

    print("a paused run goes back in the queue with its checkpoint")
    fleet.connections["run_x"] = FakeSocket()
    db.upsert_runner("run_x", "x", {"backend": "cuda", "vram_gb": 24})
    jp = finetune(1.7)
    db.ex("UPDATE jobs SET status='running', runner_id='run_x' WHERE id=?", (jp,))
    fleet.busy["run_x"] = jp
    asyncio.run(fleet.handle_runner_message("run_x", {"type": "job_preempted", "job_id": jp, "step": 40}))
    jr = db.get_job(jp)
    check("queued again", jr["status"], "queued")
    check("with its checkpoint", (jr["checkpoint_step"], jr["checkpoint_runner"]), (40, "run_x"))
    check("the machine is free", "run_x" in fleet.busy, False)

    print("a machine that comes back after the studio gave up waiting gets its run back")
    db.clear_checkpoint(jp)            # what _eligible does after CHECKPOINT_WAIT_S
    fleet.gave_up_waiting.add(jp)
    fleet.note_checkpoints("run_x", [jp], [{"job_id": jp, "step": 58}])
    jr = db.get_job(jp)
    check("bound to it again, at its step", (jr["checkpoint_step"], jr["checkpoint_runner"]), (58, "run_x"))
    check("and the run says so", any("back with" in (l.get("line") or "") for l in db.get_logs(jp)))
    db.set_job_status(jp, "running")
    db.clear_checkpoint(jp)
    fleet.note_checkpoints("run_x", [jp], [{"job_id": jp, "step": 58}])
    check("not once it has started over elsewhere", db.get_job(jp)["checkpoint_step"], 0)

    print("a run sent to a rented GPU goes to one, even with a machine of the studio free")
    from controller.scheduler import CLOUD_PIN
    jc = finetune(1.7, required_runner=CLOUD_PIN, cloud="secure")
    job = db.get_job(jc)
    local = {"backend": "cuda", "vram_gb": 48, "_id": "run_x"}
    check("the studio's own machine may not take it", fleet.can_run(job, local)[0], False)
    check("so a pod is wanted for it", m._wants_cloud(fleet, job, manager.settings()))
    db.ex("INSERT INTO cloud_pods (id, provider, name, gpu_id, cloud, price_per_hour, status, created_at) "
          "VALUES ('cp_pin', 'runpod', 'ai-studio-cp_pin', 'NVIDIA RTX A6000', 'SECURE', 0.49, 'running', ?)",
          (time.time(),))
    db.upsert_runner("run_pin", "ai-studio-cp_pin", {"backend": "cuda", "vram_gb": 48})
    check("and the pod's runner may, before it is even linked",
          fleet.can_run(job, {"backend": "cuda", "vram_gb": 48, "_id": "run_pin"})[0])

    print()
    if FAILED:
        print("%d check(s) failed:" % len(FAILED))
        for f in FAILED:
            print("  - " + f)
        return 1
    print("all cloud checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
