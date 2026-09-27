"""Rented GPUs: when to start one, which one, and when to give it back.

The studio's own machines come first. Only when a queued GPU run has no machine
in the fleet that could take it -- none online at all, not merely all busy --
does the studio rent one: the fastest (or cheapest) RunPod GPU the run fits on,
under the per-hour price cap, within the number of pods allowed at once and the
day's budget. The pod starts the ordinary CUDA runner image, which dials in with
the join token like any machine, under a name that ties it to its pod record.

A pod is not kept. Once it has been idle for a while -- and nothing queued could
use it -- it is drained: whatever exists only on its disk goes to the studio
first (finished models already do, at the end of every run; the checkpoints of
unfinished runs are uploaded on request), then the pod is terminated and its
runner forgotten. Its models and caches are not copied: everything else on it can
be downloaded again, and paying for storage to keep it would make an idle pod
cost money after all.

Money is watched from three sides: a price cap per GPU-hour, a cap on pods at
once, and a daily spend cap. Spend is accrued here from each pod's hourly price
every time the loop runs -- an estimate, and on the high side (a pod is counted
from the moment it is requested). With the cap reached no new pod starts, and,
unless told otherwise, running ones are stopped: their runs checkpointed and
handed to the studio, the pods deleted.
"""
from __future__ import annotations

import asyncio
import copy
import datetime
import json
import os
import time
import uuid
from typing import Any

from .. import auth, config, db
from .runpod import RunPod, RunPodError

SETTING_KEY = "cloud_gpus"
PROVIDER = "runpod"
NAME_PREFIX = "ai-studio-"
STUDIO_HOLDER = "studio"          # jobs.checkpoint_runner when the studio holds it

DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "max_price_per_hour": 0.80,
    "max_pods": 1,
    "daily_cap": 10.0,
    # When the day's cap is reached: stop the running pods too (their runs are
    # checkpointed and continue later), or only refuse to start new ones.
    "hard_cap": True,
    "idle_minutes": 15,
    "max_hours_per_pod": 12,
    # A run's own setting wins; this is what a run that says nothing gets.
    "default_cloud": "secure",       # never | any | secure
    "strategy": "fastest",           # fastest | cheapest
    "allowed_gpus": [],              # GPU type ids; empty means any
    "image": "ghcr.io/mkenfenheuer/ai-studio-runner:cuda",
    "disk_gb": 40,
    "volume_gb": 80,
    "controller_url": "",            # empty: AI_STUDIO_PUBLIC_URL
    # The RunPod pod template the studio keeps in step with these settings and
    # starts its pods from. Found by name, so one made by hand is adopted.
    "template_name": "ai-studio",
}
MOUNT = "/data"
POLICIES = ("never", "any", "secure")
GPU_KINDS = {"finetune_llm", "pretrain_llm", "finetune_vision_cls", "finetune_vlm",
             "finetune_asr", "finetune_tts", "finetune_diffusion"}
ACTIVE = ("starting", "running", "draining")
STARTUP_TIMEOUT_S = 25 * 60       # requested to connected: image pull on a cold host
DRAIN_TIMEOUT_S = 45 * 60         # a checkpoint upload that has not finished by then
OFFER_TTL_S = 60

# Rough training throughput relative to an RTX 3090, for "fastest" and for the
# estimates. Bandwidth and bf16 tensor rate, not a benchmark -- a ranking, not a
# promise. A GPU not listed is scored from its price.
SPEED = {
    "B200": 7.0, "H200": 4.8, "H100 SXM": 4.2, "H100 NVL": 3.8, "H100 PCIe": 3.3, "H100": 3.5,
    "A100 SXM": 2.7, "A100 80GB": 2.6, "A100": 2.4, "MI300X": 4.0, "L40S": 2.5, "L40": 1.9,
    "RTX 6000 Ada": 2.3, "RTX PRO 6000": 3.4, "RTX 5090": 2.8, "RTX 4090": 2.1, "RTX 4080": 1.5,
    "RTX 5080": 1.6, "RTX A6000": 1.25, "A40": 1.15, "RTX 3090": 1.0, "RTX A5000": 0.85,
    "RTX A4500": 0.75, "RTX 4000 Ada": 0.7, "RTX A4000": 0.55, "L4": 0.5, "RTX 2000 Ada": 0.4,
    "V100": 0.7, "A10": 0.65, "A30": 0.9, "T4": 0.25,
}
# Measured: the lab's RX 6900 XT (~0.9 on this scale) fine-tuned a 1.7B model at
# about 640 tokens/s. Tokens per second ≈ K × speed / billions of parameters.
TOKENS_PER_SECOND_K = 640 * 1.7 / 0.9


# ------------------------------------------------------------------ settings

def settings() -> dict:
    raw = db.get_setting(SETTING_KEY)
    stored = json.loads(raw) if raw else {}
    out = {**DEFAULTS, **{k: v for k, v in stored.items() if k in DEFAULTS}}
    key = _api_key(stored)
    out["api_key_set"] = bool(key)
    out["api_key_hint"] = ("…" + key[-4:]) if key else ""
    out["template_id"] = stored.get("template_id") or ""
    out["template_synced_at"] = stored.get("template_synced_at") or 0
    return out


def _api_key(stored: dict | None = None) -> str | None:
    if stored is None:
        raw = db.get_setting(SETTING_KEY)
        stored = json.loads(raw) if raw else {}
    enc = stored.get("api_key_enc")
    return auth.decrypt_secret(enc) if enc else None


def save_settings(payload: dict, user_id: str | None) -> dict:
    raw = db.get_setting(SETTING_KEY)
    stored = json.loads(raw) if raw else {}
    for k, default in DEFAULTS.items():
        if k not in payload:
            continue
        v = payload[k]
        if isinstance(default, bool):
            v = bool(v)
        elif isinstance(default, int):
            v = int(v)
            if v < 0:
                raise ValueError("%s cannot be negative" % k)
        elif isinstance(default, float):
            v = float(v)
            if v < 0:
                raise ValueError("%s cannot be negative" % k)
        elif isinstance(default, list):
            v = [str(x) for x in (v or [])]
        else:
            v = str(v or "").strip()
        if k == "default_cloud" and v not in POLICIES:
            raise ValueError("default_cloud must be one of %s" % ", ".join(POLICIES))
        if k == "strategy" and v not in ("fastest", "cheapest"):
            raise ValueError("strategy must be fastest or cheapest")
        if k in ("disk_gb",) and v < 20:
            raise ValueError("The container disk needs at least 20 GB: a run's merged model is written there.")
        if k in ("volume_gb",) and v < 20:
            raise ValueError("The /data volume needs at least 20 GB for models and checkpoints.")
        if k == "template_name" and not v:
            raise ValueError("The template needs a name.")
        stored[k] = v
    if payload.get("api_key"):
        stored["api_key_enc"] = auth.encrypt_secret(str(payload["api_key"]).strip())
    if payload.get("clear_api_key"):
        stored.pop("api_key_enc", None)
    db.set_setting(SETTING_KEY, json.dumps(stored), user_id)
    return settings()


def _remember(**fields: Any) -> None:
    """Studio-kept facts beside the settings (the template's id), not settings."""
    raw = db.get_setting(SETTING_KEY)
    stored = json.loads(raw) if raw else {}
    stored.update(fields)
    db.set_setting(SETTING_KEY, json.dumps(stored), None)


# ---------------------------------------------------------------- template

def controller_url(s: dict) -> str:
    return (s["controller_url"] or config.PUBLIC_URL or "").rstrip("/")


def template_body(s: dict) -> dict:
    """The pod template these settings describe.

    Everything a pod of this studio has in common, and nothing secret: a
    template is shown in the RunPod console, so the join token goes with each
    pod instead, as do the runner's name and the pod's id. SSH and Jupyter are
    off -- the runner dials out, and nothing on the pod needs to be reached.
    """
    return {
        "name": s["template_name"],
        "image": s["image"],
        "disk": int(s["disk_gb"]),
        "mounts": {"persistent": {"size": int(s["volume_gb"]), "path": MOUNT}},
        "env": {"AI_STUDIO_CONTROLLER": controller_url(s)},
        "ports": [],
        "startSsh": False,
        "startJupyter": False,
    }


def template_drift(have: dict, want: dict) -> list[str]:
    """Which of the template's fields differ from what the settings describe."""
    out = []
    for k in ("name", "image", "disk", "startSsh", "startJupyter"):
        if have.get(k) != want[k]:
            out.append(k)
    if ((have.get("mounts") or {}).get("persistent") or {}) != want["mounts"]["persistent"]:
        out.append("mounts")
    if (have.get("env") or {}) != want["env"]:
        out.append("env")
    if list(have.get("ports") or []) != want["ports"]:
        out.append("ports")
    return out


async def ensure_template(rp: RunPod, s: dict) -> tuple[str, str]:
    """Create or update the studio's template; returns (id, what was done).

    Found by the id remembered from last time, else by name -- so a template
    somebody made by hand in the console under that name is taken over and
    brought in line rather than duplicated. Only its own fields are sent: a
    template that already matches is left alone.
    """
    want = template_body(s)
    templates = await rp.list_templates()
    have = next((t for t in templates if s.get("template_id") and t.get("id") == s["template_id"]), None) \
        or next((t for t in templates if t.get("name") == want["name"] and not t.get("serverless")), None)
    if have is None:
        made = await rp.create_template(want)
        tid, done = (made or {}).get("id") or "", "created"
    else:
        tid = have["id"]
        drift = template_drift(have, want)
        if drift:
            await rp.update_template(tid, want)
            done = "updated (%s)" % ", ".join(drift)
        else:
            done = "already up to date"
    _remember(template_id=tid, template_synced_at=time.time())
    return tid, done


def policy_of(job: dict, s: dict | None = None) -> str:
    p = (job.get("config") or {}).get("cloud")
    return p if p in POLICIES else (s or settings())["default_cloud"]


# -------------------------------------------------------------- the ledger

def _today() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")


def spend(day: str | None = None) -> float:
    row = db.q1("SELECT usd FROM cloud_spend WHERE day=?", (day or _today(),))
    return float(row["usd"]) if row else 0.0


def spend_history(days: int = 14) -> list[dict]:
    return db.q("SELECT day, usd FROM cloud_spend ORDER BY day DESC LIMIT ?", (days,))


def _add_spend(usd: float) -> None:
    db.ex("INSERT INTO cloud_spend (day, usd) VALUES (?, ?) "
          "ON CONFLICT(day) DO UPDATE SET usd = usd + excluded.usd", (_today(), usd))


def pods(include_ended: bool = False, limit: int = 50) -> list[dict]:
    if include_ended:
        return db.q("SELECT * FROM cloud_pods ORDER BY created_at DESC LIMIT ?", (limit,))
    return db.q("SELECT * FROM cloud_pods WHERE status IN (%s) ORDER BY created_at"
                % ",".join("?" * len(ACTIVE)), ACTIVE)


def pod(pod_id: str) -> dict | None:
    return db.q1("SELECT * FROM cloud_pods WHERE id=?", (pod_id,))


def _update(pod_id: str, **fields: Any) -> None:
    sets = ", ".join("%s=?" % k for k in fields)
    db.ex("UPDATE cloud_pods SET %s WHERE id=?" % sets, (*fields.values(), pod_id))


# ------------------------------------------------------------------ offers

def speed_of(name: str, price: float) -> float:
    for key in sorted(SPEED, key=len, reverse=True):
        if key.lower() in (name or "").lower():
            return SPEED[key]
    return round(max(price, 0.05) / 0.35, 2)        # unknown: price as a proxy


_offer_cache: dict[str, tuple[float, list[dict]]] = {}


async def offers(policy: str = "any", *, refresh: bool = False) -> list[dict]:
    """What can be rented right now, one entry per GPU type and cloud."""
    key = _api_key()
    if not key:
        return []
    clouds = ["SECURE"] if policy == "secure" else ["SECURE", "COMMUNITY"]
    out: list[dict] = []
    rp = RunPod(key)
    for cloud in clouds:
        cached = _offer_cache.get(cloud)
        if cached and not refresh and time.time() - cached[0] < OFFER_TTL_S:
            gpus = cached[1]
        else:
            gpus = await rp.gpus(cloud)
            _offer_cache[cloud] = (time.time(), gpus)
        for g in gpus:
            if g.get("manufacturer") not in (None, "NVIDIA"):   # the image is CUDA
                continue
            price = float(((g.get("price") or {}).get(cloud.lower())) or 0)
            if price <= 0 or not g.get(cloud.lower(), True):
                continue
            speed = speed_of(g.get("name") or g["id"], price)
            out.append({"id": g["id"], "name": g.get("name") or g["id"], "vram_gb": g.get("memory"),
                        "cloud": cloud, "price": price, "availability": g.get("availability") or "UNKNOWN",
                        "speed": speed, "usd_per_speed": round(price / speed, 3)})
    return out


def caps_for(offer: dict) -> dict:
    """What a runner on this GPU will report, worked out the way the runner does.

    The ceilings follow runner/capabilities.py (_estimate_max_model,
    _estimate_max_scratch) for a CUDA card with fused attention, 4-bit and the
    8-bit optimiser -- so the wizard plans a cloud run exactly as it would plan
    one for the same card connected by hand.
    """
    vram = float(offer.get("vram_gb") or 0)
    return {"backend": "cuda", "device_name": offer["name"], "vram_gb": vram,
            "quantization": {"4bit": True, "8bit": True, "optim_8bit": True},
            "attention": {"flash": False, "mem_efficient": True},
            "dtypes": {"bfloat16": True, "float16": True}, "recommended_dtype": "bfloat16",
            "modalities": ["text", "vision"], "kinds": None, "_id": None,
            "max_finetune_params_b": round(max(0.0, vram - 2.0) * 0.85 / 0.5, 1) or None,
            "max_scratch_params_m": round(max(0.0, vram - 3.0) * 1024 ** 3 / 12 / 1e6) or None,
            "max_recommended_seq_len": 8192 if vram >= 40 else 4096,
            "ram_gb": 32, "cpu_cores": 8}


CLOUD_RUNNER_ID = "cloud"
_virtual: dict | None = None


async def virtual_runner() -> dict | None:
    """The cloud as one machine the wizard can choose: the largest GPU rentable now.

    Largest, not fastest: the question the wizard asks a machine is "what can
    I train here", and memory answers it. The run is then not pinned -- the
    manager rents whichever GPU under the limits fits it best when it starts.
    None when renting is off, there is no key, or nothing is in stock.
    """
    global _virtual
    s = settings()
    if not (s["enabled"] and s["api_key_set"]) or s["default_cloud"] == "never":
        _virtual = None
        return None
    try:
        cands = await offers(s["default_cloud"])
    except RunPodError:
        return _virtual
    usable = [o for o in cands if o["price"] <= s["max_price_per_hour"] and o["availability"] != "NONE"
              and (not s["allowed_gpus"] or o["id"] in s["allowed_gpus"])]
    if not usable:
        _virtual = None
        return None
    best = max(usable, key=lambda o: (o.get("vram_gb") or 0, -o["price"]))
    _virtual = {"id": CLOUD_RUNNER_ID, "name": "Cloud GPU (RunPod)", "status": "online", "cloud": True,
                "capabilities": caps_for(best),
                "offer": {"name": best["name"], "vram_gb": best.get("vram_gb"), "price": best["price"],
                          "cloud": best["cloud"], "max_price": s["max_price_per_hour"],
                          "policy": s["default_cloud"]}}
    return _virtual


def virtual_runner_cached() -> dict | None:
    return _virtual


def fits(fleet: Any, job: dict, offer: dict) -> tuple[bool, str]:
    # A copy without an id: can_run writes its memory estimate into the run it
    # checks, and a GPU nobody has rented yet must not leave one behind.
    probe = {**job, "config": copy.deepcopy(job.get("config") or {})}
    probe.pop("id", None)
    probe["config"].pop("required_runner", None)
    return fleet.can_run(probe, caps_for(offer))


def estimate(job: dict, offer: dict) -> dict:
    """Hours and dollars for this run on this GPU. Rough, and labelled as such."""
    cfg = job.get("config") or {}
    params_b = float(cfg.get("params_b") or 1.5)
    seq = int(cfg.get("max_seq_len") or 1024)
    per_step = int(cfg.get("batch_size") or 1) * int(cfg.get("grad_accum") or 1)
    steps = int(cfg.get("max_steps") or 0)
    if not steps:
        rows = (((cfg.get("dataset_fingerprint") or {}).get("splits") or {}).get(cfg.get("dataset_split") or "train")
                or (cfg.get("dataset_fingerprint") or {}).get("rows") or 1000)
        steps = max(1, int(float(cfg.get("epochs") or 1) * rows / max(per_step, 1)))
    tokens = steps * per_step * seq * 0.6        # sequences are rarely full
    tps = TOKENS_PER_SECOND_K * offer["speed"] / max(params_b, 0.3)
    hours = tokens / tps / 3600 + 0.25           # + start, model download, upload
    return {"hours": round(hours, 2), "usd": round(hours * offer["price"], 2),
            "basis": "%d steps × %d sequences of up to %d tokens, %.1fB parameters" % (steps, per_step, seq, params_b)}


def choose(fleet: Any, job: dict, candidates: list[dict], s: dict) -> tuple[dict | None, str]:
    policy = policy_of(job, s)
    usable = [o for o in candidates
              if (policy != "secure" or o["cloud"] == "SECURE")
              and o["price"] <= s["max_price_per_hour"]
              and o["availability"] != "NONE"
              and (not s["allowed_gpus"] or o["id"] in s["allowed_gpus"])]
    if not usable:
        return None, ("no RunPod GPU is in stock under $%.2f/h%s"
                      % (s["max_price_per_hour"], " on Secure Cloud" if policy == "secure" else ""))
    fitting = [o for o in usable if fits(fleet, job, o)[0]]
    if not fitting:
        return None, "no RunPod GPU under $%.2f/h has the memory this run needs" % s["max_price_per_hour"]
    if s["strategy"] == "cheapest":
        fitting.sort(key=lambda o: (o["price"], -o["speed"]))
    else:
        fitting.sort(key=lambda o: (-o["speed"], o["price"]))
    return fitting[0], ""


# ------------------------------------------------------------------ the loop

class Manager:
    def __init__(self) -> None:
        self.notes: dict[str, str] = {}             # job id -> why it waits on the cloud
        self.upload_requested: dict[tuple[str, str], float] = {}
        self.last_error = ""
        self._lock = asyncio.Lock()
        self._fleet_ref: Any = None

    def note_for(self, job_id: str) -> str:
        return self.notes.get(job_id, "")

    def status(self) -> dict:
        s = settings()
        return {"settings": s, "spend_today": round(spend(), 2), "pods": pods(include_ended=True, limit=30),
                "history": spend_history(), "last_error": self.last_error}

    async def reconcile(self, fleet: Any) -> bool:
        """One pass. Returns whether anything changed."""
        async with self._lock:
            self._fleet_ref = fleet
            s = settings()
            key = _api_key()
            if not key:
                return False
            rp = RunPod(key)
            changed = await self._refresh(rp, fleet)
            changed |= await self._link_and_drain(rp, fleet, s)
            if s["enabled"]:
                changed |= await self._start_for_waiting(rp, fleet, s)
            self._forget_finished_checkpoints()
            return changed

    # -- 1. what the provider says, and what it has cost
    async def _refresh(self, rp: RunPod, fleet: Any) -> bool:
        changed, now = False, time.time()
        for p in pods():
            if p["provider_id"]:
                try:
                    info = await rp.get_pod(p["provider_id"])
                except RunPodError as e:
                    self.last_error = str(e)
                    info = {"status": "UNKNOWN"}
            else:
                info = None
            usd = float(p["price_per_hour"]) * max(0.0, now - float(p["accrued_at"] or p["created_at"])) / 3600
            _add_spend(usd)
            fields: dict[str, Any] = {"spent": float(p["spent"]) + usd, "accrued_at": now}
            st = (info or {}).get("status")
            if info is None or st == "TERMINATED":
                fields.update(status="terminated", ended_at=now)
                changed = True
            elif st in ("EXITED", "ERROR") and p["status"] != "draining":
                fields["note"] = "RunPod reports the pod %s; it is being removed." % st.lower()
                fields["status"] = "draining"
                changed = True
            elif st == "RUNNING" and p["status"] == "starting" and not p["runner_id"]:
                pass                    # container up, runner not connected yet
            if (cost := (info or {}).get("cost")) and float(cost) > 0:
                fields["price_per_hour"] = float(cost)
            _update(p["id"], **fields)
        return changed

    # -- 2. which runner is which pod, and when a pod is given back
    async def _link_and_drain(self, rp: RunPod, fleet: Any, s: dict) -> bool:
        changed, now = False, time.time()
        over_cap = s["hard_cap"] and spend() >= s["daily_cap"] > 0
        queued = db.queued_jobs()
        for p in pods():
            runner = self._runner_for(p)
            if runner and not p["runner_id"]:
                _update(p["id"], runner_id=runner["id"], connected_at=now, status="running", last_busy_at=now)
                # A rented GPU is for training; serving would pin it busy forever.
                db.set_runner_role(runner["id"], "training")
                p = pod(p["id"]) or p
                changed = True
            rid = p["runner_id"]
            online = bool(rid and rid in fleet.connections)
            busy_job = fleet.busy.get(rid) if rid else None
            if busy_job:
                _update(p["id"], last_busy_at=now)
            if p["status"] == "starting" and now - float(p["created_at"]) > STARTUP_TIMEOUT_S:
                await self._terminate(rp, p, "It never connected to the studio (image pull or host problem).")
                changed = True
                continue
            if p["status"] != "draining":
                reason = ""
                lifetime_h = (now - float(p["created_at"])) / 3600
                if over_cap:
                    reason = "The daily budget of $%.2f is reached." % s["daily_cap"]
                elif s["max_hours_per_pod"] and lifetime_h > s["max_hours_per_pod"]:
                    reason = "It has run for %.1f hours, the most one pod may." % lifetime_h
                elif (online and not busy_job
                      and now - float(p["last_busy_at"] or p["connected_at"] or now) > s["idle_minutes"] * 60
                      and not any(self._could_take(fleet, j, rid) for j in queued
                                  if not self._held_elsewhere(fleet, j, rid))):
                    reason = "Idle for %d minutes with nothing queued it could run." % s["idle_minutes"]
                if reason:
                    if busy_job:
                        # Paused, not cancelled: the run goes back in the queue
                        # with its checkpoint, which the drain below moves to
                        # the studio, and the next machine carries on from it.
                        await fleet.send_to_runner(rid, {"type": "job_preempt", "job_id": busy_job})
                        db.add_log(busy_job, "Pausing on the rented GPU: %s The run keeps its "
                                   "progress and continues on the next machine." % reason, "warn")
                    _update(p["id"], status="draining", note=reason)
                    p = pod(p["id"]) or p
                    changed = True
            if p["status"] == "draining":
                if rid:
                    fleet.draining.add(rid)
                changed |= await self._drain(rp, fleet, p)
        return changed

    def _runner_for(self, p: dict) -> dict | None:
        if p["runner_id"]:
            return db.get_runner(p["runner_id"])
        for r in db.list_runners():
            if r.get("name") == p["name"]:
                return r
        return None

    @staticmethod
    def _held_elsewhere(fleet: Any, job: dict, runner_id: str) -> bool:
        """A run waiting for the machine that holds its checkpoint is not this pod's work."""
        holder = job.get("checkpoint_runner")
        return bool(holder and holder not in (runner_id, STUDIO_HOLDER) and holder in fleet.connections)

    @staticmethod
    def _could_take(fleet: Any, job: dict, runner_id: str) -> bool:
        runner = db.get_runner(runner_id)
        if not runner:
            return False
        caps = dict(runner["capabilities"] or {})
        caps["_id"] = runner_id
        probe = {**job, "config": copy.deepcopy(job.get("config") or {})}
        probe.pop("id", None)
        return fleet.can_run(probe, caps)[0]

    async def _drain(self, rp: RunPod, fleet: Any, p: dict) -> bool:
        """Move what only this pod has to the studio, then delete the pod."""
        rid, now = p["runner_id"], time.time()
        if rid and rid in fleet.connections:
            if fleet.busy.get(rid):
                return False            # still saving the run it was stopped in
            held = [jid for jid in sorted(fleet.checkpoints.get(rid, set()))
                    if (j := db.get_job(jid)) and j["status"] not in ("succeeded", "failed", "cancelled")
                    and j.get("checkpoint_runner") != STUDIO_HOLDER]
            if held:
                for jid in held:
                    if (rid, jid) not in self.upload_requested:
                        self.upload_requested[(rid, jid)] = now
                        await fleet.send_to_runner(rid, {"type": "checkpoint_upload", "job_id": jid})
                        db.add_log(jid, "Moving this run's checkpoint from the rented GPU to the studio.")
                oldest = min(self.upload_requested[(rid, jid)] for jid in held)
                if now - oldest < DRAIN_TIMEOUT_S:
                    return False        # uploads still running
                for jid in held:
                    db.add_log(jid, "The checkpoint did not reach the studio in time; the run "
                               "will start again from its beginning.", "warn")
        await self._terminate(rp, p, p.get("note") or "Drained.")
        return True

    async def _terminate(self, rp: RunPod, p: dict, note: str) -> None:
        if p["provider_id"]:
            try:
                await rp.terminate(p["provider_id"])
            except RunPodError as e:
                self.last_error = "Could not delete pod %s: %s" % (p["name"], e)
                return                  # tried again next pass; it keeps accruing
        _update(p["id"], status="terminated", ended_at=time.time(), note=note)
        if rid := p["runner_id"]:
            self._fleet_ref and self._fleet_ref.draining.discard(rid)
            if not db.jobs_depending_on_runner(rid):
                db.delete_runner(rid)

    # -- 3. starting pods for runs nothing else can take
    async def _start_for_waiting(self, rp: RunPod, fleet: Any, s: dict) -> bool:
        active = pods()
        # Pods not yet connected are spoken for by the run they were started for.
        reserved = {p["job_id"] for p in active if p["status"] == "starting" and p["job_id"]}
        changed = False
        self.notes = {}
        starting = {p["job_id"]: p for p in active if p["status"] == "starting" and p["job_id"]}
        for job in fleet.fair_order(db.queued_jobs()):
            if job["id"] in starting:
                sp = starting[job["id"]]
                self.notes[job["id"]] = ("a rented %s (%s cloud, $%.2f/h) is starting for this run"
                                         % (sp["gpu_name"] or sp["gpu_id"], sp["cloud"].lower(), sp["price_per_hour"]))
                continue
            if not self._wants_cloud(fleet, job, s):
                continue
            if len(active) >= s["max_pods"]:
                self.notes[job["id"]] = "waiting for a rented GPU: %d of %d allowed pods are in use" % (len(active), s["max_pods"])
                continue
            try:
                candidates = await offers(policy_of(job, s))
            except RunPodError as e:
                self.last_error = str(e)
                self.notes[job["id"]] = "RunPod: %s" % e
                break
            offer, why = choose(fleet, job, candidates, s)
            if not offer:
                self.notes[job["id"]] = why
                continue
            left = s["daily_cap"] - spend()
            if s["daily_cap"] > 0 and left < offer["price"]:
                self.notes[job["id"]] = ("waiting for a rented GPU: today's budget of $%.2f has $%.2f left, "
                                         "less than an hour of a %s" % (s["daily_cap"], max(left, 0), offer["name"]))
                continue
            p = await self._create(rp, s, offer, job=job, user_id=job.get("owner_id"))
            if p:
                active.append(p)
                reserved.add(job["id"])
                changed = True
        return changed

    def _wants_cloud(self, fleet: Any, job: dict, s: dict) -> bool:
        if job["kind"] not in GPU_KINDS or policy_of(job, s) == "never":
            return False
        cfg = job.get("config") or {}
        if cfg.get("required_runner") not in (None, "", CLOUD_RUNNER_ID):
            return False                # pinned to a machine of the studio's own
        holder = job.get("checkpoint_runner")
        if holder and holder != STUDIO_HOLDER and holder in fleet.connections:
            return False                # its own machine is back; it waits for it
        # Any machine of the fleet that is online and could run it -- busy or
        # not -- means waiting is cheaper than renting.
        return not any(self._could_take(fleet, job, rid) for rid in fleet.connections
                       if rid not in fleet.draining)

    async def _create(self, rp: RunPod, s: dict, offer: dict, *, job: dict | None = None,
                      user_id: str | None = None, reason: str = "") -> dict | None:
        pid = "cp_" + uuid.uuid4().hex[:10]
        name = NAME_PREFIX + pid
        url = controller_url(s)
        if not url:
            self.last_error = "No public address for the studio: set AI_STUDIO_PUBLIC_URL or the controller URL in the cloud settings."
            return None
        now = time.time()
        est = estimate(job, offer) if job else None
        db.ex("INSERT INTO cloud_pods (id, provider, name, gpu_id, gpu_name, vram_gb, cloud, price_per_hour, "
              "status, job_id, reason, created_at, accrued_at, created_by) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
              (pid, PROVIDER, name, offer["id"], offer["name"], offer.get("vram_gb"), offer["cloud"],
               offer["price"], "starting", (job or {}).get("id"), reason or ("for run %s" % job["id"] if job else "started by hand"),
               now, now, user_id))
        # The template is brought in line first, so a pod always starts from
        # what the settings say now. If RunPod will not have the template, the
        # pod is spelled out in full instead: a template is a convenience, and
        # a run waiting on one is not.
        try:
            template_id, _ = await ensure_template(rp, s)
        except RunPodError as e:
            template_id = ""
            self.last_error = "The pod template could not be updated (%s); starting the pod without it." % e
        try:
            info = await rp.create_pod(
                name=name, gpu_id=offer["id"], cloud=offer["cloud"], template_id=template_id or None,
                image=s["image"], disk_gb=s["disk_gb"], volume_gb=s["volume_gb"], mount=MOUNT,
                env={"AI_STUDIO_CONTROLLER": url, "AI_STUDIO_JOIN_TOKEN": config.join_token(),
                     "AI_STUDIO_RUNNER_NAME": name, "AI_STUDIO_CLOUD_POD": pid})
        except RunPodError as e:
            _update(pid, status="failed", ended_at=time.time(), note=str(e))
            self.last_error = str(e)
            if job:
                self.notes[job["id"]] = "RunPod could not start a %s: %s" % (offer["name"], e)
            return None
        _update(pid, provider_id=(info or {}).get("id"),
                price_per_hour=float((info or {}).get("cost") or offer["price"]))
        if job:
            db.add_log(job["id"], "No machine of the studio can take this run, so a RunPod %s (%s cloud, "
                       "$%.2f/h) is starting for it. Rough estimate: %.1f h, about $%.2f."
                       % (offer["name"], offer["cloud"].lower(), offer["price"], est["hours"], est["usd"]))
        return pod(pid)

    async def sync_template(self) -> dict:
        """Bring the RunPod template in line with the settings now, for the admin page."""
        key = _api_key()
        if not key:
            raise ValueError("Add a RunPod API key first.")
        s = settings()
        if not controller_url(s):
            raise ValueError("Set the studio address for pods first: the template carries it.")
        tid, done = await ensure_template(RunPod(key), s)
        return {"template_id": tid, "result": done, "name": s["template_name"]}

    async def start_by_hand(self, offer_id: str, cloud: str, user_id: str | None) -> dict:
        s = settings()
        if len(pods()) >= s["max_pods"]:
            raise ValueError("Already %d of %d allowed pods running." % (len(pods()), s["max_pods"]))
        candidates = [o for o in await offers("any", refresh=True) if o["id"] == offer_id and o["cloud"] == cloud]
        if not candidates:
            raise ValueError("That GPU is not offered on %s cloud right now." % cloud.lower())
        o = candidates[0]
        if o["price"] > s["max_price_per_hour"]:
            raise ValueError("$%.2f/h is above the price cap of $%.2f/h." % (o["price"], s["max_price_per_hour"]))
        key = _api_key()
        p = await self._create(RunPod(key), s, o, user_id=user_id, reason="started by hand")
        if not p:
            raise ValueError(self.last_error or "RunPod did not start the pod.")
        return p

    async def drain_by_hand(self, pod_id: str) -> None:
        p = pod(pod_id)
        if not p or p["status"] not in ACTIVE:
            raise ValueError("No such running pod.")
        _update(pod_id, status="draining", note="Stopped by hand.")

    async def terminate_now(self, pod_id: str) -> None:
        """Delete at once, without moving checkpoints. For a pod that is stuck."""
        p = pod(pod_id)
        if not p or p["status"] not in ACTIVE:
            raise ValueError("No such running pod.")
        await self._terminate(RunPod(_api_key() or ""), p, "Deleted by hand, without moving its checkpoints.")

    # -- 4. housekeeping
    @staticmethod
    def _forget_finished_checkpoints() -> None:
        folder = checkpoint_dir()
        if not folder.exists():
            return
        for f in folder.glob("*.tar"):
            j = db.get_job(f.stem)
            if not j or j["status"] in ("succeeded", "failed", "cancelled"):
                f.unlink(missing_ok=True)


def checkpoint_dir():
    from pathlib import Path
    return Path(config.DATA_DIR) / "checkpoints"


def checkpoint_file(job_id: str):
    return checkpoint_dir() / ("%s.tar" % job_id)


MANAGER = Manager()
