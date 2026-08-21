"""Fleet state and job scheduling.

Holds the live runner sockets, matches queued work to capable idle machines,
and fans runner events out to every open browser tab.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import WebSocket

from . import architectures, db, hub


class Fleet:
    def __init__(self) -> None:
        self.connections: dict[str, WebSocket] = {}
        self.busy: dict[str, str] = {}          # runner_id -> job_id
        self.ui_clients: set[WebSocket] = set()
        self.generations: dict[str, str] = {}   # request_id -> runner_id
        self._wake = asyncio.Event()

    # ------------------------------------------------------------ plumbing
    def attach(self, runner_id: str, ws: WebSocket) -> None:
        self.connections[runner_id] = ws

    def detach(self, runner_id: str) -> None:
        self.connections.pop(runner_id, None)
        self.busy.pop(runner_id, None)

    def wake(self) -> None:
        self._wake.set()

    async def send_to_runner(self, runner_id: str, msg: dict) -> bool:
        ws = self.connections.get(runner_id)
        if ws is None:
            return False
        try:
            await ws.send_text(json.dumps(msg))
            return True
        except Exception:  # noqa: BLE001 - a dead socket is handled by its own task
            return False

    async def broadcast_ui(self, msg: dict) -> None:
        if not self.ui_clients:
            return
        payload = json.dumps(msg)
        dead = []
        for ws in list(self.ui_clients):
            try:
                await ws.send_text(payload)
            except Exception:  # noqa: BLE001
                dead.append(ws)
        for ws in dead:
            self.ui_clients.discard(ws)

    # --------------------------------------------------------- scheduling
    def can_run(self, job: dict, caps: dict) -> tuple[bool, str]:
        """Would this job actually succeed on this runner?

        Checking here rather than letting the runner discover it means the user
        gets an instant, explained rejection instead of a crash ten minutes into
        a download.
        """
        if caps.get("backend") == "cpu" and not job["config"].get("allow_cpu"):
            return False, "runner has no GPU"
        if job["config"].get("quantization") == "4bit" \
                and not caps.get("quantization", {}).get("4bit"):
            return False, "runner has no working 4-bit support"

        required = job["config"].get("required_runner")
        if required and required != caps.get("_id"):
            return False, "pinned to a different runner"

        if job["kind"] == "pretrain_llm":
            return self._can_pretrain(job, caps)

        params_b = job["config"].get("params_b")
        if params_b and caps.get("vram_gb"):
            fit = hub.fit_report(params_b, caps)
            if fit["verdict"] in ("too_big", "needs_quantization"):
                return False, fit["message"]
        return True, ""

    @staticmethod
    def _can_pretrain(job: dict, caps: dict) -> tuple[bool, str]:
        """Full training is memory-bound in a way LoRA is not.

        The plan was sized against one machine's VRAM, 8-bit optimiser support
        and attention kernels. Sending it to a different machine that lacks any
        of those turns a fitted plan into an out-of-memory crash minutes in, so
        the fit is recomputed for whichever runner is about to take it.
        """
        cfg = job["config"]
        arch = cfg.get("arch")
        vram = caps.get("vram_gb")
        if not arch or not vram:
            return True, ""
        mem = architectures.training_memory_gb(
            arch, int(cfg.get("batch_size", 1)),
            optim_8bit=bool(cfg.get("optim_8bit"))
            and bool((caps.get("quantization") or {}).get("optim_8bit")),
            checkpointing=bool(cfg.get("gradient_checkpointing")),
            flash=bool((caps.get("attention") or {}).get("flash")),
        )
        if mem["total_gb"] > vram * 0.92:
            return False, ("training every parameter of this model needs about "
                           "%.1f GB and this machine has %.1f GB"
                           % (mem["total_gb"], vram))
        return True, ""

    async def scheduler_loop(self) -> None:
        """Assign queued jobs to idle runners. Woken by events, with a periodic
        tick as a backstop so nothing can sit stuck if a wake-up is missed."""
        while True:
            try:
                await self._dispatch_once()
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                print("[scheduler] error: %r" % e)
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                pass
            self._wake.clear()

    async def _dispatch_once(self) -> None:
        queued = db.queued_jobs()
        if not queued:
            return
        idle = [rid for rid in self.connections if rid not in self.busy]
        if not idle:
            return

        for job in queued:
            for runner_id in list(idle):
                runner = db.get_runner(runner_id)
                if not runner:
                    continue
                caps = dict(runner["capabilities"])
                caps["_id"] = runner_id
                ok, why = self.can_run(job, caps)
                if not ok:
                    continue

                db.assign_job(job["id"], runner_id)
                self.busy[runner_id] = job["id"]
                sent = await self.send_to_runner(runner_id, {
                    "type": "job_assign",
                    "job": {"id": job["id"], "kind": job["kind"], "config": job["config"]},
                })
                if not sent:
                    # Socket died between the idle check and the send.
                    self.busy.pop(runner_id, None)
                    db.set_job_status(job["id"], "queued")
                    continue
                db.add_log(job["id"], "Assigned to runner '%s'." % runner["name"])
                idle.remove(runner_id)
                await self.broadcast_ui({"type": "jobs_changed", "job_id": job["id"]})
                break

    # ----------------------------------------------------- runner messages
    async def handle_runner_message(self, runner_id: str, msg: dict) -> None:
        kind = msg.get("type")
        jid = msg.get("job_id")

        # Generated text is relayed straight through rather than stored. A
        # conversation is not a training artifact, and writing every token to
        # SQLite would turn a chat into a few hundred transactions a minute.
        if kind in ("generate_delta", "generate_status", "generate_done",
                    "generate_error"):
            await self.broadcast_ui({**msg, "type": msg["type"]})
            if kind in ("generate_done", "generate_error"):
                self.generations.pop(msg.get("request_id"), None)
            return

        if kind == "heartbeat":
            db.touch_runner(runner_id, "busy" if msg.get("busy") else "online")
            return

        if kind == "capabilities":
            runner = db.get_runner(runner_id)
            db.upsert_runner(runner_id, (runner or {}).get("name", runner_id),
                             msg.get("capabilities") or {})
            await self.broadcast_ui({"type": "runners_changed"})
            return

        if kind == "job_started":
            db.set_job_status(jid, "running")
            self.busy[runner_id] = jid
            db.touch_runner(runner_id, "busy")
            await self.broadcast_ui({"type": "jobs_changed", "job_id": jid})

        elif kind == "job_log":
            db.add_log(jid, msg.get("line", ""), msg.get("level", "info"))
            await self.broadcast_ui({"type": "job_log", "job_id": jid,
                                     "line": msg.get("line", ""),
                                     "level": msg.get("level", "info")})

        elif kind == "job_metric":
            db.add_metric(jid, msg["step"], msg["data"])
            await self.broadcast_ui({"type": "job_metric", "job_id": jid,
                                     "step": msg["step"], "data": msg["data"]})

        elif kind == "job_progress":
            stage = msg.get("stage", "")
            # Only training steps are persisted. Preparation stages report
            # their own units -- documents scanned, tokens collected -- and
            # writing those into the job's step counter makes every list that
            # reads it announce "step 25000 of 60000" for a 300-step run.
            # They still stream to open tabs, which is where they belong.
            if stage in ("", "training"):
                db.set_job_progress(jid, msg.get("step", 0), msg.get("total", 0))
            await self.broadcast_ui({"type": "job_progress", "job_id": jid,
                                     "step": msg.get("step", 0),
                                     "total": msg.get("total", 0),
                                     "stage": stage})

        elif kind == "job_meta":
            meta = msg.get("meta", {})
            if sample := meta.get("sample"):
                # Generated text gets its own channel rather than being buried
                # in the log. Watching noise turn into sentences is the clearest
                # signal a from-scratch run gives that it is working, so it is
                # stored per step and shown as its own panel.
                db.add_metric(jid, int(sample.get("step") or 0),
                              {"sample_text": sample.get("text", ""),
                               "sample_prompt": sample.get("prompt", "")})
                await self.broadcast_ui({"type": "job_sample", "job_id": jid,
                                         **sample})
            else:
                db.add_log(jid, "Details: %s" % json.dumps(meta)[:800])

        elif kind == "job_done":
            db.set_job_status(jid, "succeeded")
            summary = msg.get("summary") or {}
            db.add_log(jid, "Finished successfully. %s" % json.dumps(summary)[:600])
            self.busy.pop(runner_id, None)
            db.touch_runner(runner_id, "online")
            await self.broadcast_ui({"type": "jobs_changed", "job_id": jid})
            self.wake()

        elif kind in ("job_failed", "job_cancelled"):
            status = "failed" if kind == "job_failed" else "cancelled"
            err = msg.get("error")
            db.set_job_status(jid, status, err)
            if err:
                db.add_log(jid, err, "error")
            if tb := msg.get("traceback"):
                db.add_log(jid, tb, "debug")
            self.busy.pop(runner_id, None)
            db.touch_runner(runner_id, "online")
            await self.broadcast_ui({"type": "jobs_changed", "job_id": jid})
            self.wake()

        elif kind == "job_rejected":
            db.set_job_status(jid, "queued")
            db.add_log(jid, "Runner declined the job: %s" % msg.get("reason", ""), "warn")
            self.busy.pop(runner_id, None)
            self.wake()
