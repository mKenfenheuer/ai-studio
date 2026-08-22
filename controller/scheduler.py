"""Fleet state and job scheduling.

Holds the live runner sockets, matches queued work to capable idle machines,
and fans runner events out to every open browser tab.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from fastapi import WebSocket

from . import architectures, db, hub

# How long a queued job waits for the machine holding its checkpoint before
# giving up on the progress and running somewhere else. A container restart
# takes seconds; ten minutes of silence means the machine is not coming back
# in time to be worth waiting for.
CHECKPOINT_WAIT_S = 600


class Fleet:
    def __init__(self) -> None:
        self.connections: dict[str, WebSocket] = {}
        self.busy: dict[str, str] = {}          # runner_id -> job_id
        self.ui_clients: set[WebSocket] = set()
        self.generations: dict[str, str] = {}   # request_id -> runner_id
        # When each runner was last handed work. A runner takes a moment to
        # pick a job up, and during that gap its heartbeat still says idle --
        # without this the scheduler would hand it a second job.
        self.dispatched_at: dict[str, float] = {}
        # Jobs already told the user they are waiting, so a queue that has to
        # wait an hour does not write an hour of identical log lines.
        self.declined: set[str] = set()
        # runner_id -> the jobs that machine could carry on from a checkpoint.
        self.checkpoints: dict[str, set[str]] = {}
        self.gave_up_waiting: set[str] = set()
        self._wake = asyncio.Event()

    # ------------------------------------------------------------ plumbing
    def attach(self, runner_id: str, ws: WebSocket) -> None:
        self.connections[runner_id] = ws

    def detach(self, runner_id: str) -> None:
        self.connections.pop(runner_id, None)
        self.busy.pop(runner_id, None)
        self.dispatched_at.pop(runner_id, None)
        # Deliberately NOT clearing self.checkpoints here. The commonest reason
        # a socket closes is a restart, and the disk that holds the checkpoints
        # is still there. What the scheduler needs to know is "is that machine
        # reachable", which it reads from self.connections; forgetting what the
        # machine holds would only make the run start over once it came back.

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

    async def reconcile_orphans(self) -> None:
        """Deal with work whose machine has genuinely vanished.

        Split out from the disconnect handler on purpose. A socket closing
        says nothing about whether training stopped -- restarting the
        controller closes every socket while every run carries on. Only
        prolonged silence means the work is really lost, and by then the
        runner has had several heartbeats to say otherwise.
        """
        for job in db.orphaned_jobs():
            jid = job["id"]
            if job["runner_id"] in self.connections:
                continue
            requeued, rescued = db.requeue_jobs_for_runner(job["runner_id"])
            for j in rescued:
                db.add_log(j, "The machine went away just after uploading the "
                              "result, so this run is complete.")
            for j in requeued:
                self.declined.discard(j)
                db.add_log(j, "The machine has been unreachable for a while; "
                              "the run has gone back on the queue.", "warn")
            if requeued or rescued:
                await self.broadcast_ui({"type": "jobs_changed"})
                self.wake()

    async def reconcile_runner(self, runner_id: str,
                               current_job: str | None) -> None:
        """The machine has told us what it is doing. Believe it.

        This closes a hole that only opens when a runner restarts *quickly*.
        `reconcile_orphans` waits for a machine to fall silent past the
        heartbeat deadline before touching its work -- which is right, because
        a dropped socket does not mean training stopped. But a container that
        is recreated and dials back in within the deadline never becomes
        silent for long enough: its `last_seen` is refreshed by the reconnect,
        the job stops qualifying as orphaned, and it sits marked "running" on
        an idle machine forever.

        The runner is the authority on what it is running, and it says so on
        every connect and every heartbeat. Anything else the controller has
        pinned to that machine is finished or gone.
        """
        requeued, rescued = db.requeue_jobs_for_runner(runner_id, current_job)
        name = (db.get_runner(runner_id) or {}).get("name", "that machine")
        for jid in rescued:
            db.add_log(jid, "%s restarted just after uploading the result, so "
                       "this run is complete." % name)
        for jid in requeued:
            self.declined.discard(jid)
            job = db.get_job(jid) or {}
            step = int(job.get("checkpoint_step") or 0)
            db.add_log(jid, "%s restarted and is no longer running this, so it "
                       "has gone back on the queue. %s"
                       % (name,
                          "It will carry on from its checkpoint at step %d."
                          % step if step else
                          "There is no checkpoint, so it starts from the "
                          "beginning."), "warn")
        if requeued or rescued:
            await self.broadcast_ui({"type": "jobs_changed"})
            self.wake()

    async def scheduler_loop(self) -> None:
        """Assign queued jobs to idle runners. Woken by events, with a periodic
        tick as a backstop so nothing can sit stuck if a wake-up is missed."""
        while True:
            try:
                await self.reconcile_orphans()
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

    def fair_order(self, queued: list[dict]) -> list[dict]:
        """The queue, rearranged so one person cannot hold up everyone else.

        Strict first-come-first-served is the obvious rule and the wrong one
        for a shared machine: somebody who queues eight overnight runs at five
        o'clock owns the GPU until morning, and a colleague with a twenty-
        minute job waits behind all eight. It is not that the eight runs are
        unreasonable -- it is that "queued first" stopped being a fair way to
        choose between people once there was more than one person.

        So the queue is dealt round-robin between owners, one job each per
        pass, and the order of the owners themselves is decided by who is
        using the studio least right now: fewest runs in flight, then least
        machine time in the last day, then longest wait as the tiebreak. A
        single user sees exactly first-come-first-served, because with one
        owner the round-robin degenerates to the original order.
        """
        if len({j.get("owner_id") for j in queued}) < 2:
            return queued

        running = db.running_by_owner()
        used = db.gpu_seconds_by_owner()
        groups: dict[Any, list[dict]] = {}
        for j in queued:                       # already oldest-first
            groups.setdefault(j.get("owner_id"), []).append(j)

        owners = sorted(groups, key=lambda o: (running.get(o, 0),
                                               used.get(o, 0.0),
                                               groups[o][0]["created_at"]))
        out: list[dict] = []
        for i in range(max(len(v) for v in groups.values())):
            for owner in owners:
                if i < len(groups[owner]):
                    out.append(groups[owner][i])
        return out

    def holder_of(self, job: dict) -> str | None:
        """The machine that alone can carry this job on, or None.

        A checkpoint is a directory on one runner's disk. Handing the job to a
        different machine does not fail -- it quietly starts from the
        beginning, which is the exact outcome checkpoints exist to prevent.
        """
        if not int(job.get("checkpoint_step") or 0):
            return None
        return job.get("checkpoint_runner") or None

    def _eligible(self, job: dict, idle: list[str]) -> list[str]:
        """Which of the idle machines may take this job."""
        holder = self.holder_of(job)
        if not holder:
            return idle
        if holder in idle:
            return [holder]
        if holder in self.connections:
            return []          # connected but busy: worth waiting for
        runner = db.get_runner(holder)
        silent_for = time.time() - float((runner or {}).get("last_seen") or 0)
        if silent_for < CHECKPOINT_WAIT_S:
            return []          # probably restarting; it will be back
        # Genuinely gone. Give up on the progress rather than the run.
        if job["id"] not in self.gave_up_waiting:
            self.gave_up_waiting.add(job["id"])
            db.add_log(job["id"], "The machine holding this run's checkpoint "
                       "(%s) has been away for %d minutes, so the run will "
                       "start from the beginning on whichever machine is free."
                       % ((runner or {}).get("name", "unknown"),
                          silent_for // 60), "warn")
        db.clear_checkpoint(job["id"])
        return idle

    async def _dispatch_once(self) -> None:
        queued = db.queued_jobs()
        if not queued:
            return
        idle = [rid for rid in self.connections if rid not in self.busy]
        if not idle:
            return

        for job in self.fair_order(queued):
            for runner_id in self._eligible(job, list(idle)):
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
                self.dispatched_at[runner_id] = time.time()
                sent = await self.send_to_runner(runner_id, {
                    "type": "job_assign",
                    "job": {"id": job["id"], "kind": job["kind"], "config": job["config"]},
                })
                if not sent:
                    # Socket died between the idle check and the send.
                    self.busy.pop(runner_id, None)
                    self.dispatched_at.pop(runner_id, None)
                    db.set_job_status(job["id"], "queued")
                    continue
                db.add_log(job["id"], "Assigned to runner '%s'." % runner["name"])
                idle.remove(runner_id)
                await self.broadcast_ui({"type": "jobs_changed", "job_id": job["id"]})
                break

    def note_checkpoints(self, runner_id: str, job_ids: list[str]) -> None:
        """Record what a machine can resume, and reconcile it with the database.

        Both directions matter. A checkpoint the controller forgot (its
        database was restored from a backup, or the column was added after the
        run started) is recovered from the machine that has it. A checkpoint
        the machine no longer has -- someone cleared the volume -- stops being
        promised to a queued job that would then wait for it forever.
        """
        held = set(job_ids or [])
        self.checkpoints[runner_id] = held
        for jid in db.jobs_with_checkpoint_on(runner_id):
            if jid not in held:
                db.clear_checkpoint(jid)
                db.add_log(jid, "The checkpoint for this run is no longer on "
                           "%s, so starting it again would start from the "
                           "beginning."
                           % (db.get_runner(runner_id) or {}).get("name", "that machine"),
                           "warn")

    def queue_positions(self) -> dict[str, int]:
        """Where each waiting job sits in the order work will actually be
        handed out in. Shown in the UI, because a queue nobody can see the
        shape of is indistinguishable from a queue that is stuck."""
        return {j["id"]: i + 1 for i, j in enumerate(self.fair_order(db.queued_jobs()))}

    def in_flight(self) -> list[dict]:
        """Work that would be lost, or interrupted, by restarting right now."""
        out = []
        for job in db.q("SELECT * FROM jobs WHERE status IN ('assigned','running')"):
            runner = db.get_runner(job["runner_id"]) if job["runner_id"] else None
            out.append({
                "id": job["id"], "name": job["name"], "kind": job["kind"],
                "status": job["status"], "step": job["step"],
                "total_steps": job["total_steps"],
                "runner_id": job["runner_id"],
                "runner": (runner or {}).get("name"),
                "checkpoint_step": job["checkpoint_step"] or 0,
            })
        return out

    # ----------------------------------------------------- runner messages
    async def handle_runner_message(self, runner_id: str, msg: dict) -> None:
        kind = msg.get("type")
        jid = msg.get("job_id")

        # Generated text is relayed straight through rather than stored. A
        # conversation is not a training artifact, and writing every token to
        # SQLite would turn a chat into a few hundred transactions a minute.
        if kind in ("generate_delta", "generate_status", "generate_done",
                    "generate_error"):
            # Passed through whole, reasoning field included.
            await self.broadcast_ui({**msg, "type": msg["type"]})
            if kind in ("generate_done", "generate_error"):
                self.generations.pop(msg.get("request_id"), None)
            return

        if kind == "heartbeat":
            db.touch_runner(runner_id, "busy" if msg.get("busy") else "online")
            if (held := msg.get("checkpoints")) is not None:
                self.note_checkpoints(runner_id, held)
            # The runner is the authority on what it is doing. Deriving this
            # from dispatch bookkeeping alone loses track the moment the
            # controller restarts, and then hands work to a machine that is
            # already training.
            if msg.get("busy"):
                jid = msg.get("job_id")
                self.busy[runner_id] = jid or self.busy.get(runner_id) or "?"
                # The runner is still working on this. If the controller
                # restarted and lost track, or an earlier disconnect put the
                # job back on the queue, correct that now rather than running
                # the same work twice.
                if jid:
                    job = db.get_job(jid)
                    if job and job["status"] in ("queued", "assigned"):
                        db.set_job_status(jid, "running")
                        db.assign_job_runner(jid, runner_id)
                        self.declined.discard(jid)
                        await self.broadcast_ui({"type": "jobs_changed",
                                                 "job_id": jid})
            elif time.time() - self.dispatched_at.get(runner_id, 0) > 30:
                # Genuinely idle, and not merely slow to pick up work just
                # handed to it. Anything the database still has running here
                # is not running: say so and put it back on the queue.
                await self.reconcile_runner(runner_id, None)
                if self.busy.pop(runner_id, None) is not None:
                    self.wake()
            return

        if kind == "capabilities":
            runner = db.get_runner(runner_id)
            db.upsert_runner(runner_id, (runner or {}).get("name", runner_id),
                             msg.get("capabilities") or {})
            await self.broadcast_ui({"type": "runners_changed"})
            return

        if kind == "job_started":
            # A restarted attempt begins its curve again at step 1. Leaving the
            # abandoned attempt's measurements behind gives the job two rows
            # for every step, so the chart draws itself backwards and the
            # "latest" reading comes from a run that no longer exists. The logs
            # keep the history; the measurements describe the live attempt.
            #
            # A *resumed* attempt is the opposite case: the readings up to the
            # checkpoint describe this run and are still true, and only the
            # ones past it belong to the attempt that was lost.
            resume_step = int(msg.get("resume_step") or 0)
            if resume_step:
                dropped = db.trim_metrics(jid, resume_step)
                db.add_log(jid, "Carrying on from the checkpoint at step %d.%s"
                           % (resume_step,
                              " The %d readings from past that point belonged "
                              "to the interrupted attempt and have been "
                              "cleared." % dropped if dropped else ""))
            elif db.count_metrics(jid):
                db.clear_metrics(jid)
                db.add_log(jid, "Starting again from the beginning; the "
                                "measurements from the interrupted attempt "
                                "have been cleared.", "warn")
            db.set_job_status(jid, "running")
            self.declined.discard(jid)
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
            if ckpt := meta.get("checkpoint"):
                # Persisted, not merely noted. This one fact decides whether an
                # interrupted run resumes or starts over, and keeping it only
                # in memory would lose it to the very controller restart that
                # tends to cause the interruption.
                db.set_checkpoint(jid, int(ckpt.get("step") or 0), runner_id)
                self.checkpoints.setdefault(runner_id, set()).add(jid)
                await self.broadcast_ui({"type": "job_checkpoint", "job_id": jid,
                                         "step": int(ckpt.get("step") or 0)})
            elif sample := meta.get("sample"):
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
            if summary.get("kind") == "evaluate":
                # The per-prompt answers are filed against the prompt set, so
                # the comparison outlives this run. Only the headline stays on
                # the job, or a fifty-prompt scoring would put a megabyte of
                # generated text in the jobs table.
                from .api import evals
                written = evals.record_scores(db.get_job(jid) or {"id": jid},
                                              summary)
                summary = {**summary,
                           "scores": [{k: v for k, v in s.items() if k != "items"}
                                      for s in summary.get("scores") or []]}
                if written:
                    db.add_log(jid, "Recorded %d score%s against the prompt "
                               "set, ready to compare with later runs."
                               % (written, "" if written == 1 else "s"))
            db.set_job_summary(jid, summary)
            db.clear_checkpoint(jid)
            self.checkpoints.get(runner_id, set()).discard(jid)
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
            if summary := msg.get("summary"):
                # A stopped run that kept its model has the same summary a
                # finished one does, and everything downstream -- the
                # playground, the download, the stats panel, the comparison
                # table -- reads it the same way.
                db.set_job_summary(jid, summary)
                db.add_log(jid, "Stopped, and the model was kept. %s"
                           % json.dumps(summary)[:600])
            if kind == "job_cancelled":
                db.clear_checkpoint(jid)
                self.checkpoints.get(runner_id, set()).discard(jid)
            if tb := msg.get("traceback"):
                db.add_log(jid, tb, "debug")
            self.busy.pop(runner_id, None)
            db.touch_runner(runner_id, "online")
            await self.broadcast_ui({"type": "jobs_changed", "job_id": jid})
            self.wake()

        elif kind == "job_rejected":
            # A decline is information about the runner, not just about the
            # job. Clearing `busy` here and waking the scheduler produced a
            # tight loop: the job went back on the queue, the same busy runner
            # looked free, it was handed the job again, and it declined again
            # -- 250 times in a few minutes, writing a log line each way.
            #
            # So mark the runner busy and do NOT wake. Its next heartbeat that
            # reports idle will wake the scheduler, which is the moment there
            # is actually any point in trying again.
            db.set_job_status(jid, "queued")
            self.busy[runner_id] = msg.get("current_job") or "?"
            if jid not in self.declined:
                self.declined.add(jid)
                db.add_log(jid, "Waiting for %s to finish its current run."
                           % (db.get_runner(runner_id) or {}).get("name", "that machine"),
                           "warn")
