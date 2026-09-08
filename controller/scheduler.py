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

from . import architectures, db, hub, notify

# How long a queued job waits for the machine holding its checkpoint before
# giving up on the progress and running somewhere else. A container restart
# takes seconds; ten minutes of silence means the machine is not coming back
# in time to be worth waiting for.
CHECKPOINT_WAIT_S = 600

# Stages whose counter is the run's actual progress, and so is worth storing
# on the job for every list and every tab opened later. Everything else a
# runner reports -- loading a model, tokenizing a corpus -- is preparation
# counted in its own units, and belongs only in the live stream.
PERSISTED_STAGES = ("", "training", "writing", "uploading", "evaluating")


class Fleet:
    def __init__(self) -> None:
        self.connections: dict[str, WebSocket] = {}
        self.busy: dict[str, str] = {}          # runner_id -> job_id
        # Each browser socket, and who is signed in on it. The stream carries
        # run logs, metrics and playground tokens, and every one of those has
        # an owner; a socket that does not know whose it is cannot be told
        # what it may hear.
        self.ui_clients: dict[WebSocket, dict] = {}
        self.generations: dict[str, str] = {}   # request_id -> runner_id
        self.generation_owner: dict[str, str] = {}  # request_id -> user id
        # (user id, job id) -> (may see it, when that was checked). Visibility
        # is a database question, and a run emits a metric every step; this
        # keeps it to one query per viewer per run every half minute.
        self._visible: dict[tuple[str, str], tuple[bool, float]] = {}
        # request_id -> a queue for a caller waiting on the reply over HTTP.
        # The websocket relay is fire-and-forget, which is right for a browser
        # watching tokens appear and useless for a request that has to return
        # an answer.
        self.waiters: dict[str, "asyncio.Queue"] = {}
        # When each runner was last handed work. A runner takes a moment to
        # pick a job up, and during that gap its heartbeat still says idle --
        # without this the scheduler would hand it a second job.
        self.dispatched_at: dict[str, float] = {}
        # Jobs already told the user they are waiting, so a queue that has to
        # wait an hour does not write an hour of identical log lines.
        self.declined: set[str] = set()
        # runner_id -> the jobs that machine could carry on from a checkpoint.
        self.checkpoints: dict[str, set[str]] = {}
        # What each runner is holding, from its heartbeat. This is what makes
        # a second message to the same model fast: `loaded` is on the card and
        # answers now, `cached` is on that machine's disk and needs no
        # fourteen-gigabyte download. Without it the controller sent every
        # conversation to whichever machine trained the model, which for a
        # merged one was the processor-only box that did the merging.
        self.loaded: dict[str, list[str]] = {}   # runner_id -> job ids, MRU first
        self.cached: dict[str, set[str]] = {}    # runner_id -> job ids on disk
        self.disk: dict[str, dict] = {}          # runner_id -> free/total GB
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
        #
        # `loaded` is the exception, and for the same reason: video memory does
        # not survive a restart, so a machine that comes back has an empty card
        # and a full disk. Its next heartbeat says so either way.
        self.loaded.pop(runner_id, None)

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

    # Frames that carry a run's own data, as opposed to "something changed,
    # go and look" -- which carries nothing a list page would not show anyway.
    _PER_JOB = frozenset(("job_log", "job_metric", "job_progress",
                          "job_checkpoint", "job_sample", "job_finished"))

    def _can_see(self, user: dict, job_id: str) -> bool:
        key = (user["id"], job_id)
        hit = self._visible.get(key)
        now = time.time()
        if hit and now - hit[1] < 30:
            return hit[0]
        job = db.get_job(job_id)
        ok = bool(job) and db.access_level(
            "job", job_id, job.get("owner_id"), user) is not None
        self._visible[key] = (ok, now)
        if len(self._visible) > 5000:
            self._visible.clear()
        return ok

    async def broadcast_ui(self, msg: dict, *, user_id: str | None = None) -> None:
        """Send a frame to the browsers that may have it.

        `user_id` narrows it to one person's tabs. Frames about a run go only
        to people who can see that run. Everything else -- "the queue moved",
        "a machine connected" -- goes to everyone signed in, because it says
        nothing a list page would not.
        """
        if not self.ui_clients:
            return
        payload = json.dumps(msg)
        jid = msg.get("job_id") if msg.get("type") in self._PER_JOB else None
        dead = []
        for ws, user in list(self.ui_clients.items()):
            if user_id is not None and user.get("id") != user_id:
                continue
            if jid and not self._can_see(user, jid):
                continue
            try:
                await ws.send_text(payload)
            except Exception:  # noqa: BLE001
                dead.append(ws)
        for ws in dead:
            self.ui_clients.pop(ws, None)

    # --------------------------------------------------------- scheduling
    def can_run(self, job: dict, caps: dict) -> tuple[bool, str]:
        """Would this job actually succeed on this runner?

        Checking here rather than letting the runner discover it means the user
        gets an instant, explained rejection instead of a crash ten minutes into
        a download.
        """
        # A machine that was told to take only certain kinds of work. Checked
        # first, because "this machine only does uploads" explains a skip far
        # better than the memory arithmetic further down would.
        if (kinds := caps.get("kinds")) and job["kind"] not in kinds:
            return False, ("this machine only runs %s"
                           % ", ".join(str(k) for k in kinds))
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

        # A size the job did not carry is worked out from the model's name
        # rather than shrugged at. This check reads "refuse if it does not
        # fit", so an unknown size is not a neutral state -- it disables the
        # check entirely, and a continued run had lost `params_b` on the way.
        params_b = job["config"].get("params_b") \
            or hub.params_from_name(job["config"].get("base_model") or "")
        if params_b and caps.get("vram_gb"):
            fit = hub.fit_report(params_b, caps)
            if fit["verdict"] in ("too_big", "needs_quantization"):
                return False, fit["message"]
            # `fits_quantized` means "only in 4-bit". Treating it as a pass
            # regardless of what the run actually asked for was the whole bug:
            # a 7B needs 19.6 GB in 16-bit and 4.9 GB in 4-bit, and a run
            # configured for 16-bit was dispatched to a 16 GB card because the
            # verdict said the model *could* fit -- in a precision nobody had
            # selected. It did not crash. It sat in one backward pass for
            # thirteen hours with the allocator at 99%.
            if fit["verdict"] == "fits_quantized" \
                    and job["config"].get("quantization") != "4bit":
                mem = hub.estimate_memory(params_b) or {}
                return False, (
                    "This model needs about %.1f GB in 16-bit and this machine "
                    "has %.1f GB. It fits in 4-bit, at about %.1f GB — switch "
                    "this run to 4-bit and it will run here."
                    % (mem.get("fp16_gb") or 0, caps["vram_gb"],
                       mem.get("int4_gb") or 0))
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
        """Which of the idle machines may take this job, best first."""
        holder = self.holder_of(job)
        if not holder:
            return self._by_preference(job, idle)
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
        return self._by_preference(job, idle)

    @staticmethod
    def _by_preference(job: dict, idle: list[str]) -> list[str]:
        """Machines that specialise in this kind of work, first.

        An upload needs no GPU and will run anywhere, which is exactly the
        problem: handed to the one machine with a card, it occupies it for
        twenty minutes of network while training waits. A machine restricted
        to a list of kinds is a machine somebody set aside for them, so it is
        asked first and the GPU is left for work that needs it.
        """
        def rank(rid: str) -> int:
            runner = db.get_runner(rid)
            kinds = ((runner or {}).get("capabilities") or {}).get("kinds")
            return 0 if kinds and job["kind"] in kinds else 1

        return sorted(idle, key=rank)

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

    @staticmethod
    def _refresh_cards(job_id: str, summary: dict) -> None:
        """Rewrite the model cards this run has just made out of date.

        A card is only worth having if it is true, and there are exactly two
        moments it stops being true: when the run that made the model reports
        what it did, and when somebody measures the model afterwards. The
        second one is why this is not simply done when the artifact arrives --
        an evaluation produces no artifact at all, and it is the run that adds
        the numbers people actually want on a card.

        Cards somebody has edited are left alone; cards.refresh enforces that.
        """
        from . import cards
        if summary.get("kind") == "evaluate":
            for score in summary.get("scores") or []:
                if mid := score.get("model_job_id"):
                    cards.refresh(mid, reason="scored against \"%s\""
                                  % (summary.get("eval_name") or "a prompt set"))
            return
        job = db.get_job(job_id) or {}
        if job.get("kind") in ("finetune_llm", "pretrain_llm", "merge_adapter"):
            cards.refresh(job_id, reason="the run finished")

    @staticmethod
    def _record_publication(upload_job_id: str, summary: dict) -> None:
        """File a finished upload against the run whose model it sent.

        The upload run knows the repository; the trained run is what everything
        else asks about. Without this, a fine-tune built on a model that was
        published from this studio has no way to name it on the Hub, and its
        card drops the `base_model:` line rather than print a job id.
        """
        cfg = (db.get_job(upload_job_id) or {}).get("config") or {}
        trained_by = cfg.get("trained_by") or cfg.get("source_job")
        if not trained_by or not db.get_job(trained_by):
            return
        db.record_publication(trained_by, {
            "repo_id": summary["repo_id"],
            "url": summary.get("url"),
            "artifact_kind": cfg.get("artifact_kind") or "model",
            "upload_job": upload_job_id,
        })
        db.add_log(trained_by, "Published to %s." % summary["repo_id"])

    async def announce_end(self, job_id: str, status: str) -> None:
        """A run has ended. Tell the browsers, and tell whoever owns it.

        One place, so the three ways a run can end cannot each grow their own
        slightly different idea of what "ended" means.
        """
        job = db.get_job(job_id) or {"id": job_id}
        await self.broadcast_ui({
            "type": "job_finished", "job_id": job_id, "status": status,
            "name": job.get("name"), "kind": job.get("kind"),
            "error": job.get("error"),
            "summary": {k: (job.get("summary") or {}).get(k)
                        for k in ("best_val_loss", "final_loss", "steps",
                                  "early_stopped", "kept_from_step")},
        })
        await self.broadcast_ui({"type": "jobs_changed", "job_id": job_id})
        notify.fire(job, status)

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
        # A counted answer to "how long are these rows, really". Always to a
        # waiting caller, never broadcast: nobody's browser wants it.
        if kind in ("tokenize_done", "tokenize_error"):
            if waiter := self.waiters.get(msg.get("request_id")):
                waiter.put_nowait(msg)
            return

        if kind in ("generate_delta", "generate_status", "generate_done",
                    "generate_error"):
            rid = msg.get("request_id")
            waiter = self.waiters.get(rid)
            if waiter is not None:
                # Somebody is awaiting this reply over HTTP rather than
                # watching it in a browser. It goes to them and nowhere else:
                # broadcasting an API caller's tokens to every open tab in the
                # studio would put one person's conversation on another
                # person's screen.
                waiter.put_nowait(msg)
            elif (owner := self.generation_owner.get(rid)):
                # Passed through whole, reasoning field included -- to the
                # tabs of the person who asked, and nobody else's. This used
                # to go to every signed-in browser, which put one person's
                # conversation on every other person's screen.
                await self.broadcast_ui({**msg, "type": msg["type"]},
                                        user_id=owner)
            if kind == "generate_error":
                # The one part of a conversation worth keeping. It lands in the
                # run's own log, which is where somebody looking into "the
                # model would not answer me" already is -- an API caller only
                # ever sees a 502, and until this there was nothing behind it.
                if jid := msg.get("job_id"):
                    facts = msg.get("diagnostics") or {}
                    db.add_log(jid, "Could not answer: %s%s" % (
                        msg.get("detail") or msg.get("error") or "unknown",
                        (" [%s]" % ", ".join(
                            "%s=%s" % kv for kv in sorted(facts.items()))
                         ) if facts else ""), "error")
            if kind in ("generate_done", "generate_error"):
                self.generations.pop(rid, None)
                self.generation_owner.pop(rid, None)
            return

        if kind == "heartbeat":
            db.touch_runner(runner_id, "busy" if msg.get("busy") else "online")
            if (held := msg.get("checkpoints")) is not None:
                self.note_checkpoints(runner_id, held)
            if (resident := msg.get("loaded")) is not None:
                self.loaded[runner_id] = list(resident)
            if (on_disk := msg.get("cached")) is not None:
                # `<id>-adapter` is one artifact of a run rather than a run, and
                # what the chat router asks is "does this machine have that
                # model": the bare id is the answer to that.
                self.cached[runner_id] = {str(c).split("-adapter")[0]
                                          for c in on_disk}
            if (space := msg.get("disk")) is not None:
                self.disk[runner_id] = space
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
            # Only the stage that *is* the work is persisted. Preparation
            # stages report their own units -- documents scanned, tokens
            # collected -- and writing those into the job's step counter makes
            # every list that reads it announce "step 25000 of 60000" for a
            # 300-step run. They still stream to open tabs, which is where
            # they belong.
            #
            # The named stages are the counterpart of that: rows written and
            # megabytes sent are as much the run's real progress as steps are,
            # and leaving them out is why a generation run sat at "0/?" in
            # every list from the moment it started until the moment it
            # finished.
            if stage in PERSISTED_STAGES:
                db.set_job_progress(jid, msg.get("step", 0), msg.get("total", 0))
            await self.broadcast_ui({"type": "job_progress", "job_id": jid,
                                     "step": msg.get("step", 0),
                                     "total": msg.get("total", 0),
                                     "stage": stage})

        elif kind == "job_meta":
            meta = msg.get("meta", {})
            # The size, as counted rather than guessed. The runner has the
            # model in hand and knows exactly how many parameters it has;
            # everything else in this app infers it from the model's name,
            # which works until it does not. Recorded on the run so that a
            # later run continuing from it inherits a measured number -- the
            # missing size is what let a 7B be planned in 16-bit for a 16 GB
            # card.
            if (counted := meta.get("total_params")) and int(counted) > 0:
                job = db.get_job(jid)
                measured = round(int(counted) / 1e9, 3)
                if job and job["config"].get("params_b") != measured:
                    cfg = dict(job["config"])
                    cfg["params_b"] = measured
                    db.update_job_config(jid, cfg)

            if resolved := meta.get("resolved_format"):
                # The trainer worked out a shape the plan left as "auto". The
                # job's own record is updated so that everything downstream --
                # the playground, the API, a later fine-tune of this model --
                # uses what was actually trained on.
                job = db.get_job(jid)
                if job and (job["config"].get("format") or {}).get(
                        "mode", "auto") == "auto":
                    cfg = dict(job["config"])
                    cfg["format"] = {**(cfg.get("format") or {}), **resolved}
                    db.update_job_config(jid, cfg)
                    db.add_log(jid, "Recorded how this data reads (%s), so the "
                               "finished model is spoken to the way it was "
                               "taught." % (resolved.get("mode") or "plain text"))
            elif ckpt := meta.get("checkpoint"):
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
            if summary.get("kind") == "upload" and summary.get("repo_id") \
                    and summary.get("target") == "model":
                self._record_publication(jid, summary)
            db.set_job_summary(jid, summary)
            self._refresh_cards(jid, summary)
            db.clear_checkpoint(jid)
            self.checkpoints.get(runner_id, set()).discard(jid)
            db.add_log(jid, "Finished successfully. %s" % json.dumps(summary)[:600])
            self.busy.pop(runner_id, None)
            db.touch_runner(runner_id, "online")
            await self.announce_end(jid, "succeeded")
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
                # It has a model, so it has a card -- and the card is where it
                # says the run was stopped short of its schedule, which is the
                # one thing somebody downloading this needs told.
                self._refresh_cards(jid, summary)
                db.add_log(jid, "Stopped, and the model was kept. %s"
                           % json.dumps(summary)[:600])
            if kind == "job_cancelled":
                db.clear_checkpoint(jid)
                self.checkpoints.get(runner_id, set()).discard(jid)
            if tb := msg.get("traceback"):
                db.add_log(jid, tb, "debug")
            self.busy.pop(runner_id, None)
            db.touch_runner(runner_id, "online")
            await self.announce_end(jid, status)
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
