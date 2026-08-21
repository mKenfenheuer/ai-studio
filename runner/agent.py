"""The runner agent.

Connects OUTBOUND to the controller and holds the socket open. That direction
matters: GPU boxes routinely sit behind NAT, on a laptop's wifi, or inside a
container with no routable address. Dialing out means a runner needs no inbound
port, no fixed IP and no firewall rule -- it only needs to reach the controller.
"""
from __future__ import annotations

import asyncio
import json
import os
import queue
import shutil
import tempfile
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

import httpx
import websockets

from . import capabilities, inference
from .jobs import lora_llm, scratch_llm

HEARTBEAT_S = 15
LIVENESS_FILE = os.environ.get("AI_STUDIO_LIVENESS", "/tmp/ai-studio-runner.alive")
RECONNECT_MIN_S = 2
RECONNECT_MAX_S = 30

JOB_HANDLERS = {
    "finetune_llm": lora_llm.run,
    "pretrain_llm": scratch_llm.run,
}


class JobContext:
    """Handed to a training job so it can report progress.

    The job runs in a worker thread; the websocket lives on the event loop.
    Messages cross that boundary through a thread-safe queue, which the loop
    drains. No asyncio primitives are touched from the training thread.
    """

    def __init__(self, job_id: str, outbox: queue.Queue, workdir: str,
                 caps: dict, hf_token: str | None):
        self.job_id = job_id
        self.workdir = workdir
        self.capabilities = caps
        self.hf_token = hf_token
        self._outbox = outbox
        self._cancel = threading.Event()
        self._last_metric_step = -1

    def cancel(self) -> None:
        self._cancel.set()

    def should_cancel(self) -> bool:
        return self._cancel.is_set()

    def _put(self, msg: dict) -> None:
        msg["job_id"] = self.job_id
        self._outbox.put(msg)

    def log(self, line: str, level: str = "info") -> None:
        for part in str(line).splitlines() or [""]:
            self._put({"type": "job_log", "level": level, "line": part})

    def metric(self, step: int, data: dict) -> None:
        self._put({"type": "job_metric", "step": step,
                   "data": {k: v for k, v in data.items() if v is not None}})

    def progress(self, step: int, total: int, stage: str = "") -> None:
        self._put({"type": "job_progress", "step": step, "total": total, "stage": stage})

    def emit_meta(self, meta: dict) -> None:
        self._put({"type": "job_meta", "meta": meta})


class Runner:
    def __init__(self, controller_url: str, token: str, name: str | None = None,
                 runner_id: str | None = None):
        self.controller_url = controller_url.rstrip("/")
        self.token = token
        self.name = name or capabilities.platform.node()
        self.runner_id = runner_id or self._stable_id()
        self.caps: dict = {}
        self.outbox: queue.Queue = queue.Queue()
        self.current: JobContext | None = None
        self._busy = threading.Lock()
        # Serving a finished model is a second job this runner does. It shares
        # the GPU with training, so it is deliberately kept out of the way of
        # a run in progress rather than competing with it for memory.
        self.host: inference.ModelHost | None = None
        self.generating = False

    def _stable_id(self) -> str:
        """Reuse the same identity across restarts so the controller shows one
        runner reconnecting, not a new stranger every time the process bounces."""
        state = Path(os.environ.get("AI_STUDIO_RUNNER_STATE",
                                    Path.home() / ".ai-studio-runner-id"))
        try:
            if state.exists():
                return state.read_text(encoding="utf-8").strip()
            rid = "run_" + uuid.uuid4().hex[:12]
            state.parent.mkdir(parents=True, exist_ok=True)
            state.write_text(rid, encoding="utf-8")
            return rid
        except OSError:
            return "run_" + uuid.uuid4().hex[:12]

    # ------------------------------------------------------------------
    @property
    def ws_url(self) -> str:
        base = self.controller_url.replace("https://", "wss://").replace("http://", "ws://")
        return base + "/api/runner/ws"

    async def start(self) -> None:
        print("[runner] probing hardware, this takes a moment...")
        self.caps = await asyncio.get_event_loop().run_in_executor(None, capabilities.probe)
        print("[runner] %s | %s | %s" % (
            self.caps.get("device_name"), self.caps.get("backend"),
            self.caps.get("arch") or "-"))
        for w in self.caps.get("warnings", []):
            print("[runner] note: %s" % w)
        self.host = inference.ModelHost(self.controller_url, self.token, self.caps)

        backoff = RECONNECT_MIN_S
        while True:
            try:
                await self._session()
                backoff = RECONNECT_MIN_S
            except (OSError, websockets.WebSocketException) as e:
                print("[runner] disconnected (%s); retrying in %ss" % (type(e).__name__, backoff))
            except Exception as e:  # noqa: BLE001 - never let the agent die
                print("[runner] unexpected error: %r; retrying in %ss" % (e, backoff))
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, RECONNECT_MAX_S)

    async def _session(self) -> None:
        async with websockets.connect(self.ws_url, max_size=8 * 1024 * 1024,
                                      ping_interval=20, ping_timeout=20) as ws:
            await ws.send(json.dumps({
                "type": "register",
                "token": self.token,
                "runner_id": self.runner_id,
                "name": self.name,
                "capabilities": self.caps,
            }))
            first = json.loads(await ws.recv())
            if first.get("type") == "error":
                print("[runner] controller rejected us: %s" % first.get("message"))
                raise SystemExit(1)
            print("[runner] connected to %s as %s" % (self.controller_url, self.runner_id))
            self._touch_liveness()

            await asyncio.gather(
                self._recv_loop(ws),
                self._send_loop(ws),
                self._heartbeat(ws),
            )

    async def _recv_loop(self, ws) -> None:
        async for raw in ws:
            msg = json.loads(raw)
            kind = msg.get("type")
            if kind == "job_assign":
                self._start_job(msg["job"])
            elif kind == "job_cancel":
                if self.current and self.current.job_id == msg.get("job_id"):
                    self.current.cancel()
            elif kind == "reprobe":
                self.caps = await asyncio.get_event_loop().run_in_executor(
                    None, capabilities.probe)
                self.outbox.put({"type": "capabilities", "capabilities": self.caps})
            elif kind == "generate":
                self._start_generation(msg)
            elif kind == "generate_cancel":
                if self.host:
                    self.host.cancel()
            elif kind == "unload_model":
                if self.host and not self.generating:
                    self.host.unload()

    async def _send_loop(self, ws) -> None:
        """Drain the training thread's outbox onto the socket.

        A bare blocking `outbox.get()` in an executor would never return when
        this task is cancelled on disconnect, stranding one pool thread per
        reconnect until the pool is exhausted. Polling with a timeout gives
        cancellation a place to land.
        """
        loop = asyncio.get_event_loop()

        def _next():
            try:
                return self.outbox.get(timeout=0.5)
            except queue.Empty:
                return None

        while True:
            msg = await loop.run_in_executor(None, _next)
            if msg is None:
                continue
            await ws.send(json.dumps(msg))

    async def _heartbeat(self, ws) -> None:
        while True:
            await asyncio.sleep(HEARTBEAT_S)
            if self.host and not self.generating:
                # Release the GPU if nobody has spoken to the model in a while;
                # otherwise a finished conversation would keep memory reserved
                # against the next training run.
                if self.host.maybe_unload_idle():
                    print("[runner] unloaded idle model")
            await ws.send(json.dumps({
                "type": "heartbeat",
                "busy": self.current is not None,
                "job_id": self.current.job_id if self.current else None,
                "serving": self.host.loaded_id if self.host else None,
            }))
            self._touch_liveness()

    @staticmethod
    def _touch_liveness() -> None:
        """Record that the agent is alive and connected.

        The container healthcheck reads this file's age instead of importing
        torch and opening a second HIP context. Initialising the GPU again
        every 60s, while the agent already holds it, is both wasteful and
        unreliable on consumer cards -- it reports 'unhealthy' for a runner
        that is training perfectly well.
        """
        try:
            Path(LIVENESS_FILE).write_text(str(int(time.time())), encoding="utf-8")
        except OSError:
            pass

    # ------------------------------------------------------------------
    def _start_job(self, job: dict) -> None:
        if self.current is not None:
            self.outbox.put({"type": "job_rejected", "job_id": job["id"],
                             "reason": "runner already busy"})
            return
        workdir = tempfile.mkdtemp(prefix="aistudio_%s_" % job["id"])
        # Trailing `or None` is required, not decorative: docker-compose renders
        # an unset HF_TOKEN as an empty string, and passing "" to huggingface_hub
        # builds the header "Bearer " and fails every download with
        # "Illegal header value". Absent must mean None, not "".
        token = (job.get("hf_token") or os.environ.get("HF_TOKEN") or "").strip() or None
        ctx = JobContext(job["id"], self.outbox, workdir, self.caps, token)
        self.current = ctx
        threading.Thread(target=self._run_job, args=(job, ctx, workdir),
                         daemon=True, name="job-%s" % job["id"]).start()

    # ------------------------------------------------------------ chat
    def _start_generation(self, msg: dict) -> None:
        rid = msg.get("request_id")
        if self.current is not None:
            self.outbox.put({
                "type": "generate_error", "request_id": rid,
                "error": "This machine is training right now. Wait for the run "
                         "to finish, or use another machine.",
            })
            return
        if self.generating:
            self.outbox.put({
                "type": "generate_error", "request_id": rid,
                "error": "Still answering the previous message.",
            })
            return
        self.generating = True
        threading.Thread(target=self._generate, args=(msg,), daemon=True,
                         name="gen-%s" % rid).start()

    def _generate(self, msg: dict) -> None:
        rid = msg.get("request_id")
        spec = msg.get("spec") or {}
        spec.setdefault("hf_token", os.environ.get("HF_TOKEN") or None)

        def log(line: str) -> None:
            self.outbox.put({"type": "generate_status", "request_id": rid,
                             "status": line})

        def on_token(delta: str) -> None:
            self.outbox.put({"type": "generate_delta", "request_id": rid,
                             "delta": delta})

        try:
            result = self.host.generate(spec, msg.get("prompt", ""),
                                        msg.get("params") or {}, on_token, log)
            self.outbox.put({"type": "generate_done", "request_id": rid, **result})
        except Exception as e:  # noqa: BLE001
            self.outbox.put({"type": "generate_error", "request_id": rid,
                             "error": _friendly_error(e)})
        finally:
            self.generating = False

    def _run_job(self, job: dict, ctx: JobContext, workdir: str) -> None:
        jid = job["id"]
        try:
            self.outbox.put({"type": "job_started", "job_id": jid})
            handler = JOB_HANDLERS.get(job["kind"])
            if handler is None:
                raise ValueError("this runner cannot handle job type %r" % job["kind"])
            result = handler(job["config"], ctx)

            if path := result.pop("artifact_path", None):
                ctx.log("Uploading result to the controller...")
                self._upload(jid, path)
            self.outbox.put({"type": "job_done", "job_id": jid, "summary": result})
        except lora_llm.Cancelled:
            self.outbox.put({"type": "job_cancelled", "job_id": jid})
        except Exception as e:  # noqa: BLE001
            self.outbox.put({
                "type": "job_failed", "job_id": jid,
                "error": _friendly_error(e, job.get("kind")),
                "traceback": traceback.format_exc()[-4000:],
            })
        finally:
            self.current = None
            shutil.rmtree(workdir, ignore_errors=True)

    def _upload(self, job_id: str, path: str) -> None:
        url = "%s/api/jobs/%s/artifact" % (self.controller_url, job_id)
        with open(path, "rb") as fh:
            r = httpx.post(url, files={"file": (Path(path).name, fh, "application/zip")},
                           headers={"X-Runner-Token": self.token}, timeout=600)
            r.raise_for_status()


def _friendly_error(e: Exception, kind: str | None = None) -> str:
    """Translate the errors beginners actually hit into plain language."""
    text = str(e)
    low = text.lower()
    if "out of memory" in low or "hip out of memory" in low:
        if kind == "pretrain_llm":
            return ("The GPU ran out of memory. Training every parameter needs "
                    "far more memory than fine-tuning does. Choose a smaller "
                    "size, or turn on gradient checkpointing in the advanced "
                    "settings to trade speed for memory.")
        return ("The GPU ran out of memory. Try a smaller model, a shorter "
                "sequence length, or reduce the batch size to 1.")
    if "401" in text and "huggingface" in low:
        return ("Hugging Face refused the download. This model is probably "
                "gated -- accept its licence on the model page, then add your "
                "Hugging Face token in Settings.")
    if "403" in text and "gated" in low:
        return ("This model is gated. Accept its licence on Hugging Face and "
                "add an access token in Settings.")
    if "does not appear to have a file named" in low or "not a local folder" in low:
        return ("That model ID was not found on Hugging Face. Check the "
                "spelling -- it should look like 'owner/model-name'.")
    if "trust_remote_code" in low:
        return ("This model needs custom code to run, which is disabled for "
                "safety. Pick a model built on a standard architecture.")
    return text[:600]
