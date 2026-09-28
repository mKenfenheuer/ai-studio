"""Serving on a processor through llama.cpp.

A runner with no card can already hold a model in 4-bit through bitsandbytes
(`AI_STUDIO_CPU_QUANTIZATION=4bit`), and on the lab's Ryzen 5 5600 that turned
out to be a way of saving memory, not of answering: bitsandbytes has no fast
processor kernel for AVX2, unpacks every weight to float32 for every token, and
a Mistral-7B wrote one token in nine seconds. A 6,000-token tagging prompt did
not finish being read inside the five-minute deadline.

llama.cpp is built for exactly this machine. Its Q4_K_M kernels multiply the
packed weights directly, so `AI_STUDIO_CPU_QUANTIZATION=gguf` serves through
it instead:

* **The model is converted here, once.** The run's safetensors go through the
  same converter and quantiser the GGUF export uses, and the result is cached
  beside the model as `<id>-llamacpp`. The first load of a 7B takes minutes;
  every load after that is seconds. An adapter is converted to a llama.cpp
  LoRA and applied on top of its base's GGUF.

* **One `llama-server` per model, on loopback.** A subprocess rather than a
  binding: nothing to compile into the Python environment, and a crash in the
  kernels takes down that server, not the runner.

* **Everything about the conversation stays the studio's.** The prompt is
  rendered by the same code the transformers path uses and tokenized by the
  model's own tokenizer, and llama-server is handed token ids -- so the model
  reads exactly what it would read on a card. The reply comes back as text
  and goes through the same stop, marker, reasoning and tool-call handling.
  A reply held to a JSON schema is constrained by llama.cpp's own grammar and
  then validated here, as on a card.

* **The prompt is cached between requests** (`cache_prompt`). A tagger sends
  the same few thousand tokens of instructions before every document, and
  llama.cpp keeps them: only the part after the shared prefix is read again.

Scoring jobs keep the transformers path -- they read log-probabilities off the
model directly, which a server does not expose -- so this host is what the
runner answers conversations and deployments with, and nothing else.
"""
from __future__ import annotations

import os
import re
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable

import httpx

from common import conversation, formatting

from . import artifacts, grammar as grammars, inference, vision_lm

# The server binary. The CPU image builds it beside llama-quantize; a runner
# installed by hand finds it on PATH (Homebrew's llama.cpp, for one).
SERVER = os.environ.get("AI_STUDIO_LLAMA_SERVER") or (
    "/opt/llama.cpp/llama-server" if os.path.exists("/opt/llama.cpp/llama-server")
    else shutil.which("llama-server") or "/opt/llama.cpp/llama-server")

# What a model is quantised to for serving. Q4_K_M for the reason the export
# defaults to it: a quarter of the size, and the loss small enough that it is
# what everybody ships.
QUANT = (os.environ.get("AI_STUDIO_GGUF_SERVE_QUANT") or "Q4_K_M").upper()

# The context each server is started with, capped at what the model was built
# for. The key/value cache is allocated for all of it up front -- about 2 GB
# for a Mistral-7B at 16k -- so it is a budget, not a free maximum.
CONTEXT = int(os.environ.get("AI_STUDIO_GGUF_CONTEXT", "16384"))

# How many models may be held at once. There is no free-memory figure to plan
# against on a processor, so it is a count: a deployed model and one passing
# conversation. Past it the least recently used undeployed one is stopped.
MAX_RESIDENT = max(1, int(os.environ.get("AI_STUDIO_GGUF_MAX_MODELS", "2")))

# How long a server may take to come up. It memory-maps the file, so this is
# seconds on a warm disk; the allowance is for a cold one.
START_TIMEOUT_S = 600.0


def _tools():
    # Imported late: the export job imports the training module for its
    # Cancelled exception, and nothing in this module should cost that until
    # a model is actually converted.
    from .jobs import export_gguf
    return export_gguf


def available() -> tuple[bool, str]:
    """Whether this machine can serve through llama.cpp, and why not if not.

    All three pieces, because a server with nothing to convert models for it
    is as useless as a converter with no server.
    """
    tools = _tools()
    missing = [name for name, path in (("llama-server", SERVER),
                                       ("convert_hf_to_gguf.py", tools.CONVERTER),
                                       ("llama-quantize", tools.QUANTIZE))
               if not Path(path).exists()]
    if missing:
        return False, "not found on this machine: %s" % ", ".join(missing)
    try:
        out = subprocess.run([SERVER, "--version"], capture_output=True,
                             text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, "llama-server would not start: %s" % str(e)[:160]
    text = (out.stdout or "") + (out.stderr or "")
    if out.returncode != 0:
        return False, "llama-server would not start: %s" % text.strip()[-160:]
    version = next((line.strip() for line in text.splitlines()
                    if line.strip().startswith("version")), "")
    return True, version or "llama-server"


# ----------------------------------------------------------------- converting

class _Log:
    """What the export job's helpers expect of a run: somewhere to log to.

    Less of it than an export shows. Here each line becomes a deployment's
    status, written to the database, and the quantiser reports every one of a
    7B's 291 tensors -- worth nothing to somebody watching "loading".
    """
    _PER_TENSOR = re.compile(r"^\[\s*\d+/\s*\d+\]|^INFO:hf-to-gguf:(blk|gguf:)")

    def __init__(self, log: Callable[[str], None]):
        self._out = log

    def log(self, line: str) -> None:
        if not self._PER_TENSOR.search(line):
            self._out(line)

    @staticmethod
    def should_cancel() -> bool:
        return False


def _cache_name(ref: str) -> str:
    """A cache directory name for a run id or a `hub:` reference."""
    return re.sub(r"[^A-Za-z0-9._-]+", "--", ref).strip("-") + "-llamacpp"


def _gguf_of(folder: Path, ref: str, base_model: str,
             log: Callable[[str], None], keep: set[str]) -> Path:
    """The quantised GGUF of a model folder, converting it if need be.

    Built in a `.partial` directory and renamed into place, like every other
    artifact in the cache, so a conversion interrupted by a restart is debris
    the next eviction removes rather than a half-written file that loads.
    """
    tools = _tools()
    final_dir = artifacts.CACHE_DIR / _cache_name(ref)
    target = final_dir / ("model.%s.gguf" % QUANT)
    if target.exists():
        return target
    if tools._is_vision(folder):
        raise ValueError(
            "This model looks at pictures, and serving through llama.cpp here "
            "handles text models only. Serve it on a machine with a card.")
    params_b = _params_b(folder)
    # The full-precision intermediate is the peak: two bytes a parameter,
    # plus the quantised file written beside it.
    need = int((params_b or 8.0) * 2.7 * artifacts.GB)
    artifacts.ensure_room(need, keep={*keep, final_dir.name}, log=log)

    staging = final_dir.with_name(final_dir.name + ".partial")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    shim = _Log(log)
    src = tools._readable_by_converter(folder, staging / "src")
    src = tools._with_sentencepiece(src, base_model, staging / "spm", shim)
    f16 = staging / "model.f16.gguf"
    log("Converting to GGUF for llama.cpp. This happens once for this model; "
        "a 7B takes a few minutes.")
    t0 = time.time()
    tools._stream([tools.CONVERT_PYTHON, tools.CONVERTER, str(src),
                   "--outfile", str(f16), "--outtype", "f16"], shim)
    log("Converted in %s. Quantising to %s." % (tools._took(time.time() - t0), QUANT))
    t0 = time.time()
    tools._stream([tools.QUANTIZE, str(f16), str(staging / target.name), QUANT], shim)
    f16.unlink(missing_ok=True)
    for shadow in ("src", "spm"):
        shutil.rmtree(staging / shadow, ignore_errors=True)
    shutil.rmtree(final_dir, ignore_errors=True)
    staging.rename(final_dir)
    log("Quantised in %s: %s." % (tools._took(time.time() - t0),
                                  tools._size(target)))
    return target


def _lora_of(adapter: Path, ref: str, base_args: list[str],
             log: Callable[[str], None]) -> Path:
    """An adapter as a llama.cpp LoRA, converting it if need be. Small: the
    base's weights are never read, only its config."""
    tools = _tools()
    final_dir = artifacts.CACHE_DIR / _cache_name(ref)
    target = final_dir / "adapter.lora.f16.gguf"
    if target.exists():
        return target
    staging = final_dir.with_name(final_dir.name + ".partial")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    log("Converting the adapter to a llama.cpp LoRA.")
    tools._stream([tools.CONVERT_PYTHON, tools.LORA_CONVERTER, str(adapter),
                   *base_args, "--outfile", str(staging / target.name),
                   "--outtype", "f16"], _Log(log))
    shutil.rmtree(final_dir, ignore_errors=True)
    staging.rename(final_dir)
    return target


def _params_b(folder: Path) -> float | None:
    """Billions of parameters, from the size of the weights on disk."""
    try:
        size = sum(p.stat().st_size for p in folder.glob("*.safetensors"))
    except OSError:
        return None
    return size / 2e9 if size else None


def _hub_folder(model_id: str, token: str | None,
                log: Callable[[str], None]) -> Path:
    """A Hub model's files on this disk, which the converter reads."""
    from huggingface_hub import snapshot_download
    log("Downloading %s…" % model_id)
    return Path(snapshot_download(
        model_id, token=token,
        allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt",
                        "*.jinja", "*.tiktoken"]))


# -------------------------------------------------------------------- serving

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class LlamaServer:
    """One `llama-server` holding one model, on a loopback port.

    Stands where a transformers model stands in a resident record, so it
    carries a `config` with the one field the length check reads.
    """

    def __init__(self, model: Path, lora: Path | None, n_ctx: int,
                 log: Callable[[str], None]):
        self.n_ctx = n_ctx
        self.config = type("Cfg", (), {"max_position_embeddings": n_ctx})()
        self.port = _free_port()
        self.url = "http://127.0.0.1:%d" % self.port
        # A file rather than a pipe: nobody reads the server's chatter while it
        # runs, and a full pipe would stop it mid-sentence. Kept for the one
        # moment it matters -- a server that dies saying why.
        self._log = tempfile.NamedTemporaryFile(
            "w+", prefix="llama-server-", suffix=".log", delete=False)
        cmd = [SERVER, "-m", str(model), "--host", "127.0.0.1",
               "--port", str(self.port), "-c", str(n_ctx), "-np", "1",
               "-t", str(max(1, os.cpu_count() or 1))]
        if lora:
            cmd += ["--lora", str(lora)]
        self.proc = subprocess.Popen(cmd, stdout=self._log,
                                     stderr=subprocess.STDOUT)
        self._wait_ready(log)

    def _tail(self, lines: int = 12) -> str:
        try:
            with open(self._log.name, encoding="utf-8", errors="replace") as f:
                return "\n".join(f.read().splitlines()[-lines:])
        except OSError:
            return ""

    def _wait_ready(self, log: Callable[[str], None]) -> None:
        deadline = time.time() + START_TIMEOUT_S
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError("llama-server stopped while loading the "
                                   "model (exit %s):\n%s"
                                   % (self.proc.returncode, self._tail()))
            try:
                if httpx.get(self.url + "/health", timeout=5).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
        self.close()
        raise RuntimeError("llama-server did not come up within %d seconds."
                           % START_TIMEOUT_S)

    def alive(self) -> bool:
        return self.proc.poll() is None

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self._log.close()
        with_suppressed(os.unlink, self._log.name)

    def complete(self, body: dict, on_piece: Callable[[str], bool],
                 should_stop: Callable[[], bool]) -> dict:
        """Stream one completion. `on_piece` returns False to stop early.

        Stopping is closing the connection, which llama-server notices and
        stops generating for. A watcher does it from a second thread, because
        the one thing that cannot be interrupted from this one is the prompt
        being read -- the stream says nothing at all until that is done, and a
        cancel or a deadline has to land during it too.
        """
        import json
        final: dict = {}
        done = threading.Event()
        # Started before the request, and closing the client rather than the
        # response: the server sends no headers until the prompt has been
        # read, so for the whole of a long prefill there is no response yet.
        client = httpx.Client(timeout=httpx.Timeout(None, connect=10.0))

        def watch() -> None:
            while not done.wait(0.25):
                if should_stop():
                    with_suppressed(client.close)
                    return
        threading.Thread(target=watch, daemon=True, name="llama-watch").start()
        try:
            with client.stream("POST", self.url + "/completion",
                               json={**body, "stream": True}) as response:
                if response.status_code != 200:
                    raise RuntimeError("llama-server refused the request "
                                       "(%d): %s" % (response.status_code,
                                                     response.read()[:300]))
                for line in response.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = json.loads(line[6:])
                    if piece := data.get("content"):
                        if on_piece(piece) is False:
                            break
                    if data.get("stop"):
                        final = data
                        break
        except (httpx.HTTPError, httpx.StreamError, RuntimeError) as e:
            if not should_stop():
                if isinstance(e, RuntimeError) and "llama-server" in str(e):
                    raise
                raise RuntimeError("llama-server dropped the reply: %s\n%s"
                                   % (e, self._tail())) from e
        finally:
            done.set()
            with_suppressed(client.close)
        return final


def with_suppressed(fn: Callable, *args: Any) -> None:
    try:
        fn(*args)
    except Exception:  # noqa: BLE001 - cleanup must not raise
        pass


class LlamaCppHost(inference.ModelHost):
    """The model host, answering through llama.cpp instead of transformers.

    Fetching, residency, pinning and the prompt are inherited unchanged; what
    is replaced is what a model *is* (a server) and how the next piece of the
    reply is produced.
    """

    def _plan_precision(self, spec: dict, log: Callable[[str], None],
                        path: Path | str | None = None) -> bool:
        return True                     # always quantised: that is the point

    def _make_room(self, need_gb: float | None,
                   log: Callable[[str], None]) -> None:
        """Stop the least recently used undeployed servers past the limit."""
        while len(self._residents) >= MAX_RESIDENT:
            victim = next((r for r in self._residents.values()
                           if r.job_id not in self._pinned), None)
            if victim is None:
                return
            self._unload_locked(victim.job_id)
            log("Stopped %s to make room: this machine holds %d model%s at "
                "once." % (victim.job_id, MAX_RESIDENT,
                           "" if MAX_RESIDENT == 1 else "s"))

    def _unload_locked(self, job_id: str | None = None,
                       keep_pinned: bool = False) -> None:
        before = dict(self._residents)
        super()._unload_locked(job_id, keep_pinned)
        for gone_id, resident in before.items():
            if gone_id not in self._residents:
                with_suppressed(resident.model.close)

    def _ensure_loaded_locked(self, spec: dict,
                              log: Callable[[str], None]) -> None:
        # A server that died since it was loaded is not a model on the card.
        # Dropped here so the load below puts it back, rather than the next
        # request failing on a connection refused.
        resident = self._residents.get(spec["job_id"])
        if resident is not None and not resident.model.alive():
            log("The llama.cpp server for this model had stopped; starting it "
                "again.")
            self._residents.pop(spec["job_id"], None)
            self._refresh_view()
        super()._ensure_loaded_locked(spec, log)

    def _load(self, spec: dict, path: Path | str, quantize: bool,
              log: Callable[[str], None]) -> inference._Resident:
        from transformers import AutoTokenizer

        token = spec.get("hf_token")
        keep = set(self._residents) | {spec["job_id"]}
        lora = None
        if isinstance(path, str):                       # a Hub id
            folder = _hub_folder(path, token, log)
            tok = AutoTokenizer.from_pretrained(str(folder))
            gguf = _gguf_of(folder, spec["job_id"], path, log, keep)
        elif (path / "adapter_config.json").exists():
            base = spec.get("base_model")
            base_folder: Path | None = None
            if base_job := spec.get("base_model_job"):
                fetched = artifacts.fetch(self.controller_url, self.token,
                                          base_job, log, keep=keep | {base_job})
                if not (fetched / "adapter_config.json").exists():
                    base_folder, base_ref = fetched, base_job
            if base_folder is None:
                if not base:
                    raise ValueError(
                        "This result is an adapter, which needs the model it "
                        "was trained on, but that model is not recorded on "
                        "the run.")
                base_folder, base_ref = _hub_folder(base, token, log), "hub:" + base
            log("Serving the adapter on top of %s through llama.cpp."
                % (base or base_ref))
            gguf = _gguf_of(base_folder, base_ref, base or "", log, keep)
            lora = _lora_of(path, spec["job_id"],
                            ["--base", str(base_folder)], log)
            tok = AutoTokenizer.from_pretrained(
                str(path) if (path / "tokenizer_config.json").exists()
                else str(base_folder))
        else:
            if vision_lm.is_vision_dir(path):
                raise ValueError(
                    "This model looks at pictures, and serving through "
                    "llama.cpp here handles text models only. Serve it on a "
                    "machine with a card.")
            tok = AutoTokenizer.from_pretrained(str(path))
            gguf = _gguf_of(path, spec["job_id"], spec.get("base_model") or "",
                            log, keep)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        n_ctx = CONTEXT
        try:
            from transformers import AutoConfig
            src = path if isinstance(path, str) else (
                path if (path / "config.json").exists() else None)
            if src is not None:
                built_for = int(getattr(AutoConfig.from_pretrained(
                    str(src), token=token), "max_position_embeddings", 0) or 0)
                if built_for:
                    n_ctx = min(n_ctx, built_for)
        except Exception:  # noqa: BLE001 - the default is a fine context
            pass
        log("Starting llama.cpp with %s of context…" % f"{n_ctx:,}")
        server = LlamaServer(gguf, lora, n_ctx, log)

        template = getattr(tok, "chat_template", None)
        if isinstance(template, dict):
            template = template.get("default") or next(iter(template.values()), None)
        specials = {k: v for k, v in (
            ("bos_token", tok.bos_token), ("eos_token", tok.eos_token),
            ("pad_token", tok.pad_token), ("unk_token", tok.unk_token)) if v}
        return inference._Resident(spec["job_id"], server, tok, template,
                                   specials, True, spec.get("params_b"),
                                   inference._added_tokens(tok))

    def diagnostics(self) -> dict:
        out = dict(self.last_request)
        out["quantized"] = True
        out["engine"] = "llama.cpp %s" % QUANT
        return out

    def generate(self, spec: dict, messages: list, params: dict,
                 on_token: Callable[[str, str], None] | None,
                 log: Callable[[str], None]) -> dict:
        """The same turn the transformers host writes, written by llama.cpp.

        Everything around the model is the parent's code or a copy of its
        text-level half: render, stop texts, held-back markers, the reasoning
        split, the reply parser and the schema check.
        """
        self._cancel.clear()
        with self.lock:
            self.ensure_loaded(spec, log)
            server, tok = self.model, self.tok
            self.last_used = time.time()
            want_reasoning = bool(params.get("reasoning"))

            kind, schema = grammars.schema_of(spec.get("response_format"))
            grammar = None
            if kind not in (grammars.TEXT, None):
                if kind not in (grammars.JSON_OBJECT, grammars.JSON_SCHEMA):
                    raise grammars.Unsupported(
                        "`response_format` may be \"text\", \"json_object\" or "
                        "\"json_schema\"; this run was asked for %r." % kind)
                if kind == grammars.JSON_SCHEMA and not isinstance(schema, dict):
                    raise grammars.Unsupported(
                        "`response_format: \"json_schema\"` needs a schema to "
                        "enforce, at `response_format.json_schema.schema`.")
                # No token enforcer: llama.cpp holds the reply to the schema
                # itself. The object is for its instruction and its check.
                grammar = grammars.Grammar(None, kind, schema)
                messages = inference._with_instruction(messages,
                                                       grammar.instruction())
            if any(mm.get("kind") == "image" for m in messages
                   for mm in (m.get("media") or [])):
                log("This model is served as text only here; it is answering "
                    "the words alone.")

            fmt, text = self.render(spec, messages, want_reasoning, log)
            stop_texts = spec.get("stop") or formatting.stop_sequences(
                fmt, self.specials)
            resident = self._residents.get(spec["job_id"])
            markers = formatting.boundary_markers(
                fmt, self.specials,
                added=resident.added_tokens if resident else None)

            ids = tok(text).input_ids
            prompt_len = len(ids)
            max_new = int(params.get("max_new_tokens", 512))
            self.last_request = {
                "job_id": spec.get("job_id"), "prompt_tokens": prompt_len,
                "max_new_tokens": max_new, "boundary_markers": len(markers),
                "resident": self.loaded_ids(), "engine": "llama.cpp",
                "context_length": server.n_ctx,
            }
            if refusal := inference.length_refusal(prompt_len, max_new,
                                                   server.n_ctx, None):
                raise refusal

            body: dict = {
                "prompt": ids,
                "n_predict": max_new,
                "temperature": float(params.get("temperature", 0.8)),
                "top_k": int(params.get("top_k", 50)),
                "top_p": float(params.get("top_p", 0.95)),
                "cache_prompt": True,
                "stop": list(stop_texts),
            }
            if grammar is not None:
                body["json_schema"] = schema if kind == grammars.JSON_SCHEMA \
                    else {"type": "object"}

            t0 = time.time()
            deadline = t0 + float(params.get("deadline_s")
                                  or inference.GENERATION_DEADLINE_S)
            state = {"full": "", "emitted": "", "hit_stop": False,
                     "off_template": False, "reason": None}
            shown = {"reasoning": "", "content": ""}

            def deliver(so_far: str) -> None:
                if not on_token:
                    return
                parts = conversation.split_progressive(so_far, fmt, want_reasoning)
                for channel, whole in zip(("reasoning", "content"), parts):
                    already = shown[channel]
                    if len(whole) <= len(already) or not whole.startswith(already):
                        continue
                    on_token(whole[len(already):], channel)
                    shown[channel] = whole

            def on_piece(piece: str) -> bool:
                full = state["full"] = state["full"] + piece
                hit = next((s for s in stop_texts if s in full), None)
                if hit:
                    full = full.split(hit)[0]
                    state.update(emitted=full, hit_stop=True, reason="end")
                    deliver(full)
                    return False
                if (cut := formatting.reasoning_violation(full)) is not None:
                    full = full[:cut]
                    state.update(emitted=full, hit_stop=True, reason="end",
                                 off_template=True)
                    deliver(full)
                    return False
                state["emitted"] = full[:len(full) - formatting.held(full, markers)]
                deliver(state["emitted"])
                return True

            def should_stop() -> bool:
                if self._cancel.is_set():
                    state["reason"] = "cancelled"
                    return True
                if time.time() > deadline:
                    state["reason"] = "timeout"
                    return True
                return False

            final = server.complete(body, on_piece, should_stop)

            if state["reason"] is None:
                stop_type = final.get("stop_type") or ""
                state["reason"] = "length" if stop_type == "limit" else "end"
            if not state["hit_stop"] and state["full"]:
                state["emitted"] = state["full"]
                deliver(state["emitted"])
            emitted = state["emitted"]

            produced = int(final.get("tokens_predicted") or 0)
            if not produced and state["full"]:
                # Stopped from this side, so the server never sent its count.
                produced = len(tok(state["full"], add_special_tokens=False).input_ids)
            timings = final.get("timings") or {}
            self.last_request.update(
                prompt_read=timings.get("prompt_n"),
                prompt_per_second=round(timings.get("prompt_per_second") or 0, 1),
                cached_prompt=final.get("tokens_cached"))
            self.last_used = time.time()
            elapsed = time.time() - t0
            reply = conversation.parse_reply(emitted, fmt,
                                             reasoning_on=want_reasoning)
            content = formatting.strip_special(reply["content"])
            reasoning = formatting.strip_special(reply["reasoning"])
            if grammar is not None:
                grammar.validate(content)
            return {
                "text": content or (
                    "" if reply["tool_calls"] or reasoning else emitted),
                "reasoning": reasoning,
                "tool_calls": reply["tool_calls"],
                "tokens": produced,
                "tokens_per_sec": round(produced / max(elapsed, 1e-6), 1),
                "seconds": round(elapsed, 3),
                "prompt_tokens": prompt_len,
                "stop_reason": state["reason"],
                "off_template": state["off_template"],
                "cancelled": state["reason"] == "cancelled",
                "prompt_preview": text[-600:],
            }
