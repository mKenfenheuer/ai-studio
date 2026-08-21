"""Running a finished model, so you can actually talk to the thing you trained.

A training run that ends in a download link is only half an answer. This module
loads a completed result back onto the GPU and streams text from it, which is
the only way most people can tell whether the run worked.

Three things make it more than a wrapper around `generate()`:

* **The result is fetched from the controller, not rebuilt.** The runner that
  serves a chat need not be the one that trained the model, and the machine
  that trained it may be long gone. Artifacts live on the controller; a runner
  pulls one down, caches it on its data volume, and can serve it forever.

* **Prompts are formatted the way the model was trained.** A LoRA fine-tune
  that learned "### Instruction:\\n...\\n\\n### Response:\\n" will ignore a bare
  question, and the user concludes the training failed. The controller passes
  the template that was used, and it is applied here.

* **Generation is a hand-written loop.** `TextIteratorStreamer` needs a second
  thread and a queue to stream; doing it directly gives per-token delivery,
  immediate cancellation, and correct incremental decoding of multi-byte
  characters that a byte-level BPE splits across tokens.
"""
from __future__ import annotations

import os
import shutil
import threading
import time
import zipfile
from pathlib import Path
from typing import Any, Callable

import httpx

from common import formatting

CACHE_DIR = Path(os.environ.get("AI_STUDIO_MODEL_CACHE", "/data/models"))

# How long a model may sit loaded with nobody talking to it. The GPU is shared
# with training, and holding 6 GB for a conversation that ended an hour ago
# would block the next run for no reason.
IDLE_UNLOAD_S = 15 * 60


class ModelHost:
    """Keeps at most one model resident and generates from it."""

    def __init__(self, controller_url: str, token: str, caps: dict):
        self.controller_url = controller_url.rstrip("/")
        self.token = token
        self.caps = caps
        self.lock = threading.Lock()
        self.loaded_id: str | None = None
        self.model = None
        self.tok = None
        self.chat_template: str | None = None
        self.specials: dict = {}
        self.last_used = 0.0
        self._cancel = threading.Event()

    # ------------------------------------------------------------ device
    @property
    def device(self) -> str:
        backend = self.caps.get("backend")
        if backend in ("cuda", "rocm"):
            return "cuda"
        return "mps" if backend == "mps" else "cpu"

    # ------------------------------------------------------------ loading
    def _artifact_dir(self, job_id: str) -> Path:
        return CACHE_DIR / job_id

    def _fetch(self, job_id: str, log: Callable[[str], None]) -> Path:
        """Download and unpack this job's result, unless it is already here."""
        dest = self._artifact_dir(job_id)
        if (dest / "config.json").exists() or (dest / "adapter_config.json").exists():
            return dest

        log("Downloading the trained result from the controller…")
        dest.mkdir(parents=True, exist_ok=True)
        zip_path = dest.with_suffix(".zip")
        url = "%s/api/jobs/%s/download" % (self.controller_url, job_id)
        with httpx.stream("GET", url, timeout=600,
                          headers={"X-Runner-Token": self.token}) as r:
            r.raise_for_status()
            with open(zip_path, "wb") as fh:
                for chunk in r.iter_bytes(1 << 20):
                    fh.write(chunk)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(dest)
        zip_path.unlink(missing_ok=True)
        log("Result unpacked.")
        return dest

    def unload(self) -> None:
        with self.lock:
            self._unload_locked()

    def _unload_locked(self) -> None:
        if self.model is None:
            return
        import torch
        self.model = None
        self.tok = None
        self.loaded_id = None
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def maybe_unload_idle(self) -> bool:
        if self.model is None or not self.last_used:
            return False
        if time.time() - self.last_used < IDLE_UNLOAD_S:
            return False
        self.unload()
        return True

    def ensure_loaded(self, spec: dict, log: Callable[[str], None]) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        job_id = spec["job_id"]
        if self.loaded_id == job_id and self.model is not None:
            return

        self._unload_locked()
        path = self._fetch(job_id, log)
        dtype_name = self.caps.get("recommended_dtype", "float32")
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(
            dtype_name, torch.float32)
        if self.device == "cpu":
            dtype = torch.float32

        if spec.get("kind") == "pretrain_llm":
            log("Loading your model…")
            self.tok = AutoTokenizer.from_pretrained(str(path))
            self.model = AutoModelForCausalLM.from_pretrained(str(path), dtype=dtype)
        else:
            base = spec.get("base_model")
            if not base:
                raise ValueError(
                    "This result is an adapter, which needs the model it was "
                    "trained on, but that model is not recorded on the run.")
            log("Loading %s, then applying what you trained…" % base)
            from peft import PeftModel
            self.tok = AutoTokenizer.from_pretrained(str(path)) \
                if (path / "tokenizer_config.json").exists() \
                else AutoTokenizer.from_pretrained(base, token=spec.get("hf_token"))
            self.model = AutoModelForCausalLM.from_pretrained(
                base, dtype=dtype, token=spec.get("hf_token"))
            self.model = PeftModel.from_pretrained(self.model, str(path))

        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        # The template this model actually carries, read off the tokenizer that
        # is about to run. Training reads the same field, so the playground and
        # the run it is talking to cannot disagree about the format.
        template = getattr(self.tok, "chat_template", None)
        if isinstance(template, dict):
            template = template.get("default") or next(iter(template.values()), None)
        self.chat_template = template
        self.specials = {k: v for k, v in (
            ("bos_token", self.tok.bos_token), ("eos_token", self.tok.eos_token),
            ("pad_token", self.tok.pad_token), ("unk_token", self.tok.unk_token))
            if v}

        self.model = self.model.to(self.device).eval()
        self.model.config.use_cache = True
        self.loaded_id = job_id
        self.last_used = time.time()
        log("Ready.")

    # --------------------------------------------------------- generating
    def cancel(self) -> None:
        self._cancel.set()

    def generate(self, spec: dict, messages: list, params: dict,
                 on_token: Callable[[str], None],
                 log: Callable[[str], None]) -> dict:
        import torch

        self._cancel.clear()
        with self.lock:
            self.ensure_loaded(spec, log)
            model, tok = self.model, self.tok
            self.last_used = time.time()

            fmt = dict(spec.get("format") or {})
            # A fine-tune trained with "the model's own format" stores a flag
            # rather than the template text, because the tokenizer is the
            # authoritative copy. Resolve it here, against the tokenizer that
            # is loaded.
            if fmt.get("use_model_template") and not fmt.get("chat_template"):
                if self.chat_template:
                    fmt["chat_template"] = self.chat_template
                else:
                    log("This model carries no chat template; using the plain "
                        "conversation format instead.")
            fmt.setdefault("specials", self.specials)

            want_reasoning = bool(params.get("reasoning"))
            if want_reasoning:
                fmt["reasoning"] = True
            text = formatting.render_prompt(messages, fmt,
                                            tools=spec.get("tools"),
                                            reasoning=want_reasoning)
            stop_texts = spec.get("stop") or formatting.stop_sequences(
                fmt, self.specials)

            ids = tok(text, return_tensors="pt").input_ids.to(self.device)
            prompt_len = ids.shape[1]

            max_new = int(params.get("max_new_tokens", 200))
            temperature = float(params.get("temperature", 0.8))
            top_k = int(params.get("top_k", 50))
            top_p = float(params.get("top_p", 0.95))
            hold = max((len(s) for s in stop_texts), default=0)

            past = None
            emitted = ""
            hit_stop = False
            produced: list[int] = []
            t0 = time.time()
            cur = ids

            for _ in range(max_new):
                if self._cancel.is_set():
                    break
                with torch.no_grad():
                    out = model(input_ids=cur, past_key_values=past, use_cache=True)
                past = out.past_key_values
                logits = out.logits[:, -1, :].float()

                if temperature <= 0:
                    nxt = torch.argmax(logits, dim=-1, keepdim=True)
                else:
                    logits = logits / max(temperature, 1e-5)
                    if top_k > 0:
                        kth = torch.topk(logits, min(top_k, logits.shape[-1]))[0][..., -1, None]
                        logits = logits.masked_fill(logits < kth, float("-inf"))
                    if 0 < top_p < 1:
                        srt, idx = torch.sort(logits, descending=True, dim=-1)
                        cum = torch.softmax(srt, dim=-1).cumsum(dim=-1)
                        # Keep the first token that crosses the threshold, or a
                        # very peaked distribution would leave nothing to pick.
                        drop = cum - torch.softmax(srt, dim=-1) > top_p
                        srt = srt.masked_fill(drop, float("-inf"))
                        logits = torch.full_like(logits, float("-inf")).scatter(1, idx, srt)
                    nxt = torch.multinomial(torch.softmax(logits, dim=-1), 1)

                token_id = int(nxt[0, 0])
                if token_id == tok.eos_token_id:
                    break
                produced.append(token_id)
                cur = nxt

                # Decode the whole continuation each step and emit only what is
                # new. Decoding one token at a time corrupts any character whose
                # bytes a BPE split across two tokens.
                full = tok.decode(produced, skip_special_tokens=True)

                hit = next((s for s in stop_texts if s in full), None)
                if hit:
                    full = full.split(hit)[0]
                    if len(full) > len(emitted):
                        on_token(full[len(emitted):])
                    emitted = full
                    hit_stop = True
                    break

                # Hold back the last few characters, because they may turn out
                # to be the start of a stop sequence. Without this the model
                # streams "…green.Human:" to the browser and only then notices
                # it should have stopped -- the reply is trimmed server-side but
                # the reader has already seen the text it was supposed to cut.
                safe = full[:-hold] if hold else full
                if len(safe) > len(emitted):
                    on_token(safe[len(emitted):])
                    emitted = safe

            # The loop can also end at the token limit, at end-of-text, or on a
            # cancel -- and in all three the held-back characters were never
            # suspect. Release them, or every reply stops a few letters short.
            # Only a real stop sequence means the tail should stay cut.
            if not hit_stop and produced:
                tail = tok.decode(produced, skip_special_tokens=True)
                if len(tail) > len(emitted):
                    on_token(tail[len(emitted):])
                    emitted = tail

            self.last_used = time.time()
            elapsed = time.time() - t0
            # The prompt already opened the reasoning block, so the model's
            # output continues inside it -- put the opener back before
            # splitting, or the first half of its own reply looks like prose.
            whole = emitted
            if want_reasoning and not whole.lstrip().startswith("<"):
                whole = "<think>\n" + whole if "<|channel|>" not in text \
                    else "<|channel|>analysis<|message|>" + whole
            reasoning, answer = formatting.split_reasoning(whole, fmt)
            return {
                "text": answer if want_reasoning and reasoning else emitted,
                "reasoning": reasoning,
                "tokens": len(produced),
                "tokens_per_sec": round(len(produced) / max(elapsed, 1e-6), 1),
                "prompt_tokens": prompt_len,
                "cancelled": self._cancel.is_set(),
                "prompt_preview": text[-600:],
            }


def clear_cache(job_id: str | None = None) -> None:
    if job_id:
        shutil.rmtree(CACHE_DIR / job_id, ignore_errors=True)
    else:
        shutil.rmtree(CACHE_DIR, ignore_errors=True)
