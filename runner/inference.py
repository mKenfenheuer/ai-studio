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

import contextlib
import gc
import threading
import time
from pathlib import Path
from typing import Any, Callable

from common import conversation, formatting

from . import artifacts, capabilities

CACHE_DIR = artifacts.CACHE_DIR

# How long a model may sit loaded with nobody talking to it. The GPU is shared
# with training, and holding 6 GB for a conversation that ended an hour ago
# would block the next run for no reason.
IDLE_UNLOAD_S = 15 * 60

# What a loaded model costs beyond its weights: the key/value cache the
# conversation grows into, attention's working room, and the allocator's own
# fragmentation. Used to describe what a compressed load will take, not to
# decide whether an uncompressed one is attempted -- see _plan_precision for
# why that distinction is worth keeping.
HEADROOM_GB = 1.5


class OutOfRoom(RuntimeError):
    """An out-of-memory that has already been explained in plain language.

    The agent rewrites raw GPU errors into advice, and its advice for an
    out-of-memory is about training -- batch sizes and sequence lengths, none of
    which a person in the playground can change. This says "leave my wording
    alone", so the explanation that knows whether 4-bit was already tried is the
    one that reaches the screen.
    """
    already_explained = True


# How far back a marker still arriving may reach. `<|channel|>final<|message|>`
# is the longest this studio's formats use; anything older than this is text.
_MARKER_REACH = 32


def _showable(text: str, hold: int) -> str:
    """How much of this reply is safe to put on screen.

    Two things are held back. The last `hold` characters, because they may turn
    out to be the opening of a stop sequence -- that rule is older than this
    function. And anything after an unclosed `<`, because a marker arrives one
    character at a time and `<thi` is indistinguishable from text until its
    bracket lands. Showing it means the answer bubble flashes `<think>` and
    then takes it back.

    Only a RECENT unclosed bracket counts. A model writing "5 < 6" is not
    opening anything, and holding the rest of the reply behind it would stall
    the stream for good.
    """
    cut = len(text) - hold if hold else len(text)
    opening = text.rfind("<")
    if opening >= 0 and opening >= len(text) - _MARKER_REACH \
            and ">" not in text[opening:]:
        cut = min(cut, opening)
    return text[:max(cut, 0)]


def _is_oom(e: BaseException) -> bool:
    """Whether this failure was the card running out of room.

    Matched on the message as well as the type, because the same condition
    arrives under several names: `torch.cuda.OutOfMemoryError` on CUDA, a plain
    `RuntimeError` carrying "HIP out of memory" on ROCm, and occasionally a
    bitsandbytes allocation failure that is neither.
    """
    with contextlib.suppress(Exception):
        import torch
        if isinstance(e, torch.cuda.OutOfMemoryError):
            return True
    return "out of memory" in str(e).lower()


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
        self.quantized = False
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
        return artifacts.cached_dir(job_id)

    def _fetch(self, job_id: str, log: Callable[[str], None]) -> Path:
        """Download and unpack this job's result, unless it is already here.

        Shared with training, which needs exactly the same thing to fine-tune
        a model this studio produced. One cache, one set of rules about what a
        half-finished download counts as -- see runner/artifacts.
        """
        return artifacts.fetch(self.controller_url, self.token, job_id, log)

    def unload(self) -> None:
        with self.lock:
            self._unload_locked()

    def _unload_locked(self) -> None:
        self.model = None
        self.tok = None
        self.loaded_id = None
        self.quantized = False
        self._reclaim()

    def _reclaim(self) -> None:
        """Hand the card back everything this process has finished with.

        Dropping the reference is not enough, and this is where the playground
        was running out of memory on a card with nothing loaded on it. A
        transformers model is a graph of objects that refer back to each other,
        so releasing the last name for it leaves a cycle that only the garbage
        collector can break; until it runs, every tensor is still live and the
        allocator is holding every block. `empty_cache` then frees precisely
        nothing, because nothing is free yet.

        The training that ran an hour ago is the usual culprit -- it finished,
        its thread ended, and its model sat in an uncollected cycle holding
        most of the card while the playground tried to load onto what was
        left. So collect first, THEN return the blocks.
        """
        gc.collect()
        if self.device != "cuda":
            return
        import torch
        with contextlib.suppress(Exception):
            torch.cuda.empty_cache()
        # Blocks another process borrowed and has since dropped. Not available
        # on every build, and never worth an exception.
        with contextlib.suppress(Exception):
            torch.cuda.ipc_collect()

    def _free_gb(self) -> float | None:
        """What is actually free on the card right now, not what it has.

        `vram_gb` from the capability probe is the card's size, which is the
        wrong number to plan a load against: the question is how much is free
        *now*, after whatever else this machine has been doing.
        """
        if self.device != "cuda":
            return None
        import torch
        try:
            free, _total = torch.cuda.mem_get_info()
        except Exception:  # noqa: BLE001 - not every backend reports this
            return None
        return free / 1024 ** 3

    def maybe_unload_idle(self) -> bool:
        if self.model is None or not self.last_used:
            return False
        if time.time() - self.last_used < IDLE_UNLOAD_S:
            return False
        self.unload()
        return True

    def _expert_kwargs(self) -> dict:
        """`experts_implementation`, when this backend needs it and the
        installed transformers understands it.

        Checked against the signature rather than tried-and-retried, because a
        retry here would mean downloading and loading a multi-gigabyte model
        twice. See capabilities.expert_kernel for why it is needed at all.
        """
        from transformers import AutoModelForCausalLM
        kernel = capabilities.expert_kernel(self.caps)
        if not kernel:
            return {}
        try:
            import inspect
            params = inspect.signature(
                AutoModelForCausalLM.from_pretrained).parameters
            if "experts_implementation" not in params and \
                    not any(p.kind == p.VAR_KEYWORD for p in params.values()):
                return {}
        except (TypeError, ValueError):
            pass
        return {"experts_implementation": kernel}

    # ------------------------------------------------------------ fitting
    def _can_quantize(self) -> bool:
        """Whether 4-bit can be TRUSTED here, which is not whether it runs.

        The capability probe quantizes a known layer and compares the answer
        against float16 at the one-row shape a model writing a reply uses. On
        some ROCm builds that shape comes back as noise while the wide shapes
        training uses are perfectly accurate -- nothing raises, and the model
        simply answers with rubbish. `4bit` says training may use it; only
        `4bit_decode` says a reply may be generated through it.

        A runner that has not re-probed since this distinction existed reports
        neither, and is trusted for neither. Refusing to load is recoverable;
        serving somebody noise is not.
        """
        return self.device == "cuda" \
            and bool((self.caps.get("quantization") or {}).get("4bit_decode"))

    def _plan_precision(self, spec: dict, log: Callable[[str], None]) -> bool:
        """Whether to compress this model on the way in.

        Only when full precision is *hopeless* -- when the weights alone will
        not fit in what is free. A merely tight fit is attempted as it is and
        compressed only if it actually fails, because quantizing costs answer
        quality and the estimate is not good enough to spend that on a guess:
        a 14.5 GB model on a card with 15.6 GB free fits and answers well, and
        an allowance for the key/value cache added on top of it says it does
        not.

        Decided against what is free *now* rather than against the card's size,
        and before the load rather than after: a model that cannot fit either
        way should not be read off disk twice to find that out.
        """
        params_b = spec.get("params_b")
        free = self._free_gb()
        if not params_b or free is None:
            return False
        if params_b * 2 <= free:
            return False        # room for the weights; try it and see.
        if not self._can_quantize():
            log("This model's weights need about %.1f GB and %.1f GB is free. "
                "This machine cannot compress it to fit -- it is being loaded "
                "as it is." % (params_b * 2, free))
            return False
        log("This model needs about %.1f GB at full precision and only %.1f GB "
            "is free, so it is being loaded in 4-bit (about %.1f GB) rather "
            "than not at all. Answers are slightly worse; nothing else about "
            "the model changes."
            % (params_b * 2, free, params_b * 0.5 + HEADROOM_GB))
        return True

    def ensure_loaded(self, spec: dict, log: Callable[[str], None]) -> None:
        job_id = spec["job_id"]
        if self.loaded_id == job_id and self.model is not None:
            return

        # Whatever was resident is not what is wanted, and it is sitting on the
        # memory the next model needs. This runs even when nothing is loaded:
        # the card may still be holding a finished training run's weights in a
        # cycle nobody has collected. See _reclaim.
        self._unload_locked()
        path = self._fetch(job_id, log)

        quantize = self._plan_precision(spec, log)
        try:
            self._load(spec, path, quantize, log)
        except Exception as e:  # noqa: BLE001 - re-raised below unless it fits
            if not _is_oom(e):
                raise
            # The estimate was optimistic, or something else took the card
            # between planning and loading. Compressing is the one thing left
            # to try, and it is much better than telling somebody their model
            # cannot be talked to.
            self._unload_locked()
            if quantize or not self._can_quantize():
                raise OutOfRoom(
                    "The GPU ran out of memory loading this model%s. The card "
                    "was cleared first, so nothing else is holding it -- the "
                    "model is simply too large for this machine. Serve it "
                    "somewhere with more memory%s."
                    % (" even compressed to 4-bit" if quantize else "",
                       "" if quantize else ", or on a runner with working "
                                           "4-bit support")) from e
            log("That did not fit. Trying again in 4-bit…")
            self._load(spec, path, True, log)

        self.loaded_id = job_id
        self.last_used = time.time()
        log("Ready.")

    def _load(self, spec: dict, path: Path, quantize: bool,
              log: Callable[[str], None]) -> None:
        """Put one model on the card, at the precision asked for."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype_name = self.caps.get("recommended_dtype", "float32")
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(
            dtype_name, torch.float32)
        if self.device == "cpu":
            dtype = torch.float32

        # A mixture of experts has to be told which dispatch to use here too.
        # Training got this right and serving did not, and the failure was
        # invisible until a sparse model was actually talked to: the run
        # finished, the model saved, and the first message came back
        # "grouped gemm is not supported on ROCM". Anywhere a model is
        # constructed needs this, not just the trainer.
        extra = self._expert_kwargs()
        # The same settings the trainer quantizes a frozen base with, so a
        # model served compressed behaves the way it did while it was learning.
        if quantize:
            from transformers import BitsAndBytesConfig
            extra["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            # bitsandbytes places the weights as it quantizes them; a model
            # built this way must not be moved afterwards.
            extra["device_map"] = {"": 0}
        else:
            extra["dtype"] = dtype

        if spec.get("kind") == "pretrain_llm":
            log("Loading your model…")
            self.tok = AutoTokenizer.from_pretrained(str(path))
            self.model = AutoModelForCausalLM.from_pretrained(str(path), **extra)
        else:
            base = spec.get("base_model")
            if base_job := spec.get("base_model_job"):
                # The base can be another run in this studio rather than a
                # Hugging Face id -- someone fine-tuned a model they built
                # here. Fetched the same way the adapter was, unless what that
                # run produced was itself an adapter, in which case the base
                # is still the Hub model underneath it.
                fetched = artifacts.fetch(self.controller_url, self.token,
                                          base_job, log)
                if not (fetched / "adapter_config.json").exists():
                    base = str(fetched)
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
                base, token=spec.get("hf_token"), **extra)
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

        self.quantized = quantize
        if not quantize:
            self.model = self.model.to(self.device)
        self.model = self.model.eval()
        self.model.config.use_cache = True

    # --------------------------------------------------------- generating
    def render(self, spec: dict, messages: list, reasoning: bool = False,
               log: Callable[[str], None] = lambda _s: None) -> tuple[dict, str]:
        """The exact text this model is given, and the format it came from.

        Its own method because two callers need it and they must not drift: a
        chat in the playground, and an evaluation scoring the same model. If
        the evaluation rendered prompts even slightly differently, every score
        it produced would describe a model nobody is actually talking to.
        """
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

        # A run whose recorded format is a Jinja template over the raw dataset
        # ROW rather than over a conversation. Those render a finished example
        # -- they read row['conversations'] and have no notion of "and now the
        # assistant speaks" -- so there is no way to ask one for a prompt.
        #
        # What used to happen was the worst available answer: the style came
        # out as "continue" and the model was sent the bare question, with no
        # turn markers, no system prompt and no tools. A Mistral fine-tune got
        # a naked sentence where it expected [INST], and produced a plausible
        # answer to a question nobody could have asked it properly.
        #
        # The model's own template is the honest substitute. It is the one the
        # base was built with, it is what a row template of this kind is
        # imitating, and it can produce a generation prompt.
        if fmt.get("mode") == "jinja" and self.chat_template \
                and "messages" not in (fmt.get("template") or ""):
            log("This run was trained with a template written against the "
                "dataset's own columns, which cannot be asked for a prompt. "
                "Using %s's own chat template to talk to it instead."
                % (spec.get("base_model") or "the base model"))
            fmt = {k: v for k, v in fmt.items() if k not in ("mode", "template")}
            fmt["chat_template"] = self.chat_template
        fmt.setdefault("specials", self.specials)
        if reasoning:
            fmt["reasoning"] = True
        return fmt, formatting.render_prompt(messages, fmt,
                                             tools=spec.get("tools"),
                                             reasoning=reasoning)

    def cancel(self) -> None:
        self._cancel.set()

    def generate(self, spec: dict, messages: list, params: dict,
                 on_token: Callable[[str, str], None] | None,
                 log: Callable[[str], None]) -> dict:
        """Write the next turn, streaming it as `on_token(delta, channel)`.

        `channel` is "reasoning" or "content", decided here rather than by
        whoever is displaying it: the runner is the only party that knows which
        format this model was trained in, and a reader guessing at markers is a
        second implementation of the one thing that must not disagree.
        """
        import torch

        self._cancel.clear()
        with self.lock:
            self.ensure_loaded(spec, log)
            model, tok = self.model, self.tok
            self.last_used = time.time()

            want_reasoning = bool(params.get("reasoning"))
            fmt, text = self.render(spec, messages, want_reasoning, log)
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
            # Why writing stopped, which the reader is entitled to know. A
            # reply cut off at the token limit and a reply that finished are
            # the same string of text and completely different answers, and
            # only this tells them apart. Nothing else happening means the
            # loop ran to the end of its budget.
            stop_reason = "length"

            produced: list[int] = []
            t0 = time.time()
            cur = ids

            # What the reader has already been shown, per channel. A reply is
            # split as it arrives rather than rearranged once it stops, so a
            # model's working goes to the reasoning panel from the first token
            # instead of being typed into the answer and then taken back.
            shown = {"reasoning": "", "content": ""}

            def deliver(so_far: str) -> None:
                if not on_token:
                    return
                parts = conversation.split_progressive(so_far, fmt,
                                                       want_reasoning)
                for channel, whole in zip(("reasoning", "content"), parts):
                    already = shown[channel]
                    # A split that moved rather than grew -- the working turned
                    # out to be the answer, or a tool call was recognised and
                    # taken out. Nothing is sent for it: the finished reply
                    # travels whole in `generate_done` and is what the screen
                    # ends up showing, so a rewrite here would only be a
                    # flicker on the way to the same place.
                    if len(whole) <= len(already) or not whole.startswith(already):
                        continue
                    on_token(whole[len(already):], channel)
                    shown[channel] = whole

            for _ in range(max_new):
                if self._cancel.is_set():
                    stop_reason = "cancelled"
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
                    stop_reason = "end"
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
                    deliver(full)
                    emitted = full
                    hit_stop = True
                    stop_reason = "end"
                    break

                # Hold back the last few characters, because they may turn out
                # to be the start of a stop sequence -- or of a reasoning
                # marker. Without this the model streams "…green.Human:" to the
                # browser and only then notices it should have stopped, and the
                # reader has already seen the text it was supposed to cut.
                emitted = _showable(full, hold)
                deliver(emitted)

            # The loop can also end at the token limit, at end-of-text, or on a
            # cancel -- and in all three the held-back characters were never
            # suspect. Release them, or every reply stops a few letters short.
            # Only a real stop sequence means the tail should stay cut.
            if not hit_stop and produced:
                emitted = tok.decode(produced, skip_special_tokens=True)
                deliver(emitted)

            self.last_used = time.time()
            elapsed = time.time() - t0
            # Read back through the same module that wrote it. Putting the
            # opener of the reasoning block back, separating the working from
            # the answer, and recognising a tool call are all one job and all
            # format-specific, so they live with the formats rather than here.
            reply = conversation.parse_reply(emitted, fmt,
                                             reasoning_on=want_reasoning)
            return {
                "text": reply["content"] or (emitted if not reply["tool_calls"]
                                             else ""),
                "reasoning": reply["reasoning"],
                # Calls the model actually made, as structure rather than as
                # syntax. This is what lets the playground show "it called
                # get_order(order_id=12345)" and then hand back a result so the
                # conversation can carry on past the call -- which was
                # impossible while a call arrived as an undifferentiated
                # string of angle brackets.
                "tool_calls": reply["tool_calls"],
                "tokens": len(produced),
                "tokens_per_sec": round(len(produced) / max(elapsed, 1e-6), 1),
                "prompt_tokens": prompt_len,
                # "end" the model finished, "length" it ran out of budget,
                # "cancelled" somebody pressed stop.
                "stop_reason": stop_reason,
                "cancelled": stop_reason == "cancelled",
                "prompt_preview": text[-600:],
            }


def clear_cache(job_id: str | None = None) -> None:
    artifacts.clear(job_id)
