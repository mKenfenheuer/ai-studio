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

from . import artifacts, capabilities, grammar as grammars

CACHE_DIR = artifacts.CACHE_DIR

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


# How many prompt tokens go through the model at once before writing starts.
#
# Attention without flash-attention kernels costs memory with the SQUARE of the
# length it is handed, which is why this studio's capability probe warns about
# cards that lack them. Measured on a 16 GB card serving a 7B, with 1.95 GB left
# after the weights: a 1,024-token prompt in one pass peaks at 0.51 GB, 2,048
# peaks at 1.59, and 3,072 does not fit at all. That is not a long
# conversation -- it is about fifteen exchanges.
#
# Fed in slices the same prompt costs a slice against the context so far, which
# grows linearly rather than quadratically and is bounded by this number. The
# arithmetic is identical either way; only the order changes.
PREFILL_CHUNK = 256

# And the same number where attention IS fused. A flash or memory-efficient
# kernel never materialises the scores matrix, so the quadratic term the chunk
# above exists to bound is not there to bound: the cost of a slice is linear in
# its length, and slicing finely just means more launches for the same work.
#
# Not unlimited, because the chunk still bounds the activations of one forward
# pass and the allocator still has to find a block for them. 2048 is where
# those stop being free on a 16 GB card, and it is eight times fewer passes
# over a long conversation.
PREFILL_CHUNK_FUSED = 2048

# Room kept for the working set of one prefill slice and the allocator's slack,
# on top of the conversation's own key/value cache. Measured: a 256-token slice
# peaks around 0.11 GB against a 7B, so this is that with room to be wrong.
PREFILL_HEADROOM_GB = 0.6

# ...and the same, per token of chunk, so the two move together. A larger slice
# is a larger working set, and a fused kernel raises the slice eightfold. This
# is the number above divided by the chunk it was measured at, which is what
# makes it a measurement rather than two constants that have to be kept in
# step by hand.
PREFILL_HEADROOM_PER_TOKEN = PREFILL_HEADROOM_GB / PREFILL_CHUNK

# How long one reply may take before it is stopped and handed back as it
# stands. A runner answers one message at a time, so this is not really a limit
# on the reply -- it is a limit on how long everybody else waits.
#
# Measured on the card this was written against: 8.1 tokens a second, so a
# 4,096-token budget is eight and a half minutes of the machine being busy and
# every other message refused. The reply that comes back at the deadline is a
# real reply, marked as cut short, which is worth more than a runner nobody
# else can reach.
GENERATION_DEADLINE_S = 300.0

def _added_tokens(tok) -> list[str]:
    """Every string this tokenizer treats as a token of its own.

    Part of the inventory a reply is buffered against. These are exact -- the
    model's own vocabulary, not a guess at what its template looks like -- and
    the ones that are *not* marked special are precisely the ones that survive
    `skip_special_tokens` and reach the screen a character at a time.
    """
    out: list[str] = []
    with contextlib.suppress(Exception):
        out += [str(t) for t in (tok.all_special_tokens or [])]
    with contextlib.suppress(Exception):
        out += [str(t) for t in (tok.get_added_vocab() or {})]
    return list(dict.fromkeys(t for t in out if t))


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


def _with_instruction(messages: list, instruction: str) -> list:
    """The conversation with the format instruction added to its system turn.

    Appended to the existing system message rather than inserted as a second
    one. Two system turns is a shape most chat templates have never seen -- a
    few render only the first, and a few refuse outright -- so a reply would
    come back unconstrained-looking for a reason nobody could find from the
    request. A conversation with no system turn at all gets one.
    """
    out = [dict(m) for m in messages]
    for m in out:
        if m.get("role") in ("system", "developer"):
            existing = (m.get("content") or "").rstrip()
            m["content"] = ("%s\n\n%s" % (existing, instruction) if existing
                            else instruction)
            return out
    return [{"role": "system", "content": instruction}] + out


class _Resident:
    """One model on the card, and everything that describes it.

    A record rather than fields on the host, because the host now keeps
    several: talking to two models in turn used to reload both of them on
    every message.
    """
    __slots__ = ("job_id", "model", "tok", "chat_template", "specials",
                 "quantized", "params_b", "added_tokens", "last_used",
                 # Work that belongs to this model and costs too much to redo
                 # per request. Only the grammar's vocabulary table so far,
                 # which takes tens of seconds to build on a 152k vocabulary
                 # and depends on nothing but the tokenizer.
                 "derived")

    def __init__(self, job_id: str, model, tok, chat_template, specials,
                 quantized: bool, params_b: float | None,
                 added_tokens: list[str]):
        self.job_id = job_id
        self.derived: dict = {}
        self.model = model
        self.tok = tok
        self.chat_template = chat_template
        self.specials = specials
        self.quantized = quantized
        self.params_b = params_b
        self.added_tokens = added_tokens
        self.last_used = time.time()


class ModelHost:
    """Keeps models resident on the card and generates from them.

    Residency is least-recently-used and bounded by memory rather than by
    time. A model is dropped when the card needs the room -- for another model,
    or because a training run is starting -- and not because a conversation
    went quiet: an idle timer means the 9am and the 11am chat each pay a full
    load, for space nobody was waiting for.
    """

    def __init__(self, controller_url: str, token: str, caps: dict):
        self.controller_url = controller_url.rstrip("/")
        self.token = token
        self.caps = caps
        self.lock = threading.Lock()
        # Insertion-ordered, oldest use first: the next one to go.
        self._residents: dict[str, _Resident] = {}
        # Models somebody deployed to this machine on purpose. They are exempt
        # from the eviction below: the point of a deployment is that the model
        # is *there*, and a resident that any passing conversation can push off
        # the card is not a deployment, it is a cache that happened to be warm.
        #
        # Training still takes the whole card -- see unload_all -- because a
        # run sized against an empty card and given a card with a deployment on
        # it fails two minutes in. A machine reserved for serving takes no
        # training runs, which is the arrangement this is meant for.
        self._pinned: set[str] = set()
        # A view onto the most recently used resident. Everything that reads a
        # model -- render, generate, the context budget, the diagnostics, and
        # the two job kinds that build their own host -- goes through these, so
        # holding more than one model changed nothing outside this class.
        self.loaded_id: str | None = None
        self.model = None
        self.tok = None
        self.chat_template: str | None = None
        self.specials: dict = {}
        self.last_used = 0.0
        self.quantized = False
        # Whether this transformers understands being asked for one position's
        # logits. Settled on first use, because finding out costs a forward
        # pass. See _prefill.
        self._logits_to_keep = None
        # Enough about the request in flight to explain a failure. See
        # diagnostics().
        self.last_request: dict = {}
        self._cancel = threading.Event()
        # The same attention plan a training run gets, and for the same
        # reasons: serving materialises a scores matrix the size of the
        # conversation squared on a card with no fused kernel, and it is the
        # term that decides how long a conversation can get. Settled once, in
        # the process that will load the models, because part of the plan is
        # environment `from_pretrained` reads.
        self.attn = capabilities.attention_plan(caps)
        self.prefill_chunk = (PREFILL_CHUNK if self.attn["quadratic"]
                              else PREFILL_CHUNK_FUSED)
        self.prefill_headroom_gb = (self.prefill_chunk
                                    * PREFILL_HEADROOM_PER_TOKEN)

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

        `keep` names the models on the card right now: the cache is allowed to
        evict to make room, and deleting the directory under a loaded model is
        the one thing it must never do.
        """
        return artifacts.fetch(self.controller_url, self.token, job_id, log,
                               keep=set(self._residents))

    def loaded_ids(self) -> list[str]:
        """What is on the card, most recently used first."""
        return list(reversed(list(self._residents)))

    def pinned_ids(self) -> list[str]:
        """What is on the card because somebody deployed it here."""
        return [j for j in self._pinned if j in self._residents]

    def pin(self, job_id: str) -> None:
        self._pinned.add(job_id)

    def unpin(self, job_id: str) -> None:
        self._pinned.discard(job_id)

    def unload(self, job_id: str | None = None) -> None:
        """Drop one model, or -- with no argument -- every one of them."""
        with self.lock:
            self._unload_locked(job_id, keep_pinned=job_id is None)

    def unload_all(self, force: bool = False) -> None:
        """Empty the card. `force` takes the deployed models with it.

        Training forces it, because a run was sized against a card with
        nothing on it. Everything else leaves a deployment alone -- the
        controller put it there and is watching for it to come back if it
        goes, so silently dropping it would start a loop of it being put back.
        """
        with self.lock:
            self._unload_locked(None, keep_pinned=not force)

    def _unload_locked(self, job_id: str | None = None,
                       keep_pinned: bool = False) -> None:
        if job_id is None:
            keep = {j: r for j, r in self._residents.items()
                    if keep_pinned and j in self._pinned}
            self._residents.clear()
            self._residents.update(keep)
        else:
            self._residents.pop(job_id, None)
            self._pinned.discard(job_id)
        self._refresh_view()
        self._reclaim()

    def _refresh_view(self) -> None:
        """Point the plain attributes at the most recently used resident."""
        current = next(reversed(self._residents.values()), None) \
            if self._residents else None
        self.loaded_id = current.job_id if current else None
        self.model = current.model if current else None
        self.tok = current.tok if current else None
        self.chat_template = current.chat_template if current else None
        self.specials = current.specials if current else {}
        self.quantized = bool(current.quantized) if current else False
        self.last_used = current.last_used if current else self.last_used

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
        """What this process can still allocate, which is two things added up.

        `vram_gb` from the capability probe is the card's size, which is the
        wrong number to plan against: the question is how much is free *now*.
        But the driver's answer is wrong too, in the other direction. Torch
        takes memory from the driver in blocks and keeps them after the tensors
        in them are freed, so the driver counts them as gone while this process
        can still allocate into them freely.

        Left out, the card appears to fill up as a conversation goes on -- every
        turn hands its key/value cache back to the allocator and none of it
        shows as free -- and a budget computed from that shrinks turn by turn
        until it starts refusing conversations that would have fitted.
        """
        if self.device != "cuda":
            return None
        import torch
        try:
            free, _total = torch.cuda.mem_get_info()
        except Exception:  # noqa: BLE001 - not every backend reports this
            return None
        with contextlib.suppress(Exception):
            free += torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
        return free / 1024 ** 3

    def _make_room(self, need_gb: float | None, log: Callable[[str], None]
                   ) -> None:
        """Evict least-recently-used models until this one has somewhere to go.

        Called before the precision is planned, so the decision to compress is
        made against the memory eviction actually released rather than against
        what the card looked like while somebody else's conversation was still
        resident.

        With no estimate to work against it aims at a fixed floor rather than
        clearing the card. Clearing it was the old behaviour and it was quietly
        expensive: a model with no `params_b` recorded -- a baseline off the
        Hub, a merge made before the field was written -- evicted every other
        model on a card that had room for all of them, and the next message
        evicted it again to put the first one back. Two models that both fit,
        each paying a full load every time somebody switched between them.

        If the floor turns out to be too low the load fails on memory, and the
        out-of-memory path in `ensure_loaded` clears the card and retries. The
        expensive thing still happens when it is genuinely needed; it stopped
        happening when it was not.
        """
        if need_gb is None:
            need_gb = UNKNOWN_NEED_GB
        while True:
            free = self._free_gb()
            if free is None or free >= need_gb:
                return
            # Deployed models are not candidates. If the only thing left on the
            # card is one of those, there is no more room to make: the load
            # goes on and either fits compressed or fails saying so, which is
            # the right outcome -- taking somebody's deployed model down to
            # serve one passing message is not.
            victim = next((r for r in self._residents.values()
                           if r.job_id not in self._pinned), None)
            if victim is None:
                if self._residents:
                    log("%.1f GB free, about %.1f GB needed, and everything "
                        "resident is deployed here on purpose. Loading into "
                        "what is left."
                        % (self._free_gb() or 0.0, need_gb))
                return
            self._residents.pop(victim.job_id, None)
            self._refresh_view()
            self._reclaim()
            log("Unloaded %s to make room: %.1f GB free, about %.1f GB needed."
                % (victim.job_id, self._free_gb() or 0.0, need_gb))

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

    def _attention_kwargs(self) -> dict:
        """`attn_implementation`, when the installed transformers takes it.

        Checked against the signature rather than tried-and-retried, for the
        same reason as the expert kernel above: a retry here would download and
        load a multi-gigabyte model twice.

        A model whose family has no SDPA path still refuses this at load time,
        with a ValueError naming `attn_implementation`. That one is caught
        where the model is loaded, because by then the weights are on the disk
        and the second attempt is cheap.
        """
        from transformers import AutoModelForCausalLM
        try:
            import inspect
            params = inspect.signature(
                AutoModelForCausalLM.from_pretrained).parameters
            if ("attn_implementation" not in params
                    and not any(p.kind == p.VAR_KEYWORD
                                for p in params.values())):
                return {}
        except (TypeError, ValueError):
            pass
        return {"attn_implementation": self.attn["implementation"]}

    def _load_causal_lm(self, name: str, token, extra: dict, log) -> Any:
        """`from_pretrained`, retried without the attention path if refused.

        Model families that have no SDPA implementation raise a ValueError
        rather than falling back, and the message points at the function
        instead of at the argument. Cheap to retry at this point: the weights
        are already on the disk, and the refusal happens before any of them
        are read.
        """
        from transformers import AutoModelForCausalLM
        try:
            return AutoModelForCausalLM.from_pretrained(name, token=token,
                                                        **extra)
        except (TypeError, ValueError) as e:
            message = str(e)
            if "attn_implementation" not in extra or not any(
                    hint in message for hint in
                    ("attn_implementation", "scaled_dot_product_attention")):
                raise
            log("This model has no fused-attention implementation, so it is "
                "being loaded with the library's own.")
            return AutoModelForCausalLM.from_pretrained(
                name, token=token,
                **{k: v for k, v in extra.items() if k != "attn_implementation"})

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
        if resident := self._residents.get(job_id):
            # Already here. Moved to the front of the queue and nothing else:
            # no fetch, no load, no unloading of whatever else is resident.
            # This is the whole point of keeping more than one.
            resident.last_used = time.time()
            self._residents.pop(job_id)
            self._residents[job_id] = resident
            self._refresh_view()
            return

        # A model off the Hub rather than one this studio trained. There is no
        # artifact to fetch: the id goes straight to `from_pretrained`, which
        # downloads and caches it the same way training does. This is what
        # makes a baseline possible -- "is my fine-tune better than the model
        # I started from" cannot be answered while only runs can be loaded.
        if hub_model := spec.get("hub_model"):
            path: Path | str = hub_model
            log("Loading %s from the Hub. Nothing here trained it; it is the "
                "thing being compared against." % hub_model)
        else:
            path = self._fetch(job_id, log)

        # What this model wants at full precision. The eviction aims at that
        # rather than at the compressed size: quantizing costs answer quality
        # and is worth avoiding while there is anything left to evict.
        params_b = spec.get("params_b")
        self._make_room(params_b * 2 + HEADROOM_GB if params_b else None, log)

        quantize = self._plan_precision(spec, log)
        try:
            resident = self._load(spec, path, quantize, log)
        except Exception as e:  # noqa: BLE001 - re-raised below unless it fits
            if not _is_oom(e):
                raise
            # The estimate was optimistic, or something else took the card
            # between planning and loading. Everything else goes, and then
            # compressing is the one thing left to try -- much better than
            # telling somebody their model cannot be talked to.
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
            resident = self._load(spec, path, True, log)

        self._residents[job_id] = resident
        self._refresh_view()
        log("Ready.")

    def _load(self, spec: dict, path: Path | str, quantize: bool,
              log: Callable[[str], None]) -> _Resident:
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
        # And the attention path, for exactly the same reason one paragraph
        # up: the kernel a model was fine-tuned under is the kernel it should
        # answer under, and the card's sequence limit is written against it.
        extra.update(self._attention_kwargs())
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
            # Place the weights on the card as they are read, instead of
            # building the whole model in host memory and copying it across
            # afterwards with `.to(device)`.
            #
            # This is the difference between a 7B model taking about a minute
            # to load and taking two and a half. The old path allocated 14 GB
            # of ordinary RAM, filled it from the safetensors file, allocated
            # 14 GB on the card, copied, and then waited for the host copy to
            # be collected -- and on a machine without 14 GB to spare it went
            # to swap and took very much longer than that. With a device map
            # accelerate memory-maps the file and writes each tensor straight
            # to its final home: one copy instead of two, and the peak host
            # memory is one tensor rather than one model.
            #
            # Only where there is a card to place onto. On a processor-only
            # runner there is no second copy to avoid, and `low_cpu_mem_usage`
            # gets the one part that still applies -- not building an empty
            # model first and then overwriting every weight in it.
            if self.device == "cuda":
                extra["device_map"] = {"": 0}
            elif self.device == "mps":
                extra["device_map"] = {"": "mps"}
            else:
                extra["low_cpu_mem_usage"] = True

        # Asked of the FILES rather than of the run that produced them. A
        # fine-tune now saves its merged model beside its adapter, so the kind
        # of run no longer says which of the two is in this directory -- and an
        # adapter served as a whole model loads as nothing at all.
        is_adapter = isinstance(path, Path) \
            and (path / "adapter_config.json").exists()

        if not is_adapter:
            log("Loading your model…" if isinstance(path, Path)
                else "Downloading %s…" % path)
            token = spec.get("hf_token")
            tok = AutoTokenizer.from_pretrained(str(path), token=token)
            model = self._load_causal_lm(str(path), token, extra, log)
        else:
            base = spec.get("base_model")
            if base_job := spec.get("base_model_job"):
                # The base can be another run in this studio rather than a
                # Hugging Face id -- someone fine-tuned a model they built
                # here. Fetched the same way the adapter was, unless what that
                # run produced was itself an adapter, in which case the base
                # is still the Hub model underneath it.
                fetched = artifacts.fetch(self.controller_url, self.token,
                                          base_job, log,
                                          keep=set(self._residents))
                if not (fetched / "adapter_config.json").exists():
                    base = str(fetched)
            if not base:
                raise ValueError(
                    "This result is an adapter, which needs the model it was "
                    "trained on, but that model is not recorded on the run.")
            log("Loading %s, then applying what you trained…" % base)
            from peft import PeftModel
            tok = AutoTokenizer.from_pretrained(str(path)) \
                if (path / "tokenizer_config.json").exists() \
                else AutoTokenizer.from_pretrained(base, token=spec.get("hf_token"))
            model = self._load_causal_lm(base, spec.get("hf_token"),
                                         extra, log)
            model = PeftModel.from_pretrained(model, str(path))

        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        # The template this model actually carries, read off the tokenizer that
        # is about to run. Training reads the same field, so the playground and
        # the run it is talking to cannot disagree about the format.
        template = getattr(tok, "chat_template", None)
        if isinstance(template, dict):
            template = template.get("default") or next(iter(template.values()), None)
        specials = {k: v for k, v in (
            ("bos_token", tok.bos_token), ("eos_token", tok.eos_token),
            ("pad_token", tok.pad_token), ("unk_token", tok.unk_token))
            if v}

        # Already in the right place when a device map put it there, and
        # moving a model accelerate has dispatched is both pointless and, for
        # a quantized one, actively wrong.
        if not quantize and "device_map" not in extra:
            model = model.to(self.device)
        model = model.eval()
        model.config.use_cache = True
        return _Resident(spec["job_id"], model, tok, template, specials,
                         quantize, spec.get("params_b"), _added_tokens(tok))

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
        # A prompt that is already the exact text the model should continue.
        #
        # Only one caller asks for this and it is worth naming: a published
        # benchmark score is measured on a raw completion -- five worked
        # examples and an unfinished sixth -- and wrapping that in a chat turn
        # is a different prompt with a different number. Everything else goes
        # through the template, which is the whole point of this method.
        if spec.get("raw_prompt"):
            text = "\n\n".join(str(m.get("content") or "") for m in messages)
            return fmt, text
        return fmt, formatting.render_prompt(messages, fmt,
                                             tools=spec.get("tools"),
                                             reasoning=reasoning)

    def _prefill(self, model, ids, log: Callable[[str], None],
                 deadline: float | None = None):
        """Build the key/value cache for the prompt, a slice at a time.

        Returns the cache, positioned so the caller can carry straight on from
        the last token. Nothing is sampled here -- only the last position's
        logits are ever wanted, and asking for them at every position is a
        [1, prompt, vocab] tensor thrown away immediately.
        """
        import torch
        past = None
        total = ids.shape[1]
        for i in range(0, total, self.prefill_chunk):
            if self._cancel.is_set() or (deadline and time.time() > deadline):
                break
            chunk = ids[:, i:i + self.prefill_chunk]
            with torch.no_grad():
                if self._logits_to_keep is not False:
                    try:
                        out = model(input_ids=chunk, past_key_values=past,
                                    use_cache=True, logits_to_keep=1)
                    except TypeError:
                        # An older transformers that computes every position's
                        # logits whether they are wanted or not. Asked once.
                        self._logits_to_keep = False
                        out = model(input_ids=chunk, past_key_values=past,
                                    use_cache=True)
                else:
                    out = model(input_ids=chunk, past_key_values=past,
                                use_cache=True)
            past = out.past_key_values
            del out
            if total > self.prefill_chunk and i == 0:
                log("Reading %s tokens of conversation…" % f"{total:,}")
        return past

    def _context_budget(self, model) -> int | None:
        """How many tokens of conversation this card still has room for.

        The key/value cache is the part that grows with the conversation and
        does not go away again: two tensors per layer per token, for as long as
        the exchange lasts. Everything else is bounded by the prefill chunk.

        Worth computing because the alternative is not a clean failure. A model
        given more than the card can hold does not reliably raise -- on ROCm it
        spills over PCIe and sits at 99% "busy" moving almost nothing, so the
        request hangs for minutes and then dies. Refusing in a sentence is
        better than that, and it is the same arithmetic either way.

        None when the shape cannot be read, which disables the check rather
        than guessing at it.
        """
        cfg = getattr(model, "config", None)
        free = self._free_gb()
        if cfg is None or free is None:
            return None
        layers = getattr(cfg, "num_hidden_layers", 0) or 0
        attn_heads = getattr(cfg, "num_attention_heads", 0) or 0
        # Grouped-query attention caches one key/value per *key* head, which on
        # a 7B is a quarter of the attention heads. Reading the wrong one over-
        # states the cost fourfold and refuses conversations that would fit.
        kv_heads = getattr(cfg, "num_key_value_heads", None) or attn_heads
        dim = getattr(cfg, "head_dim", None) or (
            (getattr(cfg, "hidden_size", 0) or 0) // max(attn_heads, 1))
        if not (layers and kv_heads and dim):
            return None
        per_token = 2 * layers * kv_heads * dim * 2      # key and value, fp16
        room = (free - self.prefill_headroom_gb) * 1024 ** 3
        return int(max(room, 0) // per_token)

    def diagnostics(self) -> dict:
        """What the card and the last request looked like.

        Gathered for an error report rather than for a metric. When serving
        fails the useful questions are all about size -- how long the
        conversation was, how much room was left, whether the model was
        compressed -- and none of them could be answered afterwards, because
        a chat is deliberately never written down.
        """
        out = dict(self.last_request)
        out["quantized"] = self.quantized
        if (free := self._free_gb()) is not None:
            out["free_gb"] = round(free, 2)
        with contextlib.suppress(Exception):
            import torch
            out["peak_gb"] = round(
                torch.cuda.max_memory_allocated() / 1024 ** 3, 2)
            out["vram_gb"] = round(
                torch.cuda.mem_get_info()[1] / 1024 ** 3, 2)
        return out

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

            # What the reply is being held to, if anything. Built before the
            # prompt is rendered, because a grammar puts the schema in front
            # of the model as well as behind the sampler -- see
            # `grammar.Grammar.instruction` for why both are worth doing.
            resident = self._residents.get(spec["job_id"])
            grammar = grammars.build(spec.get("response_format"), tok,
                                     resident.derived if resident else {}, log)
            if grammar is not None:
                messages = _with_instruction(messages, grammar.instruction())

            fmt, text = self.render(spec, messages, want_reasoning, log)
            stop_texts = spec.get("stop") or formatting.stop_sequences(
                fmt, self.specials)
            # Everything this exact template writes around a message, read off
            # the template itself. What the stream may not show half of.
            markers = formatting.boundary_markers(
                fmt, self.specials,
                added=resident.added_tokens if resident else None)

            ids = tok(text, return_tensors="pt").input_ids.to(self.device)
            prompt_len = ids.shape[1]
            max_new = int(params.get("max_new_tokens", 512))
            self.last_request = {
                "job_id": spec.get("job_id"),
                "prompt_tokens": prompt_len,
                "max_new_tokens": max_new,
                "boundary_markers": len(markers),
                "resident": self.loaded_ids(),
            }
            temperature = float(params.get("temperature", 0.8))
            top_k = int(params.get("top_k", 50))
            top_p = float(params.get("top_p", 0.95))

            past = None
            emitted = ""
            hit_stop = False
            # Why writing stopped, which the reader is entitled to know. A
            # reply cut off at the token limit and a reply that finished are
            # the same string of text and completely different answers, and
            # only this tells them apart. Nothing else happening means the
            # loop ran to the end of its budget.
            stop_reason = "length"
            # Set when the reply had to be cut for leaving its own format.
            # Reported rather than shown: the answer it produced is complete
            # and correct, but a model doing this is a model whose training
            # data taught it to, which is worth being able to see.
            off_template = False

            # Refused before anything is spent, where the arithmetic says it
            # cannot end well. `budget` counts the whole exchange, because the
            # reply is cached exactly as the question is.
            budget = self._context_budget(model)
            self.last_request["context_budget"] = budget
            if budget and prompt_len + max_new > budget:
                raise OutOfRoom(
                    "This conversation is too long for the memory left on this "
                    "card: %s tokens of history plus up to %s more of reply, "
                    "against room for about %s. Start a new conversation, or "
                    "lower the length limit."
                    % (f"{prompt_len:,}", f"{max_new:,}", f"{budget:,}"))

            produced: list[int] = []
            t0 = time.time()
            deadline = t0 + float(params.get("deadline_s")
                                  or GENERATION_DEADLINE_S)
            # Everything but the final token goes in as cache; the loop below
            # starts from that token and its logits are the first thing
            # sampled. Feeding the whole prompt to the loop instead is one
            # attention matrix the size of the conversation squared.
            if prompt_len > 1:
                past = self._prefill(model, ids[:, :-1], log, deadline)
            cur = ids[:, -1:]

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
                if time.time() > deadline:
                    # Stopped by the clock rather than by the budget. What has
                    # been written is kept and said to be incomplete; the
                    # alternative is a machine nobody else can use.
                    stop_reason = "timeout"
                    break
                with torch.no_grad():
                    out = model(input_ids=cur, past_key_values=past, use_cache=True)
                past = out.past_key_values
                logits = out.logits[:, -1, :].float()

                # FIRST, before temperature, top-k and top-p. Those three
                # reshape a distribution; this one decides which tokens are in
                # it at all, and a token ruled out by the grammar must not be
                # able to win a sample no matter how confident the model is.
                #
                # Applied by addition rather than assignment so a token the
                # grammar allows keeps its score: the model still chooses, it
                # simply chooses from the legal moves.
                if grammar is not None:
                    allowed = grammar.allowed(produced)
                    if not allowed:
                        # The grammar has painted itself into a corner: no
                        # token continues the reply and none ends it. Said out
                        # loud rather than falling back to unconstrained
                        # sampling, which would hand back the one thing this
                        # whole file exists to prevent -- text that does not
                        # match the format it was promised to match.
                        raise RuntimeError(
                            "The reply could not be completed in the format "
                            "it was asked for: no token left continues it. "
                            "This usually means a schema no text can satisfy.")
                    keep = torch.full_like(logits, float("-inf"))
                    keep[0, torch.tensor(allowed, device=logits.device,
                                         dtype=torch.long)] = 0.0
                    logits = logits + keep

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

                # The same idea as a stop sequence, except that what makes it
                # one depends on what came before: a reasoning block closed
                # twice, or reopened after closing. The model has left the
                # shape it was trained in, and it will not find its own way
                # back -- the end-of-turn token it learned belongs to a format
                # it is no longer writing, so it runs to the limit instead.
                if (cut := formatting.reasoning_violation(full)) is not None:
                    full = full[:cut]
                    deliver(full)
                    emitted = full
                    hit_stop = True
                    stop_reason = "end"
                    off_template = True
                    break

                # Hold back exactly as much of the tail as could still turn
                # into one of this template's boundaries, and not a character
                # more. Without it the model streams "…green.Human:" or
                # "…done.<|im_" to the browser and only then notices it should
                # have stopped, and the reader has already seen the text it was
                # supposed to cut.
                emitted = full[:len(full) - formatting.held(full, markers)]
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
            # Last line of defence, on the finished text rather than the
            # stream. A complete marker the splitter had no use for is a
            # control token that ended up in the middle of a sentence, not
            # something the model meant to say. The buffering above is what
            # keeps it off the screen while it arrives; this is what keeps it
            # out of what is stored and handed back.
            content = formatting.strip_special(reply["content"])
            reasoning = formatting.strip_special(reply["reasoning"])

            # The last word on whether the promise was kept. Checked on the
            # text that is actually about to be returned -- after the format
            # markers have been stripped and the reply reassembled -- because
            # that is the string the caller will parse, and an earlier check
            # would be checking something else. See grammar.Grammar.validate.
            if grammar is not None:
                grammar.validate(content)
            return {
                # The fallback is for a reply nothing could be made of: hand
                # back the raw text rather than nothing. It must NOT fire when
                # the working was recognised, which is what happens to every
                # reply cut off mid-thought -- there is no answer yet, and
                # returning the raw text put the reasoning on screen twice,
                # once in its panel and once with its tags showing.
                "text": content or (
                    "" if reply["tool_calls"] or reasoning else emitted),
                "reasoning": reasoning,
                # Calls the model actually made, as structure rather than as
                # syntax. This is what lets the playground show "it called
                # get_order(order_id=12345)" and then hand back a result so the
                # conversation can carry on past the call -- which was
                # impossible while a call arrived as an undifferentiated
                # string of angle brackets.
                "tool_calls": reply["tool_calls"],
                "tokens": len(produced),
                "tokens_per_sec": round(len(produced) / max(elapsed, 1e-6), 1),
                # Time on the card, not time in the request. The controller
                # files this rather than its own wall clock, so a rate quoted
                # on the operations page is the model's speed and not a
                # measurement of the network between here and the browser.
                "seconds": round(elapsed, 3),
                "prompt_tokens": prompt_len,
                # "end" the model finished, "length" it ran out of budget,
                # "cancelled" somebody pressed stop.
                "stop_reason": stop_reason,
                "off_template": off_template,
                "cancelled": stop_reason == "cancelled",
                "prompt_preview": text[-600:],
            }


def clear_cache(job_id: str | None = None) -> None:
    artifacts.clear(job_id)
