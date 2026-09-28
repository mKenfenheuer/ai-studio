"""Serving through llama.cpp, without llama.cpp.

The host in runner/llamacpp.py hands a server token ids and gets text back,
and everything between those two ends is the studio's: the prompt, where a
reply stops, what is held back from the stream, the schema, the bookkeeping of
which servers are running. That half is checked here against a fake server
that streams what it is told to, so it runs anywhere CI does. Whether
llama.cpp itself converts and answers is a question for a machine that has it.

    python scripts/check-llamacpp.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("AI_STUDIO_DATA", tempfile.mkdtemp(prefix="ai-studio-llama-"))

from runner import capabilities, grammar as grammars, inference, llamacpp  # noqa: E402

FAILED: list[str] = []


def check(name: str, got: object, want: object = True) -> None:
    ok = got == want
    print("  %s  %s%s" % ("ok  " if ok else "FAIL", name,
                          "" if ok else "   -> %r, wanted %r" % (got, want)))
    if not ok:
        FAILED.append(name)


class ToyTokenizer:
    """One id per character: enough for a prompt to become token ids."""
    eos_token, bos_token, pad_token, unk_token = "<|im_end|>", None, None, None
    chat_template = None

    def __call__(self, text: str, add_special_tokens: bool = True):
        return type("Enc", (), {"input_ids": [ord(c) for c in text]})()

    def get_added_vocab(self) -> dict:
        return {}


class FakeServer:
    """Streams `pieces`, then reports how it stopped, like llama-server."""

    def __init__(self, pieces: list[str], stop_type: str = "eos",
                 delay: float = 0.0):
        self.pieces, self.stop_type, self.delay = pieces, stop_type, delay
        self.n_ctx = 4096
        self.config = type("Cfg", (), {"max_position_embeddings": 4096})()
        self.bodies: list[dict] = []
        self.closed = False
        self.dead = False

    def alive(self) -> bool:
        return not self.dead

    def close(self) -> None:
        self.closed = True

    def complete(self, body, on_piece, should_stop) -> dict:
        self.bodies.append(body)
        for n, piece in enumerate(self.pieces):
            if self.delay:
                time.sleep(self.delay)
            if should_stop() or on_piece(piece) is False:
                return {}
        return {"stop": True, "stop_type": self.stop_type,
                "tokens_predicted": len(self.pieces),
                "timings": {"prompt_n": 7, "prompt_per_second": 40.0}}


CAPS = {"backend": "cpu", "cpu_serving": "gguf", "attention": {"math": True}}


def host_with(server: FakeServer, job_id: str = "job_a") -> llamacpp.LlamaCppHost:
    host = llamacpp.LlamaCppHost("http://controller", "t", CAPS)

    def fake_load(spec, _path, _quantize, _log):
        return inference._Resident(spec["job_id"], server, ToyTokenizer(), None,
                                   {"eos_token": "<|im_end|>"}, True, 7.0, [])
    host._load = fake_load
    host._fetch = lambda job, log: Path("/nowhere")
    return host


SPEC = {"job_id": "job_a", "format": {"mode": "chat", "chat_format": "chatml"}}
ASK = [{"role": "user", "content": "Which tags fit a power bill?"}]


def main() -> int:
    print("Choosing the engine")
    os.environ["AI_STUDIO_CPU_QUANTIZATION"] = "gguf"
    check("`gguf` asks for llama.cpp", capabilities.cpu_quantization(), "gguf")
    os.environ["AI_STUDIO_CPU_QUANTIZATION"] = "4bit"
    check("`4bit` still asks for bitsandbytes", capabilities.cpu_quantization(), "4bit")
    os.environ.pop("AI_STUDIO_CPU_QUANTIZATION")
    check("and nothing asks for neither", capabilities.cpu_quantization(), None)
    check("a processor set to gguf answers through llama.cpp",
          type(inference.host_for("u", "t", CAPS)).__name__, "LlamaCppHost")
    check("a card never does",
          type(inference.host_for("u", "t", {"backend": "cuda",
                                               "cpu_serving": "gguf"})).__name__,
          "ModelHost")

    print("\nWhat the server is asked")
    server = FakeServer(["electricity", ", utility", ", invoice"])
    host = host_with(server)
    seen: list[tuple[str, str]] = []
    out = host.generate(SPEC, ASK, {"temperature": 0, "max_new_tokens": 40},
                        lambda d, ch: seen.append((ch, d)), lambda _l: None)
    body = server.bodies[0]
    check("the prompt goes as token ids, rendered by the studio",
          isinstance(body["prompt"], list) and "".join(map(chr, body["prompt"]))
          .endswith("<|im_start|>assistant\n"))
    check("the prompt is cached between requests", body.get("cache_prompt"), True)
    check("the format's stop strings travel with it",
          "<|im_end|>" in body.get("stop", []))
    check("and the budget and sampling", (body["n_predict"], body["temperature"]),
          (40, 0.0))
    check("the reply comes back whole", out["text"], "electricity, utility, invoice")
    check("and streamed as content",
          "".join(d for ch, d in seen if ch == "content"), out["text"])
    check("an end of sequence is a finished reply", out["stop_reason"], "end")
    check("the server's token count is the one reported", out["tokens"], 3)

    print("\nWhere a reply stops")
    host = host_with(FakeServer(["tags: a, b", "<|im_end|>", "<|im_start|>user"]))
    out = host.generate(SPEC, ASK, {}, None, lambda _l: None)
    check("a stop string reaching the text cuts it there", out["text"], "tags: a, b")
    host = host_with(FakeServer(["one", " two"], stop_type="limit"))
    out = host.generate(SPEC, ASK, {"max_new_tokens": 2}, None, lambda _l: None)
    check("running out of budget says so", out["stop_reason"], "length")
    seen = []
    host = host_with(FakeServer(["done.", "<|im_", "end|>"]))
    host.generate(SPEC, ASK, {}, lambda d, ch: seen.append(d), lambda _l: None)
    check("half a marker is never shown", any("<|im_" in d for d in seen), False)
    floor = llamacpp.DEADLINE_FLOOR_S
    check("a processor gets longer than a card's five minutes", floor > 300)
    llamacpp.DEADLINE_FLOOR_S = 0.0
    slow = FakeServer(["a"] * 50, delay=0.05)
    host = host_with(slow)
    out = host.generate(SPEC, ASK, {"deadline_s": 0.3}, None, lambda _l: None)
    llamacpp.DEADLINE_FLOOR_S = floor
    check("a deadline stops it and says so", out["stop_reason"], "timeout")
    check("keeping what was written", 0 < len(out["text"]) < 50)

    print("\nA reply held to a schema")
    schema = {"type": "object", "properties": {"tags": {"type": "array"}},
              "required": ["tags"]}
    spec = {**SPEC, "response_format": {"type": "json_schema",
                                        "json_schema": {"schema": schema}}}
    server = FakeServer(['{"tags": ', '["power"]}'])
    out = host_with(server).generate(spec, ASK, {}, None, lambda _l: None)
    check("llama.cpp is given the schema", server.bodies[0].get("json_schema"), schema)
    check("the model is shown it too",
          "matching this schema" in "".join(map(chr, server.bodies[0]["prompt"])))
    check("and a reply that matches is returned", out["text"], '{"tags": ["power"]}')
    broken = False
    try:
        host_with(FakeServer(['{"labels": 1}'])).generate(spec, ASK, {}, None,
                                                           lambda _l: None)
    except grammars.Broken:
        broken = True
    except Exception as e:  # noqa: BLE001 - reported below
        print("    (raised %s: %s)" % (type(e).__name__, e))
    check("a reply that does not is refused, as on a card", broken)

    print("\nWhich servers are running")
    first, second, third = FakeServer(["x"]), FakeServer(["y"]), FakeServer(["z"])
    servers = {"job_a": first, "job_b": second, "job_c": third}
    host = llamacpp.LlamaCppHost("http://controller", "t", CAPS)
    host._fetch = lambda job, log: Path("/nowhere")
    host._load = lambda spec, _p, _q, _l: inference._Resident(
        spec["job_id"], servers[spec["job_id"]], ToyTokenizer(), None, {}, True,
        None, [])
    host.ensure_loaded({"job_id": "job_a"}, lambda _l: None)
    host.pin("job_a")
    host.ensure_loaded({"job_id": "job_b"}, lambda _l: None)
    host.ensure_loaded({"job_id": "job_c"}, lambda _l: None)
    check("past the limit the oldest undeployed server is stopped",
          (first.closed, second.closed, third.closed), (False, True, False))
    check("and the deployed one is kept", sorted(host.loaded_ids()),
          ["job_a", "job_c"])
    third.dead = True
    fresh = FakeServer(["again"])
    servers["job_c"] = fresh
    host.ensure_loaded({"job_id": "job_c"}, lambda _l: None)
    check("a server that died is started again on the next request",
          host._residents["job_c"].model is fresh)
    host.unload_all(force=True)
    check("unloading stops every server", (first.closed, fresh.closed), (True, True))

    print("\nCache names")
    check("a run", llamacpp._cache_name("job_559a7255aa06"), "job_559a7255aa06-llamacpp")
    check("a Hub model", llamacpp._cache_name("hub:Qwen/Qwen2.5-3B-Instruct"),
          "hub--Qwen--Qwen2.5-3B-Instruct-llamacpp")

    print()
    if FAILED:
        print("%d check(s) failed: %s" % (len(FAILED), ", ".join(FAILED)))
        return 1
    print("All llama.cpp serving checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
