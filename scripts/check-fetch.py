#!/usr/bin/env python
"""Check that a runner fetches a model once, resumably, and only if it is one.

Run it with `python scripts/check-fetch.py` from the repository root.

Written after a 14.5 GB Mistral-7B could not be loaded anywhere and both
runners spent a morning downloading it over and over, stalling the host on
I/O. Three faults, each of which this drives against a real HTTP server
serving the file the way the controller does -- Starlette's FileResponse, with
its ranges and ETags -- rather than against a fake:

* **Two fetches of one model destroyed each other.** Every fetch began by
  deleting the staging directory, including when it held another thread's
  download. A deployment and a chat asking for the same model at once was
  enough.
* **What unpacked was promoted without being looked at.** The fetch whose files
  were deleted mid-unpack carried on with the members still to come -- the
  tokenizer, last alphabetically -- and moved 3.6 MB of tokenizer into place
  as the finished model.
* **An interrupted download started again from zero.** At 11.8 of 14.5 GB.

The replay at the end is that morning's state exactly: a tokenizer-only
directory where the model should be, and a stale partial beside it.
"""
from __future__ import annotations

import io
import json
import os
import shutil
import socket
import sys
import tempfile
import threading
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_TMP = Path(tempfile.mkdtemp(prefix="ai-studio-fetch-"))
os.environ["AI_STUDIO_MODEL_CACHE"] = str(_TMP / "models")
os.environ["AI_STUDIO_DATA"] = str(_TMP / "data")

from runner import artifacts  # noqa: E402

FAILED: list[str] = []
NL = chr(10)


def check(name: str, got: object, want: object = True) -> None:
    ok = got == want
    print("  %s  %s%s" % ("ok  " if ok else "FAIL", name,
                          "" if ok else "   -> %r, wanted %r" % (got, want)))
    if not ok:
        FAILED.append(name)


# --------------------------------------------------------------- the server
SERVED = _TMP / "served"
SERVED.mkdir()
REQUESTS: list[dict] = []          # every download request, as it arrived
SLOW = {"delay": 0.0}              # seconds to wait per megabyte served


def make_zip(path: Path, members: dict[str, bytes]) -> Path:
    """A stored zip, as `artifacts.pack` makes them, with the given files."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
        for name, data in sorted(members.items()):
            z.writestr(name, data)
    return path


def model_members(weight_mb: int = 6) -> dict[str, bytes]:
    """A merged model's files. Big enough that a download takes a while and a
    truncation is a real truncation; the weights are random so a byte out of
    place fails the zip's own checksum."""
    return {
        "config.json": json.dumps({"model_type": "mistral"}).encode(),
        "model-00001-of-00001.safetensors": os.urandom(weight_mb << 20),
        "tokenizer.json": b"{}",
        "tokenizer_config.json": b"{}",
    }


def serve() -> str:
    from starlette.applications import Starlette
    from starlette.responses import FileResponse, Response
    from starlette.routing import Route

    class Slow(FileResponse):
        chunk_size = 1 << 20

        async def __call__(self, scope, receive, send):
            async def paced(message):
                if message.get("type") == "http.response.body" and SLOW["delay"]:
                    import asyncio
                    await asyncio.sleep(SLOW["delay"])
                await send(message)
            await super().__call__(scope, receive, paced)

    async def download(request):
        job_id = request.path_params["job_id"]
        REQUESTS.append({"job": job_id,
                         "range": request.headers.get("range"),
                         "if_range": request.headers.get("if-range")})
        path = SERVED / ("%s.zip" % job_id)
        if not path.exists():
            return Response(status_code=404)
        return Slow(path, media_type="application/zip")

    app = Starlette(routes=[Route("/api/jobs/{job_id}/download", download)])
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    import uvicorn
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port,
                                           log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    return "http://127.0.0.1:%d" % port


def fetch(url: str, job_id: str):
    return artifacts.fetch(url, "token", job_id)


def main() -> int:
    url = serve()

    # ------------------------------------------------------------------
    print(NL + "Two callers asking for one model share one download")
    make_zip(SERVED / "job_a.zip", model_members())
    SLOW["delay"] = 0.05           # long enough that the two really overlap
    results, errors = [], []

    def worker():
        try:
            results.append(fetch(url, "job_a"))
        except Exception as e:  # noqa: BLE001 - reported below
            errors.append(repr(e))

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    SLOW["delay"] = 0.0
    check("nobody failed", errors, [])
    check("every caller got the model", len(results), 3)
    check("and it is a real one, weights and all",
          all((r / "config.json").exists()
              and (r / "model-00001-of-00001.safetensors").exists()
              for r in results))
    check("from exactly one download, not three",
          len([q for q in REQUESTS if q["job"] == "job_a"]), 1)
    check("with no staging directory left behind",
          (artifacts.CACHE_DIR / "job_a.partial").exists(), False)

    # ------------------------------------------------------------------
    print(NL + "An interrupted download carries on from where it stopped")
    whole = make_zip(SERVED / "job_b.zip", model_members())
    size = whole.stat().st_size
    # Ask the server for the ETag it will use, exactly as a first attempt
    # would have recorded it before being killed.
    import httpx
    etag = httpx.get(url + "/api/jobs/job_b/download",
                     headers={"Range": "bytes=0-0"}).headers["etag"]
    REQUESTS.clear()
    staging = artifacts.CACHE_DIR / "job_b.partial"
    staging.mkdir(parents=True)
    cut = size * 3 // 5
    (staging / "artifact.zip").write_bytes(whole.read_bytes()[:cut])
    (staging / "artifact.etag").write_text(etag, encoding="utf-8")
    # ...and some debris from an unpack that did not finish, which must not
    # survive into the result.
    (staging / "tokenizer.json").write_bytes(b"half-written")
    got = fetch(url, "job_b")
    check("the model arrives whole",
          (got / "model-00001-of-00001.safetensors").stat().st_size, 6 << 20)
    check("by asking for the rest, not the start",
          REQUESTS[-1]["range"], "bytes=%d-" % cut)
    check("and naming which file the rest has to belong to",
          REQUESTS[-1]["if_range"], etag)
    check("the unpacked debris did not survive",
          (got / "tokenizer.json").read_bytes(), b"{}")

    # ------------------------------------------------------------------
    print(NL + "A partial of a file that has since changed is not spliced")
    old = make_zip(_TMP / "old.zip", model_members())
    make_zip(SERVED / "job_c.zip", model_members())
    staging = artifacts.CACHE_DIR / "job_c.partial"
    staging.mkdir(parents=True)
    (staging / "artifact.zip").write_bytes(old.read_bytes()[:old.stat().st_size // 2])
    (staging / "artifact.etag").write_text('"an-older-version"', encoding="utf-8")
    REQUESTS.clear()
    got = fetch(url, "job_c")
    check("the result is the new file, whole",
          (got / "model-00001-of-00001.safetensors").read_bytes()
          == zipfile.ZipFile(SERVED / "job_c.zip").read(
              "model-00001-of-00001.safetensors"))
    check("because the old tag was sent and the server refused to resume it",
          REQUESTS[-1]["if_range"], '"an-older-version"')

    print(NL + "A partial with no record of which file it was is not trusted")
    make_zip(SERVED / "job_d.zip", model_members())
    staging = artifacts.CACHE_DIR / "job_d.partial"
    staging.mkdir(parents=True)
    (staging / "artifact.zip").write_bytes(b"PK" + os.urandom(4096))
    REQUESTS.clear()
    got = fetch(url, "job_d")
    check("it is downloaded again from the start",
          REQUESTS[-1]["range"], None)
    check("and arrives whole", (got / "config.json").exists())

    # ------------------------------------------------------------------
    print(NL + "Something that is not a model is refused, not promoted")
    make_zip(SERVED / "job_e.zip", {"tokenizer.json": b"{}",
                                    "tokenizer_config.json": b"{}"})
    try:
        fetch(url, "job_e")
        check("a tokenizer on its own is refused", False)
    except RuntimeError as e:
        check("a tokenizer on its own is refused", True)
        check("saying the artifact is what is wrong",
              "not a model" in str(e))
    check("nothing was moved into place",
          (artifacts.CACHE_DIR / "job_e").exists(), False)
    check("and no staging is left to be resumed from",
          (artifacts.CACHE_DIR / "job_e.partial").exists(), False)

    print(NL + "A download that stops short is kept, to be carried on")
    make_zip(SERVED / "job_f.zip", model_members())

    real_stream = artifacts.httpx.stream

    class Cut:
        """httpx.stream, but the connection drops two megabytes in."""

        def __init__(self, *a, **kw):
            self.inner = real_stream(*a, **kw)

        def __enter__(self):
            r = self.inner.__enter__()
            real_iter = r.iter_bytes

            def short(size=None):
                sent = 0
                for chunk in real_iter(size):
                    if sent >= 2 << 20:
                        return
                    sent += len(chunk)
                    yield chunk
            r.iter_bytes = short
            return r

        def __exit__(self, *exc):
            return self.inner.__exit__(*exc)

    artifacts.httpx.stream = Cut
    try:
        fetch(url, "job_f")
        check("a short download is reported", False)
    except RuntimeError as e:
        check("a short download is reported", "stopped at" in str(e))
    finally:
        artifacts.httpx.stream = real_stream
    partial = artifacts.CACHE_DIR / "job_f.partial" / "artifact.zip"
    check("and what arrived is still there", partial.exists()
          and partial.stat().st_size >= 2 << 20)
    REQUESTS.clear()
    got = fetch(url, "job_f")
    check("the next attempt finishes it from there",
          (REQUESTS[-1]["range"] or "").startswith("bytes=")
          and (got / "config.json").exists())

    # ------------------------------------------------------------------
    print(NL + "This morning, replayed")
    make_zip(SERVED / "job_559a7255aa06.zip", model_members())
    broken = artifacts.CACHE_DIR / "job_559a7255aa06"
    broken.mkdir(parents=True)
    (broken / "tokenizer.json").write_bytes(b"{}")
    (broken / "tokenizer_config.json").write_bytes(b"{}")
    (broken / ".ai_studio_used").write_text("0")
    stale = artifacts.CACHE_DIR / "job_559a7255aa06.partial"
    stale.mkdir()
    (stale / "artifact.zip").write_bytes(os.urandom(1 << 20))
    check("the tokenizer-only directory is not taken for a model",
          artifacts.is_present("job_559a7255aa06"), False)
    got = fetch(url, "job_559a7255aa06")
    check("it is replaced by the model it should have been",
          (got / "config.json").exists()
          and (got / "model-00001-of-00001.safetensors").exists())
    check("and is now taken for one",
          artifacts.is_present("job_559a7255aa06"))
    check("with nothing left over beside it",
          stale.exists(), False)

    # ------------------------------------------------------------------
    print(NL + "A deployment and a conversation never load onto the card at once")
    # The second half of the same morning. With the download fixed, both
    # callers got the model -- and then both loaded it, because a deployment's
    # preload called ensure_loaded without the lock `generate` holds. Two
    # 13.65 GiB copies on a 16 GiB card: the second ran out of memory and
    # reported the model as "simply too large for this machine", which it is
    # not. Measured alone it loads with 2.3 GiB to spare.
    from runner.inference import ModelHost

    host = ModelHost("http://unused", "token", {"backend": "cpu"})
    state = {"now": 0, "most": 0, "loads": 0}
    guard = threading.Lock()

    class FakeResident:
        def __init__(self, job_id):
            self.job_id, self.last_used = job_id, time.time()
            self.model = self.tok = self.chat_template = None
            self.specials, self.quantized = {}, False

    def slow_load(spec, path, quantize, log):
        with guard:
            state["now"] += 1
            state["loads"] += 1
            state["most"] = max(state["most"], state["now"])
        time.sleep(0.3)                    # a load takes a while
        with guard:
            state["now"] -= 1
        return FakeResident(spec["job_id"])

    host._fetch = lambda job_id, log: _TMP
    host._make_room = lambda need, log: None
    host._plan_precision = lambda spec, log: False
    host._load = slow_load
    spec = {"job_id": "job_mistral", "params_b": 7.248}

    def as_generate():            # holds the card lock, as generate() does
        with host.lock:
            host.ensure_loaded(spec, lambda _s: None)

    def as_preload():             # calls straight in, as the agent's preload does
        host.ensure_loaded(spec, lambda _s: None)

    threads = [threading.Thread(target=f) for f in (as_preload, as_generate,
                                                    as_preload)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    check("no caller is left waiting for ever",
          not any(t.is_alive() for t in threads))
    check("at most one load is ever on the card at a time", state["most"], 1)
    check("and the model is loaded once, not once per caller",
          state["loads"], 1)
    check("the one holding the lock already can still take it again",
          host.loaded_id, "job_mistral")

    print()
    shutil.rmtree(_TMP, ignore_errors=True)
    if FAILED:
        print("%d check(s) failed:" % len(FAILED))
        for name in FAILED:
            print("  - %s" % name)
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
