#!/usr/bin/env python
"""Check what the OpenAI-compatible server refuses, and what it queues instead.

Run it with `python scripts/check-serving.py` from the repository root.

Two faults live here, and they are opposites of each other.

The first is accepting an option and ignoring it. The server refuses `n`,
`logprobs` and `tool_choice: "required"` for exactly this reason, and
`response_format` was the one that slipped through -- the sharpest case of the
lot, because a caller asking for a JSON schema has stopped checking the reply.
Prose returned under a schema goes straight into whatever the schema was going
to fill in.

The second is refusing something that should simply have waited. A runner holds
one model on one card and answers one message at a time, so a second request
arriving mid-reply was told the machine was busy -- as a 502, which tells a
client library the upstream is broken. An evaluation of sixty prompts got one
answer and fifty-nine errors. They are queued now, and the queue has a bound,
because an unbounded queue is just a slower way to time out.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from controller import config  # noqa: E402
from controller.api.serving import unsupported_options  # noqa: E402
from controller.scheduler import Fleet  # noqa: E402

FAILED: list[str] = []


def check(name: str, got: object, want: object = True) -> None:
    ok = got == want
    print("  %s  %s%s" % ("ok  " if ok else "FAIL", name,
                          "" if ok else "   -> %r, wanted %r" % (got, want)))
    if not ok:
        FAILED.append(name)


def refusal(payload: dict) -> tuple[int | None, str]:
    """The status and message this request would be refused with, if any."""
    out = unsupported_options(payload)
    if out is None:
        return None, ""
    return out.status_code, json.loads(bytes(out.body))["error"]["message"]


class FakeSocket:
    """A runner's connection, as far as the fleet can tell."""

    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.dead = False

    async def send_text(self, text: str) -> None:
        if self.dead:
            raise ConnectionError("gone")
        self.sent.append(json.loads(text))


def drain(queue: asyncio.Queue) -> list[str]:
    """The kinds of frame a caller was handed, in order."""
    out = []
    while not queue.empty():
        out.append(queue.get_nowait()["type"])
    return out


def generate(rid: str) -> dict:
    return {"type": "generate", "request_id": rid, "spec": {}, "messages": []}


def options() -> None:
    print("\nOptions that are refused rather than ignored")
    check("a plain request is accepted", refusal({})[0], None)
    check("`n` above one is refused", refusal({"n": 2})[0], 400)
    # `True == 1` in Python, and the check that used to live here allowed 1
    # because `n: 1` is legitimate. So this exact request was accepted.
    check("`logprobs` is refused", refusal({"logprobs": True})[0], 400)
    check("and so is `top_logprobs`",
          refusal({"top_logprobs": 5})[0], 400)
    check("`n: 1` is still fine", refusal({"n": 1})[0], None)
    check("requiring a tool call is refused",
          refusal({"tool_choice": "required"})[0], 400)
    check("naming a tool is refused",
          refusal({"tool_choice": {"function": {"name": "f"}}})[0], 400)
    check("`tool_choice: none` is not, because it can be honoured",
          refusal({"tool_choice": "none"})[0], None)

    print("\n...including response_format, which used to be accepted silently")
    check("a JSON schema is refused",
          refusal({"response_format": {"type": "json_schema",
                                       "json_schema": {"name": "x"}}})[0], 400)
    check("and says why, in terms of what it cannot promise",
          "constrain" in refusal({"response_format":
                                  {"type": "json_schema"}})[1])
    check("a bare JSON object is refused too",
          refusal({"response_format": {"type": "json_object"}})[0], 400)
    check("`text` is accepted, because it promises nothing",
          refusal({"response_format": {"type": "text"}})[0], None)
    check("an unknown type is refused",
          refusal({"response_format": {"type": "yaml"}})[0], 400)
    check("and so is one that is not an object at all",
          refusal({"response_format": "json"})[0], 400)


async def queueing() -> None:
    print("\nA second request waits its turn instead of failing")
    fleet = Fleet()
    ws = FakeSocket()
    fleet.attach("r1", ws)
    # Both dispatch paths register a caller before they submit -- `waiters`
    # for an API request, `generation_owner` for the playground -- because a
    # queued request with nobody waiting on it is one the queue skips.
    fleet.generation_owner.update({"g1": "u", "g2": "u"})

    check("the first goes straight out",
          await fleet.submit_generation("r1", "g1", generate("g1")), "sent")
    check("the second is queued, not refused",
          await fleet.submit_generation("r1", "g2", generate("g2")), "queued")
    check("and the machine has only been given the first", len(ws.sent), 1)
    check("the queue says how many are waiting", fleet.queue_depth("r1"), 1)

    await fleet.finish_generation("r1", "g1")
    check("finishing the first sends the second", len(ws.sent), 2)
    check("which is the one that was waiting", ws.sent[1]["request_id"], "g2")
    check("and the queue is empty again", fleet.queue_depth("r1"), 0)

    print("\nOrder is arrival order")
    fleet2 = Fleet()
    ws2 = FakeSocket()
    fleet2.attach("r1", ws2)
    fleet2.generation_owner.update({"q%d" % i: "u" for i in range(5)})
    for i in range(5):
        await fleet2.submit_generation("r1", "q%d" % i, generate("q%d" % i))
    for i in range(4):
        await fleet2.finish_generation("r1", "q%d" % i)
    check("five requests, sent one at a time, first come first served",
          [m["request_id"] for m in ws2.sent],
          ["q0", "q1", "q2", "q3", "q4"])

    print("\nThe queue is bounded, and says so with a status a client knows")
    fleet3 = Fleet()
    ws3 = FakeSocket()
    fleet3.attach("r1", ws3)
    await fleet3.submit_generation("r1", "a", generate("a"))
    for i in range(config.SERVING_QUEUE_MAX):
        await fleet3.submit_generation("r1", "b%d" % i, generate("b%d" % i))
    check("the line fills to its bound",
          fleet3.queue_depth("r1"), config.SERVING_QUEUE_MAX)
    check("and the next one is turned away",
          await fleet3.submit_generation("r1", "over", generate("over")),
          "full")
    check("a machine that is not there is not queued for",
          await fleet3.submit_generation("nobody", "x", generate("x")), "gone")

    print("\nA request that leaves the queue does not come back")
    fleet4 = Fleet()
    ws4 = FakeSocket()
    fleet4.attach("r1", ws4)
    fleet4.generation_owner.update({"c1": "u", "c2": "u", "c3": "u"})
    await fleet4.submit_generation("r1", "c1", generate("c1"))
    await fleet4.submit_generation("r1", "c2", generate("c2"))
    await fleet4.submit_generation("r1", "c3", generate("c3"))
    check("cancelling a queued request takes it out of the line",
          fleet4.drop_queued_generation("c2"), True)
    check("cancelling the one being answered does not",
          fleet4.drop_queued_generation("c1"), False)
    await fleet4.finish_generation("r1", "c1")
    check("so the cancelled one is never sent",
          [m["request_id"] for m in ws4.sent], ["c1", "c3"])

    print("\nNothing is left waiting on a machine that has gone")
    fleet5 = Fleet()
    ws5 = FakeSocket()
    fleet5.attach("r1", ws5)
    told: dict[str, asyncio.Queue] = {}
    for rid in ("d1", "d2", "d3"):
        told[rid] = fleet5.waiters[rid] = asyncio.Queue()
        await fleet5.submit_generation("r1", rid, generate(rid))
    await fleet5.detach("r1")
    frames = {rid: drain(q) for rid, q in told.items()}
    check("the one being answered is told", frames["d1"], ["generate_error"])
    check("and so are the ones that were queued behind it, after the frame "
          "that told them where they stood",
          [frames[r] for r in ("d2", "d3")],
          [["generate_status", "generate_error"]] * 2)
    check("and the machine holds nothing afterwards",
          (fleet5.serving_now.get("r1"), fleet5.queue_depth("r1")), (None, 0))

    print("\nA reply that never arrives does not hold the machine for ever")
    fleet6 = Fleet()
    ws6 = FakeSocket()
    fleet6.attach("r1", ws6)
    fleet6.generation_owner.update({"e1": "u", "e2": "u"})
    await fleet6.submit_generation("r1", "e1", generate("e1"))
    await fleet6.submit_generation("r1", "e2", generate("e2"))
    # Pretend the machine started that reply long enough ago to be wedged.
    fleet6.serving_since["r1"] = 0.0
    await fleet6.reconcile_serving()
    check("the stuck slot is freed and the next request takes it",
          fleet6.serving_now.get("r1"), "e2")
    check("so the one behind it is sent after all",
          [m["request_id"] for m in ws6.sent], ["e1", "e2"])


def main() -> int:
    options()
    asyncio.run(queueing())
    print()
    if FAILED:
        print("%d check(s) failed:" % len(FAILED))
        for name in FAILED:
            print("  - %s" % name)
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
