#!/usr/bin/env python
"""Check that a conversation survives the round trip into text and back.

Run it with `python scripts/check-formats.py` from the repository root. No test
framework, and deliberately so: the controller installs four pure-Python
packages and this has to be runnable on the machine somebody is debugging on,
which is frequently not a machine with a dev environment.

What it is guarding is narrow and worth guarding. The chat templates in
`common/chat_formats.py` are one-line Jinja strings full of escaped quotes, and
a mistake in one of them does not raise -- it silently emits text the model was
never trained on. Two such mistakes were live before this existed: four of the
five formats dropped every tool call, and the tool declaration for everything
but Harmony listed names without the parameters a model has to reproduce.

The properties checked here are the ones that cannot be eyeballed:

* every format renders tool calls, tool results and the tool schema
* the boundary of each turn can be measured, which is what makes `weight`
  masking possible at all -- and which silently degrades to "train on
  everything" if a template stops being prefix-stable
* a masked turn really is outside every trained range
* the linking between a tool call and its result survives import, including
  when two calls are answered in the wrong order
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import chat_formats, conversation as C, formatting as F  # noqa: E402

FAILED: list[str] = []


def check(name: str, ok: bool, detail: object = "") -> None:
    print("  %s  %s%s" % ("ok  " if ok else "FAIL", name,
                          "" if ok else "   -> %s" % (detail,)))
    if not ok:
        FAILED.append(name)


# A conversation with everything in it: a system prompt, reasoning, two tool
# calls, two results, and a final answer.
EXAMPLE = {
    "reasoning_effort": "medium",
    "messages": [
        {"role": "system",
         "content": "You are a helpful assistant which specializes in function calling."},
        {"role": "user", "content": "Where is my order #12345?"},
        {"role": "assistant",
         "reasoning": "The user provided an order ID. I should use get_order.",
         "tool_calls": [{"id": "call_001", "type": "function",
                         "function": {"name": "get_order",
                                      "arguments": '{"order_id":"12345"}'}}]},
        {"role": "tool", "tool_call_id": "call_001",
         "content": '{"order_id":"12345","status":"shipped","carrier":"DHL"}'},
        {"role": "assistant", "content": "Your order #12345 has shipped via DHL."},
        {"role": "user", "content": "When will it arrive?"},
        {"role": "assistant",
         "reasoning": "I should use get_tracking for the delivery date.",
         "tool_calls": [{"id": "call_002", "type": "function",
                         "function": {"name": "get_tracking",
                                      "arguments": '{"tracking_number":"JD01460012"}'}}]},
        {"role": "tool", "tool_call_id": "call_002",
         "content": '{"status":"in_transit","estimated_delivery":"2026-08-27"}'},
        {"role": "assistant", "content": "In transit, due on 27 August."},
    ],
    "tools": [
        {"type": "function", "function": {
            "name": "get_order", "description": "Get the status of an order.",
            "parameters": {"type": "object",
                           "properties": {"order_id": {"type": "string"}},
                           "required": ["order_id"]}}},
        {"type": "function", "function": {
            "name": "get_tracking", "description": "Get tracking information.",
            "parameters": {"type": "object",
                           "properties": {"tracking_number": {"type": "string"}},
                           "required": ["tracking_number"]}}},
    ],
}


def test_import() -> None:
    print("\nImport and repair")
    conv, notes = C.repair(C.from_row(EXAMPLE))
    msgs = conv[C.MESSAGES_KEY]
    check("nine messages survive", len(msgs) == 9, len(msgs))
    check("reasoning kept off the content",
          msgs[2]["reasoning"].startswith("The user provided")
          and not msgs[2].get("content"))
    check("arguments stay a JSON string",
          msgs[2]["tool_calls"][0]["function"]["arguments"] == '{"order_id":"12345"}')
    check("tool results named from the call they answer",
          [m.get("name") for m in msgs if m["role"] == "tool"]
          == ["get_order", "get_tracking"])
    check("tool_call_id preserved from the data",
          [m.get("tool_call_id") for m in msgs if m["role"] == "tool"]
          == ["call_001", "call_002"])
    check("reasoning_effort kept as provenance, not as a message",
          conv[C.META_KEY].get("reasoning_effort") == "medium")
    # The one thing this example leaves implicit: the results carry an id but
    # no tool name, which is exactly what real tool-calling datasets do.
    check("only the tool names had to be inferred",
          notes == ["named 2 tool results from the call before it"], notes)
    check("validates clean", C.validate(conv) == [], C.validate(conv))

    # The row survives a trip through JSON unchanged, which is what a dataset
    # file actually does to it.
    again = C.from_row(json.loads(json.dumps(C.to_row(conv))))
    check("round trips through JSONL",
          C.to_row(again) == C.to_row(conv))


def test_repair() -> None:
    print("\nPictures and clips in a turn")
    parts = [{"type": "text", "text": "What is this?"},
             {"type": "image_url", "image_url": {"url": "https://x/cat.png"}},
             {"type": "image", "image": "asset:ast_0123456789ab"}]
    conv = C.from_messages([{"role": "user", "content": parts},
                            {"role": "assistant", "content": "A cat."}])
    user = conv["messages"][0]
    check("the words are the content", user["content"], "What is this?")
    check("no [image] in the text", "[image]" not in user["content"])
    check("both pictures kept, in order",
          [m.get("ref") or m.get("url") for m in user.get("media", [])],
          ["https://x/cat.png", "asset:ast_0123456789ab"])
    check("media round-trips through a row",
          C.from_row(json.loads(json.dumps(C.to_row(conv))))["messages"][0].get("media"),
          user["media"])
    rendered = F.render_prompt(conv["messages"], {"mode": "chat"})
    check("a placeholder marks each picture", rendered.count("<image>"), 2)
    check("placed before the words", rendered.index("<image>") < rendered.index("What is this?"))
    named = F.render_prompt(conv["messages"], {"mode": "chat", "image_token": "<|vision_start|>"})
    check("a format names its own token", named.count("<|vision_start|>"), 2)
    plain = C.from_messages([{"role": "user", "content": "just words"}])
    check("a turn with no media is untouched",
          "media" not in plain["messages"][0] and
          "<image>" not in F.render_prompt(plain["messages"], {"mode": "chat"}))

    print("\nRepairing what data leaves implicit")
    bare = {"messages": [
        {"role": "user", "content": "x"},
        {"role": "assistant",
         "function_call": {"name": "f1", "arguments": {"a": 1}}},
        {"role": "function", "content": "{}"},
        {"role": "assistant", "content": "done"}]}
    conv, notes = C.repair(C.from_row(bare))
    call = conv[C.MESSAGES_KEY][1]["tool_calls"][0]
    check("a bare function_call becomes a tool call", call["function"]["name"] == "f1")
    check("its object arguments become a string",
          call["function"]["arguments"] == '{"a": 1}')
    check("the `function` role becomes `tool`",
          conv[C.MESSAGES_KEY][2]["role"] == "tool")
    check("a missing id is invented and linked",
          conv[C.MESSAGES_KEY][2]["tool_call_id"] == call["id"])
    check("the repair is reported rather than silent", len(notes) == 2, notes)

    # Two calls in flight, answered in the other order. Pairing by position
    # rather than by id gets this exactly backwards.
    parallel = {"messages": [
        {"role": "user", "content": "x"},
        {"role": "assistant", "tool_calls": [
            {"id": "a", "function": {"name": "f1", "arguments": "{}"}},
            {"id": "b", "function": {"name": "f2", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "b", "content": "{}"},
        {"role": "tool", "tool_call_id": "a", "content": "{}"},
        {"role": "assistant", "content": "done"}],
        "tools": [{"type": "function", "function": {"name": n, "parameters": {}}}
                  for n in ("f1", "f2")]}
    conv, _ = C.repair(C.from_row(parallel))
    pairs = [(m["tool_call_id"], m["name"])
             for m in conv[C.MESSAGES_KEY] if m["role"] == "tool"]
    check("parallel results pair by id, not by position",
          pairs == [("b", "f2"), ("a", "f1")], pairs)

    orphan = {"messages": [
        {"role": "user", "content": "x"},
        {"role": "assistant", "tool_calls": [
            {"id": "c1", "function": {"name": "ghost", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "nope", "content": "{}"}]}
    conv, _ = C.repair(C.from_row(orphan))
    codes = {p["code"] for p in C.validate(conv)}
    check("a result answering no call is an error", "orphan_result" in codes, codes)
    check("and is not given a borrowed name",
          not conv[C.MESSAGES_KEY][2].get("name"))
    check("a tool nobody declared is reported", "undeclared_tool" in codes, codes)


def test_other_shapes() -> None:
    print("\nThe shapes real datasets arrive in")
    sharegpt = C.from_row({"conversations": [
        {"from": "human", "value": "hi"},
        {"from": "gpt", "value": "<think>hmm</think>hello"}]})
    check("ShareGPT roles map across",
          [m["role"] for m in sharegpt[C.MESSAGES_KEY]] == ["user", "assistant"])
    check("an inline <think> block is lifted out of the answer",
          sharegpt[C.MESSAGES_KEY][1]["reasoning"] == "hmm"
          and sharegpt[C.MESSAGES_KEY][1]["content"] == "hello")

    alpaca = C.from_row({"instruction": "Add 2+2", "output": "4"},
                        {"mode": "chat"})
    check("an instruction pair becomes two turns",
          [m["role"] for m in alpaca[C.MESSAGES_KEY]] == ["user", "assistant"])

    prose = C.from_row({"text": "Once upon a time."}, {"mode": "chat"})
    check("a lone block of prose becomes one assistant turn",
          [m["role"] for m in prose[C.MESSAGES_KEY]] == ["assistant"])

    parts = C.from_row({"messages": [
        {"role": "user", "content": [{"type": "text", "text": "a"},
                                     {"type": "text", "text": "b"}]}]})
    check("typed content parts are joined into text",
          parts[C.MESSAGES_KEY][0]["content"] == "ab")


def test_formats() -> None:
    conv, _ = C.repair(C.from_row(EXAMPLE))
    masked = json.loads(json.dumps(conv))
    masked[C.MESSAGES_KEY][2]["weight"] = 0        # the first tool call

    for spec in chat_formats.CHAT_FORMATS:
        fid = spec["id"]
        print("\n%s" % spec["label"])
        fmt = {"mode": "chat", "chat_format": fid}
        text = C.render(conv, fmt)

        check("%-8s renders the tool call" % fid, "get_order" in text)
        check("%-8s renders its arguments as JSON, not as a dict repr" % fid,
              '"12345"' in text and "'order_id'" not in text)
        check("%-8s declares the parameters, not just the name" % fid,
              "order_id" in text.split("Where is my order")[0])
        check("%-8s renders the tool result" % fid, "shipped" in text)
        check("%-8s renders the reasoning" % fid, "I should use get_order" in text)
        check("%-8s has no accidental blank lines" % fid, "\n\n\n" not in text)

        prompt = C.render(conv, fmt, add_generation_prompt=True)
        check("%-8s a generation prompt extends the conversation" % fid,
              prompt.startswith(text) and len(prompt) > len(text))

        parts = C.segments(conv, fmt)
        check("%-8s turn boundaries are exact" % fid,
              all(p.get("exact") for p in parts) and len(parts) == 9,
              len(parts))
        check("%-8s the pieces reassemble into the whole" % fid,
              "".join(p["text"] for p in parts) == text)

        body, spans, exact = C.trainable_spans(masked, fmt)
        check("%-8s spans could be established" % fid, exact)
        # Four assistant turns, one of them weight 0.
        check("%-8s only assistant turns are trained on" % fid,
              len(spans) == 3, len(spans))
        for needle, why in (("Where is my order", "the user's question"),
                            ("shipped", "the tool's answer"),
                            ("I should use get_order", "the weight-0 turn")):
            at = body.index(needle)
            check("%-8s %s is outside the loss" % (fid, why),
                  not any(s <= at < e for s, e in spans))
        for needle in ("In transit, due on 27 August",):
            at = body.index(needle)
            check("%-8s the final answer is inside the loss" % fid,
                  any(s <= at < e for s, e in spans))

        everything, spans, _ = C.trainable_spans(conv, fmt, train_on="all")
        check("%-8s train_on=all masks nothing" % fid, spans == [])


def test_reading_a_reply_back() -> None:
    """Rendering has an inverse, and the playground depends on it.

    A model trained to call tools emits a call; without this the playground
    prints it as prose, a conversation cannot continue past it, and there is
    no way to tell a working tool-caller from one that learned to type angle
    brackets.
    """
    call = {"messages": [
        {"role": "user", "content": "Where is order 12345?"},
        {"role": "assistant", "reasoning": "I should look it up.",
         "tool_calls": [{"id": "c1", "type": "function", "function": {
             "name": "get_order", "arguments": '{"order_id":"12345"}'}}]}],
        "tools": [{"type": "function", "function": {
            "name": "get_order", "description": "",
            "parameters": {"type": "object",
                           "properties": {"order_id": {"type": "string"}}}}}]}
    answer = {"messages": [{"role": "user", "content": "hi"},
                           {"role": "assistant", "content": "Hello there."}]}

    for spec in chat_formats.CHAT_FORMATS:
        fid = spec["id"]
        print("\n%s -- reading a reply back" % spec["label"])
        fmt = {"mode": "chat", "chat_format": fid}

        for conv_row, thinking in ((call, True), (answer, False)):
            conv, _ = C.repair(C.from_row(conv_row))
            full = C.render(conv, fmt)
            prompt = C.render({**conv, C.MESSAGES_KEY: conv[C.MESSAGES_KEY][:1]},
                              fmt, add_generation_prompt=True,
                              reasoning=thinking, more_turns=True)
            # Exactly the substring the model itself would have produced.
            generated = full[len(prompt):] if full.startswith(prompt) else full
            got = C.parse_reply(generated, fmt, reasoning_on=thinking)

            if conv_row is call:
                check("%-8s the call is recognised" % fid,
                      len(got["tool_calls"]) == 1, got["tool_calls"])
                if got["tool_calls"]:
                    one = got["tool_calls"][0]
                    check("%-8s with the right name" % fid,
                          one["function"]["name"] == "get_order")
                    check("%-8s with arguments that parse" % fid, one["valid"])
                    check("%-8s and the right arguments" % fid,
                          json.loads(one["function"]["arguments"])
                          == {"order_id": "12345"})
                check("%-8s the reasoning is separated from the answer" % fid,
                      got["reasoning"].strip() == "I should look it up."
                      and "should look it up" not in got["content"],
                      (got["reasoning"], got["content"]))
            else:
                check("%-8s a plain answer reads back exactly" % fid,
                      got["content"] == "Hello there." and not got["tool_calls"],
                      got)

    print("\nMalformed replies")
    broken = C.parse_reply('{"name": "get_order", "parameters": {"a": }}',
                           {"mode": "chat", "chat_format": "llama3"})
    check("a call with unparseable arguments is still reported as a call",
          len(broken["tool_calls"]) == 1 and not broken["tool_calls"][0]["valid"],
          broken)
    check("an empty reply does not raise",
          C.parse_reply("", {"mode": "chat"})["content"] == "")

    print("\nA call with its announcement stripped off")
    # What a model trained on Mistral's own template actually hands back: the
    # marker in front of the call, [TOOL_CALLS], is a special token, and
    # decoding for display drops special tokens. What is left is a bare JSON
    # array that no pattern is looking for -- and a fine-tune that had learned
    # to call tools perfectly was scored as never having called one.
    bare = C.parse_reply(
        '[{"name": "execute_services", "arguments": '
        '"{\\"list\\":[{\\"domain\\":\\"cover\\"}]}", "id": "abc"}]',
        {"mode": "chat"})
    check("a bare JSON call is recognised", len(bare["tool_calls"]) == 1, bare)
    if bare["tool_calls"]:
        one = bare["tool_calls"][0]
        check("with its name", one["function"]["name"] == "execute_services",
              one["function"]["name"])
        check("its arguments parse", one["valid"], one["function"]["arguments"])
        check("and nothing is left over as text", bare["content"] == "",
              repr(bare["content"]))
    # And the other half of the promise: JSON that is an answer stays an answer.
    plain = C.parse_reply('{"answer": 42, "why": "because"}', {"mode": "chat"})
    check("a reply that is merely JSON is not a call",
          not plain["tool_calls"] and plain["content"].startswith("{"), plain)
    names = C.parse_reply('[{"name": "Ann"}, {"name": "Bob"}]', {"mode": "chat"})
    check("nor is a list of things that have names",
          not names["tool_calls"], names)


def test_shapes_in_the_wild() -> None:
    """The two layouts the widely used function-calling datasets actually use.

    Neither stores a tool call the way the format specifies, and both store
    their tool schema as a JSON *string* -- which is what Arrow does to a
    column it has no union type for. Importing either of them used to produce
    conversations with no tools declared and no calls in them, and the only
    symptom was a fine-tune that never called anything.
    """
    print("\nShapes real datasets arrive in")

    # Hermes: ShareGPT turns, calls written into the assistant's text as
    # <tool_call> blocks, schema as a JSON string in a sibling column.
    hermes = {
        "conversations": [
            {"from": "system", "value": "You are a function calling AI model."},
            {"from": "human", "value": "Show me the front door camera."},
            {"from": "gpt", "value": "\n".join([
                "<tool_call>",
                '{"name": "get_feed", "arguments": {"id": "front"}}',
                "</tool_call>",
                "<tool_call>",
                '{"name": "record", "arguments": {"id": "front", "secs": 30}}',
                "</tool_call>"])},
        ],
        "tools": json.dumps([
            {"type": "function", "function": {
                "name": "get_feed", "description": "Live feed.",
                "parameters": {"type": "object",
                               "properties": {"id": {"type": "string"}},
                               "required": ["id"]}}},
            {"type": "function", "function": {
                "name": "record", "description": "Record.",
                "parameters": {"type": "object",
                               "properties": {"id": {"type": "string"},
                                              "secs": {"type": "integer"}}}}}]),
    }
    conv, _ = C.repair(C.from_row(hermes))
    msgs = conv[C.MESSAGES_KEY]
    check("a tool schema stored as a JSON string is still read",
          [t["function"]["name"] for t in conv[C.TOOLS_KEY]] == ["get_feed", "record"],
          conv[C.TOOLS_KEY])
    check("<tool_call> blocks in the text become real calls",
          [c["function"]["name"] for c in msgs[2].get("tool_calls", [])]
          == ["get_feed", "record"], msgs[2])
    check("and are lifted out of the text rather than left in it",
          not msgs[2].get("content"), msgs[2].get("content"))
    check("the hermes row validates clean", C.validate(conv) == [], C.validate(conv))

    # A call given a role of its own, with the body a JSON array of calls, a
    # reasoning field beside it, and a flat RapidAPI-style parameter map.
    reasoning_row = {
        "conversation": [
            {"from": "human", "reasoning": None, "value": "Is apple.com archived?"},
            {"from": "function_call",
             "reasoning": "I should call availability with a timestamp.",
             "value": '[{"name": "availability", "arguments": {"url": "https://apple.com"}}]'},
        ],
        "tools": json.dumps([{
            "name": "availability",
            "description": "Checks the Wayback Machine.",
            "parameters": {
                "url": {"description": "The URL.", "type": "str"},
                "timestamp": {"description": "When.", "type": "str",
                              "default": "20090101"}}}]),
    }
    conv, _ = C.repair(C.from_row(reasoning_row))
    msgs = conv[C.MESSAGES_KEY]
    check("a `function_call` role becomes an assistant turn",
          msgs[1]["role"] == "assistant", msgs[1]["role"])
    check("a JSON array body becomes the call",
          [c["function"]["name"] for c in msgs[1].get("tool_calls", [])]
          == ["availability"], msgs[1])
    check("the reasoning beside it is kept",
          msgs[1]["reasoning"].startswith("I should call"), msgs[1].get("reasoning"))
    params = conv[C.TOOLS_KEY][0]["function"]["parameters"]
    check("a flat parameter map becomes JSON Schema",
          params.get("type") == "object" and "url" in params.get("properties", {}),
          params)
    check("python type names are translated",
          params["properties"]["url"]["type"] == "string", params["properties"]["url"])
    check("a parameter with a default is optional, one without is required",
          params.get("required") == ["url"], params.get("required"))
    check("the reasoning row validates clean", C.validate(conv) == [], C.validate(conv))

    # The guard: prose that merely quotes JSON is not a tool call.
    quoted = C.from_row({"messages": [
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": '{"name": "Ada", "age": 36}'}]})
    check("a JSON object that is not a call stays as text",
          not quoted[C.MESSAGES_KEY][1].get("tool_calls"),
          quoted[C.MESSAGES_KEY][1])


def main() -> int:
    test_import()
    test_repair()
    test_other_shapes()
    test_shapes_in_the_wild()
    test_formats()
    test_reading_a_reply_back()
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
