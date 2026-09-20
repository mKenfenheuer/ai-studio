#!/usr/bin/env python
"""Check that `response_format` is enforced rather than requested.

Run it with `python scripts/check-grammar.py` from the repository root.

The claim being checked is a strong one: a reply asked for as JSON is not
checked afterwards and never repaired, because it could not have been written
any other way. The only honest way to test that is to take the model out of it.

So the "model" here picks UNIFORMLY AT RANDOM from whatever the grammar allows.
It has no idea what JSON is, it is trying to say nothing in particular, and it
would happily write "Sure! Here's the JSON:" if the mask let it. Every reply it
produces still parses, and still matches the schema. That is the property --
not that a good model usually complies, but that a maximally unhelpful one
cannot do otherwise.

The tokenizer is a toy on purpose. A real one would make this test about
Qwen's vocabulary; this one is small enough that a failure points at the
wiring in `runner/grammar.py`, which is what this file is here to check.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from runner import grammar as grammars  # noqa: E402

FAILED: list[str] = []


def check(name: str, got: object, want: object = True) -> None:
    ok = got == want
    print("  %s  %s%s" % ("ok  " if ok else "FAIL", name,
                          "" if ok else "   -> %r, wanted %r" % (got, want)))
    if not ok:
        FAILED.append(name)


# A vocabulary with the pieces JSON is made of, plus multi-character tokens so
# the check exercises the case that actually breaks naive implementations: one
# token carrying several characters, any of which could be the one the grammar
# refuses.
PIECES = list("{}[]:,\"0123456789.-+ \n\tabcdefghijklmnopqrstuvwxyzAZ_") + [
    "true", "false", "null", "name", "age", "ok", "Sure", "Here's", "the",
    '":', '",', '{"', '"}', "],", "0.", "-1", "12", "99", "e5", "\\n", "\\\"",
]


class ToyTokenizer:
    """Enough of a tokenizer for `grammar._tokenizer_data` to describe.

    Deliberately NOT a transformers tokenizer: the point of building the
    enforcer's data by hand is that it depends on four small methods rather
    than on a library that moves every few weeks, and a fake that provides
    exactly those four is what proves it.
    """

    def __init__(self) -> None:
        self.vocab = list(dict.fromkeys(PIECES))
        self.eos_token_id = len(self.vocab)
        self.vocab.append("<eos>")
        self.all_special_ids = [self.eos_token_id]

    def __len__(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> list[int]:
        return [self.vocab.index(text)]

    def decode(self, ids, **_kw) -> str:
        return "".join("" if i == self.eos_token_id else self.vocab[i]
                       for i in ids)


def write(grammar, tok, rng: random.Random, limit: int = 400):
    """One reply, from a model that chooses at random among the legal moves."""
    produced: list[int] = []
    for _ in range(limit):
        allowed = grammar.allowed(produced)
        if not allowed:
            return None, produced, "the grammar allowed nothing"
        token = rng.choice(allowed)
        if token == tok.eos_token_id:
            return tok.decode(produced), produced, None
        produced.append(token)
    # Not a failure of the grammar: a random walk through JSON can wander for
    # a long time. Reported separately so it is never mistaken for one.
    return None, produced, "ran to the token limit"


def build(fmt: dict, tok) -> object:
    return grammars.build(fmt, tok, {})


def grammar_for(tok):
    """A throwaway grammar, only ever used for its `validate`."""
    return grammars.build({"type": "json_object"}, tok, {})


def main() -> int:
    if not grammars.AVAILABLE:
        print("\nThe grammar engine is not installed here, so there is "
              "nothing to check.\nInstall it with: pip install "
              "lm-format-enforcer")
        return 0

    tok = ToyTokenizer()
    rng = random.Random(20260920)

    print("\nWhat asks for a grammar, and what does not")
    check("plain text is unconstrained", build({"type": "text"}, tok), None)
    check("so is a request with no response_format at all", build(None, tok),
          None)
    check("json_object asks for one",
          build({"type": "json_object"}, tok).kind, "json_object")
    check("json_schema asks for one",
          build({"type": "json_schema",
                 "json_schema": {"schema": {"type": "object"}}}, tok).kind,
          "json_schema")

    print("\nA schema nothing can enforce is refused, not accepted quietly")
    try:
        build({"type": "json_schema", "json_schema": {"name": "x"}}, tok)
        check("a json_schema with no schema is refused", False)
    except grammars.Unsupported:
        check("a json_schema with no schema is refused", True)
    try:
        build({"type": "yaml"}, tok)
        check("an unknown format is refused", False)
    except grammars.Unsupported:
        check("an unknown format is refused", True)

    print("\njson_object: a model choosing at random still writes valid JSON")
    parsed, gave_up, refused = 0, 0, 0
    for _ in range(40):
        text, _ids, why = write(build({"type": "json_object"}, tok), tok, rng)
        if text is None:
            gave_up += 1
            check("a reply ended without the grammar cornering it",
                  why, "ran to the token limit")
            continue
        try:
            json.loads(text)
            parsed += 1
        except ValueError:
            # Not a failure of this check: the enforcer is known to allow raw
            # tabs and newlines inside a string, which json.loads rejects. The
            # guarantee is that such a reply is REFUSED, not that it never
            # happens -- so the only thing that matters is that validate()
            # catches it. A reply it let through would be the real fault.
            refused += 1
            try:
                grammar_for(tok).validate(text)
                check("an unparseable reply is refused (%r)" % text[:50], False)
            except grammars.Broken:
                pass
    check("replies were produced", parsed > 0)
    check("and every reply either parsed or was refused",
          parsed + gave_up + refused, 40)
    check("the guard did fire at least once, so it is really being tested",
          refused > 0)

    print("\njson_schema: and it matches the schema it was given")
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "ok": {"type": "boolean"},
        },
        "required": ["name", "age"],
        "additionalProperties": False,
    }
    fmt = {"type": "json_schema", "json_schema": {"schema": schema}}
    good, gave_up, refused = 0, 0, 0
    for _ in range(40):
        text, _ids, why = write(build(fmt, tok), tok, rng)
        if text is None:
            gave_up += 1
            continue
        try:
            grammars.build(fmt, tok, {}).validate(text)
        except grammars.Broken:
            refused += 1
            continue
        value = json.loads(text)
        if not isinstance(value, dict):
            check("every reply is an object (%r)" % text[:60], False)
            continue
        if not {"name", "age"} <= set(value):
            check("every reply has the required keys (%r)" % text[:60], False)
            continue
        if set(value) - {"name", "age", "ok"}:
            check("no reply invents a key (%r)" % text[:60], False)
            continue
        if not isinstance(value["name"], str):
            check("name is a string (%r)" % text[:60], False)
            continue
        # JSON Schema counts any number with a zero fractional part as an
        # integer, so `6e9` qualifies even though Python decodes it to a
        # float. `8e-15` does not, and is refused above by the validator --
        # which is the case worth having a guard for.
        age = value["age"]
        if isinstance(age, bool) or not (
                isinstance(age, int)
                or (isinstance(age, float) and age.is_integer())):
            check("age is an integer (%r)" % text[:60], False)
            continue
        if "ok" in value and not isinstance(value["ok"], bool):
            check("ok is a boolean (%r)" % text[:60], False)
            continue
        good += 1
    check("replies were produced", good > 0)
    check("every reply either matched the schema or was refused",
          good + gave_up + refused, 40)

    print("\nAn enum leaves the model no room at all")
    enum = {"type": "object",
            "properties": {"ok": {"type": "string", "enum": ["yes", "no"]}},
            "required": ["ok"], "additionalProperties": False}
    # Only the two words are reachable, so a random model produces one of them
    # every time -- which is the clearest demonstration available that the
    # mask, not the model, is what decides.
    seen = set()
    for _ in range(25):
        text, _ids, _why = write(
            build({"type": "json_schema", "json_schema": {"schema": enum}},
                  tok), tok, rng)
        if text is not None:
            seen.add(json.loads(text)["ok"])
    check("every value produced was one of the two allowed",
          seen <= {"yes", "no"} and bool(seen))

    print("\nThe schema is put in front of the model as well as behind it")
    g = build(fmt, tok)
    check("the instruction names the schema", "properties" in g.instruction())
    check("and says to write nothing else",
          "nothing else" in g.instruction())

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
