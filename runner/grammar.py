"""Constrained decoding: making `response_format` a promise rather than a hint.

A model asked for JSON usually writes JSON. "Usually" is the problem. A caller
that sets `response_format` has stopped checking the reply -- that is the whole
reason the field exists -- so the one time in fifty that the model opens with
"Sure! Here's the JSON:" the prose goes wherever the parsed object was going to
go. Asking nicely in the prompt cannot fix this, because nothing in the prompt
is binding.

What is binding is the sampler. At every step the model produces a score for
every token in the vocabulary, and we choose from them. So the tokens that
would break the grammar are given a score of negative infinity before the
choice is made, and the model picks from what is left. The reply is not checked
against the schema afterwards -- it could not have been written any other way.

## Why a library here, and a hand-written loop next door

The training loop in `runner/jobs/lora_llm.py` is hand-written on purpose, and
the reasoning there does not carry over. That loop is hand-written to escape an
API that churns; this is a *grammar*, and a grammar is either correct or it
quietly emits `{"a": 01.}` at three in the morning. JSON's escaping rules, its
number syntax and its unicode handling are each a well-known source of subtle
bugs, and `lm-format-enforcer` has had them beaten out of it by other people's
production traffic. Correctness-critical, bounded in scope, and already solved
is the shape of problem a dependency is for.

What is NOT taken from it is its transformers integration, which imports torch
and `transformers.generation.logits_process` at module scope and would tie this
file to a library version that moves every few weeks. The tokenizer data is
built here instead, from the tokenizer, in about twenty lines.

## The expensive part, and where it is kept

Building the token data walks the whole vocabulary and decodes every token
twice. On Qwen2.5's 151,936-token vocabulary that is tens of seconds -- far too
long to do per request, and the reason it is cached against the model that is
already resident on the card rather than computed here and thrown away.
"""
from __future__ import annotations

import importlib.util
from typing import Any

# Kept as a name rather than an import so this module loads on a runner whose
# image predates the dependency. Everything below degrades to "not available",
# which the controller turns into a refusal that says so -- see the note in
# `controller/api/serving.py` about fields that are accepted and ignored.
AVAILABLE = importlib.util.find_spec("lmformatenforcer") is not None

# Whether the finished reply can be checked against the schema as well as
# parsed. Optional, and the difference is stated rather than hidden: without
# it a reply is still guaranteed to be JSON, and its shape is still whatever
# the grammar allowed -- see `Grammar.validate`.
_CAN_VALIDATE = importlib.util.find_spec("jsonschema") is not None

# How many spaces, tabs and newlines in a row the model may emit between
# JSON tokens. The library's default is twelve, which a model with nothing
# useful to say will happily spend its whole budget on -- replies in testing
# ran to forty characters of whitespace inside one object. Two is enough for
# any formatting anybody wants and cheap to be wrong about.
MAX_CONSECUTIVE_WHITESPACES = 2

# What `response_format.type` may be, and what each one constrains to.
JSON_OBJECT = "json_object"
JSON_SCHEMA = "json_schema"
TEXT = "text"


class Broken(RuntimeError):
    """The grammar did not hold, and the reply is being refused rather than
    handed over.

    This should never happen, and it is checked anyway, because the value of
    `response_format` is entirely in the promise: a caller that sets it has
    stopped checking. A reply that is nearly JSON is worth less than an error,
    because the error is the only one of the two they will notice.

    Two real causes, both measured against `lm-format-enforcer` 0.10 and both
    the reason this class exists:

    * raw tabs and newlines are allowed inside a JSON string, which RFC 8259
      forbids and `json.loads` rejects;
    * an internal parser error is handled by allowing only end-of-text, which
      stops the reply where it stands -- in the middle of a string, if that is
      where it was.

    Failing open is a reasonable default for a library that cannot know what
    its caller promised. It is not a reasonable default here.
    """

    already_explained = True


class Unsupported(ValueError):
    """The request asked for a format this runner cannot enforce.

    Separate from a malformed schema: one is a machine that needs its image
    rebuilt, the other is a request that needs fixing. The caller sees
    different words for each.
    """

    # The agent rewrites raw exceptions into advice about memory and batch
    # sizes, none of which applies to a schema that cannot be enforced. This
    # says "the wording here is already the explanation".
    already_explained = True


def schema_of(response_format: dict | None) -> tuple[str, dict | None]:
    """The format type and the schema it carries, from an OpenAI request.

    `json_schema` is nested one level deeper than it looks -- the schema lives
    at `response_format.json_schema.schema`, with the name and `strict` beside
    it -- and a caller who passes the schema one level up is common enough to
    be worth accepting rather than refusing on a technicality.
    """
    if not isinstance(response_format, dict):
        return TEXT, None
    kind = response_format.get("type") or TEXT
    if kind == JSON_SCHEMA:
        block = response_format.get("json_schema")
        if isinstance(block, dict):
            inner = block.get("schema")
            return kind, inner if isinstance(inner, dict) else block
        return kind, None
    return kind, None


def build(response_format: dict | None, tok: Any, cache: dict,
          log: Any = None) -> "Grammar | None":
    """A grammar for this request, or None where the reply is unconstrained.

    `cache` is a per-model dictionary that outlives the request -- the
    tokenizer data goes in it, because building it costs tens of seconds and
    depends on nothing but the tokenizer.
    """
    kind, schema = schema_of(response_format)
    if kind in (TEXT, None):
        return None
    if kind not in (JSON_OBJECT, JSON_SCHEMA):
        raise Unsupported(
            "`response_format` may be \"text\", \"json_object\" or "
            "\"json_schema\"; this run was asked for %r." % kind)
    if not AVAILABLE:
        raise Unsupported(
            "This machine cannot constrain what a model writes: the grammar "
            "engine is not installed in its runner image. Update the runner, "
            "or ask for `response_format: {\"type\": \"text\"}`.")

    from lmformatenforcer import JsonSchemaParser, TokenEnforcer

    if kind == JSON_SCHEMA and not isinstance(schema, dict):
        raise Unsupported(
            "`response_format: \"json_schema\"` needs a schema to enforce, at "
            "`response_format.json_schema.schema`. Without one, ask for "
            "`json_object` instead -- that constrains the reply to valid JSON "
            "of any shape.")
    try:
        # None is not an oversight: it is how this parser spells "any valid
        # JSON", which is exactly what `json_object` promises.
        from lmformatenforcer import CharacterLevelParserConfig
        parser = JsonSchemaParser(
            schema if kind == JSON_SCHEMA else None,
            config=CharacterLevelParserConfig(
                max_consecutive_whitespaces=MAX_CONSECUTIVE_WHITESPACES))
    except Exception as e:  # noqa: BLE001 - the library raises its own types
        raise Unsupported(
            "That schema cannot be enforced: %s. Simple types, objects, "
            "arrays, enums and `required` are supported; the more unusual "
            "corners of JSON Schema are not." % str(e)[:200]) from e

    data = cache.get("token_enforcer_data")
    if data is None:
        if log:
            log("Reading the vocabulary so the reply can be held to its "
                "format. This happens once per model.")
        data = cache["token_enforcer_data"] = _tokenizer_data(tok)
    return Grammar(TokenEnforcer(data, parser), kind, schema)


def _tokenizer_data(tok: Any):
    """Describe this tokenizer to the enforcer, without importing transformers.

    Three things are wanted: every ordinary token's text, a way to decode a
    run of tokens, and which id ends the reply.

    The decoded text is taken after a leading "0" which is then stripped off.
    That looks like a trick and is load-bearing: most tokenizers encode "a
    word that starts here" and "a word that follows a space" as different
    tokens, and decoding one on its own throws the leading space away. Decoding
    it after a digit keeps it, and the difference in length is what says the
    token begins a word -- which is what lets the enforcer tell ` {` from `{`.
    """
    from lmformatenforcer import TokenEnforcerTokenizerData

    vocab_size = len(tok)
    special = set(tok.all_special_ids or ())
    zero = tok.encode("0")[-1]
    regular: list[tuple[int, str, bool]] = []
    for token_id in range(vocab_size):
        if token_id in special:
            continue
        after_zero = tok.decode([zero, token_id])[1:]
        alone = tok.decode([token_id])
        regular.append((token_id, after_zero, len(after_zero) > len(alone)))

    def decode(ids: list[int]) -> str:
        # The trailing replacement character is a half-finished multi-byte
        # sequence: a token that carries the first byte of a character whose
        # second byte is in the next token. Dropping it means the grammar sees
        # the character when it is whole rather than being handed a U+FFFD it
        # would have to reject.
        return tok.decode(ids).rstrip("�")

    return TokenEnforcerTokenizerData(
        regular_tokens=regular, decoder=decode,
        eos_token_id=tok.eos_token_id, use_bitmask=False,
        vocab_size=vocab_size)


class Grammar:
    """The set of tokens that may come next, given what has been written.

    One per request: the enforcer carries the parse state of this particular
    reply, and sharing one between two conversations would have each of them
    constrained by the other's half-finished object.
    """

    def __init__(self, enforcer: Any, kind: str, schema: dict | None) -> None:
        self.enforcer = enforcer
        self.kind = kind
        self.schema = schema

    def allowed(self, produced: list[int]) -> list[int]:
        """Token ids that keep the reply valid, given the ones so far.

        Only the generated tokens are passed, never the prompt. The enforcer
        works from the last token and the parse state it already holds, so the
        prompt would be thousands of integers copied into a new tuple at every
        step to be ignored at every step.
        """
        return list(self.enforcer.get_allowed_tokens(produced).allowed_tokens)

    def validate(self, text: str) -> None:
        """Check the finished reply really is what it was promised to be.

        The mask is what makes this almost always true; this is what makes it
        always true. Cheap -- one parse of a reply that is at most a few
        thousand characters -- and it converts the library's two known
        failures open (see `Broken`) from "the caller gets subtly wrong data"
        into "the caller gets an error", which is the only trade worth making
        for a field whose entire purpose is that nobody checks the result.

        Schema conformance is checked too where `jsonschema` is installed. It
        is a strictly smaller worry than parsing: the grammar constrains shape
        token by token, so the shape is right by construction, whereas the
        parse is what the control-character laxness actually breaks.
        """
        stripped = text.strip()
        if not stripped:
            raise Broken(
                "The model was asked for %s and produced nothing at all."
                % self.describe())
        import json
        try:
            value = json.loads(stripped)
        except ValueError as e:
            raise Broken(
                "The model was held to %s and what came back does not parse: "
                "%s. Nothing is being returned for this request rather than "
                "text that is not the format it was asked for."
                % (self.describe(), e)) from e
        if self.kind != JSON_SCHEMA or not self.schema or not _CAN_VALIDATE:
            return
        import jsonschema
        try:
            jsonschema.validate(value, self.schema)
        except jsonschema.ValidationError as e:
            raise Broken(
                "The model was held to the schema it was given and what came "
                "back does not match it: %s."
                % str(e).splitlines()[0]) from e
        except jsonschema.SchemaError:
            # The schema itself is malformed. Not the reply's fault, and not
            # worth failing a reply that the grammar was happy to produce --
            # `build` already refused the schemas the grammar cannot take.
            return

    def describe(self) -> str:
        if self.kind == JSON_OBJECT:
            return "valid JSON"
        return "JSON matching the schema it was given"

    def instruction(self) -> str:
        """What to tell the model it is being held to.

        The grammar alone guarantees the shape, so this is not what makes the
        reply valid -- it is what makes it *good*. A model that cannot see the
        schema still emits something matching it, because it has no choice, but
        it fills the fields by guessing what they wanted. Shown the schema, it
        fills them with the answer. The difference is large and costs a few
        hundred tokens of prompt.
        """
        if self.kind == JSON_OBJECT or not self.schema:
            return ("Reply with a single JSON value and nothing else: no "
                    "explanation, no code fence.")
        import json
        return ("Reply with a single JSON value matching this schema, and "
                "nothing else -- no explanation, no code fence:\n%s"
                % json.dumps(self.schema, indent=2, sort_keys=True)[:4000])
