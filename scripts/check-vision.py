"""Checks for fine-tuning and answering with pictures (roadmap V2).

Pure Python on purpose, like every other check here: the parts of the vision
path that can be wrong without a GPU -- which tokens the loss covers once one
placeholder has become three hundred image tokens, which layers the adapter
goes on, where a picture is written into the text, what a provider is sent --
are tested with a tokenizer and a processor small enough to read in full.

    python scripts/check-vision.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import apimodels  # noqa: E402
from runner import vision_lm  # noqa: E402

failures: list[str] = []


def check(ok: bool, what: str) -> None:
    print(("  ok  " if ok else "FAIL  ") + what)
    if not ok:
        failures.append(what)


# ---------------------------------------------------------------- fakes
#
# One token per character, except the placeholder, which is one token. The
# processor expands that token to IMAGE_TOKENS copies -- which is exactly what
# a Qwen-VL processor does to <|image_pad|>, at a size a person can count.

PAD = "<P>"
PAD_ID = 999
IMAGE_TOKENS = 4


class FakeTok:
    eos_token = "#"

    def _pieces(self, text: str):
        i = 0
        while i < len(text):
            if text.startswith(PAD, i):
                yield PAD_ID, (i, i + len(PAD))
                i += len(PAD)
            else:
                yield ord(text[i]), (i, i + 1)
                i += 1

    def __call__(self, text, return_offsets_mapping=False, add_special_tokens=True):
        pieces = list(self._pieces(text))
        out = {"input_ids": [p[0] for p in pieces]}
        if return_offsets_mapping:
            out["offset_mapping"] = [p[1] for p in pieces]
        return out


class FakeProcessor:
    def __init__(self, tok):
        self.tokenizer = tok

    def __call__(self, text, images=None, return_tensors=None):
        ids = []
        for tid, _ in self.tokenizer._pieces(text[0]):
            ids += [tid] * (IMAGE_TOKENS if tid == PAD_ID else 1)
        return {"input_ids": [ids], "mm_token_type_ids": [[int(t == PAD_ID) for t in ids]]}


print("loss mask across the processor's expansion")
tok = FakeTok()
proc = FakeProcessor(tok)
text = "U:" + PAD + "what?|A:yes#"
start = text.index("A:")
row = vision_lm.encode(text, [(start, len(text))], ["picture"], tok, proc, PAD_ID, 64)
check(row is not None, "an example that fits is encoded")
ids, labels = row["input_ids"], row["labels"]
check(ids.count(PAD_ID) == IMAGE_TOKENS, "one placeholder became %d image tokens" % IMAGE_TOKENS)
check(all(lab == -100 for tid, lab in zip(ids, labels) if tid == PAD_ID),
      "no image token is ever scored")
scored = "".join(chr(t) for t, lab in zip(ids, labels) if lab != -100)
check(scored == "A:yes#", "exactly the reply is scored (got %r)" % scored)
check(len(row["mm_token_type_ids"]) == len(ids), "token types travel with the ids")
check(vision_lm.encode(text, [(start, len(text))], ["picture"], tok, proc, PAD_ID, 8) is None,
      "a row longer than the context is skipped, not cut through its image")

print("pictures written where the model's template puts them")
conv = {"messages": [
    {"role": "system", "content": "s"},
    {"role": "user", "content": "Identify the meal in this image.",
     "media": [{"kind": "image", "ref": "asset:ast_000000000001"}]},
    {"role": "assistant", "content": "{}"}]}
inlined, refs = vision_lm.inline_pictures(conv, "<V><P></V>")
user = inlined["messages"][1]
check(user["content"] == "<V><P></V>Identify the meal in this image.",
      "the placeholder goes in front of the words, with nothing added between")
check("media" not in user, "the picture is not placed a second time by the renderer")
check(refs == ["asset:ast_000000000001"] and vision_lm.asset_ids(refs) == ["ast_000000000001"],
      "the picture's reference comes back in order")

print("a picture survives the serving renderer")
from common import formatting  # noqa: E402

tmpl = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
served, _ = vision_lm.inline_pictures({"messages": [dict(conv["messages"][1])]}, vision_lm.MARK)
text = formatting.render_prompt(served["messages"], {"mode": "chat", "chat_template": tmpl,
                                                     "specials": {"eos_token": "<|im_end|>"}})


class P:
    def apply_chat_template(self, msgs, tokenize=False):
        parts = msgs[0]["content"]
        return "".join("<|vision_start|><|image_pad|><|vision_end|>" if p["type"] == "image"
                       else p["text"] for p in parts)


placed = vision_lm.place(text, P())
check("<|vision_start|><|image_pad|><|vision_end|>Identify" in placed,
      "the placeholder is in the prompt the model is given (special-token text "
      "sent by a user is stripped, so it is placed after rendering)")

print("the adapter goes on the language model only")


class Module:
    def __init__(self, names):
        self.names = names

    def named_modules(self):
        return [(n, None) for n in self.names]


model = Module(["model.visual.blocks.0.attn.q_proj", "model.language_model.layers.0.self_attn.q_proj",
                "model.language_model.layers.0.mlp.down_proj"])
pattern = vision_lm.language_only_targets(model, ["q_proj", "down_proj"])
import re  # noqa: E402

hits = [n for n, _ in model.named_modules() if re.fullmatch(pattern, n)]
check(hits == ["model.language_model.layers.0.self_attn.q_proj",
               "model.language_model.layers.0.mlp.down_proj"],
      "the encoder's layers of the same name are left alone")

print("a list of rows splits the way a dataset does")
rows = vision_lm.Rows({"i": i} for i in range(10))
parts = rows.train_test_split(test_size=3, seed=1)
check(len(parts["test"]) == 3 and len(parts["train"]) == 7
      and not {r["i"] for r in parts["test"]} & {r["i"] for r in parts["train"]},
      "held back and trained on are disjoint")
check(rows.select(range(2)) == [{"i": 0}, {"i": 1}], "select keeps order")

print("a picture reaches a Responses-API provider")
conn = {"provider": "azure_v1", "endpoint": "https://x.services.ai.azure.com", "api_key": "k",
        "api_style": "responses"}
req = apimodels.chat_request(conn, "m", [
    {"role": "system", "content": "s"},
    {"role": "user", "content": "What is this?",
     "media": [{"kind": "image", "ref": "asset:a", "url": "data:image/jpeg;base64,AAAA"}]}],
    {"max_new_tokens": 10})
parts = req["json"]["input"][0]["content"]
check(isinstance(parts, list) and parts[1] == {"type": "input_image",
                                               "image_url": "data:image/jpeg;base64,AAAA"},
      "sent as an input_image part, not dropped")

print("generation carries a row's picture, and a content filter costs one row")
try:
    from runner.jobs import generate_data as g
except ImportError as e:  # the runner's own dependencies are not installed here
    print("  --  skipped (%s)" % e)
else:
    row = {"image": "asset:ast_000000000002", "prompt": "Identify the meal in this image."}
    media = g._media_of(row, {}, {})
    check(media == [{"kind": "image", "ref": "asset:ast_000000000002"}], "a flat row's picture is found")
    written = g._row("{}", {"prompt": row["prompt"], "media": media}, {})
    check(written["messages"][0].get("media") == media, "the written row keeps its picture")
    other = g._row("{}", {"prompt": row["prompt"], "media": [{"kind": "image", "ref": "asset:ast_000000000003"}]}, {})
    check(g._dedupe_key(written) != g._dedupe_key(other),
          "two pictures asked the same question are two rows, not a repeat")
    refusal = json.dumps({"error": {"message": "Image processing blocked due to content policy violation."}})
    check(bool(g._CONTENT_REFUSAL.search(refusal)), "a content-filter 400 is recognised as one row's problem")
    check(not g._CONTENT_REFUSAL.search('{"error": {"message": "Unsupported parameter: temperature"}}'),
          "a bad parameter is still the whole run's problem")

print("a model saved by transformers 5 is readable by the GGUF converter")
try:
    from runner.jobs import export_gguf
except ImportError as e:
    print("  --  skipped (%s)" % e)
else:
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        model = Path(tmp) / "model"
        model.mkdir()
        (model / "config.json").write_text(json.dumps({"vision_config": {"depth": 24}}))
        (model / "tokenizer_config.json").write_text(json.dumps(
            {"extra_special_tokens": ["<|im_start|>"], "eos_token": "<|im_end|>"}))
        got = export_gguf._readable_by_converter(model, Path(tmp) / "shadow")
        cfg = json.loads((got / "tokenizer_config.json").read_text())
        check(got != model and "extra_special_tokens" not in cfg and cfg["eos_token"] == "<|im_end|>",
              "the list-shaped field is dropped from a copy, the rest kept")
        check("extra_special_tokens" in json.loads((model / "tokenizer_config.json").read_text()),
              "the cached model itself is not touched")
        check(export_gguf._is_vision(got), "the copy is still recognised as a vision model")

if failures:
    print("\n%d check(s) failed." % len(failures))
    sys.exit(1)
print("\nAll checks passed.")
