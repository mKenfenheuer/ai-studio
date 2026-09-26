"""Fine-tuning a model that looks as well as reads (roadmap V2).

A vision-language model is a language model with an image encoder in front of
it: the picture becomes a run of embeddings that sit in the token sequence
where the chat template put a placeholder, and from there on it is the same
next-token prediction the text trainer already does. So this is not a second
trainer. `lora_llm` keeps the loop, the optimiser, the held-out loss, early
stopping, checkpoints and the merge; this module supplies the four things that
are different when a turn shows a picture:

* **which class to load, and the processor beside it.** The tokenizer alone
  cannot turn a photograph into the tensors the encoder reads, and it cannot
  say how many tokens a given photograph becomes.
* **where the picture goes in the text.** Every family spells its placeholder
  differently, and the published template decides whether a newline follows
  it. That is read off the model's own template rather than listed here, so a
  model this file has never heard of is placed the way its authors place it.
* **which tokens the loss covers.** The processor expands one placeholder
  into hundreds of image tokens, one per patch group. The assistant-only mask
  is worked out on the text first, exactly as for a text model, and carried
  across the expansion; image tokens are context and are never scored.
* **what a batch holds.** The pixels are not stored with the row -- a few
  thousand photographs as tensors would not fit in host memory -- so the row
  keeps the files and each batch turns its own pictures into pixels.

The adapter goes on the language model only. The image encoder is left as it
was trained: it already sees food, and a few thousand pictures are enough to
teach the model what to *say* about what it sees but far too few to retrain
how it sees without making that worse.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# How large a picture is allowed to be, in pixels, before it is scaled down.
# The families this was written against default to sixteen megapixels, which
# for a phone photo is thousands of image tokens per turn and most of the card.
# A plate of food is recognisable at a fraction of that; 640x640 is ~400
# tokens on a Qwen-VL model and is what the studio trains at unless told
# otherwise (`image_max_pixels`). Serving reads the same number back off the
# run, so a model is shown pictures at the size it learned from.
DEFAULT_MAX_PIXELS = 640 * 640


def is_vision_model(base_model: str, token: str | None = None) -> bool:
    """Whether this checkpoint has an image encoder, read off its config."""
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(base_model, token=token)
    except Exception:  # noqa: BLE001 - an unreadable config is a text model's problem
        return False
    return getattr(cfg, "vision_config", None) is not None


def is_vision_dir(path: Path) -> bool:
    """The same question of a saved model directory, without importing torch."""
    try:
        cfg = json.loads((Path(path) / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return bool(cfg.get("vision_config"))


def load_processor(base_model: str, token: str | None, max_pixels: int | None):
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(base_model, token=token)
    cap_pixels(processor, max_pixels)
    return processor


def cap_pixels(processor, max_pixels: int | None) -> None:
    """Hold pictures to `max_pixels`, however this image processor spells it."""
    limit = int(max_pixels or DEFAULT_MAX_PIXELS)
    ip = getattr(processor, "image_processor", None)
    if ip is None:
        return
    size = getattr(ip, "size", None)
    # Qwen-VL keeps the pixel budget in `size.longest_edge` (a pixel count,
    # despite the name); older releases kept a separate `max_pixels`.
    if size is not None and getattr(size, "longest_edge", None) is not None:
        try:
            size.longest_edge = limit
        except Exception:  # noqa: BLE001 - a dict on older releases
            size["longest_edge"] = limit
    if hasattr(ip, "max_pixels"):
        ip.max_pixels = limit


def model_class():
    """The auto class that loads an image+text-to-text checkpoint."""
    try:
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText
    except ImportError:  # transformers 4.x
        from transformers import AutoModelForVision2Seq
        return AutoModelForVision2Seq


def placeholder(processor) -> str:
    """What the model's own template writes where one picture goes.

    Measured, not listed: render a turn with a picture and the same turn
    without, and the difference is the placeholder -- `<|vision_start|>
    <|image_pad|><|vision_end|>` for Qwen-VL, `<image>` for LLaVA, a longer
    block for Gemma. Whatever the template writes around it (a newline, or
    nothing) comes with it, so the text trained on is the text the published
    template produces and every runtime outside this studio will produce.
    """
    probe = "§PROBE§"
    with_image = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": probe}]}],
        tokenize=False)
    without = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": probe}]}],
        tokenize=False)
    head = without.split(probe)[0]
    got = with_image.split(probe)[0]
    if not got.startswith(head) or got == head:
        raise ValueError("Could not work out where this model's template puts "
                         "a picture, so it cannot be trained on pictures here.")
    return got[len(head):]


# Where a picture goes, while a prompt is being rendered for serving. The
# serving renderer strips special-token text out of what people send -- a user
# typing "<|im_end|>" must not be able to close their own turn -- and the
# placeholder IS special-token text, so written in directly it vanished and the
# model was handed pixels with nowhere to put them. This marker survives the
# renderer and is swapped for the real placeholder afterwards.
MARK = "\u2063ai-studio-picture\u2063"


def place(text: str, processor) -> str:
    """The rendered prompt with each MARK replaced by the model's placeholder."""
    return text.replace(MARK, placeholder(processor))


def image_token_id(processor) -> int:
    tok = getattr(processor, "image_token", None) or "<image>"
    return processor.tokenizer.convert_tokens_to_ids(tok)


def inline_pictures(conv: dict, mark: str) -> tuple[dict, list[str]]:
    """The conversation with each picture written into its turn's text.

    Returns the conversation (media removed, so the shared renderer adds no
    placeholder of its own) and the stored references in the order the
    pictures now appear in the text -- which is the order the processor
    matches them to placeholders.
    """
    refs: list[str] = []
    msgs = []
    for m in conv.get("messages") or []:
        pics = [mm for mm in (m.get("media") or []) if mm.get("kind") == "image"]
        if not pics:
            msgs.append(m)
            continue
        refs += [mm.get("ref") or mm.get("url") or "" for mm in pics]
        rest = [mm for mm in m.get("media") or [] if mm.get("kind") != "image"]
        new = {**m, "content": mark * len(pics) + (m.get("content") or "")}
        if rest:
            new["media"] = rest
        else:
            new.pop("media", None)
        msgs.append(new)
    return {**conv, "messages": msgs}, refs


def asset_ids(refs: list[str]) -> list[str]:
    return [r[6:] for r in refs if r.startswith("asset:")]


def open_picture(path: Path):
    from PIL import Image
    with Image.open(path) as im:
        return im.convert("RGB")


def encode(text: str, spans: list[tuple[int, int]], pictures: list, tok,
           processor, image_id: int, max_seq: int) -> dict | None:
    """One example, tokenized with its pictures and masked to the reply.

    The mask is computed on the text as written -- one placeholder per
    picture -- with the same offset logic the text trainer uses, then walked
    across the processor's expansion: where the processor put N image tokens
    for the one in the text, the extra N-1 are unmasked context. A row that
    would be cut short is skipped rather than truncated: cutting through a run
    of image tokens leaves the encoder's output and the placeholder count
    disagreeing, which is an error at the first forward pass.
    """
    plain = tok(text, return_offsets_mapping=True, add_special_tokens=True)
    ids, offsets = plain["input_ids"], plain["offset_mapping"]
    labels_plain = []
    for tid, (start, end) in zip(ids, offsets):
        if not spans:
            labels_plain.append(tid)
            continue
        inside = any(s <= start and (end or start + 1) <= e for s, e in spans)
        labels_plain.append(tid if inside else -100)

    full = processor(text=[text], images=pictures or None, return_tensors=None)
    full_ids = list(full["input_ids"][0])
    if len(full_ids) > max_seq:
        return None
    labels: list[int] = []
    i = 0
    for tid in full_ids:
        if i < len(ids) and tid == ids[i]:
            labels.append(labels_plain[i])
            i += 1
        elif tid == image_id:
            labels.append(-100)
        else:
            raise ValueError("the processor changed text it should only have "
                             "expanded (token %d)" % tid)
    if i != len(ids):
        raise ValueError("the processor dropped part of the text")
    row = {"input_ids": full_ids, "attention_mask": [1] * len(full_ids),
           "labels": labels}
    if "mm_token_type_ids" in full:
        row["mm_token_type_ids"] = list(full["mm_token_type_ids"][0])
    return row


class Collator:
    """Pads a batch to its own longest row and turns its pictures into pixels."""

    def __init__(self, processor, pad_id: int, pictures: dict[str, Path]):
        self.processor = processor
        self.pad_id = pad_id
        self.pictures = pictures

    def __call__(self, rows: list[dict]) -> dict:
        import torch
        width = max(len(r["input_ids"]) for r in rows)
        out: dict[str, list] = {"input_ids": [], "attention_mask": [], "labels": []}
        typed = all("mm_token_type_ids" in r for r in rows)
        if typed:
            out["mm_token_type_ids"] = []
        images = []
        for r in rows:
            gap = width - len(r["input_ids"])
            out["input_ids"].append(list(r["input_ids"]) + [self.pad_id] * gap)
            out["attention_mask"].append(list(r["attention_mask"]) + [0] * gap)
            out["labels"].append(list(r["labels"]) + [-100] * gap)
            if typed:
                out["mm_token_type_ids"].append(list(r["mm_token_type_ids"]) + [0] * gap)
            images += [open_picture(self.pictures[a]) for a in r["pictures"]]
        batch = {k: torch.tensor(v, dtype=torch.long) for k, v in out.items()}
        if images:
            # The same image processor, at the same size, that counted the
            # tokens in `encode` -- so the pixels and the placeholders agree.
            px = self.processor.image_processor(images=images, return_tensors="pt")
            batch.update({k: v for k, v in px.items()})
        return batch


def language_only_targets(model, picked: list[str]) -> str:
    """The adapter's target modules, restricted to the language model.

    The picker names layers by their short names (`q_proj`, `down_proj`),
    and the image encoder has layers called the same thing. A regex over the
    full module path is PEFT's way of saying "these, but only over there".
    """
    names = [n for n, _ in model.named_modules()]
    lm = next((p for p in ("model.language_model", "language_model", "model.text_model")
               if any(n.startswith(p + ".") for n in names)), None)
    alternatives = "|".join(sorted(set(picked)))
    if lm is None:
        # No recognisable language-model subtree: keep the encoder out by name.
        return r"^(?!.*(visual|vision)).*\.(%s)$" % alternatives
    return r"^%s\..*\.(%s)$" % (lm.replace(".", r"\."), alternatives)


def save_processor(processor, where: Path, ctx: Any) -> None:
    """The processor goes wherever the model goes, or nothing can feed it."""
    try:
        processor.save_pretrained(str(where))
    except Exception as e:  # noqa: BLE001 - said; the weights are still good
        ctx.log("The image processor could not be saved beside the model "
                "(%s). It loads from the base model's repository instead."
                % e, "warn")


class Rows(list):
    """Examples held as a plain list, with the two dataset methods the loop uses.

    Not a `datasets.Dataset`: its rows would have to be tensors or Arrow
    types, and these carry file references that become pixels a batch at a
    time. The trainer only ever asks a dataset for its length, its rows, a
    slice, and a random split, so that is all this is.
    """

    def select(self, indices) -> "Rows":
        return Rows(self[i] for i in indices)

    def train_test_split(self, test_size: int, seed: int) -> dict:
        import random
        order = list(range(len(self)))
        random.Random(seed).shuffle(order)
        return {"test": self.select(order[:test_size]),
                "train": self.select(order[test_size:])}


def build(raw, fmt: dict, train_on: str, tok, processor, max_seq: int,
          ctx: Any, conversation, pictures: dict[str, Path]) -> tuple[Rows, dict]:
    """Every row of `raw` as a trainable example. Returns (rows, counts)."""
    mark = placeholder(processor)
    image_id = image_token_id(processor)
    out = Rows()
    counts = {"too_long": 0, "missing_picture": 0, "unreadable": 0, "inexact": 0}
    for row in raw:
        conv, _ = conversation.repair(conversation.from_row(row, fmt))
        conv, refs = inline_pictures(conv, mark)
        ids = asset_ids(refs)
        if len(ids) != len(refs) or any(a not in pictures for a in ids):
            counts["missing_picture"] += 1
            continue
        text, spans, exact = conversation.trainable_spans(conv, fmt, train_on)
        if not exact:
            counts["inexact"] += 1
        # The terminator inside the last trained span, as in the text path:
        # a reply the model is never scored on ending is a reply it never ends.
        if text and not text.rstrip().endswith(tok.eos_token):
            if spans and spans[-1][1] == len(text):
                spans[-1] = (spans[-1][0], len(text) + len(tok.eos_token))
            text += tok.eos_token
        try:
            ex = encode(text, spans, [open_picture(pictures[a]) for a in ids],
                        tok, processor, image_id, max_seq)
        except Exception as e:  # noqa: BLE001 - counted and said below
            counts["unreadable"] += 1
            if counts["unreadable"] <= 3:
                ctx.log("A row could not be prepared (%s); it is left out." % e, "warn")
            continue
        if ex is None:
            counts["too_long"] += 1
            continue
        ex["pictures"] = ids
        out.append(ex)
    return out, counts


def picture_from(mm: dict, controller_url: str, runner_token: str):
    """One picture from a message, as a PIL image.

    A request through `/v1` carries it as a data URL; the playground sends a
    reference to the studio's store, fetched through the runner's own door.
    """
    import base64
    import io

    from PIL import Image
    url = mm.get("url") or ""
    if url.startswith("data:"):
        blob = base64.b64decode(url.split(",", 1)[1])
    else:
        ref = mm.get("ref") or ""
        if not ref.startswith("asset:"):
            raise ValueError("a picture can only be read from a data URL or "
                             "the studio's store, not %r" % (url or ref)[:60])
        import httpx
        r = httpx.get("%s/api/assets/%s/file" % (controller_url, ref[6:]),
                      headers={"X-Runner-Token": runner_token}, timeout=60.0)
        r.raise_for_status()
        blob = r.content
    with Image.open(io.BytesIO(blob)) as im:
        return im.convert("RGB")


def summary_pixels(path: Path) -> int | None:
    """The picture size a saved run was trained at, if it says."""
    try:
        s = json.loads((Path(path) / "ai_studio_summary.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return ((s.get("vision") or {}).get("image_max_pixels")) or None


_REPEAT_CHECKED: set = set()


def repair_repeat_interleave(torch, device: str, log) -> bool:
    """Route `repeat_interleave` through the processor where the card gets it wrong.

    On the RX 6900 XT with torch 2.9.1+rocm6.4, `torch.repeat_interleave` with
    a tensor of repeat counts fails with hipErrorIllegalState -- on a three-
    element tensor, in fp16 and fp32 alike, and in fp32 it takes the GPU down
    with a memory access fault. Qwen3-VL's image encoder calls it first thing,
    on the picture grids, so no vision model could train or answer on that
    card. Everything around it works, and the calls vision models make are on
    a handful of integers, so computing those on the processor and moving the
    answer back costs nothing measurable.

    Probed once per process and installed only where the probe fails: a card
    that does this right keeps the native kernel.
    """
    if device != "cuda" or device in _REPEAT_CHECKED:
        return False
    _REPEAT_CHECKED.add(device)
    try:
        g = torch.tensor([30, 40], device="cuda")
        torch.repeat_interleave(g, torch.tensor([2, 1], device="cuda"))
        torch.cuda.synchronize()
        return False
    except Exception:  # noqa: BLE001 - the failure is the finding
        pass
    native = torch.repeat_interleave

    def repeat_interleave(input, repeats=None, dim=None, *, output_size=None):
        on_card = torch.is_tensor(input) and input.device.type == "cuda"
        if on_card and repeats is None:
            return native(input.cpu(), output_size=output_size).to(input.device)
        if on_card and torch.is_tensor(repeats):
            return native(input.cpu(), repeats.cpu(), dim=dim,
                          output_size=output_size).to(input.device)
        if repeats is None:
            return native(input, output_size=output_size)
        return native(input, repeats, dim=dim, output_size=output_size)

    torch.repeat_interleave = repeat_interleave
    torch.Tensor.repeat_interleave = (
        lambda self, repeats=None, dim=None, *, output_size=None:
        repeat_interleave(self, repeats, dim=dim, output_size=output_size))
    log("This card's repeat_interleave kernel is broken (it fails on a "
        "three-element tensor), and the image encoder needs it: those calls "
        "are made on the processor instead. They are a few integers each.")
    return True
