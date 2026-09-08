"""Teaching a vision model to sort pictures into categories.

The first trainer here that reads something other than text, and the one the
whole modality layer was built to make possible: a folder of labelled
pictures becomes a dataset of rows pointing at stored files, and this turns
that into a classifier.

What it does, and does not, do:

* **A pretrained vision backbone, with a new head.** `google/vit-base-patch16-224`
  by default -- a model that already knows what edges, textures and objects
  look like, given a fresh last layer with one output per category. Only that
  head and, with LoRA on, a small adapter in the attention blocks are trained,
  so it fits on a modest card and learns from a few hundred pictures per
  category rather than a few hundred thousand.
* **Held out honestly.** A `validation` split in the dataset is the test; with
  none, ten percent is held back at random with the recorded seed. Accuracy
  on it is the run's headline number -- `primary_metric`, higher better -- and
  is what early stopping watches.
* **Shows its mistakes.** After each evaluation the pictures it got most
  confidently wrong go up as a sample grid, with what it said and what was
  right, and the confusion matrix as a table. A number says a classifier is
  87% right; the grid says it cannot tell huskies from wolves, which is the
  thing you can do something about.

The pictures come from the studio's store, fetched through the runner's own
door and cached beside the model cache, so a second run on the same data
downloads nothing.
"""
from __future__ import annotations

import io
import json
import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx

from runner import artifacts, earlystop

from . import source
from .lora_llm import Cancelled, _versions

DEFAULT_MODEL = "google/vit-base-patch16-224"
DEFAULT_IMAGE_SIZE = 224
ASSET_RE = __import__("re").compile(r"^asset:(ast_[0-9a-f]{12})$")

# Where fetched pictures live between runs, keyed by asset id: the id is
# stable and the bytes behind it never change, so a file once fetched is
# right forever.
PICTURE_CACHE = artifacts.CACHE_DIR.parent / "pictures"

# How many pictures to show when the model is wrong. Enough to see a pattern,
# few enough to read.
WORST_SHOWN = 8


# ---------------------------------------------------------------- data

def _fetch_all(ids: list[str], ctx: Any) -> dict[str, Path]:
    """Every picture the run needs, on this disk, eight at a time."""
    PICTURE_CACHE.mkdir(parents=True, exist_ok=True)
    have = {i: PICTURE_CACHE / i for i in ids if (PICTURE_CACHE / i).exists()}
    missing = [i for i in ids if i not in have]
    if missing:
        ctx.log("Fetching %s pictures from the studio%s."
                % (f"{len(missing):,}",
                   " (%s already here)" % f"{len(have):,}" if have else ""))
    failed: list[str] = []
    done = [0]

    def one(aid: str) -> None:
        url = "%s/api/assets/%s/file" % (ctx.controller_url, aid)
        try:
            r = httpx.get(url, headers={"X-Runner-Token": ctx.runner_token},
                          timeout=120.0)
            r.raise_for_status()
            tmp = PICTURE_CACHE / (aid + ".part")
            tmp.write_bytes(r.content)
            tmp.replace(PICTURE_CACHE / aid)
            have[aid] = PICTURE_CACHE / aid
        except Exception as e:  # noqa: BLE001 - counted, not fatal
            failed.append("%s (%s)" % (aid, e))
        done[0] += 1
        if done[0] % 200 == 0:
            ctx.progress(done[0], len(missing), stage="loading_dataset")

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(one, missing))
    if failed:
        ctx.log("%d picture%s could not be fetched and %s left out: %s"
                % (len(failed), "" if len(failed) == 1 else "s",
                   "is" if len(failed) == 1 else "are",
                   "; ".join(failed[:3])), "warn")
    return have


def _rows(cfg: dict, ctx: Any) -> list[dict]:
    path = source.local_copy(cfg, ctx)
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict):
                    out.append(row)
    return out


def _split(rows: list[dict], image_field: str, label_field: str, seed: int,
           ctx: Any) -> tuple[list, list, str]:
    """(train, held-out, where the held-out came from)."""
    usable = [r for r in rows
              if ASSET_RE.match(str(r.get(image_field) or ""))
              and str(r.get(label_field) or "").strip()]
    dropped = len(rows) - len(usable)
    if dropped:
        ctx.log("%d row%s without a picture or a label were left out."
                % (dropped, "" if dropped == 1 else "s"), "warn")
    if not usable:
        raise ValueError(
            "No row has both a picture in \"%s\" and a label in \"%s\"."
            % (image_field, label_field))
    held = [r for r in usable if str(r.get("split") or "") in
            ("validation", "test", "eval", "dev", "val", "holdout")]
    if held:
        train = [r for r in usable if r not in held]
        return train, held, "the dataset's own held-out split"
    # Held back per category, not from the pile. A random tenth of forty-eight
    # pictures came out as four of one colour and none of the other, and the
    # model was then "100% right" about a category it had never been tested
    # on. One in ten of each category, and at least one of each once a
    # category has ten.
    if len(usable) < 20:
        return usable, [], "nothing: too few pictures"
    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = {}
    for r in usable:
        by_label.setdefault(str(r[label_field]).strip(), []).append(r)
    train, held = [], []
    for name, group in sorted(by_label.items()):
        rng.shuffle(group)
        n = len(group) // 10 if len(group) >= 10 else 0
        held += group[:n]
        train += group[n:]
    rng.shuffle(train)
    return train, held, ("one in ten of each category held back (seed %d)" % seed
                         if held else "nothing: too few pictures per category")


# ---------------------------------------------------------------- training

def run(cfg: dict, ctx: Any) -> dict:
    import torch
    from torch.utils.data import DataLoader, Dataset
    try:
        from PIL import Image
    except ImportError as e:
        raise ValueError(
            "This machine has no image library (Pillow), so it cannot read "
            "the pictures. The runner images ship one; a machine set up by "
            "hand needs `pip install Pillow`.") from e
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    image_field = cfg.get("image_field") or "image"
    label_field = cfg.get("label_field") or "label"
    base = cfg.get("base_model") or DEFAULT_MODEL
    seed = int(cfg.get("seed") or 1234)
    epochs = max(1, int(cfg.get("epochs") or 5))
    batch = max(1, int(cfg.get("batch_size") or 16))
    lr = float(cfg.get("learning_rate") or 5e-4)
    use_lora = bool(cfg.get("lora", True))
    random.seed(seed); torch.manual_seed(seed)

    ctx.progress(0, 1, stage="loading_dataset")
    rows = _rows(cfg, ctx)
    train_rows, held_rows, held_from = _split(rows, image_field, label_field,
                                              seed, ctx)
    labels = sorted({str(r[label_field]).strip() for r in train_rows + held_rows})
    if len(labels) < 2:
        raise ValueError("Only one category (\"%s\") -- there is nothing to "
                         "tell apart." % labels[0])
    index = {name: i for i, name in enumerate(labels)}
    counts = {name: 0 for name in labels}
    for r in train_rows:
        counts[str(r[label_field]).strip()] += 1
    ctx.log("%d categories: %s." % (len(labels), ", ".join(
        "%s (%d)" % (k, v) for k, v in sorted(counts.items(), key=lambda kv: -kv[1])[:12])
        + (" …" if len(labels) > 12 else "")))
    ctx.log("Training on %s pictures, measuring on %s -- %s."
            % (f"{len(train_rows):,}", f"{len(held_rows):,}", held_from))
    thin = [k for k, v in counts.items() if v < 10]
    if thin:
        ctx.log("Under ten pictures for: %s. The model will barely learn those."
                % ", ".join(thin[:8]), "warn")
    ctx.emit_meta({"dataset_rows": len(rows), "labels": labels,
                   "label_counts": counts, "held_out_from": held_from})

    ids = sorted({ASSET_RE.match(str(r[image_field])).group(1)
                  for r in train_rows + held_rows})
    files = _fetch_all(ids, ctx)
    train_rows = [r for r in train_rows if ASSET_RE.match(str(r[image_field])).group(1) in files]
    held_rows = [r for r in held_rows if ASSET_RE.match(str(r[image_field])).group(1) in files]

    # ---- the model
    ctx.progress(0, 1, stage="loading_model")
    ctx.log("Loading %s with a new head for %d categories." % (base, len(labels)))
    processor = AutoImageProcessor.from_pretrained(base, token=cfg.get("hf_token"))
    model = AutoModelForImageClassification.from_pretrained(
        base, num_labels=len(labels),
        id2label={i: n for n, i in index.items()},
        label2id=index, ignore_mismatched_sizes=True,
        token=cfg.get("hf_token"))
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if getattr(torch.backends, "mps", None)
        and torch.backends.mps.is_available() else "cpu")
    if use_lora:
        try:
            from peft import LoraConfig, get_peft_model
            model = get_peft_model(model, LoraConfig(
                r=int(cfg.get("lora_r") or 8), lora_alpha=int(cfg.get("lora_alpha") or 16),
                target_modules=["query", "value"], lora_dropout=0.05,
                modules_to_save=["classifier"]))
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            ctx.log("LoRA on the attention blocks plus the new head: %s of %s "
                    "parameters train (%.1f%%)."
                    % (f"{trainable:,}", f"{total:,}", 100 * trainable / total))
        except Exception as e:  # noqa: BLE001 - fall back to the head only
            ctx.log("LoRA could not be applied to this backbone (%s); training "
                    "the head only." % e, "warn")
            use_lora = False
    if not use_lora:
        for name, p in model.named_parameters():
            p.requires_grad = "classifier" in name
        ctx.log("Training the new head only; the backbone is frozen.")
    model.to(device)

    class Pictures(Dataset):
        def __init__(self, items: list[dict]) -> None:
            self.items = items

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, i: int):
            r = self.items[i]
            aid = ASSET_RE.match(str(r[image_field])).group(1)
            try:
                img = Image.open(files[aid]).convert("RGB")
            except Exception:  # noqa: BLE001 - a broken file is a grey square
                img = Image.new("RGB", (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE), (128, 128, 128))
            px = processor(images=img, return_tensors="pt")["pixel_values"][0]
            return px, index[str(r[label_field]).strip()], i

    # Decoded in the training process rather than in worker processes. The
    # dataset class is local to this function and cannot be pickled to a
    # spawned worker, and decoding a 224-pixel picture is a fraction of the
    # forward pass it feeds -- the workers would be a complication for a
    # speed-up nobody would measure.
    g = torch.Generator(); g.manual_seed(seed)
    train_loader = DataLoader(Pictures(train_rows), batch_size=batch, shuffle=True,
                              generator=g, num_workers=0)
    held_loader = DataLoader(Pictures(held_rows), batch_size=batch, shuffle=False,
                             num_workers=0) if held_rows else None

    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    steps_per_epoch = max(1, math.ceil(len(train_rows) / batch))
    total_steps = steps_per_epoch * epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=lr, total_steps=total_steps,
                                                pct_start=0.1)
    ctx.log("%d epochs of %d steps at batch %d on %s." % (epochs, steps_per_epoch, batch, device))

    best_acc: float | None = None
    best_state = None
    best_step = 0
    # The same stopper the language models use, told which way is up.
    stopper = earlystop.Stopper(
        ctx, int(cfg.get("early_stop_patience") or 3),
        enabled=bool(cfg.get("early_stop", True)) and held_loader is not None,
        kind="classifier", lower_better=False, metric="held-out accuracy")
    stopper.announce(steps_per_epoch)
    stopped_early = False
    last_eval: dict = {}
    step = 0
    t0 = time.time()
    model.train()
    for epoch in range(epochs):
        for px, y, _i in train_loader:
            if ctx.should_cancel():
                raise Cancelled()
            px, y = px.to(device), y.to(device)
            out = model(pixel_values=px, labels=y)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optim.step(); sched.step(); optim.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0 or step == 1:
                ctx.metric(step, {"loss": round(float(out.loss), 5),
                                  "lr": sched.get_last_lr()[0], "epoch": epoch + 1})
            ctx.progress(step, total_steps, stage="training")

        # ---- the end of every epoch: how right is it, and where is it wrong
        if held_loader is None:
            continue
        ctx.progress(step, total_steps, stage="evaluating")
        acc, f1, confusion, worst = _evaluate(model, held_loader, held_rows, labels,
                                              device, torch)
        last_eval = {"accuracy": acc, "macro_f1": f1, "step": step, "epoch": epoch + 1}
        ctx.metric(step, {"val_accuracy": round(acc, 4), "val_macro_f1": round(f1, 4)})
        ctx.log("Epoch %d: %.1f%% right on the held-out pictures (macro F1 %.3f)."
                % (epoch + 1, 100 * acc, f1))
        _show_mistakes(ctx, step, worst, labels, confusion, files, image_field,
                       held_rows, Image)
        verdict = stopper.update(step, acc)
        if verdict == "improved" or best_acc is None:
            best_acc, best_step = acc, step
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif verdict == "stop":
            stopped_early = True
            break
        model.train()

    if best_state is not None and (stopped_early or (last_eval.get("accuracy", 0) < (best_acc or 0))):
        model.load_state_dict(best_state)
        ctx.log("Restored the weights from the best epoch.")

    # ---- save
    ctx.progress(total_steps, total_steps, stage="saving")
    out_dir = Path(ctx.workdir) / "model"
    out_dir.mkdir(parents=True, exist_ok=True)
    if use_lora:
        model = model.merge_and_unload()
    model.save_pretrained(str(out_dir))
    processor.save_pretrained(str(out_dir))
    (out_dir / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=1),
                                         encoding="utf-8")
    zip_path = artifacts.pack(out_dir, Path(ctx.workdir) / "model.zip")

    summary = {
        "kind": "finetune_vision_cls",
        "base_model": base, "labels": labels, "label_counts": counts,
        "trained_on": len(train_rows), "held_out_rows": len(held_rows),
        "held_out_from": held_from,
        "steps": step, "epochs_run": last_eval.get("epoch", epochs),
        "early_stopped": stopped_early, "lora": use_lora, "seed": seed,
        "best_accuracy": round(best_acc, 4) if best_acc is not None else None,
        "best_step": best_step,
        "macro_f1": round(last_eval.get("macro_f1", 0.0), 4) if last_eval else None,
        "confusion": last_eval and confusion if held_loader is not None else None,
        "primary_metric": ({"name": "accuracy", "label": "Accuracy",
                            "value": round(best_acc, 4), "lower_better": False}
                           if best_acc is not None else None),
        "duration_s": round(time.time() - t0, 1),
        "versions": _versions(),
        "artifact_paths": {"model": str(zip_path)},
        "artifact_size": zip_path.stat().st_size,
    }
    if best_acc is not None:
        ctx.log("Best: %.1f%% of the held-out pictures named correctly." % (100 * best_acc))
    return summary


def _evaluate(model, loader, held_rows, labels, device, torch):
    """Accuracy, macro F1, the confusion matrix, and the worst mistakes."""
    model.eval()
    n = len(labels)
    confusion = [[0] * n for _ in range(n)]
    worst: list[tuple[float, int, int, int]] = []     # (confidence, row idx, said, truth)
    with torch.no_grad():
        for px, y, idx in loader:
            logits = model(pixel_values=px.to(device)).logits
            probs = torch.softmax(logits.float(), dim=-1).cpu()
            said = probs.argmax(dim=-1)
            for k in range(len(y)):
                t, s = int(y[k]), int(said[k])
                confusion[t][s] += 1
                if s != t:
                    worst.append((float(probs[k, s]), int(idx[k]), s, t))
    total = sum(map(sum, confusion))
    correct = sum(confusion[i][i] for i in range(n))
    acc = correct / total if total else 0.0
    f1s = []
    for i in range(n):
        tp = confusion[i][i]
        fp = sum(confusion[j][i] for j in range(n)) - tp
        fn = sum(confusion[i]) - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    worst.sort(reverse=True)
    return acc, sum(f1s) / n, confusion, worst[:WORST_SHOWN]


def _show_mistakes(ctx, step, worst, labels, confusion, files, image_field,
                   held_rows, Image) -> None:
    """The confusion matrix as a table, and the worst mistakes as a grid."""
    n = len(labels)
    width = max(len(l) for l in labels)
    head = " " * (width + 2) + " ".join("%6s" % l[:6] for l in labels)
    lines = [head] + ["%-*s  %s" % (width, labels[i][:width],
                                     " ".join("%6d" % confusion[i][j] for j in range(n)))
                      for i in range(n)]
    ctx.sample(step, kind="table", text="rows: what it was · columns: what it said\n"
               + "\n".join(lines))
    if not worst:
        return
    try:
        tiles = []
        for conf, i, said, truth in worst:
            aid = ASSET_RE.match(str(held_rows[i][image_field])).group(1)
            img = Image.open(files[aid]).convert("RGB")
            img.thumbnail((192, 192))
            tiles.append((img, "said %s (%.0f%%), was %s" % (labels[said], 100 * conf, labels[truth])))
        cols = min(4, len(tiles))
        rows_n = math.ceil(len(tiles) / cols)
        grid = Image.new("RGB", (cols * 200, rows_n * 220), (24, 24, 28))
        from PIL import ImageDraw
        draw = ImageDraw.Draw(grid)
        for k, (img, caption) in enumerate(tiles):
            x, y = (k % cols) * 200 + 4, (k // cols) * 220 + 4
            grid.paste(img, (x, y))
            draw.text((x, y + 196), caption[:34], fill=(230, 230, 230))
        path = Path(ctx.workdir) / ("worst-%d.png" % step)
        grid.save(path)
        ctx.sample(step, kind="image", path=str(path),
                   caption="The %d held-out pictures it was most confidently wrong about"
                           % len(tiles))
    except Exception as e:  # noqa: BLE001 - the grid is a courtesy
        ctx.log("Could not draw the mistakes grid (%s)." % e, "warn")
