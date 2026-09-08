"""Answering "what is this picture?" with a classifier this studio trained.

The playground talks to language models through `inference.ModelHost`, which
is built around a tokenizer and a stream of tokens; a classifier has neither.
This is the small counterpart: load the model a run left behind, give it one
picture, return every label with its probability.

Kept resident after the first call, one classifier at a time -- they are
small, and the question is asked in bursts from one page.
"""
from __future__ import annotations

import base64
import io
import json
import threading
import time
from typing import Any, Callable

from runner import artifacts

_lock = threading.Lock()
_loaded: dict[str, Any] = {}       # job_id -> (model, processor, labels)


def _load(controller_url: str, token: str, job_id: str,
          log: Callable[[str], None]) -> tuple:
    if job_id in _loaded:
        return _loaded[job_id]
    import torch
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    folder = artifacts.fetch(controller_url, token, job_id, log)
    processor = AutoImageProcessor.from_pretrained(str(folder))
    model = AutoModelForImageClassification.from_pretrained(str(folder)).eval()
    labels_file = folder / "labels.json"
    if labels_file.exists():
        labels = json.loads(labels_file.read_text(encoding="utf-8"))
    else:
        labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    _loaded.clear()                 # one at a time; see the module docstring
    _loaded[job_id] = (model, processor, labels, device)
    return _loaded[job_id]


def classify(controller_url: str, token: str, job_id: str, image_b64: str,
             top: int = 5, log: Callable[[str], None] = lambda _s: None) -> dict:
    """Every label with its probability, best first, for one picture."""
    import torch
    from PIL import Image
    with _lock:
        model, processor, labels, device = _load(controller_url, token, job_id, log)
        t0 = time.time()
        img = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
        px = processor(images=img, return_tensors="pt")["pixel_values"].to(device)
        with torch.no_grad():
            probs = torch.softmax(model(pixel_values=px).logits[0].float(), dim=-1).cpu()
    ranked = sorted(((float(probs[i]), labels[i] if i < len(labels) else str(i))
                     for i in range(len(probs))), reverse=True)
    return {"labels": [{"label": name, "probability": round(p, 4)}
                       for p, name in ranked[:max(1, top)]],
            "seconds": round(time.time() - t0, 3),
            "size": list(img.size)}
