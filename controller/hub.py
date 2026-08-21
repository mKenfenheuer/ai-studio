"""Hugging Face Hub integration.

Beyond proxying search, this module answers the question a beginner actually
has -- "will this model run on my machine?" -- by estimating memory needs from
the model's real file sizes and comparing them against each runner's VRAM.
"""
from __future__ import annotations

import re
from typing import Any

import httpx

from . import config

HF_API = "https://huggingface.co/api"
DATASETS_SERVER = "https://datasets-server.huggingface.co"

# A curated on-ramp. The Hub has millions of models; someone who has never
# fine-tuned anything needs five good options with plain-language notes, not a
# search box. Every entry here is ungated, standard-architecture and known to
# train cleanly with LoRA.
STARTER_MODELS = [
    {
        "id": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "label": "SmolLM2 135M",
        "params_b": 0.135,
        "blurb": "Tiny and very fast. Best for learning how training works "
                 "and checking your data is right before a long run.",
        "min_vram_gb": 2,
        "good_for": ["first time", "quick experiments"],
    },
    {
        "id": "Qwen/Qwen2.5-0.5B-Instruct",
        "label": "Qwen2.5 0.5B",
        "params_b": 0.5,
        "blurb": "Small but genuinely capable. A good default when you want "
                 "real results without waiting hours.",
        "min_vram_gb": 4,
        "good_for": ["first time", "chat", "style"],
    },
    {
        "id": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "label": "SmolLM2 1.7B",
        "params_b": 1.7,
        "blurb": "Noticeably smarter than the small ones, still fits on most "
                 "gaming GPUs.",
        "min_vram_gb": 8,
        "good_for": ["chat", "instructions"],
    },
    {
        "id": "Qwen/Qwen2.5-3B-Instruct",
        "label": "Qwen2.5 3B",
        "params_b": 3.0,
        "blurb": "Strong all-rounder. Needs a 12GB+ card in 16-bit, or 4-bit "
                 "quantization on smaller cards.",
        "min_vram_gb": 12,
        "good_for": ["chat", "instructions", "reasoning"],
    },
    {
        "id": "mistralai/Mistral-7B-Instruct-v0.3",
        "label": "Mistral 7B",
        "params_b": 7.2,
        "blurb": "Well-known and capable. Needs 4-bit quantization to fit on "
                 "a 16GB card.",
        "min_vram_gb": 24,
        "min_vram_gb_4bit": 8,
        "good_for": ["chat", "serious work"],
    },
]

STARTER_DATASETS = [
    {
        "id": "tatsu-lab/alpaca",
        "label": "Alpaca (instructions)",
        "rows": 52002,
        "blurb": "52k instruction-and-response pairs. The classic starting "
                 "point for teaching a model to follow instructions.",
        "format": "instruction",
    },
    {
        "id": "databricks/databricks-dolly-15k",
        "label": "Dolly 15k",
        "rows": 15011,
        "blurb": "15k human-written instructions across brainstorming, "
                 "classification, Q&A and summarising.",
        "format": "instruction",
    },
    {
        "id": "yahma/alpaca-cleaned",
        "label": "Alpaca (cleaned)",
        "rows": 51760,
        "blurb": "Alpaca with errors fixed. Usually the better choice of the two.",
        "format": "instruction",
    },
    {
        "id": "HuggingFaceH4/ultrachat_200k",
        "label": "UltraChat 200k",
        "rows": 207865,
        "blurb": "Large multi-turn conversations. Use a subset unless you have "
                 "hours to spare.",
        "format": "chat",
    },
]

# Corpora for training from scratch. A different job from the instruction
# datasets above: a model with no prior knowledge of language needs a large
# amount of ordinary running text, not question-and-answer pairs.
#
# `approx_tokens` is what the corpus holds in total. It matters because a
# from-scratch run consumes tokens far faster than people expect -- exhausting
# the corpus and looping over it repeatedly is the quiet way a promising loss
# curve turns into memorisation.
STARTER_CORPORA = [
    {
        "id": "roneneldan/TinyStories",
        "label": "TinyStories",
        "text_field": "text",
        "rows": 2119719,
        "approx_tokens": 470_000_000,
        "sample_prompt": "Once upon a time, there was a little",
        "blurb": "Short children's stories written with a deliberately small "
                 "vocabulary. Built so that even a very small model can learn "
                 "to write real, coherent English -- by far the best first "
                 "choice.",
        "recommended": True,
    },
    {
        "id": "Salesforce/wikitext",
        "config": "wikitext-103-raw-v1",
        "label": "Wikipedia (WikiText-103)",
        "text_field": "text",
        "rows": 1801350,
        "approx_tokens": 117_000_000,
        "sample_prompt": "The history of",
        "blurb": "Real Wikipedia articles. Huge vocabulary and dense facts, so "
                 "a small model learns the shape of encyclopaedic writing "
                 "without the knowledge to fill it in.",
    },
    {
        "id": "stas/openwebtext-10k",
        "label": "Web text (small sample)",
        "text_field": "text",
        "rows": 10000,
        "approx_tokens": 10_000_000,
        "sample_prompt": "The best way to",
        "blurb": "Ten thousand web pages. Small enough to download in seconds, "
                 "which makes it useful for testing a setup end to end before "
                 "starting something long.",
    },
    {
        "id": "codeparrot/codeparrot-clean-valid",
        "label": "Python code",
        "text_field": "content",
        "rows": 18000,
        "approx_tokens": 40_000_000,
        "sample_prompt": "def ",
        "blurb": "Python source files. Code has stricter structure than prose, "
                 "so progress is unusually easy to see -- brackets start "
                 "matching before the logic makes sense.",
    },
]

_client: httpx.AsyncClient | None = None


def client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        headers = {"User-Agent": "ai-studio/0.1"}
        if config.HF_TOKEN:
            headers["Authorization"] = "Bearer %s" % config.HF_TOKEN
        _client = httpx.AsyncClient(timeout=30, headers=headers,
                                    follow_redirects=True)
    return _client


async def search_models(query: str = "", limit: int = 30,
                        task: str = "text-generation") -> list[dict]:
    params: dict[str, Any] = {
        "limit": limit,
        "sort": "downloads",
        "direction": -1,
        "filter": task,
        "full": "false",
    }
    if query:
        params["search"] = query
    r = await client().get(HF_API + "/models", params=params)
    r.raise_for_status()
    return [_slim_model(m) for m in r.json()]


async def search_datasets(query: str = "", limit: int = 30) -> list[dict]:
    params: dict[str, Any] = {"limit": limit, "sort": "downloads", "direction": -1}
    if query:
        params["search"] = query
    r = await client().get(HF_API + "/datasets", params=params)
    r.raise_for_status()
    return [{
        "id": d.get("id"),
        "downloads": d.get("downloads", 0),
        "likes": d.get("likes", 0),
        "tags": d.get("tags", [])[:8],
        "updated": d.get("lastModified"),
    } for d in r.json()]


def _slim_model(m: dict) -> dict:
    return {
        "id": m.get("id"),
        "downloads": m.get("downloads", 0),
        "likes": m.get("likes", 0),
        "gated": bool(m.get("gated")),
        "pipeline_tag": m.get("pipeline_tag"),
        "tags": [t for t in m.get("tags", []) if not t.startswith("dataset:")][:10],
        "params_b": _params_from_name(m.get("id", "")),
    }


_PARAM_RE = re.compile(r"[-_/](\d+(?:\.\d+)?)\s*b\b", re.IGNORECASE)


def _params_from_name(model_id: str) -> float | None:
    """Model IDs almost always encode their size ('Qwen2.5-3B-Instruct').
    A cheap heuristic that avoids an extra API round-trip per search result."""
    if m := _PARAM_RE.search(model_id.replace("B-", "B-")):
        try:
            return float(m.group(1))
        except ValueError:
            return None
    if m := re.search(r"(\d+(?:\.\d+)?)m\b", model_id, re.IGNORECASE):
        try:
            return round(float(m.group(1)) / 1000, 3)
        except ValueError:
            return None
    return None


async def model_detail(model_id: str) -> dict:
    r = await client().get("%s/models/%s" % (HF_API, model_id))
    r.raise_for_status()
    m = r.json()

    total_bytes = 0
    for s in m.get("siblings", []) or []:
        name = s.get("rfilename", "")
        if name.endswith((".safetensors", ".bin")) and "training_args" not in name:
            total_bytes += s.get("size") or 0

    params_b = _params_from_name(model_id)
    if not params_b and total_bytes:
        # Weights are usually fp16/bf16, so bytes/2 approximates parameters.
        params_b = round(total_bytes / 2 / 1e9, 2)

    return {
        "id": m.get("id"),
        "gated": bool(m.get("gated")),
        "pipeline_tag": m.get("pipeline_tag"),
        "downloads": m.get("downloads", 0),
        "likes": m.get("likes", 0),
        "tags": m.get("tags", [])[:20],
        "architecture": (m.get("config", {}) or {}).get("architectures", [None])[0],
        "params_b": params_b,
        "weights_bytes": total_bytes or None,
        "memory": estimate_memory(params_b),
    }


def estimate_memory(params_b: float | None) -> dict | None:
    """VRAM needed to LoRA fine-tune a model of this size.

    LoRA freezes the base weights, so the frozen copy dominates. The 1.35x
    multiplier covers activations, LoRA gradients, optimiser state and
    allocator fragmentation -- calibrated against measured runs.
    """
    if not params_b:
        return None
    return {
        "fp16_gb": round(params_b * 2 * 1.35, 1),
        "int8_gb": round(params_b * 1 * 1.35, 1),
        "int4_gb": round(params_b * 0.5 * 1.35, 1),
        "inference_fp16_gb": round(params_b * 2 * 1.1, 1),
    }


def fit_report(params_b: float | None, runner_caps: dict) -> dict:
    """Can this runner train this model? Returns a plain-language verdict."""
    mem = estimate_memory(params_b)
    vram = runner_caps.get("vram_gb")
    has_4bit = bool(runner_caps.get("quantization", {}).get("4bit"))
    if not mem or not vram:
        return {"verdict": "unknown",
                "message": "Cannot tell how much memory this needs."}

    if mem["fp16_gb"] <= vram:
        return {"verdict": "fits", "precision": "16-bit",
                "needed_gb": mem["fp16_gb"], "available_gb": vram,
                "message": "Fits comfortably in 16-bit (about %.1f GB of your %.1f GB)."
                           % (mem["fp16_gb"], vram)}
    if has_4bit and mem["int4_gb"] <= vram:
        return {"verdict": "fits_quantized", "precision": "4-bit",
                "needed_gb": mem["int4_gb"], "available_gb": vram,
                "message": "Too big in 16-bit, but fits in 4-bit (about %.1f GB). "
                           "Quality drops slightly." % mem["int4_gb"]}
    if not has_4bit and mem["int4_gb"] <= vram:
        return {"verdict": "needs_quantization", "precision": "4-bit",
                "needed_gb": mem["int4_gb"], "available_gb": vram,
                "message": "This would fit in 4-bit, but this runner has no "
                           "working 4-bit support, so it cannot be used here."}
    return {"verdict": "too_big", "needed_gb": mem["fp16_gb"], "available_gb": vram,
            "message": "Too large for this runner's %.1f GB of memory, even "
                       "compressed. Pick a smaller model." % vram}


async def dataset_preview(dataset_id: str, config_name: str | None = None,
                          split: str = "train") -> dict:
    """Fetch the first rows so the user can SEE their data before training.

    Showing real rows catches the single most common beginner mistake -- the
    wrong column selected -- before it wastes an hour of GPU time.
    """
    params = {"dataset": dataset_id, "split": split}
    if not config_name:
        try:
            r = await client().get(DATASETS_SERVER + "/splits",
                                   params={"dataset": dataset_id})
            r.raise_for_status()
            splits = r.json().get("splits", [])
            if splits:
                config_name = splits[0]["config"]
                if not any(s["split"] == split for s in splits):
                    split = splits[0]["split"]
                params["split"] = split
        except httpx.HTTPError:
            config_name = "default"
    params["config"] = config_name or "default"

    r = await client().get(DATASETS_SERVER + "/first-rows", params=params)
    if r.status_code != 200:
        return {"available": False, "reason": "No preview available for this dataset.",
                "dataset": dataset_id}
    data = r.json()
    rows = [row.get("row", {}) for row in data.get("rows", [])[:5]]
    features = [f.get("name") for f in data.get("features", [])]
    return {
        "available": True,
        "dataset": dataset_id,
        "config": params["config"],
        "split": params["split"],
        "columns": features,
        "rows": rows,
        "detected_format": detect_format(features),
    }


def detect_format(columns: list[str]) -> dict:
    """Guess how the dataset is laid out, so the UI can pre-fill the mapping."""
    cols = {c.lower() for c in columns if c}
    if {"instruction"} & cols and cols & {"output", "response"}:
        return {"mode": "instruction", "instruction_field": "instruction",
                "response_field": "output" if "output" in cols else "response",
                "confidence": "high"}
    if "messages" in cols or "conversations" in cols:
        return {"mode": "chat",
                "messages_field": "messages" if "messages" in cols else "conversations",
                "confidence": "high"}
    if {"prompt"} & cols and cols & {"completion", "response", "answer"}:
        resp = next(c for c in ("completion", "response", "answer") if c in cols)
        return {"mode": "instruction", "instruction_field": "prompt",
                "response_field": resp, "confidence": "high"}
    for c in ("text", "content", "document"):
        if c in cols:
            return {"mode": "text", "text_field": c, "confidence": "medium"}
    return {"mode": "auto", "confidence": "low"}
