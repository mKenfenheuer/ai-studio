"""Hugging Face Hub integration.

Beyond proxying search, this module answers the question a beginner actually
has -- "will this model run on my machine?" -- by estimating memory needs from
the model's real file sizes and comparing them against each runner's VRAM.
"""
from __future__ import annotations

import re
from typing import Any

import httpx

from common import formatting

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
        "detected_format": detect_format(features, rows),
    }


def detect_format(columns: list[str], rows: list[dict] | None = None) -> dict:
    """Guess how a dataset is laid out. Delegates to the shared rules so the
    preview and the trainer always agree about what a row means."""
    return formatting.detect_format(columns, rows)


async def dataset_configs(dataset_id: str) -> dict:
    """Which configurations and splits this dataset offers.

    Many datasets on the Hub are really several datasets sharing a name, and
    `load_dataset` refuses to guess between them:

        Config name is missing. Please pick one among the available configs:
        ['harness_drop_3', 'harness_gsm8k_5', ...]

    Discovering this up front turns a run that fails minutes in into a choice
    made before anything starts.
    """
    by_config: dict[str, list[str]] = {}
    source = "datasets-server"
    try:
        r = await client().get(DATASETS_SERVER + "/splits",
                               params={"dataset": dataset_id})
        r.raise_for_status()
        for row in r.json().get("splits", []):
            by_config.setdefault(row["config"], []).append(row["split"])
    except httpx.HTTPError:
        # The dataset viewer does not cover every dataset -- it answers 501 for
        # anything it cannot index, which includes most of the leaderboard
        # result dumps. The repository's own card still lists the
        # configurations, so fall back to that rather than giving up and
        # letting the runner discover the problem an hour later.
        by_config = await _configs_from_card(dataset_id)
        source = "dataset card"

    if not by_config:
        return {"available": False, "configs": [],
                "reason": "Could not discover this dataset's configurations."}

    configs = [{"name": name, "splits": sorted(set(splits))}
               for name, splits in by_config.items()]
    return {
        "available": bool(configs),
        "dataset": dataset_id,
        "source": source,
        "configs": configs,
        # A dataset with exactly one configuration needs no decision from the
        # user, and the UI hides the choice in that case.
        "needs_choice": len(configs) > 1,
        "default_config": _default_config(configs),
    }


async def _configs_from_card(dataset_id: str) -> dict[str, list[str]]:
    """Configurations as declared in the dataset repository's own card."""
    try:
        r = await client().get("%s/datasets/%s" % (HF_API, dataset_id))
        r.raise_for_status()
        declared = ((r.json().get("cardData") or {}).get("configs")) or []
    except (httpx.HTTPError, ValueError):
        return {}

    out: dict[str, list[str]] = {}
    for entry in declared:
        if not isinstance(entry, dict):
            continue
        name = entry.get("config_name")
        if not name:
            continue
        splits = [f.get("split") for f in (entry.get("data_files") or [])
                  if isinstance(f, dict) and f.get("split")]
        out[name] = splits or ["train"]
    return out


def _default_config(configs: list[dict]) -> str | None:
    """The configuration a person would pick if forced to guess."""
    if not configs:
        return None
    names = [c["name"] for c in configs]
    for preferred in ("default", "main", "all", "en"):
        if preferred in names:
            return preferred
    return names[0]


def _pick_split(splits: list[str], wanted: str = "train") -> str:
    if wanted in splits:
        return wanted
    for fallback in ("train", "training", "validation", "test"):
        if fallback in splits:
            return fallback
    return splits[0] if splits else "train"


# A hard ceiling on what one example may send, so a pathological row
# cannot turn a preview into a multi-megabyte response. Well above
# anything real: the largest seen so far is a 13k-character tool-calling
# conversation.
MAX_PREVIEW_CHARS = 60_000


def _resolved_fields(rows: list[dict], fmt: dict) -> dict:
    """A sample of what the current mapping pulls out of the data."""
    msgs = []
    for row in rows[:3]:
        msgs += formatting.find_messages(row, fmt.get("messages_field"),
                                         fmt.get("selectors"))
    if not msgs:
        return {}
    calls = [c for m in msgs for c in m["tool_calls"]]
    return {
        "roles": list(dict.fromkeys(m["role"] for m in msgs)),
        "turns": len(msgs),
        "with_reasoning": sum(1 for m in msgs if m["reasoning"]),
        "tool_calls": [{"name": c["name"], "arguments": (c["arguments"] or "")[:120]}
                       for c in calls[:3]],
        "tool_results": [{"name": m["name"], "content": m["content"][:120]}
                         for m in msgs if m["role"] in ("tool", "function")][:3],
        "empty_content": sum(1 for m in msgs if not m["content"].strip()
                             and not m["tool_calls"]),
    }


def _row_is_blank(row: dict) -> bool:
    """Whether a row genuinely holds nothing.

    Checked across every type, not just strings. A conversation row carries all
    of its content inside a list, so a string-only test declared every row of a
    tool-calling dataset empty -- and then reported a broken template as "five
    blank rows" instead of as the template error it was.
    """
    for value in row.values():
        if value is None:
            continue
        if isinstance(value, str):
            if value.strip():
                return False
        elif isinstance(value, (list, tuple, dict, set)):
            if value:
                return False
        else:
            return False
    return True


def _clip(text: str) -> str:
    if len(text) <= MAX_PREVIEW_CHARS:
        return text
    return text[:MAX_PREVIEW_CHARS] + "\n\n... truncated ..."


async def training_preview(dataset_id: str, config_name: str | None,
                           split: str, fmt: dict | None,
                           text_field: str | None = None,
                           base_model: str | None = None) -> dict:
    """Show the exact strings the model will be trained on.

    Not the raw columns -- the rendered result, after the instruction template
    or the chat flattening has been applied. Reading the wrong column, or
    reading the right column in the wrong shape, is the most expensive mistake
    available here, and it is invisible until you look at the finished text.
    """
    base = await dataset_preview(dataset_id, config_name, split)
    if not base.get("available"):
        return base

    rows = base["rows"]
    resolved = dict(fmt or base["detected_format"])
    if text_field:
        resolved = {"mode": "text", "text_field": text_field}

    # Fill in the model's own template when the caller asked for it. Done here
    # rather than in the browser so the preview and the runner start from the
    # same source, and so a model without one is reported honestly instead of
    # silently falling back.
    if resolved.get("use_model_template") and base_model and \
            not resolved.get("chat_template"):
        found = await model_chat_template(base_model)
        if found.get("available"):
            resolved["chat_template"] = found["chat_template"]
            resolved["specials"] = found.get("specials") or {}
            base["template_source"] = "model"
        else:
            base["template_source"] = "builtin"
            base["template_note"] = found.get("reason")
    elif resolved.get("chat_format"):
        base["template_source"] = resolved["chat_format"]
    elif resolved.get("template"):
        base["template_source"] = "custom"
    elif resolved.get("chat_template"):
        base["template_source"] = "custom"
    else:
        base["template_source"] = "builtin"

    # Three outcomes, not two. A row that renders is fine; a row that is blank
    # in the source is also fine -- line-oriented corpora like WikiText are
    # full of empty lines and the trainer simply skips them. Only a row with
    # real content that the chosen columns cannot reach is a mistake, and
    # conflating the last two would raise a false alarm on half of WikiText.
    rendered = []
    template_error = None
    for row in rows:
        try:
            text = formatting.format_example(row, resolved)
        except formatting.TemplateError as e:
            # A broken template is the user's own mistake and needs saying
            # plainly, not swallowing into "0 readable rows".
            template_error = str(e)
            text = None
        blank = _row_is_blank(row)
        rendered.append({
            "status": "ok" if text else ("empty" if blank else "unreadable"),
            "ok": bool(text),
            # Sent whole. The browser shortens it for display and offers
            # to show the rest, because deciding server-side which part
            # matters gets it wrong: a tool-calling conversation hides
            # its tool schema ten thousand characters into a system
            # prompt, and any fixed window cuts away exactly what the
            # user opened this to check.
            "text": _clip(text or ""),
            "length": len(text or ""),
        })

    # Show rows that have something in them first: an empty leading row would
    # otherwise make a perfectly good dataset look broken.
    shown = [r for r in rendered if r["status"] == "ok"][:3]
    shown += [r for r in rendered if r["status"] == "unreadable"][:2]
    if not shown:
        shown = rendered[:3]

    # The template is echoed back so the editor can show what actually ran,
    # but never the special tokens dict -- that is noise in a text box.
    echo = dict(resolved)
    echo.pop("specials", None)
    base["format"] = echo
    # Offered back in the playground later. A model fine-tuned with a system
    # prompt behaves quite differently without one, and the prompt it learned
    # is rarely something anybody writes down.
    base["system_prompts"] = formatting.system_prompts(rows, resolved)
    # What the field mapping actually found, on real rows. A selector you
    # cannot see the result of is a guess with extra steps.
    base["resolved"] = _resolved_fields(rows, resolved)
    base["style"] = formatting.conversation_style(resolved)
    base["template_error"] = template_error
    base["messages_field"] = resolved.get("messages_field")
    base["tools_field"] = resolved.get("tools_field")
    base["rendered"] = shown
    base["counts"] = {
        "ok": sum(1 for r in rendered if r["status"] == "ok"),
        "empty": sum(1 for r in rendered if r["status"] == "empty"),
        "unreadable": sum(1 for r in rendered if r["status"] == "unreadable"),
        "sampled": len(rendered),
    }
    base["readable"] = base["counts"]["ok"]
    base["suggested_text_field"] = formatting.pick_text_column(
        base["columns"], rows)
    return base


# A model's chat template is the single most important thing to get right when
# fine-tuning an instruct model, and it is not something a person should have
# to know. Every model that has one ships it in tokenizer_config.json, as a
# Jinja string -- the same Jinja this app renders. So the correct default is
# simply to use the model's own, and to show it working before training starts.
_TEMPLATE_CACHE: dict[str, dict] = {}


def _token_text(value) -> str | None:
    """Special tokens are sometimes a string and sometimes an AddedToken dict."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return value.get("content")
    return None


async def model_chat_template(model_id: str) -> dict:
    """The chat template a model ships, plus the special tokens it references."""
    if model_id in _TEMPLATE_CACHE:
        return _TEMPLATE_CACHE[model_id]

    url = "https://huggingface.co/%s/resolve/main/tokenizer_config.json" % model_id
    try:
        r = await client().get(url)
        r.raise_for_status()
        conf = r.json()
    except (httpx.HTTPError, ValueError) as e:
        return {"available": False, "reason": str(e)[:200], "model": model_id}

    template = conf.get("chat_template")
    # Some repositories ship several named templates (a default and a
    # tool-using one). Prefer the default, and say which was taken.
    name = None
    if isinstance(template, list):
        entries = {t.get("name"): t.get("template") for t in template
                   if isinstance(t, dict)}
        name = "default" if "default" in entries else next(iter(entries), None)
        template = entries.get(name)

    specials = {}
    for key in ("bos_token", "eos_token", "pad_token", "unk_token"):
        if (text := _token_text(conf.get(key))) is not None:
            specials[key] = text

    out = {
        "available": bool(template),
        "model": model_id,
        "chat_template": template,
        "template_name": name,
        "specials": specials,
        "reason": None if template else
                  "This model does not ship a chat template, which usually "
                  "means it is a base model rather than an instruct one.",
    }
    _TEMPLATE_CACHE[model_id] = out
    return out
