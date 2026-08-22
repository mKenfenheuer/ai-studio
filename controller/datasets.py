"""The studio's own dataset library: import, inspect, repair, split, reuse.

Everything in this app so far pointed at a dataset on the Hub and hoped. That
covers the tutorial case and nothing after it, because the actual work of
training a useful model is mostly work on the data:

* the dataset is nearly right, but a tenth of its rows are empty
* it contains the same example four hundred times
* two thirds of it is longer than the context you can afford, so it trains on
  truncated fragments
* you want a validation slice that the training set has never seen
* you have a file on your laptop and no interest in publishing it first

Each of those is a small operation and none of them is available anywhere in
the app. So this module keeps datasets locally as JSONL, describes them
honestly, and applies transformations that always write a NEW dataset -- the
input is never modified, so an experiment that goes wrong costs nothing and
the lineage of what produced what is recorded.

JSONL rather than Arrow or Parquet: it streams, it appends, it survives being
half-written, `datasets` loads it directly on the runner, and a person can
open it in an editor and see their own data. Every one of those matters more
here than the read speed of a columnar format.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Callable, Iterator

import httpx

from common import formatting

from . import config, db

DATASET_DIR = config.DATA_DIR / "datasets"
DATASETS_SERVER = "https://datasets-server.huggingface.co"

# A ceiling on what one dataset may hold. Not a technical limit -- JSONL will
# happily grow past it -- but the controller has no GPU, no worker pool and is
# expected to stay responsive, and a 40 GB import through a JSON parser would
# take the UI down with it. Raise it deliberately if you mean to.
MAX_ROWS = int(__import__("os").environ.get("AI_STUDIO_MAX_DATASET_ROWS", "2000000"))
IMPORT_PAGE = 100          # the datasets-server maximum per request
PREVIEW_ROWS = 8


def path_for(dataset_id: str) -> Path:
    return DATASET_DIR / dataset_id / "data.jsonl"


def ensure_dirs() -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)


def iter_rows(dataset_id: str, limit: int | None = None) -> Iterator[dict]:
    """Rows of a stored dataset, skipping any line that will not parse.

    A single corrupt line should not make a dataset unreadable; it should make
    that row missing. Anything that cares about the difference counts what it
    skipped and says so.
    """
    p = path_for(dataset_id)
    if not p.exists():
        return
    with p.open("r", encoding="utf-8") as fh:
        for n, line in enumerate(fh):
            if limit is not None and n >= limit:
                return
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict):
                yield row


def write_rows(dataset_id: str, rows: Iterator[dict]) -> tuple[int, int, list[str]]:
    """Write rows out, returning (count, bytes, columns in first-seen order)."""
    p = path_for(dataset_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    seen = set()
    n = 0
    with p.open("w", encoding="utf-8") as fh:
        for row in rows:
            if n >= MAX_ROWS:
                break
            for k in row:
                if k not in seen:
                    seen.add(k)
                    columns.append(k)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n, p.stat().st_size if p.exists() else 0, columns


def delete_files(dataset_id: str) -> None:
    import shutil
    shutil.rmtree(DATASET_DIR / dataset_id, ignore_errors=True)


def register(owner_id: str | None, name: str, source: str, rows: Iterator[dict],
             **fields: Any) -> dict:
    """Create a dataset from an iterator of rows, and describe what arrived."""
    ensure_dirs()
    did = db.create_dataset(owner_id, name, source, **fields)
    count, size, columns = write_rows(did, rows)
    sample = list(iter_rows(did, PREVIEW_ROWS * 4))
    fmt = fields.get("format") or (
        formatting.detect_format(columns, sample) if sample else {})
    db.update_dataset(did, rows=count, bytes=size, columns=columns, format=fmt)
    return db.get_dataset(did)


# ---------------------------------------------------------------------------
# Getting data in
# ---------------------------------------------------------------------------

def rows_from_upload(filename: str, blob: bytes) -> Iterator[dict]:
    """Parse an uploaded file into rows, guessing the format from its shape.

    JSONL, a JSON array, a single JSON object, or CSV/TSV. Guessed from the
    content rather than the extension, because files arrive named .txt and
    .json interchangeably and the content is the thing that decides.
    """
    text = blob.decode("utf-8-sig", errors="replace")
    stripped = text.lstrip()
    name = (filename or "").lower()

    if stripped.startswith("["):
        data = json.loads(text)
        for row in data:
            yield row if isinstance(row, dict) else {"text": str(row)}
        return

    if stripped.startswith("{"):
        # Either JSONL, or one object that happens to be the whole file.
        first, _, rest = stripped.partition("\n")
        if rest.strip():
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict):
                    yield row
            return
        obj = json.loads(text)
        if isinstance(obj, dict):
            # A dict of lists is the other common export shape.
            lists = {k: v for k, v in obj.items() if isinstance(v, list)}
            if lists and len(lists) == len(obj):
                n = min(len(v) for v in lists.values())
                for i in range(n):
                    yield {k: v[i] for k, v in lists.items()}
                return
            yield obj
        return

    if name.endswith((".csv", ".tsv")) or "," in stripped.split("\n")[0] \
            or "\t" in stripped.split("\n")[0]:
        delim = "\t" if (name.endswith(".tsv") or "\t" in stripped.split("\n")[0]) else ","
        for row in csv.DictReader(io.StringIO(text), delimiter=delim):
            yield {k: v for k, v in row.items() if k}
        return

    # Plain text: one line per row, which is what a corpus file usually is.
    for line in text.splitlines():
        if line.strip():
            yield {"text": line}


async def rows_from_hub(dataset_id: str, config_name: str | None, split: str,
                        limit: int, token: str | None = None,
                        progress: Callable[[int], None] | None = None) -> list[dict]:
    """Pull rows out of a Hub dataset through the datasets-server.

    Paged at 100 rows a request, which is that service's maximum. Slow for
    large imports and it needs no `datasets` install, no Arrow, and no
    multi-gigabyte download to take the first ten thousand rows of something
    -- which is what an import here is nearly always for.
    """
    out: list[dict] = []
    headers = {"Authorization": "Bearer %s" % token} if token else {}
    limit = min(int(limit or 1000), MAX_ROWS)
    async with httpx.AsyncClient(timeout=60) as c:
        offset = 0
        while offset < limit:
            r = await c.get("%s/rows" % DATASETS_SERVER, headers=headers, params={
                "dataset": dataset_id, "config": config_name or "default",
                "split": split or "train", "offset": offset,
                "length": min(IMPORT_PAGE, limit - offset)})
            if r.status_code >= 400:
                if out:
                    break       # a partial import is still worth keeping
                raise ValueError(
                    "Hugging Face would not serve rows of this dataset (%d). "
                    "Some datasets have to be downloaded in full by a machine "
                    "with the datasets library; this one may be one of them."
                    % r.status_code)
            batch = r.json().get("rows") or []
            if not batch:
                break
            out += [row.get("row", {}) for row in batch]
            offset += len(batch)
            if progress:
                progress(len(out))
            if len(batch) < IMPORT_PAGE:
                break
    return out


# ---------------------------------------------------------------------------
# Looking at it honestly
# ---------------------------------------------------------------------------

def inspect(dataset: dict, sample: int = 2000) -> dict:
    """What is actually in this data, and what is wrong with it.

    Measured on a sample rather than the whole file, because this runs while
    someone waits. The sample size is reported so nobody reads an estimate as
    a count.
    """
    rows = list(iter_rows(dataset["id"], sample))
    if not rows:
        return {"sampled": 0, "problems": [
            {"level": "error", "message": "This dataset has no readable rows."}]}

    columns = dataset.get("columns") or sorted({k for r in rows for k in r})
    fmt = formatting.resolve_format(dataset.get("format") or {})

    fill: dict[str, int] = {c: 0 for c in columns}
    for r in rows:
        for c in columns:
            v = r.get(c)
            if v not in (None, "", [], {}):
                fill[c] += 1

    lengths: list[int] = []
    empty = 0
    seen: dict[str, int] = {}
    duplicates = 0
    for r in rows:
        text = (formatting.format_example(r, fmt) or "").strip()
        if not text:
            empty += 1
            continue
        lengths.append(len(text))
        h = hashlib.sha1(text.encode("utf-8")).hexdigest()
        if h in seen:
            duplicates += 1
        seen[h] = seen.get(h, 0) + 1

    lengths.sort()

    def pct(p: float) -> int:
        if not lengths:
            return 0
        return lengths[min(len(lengths) - 1, int(len(lengths) * p))]

    stats = {
        "sampled": len(rows),
        "total_rows": dataset.get("rows") or len(rows),
        "columns": [{"name": c, "filled": fill[c],
                     "fill_rate": round(fill[c] / len(rows), 3)} for c in columns],
        "empty_rows": empty,
        "duplicate_rows": duplicates,
        "unique_rows": len(seen),
        "chars": {
            "min": lengths[0] if lengths else 0,
            "p50": pct(0.5), "p90": pct(0.9), "p99": pct(0.99),
            "max": lengths[-1] if lengths else 0,
            "mean": int(sum(lengths) / len(lengths)) if lengths else 0,
        },
        # Roughly four characters per token for English on a small vocabulary.
        # Called an estimate everywhere it is shown, because it is one.
        "est_tokens": int(sum(lengths) / 4) if lengths else 0,
        "histogram": _histogram(lengths),
        "preview": [formatting.format_example(r, fmt) or "" for r in rows[:PREVIEW_ROWS]],
        "format": fmt,
    }
    stats["problems"] = _problems(stats, dataset)
    return stats


def _histogram(lengths: list[int], buckets: int = 12) -> list[dict]:
    if not lengths:
        return []
    lo, hi = lengths[0], lengths[-1]
    if hi <= lo:
        return [{"from": lo, "to": hi, "count": len(lengths)}]
    # Log scale: text length is heavily skewed, and a linear histogram of it is
    # one tall bar on the left and eleven empty ones.
    lo = max(lo, 1)
    step = (math.log10(hi) - math.log10(max(lo, 1))) / buckets
    edges = [10 ** (math.log10(lo) + step * i) for i in range(buckets + 1)]
    out = []
    idx = 0
    for i in range(buckets):
        count = 0
        while idx < len(lengths) and lengths[idx] <= edges[i + 1]:
            count += 1
            idx += 1
        out.append({"from": int(edges[i]), "to": int(edges[i + 1]), "count": count})
    out[-1]["count"] += len(lengths) - idx
    return out


def _problems(stats: dict, dataset: dict) -> list[dict]:
    """The things worth acting on, each with the fix that acts on it."""
    out = []
    n = max(stats["sampled"], 1)

    if stats["empty_rows"]:
        share = stats["empty_rows"] / n
        out.append({
            "level": "error" if share > 0.5 else "warn",
            "fix": "drop_empty",
            "message": "%d of %d sampled rows read as nothing at all (%.0f%%). "
                       "Those rows cost training time and teach nothing."
                       % (stats["empty_rows"], n, share * 100)})

    if stats["duplicate_rows"]:
        share = stats["duplicate_rows"] / n
        out.append({
            "level": "warn" if share > 0.02 else "info",
            "fix": "dedupe",
            "message": "%d of %d sampled rows are exact repeats (%.0f%%). "
                       "Repeats teach a model to recite rather than to "
                       "generalise, and inflate how much data you appear to "
                       "have." % (stats["duplicate_rows"], n, share * 100)})

    thin = [c["name"] for c in stats["columns"] if 0 < c["fill_rate"] < 0.5]
    if thin:
        out.append({
            "level": "info", "fix": None,
            "message": "Mostly empty: %s. If your template reads one of these, "
                       "most rows will come out short." % ", ".join(thin[:4])})

    p90 = stats["chars"]["p90"]
    if p90 > 8000:
        out.append({
            "level": "warn", "fix": "cap_length",
            "message": "A tenth of the rows are longer than %s characters, "
                       "which is far past any context length you can train on "
                       "here. Those rows will be cut off mid-sentence."
                       % f"{p90:,}"})

    if stats["chars"]["p50"] < 40:
        out.append({
            "level": "warn", "fix": None,
            "message": "Half the rows are under 40 characters. Very short "
                       "examples give the model almost nothing to predict."})

    total = dataset.get("rows") or 0
    if total and total < 200:
        out.append({
            "level": "warn", "fix": None,
            "message": "Only %d rows. That is enough to check a pipeline and "
                       "not enough to change a model's behaviour." % total})

    if not out:
        out.append({"level": "ok", "fix": None,
                    "message": "Nothing obviously wrong: no empty rows, no "
                               "exact repeats, and a sane spread of lengths."})
    return out


# ---------------------------------------------------------------------------
# Transformations
# ---------------------------------------------------------------------------
#
# Every one of these reads a dataset and writes a new one. Nothing edits in
# place. That costs disk and buys the ability to undo, to compare, and to see
# what a step actually did -- and the alternative is a destructive operation
# on the only copy of somebody's data.

def transform(dataset: dict, ops: dict, owner_id: str | None,
              name: str | None = None) -> dict:
    fmt = formatting.resolve_format(dataset.get("format") or {})
    steps: list[str] = []

    rows = list(iter_rows(dataset["id"]))
    before = len(rows)

    if ops.get("drop_empty"):
        rows = [r for r in rows
                if (formatting.format_example(r, fmt) or "").strip()]
        steps.append("Dropped rows that render as nothing (%d removed)"
                     % (before - len(rows)))

    if ops.get("dedupe"):
        n0 = len(rows)
        seen: set[str] = set()
        kept = []
        for r in rows:
            h = hashlib.sha1((formatting.format_example(r, fmt) or "")
                             .strip().encode("utf-8")).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            kept.append(r)
        rows = kept
        steps.append("Removed exact duplicates (%d removed)" % (n0 - len(rows)))

    lo = int(ops.get("min_chars") or 0)
    hi = int(ops.get("max_chars") or 0)
    if lo or hi:
        n0 = len(rows)
        def length_ok(r: dict) -> bool:
            t = (formatting.format_example(r, fmt) or "").strip()
            return (not lo or len(t) >= lo) and (not hi or len(t) <= hi)
        rows = [r for r in rows if length_ok(r)]
        steps.append("Kept rows between %s and %s characters (%d removed)"
                     % (lo or 0, hi or "any", n0 - len(rows)))

    if contains := (ops.get("contains") or "").strip():
        n0 = len(rows)
        try:
            pattern = re.compile(contains, re.IGNORECASE)
            rows = [r for r in rows
                    if pattern.search(formatting.format_example(r, fmt) or "")]
            steps.append("Kept rows matching /%s/ (%d removed)" % (contains, n0 - len(rows)))
        except re.error as e:
            raise ValueError("That search pattern is not valid: %s" % e) from e

    if exclude := (ops.get("excludes") or "").strip():
        n0 = len(rows)
        try:
            pattern = re.compile(exclude, re.IGNORECASE)
            rows = [r for r in rows
                    if not pattern.search(formatting.format_example(r, fmt) or "")]
            steps.append("Removed rows matching /%s/ (%d removed)" % (exclude, n0 - len(rows)))
        except re.error as e:
            raise ValueError("That exclusion pattern is not valid: %s" % e) from e

    if keep := ops.get("keep_columns"):
        rows = [{k: r.get(k) for k in keep if k in r} for r in rows]
        steps.append("Kept only the columns %s" % ", ".join(keep))

    if ops.get("shuffle"):
        random.Random(int(ops.get("seed") or 1234)).shuffle(rows)
        steps.append("Shuffled")

    if sample := int(ops.get("sample") or 0):
        if sample < len(rows):
            if not ops.get("shuffle"):
                # Taking the first N of an ordered file is not a sample; many
                # datasets are sorted by source, length or label, and the head
                # of one is a biased slice. Shuffle first, and say so.
                random.Random(int(ops.get("seed") or 1234)).shuffle(rows)
                steps.append("Shuffled before sampling, so the sample is not "
                             "just the front of the file")
            rows = rows[:sample]
            steps.append("Sampled %s rows" % f"{sample:,}")

    if ops.get("to_chat"):
        rows, converted = _to_chat(rows, fmt, ops)
        steps.append("Rewrote %d rows as system/user/assistant turns" % converted)

    if not rows:
        raise ValueError(
            "Those settings would leave the dataset empty. Loosen them and "
            "try again -- nothing has been changed.")

    label = name or "%s (cleaned)" % dataset["name"]
    out_fmt = {"mode": "chat"} if ops.get("to_chat") else dataset.get("format")
    created = register(
        owner_id, label, "derived", iter(rows),
        origin=dataset["id"], parent_id=dataset["id"],
        columns=sorted({k for r in rows[:200] for k in r}),
        format=out_fmt,
        recipe={"from": dataset["id"], "from_name": dataset["name"],
                "steps": steps, "rows_before": before, "rows_after": len(rows)},
        notes=ops.get("notes"))
    return created


def _to_chat(rows: list[dict], fmt: dict, ops: dict) -> tuple[list[dict], int]:
    """Rewrite rows into a single `messages` column.

    Uses exactly the normalisation the trainer and the preview already use, so
    a dataset converted here reads identically to one that arrived in that
    shape. A system prompt can be prepended, which is the usual reason for
    doing this at all.
    """
    system = (ops.get("system_prompt") or "").strip()
    selectors = ops.get("selectors") or fmt.get("selectors")
    out: list[dict] = []
    converted = 0
    for r in rows:
        msgs = formatting.find_messages(r, fmt.get("messages_field"), selectors)
        if not msgs:
            prompt = formatting.first_present(r, ["instruction", "prompt",
                                                  "question", "input"])
            answer = formatting.first_present(r, ["output", "response",
                                                  "answer", "completion"])
            if prompt and answer:
                msgs = [{"role": "user", "content": str(r.get(prompt) or "")},
                        {"role": "assistant", "content": str(r.get(answer) or "")}]
            else:
                text = formatting.format_example(r, fmt) or ""
                if not text.strip():
                    continue
                msgs = [{"role": "assistant", "content": text}]
        clean = [{k: v for k, v in m.items()
                  if k in ("role", "content", "name", "reasoning", "tool_calls") and v}
                 for m in msgs]
        if system and not any(m.get("role") == "system" for m in clean):
            clean.insert(0, {"role": "system", "content": system})
        row = {"messages": clean}
        if tools := r.get("tools"):
            row["tools"] = tools
        out.append(row)
        converted += 1
    return out, converted


def split(dataset: dict, fraction: float, owner_id: str | None,
          seed: int = 1234) -> list[dict]:
    """Cut a dataset into a training part and a held-out part.

    The single most useful thing this module does. Without a held-out slice
    there is no way to tell a model that has learned from one that has
    memorised, and the loss curve looks identical in both cases.
    """
    fraction = min(max(float(fraction or 0.1), 0.001), 0.5)
    rows = list(iter_rows(dataset["id"]))
    if len(rows) < 20:
        raise ValueError("There are too few rows here to split meaningfully.")
    random.Random(seed).shuffle(rows)
    cut = max(1, int(len(rows) * fraction))
    held, train = rows[:cut], rows[cut:]

    made = []
    for label, part in (("train", train), ("validation", held)):
        made.append(register(
            owner_id, "%s (%s)" % (dataset["name"], label), "derived",
            iter(part), origin=dataset["id"], parent_id=dataset["id"],
            columns=dataset.get("columns"), format=dataset.get("format"),
            recipe={"from": dataset["id"], "from_name": dataset["name"],
                    "steps": ["Split %s: %d%% held out, shuffled with seed %d"
                              % (label, round(fraction * 100), seed)],
                    "rows_before": len(rows), "rows_after": len(part)}))
    return made


def merge(datasets: list[dict], owner_id: str | None, name: str,
          shuffle: bool = True, seed: int = 1234) -> dict:
    rows: list[dict] = []
    steps = []
    for d in datasets:
        part = list(iter_rows(d["id"]))
        rows += part
        steps.append("%s rows from %s" % (f"{len(part):,}", d["name"]))
    if not rows:
        raise ValueError("Those datasets are all empty.")
    if shuffle:
        random.Random(seed).shuffle(rows)
        steps.append("Shuffled together")
    return register(
        owner_id, name, "derived", iter(rows),
        columns=sorted({k for r in rows[:200] for k in r}),
        format=datasets[0].get("format"),
        recipe={"steps": steps, "rows_after": len(rows)})
