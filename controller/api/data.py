"""Datasets: bringing them in, looking at them, fixing them, sharing them."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import (APIRouter, Body, File, HTTPException, Query, Request,
                     Response, UploadFile)
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from common import conversation, formatting

from .. import assets
from .. import datasets as ds
from .. import cards, db, hfaccount, hub
from .security import current_user, require_edit, require_owner, require_view

router = APIRouter(prefix="/api/datasets")

# An upload has to fit in memory to be parsed, and the controller is not the
# machine with 64 GB in it. Large corpora belong on the Hub, where they can be
# streamed; this is for the file on somebody's laptop.
MAX_UPLOAD_BYTES = 256 * 1024 * 1024


def _get(request: Request, dataset_id: str, need: str = "view") -> dict:
    d = db.get_dataset(dataset_id)
    if not d:
        raise HTTPException(404, "No such dataset.")
    if need == "edit":
        require_edit(request, "dataset", d)
    elif need == "own":
        require_owner(request, "dataset", d)
    else:
        require_view(request, "dataset", d)
    return d


@router.get("")
async def list_datasets(request: Request) -> list[dict]:
    return db.visible_datasets(current_user(request))


@router.get("/{dataset_id}")
async def get_dataset(request: Request, dataset_id: str) -> dict:
    d = _get(request, dataset_id)
    user = current_user(request)
    d["access"] = db.access_level("dataset", d["id"], d.get("owner_id"), user)
    d["mine"] = d.get("owner_id") == user["id"]
    d["shares"] = db.list_shares("dataset", dataset_id)
    d["preview"] = list(ds.iter_rows(dataset_id, ds.PREVIEW_ROWS))
    if parent := d.get("parent_id"):
        if p := db.get_dataset(parent):
            d["parent_name"] = p["name"]
    return d


@router.get("/{dataset_id}/inspect")
def inspect_dataset(request: Request, dataset_id: str,
                          sample: int = 2000) -> dict:
    return ds.inspect(_get(request, dataset_id), min(int(sample), 20000))


# How far a search will read before it stops looking. A search that scans a
# two-million-row file holds the controller for a minute; one that scans the
# first fifty thousand answers in a moment and says what it did.
SEARCH_SCAN = 50_000


@router.get("/{dataset_id}/rows")
def dataset_rows(request: Request, dataset_id: str, offset: int = 0,
                       limit: int = 25, q: str = "", split: str = "") -> dict:
    """A page of rows, rendered the way training will read them.

    `q` filters to rows containing that text, matched against the rendered
    form rather than the raw JSON -- searching for a word should find it
    whether it lives in `text`, in a message, or three keys deep.
    """
    d = _get(request, dataset_id)
    limit = min(max(int(limit), 1), 200)
    offset = max(int(offset), 0)
    fmt = formatting.resolve_format(d.get("format") or {})
    needle = (q or "").strip().lower()
    want = (split or "").strip() or None
    in_split = (d.get("splits") or {}).get(want) if want else (d.get("rows") or 0)
    rows = []

    if not needle:
        seen = 0
        for i, row in ds.iter_indexed(dataset_id, want):
            seen += 1
            if seen <= offset:
                continue
            rows.append({"index": i, "row": row,
                         "rendered": formatting.format_example(row, fmt) or ""})
            if len(rows) >= limit:
                break
        return {"rows": rows, "offset": offset, "total": d.get("rows") or 0,
                "matched": in_split or 0, "query": "", "split": want or "",
                "columns": d.get("columns") or [],
                "assets": _assets_in(rows)}

    matched = 0
    scanned = 0
    for i, row in ds.iter_indexed(dataset_id, want):
        scanned += 1
        if scanned > SEARCH_SCAN:
            break
        rendered = formatting.format_example(row, fmt) or ""
        if needle not in rendered.lower() \
                and needle not in json.dumps(row, ensure_ascii=False).lower():
            continue
        matched += 1
        if matched > offset and len(rows) < limit:
            rows.append({"index": i, "row": row, "rendered": rendered})
    return {"rows": rows, "offset": offset, "total": d.get("rows") or 0,
            "matched": matched, "query": q, "split": want or "",
            "columns": d.get("columns") or [],
            "scanned": scanned, "capped": scanned >= SEARCH_SCAN,
            "assets": _assets_in(rows)}


def _assets_in(page: list[dict]) -> dict:
    """What each asset on this page is, so the browser can draw it.

    Told rather than guessed: a cell holding `asset:ast_...` says nothing
    about whether it is a photograph or a recording, and the browser cannot
    find out without fetching it. One query for the whole page.
    """
    ids: list[str] = []
    for entry in page:
        ids += assets.ids_in_row(entry.get("row") or {})
    return {a["id"]: {"kind": a["kind"], "mime": a["mime"],
                      "filename": a["filename"], "bytes": a["size_bytes"],
                      "url": "/api/assets/%s" % a["id"]}
            for a in db.get_assets(ids)}


@router.get("/{dataset_id}/conversations")
def dataset_conversations(request: Request, dataset_id: str,
                                offset: int = 0, limit: int = 20,
                                split: str = "") -> dict:
    """Rows as conversations, each cut where a model would have to take over.

    What the playground loads when you ask to try a held-out example. Every row
    comes back three ways at once, because all three are needed and computing
    them separately is how they end up disagreeing:

      `messages`  the whole conversation, canonical
      `prompt`    everything up to and including the last user turn
      `expected`  what the data says comes next -- which for a tool-calling row
                  is a call, a result and a reply, not a single message

    The tool results in `expected` are what lets a conversation be *replayed*:
    when the model asks for `get_order`, the answer the dataset recorded can be
    handed back and the conversation carried on, instead of stopping at the
    first call.
    """
    d = _get(request, dataset_id)
    limit = min(max(int(limit), 1), 100)
    offset = max(int(offset), 0)
    fmt = formatting.resolve_format(d.get("format") or {})
    want = (split or "").strip() or None

    out = []
    seen = 0
    for i, row in ds.iter_indexed(dataset_id, want):
        seen += 1
        if seen <= offset:
            continue
        conv, notes = conversation.repair(conversation.from_row(row, fmt))
        if not conv[conversation.MESSAGES_KEY]:
            continue
        prompt, expected = conversation.split_for_trial(conv)
        out.append({
            "index": i,
            "split": row.get(ds.SPLIT_FIELD) or ds.DEFAULT_SPLIT,
            "messages": conv[conversation.MESSAGES_KEY],
            "tools": conv[conversation.TOOLS_KEY],
            "meta": conv[conversation.META_KEY],
            "prompt": prompt,
            "expected": expected,
            "repaired": notes,
            "problems": conversation.validate(conv),
        })
        if len(out) >= limit:
            break

    counts = d.get("splits") or {}
    return {"rows": out, "offset": offset, "split": want or "",
            "matched": counts.get(want) if want else (d.get("rows") or 0),
            "splits": counts, "dataset": {"id": d["id"], "name": d["name"]}}


@router.get("/{dataset_id}/conversation-report")
def conversation_report(request: Request, dataset_id: str,
                              sample: int = 2000) -> dict:
    """Whether this dataset is usable as conversations, and what is wrong.

    The same validation the conversion runs, offered on a dataset that has
    already been converted -- because a dataset can be edited afterwards, and
    "it was clean when it was made" is not the question anybody is asking.
    """
    d = _get(request, dataset_id)
    fmt = formatting.resolve_format(d.get("format") or {})
    convs = []
    for row in ds.iter_rows(dataset_id, min(int(sample), 20000)):
        conv, _ = conversation.repair(conversation.from_row(row, fmt))
        if conv[conversation.MESSAGES_KEY]:
            convs.append(conv)
    if not convs:
        return {"rows": 0, "ok": False, "problems": [
            {"level": "error", "code": "empty",
             "message": "Nothing in this dataset reads as a conversation."}]}
    report = conversation.validate_many(convs)
    report["canonical"] = sum(
        1 for row in ds.iter_rows(dataset_id, 200)
        if conversation.is_canonical(row))
    report["sampled"] = len(convs)
    return report


@router.get("/{dataset_id}/dataset-file")
async def download_dataset(request: Request, dataset_id: str):
    """The raw JSONL.

    Also how a runner fetches a studio dataset to train on -- it presents the
    join token instead of a session, which the middleware allows for this path.
    """
    if not getattr(request.state, "runner", False):
        _get(request, dataset_id)
    p = ds.path_for(dataset_id)
    if not p.exists():
        raise HTTPException(404, "This dataset has no file.")
    name = (db.get_dataset(dataset_id) or {}).get("name") or dataset_id
    safe = "".join(c for c in name if c.isalnum() or c in "-_ ").strip() or dataset_id
    return FileResponse(p, media_type="application/x-ndjson",
                        filename="%s.jsonl" % safe.replace(" ", "-"))


# ---------------------------------------------------------------------------
# Getting data in
# ---------------------------------------------------------------------------

@router.post("/upload")
async def upload_dataset(request: Request,
                         file: list[UploadFile] = File(...),
                         name: str = Query(default=""),
                         text_split: str = Query(default="auto"),
                         chunk_chars: int = Query(default=ds.DEFAULT_CHUNK_CHARS),
                         overlap: int = Query(default=ds.DEFAULT_OVERLAP),
                         delimiter: str = Query(default=""),
                         header: str = Query(default="auto"),
                         split: str = Query(default=""),
                         into: str = Query(default="")) -> dict:
    """One or many files, in whatever format, as one dataset.

    Many rather than one because data arrives as a folder at least as often as
    it arrives as a file, and uploading eighty transcripts one at a time is
    the point at which people give up and go back to a notebook. Each file's
    rows are tagged with the file they came from when there is more than one.

    `split` is the split these rows belong to -- train unless you say
    otherwise -- and `into` adds them to a dataset that already exists rather
    than making another one. `text_split` is a different thing entirely: how
    a prose file is cut into rows.
    """
    user = current_user(request)
    files = [f for f in file if (f.filename or "").strip() or f.size]
    if not files:
        raise HTTPException(400, "No file was uploaded.")
    options = {"text_split": text_split, "chunk_chars": chunk_chars,
               "overlap": overlap, "delimiter": delimiter, "header": header,
               # Whose store an image or a recording in this upload goes
               # into. The parser has no other way to know.
               "owner_id": user["id"]}
    split = (split or "").strip() or ds.DEFAULT_SPLIT
    target = _get(request, into, "edit") if into else None
    # Every file has to fit the account's quota before any of it is parsed;
    # finding out at file eighty of ninety is the annoying way to find out.
    try:
        assets.check_quota(user["id"], sum((f.size or 0) for f in files
                                          if assets.is_media(f.filename or "")))
    except ValueError as e:
        raise HTTPException(413, str(e)) from e

    rows: list[dict] = []
    failures: list[str] = []
    total = 0
    for f in files:
        blob = await f.read()
        total += len(blob)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(
                413, "That is more than %d MB in one upload. Put a corpus "
                     "this size on the Hub and import it by name instead."
                     % (MAX_UPLOAD_BYTES // (1024 * 1024)))
        try:
            # Parsing a 256 MB CSV is seconds of pure Python. On the event loop
            # that is seconds of the whole studio: nobody else's page loads,
            # no runner heartbeat is read, no training metric is stored.
            rows += list(await run_in_threadpool(
                ds.rows_from_upload,
                f.filename or "", blob, options,
                source=f.filename if len(files) > 1 else None,
                problems=failures))
        except ValueError as e:
            failures.append("%s: %s" % (f.filename or "that file", e))
        except Exception as e:  # noqa: BLE001 - a malformed file is the user's
            failures.append("%s: %s" % (f.filename or "that file", e))

    if not rows:
        raise HTTPException(
            400, "Nothing could be read. " + (" ".join(failures[:3])
                 or "The files contained no rows."))

    if target:
        # Added to a dataset that already exists, which is what "here is the
        # test set for the data I uploaded yesterday" means.
        created = await run_in_threadpool(ds.append_rows, target, rows, split)
        created["skipped"] = failures
        created["added"] = len(rows)
        return created

    label = name or (files[0].filename or "Uploaded data") if len(files) == 1 \
        else (name or "%d uploaded files" % len(files))
    origin = files[0].filename if len(files) == 1 \
        else "%d files" % len(files)
    created = await run_in_threadpool(
        ds.register, user["id"], label, "upload", iter(rows),
        split=split, origin=origin)
    # Not an error and not silence: a folder where two files of ninety could
    # not be read is a successful import with something worth knowing in it.
    created["skipped"] = failures
    return created


@router.post("/import")
async def import_dataset(request: Request, payload: dict = Body(...)) -> dict:
    user = current_user(request)
    hub_id = (payload.get("dataset") or "").strip()
    if not hub_id:
        raise HTTPException(400, "Which dataset on the Hub?")
    # One split or several. "splits" is what the picker sends; "split" is kept
    # so an older client, or a script somebody wrote against this endpoint,
    # keeps working.
    wanted = payload.get("splits") or payload.get("split") or ""
    if isinstance(wanted, str):
        wanted = [s.strip() for s in wanted.split(",") if s.strip()]
    if not wanted:
        # Nothing named means everything there is. Defaulting to "train" was a
        # decision disguised as a default: it silently left the test split --
        # the reason anybody can tell learning from memorising -- behind.
        try:
            found = await hub.dataset_configs(hub_id)
        except Exception:  # noqa: BLE001 - a lookup failure is not fatal here
            found = {}
        config_name = payload.get("config")
        for c in found.get("configs") or []:
            if not config_name or c.get("name") == config_name:
                wanted = list(c.get("splits") or [])
                if not config_name:
                    config_name = c.get("name")
                break
        payload = {**payload, "config": config_name}
        wanted = wanted or ["train"]

    # 0 means every row. The old default of 5,000 quietly truncated datasets
    # to a demo-sized slice, which is fine for a first look and wrong for
    # everything after it.
    limit = payload.get("limit")
    limit = ds.MAX_ROWS if limit in (None, "", 0, "0") else int(limit)
    incomplete: dict = {}
    try:
        rows = await ds.rows_from_hub(
            hub_id, payload.get("config"), wanted,
            limit, hfaccount.token_for(user), incomplete=incomplete)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    if not rows:
        raise HTTPException(400, "That dataset returned no rows.")

    created = await run_in_threadpool(
        ds.register,
        user["id"], payload.get("name") or hub_id.split("/")[-1], "hub",
        iter(rows), origin=hub_id)
    # The note describes what arrived, not what was asked for. Those are the
    # same thing right up until Hugging Face stops serving rows halfway.
    got = created.get("splits") or {}
    note = "Imported %s rows from %s (%s): %s." % (
        f"{created.get('rows', 0):,}", hub_id,
        payload.get("config") or "default",
        ", ".join("%s %s" % (f"{n:,}", name) for name, n in got.items()) or "none")
    if incomplete:
        note += (" Hugging Face stopped serving rows partway through %s, so "
                 "that split is incomplete." % ", ".join(incomplete))
    db.update_dataset(created["id"], notes=note)
    created["notes"] = note
    created["incomplete"] = incomplete
    return created


# ---------------------------------------------------------------------------
# Changing it
# ---------------------------------------------------------------------------

@router.patch("/{dataset_id}")
async def rename_dataset(request: Request, dataset_id: str,
                         payload: dict = Body(...)) -> dict:
    d = _get(request, dataset_id, "edit")
    fields = {}
    if "name" in payload:
        fields["name"] = (payload["name"] or "").strip()[:120] or d["name"]
    if "notes" in payload:
        fields["notes"] = (payload["notes"] or "")[:4000]
    if "format" in payload:
        fields["format"] = payload["format"] or {}
    db.update_dataset(dataset_id, **fields)
    return db.get_dataset(dataset_id)


@router.post("/{dataset_id}/transform")
def transform_dataset(request: Request, dataset_id: str,
                            payload: dict = Body(...)) -> dict:
    d = _get(request, dataset_id)
    user = current_user(request)
    try:
        return ds.transform(d, payload.get("ops") or {}, user["id"],
                            payload.get("name"))
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@router.post("/{dataset_id}/transform/preview")
def preview_transform(request: Request, dataset_id: str,
                            payload: dict = Body(...)) -> dict:
    """What a transform would do, without doing it.

    `ops` is one options dict or an ordered list of them. `sample` is how many
    rows to rehearse on -- 0 or "all" means the whole file, which is the
    caller's choice to make and to wait for. `limit` is how many resulting
    rows to send back, and `upto` which step's output they are taken from.
    """
    d = _get(request, dataset_id)
    raw_sample = payload.get("sample", 2000)
    sample = None if raw_sample in (0, "0", "all", None) else int(raw_sample)
    if sample is not None:
        sample = max(sample, 1)
    upto = payload.get("upto")
    try:
        return ds.preview_transform(
            d, payload.get("ops") or {}, sample,
            limit=min(int(payload.get("limit") or 5), 200),
            upto=None if upto is None else int(upto),
            values=(str(payload.get("values") or "").strip() or None))
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@router.post("/{dataset_id}/rows/add")
def add_rows(request: Request, dataset_id: str,
                   payload: dict = Body(...)) -> dict:
    """Write new rows by hand, into a named split.

    The smallest thing a data tool has to be able to do and the one most
    often missing: you read the rows, you see the example that is missing,
    you add it. Rows arrive as objects with the columns this dataset already
    has, or as free text for a plain-text dataset.
    """
    d = _get(request, dataset_id, "edit")
    rows = payload.get("rows")
    if isinstance(rows, str):
        # A textarea of JSONL, or of plain lines for a text dataset.
        field = (d.get("format") or {}).get("text_field") or "text"
        parsed = []
        for line in rows.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                try:
                    obj = json.loads(line)
                except ValueError as e:
                    raise HTTPException(
                        400, "That is not valid JSON: %s" % e) from e
                if isinstance(obj, dict):
                    parsed.append(obj)
            else:
                parsed.append({field: line})
        rows = parsed
    if not isinstance(rows, list) or not rows:
        raise HTTPException(400, "No rows were given.")
    if not all(isinstance(r, dict) for r in rows):
        raise HTTPException(400, "Every row has to be an object.")
    return ds.append_rows(d, _shaped(d, rows),
                          (payload.get("split") or "").strip()
                          or ds.DEFAULT_SPLIT)


def _shaped(d: dict, rows: list[dict]) -> list[dict]:
    """Conversations, written in the shape this dataset already uses.

    The playground hands a kept exchange over as a canonical conversation --
    a `messages` list -- because that is what it holds and what the trainer
    reads. Most datasets are conversations too and take it unchanged. A flat
    one, with an instruction column and a response column, would take it as a
    brand new `messages` column: a dataset half in one shape and half in
    another, which every later step then has to guess about.

    So it is mapped instead, into the columns the dataset has. A dataset that
    cannot hold a conversation at all -- a corpus of plain prose -- is refused
    rather than half-filled, because "the last thing the assistant said" is
    not a line of a corpus and pretending it is teaches the model nothing.
    """
    if not any(isinstance(r.get("messages"), list) for r in rows):
        return rows
    cols = list(d.get("columns") or [])
    fmt = formatting.resolve_format(d.get("format") or {})
    if not cols or "messages" in cols or fmt.get("mode") == "chat":
        return rows

    instr = fmt.get("instruction_field") or next(
        (c for c in ("instruction", "prompt", "question", "input") if c in cols), "")
    resp = fmt.get("response_field") or next(
        (c for c in ("output", "response", "answer", "completion")
         if c in cols and c != instr), "")
    if not instr or not resp:
        raise HTTPException(
            400, "This dataset's rows are not conversations and have no "
                 "question and answer columns to put one in — its columns "
                 "are: %s. Keep this into a dataset of conversations, or one "
                 "with a prompt column and a reply column." % ", ".join(cols))

    system_col = next((c for c in ("system", "system_prompt") if c in cols), "")
    out = []
    for r in rows:
        msgs = r.get("messages")
        if not isinstance(msgs, list):
            out.append(r)
            continue
        asked = next((m.get("content") for m in reversed(msgs)
                      if m.get("role") == "user"), "")
        said = next((m.get("content") for m in reversed(msgs)
                     if m.get("role") == "assistant"), "")
        if not asked or not said:
            raise HTTPException(
                400, "That exchange has no question and answer to write into "
                     "%s and %s." % (instr, resp))
        row = {instr: asked, resp: said}
        if system_col:
            if sys_msg := next((m.get("content") for m in msgs
                                if m.get("role") == "system"), ""):
                row[system_col] = sys_msg
        out.append(row)
    return out


@router.post("/{dataset_id}/rows/edit")
def edit_rows(request: Request, dataset_id: str,
                    payload: dict = Body(...)) -> dict:
    """Delete rows, move them to another split, or rewrite one.

    Changes the dataset in place, which nothing else here does. Curating data
    is the work, and requiring a derived copy to remove four bad rows is how a
    library fills up with near-identical datasets nobody can tell apart.
    """
    d = _get(request, dataset_id, "edit")
    try:
        return ds.edit_rows(d, delete=payload.get("delete"),
                            move=payload.get("move"),
                            update=payload.get("update"))
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@router.post("/{dataset_id}/split")
def split_dataset(request: Request, dataset_id: str,
                        payload: dict = Body(...)) -> dict:
    """Hold part of a dataset back, as a validation split of one new dataset.

    Returns the one dataset it makes. It used to return a list of two, which
    is why this is worth a note: a caller written against the old shape gets
    an object where it expected an array, which is a visible failure rather
    than a silent one.
    """
    d = _get(request, dataset_id)
    user = current_user(request)
    try:
        return ds.split(d, float(payload.get("fraction") or 0.1), user["id"],
                        int(payload.get("seed") or 1234),
                        stratify=(payload.get("stratify") or "").strip() or None,
                        name=(payload.get("name") or "").strip() or None)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@router.post("/merge")
def merge_datasets(request: Request, payload: dict = Body(...)) -> dict:
    user = current_user(request)
    ids = payload.get("datasets") or []
    if len(ids) < 2:
        raise HTTPException(400, "Choose at least two datasets to merge.")
    parts = []
    for i in ids:
        d = db.get_dataset(i)
        if not d:
            raise HTTPException(404, "No such dataset: %s" % i)
        require_view(request, "dataset", d)
        parts.append(d)
    try:
        return ds.merge(parts, user["id"],
                        payload.get("name") or "Merged dataset",
                        bool(payload.get("shuffle", True)))
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@router.delete("/{dataset_id}")
async def delete_dataset(request: Request, dataset_id: str) -> dict:
    _get(request, dataset_id, "own")     # permission check; the row is not needed
    children = db.q("SELECT id, name FROM datasets WHERE parent_id=?", (dataset_id,))
    # The files this dataset brought with it. Only the bytes nothing else
    # points at actually go: a dataset derived from this one holds its own
    # references to the same images, so deleting the original leaves them
    # working. See controller/assets.py.
    freed = assets.release_dataset(dataset_id)
    ds.delete_files(dataset_id)
    db.clear_shares("dataset", dataset_id)
    db.delete_dataset(dataset_id)
    notes = []
    if children:
        notes.append("%d dataset(s) made from this one were kept."
                     % len(children))
    if freed["rows"]:
        notes.append("%d stored file(s) released, %s freed."
                     % (freed["rows"], _size(freed["bytes"])))
    return {"ok": True, "note": " ".join(notes) or None}


def _size(n: int) -> str:
    value = float(n or 0)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return "%.1f %s" % (value, unit)
        value /= 1024
    return "%.1f GB" % value


@router.post("/{dataset_id}/publish")
async def publish_dataset(request: Request, dataset_id: str,
                          payload: dict = Body(...)) -> dict:
    """Queue an upload of this dataset to the user's Hugging Face account.

    Queued rather than done here: see the note on the model publish endpoint.
    A dataset is usually small enough to have survived being uploaded inside
    the request, but "small enough today" is not a design, and one route for
    publishing means one place where replacing an existing repo is defined.
    """
    from ..app import _create_job         # local: avoids an import cycle

    d = _get(request, dataset_id)
    repo_id = (payload.get("repo_id") or "").strip()
    jid = await _create_job(request, {
        "name": "Publishing %s" % (repo_id or d["name"]),
        "kind": "upload",
        "config": {
            "target": "dataset", "studio_dataset": dataset_id,
            "repo_id": repo_id,
            "private": bool(payload.get("private", True)),
            "replace": bool(payload.get("replace")),
            "message": payload.get("message") or "",
            "card": cards.dataset_card(d, repo_id),
        }})
    return {"job_id": jid, "repo_id": repo_id,
            "url": "https://huggingface.co/datasets/%s" % repo_id}


@router.get("/{dataset_id}/runs")
def dataset_runs(request: Request, dataset_id: str) -> list[dict]:
    """Every run that trained on this dataset, that the caller may see.

    The fact was always in each run's config; there was no way to ask it
    from the dataset's side, which is the side you are on when deciding
    whether a dataset is worth keeping.
    """
    _get(request, dataset_id)
    user = current_user(request)
    out = []
    for j in db.jobs_trained_on(dataset_id):
        if not db.access_level("job", j["id"], j.get("owner_id"), user):
            continue
        try:
            summary = json.loads(j.get("summary") or "null") or {}
        except (TypeError, ValueError):
            summary = {}
        try:
            cfg = json.loads(j.get("config") or "{}") or {}
        except (TypeError, ValueError):
            cfg = {}
        pm = summary.get("primary_metric") if isinstance(summary, dict) else None
        out.append({"id": j["id"], "name": j["name"], "kind": j["kind"],
                    "status": j["status"], "created_at": j["created_at"],
                    "finished_at": j.get("finished_at"),
                    "base_model": cfg.get("base_model_label") or cfg.get("base_model"),
                    "primary_metric": pm or (
                        {"label": "Held-out loss", "value": summary.get("best_val_loss"),
                         "lower_better": True}
                        if isinstance(summary, dict) and summary.get("best_val_loss") is not None
                        else None)})
    return out
