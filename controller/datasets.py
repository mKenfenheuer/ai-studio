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
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import httpx

from common import conversation, formatting

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

# Splits live in a reserved column rather than in separate files.
#
# A dataset here is one JSONL file, and that is worth keeping: it streams, it
# appends, the runner loads it directly, and a person can open it. Splitting it
# into a file per split would buy nothing that a column does not, and would
# cost every reader of a dataset a directory walk. So `split` is a column, the
# counts are recorded when the file is written, and everything that reads rows
# can filter on it.
#
# The name is reserved: a column called "split" that arrives in somebody's own
# data means exactly what it says here, which is the least surprising rule
# available.
SPLIT_FIELD = "split"
DEFAULT_SPLIT = "train"


def stamp_split(rows: Iterator[dict], split: str | None) -> Iterator[dict]:
    """Put rows into a named split, unless they already name their own."""
    if not split:
        yield from rows
        return
    for row in rows:
        yield row if row.get(SPLIT_FIELD) else {**row, SPLIT_FIELD: split}


def path_for(dataset_id: str) -> Path:
    return DATASET_DIR / dataset_id / "data.jsonl"


def ensure_dirs() -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)


def iter_rows(dataset_id: str, limit: int | None = None,
              split: str | None = None) -> Iterator[dict]:
    """Rows of a stored dataset, skipping any line that will not parse.

    A single corrupt line should not make a dataset unreadable; it should make
    that row missing. Anything that cares about the difference counts what it
    skipped and says so.

    `limit` counts rows *yielded*, not lines read, so asking for ten rows of
    the validation split gives ten rows of the validation split.
    """
    p = path_for(dataset_id)
    if not p.exists():
        return
    yielded = 0
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            if limit is not None and yielded >= limit:
                return
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if not isinstance(row, dict):
                continue
            if split and (row.get(SPLIT_FIELD) or DEFAULT_SPLIT) != split:
                continue
            yielded += 1
            yield row


def iter_indexed(dataset_id: str,
                 split: str | None = None) -> Iterator[tuple[int, dict]]:
    """(position, row), where position counts every row in the file.

    The position is what identifies a row for editing, so it has to mean the
    same thing whether or not a split filter is on. Numbering within the
    filtered view instead would make "delete row 3 of the test split" delete
    the third row of the file, which is somebody else's data.
    """
    for i, row in enumerate(iter_rows(dataset_id)):
        if split and (row.get(SPLIT_FIELD) or DEFAULT_SPLIT) != split:
            continue
        yield i, row


def write_rows(dataset_id: str, rows: Iterator[dict]) -> dict:
    """Write rows out, describing what was written.

    The split counts are taken here, while every row is already in hand.
    Counting them afterwards would mean reading the whole file again on every
    page that wants to say how big the validation slice is.
    """
    p = path_for(dataset_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    seen = set()
    splits: dict[str, int] = {}
    n = 0
    with p.open("w", encoding="utf-8") as fh:
        for row in rows:
            if n >= MAX_ROWS:
                break
            for k in row:
                if k not in seen:
                    seen.add(k)
                    columns.append(k)
            name = str(row.get(SPLIT_FIELD) or DEFAULT_SPLIT)
            splits[name] = splits.get(name, 0) + 1
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return {"rows": n, "bytes": p.stat().st_size if p.exists() else 0,
            "columns": columns, "splits": splits}


def delete_files(dataset_id: str) -> None:
    import shutil
    shutil.rmtree(DATASET_DIR / dataset_id, ignore_errors=True)


def register(owner_id: str | None, name: str, source: str, rows: Iterator[dict],
             split: str | None = None, **fields: Any) -> dict:
    """Create a dataset from an iterator of rows, and describe what arrived.

    `split` names the split rows go into when they do not already carry one --
    which is how the file somebody uploads becomes the validation set.
    """
    ensure_dirs()
    did = db.create_dataset(owner_id, name, source, **fields)
    written = write_rows(did, stamp_split(rows, split))
    sample = list(iter_rows(did, PREVIEW_ROWS * 4))
    fmt = fields.get("format") or (
        formatting.detect_format(written["columns"], sample) if sample else {})
    db.update_dataset(did, rows=written["rows"], bytes=written["bytes"],
                      columns=written["columns"], format=fmt,
                      splits=written["splits"])
    return db.get_dataset(did)


def append_rows(dataset: dict, rows: list[dict], split: str) -> dict:
    """Add rows to a dataset that already exists, in a named split.

    The one operation here that is not copy-on-write, and deliberately so:
    adding the test set to the dataset it belongs with is not a
    transformation of that dataset, it is the rest of it. Written by appending
    to the same file, then re-counted.
    """
    ensure_dirs()
    path = path_for(dataset["id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dataset.get("columns") or [])
    splits = dict(dataset.get("splits") or {})
    added = 0
    with path.open("a", encoding="utf-8") as fh:
        for row in stamp_split(iter(rows), split or DEFAULT_SPLIT):
            for k in row:
                if k not in columns:
                    columns.append(k)
            name = str(row.get(SPLIT_FIELD) or DEFAULT_SPLIT)
            splits[name] = splits.get(name, 0) + 1
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            added += 1
    total = sum(splits.values())
    db.update_dataset(dataset["id"], rows=total, columns=columns,
                      splits=splits,
                      bytes=path.stat().st_size if path.exists() else 0)
    return db.get_dataset(dataset["id"])


def edit_rows(dataset: dict, *, delete: list[int] | None = None,
              move: dict | None = None, update: dict | None = None) -> dict:
    """Delete rows, move them between splits, or rewrite one.

    The other exception to copy-on-write, and a deliberate one. Curating data
    is the work: a row that is wrong is not a transformation of the dataset
    into a new dataset, it is a mistake to remove. Making somebody derive a
    fresh copy to delete four bad rows is how people end up with eleven
    datasets and no idea which is current.

    What it is not is silent. Every edit is appended to the dataset's own
    history, which the page shows, and the file is rewritten atomically -- a
    half-written dataset is worse than any edit is useful.
    """
    drop = set(int(i) for i in (delete or []))
    move_to = (move or {}).get("to")
    move_set = set(int(i) for i in (move or {}).get("indices") or [])
    updates = {int(k): v for k, v in (update or {}).items()
               if isinstance(v, dict)}

    path = path_for(dataset["id"])
    if not path.exists():
        raise ValueError("This dataset has no file to edit.")
    tmp = path.with_suffix(".rewriting")

    columns: list[str] = []
    splits: dict[str, int] = {}
    kept = removed = moved = changed = 0
    with tmp.open("w", encoding="utf-8") as out:
        for i, row in enumerate(iter_rows(dataset["id"])):
            if i in drop:
                removed += 1
                continue
            if i in updates:
                row = updates[i]
                changed += 1
            if move_to and i in move_set:
                row = {**row, SPLIT_FIELD: move_to}
                moved += 1
            for k in row:
                if k not in columns:
                    columns.append(k)
            name = str(row.get(SPLIT_FIELD) or DEFAULT_SPLIT)
            splits[name] = splits.get(name, 0) + 1
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1

    if not kept:
        tmp.unlink(missing_ok=True)
        raise ValueError(
            "That would delete every row. Delete the dataset itself if that "
            "is what you mean -- nothing has been changed.")
    tmp.replace(path)

    said = []
    if removed:
        said.append("removed %d row%s" % (removed, "" if removed == 1 else "s"))
    if moved:
        said.append("moved %d row%s to %s"
                    % (moved, "" if moved == 1 else "s", move_to))
    if changed:
        said.append("edited %d row%s" % (changed, "" if changed == 1 else "s"))
    recipe = dict(dataset.get("recipe") or {})
    recipe.setdefault("edits", []).append(
        {"at": __import__("time").time(), "what": ", ".join(said) or "no change"})
    db.update_dataset(dataset["id"], rows=kept, columns=columns, splits=splits,
                      recipe=recipe,
                      bytes=path.stat().st_size if path.exists() else 0)
    out_row = db.get_dataset(dataset["id"])
    out_row["changed"] = ", ".join(said) or "nothing changed"
    return out_row


# ---------------------------------------------------------------------------
# Getting data in
# ---------------------------------------------------------------------------
#
# A file becomes rows. Which rows depends on what the file is, and the shape
# of a text file is genuinely ambiguous: a .txt may be a corpus of one-line
# examples, a book, or a hundred documents concatenated. Guessing silently is
# how somebody ends up training on 40,000 rows that are each four words long,
# so the split is a setting -- with a default that reads the file and picks.

# How a plain-text file is cut into rows. Called a "mode" and not a "split"
# on purpose: a dataset's splits are train/validation/test, and one word for
# both would eventually put a file's rows into a split called "paragraphs".
TEXT_MODES = {
    "auto": "Work it out from the file",
    "lines": "One row per line",
    "paragraphs": "One row per paragraph",
    "document": "The whole file as one row",
    "chunks": "Fixed-size chunks",
}
DEFAULT_CHUNK_CHARS = 2000
DEFAULT_OVERLAP = 200
# The most of a chunk that may be given up to end on a sentence or a paragraph
# instead of mid-word. Searching further back than this finds a break near the
# *start* of the chunk and emits a fragment instead of a chunk -- which is how
# a 1,000-character text first turned into fifty rows here.
_BOUNDARY_FRACTION = 0.25
_BOUNDARY_MAX = 400


def _looks_like(name: str, blob: bytes) -> str:
    """What kind of file this is, from its first bytes and then its name.

    Magic bytes first, because the container formats -- zip, docx, pdf,
    parquet -- are unreadable as text and produce spectacular nonsense if
    treated as such. Only then the extension, and only then the content.
    """
    head = blob[:8]
    if head.startswith(b"PK\x03\x04"):
        if name.endswith(".docx"):
            return "docx"
        if name.endswith((".xlsx", ".xlsm")):
            return "xlsx"
        return "zip"
    if head.startswith(b"%PDF"):
        return "pdf"
    if head.startswith(b"PAR1"):
        return "parquet"
    if name.endswith((".htm", ".html", ".xhtml")):
        return "html"
    if name.endswith((".csv", ".tsv")):
        return "csv"
    if name.endswith((".jsonl", ".ndjson")):
        return "jsonl"
    if name.endswith(".json"):
        return "json"

    stripped = blob[:4096].decode("utf-8-sig", errors="replace").lstrip()
    if stripped.startswith("["):
        return "json"
    if stripped.startswith("{"):
        return "json"
    if stripped[:200].lstrip().lower().startswith(("<!doctype html", "<html")):
        return "html"
    first = stripped.split("\n", 1)[0]
    if ("," in first or "\t" in first or ";" in first) and _csv_shaped(stripped):
        return "csv"
    return "text"


def _csv_shaped(sample: str) -> bool:
    """Whether the first lines really do have the same field count.

    A sentence with commas in it is not a CSV, and the difference is whether
    the shape repeats. Prose that happens to contain a comma per line would
    otherwise be imported as a one-column table with a heading taken from
    somebody's first sentence.
    """
    try:
        dialect = csv.Sniffer().sniff(sample[:4096], delimiters=",;\t|")
    except csv.Error:
        return False
    lines = [ln for ln in sample.splitlines()[:6] if ln.strip()]
    if len(lines) < 2:
        return False
    counts = {len(next(csv.reader([ln], dialect), [])) for ln in lines}
    return len(counts) == 1 and counts.pop() > 1


# ---- text ------------------------------------------------------------------

def _blocks(text: str) -> list[str]:
    """Paragraphs: runs of text separated by one or more blank lines."""
    return [b.strip() for b in re.split(r"\n\s*\n+", text) if b.strip()]


def _chunk(text: str, size: int, overlap: int) -> Iterator[str]:
    """Fixed-size pieces that try to end where a human would.

    A chunk that stops mid-sentence teaches the model that sentences stop
    mid-sentence, so the break is walked back to the nearest paragraph, then
    sentence, then space -- but only a little way, or a text with no such
    breaks at all would collapse into one enormous chunk.
    """
    size = max(200, int(size or DEFAULT_CHUNK_CHARS))
    overlap = max(0, min(int(overlap or 0), size // 2))
    text = text.strip()
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            back = min(_BOUNDARY_MAX, int(size * _BOUNDARY_FRACTION))
            window_start = max(start + 1, end - back)
            window = text[window_start:end]
            for sep in ("\n\n", ". ", ".\n", "! ", "? ", "\n", " "):
                cut = window.rfind(sep)
                if cut != -1:
                    end = window_start + cut + len(sep)
                    break
        piece = text[start:end].strip()
        if piece:
            yield piece
        if end >= len(text):
            return
        start = max(end - overlap, start + 1)


def _text_rows(text: str, opts: dict, source: str | None = None) -> Iterator[dict]:
    """Rows from prose, cut the way the import asked for."""
    mode = (opts.get("text_split") or "auto").lower()
    if mode not in TEXT_MODES:
        mode = "auto"
    if mode == "auto":
        # Blank lines are a deliberate act in a text file. Where there are
        # several, they are the document's own idea of where a row ends;
        # where there are none, the line is all there is to go on.
        mode = "paragraphs" if len(_blocks(text)) >= 3 else "lines"

    if mode == "document":
        pieces: Iterable[str] = [text.strip()]
    elif mode == "paragraphs":
        pieces = _blocks(text)
    elif mode == "chunks":
        pieces = _chunk(text, opts.get("chunk_chars"), opts.get("overlap"))
    else:
        pieces = (ln.strip() for ln in text.splitlines())

    for piece in pieces:
        if piece and piece.strip():
            yield {"text": piece, **({"source": source} if source else {})}


class _HTMLText(HTMLParser):
    """Tags out, text left, block elements turned into paragraph breaks."""

    _SKIP = {"script", "style", "noscript", "head", "svg"}
    _BREAK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6",
              "section", "article", "blockquote", "pre"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skipping = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skipping += 1
        elif tag in self._BREAK:
            self.parts.append("\n\n")

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skipping:
            self._skipping -= 1
        elif tag in self._BREAK:
            self.parts.append("\n\n")

    def handle_data(self, data):
        if not self._skipping and data.strip():
            self.parts.append(data)

    def text(self) -> str:
        return re.sub(r"\n{3,}", "\n\n", "".join(self.parts)).strip()


def _html_text(raw_text: str) -> str:
    p = _HTMLText()
    p.feed(raw_text)
    p.close()
    return p.text()


def _docx_text(blob: bytes) -> str:
    """Paragraphs out of a .docx, with no dependency.

    A .docx is a zip of XML, and the part that matters is one file inside it.
    Reading it here keeps the controller's install to the four pure-Python
    packages it promises -- the alternative is asking somebody to install a
    Word parser on a machine whose whole point is that it needs nothing.
    """
    import xml.etree.ElementTree as ET
    import zipfile
    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    try:
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            xml = z.read("word/document.xml")
    except (KeyError, zipfile.BadZipFile) as e:
        raise ValueError("this .docx could not be opened (%s)" % e) from e
    root = ET.fromstring(xml)
    paragraphs = []
    for p in root.iter(ns + "p"):
        line = "".join(t.text or "" for t in p.iter(ns + "t")).strip()
        if line:
            paragraphs.append(line)
    return "\n\n".join(paragraphs)


def _pdf_text(blob: bytes) -> str:
    """Text out of a PDF, if this install happens to have a PDF library.

    Not a dependency: PDF text extraction is a large amount of machinery for
    something most people can do once, better, with the tool that made the
    file. If it is not here, the message says exactly that rather than
    pretending the file was unreadable.
    """
    try:
        from pypdf import PdfReader           # type: ignore
    except ImportError:
        try:
            from PyPDF2 import PdfReader      # type: ignore
        except ImportError as e:
            raise ValueError(
                "PDF text extraction is not installed on this controller. "
                "Export the document as .txt, .docx or .html and upload that "
                "-- or `pip install pypdf` on the controller and try again"
            ) from e
    reader = PdfReader(io.BytesIO(blob))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


# ---- structured ------------------------------------------------------------

def _json_rows(text: str) -> Iterator[dict]:
    """JSONL, a JSON array, a dict of columns, or one lone object."""
    stripped = text.lstrip()
    if stripped.startswith("["):
        for row in json.loads(text):
            yield row if isinstance(row, dict) else {"text": str(row)}
        return

    first, _, rest = stripped.partition("\n")
    if rest.strip():
        # More than one line: JSONL, unless it is a pretty-printed object, in
        # which case no line parses and we fall through to loading it whole.
        parsed = 0
        rows: list[dict] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            parsed += 1
            if isinstance(row, dict):
                rows.append(row)
            else:
                rows.append({"text": str(row)})
        if parsed:
            yield from rows
            return

    obj = json.loads(text)
    if isinstance(obj, list):
        for row in obj:
            yield row if isinstance(row, dict) else {"text": str(row)}
        return
    if isinstance(obj, dict):
        # Two shapes wear the same clothes: {"rows": [...]} from an export,
        # and {"question": [...], "answer": [...]} from a dataframe dump.
        for key in ("rows", "data", "examples", "items"):
            if isinstance(obj.get(key), list):
                for row in obj[key]:
                    yield row if isinstance(row, dict) else {"text": str(row)}
                return
        lists = {k: v for k, v in obj.items() if isinstance(v, list)}
        if lists and len(lists) == len(obj):
            n = min(len(v) for v in lists.values())
            for i in range(n):
                yield {k: v[i] for k, v in lists.items()}
            return
        yield obj


def _is_header(cells: list[str]) -> bool:
    """Whether the first row of a table names the columns or is already data.

    `csv.Sniffer.has_header` decides by comparing column *types* down the
    file, and on a table whose every column is text -- which is most of the
    tables anyone trains on -- it says no. Then "question,answer" becomes a
    row of data and the columns are called column_1 and column_2. So instead:
    a heading row is short, non-empty, non-numeric and unique, because that is
    what headings are.
    """
    values = [(c or "").strip() for c in cells]
    if not values or any(not v for v in values):
        return False
    for v in values:
        if len(v) > 64 or "\n" in v:
            return False
        try:
            float(v.replace(",", "."))
            return False        # a number is data, not a name
        except ValueError:
            pass
    # One column is the genuinely ambiguous case: a file of short phrases
    # looks exactly like a heading followed by data. A lone cell therefore has
    # to look like a *name* -- no sentence punctuation, nothing long -- and
    # the upload options can override the guess either way.
    if len(values) == 1:
        return bool(re.fullmatch(r"[A-Za-z][\w .-]{0,31}", values[0])) \
            and not values[0].endswith((".", "!", "?"))
    return True


def _csv_rows(text: str, opts: dict) -> Iterator[dict]:
    """Rows from a delimited file, header or no header.

    The delimiter is sniffed rather than assumed: European exports are
    semicolon-separated often enough that guessing comma turns the whole
    table into one column with the data still inside it.
    """
    chosen = (opts.get("delimiter") or "").strip()
    if chosen in ("\\t", "tab"):
        chosen = "\t"
    sample = text[:8192]
    if chosen:
        delim = chosen
    else:
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            delim = dialect.delimiter
        except csv.Error:
            delim = "\t" if "\t" in sample.split("\n")[0] else ","

    # "auto" reads the first row and decides; the other two are there for the
    # files it gets wrong, which are always somebody's real files.
    want = (opts.get("header") or "auto").lower()
    reader = csv.reader(io.StringIO(text), delimiter=delim)
    try:
        first = next(reader)
    except StopIteration:
        return
    if want == "yes" or (want != "no" and _is_header(first)):
        # Blank and duplicate headings both happen; both would silently drop a
        # column, so they are named rather than skipped.
        names, seen = [], set()
        for i, h in enumerate(first):
            name = (h or "").strip() or "column_%d" % (i + 1)
            if name in seen:
                name = "%s_%d" % (name, i + 1)
            seen.add(name)
            names.append(name)
    else:
        names = ["column_%d" % (i + 1) for i in range(len(first))]
        yield dict(zip(names, first))
    for values in reader:
        if not any((v or "").strip() for v in values):
            continue
        yield dict(zip(names, values))


# ---- the front door --------------------------------------------------------

def rows_from_upload(filename: str, blob: bytes, options: dict | None = None,
                     source: str | None = None,
                     problems: list[str] | None = None) -> Iterator[dict]:
    """Parse one uploaded file into rows.

    `options` decides how prose is cut up (see TEXT_MODES) and may name a CSV
    delimiter. Structured files ignore it: a JSONL row is already a row.
    `source` adds a column naming the file the rows came from, which is what
    tells fifty files apart once they are in one dataset. `problems` collects
    the members of an archive that could not be read, so they can be reported
    rather than quietly missing.
    """
    opts = options or {}
    name = (filename or "").lower()
    kind = _looks_like(name, blob)

    if kind == "zip":
        yield from _zip_rows(blob, opts, problems)
        return
    if kind == "xlsx":
        raise ValueError(
            "spreadsheets are not read here. Save the sheet as CSV -- one "
            "sheet per file -- and upload that instead")
    if kind == "parquet":
        raise ValueError(
            "Parquet needs a columnar reader this controller deliberately "
            "does not install. If the file came from the Hub, import it by "
            "name instead; otherwise convert it to JSONL or CSV first")

    if kind in ("json", "jsonl"):
        text = blob.decode("utf-8-sig", errors="replace")
        rows = _json_rows(text)
        if source:
            rows = ({**r, "source": source} for r in rows)
        yield from rows
        return

    if kind == "csv":
        text = blob.decode("utf-8-sig", errors="replace")
        rows = _csv_rows(text, opts)
        if source:
            rows = ({**r, "source": source} for r in rows)
        yield from rows
        return

    if kind == "docx":
        text = _docx_text(blob)
    elif kind == "pdf":
        text = _pdf_text(blob)
    elif kind == "html":
        text = _html_text(blob.decode("utf-8-sig", errors="replace"))
    else:
        text = blob.decode("utf-8-sig", errors="replace")

    if not text.strip():
        raise ValueError("no text could be read out of %s"
                         % (filename or "that file"))
    yield from _text_rows(text, opts, source)


def _zip_rows(blob: bytes, opts: dict,
              problems: list[str] | None = None) -> Iterator[dict]:
    """Every readable file in an archive, in name order, tagged with its name.

    A zip of a thousand text files is how a corpus usually arrives, and
    unpacking it by hand to upload them one at a time is the kind of chore
    that stops people bringing their own data at all.
    """
    import zipfile
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        members = [m for m in z.infolist()
                   if not m.is_dir()
                   and not m.filename.startswith("__MACOSX/")
                   and not Path(m.filename).name.startswith(".")]
        if not members:
            raise ValueError("that archive has no files in it")
        failures: list[str] = problems if problems is not None else []
        produced = 0
        for m in sorted(members, key=lambda x: x.filename):
            try:
                inner = z.read(m)
            except (RuntimeError, zipfile.BadZipFile) as e:
                failures.append("%s (%s)" % (m.filename, e))
                continue
            try:
                for row in rows_from_upload(m.filename, inner, opts,
                                            source=m.filename):
                    produced += 1
                    yield row
            except ValueError as e:
                # One unreadable member is not a broken archive. Skipped, and
                # counted -- silence here is how half a corpus goes missing.
                failures.append("%s (%s)" % (m.filename, e))
        if not produced:
            raise ValueError(
                "nothing in that archive could be read: %s"
                % "; ".join(failures[:3]))


async def rows_from_hub(dataset_id: str, config_name: str | None,
                        split: str | list[str], limit: int,
                        token: str | None = None,
                        progress: Callable[[int], None] | None = None,
                        incomplete: dict | None = None) -> list[dict]:
    """Pull rows out of a Hub dataset through the datasets-server.

    Paged at 100 rows a request, which is that service's maximum. Slow for
    large imports, and it needs no `datasets` install, no Arrow, and no
    multi-gigabyte download to take the first ten thousand rows of something
    -- which is what an import here is nearly always for.

    Several splits may be named, and each row is stamped with the split it
    came from. Importing train and test as two datasets would lose the fact
    that they are the same data cut two ways, which is the fact that makes a
    held-out score mean anything.
    """
    wanted = [split] if isinstance(split, str) else list(split or [])
    wanted = [s.strip() for s in wanted if (s or "").strip()] or [DEFAULT_SPLIT]
    out: list[dict] = []
    # Where a split stopped early, if it did. Reported through the caller's
    # dict rather than swallowed: "I got 9,600 of 25,000 because the server
    # asked me to stop" is a fact somebody needs before training on it.
    stopped_at: dict[str, int] = incomplete if incomplete is not None else {}
    headers = {"Authorization": "Bearer %s" % token} if token else {}
    # The limit is per split: asking for 5,000 rows of train and test means
    # 5,000 of each, which is what "import these two splits" plainly means.
    limit = min(int(limit or 1000), MAX_ROWS)
    async with httpx.AsyncClient(timeout=60) as c:
        for name in wanted:
            taken = 0
            while taken < limit:
                r = await c.get("%s/rows" % DATASETS_SERVER, headers=headers,
                                params={
                                    "dataset": dataset_id,
                                    "config": config_name or "default",
                                    "split": name, "offset": taken,
                                    "length": min(IMPORT_PAGE, limit - taken)})
                # Importing a whole dataset is hundreds of requests, and the
                # datasets-server starts refusing partway through. Waiting is
                # the correct response to being asked to slow down; giving up
                # is how a 75,000-row import quietly became 9,600 rows of one
                # split, which is what this did before.
                if r.status_code in (429, 500, 502, 503, 504):
                    if not await _wait_out(r, c):
                        stopped_at[name] = taken
                        break
                    continue
                if r.status_code >= 400:
                    if out:
                        stopped_at[name] = taken
                        break   # a partial import is still worth keeping
                    raise ValueError(
                        "Hugging Face would not serve the %r split of this "
                        "dataset (%d). Either the collection is not called "
                        "\"default\" -- look up its splits and pick the right "
                        "one -- or this is one of the datasets that has to be "
                        "downloaded in full by a machine with the datasets "
                        "library." % (name, r.status_code))
                batch = r.json().get("rows") or []
                if not batch:
                    break
                # The budget is consecutive refusals, not refusals in total:
                # a long import may be asked to slow down a dozen times and
                # still finish, and it should.
                c._studio_retries = 0
                out += [{**(row.get("row") or {}), SPLIT_FIELD: name}
                        for row in batch]
                taken += len(batch)
                if progress:
                    progress(len(out))
                if len(batch) < IMPORT_PAGE:
                    break
    return out


# How long to wait out a rate limit before giving up on a split, in total.
_RETRY_WAITS = (2, 5, 10, 20)


async def _wait_out(response: httpx.Response, client: httpx.AsyncClient) -> bool:
    """Sleep off a rate limit. False when it has been asked for too often."""
    import asyncio
    tries = getattr(client, "_studio_retries", 0)
    if tries >= len(_RETRY_WAITS):
        return False
    delay = _RETRY_WAITS[tries]
    if after := response.headers.get("retry-after"):
        try:
            delay = max(delay, min(float(after), 60))
        except ValueError:
            pass
    client._studio_retries = tries + 1
    await asyncio.sleep(delay)
    return True


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
    rows, steps, out_fmt, before, report = apply_ops(dataset, ops)
    if not rows:
        raise ValueError(
            "Those settings would leave the dataset empty. Loosen them and "
            "try again -- nothing has been changed.")

    label = name or "%s (cleaned)" % dataset["name"]
    created = register(
        owner_id, label, "derived", iter(rows),
        origin=dataset["id"], parent_id=dataset["id"],
        columns=sorted({k for r in rows[:200] for k in r}),
        format=out_fmt,
        recipe={"from": dataset["id"], "from_name": dataset["name"],
                "steps": steps, "rows_before": before, "rows_after": len(rows),
                "report": report or None},
        notes=ops.get("notes"))
    created["report"] = report
    return created


def preview_transform(dataset: dict, ops: dict, sample: int = 2000) -> dict:
    """What these settings would do, without doing it.

    A transform that writes a new dataset is cheap to undo and expensive to
    misread: you find out it dropped four fifths of your rows only after it
    has, and then you have two datasets and a question. So the same code runs
    over a sample first and reports what came out.

    Deliberately a sample, and it says so: on two million rows a faithful
    rehearsal would take longer than the real thing anyone is trying to avoid.
    Dedupe and sampling behave differently on a slice, which is exactly why
    the answer is labelled rather than presented as a count.
    """
    rows, steps, out_fmt, before, report = apply_ops(dataset, ops, sample)
    fmt = formatting.resolve_format(out_fmt or {})
    shown = rows[:5]
    return {
        "sampled": before,
        "kept": len(rows),
        "total_rows": dataset.get("rows") or 0,
        "partial": before < (dataset.get("rows") or 0),
        "steps": steps,
        "format": out_fmt,
        "columns": sorted({k for r in rows[:200] for k in r}),
        "rows": shown,
        "rendered": [formatting.format_example(r, fmt) or "" for r in shown],
        "splits": _count_splits(rows),
        # What the conversion had to infer and what it could not make sense
        # of. Empty for every transform that is not a conversion.
        "report": report,
    }


def _count_splits(rows: list[dict]) -> dict:
    out: dict[str, int] = {}
    for r in rows:
        name = str(r.get(SPLIT_FIELD) or DEFAULT_SPLIT)
        out[name] = out.get(name, 0) + 1
    return out


def apply_ops(dataset: dict, ops: dict, sample: int | None = None
              ) -> tuple[list[dict], list[str], dict, int, dict]:
    """Run a set of operations over the rows. Reads; never writes.

    Shared by the real transform and by the preview of one, so that what the
    preview shows and what the transform does cannot be two different pieces
    of code that drift.
    """
    fmt = formatting.resolve_format(dataset.get("format") or {})
    steps: list[str] = []

    rows = list(iter_rows(dataset["id"], sample))
    before = len(rows)

    # ---- the columns first ------------------------------------------------
    #
    # Before any row is filtered, because every filter below asks how a row
    # renders, and how a row renders is exactly what these change. Rearranging
    # the columns afterwards would filter on the old shape and save the new
    # one -- the preview and the result would disagree.

    if renames := {k: v for k, v in (ops.get("rename") or {}).items()
                   if k and v and k != v}:
        rows = [{renames.get(k, k): v for k, v in r.items()} for r in rows]
        steps.append("Renamed %s" % ", ".join(
            "%s → %s" % (k, v) for k, v in renames.items()))

    # Splitting one column into several: "Ada Lovelace" into first and last,
    # a tab-separated field into its parts. The inverse of a template, and
    # the other half of what makes an arbitrary table workable.
    for spec in (ops.get("split_columns") or []):
        source = (spec.get("from") or "").strip()
        into = [n.strip() for n in (spec.get("into") or []) if n.strip()]
        sep = spec.get("by")
        if not source or not into:
            continue
        for r in rows:
            value = r.get(source)
            text = "" if value is None else str(value)
            parts = text.split(sep) if sep else text.split()
            for n, name in enumerate(into):
                r[name] = parts[n].strip() if n < len(parts) else ""
        steps.append("Split \"%s\" on %s into %s"
                     % (source, ("%r" % sep) if sep else "whitespace",
                        ", ".join(into)))

    # Calculated columns. `template` is the single-column form the first
    # version had, kept working; `columns` is the list form, which is what a
    # workbench actually needs -- three derived columns is not three passes
    # over the data with three names for the same operation.
    built: list[tuple[str, str]] = []
    if template := (ops.get("template") or "").strip():
        built.append(((ops.get("template_column") or "text").strip() or "text",
                      template))
    for spec in (ops.get("columns") or []):
        name = (spec.get("name") or "").strip()
        body = spec.get("template")
        if name and body:
            built.append((name, body))

    for target, body in built:
        # The operation that makes an arbitrary table trainable: "Q: {question}
        # A: {answer}" turns two columns nobody's trainer recognises into one
        # the whole app already understands.
        made = 0
        out = []
        for r in rows:
            filled = _fill_template(body, r)
            if filled.strip():
                made += 1
            out.append({**r, target: filled})
        rows = out
        steps.append("Built the column \"%s\" from a template (%d of %d rows "
                     "filled)" % (target, made, len(rows)))
    if built:
        # From here on the rows are read as the last column built, including
        # by every filter below and by whatever trains on the result.
        template = template or built[-1][1]
        fmt = formatting.resolve_format(
            {"mode": "text", "text_field": built[-1][0]})

    # After the template, never before it: dropping the columns a template
    # reads from is a reasonable thing to ask for -- build the text, throw the
    # parts away -- and doing it first left every row reading "{question}".
    if dropped := [c for c in (ops.get("drop_columns") or [])
                   if c and c != SPLIT_FIELD]:
        rows = [{k: v for k, v in r.items() if k not in dropped} for r in rows]
        steps.append("Dropped the columns %s" % ", ".join(dropped))

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
            # Compared as the canonical record, not as `format_example`'s plain
            # rendering. That rendering is a projection -- on a chat dataset it
            # writes "user: ... / assistant: ..." and drops the reasoning
            # entirely -- so two rows with the same question and answer but
            # different working hashed alike, and one was silently thrown away.
            #
            # On a reasoning set that is not a duplicate being removed, it is a
            # trained field being ignored. The canonical record holds
            # everything a template can render: the working, the tool calls and
            # the ids, whether or not this particular rendering shows them.
            conv = conversation.from_row(r, fmt)
            h = hashlib.sha1(json.dumps(conv, sort_keys=True, default=str)
                             .encode("utf-8")).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            kept.append(r)
        rows = kept
        steps.append("Removed exact duplicates (%d removed)" % (n0 - len(rows)))

    # Exact duplicates are rare in a generated set and repeated QUESTIONS are
    # not: a model asked thirty times for an example on one topic converges on
    # the obvious question and varies only the answer, so `dedupe` above finds
    # nothing while one question quietly takes 2% of the file. Keeping a few of
    # each is useful -- two good answers to one question is augmentation -- and
    # keeping eight is teaching that question rather than the shape.
    if per_prompt := int(ops.get("max_per_prompt") or 0):
        n0 = len(rows)
        seen: dict[str, int] = {}
        kept = []
        for r in rows:
            conv = conversation.from_row(r, fmt)
            asked = " | ".join(
                (m.get("content") or "").strip()
                for m in conv[conversation.MESSAGES_KEY]
                if m.get("role") in ("user", "system"))
            key = hashlib.sha1(asked.encode("utf-8")).hexdigest()
            if seen.get(key, 0) >= per_prompt:
                continue
            seen[key] = seen.get(key, 0) + 1
            kept.append(r)
        rows = kept
        steps.append("Kept at most %d row%s per question (%d removed)"
                     % (per_prompt, "" if per_prompt == 1 else "s",
                        n0 - len(rows)))

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
        keep = list(keep) + ([SPLIT_FIELD] if SPLIT_FIELD not in keep else [])
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

    # `to_chat` is what the first version of this called it, kept working
    # because it is in saved recipes and in whatever anybody scripted against
    # the endpoint. It means the same thing it always did, which is now this.
    report: dict = {}
    convs_made: list[dict] = []
    if ops.get("to_conversations") or ops.get("to_chat"):
        rows, report, convs_made = _to_conversations(rows, fmt, ops)
        steps.append("Converted %d rows to the standard conversation format"
                     % report["converted"])
        if report.get("dropped"):
            steps.append("Dropped %d rows with nothing conversational in them"
                         % report["dropped"])
        for fixed in report.get("repairs") or []:
            steps.append("Repaired what the data left implicit: %s (%d rows)"
                         % (fixed["what"], fixed["rows"]))
        if ops.get("train_on") == "last":
            steps.append("Marked every assistant turn but the last one "
                         "weight 0, so they are context and not lessons")

    if ops.get("to_conversations") or ops.get("to_chat"):
        # What was produced, not merely how to read it. "mode: chat" is a
        # complete description of how to read these rows and says nothing about
        # what is in them -- and the wizard decides whether to offer "teach it
        # to reason" from exactly that. A conversion that dropped the fact left
        # a dataset whose every row has a reasoning block reported as having
        # none, with the toggle greyed out over it.
        out_fmt = {"mode": "chat", "messages_field": conversation.MESSAGES_KEY}
        roles: list[str] = []
        for conv in convs_made:
            for m in conv[conversation.MESSAGES_KEY]:
                if m["role"] not in roles:
                    roles.append(m["role"])
        out_fmt["roles"] = roles
        out_fmt["has_reasoning"] = any(
            m.get("reasoning") for c in convs_made
            for m in c[conversation.MESSAGES_KEY])
        out_fmt["has_tool_calls"] = any(
            m.get("tool_calls") for c in convs_made
            for m in c[conversation.MESSAGES_KEY])
        if any(c[conversation.TOOLS_KEY] for c in convs_made):
            out_fmt["tools_field"] = conversation.TOOLS_KEY
        if train_on := (ops.get("train_on") or "").strip():
            out_fmt["train_on"] = "assistant" if train_on == "last" else train_on
    elif built:
        # Recorded, not re-detected. A dataset that still carries its original
        # columns would otherwise be read by those instead of by the column
        # just built, and the calculation would have done nothing.
        out_fmt = {"mode": "text", "text_field": built[-1][0]}
    else:
        out_fmt = dataset.get("format")
    return rows, steps, out_fmt or {}, before, report


# `{column}`, `{column|filter}`, `{column|slice:0:80}`, `{a|trim|lower}`.
_PLACEHOLDER = re.compile(
    r"\{([A-Za-z0-9_. -]{1,64})((?:\|[a-z]+(?::[-\d:]*)?)*)\}")

# What a calculated column may do to a value on the way past. Deliberately a
# short list of verbs rather than an expression language: a spreadsheet's
# worth of functions in a text box is a programming language nobody wrote
# documentation for, and these are the operations training text actually
# needs.
FILTERS = {
    "upper": "UPPERCASE",
    "lower": "lowercase",
    "title": "Title Case",
    "trim": "remove surrounding whitespace",
    "lines": "collapse to one line",
    "first": "the first line only",
    "last": "the last line only",
    "len": "the number of characters",
    "words": "the number of words",
    "json": "as compact JSON",
    "slice": "slice:0:200 — the characters between two positions",
}


def _apply_filter(value: str, name: str, arg: str) -> str:
    if name == "upper":
        return value.upper()
    if name == "lower":
        return value.lower()
    if name == "title":
        return value.title()
    if name == "trim":
        return value.strip()
    if name == "lines":
        return " ".join(value.split())
    if name in ("first", "last"):
        parts = value.splitlines()
        return (parts[0] if name == "first" else parts[-1]) if parts else ""
    if name == "len":
        return str(len(value))
    if name == "words":
        return str(len(value.split()))
    if name == "slice":
        bits = (arg or "").split(":")
        try:
            start = int(bits[0]) if bits and bits[0] else 0
            end = int(bits[1]) if len(bits) > 1 and bits[1] else None
        except ValueError:
            return value
        return value[start:end]
    return value


def _fill_template(template: str, row: dict) -> str:
    """`"Q: {question}\\nA: {answer}"` against one row.

    Deliberately not str.format: a template written by hand contains braces
    that are not placeholders -- JSON examples, code, an emoticon -- and
    str.format raises on all of them and names a column nobody wrote. An
    unknown name is left standing as written, which is visible in the
    preview, rather than throwing away the whole run.
    """
    def one(m: "re.Match[str]") -> str:
        key, pipes = m.group(1), m.group(2) or ""
        if key not in row:
            stripped = key.strip()
            if stripped not in row:
                return m.group(0)
            key = stripped
        value = row[key]
        if value is None:
            text = ""
        elif isinstance(value, (dict, list)):
            text = json.dumps(value, ensure_ascii=False)
        else:
            text = str(value)
        for piece in [p for p in pipes.split("|") if p]:
            name, _, arg = piece.partition(":")
            text = _apply_filter(text, name, arg)
        return text
    return _PLACEHOLDER.sub(one, template)


def _to_conversations(rows: list[dict], fmt: dict,
                      ops: dict) -> tuple[list[dict], dict, list[dict]]:
    """Rewrite rows into the canonical conversation format.

    This is the step the whole workbench exists to reach. Whatever the data
    was -- ShareGPT turns, three Alpaca columns, a CSV of questions and
    answers, a tool-calling set with the schema in a sibling column -- it comes
    out of here as one shape: `messages`, `tools`, `meta`. Everything after
    this point, the trainer and the playground included, reads that one shape
    and nothing else.

    Done as an explicit transform rather than silently at training time
    because the conversion is exactly where data goes wrong, and a conversion
    you cannot see is a conversion you cannot fix. What comes out is a new
    dataset, inspectable row by row, with a report of everything that had to be
    inferred to produce it.
    """
    system = (ops.get("system_prompt") or "").strip()
    selectors = ops.get("selectors") or fmt.get("selectors")
    # Which turns the eventual run should learn from, recorded per row rather
    # than only on the job: a dataset that means "learn the final answer only"
    # means it wherever it is trained.
    train_on = (ops.get("train_on") or "").strip()
    read_as = dict(fmt)
    if selectors:
        read_as["selectors"] = selectors

    out: list[dict] = []
    convs: list[dict] = []
    repairs: dict[str, int] = {}
    dropped = 0

    for r in rows:
        conv = conversation.from_row(r, read_as, selectors)
        if not conv[conversation.MESSAGES_KEY]:
            dropped += 1
            continue
        conv, notes = conversation.repair(conv)
        for note in notes:
            # Counted by what was done, not by row: "linked 4,180 tool results
            # to the call they answer" is a fact somebody can act on, and four
            # thousand copies of one sentence is not.
            kind = re.sub(r"\d+", "N", note)
            repairs[kind] = repairs.get(kind, 0) + 1

        msgs = conv[conversation.MESSAGES_KEY]
        if system and not any(m["role"] == "system" for m in msgs):
            msgs.insert(0, {"role": "system", "content": system})
        if train_on == "last":
            # Everything but the final assistant turn is rendered as context
            # and not learned from. The usual reason: a long conversation whose
            # earlier replies came from somewhere you do not want imitated.
            last = max((i for i, m in enumerate(msgs)
                        if m["role"] == "assistant"), default=None)
            for i, m in enumerate(msgs):
                if m["role"] == "assistant" and i != last:
                    m["weight"] = 0

        convs.append(conv)
        # The split survives being rewritten. Losing it here would quietly
        # merge somebody's held-out rows back into training.
        out.append(conversation.to_row(conv, split=r.get(SPLIT_FIELD)))

    report = conversation.validate_many(convs)
    report["converted"] = len(out)
    report["dropped"] = dropped
    report["repairs"] = [{"what": k, "rows": v} for k, v in repairs.items()]
    return out, report, convs


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
            iter(part), split=label,
            origin=dataset["id"], parent_id=dataset["id"],
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
