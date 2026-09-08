#!/usr/bin/env python
"""Check that a row keeps its name, and a split keeps its meaning.

Run it with `python scripts/check-datasets.py` from the repository root. No
test framework, for the same reason `check-formats.py` has none: this has to
run on the machine somebody is debugging on.

What it guards:

* Every row written gets a name, and keeps it through a filter, a rename, a
  merge, a conversion to chat, and an edit. Names are what make it possible to
  mark a row, diff two datasets, or point at a row from anywhere else — and
  they are the fix for addressing rows by position, where deleting row 3 made
  row 4 into row 3 and a second click deleted somebody else's data.
* Deleting and moving rows acts on the rows asked for, whichever way they were
  named, and does not renumber anything the caller still holds.
* Holding back a validation split produces one dataset with two splits, not
  two datasets — and a stratified hold-back keeps the mix.
* The reserved columns survive `keep_columns` and `drop_columns`, because a
  row that loses its name stops being that row and one that loses its split
  quietly rejoins the training data.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# A scratch data directory, set before anything imports the config that reads
# it. Nothing here should touch a real studio.
_TMP = Path(tempfile.mkdtemp(prefix="ai-studio-check-"))
os.environ["AI_STUDIO_DATA"] = str(_TMP)

from controller import datasets as ds  # noqa: E402
from controller import db  # noqa: E402

FAILED: list[str] = []


def check(name: str, ok: bool, detail: object = "") -> None:
    print("  %s  %s%s" % ("ok  " if ok else "FAIL", name,
                          "" if ok else "   -> %s" % (detail,)))
    if not ok:
        FAILED.append(name)


def rows_of(dataset_id: str) -> list[dict]:
    return list(ds.iter_rows(dataset_id))


def make(name: str, rows: list[dict], **fields) -> dict:
    return ds.register(None, name, "upload", iter(rows), **fields)


# ---------------------------------------------------------------------------

def test_names() -> None:
    print("\nEvery row has a name")
    d = make("plain", [{"text": "one"}, {"text": "two"}, {"text": "three"}])
    rows = rows_of(d["id"])
    ids = [r.get(ds.ROW_ID_FIELD) for r in rows]
    check("every row is named", all(ids), ids)
    check("names are unique", len(set(ids)) == 3, ids)
    check("names are not offered as a column",
          ds.ROW_ID_FIELD not in (d.get("columns") or []), d.get("columns"))

    # A name that arrives with the data is kept rather than replaced: this is
    # what lets a derived row say which row it came from.
    keep = make("named", [{"text": "a", ds.ROW_ID_FIELD: "mine00000001"}])
    check("a name that arrives is kept",
          rows_of(keep["id"])[0][ds.ROW_ID_FIELD] == "mine00000001")


def test_editing_by_name() -> None:
    print("\nEditing addresses rows, not slots")
    d = make("edit", [{"text": t} for t in "abcde"])
    ids = [r[ds.ROW_ID_FIELD] for r in rows_of(d["id"])]

    # The failure this exists to prevent: delete the second row, then use the
    # list of names read *before* that, and the right rows still go.
    ds.edit_rows(d, delete=[ids[1]])
    after = rows_of(d["id"])
    check("the named row went", [r["text"] for r in after] == list("acde"),
          [r["text"] for r in after])

    d = db.get_dataset(d["id"])
    ds.edit_rows(d, delete=[ids[3]])
    after = rows_of(d["id"])
    check("a stale position would have been wrong, a name is not",
          [r["text"] for r in after] == list("ace"), [r["text"] for r in after])

    d = db.get_dataset(d["id"])
    ds.edit_rows(d, move={"ids": [ids[0]], "to": "validation"})
    after = rows_of(d["id"])
    check("moving by name moves that row",
          after[0].get(ds.SPLIT_FIELD) == "validation", after[0])
    check("moving keeps the name",
          after[0][ds.ROW_ID_FIELD] == ids[0])

    d = db.get_dataset(d["id"])
    ds.edit_rows(d, update={ids[2]: {"text": "CORRECTED"}})
    after = rows_of(d["id"])
    fixed = [r for r in after if r["text"] == "CORRECTED"]
    check("editing a row by name rewrites that row", len(fixed) == 1, after)
    check("an edited row is still the row it was",
          fixed and fixed[0][ds.ROW_ID_FIELD] == ids[2])

    # Positions still work, for a dataset written before names existed.
    d = db.get_dataset(d["id"])
    ds.edit_rows(d, delete=[0])
    check("a position is still accepted",
          len(rows_of(d["id"])) == 2, rows_of(d["id"]))


def test_names_survive_transforms() -> None:
    print("\nA name survives being transformed")
    d = make("shape", [{"q": "one", "a": "1"}, {"q": "two", "a": "2"},
                       {"q": "", "a": "3"}])
    before = {r["q"]: r[ds.ROW_ID_FIELD] for r in rows_of(d["id"])}

    made = ds.transform(d, {"rename": {"q": "question"},
                            "where": [{"column": "question", "op": "not_empty"}]},
                        owner_id=None, name="renamed")
    after = rows_of(made["id"])
    check("a filter keeps the names of the rows it kept",
          {r["question"] for r in after} == {"one", "two"},
          [r.get("question") for r in after])
    check("a rename does not rename the row",
          all(r[ds.ROW_ID_FIELD] == before[r["question"]] for r in after),
          [(r["question"], r[ds.ROW_ID_FIELD]) for r in after])

    # Building a column and throwing the parts away is the operation that makes
    # a spreadsheet trainable; the row is still the same row afterwards.
    built = ds.transform(db.get_dataset(made["id"]),
                         {"columns": [{"name": "text",
                                       "template": "Q: {question}\nA: {a}"}],
                          "drop_columns": ["question", "a"]},
                         owner_id=None, name="built")
    rows = rows_of(built["id"])
    check("a built column keeps the row's name",
          all(r.get(ds.ROW_ID_FIELD) for r in rows), rows)
    check("dropping columns cannot drop the name",
          all(ds.ROW_ID_FIELD in r for r in rows), rows)

    kept = ds.transform(db.get_dataset(built["id"]),
                        {"keep_columns": ["text"]}, owner_id=None, name="kept")
    rows = rows_of(kept["id"])
    check("keeping one column still keeps the name and the split",
          all(ds.ROW_ID_FIELD in r for r in rows), rows)


def test_conversion_keeps_names() -> None:
    print("\nConverting to chat keeps the names")
    d = make("chatty", [{"instruction": "Say hi", "output": "Hi"},
                        {"instruction": "Say bye", "output": "Bye"}])
    before = [r[ds.ROW_ID_FIELD] for r in rows_of(d["id"])]
    made = ds.transform(d, {"to_conversations": {}}, owner_id=None, name="as-chat")
    rows = rows_of(made["id"])
    check("every converted row is a conversation",
          all("messages" in r for r in rows), rows)
    check("and is still the row it was",
          [r.get(ds.ROW_ID_FIELD) for r in rows] == before,
          [r.get(ds.ROW_ID_FIELD) for r in rows])


def test_holding_back_a_split() -> None:
    print("\nHolding back a validation split")
    d = make("holdout", [{"text": "row %d" % i, "label": "a" if i % 4 else "b"}
                         for i in range(40)])
    out = ds.split(d, 0.25, owner_id=None)
    check("it makes one dataset, not two", isinstance(out, dict), type(out))
    if isinstance(out, dict):
        splits = out.get("splits") or {}
        check("with both splits in it",
              set(splits) == {"train", "validation"}, splits)
        check("of the sizes asked for",
              splits.get("validation") == 10 and splits.get("train") == 30, splits)
        rows = rows_of(out["id"])
        check("and every row keeps its name",
              all(r.get(ds.ROW_ID_FIELD) for r in rows))
        check("no row is in both", len(rows) == 40, len(rows))

    # Stratified: the rare label is represented in the held-out part rather
    # than landing entirely on one side of the cut.
    d2 = make("strat", [{"text": "row %d" % i, "label": "rare" if i < 8 else "common"}
                        for i in range(40)])
    out2 = ds.split(d2, 0.25, owner_id=None, stratify="label")
    if isinstance(out2, dict):
        held = [r for r in rows_of(out2["id"])
                if r.get(ds.SPLIT_FIELD) == "validation"]
        rare = [r for r in held if r["label"] == "rare"]
        check("a stratified hold-back keeps the mix",
              len(rare) == 2, "%d rare of %d held out" % (len(rare), len(held)))


def test_merge_keeps_names() -> None:
    print("\nMerging")
    a = make("left", [{"text": "l%d" % i} for i in range(3)])
    b = make("right", [{"text": "r%d" % i} for i in range(3)])
    names = {r[ds.ROW_ID_FIELD] for r in rows_of(a["id"])} \
        | {r[ds.ROW_ID_FIELD] for r in rows_of(b["id"])}
    merged = ds.merge([a, b], owner_id=None, name="both", shuffle=False)
    rows = rows_of(merged["id"])
    check("every row arrives", len(rows) == 6, len(rows))
    check("with the name it had",
          {r[ds.ROW_ID_FIELD] for r in rows} == names)


def main() -> int:
    try:
        test_names()
        test_editing_by_name()
        test_names_survive_transforms()
        test_conversion_keeps_names()
        test_holding_back_a_split()
        test_merge_keeps_names()
    finally:
        shutil.rmtree(_TMP, ignore_errors=True)
    print()
    if FAILED:
        print("%d check(s) failed:" % len(FAILED))
        for name in FAILED:
            print("  - %s" % name)
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
