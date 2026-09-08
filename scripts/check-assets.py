#!/usr/bin/env python
"""Check that stored files are shared, counted and released correctly.

Run it with `python scripts/check-assets.py` from the repository root. No test
framework, for the same reason the other two have none: this has to run on the
machine somebody is debugging on.

What it guards is the three promises in `controller/assets.py`, each of which
is invisible until it is broken and then very expensive:

* **The same bytes are stored once.** Two datasets of the same ten thousand
  photographs are ten thousand files, not twenty thousand, and the usage
  report says so.
* **Deleting is safe.** A dataset derived from another shares its files but
  holds its own references, so deleting either one leaves the other working.
  A store where tidying up your copy tore an image out of a colleague's
  dataset would be worse than no store.
* **Nothing dangerous is storable.** HTML and SVG are documents that can run
  script; served from the studio's own origin they are a cross-site scripting
  hole with a database behind it.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_TMP = Path(tempfile.mkdtemp(prefix="ai-studio-check-"))
os.environ["AI_STUDIO_DATA"] = str(_TMP)

from controller import assets  # noqa: E402
from controller import datasets as ds  # noqa: E402
from controller import db  # noqa: E402

FAILED: list[str] = []

# Not real images: nothing here decodes them, and a check script that needs
# a photograph in the repository is a check script nobody runs.
PNG = bytes.fromhex("89504e470d0a1a0a") + b"x" * 200
JPG = bytes.fromhex("ffd8ffe0") + b"y" * 300


def check(name: str, got: object, want: object = True) -> None:
    ok = got == want
    print("  %s  %s%s" % ("ok  " if ok else "FAIL", name,
                          "" if ok else "   -> %r, wanted %r" % (got, want)))
    if not ok:
        FAILED.append(name)


def main() -> int:
    try:
        ann = db.create_user("ann", "Ann", "x", role="admin")
        bob = db.create_user("bob", "Bob", "x")

        print("\nStoring")
        a = assets.store([PNG], "cat.png", "image/png", ann, dataset_id="ds_A")
        again = assets.store([PNG], "cat.png", "image/png", ann, dataset_id="ds_A")
        check("the same file twice is one reference", again["id"], a["id"])
        elsewhere = assets.store([PNG], "cat.png", "image/png", ann,
                                 dataset_id="ds_B")
        check("another dataset gets its own reference", elsewhere["id"] != a["id"])
        check("pointing at the same bytes", elsewhere["sha256"], a["sha256"])
        check("stored where its hash says", assets.path_for(a["sha256"]).exists())
        check("usage counts the bytes once", assets.usage(ann)["bytes"], len(PNG))
        check("and both references", assets.usage(ann)["references"], 2)

        print("\nRefusing")
        try:
            assets.store([b""], "empty.png", "image/png", ann)
            check("an empty file is refused", False)
        except ValueError as e:
            check("an empty file is refused", "empty" in str(e))
        check("the extension beats the browser's guess",
              assets.guess_mime("clip.WAV", "application/octet-stream"),
              "audio/wav")
        check("html is not storable", assets.guess_mime("x.html", "text/html"), "")
        check("svg is not storable", assets.guess_mime("x.svg", "image/svg+xml"), "")

        print("\nFinding references in rows")
        row = {"image": assets.ref(a["id"]), "label": "cat",
               "clips": ["x", assets.ref(elsewhere["id"])]}
        check("in a cell and in a list",
              sorted(assets.ids_in_row(row)),
              sorted([a["id"], elsewhere["id"]]))
        check("a lookalike string is not one", assets.id_in("asset:nope"), "")
        check("nor is ordinary text", assets.id_in("a cat"), "")

        print("\nA derived dataset keeps its files")
        ds.write_rows("ds_A", iter([{"image": assets.ref(a["id"]),
                                     "label": "cat"}]), ann)
        copied = [dict(r) for r in ds.iter_rows("ds_A", 10)]
        ds.write_rows("ds_C", iter(copied), ann)
        child = list(ds.iter_rows("ds_C", 10))
        child_asset = assets.id_in(child[0]["image"])
        check("the copy points at its own reference",
              child_asset not in ("", a["id"]))
        check("at the same bytes",
              db.get_asset(child_asset)["sha256"], a["sha256"])

        print("\nReleasing")
        freed = assets.release_dataset("ds_A")
        check("the original's reference goes", freed["rows"], 1)
        check("but the file stays", freed["files"], 0)
        check("so the copy still reads", assets.path_for(a["sha256"]).exists())
        check("and so does the other dataset's",
              db.get_asset(elsewhere["id"]) is not None)
        assets.release_dataset("ds_C")
        freed = assets.release_dataset("ds_B")
        check("the last reference takes the file", freed["files"], 1)
        check("the bytes are gone", assets.path_for(a["sha256"]).exists(), False)

        print("\nHousekeeping")
        kept = assets.store([JPG], "dog.jpg", "image/jpeg", bob,
                            dataset_id="ds_D")
        stray = assets.path_for("0" * 64)
        stray.parent.mkdir(parents=True, exist_ok=True)
        stray.write_bytes(b"left behind by a crash")
        check("a file no row names is an orphan",
              [p.name for p in assets.orphans()], ["0" * 64])
        check("and is swept", assets.sweep()["files"], 1)
        check("the file a row does name survives",
              assets.path_for(kept["sha256"]).exists())
        check("usage is per account", assets.usage(bob)["bytes"], len(JPG))
        check("and the studio's total is everybody's",
              assets.usage()["bytes"], len(JPG))
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
