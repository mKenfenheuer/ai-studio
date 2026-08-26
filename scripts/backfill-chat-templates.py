#!/usr/bin/env python
"""Put every saved model's chat template in both places a reader looks.

Run it with `python scripts/backfill-chat-templates.py` on the controller, or
with `--dry-run` first to see what it would touch.

Transformers 5 saves a chat template to `chat_template.jinja` and leaves
`chat_template` out of `tokenizer_config.json`. Plenty of readers only ever
look in the config -- older transformers, several serving stacks, and this
studio's own artifact endpoint until recently, which is why the training wizard
announced that an instruct fine-tune "ships no chat template of its own, which
usually means it is a base model" about a model whose template it had written
in itself.

Reading now looks in both places, so nothing is broken by leaving an artifact
alone. What this fixes is the artifact somebody DOWNLOADS: a zip carrying its
template in one of the two places is a model half the ecosystem reads as a base
model, and that is not something the studio can fix from its own side.

Rewriting a zip means copying it, so each one is written beside the original
and moved into place only once it is complete and readable. A run that is
interrupted leaves the original untouched.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import chat_formats as CF          # noqa: E402


def survey(path: Path) -> tuple[str | None, bool, bool]:
    """This artifact's template, and which of the two places already hold it."""
    with zipfile.ZipFile(path) as z:
        names = {n.rsplit("/", 1)[-1]: n for n in z.namelist()}
        if CF.TOKENIZER_CONFIG not in names:
            return None, False, False
        files = {n: z.read(names[n]).decode("utf-8")
                 for n in (CF.TEMPLATE_FILE, CF.TOKENIZER_CONFIG)
                 if n in names}
    template = CF.template_in(files)
    try:
        in_conf = bool(json.loads(files.get(CF.TOKENIZER_CONFIG) or "{}")
                       .get("chat_template"))
    except ValueError:
        in_conf = False
    return template, in_conf, CF.TEMPLATE_FILE in files


def rewrite(path: Path, template: str) -> None:
    """Copy the zip with the template in both places, then swap it in."""
    tmp = path.with_suffix(".backfill.zip")
    with zipfile.ZipFile(path) as src, \
            zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as out:
        conf_name = next((n for n in src.namelist()
                          if n.rsplit("/", 1)[-1] == CF.TOKENIZER_CONFIG), None)
        jinja_name = next((n for n in src.namelist()
                           if n.rsplit("/", 1)[-1] == CF.TEMPLATE_FILE), None)
        for info in src.infolist():
            if info.filename == conf_name:
                conf = json.loads(src.read(info).decode("utf-8"))
                conf["chat_template"] = template
                out.writestr(info, json.dumps(conf, indent=2, ensure_ascii=False))
                continue
            if info.filename == jinja_name:
                out.writestr(info, template)
                continue
            # Streamed rather than read whole: a merged 7B has members far
            # larger than the memory of the machine serving it.
            with src.open(info) as fh, out.open(info, "w") as target:
                shutil.copyfileobj(fh, target, 1024 * 1024)
        if jinja_name is None:
            out.writestr(_beside(conf_name, CF.TEMPLATE_FILE), template)

    # Proven readable before anything irreversible happens to the original.
    check, in_conf, in_jinja = survey(tmp)
    if not (check == template and in_conf and in_jinja):
        tmp.unlink(missing_ok=True)
        raise RuntimeError("the rewritten copy did not come back correct")
    tmp.replace(path)


def _beside(conf_name: str | None, name: str) -> str:
    """A new file next to the tokenizer config, wherever that lives."""
    if conf_name and "/" in conf_name:
        return conf_name.rsplit("/", 1)[0] + "/" + name
    return name


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts", default="/data/artifacts",
                    help="where the run zips are (default: /data/artifacts)")
    ap.add_argument("--dry-run", action="store_true",
                    help="say what would change and change nothing")
    args = ap.parse_args()

    art = Path(args.artifacts)
    if not art.is_dir():
        print("No artifact directory at %s" % art)
        return 1

    todo, skipped = [], 0
    for p in sorted(art.glob("*.zip")):
        try:
            template, in_conf, in_jinja = survey(p)
        except (OSError, ValueError, zipfile.BadZipFile) as e:
            print("  %-26s unreadable (%s)" % (p.name, str(e)[:60]))
            continue
        if not template:
            skipped += 1
            continue
        if in_conf and in_jinja:
            skipped += 1
            continue
        todo.append((p, template, p.stat().st_size))

    print("%d artifact(s) already carry it in both places." % skipped)
    if not todo:
        print("Nothing to backfill.")
        return 0

    need = max(size for _, _, size in todo)
    free = shutil.disk_usage(art).free
    print("%d to rewrite, %.1f GB in total. Largest needs %.1f GB spare; "
          "%.1f GB free."
          % (len(todo), sum(s for _, _, s in todo) / 1024 ** 3,
             need / 1024 ** 3, free / 1024 ** 3))
    if free < need * 1.1:
        print("Not enough room to rewrite the largest one safely. Stopping.")
        return 1
    if args.dry_run:
        for p, _, size in todo:
            print("  would rewrite %-26s %6.2f GB" % (p.name, size / 1024 ** 3))
        return 0

    for p, template, size in todo:
        t0 = time.time()
        print("  rewriting %-26s %6.2f GB … " % (p.name, size / 1024 ** 3),
              end="", flush=True)
        try:
            rewrite(p, template)
        except Exception as e:                       # noqa: BLE001
            print("FAILED (%s) -- the original is untouched" % str(e)[:80])
            continue
        print("done in %.0fs" % (time.time() - t0))
    print("Backfill complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
