"""Projects: one model being made, and everything done to make it.

A studio's work used to be five lists on five pages -- runs, datasets, prompt
sets, scores, publications -- each sorted by date, with the relationships
between them recorded only in config fields nobody could see from the list.
Making a model is not five lists; it is one thing with stages: prepare the
data, train, evaluate, benchmark, publish. A project is that thing, and its
page is a map of those stages with what has been done at each and what is
next.

What belongs to a project is filed by a `project_id` on the row. Nothing is
moved or copied: the dataset library and the run history are the same rows
seen from a different side, and a dataset used by two projects sits in one of
them with the other pointing at it. Everything that existed before projects
did is "unfiled", shown as such, and can be filed with one click.
"""
from __future__ import annotations

import json

from fastapi import APIRouter, Body, HTTPException, Request

from .. import db
from .security import current_user, require_edit, require_owner, require_view

router = APIRouter(prefix="/api/projects")

STAGES = ("data", "train", "evaluate", "benchmark", "publish")


def _project_or_404(request: Request, project_id: str, need: str = "view") -> dict:
    row = db.get_project(project_id)
    if not row:
        raise HTTPException(404, "No such project.")
    {"own": require_owner, "edit": require_edit}.get(need, require_view)(
        request, "project", row)
    return row


def _decorate(row: dict, user: dict) -> dict:
    row["access"] = db.access_level("project", row["id"], row.get("owner_id"), user)
    row["mine"] = row.get("owner_id") == user["id"]
    owner = db.get_user(row["owner_id"]) if row.get("owner_id") else None
    row["owner"] = db.public_user(owner) if owner else None
    row["shares"] = db.list_shares("project", row["id"])
    return row


@router.get("")
async def list_projects(request: Request) -> dict:
    user = current_user(request)
    return {"projects": db.visible_projects(user),
            # What is not in any project yet. Everything that existed before
            # projects did lands here, and the page offers to file it.
            "unfiled": db.unfiled_counts(user)}


@router.post("")
async def create_project(request: Request, payload: dict = Body(...)) -> dict:
    user = current_user(request)
    name = (payload.get("name") or "").strip()
    if not name:
        raise HTTPException(400, "Give the project a name -- what the model is for.")
    pid = db.create_project(user["id"], name[:120], (payload.get("goal") or "")[:500],
                            (payload.get("notes") or "")[:4000])
    return _decorate(db.get_project(pid), user)


def contents(user: dict, project_id: str | None) -> dict:
    """Everything filed under a project, by stage.

    `None` is the unfiled pile. Visibility is the same as everywhere else: a
    run shared with you shows here, a run that is not does not, whatever
    project it is in.
    """
    def mine(rows):
        return [r for r in rows if (r.get("project_id") or None) == project_id]

    datasets = mine(db.visible_datasets(user))
    runs = mine(db.visible_jobs(user, 1000))
    for r in runs:
        if not r.get("config"):
            r["config"] = (db.get_job(r["id"]) or {}).get("config") or {}
    evals = mine(db.visible_evals(user))
    library = [r for r in db.library_rows(user)
               if (r.get("project_id") or None) == project_id]
    # Scores per prompt set, so the map can say "scored 3 models, best X".
    for e in evals:
        scores = db.list_scores(e["id"])
        e["scorings"] = len(scores)
        e["is_benchmark"] = bool((e.get("source") or {}).get("benchmark"))
        e["latest"] = scores[0] if scores else None
    training = [r for r in runs if r["kind"] in ("finetune_llm", "pretrain_llm",
                                                 "finetune_vision_cls")]
    # Data the project's runs actually trained on, even when it is filed
    # somewhere else. Training on another project's dataset is allowed and
    # moves nothing -- but a project with a finished model and "Data: none"
    # on its map reads as a project missing its first stage.
    used_ids = {(r.get("config") or {}).get("studio_dataset")
                for r in runs} - {None, ""}
    mine_ids = {d["id"] for d in datasets}
    borrowed = []
    for did in sorted(used_ids - mine_ids):
        row = db.get_dataset(did)
        if row and db.access_level("dataset", did, row.get("owner_id"), user):
            row["borrowed_from"] = (db.get_project(row["project_id"]) or {}).get(
                "name") if row.get("project_id") else None
            borrowed.append(row)

    return {
        "datasets": datasets,
        "borrowed_datasets": borrowed,
        "runs": runs,
        "training": training,
        "writing": [r for r in runs if r["kind"] == "generate_dataset"],
        "scorings": [r for r in runs if r["kind"] == "evaluate"],
        "exports": [r for r in runs if r["kind"] in ("export_gguf", "upload")],
        "prompt_sets": [e for e in evals if not e["is_benchmark"]],
        "benchmarks": [e for e in evals if e["is_benchmark"]],
        "library": library,
        "conversations": [c for c in db.list_conversations(user["id"])
                          if (c.get("project_id") or None) == project_id]
                         if project_id else [],
    }


def _stage_map(c: dict) -> list[dict]:
    """The five stages, each with a state and a next step. The page draws it."""
    done_runs = [r for r in c["training"] if r["status"] == "succeeded"]
    active = [r for r in c["runs"] if r["status"] in ("queued", "assigned", "running")]
    best = None
    for r in done_runs:
        pm = (r.get("summary") or {}).get("primary_metric")
        if pm and pm.get("value") is not None:
            if best is None or (pm["value"] < best[1] if pm.get("lower_better", True)
                                else pm["value"] > best[1]):
                best = (r, pm["value"], pm)
    scored = [e for e in c["prompt_sets"] if e["scorings"]]
    benched = [e for e in c["benchmarks"] if e["scorings"]]

    # Whether any of this project's data has a part held back. It is the
    # difference between measuring a model and measuring its memory, and it
    # has to be done *before* training -- afterwards the model has seen the
    # rows and no amount of splitting brings back a fair test. So the Data
    # stage asks for it rather than mentioning it in a note nobody reads at
    # the point it still matters.
    all_data = (c["datasets"] or []) + (c["borrowed_datasets"] or [])
    held = [d for d in all_data
            if any(name in (d.get("splits") or {})
                   for name in ("validation", "test", "eval", "dev", "val",
                                "holdout"))]
    return [
        {"key": "data", "label": "Data",
         # Data with nothing held back is not finished data. Amber rather
         # than green, and the next step says the one thing to do about it.
         "state": ("todo" if not all_data else "done" if held else "warn"),
         "count": len(all_data),
         "borrowed": len(c["borrowed_datasets"]),
         "held_out": len(held),
         # The dataset to do it to, so the card can go straight there rather
         # than to a library the reader then has to search.
         "split_target": next((d["id"] for d in all_data if d not in held), None),
         "next": ("Upload or import a dataset" if not all_data
                  else "Hold back a split — without one there is nothing to "
                       "score on that the model has not already seen"
                  if not held else
                  "Check the data, or add more")},
        {"key": "train", "label": "Train",
         "state": "active" if any(r["kind"] != "evaluate" for r in active)
                  else "done" if done_runs else "todo",
         "count": len(c["training"]),
         "best": {"job_id": best[0]["id"], "name": best[0]["name"],
                  "metric": best[2]} if best else None,
         "next": ("Start a training run" if not c["training"]
                  else "Run again with a change, or sweep a setting")},
        {"key": "evaluate", "label": "Evaluate",
         "state": "done" if scored else "todo",
         "count": len(c["prompt_sets"]),
         "next": (("Take a prompt set from the held-out split — the rows "
                   "nothing trained on" if held else
                   "Hold a split back first, then take a prompt set from it")
                  if not c["prompt_sets"] else
                  "Score the latest run against its base" if not scored else
                  "Score the next run on the same set")},
        {"key": "benchmark", "label": "Benchmark",
         "state": "done" if benched else "todo",
         "count": len(c["benchmarks"]),
         "next": "Run MMLU or GSM8K against the best run and its base"},
        {"key": "publish", "label": "Publish",
         "state": "done" if c["library"] else "todo",
         "count": len(c["library"]),
         "next": ("Publish the best run to the model library, or to Hugging Face"
                  if done_runs else "Train something first")},
    ]


# ---------------------------------------------------------------------------
# The graph: what was made from what
# ---------------------------------------------------------------------------

KIND_NODE = {
    "finetune_llm": ("training", "✦"), "pretrain_llm": ("training", "✦"),
    "finetune_vision_cls": ("training", "✦"),
    "generate_dataset": ("writing", "✎"), "evaluate": ("scoring", "◎"),
    "export_gguf": ("export", "⬓"), "upload": ("export", "☁"),
    "merge_adapter": ("training", "⊕"),
}


def graph(user: dict, project_id: str | None) -> dict:
    """Everything in a project and what it was made from.

    A project's page is a map of stages, which answers "how far along is
    this". It cannot answer the question people actually argue about three
    weeks later -- *which* data went into the good model, whether the set it
    was scored on came out of the rows it trained on, what the published
    version was built from. Every one of those facts is already recorded, on
    the row that resulted: a derived dataset knows its parent, a run knows the
    dataset it read and the run it continued, a prompt set knows the split it
    was taken from, a library entry knows its run.

    Read together they are a directed graph, and it is worth drawing.
    """
    c = contents(user, project_id)
    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    present = set()

    def add(node_id: str, **fields: object) -> None:
        nodes[node_id] = {"id": node_id, **fields}
        present.add(node_id)

    def link(src: str, dst: str, label: str) -> None:
        edges.append({"from": src, "to": dst, "label": label})

    datasets = (c["datasets"] or []) + (c["borrowed_datasets"] or [])
    for d in datasets:
        splits = d.get("splits") or {}
        held = [n for n in splits if n in ("validation", "test", "eval", "dev",
                                           "val", "holdout")]
        add("ds:" + d["id"], kind="dataset", icon="▤", label=d["name"],
            sub="%s rows%s" % (f"{d.get('rows') or 0:,}",
                               " · %s held back" % ", ".join(held) if held else ""),
            href="#/data/" + d["id"], at=d.get("created_at"),
            warn=None if held else "nothing held back",
            borrowed=bool(d.get("borrowed_from")))

    for r in c["runs"]:
        kind, icon = KIND_NODE.get(r["kind"], ("run", "≡"))
        metric = (r.get("summary") or {}).get("primary_metric") or {}
        add("job:" + r["id"], kind=kind, icon=icon, label=r["name"],
            sub=(("%s %s" % (metric.get("label"), round(metric["value"], 4)))
                 if metric.get("value") is not None else r["status"]),
            href="#/jobs/" + r["id"], at=r.get("created_at"),
            status=r["status"])

    for e in c["prompt_sets"] + c["benchmarks"]:
        add("ev:" + e["id"], kind="benchmark" if e.get("is_benchmark") else "eval",
            icon="◈" if e.get("is_benchmark") else "◎", label=e["name"],
            sub="%d scoring%s" % (e.get("scorings") or 0,
                                  "" if e.get("scorings") == 1 else "s"),
            href="#/evals/" + e["id"], at=e.get("created_at"))

    for m in c["library"]:
        add("lib:" + m["id"], kind="published", icon="⬢",
            label="%s %s" % (m["name"], m.get("version") or ""),
            sub="on Hugging Face" if m["location"] == "hf" else "in this studio",
            href="#/models", at=m.get("published_at"))

    # ---- what came from what
    for d in datasets:
        parent = "ds:" + (d.get("parent_id") or "")
        if d.get("parent_id") and parent in present:
            steps = ((d.get("recipe") or {}).get("steps") or [])
            link(parent, "ds:" + d["id"],
                 (steps[0][:40] if steps else "made from"))
        # A dataset a run wrote. `origin` holds the job id for those.
        if ("job:" + str(d.get("origin") or "")) in present:
            link("job:" + d["origin"], "ds:" + d["id"], "wrote")

    for r in c["runs"]:
        cfg = r.get("config") or {}
        if ("ds:" + str(cfg.get("studio_dataset") or "")) in present:
            link("ds:" + cfg["studio_dataset"], "job:" + r["id"],
                 "read" if r["kind"] == "generate_dataset" else "trained on")
        for key, label in (("base_model_job", "continued from"),
                           ("source_job", "from"), ("trained_by", "from")):
            other = "job:" + str(cfg.get(key) or "")
            if other in present and other != "job:" + r["id"]:
                link(other, "job:" + r["id"], label)
                break
        if ("ev:" + str(cfg.get("eval_id") or "")) in present:
            link("ev:" + cfg["eval_id"], "job:" + r["id"], "asked")
        for m in cfg.get("models") or []:
            scored = "job:" + str(m.get("job_id") or "")
            if scored in present:
                link(scored, "job:" + r["id"], "scored")

    for e in c["prompt_sets"] + c["benchmarks"]:
        src = (e.get("source") or {})
        if ("ds:" + str(src.get("dataset_id") or "")) in present:
            link("ds:" + src["dataset_id"], "ev:" + e["id"],
                 "taken from the %s split" % src["split"] if src.get("split")
                 else "taken from")

    for m in c["library"]:
        if ("job:" + m["job_id"]) in present:
            link("job:" + m["job_id"], "lib:" + m["id"], "published as")

    return {"nodes": list(nodes.values()), "edges": edges}


@router.get("/unfiled")
async def unfiled(request: Request) -> dict:
    user = current_user(request)
    c = contents(user, None)
    return {"contents": c, "map": _stage_map(c)}


@router.get("/{project_id}")
async def get_project(request: Request, project_id: str) -> dict:
    user = current_user(request)
    row = _decorate(_project_or_404(request, project_id), user)
    c = contents(user, project_id)
    return {**row, "contents": c, "map": _stage_map(c)}


@router.get("/{project_id}/graph")
async def project_graph(request: Request, project_id: str) -> dict:
    user = current_user(request)
    _project_or_404(request, project_id)
    return graph(user, project_id)


@router.patch("/{project_id}")
async def update_project(request: Request, project_id: str,
                         payload: dict = Body(...)) -> dict:
    user = current_user(request)
    _project_or_404(request, project_id, "edit")
    fields = {}
    if "name" in payload:
        name = (payload.get("name") or "").strip()
        if not name:
            raise HTTPException(400, "A project needs a name.")
        fields["name"] = name[:120]
    for key, cap in (("goal", 500), ("notes", 4000)):
        if key in payload:
            fields[key] = (payload.get(key) or "")[:cap] or None
    if "archived" in payload:
        fields["archived"] = 1 if payload.get("archived") else 0
    db.update_project(project_id, **fields)
    return _decorate(db.get_project(project_id), user)


@router.delete("/{project_id}")
async def delete_project(request: Request, project_id: str) -> dict:
    _project_or_404(request, project_id, "own")
    db.clear_shares("project", project_id)
    db.delete_project(project_id)
    return {"ok": True, "note": "The project is gone; what was in it is unfiled, "
                                "not deleted."}


@router.post("/{project_id}/file")
async def file_into(request: Request, project_id: str,
                    payload: dict = Body(...)) -> dict:
    """Put a run, dataset or prompt set into this project.

    Whoever may edit the project may file into it, and the thing filed must
    be one they may at least see -- filing is a pointer, not a copy, and a
    pointer to something you cannot see would be a way to find out it exists.
    """
    user = current_user(request)
    _project_or_404(request, project_id, "edit")
    kind = payload.get("kind")
    rid = payload.get("id") or ""
    getter = {"job": db.get_job, "dataset": db.get_dataset, "eval": db.get_eval}.get(kind)
    if not getter:
        raise HTTPException(400, "File a run, a dataset or a prompt set.")
    row = getter(rid)
    if not row or not db.access_level(kind, rid, row.get("owner_id"), user):
        raise HTTPException(404, "No such %s." % kind)
    db.assign_project(kind, rid, project_id)
    return {"ok": True}


@router.post("/unfile")
async def unfile(request: Request, payload: dict = Body(...)) -> dict:
    user = current_user(request)
    kind = payload.get("kind")
    rid = payload.get("id") or ""
    getter = {"job": db.get_job, "dataset": db.get_dataset, "eval": db.get_eval}.get(kind)
    if not getter:
        raise HTTPException(400, "Unfile a run, a dataset or a prompt set.")
    row = getter(rid)
    if not row:
        raise HTTPException(404, "No such %s." % kind)
    if row.get("project_id"):
        _project_or_404(request, row["project_id"], "edit")
    db.assign_project(kind, rid, None)
    return {"ok": True}
