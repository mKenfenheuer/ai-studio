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
    return {
        "datasets": datasets,
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
    return [
        {"key": "data", "label": "Data",
         "state": "done" if c["datasets"] else "todo",
         "count": len(c["datasets"]),
         "next": ("Upload or import a dataset" if not c["datasets"]
                  else "Check the data, hold back a split")},
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
         "next": ("Write a prompt set, or take one from a held-out split"
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
