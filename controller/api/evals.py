"""Prompt sets, and the scores models get on them.

The unit here is the **prompt set**, not the score. That is the whole design.
A score belongs to a pair -- these prompts, that model -- and the reason the
prompts are saved as a thing with a name and an owner is so the same ones can
be put to next month's model. Save the score and you have a number nobody can
reproduce; save the prompts and every future model is comparable with every
past one.

Scoring itself happens on a runner, as an ordinary queued job, so it inherits
the queue, the log, the progress bar, the stop button and the sharing rules
rather than reimplementing five of them. See runner/jobs/evaluate.py for what
each measure is worth.
"""
from __future__ import annotations

import json

from fastapi import APIRouter, Body, HTTPException, Request

from common import apimodels, conversation, formatting

from .. import benchmarks as bm, config, datasets as dsets, db, hfaccount, serving
from . import providers
from .security import current_user, require_edit, require_owner, require_view

router = APIRouter(prefix="/api")

# Set by the application once the fleet exists. The routes below queue work,
# and queued work has to wake the scheduler.
FLEET = None

MAX_ITEMS = 500


def _eval_or_404(request: Request, eval_id: str, need: str = "view") -> dict:
    row = db.get_eval(eval_id)
    if not row:
        raise HTTPException(404, "No such prompt set.")
    if need == "own":
        require_owner(request, "eval", row)
    elif need == "edit":
        require_edit(request, "eval", row)
    else:
        require_view(request, "eval", row)
    return row


def _clean_items(raw: object) -> list[dict]:
    """Prompts, as a list of usable rows, or a refusal that says what is wrong."""
    if not isinstance(raw, list):
        raise HTTPException(400, "Prompts must be a list.")
    out = []
    for entry in raw:
        if isinstance(entry, str):
            entry = {"prompt": entry}
        if not isinstance(entry, dict):
            continue
        prompt = str(entry.get("prompt") or "").strip()
        if not prompt:
            continue
        item = {"prompt": prompt,
                "expected": str(entry.get("expected") or "").strip(),
                "note": str(entry.get("note") or "").strip()}
        # What else a prompt may ask to be scored on: the shape the answer
        # must have, and the tool it should call. Kept only when given, so an
        # ordinary set does not grow two empty fields per row.
        for key in ("schema", "expected_tool"):
            value = entry.get(key)
            if isinstance(value, str) and value.strip():
                try:
                    value = json.loads(value)
                except ValueError:
                    if key == "schema":
                        raise HTTPException(
                            400, "The schema on \"%s\" is not valid JSON."
                                 % prompt[:60]) from None
                    value = {"name": value.strip()}
            if isinstance(value, dict) and value:
                item[key] = value
        out.append(item)
    if not out:
        raise HTTPException(400, "There are no prompts in this set.")
    if len(out) > MAX_ITEMS:
        raise HTTPException(
            400, "A prompt set is limited to %d prompts. Every model you "
                 "compare has to answer all of them, so a set this large "
                 "turns a comparison into an overnight job." % MAX_ITEMS)
    return out


def _decorate(row: dict, user: dict) -> dict:
    row["access"] = db.access_level("eval", row["id"], row.get("owner_id"), user)
    row["mine"] = row.get("owner_id") == user["id"]
    owner = db.get_user(row["owner_id"]) if row.get("owner_id") else None
    row["owner"] = db.public_user(owner) if owner else None
    row["shares"] = db.list_shares("eval", row["id"])
    return row


# ---------------------------------------------------------------- prompts

@router.get("/evals")
async def list_evals(request: Request) -> list[dict]:
    return db.visible_evals(current_user(request))


def _project_of_first_model(models: list[dict]) -> str | None:
    for m in models:
        job = db.get_job(m.get("job_id") or "") if m.get("job_id") else None
        if job and job.get("project_id"):
            return job["project_id"]
    return None


def _project_of(request, payload: dict) -> str | None:
    """Which project this prompt set belongs to, if the page said."""
    pid = (payload.get("project_id") or "").strip()
    if not pid:
        return None
    row = db.get_project(pid)
    if not row or not db.access_level("project", pid, row.get("owner_id"),
                                      current_user(request)):
        raise HTTPException(404, "No such project.")
    return pid


@router.post("/evals")
async def create_eval(request: Request, payload: dict = Body(...)) -> dict:
    user = current_user(request)
    name = (payload.get("name") or "").strip()
    if not name:
        raise HTTPException(400, "Give this prompt set a name.")
    items = _clean_items(payload.get("items") or [])
    # Tools the prompts may call, declared once for the set. Stored with the
    # set rather than per prompt: a tool-calling model is scored on whether
    # it picks the right one from the same list every time.
    source = None
    if tools := _clean_tools(payload.get("tools")):
        source = {"tools": tools}
    eid = db.create_eval(user["id"], name, items, payload.get("notes") or "",
                         source, project_id=_project_of(request, payload))
    return _decorate(db.get_eval(eid), user)


def _clean_tools(raw: object) -> list[dict]:
    if isinstance(raw, str) and raw.strip():
        try:
            raw = json.loads(raw)
        except ValueError:
            raise HTTPException(400, "The tools are not valid JSON.") from None
    if not isinstance(raw, list):
        return []
    out = []
    for t in raw:
        if not isinstance(t, dict):
            continue
        fn = t.get("function") if "function" in t else t
        if isinstance(fn, dict) and fn.get("name"):
            out.append({"name": fn["name"], "description": fn.get("description") or "",
                        "parameters": fn.get("parameters") or {}})
    return out[:40]


@router.post("/evals/from-dataset")
async def eval_from_dataset(request: Request, payload: dict = Body(...)) -> dict:
    """Turn rows of a dataset into a prompt set.

    The obvious source, and the one worth encouraging: the held-out part of
    the data a model was fine-tuned on is exactly a set of prompts with known
    good answers. Splitting a dataset and evaluating on the half that was
    never trained on is the difference between measuring a model and
    measuring its memory.
    """
    user = current_user(request)
    ds = db.get_dataset(payload.get("dataset_id") or "")
    if not ds:
        raise HTTPException(404, "No such dataset.")
    require_view(request, "dataset", ds)

    limit = min(int(payload.get("limit") or 50), MAX_ITEMS)
    prompt_field = payload.get("prompt_field")
    answer_field = payload.get("answer_field")

    # Which split. Given one, that one; given none, the held-out split when
    # the dataset has one, because that is the whole point of the exercise.
    # Reading the first N rows of the file regardless -- which is what this
    # did -- took the training split nearly every time and then let the model
    # card say the numbers were on rows the model never saw.
    splits = ds.get("splits") or {}
    split = (payload.get("split") or "").strip() or _held_out_split(splits)
    if split and split not in splits:
        raise HTTPException(400, "That dataset has no split called %r. It has: %s."
                            % (split, ", ".join(splits) or "none"))
    rows = list(dsets.iter_rows(ds["id"], limit, split or None))
    if not rows:
        raise HTTPException(400, "That dataset has no rows to take%s."
                            % (" in the %s split" % split if split else ""))

    fmt = formatting.resolve_format(ds.get("format") or {})
    if not prompt_field and (fmt.get("mode") == "chat" or "messages" in rows[0]):
        # A conversation dataset: the prompt is the last thing the user said
        # and the answer is what the data says came next. The flat-column
        # guess below would have found no prompt column and refused.
        pairs = []
        for r in rows:
            conv, _ = conversation.repair(conversation.from_row(r, fmt))
            prompt, expected = conversation.split_for_trial(conv)
            asked = next((m.get("content") for m in reversed(prompt)
                          if m.get("role") == "user"), None)
            answer = next((m.get("content") for m in expected
                           if m.get("role") == "assistant"), "")
            if asked:
                pairs.append({"prompt": asked, "expected": answer or ""})
        items = _clean_items(pairs)
        how = "last user turn -> first assistant reply"
    else:
        if not prompt_field:
            prompt_field = next((f for f in ("instruction", "prompt", "question",
                                             "input", "text")
                                 if f in rows[0]), None)
        if not answer_field:
            answer_field = next((f for f in ("output", "response", "answer",
                                             "completion")
                                 if f in rows[0] and f != prompt_field), None)
        if not prompt_field:
            raise HTTPException(
                400, "Could not tell which column holds the prompt. Its columns "
                     "are: %s." % ", ".join(map(str, rows[0].keys())))
        items = _clean_items([
            {"prompt": r.get(prompt_field), "expected": r.get(answer_field) or ""}
            for r in rows])
        how = ("%s -> %s" % (prompt_field, answer_field) if answer_field
               else prompt_field)

    held_out = bool(split) and split != dsets.DEFAULT_SPLIT
    name = (payload.get("name") or "").strip() or ("%s%s (%d prompts)"
        % (ds["name"], " · " + split if split else "", len(items)))
    notes = ("Taken from the %s of the dataset \"%s\" (%s). %s"
             % ("%s split" % split if split else "whole", ds["name"], how,
                "A held-out split: a fair test of any model that trained on "
                "the rest of this dataset." if held_out else
                "Only meaningful as a measure if the models being scored were "
                "not trained on these rows."))
    source = {"dataset_id": ds["id"], "dataset_name": ds["name"],
              "split": split or None, "held_out": held_out}
    eid = db.create_eval(user["id"], name, items, notes, source,
                         project_id=_project_of(request, payload)
                                    or ds.get("project_id"))
    return _decorate(db.get_eval(eid), user)


def _held_out_split(splits: dict) -> str:
    """The split nothing trains on, when the dataset has one."""
    for name in ("validation", "test", "eval", "dev", "val", "holdout"):
        if name in splits:
            return name
    return next((n for n in splits if n != dsets.DEFAULT_SPLIT
                 and n.lower().startswith(("val", "test", "eval", "dev"))), "")


@router.get("/evals/{eval_id}")
async def get_eval(request: Request, eval_id: str) -> dict:
    row = _eval_or_404(request, eval_id)
    out = _decorate(row, current_user(request))
    out["scores"] = db.list_scores(eval_id)
    return out


@router.patch("/evals/{eval_id}")
async def update_eval(request: Request, eval_id: str,
                      payload: dict = Body(...)) -> dict:
    row = _eval_or_404(request, eval_id, "edit")
    fields: dict = {}
    if "name" in payload:
        if not (payload.get("name") or "").strip():
            raise HTTPException(400, "A prompt set needs a name.")
        fields["name"] = payload["name"].strip()
    if "notes" in payload:
        fields["notes"] = payload.get("notes") or ""
    if "items" in payload:
        new_items = _clean_items(payload["items"])
        if db.list_scores(eval_id) and new_items != row["items"]:
            # Changing the prompts after models have been scored on them would
            # leave a table of numbers that look comparable and are not. The
            # honest move is a new set, which keeps the old scores meaningful.
            raise HTTPException(
                409, "Models have already been scored on these prompts. "
                     "Changing them now would make those scores describe "
                     "questions that were never asked. Copy this set instead.")
        fields["items"] = new_items
    db.update_eval(eval_id, **fields)
    return _decorate(db.get_eval(eval_id), current_user(request))


@router.post("/evals/{eval_id}/copy")
async def copy_eval(request: Request, eval_id: str,
                    payload: dict = Body(default=None)) -> dict:
    row = _eval_or_404(request, eval_id)
    user = current_user(request)
    name = ((payload or {}).get("name") or "").strip() or (row["name"] + " (copy)")
    eid = db.create_eval(user["id"], name, row["items"], row.get("notes") or "")
    return _decorate(db.get_eval(eid), user)


@router.delete("/evals/{eval_id}")
async def delete_eval(request: Request, eval_id: str) -> dict:
    _eval_or_404(request, eval_id, "own")
    db.clear_shares("eval", eval_id)
    db.delete_eval(eval_id)
    return {"ok": True}


# ----------------------------------------------------------------- scores

@router.get("/evals/{eval_id}/scores")
async def get_scores(request: Request, eval_id: str) -> list[dict]:
    _eval_or_404(request, eval_id)
    return db.list_scores(eval_id)


@router.get("/evals/{eval_id}/scores/{score_id}")
async def get_score(request: Request, eval_id: str, score_id: str) -> dict:
    _eval_or_404(request, eval_id)
    score = db.get_score(score_id)
    if not score or score["eval_id"] != eval_id:
        raise HTTPException(404, "No such scoring.")
    return score


@router.delete("/evals/{eval_id}/scores/{score_id}")
async def delete_score(request: Request, eval_id: str, score_id: str) -> dict:
    _eval_or_404(request, eval_id, "edit")
    score = db.get_score(score_id)
    if not score or score["eval_id"] != eval_id:
        raise HTTPException(404, "No such scoring.")
    db.delete_score(score_id)
    return {"ok": True}


MAX_MODELS = 8


def _runs_to_score(request: Request, user: dict, wanted: list) -> list[dict]:
    """The studio's own runs, as things that can be put to a prompt set."""
    models = []
    for job_id in wanted:
        job = db.get_job(job_id)
        if not job:
            raise HTTPException(404, "One of those runs does not exist.")
        # The same 404-not-403 rule as everywhere else: a run you cannot see
        # must not be distinguishable from one that is not there.
        if not db.access_level("job", job_id, job.get("owner_id"), user):
            raise HTTPException(404, "One of those runs does not exist.")
        if not (config.ARTIFACT_DIR / ("%s.zip" % job_id)).exists():
            raise HTTPException(
                400, "\"%s\" has no saved model, so there is nothing to "
                     "score." % job["name"])
        cfg = serving.resolved_config(job)
        models.append({
            "ref": "job:" + job_id,
            "source": "run",
            "job_id": job_id,
            "name": job["name"],
            "spec": serving.chat_spec(job),
            # Its OWN system prompt, not one borrowed from whichever model
            # happened to be first in the list. A run trained with a system
            # prompt and scored without it is being asked to do a job it was
            # never told about, and it answers worse for a reason that has
            # nothing to do with the training.
            "system_prompt": cfg.get("system_prompt") or "",
        })
    return models


def _baselines(request: Request, user: dict, wanted: list) -> list[dict]:
    """Models with no run behind them, put to the same prompts.

    Two kinds, and they behave differently enough to be worth knowing apart:

    * **From the Hub.** Downloaded onto the machine and loaded like any other
      model, so every measure works on it, the loss on the expected answer
      included. This is the baseline that answers "did the training help".
    * **Behind an API.** Reached over the network. It can be asked a question
      and its answer scored for overlap, exactness and JSON validity -- but
      not for loss, because that needs the model's own probabilities and no
      hosted provider hands those out. Said plainly rather than left as an
      empty cell, because an empty cell in the column everything else is
      ranked by looks like a failure.
    """
    out = []
    for entry in wanted:
        if isinstance(entry, str):
            entry = {"source": "hub", "model": entry}
        if not isinstance(entry, dict):
            continue
        source = (entry.get("source") or "hub").strip()
        name = (entry.get("model") or "").strip()
        if not name:
            raise HTTPException(400, "A baseline needs a model to name.")

        if source == "hub":
            # The format of the run this is a baseline *for*, when one was
            # named: the base is then asked the question in the shape its
            # descendant was trained to answer, which is the comparison that
            # isolates what the training added rather than measuring two
            # different prompt formats against each other.
            fmt = None
            if like := entry.get("like_run"):
                job = db.get_job(like)
                if job and db.access_level("job", like, job.get("owner_id"), user):
                    fmt = (serving.resolved_config(job).get("format") or None)
            spec = serving.hub_spec(name, fmt, hfaccount.token_for(user))
            out.append({"ref": "hub:" + name, "source": "hub", "job_id": "",
                        "name": name, "spec": spec, "system_prompt": ""})
            continue

        if source == "api":
            conn = providers.connection(user, entry.get("provider"))
            if not conn:
                raise HTTPException(
                    400, "That provider is not connected to your account. "
                         "Connect it on your account page first.")
            if problem := apimodels.problems(conn):
                raise HTTPException(400, problem)
            model = apimodels.model_name(conn, name)
            if not model:
                raise HTTPException(400, "Which model at %s should answer?"
                                    % apimodels.describe(conn))
            out.append({
                "ref": "api:%s:%s" % (conn.get("provider") or "", model),
                "source": "api", "job_id": "",
                "name": "%s · %s" % (apimodels.describe(conn), model),
                "model": model, "connection": conn,
                "system_prompt": "",
            })
            continue

        raise HTTPException(400, "A baseline is either a model on the Hub or "
                                 "one behind a connected API.")
    return out


@router.post("/evals/{eval_id}/run")
async def run_eval(request: Request, eval_id: str,
                   payload: dict = Body(...)) -> dict:
    """Queue a job that puts these prompts to each of the chosen models."""
    row = _eval_or_404(request, eval_id)
    user = current_user(request)

    models = _runs_to_score(request, user, payload.get("model_job_ids") or []) \
        + _baselines(request, user, payload.get("baselines") or [])
    if not models:
        raise HTTPException(400, "Choose at least one model to score.")
    if len(models) > MAX_MODELS:
        raise HTTPException(
            400, "Score at most %d models at a time. Each one is loaded onto "
                 "the card in turn, and a longer list is a job that runs for "
                 "hours before it tells you anything." % MAX_MODELS)
    seen = set()
    for m in models:
        if m["ref"] in seen:
            raise HTTPException(400, "%s is in the list twice." % m["name"])
        seen.add(m["ref"])

    # One system prompt for everybody, or each model's own. The override is
    # the exception and it is recorded as one: a score taken under a system
    # prompt the model was not trained with is a different measurement, and
    # six months later nothing else would say which had happened.
    override = (payload.get("system_prompt") or "").strip()
    if override:
        for m in models:
            m["system_prompt"] = override

    # A judge: a hosted model grading every answer one to five. Resolved to a
    # usable connection here, like a hosted baseline, and stripped from every
    # response that hands the job back to a browser.
    judge = None
    if isinstance(payload.get("judge"), dict) and payload["judge"].get("provider"):
        j = payload["judge"]
        conn = providers.connection(user, j.get("provider"))
        if not conn:
            raise HTTPException(400, "The judge's provider is not connected to "
                                     "your account.")
        if problem := apimodels.problems(conn):
            raise HTTPException(400, problem)
        model = apimodels.model_name(conn, j.get("model"))
        if not model:
            raise HTTPException(400, "Which model at %s should judge?"
                                % apimodels.describe(conn))
        judge = {"connection": conn, "model": model,
                 "label": "%s · %s" % (apimodels.describe(conn), model),
                 "rubric": (j.get("rubric") or "")[:2000]}

    cfg = {
        "eval_id": eval_id,
        "eval_name": row["name"],
        "items": row["items"],
        "models": models,
        "judge": judge,
        # Tools the set declares, for scoring tool calls. Stored on the set
        # as JSON text; sent to the runner parsed.
        "tools": (row.get("source") or {}).get("tools") or None,
        "max_new_tokens": min(int(payload.get("max_new_tokens") or 200), 512),
        "temperature": float(payload.get("temperature") or 0.0),
        "system_prompt": override,
        "system_prompt_override": bool(override),
    }
    # A benchmark set carries a recipe instead of prompts: the questions are
    # on the Hub and the runner fetches them. Everything above is unchanged,
    # which is the point of a benchmark being a prompt set at all -- adding a
    # model to one months later goes through the same button.
    if recipe := ((row.get("source") or {}).get("recipe")):
        bench = bm.get(recipe.get("benchmark") or "")
        if not bench:
            raise HTTPException(
                400, "This set was built from a benchmark called %r, which "
                     "this studio no longer describes."
                     % recipe.get("benchmark"))
        cfg["benchmark"] = bench
        cfg["recipe"] = recipe
        cfg.pop("items", None)
        if bench["protocol"] == "multiple_choice":
            hosted = [m["name"] for m in models if m["source"] == "api"]
            if hosted:
                raise HTTPException(
                    400, "%s is decided by the model's own probability for "
                         "each answer, and a model reached over the network "
                         "does not expose those. Leave %s out of this one."
                         % (bench["label"], ", ".join(hosted)))

    if pinned := payload.get("runner_id"):
        cfg["required_runner"] = pinned
    # Nothing here needs a graphics card if every model being scored is behind
    # somebody else's. A comparison of two hosted models should not sit in the
    # queue waiting for the one machine with a GPU in it.
    if all(m["source"] == "api" for m in models):
        cfg["allow_cpu"] = True
    # Or when asked. A studio with no card at all could not score anything,
    # and a small model on a few dozen prompts is minutes on a processor. The
    # scheduler still tries every machine with a card first; only the
    # kind-restricted CPU box is asked ahead of them, and only for the kinds
    # it was set aside for.
    if payload.get("allow_cpu"):
        cfg["allow_cpu"] = True

    name = "%s on %d model%s" % (row["name"], len(models),
                                 "" if len(models) == 1 else "s")
    # A scoring run belongs to the same project as the prompt set it runs, or
    # to the project of the model it is scoring. Either is better than the
    # unfiled pile, and the prompt set is the more specific of the two.
    jid = db.create_job(name, "evaluate", cfg, owner_id=user["id"],
                        project_id=row.get("project_id")
                                   or _project_of_first_model(models))
    db.add_log(jid, "Queued: %d prompt%s against %s."
               % (len(row["items"]), "" if len(row["items"]) == 1 else "s",
                  ", ".join(m["name"] for m in models)))
    if FLEET is not None:
        await FLEET.broadcast_ui({"type": "jobs_changed"})
        FLEET.wake()
    return {"id": jid}


def record_scores(job: dict, summary: dict) -> int:
    """File an evaluation run's results against the prompt set it used.

    Called from the scheduler when the run reports done. Kept out of the
    eval_runs table it might otherwise have earned: the comparison has to
    outlive the job that produced it, so it hangs off the prompt set instead
    of off the run.
    """
    eval_id = summary.get("eval_id")
    if not eval_id or not db.get_eval(eval_id):
        return 0
    written = 0
    cfg = job.get("config") or {}
    # The machine's name, not its id. A score is read months later and
    # "controller-cpu" is a thing somebody recognises; "run_be7c0e2bac7d" is
    # not, and the machine it names may not exist by then either.
    runner = db.get_runner(job.get("runner_id") or "") if job.get("runner_id") else None
    for score in summary.get("scores") or []:
        if score.get("metrics", {}).get("error"):
            continue
        # The verdict and whether it separated anything belong to the scoring,
        # not to one model in it -- but they are stored per row because a row
        # is what outlives the run, and the table that reads them has to know
        # whether it may draw a winner.
        metrics = {**(score.get("metrics") or {}),
                   "verdict": summary.get("verdict"),
                   "ranking_decisive": bool(summary.get("decisive")),
                   # Which measure the verdict ranked on. A scoring that
                   # included a hosted model may have had to fall back from
                   # the loss, and the table must mark the winner in the
                   # column the verdict is actually talking about.
                   "ranked_by": summary.get("ranked_by")}
        db.record_score(
            eval_id, score.get("model_job_id") or "", job["id"], metrics,
            score.get("items") or [],
            model_ref=score.get("ref") or "",
            model_label=score.get("name") or "",
            # What produced these numbers. Two scorings of one model under
            # different generation settings are two different measurements,
            # and the table that puts them in adjacent rows has to be able to
            # say so.
            settings={
                "max_new_tokens": cfg.get("max_new_tokens"),
                "temperature": cfg.get("temperature"),
                "system_prompt": score.get("system_prompt") or "",
                "system_prompt_override": bool(cfg.get("system_prompt_override")),
                "source": score.get("source") or "run",
                # Which judge, when there was one: a score from a judge is a
                # score from that judge.
                "judge": (cfg.get("judge") or {}).get("label"),
                "runner": (runner or {}).get("name") or job.get("runner_id"),
            })
        written += 1
    return written


# ------------------------------------------------------------- benchmarks
#
# A benchmark is a prompt set whose questions come from a public dataset
# rather than from you. Making it a prompt set rather than a thing of its own
# is what gives it, for free, the scores table, the trend over time, the
# champion, sharing and the "add another model to this" button -- all of which
# a benchmark wants exactly as much as a hand-written set does.


@router.get("/benchmarks")
async def list_benchmarks(request: Request) -> dict:
    current_user(request)
    return {
        "benchmarks": bm.public(),
        # Named rather than omitted. A list of benchmarks with no HumanEval in
        # it reads as an oversight; a list that says why it is missing reads
        # as a decision, which it is.
        "unavailable": [{"name": k, "why": v} for k, v in bm.UNAVAILABLE.items()],
        "default_sample": bm.DEFAULT_SAMPLE,
        "caveat": (
            "These are this studio's own measurements. A published score is "
            "the output of one particular harness with its own wording, its "
            "own worked examples and its own normalisation, and those choices "
            "move a benchmark by several points -- more than the gap between "
            "most models. The recipe used here is recorded with every result, "
            "so these numbers are exactly comparable to each other and only "
            "roughly comparable to a model card."),
    }


def _existing_set(user: dict, recipe: dict) -> dict | None:
    """A set already built for this exact recipe.

    Reused rather than duplicated so that scoring next month's model on MMLU
    lands on the same page as last month's, with the same questions and a line
    between them. A second set with the same name and a different seed would
    look identical and compare nothing.
    """
    for row in db.visible_evals(user):
        got = ((row.get("source") or {}).get("recipe")) or {}
        if got == recipe:
            return db.get_eval(row["id"])
    return None


@router.post("/benchmarks/run")
async def run_benchmark(request: Request, payload: dict = Body(...)) -> dict:
    """Queue a published benchmark against one or more models."""
    user = current_user(request)
    bench = bm.get(payload.get("benchmark") or "")
    if not bench:
        raise HTTPException(404, "No benchmark by that name.")

    shots = payload.get("shots")
    shots = bench["shots"] if shots is None else max(0, min(int(shots), 25))
    sample = int(payload.get("sample") or bm.DEFAULT_SAMPLE)
    sample = max(20, min(sample, min(bm.MAX_SAMPLE, bench["size"])))
    seed = int(payload.get("seed") or 1234)
    recipe = bm.recipe(bench, shots, sample, seed)

    models = _runs_to_score(request, user, payload.get("model_job_ids") or []) \
        + _baselines(request, user, payload.get("baselines") or [])
    if not models:
        raise HTTPException(400, "Choose at least one model to score.")
    if len(models) > MAX_MODELS:
        raise HTTPException(
            400, "Score at most %d models at a time." % MAX_MODELS)

    row = _existing_set(user, recipe)
    if not row:
        eid = db.create_eval(
            user["id"], bm.title(bench, shots, sample), [],
            "%s %s Questions come from %s; they are not stored here, and the "
            "same %s are asked every time this recipe is run."
            % (bench["what"], bench.get("published") or "",
               bench["dataset"], f"{sample:,}"),
            {"benchmark": bench["id"], "recipe": recipe},
            project_id=_project_of(request, payload))
        row = db.get_eval(eid)

    return await run_eval(request, row["id"], {
        "model_job_ids": payload.get("model_job_ids") or [],
        "baselines": payload.get("baselines") or [],
        "runner_id": payload.get("runner_id"),
    })
