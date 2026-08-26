"""What a runner needs in order to serve one particular finished run.

Lifted out of the FastAPI application so that more than one caller can use it
without importing the app: the playground asks for it when you type a message,
and an evaluation asks for it for every model it is about to score. Those two
must produce byte-identical prompts, or the scores describe a model nobody is
actually talking to.
"""
from __future__ import annotations

from . import db, hub

# The kinds of run that leave a MODEL behind. Everything else leaves a file --
# a dataset, a scorecard -- worth keeping and not something to serve, chat
# with, or fine-tune from. Shared, because two lists of this drifted apart
# once already: the OpenAI-compatible endpoint filtered on it and the
# playground did not, so a run that wrote a dataset was offered as a model.
MODEL_KINDS = ("pretrain_llm", "finetune_llm", "merge_adapter")

# What a merge inherits from the fine-tune it was made from.
#
# Merging changes where the weights live, not what the model learned to expect.
# A run trained on conversations is still a conversational model after its
# adapter has been folded in -- but the merge job's own config records only how
# to *perform the merge* (which run, which base, which dtype), so every field
# describing how to *talk to* the result was left behind. The visible symptom
# was a chat fine-tune that came out of the merge as a base model: prompted
# with a bare string instead of its own chat format, it saw a shape it had
# never been trained on and answered with nothing at all.
INHERITED = ("format", "system_prompt", "dataset", "dataset_config",
             "dataset_split", "studio_dataset", "base_model")

# Set on a merge whose config has taken what it needs from its source, so the
# lookup below happens once per run rather than once per message. Every merge
# created from now on is written with it already true.
INHERITED_FLAG = "inherited_from_source"


def resolved_config(job: dict) -> dict:
    """This run's config, with a merge's blanks filled in from its source.

    Merges made from now on carry these fields at creation, so there is nothing
    here for them to do. It exists for the ones already on disk, made before
    anything carried them, which would otherwise stay unplayable forever: their
    config is completed on first sight and written back, so the repair happens
    once and not on every message.

    What the merge already records wins. Nothing is overwritten -- the source
    is only ever asked about fields the merge has no answer for.
    """
    cfg = job.get("config") or {}
    if job.get("kind") != "merge_adapter" or cfg.get(INHERITED_FLAG):
        return cfg
    src = db.get_job(cfg.get("source_job") or "") or {}
    src_cfg = src.get("config") or {}
    cfg = {**{k: src_cfg[k] for k in INHERITED if src_cfg.get(k)}, **cfg,
           INHERITED_FLAG: True}
    db.update_job_config(job["id"], cfg)
    job["config"] = cfg
    return cfg


def chat_spec(job: dict) -> dict:
    """Everything a runner needs to serve this particular result.

    The important field is `format`: the *exact* format the run was trained
    with, carried through unchanged. Guessing it again here would let the
    playground drift from the model, and a model given a shape it never saw
    looks broken when it is not.
    """
    cfg = resolved_config(job)
    fmt = dict(cfg.get("format") or {})
    if not fmt and job["kind"] == "pretrain_llm":
        fmt = {"mode": "text"}

    # A merged model is a complete model that happens to have been made by
    # flattening an adapter into its base. The runner only distinguishes
    # "complete model" from "adapter needing a base", so it is told the former
    # -- carrying the merge's own kind through would send it looking for a
    # base model that is already inside the weights.
    kind = "pretrain_llm" if job["kind"] == "merge_adapter" else job["kind"]

    return {
        "job_id": job["id"],
        "kind": kind,
        "base_model": cfg.get("base_model"),
        # An adapter whose base is another run in this studio rather than a
        # Hugging Face id. The runner resolves it the same way it resolves the
        # adapter itself; without this the base would be a job id passed to
        # `from_pretrained`, which fails with a confusing "not found on the
        # Hub" for a model that never was on the Hub.
        "base_model_job": cfg.get("base_model_job"),
        # How large the model is, so the runner can decide what precision it
        # will fit in *before* spending minutes loading it at one that will
        # not. Measured by the runner that trained it where there is one, read
        # off the model's name where there is not.
        "params_b": cfg.get("params_b") or hub.params_from_name(
            cfg.get("base_model") or ""),
        "format": fmt,
        "style": hub.formatting.conversation_style(fmt),
        "system_prompt": cfg.get("system_prompt") or "",
        # Whether this run was actually taught to reason, so the playground
        # offers the toggle only where it means something.
        "reasoning": bool(fmt.get("reasoning")),
    }
