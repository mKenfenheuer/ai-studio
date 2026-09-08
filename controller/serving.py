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
#
# `merge_adapter` is here for the runs that already exist. Merging stopped
# being a run of its own -- a fine-tune now merges its own adapter as its last
# step and keeps both artifacts -- but the merges made before that are still
# models, still servable and still publishable, and nothing about them should
# stop working because the way they are made changed.
LEGACY_MODEL_KINDS = ("merge_adapter",)
MODEL_KINDS = ("pretrain_llm", "finetune_llm") + LEGACY_MODEL_KINDS

# What a merge inherits from the fine-tune it was made from. Legacy: only the
# separate merge runs described above ever needed this.
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


# ---------------------------------------------------------------- baselines
#
# "Is it better than what I started from?" is the question a finished run
# raises, and until now nothing here could answer it: only a run in this
# studio could be scored, so every comparison was between two of your own
# models and never against the thing they were built out of. A baseline is a
# model that has no run behind it -- one off the Hub, or one behind somebody
# else's API -- put to exactly the same prompts.


def hub_spec(model_id: str, fmt: dict | None = None,
             hf_token: str | None = None) -> dict:
    """What a runner needs to serve a model straight off the Hub.

    `job_id` is a reference rather than a run id, because it is what the
    runner keys its resident models by -- two baselines and three runs on one
    card must not collide, and `hub:` cannot be mistaken for a job.

    The format matters more here than it looks. Given none, the model's own
    chat template is used, which is the right way to talk to an instruct model
    and the only way to talk to one this studio has never seen. Given one --
    the format of the fine-tune this is a baseline *for* -- the base is asked
    the question in the same shape its descendant was trained to answer, which
    is the comparison that actually isolates what the training added.
    """
    model_id = (model_id or "").strip()
    return {
        "job_id": "hub:" + model_id,
        "hub_model": model_id,
        "kind": "hub",
        "base_model": model_id,
        "params_b": hub.params_from_name(model_id),
        "format": dict(fmt) if fmt else {"mode": "chat",
                                         "use_model_template": True},
        "system_prompt": "",
        "hf_token": hf_token or None,
    }


def base_model_of(job: dict) -> str:
    """The Hub model this run was built from, if it was built from one."""
    cfg = resolved_config(job)
    # A run fine-tuned from another run in this studio has no Hub base of its
    # own worth offering: the baseline to compare it against is that run.
    if cfg.get("base_model_job"):
        return ""
    return (cfg.get("base_model") or "").strip()
