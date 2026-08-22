"""What a runner needs in order to serve one particular finished run.

Lifted out of the FastAPI application so that more than one caller can use it
without importing the app: the playground asks for it when you type a message,
and an evaluation asks for it for every model it is about to score. Those two
must produce byte-identical prompts, or the scores describe a model nobody is
actually talking to.
"""
from __future__ import annotations

from . import hub


def chat_spec(job: dict) -> dict:
    """Everything a runner needs to serve this particular result.

    The important field is `format`: the *exact* format the run was trained
    with, carried through unchanged. Guessing it again here would let the
    playground drift from the model, and a model given a shape it never saw
    looks broken when it is not.
    """
    cfg = job["config"]
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
        "format": fmt,
        "style": hub.formatting.conversation_style(fmt),
        "system_prompt": cfg.get("system_prompt") or "",
        # Whether this run was actually taught to reason, so the playground
        # offers the toggle only where it means something.
        "reasoning": bool(fmt.get("reasoning")),
    }
