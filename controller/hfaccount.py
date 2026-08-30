"""Each user's own Hugging Face account: connecting it, and using it.

The studio already talked to the Hub anonymously, or with one shared token set
by whoever ran the container. That is fine for *reading* public models. It is
the wrong shape as soon as the studio starts writing: a model published from
here should appear under the account of the person who trained it, using a
credential they granted and can revoke, not under a shared token nobody can
attribute.

So a token belongs to a user, is validated the moment it is offered -- an
invalid one should fail while somebody is looking at the screen, not two hours
into a download -- and is stored encrypted with nothing that returns it.

What a token is allowed to do is checked and shown, because "why did publish
fail" is otherwise a support question with no answer visible from the UI. A
read-only token is perfectly good for gated downloads and cannot create a
repository, and the user should be told that before they try.
"""
from __future__ import annotations

import asyncio
import json
import time
import zipfile

import httpx

from . import auth, config, db

HF_API = "https://huggingface.co/api"

# Repository names Hugging Face accepts. Checked here so a bad name is a
# sentence on the form rather than a 400 from a library three layers down.
_REPO_OK = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._")


def token_for(user: dict | None) -> str | None:
    """The Hugging Face token to use on this user's behalf.

    Their own first, then the studio-wide one from the environment. The shared
    token stays as a fallback so an existing single-user install keeps working
    unchanged after accounts arrive.
    """
    if user and user.get("hf_token_enc"):
        if tok := auth.decrypt_secret(user["hf_token_enc"]):
            return tok
    return config.HF_TOKEN


async def whoami(token: str) -> dict:
    """Who this token belongs to, and what it may do.

    Raises ValueError with something a person can act on. Hugging Face answers
    401 for a token that is wrong, revoked, or simply mistyped, and none of
    those are distinguishable from here -- so the message covers all three
    rather than guessing.
    """
    async with httpx.AsyncClient(timeout=20) as c:
        r = await c.get("%s/whoami-v2" % HF_API,
                        headers={"Authorization": "Bearer %s" % token})
    if r.status_code == 401:
        raise ValueError(
            "Hugging Face did not accept that token. Check it was copied in "
            "full, and that it has not been revoked.")
    if r.status_code >= 400:
        raise ValueError("Hugging Face answered %d when checking the token."
                         % r.status_code)
    d = r.json()

    # Fine-grained tokens carry an explicit permission set; classic ones carry
    # a role. Both are read here rather than assuming write access, because
    # assuming it turns "publish" into a failure at the end of an upload.
    access = d.get("auth", {}).get("accessToken", {}) or {}
    role = access.get("role") or d.get("auth", {}).get("type")
    fine = access.get("fineGrained") or {}
    scopes: set[str] = set()
    for group in (fine.get("scoped") or []):
        scopes.update(group.get("permissions") or [])
    scopes.update(fine.get("global") or [])
    can_write = (role == "write") or any(
        s.startswith("repo.write") or s.startswith("repo.content.write")
        or s == "write" for s in scopes)

    return {
        "username": d.get("name"),
        "fullname": d.get("fullname") or d.get("name"),
        "avatar": d.get("avatarUrl"),
        "orgs": [o.get("name") for o in (d.get("orgs") or []) if o.get("name")],
        "can_write": bool(can_write),
        "token_name": access.get("displayName"),
    }


async def connect(user: dict, token: str) -> dict:
    """Validate a token and attach it to this account."""
    token = (token or "").strip()
    if not token:
        raise ValueError("Paste your access token first.")
    info = await whoami(token)
    db.update_user(
        user["id"],
        hf_token_enc=auth.encrypt_secret(token),
        hf_username=info["username"], hf_fullname=info["fullname"],
        hf_avatar=info["avatar"], hf_orgs=json.dumps(info["orgs"]),
        hf_can_write=1 if info["can_write"] else 0, hf_checked_at=time.time())
    return info


def disconnect(user: dict) -> None:
    db.update_user(user["id"], hf_token_enc=None, hf_username=None,
                   hf_fullname=None, hf_avatar=None, hf_orgs=None,
                   hf_can_write=0, hf_checked_at=None)


# ---------------------------------------------------------------------------
# Listing what the user already has
# ---------------------------------------------------------------------------

async def my_repos(user: dict, kind: str = "models", author: str | None = None,
                   limit: int = 100) -> list[dict]:
    """This user's own models or datasets on the Hub."""
    token = token_for(user)
    who = author or user.get("hf_username")
    if not who:
        raise ValueError("Connect a Hugging Face account first.")
    path = "datasets" if kind == "datasets" else "models"
    headers = {"Authorization": "Bearer %s" % token} if token else {}
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.get("%s/%s" % (HF_API, path),
                        params={"author": who, "limit": limit,
                                "sort": "lastModified", "direction": -1},
                        headers=headers)
    r.raise_for_status()
    out = []
    for m in r.json():
        out.append({
            "id": m.get("id"),
            "private": bool(m.get("private")),
            "gated": bool(m.get("gated")),
            "downloads": m.get("downloads", 0),
            "likes": m.get("likes", 0),
            "updated": m.get("lastModified"),
            "tags": [t for t in (m.get("tags") or []) if ":" not in t][:6],
            "kind": path,
        })
    return out


# ---------------------------------------------------------------------------
# Publishing
# ---------------------------------------------------------------------------

def repo_problem(repo_id: str) -> str | None:
    if not repo_id or "/" not in repo_id:
        return "Use the form owner/name, for example your-name/my-first-model."
    owner, _, name = repo_id.partition("/")
    if not owner or not name:
        return "Use the form owner/name."
    if len(name) > 96:
        return "That name is too long for Hugging Face."
    bad = sorted({ch for ch in owner + name if ch not in _REPO_OK})
    if bad:
        return ("Hugging Face names may only contain letters, numbers, dots, "
                "dashes and underscores. Remove: %s" % " ".join(bad))
    return None


def can_publish_to(user: dict, repo_id: str) -> str | None:
    """Why this account cannot publish there, or None if it can."""
    if not user.get("hf_token_enc"):
        return "Connect your Hugging Face account first."
    if not user.get("hf_can_write"):
        return ("This token is read-only. Create one with write access to "
                "repositories, then reconnect.")
    owner = repo_id.split("/")[0]
    allowed = {user.get("hf_username")} | set(
        json.loads(user.get("hf_orgs") or "[]"))
    if owner not in allowed:
        return ("You can publish to %s, and nowhere else with this account."
                % ", ".join(sorted(x for x in allowed if x)))
    return None


# What used to live here did the upload inside the request that asked for it,
# and that was the whole bug. A merged 7B model is fourteen gigabytes: creating
# the repository takes a moment and sending the weights takes twenty minutes.
# Long before the Hub was finished the browser had given up, the request task
# was cancelled, and the unpacked folder was deleted out from under the thread
# still reading it. What was left on the Hub was the repository the first line
# had created and nothing else -- an empty repo, no error anywhere, and every
# appearance of having worked.
#
# Uploading is now a job, in runner/jobs/upload.py, with the queue, the log,
# the progress bar and the stop button every other long-running thing here
# already has. What stays in the controller is the half that is genuinely its
# business: whether this account may write to that name, and what the model
# card should say.


def _readme_in_artifact(job_id: str) -> str:
    """The README the training run wrote, if there is one.

    Read straight out of the archive rather than by unpacking it: the file is
    two kilobytes and the archive can be fourteen gigabytes, and the point of
    keeping it is that it explains how to load the model -- which is worth
    more on the Hub than anything this file could generate.
    """
    src = config.ARTIFACT_DIR / ("%s.zip" % job_id)
    if not src.exists():
        return ""
    try:
        with zipfile.ZipFile(src) as z:
            with z.open("README.md") as fh:
                text = fh.read().decode("utf-8", errors="replace")
    except (KeyError, OSError, zipfile.BadZipFile):
        return ""
    # Existing front matter is dropped; the block below replaces it.
    return text.split("---\n", 2)[-1] if text.startswith("---") else text


async def delete_repo(user: dict, repo_id: str, kind: str) -> None:
    """Delete one of the user's own Hub repositories.

    Only ever their own: can_publish_to() restricts the owner to the account
    and its organisations, so this cannot be pointed at somebody else's work
    even by a caller that constructs the request by hand.
    """
    if problem := (repo_problem(repo_id) or can_publish_to(user, repo_id)):
        raise ValueError(problem)
    token = token_for(user)

    def _go() -> None:
        from huggingface_hub import HfApi
        HfApi(token=token).delete_repo(
            repo_id=repo_id, repo_type="dataset" if kind == "datasets" else "model")

    await asyncio.to_thread(_go)


# ---------------------------------------------------------------------------
# Model cards
# ---------------------------------------------------------------------------

def model_card(job: dict, repo_id: str) -> str:
    """The README to put at the top of the repo.

    The runner already writes one describing how to load the model. This adds
    the YAML front matter the Hub needs to file it correctly, and the training
    facts -- which are otherwise only visible inside this studio.
    """
    cfg = job.get("config") or {}
    scratch = job.get("kind") == "pretrain_llm"
    existing = _readme_in_artifact(job["id"])

    tags = ["ai-studio", "text-generation"]
    tags.append("pretrained-from-scratch" if scratch else "lora")
    front = ["---", "library_name: transformers", "tags:"]
    front += ["  - %s" % t for t in tags]
    if not scratch and cfg.get("base_model"):
        front.append("base_model: %s" % cfg["base_model"])
    # `datasets:` in the front matter is a Hub id, and the Hub renders it as a
    # link. A dataset from this studio's own library travels as a URL pointing
    # back at the controller, which is not a Hub id and is not reachable from
    # the internet, so it is named in the text instead of linked to.
    local = cfg.get("dataset_is_local")
    if cfg.get("dataset") and not local:
        front += ["datasets:", "  - %s" % cfg["dataset"]]
    front += ["pipeline_tag: text-generation", "---", ""]

    stopped = job.get("status") == "cancelled"
    head = ["# %s" % repo_id.split("/")[-1], ""]
    trained_on = (cfg.get("dataset_label") if local else cfg.get("dataset")) \
        or "a private dataset"
    head.append("Trained with [AI Studio](https://github.com/) on %s."
                % trained_on)
    if stopped:
        head += ["", "> **Stopped before the end of its schedule.** It "
                 "completed %s of %s planned steps, so its learning rate never "
                 "finished decaying and it is rougher than the same run taken "
                 "to completion."
                 % (job.get("step"), job.get("total_steps"))]
    head.append("")
    return "\n".join(front + head) + existing


def dataset_card(dataset: dict, repo_id: str) -> str:
    fmt = dataset.get("format") or {}
    lines = [
        "---", "license: unknown", "tags:", "  - ai-studio", "---", "",
        "# %s" % dataset.get("name") or repo_id.split("/")[-1], "",
        "%s rows, prepared with AI Studio." % f"{dataset.get('rows') or 0:,}", "",
        "| | |", "|---|---|",
        "| Rows | %s |" % f"{dataset.get('rows') or 0:,}",
        "| Columns | %s |" % ", ".join(dataset.get("columns") or []),
        "| Source | %s |" % (dataset.get("origin") or dataset.get("source")),
    ]
    if fmt.get("mode"):
        lines.append("| Read as | %s |" % fmt["mode"])
    if dataset.get("notes"):
        lines += ["", dataset["notes"]]
    recipe = dataset.get("recipe") or {}
    if recipe.get("steps"):
        lines += ["", "## How it was made", ""]
        lines += ["- %s" % s for s in recipe["steps"]]
    lines += ["", "```python", "from datasets import load_dataset", "",
              'ds = load_dataset("%s", split="train")' % repo_id, "```", ""]
    return "\n".join(lines)
