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
import shutil
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

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


def _unpack(job_id: str, into: Path) -> Path:
    src = config.ARTIFACT_DIR / ("%s.zip" % job_id)
    if not src.exists():
        raise ValueError("This run has no saved model to publish.")
    with zipfile.ZipFile(src) as z:
        # Refuse a zip that would write outside the directory. These archives
        # are written by our own runner, so this should never fire -- which is
        # exactly the kind of assumption worth checking before extracting.
        for member in z.namelist():
            target = (into / member).resolve()
            if not str(target).startswith(str(into.resolve())):
                raise ValueError("This model archive is not safe to unpack.")
        z.extractall(into)
    return into


def _publish_blocking(token: str, repo_id: str, folder: Path, private: bool,
                      kind: str, message: str) -> str:
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type=kind, private=private,
                    exist_ok=True)
    api.upload_folder(repo_id=repo_id, repo_type=kind, folder_path=str(folder),
                      commit_message=message)
    return "https://huggingface.co/%s%s" % (
        "" if kind == "model" else "datasets/", repo_id)


async def publish_job(user: dict, job: dict, repo_id: str, private: bool,
                      message: str = "") -> dict:
    """Upload a finished run's model to the user's Hub account."""
    if problem := (repo_problem(repo_id) or can_publish_to(user, repo_id)):
        raise ValueError(problem)
    token = token_for(user)
    if not token:
        raise ValueError("Connect your Hugging Face account first.")

    tmp = Path(tempfile.mkdtemp(prefix="aistudio_pub_"))
    try:
        folder = _unpack(job["id"], tmp)
        _write_card(folder, job, repo_id)
        url = await asyncio.to_thread(
            _publish_blocking, token, repo_id, folder, private, "model",
            message or "Trained with AI Studio")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return {"url": url, "repo_id": repo_id, "private": private}


async def publish_dataset(user: dict, dataset: dict, path: Path, repo_id: str,
                          private: bool, message: str = "") -> dict:
    if problem := (repo_problem(repo_id) or can_publish_to(user, repo_id)):
        raise ValueError(problem)
    token = token_for(user)
    if not token:
        raise ValueError("Connect your Hugging Face account first.")

    tmp = Path(tempfile.mkdtemp(prefix="aistudio_ds_"))
    try:
        shutil.copy(path, tmp / "data.jsonl")
        (tmp / "README.md").write_text(_dataset_card(dataset, repo_id),
                                       encoding="utf-8")
        url = await asyncio.to_thread(
            _publish_blocking, token, repo_id, tmp, private, "dataset",
            message or "Prepared with AI Studio")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return {"url": url, "repo_id": repo_id, "private": private}


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

def _write_card(folder: Path, job: dict, repo_id: str) -> None:
    """Put a real README at the top of the repo.

    The runner already writes one describing how to load the model. This adds
    the YAML front matter the Hub needs to file it correctly, and the training
    facts -- which are otherwise only visible inside this studio.
    """
    cfg = job.get("config") or {}
    scratch = job.get("kind") == "pretrain_llm"
    existing = ""
    readme = folder / "README.md"
    if readme.exists():
        existing = readme.read_text(encoding="utf-8", errors="replace")
        existing = existing.split("---\n", 2)[-1] if existing.startswith("---") \
            else existing

    tags = ["ai-studio", "text-generation"]
    tags.append("pretrained-from-scratch" if scratch else "lora")
    front = ["---", "library_name: transformers", "tags:"]
    front += ["  - %s" % t for t in tags]
    if not scratch and cfg.get("base_model"):
        front.append("base_model: %s" % cfg["base_model"])
    if cfg.get("dataset"):
        front += ["datasets:", "  - %s" % cfg["dataset"]]
    front += ["pipeline_tag: text-generation", "---", ""]

    stopped = job.get("status") == "cancelled"
    head = ["# %s" % repo_id.split("/")[-1], ""]
    head.append("Trained with [AI Studio](https://github.com/) on %s."
                % (cfg.get("dataset") or "a private dataset"))
    if stopped:
        head += ["", "> **Stopped before the end of its schedule.** It "
                 "completed %s of %s planned steps, so its learning rate never "
                 "finished decaying and it is rougher than the same run taken "
                 "to completion."
                 % (job.get("step"), job.get("total_steps"))]
    head.append("")
    readme.write_text("\n".join(front + head) + existing, encoding="utf-8")


def _dataset_card(dataset: dict, repo_id: str) -> str:
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
