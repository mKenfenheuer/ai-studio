#!/usr/bin/env python
"""Check the doors for other programs: studio API keys, OAuth, MCP, imports.

Run it with `python scripts/check-mcp.py` from the repository root.

**A key is only as wide as it was made.** A key made for the models must not
open /api, even though it is valid; a key made with studio access must, as
its account and nothing more.

**The OAuth flow has to refuse what it should.** An unknown redirect address,
a missing or wrong PKCE verifier, a code used twice, a refresh token used
after it was rotated -- each of these is a way in for somebody who is not the
person who said "Allow".

**MCP has to be the REST API, not a second one.** A tool call is checked by
the same routes, so a run the account cannot see stays invisible through MCP.

**An imported adapter has to be something a run can carry on from.**
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_TMP = Path(tempfile.mkdtemp(prefix="ai-studio-mcp-"))
os.environ["AI_STUDIO_DATA"] = str(_TMP)
os.environ.setdefault("AI_STUDIO_JOIN_TOKEN", "check-join-token")

from fastapi.testclient import TestClient  # noqa: E402

from controller import auth, db  # noqa: E402
from controller.api import oauth  # noqa: E402
from controller.app import app  # noqa: E402

FAILED: list[str] = []


def ok(label: str, condition: bool, note: str = "") -> None:
    print("  %s  %s %s" % ("ok  " if condition else "FAIL", label, note))
    if not condition:
        FAILED.append(label)


def key(user_id: str, access: str | None) -> str:
    raw, h, prefix = auth.new_api_key()
    db.create_api_key(user_id, "test", h, prefix, access=access)
    return raw


def rpc(c: TestClient, token: str, method: str, params: dict | None = None, mid=1):
    return c.post("/mcp", headers={"Authorization": "Bearer " + token},
                  json={"jsonrpc": "2.0", "id": mid, "method": method,
                        "params": params or {}})


def main() -> int:
    c = TestClient(app, follow_redirects=False)
    max_id = db.create_user("max", "Max", auth.hash_password("correct horse battery"))
    eve_id = db.create_user("eve", "Eve", auth.hash_password("another long password"))
    prj = db.create_project(max_id, "HA LLM")
    eve_prj = db.create_project(eve_id, "Eve's")

    print("keys")
    studio, models_only = key(max_id, "studio"), key(max_id, None)
    r = c.get("/api/projects", headers={"Authorization": "Bearer " + studio})
    ok("a studio key opens /api", r.status_code == 200
       and any(p["id"] == prj for p in r.json()["projects"]), str(r.status_code))
    ok("it sees only its own account's projects",
       not any(p["id"] == eve_prj for p in r.json()["projects"]))
    r = c.get("/api/projects", headers={"Authorization": "Bearer " + models_only})
    ok("a models-only key does not", r.status_code == 401, str(r.status_code))
    r = c.get("/api/projects", headers={"Authorization": "Bearer sk-ais-nonsense"})
    ok("nor does a made-up one", r.status_code == 401)

    print("mcp")
    r = c.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    ok("no token: 401 with where to sign in", r.status_code == 401
       and "resource_metadata=" in r.headers.get("www-authenticate", ""))
    r = rpc(c, models_only, "initialize")
    ok("a models-only key cannot use MCP", r.status_code == 401)
    r = rpc(c, studio, "initialize", {"protocolVersion": "2025-06-18"})
    ok("initialize answers with the version asked for",
       r.json()["result"]["protocolVersion"] == "2025-06-18")
    r = c.post("/mcp", headers={"Authorization": "Bearer " + studio},
               json={"jsonrpc": "2.0", "method": "notifications/initialized"})
    ok("a notification gets 202 and no body", r.status_code == 202)
    tools = {t["name"] for t in rpc(c, studio, "tools/list").json()["result"]["tools"]}
    ok("tools are listed", {"list_runs", "start_run", "get_run_metrics"} <= tools)
    r = rpc(c, studio, "tools/call", {"name": "list_projects", "arguments": {}}).json()
    got = json.loads(r["result"]["content"][0]["text"])
    ok("list_projects goes through the REST API", not r["result"]["isError"]
       and [p["id"] for p in got["projects"]] == [prj])
    eve_run = db.create_job("hers", "finetune_llm", {}, owner_id=eve_id, project_id=eve_prj)
    r = rpc(c, studio, "tools/call", {"name": "get_run",
                                      "arguments": {"run_id": eve_run}}).json()
    ok("another account's run stays out of reach", r["result"]["isError"])
    r = rpc(c, studio, "tools/call", {"name": "studio_api", "arguments": {
        "method": "GET", "path": "/v1/models"}}).json()
    ok("studio_api is limited to /api/", r["result"]["isError"])

    print("oauth")
    meta = c.get("/.well-known/oauth-authorization-server").json()
    ok("server metadata advertises client id documents",
       meta.get("client_id_metadata_document_supported") is True
       and meta["code_challenge_methods_supported"] == ["S256"])
    res = c.get("/.well-known/oauth-protected-resource/mcp").json()
    ok("resource metadata names /mcp", res["resource"].endswith("/mcp"))

    client_id = "https://client.example/oauth/client.json"
    redirect = "https://client.example/callback"

    async def fake_fetch(cid: str) -> dict:
        return {"client_id": cid, "name": "Example Client", "fetched": True,
                "redirect_uris": [redirect], "logo": None}
    oauth.fetch_client = fake_fetch

    verifier = "v" * 64
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    q = {"response_type": "code", "client_id": client_id, "redirect_uri": redirect,
         "state": "xyz", "code_challenge": challenge, "code_challenge_method": "S256",
         "scope": "studio"}
    r = c.get("/oauth/authorize", params={**q, "redirect_uri": "https://evil.example/cb"})
    ok("an unlisted redirect is refused without redirecting",
       r.status_code == 400 and "location" not in r.headers)
    r = c.get("/oauth/authorize", params={k: v for k, v in q.items()
                                          if k != "code_challenge"})
    ok("no PKCE: sent back with an error",
       r.status_code == 302 and "error=invalid_request" in r.headers["location"])
    r = c.get("/oauth/authorize", params=q)
    ok("not signed in: the login form", r.status_code == 200 and 'name="password"' in r.text)
    r = c.post("/oauth/authorize", data={**q, "decision": "login", "username": "max",
                                         "password": "wrong"})
    ok("a wrong password is refused", "do not match" in r.text)
    r = c.post("/oauth/authorize", data={**q, "decision": "login", "username": "max",
                                         "password": "correct horse battery"})
    ok("signed in: the consent page", "Allow Example Client" in r.text)
    nonce = r.text.split('name="nonce" value="')[1].split('"')[0]
    r = c.post("/oauth/authorize", data={"nonce": nonce, "decision": "allow"})
    loc = urlsplit(r.headers.get("location", ""))
    got = parse_qs(loc.query)
    ok("allow: back to the client with a code and the state",
       r.status_code == 302 and loc.netloc == "client.example"
       and got.get("state") == ["xyz"] and got.get("code"))
    code = got["code"][0]
    r = c.post("/oauth/authorize", data={"nonce": nonce, "decision": "allow"})
    ok("the consent form works once", r.status_code == 400)

    bad = c.post("/oauth/token", data={"grant_type": "authorization_code", "code": code,
                                       "client_id": client_id, "redirect_uri": redirect,
                                       "code_verifier": "w" * 64})
    ok("a wrong verifier is refused", bad.status_code == 400)
    r = c.post("/oauth/token", data={"grant_type": "authorization_code", "code": code,
                                     "client_id": client_id, "redirect_uri": redirect,
                                     "code_verifier": verifier})
    ok("...and burns the code: the right verifier comes too late",
       r.status_code == 400 and r.json()["error"] == "invalid_grant")

    # Again, properly this time.
    r = c.post("/oauth/authorize", data={**q, "decision": "login", "username": "max",
                                         "password": "correct horse battery"})
    nonce = r.text.split('name="nonce" value="')[1].split('"')[0]
    code = parse_qs(urlsplit(c.post("/oauth/authorize", data={
        "nonce": nonce, "decision": "allow"}).headers["location"]).query)["code"][0]
    r = c.post("/oauth/token", data={"grant_type": "authorization_code", "code": code,
                                     "client_id": client_id, "redirect_uri": redirect,
                                     "code_verifier": verifier})
    tok = r.json()
    ok("code + verifier: a token", r.status_code == 200 and tok.get("access_token")
       and tok.get("refresh_token") and tok["me"].endswith("/people/max"), str(tok)[:80])
    r = rpc(c, tok["access_token"], "tools/call", {"name": "list_projects", "arguments": {}})
    ok("the token works on /mcp", not r.json()["result"]["isError"])
    r = c.post("/oauth/token", data={"grant_type": "authorization_code", "code": code,
                                     "client_id": client_id, "redirect_uri": redirect,
                                     "code_verifier": verifier})
    ok("a code works once", r.status_code == 400)
    r = c.post("/oauth/token", data={"grant_type": "refresh_token",
                                     "refresh_token": tok["refresh_token"],
                                     "client_id": client_id})
    new = r.json()
    ok("refresh: a new pair", r.status_code == 200
       and new["access_token"] != tok["access_token"])
    ok("the old access token stops working",
       rpc(c, tok["access_token"], "ping").status_code == 401)
    r = c.post("/oauth/token", data={"grant_type": "refresh_token",
                                     "refresh_token": tok["refresh_token"]})
    ok("the old refresh token stops working", r.status_code == 400)
    keys = db.list_api_keys(max_id)
    ok("the sign-in shows on the account as a studio key",
       any(k["access"] == "studio" and k["client_id"] == client_id for k in keys))

    print("import")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("adapter_config.json", json.dumps({
            "base_model_name_or_path": "Qwen/Qwen2.5-3B-Instruct", "r": 32,
            "lora_alpha": 64, "target_modules": ["q_proj", "v_proj"],
            "peft_type": "LORA"}))
        z.writestr("adapter_model.safetensors", b"\0" * 16)
    hdr = {"Authorization": "Bearer " + studio}
    r = c.put("/api/jobs/import", params={"name": "EN step 330", "project_id": eve_prj},
              content=buf.getvalue(), headers=hdr)
    ok("not into somebody else's project", r.status_code == 404)
    r = c.put("/api/jobs/import", params={"name": "junk", "project_id": prj},
              content=b"not a zip", headers=hdr)
    ok("a file that is not a model archive is refused", r.status_code == 400
       and not [j for j in db.list_jobs() if j["name"] == "junk"])
    r = c.put("/api/jobs/import", params={"name": "EN step 330", "project_id": prj},
              content=buf.getvalue(), headers=hdr)
    imp = r.json()
    ok("an adapter is imported", r.status_code == 200 and imp["kind"] == "adapter"
       and imp["base_model"] == "Qwen/Qwen2.5-3B-Instruct", str(imp)[:120])
    job = db.get_job(imp["id"])
    ok("as a finished fine-tune with its settings", job["status"] == "succeeded"
       and job["kind"] == "finetune_llm" and job["config"]["lora_r"] == 32)
    arts = db.list_artifacts(imp["id"])
    ok("whose artifact is the adapter", [a["kind"] for a in arts] == ["adapter"])
    r = c.post("/api/jobs", headers=hdr, json={
        "kind": "finetune_llm", "project_id": prj,
        "config": {"base_model_job": imp["id"]}})
    ok("a fine-tune may build on it (stops only at the missing dataset)",
       r.status_code == 400 and "dataset" in r.json()["detail"], r.text[:120])
    return 1 if FAILED else 0


if __name__ == "__main__":
    try:
        code = main()
    finally:
        shutil.rmtree(_TMP, ignore_errors=True)
    print("\n%s" % ("all good" if not FAILED else "FAILED: " + ", ".join(FAILED)))
    sys.exit(code)
