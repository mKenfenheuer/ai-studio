"""The studio as an MCP server: projects, datasets, runs and runners as tools.

Streamable HTTP at `/mcp`, answered as plain JSON -- every tool here returns
at once, so there is nothing to stream. The caller signs in with OAuth
(api/oauth.py, IndieAuth-style client ids) or sends an API key made with
studio access; either way it acts as one account, with that account's rights.

Each tool is a call to the studio's own REST API, made in-process with the
caller's token. That is deliberate: the API is where every permission check
and every validation already lives, and a second implementation for MCP would
be a second set of rules about who may do what, which is how one of them ends
up wrong. The tools are a curated map of the useful parts, plus `studio_api`
for the rest.
"""
from __future__ import annotations

import json
from typing import Any

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

from . import oauth

router = APIRouter()

PROTOCOLS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
MAX_TEXT = 60_000

_STR = {"type": "string"}
_INT = {"type": "integer"}


def _tool(name: str, description: str, props: dict | None = None,
          required: list[str] | None = None, read_only: bool = True) -> dict:
    return {"name": name, "description": description,
            "inputSchema": {"type": "object", "properties": props or {},
                            "required": required or [], "additionalProperties": False},
            "annotations": {"readOnlyHint": read_only}}


TOOLS = [
    _tool("list_projects", "Every project you can see, with its goal and what is filed in it."),
    _tool("get_project", "One project: its datasets, runs, evaluations and models by stage.",
          {"project_id": _STR}, ["project_id"]),
    _tool("create_project", "Make a project. Every run and dataset belongs to one.",
          {"name": _STR, "goal": _STR}, ["name"], read_only=False),
    _tool("list_datasets", "The datasets you can see: name, rows, splits, project."),
    _tool("get_dataset", "One dataset: its columns, splits, size and format.",
          {"dataset_id": _STR}, ["dataset_id"]),
    _tool("list_runs", "Recent runs, newest first: kind, status, progress, project.",
          {"limit": _INT}),
    _tool("get_run", "One run: its settings, status, step, summary and artifacts.",
          {"run_id": _STR}, ["run_id"]),
    _tool("get_run_metrics",
          "A run's chart as numbers: training and held-out loss, learning rate "
          "and speed per step. `last` keeps only the most recent points.",
          {"run_id": _STR, "last": _INT}, ["run_id"]),
    _tool("get_run_logs", "A run's log, newest lines last.",
          {"run_id": _STR, "limit": _INT}, ["run_id"]),
    _tool("start_run",
          "Queue a run. `kind` is finetune_llm, pretrain_llm, export_gguf, upload, "
          "evaluate, ...; `config` holds its settings exactly as the studio's own "
          "form sends them (POST /api/jobs). To keep training an earlier run or an "
          "imported adapter, set config.base_model_job to that run's id.",
          {"kind": _STR, "name": _STR, "project_id": _STR,
           "config": {"type": "object"}}, ["kind", "name", "config"], read_only=False),
    _tool("cancel_run", "Stop a run. A fine-tune keeps its best model so far.",
          {"run_id": _STR}, ["run_id"], read_only=False),
    _tool("list_runners", "The machines: their GPUs, memory, what they run now."),
    _tool("list_library", "Models published to the library, with their best metric."),
    _tool("studio_api",
          "Any other call to the studio's REST API, as this account. `path` "
          "starts with /api/. Use it for what the other tools do not cover; "
          "uploading files is not possible this way -- use the REST API with an "
          "API key for that.",
          {"method": {"type": "string", "enum": ["GET", "POST", "PUT", "DELETE"]},
           "path": _STR, "body": {"type": "object"}}, ["method", "path"],
          read_only=False),
]


def _call_for(name: str, a: dict) -> tuple[str, str, Any]:
    """The REST call a tool stands for."""
    rid = lambda k: str(a.get(k) or "").strip()  # noqa: E731
    match name:
        case "list_projects": return "GET", "/api/projects", None
        case "get_project": return "GET", "/api/projects/%s" % rid("project_id"), None
        case "create_project":
            return "POST", "/api/projects", {"name": a.get("name"), "goal": a.get("goal") or ""}
        case "list_datasets": return "GET", "/api/datasets", None
        case "get_dataset": return "GET", "/api/datasets/%s" % rid("dataset_id"), None
        case "list_runs": return "GET", "/api/jobs?limit=%d" % int(a.get("limit") or 30), None
        case "get_run": return "GET", "/api/jobs/%s" % rid("run_id"), None
        case "get_run_metrics": return "GET", "/api/jobs/%s/metrics" % rid("run_id"), None
        case "get_run_logs":
            return "GET", "/api/jobs/%s/logs?limit=%d" % (rid("run_id"),
                                                          int(a.get("limit") or 100)), None
        case "start_run":
            body = {"kind": a.get("kind"), "name": a.get("name"),
                    "config": dict(a.get("config") or {})}
            if a.get("project_id"):
                body["project_id"] = a["project_id"]
            return "POST", "/api/jobs", body
        case "cancel_run": return "POST", "/api/jobs/%s/cancel" % rid("run_id"), None
        case "list_runners": return "GET", "/api/runners", None
        case "list_library": return "GET", "/api/library", None
        case "studio_api":
            path = str(a.get("path") or "")
            if not path.startswith("/api/") or ".." in path:
                raise ValueError("path has to start with /api/.")
            return str(a.get("method") or "GET").upper(), path, a.get("body")
    raise ValueError("No tool called %s." % name)


async def _run_tool(request: Request, name: str, args: dict) -> dict:
    from ..app import app  # the app this router is part of; imported late

    method, path, body = _call_for(name, args)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://studio",
                                 timeout=120) as c:
        r = await c.request(method, path, json=body, headers={
            "Authorization": request.headers.get("authorization", "")})
    try:
        data = r.json()
    except ValueError:
        data = r.text
    if name == "get_run_metrics" and isinstance(data, list) and args.get("last"):
        data = data[-int(args["last"]):]
    text = json.dumps(data, ensure_ascii=False, default=str)
    if len(text) > MAX_TEXT:
        text = text[:MAX_TEXT] + " ... [cut at %d characters; ask for less]" % MAX_TEXT
    return {"content": [{"type": "text", "text": text}],
            "isError": r.status_code >= 400}


def _result(mid: Any, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": mid, "result": result}


def _error(mid: Any, code: int, text: str) -> dict:
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": text}}


async def _handle(request: Request, msg: dict) -> dict | None:
    if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0":
        return _error(None, -32600, "Not a JSON-RPC 2.0 message.")
    mid, method, params = msg.get("id"), msg.get("method"), msg.get("params") or {}
    if "id" not in msg:                       # a notification: nothing to say
        return None
    if method == "initialize":
        asked = params.get("protocolVersion")
        return _result(mid, {
            "protocolVersion": asked if asked in PROTOCOLS else PROTOCOLS[0],
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "ai-studio", "title": "AI Studio", "version": "0.1.0"},
            "instructions": (
                "AI Studio trains and evaluates models on the user's own GPUs. "
                "Projects hold datasets and runs; a run's chart is get_run_metrics. "
                "Starting or cancelling a run uses a GPU for hours: say what you "
                "are about to start, with its settings, before you start it."),
        })
    if method == "ping":
        return _result(mid, {})
    if method == "tools/list":
        return _result(mid, {"tools": TOOLS})
    if method == "tools/call":
        name = params.get("name")
        if name not in {t["name"] for t in TOOLS}:
            return _error(mid, -32602, "No tool called %s." % name)
        try:
            return _result(mid, await _run_tool(request, name, params.get("arguments") or {}))
        except (ValueError, TypeError) as e:
            return _result(mid, {"content": [{"type": "text", "text": str(e)}],
                                 "isError": True})
    if method in ("resources/list", "prompts/list"):
        return _result(mid, {method.split("/")[0]: []})
    return _error(mid, -32601, "Method not found: %s" % method)


@router.post("/mcp")
async def mcp(request: Request) -> Response:
    oauth.require_token(request)
    try:
        payload = await request.json()
    except ValueError:
        return JSONResponse(_error(None, -32700, "The body is not JSON."), status_code=400)
    if isinstance(payload, list):
        out = [r for r in [await _handle(request, m) for m in payload] if r]
        return JSONResponse(out) if out else Response(status_code=202)
    reply = await _handle(request, payload)
    return JSONResponse(reply) if reply else Response(status_code=202)


@router.get("/mcp")
async def mcp_get(request: Request) -> Response:
    # No server-initiated stream: every reply comes back on the POST.
    oauth.require_token(request)
    return Response(status_code=405, headers={"Allow": "POST"})


@router.delete("/mcp")
async def mcp_delete(request: Request) -> Response:
    oauth.require_token(request)
    return Response(status_code=204)
