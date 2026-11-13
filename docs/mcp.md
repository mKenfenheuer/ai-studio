# MCP server, API keys with studio access, imports

AI Studio can be driven by other programs: scripts through the REST API, and
MCP clients such as Claude through the MCP server at `/mcp`. Both act as one
account and have exactly that account's rights.

## API keys with studio access

Account → API keys → tick **Studio access** when you create the key. Such a
key works as `Authorization: Bearer sk-ais-…` on every `/api/` route and on
`/mcp`. A key without the tick only talks to the models on `/v1`, as before;
it is refused on `/api` and `/mcp` even though it is valid.

```sh
curl -H "Authorization: Bearer $KEY" http://studio/api/jobs?limit=5
```

## MCP

Endpoint: `https://<your studio>/mcp` (streamable HTTP, JSON replies).

Tools: `list_projects`, `get_project`, `create_project`, `list_datasets`,
`get_dataset`, `list_runs`, `get_run`, `get_run_metrics`, `get_run_logs`,
`start_run`, `cancel_run`, `list_runners`, `list_library`, and `studio_api`
for any other `/api/` call. Every tool is a call to the REST API with the
caller's token, so the API's own checks apply.

### Signing in: OAuth in the IndieAuth way

A client is not registered in the studio first. Its `client_id` is a URL, and
the studio fetches it when you sign in: a JSON client ID metadata document
(what MCP clients publish) or an HTML page with `<link rel="redirect_uri">`.
The browser is only ever sent back to an address the client lists, or one on
the client's own host.

- Metadata: `/.well-known/oauth-authorization-server`,
  `/.well-known/oauth-protected-resource/mcp`
- Authorization code flow with PKCE (S256 required), no client secret
- Token: an API key with studio access, valid 7 days, plus a refresh token
  valid 180 days; each refresh replaces both
- The sign-in shows on the account page as a key named `MCP: <client>`;
  deleting it signs that client out

A remote client such as claude.ai has to reach the studio over HTTPS. Put it
behind a reverse proxy or tunnel and set `AI_STUDIO_PUBLIC_URL` to the public
address, so the metadata names URLs the client can reach.

## Importing a model or adapter

`PUT /api/jobs/import?name=…&project_id=…[&base_model=…]` with a zip of the
directory `save_pretrained` wrote as the body (files at the top level of the
archive). It becomes a finished fine-tune run. A new fine-tune with
`config.base_model_job` set to its id keeps training the adapter instead of
starting a fresh one.

```sh
(cd output/lora/best && zip -q -r - .) | curl -X PUT -H "Authorization: Bearer $KEY" \
  --data-binary @- "http://studio/api/jobs/import?name=EN%20step%20330&project_id=prj_…"
```

## Calibrated GGUF export

The export dialog (and `start_run` with `kind: "export_gguf"`) can calibrate
the quantisation: `config.imatrix = {"studio_dataset": "ds_…", "rows": 300,
"split": "train"}` renders that many conversations with the model's own chat
template and tools, measures an importance matrix with `llama-imatrix` on the
GPU, and quantises with it. `config.chat_template` replaces the template in
the GGUF (for example one with a default system prompt). Calibration needs a
machine with `llama-imatrix`, which the CUDA runner image ships.
