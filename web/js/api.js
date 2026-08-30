// REST calls plus the live event socket.

// Raised when the server says "sign in". The shell listens for it and puts
// the gate back up, so a session that expires mid-session does not turn into
// a page full of red error cards.
export class NotSignedIn extends Error {}

let onUnauthorized = null;
export function handleUnauthorized(fn) { onUnauthorized = fn; }

async function req(path, opts = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    // Cookies are same-origin already; stated explicitly so that a future
    // deployment serving the UI from another host does not silently stop
    // authenticating.
    credentials: "same-origin",
    ...opts,
  });
  if (res.status === 401 || (res.status === 403 && path !== "/api/auth/state")) {
    let body = {};
    try { body = await res.clone().json(); } catch { /* not JSON */ }
    if (res.status === 401 || body.must_change) {
      if (onUnauthorized) onUnauthorized(body);
      if (res.status === 401) throw new NotSignedIn(body.detail || "Please sign in.");
    }
  }
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try { detail = (await res.json()).detail || detail; } catch { /* non-JSON body */ }
    throw new Error(detail);
  }
  return res.status === 204 ? null : res.json();
}

export const api = {
  // ---- accounts --------------------------------------------------------
  authState:   () => req("/api/auth/state"),
  login:       (body) => req("/api/auth/login", { method: "POST", body: JSON.stringify(body) }),
  setup:       (body) => req("/api/auth/setup", { method: "POST", body: JSON.stringify(body) }),
  logout:      () => req("/api/auth/logout", { method: "POST" }),
  me:          () => req("/api/me"),
  updateMe:    (body) => req("/api/me", { method: "PATCH", body: JSON.stringify(body) }),
  changePassword: (body) =>
    req("/api/me/password", { method: "POST", body: JSON.stringify(body) }),
  revokeSessions: () => req("/api/me/sessions/revoke", { method: "POST" }),

  authProviders: () => req("/api/auth/providers"),

  users:       (q = "", opts = {}) =>
    req(`/api/users?q=${encodeURIComponent(q)}`
        + `&limit=${opts.limit || 200}&pending=${!!opts.pending}`),
  // Called on every keystroke of the share box, so it is a lookup rather than
  // a filter: the server ranks and caps, and the browser never holds the list.
  searchUsers: (q, exclude = []) =>
    req(`/api/users/search?q=${encodeURIComponent(q || "")}`
        + (exclude.length ? `&exclude=${encodeURIComponent(exclude.join(","))}` : "")),
  createUser:  (body) => req("/api/users", { method: "POST", body: JSON.stringify(body) }),
  updateUser:  (id, body) =>
    req(`/api/users/${encodeURIComponent(id)}`, { method: "PATCH", body: JSON.stringify(body) }),
  resetUserPassword: (id, password) =>
    req(`/api/users/${encodeURIComponent(id)}/password`,
        { method: "POST", body: JSON.stringify({ password }) }),
  deleteUser:  (id) => req(`/api/users/${encodeURIComponent(id)}`, { method: "DELETE" }),

  // ---- single sign-on --------------------------------------------------
  idps:        () => req("/api/idp"),
  idpPresets:  () => req("/api/idp/presets"),
  createIdp:   (body) => req("/api/idp", { method: "POST", body: JSON.stringify(body) }),
  updateIdp:   (id, body) =>
    req(`/api/idp/${encodeURIComponent(id)}`,
        { method: "PATCH", body: JSON.stringify(body) }),
  deleteIdp:   (id) => req(`/api/idp/${encodeURIComponent(id)}`, { method: "DELETE" }),
  rediscoverIdp: (id) =>
    req(`/api/idp/${encodeURIComponent(id)}/rediscover`, { method: "POST" }),
  syncIdp:     (id, dryRun = false) =>
    req(`/api/idp/${encodeURIComponent(id)}/sync?dry_run=${dryRun}`,
        { method: "POST" }),

  // ---- api keys --------------------------------------------------------
  apiKeys:     () => req("/api/me/api-keys"),
  createApiKey: (name) =>
    req("/api/me/api-keys", { method: "POST", body: JSON.stringify({ name }) }),
  deleteApiKey: (id) =>
    req(`/api/me/api-keys/${encodeURIComponent(id)}`, { method: "DELETE" }),

  // ---- notifications ---------------------------------------------------
  notifyState: () => req("/api/me/notifications"),
  notifySet:   (body) =>
    req("/api/me/notifications", { method: "POST", body: JSON.stringify(body) }),
  notifyTest:  () => req("/api/me/notifications/test", { method: "POST" }),
  notifyClear: () => req("/api/me/notifications", { method: "DELETE" }),

  // ---- hugging face ----------------------------------------------------
  hfState:     () => req("/api/me/huggingface"),
  hfConnect:   (token) =>
    req("/api/me/huggingface", { method: "POST", body: JSON.stringify({ token }) }),
  hfDisconnect: () => req("/api/me/huggingface", { method: "DELETE" }),
  hfRepos:     (kind = "models") =>
    req(`/api/me/huggingface/repos?kind=${encodeURIComponent(kind)}`),
  hfDeleteRepo: (repoId, kind = "models") =>
    req(`/api/me/huggingface/repos?repo_id=${encodeURIComponent(repoId)}&kind=${kind}`,
        { method: "DELETE" }),
  publishJob:  (id, body) =>
    req(`/api/jobs/${encodeURIComponent(id)}/publish`,
        { method: "POST", body: JSON.stringify(body) }),

  // ---- hosted model providers -------------------------------------------
  providers:      () => req("/api/providers"),
  saveProvider:   (id, body) =>
    req(`/api/providers/${encodeURIComponent(id)}`,
        { method: "PUT", body: JSON.stringify(body) }),
  deleteProvider: (id, scope = "account") =>
    req(`/api/providers/${encodeURIComponent(id)}?scope=${encodeURIComponent(scope)}`,
        { method: "DELETE" }),
  providerModels: (id) =>
    req(`/api/providers/${encodeURIComponent(id)}/models`),
  testProvider:   (id, model) =>
    req(`/api/providers/${encodeURIComponent(id)}/test`,
        { method: "POST", body: JSON.stringify({ model }) }),

  // ---- sharing ---------------------------------------------------------
  shares:      (kind, id) => req(`/api/${kind}s/${encodeURIComponent(id)}/shares`),
  addShare:    (kind, id, body) =>
    req(`/api/${kind}s/${encodeURIComponent(id)}/shares`,
        { method: "POST", body: JSON.stringify(body) }),
  removeShare: (kind, id, subjectType, subjectId) =>
    req(`/api/${kind}s/${encodeURIComponent(id)}/shares?subject_type=${subjectType}`
        + (subjectId ? `&subject_id=${encodeURIComponent(subjectId)}` : ""),
        { method: "DELETE" }),
  transfer:    (kind, id, userId) =>
    req(`/api/${kind}s/${encodeURIComponent(id)}/transfer`,
        { method: "POST", body: JSON.stringify({ user_id: userId }) }),

  // ---- datasets --------------------------------------------------------
  datasets:    () => req("/api/datasets"),
  dataset:     (id) => req(`/api/datasets/${encodeURIComponent(id)}`),
  datasetInspect: (id) => req(`/api/datasets/${encodeURIComponent(id)}/inspect`),
  datasetRows: (id, offset = 0, limit = 25, q = "", split = "") =>
    req(`/api/datasets/${encodeURIComponent(id)}/rows?offset=${offset}`
        + `&limit=${limit}&q=${encodeURIComponent(q)}`
        + `&split=${encodeURIComponent(split)}`),
  // Rows as conversations, each already cut where a model would take over.
  // What the playground loads to try a held-out example against the model
  // that was trained on the rest of the file.
  datasetConversations: (id, offset = 0, limit = 20, split = "") =>
    req(`/api/datasets/${encodeURIComponent(id)}/conversations?offset=${offset}`
        + `&limit=${limit}&split=${encodeURIComponent(split)}`),
  conversationReport: (id) =>
    req(`/api/datasets/${encodeURIComponent(id)}/conversation-report`),
  importDataset: (body) =>
    req("/api/datasets/import", { method: "POST", body: JSON.stringify(body) }),
  // Several files at once, under one field name: the endpoint takes a list,
  // so one file and eighty go the same way.
  uploadDataset: (files, name, options = {}) => {
    const form = new FormData();
    [...files].forEach((f) => form.append("file", f));
    const query = new URLSearchParams({ name: name || "", ...options });
    return fetch(`/api/datasets/upload?${query}`,
                 { method: "POST", body: form, credentials: "same-origin" })
      .then(async (r) => {
        if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
        return r.json();
      });
  },
  renameDataset: (id, body) =>
    req(`/api/datasets/${encodeURIComponent(id)}`, { method: "PATCH", body: JSON.stringify(body) }),
  previewTransform: (id, body) =>
    req(`/api/datasets/${encodeURIComponent(id)}/transform/preview`,
        { method: "POST", body: JSON.stringify(body) }),
  editRows: (id, body) =>
    req(`/api/datasets/${encodeURIComponent(id)}/rows/edit`,
        { method: "POST", body: JSON.stringify(body) }),
  addRows: (id, body) =>
    req(`/api/datasets/${encodeURIComponent(id)}/rows/add`,
        { method: "POST", body: JSON.stringify(body) }),
  transformDataset: (id, body) =>
    req(`/api/datasets/${encodeURIComponent(id)}/transform`,
        { method: "POST", body: JSON.stringify(body) }),
  splitDataset: (id, body) =>
    req(`/api/datasets/${encodeURIComponent(id)}/split`,
        { method: "POST", body: JSON.stringify(body) }),
  mergeDatasets: (body) =>
    req("/api/datasets/merge", { method: "POST", body: JSON.stringify(body) }),
  deleteDataset: (id) =>
    req(`/api/datasets/${encodeURIComponent(id)}`, { method: "DELETE" }),
  publishDataset: (id, body) =>
    req(`/api/datasets/${encodeURIComponent(id)}/publish`,
        { method: "POST", body: JSON.stringify(body) }),

  // ---- evaluations -----------------------------------------------------
  evals:       () => req("/api/evals"),
  eval:        (id) => req(`/api/evals/${encodeURIComponent(id)}`),
  createEval:  (body) => req("/api/evals", { method: "POST", body: JSON.stringify(body) }),
  evalFromDataset: (body) =>
    req("/api/evals/from-dataset", { method: "POST", body: JSON.stringify(body) }),
  updateEval:  (id, body) =>
    req(`/api/evals/${encodeURIComponent(id)}`, { method: "PATCH", body: JSON.stringify(body) }),
  copyEval:    (id, body) =>
    req(`/api/evals/${encodeURIComponent(id)}/copy`,
        { method: "POST", body: JSON.stringify(body || {}) }),
  deleteEval:  (id) => req(`/api/evals/${encodeURIComponent(id)}`, { method: "DELETE" }),
  runEval:     (id, body) =>
    req(`/api/evals/${encodeURIComponent(id)}/run`,
        { method: "POST", body: JSON.stringify(body) }),
  evalScores:  (id) => req(`/api/evals/${encodeURIComponent(id)}/scores`),
  evalScore:   (id, scoreId) =>
    req(`/api/evals/${encodeURIComponent(id)}/scores/${encodeURIComponent(scoreId)}`),
  deleteScore: (id, scoreId) =>
    req(`/api/evals/${encodeURIComponent(id)}/scores/${encodeURIComponent(scoreId)}`,
        { method: "DELETE" }),

  status:      () => req("/api/status"),
  runners:     () => req("/api/runners"),
  reprobe:     (id) => req(`/api/runners/${encodeURIComponent(id)}/reprobe`, { method: "POST" }),

  jobs:        () => req("/api/jobs"),
  job:         (id) => req(`/api/jobs/${encodeURIComponent(id)}`),
  jobMetrics:  (id) => req(`/api/jobs/${encodeURIComponent(id)}/metrics`),
  jobLogs:     (id) => req(`/api/jobs/${encodeURIComponent(id)}/logs`),
  jobReport:   (id) => req(`/api/jobs/${encodeURIComponent(id)}/report`),
  // The model card. Reading one never stores it -- a run that has never had
  // a card generates one to look at, and only saving or finishing writes it.
  jobCard:     (id) => req(`/api/jobs/${encodeURIComponent(id)}/card`),
  saveJobCard: (id, markdown) =>
    req(`/api/jobs/${encodeURIComponent(id)}/card`,
        { method: "PUT", body: JSON.stringify({ markdown }) }),
  resetJobCard: (id) =>
    req(`/api/jobs/${encodeURIComponent(id)}/card`,
        { method: "PUT", body: JSON.stringify({ reset: true }) }),
  // `save` decides whether the half-trained model survives the stop.
  cancelJob:   (id, save = true, force = false) =>
    req(`/api/jobs/${encodeURIComponent(id)}/cancel`,
        { method: "POST", body: JSON.stringify({ save, force }) }),
  deleteJob:   (id) => req(`/api/jobs/${encodeURIComponent(id)}`, { method: "DELETE" }),
  renameJob:   (id, name) =>
    req(`/api/jobs/${encodeURIComponent(id)}`,
        { method: "PATCH", body: JSON.stringify({ name }) }),
  // Put a stopped or failed run back on the queue, to carry on from its
  // checkpoint rather than from step one.
  resumeJob:   (id) =>
    req(`/api/jobs/${encodeURIComponent(id)}/resume`, { method: "POST" }),
  jobChatTemplate: (id) =>
    req(`/api/jobs/${encodeURIComponent(id)}/chat-template`),
  createJob:   (body) => req("/api/jobs", { method: "POST", body: JSON.stringify(body) }),

  // ---- sweeps ----------------------------------------------------------
  sweeps:      () => req("/api/sweeps"),
  sweep:       (id) => req(`/api/sweeps/${encodeURIComponent(id)}`),
  createSweep: (body) =>
    req("/api/sweeps", { method: "POST", body: JSON.stringify(body) }),

  starters:    () => req("/api/hub/starters"),
  searchModels:   (q, task = "text-generation") =>
    req(`/api/hub/models?q=${encodeURIComponent(q)}&task=${encodeURIComponent(task)}`),
  searchDatasets: (q) => req(`/api/hub/datasets?q=${encodeURIComponent(q)}`),
  modelDetail:    (id) => req(`/api/hub/model?id=${encodeURIComponent(id)}`),
  recommendations: (runnerId) =>
    req(`/api/hub/recommendations?runner_id=${encodeURIComponent(runnerId || "")}`),
  datasetPreview: (id) => req(`/api/hub/dataset-preview?id=${encodeURIComponent(id)}`),
  datasetConfigs: (id) =>
    req(`/api/hub/dataset-configs?id=${encodeURIComponent(id)}`),
  // The rendered training text, built by the same code the runner uses.
  trainingPreview: (body) =>
    req("/api/hub/training-preview", { method: "POST", body: JSON.stringify(body) }),
  modelTemplate: (id) =>
    req(`/api/hub/model-template?id=${encodeURIComponent(id)}`),
  builtinTemplate: () => req("/api/hub/builtin-template"),
  chatFormats: () => req("/api/chat-formats"),
  selectorFields: () => req("/api/selectors"),
  plan:        (body) => req("/api/plan", { method: "POST", body: JSON.stringify(body) }),

  // Training from scratch. Sizes are scored against a specific machine and a
  // specific amount of patience, so both are part of the request.
  scratchSizes: (runnerId, minutes, vocabSize, moe = null) =>
    req(`/api/scratch/sizes?runner_id=${encodeURIComponent(runnerId)}` +
        `&minutes=${minutes}&vocab_size=${vocabSize}` +
        (moe?.enabled
          ? `&experts=${moe.num_local_experts}` +
            `&experts_per_token=${moe.num_experts_per_tok}`
          : "")),
  scratchPlan: (body) =>
    req("/api/scratch/plan", { method: "POST", body: JSON.stringify(body) }),

  playground:  () => req("/api/playground"),
  chat:        (id, body) =>
    req(`/api/jobs/${encodeURIComponent(id)}/chat`,
        { method: "POST", body: JSON.stringify(body) }),
  jobSystemPrompt: (id) =>
    req(`/api/jobs/${encodeURIComponent(id)}/system-prompt`),
  chatCancel:  (requestId) =>
    req(`/api/chat/${encodeURIComponent(requestId)}/cancel`, { method: "POST" }),
};

/** Live updates from the controller, with automatic reconnect.
 *  Subscribers get every message; each view filters for what it cares about. */
class EventStream {
  constructor() {
    this.subs = new Set();
    this.ws = null;
    this.backoff = 1000;
    // Not connected on construction any more: the socket needs a session, and
    // an unauthenticated one is closed by the server the moment it opens. Left
    // to reconnect on its own that becomes a permanent retry loop behind the
    // login screen.
    this.wanted = false;
  }
  start() {
    if (this.wanted) return;
    this.wanted = true;
    this.connect();
  }
  stop() {
    this.wanted = false;
    try { this.ws?.close(); } catch { /* already gone */ }
    this.ws = null;
  }
  connect() {
    if (!this.wanted) return;
    const proto = location.protocol === "https:" ? "wss" : "ws";
    this.ws = new WebSocket(`${proto}://${location.host}/api/events`);
    this.ws.onopen = () => { this.backoff = 1000; this.emit({ type: "_connected" }); };
    this.ws.onmessage = (e) => { this.emit(JSON.parse(e.data)); };
    this.ws.onclose = (e) => {
      this.emit({ type: "_disconnected" });
      // 4401 is this server saying "you are not signed in". Retrying that is
      // pointless; the page is about to show the login screen anyway.
      if (e && e.code === 4401) { this.wanted = false; return; }
      if (!this.wanted) return;
      setTimeout(() => this.connect(), this.backoff);
      this.backoff = Math.min(this.backoff * 2, 15000);
    };
    this.ws.onerror = () => this.ws.close();
  }
  emit(msg) { this.subs.forEach((fn) => { try { fn(msg); } catch (e) { console.error(e); } }); }
  subscribe(fn) { this.subs.add(fn); return () => this.subs.delete(fn); }
}

export const events = new EventStream();
