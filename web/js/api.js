// REST calls plus the live event socket.

async function req(path, opts = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try { detail = (await res.json()).detail || detail; } catch { /* non-JSON body */ }
    throw new Error(detail);
  }
  return res.status === 204 ? null : res.json();
}

export const api = {
  status:      () => req("/api/status"),
  runners:     () => req("/api/runners"),
  reprobe:     (id) => req(`/api/runners/${encodeURIComponent(id)}/reprobe`, { method: "POST" }),

  jobs:        () => req("/api/jobs"),
  job:         (id) => req(`/api/jobs/${encodeURIComponent(id)}`),
  jobMetrics:  (id) => req(`/api/jobs/${encodeURIComponent(id)}/metrics`),
  jobLogs:     (id) => req(`/api/jobs/${encodeURIComponent(id)}/logs`),
  cancelJob:   (id) => req(`/api/jobs/${encodeURIComponent(id)}/cancel`, { method: "POST" }),
  deleteJob:   (id) => req(`/api/jobs/${encodeURIComponent(id)}`, { method: "DELETE" }),
  createJob:   (body) => req("/api/jobs", { method: "POST", body: JSON.stringify(body) }),

  starters:    () => req("/api/hub/starters"),
  searchModels:   (q, task = "text-generation") =>
    req(`/api/hub/models?q=${encodeURIComponent(q)}&task=${encodeURIComponent(task)}`),
  searchDatasets: (q) => req(`/api/hub/datasets?q=${encodeURIComponent(q)}`),
  modelDetail:    (id) => req(`/api/hub/model?id=${encodeURIComponent(id)}`),
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
  plan:        (body) => req("/api/plan", { method: "POST", body: JSON.stringify(body) }),

  // Training from scratch. Sizes are scored against a specific machine and a
  // specific amount of patience, so both are part of the request.
  scratchSizes: (runnerId, minutes, vocabSize) =>
    req(`/api/scratch/sizes?runner_id=${encodeURIComponent(runnerId)}` +
        `&minutes=${minutes}&vocab_size=${vocabSize}`),
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
    this.connect();
  }
  connect() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    this.ws = new WebSocket(`${proto}://${location.host}/api/events`);
    this.ws.onopen = () => { this.backoff = 1000; this.emit({ type: "_connected" }); };
    this.ws.onmessage = (e) => { this.emit(JSON.parse(e.data)); };
    this.ws.onclose = () => {
      this.emit({ type: "_disconnected" });
      setTimeout(() => this.connect(), this.backoff);
      this.backoff = Math.min(this.backoff * 2, 15000);
    };
    this.ws.onerror = () => this.ws.close();
  }
  emit(msg) { this.subs.forEach((fn) => { try { fn(msg); } catch (e) { console.error(e); } }); }
  subscribe(fn) { this.subs.add(fn); return () => this.subs.delete(fn); }
}

export const events = new EventStream();
