import { api, events } from "../api.js";
import { session } from "../app.js";
import { html, raw, esc, $, on, fmtAgo, toast } from "../util.js";

export async function runnersView(mount) {
  const paint = async () => {
    const [status, runners] = await Promise.all([api.status(), api.runners()]);
    const admin = session.user?.role === "admin";
    const online = runners.filter((r) => r.status !== "offline");
    mount.innerHTML = html`
      <div class="page-head">
        <h1>Machines</h1>
        <p class="sub">Each machine contributes its graphics card to the studio.
          They connect outwards to this controller, so they work from home, an
          office, or a server — no port forwarding needed.</p>
      </div>

      ${raw(online.length ? nextCard(status) : "")}

      ${raw(admin ? joinCard(status, online.length > 0) : memberNote(online.length))}

      ${raw(runners.length
        ? `<div class="grid grid-2">${runners.map((r) => card(r, admin)).join("")}</div>`
        : html`<div class="card empty"><div class="big">🖥️</div>
            <h3>No machines connected yet</h3>
            <p class="muted">${admin
              ? "Run the command above on a computer with a graphics card. It will appear here within a few seconds."
              : "An administrator connects machines. Once one is here, you can train on it."}</p></div>`)}`;

    on(mount, "click", "[data-copy]", (_e, t) => {
      navigator.clipboard.writeText(t.dataset.copy)
        .then(() => toast("Copied to clipboard.", "ok"))
        .catch(() => toast("Could not copy — select the text manually.", "err"));
    });
    on(mount, "click", "[data-reprobe]", async (_e, t) => {
      try { await api.reprobe(t.dataset.reprobe); toast("Re-checking hardware…"); }
      catch (e) { toast(e.message, "err"); }
    });
  };

  await paint();
  const unsub = events.subscribe((m) => {
    if (m.type === "runners_changed") paint();
  });
  return unsub;
}

/** Where to go from here. The onboarding sends people to this page to connect
 *  a machine, and used to leave them on it with nowhere to go once it was. */
function nextCard(status) {
  return html`
    <div class="card" style="margin-bottom:18px;border-color:var(--accent)">
      <div class="row-between" style="flex-wrap:wrap;gap:8px">
        <div>
          <h2 style="margin:0">Connected</h2>
          <p class="muted tiny" style="margin:2px 0 0">
            ${status.jobs_running ? `${status.jobs_running} running` : "Nothing is running"}${
            status.jobs_queued ? ` · ${status.jobs_queued} waiting` : ""}.
            The next step is data, or a run.</p>
        </div>
        <div class="row">
          <a class="btn btn-primary" href="#/new">Start a training run →</a>
          <a class="btn" href="#/data">Datasets</a>
          <a class="btn" href="#/jobs">Runs</a>
        </div>
      </div>
    </div>`;
}

function memberNote(onlineCount) {
  if (onlineCount) return "";
  return html`
    <div class="callout" style="margin-bottom:18px">
      <strong>Connecting a machine needs the join token</strong>
      Only an administrator can see it. Ask them to connect one, or to make
      you an administrator.</div>`;
}

function joinCard(status, collapsed) {
  const url = location.origin;
  const cmd = `ai-studio-runner --controller ${url} --token ${status.join_token}`;
  const docker = `docker run --rm \\
  --device=/dev/kfd --device=/dev/dri --group-add video \\
  -e AI_STUDIO_CONTROLLER=${url} \\
  -e AI_STUDIO_JOIN_TOKEN=${status.join_token} \\
  ghcr.io/ai-studio/runner:rocm`;
  const dockerCuda = `docker run --rm --gpus all \\
  -e AI_STUDIO_CONTROLLER=${url} \\
  -e AI_STUDIO_JOIN_TOKEN=${status.join_token} \\
  ghcr.io/ai-studio/runner:cuda`;

  // Once a machine is here the twenty-line install snippet is reference
  // material, not the point of the page; it folds away.
  return html`
    <details class="card" style="margin-bottom:18px" ${collapsed ? "" : "open"}>
      <summary style="cursor:pointer"><h2 style="display:inline;margin:0">Connect ${
        collapsed ? "another" : "a"} machine</h2></summary>
      <p class="muted tiny">Run one of these on the computer with the graphics card.
        Anyone holding this token can join your studio, so treat it like a password.</p>

      <div class="field" style="margin-top:14px">
        <label>AMD graphics card (Docker)</label>
        <div class="row row-top">
          <pre class="mono" style="flex:1;margin:0;background:var(--surface-2);
            padding:10px;border-radius:8px;overflow-x:auto">${docker}</pre>
          <button class="btn-sm" data-copy="${docker}">Copy</button>
        </div>
      </div>

      <div class="field">
        <label>NVIDIA graphics card (Docker)</label>
        <div class="row row-top">
          <pre class="mono" style="flex:1;margin:0;background:var(--surface-2);
            padding:10px;border-radius:8px;overflow-x:auto">${dockerCuda}</pre>
          <button class="btn-sm" data-copy="${dockerCuda}">Copy</button>
        </div>
      </div>

      <details class="adv">
        <summary>Without Docker (also the only option on an Apple Mac)</summary>
        <div class="row row-top" style="margin-top:8px">
          <pre class="mono" style="flex:1;margin:0;background:var(--surface-2);
            padding:10px;border-radius:8px;overflow-x:auto">${cmd}</pre>
          <button class="btn-sm" data-copy="${cmd}">Copy</button>
        </div>
        <p class="muted tiny" style="margin-top:8px">Install first with
          <code>scripts/install-runner.sh</code>, which picks the right PyTorch
          build for your hardware.</p>
      </details>
    </details>`;
}

function card(r, admin) {
  const c = r.capabilities || {};
  const offline = r.status === "offline";
  const dot = offline ? "dot-err" : r.status === "busy" ? "dot-busy" : "dot-ok";
  const dtypes = c.dtypes || {};
  const best = c.recommended_dtype;

  // A machine restricted to uploads is never training, whatever it is doing.
  const trains = !c.kinds?.length
    || c.kinds.some((k) => k === "finetune_llm" || k === "pretrain_llm");
  return html`
    <div class="card">
      <div class="row-between" style="flex-wrap:wrap;gap:6px">
        <h3 style="margin:0"><span class="dot ${dot}"></span> ${r.name}</h3>
        <span class="badge ${offline ? "badge-err" : r.status === "busy"
          ? "badge-accent" : "badge-ok"}">${offline ? "offline"
          : r.status === "busy" ? (trains ? "training" : "busy") : "ready"}</span>
      </div>
      <p class="muted tiny mono" style="margin:6px 0 10px">
        ${c.device_name || "unknown device"}${c.arch ? " · " + c.arch : ""}</p>

      <dl class="kv" style="margin-bottom:10px">
        <dt>Memory</dt><dd>${c.vram_gb ? c.vram_gb + " GB" : "—"}</dd>
        <dt>Backend</dt><dd>${(c.backend || "—").toUpperCase()}${
          c.rocm_version ? " " + String(c.rocm_version).split("-")[0]
          : c.cuda_version ? " " + c.cuda_version : ""}</dd>
        <dt>Biggest model</dt><dd>${c.max_finetune_params_b
          ? "about " + c.max_finetune_params_b + "B parameters" : "—"}</dd>
        <dt>Last seen</dt><dd>${fmtAgo(r.last_seen)}</dd>
        ${raw(r.current_job ? html`
          <dt>Working on</dt><dd><a href="#/jobs/${r.current_job}">open the run →</a></dd>` : "")}
        ${raw(r.disk?.free_gb != null ? html`
          <dt>Disk</dt><dd>${Math.round(r.disk.free_gb)} GB free of ${
            Math.round(r.disk.total_gb || 0)}${r.disk.models_gb != null
            ? ` · ${Math.round(r.disk.models_gb)} GB of models` : ""}</dd>` : "")}
        ${raw(c.kinds?.length ? html`
          <dt>Takes</dt><dd>${c.kinds.join(", ").replace(/_/g, " ")}
            <span class="muted tiny">— and nothing else</span></dd>` : "")}
      </dl>

      ${raw(Object.keys(dtypes).length ? html`
        <div class="row" style="gap:6px;flex-wrap:wrap;margin-bottom:8px">
          ${raw(Object.entries(dtypes).map(([k, v]) => v == null ? "" :
            `<span class="badge ${k === best ? "badge-ok" : ""}">${esc(k)}
              ${v} TFLOP/s${k === best ? " ✓" : ""}</span>`).join(""))}
        </div>` : "")}

      <div class="row" style="gap:6px;flex-wrap:wrap;margin-bottom:10px">
        <span class="badge ${c.quantization?.["4bit"] ? "badge-ok" : "badge-warn"}">
          4-bit ${c.quantization?.["4bit"] ? "available" : "unavailable"}</span>
        <span class="badge ${c.attention?.flash ? "badge-ok" : "badge-warn"}">
          flash attention ${c.attention?.flash ? "available" : "unavailable"}</span>
      </div>

      ${raw((c.warnings || []).map((w) =>
        `<div class="callout callout-warn tiny">${esc(w)}</div>`).join(""))}

      ${raw(!offline && admin ? `<button class="btn-sm" data-reprobe="${esc(r.id)}">
        Re-check hardware</button>` : "")}
    </div>`;
}
