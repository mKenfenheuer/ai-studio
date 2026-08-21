import { api, events } from "../api.js";
import { html, raw, esc, $, on, fmtAgo, toast } from "../util.js";

export async function runnersView(mount) {
  const paint = async () => {
    const [status, runners] = await Promise.all([api.status(), api.runners()]);
    mount.innerHTML = html`
      <div class="page-head">
        <h1>Machines</h1>
        <p class="sub">Each machine contributes its graphics card to the studio.
          They connect outwards to this controller, so they work from home, an
          office, or a server — no port forwarding needed.</p>
      </div>

      ${raw(joinCard(status))}

      ${raw(runners.length
        ? `<div class="grid grid-2">${runners.map(card).join("")}</div>`
        : html`<div class="card empty"><div class="big">🖥️</div>
            <h3>No machines connected yet</h3>
            <p class="muted">Run the command above on a computer with a graphics
              card. It will appear here within a few seconds.</p></div>`)}`;

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

function joinCard(status) {
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

  return html`
    <div class="card" style="margin-bottom:18px">
      <h2>Connect a machine</h2>
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
    </div>`;
}

function card(r) {
  const c = r.capabilities || {};
  const offline = r.status === "offline";
  const dot = offline ? "dot-err" : r.status === "busy" ? "dot-busy" : "dot-ok";
  const dtypes = c.dtypes || {};
  const best = c.recommended_dtype;

  return html`
    <div class="card">
      <div class="row-between" style="flex-wrap:wrap;gap:6px">
        <h3 style="margin:0"><span class="dot ${dot}"></span> ${r.name}</h3>
        <span class="badge ${offline ? "badge-err" : r.status === "busy"
          ? "badge-accent" : "badge-ok"}">${offline ? "offline"
          : r.status === "busy" ? "training" : "ready"}</span>
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

      ${raw(!offline ? `<button class="btn-sm" data-reprobe="${esc(r.id)}">
        Re-check hardware</button>` : "")}
    </div>`;
}
