import { api } from "../api.js";
import { html, raw, esc, $, $$, on, toast, fmtAgo, fmtNum } from "../util.js";
import { session, applyTheme, currentTheme } from "../app.js";
import { ribbon, rb, group, wireRibbon } from "../ribbon.js";
import { pageHead, copyButton, confirmDestructive } from "../components.js";

export async function settingsView(mount) {
  const status = await api.status();
  const admin = session.user?.role === "admin";
  let storage = null;
  const drawStorage = () => {
    const box = $("#storageCard", mount);
    if (box) box.innerHTML = storageCard(storage);
  };
  const loadStorage = async () => {
    if (!admin) return;
    try { storage = await api.storage(); } catch (e) { storage = { error: e.message }; }
    drawStorage();
  };
  mount.innerHTML = html`
    ${raw(pageHead({ title: "Settings", sub: "Studio-wide configuration." }))}
    ${raw(ribbon({
      tabs: [{ key: "general", label: "General" }], active: "general",
      body: group("Your things", [
        rb(null, "◉", "Your account", { href: "#/account",
          title: "Name, password, keys, Hugging Face, notifications" }),
      ]) + (admin ? group("Administration", [
        rb(null, "◍", "People", { href: "#/users" }),
        rb(null, "🔑", "Single sign-on", { href: "#/sso" }),
        rb(null, "▦", "Machines", { href: "#/runners" }),
      ]) : "") + group("Session", [
        rb("signOut", "→", "Sign out", { cls: "danger" }),
      ]),
    }))}

    ${raw(admin ? `<div id="storageCard" style="margin-bottom:14px">${storageCard(null)}</div>` : "")}
    <div class="card" style="margin-bottom:14px">
      <h3>Hugging Face</h3>
      <p class="muted tiny">Needed for <em>gated</em> models such as Llama or
        Gemma, for better download rate limits, and to publish what you
        train.</p>
      <div class="callout ${status.hf_token_set ? "callout-ok" : "callout-warn"}">
        <strong>${status.hf_token_is_yours ? "Using your own account"
          : status.hf_token_set ? "Using the studio's shared token"
          : "No Hugging Face access"}</strong>
        ${raw(status.hf_token_is_yours
          ? `Downloads and publishing run as you.
             <a href="#/account">Manage it on your account page.</a>`
          : status.hf_token_set
          ? `This studio has a token set in its environment, and everyone
             here shares it. <a href="#/account">Connect your own account</a>
             to publish models under your name.`
          : `<a href="#/account">Connect your Hugging Face account</a> to
             download gated models and publish what you train.`)}
      </div>
    </div>

    ${raw(!admin ? "" : html`
    <div class="card" style="margin-bottom:14px">
      <h3>Join token</h3>
      <p class="muted tiny">Machines present this to join the studio. Anyone with
        it can attach a machine and read job data, so share it carefully.</p>
      <div class="row row-top">
        <input type="password" id="tok" value="${status.join_token}" readonly>
        <button class="btn-sm" id="reveal">Show</button>
        ${raw(copyButton(status.join_token || ""))}
      </div>
      <div class="hint">To rotate it, delete <code class="mono">data/join_token</code>
        on the controller and restart. Every machine must then rejoin.</div>
    </div>`)}

    <div class="card" style="margin-bottom:14px">
      <h3>Appearance</h3>
      <p class="muted tiny">Follows your operating system unless you say
        otherwise. Kept in this browser, not on your account.</p>
      <div class="seg" style="margin-top:8px" role="group" aria-label="Theme">
        ${raw([["system", "Match my system"], ["light", "Light"], ["dark", "Dark"]]
          .map(([v, label]) => `<button class="btn-sm ${
            currentTheme() === v ? "on" : ""}" data-theme-set="${v}">${label}</button>`).join(""))}
      </div>
    </div>

    <div class="card">
      <h3>About</h3>
      <dl class="kv">
        <dt>Version</dt><dd>${status.version}</dd>
        <dt>Machines</dt><dd>${status.runners_online} online / ${status.runners_total} known</dd>
        <dt>Controller</dt><dd class="mono">${location.origin}</dd>
        <dt>Signed in as</dt><dd>${session.user?.display_name || "?"}
          ${raw(admin ? `<span class="badge badge-accent">administrator</span>` : "")}</dd>
      </dl>
    </div>`;

  wireRibbon(mount, () => {});
  on(mount, "click", "[data-theme-set]", (_e, t) => {
    applyTheme(t.dataset.themeSet);
    $$("[data-theme-set]", mount).forEach((b) =>
      b.classList.toggle("on", b.dataset.themeSet === t.dataset.themeSet));
  });
  // ---- storage
  loadStorage();
  on(mount, "submit", "#retentionForm", async (e) => {
    e.preventDefault();
    const f = new FormData(e.target);
    try {
      await api.saveStorageSettings(Object.fromEntries(f.entries()));
      toast("Rules saved. They apply at the next tidy, within the hour.", "ok");
      await loadStorage();
    } catch (ex) { toast(ex.message, "err"); }
  });
  on(mount, "click", "#sweepNow", async (_e, btn) => {
    btn.disabled = true;
    try {
      const r = await api.sweepStorage();
      toast(`Tidied: ${r.models_removed} model(s) removed, ${r.runs_thinned} run(s) thinned, ${
        (r.bytes_freed / 1024 ** 2).toFixed(0)} MB freed.`, "ok");
      await loadStorage();
    } catch (ex) { toast(ex.message, "err"); }
    btn.disabled = false;
  });
  on(mount, "click", "[data-drop-model]", async (_e, t) => {
    const row = (storage?.biggest || []).find((b) => b.job_id === t.dataset.dropModel);
    if (!await confirmDestructive({
      title: `Remove the model from "${row?.name || "this run"}"?`,
      consequences: [
        `${row ? (row.bytes / 1024 ** 3).toFixed(1) + " GB comes back. " : ""}The run, its chart, its log and its notes stay.`,
        "It cannot be talked to, scored, published or exported afterwards.",
      ],
      confirmLabel: "Remove the model" })) return;
    try {
      await api.dropJobModel(t.dataset.dropModel);
      toast("Removed.", "ok");
      await loadStorage();
    } catch (ex) { toast(ex.message, "err"); }
  });

  on(mount, "click", "#signOut", async () => {
    await api.logout().catch(() => {});
    location.reload();
  });
  on(mount, "click", "#reveal", (_e, t) => {
    const i = $("#tok", mount);
    const show = i.type === "password";
    i.type = show ? "text" : "password";
    t.textContent = show ? "Hide" : "Show";
  });
}


// ---- what the disk is holding --------------------------------------------

const gb = (b) => (b / 1024 ** 3).toFixed(b >= 10 * 1024 ** 3 ? 0 : 1) + " GB";
const mb = (b) => b >= 1024 ** 3 ? gb(b) : (b / 1024 ** 2).toFixed(0) + " MB";

/** Storage: the numbers, the biggest runs, and the rules for keeping them.
 *
 *  The answer to "why is the disk full" as a table rather than a `du`. The
 *  rules are settings because the right ones depend on the disk and on
 *  what the studio is for -- a research box keeps every model, a production
 *  one keeps the ones that are served. */
function storageCard(st) {
  if (!st) return `<div class="card"><h3>Storage</h3><p class="muted tiny">Reading the disk…</p></div>`;
  if (st.error) return `<div class="card"><h3>Storage</h3><p class="muted tiny">${st.error}</p></div>`;
  const d = st.disk;
  const s = st.settings;
  const last = st.last_sweep;
  const usedClass = d && d.used_pct >= 95 ? "callout-err" : d && d.used_pct >= 85 ? "callout-warn" : "callout-ok";
  return html`
    <div class="card">
      <div class="row-between" style="flex-wrap:wrap;gap:8px">
        <h3 style="margin:0">Storage</h3>
        <span class="tiny muted">${last ? `last tidied ${fmtAgo(last.at)}` : "not tidied yet"}</span>
      </div>
      ${raw(d ? html`
        <div class="callout ${usedClass}" style="margin-top:8px">
          <strong>${d.free_gb} GB free of ${d.total_gb} GB</strong>
          ${d.used_pct}% used${d.used_pct >= 95
            ? " — a training run that saves a model may fail here." : ""}
        </div>` : "")}
      <div class="row" style="gap:24px;margin-top:10px;flex-wrap:wrap">
        <div class="stat"><span class="v">${gb(st.artifacts.bytes)}</span>
          <span class="k">models · ${st.artifacts.runs} runs</span></div>
        <div class="stat"><span class="v">${gb(st.datasets_bytes)}</span><span class="k">datasets</span></div>
        <div class="stat"><span class="v">${gb(st.assets.bytes)}</span>
          <span class="k">pictures &amp; clips · ${st.assets.files} files</span></div>
        <div class="stat"><span class="v">${mb(st.database.bytes)}</span>
          <span class="k">database · ${fmtNum(st.database.metrics_rows)} metric rows</span></div>
      </div>

      ${raw(st.biggest.length ? html`
        <h4 style="margin:14px 0 6px">The biggest</h4>
        <div class="table-wrap"><table class="table"><thead><tr>
          <th>Run</th><th>Model</th><th class="hide-sm">Finished</th><th></th>
        </tr></thead><tbody>
          ${raw(st.biggest.slice(0, 10).map((b) => html`
            <tr>
              <td><a href="#/jobs/${b.job_id}">${b.name}</a>
                ${raw(b.served ? ` <span class="badge badge-ok" title="Served under a registered name; never expired">served</span>` : "")}
                ${raw(b.kept ? ` <span class="badge" title="Tagged keep; never expired">keep</span>` : "")}
                ${raw(b.owner ? `<div class="muted tiny">${esc(b.owner.display_name || b.owner.username)}</div>` : "")}</td>
              <td class="mono tiny">${gb(b.bytes)}<div class="muted">${b.kinds.join(" + ")}</div></td>
              <td class="hide-sm tiny muted">${b.finished_at ? fmtAgo(b.finished_at) : "—"}</td>
              <td>${raw(b.served ? "" : html`
                <button class="btn-sm btn-danger" data-drop-model="${b.job_id}"
                        title="Remove the model; keep the run, its chart and its log">Remove model</button>`)}</td>
            </tr>`).join(""))}
        </tbody></table></div>` : "")}

      <h4 style="margin:14px 0 6px">Rules</h4>
      <form id="retentionForm" class="row" style="gap:10px;flex-wrap:wrap;align-items:flex-end">
        <div class="field" style="margin:0">
          <label for="rtDays">Models expire after</label>
          <input id="rtDays" name="artifact_days" type="number" min="0" value="${s.artifact_days}" style="width:90px"> days
          <div class="hint">0 keeps them forever. A run served under a name or
            tagged <code>keep</code> is never touched; the run itself always stays.</div>
        </div>
        <div class="field" style="margin:0">
          <label for="rtTrim">Old runs get lighter after</label>
          <input id="rtTrim" name="trim_days" type="number" min="0" value="${s.trim_days}" style="width:90px"> days
          <div class="hint">Metrics thinned to a few hundred points, the log
            cut to its head and tail. The chart keeps its shape.</div>
        </div>
        <div class="field" style="margin:0">
          <label for="rtQuota">Models per account</label>
          <input id="rtQuota" name="artifact_quota_gb" type="number" min="0" value="${s.artifact_quota_gb}" style="width:90px"> GB
          <div class="hint">0 for no ceiling. Checked when a run starts, not
            when its model arrives.</div>
        </div>
        <button class="btn-primary btn-sm" type="submit">Save rules</button>
        <button class="btn-sm" type="button" id="sweepNow">Tidy now</button>
      </form>
      ${raw(last ? html`
        <p class="muted tiny" style="margin:8px 0 0">Last tidy: ${last.models_removed} model${last.models_removed === 1 ? "" : "s"} removed,
          ${last.runs_thinned} run${last.runs_thinned === 1 ? "" : "s"} thinned,
          ${last.orphan_files} orphan file${last.orphan_files === 1 ? "" : "s"} swept,
          ${mb(last.bytes_freed)} freed.</p>` : "")}
    </div>`;
}
