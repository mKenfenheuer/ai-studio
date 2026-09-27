/**
 * Running the studio, kept apart from using it.
 *
 * Machines, people, single sign-on, served names, what the disk is holding and
 * what is backed up are all one job — keeping this thing running — and none of
 * them is anything to do with making a model. They were spread through the
 * main navigation, so the sidebar a new user met was half tools and half
 * plumbing, and the plumbing was the half with the destructive buttons.
 *
 * This is that half, in one place, only for administrators.
 */
import { api, events } from "../api.js";
import { session } from "../app.js";
import { html, raw, esc, $, on, toast, fmtAgo, fmtNum, fmtBytes } from "../util.js";
import { ribbon, rb, group, wireRibbon } from "../ribbon.js";
import { pageHead, confirmDestructive } from "../components.js";

const TABS = [
  { key: "overview", label: "Overview" },
  { key: "storage", label: "Storage" },
  { key: "backups", label: "Backups" },
  { key: "cloud", label: "Cloud GPUs" },
];

export async function adminView(mount, [tab]) {
  if (session.user?.role !== "admin") {
    mount.innerHTML = html`
      ${raw(pageHead({ title: "Administration" }))}
      <div class="callout callout-warn"><strong>Administrators only</strong>
        This section is the studio's own plumbing — machines, accounts, the
        disk and the backups. Ask an administrator if you need something
        here.</div>`;
    return null;
  }
  let active = TABS.some((t) => t.key === tab) ? tab : "overview";
  let status = null;
  let storage = null;
  let backups = null;
  let cloud = null;       // settings, spend, pods
  let gpus = null;        // what can be rented now

  const draw = () => {
    mount.innerHTML = html`
      ${raw(pageHead({
        title: "Administration",
        sub: "The studio itself: machines, people, the disk and the backups.",
      }))}
      ${raw(ribbon({
        tabs: TABS, active,
        body: group("People and machines", [
          rb(null, "◍", "People", { href: "#/users" }),
          rb(null, "🔑", "Single sign-on", { href: "#/sso" }),
          rb(null, "▦", "Machines", { href: "#/runners" }),
        ]) + group("What it serves", [
          rb(null, "🏷", "Served names & usage", { href: "#/serving" }),
          rb(null, "⚙", "Settings", { href: "#/settings" }),
        ]) + (active === "backups" ? group("Now", [
          rb("backupNow", "⭳", "Back up now", { cls: "primary" }),
        ]) : active === "storage" ? group("Now", [
          rb("sweepNow", "🧹", "Tidy now"),
        ]) : active === "cloud" ? group("Now", [
          rb("cloudRefresh", "↻", "Refresh prices"),
        ]) : ""),
        right: status
          ? `<span class="badge">${status.runners_online}/${status.runners_total} machines online</span>` : "",
      }))}
      <div id="adminBody">${raw(
        active === "storage" ? storageCard(storage)
        : active === "backups" ? backupsCard(backups)
        : active === "cloud" ? cloudCard(cloud, gpus)
        : overview(status))}</div>`;
  };

  const loadStorage = async () => {
    try { storage = await api.storage(); } catch (e) { storage = { error: e.message }; }
    const box = $("#adminBody", mount);
    if (box && active === "storage") box.innerHTML = storageCard(storage);
  };
  const loadBackups = async () => {
    try { backups = await api.backups(); } catch (e) { backups = { error: e.message }; }
    const box = $("#adminBody", mount);
    if (box && active === "backups") box.innerHTML = backupsCard(backups);
  };
  const loadCloud = async (withPrices = true) => {
    try { cloud = await api.cloud(); } catch (e) { cloud = { error: e.message }; }
    if (withPrices && cloud && !cloud.error && cloud.settings?.api_key_set) {
      try { gpus = await api.cloudGpus("any"); } catch (e) { gpus = { error: e.message }; }
    }
    const box = $("#adminBody", mount);
    if (box && active === "cloud") box.innerHTML = cloudCard(cloud, gpus);
  };
  if (active === "storage") loadStorage();
  if (active === "backups") loadBackups();
  if (active === "cloud") loadCloud();

  status = await api.status().catch((e) => ({ error: e.message }));
  draw();
  wireRibbon(mount, (key) => {
    // Drawn here and the address corrected afterwards, rather than navigating
    // and letting the router redraw: the router would remount this view and
    // throw away the storage report it had already fetched.
    active = key;
    draw();
    if (key === "storage") loadStorage();
    if (key === "backups") loadBackups();
    if (key === "cloud") loadCloud();
    history.replaceState(null, "", `#/admin/${key}`);
  });

  // ---- storage
  on(mount, "submit", "#retentionForm", async (e) => {
    e.preventDefault();
    try {
      await api.saveStorageSettings(
        Object.fromEntries(new FormData(e.target).entries()));
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

  // ---- backups
  on(mount, "submit", "#backupForm", async (e) => {
    e.preventDefault();
    const f = new FormData(e.target);
    try {
      await api.saveBackupSettings({
        dir: f.get("dir"),
        every_hours: Number(f.get("every_hours") || 0),
        keep: Number(f.get("keep") || 7),
        include_models: f.get("include_models") === "on",
      });
      toast("Saved.", "ok");
      await loadBackups();
    } catch (ex) { toast(ex.message, "err"); }
  });
  on(mount, "click", "#backupNow", async (_e, btn) => {
    btn.disabled = true;
    toast("Backing up… the database is copied while it runs, so nothing stops.");
    try {
      const r = await api.backUpNow({
        include_models: $("#bkIncludeNow", mount)?.checked });
      toast(`Backed up: ${fmtBytes(r.bytes)} in ${r.seconds}s.`, "ok");
      await loadBackups();
    } catch (ex) { toast(ex.message, "err"); }
    btn.disabled = false;
  });
  on(mount, "click", "[data-del-backup]", async (_e, t) => {
    if (!await confirmDestructive({
      title: "Delete this backup?",
      consequences: ["It is the copy, not the original — but if the original "
                     + "is lost after this, it is lost."],
      confirmLabel: "Delete it" })) return;
    try {
      await api.deleteBackup(t.dataset.delBackup);
      toast("Deleted.", "ok");
      await loadBackups();
    } catch (ex) { toast(ex.message, "err"); }
  });

  // ---- rented GPUs
  on(mount, "submit", "#cloudForm", async (e) => {
    e.preventDefault();
    const f = new FormData(e.target);
    const body = {
      enabled: f.get("enabled") === "on",
      max_price_per_hour: Number(f.get("max_price_per_hour") || 0),
      max_pods: Number(f.get("max_pods") || 0),
      daily_cap: Number(f.get("daily_cap") || 0),
      hard_cap: f.get("hard_cap") === "on",
      idle_minutes: Number(f.get("idle_minutes") || 15),
      max_hours_per_pod: Number(f.get("max_hours_per_pod") || 0),
      default_cloud: f.get("default_cloud"),
      strategy: f.get("strategy"),
      image: f.get("image"),
      disk_gb: Number(f.get("disk_gb") || 40),
      volume_gb: Number(f.get("volume_gb") || 80),
      controller_url: f.get("controller_url") || "",
    };
    if (f.get("api_key")) body.api_key = f.get("api_key");
    try {
      await api.saveCloudSettings(body);
      toast("Saved. The studio checks for work every 30 seconds.", "ok");
      await loadCloud();
    } catch (ex) { toast(ex.message, "err"); }
  });
  on(mount, "click", "#cloudClearKey", async () => {
    if (!await confirmDestructive({
      title: "Remove the RunPod API key?",
      consequences: ["No new pods can be started, and running ones can no longer be "
                     + "stopped from here — stop them first, or in the RunPod console."],
      confirmLabel: "Remove the key" })) return;
    try { await api.saveCloudSettings({ clear_api_key: true }); await loadCloud(); }
    catch (ex) { toast(ex.message, "err"); }
  });
  on(mount, "click", "#cloudRefresh", () => loadCloud());
  on(mount, "click", "[data-start-gpu]", async (_e, t) => {
    const [id, cl] = t.dataset.startGpu.split("|");
    const o = (gpus?.gpus || []).find((g) => g.id === id && g.cloud === cl);
    if (!await confirmDestructive({
      title: `Start a ${o?.name || id} on ${cl.toLowerCase()} cloud?`,
      consequences: [`It costs about $${(o?.price ?? 0).toFixed(2)} an hour from now until it is `
                     + "deleted — automatically once it has been idle, or by hand.",
                     "It joins the machines as a runner and takes queued GPU work."],
      confirmLabel: "Start it" })) return;
    try { await api.startCloudPod(id, cl); toast("Starting. It appears under Machines once it has booted.", "ok"); await loadCloud(false); }
    catch (ex) { toast(ex.message, "err"); }
  });
  on(mount, "click", "[data-drain-pod]", async (_e, t) => {
    try { await api.drainCloudPod(t.dataset.drainPod); toast("Stopping: checkpoints move to the studio first, then the pod is deleted.", "ok"); await loadCloud(false); }
    catch (ex) { toast(ex.message, "err"); }
  });
  on(mount, "click", "[data-delete-pod]", async (_e, t) => {
    if (!await confirmDestructive({
      title: "Delete this pod now?",
      consequences: ["Right away, without moving checkpoints to the studio: a run that "
                     + "was interrupted there starts again from its beginning.",
                     "Use Stop instead unless the pod is stuck."],
      confirmLabel: "Delete now" })) return;
    try { await api.deleteCloudPod(t.dataset.deletePod); await loadCloud(false); }
    catch (ex) { toast(ex.message, "err"); }
  });

  return events.subscribe((m) => {
    if (m.type === "cloud_changed" && active === "cloud") loadCloud(false);
    if (m.type === "runners_changed") {
      api.status().then((s) => { status = s; }).catch(() => {});
    }
  });
}

function overview(status) {
  const cards = [
    ["◍", "People", "Who can sign in, who is waiting for approval, and who "
      + "may administer this studio.", "#/users"],
    ["🔑", "Single sign-on", "Sign in with an identity provider instead of a "
      + "password here.", "#/sso"],
    ["▦", "Machines", "The runners that do the work: what they have, what "
      + "they are doing, and what they will accept.", "#/runners"],
    ["🏷", "Served names & usage", "The names the API answers to, what they "
      + "point at, and who has been calling them.", "#/serving"],
    ["🗄", "Storage", "What the disk is holding, which runs are the big ones, "
      + "and the rules for letting go of them.", "#/admin/storage"],
    ["⭳", "Backups", "A copy of the database, the datasets and the uploads, "
      + "and how to put it back.", "#/admin/backups"],
    ["☁", "Cloud GPUs", "RunPod GPUs rented for queued work when no machine of "
      + "the studio can take it — within your price and budget limits.", "#/admin/cloud"],
    ["⚙", "Settings", "The join token, Hugging Face access and the rest of "
      + "the studio-wide configuration.", "#/settings"],
  ];
  return html`
    ${raw(status && status.error ? "" : html`
      <div class="row" style="gap:24px;flex-wrap:wrap;margin-bottom:14px">
        <div class="stat"><span class="v">${status?.runners_online ?? "–"}</span>
          <span class="k">machines online</span></div>
        <div class="stat"><span class="v">${status?.jobs_running ?? "–"}</span>
          <span class="k">running now</span></div>
        <div class="stat"><span class="v">${status?.jobs_queued ?? "–"}</span>
          <span class="k">queued</span></div>
        <div class="stat"><span class="v">${esc(status?.version || "–")}</span>
          <span class="k">version</span></div>
      </div>`)}
    <div class="grid grid-2">
      ${raw(cards.map(([ico, title, body, href]) => html`
        <a class="card card-link" href="${href}">
          <h3 style="margin:0">${ico} ${title}</h3>
          <p class="muted tiny" style="margin:6px 0 0">${body}</p>
        </a>`).join(""))}
    </div>`;
}

// ---- backups --------------------------------------------------------------

/** What has been copied, where it goes, and what to do with it afterwards.
 *
 *  A backup nobody has ever restored is a rumour, so the list says exactly
 *  what is in each one and every backup carries its own RESTORE.md — the
 *  instructions travel with the copy, where they will still be legible when
 *  this page is the thing that is gone.
 */
function backupsCard(b) {
  if (!b) return `<div class="card"><h3>Backups</h3>
    <p class="muted tiny">Looking…</p></div>`;
  if (b.error) return `<div class="card"><h3>Backups</h3>
    <p class="muted tiny">${esc(b.error)}</p></div>`;
  const s = b.settings;
  const rows = b.backups || [];
  const last = b.last;

  return html`
    <div class="card" style="margin-bottom:14px">
      <div class="row-between" style="flex-wrap:wrap;gap:8px">
        <h3 style="margin:0">Backups</h3>
        <span class="tiny muted">${last
          ? `last ${fmtAgo(last.taken_at)} · ${fmtBytes(last.bytes || 0)}`
          : "never backed up"}</span>
      </div>
      <p class="muted tiny" style="margin:6px 0 0;max-width:78ch">
        Each backup is a folder holding a consistent copy of the database
        (taken while it is running, so nothing has to stop), the join token,
        your datasets and the uploaded pictures and clips. Trained models are
        excluded by default: they are the largest thing here by far and they
        can be trained again, while the record of what produced them cannot.
      </p>
      ${raw(last && last.error ? `<div class="callout callout-err"
        style="margin-top:10px"><strong>The last backup failed</strong>${
          esc(last.error)}</div>` : "")}

      <form id="backupForm" class="row" style="gap:10px;flex-wrap:wrap;
            align-items:flex-end;margin-top:12px">
        <div class="field" style="margin:0;min-width:260px;flex:1">
          <label for="bkDir">Where they go</label>
          <input id="bkDir" name="dir" value="${esc(s.dir)}" class="mono">
          <div class="hint">A path on the controller. Point it at a mounted
            share and the copy leaves this machine, which is the only kind of
            backup that survives the machine.</div>
        </div>
        <div class="field" style="margin:0">
          <label for="bkEvery">Automatically every</label>
          <input id="bkEvery" name="every_hours" type="number" min="0" max="720"
                 value="${s.every_hours}" style="width:90px"> hours
          <div class="hint">0 for manual only.</div>
        </div>
        <div class="field" style="margin:0">
          <label for="bkKeep">Keep the last</label>
          <input id="bkKeep" name="keep" type="number" min="1" max="365"
                 value="${s.keep}" style="width:90px">
          <div class="hint">Older ones are deleted after a successful new one.</div>
        </div>
        <label class="check" style="margin:0 0 10px">
          <input type="checkbox" name="include_models" ${raw(s.include_models ? "checked" : "")}>
          Include trained models</label>
        <button class="btn-primary btn-sm" type="submit">Save</button>
      </form>
      <label class="check" style="margin-top:6px">
        <input type="checkbox" id="bkIncludeNow" ${raw(s.include_models ? "checked" : "")}>
        Include models in the backup started from the ribbon</label>
    </div>

    <div class="card" style="padding:0">
      <div class="row-between" style="padding:12px 16px">
        <h3 style="margin:0">Copies on disk</h3>
        <span class="muted tiny">${rows.length}</span>
      </div>
      ${raw(rows.length ? html`
        <div class="table-wrap"><table><thead><tr>
          <th>When</th><th>Size</th><th class="hide-sm">Holds</th><th></th>
        </tr></thead><tbody>
          ${raw(rows.map((r) => html`
            <tr>
              <td>${fmtAgo(r.taken_at)}<div class="muted tiny mono">${esc(r.stamp || "")}</div></td>
              <td class="mono">${fmtBytes(r.bytes || 0)}</td>
              <td class="tiny muted hide-sm">
                ${fmtNum(r.runs || 0)} runs ·
                ${fmtNum(r.datasets || 0)} datasets ·
                ${fmtNum(r.users || 0)} people ·
                ${r.include_models ? "models included" : "no models"}</td>
              <td style="text-align:right">
                <button class="btn-sm btn-danger" data-del-backup="${esc(r.path)}">Delete</button></td>
            </tr>`).join(""))}
        </tbody></table></div>` : `<p class="muted tiny" style="padding:0 16px 14px;margin:0">
          None yet. “Back up now” makes the first one.</p>`)}
      <p class="muted tiny" style="padding:12px 16px;margin:0">
        To put one back: stop the controller, run
        <code class="mono">scripts/restore.sh &lt;backup folder&gt;</code> on
        this machine, start it again. The same instructions are written into
        every backup as <code class="mono">RESTORE.md</code>.</p>
    </div>`;
}

// ---- storage (moved here from Settings, where it never belonged) ----------

const gb = (b) => (b / 1024 ** 3).toFixed(b >= 10 * 1024 ** 3 ? 0 : 1) + " GB";
const mb = (b) => (b >= 1024 ** 3 ? gb(b) : (b / 1024 ** 2).toFixed(0) + " MB");

function storageCard(st) {
  if (!st) return `<div class="card"><h3>Storage</h3><p class="muted tiny">Reading the disk…</p></div>`;
  if (st.error) return `<div class="card"><h3>Storage</h3><p class="muted tiny">${esc(st.error)}</p></div>`;
  const d = st.disk;
  const s = st.settings;
  const last = st.last_sweep;
  const usedClass = d && d.used_pct >= 95 ? "callout-err"
    : d && d.used_pct >= 85 ? "callout-warn" : "callout-ok";
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
      </form>
      ${raw(last ? html`
        <p class="muted tiny" style="margin:8px 0 0">Last tidy: ${last.models_removed} model${last.models_removed === 1 ? "" : "s"} removed,
          ${last.runs_thinned} run${last.runs_thinned === 1 ? "" : "s"} thinned,
          ${last.orphan_files} orphan file${last.orphan_files === 1 ? "" : "s"} swept,
          ${mb(last.bytes_freed)} freed.</p>` : "")}
    </div>`;
}


// ---------------------------------------------------------------- cloud GPUs

const money = (v) => "$" + Number(v || 0).toFixed(2);

function cloudCard(c, g) {
  if (!c) return `<div class="card"><h3>Cloud GPUs</h3><p class="muted tiny">Looking…</p></div>`;
  if (c.error) return `<div class="card"><h3>Cloud GPUs</h3><p class="muted tiny">${esc(c.error)}</p></div>`;
  if (!c.settings) return `<div class="card"><h3>Cloud GPUs</h3><p class="muted tiny">Looking…</p></div>`;
  const s = c.settings;
  const pct = s.daily_cap > 0 ? Math.min(100, (c.spend_today / s.daily_cap) * 100) : 0;
  const active = (c.pods || []).filter((p) => ["starting", "running", "draining"].includes(p.status));
  const ended = (c.pods || []).filter((p) => !["starting", "running", "draining"].includes(p.status));
  const opt = (v, cur, label) => `<option value="${v}" ${v === cur ? "selected" : ""}>${label}</option>`;

  const podRow = (p) => html`<tr>
      <td><strong>${p.gpu_name || p.gpu_id}</strong><div class="tiny muted">${p.name}</div></td>
      <td>${p.cloud.toLowerCase()}</td>
      <td>${money(p.price_per_hour)}/h</td>
      <td><span class="badge">${p.status}</span>${raw(p.note ? `<div class="tiny muted">${esc(p.note)}</div>` : "")}</td>
      <td>${money(p.spent)}</td>
      <td class="tiny muted">${p.reason || ""}<div>${fmtAgo(p.created_at)}</div></td>
      <td style="white-space:nowrap">${raw(["starting", "running"].includes(p.status)
        ? `<button class="btn-sm" data-drain-pod="${esc(p.id)}">Stop</button>
           <button class="btn-sm btn-danger" data-delete-pod="${esc(p.id)}">Delete now</button>`
        : p.status === "draining"
        ? `<button class="btn-sm btn-danger" data-delete-pod="${esc(p.id)}">Delete now</button>` : "")}</td>
    </tr>`;

  const offers = g && !g.error ? (g.gpus || []) : [];
  const offerRow = (o) => html`<tr class="${o.within_price_cap && o.availability !== "NONE" ? "" : "muted"}">
      <td><strong>${o.name}</strong></td>
      <td>${o.vram_gb} GB</td>
      <td>${o.cloud.toLowerCase()}</td>
      <td>${money(o.price)}/h${raw(o.within_price_cap ? "" : ` <span class="tiny">(over the cap)</span>`)}</td>
      <td>${o.availability.toLowerCase()}</td>
      <td>${o.speed.toFixed(1)}× <span class="tiny muted">${money(o.usd_per_speed)} per unit</span></td>
      <td>${raw(o.within_price_cap && o.availability !== "NONE" && s.api_key_set
        ? `<button class="btn-sm" data-start-gpu="${esc(o.id)}|${esc(o.cloud)}">Start</button>` : "")}</td>
    </tr>`;

  return html`
    <div class="card" style="margin-bottom:14px">
      <div class="row-between" style="flex-wrap:wrap;gap:8px">
        <h3 style="margin:0">Cloud GPUs (RunPod)</h3>
        <span class="tiny muted">${s.enabled ? "renting when needed" : "off — nothing is rented automatically"}</span>
      </div>
      <p class="muted tiny" style="margin:6px 0 0;max-width:80ch">
        When a GPU run is queued and no machine of the studio could take it, the studio rents
        the fastest (or cheapest) RunPod GPU the run fits on, within the limits below. The pod
        joins as an ordinary runner. Once it has been idle, whatever exists only on it — the
        checkpoints of unfinished runs — moves to the studio, and the pod is deleted. Finished
        models reach the studio at the end of every run anyway.</p>
      <div style="margin-top:12px">
        <div class="row-between tiny"><span>Spent today ${money(c.spend_today)}</span>
          <span class="muted">cap ${s.daily_cap > 0 ? money(s.daily_cap) : "none"}</span></div>
        <div class="bar" style="height:8px;background:var(--border);border-radius:4px;overflow:hidden">
          <div style="width:${pct}%;height:100%;background:${pct >= 100 ? "var(--err)" : pct > 75 ? "var(--warn)" : "var(--ok)"}"></div></div>
        <div class="tiny muted" style="margin-top:4px">Estimated from each pod's hourly price;
          RunPod's billing is the exact figure.</div>
      </div>
      ${raw(c.last_error ? `<div class="callout callout-warn" style="margin-top:10px"><strong>RunPod</strong>${esc(c.last_error)}</div>` : "")}

      <form id="cloudForm" class="row" style="gap:10px;flex-wrap:wrap;align-items:flex-end;margin-top:14px">
        <div class="field" style="margin:0;min-width:260px;flex:1">
          <label for="cgKey">RunPod API key</label>
          <input id="cgKey" name="api_key" type="password" autocomplete="off"
                 placeholder="${s.api_key_set ? "set (" + s.api_key_hint + ") — type to replace" : "rpa_…"}">
          <div class="hint">Stored encrypted; never shown again.${raw(s.api_key_set ? ` <a href="#" id="cloudClearKey">Remove</a>` : "")}</div>
        </div>
        <label class="check" style="margin:0 0 10px"><input type="checkbox" name="enabled" ${raw(s.enabled ? "checked" : "")}>
          Rent automatically</label>
        <div class="field" style="margin:0"><label>Max price per GPU-hour</label>
          $<input name="max_price_per_hour" type="number" step="0.01" min="0" value="${s.max_price_per_hour}" style="width:80px"></div>
        <div class="field" style="margin:0"><label>Pods at once</label>
          <input name="max_pods" type="number" min="0" max="16" value="${s.max_pods}" style="width:70px"></div>
        <div class="field" style="margin:0"><label>Daily budget</label>
          $<input name="daily_cap" type="number" step="1" min="0" value="${s.daily_cap}" style="width:80px">
          <div class="hint">0 for no cap.</div></div>
        <label class="check" style="margin:0 0 10px"><input type="checkbox" name="hard_cap" ${raw(s.hard_cap ? "checked" : "")}>
          Pause running pods at the cap</label>
        <div class="field" style="margin:0"><label>Give back after idle</label>
          <input name="idle_minutes" type="number" min="1" max="1440" value="${s.idle_minutes}" style="width:70px"> min</div>
        <div class="field" style="margin:0"><label>Longest life per pod</label>
          <input name="max_hours_per_pod" type="number" min="0" max="168" value="${s.max_hours_per_pod}" style="width:70px"> h
          <div class="hint">0 for no limit.</div></div>
        <div class="field" style="margin:0"><label>Runs may use</label>
          <select name="default_cloud">${raw(opt("secure", s.default_cloud, "Secure Cloud only")
            + opt("any", s.default_cloud, "Secure or Community Cloud") + opt("never", s.default_cloud, "No cloud GPUs"))}</select>
          <div class="hint">The default; each run can choose for itself.</div></div>
        <div class="field" style="margin:0"><label>Prefer</label>
          <select name="strategy">${raw(opt("fastest", s.strategy, "the fastest GPU under the cap")
            + opt("cheapest", s.strategy, "the cheapest GPU that fits"))}</select></div>
        <div class="field" style="margin:0;min-width:320px;flex:1"><label>Runner image</label>
          <input name="image" class="mono" value="${esc(s.image)}"></div>
        <div class="field" style="margin:0"><label>Disk / volume</label>
          <input name="disk_gb" type="number" min="20" value="${s.disk_gb}" style="width:70px"> /
          <input name="volume_gb" type="number" min="20" value="${s.volume_gb}" style="width:70px"> GB</div>
        <div class="field" style="margin:0;min-width:240px"><label>Studio address for pods</label>
          <input name="controller_url" class="mono" value="${esc(s.controller_url)}" placeholder="AI_STUDIO_PUBLIC_URL">
          <div class="hint">How a pod on the internet reaches this studio.</div></div>
        <button class="btn-primary btn-sm" type="submit">Save</button>
      </form>
    </div>

    <div class="card" style="padding:0;margin-bottom:14px">
      <div class="row-between" style="padding:12px 16px"><h3 style="margin:0">Pods</h3>
        <span class="tiny muted">${active.length} running · ${s.max_pods} allowed</span></div>
      ${raw(active.length || ended.length ? html`<table class="table"><thead><tr><th>GPU</th><th>Cloud</th>
        <th>Price</th><th>Status</th><th>Spent</th><th>Why</th><th></th></tr></thead>
        <tbody>${raw(active.concat(ended).map(podRow).join(""))}</tbody></table>`
        : `<p class="muted tiny" style="padding:0 16px 14px">None yet.</p>`)}
    </div>

    <div class="card" style="padding:0">
      <div class="row-between" style="padding:12px 16px"><h3 style="margin:0">What can be rented now</h3>
        <span class="tiny muted">speed relative to an RTX 3090, for training; a ranking, not a benchmark</span></div>
      ${raw(!s.api_key_set ? `<p class="muted tiny" style="padding:0 16px 14px">Add a RunPod API key to see prices and stock.</p>`
        : g && g.error ? `<p class="muted tiny" style="padding:0 16px 14px">${esc(g.error)}</p>`
        : !g ? `<p class="muted tiny" style="padding:0 16px 14px">Asking RunPod…</p>`
        : html`<table class="table"><thead><tr><th>GPU</th><th>Memory</th><th>Cloud</th><th>Price</th>
            <th>Stock</th><th>Speed</th><th></th></tr></thead>
            <tbody>${raw(offers.map(offerRow).join(""))}</tbody></table>`)}
    </div>`;
}
