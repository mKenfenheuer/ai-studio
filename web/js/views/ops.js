/**
 * Operations: what is being served, on what, how fast, and how badly.
 *
 * The studio could train a model, name it and serve it over an API, and then
 * had nothing to say about any of it. This is the page that answers the
 * questions somebody asks once a model is the thing another system depends on
 * — which is a different set of questions from the ones training raises, and
 * they had no home.
 *
 *   Overview     is it up, is it fast, is it failing, and since when.
 *   Deployments  which model is held on which card, and whether it really is.
 *   Machines     what each one is set aside for.
 *   Keys         every key in the studio, and the button that turns one off.
 *
 * Names and long-run usage stay on the Served models page: this one is the
 * live view, in hours rather than months.
 *
 * The numbers are the runner's own — real token counts, real seconds around
 * the generation. Where there is nothing to divide, a dash is drawn rather
 * than a zero: a fleet that has served nothing is not a fleet running at zero
 * tokens a second, and every one of these panels is read by somebody deciding
 * whether something is wrong.
 */
import { api, events } from "../api.js";
import { session } from "../app.js";
import { html, raw, esc, $, on, toast, fmtAgo, fmtNum, fmtDuration, modal } from "../util.js";
import { ribbon, rb, group, rbSelect, wireRibbon, tabState } from "../ribbon.js";
import { pageHead, emptyState, confirmDestructive } from "../components.js";

const TABS = [
  { key: "overview", label: "Overview" },
  { key: "deployments", label: "Deployments" },
  { key: "machines", label: "Machines" },
  { key: "keys", label: "Keys" },
];

// The windows worth offering. Hours, not days: this page is "what is
// happening", and the month-long view is the Served models usage tab.
const WINDOWS = [[1, "last hour"], [6, "last 6 hours"], [24, "last day"],
                 [72, "last 3 days"], [168, "last week"]];

const ROLES = [
  ["both", "Both", "Takes training runs and answers messages. The default."],
  ["serving", "Reserved for serving", "No training run is ever sent here, so "
    + "a deployed model is not pushed off the card by one."],
  ["training", "Reserved for training", "Never picked to answer a message."],
];

const STATE_BADGE = {
  ready: "badge-ok",
  loading: "badge-warn",
  pending: "badge-warn",
  failed: "badge-err",
};

export async function opsView(mount) {
  const tabs = tabState("ops", TABS, "overview");
  let tab = tabs.get();
  let hours = Number(localStorage.getItem("aistudio.opsHours")) || 24;
  let metrics = null, deployments = [], runners = [], keys = [], models = [];
  const admin = session.user?.role === "admin";

  const load = async () => {
    [metrics, deployments, runners] = await Promise.all([
      api.opsMetrics(hours).catch((e) => ({ error: e.message })),
      api.deployments().catch(() => []),
      api.runners().catch(() => []),
    ]);
    // Administrators only, and asked for only when the tab is open: it is the
    // one call here that is refused for everybody else, and a 403 in the
    // console on every page load is how people learn to ignore the console.
    if (admin && tab === "keys") keys = await api.studioKeys().catch(() => []);
  };

  const draw = () => {
    mount.innerHTML = html`
      ${raw(pageHead({
        title: "Operations",
        sub: "The models that are being served: where they are held, how fast "
           + "they answer, and what is going wrong.",
      }))}
      ${raw(ribbonFor({ tab, hours, admin, deployments }))}
      ${raw(
        tab === "deployments" ? deploymentsPanel(deployments, admin)
        : tab === "machines" ? machinesPanel(runners, deployments, admin)
        : tab === "keys" ? keysPanel(keys, admin)
        : overviewPanel(metrics, deployments, runners))}`;
    wire();
  };

  function wire() {
    wireRibbon(mount, async (key) => {
      tab = key; tabs.set(key);
      if (key === "keys" && admin && !keys.length) {
        keys = await api.studioKeys().catch(() => []);
      }
      draw();
    });
    on(mount, "change", "#opsWindow", async (_e, t) => {
      hours = Number(t.value) || 24;
      localStorage.setItem("aistudio.opsHours", String(hours));
      metrics = await api.opsMetrics(hours).catch((e) => ({ error: e.message }));
      draw();
    });
    on(mount, "click", "#deployNew", () => deployDialog());
    on(mount, "click", "[data-undeploy]", async (_e, t) => {
      const dep = deployments.find((d) => d.id === t.dataset.undeploy);
      if (!await confirmDestructive({
        title: `Stop holding ${dep?.job_name || "this model"} on ${dep?.runner_name || "that machine"}?`,
        consequences: [
          "The memory comes back straight away.",
          (dep?.aliases || []).length
            ? `Anything pointed at ${dep.aliases.map((a) => `"${a}"`).join(", ")} keeps working — the first message after this just pays the load again.`
            : "The model can still be talked to; the first message after this pays the load again.",
        ],
        confirmLabel: "Take it down" })) return;
      try {
        await api.undeploy(dep.id);
        await load(); draw();
        toast("Taken down.", "ok");
      } catch (e) { toast(e.message, "err"); }
    });
    on(mount, "click", "[data-retry-dep]", async (_e, t) => {
      try {
        await api.retryDeployment(t.dataset.retryDep);
        toast("Asked again. Watch the state here.", "ok");
        await load(); draw();
      } catch (e) { toast(e.message, "err"); }
    });
    on(mount, "click", "[data-role]", (_e, t) => roleDialog(
      runners.find((r) => r.id === t.dataset.role)));
    on(mount, "click", "[data-revoke-key]", async (_e, t) => {
      const key = keys.find((k) => k.id === t.dataset.revokeKey);
      if (!await confirmDestructive({
        title: `Turn off "${key?.name || "this key"}"?`,
        consequences: [
          "Anything using it starts getting a 401 immediately.",
          "It cannot be turned back on — a new key would have to be made.",
        ],
        confirmLabel: "Turn it off" })) return;
      try {
        await api.revokeStudioKey(key.id);
        keys = await api.studioKeys().catch(() => []);
        draw();
        toast("Revoked.", "ok");
      } catch (e) { toast(e.message, "err"); }
    });
  }

  // ---- deploying ---------------------------------------------------------
  function deployDialog() {
    const servers = runners.filter((r) =>
      ["cuda", "rocm", "mps"].includes((r.capabilities || {}).backend)
      && r.role !== "training");
    const dlg = modal({
      title: "Hold a model on a machine",
      width: 580,
      body: html`
        <p class="muted tiny">The model is loaded now and kept on the card, so
          the first message does not pay for a fetch and a load. It is put back
          by itself after a restart, and it is not evicted to make room for
          somebody trying a different model.</p>
        <div class="field">
          <label for="dpJob">Model</label>
          <select id="dpJob"><option value="">Loading finished runs…</option></select>
        </div>
        <div class="field">
          <label for="dpRunner">On</label>
          <select id="dpRunner">${raw(servers.length
            ? servers.map((r) => html`<option value="${r.id}">${r.name}${
                r.role === "serving" ? " · reserved for serving" : ""}${
                r.connected ? "" : " · offline"}</option>`).join("")
            : `<option value="">No machine here can serve a model</option>`)}</select>
          <div class="hint">A machine with no graphics card is not offered: it
            would answer at a word every few seconds.</div>
        </div>
        <div class="row" style="justify-content:flex-end;gap:8px;margin-top:12px">
          <button type="button" class="btn" data-modal-close>Cancel</button>
          <button type="button" class="btn btn-primary" id="dpGo">Deploy it</button>
        </div>` });

    const fill = (rows) => {
      const sel = $("#dpJob", dlg);
      if (!sel) return;
      sel.innerHTML = rows.length
        ? rows.map((m) => html`<option value="${m.id}">${m.name}</option>`).join("")
        : `<option value="">No finished runs to deploy</option>`;
    };
    if (models.length) fill(models);
    else {
      api.servableModels().then((r) => {
        models = (r.data || []).filter((m) => !m.is_alias);
        fill(models);
      }).catch(() => fill([]));
    }

    on(dlg, "click", "#dpGo", async (_e, btn) => {
      const jobId = $("#dpJob", dlg).value;
      const runnerId = $("#dpRunner", dlg).value;
      if (!jobId) return toast("Choose a model.", "err");
      if (!runnerId) return toast("Choose a machine.", "err");
      btn.disabled = true;
      try {
        await api.deploy({ job_id: jobId, runner_id: runnerId });
        dlg.close();
        tab = "deployments"; tabs.set(tab);
        await load(); draw();
        toast("Loading it onto the card. This takes a minute or two.", "ok");
      } catch (e) { toast(e.message, "err"); btn.disabled = false; }
    });
  }

  // ---- reserving ---------------------------------------------------------
  function roleDialog(runner) {
    if (!runner) return;
    const dlg = modal({
      title: `What is ${runner.name} for?`,
      width: 560,
      body: html`
        <p class="muted tiny">A studio with two cards usually wants one of them
          answering messages and the other training. Until this existed the
          scheduler handed a run to whichever machine was idle — which is
          reliably the machine that was about to be asked a question.</p>
        ${raw(ROLES.map(([value, label, why]) => html`
          <label class="choice">
            <input type="radio" name="role" value="${value}"
                   ${runner.role === value ? "checked" : ""}>
            <span><strong>${label}</strong><br>
              <span class="muted tiny">${why}</span></span>
          </label>`).join(""))}
        <div class="field" style="margin-top:10px">
          <label for="rlNote">Why <span class="muted tiny">(optional)</span></label>
          <input id="rlNote" type="text" value="${runner.note || ""}"
                 placeholder="Serves the support assistant">
          <div class="hint">For whoever finds this machine reserved next month.</div>
        </div>
        <div class="row" style="justify-content:flex-end;gap:8px;margin-top:12px">
          <button type="button" class="btn" data-modal-close>Cancel</button>
          <button type="button" class="btn btn-primary" id="rlGo">Save</button>
        </div>` });

    on(dlg, "click", "#rlGo", async (_e, btn) => {
      const role = dlg.querySelector("input[name=role]:checked")?.value;
      if (!role) return toast("Choose one.", "err");
      btn.disabled = true;
      try {
        await api.setRunnerRole(runner.id, { role, note: $("#rlNote", dlg).value });
        dlg.close();
        await load(); draw();
        toast("Saved.", "ok");
      } catch (e) { toast(e.message, "err"); btn.disabled = false; }
    });
  }

  await load();
  draw();
  // Deployments change state on their own — a load finishes, a machine comes
  // back, a run takes the card — so this page follows the fleet rather than
  // showing whatever was true when it was opened.
  return events.subscribe(async (m) => {
    if (["deployments_changed", "runners_changed"].includes(m.type)) {
      await load(); draw();
    }
  });
}

function ribbonFor({ tab, hours, admin, deployments }) {
  const ready = deployments.filter((d) => d.state === "ready").length;
  const body = group("Models", [
    rb("deployNew", "⊕", "Deploy a model", { cls: "primary" }),
  ]) + group("Window", [
    rbSelect("opsWindow", {
      title: "How far back",
      value: String(hours),
      options: WINDOWS.map(([h, label]) => [String(h), label]),
    }),
  ]) + group("Elsewhere", [
    rb(null, "🏷", "Names & usage", { href: "#/serving" }),
    rb(null, "▦", "Machines", { href: "#/runners" }),
    rb(null, "🔑", "Your keys", { href: "#/account" }),
  ]);
  return ribbon({
    tabs: admin ? TABS : TABS.filter((t) => t.key !== "keys"),
    active: tab, body,
    right: `<span class="badge${ready ? " badge-ok" : ""}">${ready} held on a card</span>`,
  });
}

// ---------------------------------------------------------------- overview

const fmtTokens = (n) => {
  n = n || 0;
  if (n < 1000) return String(n);
  if (n < 1e6) return (n / 1000).toFixed(n < 1e4 ? 1 : 0) + "k";
  return (n / 1e6).toFixed(1) + "M";
};

// A rate we have no basis for is a dash, never a zero. See the file's header.
const fmtRate = (r) => (r == null ? "—" : `${r} tok/s`);

function overviewPanel(m, deployments, runners) {
  if (!m || m.error) {
    return emptyState({ icon: "📉", title: "The numbers could not be read",
      body: m?.error || "Try again in a moment." });
  }
  const t = m.totals || {};
  if (!t.calls) {
    return emptyState({
      icon: "◷",
      title: "Nothing has been served in this window",
      body: "Every reply this studio produces — over the API and in the "
          + "playground alike — is counted here, with the runner's own token "
          + `counts. Counts are kept for ${m.kept_days} days.`,
      cta: { href: "#/play", label: "Talk to a model" },
    });
  }

  const serving = runners.filter((r) => r.role === "serving").length;
  const ready = deployments.filter((d) => d.state === "ready").length;
  const broken = deployments.filter((d) => d.state === "failed");

  return html`
    ${raw(broken.length ? `
      <div class="callout callout-err" style="margin-bottom:14px">
        <strong>${broken.length} deployment${broken.length === 1 ? " is" : "s are"} not loaded</strong>
        ${esc(broken.map((d) => `${d.job_name} on ${d.runner_name}`).join("; "))}.
        <a href="#/ops">See Deployments</a> for the reason.</div>` : "")}
    <div class="card" style="margin-bottom:14px">
      <div class="row-between">
        <h3 style="margin:0">Last ${m.hours === 1 ? "hour" : `${m.hours} hours`}</h3>
        <span class="tiny muted">${m.scope === "studio"
          ? "the whole studio" : "your own calls"}</span>
      </div>
      <div class="row" style="gap:28px;margin-top:10px;flex-wrap:wrap">
        <div class="stat"><span class="v">${fmtNum(t.calls)}</span>
          <span class="k">replies</span></div>
        <div class="stat"><span class="v">${fmtRate(t.rate)}</span>
          <span class="k">while generating</span></div>
        <div class="stat"><span class="v">${fmtTokens(t.completion_tokens)}</span>
          <span class="k">tokens out</span></div>
        <div class="stat"><span class="v">${fmtTokens(t.prompt_tokens)}</span>
          <span class="k">tokens in</span></div>
        <div class="stat"><span class="v ${t.errors ? "bad" : ""}">${fmtNum(t.errors)}</span>
          <span class="k">failed${t.error_rate != null ? ` · ${t.error_rate}%` : ""}</span></div>
        <div class="stat"><span class="v">${ready}</span>
          <span class="k">models held${serving ? ` · ${serving} machine reserved` : ""}</span></div>
      </div>
      <p class="muted tiny" style="margin:10px 0 0">Tokens per second is the
        tokens produced divided by the seconds spent producing them — not an
        average of per-reply rates, which would weight a two-token reply the
        same as a two-thousand-token one.</p>
    </div>
    ${raw(sparkCard(m.series || []))}
    ${raw(m.recent_errors?.length ? errorsCard(m.recent_errors) : "")}
    ${raw(rateTable("By model", "Run", m.by_model, (r) => r.gone
      ? `<span class="muted">${esc(r.name)}</span> <span class="badge badge-warn">deleted</span>`
      : `<a href="#/jobs/${esc(r.job_id)}">${esc(r.name)}</a>`))}
    ${raw(rateTable("By machine", "Machine", m.by_runner,
      (r) => esc(r.name)))}
    ${raw(m.by_alias?.length
      ? rateTable("By name", "Name", m.by_alias, (r) => `<code>${esc(r.key)}</code>`)
      : "")}
    ${raw(rateTable("By key", "Key", m.by_key, (r) => `${esc(r.name)}${
      r.prefix ? ` <span class="muted tiny mono">${esc(r.prefix)}…</span>` : ""}`))}`;
}

/**
 * Calls per bucket, as bars, with the failures marked in the same bar.
 *
 * Deliberately not a separate error chart. "Traffic went up and so did
 * failures" and "traffic went up and failures did not" are the same shape in
 * two charts and obviously different in one.
 */
function sparkCard(series) {
  if (series.length < 2) return "";
  const peak = Math.max(...series.map((p) => p.calls || 0), 1);
  const bars = series.map((p) => {
    const h = Math.round(100 * (p.calls || 0) / peak);
    const bad = p.calls ? Math.round(100 * (p.errors || 0) / p.calls) : 0;
    const when = new Date((p.t || 0) * 1000).toLocaleString();
    return `<div class="opsbar" style="height:${Math.max(h, 2)}%"
      title="${esc(`${when}: ${p.calls} replies, ${p.errors || 0} failed`)}">
      ${bad ? `<span class="opsbar-bad" style="height:${bad}%"></span>` : ""}
    </div>`;
  }).join("");
  return html`
    <div class="card" style="margin-bottom:14px">
      <h3 style="margin:0 0 8px">Replies over time</h3>
      <div class="opschart">${raw(bars)}</div>
      <p class="muted tiny" style="margin:8px 0 0">Each bar is one hour. The
        red part is the share of that hour's replies that failed.</p>
    </div>`;
}

function errorsCard(rows) {
  return html`
    <div class="card" style="margin-bottom:14px;padding:0">
      <h3 style="margin:0;padding:14px 16px 0">What failed</h3>
      <div class="table-wrap"><table>
        <thead><tr><th>When</th><th>Model</th><th class="hide-sm">Where from</th>
          <th>Reason</th></tr></thead>
        <tbody>
          ${raw(rows.map((r) => html`
            <tr>
              <td class="tiny muted">${fmtAgo(r.ts)}</td>
              <td class="tiny">${raw(r.alias
                ? `<code>${esc(r.alias)}</code>`
                : `<a href="#/jobs/${esc(r.job_id)}">${esc(r.job_id)}</a>`)}</td>
              <td class="tiny muted hide-sm">${r.source === "playground"
                ? "the playground" : "the API"}</td>
              <td class="tiny">${esc(r.error || "no reason recorded")}</td>
            </tr>`).join(""))}
        </tbody>
      </table></div>
    </div>`;
}

function rateTable(title, heading, rows, label) {
  if (!rows?.length) return "";
  return html`
    <div class="card" style="margin-bottom:14px;padding:0">
      <h3 style="margin:0;padding:14px 16px 0">${title}</h3>
      <div class="table-wrap"><table>
        <thead><tr><th>${heading}</th><th>Replies</th><th>Speed</th>
          <th class="hide-sm">Tokens out</th><th>Failed</th><th>Last</th></tr></thead>
        <tbody>
          ${raw(rows.map((r) => html`
            <tr>
              <td>${raw(label(r))}</td>
              <td>${fmtNum(r.calls)}</td>
              <td class="mono tiny">${fmtRate(r.rate)}</td>
              <td class="hide-sm">${fmtTokens(r.completion_tokens)}</td>
              <td>${raw(r.errors
                ? `<span class="badge badge-err">${r.errors}</span>`
                : `<span class="muted">—</span>`)}</td>
              <td class="tiny muted">${fmtAgo(r.last_used)}</td>
            </tr>`).join(""))}
        </tbody>
      </table></div>
    </div>`;
}

// ------------------------------------------------------------- deployments

function deploymentsPanel(rows, admin) {
  if (!rows.length) {
    return html`
      <div class="card empty">
        <div class="big" aria-hidden="true">📌</div>
        <h3>No model is being held on a card</h3>
        <p class="muted">A model is fetched and loaded the first time somebody
          talks to it, which takes a minute or two, and it is dropped again
          when the card is needed for something else. Deploying one keeps it
          there: the first message is as fast as the tenth, and it is put back
          by itself after a restart.</p>
        <p><button class="btn btn-primary" id="deployNew">Deploy a model</button></p>
      </div>`;
  }
  return html`
    <div class="card" style="padding:0">
      <div class="table-wrap"><table>
        <thead><tr>
          <th>Model</th><th>On</th><th>State</th>
          <th class="hide-sm">Serving</th><th></th>
        </tr></thead>
        <tbody>
          ${raw(rows.map((d) => html`
            <tr>
              <td>${raw(d.job_gone
                ? `<span class="muted">the run is gone</span>`
                : d.job_id
                  ? `<a href="#/jobs/${esc(d.job_id)}">${esc(d.job_name)}</a>`
                  : `<span class="muted tiny">${esc(d.job_name)}</span>`)}
                ${raw(d.load_s ? `<div class="muted tiny">loaded in ${
                  fmtDuration(d.load_s)}</div>` : "")}</td>
              <td>${esc(d.runner_name)}
                ${raw(d.runner_online ? "" :
                  `<div class="muted tiny">offline</div>`)}</td>
              <td>
                <span class="badge ${STATE_BADGE[d.state] || ""}">${
                  d.state === "ready" && !d.resident ? "putting it back" : esc(d.state)}</span>
                ${raw(d.detail ? `<div class="muted tiny">${esc(d.detail)}</div>` : "")}</td>
              <td class="hide-sm">${raw((d.aliases || []).length
                ? d.aliases.map((a) => `<code>${esc(a)}</code>`).join(" ")
                : `<span class="muted">—</span>`)}</td>
              <td><div class="row" style="gap:4px">
                ${raw(d.state === "failed"
                  ? `<button class="btn-sm" data-retry-dep="${esc(d.id)}">Try again</button>`
                  : "")}
                <button class="btn-sm btn-danger" data-undeploy="${esc(d.id)}"
                        title="Stop holding it">✕</button>
              </div></td>
            </tr>`).join(""))}
        </tbody>
      </table></div>
      <p class="muted tiny" style="padding:10px 16px 14px;margin:0">
        A deployed model is not evicted to make room for another one. A
        training run still takes the whole card — reserve a machine for
        serving on the Machines tab if that matters.</p>
    </div>`;
}

// ---------------------------------------------------------------- machines

function machinesPanel(runners, deployments, admin) {
  if (!runners.length) {
    return emptyState({ icon: "🖥️", title: "No machines are connected",
      body: "Connect one from the Machines page.",
      cta: { href: "#/runners", label: "Machines" } });
  }
  const held = (id) => deployments.filter((d) => d.runner_id === id);
  return html`
    <div class="card" style="padding:0">
      <div class="table-wrap"><table>
        <thead><tr><th>Machine</th><th>Set aside for</th>
          <th class="hide-sm">On its card</th><th>Doing</th><th></th></tr></thead>
        <tbody>
          ${raw(runners.map((r) => {
            const mine = held(r.id);
            const loaded = (r.loaded || []).length;
            return html`
            <tr>
              <td><a href="#/runners">${esc(r.name)}</a>
                <div class="muted tiny">${esc((r.capabilities || {}).device_name
                  || (r.capabilities || {}).backend || "unknown")}</div></td>
              <td>
                <span class="badge${r.role === "serving" ? " badge-ok" : ""}">${
                  r.role === "both" ? "anything"
                  : r.role === "serving" ? "serving" : "training"}</span>
                ${raw(r.note ? `<div class="muted tiny">${esc(r.note)}</div>` : "")}</td>
              <td class="hide-sm">${raw(mine.length
                ? `${mine.length} deployed${loaded > mine.length
                    ? `, ${loaded - mine.length} cached` : ""}`
                : loaded ? `<span class="muted">${loaded} cached</span>`
                : `<span class="muted">nothing</span>`)}</td>
              <td class="tiny">${raw(!r.connected
                ? `<span class="muted">offline</span>`
                : r.current_job
                  ? `<a href="#/jobs/${esc(r.current_job)}">training</a>`
                  : `<span class="muted">idle</span>`)}</td>
              <td>${raw(admin
                ? `<button class="btn-sm" data-role="${esc(r.id)}">Change</button>`
                : "")}</td>
            </tr>`; }).join(""))}
        </tbody>
      </table></div>
      <p class="muted tiny" style="padding:10px 16px 14px;margin:0">
        ${admin ? "Reserving a machine for serving stops the scheduler sending"
          + " training runs to it, so a deployed model is not pushed off the"
          + " card halfway through the afternoon."
        : "Only an administrator can set a machine aside."}</p>
    </div>`;
}

// -------------------------------------------------------------------- keys

function keysPanel(keys, admin) {
  if (!admin) {
    return emptyState({ icon: "🔑", title: "Administrators only",
      body: "Your own keys are on your account page.",
      cta: { href: "#/account", label: "Your account" } });
  }
  if (!keys.length) {
    return emptyState({ icon: "🔑", title: "No keys have been made",
      body: "A key is how anything outside this studio reaches a model. "
          + "Everyone makes their own from their account page.",
      cta: { href: "#/account", label: "Make one" } });
  }
  return html`
    <div class="card" style="padding:0">
      <div class="table-wrap"><table>
        <thead><tr><th>Key</th><th>Whose</th><th class="hide-sm">Reaches</th>
          <th>Calls</th><th>Last used</th><th></th></tr></thead>
        <tbody>
          ${raw(keys.map((k) => html`
            <tr>
              <td>${esc(k.name)}
                <div class="muted tiny mono">${esc(k.prefix)}…</div></td>
              <td class="tiny">${esc(k.owner?.display_name || k.owner?.username
                || "an account that is gone")}</td>
              <td class="hide-sm tiny">${raw((k.scope || []).length
                ? k.scope.map((sc) => `<code>${esc(sc)}</code>`).join(" ")
                : `<span class="muted">everything its owner can see</span>`)}</td>
              <td>${fmtNum(k.calls)}</td>
              <td class="tiny muted">${k.last_used ? fmtAgo(k.last_used) : "never"}
                ${raw(k.expired ? `<div><span class="badge badge-warn">expired</span></div>`
                  : k.expires_at ? `<div>expires ${fmtAgo(k.expires_at)}</div>` : "")}</td>
              <td><button class="btn-sm btn-danger" data-revoke-key="${esc(k.id)}"
                          title="Turn it off">✕</button></td>
            </tr>`).join(""))}
        </tbody>
      </table></div>
      <p class="muted tiny" style="padding:10px 16px 14px;margin:0">
        A key is stored as its hash, so nothing here can show one again — only
        the prefix it starts with. Turning one off takes effect on the next
        request.</p>
    </div>`;
}
