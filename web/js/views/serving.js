/**
 * Served models: the names other software is pointed at, and what they cost.
 *
 * Two things live here that used to live nowhere.
 *
 * **Names.** Everything outside this studio addresses a model by a run id or
 * a run name. The id is unreadable; the name changes the moment somebody
 * renames a run, and every client configured with it breaks quietly. So a
 * client is pointed at `assistant-prod` instead, and promoting next week's
 * model is repointing that one name rather than editing four config files on
 * three machines. What each name pointed at before is kept, because "when did
 * the answers change, and to what" is the first question asked when something
 * that was working stops.
 *
 * **Usage.** Every reply has always carried a real token count from the
 * runner, and every one of them was thrown away. The numbers here are those
 * counts, summed: which key, which model, which name.
 */
import { api } from "../api.js";
import { html, raw, esc, $, on, toast, fmtAgo, modal } from "../util.js";
import { ribbon, rb, group, wireRibbon, tabState } from "../ribbon.js";
import { pageHead, emptyState, confirmDestructive } from "../components.js";

const TABS = [
  { key: "names", label: "Names" },
  { key: "usage", label: "Usage" },
];

const STAGES = [
  ["", "no stage"],
  ["production", "production"],
  ["staging", "staging"],
  ["experiment", "experiment"],
];

export async function servingView(mount) {
  let names = [];
  let usage = null;
  let models = [];
  const tabs = tabState("serving", TABS, "names");
  let tab = tabs.get();

  const load = async () => {
    [names, usage] = await Promise.all([
      api.registeredModels().catch(() => []),
      api.usage(30).catch(() => null),
    ]);
  };

  const draw = () => {
    mount.innerHTML = layout(names, usage, tab);
    wire();
  };

  function wire() {
    wireRibbon(mount, (key) => { tab = key; tabs.set(key); draw(); });
    on(mount, "click", "#addName", () => nameDialog(null));
    on(mount, "click", "[data-edit-name]", (_e, t) =>
      nameDialog(names.find((n) => n.alias === t.dataset.editName)));
    on(mount, "click", "[data-drop-name]", async (_e, t) => {
      const alias = t.dataset.dropName;
      if (!await confirmDestructive({
        title: `Stop serving as "${alias}"?`,
        consequences: [
          "Anything configured with that name stops working immediately.",
          "The run itself is untouched, and can still be reached by its id.",
        ],
        confirmLabel: "Remove the name" })) return;
      try {
        await api.unregisterModel(alias);
        await load();
        draw();
        toast("Removed.", "ok");
      } catch (e) { toast(e.message, "err"); }
    });
  }

  function nameDialog(existing) {
    const dlg = modal({
      title: existing ? `Point "${existing.alias}" somewhere else`
                      : "Serve a run under a name",
      width: 560,
      body: html`
        <p class="muted tiny">A name other software is configured with. It
          survives a rename, and moving it to next month's model is one change
          here rather than one in every client.</p>
        <div class="field">
          <label for="nmAlias">Name</label>
          <input id="nmAlias" type="text" class="mono" value="${existing?.alias || ""}"
                 ${raw(existing ? "readonly" : "")} placeholder="assistant-prod">
          <div class="hint">Lowercase letters, digits, dot, dash or underscore.</div>
        </div>
        <div class="field">
          <label for="nmJob">Points at</label>
          <select id="nmJob"><option value="">Loading finished runs…</option></select>
        </div>
        <div class="row" style="gap:10px">
          <div class="field" style="flex:1">
            <label for="nmStage">Stage</label>
            <select id="nmStage">${raw(STAGES.map(([v, l]) => html`
              <option value="${v}"${existing?.stage === v ? " selected" : ""}>${l}</option>`).join(""))}</select>
            <div class="hint">A label, not a workflow. Nothing here enforces
              an order.</div>
          </div>
          <div class="field" style="flex:2">
            <label for="nmNotes">What it is for <span class="muted tiny">(optional)</span></label>
            <input id="nmNotes" type="text" value="${existing?.notes || ""}"
                   placeholder="The model Home Assistant talks to">
          </div>
        </div>
        <div class="row" style="justify-content:flex-end;gap:8px;margin-top:12px">
          <button type="button" class="btn" data-modal-close>Cancel</button>
          <button type="button" class="btn btn-primary" id="nmGo">
            ${existing ? "Point it here" : "Serve it"}</button>
        </div>` });

    // The runs that have a model behind them, asked for once and cached for
    // the life of the page: this list is the same for every name.
    const fill = (rows) => {
      const sel = $("#nmJob", dlg);
      if (!sel) return;
      if (!rows.length) {
        sel.innerHTML = `<option value="">No finished runs to serve</option>`;
        return;
      }
      sel.innerHTML = rows.map((m) => html`
        <option value="${m.id}"${m.id === existing?.job_id ? " selected" : ""}>${
          m.name}${m.base_model ? ` · ${m.base_model}` : ""}</option>`).join("");
    };
    if (models.length) fill(models);
    else {
      api.servableModels().then((r) => {
        models = (r.data || []).filter((m) => !m.is_alias);
        fill(models);
      }).catch(() => fill([]));
    }

    on(dlg, "click", "#nmGo", async (_e, btn) => {
      const alias = ($("#nmAlias", dlg).value || "").trim().toLowerCase();
      const jobId = $("#nmJob", dlg).value;
      if (!alias) return toast("Give it a name.", "err");
      if (!jobId) return toast("Choose the run it points at.", "err");
      btn.disabled = true;
      try {
        const r = await api.registerModel(alias, {
          job_id: jobId,
          stage: $("#nmStage", dlg).value,
          notes: $("#nmNotes", dlg).value,
        });
        dlg.close();
        await load();
        draw();
        toast(r.moved ? `"${alias}" now answers with a different model.`
                      : `Serving as "${alias}".`, "ok");
      } catch (e) { toast(e.message, "err"); btn.disabled = false; }
    });
  }

  await load();
  draw();
}

// ---------------------------------------------------------------------------

function layout(names, usage, tab) {
  const body = tab === "usage" ? usagePanel(usage) : namesPanel(names);
  return html`
    ${raw(pageHead({ title: "Served models",
      sub: "The names other software is pointed at, and what they cost." }))}
    ${raw(ribbon({
      tabs: TABS, active: tab,
      body: group("Names", [
        rb("addName", "⊕", "Serve a run", { cls: "primary" }),
      ]) + group("Elsewhere", [
        rb(null, "🔑", "API keys", { href: "#/account" }),
        rb(null, "≡", "Runs", { href: "#/jobs" }),
      ]),
      right: `<span class="badge">${names.length} name${
        names.length === 1 ? "" : "s"}</span>`,
    }))}
    ${raw(body)}`;
}

function namesPanel(names) {
  if (!names.length) {
    return html`
      <div class="card empty">
        <div class="big" aria-hidden="true">🏷</div>
        <h3>Nothing is served under a name yet</h3>
        <p class="muted">Clients can already reach a run by its id or its name.
          A registered name is the one that keeps working after a rename, and
          that you can move to a better model without touching anything
          outside this studio.</p>
        <p><button class="btn btn-primary" id="addName">Serve a run</button></p>
      </div>`;
  }
  return html`
    <div class="card" style="padding:0">
      <div class="table-wrap"><table>
        <thead><tr>
          <th>Name</th><th>Points at</th><th class="hide-sm">Stage</th>
          <th class="hide-sm">Changed</th><th></th>
        </tr></thead>
        <tbody>
          ${raw(names.map((n) => html`
            <tr>
              <td><code>${n.alias}</code>
                ${raw(n.notes ? `<div class="muted tiny">${esc(n.notes)}</div>` : "")}</td>
              <td>
                ${raw(!n.visible
                  ? `<span class="muted tiny">a run you cannot see</span>`
                  : n.job_gone
                    ? `<span class="badge badge-warn">the run is gone</span>
                       <div class="muted tiny">Anything using this name is
                       getting an error.</div>`
                    : `<a href="#/jobs/${esc(n.job_id)}">${esc(n.job_name)}</a>`)}
                ${raw((n.history || []).length ? `<div class="muted tiny">
                  moved ${n.history.length === 1 ? "once" : n.history.length + " times"}
                  before</div>` : "")}
              </td>
              <td class="hide-sm">${raw(n.stage
                ? `<span class="badge${n.stage === "production" ? " badge-ok" : ""}">${esc(n.stage)}</span>`
                : `<span class="muted">—</span>`)}</td>
              <td class="hide-sm tiny muted">${fmtAgo(n.updated_at)}
                ${raw(n.owner ? `<div>by ${esc(n.owner.display_name || n.owner.username)}</div>` : "")}</td>
              <td><div class="row" style="gap:4px">
                <button class="btn-sm" data-edit-name="${n.alias}"
                        ${raw(n.mine ? "" : "disabled")}>Repoint</button>
                <button class="btn-sm btn-danger" data-drop-name="${n.alias}"
                        ${raw(n.mine ? "" : "disabled")} title="Remove the name">✕</button>
              </div></td>
            </tr>`).join(""))}
        </tbody>
      </table></div>
      <p class="muted tiny" style="padding:10px 16px 14px;margin:0">
        A name is checked before run ids and run names, so it always means what
        it points at here. Clients list them with
        <code>GET /v1/models</code>.</p>
    </div>`;
}

const fmtTokens = (n) => {
  n = n || 0;
  if (n < 1000) return String(n);
  if (n < 1e6) return (n / 1000).toFixed(n < 1e4 ? 1 : 0) + "k";
  return (n / 1e6).toFixed(1) + "M";
};

function usagePanel(u) {
  if (!u) {
    return emptyState({ icon: "📈", title: "Usage could not be read",
      body: "Try again in a moment." });
  }

  const t = u.totals || {};
  if (!t.calls) {
    return emptyState({
      icon: "📈",
      title: "Nothing has been served yet",
      body: "Every reply this studio serves over its OpenAI-compatible API is "
          + "counted here — which key, which model, how many tokens. Counts "
          + `are kept for ${u.kept_days} days.`,
      cta: { href: "#/account", label: "Make an API key" },
    });
  }
  return html`
    <div class="card" style="margin-bottom:14px">
      <div class="row-between">
        <h3 style="margin:0">Last ${u.days} days</h3>
        <span class="tiny muted">${u.scope === "studio"
          ? "the whole studio" : "your own calls"} · kept for ${u.kept_days} days</span>
      </div>
      <div class="row" style="gap:28px;margin-top:10px;flex-wrap:wrap">
        <div class="stat"><span class="v">${t.calls}</span>
          <span class="k">replies</span></div>
        <div class="stat"><span class="v">${fmtTokens(t.prompt_tokens)}</span>
          <span class="k">tokens in</span></div>
        <div class="stat"><span class="v">${fmtTokens(t.completion_tokens)}</span>
          <span class="k">tokens out</span></div>
        <div class="stat"><span class="v">${fmtAgo(t.last_used)}</span>
          <span class="k">most recent</span></div>
      </div>
      <p class="muted tiny" style="margin:10px 0 0">These are the runner's own
        counts of what the model was actually given after its chat template was
        applied — not an estimate from the length of the text.</p>
    </div>
    ${raw(usageTable("By model", "Run", u.by_model, (r) => r.gone
      ? `<span class="muted">${esc(r.name)}</span>
         <span class="badge badge-warn">deleted</span>`
      : `<a href="#/jobs/${esc(r.job_id)}">${esc(r.name)}</a>`))}
    ${raw(u.by_alias.length
      ? usageTable("By name", "Name", u.by_alias, (r) => `<code>${esc(r.key)}</code>`)
      : "")}
    ${raw(usageTable("By key", "Key", u.by_key, (r) => `${esc(r.name)}${
      r.prefix ? ` <span class="muted tiny mono">${esc(r.prefix)}…</span>` : ""}`))}`;
}

function usageTable(title, heading, rows, label) {
  if (!rows.length) return "";
  return html`
    <div class="card" style="margin-bottom:14px;padding:0">
      <h3 style="margin:0;padding:14px 16px 0">${title}</h3>
      <div class="table-wrap"><table>
        <thead><tr><th>${heading}</th><th>Replies</th><th class="hide-sm">Tokens in</th>
          <th class="hide-sm">Tokens out</th><th>Last</th></tr></thead>
        <tbody>
          ${raw(rows.map((r) => html`
            <tr>
              <td>${raw(label(r))}</td>
              <td>${r.calls}</td>
              <td class="hide-sm">${fmtTokens(r.prompt_tokens)}</td>
              <td class="hide-sm">${fmtTokens(r.completion_tokens)}</td>
              <td class="tiny muted">${fmtAgo(r.last_used)}</td>
            </tr>`).join(""))}
        </tbody>
      </table></div>
    </div>`;
}
