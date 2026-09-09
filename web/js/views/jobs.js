/**
 * Every run, and the things you do to a run.
 *
 * This is the list that grows without bound — one row per training run, per
 * dataset written, per model published, per scoring — and it was the one list
 * in the app with no search, no sort and no filter, in an app where two runs
 * of the same model on the same data are given the same name. Finding last
 * Tuesday's run meant reading.
 */
import { api, events } from "../api.js";
import { html, raw, esc, $, on, fmtAgo, fmtDuration, statusBadge, toast, hashParam } from "../util.js";
import { KINDS, kindOf, subjectOf, primaryMetric } from "../kinds.js";
import { ribbon, rb, group, rbSelect, rbSeg, rbSearch, wireRibbon, tabState } from "../ribbon.js";
import { pageHead, emptyState, confirmDestructive } from "../components.js";

const TABS = [
  { key: "home", label: "Home" },
  { key: "filter", label: "Find" },
  { key: "view", label: "View" },
];

const STATUS_GROUPS = [
  ["all", "All"],
  ["active", "Running"],
  ["succeeded", "Finished"],
  ["failed", "Failed"],
];

const SORTS = [
  ["created", "Newest first"],
  ["name", "Name"],
  ["loss", "Best held-out loss"],
  ["duration", "Longest"],
];

export async function jobsView(mount) {
  let jobs = [];
  let picked = new Set();
  let q = "";
  let status = "all";
  let kind = "all";
  let scope = "all";
  let sort = "created";
  let bySweep = false;
  // Which project's runs. The run history is every run the studio has ever
  // made; a project's own runs are what anybody is looking for in it.
  let project = hashParam("project") || "";
  let projects = [];

  const tabs = tabState("jobs", TABS, "home");
  let tab = tabs.get();

  const chosen = () => jobs.filter((j) => picked.has(j.id));

  const paint = async () => {
    jobs = await api.jobs();
    // A run that finished while the page was open is still selected, but one
    // that was deleted elsewhere is not there to be acted on.
    const alive = new Set(jobs.map((j) => j.id));
    picked = new Set([...picked].filter((id) => alive.has(id)));
    draw();
  };

  const draw = () => {
    mount.innerHTML = layout({ jobs, picked, q, status, kind, scope, sort,
                              bySweep, tab, project, projects });
    wire();
  };

  api.projects().then((d) => {
    projects = (d.projects || []).filter((p) => !p.archived);
    if (projects.length) draw();
  }).catch(() => {});

  function wire() {
    wireRibbon(mount, (key) => { tab = key; tabs.set(key); draw(); });

    on(mount, "input", "#jobQ", (_e, t) => {
      q = t.value.toLowerCase();
      $("#jobList", mount).innerHTML =
        listing({ jobs, picked, q, status, kind, scope, sort, bySweep, project });
    });
    on(mount, "click", "[data-status]", (_e, t) => { status = t.dataset.status; draw(); });
    on(mount, "click", "[data-scope]", (_e, t) => { scope = t.dataset.scope; draw(); });
    on(mount, "change", "#jobKind", (_e, t) => { kind = t.value; draw(); });
    on(mount, "change", "#jobProject", (_e, t) => { project = t.value; draw(); });
    on(mount, "change", "#jobSort", (_e, t) => { sort = t.value; draw(); });
    on(mount, "click", "[data-sweepgroup]", (_e, t) => {
      bySweep = t.dataset.sweepgroup === "1"; draw();
    });

    on(mount, "change", "[data-pick]", (_e, t) => {
      if (t.checked) picked.add(t.dataset.pick); else picked.delete(t.dataset.pick);
      draw();
    });
    on(mount, "click", "#pickNone", () => { picked = new Set(); draw(); });

    on(mount, "click", "#stopPicked", async () => {
      const running = chosen().filter((j) =>
        ["running", "assigned", "queued"].includes(j.status));
      if (!await confirmDestructive({
        title: running.length === 1 ? `Stop "${running[0].name}"?`
                                    : `Stop ${running.length} runs?`,
        body: "Each one keeps whatever it has finished. A training run that has "
            + "saved a checkpoint can be resumed from it afterwards.",
        confirmLabel: "Stop", tone: "danger" })) return;
      for (const j of running) {
        try { await api.cancelJob(j.id, true); }
        catch (e) { toast(`${j.name}: ${e.message}`, "err"); }
      }
      picked = new Set();
      await paint();
    });

    on(mount, "click", "#resumePicked", async () => {
      for (const j of chosen().filter((x) => x.checkpoint_step)) {
        try { await api.resumeJob(j.id); toast(`${j.name}: carrying on.`, "ok"); }
        catch (e) { toast(`${j.name}: ${e.message}`, "err"); }
      }
      picked = new Set();
      await paint();
    });

    on(mount, "click", "#deletePicked", async () => {
      const list = chosen().filter((j) =>
        ["succeeded", "failed", "cancelled"].includes(j.status));
      if (!list.length) return toast("A run has to have finished before it can be deleted.", "err");
      if (!await confirmDestructive({
        title: list.length === 1 ? `Delete "${list[0].name}"?`
                                 : `Delete ${list.length} runs?`,
        consequences: [
          "The model file, the adapter, the logs and every measurement go with it.",
          "Cached copies on the machines that ran it are removed too.",
          "Scores already recorded against a prompt set are kept.",
        ],
        confirmLabel: "Delete" })) return;
      for (const j of list) {
        try { await api.deleteJob(j.id); }
        catch (e) { toast(`${j.name}: ${e.message}`, "err"); }
      }
      picked = new Set();
      toast("Deleted.", "ok");
      await paint();
    });
  }

  await paint();
  return events.subscribe((m) => { if (m.type === "jobs_changed") paint(); });
}

// ---------------------------------------------------------------------------

function layout(s) {
  return html`
    ${raw(pageHead({ title: "Runs",
                     sub: "Every run this studio has made, newest first." }))}
    ${raw(ribbonFor(s))}
    <div class="card" style="padding:0">
      <div id="jobList">${raw(listing(s))}</div>
    </div>`;
}

function ribbonFor(s) {
  const { tab, jobs, picked, q, status, kind, scope, sort, bySweep,
          project, projects } = s;
  const list = jobs.filter((j) => picked.has(j.id));
  const n = list.length;
  const one = n === 1 ? list[0] : null;
  const canStop = list.some((j) => ["running", "assigned", "queued"].includes(j.status));
  const canResume = list.some((j) => j.checkpoint_step
    && ["failed", "cancelled"].includes(j.status));
  const done = list.filter((j) => ["succeeded", "failed", "cancelled"].includes(j.status));
  const playable = one && one.has_model && kindOf(one).leavesModel
    && ["succeeded", "cancelled"].includes(one.status);

  let body = "";
  if (tab === "home") {
    body = group("Start", [
      rb(null, "✦", "Training run", { cls: "primary", href: "#/new" }),
      rb(null, "▤", "Write a dataset", { href: "#/generate" }),
      rb(null, "◎", "Score models", { href: "#/evals" }),
      rb(null, "🏷", "Served models", { href: "#/serving",
        title: "The names other software is pointed at, and what they cost" }),
    ]) + group(n ? `${n} selected` : "Selected", [
      rb(null, "▤", "Open", { disabled: !one, href: one ? `#/jobs/${one.id}` : "" }),
      rb(null, "▷", "Try it", { cls: "primary", disabled: !playable,
        href: playable ? `#/play/${one.id}` : "" }),
      rb(null, "⟳", "Run again", { disabled: !one || !done.length,
        href: one && done.length
          ? (one.kind === "generate_dataset" ? `#/generate/from/${one.id}`
                                             : `#/jobs/${one.id}/again`) : "" }),
      rb("resumePicked", "▶", "Resume", { disabled: !canResume,
        title: "Carry on from the last checkpoint" }),
      rb("stopPicked", "■", "Stop", { disabled: !canStop, cls: "danger" }),
      rb(null, "⚖", "Compare", { href: "#/compare", disabled: n < 2 }),
      rb("deletePicked", "🗑", "Delete", { cls: "danger", disabled: !done.length }),
      rb("pickNone", "✕", "Clear", { disabled: !n }),
    ]);
  } else if (tab === "filter") {
    body = group("Find", [
      rbSearch("jobQ", { placeholder: "Name, model or dataset…", value: q }),
    ]) + group("Status", [
      rbSeg(STATUS_GROUPS.map(([v, label]) =>
        ({ label, on: status === v, data: `data-status="${v}"` }))),
    ]) + group("Where", [
      rbSelect("jobProject", { title: "Project", value: project,
        options: [["", "Every project"], ["none", "Not in a project"],
                  ...projects.map((p) => [p.id, p.name])] }),
    ]) + group("Kind", [
      rbSelect("jobKind", { title: "Kind of run", value: kind,
        options: [["all", "Every kind"]].concat(
          Object.entries(KINDS).filter(([, k]) => !k.legacy)
            .map(([id, k]) => [id, k.label])) }),
    ]) + group("Whose", [
      rbSeg([{ label: "All", on: scope === "all", data: `data-scope="all"` },
             { label: "Mine", on: scope === "mine", data: `data-scope="mine"` },
             { label: "Shared with me", on: scope === "shared", data: `data-scope="shared"` }]),
    ]);
  } else {
    body = group("Order", [
      rbSelect("jobSort", { title: "Order", value: sort, options: SORTS }),
    ]) + group("Arrange", [
      rbSeg([{ label: "Flat", on: !bySweep, data: `data-sweepgroup="0"` },
             { label: "Group sweeps", on: bySweep, data: `data-sweepgroup="1"` }]),
      rb(null, "⚖", "All sweeps", { href: "#/sweeps",
        title: "Every sweep this studio has run" }),
    ]);
  }
  return ribbon({ tabs: TABS, active: tab, body });
}

const heldOut = (j) => primaryMetric(j)?.value ?? null;
// Sorting "by loss" is really sorting by the run's own headline number, in
// its own direction: a classifier's 91% belongs above its 87%, not below.
const rankKey = (j) => {
  const m = primaryMetric(j);
  return m ? (m.lower ? m.value : -m.value) : 1e9;
};

function filtered({ jobs, q, status, kind, scope, sort, bySweep, project }) {
  let out = jobs.filter((j) => {
    if (kind !== "all" && j.kind !== kind) return false;
    // "" every project, "none" the ones in none, otherwise that project.
    if (project && (project === "none" ? j.project_id
                                       : j.project_id !== project)) return false;
    if (scope === "mine" && !j.mine) return false;
    if (scope === "shared" && j.mine) return false;
    if (status === "active" && !["running", "assigned", "queued"].includes(j.status)) return false;
    if (status === "succeeded" && j.status !== "succeeded") return false;
    if (status === "failed" && !["failed", "cancelled"].includes(j.status)) return false;
    if (!q) return true;
    // Everything on the row, so searching for the dataset finds the runs that
    // trained on it and searching for a repository finds what was published.
    const hay = [j.name, j.notes, subjectOf(j), j.config?.dataset_label, j.config?.dataset,
                 j.config?.sweep_name, ...(j.tags || []),
                 ...(j.config?.published || []).map((p) => p.repo_id)]
      .filter(Boolean).join(" ").toLowerCase();
    return hay.includes(q);
  });

  out.sort(sort === "name" ? (a, b) => a.name.localeCompare(b.name)
    : sort === "loss" ? (a, b) => rankKey(a) - rankKey(b)
    : sort === "duration" ? (a, b) => (b.summary?.duration_s || 0) - (a.summary?.duration_s || 0)
    : (a, b) => (b.created_at || 0) - (a.created_at || 0));

  if (bySweep) {
    // Members of a sweep next to each other, each sweep where its first member
    // fell in the sort. Eight runs called "Mistral · trying lora_r" scattered
    // through a list is the case this exists for.
    const seen = [];
    const groups = new Map();
    for (const j of out) {
      const key = j.config?.sweep_id || j.id;
      if (!groups.has(key)) { groups.set(key, []); seen.push(key); }
      groups.get(key).push(j);
    }
    out = seen.flatMap((k) => groups.get(k));
  }
  return out;
}

function listing(s) {
  const shown = filtered(s);
  const { picked, jobs } = s;

  if (!shown.length) {
    return emptyState({
      icon: "📋",
      title: jobs.length ? "Nothing matches that" : "Nothing here yet",
      body: jobs.length ? "Try a different word, or clear the filters."
                        : "Start a run and it will show up here.",
      cta: jobs.length ? null : { href: "#/new", label: "Start a training run" },
      card: false,
    });
  }

  return html`
    <div class="table-wrap"><table>
      <thead><tr>
        <th style="width:28px"></th>
        <th>Name</th><th>Status</th><th>Progress</th>
        <th class="hide-sm">Working on</th><th class="hide-sm">Project</th>
        <th class="hide-sm">When</th><th></th>
      </tr></thead>
      <tbody>${raw(shown.map((j) => row(j, picked)).join(""))}</tbody>
    </table></div>`;
}

/** The repositories this run went to, if any.
 *
 *  Recorded when an upload finishes, so this is the list of a run's models
 *  that actually exist on the Hub -- and the name a fine-tune of it will give
 *  as its base model. Worth a line here because the alternative is opening
 *  every run to find out which ones are already out.
 */
function hubLinks(j) {
  const rows = j.config?.published || [];
  if (!rows.length) return "";
  return html`<div class="tiny muted">${raw(rows.map((p) => html`
    <a href="${p.url || `https://huggingface.co/${p.repo_id}`}"
       target="_blank" rel="noopener">↗ ${p.repo_id}</a>`).join(" "))}</div>`;
}

function row(j, picked) {
  const pct = j.total_steps ? Math.min(100, (j.step / j.total_steps) * 100) : 0;
  const dur = j.finished_at && j.started_at ? j.finished_at - j.started_at : null;
  const k = kindOf(j);
  return html`
    <tr class="${picked.has(j.id) ? "row-picked" : ""}">
      <td><input type="checkbox" data-pick="${j.id}" aria-label="Select ${j.name}"
                 ${picked.has(j.id) ? "checked" : ""}></td>
      <td><a href="#/jobs/${j.id}"><strong>${j.name}</strong></a>${raw((j.tags || [])
          .map((t) => ` <span class="badge badge-soft">${esc(t)}</span>`).join(""))}
        ${raw(j.config.sweep_id
          ? `<div class="tiny"><a href="#/sweeps/${esc(j.config.sweep_id)}"
               >part of a sweep</a></div>` : "")}
        ${raw(hubLinks(j))}</td>
      <td>${raw(k.icon)} ${statusBadge(j.status, j.kind)}${raw(
        j.status === "cancelled" && j.has_model
          ? ` <span class="badge badge-ok">model kept</span>` : "")}${raw(
        j.checkpoint_step && ["failed", "cancelled"].includes(j.status)
          ? ` <span class="badge badge-accent">can resume</span>` : "")}</td>
      <td style="min-width:120px">
        ${raw(["running", "assigned"].includes(j.status)
          ? `<div class="progress"><i style="width:${pct}%"></i></div>
             <span class="tiny muted">${j.step}/${j.total_steps || "?"} ${
               esc(k.unit || "steps")}</span>`
          : j.status === "queued" && j.queue_position
          // Where it actually sits in the order work is handed out, which is
          // not the order runs were created: the queue is dealt round-robin
          // between people so one person's overnight batch cannot block
          // everyone else's twenty-minute job.
          ? `<span class="tiny muted">${j.queue_position} of ${
               j.queue_length} waiting</span>`
          : `<span class="tiny muted">${esc(dur ? fmtDuration(dur) : "—")}</span>`)}
      </td>
      <td class="mono tiny hide-sm">${subjectOf(j) || "—"}</td>
      <td class="tiny hide-sm">${raw(j.project_id
        ? `<a href="#/projects/${esc(j.project_id)}">${esc(j.project_name || "a project")}</a>`
        : `<span class="muted">—</span>`)}</td>
      <td class="tiny muted hide-sm">${fmtAgo(j.created_at)}</td>
      <td><div class="row" style="gap:5px">
        ${raw(j.has_model && k.leavesModel
          && ["succeeded", "cancelled"].includes(j.status)
          ? `<a class="btn btn-sm btn-primary" href="#/play/${esc(j.id)}">Try</a>` : "")}
        ${raw(j.kind === "generate_dataset" && j.summary?.dataset_id
          ? `<a class="btn btn-sm btn-primary" href="#/data/${
               esc(j.summary.dataset_id)}">Rows</a>` : "")}
        <a class="btn btn-sm" href="#/jobs/${j.id}">Open</a>
      </div></td>
    </tr>`;
}
