/**
 * Projects: the work as it is actually done, rather than as five lists.
 *
 * Making a model is one job with stages — prepare the data, train, evaluate,
 * benchmark, publish — and every one of those stages used to be a separate
 * page sorted by date, with the connections between them recorded only in
 * config fields nobody could see from a list. Two weeks later the question
 * "which dataset was the good run trained on, and did we ever score it?" took
 * ten minutes and a guess.
 *
 * A project page is a map of the five stages: what has been done at each, what
 * the best result so far is, and the one button that does the next thing.
 * Nothing is copied into a project — a project is a label on rows that already
 * exist, so the dataset library and the run history are the same rows seen
 * from the side that matters.
 */
import { api, events } from "../api.js";
import { html, raw, esc, on, toast, fmtAgo, fmtNum, fmtBytes, modal, $,
         statusBadge } from "../util.js";
import { ribbon, rb, group } from "../ribbon.js";
import { pageHead, emptyState, breadcrumb, confirmDestructive } from "../components.js";
import { primaryMetric } from "../kinds.js";

// ---------------------------------------------------------------------------
// Every project
// ---------------------------------------------------------------------------

export async function projectsView(mount) {
  let data = null;
  const paint = async () => {
    data = await api.projects();
    rememberProjects(data.projects);
    mount.innerHTML = index(data);
  };
  await paint();

  on(mount, "click", "#newProject", () => newProjectDialog());
  on(mount, "click", "#showArchived", () => {
    mount.dataset.archived = mount.dataset.archived ? "" : "1";
    mount.innerHTML = index(data, !!mount.dataset.archived);
  });
  return events.subscribe((m) => {
    if (["jobs_changed", "job_finished"].includes(m.type)) paint();
  });
}

/** Start a project. Two fields, because a third would be answered with "…". */
export function newProjectDialog(prefill = {}) {
  const dlg = modal({
    title: "New project",
    body: html`
      <p class="muted tiny" style="margin-top:0">A project holds one model
        being made: its data, its runs, the prompt sets it is scored on and
        whatever gets published at the end.</p>
      <form id="npForm">
        <div class="field">
          <label for="npName">What are you making?</label>
          <input id="npName" name="name" required maxlength="120"
                 placeholder="Support replies in our tone"
                 value="${esc(prefill.name || "")}">
        </div>
        <div class="field">
          <label for="npGoal">What would make it finished? <span class="muted">(optional)</span></label>
          <input id="npGoal" name="goal" maxlength="500"
                 placeholder="Answers our top 50 tickets without naming a competitor">
          <div class="hint">Worth a sentence now: it is what you will compare
            the scores against in three weeks.</div>
        </div>
        <div class="row" style="justify-content:flex-end;gap:8px">
          <button type="button" class="btn" data-close>Cancel</button>
          <button class="btn btn-primary" type="submit">Create</button>
        </div>
      </form>`,
  });
  on(dlg, "submit", "#npForm", async (e) => {
    e.preventDefault();
    const f = Object.fromEntries(new FormData(e.target).entries());
    try {
      const p = await api.createProject(f);
      dlg.close();
      location.hash = `#/projects/${p.id}`;
    } catch (ex) { toast(ex.message, "err"); }
  });
  return dlg;
}

function index(data, showArchived = false) {
  const all = data.projects || [];
  const live = all.filter((p) => !p.archived);
  const archived = all.filter((p) => p.archived);
  const u = data.unfiled || {};
  const loose = (u.jobs || 0) + (u.datasets || 0) + (u.evals || 0);

  return html`
    ${raw(pageHead({
      title: "Projects",
      sub: "One project per model you are making — its data, its runs, its "
         + "scores and whatever it ends up published as.",
    }))}
    ${raw(ribbon({
      tabs: [{ key: "home", label: "Projects" }], active: "home",
      body: group("Start", [
        rb("newProject", "◇", "New project", { cls: "primary",
          title: "A place to keep one model's data, runs and scores" }),
        rb(null, "✦", "Training run", { href: "#/new",
          title: "Straight to the wizard, filed afterwards" }),
      ]) + group("Libraries", [
        rb(null, "▤", "Datasets", { href: "#/data" }),
        rb(null, "⬢", "Models", { href: "#/models" }),
      ]) + group("Everything", [
        // The old top-level lists. Not in the sidebar any more -- they are
        // views onto what is inside projects -- but a studio with six months
        // of history in it needs a way to see all of something at once.
        rb(null, "≡", "All runs", { href: "#/jobs" }),
        rb(null, "◎", "All prompt sets", { href: "#/evals" }),
        rb(null, "⌂", "Activity", { href: "#/dashboard",
          title: "What the machines have been doing" }),
      ]),
      right: live.length
        ? `<span class="badge">${live.length} project${live.length > 1 ? "s" : ""}</span>` : "",
    }))}

    ${raw(live.length ? html`
      <div class="grid grid-2">${raw(live.map(projectCard).join(""))}</div>`
      : emptyState({
          icon: "◇",
          title: "No projects yet",
          body: "A project is one model being made, with everything that goes "
              + "into making it kept together: the dataset you prepared, the "
              + "runs you tried, the prompt set you judged them on, and the "
              + "version you finally published.",
        }) + html`
        <p style="text-align:center;margin-top:-6px">
          <button class="btn btn-primary" id="newProject">Start a project</button></p>`)}

    ${raw(loose ? html`
      <div class="card" style="margin-top:14px">
        <div class="row-between" style="gap:8px;flex-wrap:wrap">
          <div>
            <h3 style="margin:0">Not in any project</h3>
            <p class="muted tiny" style="margin:2px 0 0">
              ${[u.jobs && `${u.jobs} run${u.jobs > 1 ? "s" : ""}`,
                 u.datasets && `${u.datasets} dataset${u.datasets > 1 ? "s" : ""}`,
                 u.evals && `${u.evals} prompt set${u.evals > 1 ? "s" : ""}`]
                .filter(Boolean).join(", ")} — everything from before there
              were projects, and anything started outside one. It all still
              works; filing it just makes it findable.</p>
          </div>
          <a class="btn" href="#/projects/unfiled">Open</a>
        </div>
      </div>` : "")}

    ${raw(archived.length ? html`
      <p class="muted tiny" style="margin-top:14px">
        <button class="link-btn" id="showArchived">${showArchived ? "Hide" : "Show"}
          ${archived.length} archived</button></p>
      ${raw(showArchived
        ? `<div class="grid grid-2">${archived.map(projectCard).join("")}</div>` : "")}` : "")}`;
}

function projectCard(p) {
  const bits = [
    [p.datasets, "dataset"], [p.runs, "run"], [p.evals, "prompt set"],
    [p.published, "published"],
  ].filter(([n]) => n).map(([n, w]) =>
    `${n} ${w}${n > 1 && w !== "published" ? "s" : ""}`);
  return html`
    <a class="card card-link" href="#/projects/${p.id}">
      <div class="row-between" style="gap:8px">
        <h3 style="margin:0">${p.name}</h3>
        ${raw(p.active
          ? `<span class="badge badge-accent">${p.active} running</span>`
          : p.published ? `<span class="badge badge-ok">published</span>` : "")}
      </div>
      ${raw(p.goal ? `<p class="muted tiny" style="margin:4px 0 0">${esc(p.goal)}</p>` : "")}
      <p class="muted tiny" style="margin:8px 0 0">
        ${bits.join(" · ") || "Nothing in it yet"}
        ${raw(p.mine ? "" : ` · <span class="badge">shared with you</span>`)}
      </p>
      <p class="muted tiny" style="margin:4px 0 0">Touched ${fmtAgo(p.updated_at)}</p>
    </a>`;
}

// ---------------------------------------------------------------------------
// One project: the map
// ---------------------------------------------------------------------------

export async function projectView(mount, [projectId]) {
  const unfiled = projectId === "unfiled";
  let p = null;
  const paint = async () => {
    p = unfiled
      ? { id: null, name: "Not in any project", unfiled: true,
          goal: "Everything from before there were projects. File a thing into "
              + "a project and it leaves this page.", ...(await api.unfiled()) }
      : await api.project(projectId);
    mount.innerHTML = projectLayout(p);
  };
  await paint();
  if (unfiled) {
    // The picker needs somewhere to file things into, and this is the only
    // page that offers it. Drawn again once the list arrives rather than
    // holding the whole page up for it.
    api.projects().then((d) => {
      rememberProjects(d.projects);
      mount.innerHTML = projectLayout(p);
    }).catch(() => {});
  }

  on(mount, "click", "#renameProject", () => editProjectDialog(p, paint));
  on(mount, "click", "#archiveProject", async () => {
    try {
      await api.updateProject(p.id, { archived: !p.archived });
      toast(p.archived ? "Back in the list." : "Archived — still there, out of the way.", "ok");
      await paint();
    } catch (e) { toast(e.message, "err"); }
  });
  on(mount, "click", "#deleteProject", async () => {
    if (!await confirmDestructive({
      title: `Delete the project "${p.name}"?`,
      consequences: [
        "Nothing inside it is deleted: the runs, datasets, prompt sets and "
        + "published models all stay, and become unfiled.",
        "The goal, the notes and the grouping are lost.",
      ],
      confirmLabel: "Delete the project" })) return;
    try {
      await api.deleteProject(p.id);
      toast("Deleted. What was in it is unfiled.", "ok");
      location.hash = "#/projects";
    } catch (e) { toast(e.message, "err"); }
  });
  on(mount, "click", "[data-unfile]", async (_e, t) => {
    try {
      await api.unfile({ kind: t.dataset.kind, id: t.dataset.unfile });
      toast("Taken out of the project.", "ok");
      await paint();
    } catch (e) { toast(e.message, "err"); }
  });
  on(mount, "click", "[data-file-into]", async (_e, t) => {
    const target = $("#fileTarget", mount)?.value;
    if (!target) return toast("Choose a project first.", "err");
    try {
      await api.fileIntoProject(target, { kind: t.dataset.kind, id: t.dataset.fileInto });
      toast("Filed.", "ok");
      await paint();
    } catch (e) { toast(e.message, "err"); }
  });
  on(mount, "click", "[data-publish-run]", (_e, t) =>
    publishDialog(t.dataset.publishRun, t.dataset.name, p.id, paint));

  return events.subscribe((m) => {
    if (["jobs_changed", "job_finished", "job_progress"].includes(m.type)) paint();
  });
}

function editProjectDialog(p, done) {
  const dlg = modal({
    title: "Project",
    body: html`
      <form id="epForm">
        <div class="field"><label for="epName">Name</label>
          <input id="epName" name="name" required maxlength="120" value="${esc(p.name)}"></div>
        <div class="field"><label for="epGoal">What would make it finished?</label>
          <input id="epGoal" name="goal" maxlength="500" value="${esc(p.goal || "")}"></div>
        <div class="field"><label for="epNotes">Notes</label>
          <textarea id="epNotes" name="notes" rows="6" maxlength="4000"
            placeholder="What you tried, what it did, what to try next.">${esc(p.notes || "")}</textarea>
          <div class="hint">The place for the sentence you would otherwise
            put in a run's name and lose.</div></div>
        <div class="row" style="justify-content:flex-end;gap:8px">
          <button type="button" class="btn" data-close>Cancel</button>
          <button class="btn btn-primary" type="submit">Save</button>
        </div>
      </form>`,
  });
  on(dlg, "submit", "#epForm", async (e) => {
    e.preventDefault();
    try {
      await api.updateProject(p.id, Object.fromEntries(new FormData(e.target).entries()));
      dlg.close();
      await done();
    } catch (ex) { toast(ex.message, "err"); }
  });
  return dlg;
}

/** Publish a finished run into the model library, under a name and version. */
export function publishDialog(jobId, runName, projectId, done) {
  const dlg = modal({
    title: "Publish to the model library",
    body: html`
      <p class="muted tiny" style="margin-top:0">The library is the shortlist:
        the versions somebody decided were finished, with a name that is not a
        run id. The run, its chart and its weights stay exactly where they are
        — this only gives them a name other people can find.</p>
      <form id="pubForm">
        <div class="field"><label for="pubName">Name</label>
          <input id="pubName" name="name" required maxlength="120"
                 value="${esc(runName || "")}"></div>
        <div class="field"><label for="pubVer">Version <span class="muted">(optional)</span></label>
          <input id="pubVer" name="version" maxlength="40" placeholder="v1"></div>
        <div class="field"><label for="pubNotes">What changed <span class="muted">(optional)</span></label>
          <textarea id="pubNotes" name="notes" rows="3" maxlength="2000"
            placeholder="Trained on the cleaned ticket set; beats v0 on the tone prompts."></textarea></div>
        <div class="row" style="justify-content:flex-end;gap:8px">
          <button type="button" class="btn" data-close>Cancel</button>
          <button class="btn btn-primary" type="submit">Publish here</button>
        </div>
      </form>
      <p class="muted tiny" style="margin-bottom:0">Publishing to Hugging Face
        instead is on the run's own page, and lands in this library too.</p>`,
  });
  on(dlg, "submit", "#pubForm", async (e) => {
    e.preventDefault();
    const f = Object.fromEntries(new FormData(e.target).entries());
    try {
      await api.publishLocally({ ...f, job_id: jobId, project_id: projectId });
      dlg.close();
      toast("In the library.", "ok");
      if (done) await done();
    } catch (ex) { toast(ex.message, "err"); }
  });
  return dlg;
}

const STAGE_ICON = { data: "▤", train: "✦", evaluate: "◎", benchmark: "◈", publish: "⬢" };

function projectLayout(p) {
  const c = p.contents || {};
  const map = p.map || [];
  const running = (c.runs || []).filter((r) =>
    ["queued", "assigned", "running"].includes(r.status));

  return html`
    ${raw(pageHead({
      title: p.name,
      sub: p.goal || "",
      back: [{ href: "#/projects", label: "Projects" }],
    }))}
    ${raw(ribbon({
      tabs: [{ key: "map", label: "The project" }], active: "map",
      body: group("Do the next thing", [
        rb(null, "▤", "Add data", { href: "#/data" + q(p, "") ,
          title: "Upload, import or generate a dataset for this project" }),
        rb(null, "✦", "Train", { cls: "primary", href: "#/new" + q(p, "") }),
        rb(null, "◎", "Evaluate", { href: "#/evals" + q(p, "") }),
      ]) + (p.unfiled ? "" : group("This project", [
        rb("renameProject", "✎", "Name, goal, notes"),
        rb("archiveProject", "▣", p.archived ? "Unarchive" : "Archive"),
        rb("deleteProject", "␥", "Delete", { cls: "danger",
          title: "Deletes the grouping only — nothing inside it" }),
      ])),
      right: running.length
        ? `<span class="badge badge-accent">${running.length} running</span>` : "",
    }))}

    ${raw(p.unfiled ? html`
      <div class="callout" style="margin-bottom:14px">
        <strong>Everything that is not in a project</strong>
        Nothing here is lost or broken — this is just the pile from before
        projects existed. Use <em>File into…</em> on anything you still care
        about; the rest can stay here indefinitely.
        ${raw(fileTargetPicker())}
      </div>` : "")}

    <div class="stage-map">${raw(map.map((s) => stageCard(s, p)).join(""))}</div>

    ${raw(p.notes ? html`
      <div class="card" style="margin-top:14px">
        <h3 style="margin:0 0 6px">Notes</h3>
        <p class="muted" style="white-space:pre-wrap;margin:0">${p.notes}</p>
      </div>` : "")}

    ${raw(section("Data", c.datasets, datasetRow, "dataset",
      "Nothing prepared yet.", p))}
    ${raw(section("Runs", c.runs, runRow, "job",
      "No runs in this project yet.", p))}
    ${raw(section("Prompt sets and benchmarks",
      [...(c.prompt_sets || []), ...(c.benchmarks || [])], evalRow, "eval",
      "Nothing to judge it by yet.", p))}
    ${raw(libraryPanel(c.library || []))}`;
}

/** `?project=` for the pages that can start something inside a project. */
const q = (p, sep = "") => (p.unfiled || !p.id ? "" : `?project=${encodeURIComponent(p.id)}${sep}`);

function stageCard(s, p) {
  const cls = { done: "stage-done", active: "stage-active", todo: "stage-todo" }[s.state];
  const href = {
    data: "#/data" + q(p), train: "#/new" + q(p), evaluate: "#/evals" + q(p),
    benchmark: "#/evals" + q(p), publish: "#/models",
  }[s.key];
  return html`
    <a class="stage ${cls}" href="${href}">
      <span class="stage-ico" aria-hidden="true">${STAGE_ICON[s.key]}</span>
      <span class="stage-label">${s.label}</span>
      <span class="stage-count">${s.count ? fmtNum(s.count) : "—"}</span>
      ${raw(s.best ? html`
        <span class="stage-best">best ${s.best.metric.label || "score"}:
          <strong>${typeof s.best.metric.value === "number"
            ? s.best.metric.value.toFixed(4) : s.best.metric.value}</strong></span>` : "")}
      <span class="stage-next">${s.next}</span>
    </a>`;
}

function section(title, rows, render, kind, empty, p) {
  return html`
    <div class="card" style="margin-top:14px;padding:0">
      <div class="row-between" style="padding:12px 16px">
        <h3 style="margin:0">${title}</h3>
        <span class="muted tiny">${(rows || []).length}</span>
      </div>
      ${raw((rows || []).length ? html`
        <div class="table-wrap"><table><tbody>
          ${raw(rows.map((r) => render(r, kind, p)).join(""))}
        </tbody></table></div>`
        : `<p class="muted tiny" style="padding:0 16px 14px;margin:0">${esc(empty)}</p>`)}
    </div>`;
}

/** The one control that differs between a project and the unfiled pile. */
const fileButton = (kind, id, p) => (p.unfiled
  ? `<button class="btn-sm" data-file-into="${esc(id)}" data-kind="${kind}">File into…</button>`
  : `<button class="btn-sm" data-unfile="${esc(id)}" data-kind="${kind}"
       title="Take it out of this project — it is not deleted">Remove</button>`);

let fileTargets = [];
function fileTargetPicker() {
  // Filled by the projects list the page already fetched; a select rather than
  // a dialog per row, because filing twenty loose runs one dialog at a time is
  // how the pile stays a pile.
  return html`
    <div class="row" style="gap:8px;margin-top:8px;align-items:center">
      <label for="fileTarget" class="tiny muted">File into</label>
      <select id="fileTarget">
        ${raw(fileTargets.map((t) =>
          `<option value="${esc(t.id)}">${esc(t.name)}</option>`).join(""))}
      </select>
    </div>`;
}

/** Kept fresh by the index view, which is the only page that lists them. */
export function rememberProjects(list) {
  fileTargets = (list || []).filter((x) => !x.archived).map(
    ({ id, name }) => ({ id, name }));
}

function datasetRow(d, kind, p) {
  return html`
    <tr>
      <td><a href="#/data/${d.id}"><strong>${d.name}</strong></a>
        <div class="muted tiny">${fmtNum(d.rows || 0)} rows · ${fmtBytes(d.bytes || 0)}
          ${raw(d.origin ? ` · ${esc(d.origin)}` : "")}</div></td>
      <td class="tiny muted hide-sm">${fmtAgo(d.created_at)}</td>
      <td style="text-align:right">${raw(fileButton("dataset", d.id, p))}</td>
    </tr>`;
}

function runRow(r, kind, p) {
  const pm = primaryMetric(r);
  const done = r.status === "succeeded";
  return html`
    <tr>
      <td><a href="#/jobs/${r.id}"><strong>${r.name}</strong></a>
        <div class="muted tiny">${r.kind.replace(/_/g, " ")} · ${fmtAgo(r.created_at)}</div></td>
      <td>${statusBadge(r.status)}</td>
      <td class="mono tiny">${pm && pm.value != null
        ? `${pm.label}: ${typeof pm.value === "number" ? pm.value.toFixed(4) : pm.value}` : ""}</td>
      <td style="text-align:right"><div class="row" style="gap:4px;justify-content:flex-end">
        ${raw(done && r.has_model
          ? `<button class="btn-sm" data-publish-run="${esc(r.id)}"
               data-name="${esc(r.name)}">Publish</button>` : "")}
        ${raw(fileButton("job", r.id, p))}
      </div></td>
    </tr>`;
}

/** "· 40 prompts", or nothing at all for a benchmark, whose questions are
 *  fetched by recipe and never stored here. */
function prompts(e) {
  const n = (e.items || []).length || e.item_count || 0;
  if (!n) return e.is_benchmark ? "" : " · no prompts yet";
  return ` · ${fmtNum(n)} prompt${n === 1 ? "" : "s"}`;
}

function evalRow(e, kind, p) {
  return html`
    <tr>
      <td><a href="#/evals/${e.id}"><strong>${e.name}</strong></a>
        <div class="muted tiny">${e.is_benchmark ? "benchmark" : "prompt set"}
          ${raw(prompts(e))}</div></td>
      <td class="tiny">${e.scorings
        ? `${e.scorings} scoring${e.scorings > 1 ? "s" : ""}`
        : `<span class="muted">not run yet</span>`}</td>
      <td style="text-align:right">${raw(fileButton("eval", e.id, p))}</td>
    </tr>`;
}

function libraryPanel(rows) {
  return html`
    <div class="card" style="margin-top:14px;padding:0">
      <div class="row-between" style="padding:12px 16px">
        <h3 style="margin:0">Published</h3>
        <a class="btn-sm" href="#/models">Model library</a>
      </div>
      ${raw(rows.length ? html`
        <div class="table-wrap"><table><tbody>
          ${raw(rows.map((r) => html`
            <tr>
              <td><strong>${r.name}</strong> ${r.version || ""}
                <div class="muted tiny">${r.location === "hf"
                  ? `Hugging Face · ${esc(r.repo_id || "")}` : "in this studio"}
                  · ${fmtAgo(r.published_at)}</div></td>
              <td style="text-align:right">
                <a class="btn-sm" href="#/jobs/${r.job_id}">The run</a></td>
            </tr>`).join(""))}
        </tbody></table></div>`
        : `<p class="muted tiny" style="padding:0 16px 14px;margin:0">Nothing
            published from this project yet. Publishing is what turns a run
            into something other people can find and use.</p>`)}
    </div>`;
}
