import { api, events } from "../api.js";
import { html, raw, esc, $, fmtAgo, statusBadge } from "../util.js";
import { kindOf, subjectOf } from "../kinds.js";
import { pageHead, emptyState } from "../components.js";

export async function dashboardView(mount) {
  const paint = async () => {
    const [status, runners, jobs, datasets] = await Promise.all([
      api.status(), api.runners(), api.jobs(), api.datasets().catch(() => []),
    ]);
    const online = runners.filter((r) => r.status !== "offline");
    const active = jobs.filter((j) => ["running", "assigned", "queued"].includes(j.status));
    const recent = jobs.slice(0, 6);

    mount.innerHTML = html`
      ${raw(pageHead({
        title: "Dashboard",
        tab: "AI Studio",
        sub: "Train and fine-tune language models on your own hardware.",
      }))}

      ${raw(nextStep({ status, online, jobs, datasets }))}

      <div class="grid grid-3" style="margin-bottom:18px">
        <div class="card stat"><span class="k">Machines ready</span>
          <span class="v">${online.length}</span>
          <span class="tiny muted">${runners.length} known in total</span></div>
        <div class="card stat"><span class="k">Running now</span>
          <span class="v">${status.jobs_running}</span>
          <span class="tiny muted">${status.jobs_queued} waiting</span></div>
        <div class="card stat"><span class="k">Finished runs</span>
          <span class="v">${jobs.filter((j) => j.status === "succeeded").length}</span>
          <span class="tiny muted">of ${jobs.length} total</span></div>
      </div>

      ${raw(active.length ? html`
        <h2>In progress</h2>
        <div class="grid" style="margin-bottom:18px">
          ${raw(active.map(jobRow).join(""))}
        </div>` : "")}

      <div class="row-between"><h2>Recent runs</h2>
        <a href="#/jobs" class="tiny">See all →</a></div>
      ${raw(recent.length
        ? `<div class="grid">${recent.map(jobRow).join("")}</div>`
        : emptyState({
            title: "No runs yet",
            body: "Your finished models will appear here.",
            cta: online.length ? { href: "#/new", label: "Start a training run" } : null,
          }))}`;
  };

  await paint();
  const unsub = events.subscribe((m) => {
    if (["jobs_changed", "runners_changed"].includes(m.type)) paint();
  });
  return unsub;
}

/**
 * The one thing worth doing next.
 *
 * There was a card for exactly one empty state — no machines — and nothing for
 * the other two that stop a run dead. Somebody with a machine connected and no
 * data saw three tiles reading 1 / 0 / 0, a button, and a seed emoji; nothing
 * on the page mentioned that the four guided steps need a dataset, or that one
 * can be imported, uploaded, or written for you.
 */
function nextStep({ status, online, jobs, datasets }) {
  const card = (title, body, actions) => html`
    <div class="card" style="margin-bottom:18px;border-color:var(--accent)">
      <div class="row-between" style="flex-wrap:wrap;gap:10px;align-items:center">
        <div style="min-width:0">
          <h2 style="margin:0">${title}</h2>
          <p class="muted" style="margin:4px 0 0">${body}</p>
        </div>
        <div class="row" style="flex-wrap:wrap;gap:8px">${raw(actions)}</div>
      </div>
    </div>`;

  if (!online.length) {
    return card("👋 Let's get you set up",
      "AI Studio is two parts: this control panel, and one or more machines "
      + "that own the graphics cards and do the training. Nothing can train "
      + "until you connect at least one.",
      `<a class="btn btn-primary btn-lg" href="#/runners">Connect a machine →</a>`);
  }
  if (!datasets.length) {
    return card("Now you need something to train on",
      "A dataset of examples. Import one from Hugging Face, upload your own "
      + "files, or have a model write one for you.",
      `<a class="btn btn-primary btn-lg" href="#/data">Get a dataset →</a>
       <a class="btn btn-lg" href="#/generate">✦ Write one</a>`);
  }
  if (!jobs.length) {
    return card("Ready when you are",
      "Four guided steps, with every technical choice made for you from what "
      + "your hardware measured about itself.",
      `<a class="btn btn-primary btn-lg" href="#/new">Start a training run →</a>`);
  }
  const finished = jobs.filter((j) => j.status === "succeeded");
  if (finished.length && !status.hf_token_set) {
    // Not a blocker, and the one setting whose absence surfaces as a run
    // failing an hour in with "you are not authorised" on a gated model.
    return card("You have a model — and no Hugging Face token",
      "Gated models like Llama and Gemma need one to download, and publishing "
      + "needs one to push. Connect yours on your account page.",
      `<a class="btn btn-primary" href="#/account">Connect Hugging Face →</a>
       <a class="btn" href="#/play">▷ Playground</a>`);
  }
  return card("Ready when you are",
    "Four guided steps, with every technical choice made for you.",
    `<a class="btn btn-primary btn-lg" href="#/new">Start a training run →</a>`
    + (finished.length ? `<a class="btn btn-lg" href="#/play">▷ Playground</a>` : ""));
}

function jobRow(j) {
  const pct = j.total_steps ? Math.min(100, (j.step / j.total_steps) * 100) : 0;
  return html`
    <a class="card" href="#/jobs/${j.id}" style="color:inherit;display:block">
      <div class="row-between" style="flex-wrap:wrap;gap:6px">
        <strong>${j.name}</strong>${statusBadge(j.status, j.kind)}
      </div>
      <div class="tiny muted mono" style="margin:4px 0 8px">${subjectOf(j)}</div>
      ${raw(["running", "assigned"].includes(j.status)
        ? `<div class="progress"><i style="width:${pct}%"></i></div>
           <div class="tiny muted" style="margin-top:5px">${
             esc(progressText(j))}</div>`
        : `<div class="tiny muted">${esc(fmtAgo(j.finished_at || j.created_at))}</div>`)}
    </a>`;
}

/** What the numbers on the bar are counting, in this run's own units. */
function progressText(j) {
  const unit = kindOf(j).unit;
  if (!unit) return `step ${j.step}${j.total_steps ? " of " + j.total_steps : ""}`;
  return j.total_steps ? `${j.step} of ${j.total_steps} ${unit}`
                       : `${j.step} ${unit}`;
}
