import { api, events } from "../api.js";
import { html, raw, esc, $, fmtAgo, statusBadge } from "../util.js";

export async function dashboardView(mount) {
  const paint = async () => {
    const [status, runners, jobs] = await Promise.all([
      api.status(), api.runners(), api.jobs(),
    ]);
    const online = runners.filter((r) => r.status !== "offline");
    const active = jobs.filter((j) => ["running", "assigned", "queued"].includes(j.status));
    const recent = jobs.slice(0, 6);

    mount.innerHTML = html`
      <div class="page-head">
        <h1>Dashboard</h1>
        <p class="sub">Train and fine-tune language and vision models on your own hardware.</p>
      </div>

      ${raw(online.length === 0 ? firstRun(status) : "")}

      <div class="grid grid-3" style="margin-bottom:18px">
        <div class="card stat"><span class="k">Machines ready</span>
          <span class="v">${online.length}</span>
          <span class="tiny muted">${runners.length} known in total</span></div>
        <div class="card stat"><span class="k">Training now</span>
          <span class="v">${status.jobs_running}</span>
          <span class="tiny muted">${status.jobs_queued} waiting</span></div>
        <div class="card stat"><span class="k">Finished runs</span>
          <span class="v">${jobs.filter((j) => j.status === "succeeded").length}</span>
          <span class="tiny muted">of ${jobs.length} total</span></div>
      </div>

      ${raw(online.length ? html`
        <div class="card" style="margin-bottom:18px">
          <div class="row-between" style="flex-wrap:wrap;gap:8px">
            <div>
              <h2 style="margin:0">Ready when you are</h2>
              <p class="muted tiny" style="margin:2px 0 0">
                Four guided steps, with every technical choice made for you.</p>
            </div>
            <div class="row">
              <a class="btn btn-primary btn-lg" href="#/new">Start a training run →</a>
              ${raw(jobs.some((j) => j.status === "succeeded")
                ? `<a class="btn btn-lg" href="#/play">▷ Playground</a>` : "")}
            </div>
          </div>
        </div>` : "")}

      ${raw(active.length ? html`
        <h2>In progress</h2>
        <div class="grid" style="margin-bottom:18px">
          ${raw(active.map(jobRow).join(""))}
        </div>` : "")}

      <div class="row-between"><h2>Recent runs</h2>
        <a href="#/jobs" class="tiny">See all →</a></div>
      ${raw(recent.length
        ? `<div class="grid">${recent.map(jobRow).join("")}</div>`
        : html`<div class="card empty"><div class="big">🌱</div>
            <h3>No runs yet</h3>
            <p class="muted">Your finished models will appear here.</p></div>`)}`;
  };

  await paint();
  const unsub = events.subscribe((m) => {
    if (["jobs_changed", "runners_changed"].includes(m.type)) paint();
  });
  return unsub;
}

function firstRun(status) {
  return html`
    <div class="card" style="margin-bottom:18px;border-color:var(--accent)">
      <h2>👋 Let's get you set up</h2>
      <p class="muted">AI Studio splits into two parts: this control panel, and one
        or more <strong>machines</strong> that own the graphics cards and do the
        actual training. Nothing can train until you connect at least one.</p>
      <p><a class="btn btn-primary" href="#/runners">Connect your first machine →</a></p>
    </div>`;
}

function jobRow(j) {
  const pct = j.total_steps ? Math.min(100, (j.step / j.total_steps) * 100) : 0;
  return html`
    <a class="card" href="#/jobs/${j.id}" style="color:inherit;display:block">
      <div class="row-between" style="flex-wrap:wrap;gap:6px">
        <strong>${j.name}</strong>${statusBadge(j.status)}
      </div>
      <div class="tiny muted mono" style="margin:4px 0 8px">${
        j.kind === "pretrain_llm" ? "from scratch · " + (j.config.dataset || "")
                                  : (j.config.base_model || "")}</div>
      ${raw(["running", "assigned"].includes(j.status)
        ? `<div class="progress"><i style="width:${pct}%"></i></div>
           <div class="tiny muted" style="margin-top:5px">step ${j.step}${
             j.total_steps ? " of " + j.total_steps : ""}</div>`
        : `<div class="tiny muted">${esc(fmtAgo(j.finished_at || j.created_at))}</div>`)}
    </a>`;
}
