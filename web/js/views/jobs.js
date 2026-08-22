import { api, events } from "../api.js";
import { html, raw, esc, on, fmtAgo, fmtDuration, statusBadge, toast } from "../util.js";

export async function jobsView(mount) {
  const paint = async () => {
    const jobs = await api.jobs();
    mount.innerHTML = html`
      <div class="page-head">
        <div class="row-between" style="flex-wrap:wrap;gap:8px">
          <div><h1>Runs</h1>
            <p class="sub">Every training run, newest first.</p></div>
          <a class="btn btn-primary" href="#/new">New run</a>
        </div>
      </div>
      ${raw(jobs.length ? html`
        <div class="card" style="padding:0">
          <div class="table-wrap"><table>
            <thead><tr>
              <th>Name</th><th>Status</th><th>Progress</th>
              <th class="hide-sm">Model</th><th class="hide-sm">When</th><th></th>
            </tr></thead>
            <tbody>${raw(jobs.map(row).join(""))}</tbody>
          </table></div>
        </div>` : html`
        <div class="card empty"><div class="big">📋</div>
          <h3>Nothing here yet</h3>
          <p class="muted">Start a run and it will show up here.</p>
          <p><a class="btn btn-primary" href="#/new">Start a training run</a></p>
        </div>`)}`;
  };
  await paint();
  // Delegated to the mount, which survives every repaint. Bound to the buttons
  // directly it would be lost the next time a job changed status.
  on(mount, "click", "[data-del]", async (_e, t) => {
    if (!confirm(`Delete "${t.dataset.name}"?\n\nIts model file, logs and `
                 + "measurements are deleted and cannot be recovered.")) return;
    try { await api.deleteJob(t.dataset.del); toast("Run deleted.", "ok"); }
    catch (e) { toast(e.message, "err"); }
  });
  return events.subscribe((m) => { if (m.type === "jobs_changed") paint(); });
}

function row(j) {
  const pct = j.total_steps ? Math.min(100, (j.step / j.total_steps) * 100) : 0;
  const dur = j.finished_at && j.started_at ? j.finished_at - j.started_at : null;
  return html`
    <tr>
      <td><a href="#/jobs/${j.id}"><strong>${j.name}</strong></a></td>
      <td>${statusBadge(j.status)}${raw(
        j.status === "cancelled" && j.has_model
          ? ` <span class="badge badge-ok">model kept</span>` : "")}${raw(
        j.checkpoint_step && ["failed", "cancelled"].includes(j.status)
          ? ` <span class="badge badge-accent">can resume</span>` : "")}</td>
      <td style="min-width:120px">
        ${raw(["running", "assigned"].includes(j.status)
          ? `<div class="progress"><i style="width:${pct}%"></i></div>
             <span class="tiny muted">${j.step}/${j.total_steps || "?"}</span>`
          : j.status === "queued" && j.queue_position
          // Where it actually sits in the order work is handed out, which is
          // not the order runs were created: the queue is dealt round-robin
          // between people so one person's overnight batch cannot block
          // everyone else's twenty-minute job.
          ? `<span class="tiny muted">${j.queue_position} of ${
               j.queue_length} waiting</span>`
          : `<span class="tiny muted">${esc(dur ? fmtDuration(dur) : "—")}</span>`)}
      </td>
      <td class="mono tiny hide-sm">${j.kind === "pretrain_llm"
        ? "from scratch" : (j.config.base_model || "—")}</td>
      <td class="tiny muted hide-sm">${fmtAgo(j.created_at)}</td>
      <td><div class="row" style="gap:5px">
        ${raw(j.has_model && ["succeeded", "cancelled"].includes(j.status)
          ? `<a class="btn btn-sm btn-primary" href="#/play/${esc(j.id)}">Try</a>` : "")}
        <a class="btn btn-sm" href="#/jobs/${j.id}">Open</a>
        ${raw(["succeeded", "failed", "cancelled"].includes(j.status)
          ? `<button class="btn-sm btn-danger" data-del="${esc(j.id)}"
                     data-name="${esc(j.name)}" title="Delete this run">✕</button>` : "")}
      </div></td>
    </tr>`;
}
