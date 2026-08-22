/**
 * One sweep: the same run, several settings, side by side.
 *
 * Launching several runs was never the hard part. The hard part was comparing
 * them a week later and remembering which was which, so every variant carries
 * the exact values that make it different and this page puts them in a column
 * rather than leaving the reader to diff the names.
 *
 * Ranked by held-out loss, and only by held-out loss. It is the one number
 * that means the same thing across two runs of the same data, which is
 * precisely the situation a sweep creates.
 */
import { api, events } from "../api.js";
import { html, raw, esc, on, toast, fmtAgo, fmtDuration, fmtNum,
         statusBadge } from "../util.js";

export async function sweepView(mount, [sweepId]) {
  const paint = async () => {
    const s = await api.sweep(sweepId);
    mount.innerHTML = layout(s);
  };
  await paint();
  on(mount, "click", "[data-stop-run]", async (_e, t) => {
    try { await api.cancelJob(t.dataset.stopRun, true); toast("Stopping…"); }
    catch (e) { toast(e.message, "err"); }
  });
  return events.subscribe((m) => {
    if (["jobs_changed", "job_finished"].includes(m.type)) paint();
  });
}

const heldOut = (r) => r.summary?.best_val_loss ?? null;

function layout(s) {
  const runs = s.runs || [];
  const scored = runs.filter((r) => heldOut(r) != null);
  const best = scored.length ? Math.min(...scored.map(heldOut)) : null;
  const worst = scored.length ? Math.max(...scored.map(heldOut)) : null;
  const span = (worst ?? 0) - (best ?? 0) || 1;
  const running = runs.filter((r) =>
    ["queued", "assigned", "running"].includes(r.status));

  return html`
    <div class="page-head">
      <a href="#/jobs" class="tiny">← All runs</a>
      <div class="row-between" style="flex-wrap:wrap;gap:8px;margin-top:6px">
        <h1 style="margin:0">${s.name}</h1>
        <div class="row" style="gap:6px">
          <span class="badge">${runs.length} runs</span>
          ${raw(running.length
            ? `<span class="badge badge-accent">${running.length} still going</span>`
            : `<span class="badge badge-ok">all finished</span>`)}
        </div>
      </div>
      <p class="sub">Varying ${(s.varied || []).join(", ") || "settings"} —
        everything else is identical, which is what makes the comparison mean
        something.</p>
    </div>

    ${raw(running.length && scored.length < 2 ? html`
      <div class="callout" style="margin-bottom:14px">
        <strong>Still running</strong>
        ${runs.length - running.length} of ${runs.length} have finished. The
        table fills in as they do. They take turns on the card rather than
        running at once, and the queue is dealt between people, so a sweep does
        not lock anyone else out — it just takes longer.</div>` : "")}

    <div class="card" style="padding:0">
      <div class="table-wrap"><table>
        <thead><tr>
          <th>Setting</th><th>Status</th>
          <th>Held-out loss</th>
          <th class="hide-sm">Training loss</th>
          <th class="hide-sm">Steps</th>
          <th class="hide-sm">Took</th>
          <th></th>
        </tr></thead>
        <tbody>
          ${raw(runs.map((r) => {
            const held = heldOut(r);
            const sm = r.summary || {};
            const width = held != null ? 12 + 88 * (1 - (held - best) / span) : 0;
            const live = ["queued", "assigned", "running"].includes(r.status);
            const pct = r.total_steps ? Math.min(100, (r.step / r.total_steps) * 100) : 0;
            return html`
              <tr class="${r.id === s.best ? "row-best" : ""}">
                <td class="mono">${Object.entries(r.values || {})
                    .map(([k, v]) => `${k} = ${v}`).join(", ") || "—"}
                  ${raw(r.id === s.best
                    ? ` <span class="badge badge-ok">best</span>` : "")}</td>
                <td>${statusBadge(r.status)}
                  ${raw(live && r.total_steps
                    ? `<div class="progress" style="margin-top:4px"><i
                         style="width:${pct}%"></i></div>
                       <span class="tiny muted">${r.step}/${r.total_steps}</span>` : "")}</td>
                <td style="min-width:150px">
                  ${raw(held != null ? html`
                    <strong>${held.toFixed(4)}</strong>
                    <div class="meter"><i style="width:${width.toFixed(1)}%"></i></div>`
                    : `<span class="muted tiny">${
                        r.status === "running" ? "still running"
                          : live ? "waiting its turn" : "not measured"}</span>`)}
                </td>
                <td class="hide-sm">${sm.final_loss != null ? sm.final_loss.toFixed(4) : "—"}</td>
                <td class="hide-sm">${sm.steps ? fmtNum(sm.steps) : (r.step || "—")}
                  ${raw(sm.early_stopped
                    ? `<div class="muted tiny">stopped early</div>` : "")}</td>
                <td class="hide-sm tiny muted">${sm.duration_s
                  ? fmtDuration(sm.duration_s)
                  : (r.finished_at ? fmtAgo(r.finished_at) : "—")}</td>
                <td><div class="row" style="gap:4px">
                  <a class="btn btn-sm" href="#/jobs/${r.id}">Open</a>
                  ${raw(live
                    ? `<button class="btn-sm btn-danger" data-stop-run="${esc(r.id)}"
                        title="Stop this variant">✕</button>` : "")}
                </div></td>
              </tr>`;
          }).join(""))}
        </tbody>
      </table></div>
      <p class="muted tiny" style="padding:12px 16px;margin:0">
        ${raw(scored.length >= 2
          ? html`Ranked by held-out loss, which is measured on examples none of
              these models trained on — the only number comparable between two
              runs. A gap smaller than the difference between neighbouring
              settings is not a result;
              <a href="#/evals">a prompt set</a> can say whether the winner is
              actually better at your task, which loss alone cannot.`
          : html`Nothing to rank yet. Held-out loss appears as each run
              finishes.`)}</p>
    </div>`;
}
