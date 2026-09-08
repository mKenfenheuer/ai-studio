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
import { ribbon, rb, group } from "../ribbon.js";
import { primaryMetric } from "../kinds.js";
import { openServeDialog } from "../registry.js";
import { breadcrumb, pageHead, emptyState } from "../components.js";

export async function sweepView(mount, [sweepId]) {
  let sweep = null;
  const paint = async () => {
    sweep = await api.sweep(sweepId);
    mount.innerHTML = layout(sweep);
  };
  await paint();
  on(mount, "click", "#serveWinner", () => {
    const w = (sweep?.runs || []).find((r) => r.id === sweep?.best);
    if (w) openServeDialog({ id: w.id, name: w.name });
  });
  on(mount, "click", "[data-stop-run]", async (_e, t) => {
    try { await api.cancelJob(t.dataset.stopRun, true); toast("Stopping…"); }
    catch (e) { toast(e.message, "err"); }
  });
  return events.subscribe((m) => {
    if (["jobs_changed", "job_finished"].includes(m.type)) paint();
  });
}

/**
 * Every sweep, which had no page at all.
 *
 * `GET /api/sweeps` has existed since sweeps did, and so has `api.sweeps()`.
 * There was no route to reach either from, so the only way to a sweep was
 * through one of its member runs -- and the way to find a member run was to
 * recognise one of eight identically-named rows in the list of every run this
 * studio has ever made. Close the tab and the sweep was gone.
 */
export async function sweepsView(mount) {
  const paint = async () => {
    const sweeps = await api.sweeps();
    mount.innerHTML = html`
      ${raw(pageHead({
        title: "Sweeps",
        sub: "The same run, several settings, so the comparison means something.",
      }))}
      ${raw(ribbon({
        tabs: [{ key: "home", label: "Sweeps" }], active: "home",
        body: group("New", [
          rb(null, "✦", "Start one", { cls: "primary", href: "#/new",
            title: "The review step of any run can vary one setting" }),
        ]) + group("Elsewhere", [
          rb(null, "≡", "All runs", { href: "#/jobs" }),
          rb(null, "⚖", "Compare", { href: "#/compare" }),
        ]),
      }))}
      ${raw(sweeps.length ? html`
        <div class="card" style="padding:0">
          <div class="table-wrap"><table>
            <thead><tr>
              <th>Sweep</th><th>Varying</th><th>Runs</th>
              <th>Best held-out loss</th><th class="hide-sm">When</th><th></th>
            </tr></thead>
            <tbody>${raw(sweeps.map(sweepRow).join(""))}</tbody>
          </table></div>
        </div>` : emptyState({
          icon: "⚖",
          title: "No sweeps yet",
          body: "A sweep is the same run started several times with one setting "
              + "changed, so you can see which value was actually better instead "
              + "of guessing. The review step of any run can start one.",
          cta: { href: "#/new", label: "Start a training run" },
        }))}`;
  };
  await paint();
  return events.subscribe((m) => {
    if (["jobs_changed", "job_finished"].includes(m.type)) paint();
  });
}

function sweepRow(s) {
  const runs = s.runs || [];
  const scored = runs.filter((r) => heldOut(r) != null);
  const best = scored.length
    ? (lowerBetter(scored) ? Math.min : Math.max)(...scored.map(heldOut)) : null;
  const running = runs.filter((r) =>
    ["queued", "assigned", "running"].includes(r.status)).length;
  const varied = [...new Set(runs.flatMap((r) => Object.keys(r.values || {})))];
  return html`
    <tr>
      <td><a href="#/sweeps/${s.id}"><strong>${s.name}</strong></a></td>
      <td class="mono tiny">${varied.join(", ") || "—"}</td>
      <td class="tiny">${runs.length}${raw(running
        ? ` <span class="badge badge-accent">${running} going</span>`
        : ` <span class="badge badge-ok">done</span>`)}</td>
      <td class="mono">${best != null ? best.toFixed(4) : "—"}</td>
      <td class="tiny muted hide-sm">${fmtAgo(s.created_at)}</td>
      <td><a class="btn btn-sm" href="#/sweeps/${s.id}">Open</a></td>
    </tr>`;
}

const heldOut = (r) => primaryMetric(r)?.value ?? null;
// Which way is up, from the first run that says. Mixed polarities in one
// table would be a comparison of unrelated numbers, which the family
// switch already prevents.
const lowerBetter = (rows) => (rows.map(primaryMetric).find(Boolean)?.lower) !== false;

function layout(s) {
  const runs = s.runs || [];
  const scored = runs.filter((r) => heldOut(r) != null);
  const lower = lowerBetter(scored);
  const best = scored.length ? (lower ? Math.min : Math.max)(...scored.map(heldOut)) : null;
  const worst = scored.length ? (lower ? Math.max : Math.min)(...scored.map(heldOut)) : null;
  const span = (worst ?? 0) - (best ?? 0) || 1;
  const running = runs.filter((r) =>
    ["queued", "assigned", "running"].includes(r.status));
  // The variant that actually won, as a run rather than as a number. It was
  // computed to draw a bar next to and then nothing was offered to do with it:
  // the point of a sweep is the winner, and the page stopped at naming it.
  const winner = scored.length
    ? scored.reduce((a, b) => ((lower ? heldOut(a) <= heldOut(b)
                                       : heldOut(a) >= heldOut(b)) ? a : b)) : null;

  return html`
    <div class="page-head">
      ${raw(breadcrumb([{ href: "#/jobs", label: "Runs" },
                        { href: "#/sweeps", label: "Sweeps" }]))}
      <h1 style="margin:6px 0 0">${s.name}</h1>
      <p class="sub">Varying ${(s.varied || []).join(", ") || "settings"} —
        everything else is identical, which is what makes the comparison mean
        something.</p>
    </div>
    ${raw(ribbon({
      tabs: [{ key: "home", label: "Variants" }], active: "home",
      body: group("The winner", [
        // Promotion from the page that decided it. The sweep is where you
        // learn which run deserves the name; sending somebody off to find it
        // again on the Served models page was the gap.
        rb("serveWinner", "🏷", "Serve it under a name", { disabled: !winner,
          title: winner ? `Point a registered name at ${winner.name}` : "No finished variant yet" }),
        rb(null, "▷", "Try it", { cls: winner ? "primary" : "", disabled: !winner,
          href: winner ? `#/play/${esc(winner.id)}` : "",
          title: "Talk to the variant with the lowest held-out loss" }),
        rb(null, "⟳", "Run it again", { disabled: !winner,
          href: winner ? `#/jobs/${esc(winner.id)}/again` : "",
          title: "Start from the winner's settings — for longer, or on more data" }),
        rb(null, "≡", "Open it", { disabled: !winner,
          href: winner ? `#/jobs/${esc(winner.id)}` : "" }),
      ]) + group("All of them", [
        rb(null, "⚖", "Compare", { href: "#/compare" }),
        rb(null, "◎", "Score them", { href: "#/evals" }),
        rb(null, "✦", "Another sweep", { href: "#/new" }),
      ]),
      right: `<span class="badge">${runs.length} runs</span>
              ${running.length
                ? `<span class="badge badge-accent">${running.length} still going</span>`
                : `<span class="badge badge-ok">all finished</span>`}`,
    }))}

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
          <th>${(scored.map(primaryMetric).find(Boolean)?.label) || "Held-out loss"}</th>
          <th class="hide-sm">Training loss</th>
          <th class="hide-sm">Steps</th>
          <th class="hide-sm">Took</th>
          <th></th>
        </tr></thead>
        <tbody>
          ${raw(runs.map((r) => {
            const held = heldOut(r);
            const sm = r.summary || {};
            const width = held != null ? 12 + 88 * (1 - Math.abs(held - best) / span) : 0;
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
