/**
 * Runs, side by side.
 *
 * The one number here that is honestly comparable between two runs is the
 * **held-out loss**: measured on text or examples the model never trained on,
 * so it cannot be lowered by memorising. Training loss is shown next to it and
 * deliberately not ranked — two runs on different data, with different
 * vocabularies, produce training losses that are not the same quantity, and
 * sorting by it would invent a league table out of unrelated numbers.
 *
 * Two runs can be overlaid as curves. Exactly two, because this app has two
 * validated series colours and no more; comparing eight runs is a magnitude
 * question, and magnitude is what the table's bars are for.
 */
import { api } from "../api.js";
import { html, raw, $, $$, on, toast, fmtAgo, fmtNum, fmtDuration } from "../util.js";
import { LineChart } from "../chart.js";

export async function compareView(mount) {
  const jobs = (await api.jobs())
    .filter((j) => ["succeeded", "cancelled"].includes(j.status) && j.has_model);
  const picked = new Set();
  let chart = null;

  const draw = () => {
    mount.innerHTML = layout(jobs, picked);
    if (chart) { chart.destroy(); chart = null; }
    wire();
    paintCurves();
  };

  function wire() {
    on(mount, "change", "[data-cmp]", (_e, t) => {
      if (t.checked) picked.add(t.dataset.cmp); else picked.delete(t.dataset.cmp);
      draw();
    });
    on(mount, "click", "#cmpClear", () => { picked.clear(); draw(); });
  }

  async function paintCurves() {
    const box = $("#cmpChart", mount);
    if (!box || picked.size !== 2) return;
    const [a, b] = [...picked];
    const nameOf = (id) => (jobs.find((j) => j.id === id) || {}).name || id;
    try {
      const [ma, mb] = await Promise.all([api.jobMetrics(a), api.jobMetrics(b)]);
      chart = new LineChart(box, {
        title: "Held-out loss, both runs",
        height: 260,
        format: (v) => v.toFixed(4),
        series: [{ key: "a", label: nameOf(a) },
                 { key: "b", label: nameOf(b), dashed: true }],
      });
      const held = (rows) => rows.filter((m) => m.val_loss != null)
        .map((m) => ({ x: m.step, y: m.val_loss }));
      const train = (rows) => rows.filter((m) => m.loss != null)
        .map((m) => ({ x: m.step, y: m.loss }));
      // Falls back to training loss only when neither run measured a held-out
      // one, and says so rather than quietly drawing a different quantity
      // under the same title.
      const anyHeld = held(ma).length || held(mb).length;
      chart.setSeries("a", anyHeld ? held(ma) : train(ma));
      chart.setSeries("b", anyHeld ? held(mb) : train(mb));
      if (!anyHeld) {
        $("#cmpChartNote", mount).textContent =
          "Neither run measured a held-out loss, so these are training losses "
          + "— comparable only if both runs trained on the same data.";
      }
    } catch (e) { toast(e.message, "err"); }
  }

  draw();
  return () => { chart?.destroy(); };
}

const heldOut = (j) => j.summary?.best_val_loss ?? null;

function layout(jobs, picked) {
  const withHeld = jobs.filter((j) => heldOut(j) != null);
  const best = withHeld.length ? Math.min(...withHeld.map(heldOut)) : null;
  const worst = withHeld.length ? Math.max(...withHeld.map(heldOut)) : null;
  const span = (worst ?? 0) - (best ?? 0) || 1;

  return html`
    <div class="page-head">
      <div class="row-between" style="flex-wrap:wrap;gap:8px">
        <div>
          <h1>Compare runs</h1>
          <p class="sub">Held-out loss is the number that carries between runs.</p>
        </div>
        <a class="btn" href="#/evals">Prompt sets →</a>
      </div>
    </div>

    ${raw(picked.size === 2 ? html`
      <div class="card" style="margin-bottom:14px">
        <div id="cmpChart"></div>
        <p class="muted tiny" id="cmpChartNote" style="margin:8px 0 0"></p>
      </div>` : picked.size > 2 ? html`
      <div class="callout callout-warn" style="margin-bottom:14px">
        <strong>${picked.size} runs selected.</strong> The table below compares
        as many as you like. Curves are drawn for exactly two — beyond that the
        lines need a colour each, and this app has two colours that are
        verified to stay distinguishable for colourblind readers rather than
        eight that are not.
        <button class="btn-sm" id="cmpClear" style="margin-left:8px">Clear</button>
      </div>` : "")}

    ${raw(jobs.length ? html`
      <div class="card" style="padding:0">
        <div class="table-wrap"><table>
          <thead><tr>
            <th style="width:34px"></th>
            <th>Run</th>
            <th>Held-out loss</th>
            <th class="hide-sm">Training loss</th>
            <th class="hide-sm">Steps</th>
            <th class="hide-sm">Trained on</th>
            <th class="hide-sm">Took</th>
            <th>When</th>
          </tr></thead>
          <tbody>${raw(jobs.map((j) => {
            const held = heldOut(j);
            const s = j.summary || {};
            const width = held != null ? 12 + 88 * (1 - (held - best) / span) : 0;
            return html`
              <tr class="${held != null && held === best ? "row-best" : ""}">
                <td><input type="checkbox" data-cmp="${j.id}"
                           ${picked.has(j.id) ? "checked" : ""}></td>
                <td><a href="#/jobs/${j.id}">${j.name}</a>
                  <div class="muted tiny">${j.kind === "pretrain_llm"
                    ? "from scratch" : (j.config.base_model || "fine-tune")}
                    ${raw(s.continued_from
                      ? ` · <span class="badge">continued</span>` : "")}
                    ${raw(j.status === "cancelled"
                      ? ` · <span class="badge">stopped early</span>` : "")}</div></td>
                <td style="min-width:150px">
                  ${raw(held != null ? html`
                    <strong>${held.toFixed(4)}</strong>
                    <div class="meter"><i style="width:${width.toFixed(1)}%"></i></div>`
                    : `<span class="muted tiny">not measured</span>`)}
                </td>
                <td class="hide-sm">${s.final_loss != null ? s.final_loss.toFixed(4) : "—"}</td>
                <td class="hide-sm">${fmtNum(s.steps ?? j.step)}
                  ${raw(s.resumed_from_step
                    ? `<div class="muted tiny">resumed at ${s.resumed_from_step}</div>` : "")}</td>
                <td class="hide-sm tiny muted">${s.tokens_seen
                  ? fmtNum(s.tokens_seen) + " tokens"
                  : (s.held_out_rows ? s.held_out_rows + " held-out rows" : "—")}</td>
                <td class="hide-sm tiny muted">${s.duration_s ? fmtDuration(s.duration_s) : "—"}</td>
                <td class="tiny muted">${fmtAgo(j.finished_at || j.created_at)}</td>
              </tr>`;
          }).join(""))}</tbody>
        </table></div>
        <p class="muted tiny" style="padding:12px 16px;margin:0">
          Runs with no held-out loss finished before this app measured one, or
          trained on a dataset too small to hold any of it back. Their training
          loss is shown and deliberately not ranked: two runs on different data
          produce training losses that are not the same quantity.</p>
      </div>` : html`
      <div class="card empty"><div class="big">📈</div>
        <h3>Nothing to compare yet</h3>
        <p class="muted">Finished runs that produced a model appear here.</p>
        <p><a class="btn btn-primary" href="#/new">Start a run</a></p>
      </div>`)}`;
}
