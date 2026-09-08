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
import { html, raw, esc, $, $$, on, toast, fmtAgo, fmtNum, fmtDuration } from "../util.js";
import { LineChart } from "../chart.js";
import { ribbon, rb, group, rbSeg, rbSelect, rbSearch, wireRibbon } from "../ribbon.js";
import { primaryMetric } from "../kinds.js";
import { pageHead, emptyState } from "../components.js";

// The kinds of run that leave a model behind. `has_model` alone let
// dataset-generation runs in -- they register an artifact too -- and they
// sat in the table labelled "fine-tune" with nothing to rank.
const MODEL_KINDS = { finetune_llm: "Fine-tunes", pretrain_llm: "From scratch",
                      merge_adapter: "Fine-tunes" };
const family = (j) => MODEL_KINDS[j.kind];

export async function compareView(mount) {
  const all = (await api.jobs())
    .filter((j) => ["succeeded", "cancelled"].includes(j.status) && j.has_model
                   && family(j));
  // One family at a time. A from-scratch model's held-out loss is measured
  // against its own small vocabulary and a fine-tune's against a 150k one;
  // ranking the two together was a league table of unrelated numbers.
  const families = [...new Set(all.map(family))];
  let shown = families.includes("Fine-tunes") ? "Fine-tunes" : families[0];
  const picked = new Set();
  let chart = null;
  let q = "";
  let onlyBase = "";
  let onlyData = "";

  const jobsShown = () => all.filter((j) => {
    if (family(j) !== shown) return false;
    if (onlyBase && (j.config?.base_model || "") !== onlyBase) return false;
    if (onlyData && (j.config?.dataset_label || j.config?.dataset || "") !== onlyData) return false;
    if (!q) return true;
    return [j.name, j.notes, j.config?.base_model,
            j.config?.dataset_label, j.config?.dataset]
      .filter(Boolean).join(" ").toLowerCase().includes(q);
  });

  const draw = () => {
    mount.innerHTML = layout(jobsShown(), picked, families, shown,
                             { all, q, onlyBase, onlyData });
    if (chart) { chart.destroy(); chart = null; }
    wire();
    paintCurves();
  };

  function wire() {
    on(mount, "change", "[data-cmp]", (_e, t) => {
      if (t.checked) picked.add(t.dataset.cmp); else picked.delete(t.dataset.cmp);
      draw();
    });
    on(mount, "click", "#cmpClear, [data-clear-picks]", () => { picked.clear(); draw(); });
    on(mount, "click", "[data-family]", (_e, t) => {
      shown = t.dataset.family; picked.clear(); draw();
    });
    on(mount, "input", "#cmpQ", (_e, t) => { q = t.value.toLowerCase(); draw(); });
    on(mount, "change", "#cmpBase", (_e, t) => { onlyBase = t.value; draw(); });
    on(mount, "change", "#cmpData", (_e, t) => { onlyData = t.value; draw(); });
    on(mount, "click", "#cmpCsv", () => downloadCsv(jobsShown()));
    wireRibbon(mount, () => {});
  }

  async function paintCurves() {
    const box = $("#cmpChart", mount);
    if (!box || picked.size !== 2) return;
    const [a, b] = [...picked];
    const nameOf = (id) => (all.find((j) => j.id === id) || {}).name || id;
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

/** The distinct values of one field across these runs, as select options. */
function distinct(jobs, pick) {
  const seen = [...new Set((jobs || []).map(pick).filter(Boolean))].sort();
  return seen.map((v) => [v, v.length > 44 ? "…" + v.slice(-42) : v]);
}

const heldOut = (j) => primaryMetric(j)?.value ?? null;
// Which way is up, from the first run that says. Mixed polarities in one
// table would be a comparison of unrelated numbers, which the family
// switch already prevents.
const lowerBetter = (rows) => (rows.map(primaryMetric).find(Boolean)?.lower) !== false;

/** The table as a file, because the next question is always asked in a
 *  spreadsheet and there was no way to get the numbers out of here. */
function downloadCsv(jobs) {
  const cell = (v) => {
    const s = v == null ? "" : String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  const head = ["name", "kind", "base_model", "dataset", "held_out_loss",
                "final_loss", "steps", "tokens_seen", "duration_s", "seed",
                "status", "finished_at", "notes"];
  const rows = jobs.map((j) => {
    const s = j.summary || {};
    return [j.name, j.kind, j.config?.base_model || "",
            j.config?.dataset_label || j.config?.dataset || "",
            s.best_val_loss ?? "", s.final_loss ?? "", s.steps ?? j.step ?? "",
            s.tokens_seen ?? "", s.duration_s ?? "", s.seed ?? "",
            j.status, j.finished_at ? new Date(j.finished_at * 1000).toISOString() : "",
            j.notes || ""].map(cell).join(",");
  });
  const blob = new Blob([[head.join(","), ...rows].join("\n")],
                        { type: "text/csv" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `runs-${new Date().toISOString().slice(0, 10)}.csv`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}

function layout(jobs, picked, families, shown, filters = {}) {
  const withHeld = jobs.filter((j) => heldOut(j) != null);
  const lower = lowerBetter(withHeld);
  const best = withHeld.length
    ? (lower ? Math.min : Math.max)(...withHeld.map(heldOut)) : null;
  const worst = withHeld.length
    ? (lower ? Math.max : Math.min)(...withHeld.map(heldOut)) : null;
  const metricLabel = (withHeld.map(primaryMetric).find(Boolean)?.label) || "Held-out loss";
  const span = (worst ?? 0) - (best ?? 0) || 1;

  return html`
    ${raw(pageHead({
      title: "Compare runs",
      sub: "Held-out loss is the number that carries between runs.",
    }))}
    ${raw(ribbon({
      tabs: [{ key: "home", label: "Compare" }], active: "home",
      body: group("Which runs", [
        families.length > 1
          ? rbSeg(families.map((f) => ({ label: f, on: f === shown,
                                         data: `data-family="${esc(f)}"` })))
          : "",
        rb("cmpClear", "✕", "Clear selection", { disabled: !picked.size }),
      ]) + group("Narrow it", [
        rbSearch("cmpQ", { placeholder: "Name, model or dataset…",
                           value: filters.q || "" }),
        rbSelect("cmpBase", { title: "Base model", value: filters.onlyBase || "",
          options: [["", "Any base model"]].concat(
            distinct(filters.all, (j) => j.config?.base_model)) }),
        rbSelect("cmpData", { title: "Dataset", value: filters.onlyData || "",
          options: [["", "Any dataset"]].concat(
            distinct(filters.all, (j) => j.config?.dataset_label || j.config?.dataset)) }),
      ]) + group("Take it away", [
        rb("cmpCsv", "↓", "As a spreadsheet", { disabled: !jobs.length,
          title: "Every row in this table, as CSV" }),
      ]) + group("Elsewhere", [
        rb(null, "◎", "Prompt sets", { href: "#/evals",
          title: "The same questions, put to every model" }),
        rb(null, "≡", "All runs", { href: "#/jobs" }),
      ]),
    }))}

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
        <button class="btn-sm" data-clear-picks style="margin-left:8px">Clear</button>
      </div>` : "")}

    ${raw(jobs.length ? html`
      <div class="card" style="padding:0">
        <div class="table-wrap"><table>
          <thead><tr>
            <th style="width:34px"></th>
            <th>Run</th>
            <th>${metricLabel}</th>
            <th class="hide-sm">Training loss</th>
            <th class="hide-sm">Steps</th>
            <th class="hide-sm">Trained on</th>
            <th class="hide-sm">Took</th>
            <th>When</th>
          </tr></thead>
          <tbody>${raw(jobs.map((j) => {
            const held = heldOut(j);
            const s = j.summary || {};
            // Distance from the best, whichever direction the metric runs.
            const width = held != null ? 12 + 88 * (1 - Math.abs(held - best) / span) : 0;
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
      </div>` : emptyState({
        icon: "📈",
        title: "Nothing to compare yet",
        body: "Finished runs that produced a model appear here.",
        cta: { href: "#/new", label: "Start a run" },
      }))}`;
}
