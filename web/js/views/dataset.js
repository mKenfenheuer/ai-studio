/**
 * One dataset: what is in it, what is wrong with it, and how to fix that.
 *
 * The problems panel is the point of this page. A dataset name tells you
 * nothing; "a fifth of these rows are the same row" changes what you do next,
 * and every problem it reports comes with the button that fixes it.
 */
import { api } from "../api.js";
import { conversationHtml, toolsHtml, isConversation } from "../conversation.js";
import { html, raw, esc, $, $$, on, toast, modal, fmtNum, fmtAgo } from "../util.js";
import { shareButton, wireShareBox } from "./share.js";

export async function datasetView(mount, [id]) {
  let d = await api.dataset(id);
  let stats = null;
  let rows = null;

  // How the rows are shown, and which of them are ticked. Selection is kept
  // across a redraw because the actions that use it -- delete these, move
  // these -- happen after several pages have been looked at.
  let view = "table";
  let selected = new Set();

  const draw = () => {
    mount.innerHTML = layout(d, stats, rows, view, selected);
    wire();
  };

  const reload = async (offset = rows?.offset || 0) => {
    d = await api.dataset(id);
    await showRows(offset);
  };

  // One path for browsing, searching and paging: they differ only in where
  // they start and what they are looking for.
  async function showRows(offset = 0, query = null) {
    const q = query ?? ($("#rowSearch", mount)?.value || "");
    const split = $("#rowSplit", mount)?.value || "";
    try {
      rows = await api.datasetRows(id, offset, PAGE, q, split);
      draw();
    } catch (ex) { toast(ex.message, "err"); }
  }

  function wire() {
    wireShareBox(mount, "dataset", d, async () => { d = await api.dataset(id); draw(); });

    on(mount, "click", "#inspect", async () => {
      $("#inspectBox", mount).innerHTML = `<div class="muted tiny">Reading the data…</div>`;
      try {
        stats = await api.datasetInspect(id);
        draw();
      } catch (ex) { toast(ex.message, "err"); }
    });

    on(mount, "click", "#loadRows", () => showRows(0));
    on(mount, "click", "[data-view]", (_e, t) => { view = t.dataset.view; draw(); });

    // ---- selection -------------------------------------------------------
    on(mount, "change", "[data-row]", (_e, t) => {
      const i = +t.dataset.row;
      if (t.checked) selected.add(i); else selected.delete(i);
      draw();
    });
    on(mount, "change", "#selectAll", (_e, t) => {
      (rows?.rows || []).forEach((r) => {
        if (t.checked) selected.add(r.index); else selected.delete(r.index);
      });
      draw();
    });
    on(mount, "click", "#clearPick", () => { selected = new Set(); draw(); });

    on(mount, "click", "#deleteRows", async () => {
      const n = selected.size;
      if (!confirm(`Delete ${n} row${n === 1 ? "" : "s"} from "${d.name}"?

`
                   + "This changes the dataset itself. Anything already "
                   + "trained on it is unaffected.")) return;
      try {
        const r = await api.editRows(id, { delete: [...selected] });
        selected = new Set();
        toast(r.changed || "Done.", "ok");
        await reload();
      } catch (ex) { toast(ex.message, "err"); }
    });

    on(mount, "change", "#moveTo", async (_e, t) => {
      const to = t.value;
      if (!to) return;
      try {
        const r = await api.editRows(id, { move: { indices: [...selected], to } });
        selected = new Set();
        toast(r.changed || "Done.", "ok");
        await reload();
      } catch (ex) { toast(ex.message, "err"); }
    });

    // ---- one row ---------------------------------------------------------
    on(mount, "click", "[data-open-row]", (e, t) => {
      e.preventDefault();
      const i = +t.dataset.openRow;
      const found = (rows?.rows || []).find((r) => r.index === i);
      if (!found) return;
      const dlg = modal({ title: `Row ${i}`, width: 720,
                          body: rowEditor(found.row, found.rendered) });
      on(dlg, "click", "#saveRow", async () => {
        const next = {};
        let bad = null;
        $$("[data-field]", dlg).forEach((el) => {
          const key = el.dataset.field;
          if (el.dataset.json) {
            try { next[key] = JSON.parse(el.value); }
            catch (ex) { bad = `${key}: ${ex.message}`; }
          } else {
            next[key] = el.value;
          }
        });
        if (bad) {
          $("#rowEditError", dlg).innerHTML =
            `<div class="callout callout-err">${esc(bad)}</div>`;
          return;
        }
        try {
          await api.editRows(id, { update: { [i]: next } });
          dlg.close();
          toast("Row saved.", "ok");
          await reload();
        } catch (ex) {
          $("#rowEditError", dlg).innerHTML =
            `<div class="callout callout-err">${esc(ex.message)}</div>`;
        }
      });
    });

    // A cell holding a conversation or a schema opens in full rather than
    // being truncated into meaninglessness.
    on(mount, "click", "td.clickable", (_e, t) => {
      const found = (rows?.rows || []).find((r) => r.index === +t.dataset.cell);
      if (!found) return;
      modal({ title: `${t.dataset.col} — row ${t.dataset.cell}`, width: 700,
              body: `<p class="mono tiny" style="white-space:pre-wrap;
                        max-height:60vh;overflow:auto">${
                        esc(JSON.stringify(found.row[t.dataset.col], null, 2))}</p>` });
    });

    on(mount, "click", "#addRowsBtn", () => {
      const field = (d.format || {}).text_field || "text";
      const dlg = modal({ title: "Add rows", width: 640, body: html`
        <div class="field">
          <label for="newRows">One row per line</label>
          <textarea id="newRows" rows="10" class="mono"
            placeholder='{"${field}": "an example"}'></textarea>
          <div class="hint">A JSON object per line, using this dataset's
            columns — or plain text, which becomes the
            <code>${field}</code> column.</div>
        </div>
        <div class="field">
          <label for="newRowsSplit">Into the split</label>
          <input id="newRowsSplit" class="mono" value="train">
        </div>
        <div class="row" style="gap:6px">
          <button class="btn-sm btn-primary" id="saveNewRows">Add them</button>
          <button class="btn-sm" data-modal-close>Cancel</button>
        </div>
        <div id="addRowsError"></div>` });
      on(dlg, "click", "#saveNewRows", async () => {
        try {
          const r = await api.addRows(id, {
            rows: $("#newRows", dlg).value,
            split: $("#newRowsSplit", dlg).value });
          dlg.close();
          toast(`Added ${fmtNum(r.added || 0)} rows.`, "ok");
          await reload();
        } catch (ex) {
          $("#addRowsError", dlg).innerHTML =
            `<div class="callout callout-err">${esc(ex.message)}</div>`;
        }
      });
    });
    on(mount, "change", "#rowSplit", () => showRows(0));
    on(mount, "click", "[data-page]", (_e, t) => showRows(+t.dataset.page));
    on(mount, "keydown", "#rowSearch", (e) => {
      if (e.key === "Enter") { e.preventDefault(); showRows(0); }
    });

    on(mount, "submit", "#renameForm", async (e) => {
      e.preventDefault();
      const f = Object.fromEntries(new FormData(e.target).entries());
      try {
        d = await api.renameDataset(id, f);
        toast("Saved.", "ok");
        draw();
      } catch (ex) { toast(ex.message, "err"); }
    });

    // Each reported problem carries the operation that fixes it, so acting on
    // it is one click rather than a trip to the transform form.
    on(mount, "click", "[data-fix]", async (_e, t) => {
      const op = t.dataset.fix;
      const ops = { [op]: true };
      if (op === "cap_length") { delete ops.cap_length; ops.max_chars = 8000; }
      await runTransform(ops, `${d.name} (fixed)`);
    });

    on(mount, "submit", "#transformForm", async (e) => {
      e.preventDefault();
      const f = Object.fromEntries(new FormData(e.target).entries());
      await runTransform(opsFrom(f), f.name);
    });

    on(mount, "click", "#dryRun", async () => {
      const form = $("#transformForm", mount);
      const f = Object.fromEntries(new FormData(form).entries());
      const box = $("#dryRunBox", mount);
      box.innerHTML = `<div class="muted tiny" style="margin-top:8px">Working it out…</div>`;
      try {
        box.innerHTML = dryRunPanel(await api.previewTransform(id, { ops: opsFrom(f) }));
      } catch (ex) {
        box.innerHTML = `<div class="callout callout-err" style="margin-top:8px">${
          esc(ex.message)}</div>`;
      }
    });

    async function runTransform(ops, name) {
      try {
        const made = await api.transformDataset(id, { ops, name });
        toast(`Made "${made.name}" with ${fmtNum(made.rows)} rows.`, "ok");
        location.hash = `#/data/${made.id}`;
      } catch (ex) { toast(ex.message, "err"); }
    }

    on(mount, "submit", "#splitForm", async (e) => {
      e.preventDefault();
      const f = Object.fromEntries(new FormData(e.target).entries());
      try {
        const made = await api.splitDataset(id, { fraction: +f.fraction / 100 });
        toast(`Split into ${made.map((m) => fmtNum(m.rows)).join(" and ")} rows.`, "ok");
        location.hash = `#/data/${made[0].id}`;
      } catch (ex) { toast(ex.message, "err"); }
    });

    on(mount, "submit", "#publishForm", async (e) => {
      e.preventDefault();
      const f = Object.fromEntries(new FormData(e.target).entries());
      const btn = $("#pubGo", mount);
      btn.disabled = true;
      btn.textContent = "Uploading…";
      try {
        const r = await api.publishDataset(id, {
          repo_id: f.repo_id, private: f.visibility === "private" });
        toast("Published.", "ok");
        $("#publishResult", mount).innerHTML =
          `<div class="callout callout-ok"><strong>Published</strong>
            <a href="${esc(r.url)}" target="_blank" rel="noopener">${esc(r.url)}</a></div>`;
      } catch (ex) {
        $("#publishResult", mount).innerHTML =
          `<div class="callout callout-err">${esc(ex.message)}</div>`;
      } finally {
        btn.disabled = false;
        btn.textContent = "Publish to Hugging Face";
      }
    });

    on(mount, "click", "#useForTraining", () => {
      sessionStorage.setItem("aistudio.dataset", JSON.stringify(
        { id: d.id, name: d.name, splits: d.splits || {} }));
      location.hash = "#/new";
    });

    on(mount, "click", "#deleteDs", async () => {
      if (!confirm(`Delete "${d.name}"?\n\nThe rows are removed from this `
                   + "studio permanently. Anything made from it is kept.")) return;
      try {
        const r = await api.deleteDataset(id);
        toast(r.note || "Deleted.", "ok");
        location.hash = "#/data";
      } catch (ex) { toast(ex.message, "err"); }
    });
  }

  draw();
  return () => {};
}

// ---------------------------------------------------------------------------

/** Everything the transform form is asking for, in one object. */
function opsFrom(f) {
  return {
    drop_empty: !!f.drop_empty, dedupe: !!f.dedupe, shuffle: !!f.shuffle,
    max_per_prompt: +f.max_per_prompt || 0,
    min_chars: +f.min_chars || 0, max_chars: +f.max_chars || 0,
    sample: +f.sample || 0, contains: f.contains, excludes: f.excludes,
    to_conversations: !!f.to_conversations, system_prompt: f.system_prompt,
    train_on: f.train_on || "",
    columns: parseCalc(f.calc),
    split_columns: parseSplitColumn(f.split_column),
    drop_columns: splitList(f.drop_columns), rename: parsePairs(f.rename),
  };
}

/** "name = template" per line. The template may itself contain "=" and
 *  newlines, so only the first "=" of a line starts one — and a line that
 *  does not look like a new column is a continuation of the last one. */
function parseCalc(text) {
  const out = [];
  (text || "").split(NEWLINE).forEach((line) => {
    const m = line.match(/^\s*([A-Za-z0-9_.-]{1,64})\s*=\s*([\s\S]*)$/);
    if (m) out.push({ name: m[1], template: m[2] });
    else if (out.length) out[out.length - 1].template += NEWLINE + line;
  });
  return out.filter((c) => c.template.trim());
}

/** "full -> first, last  on  ," */
function parseSplitColumn(text) {
  const out = [];
  (text || "").split(NEWLINE).forEach((line) => {
    const m = line.match(/^\s*([^->]+?)\s*->\s*([^]+?)(?:\s+on\s+([^]*))?$/);
    if (!m) return;
    out.push({ from: m[1].trim(),
               into: m[2].split(",").map((s) => s.trim()).filter(Boolean),
               by: m[3] === undefined ? null : m[3] });
  });
  return out.filter((s) => s.from && s.into.length);
}

/** What a transform would do, before it does it. */
function dryRunPanel(r) {
  const dropped = r.sampled - r.kept;
  const pct = r.sampled ? Math.round((r.kept / r.sampled) * 100) : 0;
  return html`
    <div class="callout ${r.kept ? (pct < 50 ? "callout-warn" : "callout-ok") : "callout-err"}"
         style="margin-top:10px">
      <strong>${fmtNum(r.kept)} of ${fmtNum(r.sampled)} rows would remain${
        dropped ? ` — ${fmtNum(dropped)} dropped` : ""}</strong>
      ${raw(r.partial
        ? `Measured on the first ${fmtNum(r.sampled)} of ${fmtNum(r.total_rows)}
           rows. Deduplication and sampling see only that slice, so treat
           those counts as a shape rather than a total.`
        : "That is the whole dataset, not a sample.")}
      ${raw(Object.keys(r.splits || {}).length > 1
        ? `<div class="tiny" style="margin-top:6px">${Object.entries(r.splits)
            .map(([n, c]) => `<span class="badge">${esc(n)} ${fmtNum(c)}</span>`)
            .join(" ")}</div>` : "")}
      ${raw(r.steps.length ? `<ul class="tiny" style="margin:8px 0 0;padding-left:18px">${
        r.steps.map((s) => `<li>${esc(s)}</li>`).join("")}</ul>` : "")}
    </div>
    ${raw(conversionReport(r.report))}
    ${raw(r.rows.length ? html`
      <details class="adv" open style="margin-top:8px">
        <summary>The first rows, after</summary>
        <div class="table-wrap" style="max-height:260px;overflow:auto;margin-top:8px">
          <table class="table grid-table"><thead><tr>
            ${raw((r.columns || []).map((c) => `<th class="mono">${esc(c)}</th>`).join(""))}
          </tr></thead><tbody>
            ${raw(r.rows.map((row) => html`<tr>
              ${raw((r.columns || []).map((c) => {
                const v = row[c];
                const text = v === null || v === undefined ? ""
                  : typeof v === "object" ? JSON.stringify(v) : String(v);
                return `<td>${esc(text.length > 120 ? text.slice(0, 120) + "…" : text)
                  || '<span class="muted tiny">—</span>'}</td>`;
              }).join(""))}
            </tr>`).join(""))}
          </tbody></table>
        </div>
        <p class="muted tiny" style="margin:8px 0 0">Read as the trainer would:</p>
        <p class="mono tiny" style="white-space:pre-wrap;background:var(--surface-2);
           padding:10px;border-radius:8px;max-height:200px;overflow:auto">${
          (r.rendered || [])[0] || "(reads as nothing)"}</p>
      </details>` : "")}`;
}

/** What the conversion found, and what it had to work out for itself.
 *
 *  The conversion is where data goes wrong, and it goes wrong quietly: a tool
 *  result that answers a call which never happened still renders, still
 *  trains, and still produces a model that has learned something false. This
 *  is the panel that makes that visible while there is still a form on screen
 *  to change.
 */
function conversionReport(rep) {
  if (!rep || !rep.problems) return "";
  const bad = (rep.problems || []).filter((p) => p.level === "error");
  const warn = (rep.problems || []).filter((p) => p.level === "warn");
  if (!rep.converted && !bad.length && !warn.length) return "";

  const line = (p) => `<li><strong>${esc(String(p.count))} row${
    p.count === 1 ? "" : "s"}</strong> — ${esc(p.message)}${
    p.rows?.length ? ` <span class="muted">(e.g. row ${
      p.rows.slice(0, 3).join(", ")})</span>` : ""}</li>`;

  return html`
    <div class="callout ${bad.length ? "callout-err" : warn.length ? "callout-warn" : "callout-ok"}"
         style="margin-top:10px">
      <strong>${fmtNum(rep.converted || 0)} rows converted${
        rep.rows_with_errors ? ` — ${fmtNum(rep.rows_with_errors)} with problems` : ""}</strong>
      ${raw(bad.length || warn.length
        ? ""
        : "Every row reads as a well-formed conversation: no tool result "
          + "answering a call that never happened, no assistant turn with "
          + "nothing in it, no tool called that was never declared.")}
      ${raw(bad.length ? `<div class="tiny" style="margin-top:6px">
        <strong>These would teach the model something false:</strong>
        <ul style="margin:4px 0 0;padding-left:18px">${
          bad.map(line).join("")}</ul></div>` : "")}
      ${raw(warn.length ? `<div class="tiny" style="margin-top:6px">
        <strong>Worth a look:</strong>
        <ul style="margin:4px 0 0;padding-left:18px">${
          warn.map(line).join("")}</ul></div>` : "")}
      ${raw((rep.repairs || []).length ? `<div class="tiny" style="margin-top:6px">
        <strong>Worked out rather than read:</strong>
        <ul style="margin:4px 0 0;padding-left:18px">${
          rep.repairs.map((f) => `<li>${esc(f.what)} — ${fmtNum(f.rows)} rows</li>`)
            .join("")}</ul>
        <span class="muted">Real data leaves the link between a tool call and
        its result implicit. It is reconstructed here and written down, so
        nothing downstream has to guess again.</span></div>` : "")}
      ${raw(rep.dropped ? `<div class="tiny" style="margin-top:6px">${
        fmtNum(rep.dropped)} row(s) had nothing conversational in them and were
        left out.</div>` : "")}
    </div>`;
}

/** "a, b, c" as a list; blank entries ignored. */
function splitList(text) {
  return (text || "").split(",").map((s) => s.trim()).filter(Boolean);
}

/** "old=new, other=thing" as an object; anything malformed ignored. */
function parsePairs(text) {
  const out = {};
  (text || "").split(",").forEach((pair) => {
    const [from, to] = pair.split("=").map((s) => (s || "").trim());
    if (from && to) out[from] = to;
  });
  return out;
}

/** "9,000 train · 1,000 test", or nothing at all for a dataset with one
 *  unnamed split, where saying "train" would be noise. */
function splitBadges(d) {
  const splits = Object.entries(d.splits || {});
  if (splits.length < 2) return "";
  return " · " + splits.map(([name, n]) =>
    `<span class="badge">${esc(name)} ${fmtNum(n)}</span>`).join(" ");
}

function layout(d, stats, rows, view, selected) {
  const canEdit = d.access === "edit";
  return html`
    <div class="page-head">
      <a href="#/data" class="tiny">← All datasets</a>
      <div class="row-between" style="flex-wrap:wrap;gap:8px;margin-top:6px">
        <h1 style="margin:0">${d.name}</h1>
        <div class="row" style="gap:6px;flex-wrap:wrap">
          <button class="btn-sm btn-primary" id="useForTraining">Train on this</button>
          ${raw(shareButton("dataset", d))}
          <a class="btn btn-sm" href="/api/datasets/${d.id}/dataset-file">↓ JSONL</a>
          ${raw(d.mine ? `<button class="btn-sm btn-danger" id="deleteDs">Delete</button>` : "")}
        </div>
      </div>
      <p class="sub tiny">
        ${fmtNum(d.rows)} rows${raw(splitBadges(d))} ·
        ${(d.bytes / 1048576).toFixed(1)} MB ·
        ${d.owner_name || "unowned"} · updated ${fmtAgo(d.updated_at)}
        ${raw(d.access === "view"
          ? ` · <span class="badge">shared with you — read only</span>` : "")}
      </p>
    </div>

    ${raw(d.recipe?.steps?.length ? html`
      <div class="callout" style="margin-bottom:14px">
        <strong>How this was made</strong>
        ${raw(d.parent_name
          ? `From <a href="#/data/${esc(d.parent_id)}">${esc(d.parent_name)}</a>: ` : "")}
        ${d.recipe.steps.join(" · ")}
      </div>` : "")}
    ${raw(d.notes ? `<div class="callout" style="margin-bottom:14px">${esc(d.notes)}</div>` : "")}

    <div class="grid grid-2" style="align-items:start">
      <div>
        <div class="card" style="margin-bottom:14px">
          <div class="row-between" style="margin-bottom:8px">
            <h3 style="margin:0">What is in it</h3>
            <button class="btn-sm" id="inspect">${stats ? "Check again" : "Check the data"}</button>
          </div>
          <div id="inspectBox">${raw(stats ? statsPanel(stats) : html`
            <p class="muted tiny">Samples the rows and reports what is wrong
              with them: empty rows, exact repeats, and the spread of lengths.
              None of that is visible from a row count.</p>`)}</div>
        </div>

        ${raw(workbench(d, rows, view, selected, canEdit))}
      </div>

      <div>
        ${raw(canEdit ? toolsCard(d) : "")}
        ${raw(publishCard(d))}
        ${raw(canEdit ? metaCard(d) : "")}
      </div>
    </div>`;
}

function statsPanel(s) {
  const worst = { error: 0, warn: 1, info: 2, ok: 3 };
  const problems = [...s.problems].sort((a, b) => worst[a.level] - worst[b.level]);
  return html`
    <div style="margin-bottom:12px">
      ${raw(problems.map((p) => html`
        <div class="callout ${p.level === "error" ? "callout-err"
          : p.level === "warn" ? "callout-warn"
          : p.level === "ok" ? "callout-ok" : ""}" style="margin-bottom:8px">
          ${p.message}
          ${raw(p.fix ? `<button class="btn-sm btn-primary" data-fix="${esc(p.fix)}"
                  style="margin-top:6px">Fix it — makes a new dataset</button>` : "")}
        </div>`).join(""))}
    </div>

    <dl class="kv">
      <dt>Sampled</dt><dd>${fmtNum(s.sampled)} of ${fmtNum(s.total_rows)} rows</dd>
      <dt>Unique</dt><dd>${fmtNum(s.unique_rows)}</dd>
      <dt>Length</dt><dd>${fmtNum(s.chars.p50)} characters typical,
        ${fmtNum(s.chars.p90)} at the 90th percentile, ${fmtNum(s.chars.max)} longest</dd>
      <dt>Rough size</dt><dd>~${fmtNum(s.est_tokens)} tokens in the sample</dd>
    </dl>

    <div style="margin-top:12px">
      <div class="muted tiny" style="margin-bottom:6px">Length distribution
        (characters, log scale)</div>
      <div class="row" style="gap:2px;align-items:flex-end;height:60px">
        ${raw(bars(s.histogram))}
      </div>
    </div>

    <details class="adv" style="margin-top:12px">
      <summary>Columns</summary>
      <table class="table" style="margin-top:8px"><tbody>
        ${raw(s.columns.map((c) => html`
          <tr><td class="mono tiny">${c.name}</td>
            <td class="tiny muted">${Math.round(c.fill_rate * 100)}% filled</td></tr>`).join(""))}
      </tbody></table>
    </details>

    <details class="adv" style="margin-top:8px">
      <summary>How the trainer will read these rows</summary>
      <div class="mono tiny" style="white-space:pre-wrap;max-height:240px;overflow:auto;
           background:var(--surface-2);padding:10px;border-radius:6px;margin-top:8px">
        ${(s.preview || []).slice(0, 3).join("\n\n───\n\n")}</div>
    </details>`;
}

function bars(hist) {
  if (!hist?.length) return "";
  const max = Math.max(...hist.map((h) => h.count), 1);
  return hist.map((h) => html`
    <span title="${fmtNum(h.from)}–${fmtNum(h.to)} chars: ${fmtNum(h.count)} rows"
          style="flex:1;background:var(--accent);opacity:.75;border-radius:2px 2px 0 0;
                 height:${Math.max(2, Math.round((h.count / max) * 100))}%"></span>`).join("");
}

// The filters a calculated column may use. Listed for the hint text; the
// controller is the authority on what they do.
const FILTERS = ["upper", "lower", "title", "trim", "lines", "first", "last",
                 "len", "words", "json", "slice:0:200"];

// Written once rather than inline, because a newline inside a string
// literal in this file is a newline in the source: the escape has to
// survive every editor that touches it.
const NEWLINE = String.fromCharCode(10);

const PAGE = 25;
// How much of a cell is shown before it is cut. Long enough to recognise the
// value, short enough that twenty rows still look like a table.
const CELL = 140;

/** Is this a value a table cell can just print? */
const scalar = (v) => v === null || v === undefined
  || ["string", "number", "boolean"].includes(typeof v);

/** A short, honest stand-in for a value too big for a cell. */
function summarise(value) {
  if (Array.isArray(value)) {
    // A conversation is the common case, and "4 messages" says more than the
    // first forty characters of its JSON would.
    const kind = value.every((v) => v && typeof v === "object" && v.role)
      ? "message" : "item";
    return `${value.length} ${kind}${value.length === 1 ? "" : "s"}`;
  }
  if (value && typeof value === "object") {
    const keys = Object.keys(value);
    return `${keys.length} field${keys.length === 1 ? "" : "s"}: ${
      keys.slice(0, 3).join(", ")}${keys.length > 3 ? "…" : ""}`;
  }
  return String(value ?? "");
}

function cell(value) {
  if (value === null || value === undefined || value === "") {
    return `<span class="muted tiny">—</span>`;
  }
  if (!scalar(value)) {
    return `<span class="badge">${esc(summarise(value))}</span>`;
  }
  const text = String(value);
  return esc(text.length > CELL ? text.slice(0, CELL) + "…" : text);
}

/** The rows, as a table where the data is a table.
 *
 *  Reading data as JSON is reading it through a keyhole: you cannot compare
 *  two rows, you cannot see that a column is empty in half of them, and the
 *  punctuation outweighs the content. So the columns become columns. Values
 *  that are not scalars — a conversation, a tool schema — are summarised in
 *  place and open in full on a click, because a table cell is the wrong shape
 *  for a nested object but the right shape for knowing one is there.
 */
function rowTable(d, data, view, selected) {
  const shown = data.rows.length;
  const matched = data.matched ?? data.total;
  const searching = !!data.query;
  const columns = columnsOf(d, data);

  if (!shown) {
    return html`
      <div class="callout callout-warn">
        <strong>${searching ? `Nothing matched "${data.query}"`
                            : "No rows here"}</strong>
        ${raw(data.capped
          ? `Only the first ${fmtNum(data.scanned)} rows were searched — this
             dataset is larger than the search reads.` : "")}
      </div>`;
  }

  return html`
    ${raw(view === "text" ? renderedList(data) : html`
      <div class="table-wrap" style="max-height:460px;overflow:auto">
        <table class="table grid-table">
          <thead><tr>
            <th style="width:26px"><input type="checkbox" id="selectAll"
                title="Select every row on this page"></th>
            <th style="width:56px">#</th>
            ${raw(columns.map((c) => html`
              <th><span class="mono">${c}</span></th>`).join(""))}
          </tr></thead>
          <tbody>
            ${raw(data.rows.map((r) => html`
              <tr class="${selected.has(r.index) ? "picked" : ""}">
                <td><input type="checkbox" data-row="${r.index}"${
                  selected.has(r.index) ? " checked" : ""}></td>
                <td class="muted tiny mono">
                  <a href="#" data-open-row="${r.index}">${r.index}</a></td>
                ${raw(columns.map((c) => html`
                  <td data-cell="${r.index}" data-col="${c}"
                      class="${scalar(r.row[c]) ? "" : "clickable"}">${
                    raw(cell(r.row[c]))}</td>`).join(""))}
              </tr>`).join(""))}
          </tbody>
        </table>
      </div>`)}

    ${raw(pager(data))}`;
}

/** Where you are in the rows, and how to get to the rest of them. */
function pager(data) {
  const shown = data.rows.length;
  const matched = data.matched ?? data.total;
  const searching = !!data.query;
  return html`
    <div class="row-between" style="margin-top:8px;flex-wrap:wrap;gap:8px">
      <span class="muted tiny">
        Rows ${fmtNum(data.offset + 1)}–${fmtNum(data.offset + shown)} of
        ${fmtNum(matched)}${searching ? " matching" : ""}${
          data.split ? ` in ${data.split}` : ""}.
        ${raw(searching && data.capped
          ? `Searched the first ${fmtNum(data.scanned)} rows.` : "")}
      </span>
      <span class="row" style="gap:6px">
        <button class="btn-sm" data-page="${Math.max(0, data.offset - PAGE)}"
                ${data.offset ? "" : "disabled"}>← Back</button>
        <button class="btn-sm" data-page="${data.offset + PAGE}"
                ${data.offset + shown < matched ? "" : "disabled"}>Next →</button>
      </span>
    </div>`;
}

/** Which columns to show, in a stable order, with `split` last because it is
 *  bookkeeping rather than content. */
function columnsOf(d, data) {
  const seen = [];
  (d.columns || []).forEach((c) => seen.push(c));
  data.rows.forEach((r) => Object.keys(r.row).forEach((k) => {
    if (!seen.includes(k)) seen.push(k);
  }));
  return seen.filter((c) => c !== "split")
    .concat(seen.includes("split") ? ["split"] : []);
}

/** The other view: what the trainer will actually read, row by row. */
function renderedList(data) {
  return html`
    <div style="max-height:460px;overflow:auto">
      ${raw(data.rows.map((r) => html`
        <div style="border-bottom:1px solid var(--border);padding:8px 0">
          <div class="row-between">
            <a class="muted tiny mono" href="#" data-open-row="${r.index}">row ${r.index}</a>
            <span class="muted tiny">${fmtNum((r.rendered || "").length)} characters</span>
          </div>
          <div class="mono tiny" style="white-space:pre-wrap">${
            r.rendered || "(reads as nothing)"}</div>
        </div>`).join(""))}
    </div>`;
}

function toolsCard(d) {
  return html`
    <div class="card" style="margin-bottom:14px">
      <h3>Clean it up</h3>
      <p class="muted tiny">Every one of these writes a <strong>new</strong>
        dataset. This one is never changed, so nothing here can go wrong
        expensively.</p>

      <form id="transformForm" style="margin-top:12px">
        <div class="field">
          <label for="tName">Call the result</label>
          <input id="tName" name="name" type="text"
                 placeholder="${d.name} (cleaned)">
        </div>
        <label class="check"><input type="checkbox" name="drop_empty" checked>
          Drop rows that render as nothing</label>
        <label class="check"><input type="checkbox" name="dedupe" checked>
          Remove exact duplicates</label>
        <div class="field">
          <label for="tPerPrompt">Most rows to keep per question</label>
          <input id="tPerPrompt" name="max_per_prompt" type="number"
                 min="0" step="1" placeholder="no limit">
          <div class="hint">Different from the box above, and the one that
            bites on generated data: a model asked repeatedly for examples on a
            topic converges on the same obvious question and varies only the
            answer, so no two rows are identical and one question still takes a
            large share of the file. Two or three keeps the useful part of that
            — several good answers to one question — without teaching the
            question.</div>
        </div>
        <label class="check"><input type="checkbox" name="shuffle">
          Shuffle</label>
        <div class="callout" style="margin:12px 0">
          <strong>🧩 Convert to the standard conversation format</strong>
          Whatever this data is now — ShareGPT turns, two flat columns, a
          tool-calling set with the schema in a sibling column — this rewrites
          every row into the one shape the trainer, the playground and the API
          all read: <code>messages</code>, <code>tools</code>,
          <code>meta</code>. It is the same format OpenAI's fine-tuning files
          use, which is what every published chat template already knows how to
          render.
        </div>
        <label class="check"><input type="checkbox" name="to_conversations">
          Convert to conversations</label>
        <div class="field">
          <label for="tSys">System prompt to add (when converting)</label>
          <input id="tSys" name="system_prompt" type="text"
                 placeholder="You are a helpful assistant.">
          <div class="hint">Added only to rows that do not already have one.
            A model fine-tuned with a system prompt behaves noticeably
            differently without it, so it is worth setting deliberately.</div>
        </div>
        <div class="field">
          <label for="tTrainOn">What a run should learn from these rows</label>
          <select id="tTrainOn" name="train_on">
            <option value="">Every assistant turn (the usual choice)</option>
            <option value="last">Only the final answer in each conversation</option>
            <option value="all">Every token, questions included</option>
          </select>
          <div class="hint">The questions and the tool results are always
            rendered — the model reads them. This is about which tokens it is
            scored on. Training on the questions teaches it to write the next
            question itself.</div>
        </div>
        <div class="grid grid-3">
          <div class="field">
            <label for="tMin">Shortest</label>
            <input id="tMin" name="min_chars" type="number" min="0" placeholder="any">
          </div>
          <div class="field">
            <label for="tMax">Longest</label>
            <input id="tMax" name="max_chars" type="number" min="0" placeholder="any">
          </div>
          <div class="field">
            <label for="tSample">Keep at most</label>
            <input id="tSample" name="sample" type="number" min="0" placeholder="all">
          </div>
        </div>
        <details class="adv">
          <summary>Filter by content</summary>
          <div class="field" style="margin-top:8px">
            <label for="tHas">Keep rows matching</label>
            <input id="tHas" name="contains" type="text" class="mono"
                   placeholder="regular expression">
          </div>
          <div class="field">
            <label for="tNot">Remove rows matching</label>
            <input id="tNot" name="excludes" type="text" class="mono"
                   placeholder="regular expression">
          </div>
        </details>

        <details class="adv" open>
          <summary>Columns</summary>
          <p class="muted tiny" style="margin:8px 0 0">This dataset has
            ${raw((d.columns || []).map((c) =>
              `<code>${esc(c)}</code>`).join(" ") || "no columns recorded yet")}.</p>

          <div class="field" style="margin-top:8px">
            <label for="tCalc">Calculated columns</label>
            <textarea id="tCalc" name="calc" rows="4" class="mono"
              placeholder="text = Q: {${(d.columns || ["question"])[0]}}&#10;A: {${
                (d.columns || ["x", "answer"])[1] || "answer"}}"></textarea>
            <div class="hint">
              One per line, <code>name = template</code>. Any
              <code>{column}</code> is replaced with that row's value, so this
              is how columns are concatenated and how a spreadsheet becomes
              something a model can be trained on. Values can be passed
              through <code>|</code> filters:
              <code>{title|trim|upper}</code>, <code>{body|first}</code>,
              <code>{body|slice:0:400}</code>, <code>{tools|json}</code>.
              Available: ${raw(FILTERS.map((f) => `<code>${f}</code>`).join(" "))}.
              The last column built is what the trainer then reads.
            </div>
          </div>

          <div class="field">
            <label for="tSplitCol">Split a column into several</label>
            <input id="tSplitCol" name="split_column" type="text" class="mono"
                   placeholder="name -> first, last  on  ,">
            <div class="hint">
              <code>source -&gt; a, b</code> splits on whitespace;
              add <code>on ,</code> to split on something else.
            </div>
          </div>

          <div class="grid grid-2">
            <div class="field">
              <label for="tDrop">Drop columns</label>
              <input id="tDrop" name="drop_columns" type="text" class="mono"
                     placeholder="question, answer">
              <div class="hint">Applied after the calculations, so a column a
                template reads from can still be thrown away.</div>
            </div>
            <div class="field">
              <label for="tRename">Rename columns</label>
              <input id="tRename" name="rename" type="text" class="mono"
                     placeholder="prompt=instruction, reply=output">
              <div class="hint">Renaming to a name the trainer knows —
                <code>instruction</code>, <code>output</code>,
                <code>messages</code>, <code>text</code> — is often all a
                dataset needs.</div>
            </div>
          </div>
        </details>

        <div class="row" style="gap:6px;flex-wrap:wrap">
          <button class="btn-sm" type="button" id="dryRun">Show me what it would do</button>
          <button class="btn-sm btn-primary" type="submit">Make a new dataset</button>
        </div>
        <div id="dryRunBox"></div>
      </form>

      <hr style="margin:16px 0;border:0;border-top:1px solid var(--border)">

      <h3>Hold some back</h3>
      <p class="muted tiny">Splits into a training set and a validation set.
        Without one there is no way to tell a model that has learned from one
        that has memorised — the training loss looks identical either way.</p>
      <form id="splitForm" class="row row-top" style="margin-top:10px">
        <div class="field" style="max-width:130px">
          <label for="sFrac">Hold back %</label>
          <input id="sFrac" name="fraction" type="number" value="10" min="1" max="50">
        </div>
        <button class="btn-sm btn-primary" type="submit">Split</button>
      </form>
    </div>`;
}

function publishCard(d) {
  return html`
    <div class="card" style="margin-bottom:14px">
      <h3>Publish to Hugging Face</h3>
      <p class="muted tiny">Uploads the rows and a dataset card to your own
        account. Needs a connected account with write access —
        <a href="#/account">set that up here</a>.</p>
      <form id="publishForm" style="margin-top:10px">
        <div class="field">
          <label for="pubRepo">Repository</label>
          <input id="pubRepo" name="repo_id" type="text" class="mono" required
                 placeholder="your-name/${slug(d.name)}">
        </div>
        <div class="field">
          <label for="pubVis">Visibility</label>
          <select id="pubVis" name="visibility">
            <option value="private">Private</option>
            <option value="public">Public — anyone can download it</option>
          </select>
        </div>
        <button class="btn-sm" type="submit" id="pubGo">Publish to Hugging Face</button>
      </form>
      <div id="publishResult"></div>
    </div>`;
}

function metaCard(d) {
  return html`
    <div class="card">
      <h3>Name and notes</h3>
      <form id="renameForm" style="margin-top:10px">
        <div class="field">
          <label for="mName">Name</label>
          <input id="mName" name="name" type="text" value="${d.name}" maxlength="120">
        </div>
        <div class="field">
          <label for="mNotes">Notes</label>
          <textarea id="mNotes" name="notes" rows="3"
                    placeholder="What is this, and what is it for?">${d.notes || ""}</textarea>
        </div>
        <button class="btn-sm" type="submit">Save</button>
      </form>
    </div>`;
}

const slug = (s) => (s || "dataset").toLowerCase().replace(/[^a-z0-9]+/g, "-")
  .replace(/^-|-$/g, "").slice(0, 60) || "dataset";

/** The rows, and everything you can do to them.
 *
 *  One panel rather than a viewer and a separate set of tools: curating data
 *  is a loop — look, notice, change, look again — and every step of that loop
 *  that costs a page navigation is a step people stop taking.
 */
function workbench(d, rows, view, selected, canEdit) {
  const splits = Object.entries(d.splits || {});
  const picked = selected.size;
  return html`
    <div class="card" style="margin-bottom:14px">
      <div class="row-between" style="margin-bottom:8px;flex-wrap:wrap;gap:8px">
        <h3 style="margin:0">The rows themselves</h3>
        <div class="row" style="gap:6px;flex-wrap:wrap">
          ${raw(splits.length > 1 ? html`
            <select id="rowSplit" style="max-width:150px">
              <option value="">every split</option>
              ${raw(splits.map(([name, n]) => html`
                <option value="${name}"${rows?.split === name ? " selected" : ""}>${
                  name} (${fmtNum(n)})</option>`).join(""))}
            </select>` : "")}
          <input type="search" id="rowSearch" placeholder="Find text…"
                 value="${rows?.query || ""}" style="max-width:190px">
          <button class="btn-sm" id="loadRows">${rows ? "Search" : "Open the rows"}</button>
          ${raw(rows ? html`
            <div class="seg">
              <button class="btn-sm ${view === "table" ? "on" : ""}"
                      data-view="table">Table</button>
              <button class="btn-sm ${view === "text" ? "on" : ""}"
                      data-view="text">As trained</button>
            </div>` : "")}
          ${raw(canEdit ? `<button class="btn-sm" id="addRowsBtn">+ Add rows</button>` : "")}
        </div>
      </div>

      ${raw(picked && canEdit ? html`
        <div class="callout" style="margin:0 0 10px">
          <div class="row-between" style="flex-wrap:wrap;gap:8px">
            <strong class="tiny">${picked} row${picked === 1 ? "" : "s"} selected</strong>
            <div class="row" style="gap:6px;flex-wrap:wrap">
              <select id="moveTo" style="max-width:150px">
                <option value="">move to split…</option>
                ${raw(["train", "validation", "test"].concat(
                  splits.map(([n]) => n)).filter((v, i, a) => a.indexOf(v) === i)
                  .map((n) => `<option value="${esc(n)}">${esc(n)}</option>`).join(""))}
              </select>
              <button class="btn-sm btn-danger" id="deleteRows">Delete them</button>
              <button class="btn-sm" id="clearPick">Clear</button>
            </div>
          </div>
        </div>` : "")}

      ${raw(rows ? rowTable(d, rows, view, selected) : html`
        <p class="muted tiny" style="margin:0">${fmtNum(d.rows)} rows across
          ${splits.length || 1} split${splits.length === 1 ? "" : "s"}. Open
          them to read, search, correct and curate — the table is editable.</p>`)}
    </div>`;
}

/** One row, opened.
 *
 *  For a conversation, the conversation is the row: bubbles by role, the
 *  model's working folded away beside its answer, a tool call shown as a call.
 *  Reading that as JSON is reading it through a keyhole -- the punctuation
 *  outweighs the content and a call's arguments are an escaped string inside a
 *  string.
 *
 *  The fields are still there, underneath, because they are how it is edited:
 *  no set of inputs is the right shape for a turn that calls two tools, so the
 *  JSON stays as the editing surface rather than being replaced by one.
 *  Anything that is not a conversation gets the fields alone, as before.
 */
function rowEditor(row, rendered) {
  const fields = Object.entries(row);
  const chat = isConversation(row);
  const editors = html`
    ${raw(fields.map(([k, v]) => {
      const simple = v === null || v === undefined
        || ["string", "number", "boolean"].includes(typeof v);
      return html`
        <div class="field" style="margin:0 0 10px">
          <label for="rf_${k}">${k}</label>
          ${raw(simple
            ? `<textarea id="rf_${k}" data-field="${esc(k)}" rows="${
                String(v ?? "").length > 90 ? 4 : 1
              }" class="mono">${esc(String(v ?? ""))}</textarea>`
            : `<textarea id="rf_${k}" data-field="${esc(k)}" data-json="1"
                  rows="8" class="mono">${esc(JSON.stringify(v, null, 2))}</textarea>
               <div class="hint">Edited as JSON — this value is a ${
                 Array.isArray(v) ? "list" : "structure"}.</div>`)}
        </div>`;
    }).join(""))}`;

  return html`
    <div class="grid" style="gap:12px">
      ${raw(!chat ? editors : html`
        ${raw(toolsHtml(row.tools))}
        <div class="convo">${raw(conversationHtml(row.messages))}</div>
        <details class="adv">
          <summary>Edit this row</summary>
          <div style="margin-top:10px">${raw(editors)}</div>
        </details>`)}
      <details class="adv">
        <summary>What the trainer reads</summary>
        <p class="mono tiny" style="white-space:pre-wrap;background:var(--surface-2);
           padding:10px;border-radius:8px;max-height:200px;overflow:auto">${
          rendered || "(reads as nothing)"}</p>
      </details>
      <div class="row" style="gap:6px">
        <button class="btn-sm btn-primary" id="saveRow">Save this row</button>
        <button class="btn-sm" data-modal-close>Cancel</button>
      </div>
      <div id="rowEditError"></div>
    </div>`;
}
