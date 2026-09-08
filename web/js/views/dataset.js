/**
 * One dataset, opened in an editor shaped like a query editor.
 *
 * Three panes and a ribbon. The rows sit in the middle, because they are the
 * thing being worked on. Everything that can be done to them is a button on
 * a tabbed ribbon above, and every step taken is a line in the list on the
 * right -- selectable, so the table shows the rows as they stood after that
 * step; removable, so a step that turned out wrong is one click gone; and
 * editable, so the number in it can be changed without redoing what came
 * after. Nothing is written until "Apply", and Apply writes a new dataset.
 *
 * With no steps, the table is the file itself: pageable, searchable, and
 * editable row by row, which is the other half of curating data and the half
 * a pipeline cannot do.
 */
import { api } from "../api.js";
import { conversationHtml, toolsHtml, isConversation } from "../conversation.js";
import { html, raw, esc, $, $$, on, toast, modal, fmtNum, fmtAgo,
         debounce, inlineRename, resource, ensure } from "../util.js";
import { ribbon, rb, group, rbSelect, rbSeg, wireRibbon, tabState } from "../ribbon.js";
import { confirmDestructive, breadcrumb } from "../components.js";
import { shareButton, wireShareBox } from "./share.js";
import { publishCard, wirePublish } from "./publish.js";
import { qualityBadge } from "./data.js";
import { STEPS, TABS, stepsOnTab, stepFrom, describe } from "./dataset-steps.js";

const PAGE = 25;
// The reserved column holding a row's own name. A row is referred to by that
// name rather than by where it happens to sit, because deleting one row used
// to renumber every row after it -- so a second click, built from a list read
// before the first, acted on somebody else's data.
const ROW_ID = "_id";
/** How to refer to this row: its name, or its position on a dataset written
 *  before names existed. */
const handleOf = (r) => r.row?.[ROW_ID] ?? r.index;
// How much of a cell is shown before it is cut. Long enough to recognise the
// value, short enough that fifty rows still look like a table.
const CELL = 140;
const SAMPLES = [500, 2000, 10000, 0];
const draftKey = (id) => `aistudio.steps.${id}`;

export async function datasetView(mount, [id]) {
  let d = await api.dataset(id);
  // The editor wants the width the rest of the app does not.
  mount.classList.add("pq-wide");

  // ---- state --------------------------------------------------------------
  let library = [];
  let rows = null;                 // a page of the file itself
  let stats = null;                // the quality report, once asked for
  let trainedOn = [];              // the runs that read this data
  api.datasetRuns(id).then((r) => { trainedOn = r; draw(); }).catch(() => {});
  const training = resource();     // what the trainer will read, once asked
  const convReport = resource();   // whether it reads as conversations
  let trainSplit = "";             // which split the training tab is showing
  let contextLen = +(localStorage.getItem("aistudio.ctxLen") || 2048);
  let checkSample = +(localStorage.getItem("aistudio.checkSample") || 2000);
  let steps = loadDraft(id);       // [{type, ops}]
  let selected = steps.length - 1; // -1 is the source
  let history = [], future = [];   // undo and redo, as whole step lists
  let preview = null;              // what the pipeline does, up to `selected`
  let previewErr = null;
  let stages = null;               // per-step counts, for the whole pipeline
  let loading = false;
  const tabs = tabState("dataset", TABS, "home");
  let tab = tabs.get();
  let view = "table";
  let picked = new Set();
  let sample = +(localStorage.getItem("aistudio.previewSample") ?? 2000);
  let leftOpen = localStorage.getItem("aistudio.pqLeft") !== "0";
  let colMenu = null;              // {column, x, y}
  let gen = 0;                     // bumped whenever a preview answer would be stale
  const previewCache = new Map();

  const inSource = () => selected < 0;
  const pipeline = () => steps.map((s) => s.ops);

  // ---- drawing -----------------------------------------------------------
  const draw = () => {
    mount.innerHTML = layout({ d, library, rows, steps, selected, preview, previewErr, trainedOn,
                               stages, loading, tab, view, picked, sample, leftOpen,
                               history, future, colMenu, stats, checkSample,
                               training, convReport, trainSplit, contextLen });
    wire();
    // Everything the training tab shows is fetched the first time it is
    // opened, and again when the split changes -- never on the way past.
    if (tab === "train") {
      ensure(training, `${id}:${trainSplit}`,
             () => api.trainingPreview({ studio_dataset: id, split: trainSplit }),
             draw);
      if ((d.format || {}).mode === "chat") {
        ensure(convReport, id, () => api.conversationReport(id), draw);
      }
    }
  };

  // ---- the review queue --------------------------------------------------
  //
  // One row at a time from the `review` split. Keep moves it to train; drop
  // deletes it; skip leaves it for later. The row is drawn the way the
  // trainer will read it, because that is the thing being judged.
  async function reviewQueue() {
    let page = null;
    let decided = 0;
    const dlg = modal({ title: "Review", width: 720, body: `<div id="rqBody"></div>` });
    const body = $("#rqBody", dlg);
    const next = async () => {
      try { page = await api.datasetRows(id, 0, 1, "", "review"); }
      catch (ex) { toast(ex.message, "err"); return; }
      const r = page.rows?.[0];
      if (!r) {
        body.innerHTML = html`<p class="muted">Nothing left to review${decided ? ` — ${decided} decided` : ""}.</p>
          <p><button class="btn btn-primary" data-modal-close>Done</button></p>`;
        await reload();
        return;
      }
      const conv = Array.isArray(r.row.messages) ? r.row.messages : null;
      body.innerHTML = html`
        <p class="muted tiny">${fmtNum(page.matched)} waiting${decided ? ` · ${decided} decided` : ""}.
          Keep sends this row to <code>train</code>; drop deletes it.</p>
        <div class="card" style="max-height:50vh;overflow:auto">
          ${raw(conv ? conversationHtml(conv, {}) : `<pre class="mono tiny" style="white-space:pre-wrap">${esc(r.rendered || JSON.stringify(r.row, null, 2))}</pre>`)}
        </div>
        <div class="row" style="gap:8px;margin-top:10px">
          <button class="btn btn-primary" data-rq="keep">Keep → train</button>
          <button class="btn btn-danger" data-rq="drop">Drop</button>
          <button class="btn" data-rq="skip">Skip</button>
          <span class="muted tiny" style="margin-left:auto">k · d · s</span>
        </div>`;
      body.dataset.rowId = r.row._id || "";
    };
    const act = async (what) => {
      const rid = body.dataset.rowId;
      if (what === "skip" || !rid) return next();
      try {
        if (what === "keep") await api.editRows(id, { move: { ids: [rid], to: "train" } });
        else await api.editRows(id, { delete: [rid] });
        decided++;
      } catch (ex) { toast(ex.message, "err"); }
      await next();
    };
    on(dlg, "click", "[data-rq]", (_e, t) => act(t.dataset.rq));
    dlg.addEventListener("keydown", (e) => {
      const k = { k: "keep", d: "drop", s: "skip" }[e.key];
      if (k && !e.target.matches("input,textarea")) { e.preventDefault(); act(k); }
    });
    await next();
  }

  // ---- the file itself ---------------------------------------------------
  async function showRows(offset = 0, query = null) {
    const q = query ?? ($("#rowSearch", mount)?.value || "");
    const split = $("#rowSplit", mount)?.value || "";
    loading = true; draw();
    try { rows = await api.datasetRows(id, offset, PAGE, q, split); }
    catch (ex) { toast(ex.message, "err"); }
    loading = false; draw();
  }

  async function reload(offset = rows?.offset || 0) {
    d = await api.dataset(id);
    api.datasetRuns(id).then((r) => { trainedOn = r; draw(); }).catch(() => {});
    previewCache.clear();
    await showRows(offset);
  }

  // ---- the pipeline ------------------------------------------------------
  /** The preview at step `k`, fetched once per (steps, sample, k). */
  async function fetchPreview(k, ops = pipeline(), limit = 50) {
    const key = JSON.stringify([ops, sample, k, limit]);
    if (previewCache.has(key)) return previewCache.get(key);
    const p = api.previewTransform(id, { ops, sample: sample || "all", limit, upto: k });
    previewCache.set(key, p);
    try { return await p; } catch (ex) { previewCache.delete(key); throw ex; }
  }

  async function refreshPreview() {
    if (!steps.length) { preview = null; stages = null; previewErr = null; return; }
    const k = inSource() ? steps.length - 1 : selected;
    const mine = ++gen;
    loading = true; previewErr = null; draw();
    try {
      const r = await fetchPreview(k);
      // Somebody moved on while this was in flight; theirs will arrive too.
      if (mine !== gen) return;
      stages = r.stages;
      preview = inSource() ? null : r;
    } catch (ex) {
      if (mine !== gen) return;
      previewErr = ex.message;
      preview = null;
    }
    loading = false; draw();
  }

  /** The step list changed: forget what was known about the old one. */
  function reset(select) {
    selected = Math.max(-1, Math.min(select, steps.length - 1));
    picked = new Set();
    preview = null; stages = null; previewErr = null;
    previewCache.clear();
    gen++;
    saveDraft(id, steps);
    draw();
    refreshPreview();
  }

  function commit(next, select = null) {
    history.push(steps);
    if (history.length > 100) history.shift();
    future = [];
    steps = next;
    reset(select === null ? steps.length - 1 : select);
  }

  function undo() {
    if (!history.length) return;
    future.push(steps);
    steps = history.pop();
    reset(steps.length - 1);
  }
  function redo() {
    if (!future.length) return;
    history.push(steps);
    steps = future.pop();
    reset(steps.length - 1);
  }

  /** The columns as they stand after step `k` (-1: the file's own). */
  function columnsAt(k) {
    if (k >= 0 && stages?.[k]) return stages[k].columns || [];
    if (rows) return columnsOf(d, rows);
    return (d.columns || []).filter((c) => c !== "split");
  }

  /** Add a step after the selected one, or replace the one at `at`. */
  function putStep(step, at = null) {
    const next = steps.slice();
    if (at === null) { next.splice(selected + 1, 0, step); commit(next, selected + 1); }
    else { next[at] = step; commit(next, at); }
  }

  /** Open a step's form, with its own live preview, then add or replace it. */
  function openStep(type, { at = null, ops = null } = {}) {
    const def = STEPS[type];
    const existing = at !== null ? steps[at] : null;
    const initial = ops || existing?.ops || {};
    // The rows a new step sees are the rows after the step before it.
    const before = at !== null ? at - 1 : selected;
    const columns = columnsAt(before);
    // What the form may ask of the page: the columns as they stand here, and
    // what any one of them holds -- counted on the rows after the step before.
    const ctx = {
      columns,
      values: async (column) => {
        const r = await api.previewTransform(id, {
          ops: pipeline().slice(0, before + 1), sample: sample || "all",
          limit: 1, upto: before, values: column });
        return r.values || { values: [], distinct: 0, rows: 0 };
      },
    };

    if (def.instant && at === null && !ops) {
      putStep({ type, ops: def.read(null) });
      return;
    }

    const dlg = modal({
      title: existing ? `Edit: ${def.label}` : def.label, width: 640,
      body: html`
        <p class="muted tiny" style="margin:0 0 12px">${def.blurb}</p>
        <form id="stepForm">${raw(def.form(initial, ctx))}</form>
        <div id="stepLive" class="step-live"></div>
        <div class="row" style="gap:6px;margin-top:12px">
          <button class="btn-sm btn-primary" id="stepOk">${existing ? "Save step" : "Add step"}</button>
          <button class="btn-sm" data-modal-close>Cancel</button>
          <span class="muted tiny" id="stepErr"></span>
        </div>` });

    def.wire?.(dlg, ctx);

    // Chips insert a placeholder at the caret of the template box.
    on(dlg, "click", "[data-insert]", (_e, t) => {
      const ta = $("#sf_tpl", dlg);
      if (!ta) return;
      const s = ta.selectionStart ?? ta.value.length, e = ta.selectionEnd ?? s;
      ta.value = ta.value.slice(0, s) + t.dataset.insert + ta.value.slice(e);
      ta.focus();
      ta.selectionStart = ta.selectionEnd = s + t.dataset.insert.length;
      live();
    });

    // What this step would do, recomputed as the form is typed into. The
    // pipeline up to here plus the candidate, on the same sample.
    let seq = 0;
    const live = debounce(async () => {
      const box = $("#stepLive", dlg);
      if (!box) return;
      let candidate;
      try { candidate = def.read(dlg); }
      catch { box.innerHTML = ""; return; }
      const upto = before + 1;
      const ops = pipeline().slice(0, upto).concat([candidate]);
      const mine = ++seq;
      box.innerHTML = `<div class="muted tiny">Trying it…</div>`;
      try {
        const r = await fetchPreview(upto, ops, 4);
        if (mine !== seq) return;
        box.innerHTML = livePanel(r, upto);
      } catch (ex) {
        if (mine !== seq) return;
        box.innerHTML = `<div class="callout callout-err" style="margin:0">${esc(ex.message)}</div>`;
      }
    }, 350);
    dlg.addEventListener("input", live);
    dlg.addEventListener("change", live);
    live();

    on(dlg, "submit", "#stepForm", (e) => { e.preventDefault(); $("#stepOk", dlg).click(); });
    on(dlg, "click", "#stepOk", () => {
      try {
        const next = { type, ops: def.read(dlg) };
        dlg.close();
        putStep(next, at);
      } catch (ex) {
        $("#stepErr", dlg).textContent = ex.message;
      }
    });
  }

  async function applySteps() {
    if (!steps.length) return toast("There are no steps to apply yet.", "err");
    const stage = stages?.[steps.length - 1];
    const dlg = modal({ title: "Apply as a new dataset", width: 520, body: html`
      <p class="muted tiny">Runs these ${steps.length} step${steps.length === 1 ? "" : "s"}
        over the whole file and writes the result as a new dataset. This one is
        not changed.</p>
      ${raw(stage ? `<p class="tiny">On the preview sample: <strong>${fmtNum(stage.kept)}</strong>
        rows remain of ${fmtNum(stages ? (stages[0].kept + stages[0].removed) : 0)}.</p>` : "")}
      <ol class="tiny" style="padding-left:18px;margin:0 0 12px">
        ${raw(steps.map((s) => `<li>${esc(describe(s))}</li>`).join(""))}</ol>
      <div class="field"><label for="applyName">Call the result</label>
        <input id="applyName" value="${d.name} (cleaned)" maxlength="120"></div>
      <div class="row" style="gap:6px">
        <button class="btn-sm btn-primary" id="applyGo">Apply</button>
        <button class="btn-sm" data-modal-close>Cancel</button>
      </div>
      <div id="applyErr"></div>` });
    on(dlg, "click", "#applyGo", async (_e, btn) => {
      btn.disabled = true; btn.textContent = "Working…";
      try {
        const made = await api.transformDataset(id, { ops: pipeline(),
                                                      name: $("#applyName", dlg).value });
        sessionStorage.removeItem(draftKey(id));
        dlg.close();
        toast(`Made "${made.name}" with ${fmtNum(made.rows)} rows.`, "ok");
        location.hash = `#/data/${made.id}`;
      } catch (ex) {
        btn.disabled = false; btn.textContent = "Apply";
        $("#applyErr", dlg).innerHTML = `<div class="callout callout-err" style="margin-top:10px">${
          esc(ex.message)}</div>`;
      }
    });
  }

  // ---- wiring ------------------------------------------------------------
  function wire() {
    wireShareBox(mount, "dataset", d, async () => { d = await api.dataset(id); draw(); });

    wireRibbon(mount, (key) => { tab = key; tabs.set(key); draw(); });
    on(mount, "click", "#toggleLeft", () => {
      leftOpen = !leftOpen; localStorage.setItem("aistudio.pqLeft", leftOpen ? "1" : "0"); draw();
    });
    on(mount, "input", "#qFilter", (_e, t) => {
      const f = t.value.toLowerCase();
      $$("#qList .q-item", mount).forEach((el) =>
        { el.hidden = !!f && !el.dataset.name.includes(f); });
    });

    // ---- steps -----------------------------------------------------------
    on(mount, "click", "[data-add]", (_e, t) => openStep(t.dataset.add));
    // The buttons on a step line sit inside the line, and a click on one is
    // not a click on the line.
    on(mount, "click", "[data-select]", (e, t) => {
      if (e.target.closest(".step-b")) return;
      selected = +t.dataset.select;
      picked = new Set();
      preview = null;
      gen++;
      draw();
      if (inSource()) { if (!rows) showRows(0); } else refreshPreview();
    });
    on(mount, "click", "[data-edit]", (_e, t) => {
      const i = +t.dataset.edit;
      openStep(steps[i].type, { at: i });
    });
    on(mount, "click", "[data-remove]", (_e, t) => {
      const i = +t.dataset.remove;
      // Stay on the same step if it is still there; otherwise on the one before.
      const next = selected > i ? selected - 1 : selected === i ? i - 1 : selected;
      commit(steps.filter((_, j) => j !== i), next);
    });
    on(mount, "click", "[data-move]", (_e, t) => {
      const [i, dir] = t.dataset.move.split(":").map(Number);
      const j = i + dir;
      if (j < 0 || j >= steps.length) return;
      const next = steps.slice();
      [next[i], next[j]] = [next[j], next[i]];
      commit(next, j);
    });
    on(mount, "click", "#undo", undo);
    on(mount, "click", "#redo", redo);
    on(mount, "click", "#applySteps", applySteps);
    on(mount, "click", "#discardSteps", async () => {
      if (!steps.length) return;
      if (!await confirmDestructive({
        title: "Remove every step?",
        body: "The file is untouched either way — nothing has been applied yet.",
        confirmLabel: "Remove them", tone: "danger" })) return;
      commit([], -1);
      if (!rows) showRows(0);
    });
    on(mount, "click", "#reopenRecipe", () => {
      const ops = d.recipe?.ops;
      const list = Array.isArray(ops) ? ops : ops && typeof ops === "object" ? [ops] : [];
      if (!list.length || !d.parent_id) return;
      saveDraft(d.parent_id, list.map(stepFrom));
      location.hash = `#/data/${d.parent_id}`;
    });

    // The formula bar: the selected step's options, as they will be sent.
    // Enter replaces the step with whatever was typed, if it parses.
    on(mount, "keydown", "#fx", (e, t) => {
      if (e.key !== "Enter" || inSource()) return;
      e.preventDefault();
      try {
        const ops = JSON.parse(t.value);
        if (!ops || typeof ops !== "object" || Array.isArray(ops)) throw new Error("not an object");
        putStep(stepFrom(ops), selected);
      } catch (ex) { toast(`That is not a valid step: ${ex.message}`, "err"); }
    });

    on(mount, "change", "#previewSample", (_e, t) => {
      sample = +t.value;
      localStorage.setItem("aistudio.previewSample", String(sample));
      previewCache.clear();
      refreshPreview();
    });

    // ---- column menu -----------------------------------------------------
    on(mount, "click", "[data-colmenu]", (e, t) => {
      e.stopPropagation();
      const r = t.getBoundingClientRect();
      colMenu = { column: t.dataset.colmenu, x: r.left, y: r.bottom + 2 };
      draw();
    });
    on(mount, "click", "[data-colact]", (_e, t) => {
      const [act, col] = [t.dataset.colact, colMenu?.column];
      colMenu = null; draw();
      if (!col) return;
      if (act === "rename") openStep("rename", { ops: { rename: { [col]: "" } } });
      if (act === "filter") openStep("where", { ops: { where: [{ column: col, op: "in", values: [] }] } });
      if (act === "remove") putStep({ type: "drop_columns", ops: { drop_columns: [col] } });
      if (act === "keep") putStep({ type: "keep_columns", ops: { keep_columns: [col] } });
      if (act === "split") openStep("split_column", { ops: { split_columns: [{ from: col, into: [], by: null }] } });
      if (act === "calc") openStep("calc", { ops: { columns: [{ name: "text", template: `{${col}}` }] } });
    });

    // ---- the file: search, page, select, edit -----------------------------
    on(mount, "click", "#loadRows", () => showRows(0));
    on(mount, "click", "[data-view]", (_e, t) => { view = t.dataset.view; draw(); });
    on(mount, "change", "#rowSplit", () => showRows(0));
    on(mount, "click", "[data-page]", (_e, t) => showRows(+t.dataset.page));
    on(mount, "keydown", "#rowSearch", (e) => {
      if (e.key === "Enter") { e.preventDefault(); showRows(0); }
    });

    on(mount, "change", "[data-row]", (_e, t) => {
      const h = t.dataset.row;
      if (t.checked) picked.add(h); else picked.delete(h);
      draw();
    });
    on(mount, "change", "#selectAll", (_e, t) => {
      (rows?.rows || []).forEach((r) => {
        const h = String(handleOf(r));
        if (t.checked) picked.add(h); else picked.delete(h);
      });
      draw();
    });
    on(mount, "click", "#clearPick", () => { picked = new Set(); draw(); });

    on(mount, "click", "#deleteRows", async () => {
      const n = picked.size;
      if (!await confirmDestructive({
        title: `Delete ${n} row${n === 1 ? "" : "s"}?`,
        body: `From "${d.name}".`,
        consequences: ["This changes the file itself — it is not a step, and Undo does not reach it.",
                       "Anything already trained on this data is unaffected."],
        confirmLabel: `Delete ${n} row${n === 1 ? "" : "s"}`,
      })) return;
      try {
        const r = await api.editRows(id, { delete: [...picked] });
        picked = new Set();
        toast(r.changed || "Done.", "ok");
        await reload();
      } catch (ex) { toast(ex.message, "err"); }
    });
    on(mount, "change", "#moveTo", async (_e, t) => {
      const to = t.value;
      if (!to) return;
      try {
        const r = await api.editRows(id, { move: { ids: [...picked], to } });
        picked = new Set();
        toast(r.changed || "Done.", "ok");
        await reload();
      } catch (ex) { toast(ex.message, "err"); }
    });

    on(mount, "click", "[data-open-row]", (e, t) => {
      e.preventDefault();
      const h = t.dataset.openRow;
      const found = (rows?.rows || []).find((r) => String(handleOf(r)) === h);
      if (!found) return;
      const canEdit = d.access === "edit";
      const dlg = modal({ title: `Row ${found.index}`,
                          width: isConversation(found.row) ? 860 : 720,
                          body: rowEditor(found.row, found.rendered, canEdit) });
      on(dlg, "click", "#saveRow", async () => {
        const next = {};
        let bad = null;
        $$("[data-field]", dlg).forEach((el) => {
          const key = el.dataset.field;
          if (el.dataset.json) {
            try { next[key] = JSON.parse(el.value); } catch (ex) { bad = `${key}: ${ex.message}`; }
          } else next[key] = el.value;
        });
        if (bad) { $("#rowEditError", dlg).innerHTML = `<div class="callout callout-err">${esc(bad)}</div>`; return; }
        try {
          await api.editRows(id, { update: { [h]: next } });
          dlg.close();
          toast("Row saved.", "ok");
          await reload();
        } catch (ex) {
          $("#rowEditError", dlg).innerHTML = `<div class="callout callout-err">${esc(ex.message)}</div>`;
        }
      });
    });

    // A cell holding a conversation or a schema opens in full rather than
    // being truncated into meaninglessness.
    on(mount, "click", "td.clickable", (_e, t) => {
      const source = inSource() ? (rows?.rows || []) : (preview?.rows || []).map((row, n) => ({ index: n, row }));
      const found = source.find((r) => r.index === +t.dataset.cell);
      if (!found) return;
      const media = assetOf(found.row[t.dataset.col], rows?.assets);
      if (media) {
        modal({ title: `${media.filename} — row ${t.dataset.cell}`, width: 820,
                body: media.kind === "image"
                  ? `<img src="${esc(media.url)}" alt="${esc(media.filename)}"
                          style="max-width:100%;max-height:70vh;display:block;margin:0 auto">`
                  : media.kind === "audio"
                    ? `<audio controls src="${esc(media.url)}" style="width:100%"></audio>`
                    : `<p><a class="btn" href="${esc(media.url)}?download=1">Download ${esc(media.filename)}</a></p>` });
        return;
      }
      modal({ title: `${t.dataset.col} — row ${t.dataset.cell}`, width: 700,
              body: `<p class="mono tiny" style="white-space:pre-wrap;max-height:60vh;overflow:auto">${
                esc(JSON.stringify(found.row[t.dataset.col], null, 2))}</p>` });
    });

    on(mount, "click", "#addRowsBtn", () => {
      const field = (d.format || {}).text_field || "text";
      const dlg = modal({ title: "Add rows", width: 640, body: html`
        <div class="field">
          <label for="newRows">One row per line</label>
          <textarea id="newRows" rows="10" class="mono" placeholder='{"${field}": "an example"}'></textarea>
          <div class="hint">A JSON object per line, using this dataset's columns — or
            plain text, which becomes the <code>${field}</code> column.</div>
        </div>
        <div class="field"><label for="newRowsSplit">Into the split</label>
          <input id="newRowsSplit" class="mono" value="train"></div>
        <div class="row" style="gap:6px">
          <button class="btn-sm btn-primary" id="saveNewRows">Add them</button>
          <button class="btn-sm" data-modal-close>Cancel</button>
        </div>
        <div id="addRowsError"></div>` });
      on(dlg, "click", "#saveNewRows", async () => {
        try {
          const r = await api.addRows(id, { rows: $("#newRows", dlg).value, split: $("#newRowsSplit", dlg).value });
          dlg.close();
          toast(`Added ${fmtNum(r.added || 0)} rows.`, "ok");
          await reload();
        } catch (ex) {
          $("#addRowsError", dlg).innerHTML = `<div class="callout callout-err">${esc(ex.message)}</div>`;
        }
      });
    });

    // ---- properties --------------------------------------------------------
    on(mount, "click", "#renameBtn", () => inlineRename($("#dsTitle", mount), async (name) => {
      d = await api.renameDataset(id, { name });
      const p = $("#propName", mount); if (p) p.value = d.name;
    }));
    on(mount, "change", "#propName", async (_e, t) => {
      try { d = await api.renameDataset(id, { name: t.value }); draw(); toast("Renamed.", "ok"); }
      catch (ex) { toast(ex.message, "err"); }
    });
    on(mount, "click", "#reviewQueue", () => reviewQueue());
    on(mount, "change", "#propTags", async (_e, t) => {
      try { d = await api.renameDataset(id, { tags: t.value }); t.value = (d.tags || []).join(", "); toast("Tagged.", "ok"); }
      catch (ex) { toast(ex.message, "err"); }
    });
    on(mount, "change", "#propNotes", async (_e, t) => {
      try { d = await api.renameDataset(id, { notes: t.value }); toast("Notes saved.", "ok"); }
      catch (ex) { toast(ex.message, "err"); }
    });

    // ---- the training tab --------------------------------------------------
    on(mount, "change", "#trainSplit", (_e, t) => { trainSplit = t.value; draw(); });
    on(mount, "click", "#recheckTraining", () => {
      training.key = null;          // force the next ensure() to fetch again
      convReport.key = null;
      draw();
    });
    on(mount, "click", "#measureLengths", async () => {
      try { stats = await api.datasetInspect(id); draw(); }
      catch (ex) { toast(ex.message, "err"); }
    });
    on(mount, "change", "#ctxLen", (_e, t) => {
      contextLen = Math.max(128, +t.value || 2048);
      localStorage.setItem("aistudio.ctxLen", String(contextLen));
      draw();
    });

    on(mount, "click", "#editMapping", () => {
      const cols = (d.columns || []).filter((c) => c !== "split");
      const fmt = d.format || {};
      const pick = (id_, label, value, hint) => html`
        <div class="field">
          <label for="${id_}">${label}</label>
          <select id="${id_}">
            <option value="">— none —</option>
            ${raw(cols.map((c) => `<option value="${esc(c)}"${
              c === value ? " selected" : ""}>${esc(c)}</option>`).join(""))}
          </select>
          ${raw(hint ? `<div class="hint">${esc(hint)}</div>` : "")}
        </div>`;
      const dlg = modal({ title: "How to read these rows", width: 560, body: html`
        <p class="muted tiny">Saved on the dataset, so every run started from
          it reads the rows the same way. This was worked out inside the wizard
          and thrown away afterwards, which meant doing it again every time.</p>
        <div class="field">
          <label for="fmMode">Each row is</label>
          <select id="fmMode">
            ${raw([["auto", "Work it out from the columns"],
                   ["chat", "A conversation"],
                   ["instruction", "An instruction and a response"],
                   ["text", "One column of text"]].map(([v, l]) =>
              `<option value="${v}"${v === (fmt.mode || "auto") ? " selected" : ""}>${l}</option>`).join(""))}
          </select>
        </div>
        <div id="fmFields"></div>
        <div class="row" style="justify-content:flex-end;gap:8px;margin-top:12px">
          <button type="button" class="btn" data-modal-close>Cancel</button>
          <button type="button" class="btn btn-primary" id="fmSave">Save the mapping</button>
        </div>` });

      const fields = () => {
        const mode = $("#fmMode", dlg).value;
        $("#fmFields", dlg).innerHTML =
          mode === "chat" ? pick("fmMessages", "The conversation is in", fmt.messages_field,
                                 "A list of messages, each with a role and content.")
                            + pick("fmTools", "Tool definitions are in", fmt.tools_field, "")
          : mode === "instruction"
            ? pick("fmInstruction", "The instruction is in", fmt.instruction_field, "")
              + pick("fmInput", "Extra input is in", fmt.input_field,
                     "Optional. Appended to the instruction when a row has one.")
              + pick("fmResponse", "The answer is in", fmt.response_field, "")
          : mode === "text"
            ? pick("fmText", "The text is in", fmt.text_field,
                   "The whole row, trained on as-is.")
          : `<p class="muted tiny">The column names are matched against the shapes
               this app knows. Pick one of the others to say so yourself.</p>`;
      };
      fields();
      on(dlg, "change", "#fmMode", fields);
      on(dlg, "click", "#fmSave", async () => {
        const mode = $("#fmMode", dlg).value;
        const val = (sel) => $(sel, dlg)?.value || undefined;
        const next = mode === "auto" ? {} : { mode };
        if (mode === "chat") {
          next.messages_field = val("#fmMessages");
          next.tools_field = val("#fmTools");
        } else if (mode === "instruction") {
          next.instruction_field = val("#fmInstruction");
          next.input_field = val("#fmInput");
          next.response_field = val("#fmResponse");
        } else if (mode === "text") {
          next.text_field = val("#fmText");
        }
        // Said by a person rather than guessed, so nothing downstream should
        // hedge about it.
        if (mode !== "auto") next.confidence = "high";
        try {
          d = await api.renameDataset(id, { format: next });
          dlg.close();
          training.key = null;
          convReport.key = null;
          toast("Saved. Runs started from this dataset will read it this way.", "ok");
          draw();
        } catch (ex) { toast(ex.message, "err"); }
      });
    });

    // ---- home actions ------------------------------------------------------
    on(mount, "click", "#inspect", async () => {
      if (tab !== "check") { tab = "check"; tabs.set(tab); }
      stats = null;
      draw();
      try { stats = await api.datasetInspect(id, checkSample); }
      catch (ex) { toast(ex.message, "err"); }
      draw();
    });
    on(mount, "change", "#checkSample", (_e, t) => {
      checkSample = +t.value || 2000;
      localStorage.setItem("aistudio.checkSample", String(checkSample));
    });
    // Each reported finding carries the step that fixes it. Fixing adds the
    // step to the pipeline; nothing is written until Apply.
    on(mount, "click", "[data-fix]", (_e, t) => {
      const fix = t.dataset.fix;
      if (fix === "cap_length") {
        putStep({ type: "length", ops: { min_chars: 0, max_chars: 8000 } });
      } else if (STEPS[fix]) {
        putStep({ type: fix, ops: STEPS[fix].read(null) });
      } else { return; }
      toast("Added as a step. Apply when you are happy with the preview.", "ok",
            { href: `#/data/${id}`, label: "See it" });
    });

    on(mount, "click", "#holdBack", () => {
      const cols = (d.columns || []).filter((c) => c !== "split");
      const dlg = modal({ title: "Hold some back", width: 520, body: html`
        <p class="muted tiny">Makes one new dataset with a
          <strong>validation</strong> split in it. Without one there is no way
          to tell a model that has learned from one that has memorised — the
          training loss looks identical either way.</p>
        <form id="splitForm" style="margin-top:10px">
          <div class="row row-top">
            <div class="field" style="max-width:140px">
              <label for="sFrac">Hold back %</label>
              <input id="sFrac" name="fraction" type="number" value="10" min="1" max="50">
            </div>
            <div class="field" style="flex:1">
              <label for="sStrat">Keep the mix of</label>
              <select id="sStrat" name="stratify">
                <option value="">Nothing — cut at random</option>
                ${raw(cols.map((c) => `<option value="${esc(c)}">${esc(c)}</option>`).join(""))}
              </select>
              <div class="hint">Pick the column holding a label or a source, and
                each of its values is represented on both sides in proportion.
                A category held by 3% of the rows otherwise lands entirely on
                one side often enough to matter — and a held-out score that
                contains none of the rare case cannot see the thing most likely
                to be wrong.</div>
            </div>
          </div>
          <div class="row" style="justify-content:flex-end;gap:8px">
            <button type="button" class="btn" data-modal-close>Cancel</button>
            <button class="btn btn-primary" type="submit">Hold it back</button>
          </div>
        </form>` });
      on(dlg, "submit", "#splitForm", async (e) => {
        e.preventDefault();
        const f = Object.fromEntries(new FormData(e.target).entries());
        try {
          const made = await api.splitDataset(id, {
            fraction: +f.fraction / 100, stratify: f.stratify || "" });
          dlg.close();
          const s = made.splits || {};
          toast(`Made "${made.name}" — ${fmtNum(s.train || 0)} to train on, `
                + `${fmtNum(s.validation || 0)} held back.`, "ok");
          location.hash = `#/data/${made.id}`;
        } catch (ex) { toast(ex.message, "err"); }
      });
    });

    on(mount, "click", "#publishBtn", () => {
      const dlg = modal({ title: "Publish", width: 560,
        body: publishCard({ kind: "dataset", slug: slug(d.name),
                            blurb: "Uploads the rows and a dataset card to your own account." }) });
      wirePublish(dlg, "dataset", (body) => api.publishDataset(id, body));
    });

    on(mount, "click", "#useForTraining", () => {
      // A link, not a stashed value: it survives a reload, it can be sent to
      // somebody, and it is what the wizard reads on the way in.
      location.hash = `#/new?dataset=${encodeURIComponent(d.id)}`;
    });

    on(mount, "click", "#deleteDs", async () => {
      if (!await confirmDestructive({
        title: `Delete "${d.name}"?`,
        consequences: [
          `All ${fmtNum(d.rows)} rows are removed from this studio permanently.`,
          "Datasets made from this one are kept, and keep working.",
          "Runs that trained on it keep their models and their history.",
        ],
        confirmLabel: "Delete the dataset",
        confirmWord: d.rows > 5000 ? d.name : null })) return;
      try {
        const r = await api.deleteDataset(id);
        sessionStorage.removeItem(draftKey(id));
        toast(r.note || "Deleted.", "ok");
        location.hash = "#/data";
      } catch (ex) { toast(ex.message, "err"); }
    });
  }

  // Clicks anywhere else close the column menu; Ctrl+Z and Ctrl+Shift+Z work
  // when the focus is not in a box that has its own idea of undo.
  const onDocClick = () => { if (colMenu) { colMenu = null; draw(); } };
  const onKey = (e) => {
    const tag = (e.target?.tagName || "").toLowerCase();
    if (["input", "textarea", "select"].includes(tag) || e.target?.isContentEditable) return;
    if (!(e.ctrlKey || e.metaKey) || e.key.toLowerCase() !== "z") return;
    e.preventDefault();
    if (e.shiftKey) redo(); else undo();
  };
  document.addEventListener("click", onDocClick);
  document.addEventListener("keydown", onKey);

  // ---- go ----------------------------------------------------------------
  draw();
  api.datasets().then((l) => { library = l; draw(); }).catch(() => {});
  if (inSource()) showRows(0); else { showRows(0); refreshPreview(); }

  return () => {
    mount.classList.remove("pq-wide");
    document.removeEventListener("click", onDocClick);
    document.removeEventListener("keydown", onKey);
  };
}

// ---------------------------------------------------------------------------
// The draft: steps not yet applied survive a reload and a trip elsewhere.

function loadDraft(id) {
  try {
    const raw = sessionStorage.getItem(draftKey(id));
    const list = raw ? JSON.parse(raw) : [];
    return Array.isArray(list) ? list.filter((s) => s && s.ops).map((s) =>
      STEPS[s.type] ? s : stepFrom(s.ops)) : [];
  } catch { return []; }
}
function saveDraft(id, steps) {
  try {
    if (steps.length) sessionStorage.setItem(draftKey(id), JSON.stringify(steps));
    else sessionStorage.removeItem(draftKey(id));
  } catch { /* storage full or blocked; the steps are still on screen */ }
}

// ---------------------------------------------------------------------------
// Layout

function layout(s) {
  const { d, steps, selected, leftOpen, tab } = s;
  if (tab === "train" || tab === "check") {
    return html`
      ${raw(header(d))}
      ${raw(ribbonFor(s))}
      ${raw(tab === "train" ? trainPanel(s) : checkPanel(s))}`;
  }
  return html`
    ${raw(header(d))}
    ${raw(ribbonFor(s))}
    <div class="pq ${leftOpen ? "" : "no-left"}">
      ${raw(leftOpen ? leftPane(s) : "")}
      <div class="pq-pane pq-mid">
        ${raw(fxBar(s))}
        ${raw(bulkBar(s))}
        ${raw(grid(s))}
        ${raw(statusBar(s))}
      </div>
      ${raw(rightPane(s))}
    </div>
    ${raw(s.colMenu ? colMenuHtml(s.colMenu) : "")}`;
}

function header(d) {
  document.title = `${d.name} · Datasets · AI Studio`;
  return html`
    <div class="pq-head">
      ${raw(breadcrumb({ href: "#/data", label: "Datasets" }))}
      <div class="row" style="gap:8px;min-width:0;flex-wrap:wrap">
        <div class="title-row row" style="gap:4px;min-width:0">
          <h1 id="dsTitle" style="margin:0;font-size:1.3rem">${d.name}</h1>
          ${raw(d.access === "edit" ? `<button class="btn-quiet" id="renameBtn" title="Rename">✎</button>` : "")}
        </div>
        <span class="muted tiny">
          ${raw(qualityBadge(d))}
          ${fmtNum(d.rows)} rows${raw(splitBadges(d))} ·
          ${(d.bytes / 1048576).toFixed(1)} MB ·
          ${d.owner_name || "unowned"} · updated ${fmtAgo(d.updated_at)}
          ${raw(d.access === "view" ? ` · <span class="badge">shared with you — read only</span>` : "")}
        </span>
      </div>
    </div>`;
}

// ---- the ribbon ------------------------------------------------------------

function ribbonFor(s) {
  const { d, tab, steps, history, future, view, sample, leftOpen } = s;
  const canEdit = d.access === "edit";
  let body = "";
  if (tab === "home") {
    body = group("Apply", [
      rb("applySteps", "✔", "Apply as new dataset", { cls: "primary", disabled: !steps.length,
        title: "Run every step over the whole file and save the result" }),
      rb("discardSteps", "✕", "Discard steps", { disabled: !steps.length }),
      rb("undo", "↶", "Undo", { disabled: !history.length, title: "Ctrl+Z" }),
      rb("redo", "↷", "Redo", { disabled: !future.length, title: "Ctrl+Shift+Z" }),
    ]) + group("Source", [
      rb("inspect", "◉", "Check the data", { title: "Empty rows, repeats, lengths — with a fix for each" }),
      canEdit ? rb("addRowsBtn", "＋", "Add rows") : "",
      rb("holdBack", "◫", "Hold back a split"),
      // Rows a model or a person put here to be looked at -- a kept exchange
      // from the playground, a generated set -- one at a time, with a keep
      // and a drop. The workbench could already move and delete rows; what
      // it lacked was a way to go through them without picking each one.
      (d.splits || {}).review ? rb("reviewQueue", "☑", `Review ${fmtNum(d.splits.review)}`, {
        cls: "primary", disabled: !canEdit,
        title: "Go through the rows waiting in the review split" }) : "",
    ]) + group("Use", [
      rb("useForTraining", "✦", "Train on this", { cls: "primary" }),
      // The other thing to do with a set of questions: have a model answer
      // them. Reachable only from the generator until now, where you had to
      // remember the dataset's name.
      rb(null, "▷", "Answer it with a model", { href: "#/generate",
        title: "Write a new dataset of a model's answers to these rows" }),
      rb(null, "↓", "Download", { href: `/api/datasets/${esc(d.id)}/dataset-file`,
                                  title: "The whole file, as JSONL" }),
      rb("publishBtn", "☁", "Publish"),
      d.mine ? rb("deleteDs", "🗑", "Delete", { cls: "danger" }) : "",
    ]);
  } else if (tab === "view") {
    body = group("Rows", [
      rbSeg([{ label: "Table", on: view === "table", data: `data-view="table"` },
             { label: "As trained", on: view === "text", data: `data-view="text"` }]),
    ]) + group("Preview on", [
      rbSelect("previewSample", {
        title: "How many rows the preview rehearses on", value: sample,
        options: SAMPLES.map((n) =>
          [n, n ? `first ${fmtNum(n)} rows` : "the whole file"]) }),
    ]) + group("Panes", [
      rb("toggleLeft", "▤", leftOpen ? "Hide library" : "Show library"),
    ]);
  } else if (tab === "check") {
    body = group("The data", [
      rb("inspect", "◉", s.stats ? "Check it again" : "Check the data",
        { cls: s.stats ? "" : "primary",
          title: "Empty rows, repeats, near-copies, lengths, leakage, secrets" }),
      rbSelect("checkSample", { title: "How many rows to read", value: s.checkSample,
        options: [[500, "the first 500 rows"], [2000, "the first 2,000"],
                  [10000, "the first 10,000"]] }),
    ]) + group("Act on it", [
      rb("applySteps", "✔", "Apply the fixes", { cls: "primary", disabled: !steps.length,
        title: "Every fix you accepted becomes a step; this writes them" }),
      rb("discardSteps", "✕", "Discard them", { disabled: !steps.length }),
    ]);
  } else if (tab === "train") {
    body = group("Check it", [
      rb("recheckTraining", "↻", "Read it again",
        { title: "Render these rows the way the trainer will" }),
      rbSelect("trainSplit", { title: "Which split to look at", value: s.trainSplit,
        options: [["", "Every split"]].concat(
          Object.keys(d.splits || {}).map((n) => [n, n])) }),
    ]) + group("How to read it", [
      rb("editMapping", "⚙", "Change the mapping",
        { disabled: d.access !== "edit",
          title: "Which columns hold the question, the answer, the conversation" }),
    ]) + group("Then", [
      rb("useForTraining", "✦", "Train on this", { cls: "primary" }),
      rb("holdBack", "◫", "Hold back a split"),
      // Rows a model or a person put here to be looked at -- a kept exchange
      // from the playground, a generated set -- one at a time, with a keep
      // and a drop. The workbench could already move and delete rows; what
      // it lacked was a way to go through them without picking each one.
      (d.splits || {}).review ? rb("reviewQueue", "☑", `Review ${fmtNum(d.splits.review)}`, {
        cls: "primary", disabled: !canEdit,
        title: "Go through the rows waiting in the review split" }) : "",
    ]);
  } else {
    const items = stepsOnTab(tab).map((def) =>
      rb(null, def.icon, def.label, { data: `data-add="${def.key}"`, title: def.blurb }));
    const label = tab === "transform" ? "Columns and shape" : tab === "columns" ? "New column" : "Filter, order, size";
    body = group(label, items);
  }
  return ribbon({ tabs: TABS, active: tab, body,
                  right: shareButton("dataset", d) });
}

// ---- the check tab -----------------------------------------------------------
//
// The quality report was a modal: you read it, closed it, and every number in
// it was gone. Nothing on the library row or the page header said a dataset
// was a third duplicates. It lives on a tab now, and each finding still
// carries the step that fixes it.

function checkPanel(s) {
  const { stats, checkSample } = s;
  if (!stats) {
    return html`
      <div class="card empty">
        <div class="big" aria-hidden="true">◉</div>
        <h3>Look at the data before you train on it</h3>
        <p class="muted">Empty rows, exact repeats, rows that say the same thing
          in nearly the same words, the spread of lengths, whether the held-out
          split has leaked into the training data, and whether there are
          secrets in here.</p>
        <p><button class="btn btn-primary" id="inspect">Check ${
          fmtNum(checkSample)} rows</button></p>
      </div>`;
  }
  const worst = { error: 0, warn: 1, info: 2, ok: 3 };
  const problems = [...(stats.problems || [])]
    .sort((a, b) => worst[a.level] - worst[b.level]);
  const leak = stats.leakage;
  const near = stats.near_duplicates || {};

  return html`
    <div class="grid grid-2" style="align-items:start;gap:14px">
      <div>
        <div class="card" style="margin-bottom:14px">
          <div class="row-between" style="align-items:baseline;margin-bottom:10px">
            <h3 style="margin:0">What is worth acting on</h3>
            <span class="muted tiny">${fmtNum(stats.sampled)} of ${
              fmtNum(stats.total_rows)} rows read</span>
          </div>
          ${raw(problems.map((p) => html`
            <div class="callout ${p.level === "error" ? "callout-err"
              : p.level === "warn" ? "callout-warn"
              : p.level === "ok" ? "callout-ok" : ""}" style="margin-bottom:8px">
              ${p.message}
              ${raw(p.fix ? `<div style="margin-top:6px"><button class="btn-sm btn-primary"
                data-fix="${esc(p.fix)}">Fix it — adds a step</button></div>` : "")}
            </div>`).join(""))}
        </div>

        ${raw(leak ? html`
          <div class="card" style="margin-bottom:14px">
            <h3 style="margin:0 0 6px">Has the held-out split leaked?</h3>
            <p class="muted tiny">Rows in the <strong>${leak.split}</strong>
              split that also appear in the training data. This is the statistic
              that decides whether a held-out score means anything at all: a
              validation set copied from the training set produces a beautiful
              number and measures memory.</p>
            <div class="callout ${leak.leaked ? "callout-err" : "callout-ok"}"
                 style="margin-top:8px">
              <strong>${leak.leaked
                ? `${leak.leaked} of ${leak.checked} checked rows have leaked`
                : `None of the ${leak.checked} rows checked has leaked`}</strong>
              ${raw(leak.leaked
                ? "Hold the split back again from the deduplicated data."
                : "The held-out score from this dataset can be believed.")}
            </div>
            ${raw((leak.examples || []).length ? html`
              <details class="adv" style="margin-top:8px">
                <summary>What leaked</summary>
                ${raw(leak.examples.map((t) =>
                  `<p class="mono tiny" style="white-space:pre-wrap">${esc(t)}…</p>`).join(""))}
              </details>` : "")}
          </div>` : "")}

        ${raw(near.pairs ? html`
          <div class="card" style="margin-bottom:14px">
            <h3 style="margin:0 0 6px">Rows that say the same thing</h3>
            <p class="muted tiny">${fmtNum(near.rows)} rows are involved${
              near.pairs > near.rows ? ` (in ${fmtNum(near.pairs)} pairs — a group
              of thirty alike rows is four hundred pairs, so the row count is the
              one to read)` : ""}. Not byte-identical: these differ by a word or a
              comma, which is what a model writes when it is asked the same thing
              thirty times.</p>
            ${raw((near.examples || []).map(([a, b]) => html`
              <details class="adv" style="margin-top:6px">
                <summary>A pair</summary>
                <p class="mono tiny" style="white-space:pre-wrap">${a}…</p>
                <p class="mono tiny" style="white-space:pre-wrap;border-top:1px solid var(--border);padding-top:6px">${b}…</p>
              </details>`).join(""))}
          </div>` : "")}
      </div>

      <div>
        <div class="card" style="margin-bottom:14px">
          <h3 style="margin:0 0 8px">Lengths</h3>
          <dl class="kv">
            <dt>Typical</dt><dd>${fmtNum(stats.chars.p50)} characters</dd>
            <dt>Long ones</dt><dd>${fmtNum(stats.chars.p90)} at the 90th percentile</dd>
            <dt>Longest</dt><dd>${fmtNum(stats.chars.max)}</dd>
            <dt>Unique rows</dt><dd>${fmtNum(stats.unique_rows)} of ${fmtNum(stats.sampled)}</dd>
          </dl>
          <div style="margin-top:12px">
            <div class="muted tiny" style="margin-bottom:6px">Distribution
              (characters, log scale)</div>
            <div class="row" style="gap:2px;align-items:flex-end;height:60px">${
              raw(bars(stats.histogram))}</div>
          </div>
        </div>

        <div class="card">
          <h3 style="margin:0 0 8px">Columns</h3>
          <div class="table-wrap"><table class="table"><thead><tr>
            <th>Name</th><th>Holds</th><th>Filled</th><th class="hide-sm">Typical</th>
          </tr></thead><tbody>
            ${raw((stats.column_types || stats.columns).map((c) => {
              const fill = (stats.columns.find((x) => x.name === c.name) || {}).fill_rate;
              return html`
                <tr>
                  <td class="mono tiny">${c.name}</td>
                  <td class="tiny">${c.type || "—"}${raw(c.mixed
                    ? ` <span class="badge badge-warn">mixed</span>` : "")}${raw(
                    c.categorical ? ` <span class="badge">${c.distinct} values</span>` : "")}</td>
                  <td class="tiny muted">${Math.round((fill ?? 0) * 100)}%</td>
                  <td class="tiny muted hide-sm">${c.median_chars
                    ? fmtNum(c.median_chars) + " chars" : "—"}</td>
                </tr>`;
            }).join(""))}
          </tbody></table></div>
        </div>
        ${raw((stats.media || []).map(mediaCard).join(""))}
      </div>
    </div>`;
}

/** What the stored files in a column actually are.
 *
 *  Read off the file headers, not decoded -- see common/media.py. The three
 *  things worth knowing before a run: files that have gone, files whose bytes
 *  are not what their name says, and the spread of sizes. */
function mediaCard(m) {
  const img = m.images;
  const aud = m.audio;
  const kinds = Object.entries(m.kinds || {}).map(([k, n]) => `${fmtNum(n)} ${k}`).join(", ");
  return html`
    <div class="card">
      <h3 style="margin:0 0 8px">Stored files in <span class="mono">${m.column}</span></h3>
      <p class="muted tiny">${fmtNum(m.referenced)} rows point at ${fmtNum(m.files)}
        distinct file${m.files === 1 ? "" : "s"}${kinds ? ` (${kinds})` : ""}.
        ${raw(m.referenced > m.files
          ? `Rows that share a picture share the file — it is stored once.` : "")}</p>
      ${raw((m.missing_rows || m.missing_files) ? html`
        <div class="callout callout-err">
          <strong>${fmtNum(m.missing_rows + m.missing_files)} cannot be read</strong>
          ${m.missing_files ? `${fmtNum(m.missing_files)} gone from disk. ` : ""}${
            m.missing_rows ? `${fmtNum(m.missing_rows)} whose record is gone.` : ""}
        </div>` : "")}
      ${raw(m.mismatched ? html`
        <div class="callout callout-warn">
          <strong>${fmtNum(m.mismatched)} ${m.mismatched === 1 ? "is" : "are"} not what
            ${m.mismatched === 1 ? "its" : "their"} name says</strong>
          ${raw((m.examples || []).map((e) => `<div class="tiny mono">${esc(e)}</div>`).join(""))}
        </div>` : "")}
      ${raw(img ? html`
        <table class="table" style="margin-top:8px"><tbody>
          <tr><td class="muted tiny">Width</td><td class="tiny mono">${img.width.min} · ${img.width.median} · ${img.width.max} px</td></tr>
          <tr><td class="muted tiny">Height</td><td class="tiny mono">${img.height.min} · ${img.height.median} · ${img.height.max} px</td></tr>
          ${raw(img.tiny ? `<tr><td class="muted tiny">Under 64 px</td><td class="tiny">${fmtNum(img.tiny)} — nothing learns from a thumbnail</td></tr>` : "")}
          ${raw(img.huge ? `<tr><td class="muted tiny">Over 2048 px</td><td class="tiny">${fmtNum(img.huge)} — resized on the way in, slowly</td></tr>` : "")}
        </tbody></table>
        <div class="hint">min · median · max, over ${fmtNum(img.measured)} measured</div>` : "")}
      ${raw(aud ? html`
        <table class="table" style="margin-top:8px"><tbody>
          <tr><td class="muted tiny">Length</td><td class="tiny mono">${aud.seconds.min}s · ${aud.seconds.median}s · ${aud.seconds.max}s</td></tr>
          <tr><td class="muted tiny">In total</td><td class="tiny">${fmtDuration ? fmtDuration(aud.seconds.total) : aud.seconds.total + "s"}</td></tr>
          ${raw(aud.long ? `<tr><td class="muted tiny">Over 30 s</td><td class="tiny">${fmtNum(aud.long)} — most speech models want clips under that</td></tr>` : "")}
        </tbody></table>
        <div class="hint">WAV only; other formats need decoding to measure</div>` : "")}
    </div>`;
}

// ---- the training tab --------------------------------------------------------
//
// The one question this page could not answer: what will the trainer actually
// read out of these rows, and can it? The renderer that answers it existed and
// was reachable only from step three of the wizard -- so the mapping worked out
// there was never written back here, and had to be worked out again on every
// future run. The preview also has no way back to the editor, so a problem it
// reveals is a problem you go and fix somewhere else.

function trainPanel(s) {
  const { d, training, convReport, trainSplit } = s;
  const t = training.data;
  const chat = (d.format || {}).mode === "chat";

  return html`
    <div class="grid grid-2" style="align-items:start;gap:14px">
      <div>
        ${raw(formatCard(d, t))}
        ${raw(lengthCard(s))}
        ${raw(chat ? conversationCard(convReport) : "")}
      </div>
      <div>${raw(renderedCard(training, trainSplit))}</div>
    </div>`;
}

/** How these rows are read, and how sure anybody is about that. */
function formatCard(d, t) {
  const fmt = d.format || {};
  const mode = fmt.mode || "auto";
  const conf = fmt.confidence || (t?.format || {}).confidence;
  const resolved = t?.resolved || {};
  const MODE_WORDS = {
    chat: "A conversation per row",
    instruction: "An instruction and a response per row",
    text: "One column of text per row",
    auto: "Not worked out yet",
  };
  const level = conf === "high" ? "ok" : conf === "low" ? "warn" : "";
  return html`
    <div class="card" style="margin-bottom:14px">
      <h3 style="margin:0 0 8px">How the trainer reads these rows</h3>
      <div class="callout ${level ? "callout-" + level : ""}">
        <strong>${MODE_WORDS[mode] || mode}</strong>
        ${raw(conf === "high"
          ? "Read from the column names, which match a shape this app knows."
          : conf === "low"
          ? "Guessed. Check the rendered text on the right before training on it — "
            + "a wrong guess trains the model on the wrong half of every row."
          : "Worked out from the rows themselves.")}
      </div>
      ${raw(Object.keys(resolved).length ? html`
        <dl class="kv" style="margin-top:10px">
          ${raw(Object.entries(resolved).map(([k, v]) => html`
            <dt>${k.replace(/_/g, " ")}</dt>
            <dd class="mono">${v == null || v === "" ? "—" : String(v)}</dd>`).join(""))}
        </dl>` : "")}
      ${raw((fmt.roles || []).length ? html`
        <p class="muted tiny" style="margin:8px 0 0">Roles in the data:
          ${(fmt.roles || []).join(", ")}${fmt.has_tool_calls ? " · with tool calls" : ""}${
          fmt.has_reasoning ? " · with reasoning" : ""}</p>` : "")}
    </div>`;
}

/** How long the rows are, and how much of them a context window would keep. */
function lengthCard(s) {
  const { stats, contextLen } = s;
  if (!stats) {
    return html`
      <div class="card" style="margin-bottom:14px">
        <h3 style="margin:0 0 6px">How long are these rows?</h3>
        <p class="muted tiny">A row longer than the context length is cut, and
          nothing says so at training time — the end of every long example is
          simply not learned.</p>
        <button class="btn-sm" id="measureLengths">Measure them</button>
      </div>`;
  }
  const chars = stats.chars || {};
  // Four characters to a token is the usual rule for English. Said out loud
  // rather than presented as a count, because it is wrong by a third on code
  // and much worse on languages that do not space their words.
  const est = (n) => Math.round((n || 0) / 4);
  return html`
    <div class="card" style="margin-bottom:14px">
      <h3 style="margin:0 0 8px">How long are these rows?</h3>
      <dl class="kv">
        <dt>Typical</dt><dd>${fmtNum(chars.p50 || 0)} characters
          <span class="muted tiny">≈ ${fmtNum(est(chars.p50))} tokens</span></dd>
        <dt>Long ones</dt><dd>${fmtNum(chars.p90 || 0)} characters
          <span class="muted tiny">≈ ${fmtNum(est(chars.p90))} tokens</span></dd>
        <dt>Longest seen</dt><dd>${fmtNum(chars.max || 0)} characters
          <span class="muted tiny">≈ ${fmtNum(est(chars.max))} tokens</span></dd>
      </dl>
      <div class="row row-top" style="margin-top:10px;align-items:flex-end">
        <div class="field" style="max-width:170px;margin:0">
          <label for="ctxLen">Against a context of</label>
          <input id="ctxLen" type="number" value="${contextLen}" min="128" step="128">
        </div>
      </div>
      ${raw(est(chars.p90) > contextLen ? html`
        <div class="callout callout-warn" style="margin-top:10px">
          <strong>More than a tenth of these rows will be cut</strong>
          At ${fmtNum(contextLen)} tokens, everything past the cut is not
          learned — and for an instruction dataset that is usually the answer.
          Either raise the context length on the run, or filter the long rows
          out on the Rows tab.</div>`
        : est(chars.max) > contextLen ? html`
        <div class="callout" style="margin-top:10px">
          The longest rows here will be cut at ${fmtNum(contextLen)} tokens.
          Typical ones fit comfortably.</div>`
        : html`
        <div class="callout callout-ok" style="margin-top:10px">
          Every row measured fits in ${fmtNum(contextLen)} tokens.</div>`)}
      <p class="muted tiny" style="margin:8px 0 0">Character counts are exact;
        the token figures are the usual four-characters-a-token rule and are
        wrong by a third on code. Measured on ${fmtNum(stats.sampled || 0)} rows.</p>
    </div>`;
}

/** Whether it reads as conversations, from the validator the conversion uses. */
function conversationCard(res) {
  if (res.status === "loading") {
    return `<div class="card muted tiny" aria-busy="true">Reading the conversations…</div>`;
  }
  if (res.status === "error") {
    return `<div class="callout callout-err">${esc(res.error)}</div>`;
  }
  const r = res.data;
  if (!r) return "";
  const problems = r.problems || [];
  return html`
    <div class="card" style="margin-bottom:14px">
      <h3 style="margin:0 0 8px">Do they read as conversations?</h3>
      <p class="muted tiny">${fmtNum(r.sampled || 0)} checked${
        r.canonical != null ? ` · ${r.canonical} of the first 200 already in the standard shape` : ""}.</p>
      ${raw(!problems.length
        ? `<div class="callout callout-ok" style="margin-top:8px">
             Nothing wrong found.</div>`
        : problems.map((p) => `<div class="callout callout-${
            p.level === "error" ? "err" : "warn"} tiny" style="margin-top:8px">${
            esc(p.message)}${p.rows ? ` <span class="muted">(${p.rows} rows)</span>` : ""}</div>`).join(""))}
    </div>`;
}

/** The exact text the model will be trained on. */
function renderedCard(res, split) {
  if (res.status === "idle" || res.status === "loading") {
    return html`<div class="card muted tiny" aria-busy="true">
      Rendering these rows the way the trainer will…
      <div class="sk-line shimmer" style="margin-top:10px;width:80%"></div>
      <div class="sk-line shimmer" style="margin-top:6px;width:60%"></div></div>`;
  }
  if (res.status === "error") {
    return `<div class="callout callout-err"><strong>Could not render these rows</strong>${esc(res.error)}</div>`;
  }
  const t = res.data || {};
  if (!t.available) {
    return `<div class="callout callout-warn"><strong>Nothing to render</strong>${
      esc(t.reason || "That split has no rows in it.")}</div>`;
  }
  const c = t.counts || {};
  return html`
    <div class="card">
      <div class="row-between" style="align-items:baseline;margin-bottom:8px">
        <h3 style="margin:0">What the model will read</h3>
        <span class="muted tiny">${split ? split + " split" : "every split"}</span>
      </div>
      <div class="row" style="gap:6px;flex-wrap:wrap;margin-bottom:10px">
        <span class="badge badge-ok">${c.ok || 0} readable</span>
        ${raw(c.empty ? `<span class="badge">${c.empty} blank</span>` : "")}
        ${raw(c.unreadable
          ? `<span class="badge badge-err">${c.unreadable} unreadable</span>` : "")}
        <span class="badge">via ${esc(t.template_source || "the builtin format")}</span>
      </div>
      ${raw(c.unreadable ? `<div class="callout callout-err" style="margin-bottom:10px">
        <strong>${c.unreadable} of ${c.sampled} rows have content the chosen columns
        cannot reach</strong> Those rows are skipped at training time. Change the
        mapping, or look at one on the table.</div>` : "")}
      ${raw(t.template_error ? `<div class="callout callout-err" style="margin-bottom:10px">
        <strong>The template did not render</strong>${esc(t.template_error)}</div>` : "")}
      ${raw((t.rendered || []).map((r) => html`
        <div class="sample ${r.status !== "ok" ? "bad" : ""}">
          <div class="row-between tiny muted" style="margin-bottom:4px">
            <span>${r.status === "ok" ? "row as the model sees it"
              : r.status === "empty" ? "blank in the source — skipped"
              : "has content the mapping cannot reach"}</span>
            <span>${fmtNum(r.length || 0)} characters</span>
          </div>
          <p class="txt mono tiny" style="white-space:pre-wrap;margin:0">${
            r.text || "(nothing)"}</p>
        </div>`).join(""))}
      ${raw((t.system_prompts || []).length ? html`
        <details class="adv" style="margin-top:10px">
          <summary>The system prompt in this data</summary>
          <p class="mono tiny" style="white-space:pre-wrap">${t.system_prompts[0]}</p>
          <p class="muted tiny">Recorded with any run started from here, and
            offered back in the playground — a model fine-tuned with one behaves
            noticeably worse without it.</p>
        </details>` : "")}
    </div>`;
}

// ---- left: the library -----------------------------------------------------

function leftPane({ d, library }) {
  return html`
    <div class="pq-pane pq-left">
      <div class="pq-pane-h">Datasets <span class="muted">${library.length || ""}</span></div>
      <div style="padding:6px 8px 0"><input type="search" id="qFilter" placeholder="Filter…" class="tiny"></div>
      <div id="qList" class="q-list">
        ${raw(d.parent_id ? `<a class="q-item lineage" href="#/data/${esc(d.parent_id)}" data-name="${
          esc((d.parent_name || "").toLowerCase())}" title="This dataset was made from that one">↑ ${esc(d.parent_name || "source")}</a>` : "")}
        ${raw((library.length ? library : [d]).map((x) => html`
          <a class="q-item ${x.id === d.id ? "on" : ""}" href="#/data/${x.id}" data-name="${x.name.toLowerCase()}">
            <span class="q-name">${x.name}</span><span class="n">${fmtNum(x.rows)}</span></a>`).join(""))}
      </div>
      <div style="padding:8px"><a class="btn btn-sm" href="#/data" style="width:100%;text-align:center">+ New source</a></div>
    </div>`;
}

// ---- middle ------------------------------------------------------------------

function fxBar({ steps, selected, d }) {
  const value = selected < 0
    ? `Source: ${d.name} — ${fmtNum(d.rows)} rows`
    : JSON.stringify(steps[selected].ops);
  return html`
    <div class="fx-bar">
      <span class="fx" title="The selected step, as it is sent to the controller">ƒx</span>
      <input id="fx" value="${value}" ${selected < 0 ? "readonly" : ""} spellcheck="false"
             title="${selected < 0 ? "" : "Edit and press Enter to replace this step"}">
    </div>`;
}

function bulkBar({ d, picked, selected }) {
  if (selected >= 0 || !picked.size || d.access !== "edit") return "";
  const splits = Object.keys(d.splits || {});
  return html`
    <div class="bulk-bar">
      <strong class="tiny">${picked.size} row${picked.size === 1 ? "" : "s"} selected</strong>
      <select id="moveTo" style="max-width:150px">
        <option value="">move to split…</option>
        ${raw(["train", "validation", "test"].concat(splits).filter((v, i, a) => a.indexOf(v) === i)
          .map((n) => `<option value="${esc(n)}">${esc(n)}</option>`).join(""))}
      </select>
      <button class="btn-sm btn-danger" id="deleteRows">Delete them</button>
      <button class="btn-sm" id="clearPick">Clear</button>
    </div>`;
}

function grid(s) {
  const { d, rows, steps, selected, preview, previewErr, loading, view, picked } = s;
  if (selected < 0) return sourceGrid(d, rows, view, picked, loading);

  if (previewErr) {
    return `<div class="pq-grid"><div class="callout callout-err" style="margin:14px">
      <strong>This step cannot run</strong>${esc(previewErr)}
      <div class="tiny" style="margin-top:6px">Edit the step, or remove it from the list on the right.</div></div></div>`;
  }
  if (!preview) {
    return `<div class="pq-grid"><div class="muted tiny" style="padding:20px">${
      loading ? "Running the steps over the sample…" : "No preview yet."}</div></div>`;
  }
  const items = preview.rows.map((row, n) =>
    ({ index: n, label: n + 1, row, rendered: (preview.rendered || [])[n] }));
  if (!items.length) {
    return `<div class="pq-grid"><div class="callout callout-warn" style="margin:14px">
      <strong>Nothing remains after this step</strong>On the sample, every row was removed.
      Applying this would be refused.</div></div>`;
  }
  const columns = preview.columns.filter((c) => c !== "split").concat(preview.columns.includes("split") ? ["split"] : []);
  const stage = preview.stages?.[selected];
  return html`
    <div class="pq-grid ${loading ? "stale" : ""}">
      ${raw(view === "text" ? renderedList(items) : table(columns, items, null))}
    </div>
    ${raw(stepDetails(preview, stage, steps[selected]))}`;
}

function sourceGrid(d, rows, view, picked, loading) {
  const splits = Object.entries(d.splits || {});
  const bar = html`
    <div class="grid-tools">
      ${raw(splits.length > 1 ? html`
        <select id="rowSplit" style="max-width:150px">
          <option value="">every split</option>
          ${raw(splits.map(([name, n]) => html`
            <option value="${name}"${rows?.split === name ? " selected" : ""}>${name} (${fmtNum(n)})</option>`).join(""))}
        </select>` : "")}
      <input type="search" id="rowSearch" placeholder="Find text in rows…" value="${rows?.query || ""}" style="max-width:240px">
      <button class="btn-sm" id="loadRows">Search</button>
      <span class="spacer"></span>
      ${raw(rows ? pager(rows) : "")}
    </div>`;
  if (!rows) {
    return bar + `<div class="pq-grid"><div class="muted tiny" style="padding:20px">${
      loading ? "Opening the rows…" : "No rows loaded."}</div></div>`;
  }
  if (!rows.rows.length) {
    return bar + `<div class="pq-grid"><div class="callout callout-warn" style="margin:14px">
      <strong>${rows.query ? `Nothing matched "${esc(rows.query)}"` : "No rows here"}</strong>
      ${rows.capped ? `Only the first ${fmtNum(rows.scanned)} rows were searched.` : ""}</div></div>`;
  }
  const columns = columnsOf(d, rows);
  return bar + html`
    <div class="pq-grid ${loading ? "stale" : ""}">
      ${raw(view === "text" ? renderedList(rows.rows) : table(columns, rows.rows, picked, rows.assets))}
    </div>`;
}

/** The rows, as a table where the data is a table.
 *
 *  Values that are not scalars -- a conversation, a tool schema -- are
 *  summarised in place and open in full on a click, because a table cell is
 *  the wrong shape for a nested object but the right shape for knowing one is
 *  there. `picked` is null for a preview, where rows are not addressable.
 */
function table(columns, items, picked, assets = {}) {
  const types = Object.fromEntries(columns.map((c) => [c, typeOf(items, c, assets)]));
  return html`
    <table>
      <thead><tr>
        ${raw(picked ? `<th class="sel"><input type="checkbox" id="selectAll" title="Select every row on this page"></th>` : "")}
        <th class="idx">#</th>
        ${raw(columns.map((c) => html`
          <th><span class="ty" title="${types[c].name}">${types[c].mark}</span>${c}
            <button class="colmenu" data-colmenu="${c}" title="Column actions">▾</button></th>`).join(""))}
      </tr></thead>
      <tbody>
        ${raw(items.map((r) => html`
          <tr class="${picked?.has(String(handleOf(r))) ? "picked" : ""}">
            ${raw(picked ? `<td class="sel"><input type="checkbox" data-row="${esc(String(handleOf(r)))}"${
              picked.has(String(handleOf(r))) ? " checked" : ""}></td>` : "")}
            <td class="idx">${raw(picked
              ? `<a href="#" data-open-row="${esc(String(handleOf(r)))}" title="Open this row">${r.index}</a>`
              : String(r.label ?? r.index))}</td>
            ${raw(columns.map((c) => {
              const v = r.row[c];
              const media = assetOf(v, assets);
              return html`<td data-cell="${r.index}" data-col="${c}"
                class="${media ? "clickable media" : scalar(v) ? (typeof v === "number" ? "num" : "") : "clickable"}">${raw(cell(v, assets))}</td>`;
            }).join(""))}
          </tr>`).join(""))}
      </tbody>
    </table>`;
}

/** What kind of thing a column holds, from the first value that is not empty. */
function typeOf(items, c, assets = {}) {
  for (const r of items) {
    const v = r.row[c];
    if (v === null || v === undefined || v === "") continue;
    // A column of stored files is a column of pictures or recordings, not
    // of text, whatever the JSON says.
    const a = assetOf(v, assets);
    if (a) {
      return a.kind === "image" ? { mark: "IMG", name: "picture" }
        : a.kind === "audio" ? { mark: "♪", name: "recording" }
        : { mark: "FILE", name: "stored file" };
    }
    if (typeof v === "number") return { mark: "123", name: "number" };
    if (typeof v === "boolean") return { mark: "✓✗", name: "true/false" };
    if (Array.isArray(v)) {
      return v.every((m) => m && typeof m === "object" && m.role)
        ? { mark: "🗨", name: "conversation" } : { mark: "[ ]", name: "list" };
    }
    if (typeof v === "object") return { mark: "{ }", name: "object" };
    return { mark: "ABC", name: "text" };
  }
  return { mark: "—", name: "empty" };
}

/** The other view: what the trainer will actually read, row by row. */
function renderedList(items) {
  return html`
    <div class="rendered-list">
      ${raw(items.map((r) => html`
        <div class="rendered-row">
          <div class="row-between">
            <span class="muted tiny mono">row ${r.label ?? r.index}</span>
            <span class="muted tiny">${fmtNum((r.rendered || "").length)} characters</span>
          </div>
          <div class="mono tiny" style="white-space:pre-wrap">${r.rendered || "(reads as nothing)"}</div>
        </div>`).join(""))}
    </div>`;
}

/** Where you are in the rows, and how to get to the rest of them. */
function pager(data) {
  const shown = data.rows.length;
  const matched = data.matched ?? data.total;
  return html`
    <span class="muted tiny">
      ${fmtNum(data.offset + 1)}–${fmtNum(data.offset + shown)} of ${fmtNum(matched)}${
        data.query ? " matching" : ""}${data.split ? ` in ${data.split}` : ""}
      ${raw(data.query && data.capped ? ` (first ${fmtNum(data.scanned)} searched)` : "")}
    </span>
    <button class="btn-sm" data-page="${Math.max(0, data.offset - PAGE)}" ${data.offset ? "" : "disabled"}>←</button>
    <button class="btn-sm" data-page="${data.offset + PAGE}" ${data.offset + shown < matched ? "" : "disabled"}>→</button>`;
}

/** What a step did to the sample, and anything it had to work out. */
function stepDetails(preview, stage, step) {
  if (!stage) return "";
  const conv = step.type === "to_conversations" || step.ops.to_conversations || step.ops.to_chat;
  return html`
    <details class="step-details" ${conv ? "open" : ""}>
      <summary>What this step did${stage.removed ? ` — ${fmtNum(stage.removed)} rows removed` : ""}</summary>
      <ul class="tiny" style="margin:6px 0 0;padding-left:18px">
        ${raw((stage.steps || []).map((t) => `<li>${esc(t)}</li>`).join("") || "<li>Nothing changed on this sample.</li>")}
      </ul>
      ${raw(conversionReport(stage.report))}
      ${raw(preview.rendered?.length ? html`
        <p class="muted tiny" style="margin:8px 0 2px">The first row, as the trainer reads it after this step:</p>
        <p class="mono tiny rendered-box">${preview.rendered[0] || "(reads as nothing)"}</p>` : "")}
    </details>`;
}

function statusBar({ d, rows, steps, selected, preview, stages, sample, loading }) {
  const cols = selected < 0 ? (rows ? columnsOf(d, rows).length : (d.columns || []).length)
    : (preview?.columns?.length ?? "…");
  const n = selected < 0 ? fmtNum(d.rows) : (preview ? fmtNum(preview.kept) : "…");
  const where = selected < 0 ? "the file itself"
    : preview?.partial ? `a preview on the first ${fmtNum(preview.sampled)} of ${fmtNum(preview.total_rows)} rows`
    : "a preview on the whole file";
  const final = steps.length && stages ? `after every step: ${fmtNum(stages[stages.length - 1].kept)} rows` : "";
  return html`
    <div class="pq-status">
      <span>${cols} columns, ${n} rows</span>
      <span>${where}</span>
      ${raw(final ? `<span>${final}</span>` : "")}
      <span class="spacer"></span>
      ${raw(loading ? `<span class="muted">working…</span>` : "")}
      ${raw(selected >= 0 && preview?.partial ? `<span class="muted" title="Dedupe and sampling see only the sample; the counts are a shape, not a total">sample counts are approximate</span>` : "")}
    </div>`;
}

// ---- right: properties and applied steps -------------------------------------

function rightPane(s) {
  const { d, steps, selected, stages, previewErr , trainedOn } = s;
  const canEdit = d.access === "edit";
  const total = stages ? stages[0].kept + stages[0].removed : null;
  const edits = (d.recipe?.edits || []).slice(-8).reverse();
  const made = d.recipe?.steps || [];
  return html`
    <div class="pq-pane pq-right">
      <div class="pq-pane-h">Query settings</div>
      <div class="pq-props">
        <div class="pq-sub">Properties</div>
        <div class="field" style="margin-bottom:8px"><label for="propName">Name</label>
          <input id="propName" value="${d.name}" maxlength="120" ${canEdit ? "" : "disabled"}></div>
        <div class="field" style="margin-bottom:8px"><label for="propTags">Tags</label>
          <input id="propTags" class="mono" value="${(d.tags || []).join(", ")}"
                 placeholder="comma-separated — raw, cleaned, v2" ${canEdit ? "" : "disabled"}></div>
        <div class="field" style="margin-bottom:0"><label for="propNotes">Notes</label>
          <textarea id="propNotes" rows="2" placeholder="What is this, and what is it for?"
                    ${canEdit ? "" : "disabled"}>${d.notes || ""}</textarea></div>
      </div>
      <div class="pq-props" style="flex:1;min-height:0;display:flex;flex-direction:column">
        <div class="pq-sub row-between">Applied steps
          <span class="muted tiny" style="text-transform:none;letter-spacing:0;font-weight:500">${
            steps.length ? `${steps.length} · not applied yet` : "none yet"}</span></div>
        <ol class="steps-list">
          <li class="step ${selected < 0 ? "on" : ""}" data-select="-1">
            <span class="step-ico">▤</span><span class="step-t">Source</span>
            <span class="delta">${total !== null ? fmtNum(total) : fmtNum(d.rows)}</span>
          </li>
          ${raw(steps.map((st, i) => {
            const stage = stages?.[i];
            const broken = previewErr && i === selected;
            return html`
              <li class="step ${i === selected ? "on" : ""} ${broken ? "bad" : ""}" data-select="${i}"
                  title="${describe(st)}">
                <span class="step-ico">${(STEPS[st.type] || STEPS.custom).icon}</span>
                <span class="step-t">${describe(st)}</span>
                <span class="delta">${broken ? "!" : stage
                  ? (stage.removed ? `−${fmtNum(stage.removed)}` : fmtNum(stage.kept)) : "…"}</span>
                <button class="step-b" data-move="${i}:-1" title="Move up" ${i ? "" : "disabled"}>↑</button>
                <button class="step-b" data-move="${i}:1" title="Move down" ${i < steps.length - 1 ? "" : "disabled"}>↓</button>
                <button class="step-b" data-edit="${i}" title="Edit this step">⚙</button>
                <button class="step-b" data-remove="${i}" title="Remove this step">✕</button>
              </li>`;
          }).join(""))}
        </ol>
        ${raw(steps.length ? "" : `<p class="muted tiny" style="padding:4px 8px">Pick something from
          the Transform, Add Column or Rows tabs. Each one becomes a step here,
          with a preview of what it does — nothing is written until you apply.</p>`)}
      </div>
      ${raw(trainedOn.length ? html`
        <div class="pq-props">
          <div class="pq-sub">Trained on this</div>
          <ul class="tiny" style="margin:4px 0;padding-left:16px">${raw(trainedOn.slice(0, 8).map((r) => html`
            <li><a href="#/jobs/${r.id}">${r.name}</a>
              <span class="muted">· ${r.status}${r.primary_metric?.value != null
                ? ` · ${r.primary_metric.label.toLowerCase()} ${r.primary_metric.lower_better
                    ? r.primary_metric.value.toFixed(4) : (r.primary_metric.value * 100).toFixed(1) + "%"}`
                : ""}</span></li>`).join(""))}</ul>
          ${raw(trainedOn.length > 8 ? `<div class="tiny muted">and ${trainedOn.length - 8} more</div>` : "")}
          <div class="tiny muted">Delete this dataset and those runs keep their
            models, but lose the record of what they were trained on.</div>
        </div>` : "")}
      ${raw(made.length || edits.length ? html`
        <div class="pq-props">
          <div class="pq-sub">History</div>
          ${raw(made.length ? html`
            <div class="tiny" style="margin-bottom:6px">
              <strong>How this was made</strong>${raw(d.parent_name
                ? ` from <a href="#/data/${esc(d.parent_id)}">${esc(d.parent_name)}</a>` : "")}
              <ul style="margin:4px 0;padding-left:16px">${raw(made.map((t) => `<li>${esc(t)}</li>`).join(""))}</ul>
              ${raw(d.recipe?.ops && d.parent_id ? `<button class="btn-sm" id="reopenRecipe"
                title="Open the source with these steps loaded, to change them">Open recipe on the source</button>` : "")}
            </div>` : "")}
          ${raw(edits.length ? html`
            <div class="tiny"><strong>Edited in place</strong>
              <ul style="margin:4px 0;padding-left:16px">${raw(edits.map((e) =>
                `<li>${esc(e.what)} <span class="muted">· ${fmtAgo(e.at)}</span></li>`).join(""))}</ul></div>` : "")}
        </div>` : "")}
    </div>`;
}

function colMenuHtml({ column, x, y }) {
  return html`
    <div class="ctx-menu" style="left:${x}px;top:${y}px" data-colmenu-box>
      <div class="ctx-title mono">${column}</div>
      <button data-colact="filter">Filter rows by this column…</button>
      <button data-colact="rename">Rename…</button>
      <button data-colact="split">Split column…</button>
      <button data-colact="calc">Add column from this…</button>
      <button data-colact="keep">Keep only this column</button>
      <button data-colact="remove" class="danger">Remove column</button>
    </div>`;
}

/** The live preview inside a step's form. */
function livePanel(r, upto) {
  const stage = r.stages?.[upto] || { kept: r.kept, removed: 0, steps: r.steps };
  const into = stage.kept + stage.removed;
  const pct = into ? Math.round((stage.kept / into) * 100) : 0;
  const cls = !stage.kept ? "callout-err" : pct < 50 ? "callout-warn" : "callout-ok";
  const cols = (r.columns || []).filter((c) => c !== "split");
  return html`
    <div class="callout ${cls}" style="margin:10px 0 0">
      <strong>${fmtNum(stage.kept)} of ${fmtNum(into)} rows would remain${
        stage.removed ? ` — ${fmtNum(stage.removed)} removed` : ""}</strong>
      ${raw((stage.steps || []).map((t) => `<div class="tiny">${esc(t)}</div>`).join(""))}
      ${raw(r.partial ? `<div class="muted tiny">On the first ${fmtNum(r.sampled)} rows.</div>` : "")}
    </div>
    ${raw(r.rows?.length ? html`
      <div class="table-wrap live-table">
        <table><thead><tr>${raw(cols.map((c) => `<th class="mono">${esc(c)}</th>`).join(""))}</tr></thead>
        <tbody>${raw(r.rows.map((row) => `<tr>${cols.map((c) => `<td>${cell(row[c])}</td>`).join("")}</tr>`).join(""))}</tbody></table>
      </div>` : "")}
    ${raw(conversionReport(r.report))}`;
}

// ---------------------------------------------------------------------------
// Pieces kept from the previous page

/** What the conversion found, and what it had to work out for itself.
 *
 *  The conversion is where data goes wrong, and it goes wrong quietly: a tool
 *  result that answers a call which never happened still renders, still
 *  trains, and still produces a model that has learned something false. This
 *  is the panel that makes that visible while there is still a step on screen
 *  to change.
 */
function conversionReport(rep) {
  if (!rep || !rep.problems) return "";
  const bad = (rep.problems || []).filter((p) => p.level === "error");
  const warn = (rep.problems || []).filter((p) => p.level === "warn");
  if (!rep.converted && !bad.length && !warn.length) return "";
  const line = (p) => `<li><strong>${esc(String(p.count))} row${p.count === 1 ? "" : "s"}</strong> — ${esc(p.message)}${
    p.rows?.length ? ` <span class="muted">(e.g. row ${p.rows.slice(0, 3).join(", ")})</span>` : ""}</li>`;
  return html`
    <div class="callout ${bad.length ? "callout-err" : warn.length ? "callout-warn" : "callout-ok"}" style="margin-top:10px">
      <strong>${fmtNum(rep.converted || 0)} rows converted${
        rep.rows_with_errors ? ` — ${fmtNum(rep.rows_with_errors)} with problems` : ""}</strong>
      ${raw(bad.length || warn.length ? "" : "Every row reads as a well-formed conversation: no tool result "
          + "answering a call that never happened, no assistant turn with nothing in it, no tool called that was never declared.")}
      ${raw(bad.length ? `<div class="tiny" style="margin-top:6px"><strong>These would teach the model something false:</strong>
        <ul style="margin:4px 0 0;padding-left:18px">${bad.map(line).join("")}</ul></div>` : "")}
      ${raw(warn.length ? `<div class="tiny" style="margin-top:6px"><strong>Worth a look:</strong>
        <ul style="margin:4px 0 0;padding-left:18px">${warn.map(line).join("")}</ul></div>` : "")}
      ${raw((rep.repairs || []).length ? `<div class="tiny" style="margin-top:6px"><strong>Worked out rather than read:</strong>
        <ul style="margin:4px 0 0;padding-left:18px">${rep.repairs.map((f) => `<li>${esc(f.what)} — ${fmtNum(f.rows)} rows</li>`).join("")}</ul>
        <span class="muted">Real data leaves the link between a tool call and its result implicit. It is
        reconstructed here and written down, so nothing downstream has to guess again.</span></div>` : "")}
      ${raw(rep.dropped ? `<div class="tiny" style="margin-top:6px">${fmtNum(rep.dropped)} row(s) had nothing conversational in them and were left out.</div>` : "")}
    </div>`;
}

function statsPanel(s) {
  const worst = { error: 0, warn: 1, info: 2, ok: 3 };
  const problems = [...s.problems].sort((a, b) => worst[a.level] - worst[b.level]);
  return html`
    <div style="margin-bottom:12px">
      ${raw(problems.map((p) => html`
        <div class="callout ${p.level === "error" ? "callout-err" : p.level === "warn" ? "callout-warn"
          : p.level === "ok" ? "callout-ok" : ""}" style="margin-bottom:8px">
          ${p.message}
          ${raw(p.fix ? `<button class="btn-sm btn-primary" data-fix="${esc(p.fix)}" style="margin-top:6px">Fix it — adds a step</button>` : "")}
        </div>`).join(""))}
    </div>
    <dl class="kv">
      <dt>Sampled</dt><dd>${fmtNum(s.sampled)} of ${fmtNum(s.total_rows)} rows</dd>
      <dt>Unique</dt><dd>${fmtNum(s.unique_rows)}</dd>
      <dt>Length</dt><dd>${fmtNum(s.chars.p50)} characters typical, ${fmtNum(s.chars.p90)} at the 90th percentile, ${fmtNum(s.chars.max)} longest</dd>
      <dt>Rough size</dt><dd>~${fmtNum(s.est_tokens)} tokens in the sample</dd>
    </dl>
    <div style="margin-top:12px">
      <div class="muted tiny" style="margin-bottom:6px">Length distribution (characters, log scale)</div>
      <div class="row" style="gap:2px;align-items:flex-end;height:60px">${raw(bars(s.histogram))}</div>
    </div>
    <details class="adv" style="margin-top:12px"><summary>Columns</summary>
      <table class="table" style="margin-top:8px"><tbody>
        ${raw(s.columns.map((c) => html`<tr><td class="mono tiny">${c.name}</td>
          <td class="tiny muted">${Math.round(c.fill_rate * 100)}% filled</td></tr>`).join(""))}
      </tbody></table></details>
    <details class="adv" style="margin-top:8px"><summary>How the trainer will read these rows</summary>
      <div class="mono tiny rendered-box" style="max-height:240px;margin-top:8px">${
        (s.preview || []).slice(0, 3).join("\n\n───\n\n")}</div></details>`;
}

function bars(hist) {
  if (!hist?.length) return "";
  const max = Math.max(...hist.map((h) => h.count), 1);
  return hist.map((h) => html`
    <span title="${fmtNum(h.from)}–${fmtNum(h.to)} chars: ${fmtNum(h.count)} rows"
          style="flex:1;background:var(--accent);opacity:.75;border-radius:2px 2px 0 0;
                 height:${Math.max(2, Math.round((h.count / max) * 100))}%"></span>`).join("");
}

/** One row, opened.
 *
 *  For a conversation, the conversation is the row: bubbles by role, the
 *  model's working folded away beside its answer, a tool call shown as a call.
 *  The fields are still there, underneath, because they are how it is edited.
 */
function rowEditor(row, rendered, canEdit) {
  // The row's own name is not a field to edit. It survives whatever is saved
  // here -- a corrected row is still the row it was.
  const fields = Object.entries(row).filter(([k]) => k !== ROW_ID);
  const chat = isConversation(row);
  const editors = html`
    ${raw(fields.map(([k, v]) => {
      const simple = v === null || v === undefined || ["string", "number", "boolean"].includes(typeof v);
      return html`
        <div class="field" style="margin:0 0 10px">
          <label for="rf_${k}">${k}</label>
          ${raw(simple
            ? `<textarea id="rf_${k}" data-field="${esc(k)}" rows="${String(v ?? "").length > 90 ? 4 : 1}" class="mono">${esc(String(v ?? ""))}</textarea>`
            : `<textarea id="rf_${k}" data-field="${esc(k)}" data-json="1" rows="8" class="mono">${esc(JSON.stringify(v, null, 2))}</textarea>
               <div class="hint">Edited as JSON — this value is a ${Array.isArray(v) ? "list" : "structure"}.</div>`)}
        </div>`;
    }).join(""))}`;
  return html`
    <div class="grid" style="gap:12px">
      ${raw(!chat ? editors : html`
        ${raw(toolsHtml(row.tools))}
        <div class="convo">${raw(conversationHtml(row.messages))}</div>
        <details class="adv"><summary>Edit this row</summary><div style="margin-top:10px">${raw(editors)}</div></details>`)}
      <details class="adv"><summary>What the trainer reads</summary>
        <p class="mono tiny rendered-box">${rendered || "(reads as nothing)"}</p></details>
      ${raw(canEdit ? html`
        <div class="row" style="gap:6px">
          <button class="btn-sm btn-primary" id="saveRow">Save this row</button>
          <button class="btn-sm" data-modal-close>Cancel</button>
        </div>` : "")}
      <div id="rowEditError"></div>
    </div>`;
}

// ---- small helpers -----------------------------------------------------------

/** Which columns to show, in a stable order, with `split` last because it is
 *  bookkeeping rather than content. */
function columnsOf(d, data) {
  const seen = [];
  (d.columns || []).forEach((c) => seen.push(c));
  (data?.rows || []).forEach((r) => Object.keys(r.row || {}).forEach(
    (k) => { if (!seen.includes(k)) seen.push(k); }));
  // `_id` is the row's own name and `split` is bookkeeping. Neither is
  // content: the first is never shown, the second goes last.
  const kept = seen.filter((c) => c !== "split" && c !== ROW_ID);
  return kept.concat(seen.includes("split") ? ["split"] : []);
}

const scalar = (v) => v === null || v === undefined || ["string", "number", "boolean"].includes(typeof v);

function summarise(value) {
  if (Array.isArray(value)) {
    const kind = value.every((v) => v && typeof v === "object" && v.role) ? "message" : "item";
    return `${value.length} ${kind}${value.length === 1 ? "" : "s"}`;
  }
  if (value && typeof value === "object") {
    const keys = Object.keys(value);
    return `${keys.length} field${keys.length === 1 ? "" : "s"}: ${keys.slice(0, 3).join(", ")}${keys.length > 3 ? "…" : ""}`;
  }
  return String(value ?? "");
}

/** The stored file a cell points at, if the page said what it is. */
function assetOf(value, assets) {
  const m = typeof value === "string" && /^asset:(ast_[0-9a-f]{12})$/.exec(value);
  return m && assets ? assets[m[1]] || null : null;
}

/** A stored file, drawn as what it is rather than as its id.
 *
 *  A thumbnail for a picture and a player for a recording, because a column
 *  of `asset:ast_3f2a…` is a column of nothing anybody can check -- and
 *  checking the data is what the workbench is for. The kind comes from the
 *  page, not from guessing at the id: the browser cannot tell a photograph
 *  from a recording without fetching it. */
function mediaCell(a) {
  if (a.kind === "image") {
    return `<img class="thumb" src="${esc(a.url)}" alt="${esc(a.filename)}" loading="lazy">`;
  }
  if (a.kind === "audio") {
    return `<audio class="clip" controls preload="none" src="${esc(a.url)}"></audio>`;
  }
  return `<span class="badge" title="${esc(a.filename)}">${esc(a.kind || "file")} · ${esc(a.filename)}</span>`;
}

function cell(value, assets) {
  if (value === null || value === undefined || value === "") return `<span class="muted tiny">—</span>`;
  const media = assetOf(value, assets);
  if (media) return mediaCell(media);
  if (!scalar(value)) return `<span class="badge">${esc(summarise(value))}</span>`;
  const text = String(value);
  return esc(text.length > CELL ? text.slice(0, CELL) + "…" : text);
}

function splitBadges(d) {
  const splits = Object.entries(d.splits || {});
  if (splits.length < 2) return "";
  return " · " + splits.map(([name, n]) => `<span class="badge">${esc(name)} ${fmtNum(n)}</span>`).join(" ");
}

const slug = (s) => (s || "dataset").toLowerCase().replace(/[^a-z0-9]+/g, "-")
  .replace(/^-|-$/g, "").slice(0, 60) || "dataset";
