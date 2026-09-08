/**
 * The dataset library: what you have, and the ways to get more.
 *
 * Shaped like the editor one click further in — a ribbon of things you can do,
 * and the list they act on underneath. It used to be three cards of forms
 * above the library, which meant two thirds of the page was a form nobody was
 * filling in, the library itself was below the fold, and the answer to "what
 * have I got" was further from the top of the page than the answer to "how do
 * I get more".
 *
 * The forms did not go anywhere. They open from the ribbon, which is where the
 * verbs live now.
 */
import { api } from "../api.js";
import { html, raw, esc, $, $$, on, toast, modal, fmtNum, fmtAgo } from "../util.js";
import { ribbon, rb, group, rbSelect, rbSeg, rbSearch, wireRibbon, tabState } from "../ribbon.js";
import { pageHead, emptyState, confirmDestructive } from "../components.js";

/** What the last quality check found, if one has been run.
 *
 *  Stamped with the row count it was measured at, so a dataset edited since is
 *  shown as needing another look rather than wearing a verdict that no longer
 *  describes it. */
export function qualityBadge(d) {
  const q = d.quality;
  if (!q) return "";
  const stale = (q.rows_at || 0) !== (d.rows || 0);
  if (stale) {
    return `<span class="badge" title="Checked when this had ${
      fmtNum(q.rows_at || 0)} rows; it has ${fmtNum(d.rows || 0)} now">checked earlier</span>`;
  }
  const cls = q.level === "error" ? "badge-err"
    : q.level === "warn" ? "badge-warn" : "badge-ok";
  const label = q.level === "ok" ? "checked"
    : `${q.findings} to look at`;
  return `<span class="badge ${cls}" title="${esc(q.headline || "")}">${label}</span>`;
}

const SOURCE = {
  upload: ["badge", "uploaded"],
  hub: ["badge badge-accent", "from the Hub"],
  derived: ["badge", "made here"],
  generated: ["badge badge-ok", "written by a model"],
};

// How a text file is cut into rows -- not to be confused with a dataset's
// train/validation splits, which is why this is called a mode. The same names
// the controller takes, and a sentence each, because "paragraphs" and
// "chunks" produce datasets that train very differently and the difference is
// not visible afterwards.
const TEXT_MODES = [
  ["auto", "Whatever the file suggests",
   "Paragraphs when the file has blank lines between them, otherwise one row "
   + "per line."],
  ["paragraphs", "One row per paragraph",
   "Split on blank lines. Right for articles, transcripts and books."],
  ["lines", "One row per line",
   "Right for a file where every line is already one example."],
  ["chunks", "Fixed-size chunks",
   "Cut long documents into pieces of about the length you train on."],
  ["document", "One row per file",
   "The whole file as a single example. Only sensible with many files."],
];

const TABS = [
  { key: "home", label: "Home" },
  { key: "view", label: "View" },
];

const SORTS = [
  ["updated", "Recently updated"],
  ["name", "Name"],
  ["rows", "Most rows"],
];

export async function dataView(mount) {
  let items = await api.datasets();
  let filter = "";
  let sort = "updated";
  let scope = "all";            // all | mine | shared
  let tree = false;             // group children under the dataset they came from
  let picked = new Set();

  const tabs = tabState("data", TABS, "home");
  let tab = tabs.get();

  const chosen = () => items.filter((d) => picked.has(d.id));

  const draw = () => {
    mount.innerHTML = layout({ items, filter, sort, scope, tree, picked, tab });
    wire();
  };
  const refresh = async () => { items = await api.datasets(); draw(); };

  // ---- the ways in ---------------------------------------------------------

  /** Files, from the picker, from a drop, or built out of pasted text. */
  async function upload(files, options = {}, box = null) {
    if (!files || !files.length) return;
    const label = files.length === 1 ? files[0].name : `${files.length} files`;
    if (box) {
      box.innerHTML = `<div class="callout" style="margin-top:10px">
        <strong>Reading ${esc(label)}…</strong>
        Large files take a moment — the rows are parsed on the controller,
        not in your browser.</div>`;
    }
    const name = files.length === 1 ? files[0].name.replace(/\.[^.]+$/, "") : "";
    try {
      const d = await api.uploadDataset(files, name, options);
      const skipped = (d.skipped || []).length;
      toast(`${d.added ? `Added ${fmtNum(d.added)} rows` : `Read ${fmtNum(d.rows)} rows`}${
        skipped ? ` — ${skipped} file(s) skipped` : ""}.`, skipped ? "warn" : "ok");
      location.hash = `#/data/${d.id}`;
    } catch (ex) {
      if (box) {
        box.innerHTML = `<div class="callout callout-err" style="margin-top:10px">
          <strong>That did not import</strong>${esc(ex.message)}</div>`;
      } else {
        toast(ex.message, "err");
      }
    }
  }

  function uploadDialog() {
    const dlg = modal({ title: "Upload your own files", width: 620, body: html`
      <p class="muted tiny">JSONL, JSON, CSV, TSV, plain text, Markdown, HTML,
        Word (.docx) — or a .zip of any of those. Several at once become one
        dataset, each row tagged with the file it came from.</p>

      <div id="dropZone" class="dropzone">
        <strong class="tiny">Drop files here</strong>
        <span class="muted tiny">or</span>
        <label for="fileInput" class="btn btn-sm">Choose files…</label>
        <input id="fileInput" type="file" hidden multiple
               accept=".jsonl,.ndjson,.json,.csv,.tsv,.txt,.md,.markdown,.htm,.html,.docx,.zip,.pdf,text/plain,application/json">
      </div>

      <details class="adv" style="margin-top:10px">
        <summary>How to cut text files into rows</summary>
        <div class="field" style="margin-top:8px">
          <label for="splitMode">Prose becomes</label>
          <select id="splitMode">
            ${raw(TEXT_MODES.map(([v, l]) => `<option value="${v}">${esc(l)}</option>`).join(""))}
          </select>
          <div class="hint" id="splitHint">${TEXT_MODES[0][2]}</div>
        </div>
        <div class="grid grid-2" id="chunkFields" hidden>
          <div class="field">
            <label for="chunkChars">Chunk size</label>
            <input id="chunkChars" type="number" value="2000" min="200" step="100">
            <div class="hint">Characters. Breaks at the nearest sentence.</div>
          </div>
          <div class="field">
            <label for="chunkOverlap">Overlap</label>
            <input id="chunkOverlap" type="number" value="200" min="0" step="50">
            <div class="hint">Repeated from the end of the one before.</div>
          </div>
        </div>
        <div class="field">
          <label for="uploadSplit">These rows are the</label>
          <input id="uploadSplit" type="text" value="train" list="splitNames">
          <datalist id="splitNames">
            <option value="train"><option value="validation"><option value="test">
          </datalist>
          <div class="hint">A dataset holds all of its splits together. Upload
            the training data, then upload the test set with <em>test</em>
            here and add it to the same dataset below.</div>
        </div>
        <div class="field">
          <label for="uploadInto">Add to an existing dataset</label>
          <select id="uploadInto">
            <option value="">No — make a new one</option>
            ${raw(items.filter((d) => d.mine).map((d) => html`
              <option value="${d.id}">${d.name}</option>`).join(""))}
          </select>
          <div class="hint">The only operation here that changes a dataset in
            place, because the other half of a dataset is not a new dataset.</div>
        </div>
        <div class="field">
          <label for="csvHeader">First row of a CSV</label>
          <select id="csvHeader">
            <option value="auto">Work it out</option>
            <option value="yes">Is the column names</option>
            <option value="no">Is data</option>
          </select>
        </div>
        <p class="muted tiny" style="margin:0">Only affects text and CSV. JSONL,
          JSON and the rest already say where a row ends.</p>
      </details>

      <div id="uploadStatus"></div>` });

    const options = () => {
      const into = $("#uploadInto", dlg)?.value || "";
      return {
        text_split: $("#splitMode", dlg)?.value || "auto",
        chunk_chars: $("#chunkChars", dlg)?.value || 2000,
        overlap: $("#chunkOverlap", dlg)?.value || 200,
        header: $("#csvHeader", dlg)?.value || "auto",
        split: ($("#uploadSplit", dlg)?.value || "train").trim() || "train",
        ...(into ? { into } : {}),
      };
    };
    on(dlg, "change", "#fileInput", (_e, t) =>
      upload(t.files, options(), $("#uploadStatus", dlg)));
    on(dlg, "change", "#splitMode", (_e, t) => {
      const row = TEXT_MODES.find((s) => s[0] === t.value);
      $("#splitHint", dlg).textContent = row ? row[2] : "";
      $("#chunkFields", dlg).hidden = t.value !== "chunks";
    });
    dropTarget($("#dropZone", dlg), (files) =>
      upload(files, options(), $("#uploadStatus", dlg)));
  }

  function importDialog() {
    const dlg = modal({ title: "Import from Hugging Face", width: 620, body: html`
      <p class="muted tiny">Pulls the rows here so you can look at them, clean
        them and split them before training. By default it brings in
        <strong>every split, in full</strong> — a big dataset takes a few
        minutes.</p>
      <form id="importForm" style="margin-top:10px">
        <div class="field">
          <label for="imId">Dataset</label>
          <input id="imId" name="dataset" type="text" class="mono" required
                 placeholder="tatsu-lab/alpaca">
        </div>
        <div class="grid grid-2">
          <div class="field">
            <label for="imSplit">Splits</label>
            <input id="imSplit" name="split" type="text" placeholder="all of them">
            <div class="hint">Blank means every split it has. Or look them up
              and tick the ones you want.</div>
          </div>
          <div class="field">
            <label for="imLimit">Rows per split</label>
            <input id="imLimit" name="limit" type="number" placeholder="all"
                   min="1" max="2000000">
            <div class="hint">Blank means all of them.</div>
          </div>
        </div>
        <div id="imSplits"></div>
        <div class="row" style="gap:6px;margin-top:12px;justify-content:flex-end">
          <button class="btn-sm" type="button" id="lookUp">Look up its splits</button>
          <button class="btn-sm btn-primary" type="submit" id="importGo">Import</button>
        </div>
      </form>` });

    on(dlg, "click", "#lookUp", async () => {
      const id = ($("#imId", dlg)?.value || "").trim();
      const box = $("#imSplits", dlg);
      if (!id) return toast("Which dataset?", "err");
      box.innerHTML = `<span class="muted tiny">Asking Hugging Face…</span>`;
      try {
        box.innerHTML = splitPicker(await api.datasetConfigs(id));
      } catch (ex) {
        box.innerHTML = `<div class="callout callout-err">${esc(ex.message)}</div>`;
      }
    });
    // Choosing a configuration changes which splits exist, so the tick boxes
    // are rebuilt rather than left showing the previous one's.
    on(dlg, "change", "#imCfgPick", (_e, t) => {
      const chosenSplits = JSON.parse(t.selectedOptions[0].dataset.splits || "[]");
      $("#imSplitList", dlg).innerHTML = splitBoxes(chosenSplits);
    });
    on(dlg, "submit", "#importForm", async (e) => {
      e.preventDefault();
      const f = Object.fromEntries(new FormData(e.target).entries());
      const splits = $$("[data-split]:checked", dlg).map((c) => c.dataset.split);
      const btn = $("#importGo", dlg);
      btn.disabled = true;
      btn.textContent = "Fetching rows…";
      try {
        const d = await api.importDataset({
          ...f,
          config: $("#imCfgPick", dlg)?.value || f.config,
          // Nothing ticked and nothing typed means everything there is -- the
          // server looks up which splits exist. Same for the row count: blank
          // is all of them, not a demo-sized slice of one split.
          splits: splits.length ? splits : splitList(f.split),
          limit: +f.limit || 0 });
        toast(`Imported ${fmtNum(d.rows)} rows across ${
          Object.keys(d.splits || {}).length} split(s).`, "ok");
        dlg.close();
        location.hash = `#/data/${d.id}`;
      } catch (ex) {
        toast(ex.message, "err");
        btn.disabled = false;
        btn.textContent = "Import";
      }
    });
  }

  /** Rows typed or pasted straight in.
   *
   *  The shortest path from "I have five examples in mind" to a dataset used
   *  to be: write a file, find it in a picker, upload it. Everything after
   *  this point already works on rows, so this only has to make some. */
  function pasteDialog() {
    const dlg = modal({ title: "Paste some rows", width: 620, body: html`
      <p class="muted tiny">One example per line as JSON, or plain text with one
        example per line. Both end up as a dataset you can open, edit and
        train on.</p>
      <form id="pasteForm">
        <div class="field">
          <label for="pasteName">Call it</label>
          <input id="pasteName" value="Pasted rows" maxlength="120">
        </div>
        <div class="field">
          <label for="pasteBody">Rows</label>
          <textarea id="pasteBody" rows="12" class="mono" spellcheck="false"
            placeholder='{"instruction": "Say hello", "output": "Hello!"}
{"instruction": "Say goodbye", "output": "Goodbye!"}'></textarea>
          <div class="hint" id="pasteHint">Nothing yet.</div>
        </div>
        <div class="field">
          <label for="pasteSplit">These rows are the</label>
          <input id="pasteSplit" value="train" list="splitNames">
          <datalist id="splitNames">
            <option value="train"><option value="validation"><option value="test">
          </datalist>
        </div>
        <div class="row" style="justify-content:flex-end;gap:8px">
          <button type="button" class="btn" data-modal-close>Cancel</button>
          <button type="submit" class="btn btn-primary" id="pasteGo">Make the dataset</button>
        </div>
      </form>` });

    const read = () => {
      const text = $("#pasteBody", dlg).value;
      const lines = text.split("\n").map((l) => l.trim()).filter(Boolean);
      const json = lines.filter((l) => l.startsWith("{")).length;
      return { text, lines, json };
    };
    on(dlg, "input", "#pasteBody", () => {
      const { lines, json } = read();
      $("#pasteHint", dlg).textContent = !lines.length ? "Nothing yet."
        : json === lines.length ? `${lines.length} rows, read as JSON.`
        : json ? `${lines.length} lines, ${json} of them JSON — the rest become plain text rows.`
        : `${lines.length} lines, one row each.`;
    });
    on(dlg, "submit", "#pasteForm", async (e) => {
      e.preventDefault();
      const { text, lines, json } = read();
      if (!lines.length) return toast("Nothing to make a dataset from.", "err");
      // Whichever it mostly is. A mixed paste goes in as text, because a file
      // of half-JSON is read line by line either way and this keeps every
      // line rather than dropping the ones that do not parse.
      const asJson = json === lines.length;
      const name = ($("#pasteName", dlg).value || "").trim() || "Pasted rows";
      const file = new File([lines.join("\n")],
                            `${name}.${asJson ? "jsonl" : "txt"}`,
                            { type: asJson ? "application/json" : "text/plain" });
      dlg.close();
      await upload([file], { text_split: "lines",
                             split: ($("#pasteSplit", dlg)?.value || "train").trim() || "train" });
    });
  }

  function mergeDialog() {
    const list = chosen();
    const dlg = modal({ title: `Merge ${list.length} datasets`, width: 520, body: html`
      <p class="muted tiny">One dataset with every row of these in it. The
        originals are untouched.</p>
      <ul class="consequences">${raw(list.map((d) =>
        `<li>${esc(d.name)} · ${fmtNum(d.rows)} rows</li>`).join(""))}</ul>
      <div class="field" style="margin-top:12px">
        <label for="mergeName">Call it</label>
        <input id="mergeName" value="Merged dataset" maxlength="120">
      </div>
      <label class="check"><input type="checkbox" id="mergeShuffle" checked>
        Shuffle the rows together</label>
      <p class="hint">Off keeps them end to end, which trains on one dataset
        and then the next rather than on a mixture of both.</p>
      <div class="row" style="justify-content:flex-end;gap:8px;margin-top:12px">
        <button type="button" class="btn" data-modal-close>Cancel</button>
        <button type="button" class="btn btn-primary" id="mergeGo">Merge them</button>
      </div>` });

    on(dlg, "click", "#mergeGo", async (_e, btn) => {
      btn.disabled = true;
      btn.textContent = "Merging…";
      try {
        const d = await api.mergeDatasets({
          datasets: list.map((x) => x.id),
          name: ($("#mergeName", dlg).value || "").trim() || "Merged dataset",
          shuffle: $("#mergeShuffle", dlg).checked });
        toast(`Merged into ${fmtNum(d.rows)} rows.`, "ok");
        dlg.close();
        location.hash = `#/data/${d.id}`;
      } catch (ex) {
        toast(ex.message, "err");
        btn.disabled = false;
        btn.textContent = "Merge them";
      }
    });
  }

  // ---- wiring --------------------------------------------------------------
  function wire() {
    wireRibbon(mount, (key) => { tab = key; tabs.set(key); draw(); });

    on(mount, "input", "#dsFilter", (_e, t) => {
      filter = t.value.toLowerCase();
      $("#dsList", mount).innerHTML = listing({ items, filter, sort, scope, tree, picked });
    });
    on(mount, "change", "#dsSort", (_e, t) => { sort = t.value; draw(); });
    on(mount, "click", "[data-scope]", (_e, t) => { scope = t.dataset.scope; draw(); });
    on(mount, "click", "[data-tree]", (_e, t) => { tree = t.dataset.tree === "1"; draw(); });

    on(mount, "change", "[data-pick]", (_e, t) => {
      if (t.checked) picked.add(t.dataset.pick); else picked.delete(t.dataset.pick);
      draw();
    });
    on(mount, "click", "#pickNone", () => { picked = new Set(); draw(); });

    on(mount, "click", "#openUpload", uploadDialog);
    on(mount, "click", "#openImport", importDialog);
    on(mount, "click", "#openPaste", pasteDialog);
    on(mount, "click", "#mergePicked", mergeDialog);

    on(mount, "click", "#trainOnPicked", () => {
      const d = chosen()[0];
      if (!d) return;
      sessionStorage.setItem("aistudio.dataset", JSON.stringify(
        { id: d.id, name: d.name, splits: d.splits || {}, rows: d.rows }));
      location.hash = "#/new";
    });

    on(mount, "click", "#deletePicked", async () => {
      const list = chosen().filter((d) => d.mine);
      if (!list.length) return toast("Only the owner can delete a dataset.", "err");
      if (!await confirmDestructive({
        title: list.length === 1 ? `Delete "${list[0].name}"?`
                                 : `Delete ${list.length} datasets?`,
        consequences: [
          `${fmtNum(list.reduce((n, d) => n + (d.rows || 0), 0))} rows are removed permanently.`,
          "Datasets made from these are kept, and keep working.",
          "Runs that trained on them keep their models and their history.",
        ],
        confirmLabel: "Delete" })) return;
      for (const d of list) {
        try { await api.deleteDataset(d.id); }
        catch (ex) { toast(`${d.name}: ${ex.message}`, "err"); }
      }
      picked = new Set();
      toast("Deleted.", "ok");
      await refresh();
    });

    // Files dropped anywhere on the page, not only into a dialog nobody has
    // opened yet. Dropping a file on a list of datasets means one obvious
    // thing, and making somebody find the dialog first is a step for nothing.
    dropTarget(mount, (files) => upload(files, { text_split: "auto", split: "train" }));
  }

  draw();
  return () => {};
}

/** Drag and drop onto one element.
 *
 *  Both handlers must preventDefault or the browser navigates to the file and
 *  loses the page along with the upload.
 *
 *  Registered once per element and then re-pointed, for the same reason the
 *  delegated click helper is: the page mount is a drop target and `wire()`
 *  runs on every redraw, so a plain addEventListener would have the tenth
 *  redraw upload your file ten times. */
function dropTarget(el, onFiles) {
  if (!el) return;
  if (el.__dropHandler) { el.__dropHandler = onFiles; return; }
  el.__dropHandler = onFiles;
  ["dragenter", "dragover"].forEach((ev) => el.addEventListener(ev, (e) => {
    if (!e.dataTransfer?.types?.includes("Files")) return;
    e.preventDefault();
    el.classList.add("over");
  }));
  ["dragleave", "drop"].forEach((ev) => el.addEventListener(ev, (e) => {
    e.preventDefault();
    el.classList.remove("over");
  }));
  el.addEventListener("drop", (e) => el.__dropHandler(e.dataTransfer?.files));
}

// ---------------------------------------------------------------------------

function layout(s) {
  const { items, tab } = s;
  return html`
    ${raw(pageHead({
      title: "Datasets",
      sub: `${items.length} in your library. Yours are private until you share them.`,
    }))}
    ${raw(ribbonFor(s))}
    <div class="card" style="padding:0">
      <div id="dsList">${raw(listing(s))}</div>
    </div>`;
}

function ribbonFor(s) {
  const { tab, picked, items, filter, sort, scope, tree } = s;
  const n = picked.size;
  const mine = items.filter((d) => picked.has(d.id) && d.mine).length;
  let body = "";

  if (tab === "home") {
    body = group("Get data", [
      rb("openUpload", "↑", "Upload files", { cls: "primary",
        title: "JSONL, CSV, text, Word, or a zip of them" }),
      rb("openImport", "⇩", "From the Hub",
        { title: "Import a dataset from Hugging Face" }),
      rb(null, "✦", "Write one", { href: "#/generate",
        title: "Have a model write a dataset for you" }),
      rb("openPaste", "✎", "Paste rows",
        { title: "Type or paste examples straight in" }),
    ]) + group(n ? `${n} selected` : "Selected", [
      rb(null, "▤", "Open", { disabled: n !== 1, href: n === 1 ? `#/data/${[...picked][0]}` : "" }),
      rb("trainOnPicked", "✦", "Train on this", { cls: "primary", disabled: n !== 1 }),
      rb("mergePicked", "⧉", "Merge", { disabled: n < 2,
        title: "One dataset with every row of the selected ones" }),
      rb("deletePicked", "🗑", "Delete", { cls: "danger", disabled: !mine }),
      rb("pickNone", "✕", "Clear", { disabled: !n }),
    ]);
  } else {
    body = group("Find", [
      rbSearch("dsFilter", { placeholder: "Filter by name or origin…", value: filter }),
    ]) + group("Show", [
      rbSeg([{ label: "All", on: scope === "all", data: `data-scope="all"` },
             { label: "Mine", on: scope === "mine", data: `data-scope="mine"` },
             { label: "Shared with me", on: scope === "shared", data: `data-scope="shared"` }]),
    ]) + group("Arrange", [
      rbSeg([{ label: "Flat", on: !tree, data: `data-tree="0"` },
             { label: "By lineage", on: tree, data: `data-tree="1"` }]),
      rbSelect("dsSort", { title: "Order", value: sort, options: SORTS }),
    ]);
  }
  return ribbon({ tabs: TABS, active: tab, body });
}

/** "a, b" as a list; blank entries ignored. */
function splitList(text) {
  return (text || "").split(",").map((s) => s.trim()).filter(Boolean);
}

/** The configurations and splits a Hub dataset actually has. */
function splitPicker(r) {
  const configs = r.configs || [];
  if (!configs.length) {
    return html`<div class="callout callout-warn" style="margin-top:8px">
      <strong>Could not read its splits</strong>${r.reason
        || "Type the split name instead."}</div>`;
  }
  const first = configs[0];
  return html`
    <div class="field" style="margin-top:8px">
      <label for="imCfgPick">Collection</label>
      <select id="imCfgPick" name="config">
        ${raw(configs.map((c) => html`
          <option value="${c.name}" data-splits="${JSON.stringify(c.splits || [])}"${
            c.name === r.default_config ? " selected" : ""}>${c.name}</option>`).join(""))}
      </select>
    </div>
    <div id="imSplitList">${raw(splitBoxes(
      (configs.find((c) => c.name === r.default_config) || first).splits || []))}</div>`;
}

/** One tick box per split, with train pre-ticked because it always exists and
 *  is always wanted; the others are the point of asking. */
function splitBoxes(splits) {
  if (!splits.length) return `<span class="muted tiny">No splits listed.</span>`;
  return html`
    <p class="muted tiny" style="margin:8px 0 4px">Bring in:</p>
    ${raw(splits.map((name) => html`
      <label class="check"><input type="checkbox" data-split="${name}" checked>
        ${name}</label>`).join(""))}
    <p class="muted tiny" style="margin:4px 0 0">They arrive as one dataset,
      each row remembering which split it came from — which is what makes a
      held-out score mean anything later.</p>`;
}

/** The library, in the order and shape the View tab asked for. */
function listing({ items, filter, sort, scope, tree, picked }) {
  let shown = items.filter((d) =>
    (!filter || d.name.toLowerCase().includes(filter)
      || (d.origin || "").toLowerCase().includes(filter))
    && (scope === "all" || (scope === "mine" ? d.mine : !d.mine)));

  shown.sort(sort === "name" ? (a, b) => a.name.localeCompare(b.name)
    : sort === "rows" ? (a, b) => (b.rows || 0) - (a.rows || 0)
    : (a, b) => (b.updated_at || 0) - (a.updated_at || 0));

  if (tree) shown = byLineage(shown);

  if (!shown.length) {
    return emptyState({
      icon: "▤",
      title: filter || scope !== "all" ? "Nothing matches that" : "No datasets yet",
      body: filter || scope !== "all" ? "Try a different word, or show everything."
        : "Import one from the Hub, upload a file, or have a model write one.",
      card: false,
    });
  }

  // In its own scroll container: a wide table in a card is a page that scrolls
  // sideways, and on a phone that takes the tab bar with it.
  return html`<div class="table-wrap"><table class="table">
    <thead><tr>
      <th style="width:28px"></th><th>Name</th><th>Rows</th>
      <th class="hide-sm">Where from</th><th class="hide-sm">Owner</th>
      <th class="hide-sm">Updated</th>
    </tr></thead>
    <tbody>
      ${raw(shown.map((d) => {
        const [cls, label] = SOURCE[d.source] || ["badge", d.source];
        return html`
          <tr class="${picked.has(d.id) ? "row-picked" : ""}">
            <td><input type="checkbox" data-pick="${d.id}"
                       aria-label="Select ${d.name}"
                       ${picked.has(d.id) ? "checked" : ""}></td>
            <td>
              ${raw(d.depth ? `<span class="lineage-rail" style="--depth:${d.depth}"></span>` : "")}
              <a href="#/data/${d.id}"><strong>${d.name}</strong></a>
              ${raw(qualityBadge(d))}
              ${raw(d.mine ? "" : `<span class="badge badge-accent">shared with you</span>`)}
              ${raw(d.mine && d.is_shared ? `<span class="badge">shared</span>` : "")}
            </td>
            <td>${fmtNum(d.rows)}</td>
            <td class="hide-sm"><span class="${cls}">${label}</span>
              ${raw(d.origin ? `<div class="muted tiny mono">${esc(d.origin)}</div>` : "")}</td>
            <td class="tiny muted hide-sm">${d.owner_name || "—"}</td>
            <td class="tiny muted hide-sm">${fmtAgo(d.updated_at)}</td>
          </tr>`;
      }).join(""))}
    </tbody></table></div>`;
}

/**
 * Children under the dataset they were made from.
 *
 * Every transform writes a new dataset, which is the right rule and fills a
 * library with `X`, `X (cleaned)`, `X (cleaned) (cleaned)` — a flat list where
 * nothing says which of those three is the one to train on, or that two of
 * them are drafts of the third. The parentage was recorded all along.
 */
function byLineage(rows) {
  const byId = new Map(rows.map((d) => [d.id, d]));
  const kids = new Map();
  const roots = [];
  for (const d of rows) {
    if (d.parent_id && byId.has(d.parent_id)) {
      if (!kids.has(d.parent_id)) kids.set(d.parent_id, []);
      kids.get(d.parent_id).push(d);
    } else {
      roots.push(d);
    }
  }
  const out = [];
  const walk = (d, depth) => {
    out.push({ ...d, depth });
    // Guarded against a cycle: parentage is a database field, and a restored
    // backup or a hand-edited row can point two datasets at each other.
    if (depth > 12) return;
    (kids.get(d.id) || []).forEach((k) => walk(k, depth + 1));
  };
  roots.forEach((d) => walk(d, 0));
  return out;
}
