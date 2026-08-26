/** The dataset library: what you have, and the three ways to get more. */
import { api } from "../api.js";
import { html, raw, esc, $, on, toast, fmtNum, fmtAgo } from "../util.js";

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

export async function dataView(mount) {
  let items = await api.datasets();
  let filter = "";

  const draw = () => {
    mount.innerHTML = layout(items, filter);
    wire();
  };

  const refresh = async () => { items = await api.datasets(); draw(); };

  function wire() {
    on(mount, "input", "#dsFilter", (_e, t) => {
      filter = t.value.toLowerCase();
      $("#dsList", mount).innerHTML = listing(items, filter);
    });

    on(mount, "click", "#lookUp", async () => {
      const id = ($("#imId", mount)?.value || "").trim();
      const box = $("#imSplits", mount);
      if (!id) return toast("Which dataset?", "err");
      box.innerHTML = `<span class="muted tiny">Asking Hugging Face…</span>`;
      try {
        const r = await api.datasetConfigs(id);
        box.innerHTML = splitPicker(r);
      } catch (ex) {
        box.innerHTML = `<div class="callout callout-err">${esc(ex.message)}</div>`;
      }
    });

    // Choosing a configuration changes which splits exist, so the tick boxes
    // are rebuilt rather than left showing the previous one's.
    on(mount, "change", "#imCfgPick", (_e, t) => {
      const chosen = JSON.parse(t.selectedOptions[0].dataset.splits || "[]");
      $("#imSplitList", mount).innerHTML = splitBoxes(chosen);
    });

    on(mount, "submit", "#importForm", async (e) => {
      e.preventDefault();
      const f = Object.fromEntries(new FormData(e.target).entries());
      const splits = [...mount.querySelectorAll("[data-split]:checked")]
        .map((c) => c.dataset.split);
      const btn = $("#importGo", mount);
      btn.disabled = true;
      btn.textContent = "Fetching rows…";
      try {
        const d = await api.importDataset({
          ...f,
          config: $("#imCfgPick", mount)?.value || f.config,
          // Nothing ticked and nothing typed means everything there is --
          // the server looks up which splits exist. Same for the row count:
          // blank is all of them, not a demo-sized slice of one split.
          splits: splits.length ? splits : splitList(f.split),
          limit: +f.limit || 0 });
        toast(`Imported ${fmtNum(d.rows)} rows across ${
          Object.keys(d.splits || {}).length} split(s).`, "ok");
        location.hash = `#/data/${d.id}`;
      } catch (ex) {
        toast(ex.message, "err");
        btn.disabled = false;
        btn.textContent = "Import";
      }
    });

    // The two ways in are the same upload: the picker and the drop zone both
    // hand over a FileList, and everything after that is one path.
    async function upload(files) {
      if (!files || !files.length) return;
      const box = $("#uploadStatus", mount);
      const label = files.length === 1
        ? files[0].name : `${files.length} files`;
      box.innerHTML = `<div class="callout" style="margin-top:10px">
        <strong>Reading ${esc(label)}…</strong>
        Large files take a moment — the rows are parsed here, not on your
        machine.</div>`;
      const name = files.length === 1
        ? files[0].name.replace(/\.[^.]+$/, "") : "";
      try {
        const d = await api.uploadDataset(files, name, uploadOptions());
        const skipped = (d.skipped || []).length;
        toast(`${d.added ? `Added ${fmtNum(d.added)} rows` : `Read ${fmtNum(d.rows)} rows`}${
          skipped ? ` — ${skipped} file(s) skipped` : ""}.`, skipped ? "warn" : "ok");
        location.hash = `#/data/${d.id}`;
      } catch (ex) {
        box.innerHTML = `<div class="callout callout-err" style="margin-top:10px">
          <strong>That did not import</strong>${esc(ex.message)}</div>`;
      }
    }

    function uploadOptions() {
      const mode = $("#splitMode", mount)?.value || "auto";
      const into = $("#uploadInto", mount)?.value || "";
      return {
        // How prose is cut into rows, and -- a different thing entirely --
        // which split of the dataset these rows belong to.
        text_split: mode,
        chunk_chars: $("#chunkChars", mount)?.value || 2000,
        overlap: $("#chunkOverlap", mount)?.value || 200,
        header: $("#csvHeader", mount)?.value || "auto",
        split: ($("#uploadSplit", mount)?.value || "train").trim() || "train",
        ...(into ? { into } : {}),
      };
    }

    on(mount, "change", "#fileInput", (_e, t) => upload(t.files));

    on(mount, "change", "#splitMode", (_e, t) => {
      const row = TEXT_MODES.find((s) => s[0] === t.value);
      $("#splitHint", mount).textContent = row ? row[2] : "";
      $("#chunkFields", mount).hidden = t.value !== "chunks";
    });

    const zone = $("#dropZone", mount);
    if (zone) {
      // Both handlers must preventDefault or the browser navigates to the
      // file instead, which loses the page and the upload with it.
      ["dragenter", "dragover"].forEach((ev) => zone.addEventListener(ev, (e) => {
        e.preventDefault();
        zone.classList.add("over");
      }));
      ["dragleave", "drop"].forEach((ev) => zone.addEventListener(ev, (e) => {
        e.preventDefault();
        zone.classList.remove("over");
      }));
      zone.addEventListener("drop", (e) => upload(e.dataTransfer?.files));
    }

    // The name is typed into the bar rather than asked for with `prompt`.
    // A browser is free to suppress that dialog -- Chrome does, permanently,
    // once somebody ticks "prevent this page from creating additional
    // dialogs" -- and a suppressed prompt returns null, which this code read
    // as "cancelled" and obeyed in silence. The button did nothing, said
    // nothing, and logged nothing, which is the worst of the three.
    on(mount, "click", "#mergeGo", async (_e, btn) => {
      const picked = [...mount.querySelectorAll("[data-pick]:checked")]
        .map((c) => c.dataset.pick);
      if (picked.length < 2) return toast("Tick at least two datasets.", "err");
      const name = ($("#mergeName", mount)?.value || "").trim() || "Merged dataset";
      btn.disabled = true;
      btn.textContent = "Merging…";
      try {
        const d = await api.mergeDatasets({ datasets: picked, name, shuffle: true });
        toast(`Merged into ${fmtNum(d.rows)} rows.`, "ok");
        location.hash = `#/data/${d.id}`;
      } catch (ex) {
        toast(ex.message, "err");
      } finally {
        btn.disabled = false;
        btn.textContent = "Merge them into one";
      }
    });

    on(mount, "change", "[data-pick]", () => {
      const n = mount.querySelectorAll("[data-pick]:checked").length;
      const bar = $("#mergeBar", mount);
      bar.hidden = n < 2;
      $("#mergeCount", mount).textContent = n;
    });
  }

  draw();
  return () => {};
}

// ---------------------------------------------------------------------------

function layout(items, filter) {
  return html`
    <div class="page-head">
      <div class="row-between" style="flex-wrap:wrap;gap:8px">
        <div>
          <h1>Datasets</h1>
          <p class="sub">${items.length} in your library. Yours are private
            until you share them.</p>
        </div>
        <a class="btn btn-primary" href="#/generate">✦ Write one with a model</a>
      </div>
    </div>

    <div class="grid grid-3" style="margin-bottom:16px;align-items:start">
      <div class="card">
        <h3>Import from Hugging Face</h3>
    <p class="muted tiny">Pulls the rows here so you can look at them,
          clean them and split them before training. By default it brings in
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
              <div class="hint">Blank means every split it has. Or look them
                up and tick the ones you want.</div>
            </div>
            <div class="field">
              <label for="imLimit">Rows per split</label>
              <input id="imLimit" name="limit" type="number" placeholder="all"
                     min="1" max="2000000">
              <div class="hint">Blank means all of them.</div>
            </div>
          </div>
          <div id="imSplits"></div>
          <div class="row" style="gap:6px;margin-top:8px">
            <button class="btn-sm" type="button" id="lookUp">Look up its splits</button>
            <button class="btn-sm btn-primary" type="submit" id="importGo">Import</button>
          </div>
        </form>
      </div>

      <div class="card">
        <h3>Upload your own files</h3>
        <p class="muted tiny">JSONL, JSON, CSV, TSV, plain text, Markdown,
          HTML, Word (.docx) — or a .zip of any of those. Drop several at
          once and they become one dataset, each row tagged with the file it
          came from.</p>

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
              ${raw(TEXT_MODES.map(([v, l, hint]) => `<option value="${v}">${esc(l)}</option>`).join(""))}
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
            <div class="hint">A dataset holds all of its splits together.
              Upload the training data, then upload the test set with
              <em>test</em> here and add it to the same dataset below.</div>
          </div>
          <div class="field">
            <label for="uploadInto">Add to an existing dataset</label>
            <select id="uploadInto">
              <option value="">No — make a new one</option>
              ${raw(items.filter((d) => d.mine).map((d) => html`
                <option value="${d.id}">${d.name}</option>`).join(""))}
            </select>
            <div class="hint">The only operation here that changes a dataset
              in place, because the other half of a dataset is not a new
              dataset.</div>
          </div>
          <div class="field">
            <label for="csvHeader">First row of a CSV</label>
            <select id="csvHeader">
              <option value="auto">Work it out</option>
              <option value="yes">Is the column names</option>
              <option value="no">Is data</option>
            </select>
          </div>
          <p class="muted tiny" style="margin:0">Only affects text and CSV.
            JSONL, JSON and the rest already say where a row ends.</p>
        </details>

        <div id="uploadStatus"></div>
      </div>

      <div class="card">
        <h3>Why bother</h3>
        <p class="muted tiny">Most of the work of getting a good model is work
          on the data, and none of it is visible from a dataset name. Bringing
          one in here lets you see the empty rows, the duplicates and the
          lengths — and hold a slice back so you can tell learning from
          memorising.</p>
      </div>
    </div>

    <div class="card">
      <div class="row-between" style="margin-bottom:10px;flex-wrap:wrap;gap:8px">
        <h3 style="margin:0">Your library</h3>
        <input type="search" id="dsFilter" placeholder="Filter…"
               value="${filter}" style="max-width:220px">
      </div>
      <div id="mergeBar" class="callout" hidden style="margin-bottom:10px">
        <div class="row" style="flex-wrap:wrap;gap:8px;align-items:center">
          <strong><span id="mergeCount">0</span> selected</strong>
          <input id="mergeName" type="text" placeholder="Merged dataset"
                 aria-label="Name for the merged dataset" style="max-width:260px">
          <button class="btn-sm btn-primary" id="mergeGo">Merge them into one</button>
        </div>
      </div>
      <div id="dsList">${raw(listing(items, filter))}</div>
    </div>`;
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

function listing(items, filter) {
  const shown = items.filter((d) =>
    !filter || d.name.toLowerCase().includes(filter)
    || (d.origin || "").toLowerCase().includes(filter));

  if (!shown.length) {
    return html`<div class="empty"><div class="big">▤</div>
      <h2>${filter ? "Nothing matches that" : "No datasets yet"}</h2>
      <p class="muted tiny">${filter ? "Try a different word."
        : "Import one from the Hub, upload a file, or have a model write one."}</p>
    </div>`;
  }

  return html`<table class="table">
    <thead><tr>
      <th style="width:28px"></th><th>Name</th><th>Rows</th>
      <th class="hide-sm">Where from</th><th class="hide-sm">Owner</th>
      <th class="hide-sm">Updated</th>
    </tr></thead>
    <tbody>
      ${raw(shown.map((d) => {
        const [cls, label] = SOURCE[d.source] || ["badge", d.source];
        return html`
          <tr>
            <td><input type="checkbox" data-pick="${d.id}"></td>
            <td>
              <a href="#/data/${d.id}"><strong>${d.name}</strong></a>
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
    </tbody></table>`;
}
