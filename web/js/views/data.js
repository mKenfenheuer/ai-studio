/** The dataset library: what you have, and the three ways to get more. */
import { api } from "../api.js";
import { html, raw, esc, $, on, toast, fmtNum, fmtAgo } from "../util.js";

const SOURCE = {
  upload: ["badge", "uploaded"],
  hub: ["badge badge-accent", "from the Hub"],
  derived: ["badge", "made here"],
  generated: ["badge badge-ok", "written by a model"],
};

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

    on(mount, "submit", "#importForm", async (e) => {
      e.preventDefault();
      const f = Object.fromEntries(new FormData(e.target).entries());
      const btn = $("#importGo", mount);
      btn.disabled = true;
      btn.textContent = "Fetching rows…";
      try {
        const d = await api.importDataset({ ...f, limit: +f.limit || 5000 });
        toast(`Imported ${fmtNum(d.rows)} rows.`, "ok");
        location.hash = `#/data/${d.id}`;
      } catch (ex) {
        toast(ex.message, "err");
        btn.disabled = false;
        btn.textContent = "Import";
      }
    });

    on(mount, "change", "#fileInput", async (_e, t) => {
      const file = t.files[0];
      if (!file) return;
      const box = $("#uploadStatus", mount);
      box.innerHTML = `<span class="muted tiny">Reading ${esc(file.name)}…</span>`;
      try {
        const d = await api.uploadDataset(file, file.name.replace(/\.[^.]+$/, ""));
        toast(`Read ${fmtNum(d.rows)} rows.`, "ok");
        location.hash = `#/data/${d.id}`;
      } catch (ex) {
        box.innerHTML = `<div class="callout callout-err">${esc(ex.message)}</div>`;
      }
    });

    on(mount, "click", "#mergeGo", async () => {
      const picked = [...mount.querySelectorAll("[data-pick]:checked")]
        .map((c) => c.dataset.pick);
      if (picked.length < 2) return toast("Tick at least two datasets.", "err");
      const name = prompt("Name for the merged dataset?", "Merged dataset");
      if (!name) return;
      try {
        const d = await api.mergeDatasets({ datasets: picked, name, shuffle: true });
        toast(`Merged into ${fmtNum(d.rows)} rows.`, "ok");
        location.hash = `#/data/${d.id}`;
      } catch (ex) { toast(ex.message, "err"); }
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
          clean them and split them before training.</p>
        <form id="importForm" style="margin-top:10px">
          <div class="field">
            <label for="imId">Dataset</label>
            <input id="imId" name="dataset" type="text" class="mono" required
                   placeholder="tatsu-lab/alpaca">
          </div>
          <div class="grid grid-3">
            <div class="field">
              <label for="imCfg">Config</label>
              <input id="imCfg" name="config" type="text" placeholder="default">
            </div>
            <div class="field">
              <label for="imSplit">Split</label>
              <input id="imSplit" name="split" type="text" value="train">
            </div>
            <div class="field">
              <label for="imLimit">Rows</label>
              <input id="imLimit" name="limit" type="number" value="5000"
                     min="1" max="200000">
            </div>
          </div>
          <button class="btn-sm btn-primary" type="submit" id="importGo">Import</button>
        </form>
      </div>

      <div class="card">
        <h3>Upload a file</h3>
        <p class="muted tiny">JSONL, JSON, CSV, TSV or plain text — one example
          per line. The shape is worked out from the content, not the
          extension.</p>
        <div class="field" style="margin-top:10px">
          <label for="fileInput" class="btn btn-sm">Choose a file…</label>
          <input id="fileInput" type="file" hidden
                 accept=".jsonl,.json,.csv,.tsv,.txt,text/plain,application/json">
        </div>
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
        <strong><span id="mergeCount">0</span> selected</strong>
        <button class="btn-sm btn-primary" id="mergeGo">Merge them into one</button>
      </div>
      <div id="dsList">${raw(listing(items, filter))}</div>
    </div>`;
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
