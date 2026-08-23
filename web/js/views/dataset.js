/**
 * One dataset: what is in it, what is wrong with it, and how to fix that.
 *
 * The problems panel is the point of this page. A dataset name tells you
 * nothing; "a fifth of these rows are the same row" changes what you do next,
 * and every problem it reports comes with the button that fixes it.
 */
import { api } from "../api.js";
import { html, raw, esc, $, $$, on, toast, fmtNum, fmtAgo } from "../util.js";
import { shareButton, wireShareBox } from "./share.js";

export async function datasetView(mount, [id]) {
  let d = await api.dataset(id);
  let stats = null;
  let rows = null;

  const draw = () => { mount.innerHTML = layout(d, stats, rows); wire(); };

  function wire() {
    wireShareBox(mount, "dataset", d, async () => { d = await api.dataset(id); draw(); });

    on(mount, "click", "#inspect", async () => {
      $("#inspectBox", mount).innerHTML = `<div class="muted tiny">Reading the data…</div>`;
      try {
        stats = await api.datasetInspect(id);
        draw();
      } catch (ex) { toast(ex.message, "err"); }
    });

    on(mount, "click", "#loadRows", async () => {
      try {
        rows = await api.datasetRows(id, 0, 25);
        draw();
      } catch (ex) { toast(ex.message, "err"); }
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
      const ops = {
        drop_empty: !!f.drop_empty, dedupe: !!f.dedupe, shuffle: !!f.shuffle,
        min_chars: +f.min_chars || 0, max_chars: +f.max_chars || 0,
        sample: +f.sample || 0, contains: f.contains, excludes: f.excludes,
        to_chat: !!f.to_chat, system_prompt: f.system_prompt,
      };
      await runTransform(ops, f.name);
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
        { id: d.id, name: d.name }));
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

function layout(d, stats, rows) {
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
        ${fmtNum(d.rows)} rows · ${(d.bytes / 1048576).toFixed(1)} MB ·
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

        <div class="card" style="margin-bottom:14px">
          <div class="row-between" style="margin-bottom:8px">
            <h3 style="margin:0">The rows themselves</h3>
            <button class="btn-sm" id="loadRows">Show 25 rows</button>
          </div>
          ${raw(rows ? rowTable(rows) : html`
            <div class="mono tiny" style="white-space:pre-wrap;max-height:220px;
                 overflow:auto;background:var(--surface-2);padding:10px;border-radius:6px">
              ${JSON.stringify(d.preview?.[0] || {}, null, 2)}</div>`)}
        </div>
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

function rowTable(data) {
  return html`
    <div style="max-height:340px;overflow:auto">
      ${raw(data.rows.map((r) => html`
        <div style="border-bottom:1px solid var(--border);padding:8px 0">
          <div class="muted tiny mono">row ${r.index}</div>
          <div class="mono tiny" style="white-space:pre-wrap">${r.rendered || "(reads as nothing)"}</div>
        </div>`).join(""))}
    </div>
    <p class="muted tiny" style="margin-top:8px">Showing ${data.rows.length} of
      ${fmtNum(data.total)} rows, rendered the way training will read them.</p>`;
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
        <label class="check"><input type="checkbox" name="shuffle">
          Shuffle</label>
        <label class="check"><input type="checkbox" name="to_chat">
          Rewrite as system/user/assistant turns</label>
        <div class="field">
          <label for="tSys">System prompt to add (when rewriting)</label>
          <input id="tSys" name="system_prompt" type="text"
                 placeholder="You are a helpful assistant.">
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
        <button class="btn-sm btn-primary" type="submit">Make a cleaned copy</button>
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
