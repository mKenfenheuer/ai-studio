/**
 * Prompt sets: the fixed part of an experiment.
 *
 * The list page exists to make one habit easy — write the questions down once,
 * then ask every future model the same ones. A studio that only has the
 * playground can tell you what a model said; it cannot tell you whether the
 * model you trained today is better than the one you trained last week,
 * because nothing was held constant between them.
 */
import { api } from "../api.js";
import { html, raw, esc, $, on, toast, fmtAgo, fmtNum } from "../util.js";

export async function evalsView(mount) {
  let items = await api.evals();
  let datasets = [];

  const draw = () => { mount.innerHTML = layout(items, datasets); wire(); };
  const refresh = async () => { items = await api.evals(); draw(); };

  function wire() {
    on(mount, "submit", "#newEvalForm", async (e) => {
      e.preventDefault();
      const f = Object.fromEntries(new FormData(e.target).entries());
      const parsed = parsePrompts(f.prompts || "");
      if (!parsed.items.length) {
        return toast("Write at least one prompt.", "err");
      }
      try {
        const created = await api.createEval({
          name: f.name, notes: f.notes || "", items: parsed.items });
        toast(`Saved ${parsed.items.length} prompts.`, "ok");
        location.hash = `#/evals/${created.id}`;
      } catch (ex) { toast(ex.message, "err"); }
    });

    on(mount, "input", "#promptsBox", (_e, t) => {
      const p = parsePrompts(t.value);
      $("#promptsCount", mount).textContent = p.items.length
        ? `${p.items.length} prompt${p.items.length === 1 ? "" : "s"}, `
          + `${p.withAnswers} with an expected answer`
        : "nothing yet";
    });

    on(mount, "submit", "#fromDataForm", async (e) => {
      e.preventDefault();
      const f = Object.fromEntries(new FormData(e.target).entries());
      if (!f.dataset_id) return toast("Choose a dataset first.", "err");
      try {
        const created = await api.evalFromDataset({
          dataset_id: f.dataset_id, limit: +f.limit || 50 });
        toast("Prompt set created.", "ok");
        location.hash = `#/evals/${created.id}`;
      } catch (ex) { toast(ex.message, "err"); }
    });

    on(mount, "click", "[data-del-eval]", async (_e, t) => {
      if (!confirm(`Delete "${t.dataset.name}"?\n\nEvery score recorded `
                   + "against it is deleted too, and those cannot be "
                   + "recomputed without re-running the models.")) return;
      try { await api.deleteEval(t.dataset.delEval); toast("Deleted.", "ok"); await refresh(); }
      catch (ex) { toast(ex.message, "err"); }
    });
  }

  draw();
  // Fetched after the first paint: most visits here are to open an existing
  // prompt set, not to build one out of a dataset.
  api.datasets().then((d) => { datasets = d; draw(); }).catch(() => {});
  return () => {};
}

/** Free text to prompts. One prompt per line; "prompt => expected" splits. */
function parsePrompts(text) {
  const items = [];
  let withAnswers = 0;
  for (const line of String(text).split("\n")) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    const at = trimmed.indexOf("=>");
    if (at > 0) {
      const expected = trimmed.slice(at + 2).trim();
      items.push({ prompt: trimmed.slice(0, at).trim(), expected });
      if (expected) withAnswers++;
    } else {
      items.push({ prompt: trimmed, expected: "" });
    }
  }
  return { items, withAnswers };
}

function layout(items, datasets) {
  return html`
    <div class="page-head">
      <h1>Evaluate</h1>
      <p class="sub">Ask every model the same questions, and keep the answers.</p>
    </div>

    <div class="card callout" style="margin-bottom:16px">
      <p class="tiny" style="margin:0">A model's training loss cannot be
        compared with another run's — different data, different vocabulary,
        different length. Two things here can. A <strong>held-out loss</strong>
        is measured on examples the model never trained on, and every run
        records one; <a href="#/compare">compare runs</a> puts them side by
        side. A <strong>prompt set</strong> goes further: the same questions,
        put to each model in turn, scored the same way.</p>
    </div>

    <div class="grid grid-2" style="margin-bottom:16px">
      <div class="card">
        <h3>New prompt set</h3>
        <p class="muted tiny">One prompt per line. Add the answer you would
          call correct after <code>=&gt;</code> — without it there is nothing to
          score against, only text to read.</p>
        <form id="newEvalForm" style="margin-top:10px">
          <div class="field">
            <label for="evName">Name</label>
            <input id="evName" name="name" type="text" required
                   placeholder="e.g. Home assistant commands">
          </div>
          <div class="field">
            <label for="promptsBox">Prompts</label>
            <textarea id="promptsBox" name="prompts" rows="7" class="mono"
              placeholder="Turn off the kitchen light => {&quot;action&quot;: &quot;light.turn_off&quot;}
What is the capital of France? => Paris
Write a haiku about rain"></textarea>
            <div class="hint"><span id="promptsCount">nothing yet</span></div>
          </div>
          <div class="field">
            <label for="evNotes">Notes <span class="muted tiny">(optional)</span></label>
            <input id="evNotes" name="notes" type="text"
                   placeholder="What this set is meant to measure">
          </div>
          <button class="btn-primary btn-sm" type="submit">Save prompt set</button>
        </form>
      </div>

      <div class="card">
        <h3>From a dataset</h3>
        <p class="muted tiny">Takes rows straight out of your dataset library,
          using one column as the prompt and another as the expected answer.
          The honest way to do this is to <a href="#/data">split a dataset</a>
          first and build the prompt set from the part you did
          <em>not</em> train on — otherwise you are measuring the model's
          memory, not what it learned.</p>
        <form id="fromDataForm" style="margin-top:10px">
          <div class="field">
            <label for="dsPick">Dataset</label>
            <select id="dsPick" name="dataset_id">
              <option value="">Choose one…</option>
              ${raw(datasets.map((d) => html`
                <option value="${d.id}">${d.name} · ${fmtNum(d.rows)} rows</option>`).join(""))}
            </select>
          </div>
          <div class="field">
            <label for="dsLimit">How many rows</label>
            <input id="dsLimit" name="limit" type="number" value="50" min="1" max="500">
            <div class="hint">Every model you compare answers all of them, so
              fifty is usually plenty and five hundred is an overnight job.</div>
          </div>
          <button class="btn-sm" type="submit">Build prompt set</button>
        </form>
      </div>
    </div>

    ${raw(items.length ? html`
      <div class="card" style="padding:0">
        <div class="table-wrap"><table>
          <thead><tr>
            <th>Prompt set</th><th>Prompts</th><th>Scorings</th>
            <th class="hide-sm">Owner</th><th class="hide-sm">Updated</th><th></th>
          </tr></thead>
          <tbody>${raw(items.map(row).join(""))}</tbody>
        </table></div>
      </div>` : html`
      <div class="card empty"><div class="big">🎯</div>
        <h3>No prompt sets yet</h3>
        <p class="muted">Write down the questions you actually care about, once.
          Every model you train from now on can be put to the same ones.</p>
      </div>`)}`;
}

function row(e) {
  const answered = (e.items || []).filter((i) => i.expected).length;
  return html`
    <tr>
      <td><a href="#/evals/${e.id}"><strong>${e.name}</strong></a>
        ${raw(e.notes ? `<div class="muted tiny">${esc(e.notes)}</div>` : "")}</td>
      <td>${(e.items || []).length}
        <div class="muted tiny">${answered} scorable</div></td>
      <td>${e.score_count || 0}</td>
      <td class="tiny muted hide-sm">${e.mine ? "you" : (e.owner_name || "—")}
        ${raw(e.is_shared ? ` <span class="badge">shared</span>` : "")}</td>
      <td class="tiny muted hide-sm">${fmtAgo(e.updated_at)}</td>
      <td><div class="row" style="gap:5px">
        <a class="btn btn-sm" href="#/evals/${e.id}">Open</a>
        ${raw(e.mine ? `<button class="btn-sm btn-danger" data-del-eval="${esc(e.id)}"
                 data-name="${esc(e.name)}" title="Delete">✕</button>` : "")}
      </div></td>
    </tr>`;
}
