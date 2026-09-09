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
import { html, raw, esc, $, on, toast, modal, fmtAgo, fmtNum, hashParam } from "../util.js";
import { ribbon, rb, group, rbSearch, wireRibbon, tabState } from "../ribbon.js";
import { pageHead, emptyState, confirmDestructive } from "../components.js";

const TABS = [
  { key: "home", label: "Home" },
  { key: "benchmarks", label: "Benchmarks" },
  { key: "about", label: "How to read this" },
];

export async function evalsView(mount) {
  let items = await api.evals();
  let datasets = [];
  let bench = null;
  let models = [];
  let filter = "";
  let picked = new Set();
  const tabs = tabState("evals", TABS, "home");
  let tab = tabs.get();

  const draw = () => {
    mount.innerHTML = layout({ items, filter, picked, tab, bench });
    wire();
  };
  const refresh = async () => { items = await api.evals(); draw(); };

  const loadBenchmarks = async () => {
    if (bench) return;
    try {
      [bench, models] = await Promise.all([
        api.benchmarks(),
        api.playground().catch(() => []),
      ]);
    } catch (e) { toast(e.message, "err"); return; }
    draw();
  };

  /** Which benchmarks, on which models, at what sample size.
   *
   *  Several at once because that is how a model is reported -- nobody quotes
   *  MMLU on its own -- and because the decisions that surround them (which
   *  models, how many questions) are the same decisions for all of them.
   */
  function benchDialog(b) {
    const runs = models || [];
    const all = bench.benchmarks || [];
    const dlg = modal({
      title: b ? `Run ${b.label}` : "Run benchmarks", width: 620, body: html`
      <p class="muted tiny">${b ? b.what
        : "Pick the ones you want. Each is queued as its own run against its "
        + "own questions, and they take their turn on the machine."}</p>
      <div class="field">
        <label>Benchmarks</label>
        <div class="picklist">
          ${raw(all.map((x) => html`
            <label class="check">
              <input type="checkbox" data-bm-pick value="${x.id}"${
                b && x.id === b.id ? " checked" : ""}>
              <span>${x.label}
                <span class="muted tiny">· ${fmtNum(x.size)} questions ·
                  ${x.shots}-shot as published</span></span>
            </label>`).join(""))}
        </div>
        <div class="hint">The number of questions below applies to each of
          them; the worked examples are each benchmark's own unless you say
          otherwise.</div>
      </div>
      <div class="callout callout-warn">
        <strong>This is not the number on the model card</strong>
        ${bench.caveat}
      </div>
      ${raw(runs.length ? html`
        <div class="field">
          <label>Models</label>
          <div class="picklist">
            ${raw(runs.map((r, i) => html`
              <label class="check">
                <input type="checkbox" data-bm-run value="${r.id}"${i === 0 ? " checked" : ""}>
                <span>${r.name}
                  <span class="muted tiny">· ${r.base_model || r.kind}</span></span>
              </label>`).join(""))}
          </div>
        </div>` : html`
        <p class="muted tiny">No finished models of your own yet — a model off
          the Hub can still be run on its own.</p>`)}
      <div class="field">
        <label for="bmBase">And a model off the Hub <span class="muted tiny">(optional)</span></label>
        <input id="bmBase" type="text" class="mono" placeholder="owner/name"
               value="${esc(runs[0]?.base_model || "")}">
        <div class="hint">The base a run was trained from is the comparison
          worth having: it says whether the training helped or cost you
          something.</div>
      </div>
      <div class="row" style="gap:10px">
        <div class="field" style="flex:1">
          <label for="bmSample">How many questions each</label>
          <input id="bmSample" type="number" min="20" max="${bench.max_sample || 2000}"
                 value="${b ? Math.min(bench.default_sample, b.size)
                            : bench.default_sample}">
          <div class="hint">Fewer is faster and wider: a sample cannot
            separate two models a couple of points apart, and every result
            says by how much. A benchmark smaller than this is asked in
            full.</div>
        </div>
        <div class="field" style="flex:1">
          <label for="bmShots">Worked examples</label>
          <input id="bmShots" type="number" min="0" max="25"
                 placeholder="${b ? b.shots : "as published"}"
                 value="${b ? b.shots : ""}">
          <div class="hint">${b ? `${b.shots} is what this one is published at.`
            : "Blank leaves each benchmark at what it is published with."}
            Changing it changes the number.</div>
        </div>
      </div>
      ${raw((b ? [b] : all).some((x) => x.needs_local) ? html`
        <p class="muted tiny">${b || all.every((x) => x.needs_local)
          ? "Answered by scoring each option's probability, so it needs the "
            + "weights on a machine here — a hosted model cannot be run on it."
          : "Some of these are answered by scoring each option's probability, "
            + "which needs the weights on a machine here: a hosted model "
            + "cannot be given those, and is left out of the ones that do."}
        </p>` : "")}
      <div class="row" style="justify-content:flex-end;gap:8px;margin-top:12px">
        <button type="button" class="btn" data-modal-close>Cancel</button>
        <button type="button" class="btn btn-primary" id="bmGo">Run it</button>
      </div>` });

    on(dlg, "click", "#bmGo", async (_e, btn) => {
      const chosen = [...dlg.querySelectorAll("[data-bm-run]:checked")]
        .map((c) => c.value);
      const picked = [...dlg.querySelectorAll("[data-bm-pick]:checked")]
        .map((c) => c.value);
      const hub = ($("#bmBase", dlg).value || "").trim();
      if (!picked.length) return toast("Choose at least one benchmark.", "err");
      if (!chosen.length && !hub) {
        return toast("Choose at least one model.", "err");
      }
      const shots = ($("#bmShots", dlg).value || "").trim();
      btn.disabled = true;
      btn.textContent = "Queueing…";
      try {
        const r = await api.runBenchmark({
          benchmarks: picked,
          model_job_ids: chosen,
          baselines: hub ? [{ source: "hub", model: hub }] : [],
          sample: +$("#bmSample", dlg).value || undefined,
          shots: shots === "" ? null : +shots,
          project_id: hashParam("project"),
        });
        dlg.close();
        const n = (r.jobs || []).length || 1;
        toast(n === 1 ? "Queued." : `${n} benchmarks queued.`, "ok",
              { href: `#/jobs/${r.id}`, label: "Watch the first" });
        location.hash = n === 1 ? `#/jobs/${r.id}` : "#/jobs";
      } catch (e) {
        toast(e.message, "err");
        btn.disabled = false;
        btn.textContent = "Run it";
      }
    });
  }

  /** Prompts typed straight in. */
  function newSetDialog() {
    const dlg = modal({ title: "New prompt set", width: 620, body: html`
      <p class="muted tiny">One prompt per line. Add the answer you would call
        correct after <code>=&gt;</code> — without it there is nothing to score
        against, only text to read. A line that is a JSON object can say more:
        <code>{"prompt": …, "schema": {…}}</code> scores the answer against a
        shape, <code>{"prompt": …, "expected_tool": {"name": …, "arguments": {…}}}</code>
        against a call.</p>
      <form id="newEvalForm">
        <div class="field">
          <label for="evName">Name</label>
          <input id="evName" name="name" type="text" required
                 placeholder="e.g. Home assistant commands">
        </div>
        <div class="field">
          <label for="promptsBox">Prompts</label>
          <textarea id="promptsBox" name="prompts" rows="9" class="mono"
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
        <div class="row" style="justify-content:flex-end;gap:8px">
          <button type="button" class="btn" data-modal-close>Cancel</button>
          <button class="btn btn-primary" type="submit">Save prompt set</button>
        </div>
      </form>` });
    wireNewSet(dlg);
  }

  /** Prompts taken out of a dataset split. */
  function fromDatasetDialog() {
    const dlg = modal({ title: "Prompt set from a dataset", width: 560, body: html`
      <p class="muted tiny">Takes rows straight out of your library, using one
        column as the prompt and another as the expected answer. Build it from
        a split you did <em>not</em> train on, or you are measuring the model's
        memory rather than what it learned.</p>
      <form id="fromDataForm">
        <div class="field">
          <label for="dsPick">Dataset</label>
          <select id="dsPick" name="dataset_id">
            <option value="">Choose one…</option>
            ${raw(datasets.map((d) => html`
              <option value="${d.id}">${d.name} · ${fmtNum(d.rows)} rows</option>`).join(""))}
          </select>
        </div>
        <div id="dsSplitBox">${raw(splitPicker(null))}</div>
        <div class="field">
          <label for="dsLimit">How many rows</label>
          <input id="dsLimit" name="limit" type="number" value="50" min="1" max="500">
          <div class="hint">Every model you compare answers all of them, so
            fifty is usually plenty and five hundred is an overnight job.</div>
        </div>
        <div class="row" style="justify-content:flex-end;gap:8px">
          <button type="button" class="btn" data-modal-close>Cancel</button>
          <button class="btn btn-primary" type="submit">Build prompt set</button>
        </div>
      </form>` });
    wireFromDataset(dlg);
  }

  function wireNewSet(root) {
    on(root, "input", "#promptsBox", (_e, t) => {
      const p = parsePrompts(t.value);
      $("#promptsCount", root).textContent = p.items.length
        ? `${p.items.length} prompt${p.items.length === 1 ? "" : "s"}, `
          + `${p.withAnswers} with an expected answer`
        : "nothing yet";
    });
    on(root, "submit", "#newEvalForm", async (e) => {
      e.preventDefault();
      const f = Object.fromEntries(new FormData(e.target).entries());
      const parsed = parsePrompts(f.prompts || "");
      if (!parsed.items.length) {
        return toast("Write at least one prompt.", "err");
      }
      try {
        const created = await api.createEval({
          name: f.name, notes: f.notes || "", items: parsed.items,
          project_id: hashParam("project") });
        toast(`Saved ${parsed.items.length} prompts.`, "ok");
        location.hash = `#/evals/${created.id}`;
      } catch (ex) { toast(ex.message, "err"); }
    });

  }

  function wireFromDataset(root) {
    // The split boxes follow the dataset: a held-out split is offered first
    // when there is one, and the hint says which kind of set this will make.
    on(root, "change", "#dsPick", (_e, t) => {
      const d = datasets.find((x) => x.id === t.value);
      const box = $("#dsSplitBox", root);
      if (box) box.innerHTML = splitPicker(d);
    });
    on(root, "submit", "#fromDataForm", async (e) => {
      e.preventDefault();
      const f = Object.fromEntries(new FormData(e.target).entries());
      if (!f.dataset_id) return toast("Choose a dataset first.", "err");
      try {
        const created = await api.evalFromDataset({
          dataset_id: f.dataset_id, limit: +f.limit || 50,
          split: f.split || "" });
        toast("Prompt set created.", "ok");
        location.hash = `#/evals/${created.id}`;
      } catch (ex) { toast(ex.message, "err"); }
    });
  }

  function wire() {
    wireRibbon(mount, (key) => {
      tab = key; tabs.set(key); draw();
      if (key === "benchmarks") loadBenchmarks();
    });
    on(mount, "click", "[data-run-bm]", (_e, t) =>
      benchDialog((bench.benchmarks || []).find((b) => b.id === t.dataset.runBm)));
    on(mount, "click", "#runSeveral", () => benchDialog(null));
    on(mount, "click", "#newSet", newSetDialog);
    on(mount, "click", "#fromDataset", fromDatasetDialog);
    on(mount, "input", "#evFilter", (_e, t) => {
      filter = t.value.toLowerCase();
      $("#evList", mount).innerHTML = listing({ items, filter, picked });
    });
    on(mount, "change", "[data-pick]", (_e, t) => {
      if (t.checked) picked.add(t.dataset.pick); else picked.delete(t.dataset.pick);
      draw();
    });
    on(mount, "click", "#pickNone", () => { picked = new Set(); draw(); });

    on(mount, "click", "#copySet", async () => {
      const one = items.find((e) => picked.has(e.id));
      if (!one) return;
      try {
        const made = await api.copyEval(one.id, {});
        toast("Copied.", "ok");
        location.hash = `#/evals/${made.id}`;
      } catch (ex) { toast(ex.message, "err"); }
    });

    on(mount, "click", "#deleteSet", async () => {
      const list = items.filter((e) => picked.has(e.id) && e.mine);
      if (!list.length) return toast("Only the owner can delete a prompt set.", "err");
      if (!await confirmDestructive({
        title: list.length === 1 ? `Delete "${list[0].name}"?`
                                 : `Delete ${list.length} prompt sets?`,
        consequences: [
          "Every score recorded against them goes too.",
          "Those cannot be recomputed without running the models again.",
        ],
        confirmLabel: "Delete" })) return;
      for (const e of list) {
        try { await api.deleteEval(e.id); } catch (ex) { toast(ex.message, "err"); }
      }
      picked = new Set();
      toast("Deleted.", "ok");
      await refresh();
    });
  }

  draw();
  // Fetched after the first paint: most visits here are to open an existing
  // prompt set, not to build one out of a dataset.
  api.datasets().then((d) => { datasets = d; draw(); }).catch(() => {});
  return () => {};
}

/** Free text to prompts. One prompt per line; "prompt => expected" splits. */
const HELD_OUT = /^(validation|test|eval|dev|val|holdout)/i;

/** Which split the prompts come from, held-out ones first. */
function splitPicker(d) {
  const names = Object.keys(d?.splits || {});
  if (!d || !names.length) return "";
  const held = names.filter((n) => HELD_OUT.test(n));
  const rest = names.filter((n) => !HELD_OUT.test(n));
  const ordered = held.concat(rest);
  return html`
    <div class="field">
      <label for="dsSplit">Which split</label>
      <select id="dsSplit" name="split">
        ${raw(ordered.map((n, i) => html`
          <option value="${n}"${i === 0 ? " selected" : ""}>${n} · ${
            fmtNum(d.splits[n])} rows${HELD_OUT.test(n) ? " · held out" : ""}</option>`).join(""))}
      </select>
      <div class="hint">${held.length
        ? "A held-out split is a fair test of anything that trained on the rest."
        : "This dataset has no held-out split. Hold one back in the dataset editor first, or accept that these scores measure memory as much as skill."}</div>
    </div>`;
}

function parsePrompts(text) {
  const items = [];
  let withAnswers = 0;
  for (const line of String(text).split("\n")) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    // A line that is a JSON object is a prompt with more on it: the shape
    // the answer must fit (`schema`), the tool it should call
    // (`expected_tool`), and the expected answer beside them.
    if (trimmed.startsWith("{")) {
      try {
        const o = JSON.parse(trimmed);
        if (o && typeof o === "object" && o.prompt) {
          items.push(o);
          if (o.expected) withAnswers++;
          continue;
        }
      } catch { /* not JSON after all: a prompt that happens to start with a brace */ }
    }
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

function layout(s) {
  return html`
    ${raw(pageHead({
      title: "Evaluate",
      sub: "Ask every model the same questions, and keep the answers.",
    }))}
    ${raw(ribbonFor(s))}
    ${raw(s.tab === "about" ? about()
      : s.tab === "benchmarks" ? benchmarkPanel(s.bench)
      : html`
      <div class="card" style="padding:0">
        <div id="evList">${raw(listing(s))}</div>
      </div>`)}`;
}

/** The published benchmarks, and the warning that has to come with them. */
function benchmarkPanel(bench) {
  if (!bench) {
    return `<div class="card empty"><p class="muted">Loading the catalogue…</p></div>`;
  }
  return html`
    <div class="callout callout-warn" style="margin-bottom:14px">
      <strong>These will not match a model card, and cannot be made to</strong>
      ${bench.caveat}
    </div>
    <p style="margin:0 0 14px">
      <button class="btn btn-primary" id="runSeveral">Run several at once</button>
      <span class="muted tiny" style="margin-left:8px">One decision about
        which models and how many questions, and each benchmark queued as its
        own run.</span></p>
    <div class="card" style="padding:0;margin-bottom:14px">
      <div class="table-wrap"><table>
        <thead><tr>
          <th>Benchmark</th><th class="hide-sm">What it asks</th>
          <th class="hide-sm">Questions</th><th class="hide-sm">Scored by</th>
          <th></th>
        </tr></thead>
        <tbody>
          ${raw(bench.benchmarks.map((b) => html`
            <tr>
              <td><strong>${b.label}</strong>
                <div class="muted tiny mono">${b.dataset}</div></td>
              <td class="hide-sm tiny muted">${b.what}
                ${raw(b.published ? `<div style="margin-top:4px">${esc(b.published)}</div>` : "")}</td>
              <td class="hide-sm tiny">${fmtNum(b.size)}
                <div class="muted">${b.shots}-shot</div></td>
              <td class="hide-sm tiny muted">${b.protocol === "multiple_choice"
                ? "the probability it gives each option"
                : "a number pulled out of what it writes"}</td>
              <td><button class="btn-sm btn-primary" data-run-bm="${b.id}">Run it</button></td>
            </tr>`).join(""))}
        </tbody>
      </table></div>
    </div>
    <div class="card">
      <h3 style="margin:0 0 6px">Not here, and why</h3>
      <p class="muted tiny">A list with these missing would read as an
        oversight rather than a decision.</p>
      <div class="picklist" style="margin-top:8px">
        ${raw(bench.unavailable.map((u) => html`
          <div style="padding:6px 0">
            <strong class="tiny">${u.name}</strong>
            <div class="muted tiny">${u.why}</div>
          </div>`).join(""))}
      </div>
    </div>`;
}

function ribbonFor({ tab, items, picked, filter }) {
  const chosen = items.filter((e) => picked.has(e.id));
  const one = chosen.length === 1 ? chosen[0] : null;
  const mine = chosen.filter((e) => e.mine).length;
  const body = tab === "home"
    ? group("New prompt set", [
        rb("newSet", "✎", "Type prompts", { cls: "primary" }),
        rb("fromDataset", "▤", "From a dataset",
          { title: "Take rows out of a held-out split" }),
      ]) + group(chosen.length ? `${chosen.length} selected` : "Selected", [
        rb(null, "◎", "Open", { disabled: !one, href: one ? `#/evals/${one.id}` : "" }),
        rb("copySet", "⧉", "Copy", { disabled: !one,
          title: "A copy to edit, so the scores already taken keep their meaning" }),
        rb("deleteSet", "🗑", "Delete", { cls: "danger", disabled: !mine }),
        rb("pickNone", "✕", "Clear", { disabled: !chosen.length }),
      ]) + group("Compare", [
        rb(null, "⚖", "Held-out loss", { href: "#/compare",
          title: "Rank finished runs by the loss they measured on unseen data" }),
      ]) + group("Find", [
        rbSearch("evFilter", { placeholder: "Filter…", value: filter }),
      ])
    : tab === "benchmarks"
    ? group("Benchmarks", [
        rb(null, "≡", "Prompt sets", { href: "#/evals" }),
        rb(null, "⚖", "Compare runs", { href: "#/compare" }),
      ])
    : group("Elsewhere", [
        rb(null, "⚖", "Compare runs", { href: "#/compare" }),
        rb(null, "▤", "Datasets", { href: "#/data" }),
      ]);
  return ribbon({ tabs: TABS, active: tab, body });
}

/** Why there are two numbers here and what each is worth. */
function about() {
  return html`
    <div class="card">
      <h3>Two numbers, and what each one means</h3>
      <p class="muted tiny">A model's <strong>training loss</strong> cannot be
        compared with another run's — different data, different vocabulary,
        different length. Two things here can.</p>
      <p class="muted tiny">A <strong>held-out loss</strong> is measured on
        examples the model never trained on, and every run records one.
        <a href="#/compare">Compare runs</a> puts them side by side. It is free,
        it is honest, and it only tells you which model fits its own held-out
        data best — not whether either is any good at your task.</p>
      <p class="muted tiny">A <strong>prompt set</strong> goes further: the same
        questions, put to each model in turn, scored the same way. It is the
        only thing here that answers "is the one I trained today better than
        the one I trained last week", and it is only worth anything if the
        models were not trained on these prompts. Build one from a held-out
        split and that is guaranteed rather than hoped for.</p>
      <p class="muted tiny">When two models are close, this studio will say so
        rather than crowning one: a difference smaller than the spread between
        prompts is not a difference.</p>
    </div>`;
}

function listing({ items, filter, picked }) {
  const shown = items.filter((e) => !filter
    || e.name.toLowerCase().includes(filter)
    || (e.notes || "").toLowerCase().includes(filter));

  if (!shown.length) {
    return emptyState({
      icon: "🎯",
      title: items.length ? "Nothing matches that" : "No prompt sets yet",
      body: items.length ? "Try a different word."
        : "Write down the questions you actually care about, once. Every model "
          + "you train from now on can be put to the same ones.",
      card: false,
    });
  }
  return html`
    <div class="table-wrap"><table>
      <thead><tr>
        <th style="width:28px"></th>
        <th>Prompt set</th><th>Prompts</th><th>Scorings</th>
        <th class="hide-sm">Owner</th><th class="hide-sm">Updated</th><th></th>
      </tr></thead>
      <tbody>${raw(shown.map((e) => row(e, picked)).join(""))}</tbody>
    </table></div>`;
}

function row(e, picked) {
  const answered = (e.items || []).filter((i) => i.expected).length;
  return html`
    <tr class="${picked.has(e.id) ? "row-picked" : ""}">
      <td><input type="checkbox" data-pick="${e.id}" aria-label="Select ${e.name}"
                 ${picked.has(e.id) ? "checked" : ""}></td>
      <td><a href="#/evals/${e.id}"><strong>${e.name}</strong></a>
        ${raw(e.notes ? `<div class="muted tiny">${esc(e.notes)}</div>` : "")}</td>
      <td>${(e.items || []).length}
        <div class="muted tiny">${answered} scorable</div></td>
      <td>${e.score_count || 0}</td>
      <td class="tiny muted hide-sm">${e.mine ? "you" : (e.owner_name || "—")}
        ${raw(e.is_shared ? ` <span class="badge">shared</span>` : "")}</td>
      <td class="tiny muted hide-sm">${fmtAgo(e.updated_at)}</td>
      <td><a class="btn btn-sm" href="#/evals/${e.id}">Open</a></td>
    </tr>`;
}
