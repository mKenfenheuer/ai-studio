/**
 * One prompt set: its questions, and every model that has answered them.
 *
 * The table is the point of the page. Its rows are (model, date) pairs and its
 * columns are the four measures the runner computes — see
 * runner/jobs/evaluate.py for what each is worth. They are shown together
 * rather than reduced to a single score because they disagree, and a page that
 * hid the disagreement would be inventing confidence it does not have.
 *
 * Magnitude is drawn as a bar in one hue, not as a colour per model: the
 * palette here has exactly two validated series colours, and a comparison
 * across eight models is a magnitude question anyway.
 */
import { api, events } from "../api.js";
import { LineChart } from "../chart.js";
import { html, raw, esc, $, $$, on, toast, modal, fmtAgo, fmtNum, fmtDuration } from "../util.js";
import { shareButton, wireShareBox } from "./share.js";
import { ribbon, rb, group, wireRibbon, tabState } from "../ribbon.js";
import { breadcrumb, confirmDestructive, emptyState } from "../components.js";
import { openServeDialog } from "../registry.js";

const TABS = [
  { key: "scores", label: "Scores" },
  { key: "trend", label: "Over time" },
  { key: "run", label: "Score models" },
  { key: "prompts", label: "The prompts" },
];

export async function evalView(mount, [evalId]) {
  let ev = await api.eval(evalId);
  let candidates = [];
  let openScore = null;
  let hosted = { providers: [], connected: [] };
  // What is going to be scored, held here rather than read off the checkboxes
  // at the end: adding a baseline redraws the panel, and a redraw that forgot
  // which models were ticked would be worse than no baselines at all.
  const picked = new Set();
  let baselines = [];
  const tabs = tabState("evalview", TABS, "scores");
  let tab = tabs.get();

  const addBaseline = (b) => {
    const key = b.source === "api" ? `api:${b.provider}:${b.model}`
                                   : `hub:${b.model}`;
    if (baselines.some((x) => x.key === key)) return;
    baselines.push({ ...b, key });
    draw();
  };

  const draw = () => {
    mount.innerHTML = layout(ev, candidates, openScore, tab,
                             { picked, baselines, hosted });
    wire();
  };

  const refresh = async () => { ev = await api.eval(evalId); draw(); };

  function wire() {
    wireShareBox(mount, "eval", ev, refresh);
    drawTrend(mount, ev, tab);
    wireRibbon(mount, (key) => { tab = key; tabs.set(key); draw(); });

    // A prompt set with no scores yet opens on the tab that does something
    // about that, rather than on an empty table.
    on(mount, "click", "#goScore", () => { tab = "run"; tabs.set(tab); draw(); });

    on(mount, "click", "#deleteEval", async () => {
      if (!await confirmDestructive({
        title: `Delete "${ev.name}"?`,
        consequences: [
          "Every score recorded against it goes too.",
          "Those cannot be recomputed without running the models again.",
        ],
        confirmLabel: "Delete the prompt set" })) return;
      try {
        await api.deleteEval(evalId);
        toast("Deleted.", "ok");
        location.hash = "#/evals";
      } catch (ex) { toast(ex.message, "err"); }
    });

    on(mount, "change", "[data-model-pick]", (_e, t) => {
      if (t.checked) picked.add(t.value); else picked.delete(t.value);
      // Redrawn because the suggested baselines are the base models of
      // whatever is ticked: tick a fine-tune and the model it came from is
      // offered on the spot, which is the whole point.
      draw();
    });

    on(mount, "click", "[data-add-base]", (_e, t) =>
      addBaseline({ source: "hub", model: t.dataset.addBase,
                    like_run: t.dataset.likeRun || "" }));

    on(mount, "click", "#addHub", () => {
      const box = $("#hubModel", mount);
      const id = (box.value || "").trim();
      if (!id) return toast("Name a model on the Hub.", "err");
      if (!id.includes("/")) {
        return toast("A Hub model looks like owner/name.", "err");
      }
      box.value = "";
      addBaseline({ source: "hub", model: id });
    });

    on(mount, "click", "#addApi", () => {
      const provider = $("#apiProvider", mount)?.value;
      const model = ($("#apiModel", mount)?.value || "").trim();
      if (!provider) return toast("Connect a provider first.", "err");
      if (!model) return toast("Which model at that provider?", "err");
      addBaseline({ source: "api", provider, model });
    });

    on(mount, "click", "[data-drop-base]", (_e, t) => {
      baselines = baselines.filter((b) => b.key !== t.dataset.dropBase);
      draw();
    });

    on(mount, "click", "#runEval", async () => {
      if (!picked.size && !baselines.length) {
        return toast("Choose at least one model.", "err");
      }
      const btn = $("#runEval", mount);
      btn.disabled = true;
      btn.textContent = "Queueing…";
      try {
        const { id } = await api.runEval(evalId, {
          model_job_ids: [...picked],
          baselines: baselines.map(({ key, ...b }) => b),
          max_new_tokens: +$("#evMaxTokens", mount).value || 200,
          temperature: +$("#evTemp", mount).value || 0,
          system_prompt: $("#evSystem", mount).value || "",
          judge: $("#evJudge", mount)?.value ? {
            provider: $("#evJudge", mount).value,
            model: $("#evJudgeModel", mount)?.value || "",
            rubric: $("#evRubric", mount)?.value || "",
          } : null,
        });
        toast("Scoring queued.", "ok");
        location.hash = `#/jobs/${id}`;
      } catch (ex) {
        toast(ex.message, "err");
        btn.disabled = false;
        btn.textContent = "Score these models";
      }
    });

    on(mount, "click", "[data-open-score]", async (_e, t) => {
      const id = t.dataset.openScore;
      if (openScore?.id === id) { openScore = null; return draw(); }
      try {
        openScore = await api.evalScore(evalId, id);
        draw();
      } catch (ex) { toast(ex.message, "err"); }
    });

    on(mount, "click", "[data-del-score]", async (_e, t) => {
      if (!await confirmDestructive({
        title: "Remove this scoring?",
        body: "It leaves the comparison. The run that produced it is untouched.",
        confirmLabel: "Remove it" })) return;
      try { await api.deleteScore(evalId, t.dataset.delScore); await refresh(); }
      catch (ex) { toast(ex.message, "err"); }
    });

    // Promotion from the eval page: the run that won this prompt set is
    // exactly the one that should answer to the name other software uses, and
    // getting there meant remembering which run it was and finding it again.
    // Promotion from the eval page: the run that won this prompt set is
    // exactly the one that should answer to the name other software uses.
    on(mount, "click", "#promoteBest", (_e, t) =>
      openServeDialog({ id: t.dataset.job, name: t.dataset.name || "this run" },
                      { title: "Serve the best model under a name" }));

    on(mount, "change", "#trendOnly", (_e, t) => { ev._trendOnly = t.value; draw(); });

    on(mount, "click", "#copyEval", async () => {
      try {
        const copy = await api.copyEval(evalId, {});
        toast("Copied. Edit the copy freely.", "ok");
        location.hash = `#/evals/${copy.id}`;
      } catch (ex) { toast(ex.message, "err"); }
    });
  }

  draw();
  api.playground().then((p) => { candidates = p; draw(); }).catch(() => {});
  api.providers().then((h) => { hosted = h; draw(); }).catch(() => {});
  const unsub = events.subscribe((m) => { if (m.type === "jobs_changed") refresh(); });
  return () => unsub();
}

/** Fill the trend chart, once its element exists in the page. */
function drawTrend(mount, ev, tab) {
  const el = tab === "trend" ? $("#trendChart", mount) : null;
  if (!el) return;
  const key = el.dataset.trend;
  const measure = MEASURES[key] || MEASURES.expected_loss;
  const only = el.dataset.only || "";
  const scores = (ev.scores || []).slice()
    .sort((a, b) => a.created_at - b.created_at)
    .filter((s) => s.metrics[key] != null && (!only || s.model_ref === only));

  const each = scores.map((s) => ({ x: s.created_at, y: s.metrics[key] }));
  let running = null;
  const frontier = scores.map((s) => {
    const v = s.metrics[key];
    running = running === null ? v
      : (measure.lower ? Math.min(running, v) : Math.max(running, v));
    return { x: s.created_at, y: running };
  });

  const chart = new LineChart(el, {
    title: `${measure.label} on this set${measure.lower ? " — lower is better"
                                                       : " — higher is better"}`,
    height: 260,
    xLabel: "Scored",
    format: (v) => measure.fmt(v),
    // Seconds since the epoch on the axis would read as a nine-digit number.
    xFormat: (v) => new Date(v * 1000).toLocaleDateString(undefined,
      { month: "short", day: "numeric" }),
    series: [
      { key: "each", label: "Each scoring" },
      { key: "best", label: "Best so far", dashed: true },
    ],
  });
  chart.setSeries("each", each);
  chart.setSeries("best", frontier);
}

// ---------------------------------------------------------------------------

function layout(ev, candidates, openScore, tab, state) {
  const scores = ev.scores || [];
  const items = ev.items || [];
  const answered = items.filter((i) => i.expected).length;
  document.title = `${ev.name} · Evaluate · AI Studio`;

  const body = tab === "trend" ? trendPanel(ev)
    : tab === "run" ? runPanel(ev, candidates, state)
    : tab === "prompts" ? ((ev.source || {}).benchmark
        ? benchmarkPrompts(ev) : promptsPanel(items, answered))
    : html`${raw((ev.source || {}).benchmark
             ? benchmarkTable(ev, scores)
             : scoreTable(scores, answered, items.length))}
           ${raw(openScore ? scoreDetail(openScore) : "")}`;

  return html`
    <div class="page-head">
      ${raw(breadcrumb({ href: "#/evals", label: "Prompt sets" }))}
      <h1 style="margin:6px 0 0">${ev.name}</h1>
      ${raw(ev.notes ? `<p class="sub">${esc(ev.notes)}</p>` : "")}
    </div>
    ${raw(ribbonFor(ev, candidates, tab))}
    ${raw(body)}`;
}

function ribbonFor(ev, candidates, tab) {
  const items = ev.items || [];
  const scored = (ev.scores || []).length;
  const body = group("This set", [
    // Not disabled when there are no runs any more: these prompts can be put
    // to a model off the Hub or one behind an API without this studio having
    // trained anything at all.
    rb("goScore", "◎", "Score models", { cls: "primary",
      title: "Put these prompts to a model" }),
    rb("copyEval", "⧉", "Copy",
      { title: "An editable copy, so scores already taken keep their meaning" }),
    rb("deleteEval", "🗑", "Delete", { cls: "danger", disabled: !ev.mine }),
  ]) + group("Compare", [
    rb(null, "⚖", "Held-out loss", { href: "#/compare" }),
    rb(null, "≡", "Runs", { href: "#/jobs" }),
  ]);
  return ribbon({
    tabs: TABS, active: tab, body,
    right: `<span class="badge">${items.length} prompts</span>
            <span class="badge">${scored} scoring${scored === 1 ? "" : "s"}</span>
            ${shareButton("eval", ev)}`,
  });
}

/** Choosing what to ask, and how. */
function runPanel(ev, candidates, state) {
  const items = ev.items || [];
  const { picked, baselines, hosted } = state;
  const none = !candidates.length;
  const chosen = picked.size + baselines.length;
  return html`
    ${raw(none ? html`
      <div class="card empty"><div class="big" aria-hidden="true">🌱</div>
        <h3>No finished models of your own yet</h3>
        <p class="muted">Train something and it appears here. In the meantime
          these prompts can still be put to models you have not trained —
          anything on the Hub, or a model behind a connected API.</p>
        <p><a class="btn" href="#/new">Start a run</a></p></div>` : html`
      <div class="card">
        <p class="muted tiny">Each model is loaded onto a machine in turn and
          asked all ${items.length} prompts. That takes a while, so it runs as
          a queued job you can watch and stop.</p>
        <div class="picklist" style="margin-top:10px">
          ${raw(candidates.map((c) => html`
            <label class="check">
              <input type="checkbox" data-model-pick value="${c.id}"${
                raw(picked.has(c.id) ? " checked" : "")}>
              <span>${c.name}
                <span class="muted tiny">· ${c.kind === "pretrain_llm"
                  ? "built from scratch" : (c.base_model || "fine-tune")}
                  ${raw(c.stopped_early ? ' · <span class="badge">stopped early</span>' : "")}
                </span></span>
            </label>`).join(""))}
        </div>
      </div>`)}
    ${raw(baselinePanel(candidates, picked, baselines, hosted))}
    <div class="card">
      <details class="adv">
        <summary>How they are asked</summary>
        <div class="field">
          <label for="evSystem">System prompt for every model
            <span class="muted tiny">(optional)</span></label>
          <input id="evSystem" type="text"
                 placeholder="Left empty, each model gets the one it was trained with">
          <div class="hint">Left empty, every run is asked with its own
            recorded system prompt and every baseline with none — which is how
            each model is actually meant to be used. Filling this in overrides
            all of them, which is a fair test of a specific instruction and a
            different measurement from the one above.</div>
        </div>
        ${raw((hosted.connected || []).length ? html`
          <div class="field">
            <label for="evJudge">A judge <span class="muted tiny">(optional)</span></label>
            <div class="row" style="gap:8px">
              <select id="evJudge">
                <option value="">none — measured numbers only</option>
                ${raw((hosted.connected || []).map((c) => html`
                  <option value="${c.provider}">${((hosted.providers || []).find((p) => p.id === c.provider) || {}).label || c.provider}</option>`).join(""))}
              </select>
              <input id="evJudgeModel" type="text" class="mono" style="flex:1"
                     value="${esc((hosted.connected || [])[0]?.model || "")}"
                     placeholder="which model judges">
            </div>
            <textarea id="evRubric" rows="2" style="margin-top:6px"
              placeholder="What a good answer looks like — optional. The judge grades one to five."></textarea>
            <div class="hint">A hosted model reads each answer and gives it a
              grade. Use it where nothing else can measure — free text with no
              single right answer. The grade is that judge's opinion: shown
              beside the measured numbers, never instead of them, and the
              judge's name is recorded with every score.</div>
          </div>` : "")}
        <div class="row" style="gap:10px">
          <div class="field" style="flex:1">
            <label for="evMaxTokens">Longest answer</label>
            <input id="evMaxTokens" type="number" value="200" min="16" max="512">
          </div>
          <div class="field" style="flex:1">
            <label for="evTemp">Temperature</label>
            <input id="evTemp" type="number" value="0" min="0" max="2" step="0.1">
            <div class="hint">Zero means the same answer every time, which is
              what makes two scorings comparable.</div>
          </div>
        </div>
      </details>
      <button class="btn-primary btn-sm" id="runEval" style="margin-top:10px">
        ${chosen === 0 ? "Score models"
          : chosen === 1 ? "Score 1 model" : `Score ${chosen} models`}</button>
    </div>`;
}

/** Models with no run behind them: the thing to be better *than*. */
function baselinePanel(candidates, picked, baselines, hosted) {
  // The base models of whatever is ticked, offered by name. This is the
  // comparison somebody actually wants and the one that was impossible: not
  // "which of my two fine-tunes won" but "did fine-tuning help at all".
  const suggested = [];
  for (const c of candidates) {
    if (!picked.has(c.id) || !c.base_model) continue;
    if (suggested.some((s) => s.model === c.base_model)) continue;
    if (baselines.some((b) => b.model === c.base_model)) continue;
    suggested.push({ model: c.base_model, run: c.id, name: c.name });
  }
  const connected = hosted.connected || [];
  const labelOf = Object.fromEntries(
    (hosted.providers || []).map((p) => [p.id, p.label]));

  return html`
    <div class="card">
      <div class="row-between" style="margin-bottom:6px">
        <h3 style="margin:0">Compare against</h3>
        <span class="tiny muted">optional</span>
      </div>
      <p class="muted tiny">A model that is not a run of this studio, asked the
        same prompts. Without one, a comparison can only say which of your own
        models won — never whether any of them beat what you started from.</p>

      ${raw(suggested.length ? html`
        <div class="row" style="gap:6px;flex-wrap:wrap;margin-top:10px">
          ${raw(suggested.map((sg) => html`
            <button class="btn-sm" data-add-base="${sg.model}"
                    data-like-run="${sg.run}"
                    title="The model ${esc(sg.name)} was trained from, asked in the same format that run was trained in">
              + ${sg.model}</button>`).join(""))}
        </div>` : "")}

      ${raw(baselines.length ? html`
        <div class="picklist" style="margin-top:10px">
          ${raw(baselines.map((b) => html`
            <div class="row-between" style="padding:4px 0">
              <span>${b.source === "api" ? "◇" : "⌂"} ${b.model}
                <span class="muted tiny">· ${b.source === "api"
                  ? `${labelOf[b.provider] || b.provider} · scored on what it writes, not on loss`
                  : "from the Hub"}</span></span>
              <button class="btn-sm btn-danger" data-drop-base="${b.key}"
                      title="Remove">✕</button>
            </div>`).join(""))}
        </div>` : "")}

      <div class="row" style="gap:8px;margin-top:10px;align-items:flex-end">
        <div class="field" style="flex:1;margin:0">
          <label for="hubModel">A model on the Hub</label>
          <input id="hubModel" type="text" placeholder="owner/name">
        </div>
        <button class="btn-sm" id="addHub">Add</button>
      </div>

      ${raw(connected.length ? html`
        <div class="row" style="gap:8px;margin-top:8px;align-items:flex-end">
          <div class="field" style="margin:0">
            <label for="apiProvider">A hosted model</label>
            <select id="apiProvider">
              ${raw(connected.map((c) => html`
                <option value="${c.provider}">${labelOf[c.provider] || c.provider}</option>`).join(""))}
            </select>
          </div>
          <div class="field" style="flex:1;margin:0">
            <label for="apiModel">Which model</label>
            <input id="apiModel" type="text"
                   value="${connected[0].model || ""}"
                   placeholder="exactly as the provider names it">
          </div>
          <button class="btn-sm" id="addApi">Add</button>
        </div>
        <p class="muted tiny" style="margin-top:6px">A hosted model is billed to
          the account you connected, and it cannot be scored on the loss — that
          needs the model's own probabilities, which no provider hands out. It
          is scored on what it writes.</p>`
        : html`<p class="muted tiny" style="margin-top:8px">
          <a href="#/account">Connect a provider</a> to compare against a
          hosted model too.</p>`)}
    </div>`;
}

/** Is this prompt set's best model getting better?
 *
 *  The table answers "which of these won". It cannot answer the question a
 *  prompt set exists for, which is whether three months of work moved the
 *  number at all -- for that the scorings have to be read in the order they
 *  happened, with the best-so-far drawn beside them. Two series, one unit,
 *  which is the only condition under which this chart puts two lines on one
 *  plot.
 */
function trendPanel(ev) {
  const scores = (ev.scores || []).slice()
    .sort((a, b) => a.created_at - b.created_at);
  const key = (ev.scores?.[0]?.metrics || {}).ranked_by || "expected_loss";
  const measure = MEASURES[key] || MEASURES.expected_loss;
  const usable = scores.filter((s) => s.metrics[key] != null);
  if (usable.length < 2) {
    return emptyState({
      icon: "📉",
      title: "Not enough scorings yet",
      body: "Two of them, and this is where the line goes. It is the question "
          + "a prompt set exists to answer: not which model won today, but "
          + "whether the best one is better than the best one last month.",
    });
  }

  // One model at a time, when asked. The chart has two validated series
  // colours and a prompt set may have been used by ten models; a line per
  // model would be a tangle in colours nobody could tell apart, so the
  // choice is every scoring together, or one model's own history.
  const models = [...new Map(usable.map((s) => [s.model_ref, s.model_name])).entries()];
  const only = ev._trendOnly || "";
  const shown = only ? usable.filter((s) => s.model_ref === only) : usable;
  const best = (shown.length ? shown : usable).reduce((a, b) =>
    (measure.lower ? b.metrics[key] < a.metrics[key]
                   : b.metrics[key] > a.metrics[key]) ? b : a);
  const first = (shown.length ? shown : usable)[0];
  const moved = measure.lower ? first.metrics[key] - best.metrics[key]
                              : best.metrics[key] - first.metrics[key];

  return html`
    <div class="card" style="margin-bottom:14px">
      ${raw(models.length > 1 ? html`
        <div class="row-between" style="margin-bottom:6px">
          <span class="tiny muted">Show</span>
          <select id="trendOnly" class="tiny">
            <option value="">every model</option>
            ${raw(models.map(([ref, name]) => html`
              <option value="${ref}"${ref === only ? " selected" : ""}>${name}</option>`).join(""))}
          </select>
        </div>` : "")}
      <div id="trendChart" data-trend="${esc(key)}" data-only="${esc(only)}"></div>
      <p class="muted tiny" style="margin:8px 0 0">Every scoring of this set,
        in the order it happened. The second line is the best result so far,
        which is the one that answers whether the work is going anywhere —
        a single bad scoring is a bad model, not a regression.</p>
    </div>
    <div class="card">
      <div class="row-between" style="flex-wrap:wrap;gap:10px">
        <div>
          <h3 style="margin:0 0 4px">Best on this set</h3>
          <div>${raw(best.is_run
            ? `<a href="#/jobs/${esc(best.model_job_id)}">${esc(best.model_name)}</a>`
            : `<span>${esc(best.model_name)}</span>`)}
            ${raw(baselineBadge(best))}
            <span class="muted tiny">· ${measure.label.toLowerCase()}
              ${measure.fmt(best.metrics[key])} · scored ${fmtAgo(best.created_at)}</span>
          </div>
          <div class="muted tiny" style="margin-top:4px">
            ${moved > 0
              ? `${measure.fmt(Math.abs(moved))} better than the first scoring on this set.`
              : "No better than the first scoring on this set."}</div>
        </div>
        ${raw(best.is_run ? html`
          <button class="btn btn-primary btn-sm" id="promoteBest"
                  data-job="${esc(best.model_job_id)}" data-name="${esc(best.model_name)}">Serve it under a name</button>`
          : `<span class="muted tiny">A baseline, not a run here — there is
             nothing of yours to serve.</span>`)}
      </div>
    </div>`;
}

/** Where a benchmark's questions come from, since they are not held here. */
function benchmarkPrompts(ev) {
  const r = (ev.source || {}).recipe || {};
  return html`
    <div class="card">
      <h3 style="margin:0 0 8px">The questions are not stored here</h3>
      <p class="muted tiny">They are fetched from
        <a href="https://huggingface.co/datasets/${esc(r.dataset)}"
           target="_blank" rel="noopener">${esc(r.dataset)}</a> by whichever
        machine runs the scoring — the same way the tools that publish these
        numbers get them. Keeping a copy here would let it drift from the
        dataset everybody else is measuring against.</p>
      <table class="table" style="margin-top:10px"><tbody>
        ${raw([
          ["Dataset", `${esc(r.dataset)}${r.config ? ` · ${esc(r.config)}` : ""}`],
          ["Split", esc(r.split || "")],
          ["Worked examples", `${r.shots} from the ${esc(r.fewshot_split || "same")} split`],
          ["Questions asked", `${fmtNum(r.sample)} of them, chosen with seed ${r.seed}`],
          ["Scored by", r.protocol === "multiple_choice"
            ? (r.style === "letter"
               ? "the probability the model gives each answer's letter"
               : "the probability of each answer, normalised by its length")
            : "the last number in what the model wrote"],
        ].map(([k, v]) => `<tr><td class="muted tiny">${k}</td><td class="tiny">${v}</td></tr>`).join(""))}
      </tbody></table>
      <p class="muted tiny" style="margin-top:10px">Every one of those lines
        moves the score. That is why they are written down rather than
        assumed.</p>
    </div>`;
}

function promptsPanel(items, answered) {
  return html`
    <div class="card">
      <p class="muted tiny">${answered} of ${items.length} have an expected
        answer. Prompts without one still get asked, and their answers are kept
        to read — they simply cannot be scored.</p>
      <div class="promptlist">
        ${raw(items.slice(0, 200).map((i) => html`
          <div class="prompt-row">
            <div class="p">${i.prompt}</div>
            ${raw(i.expected
              ? `<div class="e">→ ${esc(i.expected)}</div>`
              : `<div class="e muted">no expected answer</div>`)}
          </div>`).join(""))}
        ${raw(items.length > 200
          ? `<p class="muted tiny">…and ${items.length - 200} more.</p>` : "")}
      </div>
    </div>`;
}

// Below this, a difference between two models is a difference between a
// handful of prompts. The runner says so in words after each scoring; the
// table has to stop drawing a winner's rosette on it.
const ENOUGH_PROMPTS = 10;

// What each measure is called, whether more is better, and how to print it.
// One table, because the column heading, the bar, the winner's rosette and
// the sentence underneath all have to agree about it.
const MEASURES = {
  accuracy: { label: "Accuracy", lower: false,
              fmt: (v) => (v * 100).toFixed(1) + "%" },
  expected_loss: { label: "Loss on expected", lower: true,
                   fmt: (v) => v.toFixed(4) },
  chrf: { label: "Character overlap", lower: false,
          fmt: (v) => (v * 100).toFixed(0) + "%" },
  f1: { label: "Token overlap", lower: false,
        fmt: (v) => (v * 100).toFixed(0) + "%" },
  judge_score: { label: "Judge's score", lower: false, fmt: (v) => v.toFixed(2) + "/5" },
};

/** A published benchmark's results: one number, and how wide it is. */
function benchmarkTable(ev, scores) {
  const recipe = (ev.source || {}).recipe || {};
  if (!scores.length) {
    return html`
      <div class="card empty" style="margin-bottom:14px">
        <div class="big">📊</div>
        <h3>Not run yet</h3>
        <p class="muted">Pick some models on the next tab. Every model is
          asked the same ${fmtNum(recipe.sample)} questions, chosen with the
          same seed.</p>
      </div>`;
  }
  const latest = scores[0]?.metrics || {};
  const decisive = latest.ranking_decisive === true;
  const best = Math.max(...scores.map((s) => s.metrics.accuracy ?? -1));

  return html`
    <div class="card" style="margin-bottom:14px;padding:0">
      <div class="row-between" style="padding:14px 16px 0">
        <h3 style="margin:0">${esc(recipe.label || "Benchmark")}</h3>
        <span class="tiny muted">${recipe.shots}-shot ·
          ${fmtNum(recipe.sample)} questions · seed ${recipe.seed}</span>
      </div>
      <div class="table-wrap"><table>
        <thead><tr>
          <th>Model</th><th>Accuracy</th>
          <th class="hide-sm" title="95% confidence interval on this sample">Within</th>
          <th class="hide-sm">Asked</th><th class="hide-sm">Took</th>
          <th>When</th><th></th>
        </tr></thead>
        <tbody>
          ${raw(scores.map((s) => {
            const m = s.metrics || {};
            const isBest = decisive && m.accuracy === best && scores.length > 1;
            return html`
              <tr class="${isBest ? "row-best" : ""}">
                <td>
                  ${raw(s.is_run
                    ? `<a href="#/jobs/${esc(s.model_job_id)}">${esc(s.model_name)}</a>`
                    : `<span>${esc(s.model_name)}</span>`)}
                  ${raw(isBest ? ` <span class="badge badge-ok">best</span>` : "")}
                  ${raw(baselineBadge(s))}
                </td>
                <td style="min-width:130px">
                  ${raw(m.accuracy != null ? html`
                    <strong>${(m.accuracy * 100).toFixed(1)}%</strong>
                    <div class="meter"><i style="width:${(m.accuracy * 100).toFixed(1)}%"></i></div>`
                    : `<span class="muted">—</span>`)}
                </td>
                <td class="hide-sm tiny muted">${m.accuracy_low != null
                  ? `${(m.accuracy_low * 100).toFixed(1)}–${(m.accuracy_high * 100).toFixed(1)}%`
                  : "—"}</td>
                <td class="hide-sm tiny">${fmtNum(m.items || 0)}</td>
                <td class="hide-sm tiny muted">${m.seconds
                  ? fmtDuration(m.seconds) : "—"}</td>
                <td class="tiny muted">${fmtAgo(s.created_at)}</td>
                <td><div class="row" style="gap:4px">
                  <button class="btn-sm" data-open-score="${s.id}">Answers</button>
                  <button class="btn-sm btn-danger" data-del-score="${s.id}"
                          title="Remove from the comparison">✕</button>
                </div></td>
              </tr>`;
          }).join(""))}
        </tbody>
      </table></div>
      ${raw(latest.verdict ? html`
        <div class="callout ${decisive ? "callout-ok" : "callout-warn"}"
             style="margin:0 16px 12px">
          <strong>${decisive ? "This sample separated them"
                             : "This sample could not separate them"}</strong>
          ${latest.verdict}
        </div>` : "")}
      <p class="muted tiny" style="padding:10px 16px 14px;margin:0">
        Measured here, not copied from anywhere: ${esc(recipe.dataset)},
        ${esc(recipe.split)} split, ${recipe.shots}-shot,
        ${recipe.protocol === "multiple_choice"
          ? (recipe.style === "letter"
             ? "scored on the probability of each answer's letter"
             : "scored on the probability of each answer, normalised by its length")
          : "scored on the last number in what the model wrote"}.
        A published score uses a different harness and will differ by
        several points; these are exactly comparable to each other.</p>
    </div>`;
}

function scoreTable(scores, answered, total) {
  if (!scores.length) {
    return html`
      <div class="card empty" style="margin-bottom:14px">
        <div class="big">📊</div>
        <h3>Nothing scored yet</h3>
        <p class="muted">Pick some models below. Once two of them have answered
          these prompts, this is where the comparison appears.</p>
      </div>`;
  }

  const latest = scores[0]?.metrics || {};
  // The measure the runner ranked on. Older scores, taken before a scoring
  // could fall back from the loss, recorded nothing here and were always
  // ranked on the loss.
  const key = latest.ranked_by || "expected_loss";
  const measure = MEASURES[key] || MEASURES.expected_loss;
  const usable = scores.filter((s) => s.metrics[key] != null);
  const values = usable.map((s) => s.metrics[key]);
  const best = values.length
    ? (measure.lower ? Math.min(...values) : Math.max(...values)) : null;
  const worst = values.length
    ? (measure.lower ? Math.max(...values) : Math.min(...values)) : null;
  // The bar is a magnitude, so it gets one hue and a shared scale. Anchored at
  // zero would make every model look identical -- the differences that matter
  // between two trained models are small in absolute terms.
  const span = Math.abs((worst ?? 0) - (best ?? 0)) || 1;
  // Two conditions, and both are the runner's own judgement rather than this
  // page's: enough prompts to be worth measuring, and a most-recent scoring
  // whose difference actually survived being measured against the spread
  // between prompts. Marking a winner the log has just called a coin toss
  // would be the table contradicting its own evidence.
  const decisive = total >= ENOUGH_PROMPTS && latest.ranking_decisive !== false;
  const verdict = latest.verdict;
  const anyJson = scores.some((s) => s.metrics.json_valid != null);
  const anyJudge = scores.some((s) => s.metrics.judge_score != null);
  const anySchema = scores.some((s) => s.metrics.schema_valid != null);
  const anyTool = scores.some((s) => s.metrics.tool_name_ok != null);

  return html`
    <div class="card" style="margin-bottom:14px;padding:0">
      <div class="row-between" style="padding:14px 16px 0">
        <h3 style="margin:0">Results</h3>
        <span class="tiny muted">newest first · ranked on
          ${measure.label.toLowerCase()}</span>
      </div>
      <div class="table-wrap"><table>
        <thead><tr>
          <th>Model</th>
          <th title="Teacher-forced loss on the answer you called correct">
            Loss on expected</th>
          <th class="hide-sm" title="Character n-gram overlap: survives a right answer worded or inflected differently">
            chrF</th>
          <th class="hide-sm" title="Token overlap with the expected answer">Overlap</th>
          <th class="hide-sm">Exact</th>
          ${raw(anyJson ? `<th class="hide-sm" title="Answers that parsed as JSON, and answers that parsed to the expected value">JSON</th>` : "")}
          ${raw(anySchema ? `<th class="hide-sm" title="Answers that fit the shape the prompt asked for">Schema</th>` : "")}
          ${raw(anyTool ? `<th class="hide-sm" title="Called the tool the prompt expected; and with the expected arguments">Tool</th>` : "")}
          ${raw(anyJudge ? `<th title="A hosted model's grade, one to five. Its opinion, not a measurement.">Judge</th>` : "")}
          <th class="hide-sm">Speed</th>
          <th>When</th><th></th>
        </tr></thead>
        <tbody>
          ${raw(scores.map((s) => {
            const m = s.metrics || {};
            const isBest = decisive && best !== null && m[key] === best
                           && usable.length > 1;
            const width = m[key] != null
              ? 12 + 88 * (1 - Math.abs(m[key] - best) / span) : 0;
            return html`
              <tr class="${isBest ? "row-best" : ""}">
                <td>
                  ${raw(s.is_run
                    ? `<a href="#/jobs/${esc(s.model_job_id)}">${esc(s.model_name)}</a>`
                    : `<span>${esc(s.model_name)}</span>`)}
                  ${raw(isBest ? ` <span class="badge badge-ok">best</span>` : "")}
                  ${raw(baselineBadge(s))}
                  ${raw(s.model_gone
                    ? ` <span class="badge badge-warn">run deleted</span>` : "")}
                  ${raw(settingsNote(s))}
                </td>
                <td style="min-width:150px">
                  ${raw(m.expected_loss != null ? html`
                    <strong>${m.expected_loss.toFixed(4)}</strong>
                    <div class="muted tiny">${m.expected_perplexity != null
                      ? `1 in ${m.expected_perplexity} surprise` : ""}</div>
                    ${raw(key === "expected_loss" ? html`
                      <div class="meter"><i style="width:${width.toFixed(1)}%"></i></div>` : "")}`
                    : m.loss_unavailable
                      ? `<span class="muted tiny" title="${esc(m.loss_unavailable)}">not measurable</span>`
                      : `<span class="muted">—</span>`)}
                </td>
                <td class="hide-sm">${m.chrf != null ? (m.chrf * 100).toFixed(0) + "%" : "—"}
                  ${raw(key === "chrf" && m.chrf != null
                    ? `<div class="meter"><i style="width:${width.toFixed(1)}%"></i></div>` : "")}</td>
                <td class="hide-sm">${m.f1 != null ? (m.f1 * 100).toFixed(0) + "%" : "—"}
                  ${raw(key === "f1" && m.f1 != null
                    ? `<div class="meter"><i style="width:${width.toFixed(1)}%"></i></div>` : "")}</td>
                <td class="hide-sm">${m.exact != null ? (m.exact * 100).toFixed(0) + "%" : "—"}</td>
                ${raw(anyJson ? html`
                  <td class="hide-sm tiny">${m.json_valid != null
                    ? `${(m.json_valid * 100).toFixed(0)}% valid`
                    : "—"}${raw(m.json_match != null
                      ? `<div class="muted">${(m.json_match * 100).toFixed(0)}% right</div>` : "")}</td>` : "")}
                ${raw(anySchema ? html`
                  <td class="hide-sm tiny">${m.schema_valid != null ? `${(m.schema_valid * 100).toFixed(0)}%` : "—"}</td>` : "")}
                ${raw(anyTool ? html`
                  <td class="hide-sm tiny">${m.tool_name_ok != null ? `${(m.tool_name_ok * 100).toFixed(0)}% right tool` : "—"}${raw(
                    m.tool_args_ok != null ? `<div class="muted">${(m.tool_args_ok * 100).toFixed(0)}% right arguments</div>` : "")}</td>` : "")}
                ${raw(anyJudge ? html`
                  <td>${m.judge_score != null ? html`<strong>${m.judge_score.toFixed(2)}</strong><span class="muted tiny"> /5</span>${raw(
                    key === "judge_score" ? `<div class="meter"><i style="width:${width.toFixed(1)}%"></i></div>` : "")}${raw(
                    s.settings?.judge ? `<div class="muted tiny">${esc(s.settings.judge)}</div>` : "")}` : `<span class="muted">—</span>`}</td>` : "")}
                <td class="hide-sm tiny muted">${m.tokens_per_sec
                  ? m.tokens_per_sec.toFixed(0) + " tok/s" : "—"}
                  ${raw(m.seconds ? `<div>${esc(fmtDuration(m.seconds))}</div>` : "")}</td>
                <td class="tiny muted">${fmtAgo(s.created_at)}</td>
                <td><div class="row" style="gap:4px">
                  <button class="btn-sm" data-open-score="${s.id}">Answers</button>
                  <button class="btn-sm btn-danger" data-del-score="${s.id}"
                          title="Remove from the comparison">✕</button>
                </div></td>
              </tr>`;
          }).join(""))}
        </tbody>
      </table></div>
      ${raw(verdict ? html`
        <div class="callout ${decisive ? "callout-ok" : "callout-warn"}"
             style="margin:0 16px 12px">
          <strong>${decisive ? "The latest scoring separated them"
                             : "The latest scoring could not separate them"}</strong>
          ${verdict}
          ${raw(!decisive && total < ENOUGH_PROMPTS ? html`
            <div style="margin-top:6px">Around ${ENOUGH_PROMPTS} prompts is
              where a comparison starts to be worth reading, and more is
              better.</div>` : "")}
        </div>` : !decisive && scores.length > 1 ? html`
        <div class="callout callout-warn" style="margin:0 16px 12px">
          <strong>${total} prompt${total === 1 ? "" : "s"} is not enough to
          rank these.</strong> The gap between two trained models is usually
          smaller than the gap between one prompt and the next, so with a set
          this small the lowest number is as likely to be luck as skill.</div>`
        : "")}
      ${raw(scores.length === 1 && scores[0].is_run ? html`
        <div class="callout" style="margin:0 16px 12px">
          <strong>One model is not a comparison.</strong> Score it against the
          model it was trained from — that is the number that says whether the
          training helped.</div>` : "")}
      <p class="muted tiny" style="padding:10px 16px 14px;margin:0">
        ${raw(answered
          ? html`<strong>Loss on expected</strong> is how surprised the model was
              by the answer you called correct, scored on the answer only. It is
              the measure to trust: it does not care about wording, and it can
              separate two models that both scored zero exact matches.
              <strong>chrF</strong> and <strong>Overlap</strong> credit a right
              answer worded differently, and just as happily credit a wrong
              answer that reuses the right words.`
          : html`None of these prompts has an expected answer, so there is
              nothing to score against — only the text each model produced.
              Add expected answers and score again to get numbers.`)}</p>
    </div>`;
}

/** Where a scored model came from, when it was not a run of this studio. */
function baselineBadge(s) {
  const ref = s.model_ref || "";
  if (ref.startsWith("hub:")) {
    return ` <span class="badge" title="A model off the Hub, scored as a baseline">baseline</span>`;
  }
  if (ref.startsWith("api:")) {
    return ` <span class="badge" title="Reached over the network. No loss: a hosted model does not expose its probabilities">hosted</span>`;
  }
  return "";
}

/** The settings that produced a score, when they were not the ordinary ones.
 *
 *  Two scorings of one model under different settings are two different
 *  measurements, and the table putting them in adjacent rows has to say so.
 *  Only the differences are shown: a line reading "temperature 0" under every
 *  row is noise. */
function settingsNote(s) {
  const st = s.settings;
  if (!st) return "";
  const bits = [];
  if (st.system_prompt_override) bits.push("a system prompt set for the scoring");
  else if (st.system_prompt) bits.push("its own system prompt");
  if (st.temperature) bits.push(`temperature ${st.temperature}`);
  if (!bits.length) return "";
  return `<div class="muted tiny">${esc(bits.join(" · "))}</div>`;
}

function scoreDetail(score) {
  return html`
    <div class="card" style="margin-bottom:14px">
      <div class="row-between" style="margin-bottom:8px">
        <h3 style="margin:0">What ${score.model_name || "it"} answered</h3>
        <span class="tiny muted">${(score.items || []).length} prompts</span>
      </div>
      <div class="table-wrap"><table>
        <thead><tr><th>Prompt</th><th>Expected</th><th>Answered</th><th>Score</th></tr></thead>
        <tbody>
          ${raw((score.items || []).map((i) => html`
            <tr>
              <td class="tiny">${i.prompt}</td>
              <td class="tiny muted">${i.expected || "—"}</td>
              <td class="tiny">${i.answer || "(nothing)"}</td>
              <td class="tiny mono">
                ${raw(i.expected_loss != null
                  ? `${i.expected_loss.toFixed(3)}` : "—")}
                ${raw(i.chrf != null ? ` ${(i.chrf * 100).toFixed(0)}%` : "")}
                ${raw(i.exact ? ` <span class="badge badge-ok">exact</span>`
                  : i.contains ? ` <span class="badge">contains</span>` : "")}
                ${raw(i.json_valid === false
                  ? ` <span class="badge badge-warn">not JSON</span>` : "")}
                ${raw(i.schema_valid === false
                  ? ` <span class="badge badge-warn" title="${esc(i.schema_error || "")}">off-schema</span>`
                  : i.schema_valid ? ` <span class="badge badge-ok">fits</span>` : "")}
                ${raw(i.tool_name_ok === true ? ` <span class="badge badge-ok">right tool${i.tool_args_ok === false ? ", wrong arguments" : ""}</span>`
                  : i.tool_name_ok === false ? ` <span class="badge badge-warn">${i.tool_called ? "wrong tool" : "no call"}</span>` : "")}
                ${raw(i.judge_score != null
                  ? ` <span class="badge" title="${esc(i.judge_reason || "")}">judge ${i.judge_score}/5</span>` : "")}
              </td>
            </tr>`).join(""))}
        </tbody>
      </table></div>
    </div>`;
}
