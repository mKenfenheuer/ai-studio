import { api } from "../api.js";
import { html, raw, esc, on, $, $$, fmtNum, fmtDuration, toast } from "../util.js";

// Two fundamentally different jobs behind one wizard. They share a machine
// picker and a review step; everything between them differs, because the
// decisions differ. Fine-tuning asks "which model, which examples"; training
// from scratch asks "how big, and how long are you prepared to wait" -- and
// that second question has no counterpart in fine-tuning at all.
const MODES = [
  { id: "finetune", icon: "🎯", title: "Improve an existing model",
    desc: "Start from a model that already understands language and teach it "
        + "your task. This is what you want almost every time.",
    hint: "Minutes to hours" },
  { id: "scratch", icon: "🌱", title: "Build a model from scratch",
    desc: "Start from random numbers and teach a brand-new model to write, "
        + "using only the text you give it. Slower, and only small models are "
        + "realistic on one graphics card — but the result is entirely yours.",
    hint: "Hours · small models only" },
];

// Goals are phrased as outcomes, not techniques. Someone who has never trained
// a model knows what they want to achieve; they do not know what "SFT" means.
const GOALS = [
  { id: "instructions", icon: "💬", title: "Follow instructions better",
    desc: "Teach the model to answer questions and carry out tasks the way your examples do.",
    hint: "The most common starting point." },
  { id: "style", icon: "✍️", title: "Write in a particular style",
    desc: "Make the model imitate a tone or voice — your company's, or your own.",
    hint: "Works well with only a few hundred examples." },
  { id: "domain", icon: "🔬", title: "Learn a specialist subject",
    desc: "Adapt the model to vocabulary and knowledge from your own field.",
    hint: "Needs more examples than the others." },
  { id: "format", icon: "📐", title: "Always answer in a fixed format",
    desc: "Force replies into a shape you can parse, like JSON or a template.",
    hint: "Very reliable, even with small datasets." },
];

const STEPS = {
  finetune: ["Goal", "Model", "Data", "Review"],
  scratch: ["Goal", "Text", "Size", "Review"],
};

const VERDICT_CLASS = { ok: "badge-ok", warn: "badge-warn", err: "badge-err" };

const runnerName = (runners, id) =>
  (runners.find((r) => r.id === id) || {}).name || "this machine";

export async function wizardView(mount) {
  const state = {
    mode: "finetune", step: 0, goal: "instructions", runnerId: null,
    model: null, modelDetail: null,
    dataset: null, preview: null,
    plan: null, overrides: {},
    // from-scratch only
    corpus: null, corpusInfo: null, minutes: 60, size: null, vocab: 8192,
    sizes: null, sizesKey: null, scratchPlan: null,
  };

  const runners = (await api.runners()).filter((r) => r.status !== "offline");
  const starters = await api.starters();
  state.runnerId = runners[0]?.id ?? null;
  state.vocab = starters.default_vocab || 8192;

  const draw = () => {
    mount.innerHTML = shell(state, runners);
    stepBody(mount, state, runners, starters, draw);
    wireNav(mount, state, draw);
  };
  draw();
  return () => {};
}

function shell(state, runners) {
  if (!runners.length) {
    return html`
      <div class="page-head"><h1>New training run</h1></div>
      <div class="card empty">
        <div class="big">🔌</div>
        <h2>No machines are connected</h2>
        <p class="muted">A training run needs a machine with a GPU. Connect one
          from the <a href="#/runners">Machines</a> page — it takes one command.</p>
        <p><a class="btn btn-primary" href="#/runners">Connect a machine</a></p>
      </div>`;
  }
  const steps = STEPS[state.mode];
  const chips = steps.map((s, i) => html`
    <div class="step-chip ${i === state.step ? "current" : i < state.step ? "done" : ""}">
      <span class="n">${i < state.step ? "✓" : String(i + 1)}</span>${s}
    </div>`).join("");
  return html`
    <div class="page-head">
      <h1>New training run</h1>
      <p class="sub">Four steps. Everything technical is chosen for you, and
        every choice is explained.</p>
    </div>
    <div class="steps">${raw(chips)}</div>
    <div id="stepBody"></div>
    <div class="wizard-actions">
      <button id="backBtn" ${state.step === 0 ? "disabled" : ""}>← Back</button>
      <div class="spacer"></div>
      <span id="navHint" class="tiny muted"></span>
      <button id="nextBtn" class="btn-primary btn-lg">
        ${state.step === steps.length - 1 ? "Start training" : "Continue →"}
      </button>
    </div>`;
}

// ===========================================================================

function stepBody(mount, state, runners, starters, draw) {
  const body = $("#stepBody", mount);
  const steps = state.mode === "scratch"
    ? { 0: stepGoal, 1: stepCorpus, 2: stepSize, 3: stepScratchReview }
    : { 0: stepGoal, 1: stepModel, 2: stepData, 3: stepReview };
  steps[state.step](body, state, runners, starters, draw);
}

// ---- 1. what kind of run, and on which machine ----------------------------
function stepGoal(body, state, runners, starters, draw) {
  const scratch = state.mode === "scratch";
  body.innerHTML = html`
    <div class="grid grid-2" style="margin-bottom:22px">
      ${raw(MODES.map((m) => html`
        <button class="pick ${state.mode === m.id ? "selected" : ""}" data-mode="${m.id}">
          <span class="t"><span style="font-size:17px">${m.icon}</span>${m.title}</span>
          <span class="d">${m.desc}</span>
          <span class="badge">${m.hint}</span>
        </button>`).join(""))}
    </div>

    ${raw(scratch ? html`
      <div class="callout" style="margin-bottom:22px">
        <strong>What "from scratch" really means</strong>
        The model starts as pure noise and has to learn spelling, grammar and
        meaning from nothing but the text you choose. That is far more work
        than fine-tuning, so on a single graphics card only small models — up
        to a few hundred million parameters — can be trained properly. A small
        model trained well beats a large one trained badly, and the next steps
        will show you exactly where that line falls on your machine.
      </div>` : html`
      <div class="grid grid-2" style="margin-bottom:22px">
        ${raw(GOALS.map((g) => html`
          <button class="pick ${state.goal === g.id ? "selected" : ""}" data-goal="${g.id}">
            <span class="t"><span style="font-size:17px">${g.icon}</span>${g.title}</span>
            <span class="d">${g.desc}</span>
            <span class="badge">${g.hint}</span>
          </button>`).join(""))}
      </div>`)}

    <div class="card">
      <h3>Which machine should do the work?</h3>
      <p class="muted tiny">Training runs on a connected machine's graphics card.
        What each one can handle differs, so this choice shapes the options later.</p>
      <div class="grid grid-2" style="margin-top:12px">
        ${raw(runners.map((r) => {
          const c = r.capabilities || {};
          const ceiling = scratch
            ? (c.max_scratch_params_m ? `up to ~${fmtNum(c.max_scratch_params_m * 1e6)} params` : "")
            : (c.max_finetune_params_b ? `up to ~${c.max_finetune_params_b}B params` : "");
          return html`
            <button class="pick ${state.runnerId === r.id ? "selected" : ""}"
                    data-runner="${r.id}">
              <span class="t">
                <span class="dot ${r.status === "busy" ? "dot-busy" : "dot-ok"}"></span>
                ${r.name}
              </span>
              <span class="d">${c.device_name || "Unknown device"}</span>
              <span class="row" style="gap:6px;flex-wrap:wrap">
                ${raw(c.vram_gb ? `<span class="badge">${c.vram_gb} GB memory</span>` : "")}
                ${raw(c.backend ? `<span class="badge">${esc(c.backend.toUpperCase())}</span>` : "")}
                ${raw(ceiling ? `<span class="badge badge-accent">${esc(ceiling)}</span>` : "")}
                ${raw(r.status === "busy" ? `<span class="badge badge-warn">busy</span>` : "")}
              </span>
            </button>`;
        }).join(""))}
      </div>
    </div>`;

  on(body, "click", "[data-mode]", (_e, t) => {
    if (state.mode === t.dataset.mode) return;
    state.mode = t.dataset.mode;
    // The two paths share nothing past this point, so a half-made choice from
    // the other one must not survive and quietly end up in the job config.
    Object.assign(state, { model: null, modelDetail: null, dataset: null,
      preview: null, plan: null, overrides: {}, corpus: null, corpusInfo: null,
      size: null, sizes: null, sizesKey: null, scratchPlan: null });
    draw();
  });
  on(body, "click", "[data-goal]", (_e, t) => { state.goal = t.dataset.goal; draw(); });
  on(body, "click", "[data-runner]", (_e, t) => {
    state.runnerId = t.dataset.runner;
    state.sizes = null; state.sizesKey = null;   // scored per machine
    draw();
  });
}

// ---- 2. model (fine-tune) -------------------------------------------------
function stepModel(body, state, runners, starters, draw) {
  const runner = runners.find((r) => r.id === state.runnerId);
  const caps = runner?.capabilities || {};
  const ceiling = caps.max_finetune_params_b;

  body.innerHTML = html`
    <div class="card" style="margin-bottom:14px">
      <div class="row-between">
        <div>
          <h3 style="margin:0">Choose a base model</h3>
          <p class="muted tiny" style="margin:2px 0 0">
            You are not starting from nothing — you start from a model that already
            understands language, and teach it your task.</p>
        </div>
        ${raw(ceiling ? `<span class="badge badge-accent">${esc(runner.name)} handles
          about ${ceiling}B parameters</span>` : "")}
      </div>
    </div>

    <div class="grid grid-2">
      ${raw(starters.models.map((m) => {
        const tooBig = ceiling && m.params_b > ceiling;
        return html`
          <button class="pick ${state.model === m.id ? "selected" : ""}"
                  data-model="${m.id}" data-params="${m.params_b}">
            <span class="t">${m.label}
              <span class="badge">${m.params_b < 1
                ? Math.round(m.params_b * 1000) + "M" : m.params_b + "B"}</span>
              ${raw(tooBig ? `<span class="badge badge-warn">likely too big</span>`
                           : `<span class="badge badge-ok">fits</span>`)}
            </span>
            <span class="d">${m.blurb}</span>
          </button>`;
      }).join(""))}
    </div>

    <details class="adv">
      <summary>Or search all of Hugging Face</summary>
      <div class="card" style="margin-top:10px">
        <div class="row">
          <input type="text" id="modelSearch" placeholder="e.g. llama, phi, gemma…">
          <button id="modelSearchBtn">Search</button>
        </div>
        <div id="modelResults" style="margin-top:12px"></div>
      </div>
    </details>

    <div id="modelInfo" style="margin-top:14px"></div>`;

  const choose = async (id, paramsB) => {
    state.model = id;
    state.modelDetail = null;
    draw();
    const info = $("#modelInfo");
    info.innerHTML = `<div class="card muted tiny">Checking whether this fits…</div>`;
    try {
      const d = await api.modelDetail(id);
      state.modelDetail = d;
      const fit = d.fit?.[state.runnerId];
      const cls = { fits: "callout-ok", fits_quantized: "callout",
                    needs_quantization: "callout-warn", too_big: "callout-err"
                  }[fit?.verdict] || "callout";
      info.innerHTML = html`
        <div class="card">
          <div class="row-between" style="margin-bottom:10px">
            <h3 style="margin:0">${d.id}</h3>
            ${raw(d.gated ? `<span class="badge badge-warn">licence required</span>` : "")}
          </div>
          ${raw(fit ? `<div class="callout ${cls}"><strong>Will it fit?</strong>${
            esc(fit.message)}</div>` : "")}
          <dl class="kv">
            <dt>Size</dt><dd>${d.params_b ? d.params_b + "B parameters" : "unknown"}</dd>
            <dt>Architecture</dt><dd>${d.architecture || "—"}</dd>
            <dt>Downloads</dt><dd>${fmtNum(d.downloads)}</dd>
            ${raw(d.memory ? `<dt>Memory needed</dt><dd>${d.memory.fp16_gb} GB in
              16-bit · ${d.memory.int4_gb} GB in 4-bit</dd>` : "")}
          </dl>
          ${raw(d.gated ? `<div class="callout callout-warn" style="margin-top:10px">
            <strong>This model is gated</strong>You must accept its licence on the
            Hugging Face model page, and add an access token in Settings, before
            training can download it.</div>` : "")}
        </div>`;
    } catch (e) {
      info.innerHTML = html`<div class="callout callout-err">
        <strong>Could not load that model</strong>${e.message}</div>`;
    }
  };

  on(body, "click", "[data-model]", (_e, t) => choose(t.dataset.model, +t.dataset.params));

  const doSearch = async () => {
    const q = $("#modelSearch", body).value.trim();
    const box = $("#modelResults", body);
    box.innerHTML = `<span class="muted tiny">Searching…</span>`;
    try {
      const rows = await api.searchModels(q);
      box.innerHTML = rows.length ? html`<div class="table-wrap"><table>
        <thead><tr><th>Model</th><th>Size</th><th>Downloads</th><th></th></tr></thead>
        <tbody>${raw(rows.map((r) => html`<tr>
          <td class="mono">${r.id}${raw(r.gated
            ? ` <span class="badge badge-warn">gated</span>` : "")}</td>
          <td>${r.params_b ? r.params_b + "B" : "—"}</td>
          <td>${fmtNum(r.downloads)}</td>
          <td><button class="btn-sm" data-model="${r.id}"
                      data-params="${r.params_b || ""}">Use this</button></td>
        </tr>`).join(""))}</tbody></table></div>`
        : `<span class="muted tiny">Nothing matched that search.</span>`;
    } catch (e) {
      box.innerHTML = html`<div class="callout callout-err">${e.message}</div>`;
    }
  };
  $("#modelSearchBtn", body)?.addEventListener("click", doSearch);
  $("#modelSearch", body)?.addEventListener("keydown", (e) => {
    if (e.key === "Enter") doSearch();
  });

  if (state.model && state.modelDetail) {
    // Re-render of an already-inspected model: restore the detail panel.
    choose(state.model, state.modelDetail.params_b);
  }
}

// ---- 3. data (fine-tune) --------------------------------------------------
function stepData(body, state, runners, starters, draw) {
  body.innerHTML = html`
    <div class="card" style="margin-bottom:14px">
      <h3 style="margin:0">Choose your examples</h3>
      <p class="muted tiny" style="margin:2px 0 0">
        This is what the model learns from. Quality matters far more than
        quantity — a few hundred good examples beat thousands of sloppy ones.</p>
    </div>

    <div class="grid grid-2">
      ${raw(starters.datasets.map((d) => html`
        <button class="pick ${state.dataset === d.id ? "selected" : ""}" data-ds="${d.id}">
          <span class="t">${d.label}<span class="badge">${fmtNum(d.rows)} examples</span></span>
          <span class="d">${d.blurb}</span>
        </button>`).join(""))}
    </div>

    <details class="adv">
      <summary>Or search all of Hugging Face</summary>
      <div class="card" style="margin-top:10px">
        <div class="row">
          <input type="text" id="dsSearch" placeholder="e.g. code, medical, german…">
          <button id="dsSearchBtn">Search</button>
        </div>
        <div id="dsResults" style="margin-top:12px"></div>
      </div>
    </details>

    <div id="dsPreview" style="margin-top:14px"></div>`;

  const choose = async (id) => {
    state.dataset = id;
    state.preview = null;
    draw();
    const box = $("#dsPreview");
    box.innerHTML = `<div class="card muted tiny">Loading a few real examples…</div>`;
    try {
      const p = await api.datasetPreview(id);
      state.preview = p;
      if (!p.available) {
        box.innerHTML = html`<div class="callout callout-warn">
          <strong>No preview available</strong>${p.reason || ""} You can still use
          it, but check the column names are right before a long run.</div>`;
        return;
      }
      const cols = p.columns.slice(0, 4);
      box.innerHTML = html`
        <div class="card">
          <div class="row-between" style="margin-bottom:8px">
            <h3 style="margin:0">What your data actually looks like</h3>
            <span class="badge badge-accent">${p.detected_format.mode} format detected</span>
          </div>
          <p class="muted tiny">These are real rows from the dataset. Check they
            look like what you expect — the single most common mistake is training
            on the wrong column.</p>
          <div class="table-wrap"><table class="preview-table">
            <thead><tr>${raw(cols.map((c) => `<th>${esc(c)}</th>`).join(""))}</tr></thead>
            <tbody>${raw(p.rows.slice(0, 3).map((r) => html`<tr>
              ${raw(cols.map((c) => `<td><div>${
                esc(String(r[c] ?? "")).slice(0, 300)}</div></td>`).join(""))}
            </tr>`).join(""))}</tbody>
          </table></div>
        </div>`;
    } catch (e) {
      box.innerHTML = html`<div class="callout callout-err">${e.message}</div>`;
    }
  };

  on(body, "click", "[data-ds]", (_e, t) => choose(t.dataset.ds));

  const doSearch = async () => {
    const q = $("#dsSearch", body).value.trim();
    const box = $("#dsResults", body);
    box.innerHTML = `<span class="muted tiny">Searching…</span>`;
    try {
      const rows = await api.searchDatasets(q);
      box.innerHTML = rows.length ? html`<div class="table-wrap"><table>
        <thead><tr><th>Dataset</th><th>Downloads</th><th></th></tr></thead>
        <tbody>${raw(rows.map((r) => html`<tr>
          <td class="mono">${r.id}</td><td>${fmtNum(r.downloads)}</td>
          <td><button class="btn-sm" data-ds="${r.id}">Use this</button></td>
        </tr>`).join(""))}</tbody></table></div>`
        : `<span class="muted tiny">Nothing matched.</span>`;
    } catch (e) {
      box.innerHTML = html`<div class="callout callout-err">${e.message}</div>`;
    }
  };
  $("#dsSearchBtn", body)?.addEventListener("click", doSearch);
  $("#dsSearch", body)?.addEventListener("keydown", (e) => {
    if (e.key === "Enter") doSearch();
  });

  if (state.dataset && state.preview) choose(state.dataset);
}

// ---- 2. text (from scratch) ----------------------------------------------
function stepCorpus(body, state, runners, starters, draw) {
  const corpora = starters.corpora || [];
  body.innerHTML = html`
    <div class="card" style="margin-bottom:14px">
      <h3 style="margin:0">Choose the text it learns from</h3>
      <p class="muted tiny" style="margin:2px 0 0">
        Everything your model will ever know comes from here. It needs ordinary
        running text — books, articles, stories — not questions and answers.
        Simple, consistent writing works far better at small sizes than varied,
        difficult writing does.</p>
    </div>

    <div class="grid grid-2">
      ${raw(corpora.map((c) => html`
        <button class="pick ${state.corpus === c.id ? "selected" : ""}" data-corpus="${c.id}">
          <span class="t">${c.label}
            <span class="badge">${fmtNum(c.approx_tokens)} tokens</span>
            ${raw(c.recommended ? `<span class="badge badge-ok">best first choice</span>` : "")}
          </span>
          <span class="d">${c.blurb}</span>
        </button>`).join(""))}
    </div>

    <details class="adv">
      <summary>Or use any text dataset from Hugging Face</summary>
      <div class="card" style="margin-top:10px">
        <div class="row">
          <input type="text" id="dsSearch" placeholder="e.g. poetry, german, legal…">
          <button id="dsSearchBtn">Search</button>
        </div>
        <p class="muted tiny" style="margin:8px 0 0">It must contain plain text.
          Instruction datasets will train, but the result continues text rather
          than answering, because that is all a fresh model can learn to do.</p>
        <div id="dsResults" style="margin-top:12px"></div>
      </div>
    </details>

    <div id="corpusPreview" style="margin-top:14px"></div>`;

  const choose = async (id, known) => {
    state.corpus = id;
    state.corpusInfo = known || { id, text_field: null, approx_tokens: null };
    state.scratchPlan = null;
    draw();
    const box = $("#corpusPreview");
    box.innerHTML = `<div class="card muted tiny">Loading a few real examples…</div>`;
    try {
      const p = await api.datasetPreview(id);
      if (!p.available) {
        state.corpusInfo.text_field = state.corpusInfo.text_field || "text";
        box.innerHTML = html`<div class="callout callout-warn">
          <strong>No preview available</strong>${p.reason || ""} Training will
          assume the text is in a column called
          "${state.corpusInfo.text_field}".</div>`;
        return;
      }
      // Longest string column is the text: on a corpus that is what the body
      // is, and it beats guessing by name across datasets that call it
      // "content", "document" or "raw".
      const detected = p.detected_format?.text_field
        || pickTextColumn(p.columns, p.rows)
        || state.corpusInfo.text_field || "text";
      state.corpusInfo.text_field = detected;
      const sample = String(p.rows[0]?.[detected] ?? "");
      box.innerHTML = html`
        <div class="card">
          <div class="row-between" style="margin-bottom:8px">
            <h3 style="margin:0">What your model will read</h3>
            <span class="badge badge-accent">using column "${detected}"</span>
          </div>
          <p class="sample-preview mono tiny" style="white-space:pre-wrap;
             background:var(--surface-2);padding:10px 12px;border-radius:8px;
             max-height:160px;overflow:auto">${sample.slice(0, 900)}</p>
          <div class="field" style="margin:12px 0 0">
            <label for="textField">Which column holds the text?</label>
            <select id="textField">
              ${raw(p.columns.map((c) =>
                `<option value="${esc(c)}" ${c === detected ? "selected" : ""}>${
                  esc(c)}</option>`).join(""))}
            </select>
            <div class="hint">Getting this wrong is the most common way a
              from-scratch run wastes hours. Check the text above looks right.</div>
          </div>
        </div>`;
      $("#textField").addEventListener("change", (e) => {
        state.corpusInfo.text_field = e.target.value;
        choose(id, state.corpusInfo);
      });
    } catch (e) {
      box.innerHTML = html`<div class="callout callout-err">${e.message}</div>`;
    }
  };

  on(body, "click", "[data-corpus]", (_e, t) => {
    const known = corpora.find((c) => c.id === t.dataset.corpus);
    choose(t.dataset.corpus, known ? { ...known } : null);
  });

  const doSearch = async () => {
    const q = $("#dsSearch", body).value.trim();
    const box = $("#dsResults", body);
    box.innerHTML = `<span class="muted tiny">Searching…</span>`;
    try {
      const rows = await api.searchDatasets(q);
      box.innerHTML = rows.length ? html`<div class="table-wrap"><table>
        <thead><tr><th>Dataset</th><th>Downloads</th><th></th></tr></thead>
        <tbody>${raw(rows.map((r) => html`<tr>
          <td class="mono">${r.id}</td><td>${fmtNum(r.downloads)}</td>
          <td><button class="btn-sm" data-corpus="${r.id}">Use this</button></td>
        </tr>`).join(""))}</tbody></table></div>`
        : `<span class="muted tiny">Nothing matched.</span>`;
    } catch (e) {
      box.innerHTML = html`<div class="callout callout-err">${e.message}</div>`;
    }
  };
  $("#dsSearchBtn", body)?.addEventListener("click", doSearch);
  $("#dsSearch", body)?.addEventListener("keydown", (e) => {
    if (e.key === "Enter") doSearch();
  });

  if (state.corpus) choose(state.corpus, state.corpusInfo);
}

function pickTextColumn(columns, rows) {
  let best = null, bestLen = 0;
  for (const c of columns || []) {
    const len = Math.max(...(rows || []).map((r) =>
      typeof r[c] === "string" ? r[c].length : 0), 0);
    if (len > bestLen) { bestLen = len; best = c; }
  }
  return bestLen > 40 ? best : null;
}

// ---- 3. size (from scratch) ----------------------------------------------
async function stepSize(body, state, runners, starters, draw) {
  const key = `${state.runnerId}|${state.minutes}|${state.vocab}`;
  if (state.sizesKey !== key) {
    body.innerHTML = `<div class="card muted">Working out what this machine can train…</div>`;
    try {
      state.sizes = await api.scratchSizes(state.runnerId, state.minutes, state.vocab);
      state.sizesKey = key;
    } catch (e) {
      body.innerHTML = html`<div class="callout callout-err">${e.message}</div>`;
      return;
    }
  }
  const { sizes, time_budgets: budgets, vocab_presets: vocabs } = state.sizes;
  if (!state.size) state.size = (sizes.find((s) => s.recommended) || sizes[0]).id;

  body.innerHTML = html`
    <div class="card" style="margin-bottom:14px">
      <h3 style="margin:0 0 4px">How long can you leave it running?</h3>
      <p class="muted tiny" style="margin:0 0 12px">This is the real limit on
        how big a model you can build. Everything below is recalculated from
        your answer.</p>
      <div class="grid grid-2">
        ${raw(budgets.map((b) => html`
          <button class="pick ${state.minutes === b.minutes ? "selected" : ""}"
                  data-minutes="${b.minutes}">
            <span class="t">${b.label}</span>
            <span class="d">${b.hint}</span>
          </button>`).join(""))}
      </div>
    </div>

    <div class="card" style="margin-bottom:14px">
      <h3 style="margin:0 0 4px">Choose a size</h3>
      <p class="muted tiny" style="margin:0 0 6px">
        A model needs about <strong>20 tokens of text per parameter</strong> to
        finish learning. Anything less and it stays half-taught — which is why
        the biggest option is usually the wrong one.</p>
    </div>

    <div class="grid">
      ${raw(sizes.map((s) => sizeCard(s, state)).join(""))}
    </div>

    <details class="adv">
      <summary>Change the vocabulary size</summary>
      <div class="card" style="margin-top:10px">
        <p class="muted tiny">The model reads text as tokens from a vocabulary
          built out of your own text. A bigger vocabulary reads text in fewer,
          longer pieces, but the lookup table costs parameters that the network
          itself then cannot use.</p>
        <div class="grid grid-2" style="margin-top:10px">
          ${raw(vocabs.map((v) => html`
            <button class="pick ${state.vocab === v.id ? "selected" : ""}"
                    data-vocab="${v.id}">
              <span class="t">${v.label} tokens</span>
              <span class="d">${v.hint}</span>
            </button>`).join(""))}
        </div>
      </div>
    </details>`;

  on(body, "click", "[data-minutes]", (_e, t) => {
    state.minutes = +t.dataset.minutes;
    state.scratchPlan = null;
    draw();
  });
  on(body, "click", "[data-vocab]", (_e, t) => {
    state.vocab = +t.dataset.vocab;
    state.scratchPlan = null;
    draw();
  });
  on(body, "click", "[data-size]", (_e, t) => {
    state.size = t.dataset.size;
    state.scratchPlan = null;
    draw();
  });
}

function sizeCard(s, state) {
  // The bar is the argument: it shows how much of the training a model of this
  // size actually gets, against the amount it needs. A quarter-full bar makes
  // "undertrained" concrete in a way a ratio does not.
  const pct = Math.min(100, Math.round((s.coverage || 0) * 100));
  const tone = VERDICT_CLASS[s.tone] || "badge";
  const barColor = s.tone === "ok" ? "var(--ok)"
    : s.tone === "warn" ? "var(--warn)" : "var(--err)";
  return html`
    <button class="pick ${state.size === s.id ? "selected" : ""}" data-size="${s.id}"
            ${!s.fits ? "disabled" : ""} style="${!s.fits ? "opacity:.5" : ""}">
      <span class="t">${s.label}
        <span class="badge">${s.params_label} parameters</span>
        ${raw(s.recommended ? `<span class="badge badge-ok">best for your time</span>` : "")}
        ${raw(!s.fits ? `<span class="badge badge-err">will not fit in memory</span>` : "")}
      </span>
      <span class="d">${s.blurb}</span>
      <span style="width:100%;margin-top:2px">
        <span class="row-between tiny" style="margin-bottom:4px">
          <span class="muted">Training it will get ${
            pct >= 100 ? "all" : pct + "%"} of the text it needs</span>
          <span class="badge ${tone}">${s.ratio ?? "?"} tokens per parameter</span>
        </span>
        <span class="progress" style="display:block">
          <i style="width:${pct}%;background:${barColor}"></i></span>
      </span>
      <span class="d" style="margin-top:2px">${s.message}
        ${raw(s.minutes_for_full ? `<em class="muted">Training it fully would take
          about ${esc(fmtDuration(s.minutes_for_full * 60))}.</em>` : "")}</span>
    </button>`;
}

// ---- 4. review (fine-tune) ------------------------------------------------
async function stepReview(body, state, runners, starters, draw) {
  body.innerHTML = `<div class="card muted">Working out the best settings for your machine…</div>`;
  let plan;
  try {
    plan = await api.plan({
      runner_id: state.runnerId,
      params_b: state.modelDetail?.params_b ?? null,
      goal: state.goal,
      dataset_rows: 2000,
    });
  } catch (e) {
    body.innerHTML = html`<div class="callout callout-err">${e.message}</div>`;
    return;
  }
  state.plan = plan;
  const s = { ...plan.settings, ...state.overrides };
  const runner = runners.find((r) => r.id === state.runnerId);
  const caps = runner?.capabilities || {};

  // A model that cannot fit is a dead end, not a warning. Say so here rather
  // than letting the controller reject the run a click later.
  const blocked = ["too_big", "needs_quantization"].includes(plan.fit?.verdict);
  state.blocked = blocked;

  body.innerHTML = html`
    ${raw(blocked ? html`
      <div class="callout callout-err">
        <strong>This model will not run on ${runnerName(runners, state.runnerId)}</strong>
        ${plan.fit.message} Go back and choose a smaller model, or pick a
        machine with more memory.
      </div>` : "")}

    <div class="card" style="margin-bottom:14px">
      <h3 style="margin:0 0 8px">Ready to train</h3>
      <dl class="kv">
        <dt>Teaching it to</dt><dd>${(GOALS.find((g) => g.id === state.goal) || {}).title}</dd>
        <dt>Starting from</dt><dd class="mono">${state.model}</dd>
        <dt>Learning from</dt><dd class="mono">${state.dataset}</dd>
        <dt>Running on</dt><dd>${runner?.name} — ${caps.device_name || ""}</dd>
        ${raw(plan.estimated_minutes
          ? `<dt>Rough duration</dt><dd>about ${esc(fmtDuration(plan.estimated_minutes * 60))}</dd>`
          : "")}
      </dl>
    </div>

    ${raw((caps.warnings || []).map((w) => html`
      <div class="callout callout-warn"><strong>About this machine</strong>${w}</div>`).join(""))}

    <div class="card" style="margin-bottom:14px">
      <h3>Settings chosen for you</h3>
      <p class="muted tiny">Picked from what this machine measured about itself.
        You can change any of them, but the defaults are sound.</p>
      <div class="grid grid-2" style="margin-top:12px">
        ${raw(plan.explanations.map(explainCard).join(""))}
      </div>

      <details class="adv">
        <summary>Change the technical settings</summary>
        <div class="grid grid-3" style="margin-top:12px">
          ${raw(fields([
            ["epochs", "Passes over the data", s.epochs, "number", "0.1"],
            ["learning_rate", "Learning rate", s.learning_rate, "text", ""],
            ["batch_size", "Batch size", s.batch_size, "number", "1"],
            ["grad_accum", "Gradient accumulation", s.grad_accum, "number", "1"],
            ["max_seq_len", "Max sequence length", s.max_seq_len, "number", "1"],
            ["lora_r", "LoRA rank", s.lora_r, "number", "1"],
            ["max_steps", "Maximum steps", s.max_steps, "number", "1"],
          ]))}
          <div class="field">
            <label for="f_dtype">Number precision</label>
            <select id="f_dtype" data-setting="dtype">
              ${raw(["float16", "bfloat16", "float32"].map((d) =>
                `<option value="${d}" ${d === s.dtype ? "selected" : ""}>${d}</option>`).join(""))}
            </select>
            <div class="hint">Recommended here: ${caps.recommended_dtype}</div>
          </div>
        </div>
      </details>
    </div>

    <div class="field card">
      <label for="jobName">Name this run</label>
      <input type="text" id="jobName"
             value="${String(state.model || "").split("/").pop()} on ${
               String(state.dataset || "").split("/").pop()}">
      <div class="hint">Just so you can find it later.</div>
    </div>`;

  gateNext(blocked, "Choose a smaller model to continue.");
  wireOverrides(body, state);
}

// ---- 4. review (from scratch) --------------------------------------------
async function stepScratchReview(body, state, runners, starters, draw) {
  body.innerHTML = `<div class="card muted">Working out the settings for your machine…</div>`;
  let plan;
  try {
    plan = await api.scratchPlan({
      runner_id: state.runnerId,
      size: state.size,
      minutes: state.minutes,
      vocab_size: state.vocab,
      corpus_tokens: state.corpusInfo?.approx_tokens ?? null,
      sample_prompt: state.corpusInfo?.sample_prompt || "Once upon a time",
      text_field: state.corpusInfo?.text_field || "text",
    });
  } catch (e) {
    body.innerHTML = html`<div class="callout callout-err">${e.message}</div>`;
    gateNext(true, "This size will not run here.");
    return;
  }
  state.scratchPlan = plan;
  const s = { ...plan.settings, ...state.overrides };
  const runner = runners.find((r) => r.id === state.runnerId);
  const caps = runner?.capabilities || {};
  const size = state.sizes?.sizes.find((x) => x.id === state.size) || {};
  const blocked = plan.verdict.verdict === "not_viable";
  state.blocked = blocked;

  body.innerHTML = html`
    ${raw(blocked ? html`
      <div class="callout callout-err">
        <strong>This will not produce a usable model</strong>
        ${plan.verdict.message} Go back and pick a smaller size, or allow more
        time. You can start it anyway from the previous step by choosing a
        different size — but at this ratio the result will be noise.
      </div>` : "")}

    <div class="card" style="margin-bottom:14px">
      <h3 style="margin:0 0 8px">Ready to build a model</h3>
      <dl class="kv">
        <dt>Building</dt><dd>${size.label || state.size} —
          ${plan.params_label} parameters, from random weights</dd>
        <dt>Learning from</dt><dd class="mono">${state.corpus}</dd>
        <dt>Running on</dt><dd>${runner?.name} — ${caps.device_name || ""}</dd>
        <dt>Training time</dt><dd>about ${fmtDuration(state.minutes * 60)}</dd>
        <dt>Text it will read</dt><dd>${fmtNum(s.token_budget)} tokens over
          ${fmtNum(s.max_steps)} steps</dd>
        <dt>Memory needed</dt><dd>about ${plan.memory_gb} GB of
          ${caps.vram_gb || "?"} GB</dd>
      </dl>
    </div>

    ${raw((plan.notes || []).map((n) => html`
      <div class="callout callout-warn"><strong>Worth knowing</strong>${n}</div>`).join(""))}

    ${raw(plan.minutes_for_full && plan.minutes_for_full > state.minutes * 1.5 ? html`
      <div class="callout">
        <strong>For reference</strong>
        Training this size to completion would take about
        ${fmtDuration(plan.minutes_for_full * 60)} on this machine. You are
        giving it ${fmtDuration(state.minutes * 60)}, which is
        ${Math.round((state.minutes / plan.minutes_for_full) * 100)}% of that.
      </div>` : "")}

    <div class="card" style="margin-bottom:14px">
      <h3>Settings chosen for you</h3>
      <p class="muted tiny">Derived from your machine's measured speed and
        memory. Every one of them can be changed.</p>
      <div class="grid grid-2" style="margin-top:12px">
        ${raw(plan.explanations.map(explainCard).join(""))}
      </div>

      <details class="adv">
        <summary>Change the technical settings</summary>
        <div class="grid grid-3" style="margin-top:12px">
          ${raw(fields([
            ["learning_rate", "Learning rate", s.learning_rate, "text", ""],
            ["batch_size", "Batch size", s.batch_size, "number", "1"],
            ["grad_accum", "Gradient accumulation", s.grad_accum, "number", "1"],
            ["max_steps", "Steps", s.max_steps, "number", "1"],
            ["warmup_steps", "Warmup steps", s.warmup_steps, "number", "1"],
            ["weight_decay", "Weight decay", s.weight_decay, "text", ""],
            ["eval_every", "Check held-out loss every", s.eval_every, "number", "1"],
            ["sample_every", "Write a sample every", s.sample_every, "number", "1"],
          ]))}
        </div>
      </details>
    </div>

    <div class="card">
      <div class="field">
        <label for="samplePrompt">Sample opening</label>
        <input type="text" id="samplePrompt" data-setting="sample_prompt"
               value="${s.sample_prompt}">
        <div class="hint">During training the model will be asked to continue
          this, every ${s.sample_every} steps. Watching those samples turn from
          noise into sentences is the clearest sign it is working.</div>
      </div>
      <div class="field" style="margin-bottom:0">
        <label for="jobName">Name this run</label>
        <input type="text" id="jobName" value="${size.label || "Model"} on ${
          String(state.corpus || "").split("/").pop()}">
        <div class="hint">Just so you can find it later.</div>
      </div>
    </div>`;

  gateNext(blocked, "Pick a smaller size or allow more time.");
  wireOverrides(body, state);
}

// ---- shared review pieces -------------------------------------------------
const explainCard = (e) => html`
  <div class="card" style="box-shadow:none;background:var(--surface-2)">
    <div class="row-between"><strong class="tiny">${e.setting}</strong>
      <span class="badge badge-accent">${e.value}</span></div>
    <p class="muted tiny" style="margin:6px 0 0">${e.why}</p>
  </div>`;

const fields = (rows) => rows.map(([k, label, v, type, step]) => html`
  <div class="field">
    <label for="f_${k}">${label}</label>
    <input id="f_${k}" data-setting="${k}" type="${type}"
           ${raw(step ? `step="${step}"` : "")} value="${v}">
  </div>`).join("");

function gateNext(blocked, hintText) {
  const nextBtn = document.getElementById("nextBtn");
  if (!nextBtn) return;
  nextBtn.disabled = blocked;
  const hint = document.getElementById("navHint");
  if (hint) hint.textContent = blocked ? hintText : "";
}

function wireOverrides(body, state) {
  const numeric = new Set(["learning_rate", "weight_decay"]);
  on(body, "change", "[data-setting]", (_e, t) => {
    const k = t.dataset.setting;
    state.overrides[k] = t.type === "number" ? +t.value
      : numeric.has(k) ? parseFloat(t.value) : t.value;
  });
}

// ===========================================================================

function wireNav(mount, state, draw) {
  const next = $("#nextBtn", mount);
  const back = $("#backBtn", mount);
  const hint = $("#navHint", mount);
  if (!next) return;

  const last = STEPS[state.mode].length - 1;
  const blockers = state.mode === "scratch" ? {
    0: () => (!state.runnerId ? "Choose a machine to continue." : null),
    1: () => (!state.corpus ? "Choose some text to continue." : null),
    2: () => (!state.size ? "Choose a size to continue." : null),
    3: () => null,
  } : {
    0: () => (!state.runnerId ? "Choose a machine to continue." : null),
    1: () => (!state.model ? "Choose a model to continue." : null),
    2: () => (!state.dataset ? "Choose a dataset to continue." : null),
    3: () => null,
  };
  const blocker = blockers[state.step]();

  next.disabled = !!blocker;
  hint.textContent = blocker || "";

  back?.addEventListener("click", () => { state.step--; draw(); });
  next.addEventListener("click", async () => {
    if (state.step < last) { state.step++; draw(); return; }

    next.disabled = true;
    next.textContent = "Starting…";
    try {
      const { id } = await api.createJob(buildJob(mount, state));
      toast("Training run created.", "ok");
      location.hash = `#/jobs/${id}`;
    } catch (e) {
      toast(e.message, "err");
      next.disabled = false;
      next.textContent = "Start training";
    }
  });
}

function buildJob(mount, state) {
  const name = $("#jobName", mount)?.value || undefined;
  if (state.mode === "scratch") {
    const s = { ...state.scratchPlan.settings, ...state.overrides };
    return {
      name, kind: "pretrain_llm",
      config: {
        dataset: state.corpus,
        dataset_config: state.corpusInfo?.config || null,
        text_field: state.corpusInfo?.text_field || "text",
        required_runner: state.runnerId,
        ...s,
      },
    };
  }
  const s = { ...state.plan.settings, ...state.overrides };
  return {
    name, kind: "finetune_llm",
    config: {
      base_model: state.model,
      dataset: state.dataset,
      params_b: state.modelDetail?.params_b ?? null,
      goal: state.goal,
      required_runner: state.runnerId,
      format: state.preview?.detected_format || { mode: "auto" },
      ...s,
    },
  };
}
