import { api } from "../api.js";
import { html, raw, esc, on, $, $$, fmtNum, fmtDuration, toast } from "../util.js";

// Two fundamentally different jobs behind one wizard. They share a machine
// picker, a data step and a review; everything between differs, because the
// decisions differ. Fine-tuning asks "which model, which examples"; training
// from scratch asks "how big, and how long are you prepared to wait" -- a
// question fine-tuning never has to answer.
//
// RENDERING RULE, and the reason this file was rewritten: `draw()` renders
// purely from state, and no render may start work that causes another render
// synchronously. Every asynchronous load goes through `ensure()`, which marks
// itself in-flight *before* awaiting, so the re-render it eventually triggers
// finds the work already done rather than starting it again. The previous
// version called draw() from inside the click handler that each step re-ran on
// render, which recursed until the stack gave out.

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

const STEP_NAMES = {
  finetune: ["Goal", "Model", "Data", "Review"],
  scratch: ["Goal", "Text", "Design", "Review"],
};

const VERDICT_CLASS = { ok: "badge-ok", warn: "badge-warn", err: "badge-err" };
const LEVEL_CLASS = { error: "callout-err", warn: "callout-warn", info: "callout" };

// Every from-scratch training setting, with the sentence that explains it.
// Declared rather than hand-written so the review step cannot quietly omit one.
const SCRATCH_FIELDS = [
  ["learning_rate", "Learning rate", "text",
   "How big a step each update takes. Scales with model width — too high and "
   + "the loss spikes and never recovers; too low and you waste the run."],
  ["batch_size", "Batch size", "number",
   "Sequences processed at once. Limited by memory, not by preference."],
  ["grad_accum", "Gradient accumulation", "number",
   "How many batches are combined before each update. Multiplies the effective "
   + "batch without needing the memory for it."],
  ["max_steps", "Training steps", "number",
   "How many updates to make. Together with tokens-per-step this is the whole "
   + "training budget."],
  ["warmup_steps", "Warmup steps", "number",
   "Steps spent ramping the learning rate up from zero. A model of random "
   + "weights takes a large, badly-aimed first step without it."],
  ["min_lr_ratio", "Final learning rate", "text",
   "Fraction of the peak rate to decay down to by the end. 0.1 is standard."],
  ["weight_decay", "Weight decay", "text",
   "Gentle pressure pulling weights toward zero, which limits overfitting. "
   + "Applied to matrices only, never to norms or biases."],
  ["grad_clip", "Gradient clipping", "text",
   "Caps the size of an update so one strange batch cannot wreck the model."],
  ["eval_every", "Check held-out loss every", "number",
   "How often to measure against text the model never trains on. The only "
   + "number that can tell you it is memorising."],
  ["sample_every", "Write a sample every", "number",
   "How often the model is asked to write something, so you can watch it learn."],
  ["seed", "Random seed", "number",
   "Fixes the shuffling and the initial weights, so a run can be repeated."],
  ["max_corpus_tokens", "Maximum text held in memory", "number",
   "Tokens kept in RAM at once. Beyond this the run makes repeated passes."],
  ["tokenizer_sample_rows", "Rows used to build the vocabulary", "number",
   "More rows make a better vocabulary and take longer to scan."],
];

const ARCH_FIELDS = [
  ["num_hidden_layers", "Layers", "How many transformer blocks are stacked. "
   + "Depth lets the model compose ideas across steps."],
  ["hidden_size", "Width", "The size of the vector carried between layers. "
   + "The single biggest influence on capacity and on speed."],
  ["num_attention_heads", "Attention heads", "The width is split evenly across "
   + "heads, each looking at a different relationship. Must divide the width."],
  ["max_position_embeddings", "Context length", "How many tokens the model can "
   + "see at once. Without a fused attention kernel, memory grows with the "
   + "square of this."],
  ["intermediate_size", "Feed-forward width", "The width inside each block's "
   + "feed-forward layer, normally about 2.7x the model width. Most of a "
   + "transformer's capacity lives here."],
];

// ===========================================================================
// A resource that loads once per key, never in a render loop.
// ===========================================================================

const resource = () => ({ status: "idle", key: null, data: null, error: null });

function ensure(res, key, fetcher, draw) {
  // Already loading or loaded for this exact key: do nothing. This is the
  // guard that makes it safe for a render to call ensure() unconditionally.
  if (res.key === key && res.status !== "idle") return res;
  res.key = key;
  res.status = "loading";
  res.data = null;
  res.error = null;
  (async () => {
    try {
      const data = await fetcher();
      if (res.key !== key) return;      // a newer request has superseded this
      res.data = data;
      res.status = "ready";
    } catch (e) {
      if (res.key !== key) return;
      res.error = e.message || String(e);
      res.status = "error";
    }
    draw();
  })();
  return res;
}

const loading = (what) => html`<div class="card muted tiny">${what}</div>`;
const failed = (msg) => html`<div class="callout callout-err">${msg}</div>`;

// ===========================================================================

export async function wizardView(mount) {
  const state = {
    mode: "finetune", step: 0, goal: "instructions", runnerId: null,
    model: null, modelDetail: resource(),
    // The data step, shared by both paths.
    dataset: null, configs: resource(), config: null, split: "train",
    preview: resource(), textField: null, formatMode: null,
    // From scratch.
    minutes: 60, size: null, vocab: 8192, custom: null,
    sizes: resource(), plan: resource(), ftPlan: resource(),
    overrides: {}, archOverrides: {},
    starting: false,
  };

  const runners = (await api.runners()).filter((r) => r.status !== "offline");
  const starters = await api.starters();
  state.runnerId = runners[0]?.id ?? null;
  state.vocab = starters.default_vocab || 8192;

  const ctx = { state, runners, starters, draw: () => draw() };

  function draw() {
    // Clamp rather than trust: a step index that runs past the end used to
    // throw "steps[state.step] is not a function" and blank the page.
    const names = STEP_NAMES[state.mode] || STEP_NAMES.finetune;
    state.step = Math.max(0, Math.min(state.step | 0, names.length - 1));
    mount.innerHTML = shell(state, runners, names);
    const body = $("#stepBody", mount);
    if (body) STEPS[state.mode][state.step](body, ctx);
    wireNav(mount, ctx);
  }

  draw();
  return () => {};
}

function shell(state, runners, names) {
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
  const chips = names.map((s, i) => html`
    <div class="step-chip ${i === state.step ? "current" : i < state.step ? "done" : ""}">
      <span class="n">${i < state.step ? "✓" : String(i + 1)}</span>${s}
    </div>`).join("");
  return html`
    <div class="page-head">
      <h1>New training run</h1>
      <p class="sub">Four steps. Everything technical is chosen for you, and
        every choice is explained — and every one can be changed.</p>
    </div>
    <div class="steps">${raw(chips)}</div>
    <div id="stepBody"></div>
    <div class="wizard-actions">
      <button id="backBtn" ${state.step === 0 ? "disabled" : ""}>← Back</button>
      <div class="spacer"></div>
      <span id="navHint" class="tiny muted"></span>
      <button id="nextBtn" class="btn-primary btn-lg">
        ${state.step === names.length - 1 ? "Start training" : "Continue →"}
      </button>
    </div>`;
}

// ===========================================================================
// Step 1 — what kind of run, on which machine
// ===========================================================================

function stepGoal(body, { state, runners, draw }) {
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
        than fine-tuning, so on a single graphics card only small models can be
        trained properly. A small model trained well beats a large one trained
        badly, and the next steps show you exactly where that line falls on
        your machine.
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
    // The two paths share nothing past this point. A half-made choice from the
    // other one must not survive and end up in the job config.
    Object.assign(state, {
      model: null, modelDetail: resource(), dataset: null, configs: resource(),
      config: null, split: "train", preview: resource(), textField: null,
      formatMode: null, size: null, custom: null, sizes: resource(),
      plan: resource(), ftPlan: resource(), overrides: {}, archOverrides: {},
    });
    draw();
  });
  on(body, "click", "[data-goal]", (_e, t) => { state.goal = t.dataset.goal; draw(); });
  on(body, "click", "[data-runner]", (_e, t) => {
    state.runnerId = t.dataset.runner;
    state.sizes = resource();
    state.plan = resource();
    state.ftPlan = resource();
    draw();
  });
}

// ===========================================================================
// Step 2 (fine-tune) — the base model
// ===========================================================================

function stepModel(body, ctx) {
  const { state, runners, starters, draw } = ctx;
  const runner = runners.find((r) => r.id === state.runnerId);
  const caps = runner?.capabilities || {};
  const ceiling = caps.max_finetune_params_b;

  if (state.model) {
    ensure(state.modelDetail, state.model,
           () => api.modelDetail(state.model), draw);
  }

  body.innerHTML = html`
    <div class="card" style="margin-bottom:14px">
      <div class="row-between">
        <div>
          <h3 style="margin:0">Choose a base model</h3>
          <p class="muted tiny" style="margin:2px 0 0">
            You are not starting from nothing — you start from a model that
            already understands language, and teach it your task.</p>
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
                  data-model="${m.id}">
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

    <div style="margin-top:14px">${raw(modelInfo(state))}</div>`;

  on(body, "click", "[data-model]", (_e, t) => {
    state.model = t.dataset.model;
    state.modelDetail = resource();
    state.ftPlan = resource();
    draw();
  });

  wireSearch(body, "model", async (q) => {
    const rows = await api.searchModels(q);
    return rows.length ? html`<div class="table-wrap"><table>
      <thead><tr><th>Model</th><th>Size</th><th>Downloads</th><th></th></tr></thead>
      <tbody>${raw(rows.map((r) => html`<tr>
        <td class="mono">${r.id}${raw(r.gated
          ? ` <span class="badge badge-warn">gated</span>` : "")}</td>
        <td>${r.params_b ? r.params_b + "B" : "—"}</td>
        <td>${fmtNum(r.downloads)}</td>
        <td><button class="btn-sm" data-model="${r.id}">Use this</button></td>
      </tr>`).join(""))}</tbody></table></div>`
      : `<span class="muted tiny">Nothing matched that search.</span>`;
  });
}

function modelInfo(state) {
  const r = state.modelDetail;
  if (!state.model) return "";
  if (r.status === "loading") return loading("Checking whether this fits…");
  if (r.status === "error") return failed(r.error);
  if (r.status !== "ready") return "";
  const d = r.data;
  const fit = d.fit?.[state.runnerId];
  const cls = { fits: "callout-ok", fits_quantized: "callout",
                needs_quantization: "callout-warn", too_big: "callout-err"
              }[fit?.verdict] || "callout";
  return html`
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
}

// ===========================================================================
// Step 3 — the data, and what the model will actually read
// ===========================================================================

function stepData(body, ctx) {
  const { state, starters, draw } = ctx;
  const scratch = state.mode === "scratch";
  const catalogue = scratch ? (starters.corpora || []) : starters.datasets;

  if (state.dataset) {
    ensure(state.configs, state.dataset,
           () => api.datasetConfigs(state.dataset), draw);
    // Adopt the dataset's default configuration the moment we learn it.
    const c = state.configs;
    if (c.status === "ready" && c.key === state.dataset && state.config === null) {
      state.config = c.data.default_config ?? "";
      const chosen = c.data.configs.find((x) => x.name === state.config);
      if (chosen && !chosen.splits.includes(state.split)) {
        state.split = chosen.splits.includes("train") ? "train" : chosen.splits[0];
      }
    }
    if (c.status === "ready") {
      const key = [state.dataset, state.config, state.split,
                   state.formatMode || "", state.textField || ""].join("|");
      ensure(state.preview, key, () => api.trainingPreview({
        dataset: state.dataset,
        config: state.config,
        split: state.split,
        format: state.formatMode ? { mode: state.formatMode } : null,
        text_field: scratch ? state.textField : null,
      }), draw);
    }
  }

  body.innerHTML = html`
    <div class="card" style="margin-bottom:14px">
      <h3 style="margin:0">${scratch ? "Choose the text it learns from"
                                     : "Choose your examples"}</h3>
      <p class="muted tiny" style="margin:2px 0 0">
        ${scratch
          ? raw("Everything your model will ever know comes from here. It needs "
              + "ordinary running text — books, articles, stories — not questions "
              + "and answers. Simple, consistent writing works far better at small "
              + "sizes than varied, difficult writing does.")
          : raw("This is what the model learns from. Quality matters far more "
              + "than quantity — a few hundred good examples beat thousands of "
              + "sloppy ones.")}</p>
    </div>

    <div class="grid grid-2">
      ${raw(catalogue.map((d) => html`
        <button class="pick ${state.dataset === d.id ? "selected" : ""}" data-ds="${d.id}"
                data-config="${d.config || ""}" data-field="${d.text_field || ""}"
                data-tokens="${d.approx_tokens || ""}"
                data-prompt="${d.sample_prompt || ""}">
          <span class="t">${d.label}
            <span class="badge">${scratch ? fmtNum(d.approx_tokens) + " tokens"
                                          : fmtNum(d.rows) + " examples"}</span>
            ${raw(d.recommended ? `<span class="badge badge-ok">best first choice</span>` : "")}
          </span>
          <span class="d">${d.blurb}</span>
        </button>`).join(""))}
    </div>

    <details class="adv">
      <summary>Or use any dataset from Hugging Face</summary>
      <div class="card" style="margin-top:10px">
        <div class="row">
          <input type="text" id="dsSearch" placeholder="e.g. code, medical, german…">
          <button id="dsSearchBtn">Search</button>
        </div>
        <div id="dsResults" style="margin-top:12px"></div>
      </div>
    </details>

    <div style="margin-top:14px">${raw(dataDetail(state, scratch))}</div>`;

  on(body, "click", "[data-ds]", (_e, t) => {
    state.dataset = t.dataset.ds;
    state.configs = resource();
    state.preview = resource();
    state.config = t.dataset.config || null;
    state.split = "train";
    state.textField = t.dataset.field || null;
    state.formatMode = null;
    // How much text the corpus holds, and a natural opening to sample with.
    // Both feed the plan: the first decides whether the token budget would
    // exhaust the dataset, the second is what the model is asked to continue.
    state.corpusTokens = +t.dataset.tokens || null;
    state.samplePrompt = t.dataset.prompt || null;
    state.plan = resource();
    state.sizes = resource();
    draw();
  });

  on(body, "change", "#cfgSelect", (_e, t) => {
    state.config = t.value;
    state.preview = resource();
    const chosen = state.configs.data?.configs.find((x) => x.name === t.value);
    if (chosen && !chosen.splits.includes(state.split)) {
      state.split = chosen.splits.includes("train") ? "train" : chosen.splits[0];
    }
    draw();
  });
  on(body, "change", "#splitSelect", (_e, t) => {
    state.split = t.value; state.preview = resource(); draw();
  });
  on(body, "change", "#fieldSelect", (_e, t) => {
    state.textField = t.value; state.preview = resource(); draw();
  });
  on(body, "change", "#formatSelect", (_e, t) => {
    state.formatMode = t.value || null; state.preview = resource(); draw();
  });

  wireSearch(body, "ds", async (q) => {
    const rows = await api.searchDatasets(q);
    return rows.length ? html`<div class="table-wrap"><table>
      <thead><tr><th>Dataset</th><th>Downloads</th><th></th></tr></thead>
      <tbody>${raw(rows.map((r) => html`<tr>
        <td class="mono">${r.id}</td><td>${fmtNum(r.downloads)}</td>
        <td><button class="btn-sm" data-ds="${r.id}">Use this</button></td>
      </tr>`).join(""))}</tbody></table></div>`
      : `<span class="muted tiny">Nothing matched.</span>`;
  });
}

function dataDetail(state, scratch) {
  if (!state.dataset) return "";
  const c = state.configs;
  if (c.status === "loading") return loading("Looking at this dataset…");
  if (c.status === "error") return failed(c.error);
  if (c.status !== "ready") return "";

  const configs = c.data.configs || [];
  const chosen = configs.find((x) => x.name === state.config) || configs[0];
  const splits = chosen?.splits || ["train"];

  return html`
    <div class="card">
      <h3 style="margin:0 0 4px">Which part of it?</h3>
      <p class="muted tiny" style="margin:0 0 12px">
        ${raw(c.data.needs_choice
          ? "This dataset contains several separate collections under one name. "
          + "Training cannot guess between them, so pick the one you want."
          : "This dataset has a single collection, so there is nothing to choose "
          + "here beyond the split.")}</p>

      <div class="grid grid-3">
        ${raw(configs.length ? html`
          <div class="field">
            <label for="cfgSelect">Collection</label>
            <select id="cfgSelect">
              ${raw(configs.map((x) => `<option value="${esc(x.name)}"${
                x.name === state.config ? " selected" : ""}>${esc(x.name)}</option>`).join(""))}
            </select>
            <div class="hint">${configs.length} available.</div>
          </div>` : "")}
        <div class="field">
          <label for="splitSelect">Split</label>
          <select id="splitSelect">
            ${raw(splits.map((sp) => `<option value="${esc(sp)}"${
              sp === state.split ? " selected" : ""}>${esc(sp)}</option>`).join(""))}
          </select>
          <div class="hint">Usually "train".</div>
        </div>
        ${raw(previewControls(state, scratch))}
      </div>
    </div>

    <div style="margin-top:14px">${raw(trainingText(state, scratch))}</div>`;
}

function previewControls(state, scratch) {
  const p = state.preview;
  if (p.status !== "ready" || !p.data.available) return "";
  const cols = p.data.columns || [];
  if (scratch) {
    return html`
      <div class="field">
        <label for="fieldSelect">Text column</label>
        <select id="fieldSelect">
          ${raw(cols.map((col) => `<option value="${esc(col)}"${
            col === (state.textField || p.data.format?.text_field)
              ? " selected" : ""}>${esc(col)}</option>`).join(""))}
        </select>
        <div class="hint">Which column holds the writing.</div>
      </div>`;
  }
  const mode = state.formatMode || p.data.format?.mode || "auto";
  return html`
    <div class="field">
      <label for="formatSelect">How to read a row</label>
      <select id="formatSelect">
        ${raw([["", "Detect automatically"], ["instruction", "Instruction and response"],
               ["chat", "Chat messages"], ["text", "Plain text"]].map(([v, l]) =>
          `<option value="${v}"${v === (state.formatMode || "") ? " selected" : ""}>${
            esc(l)}</option>`).join(""))}
      </select>
      <div class="hint">Detected: ${esc(mode)}.</div>
    </div>`;
}

function trainingText(state, scratch) {
  const p = state.preview;
  if (p.status === "loading") return loading("Reading a few real rows…");
  if (p.status === "error") return failed(p.error);
  if (p.status !== "ready") return "";
  if (!p.data.available) {
    return html`<div class="callout callout-warn">
      <strong>No preview available</strong>${p.data.reason || ""} You can still
      use it, but check the column names are right before a long run.</div>`;
  }

  const rendered = p.data.rendered || [];
  const counts = p.data.counts || {};
  return html`
    ${raw(counts.unreadable ? html`
      <div class="callout callout-err">
        <strong>${counts.unreadable} of ${counts.sampled} sampled rows could not be read</strong>
        These rows have content, but the columns chosen above do not reach it,
        so training would skip them. The dataset's columns are:
        ${(p.data.columns || []).join(", ")}.
      </div>` : "")}
    ${raw(counts.empty ? html`
      <div class="callout">
        <strong>${counts.empty} of ${counts.sampled} sampled rows are blank</strong>
        Perfectly normal in line-by-line corpora — blank lines separate
        paragraphs. Training skips them and reads the rest.
      </div>` : "")}
    <div class="card">
      <div class="row-between" style="margin-bottom:6px">
        <h3 style="margin:0">Exactly what the model will read</h3>
        <span class="badge badge-accent">${p.data.split}</span>
      </div>
      <p class="muted tiny">Not the raw columns — the finished text, built by
        the same code that will build the training batches.
        ${raw(scratch ? "The model reads these one after another, with no gaps."
                      : "Everything below, including the headings, is learned.")}</p>
      <div class="samples" style="max-height:380px">
        ${raw(rendered.map((r, i) => html`
          <div class="sample">
            <div class="hd"><span class="badge">example ${i + 1}</span>
              ${raw(r.status === "unreadable"
                ? `<span class="badge badge-err">columns do not match</span>`
                : r.status === "empty"
                ? `<span class="badge">blank row</span>` : "")}</div>
            <p class="txt">${r.text || "(nothing)"}</p>
          </div>`).join(""))}
      </div>
    </div>`;
}

// ===========================================================================
// Step 3 (from scratch) — designing the model
// ===========================================================================

function stepDesign(body, ctx) {
  const { state, draw } = ctx;
  const key = `${state.runnerId}|${state.minutes}|${state.vocab}`;
  ensure(state.sizes, key,
         () => api.scratchSizes(state.runnerId, state.minutes, state.vocab), draw);

  if (state.sizes.status === "loading") {
    body.innerHTML = loading("Working out what this machine can train…");
    return;
  }
  if (state.sizes.status === "error") { body.innerHTML = failed(state.sizes.error); return; }
  if (state.sizes.status !== "ready") { body.innerHTML = ""; return; }

  const { sizes, time_budgets: budgets, vocab_presets: vocabs } = state.sizes.data;
  if (!state.size) state.size = (sizes.find((s) => s.recommended) || sizes[0]).id;
  const custom = state.size === "custom";

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
      <details class="adv">
        <summary>Or set an exact number of minutes</summary>
        <div class="field" style="margin-top:10px;max-width:240px">
          <label for="customMinutes">Minutes</label>
          <input type="number" id="customMinutes" min="1" max="20160"
                 value="${state.minutes}">
        </div>
      </details>
    </div>

    <div class="card" style="margin-bottom:14px">
      <h3 style="margin:0 0 4px">Choose a size</h3>
      <p class="muted tiny" style="margin:0">
        A model needs about <strong>20 tokens of text per parameter</strong> to
        finish learning. Anything less and it stays half-taught — which is why
        the biggest option is usually the wrong one.</p>
    </div>

    <div class="grid">
      ${raw(sizes.map((s) => sizeCard(s, state)).join(""))}
      <button class="pick ${custom ? "selected" : ""}" data-size="custom">
        <span class="t">⚙ Design it yourself
          ${raw(custom ? `<span class="badge badge-accent">custom</span>` : "")}</span>
        <span class="d">Set the layers, width, heads and context length by hand.
          Every number is checked against your machine as you type, and anything
          that will not work is explained rather than simply refused.</span>
      </button>
    </div>

    <div id="designer">${raw(custom ? designer(state) : "")}</div>

    <details class="adv" ${custom ? "open" : ""}>
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
    state.sizes = resource(); state.plan = resource();
    draw();
  });
  on(body, "change", "#customMinutes", (_e, t) => {
    const v = Math.max(1, Math.min(20160, +t.value || 60));
    state.minutes = v; state.sizes = resource(); state.plan = resource();
    draw();
  });
  on(body, "click", "[data-vocab]", (_e, t) => {
    state.vocab = +t.dataset.vocab;
    state.sizes = resource(); state.plan = resource();
    draw();
  });
  on(body, "click", "[data-size]", (_e, t) => {
    state.size = t.dataset.size;
    state.plan = resource();
    if (state.size === "custom" && !state.custom) {
      // Seed the designer from whichever preset was highlighted, so it starts
      // from something that works rather than from an empty form.
      const from = sizes.find((s) => s.recommended) || sizes[0];
      state.custom = {
        num_hidden_layers: from.layers, hidden_size: from.dim,
        num_attention_heads: from.heads, max_position_embeddings: from.seq,
        intermediate_size: null,
      };
    }
    draw();
  });

  if (custom) wireDesigner(body, ctx);
}

function sizeCard(s, state) {
  // The bar is the argument: how much of the training a model of this size
  // actually gets, against how much it needs. A quarter-full bar makes
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
          <span class="muted">Gets ${pct >= 100 ? "all" : pct + "%"} of the text it needs</span>
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

function designer(state) {
  const c = state.custom || {};
  return html`
    <div class="card" style="margin-top:14px">
      <h3 style="margin:0 0 4px">Design your model</h3>
      <p class="muted tiny">Change anything. The numbers on the right update as
        you type, and problems are explained underneath rather than hidden.</p>
      <div class="grid grid-2" style="margin-top:12px;align-items:start">
        <div>
          ${raw(ARCH_FIELDS.map(([k, label, hint]) => html`
            <div class="field">
              <label for="a_${k}">${label}</label>
              <input id="a_${k}" data-arch="${k}" type="number" min="1"
                     value="${c[k] ?? ""}"
                     placeholder="${k === "intermediate_size" ? "auto" : ""}">
              <div class="hint">${hint}</div>
            </div>`).join(""))}
        </div>
        <div id="designPreview"></div>
      </div>
    </div>`;
}

function wireDesigner(body, ctx) {
  const { state } = ctx;
  const refresh = () => refreshPlan(ctx, "#designPreview", designSummary);
  refresh();

  let timer = null;
  // Rendered into its own panel rather than through draw(), so that typing in
  // a field does not destroy the field being typed into.
  on(body, "input", "[data-arch]", (_e, t) => {
    const k = t.dataset.arch;
    state.custom = { ...state.custom, [k]: t.value === "" ? null : +t.value };
    state.plan = resource();
    clearTimeout(timer);
    timer = setTimeout(refresh, 350);
  });
}

function designSummary(state, plan) {
  if (!plan) return "";
  const p = plan.params || {};
  const v = plan.verdict || {};
  const tone = VERDICT_CLASS[v.tone] || "badge";
  return html`
    <div class="card" style="box-shadow:none;background:var(--surface-2)">
      <h4 style="margin:0 0 10px">This model</h4>
      <dl class="kv">
        <dt>Parameters</dt><dd>${plan.params_label}</dd>
        <dt>Of which vocabulary</dt><dd>${Math.round((p.embedding_share || 0) * 100)}%</dd>
        <dt>Memory needed</dt><dd>${plan.memory_gb} GB</dd>
        <dt>Batch</dt><dd>${plan.settings?.batch_size} × ${plan.settings?.grad_accum}
          = ${fmtNum(plan.tokens_per_step)} tokens per step</dd>
        <dt>Steps</dt><dd>${fmtNum(plan.settings?.max_steps)}</dd>
        <dt>Learning rate</dt><dd>${(plan.recommended_lr ?? 0).toExponential(1)}</dd>
        <dt>Training it fully</dt><dd>${plan.minutes_for_full
          ? fmtDuration(plan.minutes_for_full * 60) : "—"}</dd>
      </dl>
      <div style="margin-top:10px">
        <span class="badge ${tone}">${v.ratio ?? "?"} tokens per parameter</span>
      </div>
    </div>
    ${raw(issueList(plan.issues))}`;
}

// ===========================================================================
// Step 4 — review
// ===========================================================================

function stepReview(body, ctx) {
  const { state, runners, draw } = ctx;
  const runner = runners.find((r) => r.id === state.runnerId);
  const caps = runner?.capabilities || {};

  if (state.mode === "finetune") {
    const key = `${state.runnerId}|${state.model}|${state.goal}`;
    ensure(state.ftPlan, key, () => api.plan({
      runner_id: state.runnerId,
      params_b: state.modelDetail.data?.params_b ?? null,
      goal: state.goal,
      dataset_rows: 2000,
    }), draw);
    body.innerHTML = finetuneReview(state, runner, caps);
    if (state.ftPlan.status === "ready") wireOverrides(body, ctx);
    return;
  }

  ensure(state.plan, planKey(state), () => api.scratchPlan(planPayload(state)), draw);
  body.innerHTML = scratchReview(state, runner, caps);
  if (state.plan.status === "ready") {
    wireOverrides(body, ctx);
    refreshPlan(ctx, "#reviewIssues", (_s, plan) => issueList(plan.issues));
  }
}

function finetuneReview(state, runner, caps) {
  const r = state.ftPlan;
  if (r.status === "loading") return loading("Working out the best settings for your machine…");
  if (r.status === "error") return failed(r.error);
  if (r.status !== "ready") return "";
  const plan = r.data;
  const s = { ...plan.settings, ...state.overrides };
  const blocked = ["too_big", "needs_quantization"].includes(plan.fit?.verdict);
  state.blocked = blocked;

  return html`
    ${raw(blocked ? html`
      <div class="callout callout-err">
        <strong>This model will not run on ${runner?.name || "this machine"}</strong>
        ${plan.fit.message} Go back and choose a smaller model, or pick a
        machine with more memory.
      </div>` : "")}

    <div class="card" style="margin-bottom:14px">
      <h3 style="margin:0 0 8px">Ready to train</h3>
      <dl class="kv">
        <dt>Teaching it to</dt><dd>${(GOALS.find((g) => g.id === state.goal) || {}).title}</dd>
        <dt>Starting from</dt><dd class="mono">${state.model}</dd>
        <dt>Learning from</dt><dd class="mono">${state.dataset}${
          state.config ? " · " + state.config : ""} · ${state.split}</dd>
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
            ["epochs", "Passes over the data", s.epochs, "number",
             "How many times the model sees every example."],
            ["learning_rate", "Learning rate", s.learning_rate, "text",
             "How big a step each update takes."],
            ["batch_size", "Batch size", s.batch_size, "number",
             "Examples processed at once."],
            ["grad_accum", "Gradient accumulation", s.grad_accum, "number",
             "Batches combined before each update."],
            ["max_seq_len", "Max sequence length", s.max_seq_len, "number",
             "Longer examples are cut to this length."],
            ["lora_r", "LoRA rank", s.lora_r, "number",
             "Size of the trained adapter. Higher learns more and risks more."],
            ["max_steps", "Maximum steps", s.max_steps, "number",
             "Training stops here even if the data has not run out."],
          ]))}
          <div class="field">
            <label for="f_dtype">Number precision</label>
            <select id="f_dtype" data-setting="dtype">
              ${raw(["float16", "bfloat16", "float32"].map((d) =>
                `<option value="${d}"${d === s.dtype ? " selected" : ""}>${d}</option>`).join(""))}
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
}

function scratchReview(state, runner, caps) {
  const r = state.plan;
  if (r.status === "loading") return loading("Working out the settings for your machine…");
  if (r.status === "error") return failed(r.error);
  if (r.status !== "ready") return "";
  const plan = r.data;
  const s = { ...plan.settings, ...state.overrides };
  const a = plan.settings.arch;
  const blocked = plan.blocked;
  state.blocked = blocked;

  return html`
    <div class="card" style="margin-bottom:14px">
      <h3 style="margin:0 0 8px">Ready to build a model</h3>
      <dl class="kv">
        <dt>Building</dt><dd>${plan.params_label} parameters from random weights —
          ${a.num_hidden_layers} layers, ${a.hidden_size} wide,
          ${a.num_attention_heads} heads, ${a.max_position_embeddings} context</dd>
        <dt>Vocabulary</dt><dd>${fmtNum(a.vocab_size)} tokens, built from your text</dd>
        <dt>Learning from</dt><dd class="mono">${state.dataset}${
          state.config ? " · " + state.config : ""} · ${state.split}</dd>
        <dt>Running on</dt><dd>${runner?.name} — ${caps.device_name || ""}</dd>
        <dt>Training time</dt><dd>about ${fmtDuration(state.minutes * 60)}</dd>
        <dt>Text it will read</dt><dd>${fmtNum(s.token_budget)} tokens over
          ${fmtNum(s.max_steps)} steps</dd>
        <dt>Memory needed</dt><dd>about ${plan.memory_gb} GB of
          ${caps.vram_gb || "?"} GB</dd>
      </dl>
    </div>

    <div id="reviewIssues">${raw(issueList(plan.issues))}</div>

    ${raw((plan.notes || []).map((n) => html`
      <div class="callout callout-warn"><strong>Worth knowing</strong>${n}</div>`).join(""))}

    <div class="card" style="margin-bottom:14px">
      <h3>Settings chosen for you</h3>
      <p class="muted tiny">Derived from your machine's measured speed and
        memory. Every one of them can be changed below.</p>
      <div class="grid grid-2" style="margin-top:12px">
        ${raw(plan.explanations.map(explainCard).join(""))}
      </div>
    </div>

    <div class="card" style="margin-bottom:14px">
      <h3 style="margin:0 0 4px">Every training setting</h3>
      <p class="muted tiny">Changing one re-checks the whole plan, and anything
        that looks wrong is explained above.</p>
      <div class="grid grid-3" style="margin-top:12px">
        ${raw(fields(SCRATCH_FIELDS.map(([k, label, type, hint]) =>
          [k, label, s[k] ?? "", type, hint])))}
        <div class="field">
          <label for="f_dtype">Number precision</label>
          <select id="f_dtype" data-setting="dtype">
            ${raw(["float16", "bfloat16", "float32"].map((d) =>
              `<option value="${d}"${d === s.dtype ? " selected" : ""}>${d}</option>`).join(""))}
          </select>
          <div class="hint">Measured fastest here: ${caps.recommended_dtype}.
            Weights are always kept in 32-bit regardless; this only sets what
            the matrix multiplications use.</div>
        </div>
        ${raw(toggle("gradient_checkpointing", "Recompute activations",
          s.gradient_checkpointing,
          "Saves a lot of memory by recomputing during the backward pass "
          + "instead of storing. Costs about 30% of the speed."))}
        ${raw(toggle("optim_8bit", "8-bit optimiser", s.optim_8bit,
          "Stores Adam's two running averages in 8 bits instead of 32. Frees "
          + "memory for a bigger batch at no measurable quality cost."))}
      </div>
    </div>

    <div class="card">
      <div class="field">
        <label for="f_sample_prompt">Sample opening</label>
        <input type="text" id="f_sample_prompt" data-setting="sample_prompt"
               value="${s.sample_prompt}">
        <div class="hint">During training the model is asked to continue this.
          Watching those samples turn from noise into sentences is the clearest
          sign it is working.</div>
      </div>
      <div class="field" style="margin-bottom:0">
        <label for="jobName">Name this run</label>
        <input type="text" id="jobName" value="${plan.params_label} on ${
          String(state.dataset || "").split("/").pop()}">
        <div class="hint">Just so you can find it later.</div>
      </div>
    </div>`;
}

// ===========================================================================
// Shared pieces
// ===========================================================================

function issueList(issues) {
  if (!issues || !issues.length) return "";
  const order = { error: 0, warn: 1, info: 2 };
  const sorted = [...issues].sort((a, b) => order[a.level] - order[b.level]);
  const title = { error: "This will not run", warn: "Worth changing",
                  info: "Worth knowing" };
  return sorted.map((i) => html`
    <div class="callout ${LEVEL_CLASS[i.level] || "callout"}">
      <strong>${title[i.level] || ""}</strong>${i.message}
      ${raw(i.fix ? `<em class="muted" style="display:block;margin-top:4px">${
        esc(i.fix)}</em>` : "")}
    </div>`).join("");
}

const explainCard = (e) => html`
  <div class="card" style="box-shadow:none;background:var(--surface-2)">
    <div class="row-between"><strong class="tiny">${e.setting}</strong>
      <span class="badge badge-accent">${e.value}</span></div>
    <p class="muted tiny" style="margin:6px 0 0">${e.why}</p>
  </div>`;

const fields = (rows) => rows.map(([k, label, v, type, hint]) => html`
  <div class="field">
    <label for="f_${k}">${label}</label>
    <input id="f_${k}" data-setting="${k}" type="${type}" value="${v}">
    ${raw(hint ? `<div class="hint">${esc(hint)}</div>` : "")}
  </div>`).join("");

const toggle = (k, label, on_, hint) => html`
  <div class="field">
    <label for="f_${k}">${label}</label>
    <select id="f_${k}" data-setting="${k}">
      <option value="1"${on_ ? " selected" : ""}>On</option>
      <option value="0"${on_ ? "" : " selected"}>Off</option>
    </select>
    <div class="hint">${hint}</div>
  </div>`;

// Settings that must survive as text/float rather than being coerced to int.
const FLOAT_SETTINGS = new Set(["learning_rate", "weight_decay", "grad_clip",
                                "min_lr_ratio", "epochs"]);
const BOOL_SETTINGS = new Set(["gradient_checkpointing", "optim_8bit"]);

function wireOverrides(body, ctx) {
  const { state, draw } = ctx;
  on(body, "change", "[data-setting]", (_e, t) => {
    const k = t.dataset.setting;
    let v;
    if (BOOL_SETTINGS.has(k)) v = t.value === "1";
    else if (FLOAT_SETTINGS.has(k)) v = parseFloat(t.value);
    else if (t.type === "number") v = +t.value;
    else v = t.value;
    state.overrides[k] = v;
    if (state.mode === "scratch") {
      state.plan = resource();
      refreshPlan(ctx, "#reviewIssues", (_s, plan) => issueList(plan.issues));
    }
  });
}

/** Fetch a scratch plan and paint one panel with it, without a full redraw. */
function refreshPlan(ctx, selector, render) {
  const { state } = ctx;
  const box = $(selector);
  if (box) box.innerHTML = loading("Checking…");
  const key = planKey(state);
  ensure(state.plan, key, () => api.scratchPlan(planPayload(state)), () => {
    const target = $(selector);
    if (!target) return;
    if (state.plan.status === "error") { target.innerHTML = failed(state.plan.error); return; }
    if (state.plan.status !== "ready") return;
    target.innerHTML = render(state, state.plan.data);
    gateNext(state.plan.data.blocked,
             "Fix the problems above before starting.");
  });
  if (state.plan.status === "ready" && box) {
    box.innerHTML = render(state, state.plan.data);
  }
}

const planKey = (state) => JSON.stringify([
  state.runnerId, state.size, state.minutes, state.vocab,
  state.custom, state.overrides, state.dataset, state.corpusTokens]);

function planPayload(state) {
  return {
    runner_id: state.runnerId,
    size: state.size,
    custom: state.custom,
    minutes: state.minutes,
    vocab_size: state.vocab,
    corpus_tokens: state.corpusTokens ?? null,
    sample_prompt: state.samplePrompt || "Once upon a time",
    text_field: state.textField || "text",
    overrides: state.overrides,
  };
}

function gateNext(blocked, hintText) {
  const nextBtn = document.getElementById("nextBtn");
  if (!nextBtn) return;
  nextBtn.disabled = !!blocked;
  const hint = document.getElementById("navHint");
  if (hint) hint.textContent = blocked ? hintText : "";
}

function wireSearch(body, prefix, run) {
  const box = $(`#${prefix}Results`, body);
  const input = $(`#${prefix}Search`, body);
  const go = async () => {
    if (!box) return;
    box.innerHTML = `<span class="muted tiny">Searching…</span>`;
    try { box.innerHTML = await run(input.value.trim()); }
    catch (e) { box.innerHTML = failed(e.message); }
  };
  $(`#${prefix}SearchBtn`, body)?.addEventListener("click", go);
  input?.addEventListener("keydown", (e) => { if (e.key === "Enter") go(); });
}

const STEPS = {
  finetune: [stepGoal, stepModel, stepData, stepReview],
  scratch: [stepGoal, stepData, stepDesign, stepReview],
};

// ===========================================================================

function wireNav(mount, ctx) {
  const { state, draw } = ctx;
  const next = $("#nextBtn", mount);
  const back = $("#backBtn", mount);
  const hint = $("#navHint", mount);
  if (!next) return;

  const names = STEP_NAMES[state.mode];
  const last = names.length - 1;
  const blockers = state.mode === "scratch" ? [
    () => (!state.runnerId ? "Choose a machine to continue." : null),
    () => (!state.dataset ? "Choose some text to continue." : null),
    () => (!state.size ? "Choose a size to continue." : null),
    () => (state.plan.status !== "ready" ? "Working out the settings…"
           : state.blocked ? "Fix the problems above before starting." : null),
  ] : [
    () => (!state.runnerId ? "Choose a machine to continue." : null),
    () => (!state.model ? "Choose a model to continue." : null),
    () => (!state.dataset ? "Choose a dataset to continue." : null),
    () => (state.ftPlan.status !== "ready" ? "Working out the settings…"
           : state.blocked ? "Choose a smaller model to continue." : null),
  ];
  const blocker = blockers[state.step]();

  next.disabled = !!blocker;
  hint.textContent = blocker || "";

  back?.addEventListener("click", () => {
    if (state.step > 0) { state.step--; draw(); }
  });
  next.addEventListener("click", async () => {
    if (state.starting) return;
    if (state.step < last) { state.step++; draw(); return; }

    const job = buildJob(mount, state);
    if (!job) { toast("The plan is not ready yet.", "err"); return; }
    state.starting = true;
    next.disabled = true;
    next.textContent = "Starting…";
    try {
      const { id } = await api.createJob(job);
      toast("Training run created.", "ok");
      location.hash = `#/jobs/${id}`;
    } catch (e) {
      toast(e.message, "err");
      state.starting = false;
      next.disabled = false;
      next.textContent = "Start training";
    }
  });
}

function buildJob(mount, state) {
  const name = $("#jobName", mount)?.value || undefined;
  const dataBits = {
    dataset: state.dataset,
    dataset_config: state.config || null,
    dataset_split: state.split || "train",
  };

  if (state.mode === "scratch") {
    // Guarded: the review used to read .settings straight off a plan that
    // could still be null, which threw instead of explaining itself.
    if (state.plan.status !== "ready") return null;
    const s = { ...state.plan.data.settings, ...state.overrides };
    delete s.fits;
    return {
      name, kind: "pretrain_llm",
      config: { ...dataBits, text_field: state.textField || "text",
                required_runner: state.runnerId, ...s },
    };
  }

  if (state.ftPlan.status !== "ready") return null;
  const s = { ...state.ftPlan.data.settings, ...state.overrides };
  const fmt = state.formatMode
    ? { mode: state.formatMode }
    : (state.preview.data?.format || { mode: "auto" });
  return {
    name, kind: "finetune_llm",
    config: {
      ...dataBits,
      base_model: state.model,
      params_b: state.modelDetail.data?.params_b ?? null,
      goal: state.goal,
      required_runner: state.runnerId,
      format: fmt,
      ...s,
    },
  };
}
