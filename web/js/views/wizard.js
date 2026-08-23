import { api, events } from "../api.js";
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
  ["early_stop", "Stop when it stops improving", "bool",
   "Ends the run once the held-out loss has gone several checks without "
   + "getting better, and keeps whichever version scored best rather than "
   + "whichever came last. The last weights are usually not the best ones."],
  ["early_stop_patience", "Checks before giving up", "number",
   "How many held-out measurements in a row may fail to improve before the "
   + "run ends. Counted in checks, not steps, so it means the same thing in a "
   + "300-step run and a 30,000-step one."],
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

// The words say what is being waited for -- they are different every time and
// worth reading -- and the bar under them holds the space the answer will
// need, so the panel does not grow when it arrives.
const loading = (what) => html`<div class="card muted tiny" aria-busy="true">
  ${what}
  <div class="sk-line shimmer" style="margin-top:10px;width:70%"></div>
  <div class="sk-line shimmer" style="margin-top:6px;width:45%"></div>
</div>`;
const failed = (msg) => html`<div class="callout callout-err">${msg}</div>`;

// ===========================================================================

export async function wizardView(mount) {
  const state = {
    mode: "finetune", step: 0, goal: "instructions", runnerId: null,
    model: null, modelDetail: resource(),
    // A base model that is one of this studio's own finished runs rather than
    // a Hugging Face id. Kept beside `model` instead of inside it because the
    // two are resolved completely differently -- one is downloaded from the
    // Hub, the other is fetched from the controller with the join token -- and
    // collapsing them into one string is how a job id ends up being passed to
    // `from_pretrained`.
    sourceRun: null, myModels: resource(), sourceTemplate: resource(),
    // The data step, shared by both paths.
    dataset: null, configs: resource(), config: null, split: "train",
    // A dataset from this studio's library rather than from the Hub. It is
    // fetched by the runner with the join token, so a private dataset never
    // has to be published to be trained on.
    studioDataset: null,
    preview: resource(), textField: null, formatMode: null,
    // How a conversation becomes training text. "model" uses the base model's
    // own chat template, which is what an instruct model was trained to expect
    // and therefore the right default; "custom" is a Jinja template the user
    // writes; "builtin" is a plain readable rendering.
    templateSource: "model", customTemplate: "", builtin: resource(),
    // Which set of message-boundary tokens a from-scratch model is taught.
    chatFormat: "chatml", formats: resource(), teachReasoning: false,
    // Explicit paths for datasets auto-detection reads wrongly.
    selectors: {}, roleMap: "", selectorFields: resource(),
    // From scratch.
    minutes: 60, size: null, vocab: 8192, custom: null,
    // A mixture of experts is something done to whichever size was chosen,
    // not a size of its own, so it lives beside the size rather than in it.
    moe: { enabled: false, num_local_experts: 8, num_experts_per_tok: 2 },
    sizes: resource(), plan: resource(), ftPlan: resource(),
    overrides: {}, archOverrides: {},
    starting: false,
  };

  try {
    const handed = sessionStorage.getItem("aistudio.dataset");
    if (handed) {
      const d = JSON.parse(handed);
      sessionStorage.removeItem("aistudio.dataset");
      state.studioDataset = d;
      state.dataset = d.name;
    }
  } catch { /* nothing was handed over */ }

  // In parallel: neither needs the other, and doing them in turn doubled the
  // wait before anything at all appeared.
  const [allRunners, starters] = await Promise.all([
    api.runners(), api.starters(),
  ]);
  let known = allRunners;
  let runners = allRunners.filter((r) => r.status !== "offline");
  state.runnerId = runners[0]?.id ?? null;
  state.vocab = starters.default_vocab || 8192;

  const ctx = { state, runners, known, starters, draw: () => draw() };
  // `draw` reads ctx.runners, so the subscription below can refresh the list
  // without rebuilding the context.

  function draw() {
    // Clamp rather than trust: a step index that runs past the end used to
    // throw "steps[state.step] is not a function" and blank the page.
    const names = STEP_NAMES[state.mode] || STEP_NAMES.finetune;
    state.step = Math.max(0, Math.min(state.step | 0, names.length - 1));
    mount.innerHTML = shell(state, ctx.runners, names, ctx.known);
    const body = $("#stepBody", mount);
    if (body) STEPS[state.mode][state.step](body, ctx);
    wireNav(mount, ctx);
  }

  draw();

  // A runner that is reconnecting is offline for a second or two -- after a
  // controller restart, every one of them is. Opening the wizard in that
  // window used to give "No machines are connected" and leave it there
  // forever, because the list was read exactly once. It is not a state to
  // report; it is a state to wait a moment for.
  const recheck = async () => {
    const fresh = await api.runners().catch(() => null);
    if (!fresh) return;
    const live = fresh.filter((r) => r.status !== "offline");
    const changed = live.length !== runners.length
      || live.some((r, i) => r.id !== runners[i]?.id);
    known = fresh;
    ctx.known = fresh;
    if (!changed) return;
    runners = live;
    ctx.runners = live;
    if (!state.runnerId || !live.some((r) => r.id === state.runnerId)) {
      state.runnerId = live[0]?.id ?? null;
    }
    // Only when the screen would actually be wrong. Redrawing under somebody
    // halfway through a form is worse than a slightly stale machine list.
    if (state.step === 0 || !live.length) draw();
  };

  const stop = events.subscribe((msg) => {
    if (msg.type === "runners_changed" || msg.type === "_connected") recheck();
  });
  // Belt as well as braces, and only while there is nothing to work with: the
  // socket event is the fast path, but a page opened during a restart may
  // have connected before the runner did and would then wait for an event
  // that already happened.
  const timer = setInterval(() => { if (!runners.length) recheck(); }, 3000);
  return () => { stop(); clearInterval(timer); };
}

function shell(state, runners, names, known = []) {
  if (!runners.length && known.length) {
    // A machine this studio knows about, currently not answering. After a
    // controller restart that is every machine, for as long as it takes them
    // to dial back in -- and telling somebody to go and connect a machine
    // they already own, while it is in the middle of reconnecting, is the
    // wrong instruction as well as the wrong diagnosis.
    return html`
      <div class="page-head"><h1>New training run</h1></div>
      <div class="card">
        <div class="row" style="gap:12px;align-items:flex-start">
          <div class="sk-value shimmer" style="min-width:34px;height:34px;
               border-radius:50%;flex:none"></div>
          <div>
            <h3 style="margin:0 0 2px">Waiting for
              ${known.length === 1 ? known[0].name : "your machines"}</h3>
            <p class="muted tiny" style="margin:0">Connected before, not
              answering right now — usually a few seconds after the studio
              itself restarts. This page carries on by itself the moment it
              is back.</p>
            <p class="tiny" style="margin:8px 0 0">
              <a href="#/runners">See what the machines are doing</a></p>
          </div>
        </div>
      </div>`;
  }
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
      moe: { enabled: false, num_local_experts: 8, num_experts_per_tok: 2 },
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
  // Exactly the runs that produced a usable model, already filtered to what
  // this account may see -- the same list the playground offers, for the same
  // reason.
  ensure(state.myModels, "all", () => api.playground(), draw);
  if (state.sourceRun) {
    // The template that model carries, read out of its own tokenizer, so the
    // preview shows the shape the run will actually train on rather than a
    // generic rendering of it.
    ensure(state.sourceTemplate, state.sourceRun.id,
           () => api.jobChatTemplate(state.sourceRun.id), draw);
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

    ${raw(ownModelPanel(state))}

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
    state.sourceRun = null;
    state.sourceTemplate = resource();
    state.modelDetail = resource();
    state.ftPlan = resource();
    draw();
  });

  on(body, "click", "[data-own-model]", (_e, t) => {
    const own = (state.myModels.data || []).find((m) => m.id === t.dataset.ownModel);
    state.sourceRun = own || null;
    state.model = null;
    state.modelDetail = resource();
    state.sourceTemplate = resource();
    state.ftPlan = resource();
    draw();
  });

  on(body, "click", "#clearOwnModel", () => {
    state.sourceRun = null;
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

function ownModelPanel(state) {
  const r = state.myModels;
  const rows = (r.data || []).filter((m) => !m.stopped_early || m.kind);
  if (state.sourceRun) {
    const src = state.sourceRun;
    const tmpl = state.sourceTemplate;
    return html`
      <div class="card callout-ok" style="margin-top:14px">
        <div class="row-between" style="gap:8px;flex-wrap:wrap">
          <div>
            <h3 style="margin:0">Starting from your own model</h3>
            <p class="muted tiny" style="margin:4px 0 0">
              <strong>${src.name}</strong> —
              ${src.kind === "pretrain_llm"
                ? "a complete model this studio built. It becomes the base, and this run teaches it your task."
                : "an adapter. This run carries on training it, keeping everything it already learned."}
            </p>
            ${raw(src.kind !== "pretrain_llm" ? html`
              <p class="muted tiny" style="margin:4px 0 0">Its base model,
                <code>${src.base_model || "—"}</code>, stays the same.</p>` : "")}
            ${raw(tmpl.status === "ready" && !tmpl.data.available ? html`
              <p class="muted tiny" style="margin:4px 0 0">It carries no chat
                template of its own, so the plain readable format is used.</p>` : "")}
          </div>
          <button class="btn-sm" id="clearOwnModel">Use a Hugging Face model instead</button>
        </div>
      </div>`;
  }
  if (r.status === "loading") return "";
  if (!rows.length) return "";
  return html`
    <details class="adv" style="margin-top:14px">
      <summary>Or start from one of your own models
        <span class="muted tiny">(${rows.length})</span></summary>
      <p class="muted tiny" style="margin:10px 0 0">
        A model this studio produced can be a base like any other. For a model
        built from scratch this teaches it your task; for a fine-tune it
        carries the same adapter on with new data instead of starting a
        second one.</p>
      <div class="grid grid-2" style="margin-top:10px">
        ${raw(rows.map((m) => html`
          <button class="pick" data-own-model="${m.id}">
            <span class="t">${m.name}
              <span class="badge">${m.kind === "pretrain_llm"
                ? "from scratch" : "adapter"}</span>
              ${raw(m.stopped_early
                ? `<span class="badge badge-warn">stopped early</span>` : "")}
            </span>
            <span class="d">${m.kind === "pretrain_llm"
              ? (m.size || "built here") : (m.base_model || "")}</span>
          </button>`).join(""))}
      </div>
    </details>`;
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
        <dt>Architecture</dt><dd>${d.architecture || "—"}${raw(
          d.moe ? ` <span class="badge badge-accent">mixture of experts</span>` : "")}</dd>
        <dt>Downloads</dt><dd>${fmtNum(d.downloads)}</dd>
        ${raw(d.memory ? `<dt>Memory needed</dt><dd>${d.memory.fp16_gb} GB in
          16-bit · ${d.memory.int4_gb} GB in 4-bit</dd>` : "")}
      </dl>
      ${raw(d.moe ? `<div class="callout" style="margin-top:10px">
        <strong>This is a mixture-of-experts model</strong>Each block holds
        several feed-forward networks and a router picks a couple of them per
        token. Only some of it does the work on any given token, but
        <em>all</em> of it has to be in memory — so judge it by the size above,
        not by any smaller "active" figure in its name. The adapter goes on
        attention, which every token passes through whatever the router
        decides.</div>` : "")}
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
      ensure(state.preview, previewKey(state, scratch),
             () => api.trainingPreview(previewRequest(state, scratch)), draw);
      if (state.preview.status === "ready") {
        state.previewIsChat = state.preview.data.format?.mode === "chat";
      }
    }
  }
  // Fetched once, and only used to seed the editor when someone chooses to
  // write their own.
  if (!scratch) ensure(state.builtin, "builtin", () => api.builtinTemplate(), draw);
  if (scratch) ensure(state.formats, "formats", () => api.chatFormats(), draw);
  ensure(state.selectorFields, "sel", () => api.selectorFields(), draw);

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
    state.templateSource = "model";
    state.customTemplate = "";
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
  on(body, "click", "#teachReasoning", (_e, t) => {
    state.teachReasoning = !state.teachReasoning;
    state.preview = resource();
    draw();
  });
  on(body, "click", "[data-chatfmt]", (_e, t) => {
    state.chatFormat = t.dataset.chatfmt;
    state.templateSource = "builtin";
    state.preview = resource();
    draw();
  });
  on(body, "click", "[data-tmplsrc]", (_e, t) => {
    state.templateSource = t.dataset.tmplsrc;
    if (state.templateSource === "custom" && !state.customTemplate) {
      // Start from something that already works rather than a blank box.
      const b = state.builtin.data || {};
      const det = state.preview.data?.format || {};
      state.customTemplate = det.mode === "chat"
        ? (b.template || "") : (b.instruction_template || "");
    }
    state.preview = resource();
    draw();
  });
  // Applied on demand, not per keystroke: re-rendering four examples on every
  // character would fight the cursor and hammer the dataset server.
  // Expanding rewrites one paragraph in place. Going through draw() would
  // re-render the step and fold it straight back up.
  on(body, "click", "[data-expand]", (_e, t) => {
    const card = t.closest(".sample");
    const para = card && card.querySelector(".txt");
    const idx = [...body.querySelectorAll(".sample")].indexOf(card);
    const full = state.preview.data?.rendered?.[idx]?.text;
    if (!para || full == null) return;
    const wasExpanded = para.dataset.full === "0";
    para.textContent = wasExpanded ? shorten(full) : full;
    para.dataset.full = wasExpanded ? "1" : "0";
    t.textContent = wasExpanded
      ? "Show all " + fmtNum(full.length) + " characters" : "Show less";
  });
  on(body, "click", "#applySelectors", () => {
    const next = {};
    $$("[data-selector]", body).forEach((el) => {
      const v = el.value.trim();
      if (v) next[el.dataset.selector] = v;
    });
    state.selectors = next;
    state.roleMap = ($("[data-rolemap]", body) || {}).value || "";
    state.preview = resource();
    draw();
  });
  on(body, "click", "#applyTemplate", () => {
    const box = $("#templateBox", body);
    if (box) state.customTemplate = box.value;
    state.preview = resource();
    draw();
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

    <div style="margin-top:14px">${raw(templatePanel(state, scratch))}</div>
    <div style="margin-top:14px">${raw(trainingText(state, scratch))}</div>`;
}

const TEMPLATE_SOURCES = [
  { id: "model", title: "The model's own format",
    desc: "Every instruct model was trained to expect one exact layout, and "
        + "ships it in its own files. Using it is almost always right — give a "
        + "model a shape it has never seen and it ignores half of what you "
        + "taught it." },
  { id: "builtin", title: "Plain and readable",
    desc: "Roles written out as text. Fine for a base model, and easy to read "
        + "when you are checking the data rather than the format." },
  { id: "custom", title: "Write it yourself",
    desc: "A Jinja template with the conversation and tools handed to it. Full "
        + "control, for a layout neither of the others produces." },
];

function templatePanel(state, scratch) {
  const p = state.preview;
  const det = p.status === "ready" ? (p.data.format || {}) : {};
  const isChat = det.mode === "chat";
  const src = state.templateSource;
  // A model being built from scratch has no "own format" to borrow -- it has
  // never been trained on anything. It learns whichever shape it is shown, so
  // the choice is between the plain rendering and one written by hand.
  const sources = scratch
    ? TEMPLATE_SOURCES.filter((t) => t.id !== "model")
    : TEMPLATE_SOURCES;
  if (scratch && !isChat && src === "model") state.templateSource = "builtin";

  return html`
    <div class="card">
      <div class="row-between" style="margin-bottom:4px">
        <h3 style="margin:0">How a row becomes training text</h3>
        ${raw(p.status === "ready" && p.data.template_source
          ? `<span class="badge badge-accent">using: ${esc(p.data.template_source)}</span>` : "")}
      </div>
      ${raw(scratch && isChat ? html`
        <div class="callout" style="margin:0 0 12px">
          <strong>A conversation, learned from nothing</strong>
          This dataset is a conversation, and a model built from scratch can
          learn its shape along with the language — the roles below become part
          of what it writes. Whatever you choose here is exactly what the
          Playground will speak to it afterwards.
        </div>` : "")}
      ${raw(isChat ? html`
        <p class="muted tiny" style="margin:0 0 10px">
          This is a conversation dataset. Roles found:
          ${raw((det.roles || []).map((r) =>
            `<span class="badge">${esc(r)}</span>`).join(" "))}
          ${raw(det.tools_field
            ? `<span class="badge badge-accent">tool definitions in
               "${esc(det.tools_field)}"</span>` : "")}
          ${raw(det.has_tool_calls
            ? `<span class="badge badge-accent">tool calls</span>` : "")}
        </p>` : html`
        <p class="muted tiny" style="margin:0 0 10px">
          Rows are read as ${det.mode || "…"}. Change the template below if
          that is not the shape you want the model to learn.</p>`)}

      ${raw(scratch && isChat
        ? formatPicker(state)
        : html`<div class="grid grid-3">
        ${raw(sources.map((t) => html`
          <button class="pick ${src === t.id ? "selected" : ""}" data-tmplsrc="${t.id}">
            <span class="t">${t.title}
              ${raw(t.id === "model" ? `<span class="badge badge-ok">recommended</span>` : "")}
            </span>
            <span class="d">${t.desc}</span>
          </button>`).join(""))}
      </div>`)}

      ${raw(p.status === "ready" && p.data.template_note ? html`
        <div class="callout callout-warn" style="margin-top:12px">
          <strong>No template on this model</strong>${p.data.template_note}
          Falling back to the plain readable form.
        </div>` : "")}

      ${raw(src === "custom" ? html`
        <div class="field" style="margin-top:12px">
          <label for="templateBox">Jinja template</label>
          <textarea id="templateBox" rows="10" spellcheck="false"
                    class="mono">${state.customTemplate}</textarea>
          <div class="hint">
            Available: <code>messages</code> (each with
            <code>role</code>, <code>content</code>, <code>tool_calls</code>,
            <code>train</code>), <code>tools</code>, and every column of the row
            by name. Example:
            <code>{% for m in messages %}{{ m.role }}: {{ m.content }}
            {% endfor %}</code>
          </div>
          <div class="row" style="margin-top:8px">
            <button class="btn-primary btn-sm" id="applyTemplate">Apply and preview</button>
          </div>
        </div>` : "")}

      ${raw(p.status === "ready" && (p.data.system_prompts || []).length ? html`
        <details class="adv" style="margin-top:10px">
          <summary>System prompt found in this data</summary>
          <p class="muted tiny" style="margin:8px 0 4px">Kept with the run, and
            offered again in the Playground — a model trained with a system
            prompt behaves differently without it.</p>
          <p class="txt mono tiny" style="white-space:pre-wrap;max-height:180px;
             overflow:auto;background:var(--surface-2);padding:10px;
             border-radius:8px">${p.data.system_prompts[0].slice(0, 1500)}</p>
        </details>` : "")}

      ${raw(selectorPanel(state))}

      ${raw(p.status === "ready" && p.data.template_error ? html`
        <div class="callout callout-err" style="margin-top:12px">
          <strong>That template did not work</strong>${p.data.template_error}
        </div>` : "")}
    </div>`;
}

// How much of an example is shown before it is folded. Long enough to see
// the shape of a conversation, short enough that several still fit a screen.
const SHORT = 2600;

function shorten(text) {
  if (!text || text.length <= SHORT) return text;
  const head = Math.round(SHORT * 0.6);
  const tail = SHORT - head;
  return text.slice(0, head)
    + "\n\n… " + fmtNum(text.length - SHORT) + " characters hidden …\n\n"
    + text.slice(-tail);
}

function formatPicker(state) {
  const r = state.formats;
  if (r.status === "loading") return loading("Loading the formats…");
  if (r.status === "error") return failed(r.error);
  if (r.status !== "ready") return "";
  const formats = r.data.formats || [];

  return html`
    <div class="callout" style="margin:0 0 12px">
      <strong>Why this matters</strong>
      Written as plain text, "assistant" is just a word — the model has to
      guess where a turn ends from punctuation, and generation has nothing
      dependable to stop on. These formats reserve <em>single, atomic tokens</em>
      for the boundaries before the vocabulary is trained, which only a
      from-scratch run can do: adding tokens to an existing model's tokenizer
      would leave its embedding table the wrong size.
    </div>
    <div class="grid grid-2">
      ${raw(formats.map((f) => html`
        <button class="pick ${state.chatFormat === f.id ? "selected" : ""}"
                data-chatfmt="${f.id}">
          <span class="t">${f.label}
            <span class="badge">${f.token_count} token${f.token_count > 1 ? "s" : ""}</span>
            ${raw(f.id === "chatml" ? `<span class="badge badge-ok">recommended</span>` : "")}
          </span>
          <span class="d">${f.blurb}</span>
          <span class="mono tiny" style="display:block;background:var(--surface-2);
                padding:7px 9px;border-radius:6px;white-space:pre-wrap;
                word-break:break-all;color:var(--text-2)">${f.sample}</span>
          <span class="row" style="gap:5px;flex-wrap:wrap">
            ${raw(f.specials.map((t) =>
              `<span class="badge">${esc(t)}</span>`).join(""))}
          </span>
        </button>`).join(""))}
    </div>
    ${raw(reasoningPanel(state, formats))}

    <details class="adv" style="margin-top:10px">
      <summary>Or write the layout yourself</summary>
      <p class="muted tiny" style="margin:8px 0 0">Choosing "write it yourself"
        below gives you the Jinja, but no reserved tokens — anything you invent
        is split into ordinary pieces by the tokenizer. Use one of the formats
        above unless you have a reason not to.</p>
      <div class="row" style="margin-top:8px">
        <button class="btn-sm ${state.templateSource === "custom" ? "btn-primary" : ""}"
                data-tmplsrc="custom">Write it yourself</button>
        ${raw(state.templateSource === "custom"
          ? `<button class="btn-sm" data-tmplsrc="builtin">Back to a standard format</button>` : "")}
      </div>
    </details>`;
}

function reasoningPanel(state, formats) {
  const det = state.preview.status === "ready"
    ? (state.preview.data.format || {}) : {};
  const chosen = formats.find((f) => f.id === state.chatFormat) || {};
  const on = state.teachReasoning;
  return html`
    <div class="card" style="margin-top:12px;box-shadow:none;background:var(--surface-2)">
      <div class="row-between" style="flex-wrap:wrap;gap:8px">
        <div style="flex:1;min-width:240px">
          <strong class="tiny">Teach it to reason before answering</strong>
          <p class="muted tiny" style="margin:4px 0 0">
            ${raw(det.has_reasoning
              ? "This dataset contains worked reasoning as well as answers. "
              + "Trained on it, the model writes out its thinking first and "
              + "the answer after — and the two stay separable afterwards."
              : "This dataset has no reasoning in it, so there is nothing to "
              + "learn from. Turning this on would only teach the model to "
              + "open an empty block.")}
          </p>
          ${raw(on && chosen.reasoning_note
            ? `<p class="muted tiny" style="margin:6px 0 0"><em>${
                esc(chosen.label)}: ${esc(chosen.reasoning_note)}</em></p>` : "")}
        </div>
        <button class="btn-sm ${on ? "btn-primary" : ""}" id="teachReasoning"
                ${det.has_reasoning ? "" : "disabled"}>
          ${on ? "On" : "Off"}
        </button>
      </div>
    </div>`;
}

function selectorPanel(state) {
  const r = state.selectorFields;
  if (r.status !== "ready") return "";
  const found = state.preview.status === "ready"
    ? (state.preview.data.resolved || {}) : {};
  const anySet = Object.values(state.selectors).some((v) => v)
    || state.roleMap.trim();

  return html`
    <details class="adv" style="margin-top:10px" ${anySet ? "open" : ""}>
      <summary>Where the fields are${raw(anySet
        ? ` <span class="badge badge-accent">mapped by hand</span>`
        : ` <span class="badge">found automatically</span>`)}</summary>

      <div class="card" style="margin-top:10px">
        <p class="muted tiny" style="margin:0 0 10px">
          Detection covers the shapes that recur, and there are always datasets
          it reads wrongly. Point at a field directly with a dotted path —
          <code>invoke.tool</code>, <code>args[0].value</code>, or
          <code>content.tool_name</code> to look inside a message whose content
          is itself JSON. Leave one blank to keep detecting it.
        </p>

        <div class="grid grid-3">
          ${raw(r.data.fields.map((f) => html`
            <div class="field">
              <label for="sel_${f.id}">${f.id}</label>
              <input id="sel_${f.id}" data-selector="${f.id}" type="text"
                     class="mono" placeholder="detect"
                     value="${state.selectors[f.id] || ""}">
              <div class="hint">${f.hint}</div>
            </div>`).join(""))}
          <div class="field">
            <label for="sel_rolemap">role names</label>
            <input id="sel_rolemap" data-rolemap type="text" class="mono"
                   placeholder="tool_out=tool, narrator=system"
                   value="${state.roleMap}">
            <div class="hint">Rename this dataset's own role words to
              ${(r.data.roles || []).join(", ")}.</div>
          </div>
        </div>

        <button class="btn-primary btn-sm" id="applySelectors">Apply and preview</button>

        ${raw(found.turns ? html`
          <div class="callout" style="margin-top:12px">
            <strong>What that found</strong>
            ${found.turns} turns across the sampled rows · roles
            ${raw((found.roles || []).map((x) =>
              `<span class="badge">${esc(x)}</span>`).join(" "))}
            ${raw(found.with_reasoning
              ? ` · <span class="badge badge-accent">${found.with_reasoning} with reasoning</span>` : "")}
            ${raw(found.empty_content
              ? ` · <span class="badge badge-warn">${found.empty_content} with no content</span>` : "")}
            ${raw((found.tool_calls || []).length ? html`
              <div class="mono tiny" style="margin-top:8px">
                ${raw(found.tool_calls.map((c) =>
                  `<div>call ${esc(c.name)}(${esc(c.arguments)})</div>`).join(""))}
                ${raw((found.tool_results || []).map((t) =>
                  `<div>result from ${esc(t.name || "?")}: ${esc(t.content)}</div>`).join(""))}
              </div>` : "")}
          </div>` : "")}
      </div>
    </details>`;
}

function previewKey(state, scratch) {
  return JSON.stringify([state.dataset, state.config, state.split,
                         state.formatMode, state.textField, state.model,
                         // Included, or switching to a studio model would keep
                         // showing the preview built for the previous base.
                         state.sourceRun?.id,
                         state.sourceTemplate?.status,
                         state.templateSource, state.customTemplate,
                         state.chatFormat, state.teachReasoning,
                         state.selectors, state.roleMap]);
}

/** "a=b, c=d" as an object; blank entries ignored. */
function parseRoleMap(text) {
  const out = {};
  (text || "").split(",").forEach((pair) => {
    const [from, to] = pair.split("=").map((x) => (x || "").trim());
    if (from && to) out[from] = to;
  });
  return out;
}

function selectorsFor(state) {
  const sel = { ...state.selectors };
  const map = parseRoleMap(state.roleMap);
  if (Object.keys(map).length) sel.role_map = map;
  return Object.keys(sel).length ? sel : null;
}

function previewRequest(state, scratch) {
  const fmt = {};
  if (state.formatMode) fmt.mode = state.formatMode;
  // The same three sources for both paths, minus the one a from-scratch model
  // cannot have. Building the format here rather than per-path is what lets a
  // conversation dataset train a model from nothing and then be talked to in
  // the same shape afterwards.
  if (!scratch && state.templateSource === "model" && state.sourceRun) {
    // The model is one of ours, so its template is text we already fetched
    // rather than something the controller can look up by id. Training still
    // records `use_model_template`, and the runner reads it back off the same
    // tokenizer -- so what the preview shows and what trains agree.
    const t = state.sourceTemplate;
    if (t.status === "ready" && t.data.chat_template) {
      fmt.mode = "jinja";
      fmt.template = t.data.chat_template;
    }
  } else if (!scratch && state.templateSource === "model") {
    fmt.use_model_template = true;
  } else if (scratch && state.templateSource !== "custom" && state.previewIsChat) {
    // A named format carries its own Jinja and its own reserved tokens; the
    // controller resolves the name so the preview shows what will be trained.
    fmt.chat_format = state.chatFormat;
    fmt.mode = "chat";
    if (state.teachReasoning) fmt.reasoning = true;
  } else if (state.templateSource === "custom" && state.customTemplate) {
    // "jinja" rather than a chat template, so the same box works whether or
    // not the dataset is a conversation.
    fmt.mode = "jinja";
    fmt.template = state.customTemplate;
  }
  const sel = selectorsFor(state);
  if (sel) fmt.selectors = sel;
  return {
    dataset: state.dataset,
    config: state.config,
    split: state.split,
    format: Object.keys(fmt).length ? fmt : null,
    // A plain-text corpus still names its column; a conversation does not.
    text_field: (scratch && state.templateSource === "builtin"
                 && !state.previewIsChat) ? state.textField : null,
    // A studio model is not on the Hub, so there is nothing to look its
    // template up by. Its template is passed as text instead, fetched from
    // the run itself -- see the chat_template branch above.
    base_model: scratch ? null : state.model,
  };
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
      <div class="hint">Detected: ${mode}.</div>
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
  // A failed template makes every row unreadable. Saying so *and* accusing the
  // data of being empty sends the reader looking in the wrong place; the
  // template panel above already shows the real reason.
  const blame = !p.data.template_error;
  return html`
    ${raw(blame && counts.unreadable ? html`
      <div class="callout callout-err">
        <strong>${counts.unreadable} of ${counts.sampled} sampled rows could not be read</strong>
        These rows have content, but the columns chosen above do not reach it,
        so training would skip them. The dataset's columns are:
        ${(p.data.columns || []).join(", ")}.
      </div>` : "")}
    ${raw(blame && counts.empty ? html`
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
              <span class="row" style="gap:6px">
              ${raw(r.length ? `<span class="badge">${fmtNum(r.length)} characters</span>` : "")}
              ${raw(r.status === "unreadable"
                ? `<span class="badge badge-err">columns do not match</span>`
                : r.status === "empty"
                ? `<span class="badge">blank row</span>` : "")}</span></div>
            <p class="txt" data-full="${r.length > SHORT ? "1" : ""}">${
                r.length > SHORT ? shorten(r.text) : (r.text || "(nothing)")}</p>
              ${raw(r.length > SHORT ? `<button class="btn-sm" data-expand
                  style="margin-top:6px">Show all ${fmtNum(r.length)} characters</button>` : "")}
          </div>`).join(""))}
      </div>
    </div>`;
}

// ===========================================================================
// Step 3 (from scratch) — designing the model
// ===========================================================================

function stepDesign(body, ctx) {
  const { state, draw } = ctx;
  const key = `${state.runnerId}|${state.minutes}|${state.vocab}|` +
              JSON.stringify(state.moe);
  ensure(state.sizes, key,
         () => api.scratchSizes(state.runnerId, state.minutes, state.vocab,
                                state.moe), draw);

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

    ${raw(moePanel(state))}

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
  on(body, "click", "[data-moe]", (_e, t) => {
    state.moe = { ...state.moe, enabled: t.dataset.moe === "on" };
    state.sizes = resource(); state.plan = resource();
    draw();
  });
  // `change`, not `input`: every keystroke here re-scores all five sizes on
  // the server, and a half-typed "1" on the way to "16" is a different model.
  on(body, "change", "[data-moef]", (_e, t) => {
    const v = Math.max(1, +t.value || 1);
    state.moe = { ...state.moe, [t.dataset.moef]: v };
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
      // Heads and feed-forward width start as null, meaning "work it out
      // from the width". They are shown as whatever the plan resolved them
      // to, and they follow the width until somebody moves them by hand.
      state.custom = {
        num_hidden_layers: from.layers, hidden_size: from.dim,
        max_position_embeddings: from.seq,
        num_attention_heads: null, intermediate_size: null,
      };
      state.touched = {};
    }
    draw();
  });

  if (custom) wireDesigner(body, ctx);
}

function moePanel(state) {
  const m = state.moe;
  const chosen = (state.sizes.data?.sizes || []).find((s) => s.id === state.size);
  return html`
    <details class="adv" ${m.enabled ? "open" : ""} style="margin-top:14px">
      <summary>Make it a mixture of experts${raw(m.enabled
        ? ` <span class="badge badge-accent">${m.num_local_experts} experts,
             ${m.num_experts_per_tok} per token</span>` : "")}</summary>
      <div class="card" style="margin-top:10px">
        <p class="muted tiny" style="margin:0 0 10px">
          Instead of one feed-forward network per block, the model gets several
          — the <em>experts</em> — plus a small router that sends each token to
          just a few of them. Parameters go up; the arithmetic per token
          does not.</p>
        <p class="muted tiny" style="margin:0 0 12px">
          <strong>What it does not do is save memory.</strong> Every expert is
          held on the card and every expert is trained, whether or not a
          particular token visits it. And because each expert only learns from
          the tokens the router sends it, the model needs considerably more
          text than its shape suggests. On one GPU this is usually a worse
          trade than simply choosing a bigger size — it is here because it is
          how the large open models are built, and worth being able to try.</p>

        <div class="row" style="gap:8px;flex-wrap:wrap;margin-bottom:12px">
          <button class="btn-sm ${!m.enabled ? "btn-primary" : ""}"
                  data-moe="off">One network per block</button>
          <button class="btn-sm ${m.enabled ? "btn-primary" : ""}"
                  data-moe="on">Mixture of experts</button>
        </div>

        ${raw(!m.enabled ? "" : html`
          <div class="grid grid-2">
            <div class="field">
              <label for="moeExperts">Experts per block</label>
              <input id="moeExperts" data-moef="num_local_experts" type="number"
                     min="2" max="64" value="${m.num_local_experts}">
              <div class="hint">How many separate feed-forward networks each
                block holds. Every one of them takes memory and has to be
                trained.</div>
            </div>
            <div class="field">
              <label for="moeActive">Used per token</label>
              <input id="moeActive" data-moef="num_experts_per_tok" type="number"
                     min="1" max="16" value="${m.num_experts_per_tok}">
              <div class="hint">How many the router picks for each token. Two
                is the usual choice; picking all of them makes it an ordinary
                model that costs far more to run.</div>
            </div>
          </div>
          ${raw(chosen ? html`
            <div class="callout" style="margin-top:4px">
              <strong>${chosen.label} as a mixture of experts</strong>
              holds ${chosen.params_label} parameters and uses
              ${chosen.active_params_label} of them on any given token.
              Memory and the amount of text it needs follow the first number;
              speed follows the second.
              <em class="muted">Its weights and optimiser state alone take
                ${chosen.weights_gb} GB, before a single batch. Read the
                memory figure on the cards with care — a bigger model is given
                a smaller batch to fit, so its total can come out lower while
                the model itself is several times larger.</em>
            </div>` : "")}`)}
      </div>
    </details>`;
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
        ${raw(s.experts
          ? `<span class="badge badge-accent">${esc(s.active_params_label)}
               active</span>` : "")}
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

// Fallback stops, used for the first paint only. The real ones come back with
// every plan, because what counts as a sensible head count depends on the
// width and what fits depends on the whole shape.
const FALLBACK_STOPS = {
  num_hidden_layers: [1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 64],
  hidden_size: [128, 192, 256, 320, 384, 448, 512, 640, 768, 896, 1024, 1152,
                1280, 1536, 1792, 2048, 2560, 3072, 4096],
  num_attention_heads: [1, 2, 4, 6, 8, 12, 16, 24, 32],
  max_position_embeddings: [128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096,
                            6144, 8192],
  intermediate_size: [512, 1024, 1536, 2048, 2752, 3072, 4096, 5504, 8192],
};

const nearestIndex = (stops, value) => {
  let best = 0;
  stops.forEach((v, i) => {
    if (Math.abs(v - value) < Math.abs(stops[best] - value)) best = i;
  });
  return best;
};

/** The scales this designer is currently drawing against.
 *  The plan carries them; until the first plan lands, the fallbacks do. */
function scalesFor(state) {
  const fromPlan = state.plan?.data?.scales;
  const out = {};
  for (const [k, stops] of Object.entries(FALLBACK_STOPS)) {
    const live = fromPlan?.[k];
    out[k] = { stops: live?.stops?.length ? live.stops : stops,
               fits_up_to: live?.fits_up_to ?? null };
  }
  return out;
}

function designer(state) {
  // What the server actually built, with anything explicitly set on top. A
  // field left alone is DERIVED -- feed-forward width from model width, for
  // instance -- and showing a made-up default for it meant the box read 2752
  // while the model being planned had 704. The number on the screen has to be
  // the number that gets trained.
  const planned = state.plan?.data?.settings?.arch || {};
  const c = { ...planned };
  for (const [k, v] of Object.entries(state.custom || {})) {
    if (v != null) c[k] = v;
  }
  const scales = scalesFor(state);
  return html`
    <div class="card" style="margin-top:14px">
      <h3 style="margin:0 0 4px">Design your model</h3>
      <p class="muted tiny">Drag for sensible values, or type an exact one. The
        sliders click through shapes that actually work — head counts that
        divide the width, context lengths that tile evenly — and the marked
        point on each track is where this machine runs out of memory.</p>
      <div class="grid grid-2" style="margin-top:12px;align-items:start">
        <div>
          ${raw(ARCH_FIELDS.map(([k, label, hint]) =>
            sliderRow(k, label, hint, c[k], scales[k])).join(""))}
        </div>
        <div id="designPreview"></div>
      </div>
    </div>`;
}

function sliderRow(key, label, hint, value, scale) {
  const stops = scale.stops;
  // Only reached before the first plan comes back, when there is nothing
  // authoritative to show yet.
  const current = value ?? stops[Math.floor(stops.length / 2)];
  const index = nearestIndex(stops, current);
  const limit = scale.fits_up_to;
  // Where along the track the machine gives out, as a percentage, so the part
  // beyond it can be shaded rather than explained in a sentence nobody reads.
  // `null` means there is no machine to check against; `0` means nothing on
  // this track fits, which is the opposite thing and must not shade green.
  // The thumb sits at index/(N-1) of the track, so the boundary between the
  // last stop that fits and the first that does not is halfway between them.
  // Dividing by N instead of N-1 left a sliver of red on a track where
  // everything fits, which reads as a limit that is not there.
  const fitPct = limit == null ? 100
    : limit === 0 ? 0
    : limit >= stops[stops.length - 1] ? 100
    : Math.min(100, Math.round(100 * (nearestIndex(stops, limit) + 0.5)
                               / Math.max(stops.length - 1, 1)));
  const over = limit != null && limit !== null && current > limit;
  return html`
    <div class="field slider-field">
      <label for="a_${key}">${label}
        <span class="slider-value ${over ? "over" : ""}">${current}</span></label>
      <div class="slider-line">
        <input type="range" id="a_${key}" data-arch-range="${key}"
               min="0" max="${stops.length - 1}" step="1" value="${index}"
               style="--fit:${fitPct}%"
               aria-valuetext="${current}">
        <input type="number" class="slider-num" data-arch="${key}"
               min="1" value="${current}" inputmode="numeric">
      </div>
      <div class="hint">${hint}${raw(!over ? "" : limit === 0
        ? ` <strong class="warn-text">Nothing on this scale fits while the
            rest of the model is this big. Bring the width, the depth or the
            context down first.</strong>`
        : ` <strong class="warn-text">Past ${limit} this will not fit on this
            machine unless something else comes down.</strong>`)}</div>
    </div>`;
}

function wireDesigner(body, ctx) {
  const { state } = ctx;
  const refresh = () => refreshPlan(ctx, "#designPreview", designSummary, () => {
    // A new plan may have re-derived the fields nobody has touched, and may
    // have moved the memory marks on every track. Redrawn only when nothing
    // is being dragged, so the panel never changes under a finger.
    if (document.activeElement?.closest?.("#designer")) return;
    const host = $("#designer", body);
    if (!host) return;
    host.innerHTML = designer(state);
    // `designer()` rebuilds the summary panel along with the sliders, so the
    // answer that just arrived has to be put back into it -- otherwise the
    // redraw blanks the very thing that triggered it.
    const preview = $("#designPreview", host);
    if (preview) preview.innerHTML = designSummary(state, state.plan.data);
  });
  refresh();

  let timer = null;
  const settle = () => {
    state.plan = resource();
    clearTimeout(timer);
    timer = setTimeout(refresh, 350);
  };

  // Redrawn in place rather than through draw(), so that dragging a slider
  // does not destroy the slider being dragged. Only the readout and the
  // partner field move while the finger is down.
  // Which dimensions the person has actually set. Everything else is derived
  // and keeps following what it is derived from -- widen the model and the
  // head count and feed-forward width move with it, which is what makes the
  // sliders feel like they know about each other.
  const DERIVED_FROM_WIDTH = ["num_attention_heads", "intermediate_size"];
  state.touched = state.touched || {};

  const setValue = (key, value, from) => {
    state.touched[key] = true;
    state.custom = { ...state.custom, [key]: value };
    if (key === "hidden_size") {
      for (const dep of DERIVED_FROM_WIDTH) {
        if (!state.touched[dep]) state.custom[dep] = null;
      }
    }
    const field = $(`#a_${key}`, body)?.closest(".slider-field");
    if (field) {
      const readout = $(".slider-value", field);
      if (readout) readout.textContent = value;
      if (from !== "range") {
        const scale = scalesFor(state)[key];
        $(`#a_${key}`, field).value = nearestIndex(scale.stops, value);
      }
      if (from !== "number") $(".slider-num", field).value = value;
    }
    settle();
  };

  on(body, "input", "[data-arch-range]", (_e, t) => {
    const key = t.dataset.archRange;
    const stops = scalesFor(state)[key].stops;
    setValue(key, stops[Math.min(+t.value, stops.length - 1)], "range");
  });

  // `change`, not `input`, for the box: a half-typed "1" on the way to "16"
  // is a different model, and re-snapping the slider under a moving cursor
  // makes the field impossible to type in.
  on(body, "change", "[data-arch]", (_e, t) => {
    const value = t.value === "" ? null : Math.max(1, +t.value);
    setValue(t.dataset.arch, value, "number");
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
        ${raw(p.experts ? html`
          <dt>Used per token</dt><dd>${plan.active_params_label}</dd>
          <dt>Experts</dt><dd>${p.experts}, ${p.experts_per_token} per token</dd>` : "")}
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
    const key = `${state.runnerId}|${state.model}|${state.sourceRun?.id}|${state.goal}`;
    ensure(state.ftPlan, key, () => api.plan({
      runner_id: state.runnerId,
      params_b: state.modelDetail.data?.params_b ?? null,
      goal: state.goal,
      dataset_rows: 2000,
    }), draw);
    body.innerHTML = finetuneReview(state, runner, caps);
    if (state.ftPlan.status === "ready") wireOverrides(body, ctx);
    wireSweep(body, ctx);
    return;
  }

  ensure(state.plan, planKey(state), () => api.scratchPlan(planPayload(state)), draw);
  body.innerHTML = scratchReview(state, runner, caps);
  wireSweep(body, ctx);
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
        <dt>Starting from</dt><dd class="mono">${state.sourceRun
          ? state.sourceRun.name : state.model}</dd>
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
          ${raw(state.modelDetail.data?.moe
            ? toggle("adapt_experts", "Also adapt the experts",
                s.adapt_experts,
                "Off puts the adapter on attention only, which every token "
                + "passes through. On adapts the expert networks too, where "
                + "each expert only learns from the tokens its router happened "
                + "to send it — a much larger adapter that needs a lot more "
                + "data to be worth it. Many models store their experts in a "
                + "form no adapter can attach to; the run says so in its log "
                + "and falls back to attention. The router is never adapted.")
            : "")}
        </div>
      </details>
    </div>

    ${raw(sweepPanel(state))}

    <div class="field card">
      <label for="jobName">Name this run</label>
      <input type="text" id="jobName"
             value="${state.sourceRun
               ? `${state.sourceRun.name} (continued)`
               : String(state.model || "").split("/").pop()} on ${
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
        ${raw(a.num_local_experts > 1 ? html`
          <dt>Experts</dt><dd>${a.num_local_experts} per block,
            ${a.num_experts_per_tok} used per token — ${plan.active_params_label}
            of the ${plan.params_label} does the work on any one token, and all
            of it has to be held and trained</dd>` : "")}
        <dt>Vocabulary</dt><dd>${fmtNum(a.vocab_size)} tokens, built from your text</dd>
        <dt>Learning from</dt><dd class="mono">${state.dataset}${
          state.config ? " · " + state.config : ""} · ${state.split}</dd>
        <dt>Running on</dt><dd>${runner?.name} — ${caps.device_name || ""}</dd>
        <dt>Training time</dt><dd>about ${fmtDuration(
          (plan.estimated_minutes ?? state.minutes) * 60)}${
          plan.estimated_minutes && plan.requested_minutes
            && plan.estimated_minutes < plan.requested_minutes * 0.9
          ? ` — less than the ${fmtDuration(plan.requested_minutes * 60)} you `
            + `allowed, for the reason below`
          : ""}</dd>
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
    </div>

    ${raw(sweepPanel(state))}`;
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

const toggle = (k, label, on_, hint) => html`
  <div class="field">
    <label for="f_${k}">${label}</label>
    <select id="f_${k}" data-setting="${k}">
      <option value="1"${on_ ? " selected" : ""}>On</option>
      <option value="0"${on_ ? "" : " selected"}>Off</option>
    </select>
    <div class="hint">${hint}</div>
  </div>`;

// A settings row renders as an input unless it is a yes/no, which renders as
// the same On/Off control used elsewhere. Dispatching here rather than at each
// call site keeps the field list a plain declaration -- `type: "bool"` fed to
// an <input> silently becomes a text box that accepts anything.
const fields = (rows) => rows.map(([k, label, v, type, hint]) =>
  (type === "bool"
    ? toggle(k, label, !!v && v !== "0" && v !== "false", hint)
    : html`
  <div class="field">
    <label for="f_${k}">${label}</label>
    <input id="f_${k}" data-setting="${k}" type="${type}" value="${v}">
    ${raw(hint ? `<div class="hint">${esc(hint)}</div>` : "")}
  </div>`)).join("");


// Settings that must survive as text/float rather than being coerced to int.
const FLOAT_SETTINGS = new Set(["learning_rate", "weight_decay", "grad_clip",
                                "min_lr_ratio", "epochs"]);
const BOOL_SETTINGS = new Set(["gradient_checkpointing", "optim_8bit",
                               "adapt_experts", "early_stop"]);

function wireSweep(mount, ctx) {
  const { state, draw } = ctx;
  const sel = mount.querySelector("#sweepKey");
  if (!sel) return;
  sel.addEventListener("change", () => {
    state.sweepKey = sel.value || null;
    const opt = (SWEEPABLE[state.mode] || []).find((o) => o[0] === state.sweepKey);
    // Prefilled with the values worth trying, because the point of this
    // feature is for somebody who does not know what to try.
    state.sweepValues = opt ? opt[2].join(", ") : "";
    draw();
  });
  const box = mount.querySelector("#sweepValues");
  box?.addEventListener("input", () => { state.sweepValues = box.value; });
}

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
function refreshPlan(ctx, selector, render, onReady) {
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
    onReady?.();
  });
  if (state.plan.status === "ready" && box) {
    box.innerHTML = render(state, state.plan.data);
  }
}

const planKey = (state) => JSON.stringify([
  state.runnerId, state.size, state.minutes, state.vocab,
  state.custom, state.overrides, state.dataset, state.corpusTokens, state.moe]);

function planPayload(state) {
  return {
    runner_id: state.runnerId,
    size: state.size,
    custom: state.custom,
    moe: state.moe?.enabled ? state.moe : null,
    minutes: state.minutes,
    vocab_size: state.vocab,
    corpus_tokens: state.corpusTokens ?? null,
    sample_prompt: state.samplePrompt || "Once upon a time",
    text_field: state.textField || "text",
    overrides: state.overrides,
  };
}

// Settings worth trying several values of, and why each one. Deliberately
// short: a sweep costs a full training run per value, so the list is the
// handful where the right answer is genuinely unknown in advance rather than
// everything that happens to be a number.
const SWEEPABLE = {
  finetune: [
    ["learning_rate", "Learning rate", [5e-5, 1e-4, 2e-4, 4e-4],
     "The setting most likely to be wrong, and the one with the largest "
     + "effect. Too high and the loss spikes; too low and the run is wasted."],
    ["lora_r", "Adapter size", [8, 16, 32, 64],
     "How much capacity the adapter has. Larger learns more and overfits "
     + "sooner on a small dataset."],
    ["epochs", "Passes over the data", [1, 2, 3, 4],
     "More passes learn more from the same examples, until they start "
     + "memorising them."],
  ],
  scratch: [
    ["learning_rate", "Learning rate", [3e-4, 6e-4, 1e-3, 2e-3],
     "Scales with model width, and the recommendation is an estimate. The "
     + "loss curve tells you within a few hundred steps whether it was right."],
    ["weight_decay", "Weight decay", [0.0, 0.05, 0.1, 0.2],
     "Gentle pressure toward smaller weights. Matters most when the corpus "
     + "is small enough to memorise."],
    ["warmup_steps", "Warmup steps", [10, 50, 100, 200],
     "How long the learning rate takes to reach full size. Too short and the "
     + "first update can wreck a model of random weights."],
  ],
};

function sweepPanel(state) {
  const options = SWEEPABLE[state.mode] || [];
  return html`
    <details class="card" style="margin-bottom:14px">
      <summary><strong>Or try several values at once</strong>
        <span class="muted tiny"> — launch a few variants and compare them</span>
      </summary>
      <p class="muted tiny" style="margin:10px 0 0">
        Runs one variant per value, identical in every other respect, and ranks
        them by held-out loss when they finish. Each variant is a whole
        training run on the same card, so three values take three times as
        long — they take turns, and the queue is dealt between people so this
        does not lock anyone else out.</p>
      <div class="field" style="margin-top:10px">
        <label for="sweepKey">Vary</label>
        <select id="sweepKey">
          <option value="">Nothing — just one run</option>
          ${raw(options.map(([k, label]) => html`
            <option value="${k}"${state.sweepKey === k ? " selected" : ""}
            >${label}</option>`).join(""))}
        </select>
        <div class="hint" id="sweepWhy">${
          (options.find((o) => o[0] === state.sweepKey) || [])[3] || ""}</div>
      </div>
      <div class="field">
        <label for="sweepValues">Values to try</label>
        <input type="text" id="sweepValues" class="mono"
               value="${state.sweepValues || ""}"
               placeholder="${((options.find(
                 (o) => o[0] === state.sweepKey) || [])[2] || [])
                 .join(", ")}">
        <div class="hint">Separated by commas. Up to eight in total.</div>
      </div>
    </details>`;
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
    () => (!state.model && !state.sourceRun
           ? "Choose a model to continue." : null),
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
    const varied = parseSweep(state);
    if (varied && varied.error) { toast(varied.error, "err"); return; }
    state.starting = true;
    next.disabled = true;
    next.textContent = varied ? "Launching variants…" : "Starting…";
    try {
      if (varied) {
        const r = await api.createSweep({
          name: `${job.name || "Run"} · trying ${varied.key}`,
          base: job, vary: { [varied.key]: varied.values } });
        toast(`${r.jobs.length} runs queued.`, "ok");
        location.hash = `#/sweeps/${r.sweep_id}`;
      } else {
        const { id } = await api.createJob(job);
        toast("Training run created.", "ok");
        location.hash = `#/jobs/${id}`;
      }
    } catch (e) {
      toast(e.message, "err");
      state.starting = false;
      next.disabled = false;
      next.textContent = "Start training";
    }

  });
}

/** The sweep the user asked for, or null, or an explanation of what is wrong.
 *  Parsed here rather than server-side so a typo is caught before eight runs
 *  are queued and then deleted again. */
function parseSweep(state) {
  if (!state.sweepKey) return null;
  const raw_ = (state.sweepValues || "").split(",")
    .map((v) => v.trim()).filter(Boolean);
  if (raw_.length < 2) {
    return { error: "Give at least two values to compare, separated by commas." };
  }
  if (raw_.length > 8) {
    return { error: `That is ${raw_.length} runs. Eight is the limit.` };
  }
  const values = raw_.map(Number);
  if (values.some((v) => !isFinite(v))) {
    return { error: "Those values are not all numbers." };
  }
  const whole = new Set(["lora_r", "epochs", "warmup_steps", "batch_size",
                         "grad_accum", "max_steps", "lora_alpha"]);
  return { key: state.sweepKey,
           values: whole.has(state.sweepKey) ? values.map(Math.round) : values };
}

function buildJob(mount, state) {
  const name = $("#jobName", mount)?.value || undefined;
  const dataBits = state.studioDataset
    ? { studio_dataset: state.studioDataset.id }
    : {
      dataset: state.dataset,
      dataset_config: state.config || null,
      dataset_split: state.split || "train",
    };

  // Whatever the preview proved, recorded on the job. The Playground reads it
  // back so it speaks to the finished model in the shape the model learned,
  // and the remembered system prompt is offered there as a starting point.
  const trained = { ...(state.preview.data?.format || {}) };
  delete trained.specials;
  if (state.templateSource === "model" && state.mode === "finetune") {
    trained.use_model_template = true;
    delete trained.chat_template;
  } else if (state.templateSource === "custom" && state.customTemplate) {
    trained.mode = "jinja";
    trained.template = state.customTemplate;
  }
  // The format is recorded by name, not as expanded Jinja: the runner needs
  // the name to know which tokens to reserve in the vocabulary it trains.
  if (state.mode === "scratch" && state.previewIsChat
      && state.templateSource !== "custom") {
    trained.chat_format = state.chatFormat;
    trained.mode = "chat";
    if (state.teachReasoning) trained.reasoning = true;
    delete trained.chat_template;
  }
  const sel = selectorsFor(state);
  if (sel) trained.selectors = sel;
  const systemPrompt = (state.preview.data?.system_prompts || [])[0] || "";

  if (state.mode === "scratch") {
    // Guarded: the review used to read .settings straight off a plan that
    // could still be null, which threw instead of explaining itself.
    if (state.plan.status !== "ready") return null;
    const s = { ...state.plan.data.settings, ...state.overrides };
    delete s.fits;
    return {
      name, kind: "pretrain_llm",
      config: { ...dataBits, text_field: state.textField || "text",
                format: trained, system_prompt: systemPrompt,
                required_runner: state.runnerId, ...s },
    };
  }

  if (state.ftPlan.status !== "ready") return null;
  const s = { ...state.ftPlan.data.settings, ...state.overrides };
  return {
    name, kind: "finetune_llm",
    config: {
      ...dataBits,
      ...(state.sourceRun
        ? { base_model_job: state.sourceRun.id }
        : { base_model: state.model }),
      params_b: state.modelDetail.data?.params_b ?? null,
      goal: state.goal,
      required_runner: state.runnerId,
      format: trained,
      system_prompt: systemPrompt,
      ...s,
    },
  };
}
