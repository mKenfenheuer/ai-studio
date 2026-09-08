import { api, events } from "../api.js";
import { html, raw, esc, on, $, $$, fmtNum, fmtDuration, toast, resource,
         ensure } from "../util.js";
import { ribbon, rb, group, wireRibbon } from "../ribbon.js";
import { pageHead, emptyState } from "../components.js";

// Two fundamentally different jobs behind one wizard. They share a machine
// picker, a data step and a review; everything between differs, because the
// decisions differ. Fine-tuning asks "which model, which examples"; training
// from scratch asks "how big, and how long are you prepared to wait" -- a
// question fine-tuning never has to answer.
//
// This file follows the rendering rule, which now lives with `ensure()` in
// util.js: draw() renders purely from state, and no render starts work that
// causes another render synchronously.

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

// The format step is its own step in both paths, and not an afterthought at
// the bottom of the data step, because it is the decision that decides whether
// the finished model can be talked to at all. Trained in a shape the model has
// never seen, everything else on this screen is wasted -- so it is asked
// exactly once, explicitly, and nothing continues until it is answered.
const STEP_NAMES = {
  finetune: ["Goal", "Model", "Data", "Format", "Review"],
  scratch: ["Goal", "Text", "Format", "Design", "Review"],
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
// The draft, and the step in the address bar
// ===========================================================================
//
// All of this lived in a closure. Six steps, a mandatory format decision, and
// a browser Back that left the wizard entirely and took every choice with it;
// so did a reload, and so did going to look at the dataset you were about to
// train on. The way to answer "which split was that again" was to start over.
//
// Two mechanisms, deliberately:
//
//   The STEP is in the address bar, so Back and Forward move between steps
//   rather than out of the wizard, and a step can be linked to.
//
//   Everything else is a DRAFT in localStorage, restored on mount. The step
//   in the URL then decides where you land. Resources -- the previews, the
//   plans, the model lists -- are not saved: they are caches keyed on the
//   choices, and they refill from the choices.
//
// The draft expires, because resuming a half-made run from last Tuesday
// without being told is worse than starting fresh.

const DRAFT_KEY = "aistudio.wizard";
const DRAFT_MAX_AGE_MS = 8 * 60 * 60 * 1000;

// The decisions. Everything here is plain data that came from a person; the
// resources are left out on purpose.
const DRAFT_FIELDS = [
  "mode", "goal", "runnerId", "model", "sourceRun", "dataset", "config",
  "split", "studioDataset", "textField", "formatMode", "templateSource",
  "customTemplate", "chatFormat", "teachReasoning", "selectors", "roleMap",
  "minutes", "size", "vocab", "custom", "moe", "overrides", "archOverrides",
  "sweepKey", "sweepValues",
];

/** Whatever follows the `?` in `#/new?step=2&dataset=ds_x`.
 *
 *  Read from the hash rather than from location.search, because the router
 *  puts everything after the `#`. */
function wizardParams() {
  const at = location.hash.indexOf("?");
  return new URLSearchParams(at < 0 ? "" : location.hash.slice(at + 1));
}

function saveDraft(state) {
  try {
    const keep = { at: Date.now() };
    for (const k of DRAFT_FIELDS) keep[k] = state[k];
    localStorage.setItem(DRAFT_KEY, JSON.stringify(keep));
  } catch { /* private browsing; the wizard still works, it just forgets */ }
}

function loadDraft() {
  try {
    const raw = JSON.parse(localStorage.getItem(DRAFT_KEY) || "null");
    if (!raw || !raw.at || Date.now() - raw.at > DRAFT_MAX_AGE_MS) return null;
    return raw;
  } catch { return null; }
}

export function clearWizardDraft() {
  try { localStorage.removeItem(DRAFT_KEY); } catch { /* nothing to forget */ }
}

/** Move to a step by changing the address, so the browser records it. */
function goToStep(n) {
  const params = wizardParams();
  params.set("step", String(n));
  location.hash = `#/new?${params}`;
}

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
    // Which models fit this machine, and which one uses it best.
    fits: resource(),
    // The data step, shared by both paths.
    dataset: null, configs: resource(), config: null, split: "train",
    // A dataset from this studio's library rather than from the Hub. It is
    // fetched by the runner with the join token, so a private dataset never
    // has to be published to be trained on.
    studioDataset: null, myDatasets: resource(),
    preview: resource(), textField: null, formatMode: null,
    // How a row becomes training text. "model" uses the base model's own chat
    // template, which is what an instruct model was trained to expect and
    // therefore the right suggestion; "format" is one of the named chat
    // formats, chosen by `chatFormat`; "builtin" is the plain rendering of
    // whatever shape the data already has; "custom" is a Jinja template the
    // user writes.
    //
    // Deliberately null. There is no safe default here -- the same wizard
    // fine-tunes an instruct model, continues one of this studio's own runs
    // and builds a model from nothing, and the right answer differs for all
    // three. It is suggested, and it is not decided until it is clicked.
    templateSource: null, customTemplate: "", builtin: resource(),
    // Which set of message-boundary tokens the run is written in.
    chatFormat: "chatml", formats: resource(), teachReasoning: false,
    // The base model's own template, fetched so the step can say whether the
    // model has one before the choice is made rather than after it.
    modelTemplate: resource(),
    // What the rows turned out to be, remembered across the preview being
    // refetched: {key, mode}. See rememberShape.
    shape: null,
    // Explicit paths for datasets auto-detection reads wrongly.
    selectors: {}, roleMap: "", selectorFields: resource(),
    // From scratch.
    minutes: 60, size: null, vocab: 8192, custom: null,
    // A mixture of experts is something done to whichever size was chosen,
    // not a size of its own, so it lives beside the size rather than in it.
    moe: { enabled: false, num_local_experts: 8, num_experts_per_tok: 2 },
    sizes: resource(), plan: resource(), ftPlan: resource(),
    // What would go wrong with this exact configuration, asked of the
    // controller (and, for the token counts, of a runner) before anything is
    // queued rather than an hour into a run.
    preflight: resource(),
    overrides: {}, archOverrides: {},
    starting: false,
  };

  // What was being worked on before, if it was recent.
  const draft = loadDraft();
  let resumed = false;
  if (draft) {
    for (const k of DRAFT_FIELDS) {
      if (draft[k] !== undefined) state[k] = draft[k];
    }
    resumed = true;
  }

  const params = wizardParams();

  // A dataset named in the address wins over the draft: following "train on
  // this" from a dataset page is a new intention, not a continuation. It is a
  // link rather than a stashed value so that it survives a reload and can be
  // sent to somebody.
  try {
    const handed = sessionStorage.getItem("aistudio.dataset");
    if (handed) {
      sessionStorage.removeItem("aistudio.dataset");
      const d = JSON.parse(handed);
      state.studioDataset = d;
      state.dataset = d.name;
      state.split = defaultSplit(d.splits);
      state.formatMode = null;
      resumed = false;
    }
  } catch { /* nothing was handed over */ }

  const wantDataset = params.get("dataset");
  if (wantDataset && state.studioDataset?.id !== wantDataset) {
    try {
      const d = await api.dataset(wantDataset);
      state.studioDataset = { id: d.id, name: d.name, splits: d.splits || {},
                              rows: d.rows };
      state.dataset = d.name;
      state.split = defaultSplit(d.splits);
      state.textField = (d.format || {}).text_field || null;
      state.formatMode = null;
      resumed = false;
    } catch { /* it may have been deleted; the picker still works */ }
  }

  const wantStep = parseInt(params.get("step") || "", 10);
  if (Number.isFinite(wantStep)) state.step = wantStep;
  state.resumed = resumed && (state.step > 0 || !!state.dataset || !!state.model);

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
    // Every choice, kept, so a reload or a trip to look at the dataset does
    // not start the whole thing again.
    saveDraft(state);
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
      ${raw(pageHead({ title: "New training run" }))}
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
      ${raw(pageHead({ title: "New training run" }))}
      ${raw(emptyState({
        icon: "🔌",
        title: "No machines are connected",
        body: "A training run needs a machine with a graphics card. Connecting "
            + "one takes a single command.",
        cta: { href: "#/runners", label: "Connect a machine" },
      }))}`;
  }
  // The steps are the tabs. A step you have not reached is not a tab you can
  // press: this is a sequence, and every step reads what the one before it
  // decided. Going back is free.
  const tabs = names.map((label, i) => ({
    key: String(i),
    label: `${i < state.step ? "✓" : i + 1}. ${label}`,
    disabled: i > state.step,
    hint: i > state.step ? "Finish the steps before this one first" : "",
  }));
  const last = names.length - 1;
  return html`
    ${raw(pageHead({
      title: "New training run",
      tab: `${names[state.step]} · New run`,
      sub: `${names.length} steps. Everything technical is chosen for you, and `
         + `every choice is explained — and every one can be changed.`,
    }))}
    ${raw(ribbon({
      tabs, active: String(state.step),
      body: group("This run", [
        rb(null, "←", "Back", { disabled: state.step === 0, data: 'data-back="1"' }),
        rb(null, state.step === last ? "▶" : "→",
           state.step === last ? "Start training" : "Continue",
           { cls: "primary", data: 'data-next="1"' }),
      ]) + group("Look at", [
        rb(null, "▤", "Datasets", { href: "#/data" }),
        rb(null, "▦", "Machines", { href: "#/runners" }),
        rb(null, "≡", "Runs", { href: "#/jobs" }),
      ]),
    }))}
    ${raw(state.resumed ? html`
      <div class="callout" style="margin-bottom:14px">
        <div class="row-between" style="flex-wrap:wrap;gap:8px;align-items:center">
          <span><strong>Carrying on where you left off.</strong> Every choice
            you had made is still here.</span>
          <button class="btn-sm" id="startOver">Start fresh</button>
        </div>
      </div>` : "")}
    <div id="stepBody"></div>
    <div class="wizard-actions">
      <button data-back="1" ${state.step === 0 ? "disabled" : ""}>← Back</button>
      <div class="spacer"></div>
      <span id="navHint" class="tiny muted"></span>
      <button id="nextBtn" data-next="1" class="btn-primary btn-lg">
        ${state.step === last ? "Start training" : "Continue →"}
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
    // What differs between the two paths is thrown away; what does not, is
    // kept. Which dataset you mean to train on is the same question either
    // way, and clearing it meant somebody who picked a dataset and then
    // wondered what from-scratch would look like had to go and find it again.
    Object.assign(state, {
      model: null, modelDetail: resource(), preview: resource(),
      formatMode: null, size: null, custom: null, sizes: resource(),
      // The format is asked again as well: "the model's own" means nothing
      // once there is no model, and a from-scratch run reserves its tokens
      // from a decision this one has not made yet.
      templateSource: null, customTemplate: "", teachReasoning: false,
      modelTemplate: resource(), sourceRun: null, sourceTemplate: resource(),
      plan: resource(), ftPlan: resource(), overrides: {}, archOverrides: {},
      moe: { enabled: false, num_local_experts: 8, num_experts_per_tok: 2 },
    });
    draw();
  });
  on(body, "click", "[data-goal]", (_e, t) => { state.goal = t.dataset.goal; draw(); });
  on(body, "click", "[data-runner]", (_e, t) => {
    state.runnerId = t.dataset.runner;
    // Another card, another answer to "what fits".
    state.fits = resource();
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
  // What this particular machine can actually train, costed against its own
  // measured memory. Keyed on the runner, because the answer changes entirely
  // when the run is pointed at a different card.
  ensure(state.fits, state.runnerId || "none",
         () => api.recommendations(state.runnerId), draw);
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

    ${raw(modelChoices(state, starters))}

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
    // A different model is a different "the model's own format", so the
    // format is asked again rather than silently carried over.
    forgetTemplate(state);
    draw();
  });

  on(body, "click", "[data-own-model]", (_e, t) => {
    const own = (state.myModels.data || []).find((m) => m.id === t.dataset.ownModel);
    state.sourceRun = own || null;
    state.model = null;
    state.modelDetail = resource();
    state.sourceTemplate = resource();
    state.ftPlan = resource();
    forgetTemplate(state);
    draw();
  });

  on(body, "click", "#clearOwnModel", () => {
    state.sourceRun = null;
    state.ftPlan = resource();
    forgetTemplate(state);
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

/** The models worth training here, costed against this machine.
 *
 *  The old list said "fits" or "likely too big" and left it there, which
 *  answers the question nobody asked. What a person wants to know standing in
 *  front of a 16 GB card is *which model uses it* -- because a 4-bit 7B on
 *  that card leaves ten gigabytes idle, and nothing said so.
 */
function modelChoices(state, starters) {
  const r = state.fits;
  if (r.status === "loading" || r.status === "idle") {
    return loading("Working out what this machine can train…");
  }
  // The recommendation is an improvement, not a requirement: if it could not
  // be worked out, the plain list still lets somebody choose a model.
  const rec = r.status === "ready" ? r.data : null;
  const models = rec?.models
    || (starters.models || []).map((m) => ({ ...m, fit: {} }));
  const best = rec?.best;

  const usable = models.filter((m) => m.fit.verdict !== "too_big");
  const over = models.filter((m) => m.fit.verdict === "too_big");

  const card = (m) => {
    const q = m.fit.verdict === "fits_quantized";
    const blocked = m.fit.verdict === "needs_quantization";
    return html`
      <button class="pick ${state.model === m.id ? "selected" : ""}
                     ${!state.model && m.id === best ? "suggested" : ""}"
              data-model="${m.id}">
        <span class="t">${m.label}
          <span class="badge">${m.params_b < 1
            ? Math.round(m.params_b * 1000) + "M" : m.params_b + "B"}</span>
          ${raw(m.id === best
            ? `<span class="badge badge-ok">best use of this card</span>` : "")}
          ${raw(m.gated ? `<span class="badge badge-warn">gated</span>` : "")}
        </span>
        <span class="d">${m.blurb}</span>
        ${raw(m.needed_gb ? html`
          <span class="row tiny muted" style="gap:6px;flex-wrap:wrap">
            <span class="badge ${blocked ? "badge-warn" : q ? "badge-accent" : "badge-ok"}">${
              blocked ? "needs 4-bit, unavailable here"
                      : `${m.precision} · ${m.needed_gb} GB`}</span>
            ${raw(m.spare_gb > 0 && !blocked
              ? `<span>${m.spare_gb} GB spare</span>` : "")}
          </span>` : "")}
      </button>`;
  };

  return html`
    ${raw(rec?.note ? html`
      <div class="callout" style="margin-bottom:12px">
        <strong>What this machine can do</strong>${rec.note}
      </div>` : "")}

    <div class="grid grid-2">${raw(usable.map(card).join(""))}</div>

    ${raw(over.length ? html`
      <details class="adv">
        <summary>Too big for this machine (${over.length})</summary>
        <div class="grid grid-2" style="margin-top:10px">${raw(over.map(card).join(""))}</div>
      </details>` : "")}`;
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

  if (state.dataset && state.studioDataset) {
    // A dataset from this studio's own library knows its own splits, and
    // there is nothing to look up on the Hub -- asking the Hub about a name
    // it has never heard of is how this step used to show "no preview
    // available" for every dataset somebody brought in themselves.
    const names = Object.keys(state.studioDataset.splits || {});
    if (state.configs.key !== state.dataset) {
      state.configs = { key: state.dataset, status: "ready", error: null,
                        data: { configs: [], studio: true,
                                splits: names.length ? names : ["train"] } };
      state.config = "";
      if (!names.includes(state.split)) state.split = names[0] || "train";
    }
    ensure(state.preview, previewKey(state, scratch),
           () => api.trainingPreview(previewRequest(state, scratch)), draw);
    rememberShape(state);
  } else if (state.dataset) {
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
      rememberShape(state);
    }
  }
  ensure(state.selectorFields, "sel", () => api.selectorFields(), draw);
  // Your own datasets, which is where most real training data lives once
  // anybody has used this app for a week. They were reachable only from the
  // dataset page's "Train on this" button, which meant that starting from the
  // wizard -- the obvious way to start -- could not see them at all.
  ensure(state.myDatasets, "all", () => api.datasets(), draw);

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

    ${raw(libraryPicker(state, scratch))}

    <h3 style="margin:18px 0 8px">${state.myDatasets.status === "ready"
      && (state.myDatasets.data || []).length
      ? "Or start from one of these" : "Start from one of these"}</h3>
    <div class="grid grid-2">
      ${raw(catalogue.map((d) => html`
        <button class="pick ${state.dataset === d.id && !state.studioDataset
                              ? "selected" : ""}" data-ds="${d.id}"
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

  on(body, "click", "[data-studio-ds]", (_e, t) => {
    const d = (state.myDatasets.data || []).find((x) => x.id === t.dataset.studioDs);
    if (!d) return;
    state.studioDataset = { id: d.id, name: d.name, splits: d.splits || {}, rows: d.rows };
    state.dataset = d.name;
    state.configs = resource();
    state.preview = resource();
    state.config = "";
    state.split = defaultSplit(d.splits);
    state.textField = (d.format || {}).text_field || null;
    state.formatMode = null;
    // Roughly four characters to a token. Approximate, and it only feeds the
    // warning about a corpus running out before the token budget does -- a
    // place where the right answer is "about this many" rather than silence.
    state.corpusTokens = d.bytes ? Math.round(d.bytes / 4) : null;
    state.samplePrompt = null;
    forgetTemplate(state);
    state.plan = resource();
    state.sizes = resource();
    draw();
  });

  on(body, "click", "[data-ds]", (_e, t) => {
    state.dataset = t.dataset.ds;
    // A Hub dataset is not one of ours: leaving this set would send the job a
    // studio dataset id alongside a Hugging Face name.
    state.studioDataset = null;
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
    // Different data, different shape: a format chosen for conversations is
    // the wrong answer for a column of prose, so the next step asks again.
    forgetTemplate(state);
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
  wireExpand(body, state);
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

/** The datasets in this studio's own library.
 *
 *  First, not last: once anybody has imported, uploaded or generated
 *  anything, their own data is the likeliest answer to "what shall I train
 *  on" -- and it is the only data on this screen that nobody else has
 *  already trained a model on.
 */
function libraryPicker(state, scratch) {
  const r = state.myDatasets;
  if (r.status === "loading" || r.status === "idle") {
    return loading("Looking at your dataset library…");
  }
  if (r.status === "error") return "";
  const mine = r.data || [];
  if (!mine.length) {
    return html`
      <div class="callout" style="margin-bottom:14px">
        <strong>Your library is empty</strong>
        Anything you <a href="#/data">import, upload or generate</a> appears
        here and can be trained on directly — including the splits you gave
        it.
      </div>`;
  }

  return html`
    <h3 style="margin:0 0 8px">From your library</h3>
    <div class="grid grid-2">
      ${raw(mine.map((d) => {
        const splits = Object.entries(d.splits || {});
        return html`
          <button class="pick ${state.studioDataset?.id === d.id ? "selected" : ""}"
                  data-studio-ds="${d.id}">
            <span class="t">${d.name}
              <span class="badge">${fmtNum(d.rows)} rows</span>
              ${raw(d.mine ? "" : `<span class="badge badge-accent">shared</span>`)}
            </span>
            <span class="d">
              ${raw(splits.length > 1
                ? splits.map(([n, c]) =>
                    `<span class="badge">${esc(n)} ${fmtNum(c)}</span>`).join(" ")
                : "")}
              ${raw(d.origin ? `<span class="mono tiny">${esc(d.origin)}</span>` : "")}
            </span>
          </button>`;
      }).join(""))}
    </div>`;
}

function dataDetail(state, scratch) {
  if (!state.dataset) return "";
  const c = state.configs;
  if (c.status === "loading") return loading("Looking at this dataset…");
  if (c.status === "error") return failed(c.error);
  if (c.status !== "ready") return "";

  const configs = c.data.configs || [];
  const chosen = configs.find((x) => x.name === state.config) || configs[0];
  const splits = c.data.studio
    ? (c.data.splits || ["train"]) : (chosen?.splits || ["train"]);

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
      ${raw(selectorPanel(state))}
    </div>

    <div style="margin-top:14px">${raw(trainingText(state, scratch))}</div>`;
}

// ===========================================================================
// Step — the shape the training text is written in
//
// One decision, asked once, for every kind of run. A model is only as usable
// as the format it was taught: give an instruct model a layout it has never
// seen and it ignores half of what you taught it, and give a from-scratch
// model no boundary tokens and generation has nothing dependable to stop on.
// So there is no silent default here. The right answer is suggested -- loudly
// -- and the run does not continue until it has been chosen.
// ===========================================================================

/** Un-decide the format. Called wherever the thing it was decided *about*
 *  changes: the base model, the dataset, or the kind of run itself. */
function forgetTemplate(state) {
  state.templateSource = null;
  state.customTemplate = "";
  state.teachReasoning = false;
  state.preview = resource();
}

/** The dataset itself, without any decision made about it. What shape the
 *  rows are depends on this and nothing else. */
function dataKey(state) {
  return JSON.stringify([state.dataset, state.config, state.split]);
}

/** Remember what shape a dataset turned out to be.
 *
 *  Kept on the state rather than read out of the preview each time, because
 *  the preview is emptied while the next one is in flight -- and the shape of
 *  the rows does not change just because we are asking about a different
 *  template. Without this the suggestion flickers, and the request that goes
 *  out during the gap is built from a shape nobody detected. */
function rememberShape(state) {
  const p = state.preview;
  if (p.status !== "ready") return;
  const mode = p.data.detected_format?.mode || p.data.format?.mode;
  if (mode && mode !== "auto") state.shape = { key: dataKey(state), mode };
}

/** What shape the rows are: "chat", "instruction", "text" -- or "unknown"
 *  while nothing has been read yet, which is not the same as "text" and must
 *  not be guessed as it.
 *
 *  Taken from the *detected* format rather than from the rendered one, because
 *  the rendered one is a consequence of the request we sent -- asking it what
 *  shape the data is would be asking our own last answer back. */
function dataShape(state) {
  if (state.formatMode) {
    return state.formatMode === "jinja" ? "chat" : state.formatMode;
  }
  const known = state.shape;
  return known && known.key === dataKey(state) ? known.mode : "unknown";
}

/** The base model's own template, from whichever place it lives in.
 *  A Hub model ships it in tokenizer_config.json; one of this studio's own
 *  runs carries it inside the saved tokenizer, read back out of the zip. */
function ownTemplate(state) {
  const own = !!state.sourceRun;
  const r = own ? state.sourceTemplate : state.modelTemplate;
  return {
    name: own ? state.sourceRun.name : state.model,
    waiting: r.status === "loading" || r.status === "idle",
    failed: r.status === "error",
    available: r.status === "ready" && !!r.data.available,
    template: (r.status === "ready" && r.data.chat_template) || "",
    reason: (r.status === "ready" && r.data.reason)
      || "This model ships no chat template of its own, which usually means "
       + "it is a base model rather than an instruct one.",
  };
}

/** The choice, as one string, so a grid of buttons can compare against it. */
function selectedKey(state) {
  return state.templateSource === "format"
    ? "format:" + state.chatFormat : state.templateSource;
}

/** The chosen format, in a sentence, for the review step. */
function templateLabel(state) {
  const formats = state.formats.data?.formats || [];
  switch (state.templateSource) {
    case "model":
      return state.sourceRun
        ? `${state.sourceRun.name}'s own chat template`
        : `${state.model}'s own chat template`;
    case "format": {
      const f = formats.find((x) => x.id === state.chatFormat);
      return (f ? f.label : state.chatFormat) + " — "
        + (state.mode === "scratch"
           ? "its boundary tokens reserved in the vocabulary"
           : "rendered as that format publishes it")
        + (state.teachReasoning ? ", reasoning before the answer" : "");
    }
    case "custom":
      return "a Jinja template you wrote";
    case "builtin":
      return dataShape(state) === "text"
        ? "raw text, exactly as the column holds it"
        : "the plain readable rendering";
    default:
      return "not chosen";
  }
}

/** What this particular run should almost certainly be trained in.
 *  A suggestion only: it is drawn as one, and it is not the answer until it
 *  has been clicked. */
function suggestedKey(state, scratch) {
  const fallback = "format:" + (state.formats.data?.default || "chatml");
  if (!scratch) {
    // The model's own, unless it has none -- a base model, most often, which
    // has never been taught any layout and can therefore be taught one here.
    const own = ownTemplate(state);
    return own.waiting || own.available ? "model" : fallback;
  }
  // From scratch there is no "own" template: the model learns whichever shape
  // it is shown. Conversations get boundary tokens; prose is left as prose --
  // and a corpus nothing could be read from is prose until proven otherwise,
  // which is what a from-scratch corpus almost always is.
  const shape = dataShape(state);
  return shape === "text" || shape === "unknown" ? "builtin" : fallback;
}

function chooseTemplate(state, key) {
  if (key.startsWith("format:")) {
    state.templateSource = "format";
    state.chatFormat = key.slice("format:".length);
  } else {
    state.templateSource = key;
    // Reasoning is a property of a named format's own layout -- its think
    // block or its analysis channel. Nothing else here has one to teach.
    state.teachReasoning = false;
  }
  if (state.templateSource === "custom" && !state.customTemplate) {
    // Start from something that already works rather than a blank box.
    const b = state.builtin.data || {};
    state.customTemplate = dataShape(state) === "instruction"
      ? (b.instruction_template || "") : (b.template || "");
  }
  state.preview = resource();
}

function stepTemplate(body, ctx) {
  const { state, draw } = ctx;
  const scratch = state.mode === "scratch";

  ensure(state.formats, "formats", () => api.chatFormats(), draw);
  ensure(state.builtin, "builtin", () => api.builtinTemplate(), draw);
  if (!scratch && state.model) {
    ensure(state.modelTemplate, state.model,
           () => api.modelTemplate(state.model), draw);
  }
  if (!scratch && state.sourceRun) {
    ensure(state.sourceTemplate, state.sourceRun.id,
           () => api.jobChatTemplate(state.sourceRun.id), draw);
  }
  // The same preview the data step asked for, under the same key: the choice
  // made here changes the key, and the rendered examples below re-render with
  // the format actually chosen rather than with a generic one.
  if (state.dataset && state.configs.status === "ready") {
    ensure(state.preview, previewKey(state, scratch),
           () => api.trainingPreview(previewRequest(state, scratch)), draw);
    rememberShape(state);
  }

  body.innerHTML = html`
    ${raw(formatIntro(state, scratch))}
    ${raw(formatGrid(state, scratch))}
    ${raw(state.templateSource === "format" ? reasoningPanel(state) : "")}
    ${raw(customEditor(state))}
    ${raw(formatNotes(state))}
    <div style="margin-top:14px">${raw(trainingText(state, scratch))}</div>`;

  on(body, "click", "[data-tmpl]", (_e, t) => {
    chooseTemplate(state, t.dataset.tmpl);
    draw();
  });
  on(body, "click", "#teachReasoning", () => {
    state.teachReasoning = !state.teachReasoning;
    state.preview = resource();
    draw();
  });
  on(body, "click", "#applyTemplate", () => {
    const box = $("#templateBox", body);
    if (box) state.customTemplate = box.value;
    state.preview = resource();
    draw();
  });
  wireExpand(body, state);
}

function formatIntro(state, scratch) {
  const chosen = state.templateSource;
  const shape = dataShape(state);
  const own = scratch ? null : ownTemplate(state);
  const source = state.preview.status === "ready"
    ? state.preview.data.template_source : null;

  return html`
    <div class="card" style="margin-bottom:14px">
      <div class="row-between" style="gap:8px;flex-wrap:wrap">
        <div>
          <h3 style="margin:0">How a row becomes training text</h3>
          <p class="muted tiny" style="margin:2px 0 0;max-width:70ch">
            ${raw(scratch
              ? "Your model will only ever speak the shape it is trained in — "
              + "there is no other source for it. Choose that shape now: it "
              + "decides which boundary tokens the new vocabulary reserves, "
              + "and it is what the Playground will speak to the finished "
              + "model afterwards."
              : "Every instruct model was trained to expect one exact layout, "
              + "and it is the one thing that has to be right. Trained in a "
              + "shape it has never seen, a model ignores much of what you "
              + "taught it — and nothing in the loss curve tells you so.")}</p>
        </div>
        ${raw(chosen && source
          ? `<span class="badge badge-accent">using: ${esc(source)}</span>` : "")}
      </div>

      ${raw(chosen ? "" : html`
        <div class="callout callout-warn" style="margin:12px 0 0">
          <strong>Choose one to continue</strong>
          This is not decided for you, because the right answer depends on
          where the model came from and what your rows look like. The one
          marked <em>suggested</em> is right for almost every run of this kind.
        </div>`)}

      ${raw(!scratch && own && own.waiting ? html`
        <p class="muted tiny" style="margin:12px 0 0">Reading
          ${own.name || "the model"}'s own template…</p>` : "")}
      ${raw(!scratch && own && !own.waiting && !own.available ? html`
        <div class="callout" style="margin:12px 0 0">
          <strong>${own.name || "This model"} has no template of its own</strong>
          ${own.reason} Pick the layout you want it taught instead — anything
          here works, as long as you talk to it the same way afterwards.
        </div>` : "")}

      ${raw(shape === "unknown" ? "" : html`
        <p class="muted tiny" style="margin:12px 0 0">
          Your rows are read as <span class="badge">${shape}</span>
          ${raw(shape === "instruction"
            ? " — instruction and response columns, which whichever format you "
            + "choose below turns into a one-turn conversation." : "")}
          ${raw(shape === "text"
            ? " — a column of prose, with no turns in it to lay out." : "")}
          ${raw(shape === "chat"
            ? " — turns, already laid out, waiting for a format to write them "
            + "in." : "")}
        </p>`)}
    </div>`;
}

function formatGrid(state, scratch) {
  const r = state.formats;
  if (r.status === "loading" || r.status === "idle") {
    return loading("Loading the formats…");
  }
  if (r.status === "error") return failed(r.error);
  // Nothing is offered until the rows have been read once. Which layouts make
  // sense, and which of them is suggested, both depend on what the data
  // actually is -- and a suggestion that changes under the cursor is worse
  // than one that arrives a second later.
  if (["loading", "idle"].includes(state.preview.status)) {
    return loading("Reading a few real rows to see what shape they are…");
  }
  const formats = r.data.formats || [];
  const shape = dataShape(state);
  const picked = selectedKey(state);
  const suggested = suggestedKey(state, scratch);

  // Dashed while it is only a suggestion, solid once it has been chosen: a
  // step that must be answered must not look as though it already was.
  const mark = (key) => (picked === key ? "selected"
    : !picked && suggested === key ? "suggested" : "");
  const tile = (key, title, desc, extra = "") => html`
    <button class="pick ${mark(key)}" data-tmpl="${key}">
      <span class="t">${title}
        ${raw(suggested === key
          ? `<span class="badge badge-ok">recommended</span>` : "")}
      </span>
      <span class="d">${desc}</span>
      ${raw(extra)}
    </button>`;

  const tiles = [];

  if (!scratch) {
    const own = ownTemplate(state);
    tiles.push(tile("model", "The model's own format",
      own.available
        ? `The layout ${own.name} was trained to expect, taken from its own `
        + `tokenizer. Almost always the right answer when you are improving `
        + `a model that already talks.`
        : `Taken from the model's own tokenizer — ${own.waiting
            ? "still reading it" : "this one does not ship one"}.`,
      own.available && own.template ? html`
        <span class="mono tiny" style="display:block;background:var(--surface-2);
              padding:7px 9px;border-radius:6px;white-space:pre-wrap;
              word-break:break-all;color:var(--text-2);max-height:96px;
              overflow:auto">${own.template.slice(0, 400)}${
                own.template.length > 400 ? " …" : ""}</span>` : ""));
  }

  formats.forEach((f) => {
    tiles.push(tile("format:" + f.id, f.label, f.blurb, html`
      <span class="mono tiny" style="display:block;background:var(--surface-2);
            padding:7px 9px;border-radius:6px;white-space:pre-wrap;
            word-break:break-all;color:var(--text-2)">${f.sample}</span>
      <span class="row" style="gap:5px;flex-wrap:wrap">
        <span class="badge">${f.token_count} token${f.token_count > 1 ? "s" : ""}</span>
        ${raw(f.specials.map((t) => `<span class="badge">${esc(t)}</span>`).join(""))}
      </span>`));
  });

  tiles.push(tile("builtin",
    shape === "text" ? "Raw text, exactly as it is" : "Plain and readable",
    shape === "text"
      ? "No chat layout at all: the column you chose is fed to the model "
      + "unchanged. The right answer for a corpus of prose, and the only one "
      + "that does not teach the model a shape its text never contains."
      : "Roles written out as ordinary words, with nothing reserved. Fine for "
      + "a base model, and the easiest to read while you are checking the "
      + "data rather than the format."));

  tiles.push(tile("custom", "Write it yourself",
    "A Jinja template, handed the conversation and any tools. Full control, "
    + "for a layout none of the others produces — and no reserved tokens, so "
    + "anything you invent is split into ordinary pieces by the tokenizer."));

  return html`
    ${raw(formatCaveat(state, scratch, shape))}
    <div class="grid grid-2">${raw(tiles.join(""))}</div>`;
}

/** The one thing that is different about naming a format on each path. */
function formatCaveat(state, scratch, shape) {
  if (scratch && shape === "text") {
    return html`
      <div class="callout" style="margin:0 0 12px">
        <strong>This corpus has no conversations in it</strong>
        Choosing a chat format still reserves its boundary tokens in the
        vocabulary and writes its layout onto the finished tokenizer — useful
        if you mean to fine-tune this model on conversations afterwards, and
        misleading if you do not, because the model will never have been shown
        those tokens. Raw text is the honest choice for prose.
      </div>`;
  }
  if (scratch) {
    return html`
      <div class="callout" style="margin:0 0 12px">
        <strong>Why the tokens matter</strong>
        Written as plain text, "assistant" is just a word — the model has to
        guess where a turn ends from punctuation, and generation has nothing
        dependable to stop on. These formats reserve <em>single, atomic
        tokens</em> for the boundaries before the vocabulary is trained, which
        only a from-scratch run can do: adding tokens to an existing model's
        tokenizer would leave its embedding table the wrong size.
      </div>`;
  }
  // Nothing to change *from* when the model arrived without a format of its
  // own, and telling somebody they are departing from a layout that does not
  // exist is worse than saying nothing.
  if (!ownTemplate(state).available) return "";
  return html`
    <div class="callout" style="margin:0 0 12px">
      <strong>Anything other than the model's own is a change of language</strong>
      A named format below is rendered exactly as it is published, and it
      trains perfectly well — but its boundary tokens are only single tokens if
      this model's tokenizer already knows them, and the model has to unlearn
      the layout it arrived with. Worth it when you are standardising on one
      format across models; not worth it otherwise.
    </div>`;
}

function customEditor(state) {
  if (state.templateSource !== "custom") return "";
  return html`
    <div class="card" style="margin-top:14px">
      <div class="field" style="margin:0">
        <label for="templateBox">Jinja template</label>
        <textarea id="templateBox" rows="10" spellcheck="false"
                  class="mono">${state.customTemplate}</textarea>
        <div class="hint">
          Available: <code>messages</code> (each with <code>role</code>,
          <code>content</code>, <code>tool_calls</code>, <code>train</code>),
          <code>tools</code>, and every column of the row by name. Example:
          <code>{% for m in messages %}{{ m.role }}: {{ m.content }}
          {% endfor %}</code>
        </div>
        <div class="row" style="margin-top:8px">
          <button class="btn-primary btn-sm" id="applyTemplate">Apply and preview</button>
        </div>
      </div>
    </div>`;
}

function formatNotes(state) {
  const p = state.preview;
  if (p.status !== "ready") return "";
  return html`
    ${raw(p.data.template_note ? html`
      <div class="callout callout-warn" style="margin-top:12px">
        <strong>No template on this model</strong>${p.data.template_note}
        Falling back to the plain readable form.
      </div>` : "")}
    ${raw(p.data.template_error ? html`
      <div class="callout callout-err" style="margin-top:12px">
        <strong>That template did not work</strong>${p.data.template_error}
      </div>` : "")}
    ${raw((p.data.system_prompts || []).length ? html`
      <details class="adv" style="margin-top:10px">
        <summary>System prompt found in this data</summary>
        <p class="muted tiny" style="margin:8px 0 4px">Kept with the run, and
          offered again in the Playground — a model trained with a system
          prompt behaves differently without it.</p>
        <p class="txt mono tiny" style="white-space:pre-wrap;max-height:180px;
           overflow:auto;background:var(--surface-2);padding:10px;
           border-radius:8px">${p.data.system_prompts[0].slice(0, 1500)}</p>
      </details>` : "")}`;
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

/** Reasoning is a property of the chosen format's own layout -- ChatML's
 *  think block, Harmony's analysis channel -- so it is offered here, beside
 *  the format, and only when the data has any reasoning to learn from. */
function reasoningPanel(state) {
  const formats = state.formats.data?.formats || [];
  const det = state.preview.status === "ready"
    ? (state.preview.data.format || {}) : {};
  const chosen = formats.find((f) => f.id === state.chatFormat) || {};
  const on_ = state.teachReasoning;
  return html`
    <div class="card" style="margin-top:14px;box-shadow:none;background:var(--surface-2)">
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
          ${raw(on_ && chosen.reasoning_note
            ? `<p class="muted tiny" style="margin:6px 0 0"><em>${
                esc(chosen.label)}: ${esc(chosen.reasoning_note)}</em></p>` : "")}
        </div>
        <button class="btn-sm ${on_ ? "btn-primary" : ""}" id="teachReasoning"
                ${det.has_reasoning ? "" : "disabled"}>
          ${on_ ? "On" : "Off"}
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
  return JSON.stringify([state.dataset, state.studioDataset?.id,
                         state.config, state.split,
                         state.formatMode, state.textField, state.model,
                         // Included, or switching to a studio model would keep
                         // showing the preview built for the previous base.
                         state.sourceRun?.id,
                         state.sourceTemplate?.status,
                         state.templateSource, state.customTemplate,
                         state.chatFormat, state.teachReasoning,
                         // The request differs by the shape of the rows, and
                         // the shape is only known once one preview has come
                         // back. Keyed on it, that first answer corrects the
                         // request instead of standing as the last word.
                         dataShape(state),
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

/** The format decision, in the shape the controller and the runner read.
 *
 *  One function for the preview request and for the job that is finally
 *  created, because the whole promise of the preview is that what you were
 *  shown is what trains. Two copies of this logic were two chances to differ.
 *
 *  `forJob` is the one honest difference: for one of this studio's own models
 *  the preview needs the template as *text* (the controller cannot look a job
 *  id up on the Hub), while the job records `use_model_template` and lets the
 *  runner read it back off the very tokenizer it loads.
 */
function formatOverlay(state, forJob = false) {
  const fmt = {};
  const src = state.templateSource;
  const shape = dataShape(state);

  if (src === "model" && state.sourceRun && !forJob) {
    const t = state.sourceTemplate;
    if (t.status === "ready" && t.data.chat_template) {
      fmt.mode = "jinja";
      fmt.template = t.data.chat_template;
    }
  } else if (src === "model") {
    fmt.use_model_template = true;
  } else if (src === "format") {
    // Recorded by name, never as expanded Jinja: a from-scratch run needs the
    // name to know which tokens to reserve in the vocabulary it trains, and
    // the playground needs it to know where a reply stops.
    fmt.chat_format = state.chatFormat;
    // Prose has no turns to lay out. The format still travels with the run --
    // it decides the reserved tokens -- but the rows are read as text.
    if (shape === "chat" || shape === "instruction") fmt.mode = "chat";
    if (state.teachReasoning) fmt.reasoning = true;
  } else if (src === "custom" && state.customTemplate) {
    // "jinja" rather than a chat template, so the same box works whether or
    // not the dataset is a conversation.
    fmt.mode = "jinja";
    fmt.template = state.customTemplate;
  }
  return fmt;
}

function previewRequest(state, scratch) {
  const fmt = {};
  if (state.formatMode) fmt.mode = state.formatMode;
  Object.assign(fmt, formatOverlay(state));
  const sel = selectorsFor(state);
  if (sel) fmt.selectors = sel;
  return {
    dataset: state.dataset,
    config: state.config,
    split: state.split,
    format: Object.keys(fmt).length ? fmt : null,
    // A column of prose still names its column; a conversation does not. A
    // from-scratch corpus nothing has been read from yet is treated as prose,
    // which is what a corpus almost always is.
    text_field: (dataShape(state) === "text"
                 || (scratch && dataShape(state) === "unknown"))
      ? state.textField : null,
    // A studio model is not on the Hub, so there is nothing to look its
    // template up by. Its template is passed as text instead, fetched from
    // the run itself -- see formatOverlay.
    base_model: scratch ? null : state.model,
    // And a studio *dataset* is not on the Hub either: the controller reads
    // its rows off the disk, from the split chosen above.
    studio_dataset: state.studioDataset?.id || null,
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
        <h3 style="margin:0">${state.templateSource
          ? "Exactly what the model will read"
          : "What these rows contain"}</h3>
        <span class="badge badge-accent">${p.data.split}</span>
      </div>
      <p class="muted tiny">Not the raw columns — the finished text, built by
        the same code that will build the training batches.
        ${raw(state.templateSource
          ? (scratch ? "The model reads these one after another, with no gaps."
                     : "Everything below, including the headings, is learned.")
          : "Rendered plainly for now: the next step decides the layout these "
          + "are actually written in.")}</p>
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

/**
 * What would go wrong, before it does.
 *
 * Everything here was found out an hour into a run by reading a log on a
 * machine: rows cut off at the context length, a split with no rows in it, a
 * dataset too small to hold anything back, a corpus read six times over. The
 * token counts come from a runner, because counting them needs the tokenizer
 * the run will use and the controller deliberately has no such thing.
 */
/** Ask the controller what is wrong with this configuration.
 *
 *  Keyed on the decisions that change the answer, so adjusting a learning rate
 *  does not re-count two hundred rows on a runner, and changing the context
 *  length does. */
/** The reason the pre-flight check gives for not starting, if it has one. */
function preflightBlocker(state) {
  const r = state.preflight;
  if (r.status !== "ready" || !r.data?.blocked) return null;
  const first = (r.data.issues || []).find((i) => i.level === "error");
  return first ? first.message.slice(0, 120) : "Fix the problems above first.";
}

function askPreflight(state, ctx) {
  const job = buildJob(null, state);
  if (!job) return;
  const c = job.config;
  const key = JSON.stringify([
    c.studio_dataset || c.dataset, c.dataset_split, c.base_model,
    c.max_seq_len, c.token_budget, c.train_on,
    (c.format || {}).mode, (c.format || {}).chat_format,
    (c.format || {}).use_model_template, (c.format || {}).template,
  ]);
  ensure(state.preflight, key,
         () => api.preflight({ config: { ...c, kind: job.kind } }),
         ctx.draw);
}

function preflightPanel(state) {
  const res = state.preflight;
  if (res.status === "idle") return "";
  if (res.status === "loading") {
    return html`<div class="card muted tiny" aria-busy="true">
      Checking the data against these settings…
      <div class="sk-line shimmer" style="margin-top:10px;width:65%"></div></div>`;
  }
  if (res.status === "error") {
    // A check that cannot run is not a reason to stop: it is a reason to say
    // it could not run.
    return html`<div class="callout callout-warn"><strong>Could not check the
      data first</strong>${res.error} The run can still be started.</div>`;
  }
  const r = res.data || {};
  const f = r.facts || {};
  const counted = f.tokens?.available;
  if (!(r.issues || []).length && !counted) return "";
  return html`
    <div style="margin-bottom:14px">
      ${raw(issueList((r.issues || []).filter((i) => i.level !== "ok")))}
      ${raw(counted ? html`
        <div class="card" style="box-shadow:none;background:var(--surface-2)">
          <div class="row-between" style="flex-wrap:wrap;gap:8px">
            <strong class="tiny">Counted on ${f.tokens.runner}, with this
              model's own tokenizer</strong>
            <span class="muted tiny">${fmtNum(f.rows || 0)} rows in the split</span>
          </div>
          <p class="muted tiny" style="margin:6px 0 0">
            Typical row ${fmtNum(f.token_p50 || 0)} tokens · nine in ten under
            ${fmtNum(f.token_p90 || 0)} · longest ${fmtNum(f.token_max || 0)}${
            f.over_limit ? ` · ${fmtNum(f.over_limit)} over the limit` : ""}${
            f.held_out_split ? ` · measuring on the ${f.held_out_split} split` : ""}
          </p>
        </div>` : "")}
    </div>`;
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
      // The name too. Continuing an earlier run never looks the model up, so
      // its size is not in hand -- and without a size the plan cannot tell
      // whether 16-bit fits and quietly assumes it does.
      base_model: state.model || state.sourceRun?.base_model || null,
      goal: state.goal,
      // The plan derives epochs and the step budget from this. It was a
      // constant 2,000 for every dataset, so a 200-row set got the schedule
      // of a 2,000-row one and a 200,000-row set got it too.
      dataset_rows: datasetRows(state),
    }), draw);
    body.innerHTML = finetuneReview(state, runner, caps);
    if (state.ftPlan.status === "ready") {
      wireOverrides(body, ctx);
      askPreflight(state, ctx);
    }
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

/** How the frozen base model is held in memory.
 *
 *  The single setting that decides whether a model fits on a card at all: a 7B
 *  is about 19 GB in 16-bit and 5 GB in 4-bit. The plan picks it, and until
 *  now that pick was invisible and unchangeable -- so a run refused at
 *  dispatch with "switch this run to 4-bit" offered no way to do that.
 *
 *  Only the base is quantised. The adapter being trained stays in full
 *  precision either way, which is why the quality cost is small and why this
 *  is not the same choice as `dtype` above.
 */
function quantField(s, plan, caps) {
  const has4bit = !!((caps.quantization || {})["4bit"]);
  const mem = plan.memory || {};
  const fits16 = plan.fit?.verdict === "fits";
  const only4 = plan.fit?.verdict === "fits_quantized";
  return html`
    <div class="field">
      <label for="f_quantization">Base model precision</label>
      <select id="f_quantization" data-setting="quantization">
        <option value="none"${s.quantization !== "4bit" ? " selected" : ""}
          ${only4 ? " disabled" : ""}>16-bit — full quality${
            mem.fp16_gb ? ` · about ${mem.fp16_gb} GB` : ""}${
            only4 ? " · too large for this machine" : ""}</option>
        <option value="4bit"${s.quantization === "4bit" ? " selected" : ""}
          ${has4bit ? "" : " disabled"}>4-bit — fits far more${
            mem.int4_gb ? ` · about ${mem.int4_gb} GB` : ""}${
            has4bit ? "" : " · not available on this machine"}</option>
      </select>
      <div class="hint">${raw(only4
        ? "This model only fits on this card in 4-bit, so that is what the "
          + "plan chose. Quality drops slightly; not running at all drops it "
          + "further."
        : fits16
          ? "This model fits either way here. 16-bit is the better of the two "
            + "unless you want the memory back for a longer sequence or a "
            + "bigger batch."
          : "Only the frozen base is compressed — the adapter you are training "
            + "stays at full precision either way.")}</div>
    </div>`;
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
        <dt>Written as</dt><dd>${templateLabel(state)}</dd>
        <dt>Running on</dt><dd>${runner?.name} — ${caps.device_name || ""}</dd>
        ${raw(plan.estimated_minutes
          ? `<dt>Rough duration</dt><dd>about ${esc(fmtDuration(plan.estimated_minutes * 60))}</dd>`
          : "")}
      </dl>
    </div>

    <div id="preflight">${raw(preflightPanel(state))}</div>

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
          ${raw(quantField(s, plan, caps))}
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
          ${raw(toggle("merge_after", "Also produce a standalone model",
            s.merge_after !== false,
            "A fine-tune produces an adapter, which needs the exact base model "
            + "it was trained against in order to run anywhere. Merging folds it "
            + "into the weights, so what you are left with needs nothing else — "
            + "which is what every tool outside this studio wants. It is the "
            + "last step of this run, on the machine that still has the weights "
            + "in memory: no second run appears and nothing is downloaded again. "
            + "The cost is disk: a merged 7B is about 14 GB where its adapter "
            + "was 50 MB. The adapter is kept as well — it is what a later run "
            + "continues from, and either can be published."))}
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
        <dt>Speaking</dt><dd>${templateLabel(state)}</dd>
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
    <div id="preflight">${raw(preflightPanel(state))}</div>

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
                               "adapt_experts", "early_stop", "merge_after"]);

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

/** Whether the run can move on, said in both places it is asked.
 *
 *  Continue exists twice — on the ribbon, where it is always in view, and at
 *  the end of the step, where the reading order puts it. They are one control
 *  wearing two coats, so they are gated together. */
function gateNext(blocked, hintText) {
  const buttons = document.querySelectorAll("[data-next]");
  if (!buttons.length) return;
  buttons.forEach((b) => { b.disabled = !!blocked; });
  const hint = document.getElementById("navHint");
  if (hint) hint.textContent = blocked ? hintText : "";
}

/** "Show all N characters" on a rendered example.
 *
 *  Rewrites the one paragraph in place rather than going through draw(),
 *  which would re-render the step and fold it straight back up. Shared by the
 *  two steps that show the rendered rows. */
function wireExpand(body, state) {
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
  finetune: [stepGoal, stepModel, stepData, stepTemplate, stepReview],
  scratch: [stepGoal, stepData, stepTemplate, stepDesign, stepReview],
};

// ===========================================================================

function wireNav(mount, ctx) {
  const { state, draw } = ctx;
  const next = $("#nextBtn", mount);
  const hint = $("#navHint", mount);
  if (!next) return;

  // A step you have already been through can be reopened from the ribbon; one
  // you have not reached cannot, because each step reads what the one before
  // it decided.
  wireRibbon(mount, (key) => {
    const want = Number(key);
    if (Number.isFinite(want) && want <= state.step) goToStep(want);
  });

  const names = STEP_NAMES[state.mode];
  const last = names.length - 1;
  // One per step, in order. The format step has no default answer on purpose:
  // it is the one decision that quietly ruins a finished model, so it is
  // asked rather than assumed.
  const noFormat = () => (!state.templateSource
    ? "Choose the format your training text is written in." : null);
  const blockers = state.mode === "scratch" ? [
    () => (!state.runnerId ? "Choose a machine to continue." : null),
    () => (!state.dataset ? "Choose some text to continue." : null),
    noFormat,
    () => (!state.size ? "Choose a size to continue." : null),
    () => (state.plan.status !== "ready" ? "Working out the settings…"
           : state.blocked ? "Fix the problems above before starting."
           : preflightBlocker(state)),
  ] : [
    () => (!state.runnerId ? "Choose a machine to continue." : null),
    () => (!state.model && !state.sourceRun
           ? "Choose a model to continue." : null),
    () => (!state.dataset ? "Choose a dataset to continue." : null),
    noFormat,
    () => (state.ftPlan.status !== "ready" ? "Working out the settings…"
           : state.blocked ? "Choose a smaller model to continue."
           : preflightBlocker(state)),
  ];
  const blocker = blockers[state.step]();

  $$("[data-next]", mount).forEach((b) => { b.disabled = !!blocker; });
  hint.textContent = blocker || "";

  on(mount, "click", "#startOver", () => {
    clearWizardDraft();
    // Replaced rather than pushed: "start fresh" should not leave the draft
    // one press of Back away.
    history.replaceState(null, "", "#/new");
    location.reload();
  });

  on(mount, "click", "[data-back]", () => {
    // Through the address bar, so the browser's own Back does the same thing
    // this button does rather than leaving the wizard and losing everything.
    if (state.step > 0) goToStep(state.step - 1);
  });
  on(mount, "click", "[data-next]", async () => {
    if (state.starting) return;
    if (state.step < last) { goToStep(state.step + 1); return; }

    const job = buildJob(mount, state);
    if (!job) { toast("The plan is not ready yet.", "err"); return; }
    const varied = parseSweep(state);
    if (varied && varied.error) { toast(varied.error, "err"); return; }
    state.starting = true;
    const buttons = $$("[data-next]", mount);
    buttons.forEach((b) => { b.disabled = true; });
    next.textContent = varied ? "Launching variants…" : "Starting…";
    try {
      if (varied) {
        const r = await api.createSweep({
          name: `${job.name || "Run"} · trying ${varied.key}`,
          base: job, vary: { [varied.key]: varied.values } });
        toast(`${r.jobs.length} runs queued.`, "ok");
        clearWizardDraft();
        location.hash = `#/sweeps/${r.sweep_id}`;
      } else {
        const { id } = await api.createJob(job);
        toast("Training run created.", "ok");
        clearWizardDraft();
        location.hash = `#/jobs/${id}`;
      }
    } catch (e) {
      toast(e.message, "err");
      state.starting = false;
      buttons.forEach((b) => { b.disabled = false; });
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

/** How many rows the run will actually see: the chosen split of a studio
 *  dataset when its count is known, the whole dataset otherwise, and a guess
 *  only for a Hub dataset whose size nobody has measured yet. */
/** The split a run trains on unless told otherwise. It was the first key
 *  of the counts dict -- insertion order -- so a file that happened to start
 *  with a validation row trained on the validation split. */
function defaultSplit(splits) {
  const names = Object.keys(splits || {});
  if (!names.length || names.includes("train")) return "train";
  return names.find((n) => !/^(val|validation|dev|test|eval)/i.test(n)) || names[0];
}

function datasetRows(state) {
  const d = state.studioDataset;
  if (d) {
    const inSplit = (d.splits || {})[state.split];
    if (inSplit) return inSplit;
    if (d.rows) return d.rows;
  }
  return state.preview.data?.rows_total || 2000;
}

function buildJob(mount, state) {
  // `mount` may be null: the pre-flight check builds a job to ask about
  // without having a page in hand. The name box is in the document either way.
  const name = $("#jobName", mount || document)?.value || undefined;
  const dataBits = state.studioDataset
    // The split travels with a studio dataset too: it is one file holding
    // every split, and the runner reads the one named here.
    ? { studio_dataset: state.studioDataset.id,
        dataset_split: state.split || "train" }
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
  // The expanded Jinja is dropped: it can be tens of kilobytes, and the
  // decision that produced it -- a name, or the model's own tokenizer -- is
  // recorded instead, so the runner and the playground resolve it the same
  // way this preview did.
  delete trained.chat_template;
  delete trained.use_model_template;
  delete trained.chat_format;
  delete trained.reasoning;
  Object.assign(trained, formatOverlay(state, true));
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
