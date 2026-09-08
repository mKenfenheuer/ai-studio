/**
 * Writing a dataset with a model.
 *
 * The screen leads with the caveats rather than burying them, because
 * generated data fails in ways that look like success — it is fluent, it is
 * plentiful, and it can be uniformly wrong.
 */
import { api } from "../api.js";
import { html, raw, esc, $, $$, on, toast, fmtNum } from "../util.js";
import { requireProject } from "./projects.js";
import { ribbon, rb, group, rbSelect } from "../ribbon.js";
import { breadcrumb } from "../components.js";

const MODES = [
  {
    id: "from_prompts",
    title: "Answer my questions",
    blurb: "You supply the questions; the model writes the answers. This is "
         + "distillation — a capable model teaching a smaller one. The most "
         + "reliable of the three, because you control half of every example.",
  },
  {
    id: "from_topics",
    title: "Cover these topics",
    blurb: "You supply subject matter; the model invents examples about each. "
         + "Good for breadth when you know the shape of the task but not the "
         + "individual cases.",
  },
  {
    id: "from_seeds",
    title: "More like these",
    blurb: "You supply a handful of examples; the model writes variations. "
         + "Everything it produces will resemble your seeds — that is the "
         + "point, and it is also the ceiling.",
  },
  {
    id: "conversations",
    title: "Write whole conversations",
    blurb: "You supply a brief and the tools; the model writes the whole "
         + "exchange — what the person asks, the call it makes, what the "
         + "function returned, and the answer. The reply is held to a schema "
         + "by the provider, so it arrives as a conversation rather than as "
         + "prose describing one.",
    brief: true,
  },
  {
    id: "from_dataset",
    title: "Answer a dataset I already have",
    blurb: "You supply a dataset of questions; the model answers every one of "
         + "them. Distillation onto prompts you already collected, or your "
         + "own model's answers to a split, written down where they can be "
         + "read, corrected and trained on.",
    dataset: true,
    answers: true,
  },
  {
    id: "extend_conversations",
    title: "Carry these conversations further",
    blurb: "You supply a dataset; the model adds more turns to every "
         + "conversation in it. The mode most tool-calling datasets need — "
         + "they are almost all single exchanges, and a model trained only on "
         + "those never learns the fourth turn.",
    dataset: true,
  },
];

export async function generateView(mount, [fromJob] = []) {
  // Writing a dataset is a run like any other, and lands in a project like
  // any other -- which is also where the rows it writes end up.
  const projectId = await requireProject(mount, {
    title: "Which project is this data for?",
    blurb: "The rows a model writes are filed with the project that asked "
         + "for them, so the run that trains on them is beside the run that "
         + "wrote them.",
  });
  if (!projectId) return () => {};
  const [runners, playable, hosted, datasets, jobs] = await Promise.all([
    api.runners(), api.playground(),
    // A studio with no keys connected simply has no hosted options; it must
    // not be a reason for this page to fail to open.
    api.providers().catch(() => ({ providers: [], connected: [] })),
    api.datasets().catch(() => []),
    api.jobs().catch(() => []),
  ]);
  const online = runners.filter((r) => r.status !== "offline");
  const connected = hosted.connected || [];
  const labelOf = Object.fromEntries(
    (hosted.providers || []).map((p) => [p.id, p.label]));
  const earlier = (jobs || []).filter((j) => j.kind === "generate_dataset");

  // A run started from an earlier one begins as that one's settings. Nothing
  // is copied silently: every value lands in a field on this page, where it
  // can be read and changed before anything is spent on it.
  let pre = {};
  let preName = "";
  if (fromJob) {
    try {
      const job = await api.job(fromJob);
      pre = job.config || {};
      preName = job.name || fromJob;
    } catch {
      toast("That run's settings could not be read; starting blank.", "err");
    }
  }

  const state = {
    mode: pre.mode || "from_prompts",
    runnerId: online.some((r) => r.id === pre.required_runner)
      ? pre.required_runner : (online[0]?.id || null),
    source: sourceOf(pre, connected, playable),
    // Which model at the provider. Free text, because the list of models a
    // provider offers changes weekly and a dropdown baked in here would be
    // wrong by the time anyone read it.
    apiModel: pre.model?.model || connected[0]?.model || "",
    count: pre.count || 200,
    pre,
    preName,
    earlier,
    from: fromJob || "",
  };

  const draw = () => {
    mount.innerHTML = layout(state, online, playable, connected, labelOf,
                             datasets);
    wire();
  };

  function wire() {
    on(mount, "click", "[data-mode]", (_e, t) => { state.mode = t.dataset.mode; draw(); });
    on(mount, "change", "#genFrom", (_e, t) => {
      // A whole page reload of the same view: the earlier run's settings are
      // read once, at the top, and everything below is drawn from them.
      location.hash = t.value ? `#/generate/from/${t.value}` : "#/generate";
    });
    on(mount, "change", "#genRunner", (_e, t) => { state.runnerId = t.value; });
    on(mount, "change", "#genSource", (_e, t) => {
      state.source = t.value;
      // The model box only exists for a hosted provider, so the panel has to
      // be redrawn rather than updated in place.
      draw();
    });
    on(mount, "input", "#genApiModel", (_e, t) => { state.apiModel = t.value; });

    // The ribbon's Start is the form's submit, one screen higher.
    on(mount, "click", "[data-submit]", (_e, t) => {
      $(`#${t.dataset.submit}`, mount)?.requestSubmit();
    });

    on(mount, "submit", "#genForm", async (e) => {
      e.preventDefault();
      const f = Object.fromEntries(new FormData(e.target).entries());
      const src = state.source || f.source;
      if (!src) return toast("Choose a model to write with.", "err");

      const model = src.startsWith("job:")
        ? { job_id: src.slice(4), kind: kindOf(playable, src.slice(4)) }
        : src.startsWith("api:")
        ? { provider: src.slice(4), model: (state.apiModel || "").trim() }
        : { base_model: src, kind: "finetune_llm" };
      if (model.provider && !model.model
          && model.provider !== "azure") {
        return toast("Which model at that provider should write it?", "err");
      }

      const cfg = {
        mode: state.mode,
        model,
        count: +f.count || 100,
        required_runner: state.runnerId,
        dataset_name: f.dataset_name || "Generated dataset",
        system_prompt: f.system_prompt || "",
        dataset_system_prompt: f.dataset_system_prompt || "",
        instruction: f.instruction || "",
        temperature: +f.temperature || 1.05,
        max_new_tokens: +f.max_new_tokens || 512,
        output: f.output || "chat",
      };
      if (+f.workers > 0) cfg.workers = +f.workers;
      if (state.mode === "conversations") {
        cfg.instruction = f.body;
        cfg.topics = f.topics || "";
        cfg.languages = f.languages || "";
        cfg.tools = f.tools || "";
        cfg.with_reasoning = f.with_reasoning === "on";
        cfg.system_from_model = f.system_from_model === "on";
        cfg.require_tool_call = f.require_tool_call === "on";
        // Caught here rather than on row 1 of 500, where it is a job that
        // fails a minute after it was started.
        if (cfg.tools.trim()) {
          try {
            JSON.parse(cfg.tools);
          } catch (ex) {
            return toast(`The tools are not valid JSON: ${ex.message}`, "err");
          }
        }
      }
      if (state.mode === "from_prompts") cfg.prompts = f.body;
      if (state.mode === "from_topics") cfg.topics = f.body;
      if (state.mode === "from_seeds") cfg.seeds = f.body;
      if (state.mode === "from_dataset") {
        if (!f.source_dataset_id) {
          return toast("Which dataset holds the questions?", "err");
        }
        cfg.source_dataset_id = f.source_dataset_id;
        cfg.source_split = f.source_split || "";
        cfg.prompt_field = f.prompt_field || "";
        cfg.instruction = f.instruction || "";
        // Blank means every row in the split, which the controller fills in
        // from the split's own size. A number here is a sample of it.
        cfg.count = +f.count || 0;
      }
      if (state.mode === "extend_conversations") {
        if (!f.source_dataset_id) {
          return toast("Which dataset should get the extra turns?", "err");
        }
        cfg.source_dataset_id = f.source_dataset_id;
        cfg.source_split = f.source_split || "";
        cfg.extra_turns = +f.extra_turns || 2;
        cfg.persona = f.persona || "";
        cfg.invent_tool_results = f.invent_tool_results === "on";
        // Every row, unless the box says fewer. Extending half a dataset and
        // leaving the rest is rarely what anybody means.
        cfg.count = +f.count || 0;
      }

      const btn = $("#genGo", mount);
      btn.disabled = true;
      try {
        const r = await api.createJob({
          name: f.dataset_name || "Generating a dataset",
          kind: "generate_dataset", config: cfg, project_id: projectId });
        toast("Started. Watch it write.", "ok");
        location.hash = `#/jobs/${r.id}`;
      } catch (ex) {
        toast(ex.message, "err");
        btn.disabled = false;
      }
    });
  }

  draw();
  return () => {};
}

/** Picking the dataset to lengthen, and how far.
 *
 *  Deliberately blunt about the one thing that is not true in the result: on a
 *  tool-calling dataset the model has to invent what the tool returned,
 *  because no tool ran. That is the right trade for teaching the *shape* of a
 *  multi-step conversation and the wrong one for teaching facts, and which of
 *  those you are doing is not something this page can work out for you.
 */
/** Answering questions that already exist, rather than inventing them. */
function answerPanel(datasets, p = {}) {
  const rows = (datasets || []).filter((d) => (d.rows || 0) > 0);
  return html`
    <div class="field">
      <label for="srcDs">Dataset holding the questions</label>
      <select id="srcDs" name="source_dataset_id" required>
        <option value="">Choose a dataset…</option>
        ${raw(rows.map((d) => `<option value="${esc(d.id)}"${
          sel(d.id, val(p, "source_dataset_id"))}>${esc(d.name)} — ${
          fmtNum(d.rows)} rows</option>`).join(""))}
      </select>
      <div class="hint">One answer per row. The original is not touched — this
        writes a new dataset, like every other transformation here.</div>
    </div>
    <div class="grid grid-2">
      <div class="field">
        <label for="srcSplit">Split</label>
        <input id="srcSplit" name="source_split" class="mono"
               value="${val(p, "source_split")}" placeholder="every row">
        <div class="hint">A held-out split is the interesting one: those are
          the questions nothing was trained on.</div>
      </div>
      <div class="field">
        <label for="promptField">Which column holds the question</label>
        <input id="promptField" name="prompt_field" class="mono"
               value="${val(p, "prompt_field")}" placeholder="work it out">
        <div class="hint">Left blank: the last thing the user said, for a
          dataset of conversations, or the first column that looks like a
          question.</div>
      </div>
    </div>
    <div class="field">
      <label for="genInstr">Say something about how to answer (optional)</label>
      <textarea id="genInstr" name="instruction" rows="3" class="mono"
                placeholder="Answer as a support agent would: two sentences, no apology.">${
                  val(p, "instruction")}</textarea>
      <div class="hint">Put <code>{prompt}</code> where the question should go,
        or leave it out and the question follows on its own line.</div>
    </div>
    <div class="callout" style="margin-top:10px">
      <strong>The answers are the model's, not the truth</strong>
      A dataset written this way teaches a smaller model to imitate this one,
      including where it is wrong. If the source split already has expected
      answers, score against them instead of training on these.</div>`;
}

function extendPanel(datasets, p = {}) {
  const rows = (datasets || []).filter((d) => (d.rows || 0) > 0);
  return html`
    <div class="field">
      <label for="srcDs">Dataset to lengthen</label>
      <select id="srcDs" name="source_dataset_id" required>
        <option value="">Choose a dataset…</option>
        ${raw(rows.map((d) => `<option value="${esc(d.id)}"${
          sel(d.id, val(p, "source_dataset_id"))}>${esc(d.name)} — ${
          fmtNum(d.rows)} rows</option>`).join(""))}
      </select>
      <div class="hint">Every conversation in it gets carried further. The
        original is not touched — this writes a new dataset, like every other
        transformation here.</div>
    </div>
    <div class="grid grid-2">
      <div class="field">
        <label for="srcSplit">Split</label>
        <input id="srcSplit" name="source_split" class="mono"
               value="${val(p, "source_split")}" placeholder="every row">
        <div class="hint">Blank for all of them.</div>
      </div>
      <div class="field">
        <label for="extraTurns">Extra exchanges per conversation</label>
        <input id="extraTurns" name="extra_turns" type="number" min="1" max="12"
               value="${val(p, "extra_turns", 2)}">
        <div class="hint">Each one is a new question and its answer — plus any
          tool calls and results in between.</div>
      </div>
    </div>
    <div class="field">
      <label for="persona">Who is the person? (optional)</label>
      <input id="persona" name="persona" class="mono"
             value="${val(p, "persona")}"
             placeholder="a busy warehouse supervisor, terse, types in lower case">
      <div class="hint">The model writes the user's turns as well as the
        assistant's, and left to itself it writes a user who talks like an
        assistant. A sentence here is the difference between realistic
        follow-ups and a second assistant interviewing the first.</div>
    </div>
    <label class="check"><input type="checkbox" name="invent_tool_results"${
      checked(p, "invent_tool_results", true)}>
      When the model calls a tool, invent a plausible result so the
      conversation can carry on</label>
    <div class="callout callout-warn" style="margin-top:10px">
      <strong>Invented results are plausible, not true</strong>
      No tool actually runs. What comes back is what the model imagines the
      function would return. That is what you want for teaching the shape of a
      multi-step tool conversation, and never what you want for teaching facts
      about your systems.
    </div>`;
}

/** A brief, the tools, and the languages to write it in.
 *
 *  The brief is the whole of the instruction — what the assistant is, who is
 *  talking to it, what a good exchange looks like. Situations and languages
 *  are cycled across it independently, so a dozen of each is a hundred and
 *  forty-four combinations rather than a dozen.
 */
function conversationPanel(p = {}) {
  return html`
    <div class="field">
      <label for="genBody">What should these conversations be?</label>
      <textarea id="genBody" name="body" rows="10" class="mono"
                placeholder="${BRIEF_PLACEHOLDER}" required>${
                  val(p, "instruction")}</textarea>
      <div class="hint">Describe the assistant, the person, and what a good
        exchange looks like. Use <code>{topic}</code> and
        <code>{language}</code> to place them yourself; otherwise they are
        appended.</div>
    </div>
    <div class="grid grid-2">
      <div class="field">
        <label for="genTopics">Situations, one per line</label>
        <textarea id="genTopics" name="topics" rows="6" class="mono"
                  placeholder="light control&#10;climate control&#10;weather question&#10;ambiguous request">${
                    val(p, "topics")}</textarea>
        <div class="hint">Optional, and the single biggest lever on variety. A
          short label is enough — the brief says what a good exchange looks
          like, these only say which one this row is.</div>
      </div>
      <div class="field">
        <label for="genLangs">Languages, one per line</label>
        <textarea id="genLangs" name="languages" rows="6" class="mono"
                  placeholder="German&#10;English&#10;French&#10;Spanish">${
                    val(p, "languages")}</textarea>
        <div class="hint">Optional. Each conversation is written entirely in
          one of them — question and answer both.</div>
      </div>
    </div>
    <div class="field">
      <label for="genTools">Tools the conversations may call (JSON)</label>
      <textarea id="genTools" name="tools" rows="8" class="mono"
                placeholder="${TOOLS_PLACEHOLDER}">${val(p, "tools")}</textarea>
      <div class="hint">A JSON array of function definitions —
        <code>name</code>, <code>description</code>, <code>parameters</code>.
        They are stored on every row, so the dataset carries its own tool
        schema.</div>
    </div>
    <label class="check"><input type="checkbox" name="with_reasoning"${
      checked(p, "with_reasoning", true)}>
      Each assistant turn shows its working, including why it called
      what it called</label>
    <label class="check"><input type="checkbox" name="system_from_model"${
      checked(p, "system_from_model", false)}>
      The model writes each row's system prompt as well</label>
    <label class="check"><input type="checkbox" name="require_tool_call"${
      checked(p, "require_tool_call", false)}>
      Keep only conversations that call a tool</label>
    <div class="callout callout-warn" style="margin-top:10px">
      <strong>The results are invented</strong>
      No function runs. What the tool "returned" is what the model imagined it
      would return — right in shape, made up in substance. That is what teaches
      when to call and how to answer afterwards, and it teaches nothing true
      about your systems.
    </div>`;
}

const BRIEF_PLACEHOLDER =
  "You are writing conversations between a person and the voice assistant "
  + "that runs their home.\nThe person speaks naturally and does not name "
  + "entity ids. The assistant acts, then says what it did in one sentence.";

const TOOLS_PLACEHOLDER = `[{"name": "execute_services", "description": "…", `
  + `"parameters": {"type": "object", "properties": {…}}}]`;

const kindOf = (playable, id) =>
  playable.find((p) => p.id === id)?.kind || "finetune_llm";

/** The model selector's value for a run being copied. */
function sourceOf(pre, connected, playable) {
  const m = pre.model || {};
  if (m.provider) return `api:${m.provider}`;
  if (m.job_id) return `job:${m.job_id}`;
  if (m.base_model) return m.base_model;
  return connected.length ? `api:${connected[0].provider}`
    : (playable[0] ? `job:${playable[0].id}` : "");
}

/** A prefilled value, or the default this page has always used. */
const val = (pre, name, fallback = "") =>
  pre[name] === undefined || pre[name] === null || pre[name] === ""
    ? fallback : pre[name];

const checked = (pre, name, fallback) =>
  (pre[name] === undefined ? fallback : pre[name]) ? " checked" : "";

const sel = (a, b) => (String(a) === String(b) ? " selected" : "");

// ---------------------------------------------------------------------------

function layout(state, online, playable, connected = [], labelOf = {},
                datasets = []) {
  const mode = MODES.find((m) => m.id === state.mode);
  const p = state.pre || {};
  const hostedSource = state.source.startsWith("api:");
  return html`
    <div class="page-head">
      ${raw(breadcrumb({ href: "#/data", label: "Datasets" }))}
      <h1 style="margin:6px 0 0">Write a dataset with a model</h1>
      <p class="sub">It runs as a job: progress, a log, and a stop button that
        keeps whatever it has written so far.</p>
    </div>

    ${raw(ribbon({
      tabs: [{ key: "home", label: "The brief" }], active: "home",
      body: group("This run", [
        rb(null, "▶", "Start writing", { cls: "primary", data: 'data-submit="genForm"' }),
      ]) + (state.earlier?.length ? group("Start from", [
        rbSelect("genFrom", {
          title: "Load an earlier run's brief into this page",
          value: state.from || "",
          options: [["", "A blank brief"]].concat(state.earlier.slice(0, 40)
            .map((j) => [j.id, j.name + (j.status === "succeeded" ? "" : ` — ${j.status}`)])),
        }),
      ]) : "") + group("Elsewhere", [
        rb(null, "▤", "Datasets", { href: "#/data" }),
        rb(null, "◉", "Model keys", { href: "#/account",
          title: "Connect OpenAI, Anthropic, Azure or anything OpenAI-shaped" }),
      ]),
    }))}

    ${raw(state.earlier?.length ? html`
      <p class="muted tiny" style="margin:-4px 0 14px">Loading an earlier run
        brings its brief, situations, languages, tools and settings into this
        page. Change what you want and start a second run — the earlier one
        and its dataset are untouched.</p>` : "")}

    ${raw(state.preName ? html`
      <div class="callout" style="margin-bottom:14px">
        <strong>Copied from “${esc(state.preName)}”</strong>
        Every setting below came from that run. Nothing is sent until you
        start this one.
      </div>` : "")}

    ${raw(!online.length ? html`
      <div class="callout callout-err"><strong>No machine is connected</strong>
        A generation runs as a job, so it needs a machine to run on — even a
        hosted model is driven from one.</div>` : "")}

    <div class="callout callout-warn" style="margin-bottom:14px">
      <strong>Read this before you train on the result</strong>
      A generated dataset can be fluent and wrong at the same time, and the
      loss curve will not tell you which. Nothing here can be better than the
      model that wrote it — this is for teaching a small model what a bigger
      one already knows, not for creating knowledge neither of them has. Look
      at the rows on the dataset page before you use them.
    </div>

    <div class="grid" style="margin-bottom:14px">
      ${raw(MODES.map((m) => html`
        <button class="pick ${state.mode === m.id ? "selected" : ""}" data-mode="${m.id}">
          <span class="t">${m.title}</span>
          <span class="d">${m.blurb}</span>
        </button>`).join(""))}
    </div>

    <form id="genForm">
      <div class="grid grid-2" style="align-items:start">
        <div class="card">
          <h3>${mode.title}</h3>
          ${raw(mode.answers ? answerPanel(datasets, p)
                : mode.dataset ? extendPanel(datasets, p)
                : mode.brief ? conversationPanel(p) : html`
          <div class="field">
            <label for="genBody">${bodyLabel(state.mode)}</label>
            <textarea id="genBody" name="body" rows="10" class="mono"
                      placeholder="${bodyPlaceholder(state.mode)}"
                      required>${bodyValue(state.mode, p)}</textarea>
            <div class="hint">${bodyHint(state.mode)}</div>
          </div>`)}
          ${raw(state.mode !== "from_prompts" && !mode.dataset && !mode.brief ? html`
            <details class="adv">
              <summary>Change what it is asked for</summary>
              <div class="field" style="margin-top:8px">
                <label for="genInstr">Instruction template</label>
                <textarea id="genInstr" name="instruction" rows="3" class="mono"
                          placeholder="${templateHint(state.mode)}">${
                            val(p, "instruction")}</textarea>
                <div class="hint">${state.mode === "from_topics"
                  ? "Use {topic} where the topic should go."
                  : "Use {examples} where the seed examples should go."}</div>
              </div>
            </details>` : "")}
        </div>

        <div>
          <div class="card" style="margin-bottom:14px">
            <h3>Which model writes it</h3>
            <div class="field">
              <label for="genSource">Model</label>
              <select id="genSource" name="source">
                ${raw(connected.length ? html`
                  <optgroup label="Hosted — billed to your account">
                    ${raw(connected.map((c) => html`
                      <option value="api:${c.provider}"${
                        state.source === `api:${c.provider}` ? " selected" : ""}>
                        ${labelOf[c.provider] || c.provider}${
                          c.deployment ? ` · ${c.deployment}` : ""}</option>`).join(""))}
                  </optgroup>` : "")}
                <optgroup label="On your own machine">
                  ${raw(playable.map((p) => html`
                    <option value="job:${p.id}"${
                      state.source === `job:${p.id}` ? " selected" : ""}>
                      ${p.name} — your own</option>`).join(""))}
                  ${raw(["Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct",
                         "HuggingFaceTB/SmolLM2-1.7B-Instruct"].map((m) =>
                    `<option value="${esc(m)}"${
                      state.source === m ? " selected" : ""}>${esc(m)} — from Hugging Face</option>`).join(""))}
                </optgroup>
              </select>
              <div class="hint">A bigger model writes better data and writes it
                more slowly. This is the one place where paying for quality is
                obviously worth it — the data outlives the run.</div>
            </div>

            ${raw(state.source.startsWith("api:") ? html`
              <div class="field">
                <label for="genApiModel">Which model there</label>
                <input id="genApiModel" class="mono" value="${state.apiModel}"
                       placeholder="${state.source === "api:azure"
                         ? "leave blank to use the deployment" : "model name"}">
                <div class="hint">Exactly as the provider names it.
                  <a href="#/account">Your account page</a> lists what this
                  connection can reach.</div>
              </div>
              <div class="callout" style="margin:0">
                <strong>This does not use the GPU</strong>
                The rows are written over the network and billed to the account
                you connected. A machine is still chosen below, because the run
                is a job like any other — it just spends network rather than
                VRAM.
              </div>` : "")}
            <div class="field">
              <label for="genRunner">Machine</label>
              <select id="genRunner" name="runner">
                ${raw(online.map((r) => `<option value="${esc(r.id)}"${
                  sel(r.id, state.runnerId)}>${esc(r.name)}</option>`).join(""))}
              </select>
            </div>
          </div>

          <div class="card" style="margin-bottom:14px">
            <h3>The result</h3>
            <div class="field">
              <label for="genName">Call the dataset</label>
              <input id="genName" name="dataset_name" type="text"
                     value="${val(p, "dataset_name", "Generated dataset")}"
                     required>
            </div>
            <div class="grid grid-2">
              <div class="field">
                <label for="genCount">How many rows</label>
                ${raw(mode.answers ? html`
                  <input id="genCount" name="count" type="number"
                         value="${val(p, "count") || ""}" min="1" max="100000"
                         placeholder="every row in the split">
                  <div class="hint">There is one answer per row, so this is a
                    sample rather than a target. Blank answers all of
                    them.</div>`
                  : html`
                  <input id="genCount" name="count" type="number" value="${state.count}"
                         min="1" max="100000" required>`)}
              </div>
              ${raw(mode.brief ? "" : html`
              <div class="field">
                <label for="genOut">Shape</label>
                <select id="genOut" name="output">
                  <option value="chat"${sel("chat", val(p, "output", "chat"))}>Conversation turns</option>
                  <option value="text"${sel("text", val(p, "output"))}>Plain text</option>
                  <option value="json"${sel("json", val(p, "output"))}>JSON the model writes</option>
                </select>
              </div>`)}
            </div>
            <div class="field">
              <label for="genDsSys">System prompt to store in each row</label>
              <input id="genDsSys" name="dataset_system_prompt" type="text"
                     value="${val(p, "dataset_system_prompt")}"
                     placeholder="optional">
              <div class="hint">Goes into the dataset, so the model you later
                train sees it during training.</div>
            </div>
          </div>

          <div class="card">
            <details class="adv">
              <summary>How it is asked</summary>
              <div class="field" style="margin-top:8px">
                <label for="genSys">System prompt for the writer</label>
                <textarea id="genSys" name="system_prompt" rows="3"
                          placeholder="You write concise, factual training examples.">${
                            val(p, "system_prompt")}</textarea>
              </div>
              <div class="grid grid-2">
                <div class="field">
                  <label for="genTemp">Temperature</label>
                  <input id="genTemp" name="temperature" type="number"
                         value="${val(p, "temperature", 1.05)}" step="0.05"
                         min="0" max="2">
                  <div class="hint">Higher is more varied and less reliable.
                    Set high on purpose: below about 0.9 a long run writes the
                    same handful of examples over and over.</div>
                </div>
                <div class="field">
                  <label for="genMax">Longest reply</label>
                  <input id="genMax" name="max_new_tokens" type="number"
                         value="${val(p, "max_new_tokens", mode.brief ? 1600 : 512)}"
                         min="16" max="8192">
                  ${raw(mode.brief ? html`<div class="hint">A whole
                    conversation, not one answer — it needs the room. Cut
                    short, the JSON is unfinished and the row is dropped.</div>`
                    : "")}
                </div>
              </div>
              ${raw(hostedSource ? html`
                <div class="field">
                  <label for="genWorkers">Rows written at once</label>
                  <input id="genWorkers" name="workers" type="number"
                         value="${val(p, "workers", 6)}" min="1" max="16">
                  <div class="hint">Hosted rows are network waiting, not
                    computation, so several can be in flight together: a
                    thousand-row run is an hour rather than an afternoon. Too
                    many and the provider starts refusing — the run backs off
                    and carries on, more slowly than if you had asked for
                    fewer.</div>
                </div>` : "")}
            </details>
          </div>
        </div>
      </div>

      <div class="row" style="margin-top:14px;gap:8px">
        <button class="btn-primary" type="submit" id="genGo"
                ${!online.length ? "disabled" : ""}>Start writing</button>
        <a class="btn" href="#/data">Cancel</a>
      </div>
    </form>`;
}

const bodyLabel = (m) => ({
  from_prompts: "Questions, one per line",
  from_topics: "Topics, one per line",
  from_seeds: "Example rows, one per line",
}[m]);

const bodyPlaceholder = (m) => ({
  from_prompts: "How do I reset my password?\nWhat are your opening hours?\n"
              + "Can I change my delivery address after ordering?",
  from_topics: "refunds\ndelivery delays\nchanging an order\naccount security",
  from_seeds: "Turn the lights off in the kitchen.\nSet a timer for ten minutes.\n"
            + "What is the temperature in the bedroom?",
}[m]);

/** What the one big box held in the run being copied. */
const bodyValue = (m, p) => val(p, {
  from_prompts: "prompts", from_topics: "topics", from_seeds: "seeds",
}[m] || "");

const bodyHint = (m) => ({
  from_prompts: "Each one is asked once, then they cycle round until the row "
              + "count is reached. More questions means more varied data.",
  from_topics: "It works through these in turn. A dozen topics gives far more "
             + "variety than one repeated a hundred times.",
  from_seeds: "At least two. Three are shown at random each time, so the model "
            + "sees the pattern rather than one example.",
}[m]);

const templateHint = (m) => m === "from_topics"
  ? "Write one realistic example about: {topic}\n\nReply with the example only."
  : "Here are some examples:\n\n{examples}\n\nWrite one more in the same style.";
