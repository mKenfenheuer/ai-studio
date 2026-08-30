/**
 * Writing a dataset with a model.
 *
 * The screen leads with the caveats rather than burying them, because
 * generated data fails in ways that look like success — it is fluent, it is
 * plentiful, and it can be uniformly wrong.
 */
import { api } from "../api.js";
import { html, raw, esc, $, $$, on, toast, fmtNum } from "../util.js";

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
    id: "extend_conversations",
    title: "Carry these conversations further",
    blurb: "You supply a dataset; the model adds more turns to every "
         + "conversation in it. The mode most tool-calling datasets need — "
         + "they are almost all single exchanges, and a model trained only on "
         + "those never learns the fourth turn.",
    dataset: true,
  },
];

export async function generateView(mount) {
  const [runners, playable, hosted, datasets] = await Promise.all([
    api.runners(), api.playground(),
    // A studio with no keys connected simply has no hosted options; it must
    // not be a reason for this page to fail to open.
    api.providers().catch(() => ({ providers: [], connected: [] })),
    api.datasets().catch(() => []),
  ]);
  const online = runners.filter((r) => r.status !== "offline");
  const connected = hosted.connected || [];
  const labelOf = Object.fromEntries(
    (hosted.providers || []).map((p) => [p.id, p.label]));

  const state = {
    mode: "from_prompts",
    runnerId: online[0]?.id || null,
    source: connected.length
      ? `api:${connected[0].provider}`
      : (playable[0] ? `job:${playable[0].id}` : ""),
    // Which model at the provider. Free text, because the list of models a
    // provider offers changes weekly and a dropdown baked in here would be
    // wrong by the time anyone read it.
    apiModel: connected[0]?.model || "",
    count: 200,
  };

  const draw = () => {
    mount.innerHTML = layout(state, online, playable, connected, labelOf,
                             datasets);
    wire();
  };

  function wire() {
    on(mount, "click", "[data-mode]", (_e, t) => { state.mode = t.dataset.mode; draw(); });
    on(mount, "change", "#genRunner", (_e, t) => { state.runnerId = t.value; });
    on(mount, "change", "#genSource", (_e, t) => {
      state.source = t.value;
      // The model box only exists for a hosted provider, so the panel has to
      // be redrawn rather than updated in place.
      draw();
    });
    on(mount, "input", "#genApiModel", (_e, t) => { state.apiModel = t.value; });

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
        temperature: +f.temperature || 0.9,
        max_new_tokens: +f.max_new_tokens || 512,
        output: f.output || "chat",
      };
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
          kind: "generate_dataset", config: cfg });
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
function extendPanel(datasets) {
  const rows = (datasets || []).filter((d) => (d.rows || 0) > 0);
  return html`
    <div class="field">
      <label for="srcDs">Dataset to lengthen</label>
      <select id="srcDs" name="source_dataset_id" required>
        <option value="">Choose a dataset…</option>
        ${raw(rows.map((d) => `<option value="${esc(d.id)}">${esc(d.name)} — ${
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
               placeholder="every row">
        <div class="hint">Blank for all of them.</div>
      </div>
      <div class="field">
        <label for="extraTurns">Extra exchanges per conversation</label>
        <input id="extraTurns" name="extra_turns" type="number" min="1" max="12"
               value="2">
        <div class="hint">Each one is a new question and its answer — plus any
          tool calls and results in between.</div>
      </div>
    </div>
    <div class="field">
      <label for="persona">Who is the person? (optional)</label>
      <input id="persona" name="persona" class="mono"
             placeholder="a busy warehouse supervisor, terse, types in lower case">
      <div class="hint">The model writes the user's turns as well as the
        assistant's, and left to itself it writes a user who talks like an
        assistant. A sentence here is the difference between realistic
        follow-ups and a second assistant interviewing the first.</div>
    </div>
    <label class="check"><input type="checkbox" name="invent_tool_results" checked>
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
function conversationPanel() {
  return html`
    <div class="field">
      <label for="genBody">What should these conversations be?</label>
      <textarea id="genBody" name="body" rows="10" class="mono"
                placeholder="${BRIEF_PLACEHOLDER}" required></textarea>
      <div class="hint">Describe the assistant, the person, and what a good
        exchange looks like. Use <code>{topic}</code> and
        <code>{language}</code> to place them yourself; otherwise they are
        appended.</div>
    </div>
    <div class="grid grid-2">
      <div class="field">
        <label for="genTopics">Situations, one per line</label>
        <textarea id="genTopics" name="topics" rows="6" class="mono"
                  placeholder="turning a light off in a named room&#10;asking what the weather will do tomorrow&#10;setting the thermostat before bed&#10;closing the blinds because of the sun"></textarea>
        <div class="hint">Optional, and the single biggest lever on variety.</div>
      </div>
      <div class="field">
        <label for="genLangs">Languages, one per line</label>
        <textarea id="genLangs" name="languages" rows="6" class="mono"
                  placeholder="German&#10;English&#10;French&#10;Spanish"></textarea>
        <div class="hint">Optional. Each conversation is written entirely in
          one of them — question and answer both.</div>
      </div>
    </div>
    <div class="field">
      <label for="genTools">Tools the conversations may call (JSON)</label>
      <textarea id="genTools" name="tools" rows="8" class="mono"
                placeholder="${TOOLS_PLACEHOLDER}"></textarea>
      <div class="hint">A JSON array of function definitions —
        <code>name</code>, <code>description</code>, <code>parameters</code>.
        They are stored on every row, so the dataset carries its own tool
        schema.</div>
    </div>
    <label class="check"><input type="checkbox" name="with_reasoning" checked>
      Each assistant turn shows its working, including why it called
      what it called</label>
    <label class="check"><input type="checkbox" name="system_from_model">
      The model writes each row's system prompt as well</label>
    <label class="check"><input type="checkbox" name="require_tool_call">
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

// ---------------------------------------------------------------------------

function layout(state, online, playable, connected = [], labelOf = {},
                datasets = []) {
  const mode = MODES.find((m) => m.id === state.mode);
  return html`
    <div class="page-head">
      <a href="#/data" class="tiny">← Datasets</a>
      <h1 style="margin-top:6px">Write a dataset with a model</h1>
      <p class="sub">It runs as a job: progress, a log, and a stop button that
        keeps whatever it has written so far.</p>
    </div>

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
          ${raw(mode.dataset ? extendPanel(datasets)
                : mode.brief ? conversationPanel() : html`
          <div class="field">
            <label for="genBody">${bodyLabel(state.mode)}</label>
            <textarea id="genBody" name="body" rows="10" class="mono"
                      placeholder="${bodyPlaceholder(state.mode)}"
                      required></textarea>
            <div class="hint">${bodyHint(state.mode)}</div>
          </div>`)}
          ${raw(state.mode !== "from_prompts" && !mode.dataset && !mode.brief ? html`
            <details class="adv">
              <summary>Change what it is asked for</summary>
              <div class="field" style="margin-top:8px">
                <label for="genInstr">Instruction template</label>
                <textarea id="genInstr" name="instruction" rows="3" class="mono"
                          placeholder="${templateHint(state.mode)}"></textarea>
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
                ${raw(online.map((r) => html`
                  <option value="${r.id}">${r.name}</option>`).join(""))}
              </select>
            </div>
          </div>

          <div class="card" style="margin-bottom:14px">
            <h3>The result</h3>
            <div class="field">
              <label for="genName">Call the dataset</label>
              <input id="genName" name="dataset_name" type="text"
                     value="Generated dataset" required>
            </div>
            <div class="grid grid-2">
              <div class="field">
                <label for="genCount">How many rows</label>
                <input id="genCount" name="count" type="number" value="${state.count}"
                       min="1" max="100000" required>
              </div>
              ${raw(mode.brief ? "" : html`
              <div class="field">
                <label for="genOut">Shape</label>
                <select id="genOut" name="output">
                  <option value="chat">Conversation turns</option>
                  <option value="text">Plain text</option>
                  <option value="json">JSON the model writes</option>
                </select>
              </div>`)}
            </div>
            <div class="field">
              <label for="genDsSys">System prompt to store in each row</label>
              <input id="genDsSys" name="dataset_system_prompt" type="text"
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
                          placeholder="You write concise, factual training examples."></textarea>
              </div>
              <div class="grid grid-2">
                <div class="field">
                  <label for="genTemp">Temperature</label>
                  <input id="genTemp" name="temperature" type="number"
                         value="0.9" step="0.05" min="0" max="2">
                  <div class="hint">Higher is more varied and less reliable.
                    Below about 0.7 it repeats itself.</div>
                </div>
                <div class="field">
                  <label for="genMax">Longest reply</label>
                  <input id="genMax" name="max_new_tokens" type="number"
                         value="${mode.brief ? 1600 : 512}" min="16" max="8192">
                  ${raw(mode.brief ? html`<div class="hint">A whole
                    conversation, not one answer — it needs the room. Cut
                    short, the JSON is unfinished and the row is dropped.</div>`
                    : "")}
                </div>
              </div>
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
