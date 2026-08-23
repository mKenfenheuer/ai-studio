/**
 * Writing a dataset with a model.
 *
 * The screen leads with the caveats rather than burying them, because
 * generated data fails in ways that look like success — it is fluent, it is
 * plentiful, and it can be uniformly wrong.
 */
import { api } from "../api.js";
import { html, raw, esc, $, $$, on, toast } from "../util.js";

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
];

export async function generateView(mount) {
  const [runners, playable] = await Promise.all([api.runners(), api.playground()]);
  const online = runners.filter((r) => r.status !== "offline");

  const state = {
    mode: "from_prompts",
    runnerId: online[0]?.id || null,
    source: playable[0] ? `job:${playable[0].id}` : "",
    count: 200,
  };

  const draw = () => { mount.innerHTML = layout(state, online, playable); wire(); };

  function wire() {
    on(mount, "click", "[data-mode]", (_e, t) => { state.mode = t.dataset.mode; draw(); });
    on(mount, "change", "#genRunner", (_e, t) => { state.runnerId = t.value; });
    on(mount, "change", "#genSource", (_e, t) => { state.source = t.value; });

    on(mount, "submit", "#genForm", async (e) => {
      e.preventDefault();
      const f = Object.fromEntries(new FormData(e.target).entries());
      const src = state.source || f.source;
      if (!src) return toast("Choose a model to write with.", "err");

      const model = src.startsWith("job:")
        ? { job_id: src.slice(4), kind: kindOf(playable, src.slice(4)) }
        : { base_model: src, kind: "finetune_llm" };

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
      if (state.mode === "from_prompts") cfg.prompts = f.body;
      if (state.mode === "from_topics") cfg.topics = f.body;
      if (state.mode === "from_seeds") cfg.seeds = f.body;

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

const kindOf = (playable, id) =>
  playable.find((p) => p.id === id)?.kind || "finetune_llm";

// ---------------------------------------------------------------------------

function layout(state, online, playable) {
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
        Generating needs a GPU, the same as training does.</div>` : "")}

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
          <div class="field">
            <label for="genBody">${bodyLabel(state.mode)}</label>
            <textarea id="genBody" name="body" rows="10" class="mono"
                      placeholder="${bodyPlaceholder(state.mode)}"
                      required></textarea>
            <div class="hint">${bodyHint(state.mode)}</div>
          </div>
          ${raw(state.mode !== "from_prompts" ? html`
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
                ${raw(playable.map((p) => html`
                  <option value="job:${p.id}"${
                    state.source === `job:${p.id}` ? " selected" : ""}>
                    ${p.name} — your own</option>`).join(""))}
                <optgroup label="From Hugging Face">
                  ${raw(["Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct",
                         "HuggingFaceTB/SmolLM2-1.7B-Instruct"].map((m) =>
                    `<option value="${esc(m)}">${esc(m)}</option>`).join(""))}
                </optgroup>
              </select>
              <div class="hint">A bigger model writes better data and writes it
                more slowly. This is the one place where paying for quality is
                obviously worth it — the data outlives the run.</div>
            </div>
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
              <div class="field">
                <label for="genOut">Shape</label>
                <select id="genOut" name="output">
                  <option value="chat">Conversation turns</option>
                  <option value="text">Plain text</option>
                  <option value="json">JSON the model writes</option>
                </select>
              </div>
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
                         value="512" min="16" max="4096">
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
