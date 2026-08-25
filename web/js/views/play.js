import { api, events } from "../api.js";
import { html, raw, esc, $, $$, on, fmtAgo, toast, inlineRename } from "../util.js";
import { conversationHtml, reasoningBlock, pretty } from "../conversation.js";

// Talking to what you trained.
//
// The rule here is that the Playground speaks the shape the model was taught,
// never a shape of its own. Every run records the exact format it trained
// with -- the base model's own chat template, a plain readable rendering, or a
// Jinja template written by hand -- and this view sends the conversation back
// through that same format. Get it wrong and a perfectly good fine-tune looks
// broken: give a model a layout it has never seen and it ignores half of what
// it learned.
//
// Three styles come out of that, and they need different interfaces:
//   chat      roles, a system prompt, and turns that accumulate
//   instruct  one question at a time, wrapped in its training template
//   continue  a base model, which continues text and has never seen a question
//
// ## Trying it on data it has never seen
//
// Typing a question and reading the answer tells you whether a model produces
// *something*. It does not tell you whether it produces the right thing, and
// for a tool-calling model it barely tells you anything at all -- you have to
// know the tools, guess a question they apply to, and then judge a call
// against a schema you are holding in your head.
//
// So a conversation can be loaded from a held-out row instead: the questions
// the model was never trained on, with the reply the data says is correct
// shown beside whatever the model produces, and the row's tool definitions
// declared automatically. The prefill is editable, because the useful thing to
// do with a held-out example is nearly always to change one word of it and see
// whether the answer survives.
//
// And when the model calls a tool, the result the dataset recorded can be
// handed straight back, so a multi-step exchange runs to the end instead of
// stopping at the first call.
const STYLE_UI = {
  chat: {
    icon: "💬",
    title: "Have a conversation",
    placeholder: "Say something…",
    note: "This model was trained on conversations, so it keeps track of the "
        + "turns. Your messages go back through the exact format it learned.",
    system: true, multiturn: true,
  },
  instruct: {
    icon: "📐",
    title: "Ask it something",
    placeholder: "Ask a question, or give it a task…",
    note: "This model learned single instructions and their answers. Each "
        + "message is wrapped in the template it was trained with; it does not "
        + "remember earlier ones.",
    system: true, multiturn: false,
  },
  continue: {
    icon: "✍️",
    title: "Write an opening and it will continue",
    placeholder: "Once upon a time, there was a little…",
    note: "This is a base model: it continues text rather than answering "
        + "questions. Give it the first few words of something and see where "
        + "it goes.",
    system: false, multiturn: false,
  },
};

const styleOf = (run) => STYLE_UI[run.style || run.mode] || STYLE_UI.continue;

export async function playView(mount, [jobId]) {
  const runs = await api.playground();
  if (!jobId) return pickerView(mount, runs);

  const run = runs.find((r) => r.id === jobId);
  if (!run) {
    mount.innerHTML = html`
      <div class="card empty"><div class="big">🤷</div>
        <h2>That run has nothing to try</h2>
        <p class="muted">It may still be training, or it may have been deleted.</p>
        <p><a class="btn" href="#/play">Pick another run</a></p></div>`;
    return () => {};
  }
  return chatView(mount, run, runs);
}

// ---------------------------------------------------------------- picker
function pickerView(mount, runs) {
  mount.innerHTML = html`
    <div class="page-head">
      <h1>Playground</h1>
      <p class="sub">Every finished run stays here. Come back and talk to any of
        them whenever you like.</p>
    </div>
    ${raw(runs.length ? html`
      <div class="grid grid-2">
        ${raw(runs.map(runCard).join(""))}
      </div>` : html`
      <div class="card empty"><div class="big">🎈</div>
        <h2>Nothing to try yet</h2>
        <p class="muted">When a training run finishes, it appears here and you
          can chat with it.</p>
        <p><a class="btn btn-primary" href="#/new">Start a training run</a></p>
      </div>`)}`;
  return () => {};
}

function runCard(r) {
  const ui = styleOf(r);
  return html`
    <a class="pick" href="#/play/${r.id}" style="text-decoration:none">
      <span class="t"><span style="font-size:17px">${ui.icon}</span>${r.name}</span>
      <span class="d">${r.kind === "pretrain_llm"
        ? `Built from scratch on ${r.dataset}`
        : `${r.base_model} fine-tuned on ${r.dataset}`}</span>
      <span class="row" style="gap:6px;flex-wrap:wrap">
        <span class="badge ${r.kind === "pretrain_llm" ? "badge-ok" : "badge-accent"}">
          ${r.kind === "pretrain_llm" ? "your own model" : "fine-tune"}</span>
        <span class="badge">${ui.title.toLowerCase()}</span>
        ${raw(r.system_prompt ? `<span class="badge badge-accent">has a system prompt</span>` : "")}
        ${raw(r.held_out_split
          ? `<span class="badge badge-ok">held-out rows to try</span>` : "")}
        ${raw(r.stopped_early
          ? `<span class="badge badge-warn" title="This run was stopped before
               it finished, so the model had less practice than planned."
             >stopped early</span>` : "")}
        <span class="badge">${fmtAgo(r.finished_at)}</span>
      </span>
    </a>`;
}

// --------------------------------------------------------------- helpers
//
// Drawing a conversation lives in ../conversation.js, shared with the dataset
// workbench. A held-out row read there and the same row loaded here are the
// same conversation, so they are drawn by the same code -- two renderers would
// be two chances for "what the data says" and "what I am sending" to disagree,
// which is the one disagreement this page exists to rule out.

/** One canonical message as plain text, for an editor box. */
const bodyOf = (m) => m.role === "assistant" && !m.content && m.tool_calls?.length
  ? "" : (m.content || "");

// --------------------------------------------------------------- the view
function chatView(mount, run, runs) {
  const ui = styleOf(run);
  // The conversation, in canonical form: the same message objects the trainer
  // and the API use. Keeping the playground's own shape here is how the two
  // drift apart, so there is only the one.
  let turns = [];
  let tools = [];            // declared for this exchange
  let requestId = null;
  let early = [];            // events that beat their own POST response
  let think = !!run.reasoning;

  // The held-out example currently loaded, if any.
  let sample = null;         // {index, messages, tools, prompt, expected, ...}
  let samples = [];          // the page of rows fetched
  let at = 0;                // which of them is loaded
  let sourceId = run.dataset_id || "";
  let sourceSplit = run.held_out_split || "";
  let datasets = [];
  let autoTools = true;      // hand back recorded results without being asked

  mount.innerHTML = html`
    <div class="page-head">
      <a href="#/play" class="tiny">← All finished runs</a>
      <div class="row-between" style="flex-wrap:wrap;gap:8px;margin-top:6px">
        <div class="row title-row" style="gap:4px;min-width:0">
          <h1 style="margin:0" id="runTitle">${run.name}</h1>
          <button class="btn-sm btn-quiet" id="renameRun" title="Rename this run"
            aria-label="Rename this run">&#9998;</button>
        </div>
        <div class="row">
          <a class="btn btn-sm" href="#/jobs/${run.id}">Training details</a>
          <a class="btn btn-sm" href="/api/jobs/${run.id}/download">↓ Download</a>
        </div>
      </div>
      <p class="sub tiny" style="margin-top:4px">${ui.title}</p>
    </div>

    <div class="callout" style="margin-bottom:14px">
      <strong>${ui.icon} ${run.kind === "pretrain_llm"
        ? "This is the model you built" : "This is your fine-tune"}</strong>
      ${ui.note}
    </div>

    <div class="card" id="trialCard" style="margin-bottom:14px"></div>

    ${raw(ui.system ? html`
      <details class="adv" id="sysBox">
        <summary>System prompt${raw(run.system_prompt
          ? ` <span class="badge badge-accent">from your training data</span>` : "")}</summary>
        <div class="card" style="margin-top:10px">
          <div class="field" style="margin-bottom:8px">
            <label for="systemBox">Standing instructions, sent before every message</label>
            <textarea id="systemBox" rows="5" class="mono"
              placeholder="You are a helpful assistant.">${run.system_prompt || ""}</textarea>
            <div class="hint">${raw(run.system_prompt
              ? "This is the system prompt found in the data this model was "
              + "trained on. It behaves closest to its training with this in "
              + "place — edit it to see how much it depends on it."
              : "This model's training data had no system prompt, but you can "
              + "still set one.")}</div>
          </div>
          <div class="row">
            ${raw(run.system_prompt
              ? `<button class="btn-sm" id="resetSystem">Restore the trained one</button>` : "")}
            <button class="btn-sm" id="clearSystem">Clear</button>
          </div>
        </div>
      </details>` : "")}

    <div class="card chat" style="margin-top:14px">
      <div class="chat-log" id="chatLog">
        <div class="chat-empty" id="chatEmpty">Nothing said yet. ${ui.title}.</div>
      </div>
      <div class="tiny muted" id="chatStatus"></div>
      <div class="chat-input">
        <textarea id="chatBox" rows="2" placeholder="${ui.placeholder}"></textarea>
        <button class="btn-primary" id="sendBtn">Send</button>
        <button class="btn-primary" id="askBtn" hidden
          title="Send the conversation as it stands and let the model write the next turn"
          >Let it answer</button>
        <button class="btn-danger" id="stopBtn" hidden>Stop</button>
      </div>
      <div class="row" style="margin-top:8px;flex-wrap:wrap">
        ${raw(ui.multiturn
          ? `<button class="btn-sm" id="resetChat">New conversation</button>` : "")}
        ${raw(run.reasoning
          ? `<button class="btn-sm btn-primary" id="thinkBtn"
                     title="Ask it to work through the problem first">Reasoning: on</button>`
          : "")}
        <span class="tiny muted" id="turnCount"></span>
        <span class="tiny muted" id="turnHint">· right-click a message to edit,
          remove or regenerate it</span>
      </div>
      ${raw(run.reasoning ? html`
        <p class="muted tiny" style="margin:8px 0 0">This model was trained to
          reason before answering. Its working is shown above each reply and
          can be folded away.</p>` : "")}
      <details class="adv">
        <summary>Generation settings</summary>
        <div class="grid grid-3" style="margin-top:10px">
          <div class="field">
            <label for="temp">Creativity</label>
            <input type="number" id="temp" value="0.8" step="0.1" min="0" max="2">
            <div class="hint">0 always picks the likeliest next word. Higher is
              more varied and less reliable.</div>
          </div>
          <div class="field">
            <label for="maxTok">Length limit</label>
            <input type="number" id="maxTok" value="512" step="64" min="16" max="4096">
            <div class="hint">Most tokens it may write before it is stopped.
              You are told in the conversation when a reply reaches this, so a
              cut-off answer is never mistaken for a finished one. Long limits
              are long waits — this writes one token at a time — and Stop works
              at any point.</div>
          </div>
        </div>
        <details class="adv">
          <summary>What the model is actually being sent</summary>
          <p class="muted tiny" style="margin:8px 0 4px">The finished prompt,
            after your conversation is put through the format this run was
            trained with. Filled in after the first reply.</p>
          <pre class="txt mono tiny" id="promptPeek"
               style="white-space:pre-wrap;background:var(--surface-2);
                      padding:10px;border-radius:8px;max-height:240px;
                      overflow:auto">(nothing sent yet)</pre>
        </details>
      </details>
    </div>

    ${raw(runs.length > 1 ? html`
      <div class="card" style="margin-top:14px">
        <h3 style="margin:0 0 8px">Other finished runs</h3>
        <div class="row" style="flex-wrap:wrap;gap:8px">
          ${raw(runs.filter((r) => r.id !== run.id).slice(0, 8).map((r) =>
            `<a class="btn btn-sm" href="#/play/${esc(r.id)}">${esc(r.name)}</a>`).join(""))}
        </div>
      </div>` : "")}`;

  const log = $("#chatLog", mount);
  const box = $("#chatBox", mount);
  const sendBtn = $("#sendBtn", mount);
  const askBtn = $("#askBtn", mount);
  const stopBtn = $("#stopBtn", mount);
  const statusEl = $("#chatStatus", mount);
  const systemBox = $("#systemBox", mount);
  const trial = $("#trialCard", mount);

  const scroll = () => { log.scrollTop = log.scrollHeight; };

  // ------------------------------------------------------- trying a row
  //
  // The dataset picker, the row navigation, and the expected reply. Drawn on
  // its own so that loading a row, editing one, and moving to the next all go
  // through one place.
  function drawTrial() {
    if (!ui.multiturn && !ui.system) {
      // A base model continues text; there is no held-out "reply" to compare
      // against, so the whole panel would be a category error.
      trial.hidden = true;
      return;
    }
    const options = datasets.length ? datasets : (run.dataset_id
      ? [{ id: run.dataset_id, name: run.dataset_name || "the training data" }] : []);
    const chosen = options.find((d) => d.id === sourceId);
    const splits = chosen?.splits || run.splits || {};
    const splitNames = Object.keys(splits);

    trial.innerHTML = html`
      <div class="row-between" style="flex-wrap:wrap;gap:8px">
        <h3 style="margin:0">🎯 Try it on data it has never seen</h3>
        ${raw(sample ? html`
          <span class="tiny muted">row ${sample.index} of
            ${chosen?.name || "the dataset"}</span>` : "")}
      </div>
      <p class="muted tiny" style="margin:6px 0 10px">A held-out row is one the
        model never trained on, so what it does with it is the only honest
        answer to “did this work”. Load one and its question lands in the box
        below, yours to edit before you send it; the reply the data says is
        correct is shown here beside the model’s.</p>

      <div class="row" style="flex-wrap:wrap;gap:8px;align-items:flex-end">
        <div class="field" style="margin:0;min-width:200px">
          <label for="dsPick">Dataset</label>
          <select id="dsPick">
            ${raw(options.map((d) => `<option value="${esc(d.id)}"
              ${d.id === sourceId ? "selected" : ""}>${esc(d.name)}${
                d.id === run.dataset_id ? " — what it trained on" : ""}</option>`).join(""))}
            ${raw(options.length ? "" : `<option value="">(no datasets)</option>`)}
          </select>
        </div>
        <div class="field" style="margin:0;min-width:150px">
          <label for="splitPick">Split</label>
          <select id="splitPick">
            <option value="" ${!sourceSplit ? "selected" : ""}>Every row</option>
            ${raw(splitNames.map((s) => `<option value="${esc(s)}"
              ${s === sourceSplit ? "selected" : ""}>${esc(s)} (${splits[s]})${
                s === run.held_out_split ? " — held out" : ""}</option>`).join(""))}
          </select>
        </div>
        <button class="btn btn-primary btn-sm" id="loadRow">
          ${sample ? "Load another" : "Load a row"}</button>
        ${raw(sample ? html`
          <button class="btn btn-sm" id="prevRow" ${at <= 0 ? "disabled" : ""}>‹ Previous</button>
          <button class="btn btn-sm" id="nextRow">Next ›</button>` : "")}
      </div>

      ${raw(sample ? trialDetail(sample) : "")}`;
  }

  function trialDetail(s) {
    const problems = (s.problems || []).filter((p) => p.level !== "ok");
    return html`
      <div style="margin-top:12px;border-top:1px solid var(--border);padding-top:12px">
        ${raw(s.tools?.length ? html`
          <div class="row" style="flex-wrap:wrap;gap:6px;margin-bottom:8px">
            <span class="tiny muted">Tools declared to the model:</span>
            ${raw(s.tools.map((t) => `<span class="badge badge-accent">${
              esc(t.function?.name || "?")}</span>`).join(""))}
          </div>` : "")}
        ${raw(problems.length ? html`
          <div class="callout callout-warn" style="margin-bottom:8px">
            <strong>This row has problems</strong>
            ${raw(problems.map((p) => `<div class="tiny">${esc(p.message)}</div>`).join(""))}
          </div>` : "")}
        ${raw(s.repaired?.length ? html`
          <p class="tiny muted" style="margin:0 0 8px">The data left some of
            this implicit; it was worked out on load: ${s.repaired.join("; ")}.</p>` : "")}

        <details class="adv" ${s.expected?.length ? "open" : ""}>
          <summary>What the data says should happen next
            (${(s.expected || []).length} message${
              (s.expected || []).length === 1 ? "" : "s"})</summary>
          <div style="margin-top:8px">
            ${raw((s.expected || []).map((m) => expectedBlock(m)).join("")
                  || `<p class="muted tiny">Nothing — this row ends with the
                      question.</p>`)}
          </div>
        </details>

        <label class="check" style="margin-top:8px">
          <input type="checkbox" id="autoTools" ${autoTools ? "checked" : ""}>
          <span>When the model calls a tool, hand back the result this row
            recorded so the conversation can carry on</span>
        </label>
      </div>`;
  }

  function expectedBlock(m) {
    if (m.role === "tool") {
      return html`<div class="tiny mono" style="opacity:.75;margin:4px 0">
        <span class="badge">${m.name || "tool"} returned</span>
        <pre style="white-space:pre-wrap;margin:4px 0 0">${pretty(m.content)}</pre>
      </div>`;
    }
    const calls = (m.tool_calls || []).map((c) => html`
      <div class="tiny mono" style="margin:4px 0">⚙ ${c.function.name}(<pre
        style="white-space:pre-wrap;display:inline">${c.function.arguments}</pre>)</div>`).join("");
    return html`
      <div style="margin:6px 0">
        ${raw(m.reasoning ? `<div class="tiny muted" style="font-style:italic">${
          esc(m.reasoning)}</div>` : "")}
        ${raw(m.content ? `<div class="txt">${esc(m.content)}</div>` : "")}
        ${raw(calls)}
      </div>`;
  }

  // ------------------------------------------------------------ painting
  //
  // The log is redrawn from `turns` rather than appended to, because a turn
  // can now be edited and a tool result can be inserted in the middle. Two
  // ways of getting a message on screen is two ways for the screen to stop
  // matching what will actually be sent.
  function paint() {
    const last = turns.length - 1;
    const body = conversationHtml(turns, {
      // Every turn can be edited, removed or regenerated -- the useful thing to
      // do with a held-out example is change one word of it and ask what
      // survives. The controls live in a context menu rather than beside every
      // message: two buttons on each of twenty turns is forty buttons, and the
      // conversation is the thing you came to read.
      footer: (m) => (m.stopped_at_limit
        ? `<div class="turn-note">Stopped at the length limit — this reply is
             cut off, not finished. Raise the limit in generation settings and
             regenerate to see the rest.</div>`
        : ""),
      // Answering a call by hand, when the row recorded no result for it or
      // you want to see what a different result would do.
      callAction: (c, _n) => (turns[last]?.tool_calls || []).includes(c)
        ? `<button class="btn-sm" data-answer="${esc(c.id || "")}">Return a result…</button>`
        : "",
    });
    log.innerHTML = turns.length || live ? body : html`
      <div class="chat-empty">Nothing said yet. ${ui.title}.</div>`;
    if (live) log.insertAdjacentHTML("beforeend", liveHtml());
    // Something is loaded that the model could answer without another word
    // being typed -- a row that ends on a tool result, or turns left standing
    // after an edit. Without this the conversation simply sits there.
    const tail = turns[turns.length - 1];
    askBtn.hidden = !(tail && tail.role !== "assistant"
                      && !live && !requestId);
    const el = $("#turnCount", mount);
    if (el) el.textContent = turns.length
      ? `${turns.length} message${turns.length > 1 ? "s" : ""} in context` : "";
    scroll();
  }

  // The turn being written right now, as its two channels. Null when nothing
  // is in flight. The runner tags every delta "reasoning" or "content", so
  // this is a running copy of what it has said rather than a guess made here:
  // the working goes to the reasoning panel from its first character instead
  // of being typed into the answer and taken back when the stream ends.
  let live = null;

  /** The in-flight turn. Shaped exactly like a finished one, so it does not
   *  jump when the real message replaces it. */
  function liveHtml() {
    const reasoning = (live.reasoning || "").trim();
    const answer = live.content || "";
    // The working folds itself away once the answer starts, which is the
    // moment it stops being the interesting thing on screen.
    return html`
      <div class="turn it live" id="liveTurn" data-role="assistant">
        <div class="turn-who"><span class="turn-mark">◆</span><span>Assistant</span></div>
        ${raw(reasoning
          ? reasoningBlock(reasoning, { live: !answer }) : "")}
        ${raw(answer || !reasoning ? html`
          <div class="bubble ${answer ? "" : "pending"}"
            ><div class="bubble-text" id="liveText">${answer}</div></div>` : "")}
      </div>`;
  }

  /** Grow the live turn in place.
   *
   *  Text arrives many times a second and a full repaint each time would
   *  fight the reader: it closes a reasoning panel they just opened and drops
   *  the selection they were making. So only a *structural* change -- the
   *  working appearing, the answer starting -- repaints; everything else
   *  writes into the node that is already there.
   */
  function grow(channel) {
    const node = $(channel === "reasoning"
      ? "#liveTurn .reasoning-body" : "#liveText", mount);
    if (!node) { paint(); return; }
    node.textContent = live[channel];
    const label = $("#liveTurn .reasoning-label", mount);
    if (channel === "reasoning" && label) {
      label.textContent = `Thinking — ${live.reasoning.length.toLocaleString()} characters`;
    }
    scroll();
  }

  // ---------------------------------------------------------- generating
  const finish = () => {
    requestId = null;
    early = [];
    live = null;
    sendBtn.disabled = false;
    stopBtn.hidden = true;
    paint();
  };

  /** Send whatever is in `turns` and let the model write the next turn. */
  async function ask() {
    if (requestId) return;
    live = { reasoning: "", content: "" };
    paint();
    statusEl.textContent = "Waking the model up…";
    early = [];
    sendBtn.disabled = true;
    try {
      const r = await api.chat(run.id, {
        messages: turns,
        tools,
        system: systemBox ? systemBox.value : "",
        temperature: parseFloat($("#temp", mount).value) || 0.8,
        max_new_tokens: parseInt($("#maxTok", mount).value, 10) || 512,
        reasoning: think,
      });
      requestId = r.request_id;
      stopBtn.hidden = false;
      statusEl.textContent = `Running on ${r.runner}…`;
      const buffered = early.filter((m) => m.request_id === requestId);
      early = [];
      buffered.forEach(handle);
    } catch (e) {
      live = null;
      paint();
      statusEl.textContent = "";
      sendBtn.disabled = false;
      toast(e.message, "err");
    }
  }

  const send = async () => {
    const text = box.value.trim();
    if (!text || requestId) return;
    // A model that never learned to follow a conversation is not given one:
    // each instruction stands alone, exactly as it did in training.
    if (!ui.multiturn) turns = [];
    turns.push({ role: "user", content: text });
    box.value = "";
    await ask();
  };

  /** The result this row recorded for a call the model just made, if any. */
  function recordedResult(call) {
    for (const m of (sample?.expected || [])) {
      if (m.role !== "tool") continue;
      if (m.name && m.name === call.function.name) return m.content;
    }
    return null;
  }

  /** Hand a tool result back and let the model carry on. */
  async function returnResult(call, content) {
    turns.push({ role: "tool", tool_call_id: call.id,
                 name: call.function.name, content });
    paint();
    await ask();
  }

  // ------------------------------------------------------------- loading
  async function loadRow(index) {
    const id = sourceId;
    if (!id) { toast("Pick a dataset to try rows from.", "err"); return; }
    try {
      if (!samples.length || index === undefined) {
        const r = await api.datasetConversations(id, 0, 25, sourceSplit);
        samples = r.rows || [];
        at = 0;
        if (!samples.length) {
          toast("That split has no rows that read as conversations.", "err");
          return;
        }
      } else {
        at = Math.max(0, Math.min(index, samples.length - 1));
      }
      applyRow(samples[at]);
    } catch (e) {
      toast(e.message, "err");
    }
  }

  /** Put a held-out row into the conversation, ready to be sent or edited.
   *
   *  The row arrives already cut at the point a model would have to take over,
   *  and it lands in three places rather than one: its system turn in the
   *  system box, everything before the question in the log as context, and the
   *  question itself in the input box. The reply is deliberately NOT loaded --
   *  producing it is the model's job, and having it already on screen is how
   *  you talk yourself into believing a wrong answer was right.
   */
  function applyRow(s) {
    sample = s;
    let prompt = JSON.parse(JSON.stringify(s.prompt || []));
    tools = s.tools || [];
    const sys = prompt.find((m) => m.role === "system");
    if (sys && systemBox) systemBox.value = sys.content || "";
    // The system turn lives in the box, not in the log, so it is edited in one
    // place rather than two.
    prompt = prompt.filter((m) => m.role !== "system");

    // The question goes in the input box rather than into the log. Loading a
    // row used to put the whole prompt straight into the conversation, which
    // read as though the row had already been sent -- and with nothing left to
    // type, there was then no way to ask the model to answer it at all. The
    // box is also the point: the useful thing to do with a held-out example is
    // to change one word of it and see whether the answer survives, and a
    // message you can edit before sending is how you do that.
    const last = prompt[prompt.length - 1];
    if (last && last.role === "user" && !(last.tool_calls || []).length) {
      prompt.pop();
      box.value = last.content || "";
    } else {
      // A row that ends on a tool result or a call: there is no question to
      // type, so "Let it answer" carries it on from where it stands.
      box.value = "";
    }
    turns = prompt;

    live = null;
    statusEl.textContent = "";
    drawTrial();
    paint();
    box.focus();
    box.setSelectionRange(box.value.length, box.value.length);
  }

  // -------------------------------------------------------------- wiring
  // The name a run was given when it was created is a guess made from the
  // model and the dataset. Renaming it here saves a trip to its training page.
  on(mount, "click", "#renameRun", () => {
    inlineRename($("#runTitle", mount), async (name) => {
      await api.renameJob(run.id, name);
      run.name = name;
    });
  });
  sendBtn.addEventListener("click", send);
  askBtn.addEventListener("click", () => { if (turns.length) ask(); });
  box.addEventListener("keydown", (e) => {
    // Enter sends, Shift+Enter makes a new line — the convention every chat
    // interface uses, so nobody has to be told.
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
  });
  stopBtn.addEventListener("click", async () => {
    if (requestId) await api.chatCancel(requestId).catch(() => {});
  });

  on(mount, "click", "#thinkBtn", (_e, t) => {
    think = !think;
    t.textContent = think ? "Reasoning: on" : "Reasoning: off";
    t.classList.toggle("btn-primary", think);
  });
  on(mount, "click", "#resetChat", () => {
    turns = [];
    tools = [];
    sample = null;
    live = null;
    statusEl.textContent = "";
    drawTrial();
    paint();
    box.focus();
  });
  on(mount, "click", "#resetSystem", () => {
    if (systemBox) systemBox.value = run.system_prompt || "";
    toast("Restored the system prompt from the training data.", "ok");
  });
  on(mount, "click", "#clearSystem", () => { if (systemBox) systemBox.value = ""; });

  on(mount, "change", "#dsPick", (_e, t) => {
    sourceId = t.value;
    samples = [];
    sample = null;
    const d = datasets.find((x) => x.id === sourceId);
    sourceSplit = d?.splits?.validation ? "validation"
      : (d?.splits?.test ? "test" : "");
    drawTrial();
  });
  on(mount, "change", "#splitPick", (_e, t) => {
    sourceSplit = t.value;
    samples = [];
    drawTrial();
  });
  on(mount, "change", "#autoTools", (_e, t) => { autoTools = t.checked; });
  on(mount, "click", "#loadRow", () => loadRow());
  on(mount, "click", "#prevRow", () => loadRow(at - 1));
  on(mount, "click", "#nextRow", async () => {
    if (at + 1 >= samples.length) {
      // Past the end of the page: fetch the next one rather than stopping.
      const r = await api.datasetConversations(
        sourceId, (sample?.index ?? 0) + 1, 25, sourceSplit).catch(() => null);
      if (r?.rows?.length) { samples = r.rows; at = 0; applyRow(samples[0]); return; }
      toast("That is the last row in this split.", "");
      return;
    }
    loadRow(at + 1);
  });

  // ------------------------------------------------------- the turn menu
  //
  // Every message can be edited, removed or regenerated: the useful thing to
  // do with a held-out example is change one word of it and ask what survives,
  // and that applies to the model's own replies and to a tool's result -- "what
  // would it have said if the tool had returned something else" is a question
  // you can only ask by rewriting the answer.
  //
  // Reached by right-click, or by holding a message down on a touch screen.
  // The controls used to sit beside every turn, which put two buttons on each
  // of twenty messages and made the conversation hard to read for the sake of
  // something used occasionally.
  let menu = null;

  function closeMenu() {
    menu?.remove();
    menu = null;
  }

  /** Edit one turn in place, in a box where the words are. */
  function editTurn(i) {
    const m = turns[i];
    if (!m) return;
    const holder = $(`.turn[data-index="${i}"]`, mount);
    const target = holder?.querySelector(".bubble-text")
      || holder?.querySelector(".tool-result pre")
      || holder?.querySelector(".tool-result summary")
      || holder?.querySelector(".bubble");
    if (!target || target.dataset.editing) return;
    const area = document.createElement("textarea");
    area.className = "mono turn-edit";
    area.rows = Math.min(14, String(bodyOf(m) || m.content || "").split("\n").length + 1);
    area.value = m.role === "tool" ? pretty(m.content) : bodyOf(m);
    const bar = document.createElement("div");
    bar.className = "row";
    bar.style.marginTop = "6px";
    const save = document.createElement("button");
    save.className = "btn-sm btn-primary";
    save.textContent = "Save";
    const cancel = document.createElement("button");
    cancel.className = "btn-sm";
    cancel.textContent = "Cancel";
    save.addEventListener("click", () => { m.content = area.value; paint(); });
    cancel.addEventListener("click", () => paint());
    // Escape gets you out of a box you opened by accident; the conversation is
    // redrawn from `turns`, which the edit has not touched.
    area.addEventListener("keydown", (e) => {
      if (e.key === "Escape") { e.preventDefault(); paint(); }
    });
    bar.append(save, cancel);
    target.dataset.editing = "1";
    target.replaceWith(area);
    area.after(bar);
    area.focus();
    area.setSelectionRange(area.value.length, area.value.length);
  }

  /** Ask again from this point, throwing away what came after it. */
  function regenerate(i) {
    if (requestId) return;
    // Regenerating the model's turn replaces it. Regenerating a turn somebody
    // else took means "answer this again", so that turn stays and everything
    // after it goes.
    turns = turns.slice(0, turns[i]?.role === "assistant" ? i : i + 1);
    paint();
    if (turns.length) ask();
  }

  function openMenu(i, x, y) {
    closeMenu();
    const m = turns[i];
    if (!m) return;
    menu = document.createElement("div");
    menu.className = "ctx-menu";
    menu.innerHTML = html`
      <button data-act="edit">Edit</button>
      <button data-act="regen" ${requestId ? "disabled" : ""}>${
        m.role === "assistant" ? "Regenerate" : "Answer again"}</button>
      <button data-act="remove" class="danger">Remove</button>`;
    document.body.append(menu);
    // Placed at the pointer, then pulled back inside the window -- a menu
    // opened near the bottom edge otherwise opens where it cannot be read.
    const box = menu.getBoundingClientRect();
    menu.style.left = `${Math.min(x, window.innerWidth - box.width - 8)}px`;
    menu.style.top = `${Math.min(y, window.innerHeight - box.height - 8)}px`;
    menu.addEventListener("click", (e) => {
      const act = e.target.closest("button")?.dataset.act;
      closeMenu();
      if (act === "edit") editTurn(i);
      else if (act === "remove") { turns.splice(i, 1); paint(); }
      else if (act === "regen") regenerate(i);
    });
    $("button", menu)?.focus();
  }

  const turnAt = (target) => {
    const el = target?.closest?.(".turn[data-index]");
    return el && log.contains(el) ? +el.dataset.index : null;
  };

  log.addEventListener("contextmenu", (e) => {
    const i = turnAt(e.target);
    if (i === null) return;
    e.preventDefault();
    openMenu(i, e.clientX, e.clientY);
  });

  // Touch has no right-click, so the same menu is held open. Cancelled by any
  // movement, because a hold that turns into a scroll was a scroll.
  let held = null;
  log.addEventListener("touchstart", (e) => {
    const i = turnAt(e.target);
    if (i === null || e.touches.length !== 1) return;
    const spot = e.touches[0];
    held = setTimeout(() => {
      held = null;
      openMenu(i, spot.clientX, spot.clientY);
    }, 500);
  }, { passive: true });
  const dropHold = () => { clearTimeout(held); held = null; };
  log.addEventListener("touchmove", dropHold, { passive: true });
  log.addEventListener("touchend", dropHold);
  log.addEventListener("touchcancel", dropHold);
  // A menu that outlives what it was opened on is a menu pointing at nothing.
  log.addEventListener("scroll", closeMenu, { passive: true });
  const menuEscape = (e) => { if (e.key === "Escape") closeMenu(); };
  document.addEventListener("click", closeMenu);
  document.addEventListener("keydown", menuEscape);

  // Answering a call by hand, when the row recorded no result for it or you
  // want to see what a different result would do.
  on(mount, "click", "[data-answer]", (_e, t) => {
    const last = turns[turns.length - 1];
    const call = (last?.tool_calls || []).find((c) => c.id === t.dataset.answer);
    if (!call) return;
    const box2 = document.createElement("div");
    box2.innerHTML = html`
      <div class="field" style="margin-top:6px">
        <label>What ${call.function.name} returns</label>
        <textarea class="mono" rows="4" id="handResult">${
          recordedResult(call) || "{}"}</textarea>
      </div>
      <button class="btn-sm btn-primary" id="sendResult">Send it back</button>`;
    t.replaceWith(box2);
    $("#sendResult", box2).addEventListener("click", () => {
      returnResult(call, $("#handResult", box2).value);
    });
  });

  const unsub = events.subscribe((msg) => {
    if (!String(msg.type || "").startsWith("generate_")) return;
    if (!requestId) {
      // Our own POST has not returned yet; hold it until we can tell.
      if (live) early.push(msg);
      return;
    }
    if (msg.request_id !== requestId) return;
    handle(msg);
  });

  function handle(msg) {
    if (msg.type === "generate_status") {
      statusEl.textContent = msg.status;
    } else if (msg.type === "generate_delta") {
      if (!live) return;
      // Older runners send no channel at all; their text is the answer, which
      // is what it always was.
      const channel = msg.channel === "reasoning" ? "reasoning" : "content";
      const first = !live[channel];
      live[channel] += msg.delta;
      statusEl.textContent = "";
      // The first character of a channel is the one that needs the panel or
      // the bubble built for it; the rest just lengthen what is there.
      if (first) paint(); else grow(channel);
    } else if (msg.type === "generate_done") {
      const calls = msg.tool_calls || [];
      const reply = {
        role: "assistant",
        content: typeof msg.text === "string" ? msg.text : (live?.content || ""),
        reasoning: msg.reasoning || live?.reasoning || "",
        tool_calls: calls,
        // Cut off rather than finished. Kept on the message so the note stays
        // with the reply it describes when turns above it are edited away.
        stopped_at_limit: msg.stop_reason === "length",
      };
      const empty = !reply.content && !calls.length && !reply.reasoning;
      requestId = null;
      live = null;
      sendBtn.disabled = false;
      stopBtn.hidden = true;
      if (empty) {
        toast("It produced nothing — try a different opening, or a longer "
              + "length limit.", "");
      } else {
        turns.push(reply);
      }
      const peek = $("#promptPeek", mount);
      if (peek && msg.prompt_preview) peek.textContent = msg.prompt_preview;
      statusEl.textContent = msg.tokens
        ? `${msg.tokens} tokens · ${msg.tokens_per_sec}/s` : "";
      paint();

      // The model asked for a tool. If this row recorded what that tool
      // returned, hand it back and let the exchange run on -- otherwise the
      // conversation stops at the first call and a multi-step model can never
      // be seen doing the thing it was trained to do.
      if (calls.length && autoTools && sample) {
        const answered = calls.map(recordedResult);
        if (answered.every((a) => a !== null)) {
          calls.forEach((c, i) => turns.push({
            role: "tool", tool_call_id: c.id, name: c.function.name,
            content: answered[i] }));
          paint();
          ask();
        } else {
          statusEl.textContent = "It called a tool this row has no recorded "
            + "result for. Type one in to carry on.";
        }
      }
    } else if (msg.type === "generate_error") {
      live = null;
      requestId = null;
      sendBtn.disabled = false;
      stopBtn.hidden = true;
      toast(msg.error, "err");
      statusEl.textContent = "";
      paint();
    }
  }

  // A run made before the prompt was recorded, or made through the API, still
  // has one sitting in the data it trained on. Fetched in the background so
  // the page is usable immediately and the prompt appears when it arrives.
  if (ui.system && !run.system_prompt) {
    api.jobSystemPrompt(run.id).then((r) => {
      if (!r.system_prompt || !systemBox || systemBox.value.trim()) return;
      run.system_prompt = r.system_prompt;
      systemBox.value = r.system_prompt;
      const sum = $("#sysBox", mount)?.querySelector("summary");
      if (sum) {
        sum.innerHTML = "System prompt "
          + '<span class="badge badge-accent">recovered from your training data</span>';
      }
      toast("Found the system prompt this model was trained with.", "ok");
    }).catch(() => {});
  }

  // Every dataset this account can see, so a row can be tried from somewhere
  // other than what the model trained on -- which is the honest test when the
  // training data and the thing you actually want it to do are not the same.
  api.datasets().then((rows) => {
    datasets = rows.map((d) => ({ id: d.id, name: d.name, splits: d.splits || {} }));
    if (!sourceId && datasets.length) sourceId = datasets[0].id;
    drawTrial();
  }).catch(() => {});

  drawTrial();
  paint();
  box.focus();
  return () => {
    unsub();
    // The menu lives on <body>, so it outlives this view unless it is taken
    // down with it -- along with the two listeners that dismiss it.
    closeMenu();
    document.removeEventListener("click", closeMenu);
    document.removeEventListener("keydown", menuEscape);
    if (requestId) api.chatCancel(requestId).catch(() => {});
  };
}
