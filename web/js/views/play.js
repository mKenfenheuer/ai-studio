import { api, events } from "../api.js";
import { html, raw, esc, $, on, fmtAgo, toast } from "../util.js";

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
        ${raw(r.stopped_early
          ? `<span class="badge badge-warn" title="This run was stopped before
               it finished, so the model had less practice than planned."
             >stopped early</span>` : "")}
        <span class="badge">${esc(fmtAgo(r.finished_at))}</span>
      </span>
    </a>`;
}

// ------------------------------------------------------------------ chat
function chatView(mount, run, runs) {
  const ui = styleOf(run);
  // The conversation, in the same shape the model was trained on.
  let turns = [];
  let requestId = null;
  let pending = null;      // the bubble currently being written into
  let early = [];          // events that beat their own POST response
  // Only offered for a run that was actually taught to reason; a model that
  // never saw a reasoning block just writes prose inside one.
  let think = !!run.reasoning;

  mount.innerHTML = html`
    <div class="page-head">
      <a href="#/play" class="tiny">← All finished runs</a>
      <div class="row-between" style="flex-wrap:wrap;gap:8px;margin-top:6px">
        <h1 style="margin:0">${run.name}</h1>
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

    ${raw(ui.system ? html`
      <details class="adv" id="sysBox" ${run.system_prompt ? "" : ""}>
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
            <input type="number" id="maxTok" value="200" step="10" min="10" max="512">
            <div class="hint">Maximum tokens to write.</div>
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
  const stopBtn = $("#stopBtn", mount);
  const statusEl = $("#chatStatus", mount);
  const systemBox = $("#systemBox", mount);

  const scroll = () => { log.scrollTop = log.scrollHeight; };

  const bubble = (cls, text, role) => {
    $("#chatEmpty", mount)?.remove();
    const d = document.createElement("div");
    d.className = `bubble ${cls}`;
    if (role) d.dataset.role = role;
    d.textContent = text;
    log.appendChild(d);
    scroll();
    return d;
  };

  const paintTurns = () => {
    const el = $("#turnCount", mount);
    if (el) el.textContent = turns.length
      ? `${turns.length} message${turns.length > 1 ? "s" : ""} in context` : "";
  };

  const finish = () => {
    requestId = null;
    early = [];
    pending?.classList.remove("pending");
    pending = null;
    sendBtn.disabled = false;
    stopBtn.hidden = true;
    box.focus();
  };

  const send = async () => {
    const text = box.value.trim();
    if (!text || requestId) return;

    // A model that never learned to follow a conversation is not given one:
    // each instruction stands alone, exactly as it did in training.
    if (!ui.multiturn) turns = [];
    turns.push({ role: "user", content: text });

    bubble("me", text, "user");
    box.value = "";
    sendBtn.disabled = true;
    pending = bubble("it pending", "", "assistant");
    statusEl.textContent = "Waking the model up…";
    early = [];
    try {
      const r = await api.chat(run.id, {
        messages: turns,
        system: systemBox ? systemBox.value : "",
        temperature: parseFloat($("#temp", mount).value) || 0.8,
        max_new_tokens: parseInt($("#maxTok", mount).value, 10) || 200,
        reasoning: think,
      });
      requestId = r.request_id;
      stopBtn.hidden = false;
      statusEl.textContent = `Running on ${r.runner}…`;
      const buffered = early.filter((m) => m.request_id === requestId);
      early = [];
      buffered.forEach(handle);
    } catch (e) {
      turns.pop();
      pending.classList.remove("pending");
      pending.textContent = e.message;
      pending.classList.add("muted");
      finish();
      statusEl.textContent = "";
    }
    paintTurns();
  };

  sendBtn.addEventListener("click", send);
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
    log.innerHTML = `<div class="chat-empty" id="chatEmpty">Nothing said yet. ${
      esc(ui.title)}.</div>`;
    statusEl.textContent = "";
    paintTurns();
    box.focus();
  });
  on(mount, "click", "#resetSystem", () => {
    if (systemBox) systemBox.value = run.system_prompt || "";
    toast("Restored the system prompt from the training data.", "ok");
  });
  on(mount, "click", "#clearSystem", () => { if (systemBox) systemBox.value = ""; });

  const unsub = events.subscribe((msg) => {
    if (!String(msg.type || "").startsWith("generate_")) return;
    if (!requestId) {
      // Our own POST has not returned yet; hold it until we can tell.
      if (pending) early.push(msg);
      return;
    }
    if (msg.request_id !== requestId) return;
    handle(msg);
  });

  function handle(msg) {
    if (msg.type === "generate_status") {
      statusEl.textContent = msg.status;
    } else if (msg.type === "generate_delta") {
      if (!pending) return;
      const atBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 60;
      pending.textContent += msg.delta;
      statusEl.textContent = "";
      if (atBottom) scroll();
    } else if (msg.type === "generate_done") {
      // The runner's final text is authoritative. Normally it matches what the
      // deltas built, but if a stop sequence trimmed the tail this is where the
      // bubble catches up.
      if (pending && typeof msg.text === "string"
          && msg.text !== pending.textContent) {
        pending.textContent = msg.text;
      }
      // Its working, kept apart from its answer and folded away by default:
      // the reasoning is usually longer than the reply and rarely the thing
      // you are reading for.
      if (pending && msg.reasoning) {
        const box = document.createElement("details");
        box.className = "reasoning";
        box.innerHTML = `<summary>Its reasoning (${
          msg.reasoning.length} characters)</summary>`;
        const body = document.createElement("p");
        body.className = "txt";
        body.textContent = msg.reasoning;
        box.appendChild(body);
        pending.parentNode.insertBefore(box, pending);
      }
      const reply = pending ? pending.textContent : "";
      if (pending && !reply) {
        pending.textContent = "(it produced nothing — try a different opening, "
          + "or a longer length limit)";
        pending.classList.add("muted");
      } else {
        turns.push({ role: "assistant", content: reply });
      }
      const peek = $("#promptPeek", mount);
      if (peek && msg.prompt_preview) peek.textContent = msg.prompt_preview;
      const stats = msg.tokens
        ? `${msg.tokens} tokens · ${msg.tokens_per_sec}/s` : "";
      finish();
      statusEl.textContent = stats;
      paintTurns();
    } else if (msg.type === "generate_error") {
      turns.pop();                       // the user turn never got a reply
      if (pending) {
        pending.textContent = msg.error;
        pending.classList.add("muted");
      }
      toast(msg.error, "err");
      finish();
      statusEl.textContent = "";
      paintTurns();
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

  box.focus();
  return () => { unsub(); if (requestId) api.chatCancel(requestId).catch(() => {}); };
}
