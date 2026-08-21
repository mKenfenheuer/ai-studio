import { api, events } from "../api.js";
import { html, raw, esc, $, on, fmtAgo, toast } from "../util.js";

// Talking to what you trained.
//
// The two kinds of result behave completely differently, and pretending
// otherwise is how people conclude their training failed. A fine-tune has
// learned an instruction shape and answers questions. A model trained from
// scratch is a base language model: it continues text and has never seen a
// question in its life. The interface says which one it is holding, and asks
// for the right kind of input.
const MODE_UI = {
  instruct: {
    icon: "💬",
    title: "Ask it something",
    placeholder: "Ask a question, or give it a task…",
    note: "This model was fine-tuned on instructions, so ask it something "
        + "directly. Your message is wrapped in the same template it was "
        + "trained with.",
  },
  continue: {
    icon: "✍️",
    title: "Write an opening and it will continue",
    placeholder: "Once upon a time, there was a little…",
    note: "This is a base model: it continues text rather than answering "
        + "questions. Give it the first few words of something and see where "
        + "it goes.",
  },
};

export async function playView(mount, [jobId]) {
  const runs = await api.playground();
  if (!jobId) return pickerView(mount, runs);

  const run = runs.find((r) => r.id === jobId);
  if (!run) {
    mount.innerHTML = html`
      <div class="card empty"><div class="big">🤷</div>
        <h2>That run has nothing to try</h2>
        <p class="muted">It may still be training, or it may not have produced
          a model.</p>
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
  const ui = MODE_UI[r.mode] || MODE_UI.continue;
  return html`
    <a class="pick" href="#/play/${r.id}" style="text-decoration:none">
      <span class="t"><span style="font-size:17px">${ui.icon}</span>${r.name}</span>
      <span class="d">${r.kind === "pretrain_llm"
        ? `Built from scratch${r.size ? " · " + r.size : ""} on ${r.dataset}`
        : `${r.base_model} fine-tuned on ${r.dataset}`}</span>
      <span class="row" style="gap:6px;flex-wrap:wrap">
        <span class="badge ${r.kind === "pretrain_llm" ? "badge-ok" : "badge-accent"}">
          ${r.kind === "pretrain_llm" ? "your own model" : "fine-tune"}</span>
        <span class="badge">${esc(fmtAgo(r.finished_at))}</span>
      </span>
    </a>`;
}

// ------------------------------------------------------------------ chat
function chatView(mount, run, runs) {
  const ui = MODE_UI[run.mode] || MODE_UI.continue;
  const turns = [];
  let requestId = null;
  let pending = null;      // the bubble currently being written into

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

    <div class="card chat">
      <div class="chat-log" id="chatLog">
        <div class="chat-empty" id="chatEmpty">
          Nothing said yet. ${ui.title}.
        </div>
      </div>
      <div class="tiny muted" id="chatStatus"></div>
      <div class="chat-input">
        <textarea id="chatBox" rows="2" placeholder="${ui.placeholder}"></textarea>
        <button class="btn-primary" id="sendBtn">Send</button>
        <button class="btn-danger" id="stopBtn" hidden>Stop</button>
      </div>
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

  const scroll = () => { log.scrollTop = log.scrollHeight; };

  const bubble = (cls, text) => {
    $("#chatEmpty", mount)?.remove();
    const d = document.createElement("div");
    d.className = `bubble ${cls}`;
    d.textContent = text;
    log.appendChild(d);
    scroll();
    return d;
  };

  const finish = () => {
    requestId = null;
    pending?.classList.remove("pending");
    pending = null;
    sendBtn.disabled = false;
    stopBtn.hidden = true;
    statusEl.textContent = "";
    box.focus();
  };

  const send = async () => {
    const text = box.value.trim();
    if (!text || requestId) return;
    bubble("me", text);
    box.value = "";
    sendBtn.disabled = true;
    // Created empty and filled by the stream, so the first token appears the
    // moment the runner produces it rather than after the whole reply.
    pending = bubble("it pending", "");
    statusEl.textContent = "Waking the model up…";
    try {
      const r = await api.chat(run.id, {
        prompt: text,
        temperature: parseFloat($("#temp", mount).value) || 0.8,
        max_new_tokens: parseInt($("#maxTok", mount).value, 10) || 200,
      });
      requestId = r.request_id;
      stopBtn.hidden = false;
      statusEl.textContent = `Running on ${r.runner}…`;
    } catch (e) {
      pending.classList.remove("pending");
      pending.classList.add("err");
      pending.textContent = e.message;
      finish();
    }
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

  const unsub = events.subscribe((msg) => {
    if (!requestId || msg.request_id !== requestId) return;
    if (msg.type === "generate_status") {
      statusEl.textContent = msg.status;
    } else if (msg.type === "generate_delta") {
      if (!pending) return;
      const atBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 60;
      pending.textContent += msg.delta;
      statusEl.textContent = "";
      if (atBottom) scroll();
    } else if (msg.type === "generate_done") {
      if (pending && !pending.textContent) {
        pending.textContent = "(it produced nothing — try a different opening, "
          + "or a longer length limit)";
        pending.classList.add("muted");
      }
      statusEl.textContent = msg.tokens
        ? `${msg.tokens} tokens · ${msg.tokens_per_sec}/s` : "";
      const keep = statusEl.textContent;
      finish();
      statusEl.textContent = keep;
    } else if (msg.type === "generate_error") {
      if (pending) {
        pending.textContent = msg.error;
        pending.classList.add("muted");
      }
      toast(msg.error, "err");
      finish();
    }
  });

  box.focus();
  return () => { unsub(); if (requestId) api.chatCancel(requestId).catch(() => {}); };
}
