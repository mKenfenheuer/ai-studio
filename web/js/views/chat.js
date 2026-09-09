/**
 * Talking to the models, for people who are only here to talk to them.
 *
 * The playground is a workbench: it lists every run, compares two of them side
 * by side, exposes temperature and top-p and the system prompt, and assumes
 * you know what a fine-tune is. That is right for the person who trained the
 * model and wrong for the person the model was trained *for* — the colleague
 * who has been told "ask the assistant" and has no business seeing four
 * hundred runs, somebody's dataset, or the button that deletes a machine.
 *
 * So this is the other door. It shows the models that somebody deliberately
 * published under a name, and nothing else. A `chat` account gets this page
 * and no other; everybody else can reach it too, because a member who just
 * wants to ask a question should not have to drive the workbench either.
 *
 * It talks to `/v1/chat/completions` — the same OpenAI-compatible endpoint any
 * outside client uses, with the session cookie instead of a key. That is worth
 * being deliberate about: it means this page cannot drift from what everything
 * else gets, the usage is counted the same way, and a bug somebody hits here
 * is a bug an integration would have hit too.
 */
import { api } from "../api.js";
import { html, raw, esc, $, on, toast, fmtBytes } from "../util.js";
import { reasoningBlock } from "../conversation.js";
import { renderMarkdown } from "../markdown.js";

// Files whose content is text and can simply be put in the message. Anything
// else is refused with a sentence rather than uploaded and silently ignored:
// a model that was handed a .docx as bytes answers about nothing, and the
// person cannot tell that is what happened.
const TEXTUAL = /\.(txt|md|markdown|csv|tsv|json|jsonl|ya?ml|log|ini|conf|toml|xml|html?|css|js|ts|jsx|tsx|py|rb|go|rs|java|c|h|cpp|hpp|cs|sh|sql|r|swift|kt|php|pl)$/i;

// How much of one file to include. A model has a context window and a person
// dropping a 40 MB log has not thought about it; cutting with a line that
// says so is better than a request that fails on length.
const MAX_FILE_CHARS = 60000;

export async function chatView(mount) {
  let names = [];
  let model = localStorage.getItem("aistudio.chatModel") || "";
  // The conversation as the API wants it: {role, content}. Reasoning is kept
  // beside it for drawing and deliberately never sent back — a model given its
  // own thinking as context behaves differently from one that was not, and
  // the API's own clients do not send it either.
  let turns = [];
  let attached = [];
  let busy = false;
  let stop = null;

  try {
    names = await api.registeredModels();
  } catch (e) {
    mount.innerHTML = `<div class="callout callout-err"><strong>Could not load
      the models</strong>${esc(e.message)}</div>`;
    return null;
  }
  // A name pointing at a run that is gone, or one this account cannot see, is
  // not something to offer: choosing it produces a 404 on the first message.
  names = names.filter((n) => n.visible && !n.job_gone);
  if (!names.some((n) => n.alias === model)) {
    model = (names.find((n) => n.stage === "production") || names[0] || {}).alias || "";
  }

  draw();

  function draw() {
    mount.innerHTML = html`
      <div class="chatpage">
        <header class="chathead">
          <div>
            <h1>Assistant</h1>
            <p class="muted tiny">${names.length
              ? "Ask a question. The answer comes from a model trained in this studio."
              : "No model has been published under a name yet."}</p>
          </div>
          ${raw(names.length > 1 ? html`
            <label class="chatpick">
              <span class="muted tiny">Model</span>
              <select id="chatModel">${raw(names.map((n) => html`
                <option value="${n.alias}"${n.alias === model ? " selected" : ""}>${
                  n.notes || n.alias}${n.stage === "production" ? "" : ` · ${n.stage || "unlabelled"}`}</option>`).join(""))}</select>
            </label>` : "")}
        </header>

        <div class="chatlog" id="chatLog">${raw(
          turns.length ? turns.map(turnHtml).join("") : welcome(names))}</div>

        <form class="chatbar" id="chatForm" ${raw(names.length ? "" : "hidden")}>
          <div class="chatfiles" id="chatFiles">${raw(attached.map(fileChip).join(""))}</div>
          <div class="chatrow">
            <textarea id="chatText" rows="1" placeholder="Ask something…"
                      aria-label="Your message"></textarea>
            <label class="icon-btn" title="Attach a text file">
              <input type="file" id="chatFile" multiple hidden>📎</label>
            <button type="submit" class="btn btn-primary" id="chatSend"
                    ${raw(busy ? "disabled" : "")}>${busy ? "…" : "Send"}</button>
            ${raw(busy ? `<button type="button" class="btn" id="chatStop">Stop</button>` : "")}
          </div>
          <p class="muted tiny chatfoot">Models can be wrong. Check anything
            that matters.</p>
        </form>
      </div>`;
    wire();
    const log = $("#chatLog", mount);
    if (log) log.scrollTop = log.scrollHeight;
  }

  function wire() {
    on(mount, "change", "#chatModel", (_e, t) => {
      model = t.value;
      localStorage.setItem("aistudio.chatModel", model);
    });
    on(mount, "change", "#chatFile", (_e, t) => { addFiles(t.files); t.value = ""; });
    on(mount, "click", "[data-drop-file]", (_e, t) => {
      attached.splice(Number(t.dataset.dropFile), 1);
      $("#chatFiles", mount).innerHTML = attached.map(fileChip).join("");
    });
    on(mount, "click", "#chatStop", () => { if (stop) stop(); });
    on(mount, "submit", "#chatForm", (e) => { e.preventDefault(); send(); });

    // Enter sends, shift-enter makes a new line: what every chat does, and
    // what the muscle memory of anybody arriving here already expects.
    const box = $("#chatText", mount);
    if (box) {
      box.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
      });
      // Grows with what is being written, to a point. A one-line box for a
      // paragraph makes people write less than they meant to.
      box.addEventListener("input", () => {
        box.style.height = "auto";
        box.style.height = `${Math.min(box.scrollHeight, 200)}px`;
      });
      box.focus();
    }
  }

  async function addFiles(files) {
    for (const f of [...files]) {
      if (!TEXTUAL.test(f.name)) {
        toast(`${f.name} is not a text file, so the model could not read it. `
              + "Paste the part that matters instead.", "err");
        continue;
      }
      let text = await f.text();
      let cut = false;
      if (text.length > MAX_FILE_CHARS) {
        text = text.slice(0, MAX_FILE_CHARS);
        cut = true;
      }
      attached.push({ name: f.name, size: f.size, text, cut });
    }
    const box = $("#chatFiles", mount);
    if (box) box.innerHTML = attached.map(fileChip).join("");
  }

  /** The message actually sent: what was typed, with each file below it. */
  function compose(typed) {
    if (!attached.length) return typed;
    const blocks = attached.map((f) =>
      `--- ${f.name} ---\n${f.text}${f.cut
        ? "\n--- truncated: only the first part of this file was included ---"
        : ""}`).join("\n\n");
    return `${typed}\n\n${blocks}`;
  }

  async function send() {
    if (busy) return;
    const box = $("#chatText", mount);
    const typed = (box?.value || "").trim();
    if (!typed) return;
    if (!model) return toast("There is no model to talk to.", "err");

    turns.push({
      role: "user",
      content: compose(typed),
      // What to *show*, which is not what is sent: the file goes to the model
      // in full and appears here as a chip. A screen full of pasted CSV is not
      // a conversation anybody can read afterwards.
      shown: typed,
      files: attached.map((f) => ({ name: f.name, size: f.size })),
    });
    attached = [];
    busy = true;
    draw();

    const live = { reasoning: "", content: "" };
    turns.push({ role: "assistant", content: "", reasoning: "", live: true });
    const log = $("#chatLog", mount);
    log.insertAdjacentHTML("beforeend", turnHtml(turns[turns.length - 1],
                                                 turns.length - 1));
    log.scrollTop = log.scrollHeight;

    const controller = new AbortController();
    stop = () => controller.abort();
    try {
      await stream(controller.signal, live, () => paint(live, log));
      const last = turns[turns.length - 1];
      last.content = live.content;
      last.reasoning = live.reasoning;
      last.live = false;
      if (!last.content && !last.reasoning) {
        last.content = "_The model returned nothing._";
      }
    } catch (e) {
      const last = turns[turns.length - 1];
      last.live = false;
      last.failed = true;
      last.content = e.name === "AbortError"
        ? "_Stopped._"
        : `Something went wrong: ${e.message}`;
    } finally {
      busy = false;
      stop = null;
      draw();
    }
  }

  /**
   * Read the server-sent stream, token by token.
   *
   * Written by hand rather than with EventSource, which cannot POST. The
   * framing is the same either way: `data: {...}` lines, blank line between
   * events, `data: [DONE]` at the end.
   */
  async function stream(signal, live, onDelta) {
    const res = await fetch("/v1/chat/completions", {
      method: "POST",
      signal,
      credentials: "same-origin",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model,
        stream: true,
        messages: turns
          .filter((t) => !t.live && !t.failed && t.content)
          .map((t) => ({ role: t.role, content: t.content })),
      }),
    });
    if (!res.ok) {
      let why = `${res.status} ${res.statusText}`;
      try { why = (await res.json()).error?.message || why; } catch { /* not JSON */ }
      throw new Error(why);
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      // Events are separated by a blank line; a partial one stays in the
      // buffer until the rest of it arrives. Splitting on "\n" alone would
      // hand half a JSON object to JSON.parse on a slow connection.
      const frames = buffer.split("\n\n");
      buffer = frames.pop() || "";
      for (const evt of frames) {
        for (const line of evt.split("\n")) {
          if (!line.startsWith("data:")) continue;
          const payload = line.slice(5).trim();
          if (!payload || payload === "[DONE]") continue;
          let frame;
          try { frame = JSON.parse(payload); } catch { continue; }
          const delta = frame.choices?.[0]?.delta || {};
          // Both spellings, because the endpoint sends both and a client that
          // reads only one gets a silent gap where the thinking was.
          const think = delta.reasoning_content ?? delta.reasoning;
          if (think) live.reasoning += think;
          if (delta.content) live.content += delta.content;
          if (think || delta.content) onDelta();
        }
      }
    }
  }

  /**
   * Repaint the turn in flight.
   *
   * Only that turn: redrawing the page on every token would throw away the
   * reader's scroll position sixty times a second, and a reader who has
   * scrolled up to check something said earlier would be dragged back down.
   */
  function paint(live, log) {
    const node = $("#liveTurn", mount);
    if (!node) return;
    node.innerHTML = bodyHtml({ ...live, live: true });
    const atBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 80;
    if (atBottom) log.scrollTop = log.scrollHeight;
  }

  return () => { if (stop) stop(); };
}

// ---------------------------------------------------------------------------

function welcome(names) {
  if (!names.length) {
    return html`
      <div class="empty">
        <div class="big" aria-hidden="true">💬</div>
        <h3>Nothing has been published to talk to</h3>
        <p class="muted">A model becomes available here once somebody serves
          it under a name. Ask whoever runs this studio.</p>
      </div>`;
  }
  return html`
    <div class="empty">
      <div class="big" aria-hidden="true">💬</div>
      <h3>Ask a question</h3>
      <p class="muted">The answer comes from a model trained in this studio, on
        this organisation's own data. You can attach a text file — a log, a
        CSV, some notes — and ask about what is in it.</p>
    </div>`;
}

const fileChip = (f, i) => html`
  <span class="chip">📄 ${f.name}
    <span class="muted tiny">${fmtBytes(f.size)}</span>
    <button type="button" class="chip-x" data-drop-file="${i}"
            aria-label="Remove ${f.name}">✕</button></span>`;

function turnHtml(t, i) {
  if (t.role === "user") {
    return html`
      <div class="chatturn me">
        <div class="bubble">${raw(esc(t.shown ?? t.content).replace(/\n/g, "<br>"))}
          ${raw((t.files || []).length ? `<div class="chatattach">${
            t.files.map((f) => `<span class="chip">📄 ${esc(f.name)}</span>`).join("")
          }</div>` : "")}</div>
      </div>`;
  }
  return html`
    <div class="chatturn it${t.failed ? " failed" : ""}">
      <span class="chatmark" aria-hidden="true">◆</span>
      <div class="chatbody" ${raw(t.live ? 'id="liveTurn"' : "")}>${raw(bodyHtml(t))}</div>
    </div>`;
}

function bodyHtml(t) {
  const reasoning = (t.reasoning || "").trim();
  const answer = (t.content || "").trim();
  return (reasoning ? reasoningBlock(reasoning, { live: t.live && !answer }) : "")
    + (answer
      ? `<div class="chatanswer">${renderMarkdown(answer)}</div>`
      : t.live && !reasoning ? `<div class="chatwait">Thinking…</div>` : "");
}
