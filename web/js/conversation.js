// One way to draw a conversation, used everywhere one is shown.
//
// A canonical row is `messages`, `tools`, `meta` — and printed as JSON it is
// unreadable. The punctuation outweighs the content, a tool call's arguments
// are an escaped string inside a string, and the one thing you actually want
// to know when looking at training data — who says what, in what order — is
// the thing the shape hides hardest.
//
// So a conversation is drawn as a conversation. Each turn is one unit -- who
// spoke, what they were thinking, what they said -- rather than a run of loose
// siblings, which is what let a model's reasoning float above its answer
// belonging to nothing. A tool call is shown as a call with its arguments laid
// out, and a tool's answer is not a bubble at all, because nobody said it.
//
// The two speakers are drawn differently on purpose. A person's turn is a
// bubble, kept narrow, because questions are short. The model's is a plain
// block at full width, because that is the text anyone actually reads.
//
// This lives outside `views/` because two very different pages need exactly
// the same picture and they must not drift: the playground, where the turns
// are live, and the dataset workbench, where they are rows on disk. A row you
// read in the workbench and the same row loaded into the playground should
// look identical, because they are the same conversation.

import { html, raw, esc } from "./util.js";

// What each role is called above its bubble, and which side it sits on. The
// role is named rather than left to the alignment: a conversation with system
// and tool turns has more than two speakers, and two sides cannot say which
// of four is talking.
const ROLE = {
  user:      { label: "You",       side: "me",   mark: "" },
  assistant: { label: "Assistant", side: "it",   mark: "◆" },
  system:    { label: "System",    side: "note", mark: "" },
  developer: { label: "Developer", side: "note", mark: "" },
  tool:      { label: "Tool",      side: "note", mark: "" },
};

/** Pretty-print JSON when it is JSON, leave it alone when it is not.
 *
 *  Tool arguments are the thing you most want to read at a glance and the
 *  thing most reliably written as one unbroken line.
 */
export function pretty(text) {
  const s = String(text ?? "").trim();
  if (!s.startsWith("{") && !s.startsWith("[")) return s;
  try { return JSON.stringify(JSON.parse(s), null, 2); } catch { return s; }
}

/** `get_order(order_id="12345")` — a call on one line, when it fits. */
function callSignature(call) {
  const args = call?.function?.arguments;
  let parsed = null;
  try { parsed = JSON.parse(args || "{}"); } catch { /* shown in full below */ }
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return "";
  const parts = Object.entries(parsed).map(([k, v]) => `${k}=${JSON.stringify(v)}`);
  const line = parts.join(", ");
  return line.length > 90 ? "" : line;
}

/** One tool call, as a call rather than as syntax. */
function toolCall(call, i, opts) {
  const name = call?.function?.name || "?";
  const args = call?.function?.arguments || "";
  const signature = callSignature(call);
  const broken = call.valid === false;
  const body = pretty(args);
  // A one-line call needs no disclosure triangle; a call with a page of
  // arguments needs one, or a single row of a dataset fills the screen.
  const long = body.includes("\n") || body.length > 90;
  return html`
    <div class="tool-call ${broken ? "bad" : ""}">
      <div class="tool-call-head">
        <span class="tool-call-chip">call</span>
        <span class="tool-call-name">${name}</span>${
          raw(signature && !long
            ? `<span class="tool-call-args mono">(${esc(signature)})</span>` : "")}
        ${raw(broken
          ? `<span class="badge badge-err">arguments are not valid JSON</span>` : "")}
        ${raw(opts.callAction ? opts.callAction(call, i) : "")}
      </div>
      ${raw(!long ? "" : html`
        <details class="tool-call-body">
          <summary>arguments</summary>
          <pre class="mono">${body}</pre>
        </details>`)}
    </div>`;
}

/** The model's working, folded away.
 *
 *  Its own block above the answer rather than a line mixed into it: it is
 *  usually longer than the answer and rarely the thing you are reading for,
 *  but it is the thing you go looking for when the answer is wrong.
 *
 *  `live` means it is still being written, which is worth showing plainly --
 *  a reasoning model can think for a long time before it says anything, and a
 *  blank screen for twenty seconds reads as a hang.
 */
export function reasoningBlock(text, opts = {}) {
  const n = text.length;
  const label = opts.live
    ? `Thinking${n ? ` — ${n.toLocaleString()} characters` : "…"}`
    : `Thought for ${n.toLocaleString()} character${n === 1 ? "" : "s"}`;
  return html`
    <details class="reasoning ${opts.live ? "live" : ""}" ${
      opts.open || opts.live ? "open" : ""}>
      <summary>
        <span class="reasoning-mark" aria-hidden="true"></span>
        <span class="reasoning-label">${label}</span>
      </summary>
      <div class="reasoning-body">${text}</div>
    </details>`;
}

/** A tool result: not a bubble, because nobody said it. */
function toolResult(m, opts) {
  const body = pretty(m.content || "");
  const long = body.includes("\n") || body.length > 160;
  const head = html`<span class="tool-result-name">${m.name || "tool"}</span>
    <span class="tool-result-verb">returned</span>`;
  return html`
    <div class="turn note" data-index="${opts.index}" data-role="tool">
      <div class="tool-result">
        ${raw(long ? html`
          <details>
            <summary>${raw(head)}</summary>
            <pre class="mono">${body}</pre>
          </details>` : html`
          <div class="tool-result-head">${raw(head)}</div>
          <pre class="mono">${body}</pre>`)}
      </div>
      ${raw(opts.footer ? opts.footer(m, opts.index) : "")}
      ${raw(opts.actions ? opts.actions(m, opts.index) : "")}
    </div>`;
}

/** What was actually said, under the label and the working.
 *
 *  A turn that is nothing but a tool call gets no bubble. It used to get one,
 *  and a call already draws its own card, so the screen showed a box inside a
 *  box for every call a model made -- which reads as a mistake because it is.
 */
function body(content, calls, reasoning, per) {
  const called = calls.map((c, n) => toolCall(c, n, per)).join("");
  if (content) {
    return html`<div class="bubble"><div class="bubble-text">${content}</div>${
      raw(called)}</div>`;
  }
  if (called) return called;
  // Nothing at all -- but a turn that reasoned and then stopped has said
  // something, and does not need to be told it is empty.
  return reasoning ? "" : html`<div class="bubble"><span class="muted tiny"
    >(nothing in this turn)</span></div>`;
}

/**
 * A whole conversation as HTML.
 *
 * Every turn is one `.turn` element -- label, working, words and controls
 * together -- rather than a run of loose siblings. They used to be separate,
 * and a model's reasoning floated above its answer belonging to nothing,
 * which is exactly how it read.
 *
 * `opts`:
 *   openReasoning  show the working expanded rather than folded
 *   actions        (message, index) => html overlaid on each turn
 *   footer         (message, index) => html placed under what was said
 *   callAction     (call, index) => html appended beside a tool call
 *   empty          what to show when there are no messages
 */
export function conversationHtml(messages, opts = {}) {
  const list = Array.isArray(messages) ? messages : [];
  if (!list.length) {
    return html`<div class="chat-empty">${
      opts.empty || "This row has no messages in it."}</div>`;
  }
  return list.map((m, i) => {
    const role = String(m.role || "user").toLowerCase();
    const spec = ROLE[role] || { label: role, side: "note", mark: "" };
    const per = { ...opts, index: i };

    if (role === "tool") return toolResult(m, per);

    const calls = m.tool_calls || [];
    const content = (m.content || "").trim();
    const reasoning = (m.reasoning || "").trim();
    // weight 0 means "render this, do not learn from it". Worth showing,
    // because a row that trains on nothing looks identical to one that trains
    // on everything until you are told.
    const muted = m.weight === 0;

    return html`
      <div class="turn ${spec.side} ${muted ? "not-trained" : ""}"
           data-index="${i}" data-role="${role}">
        <div class="turn-who">
          ${raw(spec.mark ? `<span class="turn-mark">${spec.mark}</span>` : "")}
          <span>${spec.label}</span>
          ${raw(muted
            ? `<span class="badge badge-soft" title="weight 0 — rendered as
                 context, never learned from">not trained on</span>` : "")}
        </div>
        ${raw(reasoning
          ? reasoningBlock(reasoning, { open: !!opts.openReasoning }) : "")}
        ${raw(body(content, calls, reasoning, per))}
        ${raw(opts.footer ? opts.footer(m, i) : "")}
        ${raw(opts.actions ? opts.actions(m, i) : "")}
      </div>`;
  }).join("");
}

/** The tools a row declares, as a readable list rather than a schema dump. */
export function toolsHtml(tools) {
  const list = Array.isArray(tools) ? tools : [];
  if (!list.length) return "";
  return html`
    <details class="tools-declared">
      <summary>${list.length} tool${list.length === 1 ? "" : "s"} available to
        the model in this row</summary>
      <div class="tools-list">
        ${raw(list.map((t) => {
          const fn = t.function || t;
          const params = fn.parameters || {};
          const props = params.properties || {};
          const required = new Set(params.required || []);
          const names = Object.keys(props);
          return html`
            <div class="tool-def">
              <div><span class="tool-call-name mono">${fn.name || "?"}</span><span
                class="mono muted">(${names.map((n) =>
                  n + (required.has(n) ? "" : "?")).join(", ")})</span></div>
              ${raw(fn.description
                ? `<div class="muted tiny">${esc(fn.description)}</div>` : "")}
              ${raw(names.length ? html`
                <ul class="tool-params">
                  ${raw(names.map((n) => {
                    const p = props[n] || {};
                    return `<li><span class="mono">${esc(n)}</span>
                      <span class="muted">${esc(p.type || "any")}${
                        required.has(n) ? ", required" : ""}</span>${
                      p.description ? ` — ${esc(p.description)}` : ""}</li>`;
                  }).join(""))}
                </ul>` : "")}
            </div>`;
        }).join(""))}
      </div>
    </details>`;
}

/** Whether this row is worth drawing as a conversation at all. */
export function isConversation(row) {
  return Array.isArray(row?.messages) && row.messages.length > 0
    && row.messages.every((m) => m && typeof m === "object" && m.role);
}
