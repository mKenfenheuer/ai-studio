// One way to draw a conversation, used everywhere one is shown.
//
// A canonical row is `messages`, `tools`, `meta` — and printed as JSON it is
// unreadable. The punctuation outweighs the content, a tool call's arguments
// are an escaped string inside a string, and the one thing you actually want
// to know when looking at training data — who says what, in what order — is
// the thing the shape hides hardest.
//
// So a conversation is drawn as a conversation: bubbles that pick a side by
// role, the model's working folded away beside its answer rather than mixed
// into it, and a tool call shown as a call with its arguments laid out.
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
  user:      { label: "user",      side: "me" },
  assistant: { label: "assistant", side: "it" },
  system:    { label: "system",    side: "note" },
  developer: { label: "developer", side: "note" },
  tool:      { label: "tool",      side: "note" },
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
  const parts = Object.entries(parsed).map(([k, v]) =>
    `${k}=${typeof v === "string" ? JSON.stringify(v) : JSON.stringify(v)}`);
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
        <span class="tool-call-name">⚙ ${name}</span>${
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

/** The model's working, folded away. Usually longer than the answer and
 *  rarely the thing you are reading for. */
function reasoningBlock(text, open) {
  const n = text.length;
  return html`
    <details class="reasoning" ${open ? "open" : ""}>
      <summary>reasoning — ${n.toLocaleString()} character${n === 1 ? "" : "s"}</summary>
      <p class="txt">${text}</p>
    </details>`;
}

/** A tool result: not a bubble, because nobody said it. */
function toolResult(m, opts) {
  const body = pretty(m.content || "");
  const long = body.includes("\n") || body.length > 160;
  const head = html`<span class="tool-result-name">${
    m.name || "tool"}</span> returned`;
  return html`
    <div class="tool-result" data-index="${opts.index}">
      ${raw(long ? html`
        <details>
          <summary>${raw(head)}</summary>
          <pre class="mono">${body}</pre>
        </details>` : html`
        <div class="tool-result-head">${raw(head)}</div>
        <pre class="mono">${body}</pre>`)}
      ${raw(opts.actions ? opts.actions(m, opts.index) : "")}
    </div>`;
}

/**
 * A whole conversation as HTML.
 *
 * `opts`:
 *   openReasoning  show the working expanded rather than folded
 *   actions        (message, index) => html appended inside each bubble
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
    const spec = ROLE[role] || { label: role, side: "note" };
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
      ${raw(reasoning ? reasoningBlock(reasoning, !!opts.openReasoning) : "")}
      <div class="bubble ${spec.side} ${muted ? "not-trained" : ""}"
           data-role="${spec.label}" data-index="${i}">
        ${raw(muted
          ? `<span class="badge badge-soft" title="weight 0 — rendered as
               context, never learned from">not trained on</span>` : "")}
        ${raw(content ? `<div class="bubble-text">${esc(content)}</div>` : "")}
        ${raw(calls.map((c, n) => toolCall(c, n, per)).join(""))}
        ${raw(!content && !calls.length && !reasoning
          ? `<span class="muted tiny">(nothing in this turn)</span>` : "")}
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
