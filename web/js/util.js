// Small helpers. No framework: the whole UI must run with zero build step,
// so a Python-only install is all anyone ever needs.

/** Escape untrusted text for interpolation into HTML. */
export const esc = (s) => String(s ?? "").replace(/[&<>"']/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

/** Tagged template that escapes interpolations by default.
 *  Use raw(...) to opt out for already-built markup. */
export function html(strings, ...vals) {
  return strings.reduce((out, s, i) => {
    const v = vals[i - 1];
    let piece;
    if (v === undefined || v === null || v === false) piece = "";
    else if (v && v.__raw) piece = v.value;
    else if (Array.isArray(v)) piece = v.map((x) => (x && x.__raw ? x.value : esc(x))).join("");
    else piece = esc(v);
    return out + piece + s;
  });
}
export const raw = (value) => ({ __raw: true, value });

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

/** A delegated listener that survives the view being redrawn.
 *
 *  Every view here renders by replacing the contents of one mount and then
 *  re-wiring it, and delegation is what makes that work: the listener sits on
 *  the mount, which is not replaced. But that is also the trap. Registering
 *  again on the next redraw added a *second* listener to the same node, and a
 *  third, and a tenth — so one click ran the handler ten times.
 *
 *  It showed up as a delete confirmation that would not go away: the first
 *  answer deleted the dataset, and the nine stacked copies of the same
 *  handler each asked again, then failed against a dataset that was already
 *  gone. So this keeps exactly one listener per (root, event, selector) and
 *  points it at the newest handler — which closes over the newest state, and
 *  is what re-wiring was trying to achieve in the first place.
 */
export function on(root, event, selector, handler) {
  const key = event + " " + selector;
  const registry = root.__delegated || (root.__delegated = new Map());
  const existing = registry.get(key);
  if (existing) {
    existing.handler = handler;
    return;
  }
  // The listener is kept by reference so it can be removed again. An
  // AbortSignal reads better and is not used: it is honoured by browsers but
  // silently ignored by some DOM implementations, and a teardown that quietly
  // does nothing is the exact failure this is here to prevent.
  const entry = { handler, event };
  entry.listener = (e) => {
    const t = e.target.closest(selector);
    if (t && root.contains(t)) entry.handler(e, t);
  };
  registry.set(key, entry);
  root.addEventListener(event, entry.listener);
}

/** Forget every delegated listener on this root.
 *
 *  Keeping one listener per selector and swapping the handler works only while
 *  the next thing drawn registers the same selectors. It does not, when the
 *  next thing is a different page — and then a button drawn by the NEW page
 *  runs a handler closed over the OLD page's state.
 *
 *  That was not hypothetical. The job page began branching by kind, the
 *  generation branch did not register the stop controls, and the Stop button
 *  on a generation run went on calling cancelJob with the job id of whichever
 *  training run had been looked at last. It stopped the wrong run, and the
 *  right one could not be stopped at all.
 *
 *  So the router clears these between routes. A stale handler cannot fire if
 *  it is not there, and a page that forgets to wire a control now gets a
 *  button that does nothing — which is visible, unlike a button that does
 *  something to something else.
 */
export function resetDelegated(root) {
  const registry = root?.__delegated;
  if (!registry) return;
  registry.forEach((entry) => root.removeEventListener(entry.event, entry.listener));
  registry.clear();
}

export function fmtBytes(n) {
  if (!n && n !== 0) return "—";
  const u = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(n < 10 && i > 0 ? 1 : 0)} ${u[i]}`;
}

export function fmtNum(n) {
  if (n === null || n === undefined) return "—";
  if (n >= 1e9) return (n / 1e9).toFixed(1) + "B";
  if (n >= 1e6) return (n / 1e6).toFixed(1) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(1) + "k";
  return String(n);
}

export function fmtDuration(sec) {
  if (sec === null || sec === undefined) return "—";
  sec = Math.max(0, Math.round(sec));
  if (sec < 60) return `${sec}s`;
  const m = Math.floor(sec / 60), s = sec % 60;
  if (m < 60) return `${m}m ${s}s`;
  return `${Math.floor(m / 60)}h ${m % 60}m`;
}

export function fmtAgo(ts) {
  if (!ts) return "—";
  const d = Date.now() / 1000 - ts;
  if (d < 60) return "just now";
  if (d < 3600) return `${Math.floor(d / 60)}m ago`;
  if (d < 86400) return `${Math.floor(d / 3600)}h ago`;
  return `${Math.floor(d / 86400)}d ago`;
}

export const STATUS_STYLE = {
  queued:    ["badge", "Waiting for a free machine"],
  assigned:  ["badge badge-accent", "Starting"],
  running:   ["badge badge-accent", "Training"],
  succeeded: ["badge badge-ok", "Done"],
  failed:    ["badge badge-err", "Failed"],
  cancelled: ["badge", "Stopped"],
};

export function statusBadge(status) {
  const [cls, label] = STATUS_STYLE[status] || ["badge", status];
  return raw(`<span class="${cls}">${esc(label)}</span>`);
}

export function toast(message, kind = "") {
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.textContent = message;
  document.getElementById("toasts").appendChild(el);
  setTimeout(() => el.remove(), kind === "err" ? 7000 : 4000);
}

/** Ask for notification permission, but only where it makes sense to ask.
 *  Browsers refuse the request unless it follows a click, and they penalise
 *  sites that ask on load, so this is called from the account page. */
export async function askForNotifications() {
  if (!("Notification" in window)) return "unsupported";
  if (Notification.permission !== "default") return Notification.permission;
  try { return await Notification.requestPermission(); }
  catch { return "denied"; }
}

/** A modal dialog, built on the element the platform already provides.
 *
 *  `<dialog>` is used rather than a div with a z-index because it brings the
 *  three things a hand-rolled modal always gets wrong for free: focus is
 *  trapped inside it, Escape closes it, and everything behind it is inert to
 *  both the mouse and the screen reader. */
export function modal({ title, body = "", width = 580, onClose } = {}) {
  const dlg = document.createElement("dialog");
  dlg.className = "modal";
  dlg.style.setProperty("--modal-w", `${width}px`);
  dlg.innerHTML = `
    <div class="modal-head">
      <h2>${esc(title || "")}</h2>
      <button class="icon-btn" data-modal-close aria-label="Close">✕</button>
    </div>
    <div class="modal-body">${body}</div>`;
  document.body.appendChild(dlg);
  dlg.addEventListener("close", () => { dlg.remove(); onClose?.(); });
  on(dlg, "click", "[data-modal-close]", () => dlg.close());
  // A click on the backdrop is reported as a click on the dialog itself --
  // the box has its own padding-free geometry -- so comparing the target to
  // the element is what tells "outside" from "inside".
  dlg.addEventListener("click", (e) => { if (e.target === dlg) dlg.close(); });
  dlg.showModal();
  return dlg;
}

/** Run `fn` only once the typing stops. Used by every search-as-you-type box:
 *  without it each keystroke is a request, and the answers come back out of
 *  order so the list flickers between two different searches. */
export function debounce(fn, ms = 180) {
  let t = null;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
}

/** One or two letters standing in for a face. */
export function initials(name) {
  const parts = String(name || "?").trim().split(/[\s._-]+/).filter(Boolean);
  if (!parts.length) return "?";
  return (parts[0][0] + (parts.length > 1 ? parts[parts.length - 1][0] : ""))
    .toUpperCase();
}

/** A person, as a small avatar.
 *
 *  The initials are always rendered; the picture, when the directory gave us
 *  one, is laid over them. So a photo that 404s or is blocked reveals the
 *  initials underneath rather than a broken-image icon, with no error
 *  handler to get wrong. */
export function avatar(person, size = 30) {
  const px = `width:${size}px;height:${size}px;font-size:${Math.round(size * 0.38)}px`;
  const pic = person?.avatar_url
    ? `<img src="${esc(person.avatar_url)}" alt="" loading="lazy">` : "";
  return raw(`<span class="who-avatar" style="${px}">${
    esc(initials(person?.display_name || person?.username))}${pic}</span>`);
}

/** The message a failed sign-in sent back, read once and then scrubbed.
 *
 *  It arrives as a query parameter because the browser was at the identity
 *  provider when it went wrong and there was nowhere else to put it. Read
 *  once: leaving it in the address bar means a reload redisplays an error
 *  about something that already happened, and a bookmark carries it forever.
 */
export function takeSsoError() {
  const params = new URLSearchParams(location.search);
  const message = params.get("sso_error");
  if (!message) return null;
  params.delete("sso_error");
  const rest = params.toString();
  history.replaceState(null, "",
    location.pathname + (rest ? "?" + rest : "") + location.hash);
  return message;
}

/** A placeholder shaped like the thing that is coming.
 *
 *  Not decoration. A page that renders as one line of "Loading…" and then
 *  expands into nine cards moves everything below it, and on a phone that
 *  means the control you were reaching for is somewhere else by the time your
 *  thumb arrives. Taking up roughly the right room from the start is the
 *  whole point; the shimmer is just what makes it read as "coming" rather
 *  than as "broken".
 */
export function skeleton({ title = true, cards = 3, rows = 0 } = {}) {
  const card = `<div class="sk-card shimmer"></div>`;
  return `
    <div class="skeleton" aria-busy="true" aria-live="polite">
      <span class="visually-hidden">Loading</span>
      ${title ? `<div class="sk-title shimmer"></div>
                 <div class="sk-sub shimmer"></div>` : ""}
      ${cards ? `<div class="sk-row">${card.repeat(cards)}</div>` : ""}
      ${Array.from({ length: rows },
                   () => `<div class="sk-line shimmer"></div>`).join("")}
    </div>`;
}

/** A single value that has not arrived: keeps its own width and height so the
 *  card around it does not resize when it does. */
export const skeletonValue = (width = "3.5em") =>
  raw(`<span class="sk-value shimmer" style="min-width:${width}"></span>`);
