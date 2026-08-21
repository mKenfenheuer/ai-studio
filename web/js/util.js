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

export function on(root, event, selector, handler) {
  root.addEventListener(event, (e) => {
    const t = e.target.closest(selector);
    if (t && root.contains(t)) handler(e, t);
  });
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
