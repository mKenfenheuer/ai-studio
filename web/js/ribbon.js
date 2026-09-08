/**
 * The ribbon: tabs across the top, groups of labelled buttons underneath.
 *
 * Built for the dataset editor, where the shape earned itself — the rows are
 * the thing, and everything you can do to them is a button above them rather
 * than a form wedged between the rows and the page head. Every other page in
 * this studio has the same problem and had a different answer to it: an action
 * row here, three cards there, a `<details>` somewhere else, and on the run
 * page five separate layouts each re-emitting its own back link.
 *
 * So it moved out of the editor. A view says which tabs it has and what is on
 * the active one; this draws it, remembers which tab you were on, and answers
 * the arrow keys.
 *
 * Nothing here knows what a dataset or a run is. Views pass finished markup.
 */
import { esc, html, raw, $, $$, on } from "./util.js";

/** One ribbon button. A link when given `href`, a button otherwise — the
 *  difference matters for downloads, which must be a real navigation. */
export function rb(id, icon, label, opts = {}) {
  const { cls = "", title = "", disabled = false, data = "", href = "" } = opts;
  const inner = `<span class="ico" aria-hidden="true">${icon}</span><span>${esc(label)}</span>`;
  const attrs = `class="rb-btn ${cls}" ${id ? `id="${esc(id)}"` : ""} ${data}`
    + ` title="${esc(title || label)}"`;
  if (href && !disabled) return `<a ${attrs} href="${esc(href)}">${inner}</a>`;
  return `<button type="button" ${attrs}${disabled ? " disabled" : ""}>${inner}</button>`;
}

/** A labelled cluster of buttons. The label is what makes a ribbon readable:
 *  eight icons in a row is a puzzle, and "Apply · Source · Use" is a sentence. */
export const group = (label, items) => {
  const kept = items.filter(Boolean);
  if (!kept.length) return "";
  return `<div class="rb-group"><div class="rb-items">${kept.join("")}</div>`
    + `<div class="rb-label">${esc(label)}</div></div>`;
};

/** A dropdown that lives on the ribbon rather than in a form. */
export function rbSelect(id, { title = "", value = "", options = [], data = "" } = {}) {
  return `<select class="rb-select" id="${esc(id)}" ${data} title="${esc(title)}"
    aria-label="${esc(title || id)}">${options.map(([v, text]) =>
      `<option value="${esc(v)}"${String(v) === String(value) ? " selected" : ""}>${
        esc(text)}</option>`).join("")}</select>`;
}

/** A segmented control: two or three mutually exclusive choices, shown as one
 *  control because they are one decision. */
export const rbSeg = (items) =>
  `<div class="seg" role="group">${items.map((it) =>
    `<button type="button" class="btn-sm ${it.on ? "on" : ""}" ${it.data || ""}
      ${it.disabled ? "disabled" : ""}>${esc(it.label)}</button>`).join("")}</div>`;

/** A search box on the ribbon, for the pages whose main problem is finding
 *  one row in a long list. */
export const rbSearch = (id, { placeholder = "Search…", value = "" } = {}) =>
  `<input type="search" class="rb-search" id="${esc(id)}" value="${esc(value)}"
     placeholder="${esc(placeholder)}" aria-label="${esc(placeholder)}">`;

/**
 * The ribbon itself.
 *
 * `tabs`   [{key, label}]
 * `active` which key is showing
 * `body`   finished markup for that tab — normally a few group() calls
 * `right`  anything that belongs on the tab strip rather than under it
 *          (sharing, a status pill): visible on every tab
 */
export function ribbon({ tabs, active, body = "", right = "" }) {
  const strip = tabs.map((t) => `
    <button type="button" role="tab" data-tab="${esc(t.key)}"
      id="rbtab-${esc(t.key)}" aria-selected="${t.key === active}"
      ${t.disabled ? 'aria-disabled="true" disabled' : ""}
      ${t.hint ? `title="${esc(t.hint)}"` : ""}
      tabindex="${t.key === active ? 0 : -1}"
      class="${t.key === active ? "on" : ""}${t.disabled ? " off" : ""}"
      >${esc(t.label)}</button>`).join("");
  return html`
    <div class="ribbon">
      <div class="ribbon-tabs" role="tablist">
        ${raw(strip)}
        <span class="spacer"></span>
        ${raw(right ? `<span class="ribbon-right">${right}</span>` : "")}
      </div>
      <div class="ribbon-body" role="tabpanel"
           aria-labelledby="rbtab-${esc(active)}">${raw(body)}</div>
    </div>`;
}

/**
 * Which tab this page was left on.
 *
 * Remembered per view, because coming back to a page you were working on and
 * finding it reset to Home is the small daily tax that makes a tool feel like
 * it is not paying attention. Falls back cleanly when the remembered tab no
 * longer exists — tabs change between releases and a stored key outlives them.
 */
export function tabState(viewKey, tabs, fallback = null) {
  const storageKey = `aistudio.tab.${viewKey}`;
  const valid = new Set(tabs.map((t) => t.key));
  let current = fallback || tabs[0]?.key;
  try {
    const saved = localStorage.getItem(storageKey);
    if (saved && valid.has(saved)) current = saved;
  } catch { /* private browsing; the default is fine */ }
  return {
    get: () => current,
    set: (key) => {
      if (!valid.has(key)) return;
      current = key;
      try { localStorage.setItem(storageKey, key); } catch { /* as above */ }
    },
  };
}

/**
 * Clicks and arrow keys on the tab strip.
 *
 * Delegated through `on`, so it survives the redraw it causes. The arrow keys
 * are not decoration: a tab strip that cannot be traversed from the keyboard
 * is a menu that only exists for a mouse, and this one holds the primary
 * action of every page in the app.
 */
export function wireRibbon(mount, onTab) {
  on(mount, "click", "[data-tab]", (_e, t) => onTab(t.dataset.tab));
  on(mount, "keydown", ".ribbon-tabs", (e) => {
    const keys = ["ArrowLeft", "ArrowRight", "Home", "End"];
    if (!keys.includes(e.key)) return;
    const tabs = $$("[data-tab]:not([disabled])", e.currentTarget || mount);
    const at = tabs.findIndex((b) => b.getAttribute("aria-selected") === "true");
    if (at < 0) return;
    e.preventDefault();
    const next = e.key === "Home" ? 0
      : e.key === "End" ? tabs.length - 1
      : (at + (e.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length;
    onTab(tabs[next].dataset.tab);
    // The redraw replaces these nodes, so focus has to be put back by key
    // rather than held on the element that was clicked.
    requestAnimationFrame(() => $(`[data-tab="${tabs[next].dataset.tab}"]`, mount)?.focus());
  });
}
