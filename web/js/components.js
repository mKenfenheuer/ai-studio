/**
 * The pieces every page was writing out by hand.
 *
 * A page head in eighteen views, ten hand-rolled empty states, eleven back
 * links (and four pages that needed one and did not have it), six separate
 * copy-to-clipboard handlers of which exactly one coped with an insecure
 * origin, and eighteen native `confirm()` calls — one of them collecting a
 * password — in an app that already had a proper `<dialog>` helper sitting
 * unused.
 *
 * None of that is a design decision anybody made. It is what happens when
 * there is nowhere to put a small shared thing, so this is that place.
 */
import { esc, html, raw, $, on, modal, toast } from "./util.js";

/** The title of the page, and of the browser tab.
 *
 *  The tab title was never set: eighteen routes, work that runs for hours,
 *  and a workflow where having a run, a dataset and the playground open at
 *  once is normal — all of them reading "AI Studio" in the tab strip, the
 *  history and every bookmark. It is set here because this is the one thing
 *  every page draws exactly once.
 */
export function pageHead({ title, sub = "", right = "", back = null, tab = null }) {
  document.title = `${tab || title} · AI Studio`;
  return html`
    <div class="page-head">
      ${raw(back ? breadcrumb(back) : "")}
      <div class="row-between" style="flex-wrap:wrap;gap:8px;align-items:flex-start">
        <div style="min-width:0">
          <h1>${title}</h1>
          ${raw(sub ? `<p class="sub">${esc(sub)}</p>` : "")}
        </div>
        ${raw(right ? `<div class="row" style="flex-wrap:wrap;gap:8px">${right}</div>` : "")}
      </div>
    </div>`;
}

/** Where you were, one step up. Takes `{href, label}` or a list of them. */
export function breadcrumb(trail) {
  const steps = Array.isArray(trail) ? trail : [trail];
  return `<nav class="crumbs" aria-label="Breadcrumb">${steps.map((s) =>
    `<a href="${esc(s.href)}">${esc(s.label)}</a>`).join(
      `<span aria-hidden="true">/</span>`)}</nav>`;
}

/** Nothing here yet, and what to do about it.
 *
 *  The `cta` is the part that was missing from half of the hand-written ones:
 *  an empty state that only says "no runs yet" has told the reader something
 *  they could see for themselves. */
export function emptyState({ icon = "🌱", title, body = "", cta = null, card = true }) {
  const inner = html`
    <div class="big" aria-hidden="true">${icon}</div>
    <h3>${title}</h3>
    ${raw(body ? `<p class="muted">${esc(body)}</p>` : "")}
    ${raw(cta ? `<p><a class="btn btn-primary" href="${esc(cta.href)}">${
      esc(cta.label)}</a></p>` : "")}`;
  return `<div class="${card ? "card " : ""}empty">${inner}</div>`;
}

/** A button that copies a string.
 *
 *  The string travels in the attribute rather than in a closure so this works
 *  from inside markup any view builds, and the handler is installed once for
 *  the whole document below. */
export const copyButton = (text, label = "Copy") =>
  `<button type="button" class="btn-sm" data-copy="${esc(text)}">${esc(label)}</button>`;

/** Copy anything marked `data-copy`, from anywhere, once.
 *
 *  `navigator.clipboard` does not exist on a page served over plain http from
 *  another machine — which is exactly how this studio is usually reached — so
 *  the old textarea trick is kept as the fallback rather than the failure. */
async function copyText(text) {
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch { /* fall through to the fallback */ }
  try {
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.setAttribute("readonly", "");
    ta.style.cssText = "position:fixed;top:-1000px;opacity:0";
    document.body.appendChild(ta);
    ta.select();
    const ok = document.execCommand("copy");
    ta.remove();
    return ok;
  } catch { return false; }
}

on(document, "click", "[data-copy]", async (_e, t) => {
  const ok = await copyText(t.dataset.copy);
  toast(ok ? "Copied to clipboard." : "Could not copy — select the text manually.",
        ok ? "ok" : "err");
});

/**
 * "Are you sure", in a dialog this app can style, read out and lay out.
 *
 * The native `confirm()` cannot show a list of consequences as anything but
 * one run-on line, is suppressed outright by "prevent this page from creating
 * more dialogs", and on a phone puts its two buttons a thumb-width apart. It
 * was being used for deleting a run, deleting a person, and — through
 * `prompt()`, in clear text, in a box the browser may log — setting somebody's
 * password.
 *
 * `confirmWord` turns it into type-the-name, for the ones that cannot be
 * undone and cost more than a minute to redo.
 *
 * Resolves true or false. Never throws: a dialog that fails to open must not
 * take the caller's click handler with it.
 */
export function confirmDestructive({
  title, body = "", consequences = [], confirmLabel = "Delete",
  confirmWord = null, tone = "danger",
} = {}) {
  return new Promise((resolve) => {
    let answered = false;
    const done = (v) => { if (!answered) { answered = true; resolve(v); } };

    const dlg = modal({
      title,
      width: 460,
      onClose: () => done(false),
      body: html`
        ${raw(body ? `<p>${esc(body)}</p>` : "")}
        ${raw(consequences.length ? `<ul class="consequences">${
          consequences.map((c) => `<li>${esc(c)}</li>`).join("")}</ul>` : "")}
        ${raw(confirmWord ? html`
          <div class="field">
            <label for="confirmWord">Type <strong>${confirmWord}</strong> to confirm</label>
            <input id="confirmWord" autocomplete="off" spellcheck="false">
          </div>` : "")}
        <div class="row" style="justify-content:flex-end;gap:8px;margin-top:14px">
          <button type="button" class="btn" data-modal-close>Cancel</button>
          <button type="button" class="btn ${tone === "danger" ? "btn-danger" : "btn-primary"}"
                  id="confirmGo" ${confirmWord ? "disabled" : ""}>${confirmLabel}</button>
        </div>`,
    });

    const word = $("#confirmWord", dlg);
    const go = $("#confirmGo", dlg);
    if (word) {
      word.addEventListener("input", () => {
        go.disabled = word.value.trim() !== confirmWord;
      });
      // Enter in the box is the same answer as pressing the button, but only
      // once the box actually says the word.
      word.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !go.disabled) { e.preventDefault(); go.click(); }
      });
      word.focus();
    } else {
      go.focus();
    }
    go.addEventListener("click", () => { done(true); dlg.close(); });
  });
}
