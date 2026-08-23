/**
 * The sign-in screen, and the one that creates the very first account.
 *
 * It renders over the whole application rather than as a route, because a
 * route implies the shell around it is meaningful — and a sidebar full of
 * links that all answer 401 is worse than no sidebar. Nothing behind this is
 * drawn until there is a session.
 */
import { api } from "../api.js";
import { html, raw, esc, $, on, toast,
         takeSsoError } from "../util.js";

let onSignedIn = null;

export function initGate(callback) { onSignedIn = callback; }

export function showGate(state) {
  const gate = $("#gate");
  const app = $("#app");
  gate.hidden = false;
  app.style.display = "none";
  gate.innerHTML = state.setup_required ? setupForm()
    : state.must_change ? changeForm() : loginForm();
  wire(gate, state);
  showReturnedError(gate);
  setTimeout(() => $("#gateFirst", gate)?.focus(), 30);

  // The buttons arrive a moment after the form rather than delaying it. A
  // studio with no provider configured is the common case and must not pay a
  // request for the possibility, and a provider list that fails to load has to
  // leave a working password form behind, not an empty screen.
  if (!state.setup_required && !state.must_change) {
    api.authProviders()
      .then(({ providers }) => paintProviders(gate, providers || []))
      .catch(() => { /* password sign-in still works */ });
  }
}

function showReturnedError(gate) {
  const message = takeSsoError();
  if (!message) return;
  $("#gateError", gate).innerHTML =
    `<div class="callout callout-err">${esc(message)}</div>`;
}

function paintProviders(gate, providers) {
  const slot = $("#gateSso", gate);
  if (!slot || !providers.length) return;
  // Where they were heading before the session ran out, so that signing in
  // returns them to the page they asked for rather than to the dashboard.
  const next = location.hash.startsWith("#/") ? location.hash : "";
  slot.innerHTML = html`
    ${raw(providers.map((p) => html`
      <a class="btn btn-sso" href="/api/auth/sso/${p.id}/start${
          next ? "?next=" + encodeURIComponent(next) : ""}">
        <span class="sso-mark" aria-hidden="true">${raw(MARKS[p.kind] || MARKS.oidc)}</span>
        Continue with ${p.name}
      </a>`).join(""))}
    <div class="or"><span>or use a password</span></div>`;
  slot.hidden = false;
}

// Small inline marks rather than fetched logos: a strict page that loads
// nothing from another origin cannot fetch a brand asset, and a studio on a
// private network often has no route to one anyway.
const MARKS = {
  entra: `<svg viewBox="0 0 16 16" width="15" height="15">
    <rect x="0" y="0" width="7" height="7" fill="#f25022"/>
    <rect x="9" y="0" width="7" height="7" fill="#7fba00"/>
    <rect x="0" y="9" width="7" height="7" fill="#00a4ef"/>
    <rect x="9" y="9" width="7" height="7" fill="#ffb900"/></svg>`,
  google: `<svg viewBox="0 0 18 18" width="15" height="15">
    <path fill="#4285f4" d="M17.6 9.2c0-.6-.1-1.3-.2-1.9H9v3.5h4.8a4.1 4.1 0 0 1-1.8 2.7v2.2h2.9c1.7-1.6 2.7-3.9 2.7-6.5z"/>
    <path fill="#34a853" d="M9 18c2.4 0 4.5-.8 6-2.2l-2.9-2.3a5.4 5.4 0 0 1-8.1-2.8H1v2.3A9 9 0 0 0 9 18z"/>
    <path fill="#fbbc05" d="M4 10.7a5.4 5.4 0 0 1 0-3.4V5H1a9 9 0 0 0 0 8l3-2.3z"/>
    <path fill="#ea4335" d="M9 3.6c1.3 0 2.5.5 3.4 1.3L15 2.3A9 9 0 0 0 1 5l3 2.3A5.4 5.4 0 0 1 9 3.6z"/></svg>`,
  oidc: `<svg viewBox="0 0 16 16" width="15" height="15" fill="none"
    stroke="currentColor" stroke-width="1.6">
    <path d="M11 7V5a3 3 0 1 0-6 0v2"/><rect x="3" y="7" width="10" height="7" rx="1.5"/></svg>`,
};

export function hideGate() {
  $("#gate").hidden = true;
  $("#gate").innerHTML = "";
  $("#app").style.display = "";
}

// ---------------------------------------------------------------------------

const frame = (title, sub, body, foot = "") => html`
  <div class="gate-card">
    <div class="gate-brand"><span class="brand-mark">◆</span> AI Studio</div>
    <h1>${title}</h1>
    <p class="muted tiny">${sub}</p>
    <div id="gateSso" class="gate-sso" hidden></div>
    <form id="gateForm" autocomplete="on">${raw(body)}</form>
    <div id="gateError"></div>
    ${raw(foot)}
  </div>`;

const loginForm = () => frame(
  "Sign in", "Your runs, your datasets and your Hugging Face account.", html`
    <div class="field">
      <label for="gateFirst">Username</label>
      <input id="gateFirst" name="username" type="text" autocomplete="username"
             autocapitalize="none" spellcheck="false" required>
    </div>
    <div class="field">
      <label for="gatePw">Password</label>
      <input id="gatePw" name="password" type="password"
             autocomplete="current-password" required>
    </div>
    <button class="btn-primary" type="submit" id="gateGo">Sign in</button>`);

const setupForm = () => frame(
  "Set up this studio",
  "There are no accounts here yet, so this one becomes the administrator.",
  html`
    <div class="field">
      <label for="gateFirst">Username</label>
      <input id="gateFirst" name="username" type="text" autocomplete="username"
             autocapitalize="none" spellcheck="false" required
             placeholder="lower case, no spaces">
    </div>
    <div class="field">
      <label for="gateName">Your name</label>
      <input id="gateName" name="display_name" type="text" autocomplete="name"
             placeholder="shown next to your runs">
    </div>
    <div class="field">
      <label for="gatePw">Password</label>
      <input id="gatePw" name="password" type="password"
             autocomplete="new-password" required minlength="10">
      <div class="hint">At least 10 characters. Length is what makes a
        password hard to guess — a short one with symbols in it is not.</div>
    </div>
    <button class="btn-primary" type="submit" id="gateGo">Create the account</button>`,
  html`
    <div class="callout callout-warn" style="margin-top:14px">
      <strong>Until you finish this, anyone who can reach this address can
      set it up instead.</strong>
      That window closes the moment the first account exists, so do it now
      rather than later.
    </div>`);

const changeForm = () => frame(
  "Choose a new password",
  "An administrator set a temporary one. It has to be replaced before you can "
  + "do anything else.", html`
    <div class="field">
      <label for="gateFirst">New password</label>
      <input id="gateFirst" name="new_password" type="password"
             autocomplete="new-password" required minlength="10">
      <div class="hint">At least 10 characters.</div>
    </div>
    <div class="field">
      <label for="gatePw2">And again</label>
      <input id="gatePw2" name="confirm" type="password"
             autocomplete="new-password" required>
    </div>
    <button class="btn-primary" type="submit" id="gateGo">Save and continue</button>`,
  html`<button class="link-btn" id="gateSignOut" type="button"
               style="margin-top:12px">Sign out instead</button>`);

// ---------------------------------------------------------------------------

function wire(gate, state) {
  const err = (msg) => {
    $("#gateError", gate).innerHTML =
      `<div class="callout callout-err">${msg}</div>`;
  };

  on(gate, "click", "#gateSignOut", async () => {
    await api.logout().catch(() => {});
    location.reload();
  });

  $("#gateForm", gate).addEventListener("submit", async (e) => {
    e.preventDefault();
    const f = Object.fromEntries(new FormData(e.target).entries());
    const btn = $("#gateGo", gate);
    btn.disabled = true;
    $("#gateError", gate).innerHTML = "";
    try {
      if (state.setup_required) {
        const r = await api.setup(f);
        if (r.adopted_runs) {
          toast(`Signed in. ${r.adopted_runs} existing run(s) are now yours.`, "ok");
        }
      } else if (state.must_change) {
        if (f.new_password !== f.confirm) throw new Error("Those two do not match.");
        await api.changePassword({ new_password: f.new_password });
      } else {
        await api.login(f);
      }
      hideGate();
      if (onSignedIn) await onSignedIn();
    } catch (ex) {
      err(ex.message || String(ex));
      btn.disabled = false;
      $("#gatePw", gate)?.select?.();
    }
  });
}
