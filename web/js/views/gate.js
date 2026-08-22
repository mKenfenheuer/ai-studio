/**
 * The sign-in screen, and the one that creates the very first account.
 *
 * It renders over the whole application rather than as a route, because a
 * route implies the shell around it is meaningful — and a sidebar full of
 * links that all answer 401 is worse than no sidebar. Nothing behind this is
 * drawn until there is a session.
 */
import { api } from "../api.js";
import { html, raw, $, on, toast } from "../util.js";

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
  setTimeout(() => $("#gateFirst", gate)?.focus(), 30);
}

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
