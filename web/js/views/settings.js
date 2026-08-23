import { api } from "../api.js";
import { html, raw, $, on, toast } from "../util.js";
import { session } from "../app.js";

export async function settingsView(mount) {
  const status = await api.status();
  const admin = session.user?.role === "admin";
  mount.innerHTML = html`
    <div class="page-head"><h1>Settings</h1>
      <p class="sub">Studio-wide configuration.</p></div>

    <div class="card" style="margin-bottom:14px">
      <h3>Hugging Face</h3>
      <p class="muted tiny">Needed for <em>gated</em> models such as Llama or
        Gemma, for better download rate limits, and to publish what you
        train.</p>
      <div class="callout ${status.hf_token_set ? "callout-ok" : "callout-warn"}">
        <strong>${status.hf_token_is_yours ? "Using your own account"
          : status.hf_token_set ? "Using the studio's shared token"
          : "No Hugging Face access"}</strong>
        ${raw(status.hf_token_is_yours
          ? `Downloads and publishing run as you.
             <a href="#/account">Manage it on your account page.</a>`
          : status.hf_token_set
          ? `This studio has a token set in its environment, and everyone
             here shares it. <a href="#/account">Connect your own account</a>
             to publish models under your name.`
          : `<a href="#/account">Connect your Hugging Face account</a> to
             download gated models and publish what you train.`)}
      </div>
    </div>

    ${raw(!admin ? "" : html`
    <div class="card" style="margin-bottom:14px">
      <div class="row-between" style="gap:10px;flex-wrap:wrap;align-items:flex-start">
        <div style="min-width:0">
          <h3 style="margin-bottom:2px">Single sign-on</h3>
          <p class="muted tiny" style="margin:0">Let people in with the account
            they already have — Entra ID, Google, Okta, Keycloak — and find
            colleagues by name when sharing, without keeping a second list of
            who works here.</p>
        </div>
        <a class="btn btn-sm" href="#/sso">Set up</a>
      </div>
    </div>

    <div class="card" style="margin-bottom:14px">
      <h3>Join token</h3>
      <p class="muted tiny">Machines present this to join the studio. Anyone with
        it can attach a machine and read job data, so share it carefully.</p>
      <div class="row row-top">
        <input type="password" id="tok" value="${status.join_token}" readonly>
        <button class="btn-sm" id="reveal">Show</button>
        <button class="btn-sm" data-copy="${status.join_token}">Copy</button>
      </div>
      <div class="hint">To rotate it, delete <code class="mono">data/join_token</code>
        on the controller and restart. Every machine must then rejoin.</div>
    </div>`)}

    <div class="card">
      <h3>About</h3>
      <dl class="kv">
        <dt>Version</dt><dd>${status.version}</dd>
        <dt>Machines</dt><dd>${status.runners_online} online / ${status.runners_total} known</dd>
        <dt>Controller</dt><dd class="mono">${location.origin}</dd>
        <dt>Signed in as</dt><dd>${session.user?.display_name || "?"}
          ${raw(admin ? `<span class="badge badge-accent">administrator</span>` : "")}</dd>
      </dl>
      ${raw(admin ? `<a class="btn btn-sm" href="#/users"
        style="margin-top:10px">Manage people</a>` : "")}
      <button class="btn-sm btn-danger" id="signOut"
              style="margin-top:10px">Sign out</button>
    </div>`;

  $("#signOut", mount).addEventListener("click", async () => {
    await api.logout().catch(() => {});
    location.reload();
  });

  $("#reveal", mount)?.addEventListener("click", (e) => {
    const i = $("#tok", mount);
    const show = i.type === "password";
    i.type = show ? "text" : "password";
    e.target.textContent = show ? "Hide" : "Show";
  });
  on(mount, "click", "[data-copy]", (_e, t) => {
    navigator.clipboard.writeText(t.dataset.copy)
      .then(() => toast("Copied.", "ok"))
      .catch(() => toast("Could not copy.", "err"));
  });
}
