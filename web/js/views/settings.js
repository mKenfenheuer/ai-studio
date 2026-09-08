import { api } from "../api.js";
import { html, raw, esc, $, $$, on, toast } from "../util.js";
import { session, applyTheme, currentTheme } from "../app.js";
import { ribbon, rb, group, wireRibbon } from "../ribbon.js";
import { pageHead, copyButton } from "../components.js";

export async function settingsView(mount) {
  const status = await api.status();
  const admin = session.user?.role === "admin";
  mount.innerHTML = html`
    ${raw(pageHead({ title: "Settings", sub: "Studio-wide configuration." }))}
    ${raw(ribbon({
      tabs: [{ key: "general", label: "General" }], active: "general",
      body: group("Your things", [
        rb(null, "◉", "Your account", { href: "#/account",
          title: "Name, password, keys, Hugging Face, notifications" }),
      ]) + (admin ? group("Administration", [
        rb(null, "⌸", "Admin section", { href: "#/admin",
          title: "Machines, people, storage and backups" }),
        rb(null, "◍", "People", { href: "#/users" }),
        rb(null, "▦", "Machines", { href: "#/runners" }),
      ]) : "") + group("Session", [
        rb("signOut", "→", "Sign out", { cls: "danger" }),
      ]),
    }))}

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
      <h3>Join token</h3>
      <p class="muted tiny">Machines present this to join the studio. Anyone with
        it can attach a machine and read job data, so share it carefully.</p>
      <div class="row row-top">
        <input type="password" id="tok" value="${status.join_token}" readonly>
        <button class="btn-sm" id="reveal">Show</button>
        ${raw(copyButton(status.join_token || ""))}
      </div>
      <div class="hint">To rotate it, delete <code class="mono">data/join_token</code>
        on the controller and restart. Every machine must then rejoin.</div>
    </div>`)}

    <div class="card" style="margin-bottom:14px">
      <h3>Appearance</h3>
      <p class="muted tiny">Follows your operating system unless you say
        otherwise. Kept in this browser, not on your account.</p>
      <div class="seg" style="margin-top:8px" role="group" aria-label="Theme">
        ${raw([["system", "Match my system"], ["light", "Light"], ["dark", "Dark"]]
          .map(([v, label]) => `<button class="btn-sm ${
            currentTheme() === v ? "on" : ""}" data-theme-set="${v}">${label}</button>`).join(""))}
      </div>
    </div>

    <div class="card">
      <h3>About</h3>
      <dl class="kv">
        <dt>Version</dt><dd>${status.version}</dd>
        <dt>Machines</dt><dd>${status.runners_online} online / ${status.runners_total} known</dd>
        <dt>Controller</dt><dd class="mono">${location.origin}</dd>
        <dt>Signed in as</dt><dd>${session.user?.display_name || "?"}
          ${raw(admin ? `<span class="badge badge-accent">administrator</span>` : "")}</dd>
      </dl>
    </div>`;

  wireRibbon(mount, () => {});
  on(mount, "click", "[data-theme-set]", (_e, t) => {
    applyTheme(t.dataset.themeSet);
    $$("[data-theme-set]", mount).forEach((b) =>
      b.classList.toggle("on", b.dataset.themeSet === t.dataset.themeSet));
  });
  on(mount, "click", "#signOut", async () => {
    await api.logout().catch(() => {});
    location.reload();
  });
  on(mount, "click", "#reveal", (_e, t) => {
    const i = $("#tok", mount);
    const show = i.type === "password";
    i.type = show ? "text" : "password";
    t.textContent = show ? "Hide" : "Show";
  });
}
