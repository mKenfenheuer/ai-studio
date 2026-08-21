import { api } from "../api.js";
import { html, raw, $, on, toast } from "../util.js";

export async function settingsView(mount) {
  const status = await api.status();
  mount.innerHTML = html`
    <div class="page-head"><h1>Settings</h1>
      <p class="sub">Studio-wide configuration.</p></div>

    <div class="card" style="margin-bottom:14px">
      <h3>Hugging Face access token</h3>
      <p class="muted tiny">Needed only for <em>gated</em> models such as Llama or
        Gemma, where you must accept a licence first. It also raises download
        rate limits.</p>
      <div class="callout ${status.hf_token_set ? "callout-ok" : "callout-warn"}">
        <strong>${status.hf_token_set ? "A token is configured" : "No token configured"}</strong>
        ${raw(status.hf_token_set
          ? "Gated models can be downloaded."
          : `Set it on the controller and restart:
             <code class="mono">HF_TOKEN=hf_xxx</code>. It is deliberately not
             editable from this page, so a browser session can never read or
             change your credentials.`)}
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
    </div>

    <div class="card">
      <h3>About</h3>
      <dl class="kv">
        <dt>Version</dt><dd>${status.version}</dd>
        <dt>Machines</dt><dd>${status.runners_online} online / ${status.runners_total} known</dd>
        <dt>Controller</dt><dd class="mono">${location.origin}</dd>
      </dl>
    </div>`;

  $("#reveal", mount).addEventListener("click", (e) => {
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
