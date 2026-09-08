/** Your account: your name, your password, how you are told a run ended,
 *  your Hugging Face connection, and the hosted models you can write data
 *  with. */
import { api } from "../api.js";
import { ribbon, rb, group, wireRibbon, tabState } from "../ribbon.js";
import { pageHead } from "../components.js";
import { html, raw, esc, $, $$, on, toast, fmtAgo,
         askForNotifications } from "../util.js";

const TABS = [
  { key: "profile", label: "Profile" },
  { key: "keys", label: "API keys" },
  { key: "hf", label: "Hugging Face" },
  { key: "hosted", label: "Hosted models" },
  { key: "alerts", label: "Notifications" },
];

export async function accountView(mount) {
  let me = await api.me();
  let alerts = await api.notifyState();
  let keys = await api.apiKeys();
  let hosted = await api.providers();
  // The one time a key is visible. Held in memory only, and dropped as soon
  // as the page is left -- there is nowhere it could be stored that would not
  // be a worse place than the user's own password manager.
  let freshKey = null;
  // Which provider's form is open. Only one at a time: they are forms with a
  // secret in them, not a list to browse.
  let opening = null;

  const tabs = tabState("account", TABS, "profile");
  let tab = tabs.get();

  const draw = () => {
    mount.innerHTML = layout(me, alerts, keys, freshKey, hosted, opening, tab);
    wire();
  };

  function wire() {
    wireRibbon(mount, (key) => { tab = key; tabs.set(key); draw(); });
    on(mount, "submit", "#keyForm", async (e) => {
      e.preventDefault();
      const name = new FormData(e.target).get("name");
      try {
        freshKey = await api.createApiKey(name);
        keys = await api.apiKeys();
        draw();
      } catch (ex) { toast(ex.message, "err"); }
    });

    on(mount, "click", "#copyKey", async () => {
      const box = $("#freshKey", mount);
      box.select();
      try {
        await navigator.clipboard.writeText(box.value);
        toast("Copied.", "ok");
      } catch {
        // Clipboard access needs a secure context, which a studio reached
        // over plain http on a local address is not. The text is selected,
        // which is the fallback that always works.
        toast("Select it and copy — this browser will not do it for us here.");
      }
    });

    on(mount, "click", "[data-del-key]", async (_e, t) => {
      if (!confirm("Delete this key?\n\nAnything using it stops working "
                   + "immediately.")) return;
      try {
        await api.deleteApiKey(t.dataset.delKey);
        keys = await api.apiKeys();
        freshKey = null;
        draw();
      } catch (ex) { toast(ex.message, "err"); }
    });

    on(mount, "submit", "#hookForm", async (e) => {
      e.preventDefault();
      const url = $("#hookUrl", mount).value.trim();
      if (!url) return toast("Paste the address first.", "err");
      try {
        alerts = await api.notifySet({ url });
        toast("Saved. Send a test to be sure it arrives.", "ok");
        draw();
      } catch (ex) { toast(ex.message, "err"); }
    });

    on(mount, "click", "#hookTest", async (_e, t) => {
      t.disabled = true;
      t.textContent = "Sending…";
      const r = await api.notifyTest().catch((ex) => ({ ok: false, detail: ex.message }));
      $("#hookResult", mount).innerHTML = r.ok
        ? `<div class="callout callout-ok"><strong>It arrived</strong>
             The endpoint answered ${esc(r.detail)}.</div>`
        : `<div class="callout callout-err"><strong>It did not arrive</strong>
             ${esc(r.detail)}</div>`;
      t.disabled = false;
      t.textContent = "Send a test";
    });

    on(mount, "click", "#hookClear", async () => {
      try { alerts = await api.notifyClear(); toast("Removed.", "ok"); draw(); }
      catch (ex) { toast(ex.message, "err"); }
    });

    on(mount, "change", "[data-alert-event]", async () => {
      const events = $$("[data-alert-event]:checked", mount)
        .map((c) => c.dataset.alertEvent);
      try { alerts = await api.notifySet({ events }); }
      catch (ex) { toast(ex.message, "err"); }
    });

    on(mount, "click", "#askNotify", async () => {
      const r = await askForNotifications();
      if (r !== "granted") {
        toast(r === "denied"
          ? "Your browser blocked it. That has to be changed in its site settings."
          : "Notifications are not available in this browser.", "err");
      }
      draw();
    });

    on(mount, "submit", "#nameForm", async (e) => {
      e.preventDefault();
      const v = $("#displayName", mount).value.trim();
      try {
        me = { ...me, ...(await api.updateMe({ display_name: v })) };
        toast("Saved.", "ok");
        // The sidebar shows this name; reloading is the honest way to keep
        // one source of truth rather than two copies that can disagree.
        location.reload();
      } catch (ex) { toast(ex.message, "err"); }
    });

    on(mount, "submit", "#pwForm", async (e) => {
      e.preventDefault();
      const f = Object.fromEntries(new FormData(e.target).entries());
      if (f.new_password !== f.confirm) return toast("Those two do not match.", "err");
      try {
        const r = await api.changePassword(f);
        toast(r.other_sessions_ended
          ? `Password changed. ${r.other_sessions_ended} other session(s) signed out.`
          : "Password changed.", "ok");
        e.target.reset();
        me = await api.me();
        draw();
      } catch (ex) { toast(ex.message, "err"); }
    });

    on(mount, "click", "#revoke", async () => {
      if (!confirm("Sign out everywhere except this browser?")) return;
      try {
        const r = await api.revokeSessions();
        toast(`${r.ended} session(s) signed out.`, "ok");
        me = await api.me();
        draw();
      } catch (ex) { toast(ex.message, "err"); }
    });

    on(mount, "click", "[data-open-provider]", (_e, t) => {
      opening = opening === t.dataset.openProvider ? null : t.dataset.openProvider;
      draw();
    });

    on(mount, "submit", "[data-provider-form]", async (e) => {
      e.preventDefault();
      const id = e.target.dataset.providerForm;
      const body = Object.fromEntries(new FormData(e.target).entries());
      try {
        await api.saveProvider(id, body);
        hosted = await api.providers();
        opening = null;
        toast("Saved. Nothing was sent anywhere yet — use Test to check it.", "ok");
        draw();
      } catch (ex) { toast(ex.message, "err"); }
    });

    on(mount, "click", "[data-test-provider]", async (_e, t) => {
      const id = t.dataset.testProvider;
      const box = $(`[data-provider-result="${id}"]`, mount);
      const model = $(`[data-provider-model="${id}"]`, mount)?.value || "";
      box.innerHTML = `<span class="muted tiny">Asking ${esc(id)} to say hello…</span>`;
      try {
        const r = await api.testProvider(id, model);
        box.innerHTML = `<div class="callout callout-ok" style="margin-top:8px">
          <strong>${esc(r.model)} answered</strong>
          "${esc(r.reply || "(nothing)")}" —
          ${r.usage.input_tokens + r.usage.output_tokens} tokens.</div>`;
        hosted = await api.providers();
      } catch (ex) {
        box.innerHTML = `<div class="callout callout-err" style="margin-top:8px">${
          esc(ex.message)}</div>`;
      }
    });

    on(mount, "click", "[data-list-models]", async (_e, t) => {
      const id = t.dataset.listModels;
      const box = $(`[data-provider-result="${id}"]`, mount);
      box.innerHTML = `<span class="muted tiny">Asking what it can reach…</span>`;
      try {
        const r = await api.providerModels(id);
        box.innerHTML = `<div class="callout" style="margin-top:8px">
          <strong>${r.models.length} model(s)</strong>
          ${raw(r.note ? esc(r.note) : "")}
          <div class="mono tiny" style="max-height:140px;overflow:auto;margin-top:6px">${
            r.models.map((m) => esc(m)).join("<br>")}</div></div>`;
      } catch (ex) {
        box.innerHTML = `<div class="callout callout-err" style="margin-top:8px">${
          esc(ex.message)}</div>`;
      }
    });

    on(mount, "click", "[data-forget-provider]", async (_e, t) => {
      const id = t.dataset.forgetProvider;
      const scope = t.dataset.scope || "account";
      if (!confirm(`Disconnect ${id}?

`
                   + (scope === "studio"
                      ? "This one is shared with the whole studio, so it goes "
                        + "for everybody. "
                      : "The key is deleted from this studio. ")
                   + "Runs already queued keep going.")) return;
      try {
        const r = await api.deleteProvider(id, scope);
        hosted = await api.providers();
        toast(r.note || "Disconnected.", "ok");
        draw();
      } catch (ex) { toast(ex.message, "err"); }
    });

    on(mount, "submit", "#hfForm", async (e) => {
      e.preventDefault();
      const token = $("#hfToken", mount).value.trim();
      const btn = $("#hfGo", mount);
      btn.disabled = true;
      btn.textContent = "Checking with Hugging Face…";
      try {
        await api.hfConnect(token);
        toast("Connected.", "ok");
        me = await api.me();
        draw();
      } catch (ex) {
        toast(ex.message, "err");
        btn.disabled = false;
        btn.textContent = "Connect";
      }
    });

    on(mount, "click", "#hfDisconnect", async () => {
      if (!confirm("Disconnect your Hugging Face account?\n\nRuns already "
                   + "queued keep the token they were given; new ones will "
                   + "fall back to the studio's shared token, if there is one."))
        return;
      await api.hfDisconnect();
      toast("Disconnected.", "ok");
      me = await api.me();
      draw();
    });

    on(mount, "click", "[data-repos]", async (_e, t) => {
      const kind = t.dataset.repos;
      const box = $("#repoList", mount);
      box.innerHTML = `<div class="muted tiny">Loading your ${kind}…</div>`;
      $$("[data-repos]", mount).forEach((b) =>
        b.classList.toggle("btn-primary", b.dataset.repos === kind));
      try {
        box.innerHTML = repoList(await api.hfRepos(kind), kind);
      } catch (ex) {
        box.innerHTML = `<div class="callout callout-err">${esc(ex.message)}</div>`;
      }
    });

    on(mount, "click", "[data-delrepo]", async (_e, t) => {
      const id = t.dataset.delrepo;
      const kind = t.dataset.kind;
      const typed = prompt(
        `Delete ${id} from Hugging Face?\n\n`
        + "This removes it for everyone, permanently, and cannot be undone.\n"
        + `Type the repository name to confirm:`);
      if (typed !== id.split("/").pop()) {
        if (typed !== null) toast("Name did not match; nothing deleted.");
        return;
      }
      try {
        await api.hfDeleteRepo(id, kind);
        toast("Deleted on Hugging Face.", "ok");
        $(`[data-repos="${kind}"]`, mount)?.click();
      } catch (ex) { toast(ex.message, "err"); }
    });
  }

  draw();
}

// ---------------------------------------------------------------------------

function layout(me, alerts, keys, freshKey, hosted, opening, tab) {
  const hf = me.hf || {};
  const panel = tab === "keys" ? keysCard(keys, freshKey)
    : tab === "hf" ? hfCard(hf)
    : tab === "hosted" ? hostedCard(hosted, opening)
    : tab === "alerts" ? alertCard(alerts)
    : profilePanel(me);

  return html`
    ${raw(pageHead({
      title: "Your account",
      sub: `Signed in as ${me.username}${me.role === "admin" ? " · administrator" : ""}`,
    }))}
    ${raw(ribbon({
      tabs: TABS, active: tab,
      body: group("Elsewhere", [
        rb(null, "⚙", "Settings", { href: "#/settings" }),
        me.role === "admin" ? rb(null, "◍", "People", { href: "#/users" }) : "",
        me.role === "admin" ? rb(null, "🔑", "Single sign-on", { href: "#/sso" }) : "",
      ]),
    }))}
    <div class="acct-panel">${raw(panel)}</div>`;
}

/** Name, password, and the sessions you are signed in on. */
function profilePanel(me) {
  return html`
    <div class="grid grid-2" style="align-items:start">
      <div>
        <div class="card" style="margin-bottom:14px">
          <h3>Name</h3>
          <p class="muted tiny">Shown next to your runs and when somebody
            shares something with you.</p>
          <form id="nameForm" class="row row-top" style="margin-top:10px">
            <input type="text" id="displayName" value="${me.display_name}"
                   maxlength="80">
            <button class="btn-sm btn-primary" type="submit">Save</button>
          </form>
        </div>

        <div class="card" style="margin-bottom:14px">
          <h3>Password</h3>
          <form id="pwForm">
            <div class="field">
              <label for="curPw">Current password</label>
              <input id="curPw" name="current_password" type="password"
                     autocomplete="current-password" required>
            </div>
            <div class="field">
              <label for="newPw">New password</label>
              <input id="newPw" name="new_password" type="password"
                     autocomplete="new-password" minlength="10" required>
              <div class="hint">At least 10 characters.</div>
            </div>
            <div class="field">
              <label for="newPw2">And again</label>
              <input id="newPw2" name="confirm" type="password"
                     autocomplete="new-password" required>
            </div>
            <button class="btn-sm btn-primary" type="submit">Change password</button>
          </form>
        </div>

        <div class="card">
          <h3>Where you are signed in</h3>
          <p class="muted tiny">${(me.sessions || []).length} active session${
            (me.sessions || []).length === 1 ? "" : "s"}.</p>
          <ul class="muted tiny" style="margin:8px 0;padding-left:18px;line-height:1.7">
            ${raw((me.sessions || []).slice(0, 6).map((s) => html`
              <li>${shortAgent(s.user_agent)} — last used ${
                fmtAgo(s.last_used)}</li>`).join(""))}
          </ul>
          <button class="btn-sm btn-danger" id="revoke">Sign out everywhere else</button>
        </div>
      </div>

    </div>`;
}

/** Using your models from outside this app. */
function keysCard(keys, fresh) {
  const origin = location.origin;
  return html`
    <div class="card" style="margin-bottom:14px">
      <h3>Use your models from other software</h3>
      <p class="muted tiny">This studio speaks the OpenAI API, so anything with
        a &ldquo;base URL&rdquo; and an API key field can talk to a model you
        trained here — Home Assistant, a script, an editor plugin, or any
        client library.</p>

      <div class="field" style="margin-top:10px">
        <label>Base URL</label>
        <input type="text" class="mono" readonly value="${origin}/v1">
        <div class="hint">Model names come from
          <code>GET ${origin}/v1/models</code> — each finished run is one, by
          its id or its name, and so is every
          <a href="#/serving">registered name</a>. Point a client at a
          registered name rather than a run: it survives a rename, and you can
          move it to a better model without editing anything out there.</div>
      </div>

      ${raw(fresh ? html`
        <div class="callout callout-ok">
          <strong>Copy this now</strong>
          It is stored hashed and cannot be shown again. If you lose it,
          delete it and make another.
          <div class="row" style="gap:6px;margin-top:8px">
            <input type="text" class="mono" readonly id="freshKey"
                   value="${fresh.key}" style="flex:1">
            <button class="btn-sm" id="copyKey">Copy</button>
          </div>
        </div>` : "")}

      <form id="keyForm" class="row" style="gap:6px;margin-top:10px;flex-wrap:wrap">
        <input type="text" name="name" placeholder="What is it for? e.g. Home Assistant"
               style="flex:1;min-width:160px">
        <button class="btn-primary btn-sm" type="submit">Create a key</button>
      </form>

      ${raw(keys.length ? html`
        <table class="table" style="margin-top:10px"><tbody>
          ${raw(keys.map((k) => html`
            <tr>
              <td>${k.name}
                <div class="muted tiny mono">${k.prefix}…</div></td>
              <td class="tiny muted">${k.last_used
                ? `used ${fmtAgo(k.last_used)}` : "never used"}
                ${raw(k.last_used
                  ? `<div><a href="#/serving">what it served</a></div>` : "")}</td>
              <td><button class="btn-sm btn-danger" data-del-key="${k.id}"
                          title="Delete this key">✕</button></td>
            </tr>`).join(""))}
        </tbody></table>` : "")}

      <details class="adv" style="margin-top:10px">
        <summary>How to point something at it</summary>
        <pre class="code tiny">curl ${origin}/v1/chat/completions \
  -H "Authorization: Bearer YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "your-run-name",
       "messages": [{"role": "user", "content": "hello"}]}'</pre>
        <p class="muted tiny">Streaming works too — send
          <code>"stream": true</code>. A key acts as you: it can reach the runs
          you can reach and no others.</p>
      </details>
    </div>`;
}

/** Being told a run ended, without having to sit and watch it. */
function alertCard(a) {
  const browserState = typeof Notification === "undefined" ? "unsupported"
    : Notification.permission;
  return html`
    <div class="card" style="margin-bottom:14px">
      <h3>Tell me when a run ends</h3>
      <p class="muted tiny">A run here takes hours. Nothing has to be open for
        the webhook to reach you.</p>

      <form id="hookForm" style="margin-top:10px">
        <div class="field">
          <label for="hookUrl">Webhook address</label>
          <input id="hookUrl" name="url" type="url" class="mono"
                 placeholder="${a.configured ? a.url_hint
                                : "https://ntfy.sh/your-topic"}">
          <div class="hint">One JSON POST per event. Works with ntfy, Slack,
            Discord, Gotify, Home Assistant, or your own script — it carries
            the run&rsquo;s name, how it ended and its held-out loss, plus a
            <code>text</code> field for services that only render one.
            ${raw(a.configured
              ? "Stored encrypted and never shown again, because whoever "
                + "holds this address can post as you."
              : "")}</div>
        </div>
        <div class="row" style="gap:6px;flex-wrap:wrap">
          <button class="btn-primary btn-sm" type="submit">
            ${a.configured ? "Replace it" : "Save"}</button>
          ${raw(a.configured ? html`
            <button class="btn-sm" type="button" id="hookTest">Send a test</button>
            <button class="btn-sm btn-danger" type="button" id="hookClear">Remove</button>`
            : "")}
        </div>
      </form>
      <div id="hookResult"></div>

      <div style="margin-top:12px">
        <strong class="tiny">Tell me about</strong>
        ${raw(Object.entries(a.available_events || {}).map(([id, label]) => html`
          <label class="check" style="margin-top:4px">
            <input type="checkbox" data-alert-event="${id}"
                   ${a.events.includes(id) ? "checked" : ""}>
            <span>when ${label}</span>
          </label>`).join(""))}
      </div>

      <div style="margin-top:12px">
        <strong class="tiny">In this browser</strong>
        <p class="muted tiny" style="margin:4px 0 6px">${
          browserState === "granted"
            ? "On. A notification appears when a run ends and this tab is not "
              + "the one you are looking at."
            : browserState === "denied"
            ? "Blocked. Your browser is refusing notifications for this site; "
              + "that has to be changed in its own site settings."
            : browserState === "unsupported"
            ? "This browser does not offer notifications."
            : "Off. Runs still finish quietly, and the webhook above still "
              + "works."}</p>
        ${raw(browserState === "default"
          ? `<button class="btn-sm" id="askNotify">Allow notifications</button>` : "")}
      </div>
    </div>`;
}

function hfCard(hf) {
  if (!hf.connected) {
    return html`
      <div class="card">
        <h3>Hugging Face</h3>
        <p class="muted tiny">Connect your account and this studio can download
          gated models as you, publish what you train to your own profile, and
          list your models and datasets here.</p>

        <form id="hfForm" style="margin-top:12px">
          <div class="field">
            <label for="hfToken">Access token</label>
            <input id="hfToken" type="password" class="mono"
                   placeholder="hf_…" autocomplete="off" required>
            <div class="hint">Create one at
              <a href="https://huggingface.co/settings/tokens" target="_blank"
                 rel="noopener">huggingface.co/settings/tokens</a>.
              Give it <strong>write</strong> access to repositories if you want
              to publish from here; read access is enough for gated downloads.</div>
          </div>
          <button class="btn-primary btn-sm" type="submit" id="hfGo">Connect</button>
        </form>

        <div class="callout" style="margin-top:12px">
          <strong>What happens to it</strong>
          It is encrypted before it is written down, and no page in this app
          can read it back — not even this one. Runs you start use it to reach
          the Hub; other people's runs never do. Revoke it on Hugging Face at
          any time and it stops working here immediately.
        </div>
      </div>`;
  }

  return html`
    <div class="card" style="margin-bottom:14px">
      <div class="row-between">
        <h3 style="margin:0">Hugging Face</h3>
        <button class="btn-sm" id="hfDisconnect">Disconnect</button>
      </div>
      <div class="row row-top" style="gap:12px;margin-top:12px;align-items:center">
        ${raw(hf.avatar
          ? `<img src="${esc(hf.avatar)}" alt="" width="48" height="48"
                  style="border-radius:50%">`
          : `<span class="who-avatar">${esc((hf.username || "?")[0].toUpperCase())}</span>`)}
        <div>
          <div><strong>${hf.fullname || hf.username}</strong></div>
          <div class="muted tiny mono">@${hf.username}</div>
        </div>
      </div>
      <div class="row" style="gap:6px;flex-wrap:wrap;margin-top:12px">
        <span class="badge ${hf.can_write ? "badge-ok" : "badge-warn"}">
          ${hf.can_write ? "can publish" : "read only"}</span>
        ${raw((hf.orgs || []).map((o) =>
          `<span class="badge badge-accent">${esc(o)}</span>`).join(""))}
      </div>
      ${raw(!hf.can_write ? html`
        <div class="callout callout-warn" style="margin-top:12px">
          <strong>This token cannot write</strong>
          Gated downloads work. Publishing a model or a dataset from here does
          not — make a token with write access to repositories and connect it
          instead.
        </div>` : "")}
    </div>

    <div class="card">
      <div class="row-between" style="margin-bottom:10px">
        <h3 style="margin:0">Your repositories</h3>
        <div class="row" style="gap:6px">
          <button class="btn-sm" data-repos="models">Models</button>
          <button class="btn-sm" data-repos="datasets">Datasets</button>
        </div>
      </div>
      <div id="repoList"><div class="muted tiny">Choose one to load.</div></div>
    </div>`;
}

function repoList(repos, kind) {
  if (!repos.length) {
    return `<div class="muted tiny">Nothing here yet. Publish a finished run
      from its page and it will appear.</div>`;
  }
  return html`<table class="table"><tbody>
    ${raw(repos.map((r) => html`
      <tr>
        <td>
          <a href="https://huggingface.co/${kind === "datasets" ? "datasets/" : ""}${
            r.id}" target="_blank" rel="noopener" class="mono tiny">${r.id}</a>
          ${raw(r.private ? `<span class="badge">private</span>` : "")}
          ${raw(r.gated ? `<span class="badge badge-warn">gated</span>` : "")}
        </td>
        <td class="tiny muted hide-sm">${r.downloads} downloads</td>
        <td class="tiny muted hide-sm">${fmtAgo(Date.parse(r.updated) / 1000)}</td>
        <td><button class="btn-sm btn-danger" data-delrepo="${r.id}"
                    data-kind="${kind}" title="Delete on Hugging Face">✕</button></td>
      </tr>`).join(""))}
  </tbody></table>`;
}

function shortAgent(ua) {
  if (!ua) return "unknown browser";
  const m = ua.match(/(Firefox|Edg|Chrome|Safari)\/[\d.]+/);
  const os = /Windows/.test(ua) ? "Windows" : /Mac OS/.test(ua) ? "macOS"
    : /Android/.test(ua) ? "Android" : /iPhone|iPad/.test(ua) ? "iOS"
    : /Linux/.test(ua) ? "Linux" : "";
  const browser = m ? m[0].split("/")[0].replace("Edg", "Edge") : "browser";
  return os ? `${browser} on ${os}` : browser;
}

/** Hosted models: OpenAI, Azure OpenAI, Anthropic, anything OpenAI-shaped.
 *
 *  Here rather than in Settings because a key is yours and is billed to you.
 *  What it unlocks is one thing — writing a dataset with a model far larger
 *  than anything this hardware could run — so the card says that rather than
 *  presenting itself as a general integration surface. */
function hostedCard(hosted, opening) {
  const catalogue = hosted?.providers || [];
  const connected = new Map((hosted?.connected || []).map((c) => [c.provider, c]));
  return html`
    <div class="card" style="margin-bottom:14px">
      <h3>Hosted models</h3>
      <p class="muted tiny">Connect an account and this studio can write
        datasets with a model far larger than this hardware could run — then
        train your own small one on what it wrote. Keys are encrypted, are
        never shown again, and are only ever sent to the provider you gave
        them for. Every row is billed to you.</p>

      ${raw(catalogue.map((p) => {
        const c = connected.get(p.id);
        const open = opening === p.id;
        return html`
          <div style="border-top:1px solid var(--border);padding:10px 0">
            <div class="row-between" style="gap:8px;flex-wrap:wrap">
              <div style="min-width:0">
                <strong class="tiny">${p.label}</strong>
                ${raw(c ? `<span class="badge badge-ok">connected</span>` : "")}
                ${raw(c?.scope === "studio"
                  ? `<span class="badge badge-accent" title="Configured for the
                       whole studio. Anyone who can start a job can spend it."
                     >shared with the studio</span>` : "")}
                ${raw(c?.key_hint ? `<span class="badge">key ${esc(c.key_hint)}</span>` : "")}
                <p class="muted tiny" style="margin:2px 0 0">${p.blurb}</p>
              </div>
              <div class="row" style="gap:6px">
                ${raw(c ? `<button class="btn-sm" data-test-provider="${p.id}">Test</button>` : "")}
                ${raw(c && p.lists_models
                  ? `<button class="btn-sm" data-list-models="${p.id}">Models</button>` : "")}
                <button class="btn-sm" data-open-provider="${p.id}">${
                  open ? "Close" : c ? "Edit" : "Connect"}</button>
              </div>
            </div>

            ${raw(c ? html`
              <div class="row" style="gap:6px;margin-top:8px;flex-wrap:wrap">
                <input class="mono" data-provider-model="${p.id}"
                       style="max-width:280px" placeholder="${
                         p.id === "azure" ? c.deployment : "model to test"}"
                       value="${c.model || (p.id === "azure" ? c.deployment : "")}">
                ${raw(c.scope !== "studio" || hosted.can_manage_studio
                  ? `<button class="btn-sm btn-danger" data-forget-provider="${p.id}"
                             data-scope="${esc(c.scope || "account")}">Disconnect</button>`
                  : `<span class="tiny muted">An administrator set this up for
                       everyone; only an administrator can remove it.</span>`)}
              </div>` : "")}

            <div data-provider-result="${p.id}"></div>

            ${raw(!open ? "" : html`
              <form data-provider-form="${p.id}" style="margin-top:10px">
                ${raw(p.fields.map((f) => html`
                  <div class="field">
                    <label for="pf_${p.id}_${f.name}">${f.label}${
                      raw(f.required ? "" : ` <span class="muted tiny">(optional)</span>`)}</label>
                    ${raw(f.choices ? html`
                      <select id="pf_${p.id}_${f.name}" name="${f.name}">
                        ${raw(f.choices.map((o) => `<option value="${esc(o.value)}"
                          ${(c?.[f.name] || f.choices[0].value) === o.value ? "selected" : ""}
                          >${esc(o.label)} — ${esc(o.hint || "")}</option>`).join(""))}
                      </select>` : html`
                      <input id="pf_${p.id}_${f.name}" name="${f.name}" class="mono"
                             type="${f.secret ? "password" : "text"}"
                             autocomplete="off"
                             placeholder="${f.name === "base_url" ? p.base_url : ""}"
                             value="${f.secret ? "" : (c?.[f.name] || "")}"
                             ${f.required && !(f.secret && c) ? "required" : ""}>`)}
                    <div class="hint">${f.hint}</div>
                  </div>`).join(""))}
                <div class="field">
                  <label for="pf_${p.id}_model">Default model</label>
                  <input id="pf_${p.id}_model" name="model" class="mono"
                         value="${c?.model || ""}"
                         placeholder="${p.id === "azure"
                           ? "the deployment name" : "e.g. the model you use most"}">
                  <div class="hint">Offered first when writing a dataset. You
                    can change it there.</div>
                </div>
                ${raw(hosted.can_manage_studio ? html`
                  <div class="field">
                    <label for="pf_${p.id}_scope">Who can use it</label>
                    <select id="pf_${p.id}_scope" name="scope">
                      <option value="account" ${c?.scope !== "studio" ? "selected" : ""}
                        >Just me — billed to my own key</option>
                      <option value="studio" ${c?.scope === "studio" ? "selected" : ""}
                        >Everyone in this studio</option>
                    </select>
                    <div class="hint">A studio connection is for a resource set
                      up once for the whole team. Be clear-eyed about it:
                      anyone who can start a job can spend it, and nobody but
                      an administrator can take it away again.</div>
                  </div>` : "")}
                <button class="btn-sm btn-primary" type="submit">${
                  c ? "Save changes" : "Connect"}</button>
              </form>`)}
          </div>`;
      }).join(""))}
    </div>`;
}
