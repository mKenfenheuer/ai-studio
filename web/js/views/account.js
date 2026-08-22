/** Your account: your name, your password, how you are told a run ended,
 *  and your Hugging Face connection. */
import { api } from "../api.js";
import { html, raw, esc, $, $$, on, toast, fmtAgo,
         askForNotifications } from "../util.js";

export async function accountView(mount) {
  let me = await api.me();
  let alerts = await api.notifyState();
  let keys = await api.apiKeys();
  // The one time a key is visible. Held in memory only, and dropped as soon
  // as the page is left -- there is nowhere it could be stored that would not
  // be a worse place than the user's own password manager.
  let freshKey = null;

  const draw = () => {
    mount.innerHTML = layout(me, alerts, keys, freshKey);
    wire();
  };

  function wire() {
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

function layout(me, alerts, keys, freshKey) {
  const hf = me.hf || {};
  return html`
    <div class="page-head">
      <h1>Your account</h1>
      <p class="sub">Signed in as <strong>${me.username}</strong>${
        me.role === "admin" ? " · administrator" : ""}</p>
    </div>

    <div class="grid grid-2" style="align-items:start">
      <div>
        <div class="card" style="margin-bottom:14px">
          <h3>Name</h3>
          <p class="muted tiny">Shown next to your runs and when somebody
            shares something with you.</p>
          <form id="nameForm" class="row row-top" style="margin-top:10px">
            <input type="text" id="displayName" value="${esc(me.display_name)}"
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
          <p class="muted tiny">${me.sessions.length} active session${
            me.sessions.length === 1 ? "" : "s"}.</p>
          <ul class="muted tiny" style="margin:8px 0;padding-left:18px;line-height:1.7">
            ${raw(me.sessions.slice(0, 6).map((s) => html`
              <li>${esc(shortAgent(s.user_agent))} — last used ${
                esc(fmtAgo(s.last_used))}</li>`).join(""))}
          </ul>
          <button class="btn-sm btn-danger" id="revoke">Sign out everywhere else</button>
        </div>
      </div>

      <div>
        ${raw(keysCard(keys, freshKey))}
        ${raw(alertCard(alerts))}
        ${raw(hfCard(hf))}
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
        <input type="text" class="mono" readonly value="${esc(origin)}/v1">
        <div class="hint">Model names come from
          <code>GET ${esc(origin)}/v1/models</code> — each finished run is one,
          by its id or its name.</div>
      </div>

      ${raw(fresh ? html`
        <div class="callout callout-ok">
          <strong>Copy this now</strong>
          It is stored hashed and cannot be shown again. If you lose it,
          delete it and make another.
          <div class="row" style="gap:6px;margin-top:8px">
            <input type="text" class="mono" readonly id="freshKey"
                   value="${esc(fresh.key)}" style="flex:1">
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
                ? `used ${fmtAgo(k.last_used)}` : "never used"}</td>
              <td><button class="btn-sm btn-danger" data-del-key="${esc(k.id)}"
                          title="Delete this key">✕</button></td>
            </tr>`).join(""))}
        </tbody></table>` : "")}

      <details class="adv" style="margin-top:10px">
        <summary>How to point something at it</summary>
        <pre class="code tiny">curl ${esc(origin)}/v1/chat/completions \
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
                 placeholder="${a.configured ? esc(a.url_hint)
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
          <div><strong>${esc(hf.fullname || hf.username)}</strong></div>
          <div class="muted tiny mono">@${esc(hf.username)}</div>
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
            esc(r.id)}" target="_blank" rel="noopener" class="mono tiny">${esc(r.id)}</a>
          ${raw(r.private ? `<span class="badge">private</span>` : "")}
          ${raw(r.gated ? `<span class="badge badge-warn">gated</span>` : "")}
        </td>
        <td class="tiny muted hide-sm">${r.downloads} downloads</td>
        <td class="tiny muted hide-sm">${esc(fmtAgo(Date.parse(r.updated) / 1000))}</td>
        <td><button class="btn-sm btn-danger" data-delrepo="${esc(r.id)}"
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
