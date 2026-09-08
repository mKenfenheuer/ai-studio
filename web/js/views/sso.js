/**
 * Connecting the studio to the place your accounts already live.
 *
 * Setting up OpenID Connect is a two-sided job — half of it happens at the
 * provider, in a console this page cannot see — and every failure mode is a
 * mismatch between the two halves. So this screen is built around that: the
 * redirect URI is shown *before* anything is saved, in its final form, ready
 * to paste; discovery runs on save and reports what the provider actually
 * published; and a directory sync can be run as a dry run first, which reads
 * everything and writes nothing.
 *
 * Secrets go in and never come back out. The form shows whether one is set,
 * never what it is, and leaving the field empty on an edit keeps the one
 * already stored rather than clearing it.
 */
import { api } from "../api.js";
import { html, raw, esc, $, $$, on, toast, modal, fmtAgo } from "../util.js";
import { ribbon, rb, group } from "../ribbon.js";
import { breadcrumb } from "../components.js";

export async function ssoView(mount) {
  let idps = [];
  let presets = {};

  const paint = async () => {
    [idps, presets] = await Promise.all([api.idps(), api.idpPresets()]);
    mount.innerHTML = layout(idps, presets);
  };
  await paint();

  on(mount, "click", "#addIdp", () => chooseKind(presets, paint));
  on(mount, "click", "[data-edit]", (_e, t) =>
    openForm(idps.find((i) => i.id === t.dataset.edit), presets, paint));

  on(mount, "click", "[data-toggle]", async (_e, t) => {
    try {
      await api.updateIdp(t.dataset.toggle, { enabled: t.dataset.to === "1" });
      await paint();
    } catch (e) { toast(e.message, "err"); }
  });

  on(mount, "click", "[data-rediscover]", async (_e, t) => {
    t.disabled = true;
    try {
      await api.rediscoverIdp(t.dataset.rediscover);
      toast("Read the provider's configuration again.", "ok");
      await paint();
    } catch (e) { toast(e.message, "err"); t.disabled = false; }
  });

  on(mount, "click", "[data-sync]", async (_e, t) => {
    const dry = t.dataset.dry === "1";
    t.disabled = true;
    t.textContent = dry ? "Reading…" : "Syncing…";
    try {
      const r = await api.syncIdp(t.dataset.sync, dry);
      toast(dry
        ? `Read ${r.read} people. ${r.created} would be new. Nothing changed.`
        : `${r.read} read · ${r.created} added · ${r.updated} updated · `
          + `${r.deactivated} disabled.`, r.refused ? "err" : "ok");
      await paint();
    } catch (e) { toast(e.message, "err"); await paint(); }
  });

  on(mount, "click", "[data-del-idp]", async (_e, t) => {
    const idp = idps.find((i) => i.id === t.dataset.delIdp);
    if (!confirm(`Remove ${idp.name}?\n\n${idp.people} account(s) came from `
                 + `it. They keep everything they own, but they will not be `
                 + `able to sign in until somebody gives them a password.`)) return;
    try {
      const r = await api.deleteIdp(idp.id);
      toast(r.note, "ok");
      await paint();
    } catch (e) { toast(e.message, "err"); }
  });

  on(mount, "click", "[data-copy]", (_e, t) => {
    navigator.clipboard.writeText(t.dataset.copy)
      .then(() => toast("Copied.", "ok"))
      .catch(() => toast("Could not copy.", "err"));
  });
}

// ---------------------------------------------------------------------------

function layout(idps, presets) {
  return html`
    <div class="page-head">
      ${raw(breadcrumb({ href: "#/settings", label: "Settings" }))}
      <h1 style="margin:6px 0 0">Single sign-on</h1>
      <p class="sub">Let people in with the account they already have, and
        find colleagues by name when sharing — without keeping a second list
        of who works here.</p>
    </div>
    ${raw(ribbon({
      tabs: [{ key: "home", label: "Sign-in methods" }], active: "home",
      body: group("Methods", [
        rb("addIdp", "＋", "Add a method", { cls: "primary" }),
      ]) + group("Elsewhere", [
        rb(null, "◍", "People", { href: "#/users" }),
        rb(null, "⚙", "Settings", { href: "#/settings" }),
      ]),
    }))}

    ${raw(!idps.length ? empty() : idps.map(card).join(""))}

    <div class="callout" style="margin-top:16px">
      <strong>Passwords keep working.</strong>
      Connecting a provider never takes the password form away. A studio whose
      identity provider is unreachable at eight in the morning must not be a
      studio nobody can get into — and somebody has to be able to fix it.
    </div>`;
}

const empty = () => html`
  <div class="card">
    <div class="empty" style="padding:28px 16px">
      <div class="big">\u{1F511}</div>
      <h3>Nobody signs in from outside yet</h3>
      <p class="muted" style="max-width:52ch;margin:0 auto">
        Connect Entra ID, Google Workspace, Okta, Keycloak or anything else
        that speaks OpenID Connect. It takes two values from your provider \u2014
        an application ID and a secret \u2014 and one address pasted back the
        other way.</p>
    </div>
  </div>`;

function card(idp) {
  const ok = idp.endpoints?.authorization_endpoint;
  return html`
    <div class="card" style="margin-bottom:14px">
      <div class="row-between" style="gap:10px;flex-wrap:wrap;align-items:flex-start">
        <div style="min-width:0">
          <h3 style="margin-bottom:2px">${idp.name}
            ${raw(idp.enabled
              ? `<span class="badge badge-ok">on</span>`
              : `<span class="badge">off</span>`)}
          </h3>
          <p class="muted tiny mono" style="margin:0;word-break:break-all">${idp.issuer}</p>
        </div>
        <div class="row" style="gap:6px;flex-wrap:wrap">
          <button class="btn-sm" data-edit="${idp.id}">Settings</button>
          <button class="btn-sm" data-toggle="${idp.id}"
                  data-to="${idp.enabled ? "0" : "1"}">
            ${idp.enabled ? "Turn off" : "Turn on"}</button>
          <button class="btn-sm btn-danger" data-del-idp="${idp.id}">Remove</button>
        </div>
      </div>

      <div class="factrow" style="margin-top:12px">
        <div class="fact"><div class="k">People</div>
          <div class="v">${idp.people}</div>
          <div class="n">${idp.pending} not signed in yet</div></div>
        <div class="fact"><div class="k">New accounts</div>
          <div class="v">${idp.auto_create ? "Created" : "Refused"}</div>
          <div class="n">${idp.allowed_domains
            ? idp.allowed_domains : "any address"}</div></div>
        <div class="fact"><div class="k">Directory</div>
          <div class="v">${idp.sync_enabled ? "Syncing" : (idp.directory_kind ? "Off" : "n/a")}</div>
          <div class="n">${idp.last_sync_at ? fmtAgo(idp.last_sync_at)
            : (idp.directory_kind || "not available")}</div></div>
        <div class="fact"><div class="k">Configuration</div>
          <div class="v">${ok ? "Read" : "Missing"}</div>
          <div class="n">${idp.discovered_at ? fmtAgo(idp.discovered_at) : "—"}</div></div>
      </div>

      <div class="field" style="margin-top:6px">
        <label>Redirect URI — this exact string goes in the provider</label>
        <div class="row row-top">
          <input readonly value="${idp.redirect_uri}">
          <button class="btn-sm" data-copy="${idp.redirect_uri}">Copy</button>
        </div>
        <div class="hint">Providers compare it character for character. A
          mismatch is the single most common reason a sign-in fails, and it
          always fails at the provider, before this studio hears anything.</div>
      </div>

      ${raw(idp.last_sync_note ? html`
        <p class="muted tiny" style="margin:8px 0 0">
          Last sync: ${idp.last_sync_note}</p>` : "")}

      <div class="row" style="gap:6px;margin-top:10px;flex-wrap:wrap">
        <button class="btn-sm" data-rediscover="${idp.id}">Re-read configuration</button>
        ${raw(idp.directory_kind ? html`
          <button class="btn-sm" data-sync="${idp.id}" data-dry="1">Preview a sync</button>
          <button class="btn-sm btn-primary" data-sync="${idp.id}">Sync people now</button>`
        : html`<span class="muted tiny" style="align-self:center">
            This kind of provider has no list of people to read, so accounts
            here appear as people sign in.</span>`)}
      </div>
    </div>`;
}

// ---------------------------------------------------------------------------
// Adding one
// ---------------------------------------------------------------------------

function chooseKind(presets, refresh) {
  const dlg = modal({ title: "What are you connecting?", width: 560 });
  $(".modal-body", dlg).innerHTML = html`
    <div class="picklist">
      ${raw(Object.entries(presets.presets || {}).map(([kind, p]) => html`
        <button class="kind-option" data-kind="${kind}">
          <strong>${p.label}</strong>
          <span class="muted tiny">${p.blurb}</span>
          ${raw(p.directory ? `<span class="badge badge-soft">reads your
            directory</span>` : "")}
        </button>`).join(""))}
    </div>`;
  on(dlg, "click", "[data-kind]", (_e, t) => {
    dlg.close();
    openForm(null, presets, refresh, t.dataset.kind);
  });
}

function openForm(existing, presets, refresh, kind) {
  kind = kind || existing?.kind || "oidc";
  const preset = (presets.presets || {})[kind] || {};
  const redirect = existing?.redirect_uri
    || (presets.redirect_uris || {})[kind] || "";
  const dlg = modal({
    title: existing ? `${existing.name} settings` : `Connect ${preset.label}`,
    width: 640,
  });

  $(".modal-body", dlg).innerHTML = html`
    <div class="callout" style="margin-bottom:14px">
      <strong>First, at ${preset.label}.</strong>
      Register an application and give it this redirect URI:
      <div class="row row-top" style="margin-top:8px">
        <input readonly value="${redirect}">
        <button class="btn-sm" type="button" data-copy-redirect>Copy</button>
      </div>
      ${raw(preset.docs ? html`<div class="hint">${preset.docs}</div>` : "")}
    </div>

    <form id="idpForm">
      <div class="field">
        <label for="f_name">Name on the sign-in button</label>
        <input id="f_name" name="name" value="${existing?.name || preset.label || ""}"
               placeholder="${preset.label || "Single sign-on"}">
      </div>

      ${raw(preset.issuer_template === "{tenant}" || kind !== "oidc" ? html`
        <div class="field">
          <label for="f_tenant">${preset.tenant_label || "Issuer URL"}</label>
          <input id="f_tenant" name="tenant" required
                 value="${existing?.tenant || ""}">
          <div class="hint">${preset.tenant_hint || ""}</div>
        </div>` : "")}

      <div class="field">
        <label for="f_client">Application (client) ID</label>
        <input id="f_client" name="client_id" required
               value="${existing?.client_id || ""}">
      </div>

      <div class="field">
        <label for="f_secret">Client secret</label>
        <input id="f_secret" name="client_secret" type="password"
               autocomplete="new-password"
               placeholder="${existing?.has_secret
                 ? "stored — leave blank to keep it" : ""}">
        <div class="hint">Encrypted before it is written down, and returned by
          no page here, including this one.</div>
      </div>

      <div class="field">
        <label for="f_domains">Only these email domains may sign in</label>
        <input id="f_domains" name="allowed_domains"
               value="${existing?.allowed_domains || ""}"
               placeholder="example.com, example.org">
        <div class="hint">${raw(kind === "google" ? html`<strong>Set this.</strong>
          Google will happily authenticate every Gmail account on earth;
          without a domain here, all of them may sign in.`
          : "Leave empty to accept every address the provider vouches for.")}</div>
      </div>

      <div class="field">
        <label for="f_admins">Groups whose members become administrators</label>
        <input id="f_admins" name="admin_groups"
               value="${existing?.admin_groups || ""}"
               placeholder="AI Studio Admins">
        <div class="hint">Promotion only, never demotion — a group claim that
          is briefly missing must not strip you of your own studio.</div>
      </div>

      <label class="check"><input type="checkbox" name="auto_create"
        ${existing?.auto_create !== false ? "checked" : ""}>
        Create an account the first time somebody signs in</label>
      <label class="check"><input type="checkbox" name="link_by_email"
        ${existing?.link_by_email !== false ? "checked" : ""}>
        Attach to an existing account with the same email address</label>

      ${raw(!preset.directory ? "" : html`
        <h4 style="margin:18px 0 6px">Read the list of people</h4>
        <p class="muted tiny" style="margin:0 0 8px">Imports everybody as a
          findable account so you can share a run with a colleague before they
          have ever opened the studio. Runs hourly.</p>
        <label class="check"><input type="checkbox" name="sync_enabled"
          ${existing?.sync_enabled ? "checked" : ""}>
          Sync from ${preset.directory}</label>
        <div class="field">
          <label for="f_group">${kind === "google"
            ? "Read the directory as (an administrator address)"
            : "Only import members of this group (optional)"}</label>
          <input id="f_group" name="${kind === "google" ? "sync_subject" : "sync_group"}"
                 value="${(kind === "google" ? existing?.sync_subject
                                                : existing?.sync_group) || ""}"
                 placeholder="${kind === "google" ? "admin@example.com"
                   : "AI Studio Users"}">
          <div class="hint">${kind === "google"
            ? "Google's directory can only be read as a real administrator, "
              + "through domain-wide delegation."
            : "Without one, everybody in the tenant is imported."}</div>
        </div>
        <div class="field">
          <label for="f_syncsecret">${kind === "google"
            ? "Service account key (JSON)" : "Directory secret (optional)"}</label>
          ${raw(kind === "google"
            ? html`<textarea id="f_syncsecret" name="sync_secret" rows="4"
                     placeholder="${existing?.has_sync_secret
                       ? "stored — leave blank to keep it"
                       : "paste the whole JSON key file"}"></textarea>`
            : html`<input id="f_syncsecret" name="sync_secret" type="password"
                     autocomplete="new-password"
                     placeholder="${existing?.has_sync_secret
                       ? "stored — leave blank to keep it"
                       : "leave blank to reuse the client secret above"}">`)}
          <div class="hint">${kind === "google"
            ? "Needs the admin.directory.user.readonly scope."
            : "The app registration needs the Graph application permission "
              + "User.Read.All, granted with admin consent."}</div>
        </div>`)}

      <div id="idpError"></div>
      <div class="row" style="gap:8px;margin-top:16px;justify-content:flex-end">
        <button type="button" class="btn" data-modal-close>Cancel</button>
        <button type="submit" class="btn-primary" id="idpSave">
          ${existing ? "Save" : "Connect"}</button>
      </div>
    </form>`;

  on(dlg, "click", "[data-copy-redirect]", () => {
    navigator.clipboard.writeText(redirect)
      .then(() => toast("Copied.", "ok"))
      .catch(() => toast("Could not copy.", "err"));
  });

  $("#idpForm", dlg).addEventListener("submit", async (e) => {
    e.preventDefault();
    const form = e.target;
    const body = { kind };
    $$("input, textarea", form).forEach((el) => {
      if (!el.name) return;
      body[el.name] = el.type === "checkbox" ? el.checked : el.value.trim();
    });
    // An empty secret box on an edit means "keep the one you have", not
    // "delete it". Only a create may send an empty string.
    if (existing) {
      if (!body.client_secret) delete body.client_secret;
      if (!body.sync_secret) delete body.sync_secret;
      if (body.tenant === (existing.tenant || "")) delete body.tenant;
    }
    const btn = $("#idpSave", dlg);
    btn.disabled = true;
    btn.textContent = "Talking to the provider…";
    try {
      if (existing) await api.updateIdp(existing.id, body);
      else await api.createIdp(body);
      dlg.close();
      toast(existing ? "Saved." : "Connected. Try it from a private window.", "ok");
      await refresh();
    } catch (ex) {
      $("#idpError", dlg).innerHTML =
        `<div class="callout callout-err">${esc(ex.message)}</div>`;
      btn.disabled = false;
      btn.textContent = existing ? "Save" : "Connect";
    }
  });
}
