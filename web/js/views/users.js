/** Administering accounts. Visible only to administrators. */
import { api } from "../api.js";
import { html, raw, esc, $, on, toast, fmtAgo } from "../util.js";
import { session } from "../app.js";

export async function usersView(mount) {
  if (session.user?.role !== "admin") {
    mount.innerHTML = html`
      <div class="callout callout-err"><strong>Administrators only</strong>
        Managing accounts needs an administrator account.</div>`;
    return;
  }

  let users = await api.users();

  const draw = () => {
    mount.innerHTML = layout(users);
    wire();
  };

  const refresh = async () => { users = await api.users(); draw(); };

  function wire() {
    on(mount, "submit", "#newUser", async (e) => {
      e.preventDefault();
      const f = Object.fromEntries(new FormData(e.target).entries());
      try {
        await api.createUser({ ...f, must_change: true });
        toast(`Created ${f.username}. Give them that password — they will be `
              + "asked to change it when they first sign in.", "ok");
        e.target.reset();
        await refresh();
      } catch (ex) { toast(ex.message, "err"); }
    });

    on(mount, "change", "[data-role]", async (_e, t) => {
      try {
        await api.updateUser(t.dataset.role, { role: t.value });
        toast("Role changed.", "ok");
        await refresh();
      } catch (ex) { toast(ex.message, "err"); await refresh(); }
    });

    on(mount, "click", "[data-toggle]", async (_e, t) => {
      const on_ = t.dataset.active === "1";
      if (on_ && !confirm("Disable this account? They will be signed out "
                          + "everywhere and cannot sign back in.")) return;
      try {
        await api.updateUser(t.dataset.toggle, { active: !on_ });
        await refresh();
      } catch (ex) { toast(ex.message, "err"); }
    });

    on(mount, "click", "[data-reset]", async (_e, t) => {
      const pw = prompt("Temporary password for this account?\n\n"
                        + "At least 10 characters. They will have to replace "
                        + "it before they can do anything.");
      if (!pw) return;
      try {
        const r = await api.resetUserPassword(t.dataset.reset, pw);
        toast("Password reset. Sessions signed out"
              + (r.keys_revoked
                 ? `, and ${r.keys_revoked} API key${r.keys_revoked === 1
                    ? "" : "s"} revoked — anything using them stops working.`
                 : "."), "ok");
        await refresh();
      } catch (ex) { toast(ex.message, "err"); }
    });

    on(mount, "click", "[data-del]", async (_e, t) => {
      if (!confirm(`Delete ${t.dataset.name}?\n\n`
                   + "Their runs and datasets are kept, and become unowned — "
                   + "nothing they trained is destroyed.")) return;
      try {
        const r = await api.deleteUser(t.dataset.del);
        toast(r.note || "Deleted.", "ok");
        await refresh();
      } catch (ex) { toast(ex.message, "err"); }
    });
  }

  draw();
}

function layout(users) {
  const admins = users.filter((u) => u.role === "admin" && u.active).length;
  return html`
    <div class="page-head">
      <h1>People</h1>
      <p class="sub">${users.length} account${users.length === 1 ? "" : "s"} ·
        ${admins} administrator${admins === 1 ? "" : "s"}</p>
    </div>

    <div class="card" style="margin-bottom:14px">
      <h3>Add someone</h3>
      <p class="muted tiny">They sign in with this password once, then choose
        their own. Their runs and datasets are private to them until they
        share them.</p>
      <form id="newUser" class="grid grid-4" style="margin-top:12px;align-items:end">
        <div class="field">
          <label for="nu">Username</label>
          <input id="nu" name="username" type="text" required
                 autocapitalize="none" spellcheck="false"
                 placeholder="lower case">
        </div>
        <div class="field">
          <label for="nn">Name</label>
          <input id="nn" name="display_name" type="text" placeholder="optional">
        </div>
        <div class="field">
          <label for="np">Temporary password</label>
          <input id="np" name="password" type="text" class="mono" required
                 minlength="10" placeholder="at least 10 characters">
        </div>
        <div class="field">
          <label for="nr">Role</label>
          <select id="nr" name="role">
            <option value="member">Member</option>
            <option value="admin">Administrator</option>
          </select>
        </div>
      </form>
      <button class="btn-primary btn-sm" type="submit" form="newUser"
              style="margin-top:10px">Create account</button>
    </div>

    <div class="card">
      <table class="table">
        <thead><tr>
          <th>Person</th><th>Role</th><th class="hide-sm">Last signed in</th>
          <th class="hide-sm">Hugging Face</th><th></th>
        </tr></thead>
        <tbody>
          ${raw(users.map((u) => row(u)).join(""))}
        </tbody>
      </table>
      <p class="muted tiny" style="margin-top:10px">
        An administrator can see and manage every run and dataset in the
        studio. A member sees their own, plus whatever has been shared with
        them.</p>
    </div>`;
}

function row(u) {
  const me = session.user?.id === u.id;
  return html`
    <tr style="${u.active ? "" : "opacity:.55"}">
      <td>
        <strong>${esc(u.display_name)}</strong>
        ${raw(me ? `<span class="badge badge-accent">you</span>` : "")}
        ${raw(u.active ? "" : `<span class="badge">disabled</span>`)}
        <div class="muted tiny mono">${esc(u.username)}</div>
      </td>
      <td>
        <select data-role="${esc(u.id)}" ${me ? "disabled" : ""}
                title="${me ? "You cannot change your own role." : ""}">
          <option value="member"${u.role === "member" ? " selected" : ""}>Member</option>
          <option value="admin"${u.role === "admin" ? " selected" : ""}>Administrator</option>
        </select>
      </td>
      <td class="tiny muted hide-sm">${u.last_login ? esc(fmtAgo(u.last_login)) : "never"}</td>
      <td class="tiny muted hide-sm">${raw(u.hf?.connected
        ? `<span class="badge badge-ok">@${esc(u.hf.username)}</span>`
        : `<span class="muted">not connected</span>`)}</td>
      <td>
        <div class="row" style="gap:5px;flex-wrap:wrap">
          <button class="btn-sm" data-reset="${esc(u.id)}">Reset password</button>
          ${raw(me ? "" : html`
            <button class="btn-sm" data-toggle="${esc(u.id)}"
                    data-active="${u.active ? 1 : 0}">
              ${u.active ? "Disable" : "Enable"}</button>
            <button class="btn-sm btn-danger" data-del="${esc(u.id)}"
                    data-name="${esc(u.display_name)}">✕</button>`)}
        </div>
      </td>
    </tr>`;
}
