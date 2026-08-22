/**
 * The sharing panel, used by both runs and datasets.
 *
 * One component for both because the question is the same in both places, and
 * two near-identical panels would drift apart within a month.
 */
import { api } from "../api.js";
import { html, raw, esc, $, on, toast } from "../util.js";

export function shareBox(kind, resource) {
  const shares = resource.shares || [];
  const owned = resource.mine || resource.access === "edit";
  const everyone = shares.find((s) => s.subject_type === "everyone");
  const people = shares.filter((s) => s.subject_type === "user");

  if (!owned && !shares.length) return "";

  return html`
    <div class="card" style="margin-bottom:14px">
      <h3>Who can see this</h3>
      ${raw(!resource.mine ? html`
        <p class="muted tiny">Shared with you by
          ${esc(resource.owner_name || resource.owner?.display_name || "someone")}.
          You have ${resource.access === "edit" ? "edit" : "read-only"} access.</p>`
      : html`
        <p class="muted tiny">Yours, and private, unless you add somebody
          here. Everyone you add can see it; only you can share or delete it.</p>

        <div class="row row-top" style="margin-top:10px;gap:6px;flex-wrap:wrap">
          <select id="shareWho" style="flex:1;min-width:150px">
            <option value="">Choose a person…</option>
          </select>
          <select id="shareLevel" style="max-width:130px">
            <option value="view">Can view</option>
            <option value="edit">Can edit</option>
          </select>
          <button class="btn-sm btn-primary" id="shareAdd">Share</button>
        </div>

        <label class="check" style="margin-top:10px">
          <input type="checkbox" id="shareEveryone" ${everyone ? "checked" : ""}>
          Everyone with an account on this studio can view it
        </label>`)}

      ${raw(people.length ? html`
        <table class="table" style="margin-top:10px"><tbody>
          ${raw(people.map((s) => html`
            <tr>
              <td>${esc(s.display_name || s.username || "unknown")}
                <div class="muted tiny mono">${esc(s.username || "")}</div></td>
              <td><span class="badge ${s.level === "edit" ? "badge-accent" : ""}">
                ${s.level === "edit" ? "can edit" : "can view"}</span></td>
              ${raw(resource.mine ? html`
                <td><button class="btn-sm btn-danger" data-unshare="${esc(s.subject_id)}"
                            title="Stop sharing with this person">✕</button></td>` : "<td></td>")}
            </tr>`).join(""))}
        </tbody></table>` : "")}
    </div>`;
}

export function wireShareBox(mount, kind, resource, refresh) {
  const select = $("#shareWho", mount);
  if (select) {
    // Loaded once, after render, so the panel appears immediately rather than
    // waiting on a request that most visits never need.
    api.users().then((users) => {
      const already = new Set((resource.shares || [])
        .filter((s) => s.subject_type === "user").map((s) => s.subject_id));
      const ownerId = resource.owner_id || resource.owner?.id;
      users
        .filter((u) => u.active && u.id !== ownerId && !already.has(u.id))
        .forEach((u) => {
          const o = document.createElement("option");
          o.value = u.id;
          o.textContent = `${u.display_name} (${u.username})`;
          select.appendChild(o);
        });
    }).catch(() => { /* the panel still works without the list */ });
  }

  on(mount, "click", "#shareAdd", async () => {
    const who = $("#shareWho", mount).value;
    if (!who) return toast("Choose somebody first.", "err");
    try {
      await api.addShare(kind, resource.id, {
        subject_type: "user", subject_id: who,
        level: $("#shareLevel", mount).value });
      toast("Shared.", "ok");
      await refresh();
    } catch (ex) { toast(ex.message, "err"); }
  });

  on(mount, "change", "#shareEveryone", async (_e, t) => {
    try {
      if (t.checked) {
        await api.addShare(kind, resource.id,
                           { subject_type: "everyone", level: "view" });
        toast("Everyone here can now see it.", "ok");
      } else {
        await api.removeShare(kind, resource.id, "everyone");
        toast("No longer shared with everyone.", "ok");
      }
      await refresh();
    } catch (ex) { toast(ex.message, "err"); await refresh(); }
  });

  on(mount, "click", "[data-unshare]", async (_e, t) => {
    try {
      await api.removeShare(kind, resource.id, "user", t.dataset.unshare);
      await refresh();
    } catch (ex) { toast(ex.message, "err"); }
  });
}
