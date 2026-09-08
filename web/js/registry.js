/**
 * "Serve it under a name": the same dialog, from wherever a run has just won.
 *
 * A run wins on the sweep page, on a prompt set's trend, or simply on its own
 * page, and in each place the next thing to do is the same: point the name
 * other software is configured with at it. One module, so the three places
 * cannot offer three slightly different dialogs.
 */
import { api } from "./api.js";
import { html, raw, esc, $, on, toast, modal } from "./util.js";

/** @param {{id:string, name:string}} run */
export async function openServeDialog(run, { title } = {}) {
  let existing = [];
  try { existing = await api.registeredModels(); } catch { /* offer anyway */ }
  const here = existing.filter((n) => n.job_id === run.id);
  const dlg = modal({ title: title || `Serve "${run.name}" under a name`, width: 520,
    body: html`
    <p class="muted tiny">A name other software is configured with. Point it
      here and every client follows, with nothing out there to edit.</p>
    ${raw(here.length ? html`
      <div class="callout callout-ok"><strong>Already served as</strong>
        ${raw(here.map((n) => `<code>${esc(n.alias)}</code>`).join(", "))}</div>` : "")}
    <div class="field">
      <label for="srvName">Name</label>
      <input id="srvName" type="text" class="mono" placeholder="assistant-prod"
             list="srvNames" value="${esc(existing.find((n) => n.stage === "production")?.alias || "")}">
      <datalist id="srvNames">${raw(existing.map((n) =>
        `<option value="${esc(n.alias)}"></option>`).join(""))}</datalist>
      <div class="hint">An existing name is repointed; a new one is created.
        Lowercase letters, digits, dot, dash or underscore.</div>
    </div>
    <div class="row" style="justify-content:flex-end;gap:8px;margin-top:12px">
      <button type="button" class="btn" data-modal-close>Cancel</button>
      <a class="btn" href="#/serving">All names</a>
      <button type="button" class="btn btn-primary" id="srvGo">Serve it</button>
    </div>` });
  on(dlg, "click", "#srvGo", async (_e, btn) => {
    const alias = ($("#srvName", dlg).value || "").trim().toLowerCase();
    if (!alias) return toast("Give it a name.", "err");
    btn.disabled = true;
    try {
      const r = await api.registerModel(alias, { job_id: run.id });
      dlg.close();
      toast(r.moved ? `"${alias}" now answers with ${run.name}.` : `Serving as "${alias}".`,
            "ok", { href: "#/serving", label: "Served models" });
    } catch (e) { toast(e.message, "err"); btn.disabled = false; }
  });
}
