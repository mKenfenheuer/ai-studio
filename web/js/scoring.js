/**
 * "Score it": the same dialog, from wherever you are looking at a model.
 *
 * The question a finished model raises — is this better than what I started
 * with — comes up on the run page and again in the playground, three messages
 * into finding out that it is not. Answering it used to mean navigating to
 * Evaluate and finding the run again in a list of every model in the studio,
 * and once there, remembering that a baseline was possible at all.
 *
 * One module rather than one copy per view: the two would have drifted, and
 * the half that matters is the baseline checkbox that neither view had.
 */
import { api } from "./api.js";
import { html, raw, esc, $, on, toast, modal } from "./util.js";

/**
 * @param {{id:string, name:string, config?:object, base_model?:string}} model
 *        A run, as either the run page or the playground holds it.
 */
export async function openScoreDialog(model) {
  let sets = [];
  try { sets = await api.evals(); }
  catch (e) { return toast(e.message, "err"); }
  if (!sets.length) {
    toast("No prompt sets yet — write one first.", "",
          { href: "#/evals", label: "Prompt sets" });
    return;
  }
  // The base it was trained from, offered in the same breath. A run fine-tuned
  // from another run in this studio has no Hub base worth offering: the thing
  // to compare it against is that run, which the Evaluate page can pick.
  const cfg = model.config || {};
  const base = (!cfg.base_model_job && (cfg.base_model || model.base_model)) || "";

  const dlg = modal({ title: `Score "${model.name}"`, width: 480, body: html`
    <p class="muted tiny">Every model you put the same prompts to can be
      compared. Pick the set to ask.</p>
    <div class="field">
      <label for="scoreSet">Prompt set</label>
      <select id="scoreSet">${raw(sets.map((e) => html`
        <option value="${e.id}">${e.name} · ${(e.items || []).length} prompts</option>`).join(""))}</select>
    </div>
    ${raw(base ? html`
      <label class="check">
        <input type="checkbox" id="scoreBase" checked>
        <span>Score ${base} beside it
          <span class="muted tiny">· the model this was trained from, asked in
            the same format. Without it the result can only say which of your
            own models won.</span></span>
      </label>` : "")}
    <details class="adv" style="margin-top:8px">
      <summary>If no machine has a graphics card</summary>
      <label class="check">
        <input type="checkbox" id="scoreCpu">
        <span>Allow a machine without one
          <span class="muted tiny">· slow, but a small model on a few dozen
            prompts is minutes, and a studio with no card at all could not
            score anything otherwise. Never chosen over a card that is
            free.</span></span>
      </label>
    </details>
    <div class="row" style="justify-content:flex-end;gap:8px;margin-top:12px">
      <button type="button" class="btn" data-modal-close>Cancel</button>
      <button type="button" class="btn btn-primary" id="scoreGo">Score it</button>
    </div>` });

  on(dlg, "click", "#scoreGo", async (_e, btn) => {
    btn.disabled = true;
    const evalId = $("#scoreSet", dlg).value;
    try {
      await api.runEval(evalId, {
        model_job_ids: [model.id],
        baselines: base && $("#scoreBase", dlg)?.checked
          ? [{ source: "hub", model: base, like_run: model.id }] : [],
        allow_cpu: !!$("#scoreCpu", dlg)?.checked,
      });
      dlg.close();
      toast("Scoring queued.", "ok",
            { href: `#/evals/${esc(evalId)}`, label: "See the results" });
    } catch (e) { toast(e.message, "err"); btn.disabled = false; }
  });
}
