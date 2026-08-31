/** The "publish to Hugging Face" form, shared by models and datasets.
 *
 *  There were two of these, identical apart from a noun, and the second one
 *  did not learn anything the first one did. Now that publishing can replace
 *  what is already in a repository -- a choice with consequences -- there is
 *  one form and one place where that choice is worded.
 *
 *  The repository box offers what the account already has rather than only
 *  taking a new name. Publishing an update over the thing you published last
 *  week is the common case, and it used to require remembering the name
 *  exactly; a typo made a second repository instead, silently.
 */
import { api } from "../api.js";
import { html, raw, esc, $, on, toast } from "../util.js";

export function publishCard({ kind, slug, blurb, targets = [] }) {
  const dataset = kind === "dataset";
  // A fine-tune finishes holding both the merged model and the adapter it was
  // merged from. They are two different things -- one loads on its own, the
  // other is fifty megabytes and needs its base -- and a repository holding
  // both loads as neither, so "both" means two repositories.
  const choose = targets.includes("model") && targets.includes("adapter");
  // A run whose merge did not fit on the machine that trained it has only the
  // adapter. There is nothing to choose between, but the request still has to
  // say which one it means -- the default is the merged model, and asking for
  // one that does not exist is refused.
  const only = !choose && targets.includes("adapter") ? "adapter" : "";
  return html`
    <div class="card" style="margin-bottom:14px">
      <h3>Publish to Hugging Face</h3>
      <p class="muted tiny">${blurb} Needs a connected account with write
        access — <a href="#/account">set that up here</a>.</p>
      <div id="pubBaseNote"></div>
      <form id="publishForm" style="margin-top:10px">
        ${raw(!only ? "" : html`
        <input type="hidden" name="what" value="${only}">`)}
        ${raw(!choose ? "" : html`
        <div class="field">
          <label>What to publish</label>
          <label class="check">
            <input type="radio" name="what" value="model" checked>
            <span>Merged model — loads on its own, anywhere
              <span class="muted tiny">(recommended)</span></span>
          </label>
          <label class="check">
            <input type="radio" name="what" value="adapter">
            <span>Adapter only — small, and needs the base model to load</span>
          </label>
          <label class="check">
            <input type="radio" name="what" value="both">
            <span>Both — two repositories</span>
          </label>
        </div>`)}
        <div class="field">
          <label for="pubRepo" id="pubRepoLabel">${
            only === "adapter" ? "Adapter repository" : "Repository"}</label>
          <input id="pubRepo" name="repo_id" type="text" class="mono" required
                 list="pubRepoList" autocomplete="off"
                 placeholder="your-name/${slug}">
          <datalist id="pubRepoList"></datalist>
          <div class="hint" id="pubRepoHint">A name you already own updates
            that repository; a new name creates one.</div>
        </div>
        ${raw(!choose ? "" : html`
        <div class="field" id="pubAdapterField" hidden>
          <label for="pubAdapterRepo">Adapter repository</label>
          <input id="pubAdapterRepo" name="adapter_repo_id" type="text"
                 class="mono" list="pubRepoList" autocomplete="off"
                 placeholder="your-name/${slug}-lora">
          <div class="hint">Left empty, the adapter goes to the model
            repository's name with <code>-lora</code> on the end.</div>
        </div>`)}
        <div class="field">
          <label for="pubVis">Visibility</label>
          <select id="pubVis" name="visibility">
            <option value="private">Private</option>
            <option value="public">Public — anyone can download it</option>
          </select>
          <div class="hint">Only applied when the repository is created.
            Hugging Face does not let this flip an existing one.</div>
        </div>
        <label class="check">
          <input type="checkbox" name="replace" id="pubReplace">
          <span>Replace everything already there</span>
        </label>
        <div class="hint" style="margin:-4px 0 10px">
          Files in the repository that this upload does not include are deleted,
          in the same commit. Leave it off to add to what is there.
          ${raw(dataset ? "" : html`Worth turning on when publishing a merged
            model over an adapter: the old files load in preference to the new
            ones otherwise.`)}
        </div>
        <button class="btn-sm" type="submit" id="pubGo">
          Publish to Hugging Face</button>
      </form>
      <div id="publishResult"></div>
    </div>`;
}

/** Wire the form.
 *
 *  `send(body)` creates the upload run and returns it. `loadInfo`, when given,
 *  returns this run's publish-info: what it could publish, and what its base
 *  model is doing on the Hub.
 */
export function wirePublish(mount, kind, send, loadInfo = null) {
  // Best-effort: the form works without either of these, and an account that
  // is not connected yet has nothing to offer. Filled through a function
  // rather than once, because the card around it is repainted whenever the run
  // changes and that replaces the elements this wrote into.
  let repos = null, info = null;
  const fill = () => {
    const list = $("#pubRepoList", mount);
    if (list && repos && list.children.length !== repos.length) {
      list.innerHTML = repos.map((r) => html`
        <option value="${r.id}">${r.private ? "private" : "public"}</option>`)
        .join("");
    }
    const note = $("#pubBaseNote", mount);
    if (note) note.innerHTML = baseNote(info);
  };
  api.hfRepos(kind === "dataset" ? "datasets" : "models")
    .then((r) => { repos = r; fill(); })
    .catch(() => {});
  if (loadInfo) loadInfo().then((r) => { info = r; fill(); }).catch(() => {});
  on(mount, "input", "#pubRepo", fill);

  // "Both" is the only choice that needs a second name, so the second box is
  // only there when it is the one being asked for.
  on(mount, "change", "input[name=what]", (_e, el) => {
    const field = $("#pubAdapterField", mount);
    if (field) field.hidden = el.value !== "both";
    const label = $("#pubRepoLabel", mount);
    if (label) {
      label.textContent = el.value === "adapter" ? "Adapter repository"
        : el.value === "both" ? "Model repository" : "Repository";
    }
  });

  on(mount, "submit", "#publishForm", async (e) => {
    e.preventDefault();
    const f = Object.fromEntries(new FormData(e.target).entries());
    const btn = $("#pubGo", mount);
    btn.disabled = true;
    btn.textContent = "Queueing…";
    try {
      const r = await send({
        repo_id: (f.repo_id || "").trim(),
        private: f.visibility === "private",
        replace: !!f.replace,
        what: f.what || "model",
        adapter_repo_id: (f.adapter_repo_id || "").trim(),
      });
      // Uploading is a run now, with a log and a progress bar, so the useful
      // thing to do here is go and watch it rather than report "started".
      // "Both" queues two of them; the first one is what opens.
      toast(r.queued?.length > 1
        ? "Publishing both. Watching the model upload."
        : "Publishing. Watch it here.", "ok");
      location.hash = `#/jobs/${r.job_id}`;
    } catch (ex) {
      $("#publishResult", mount).innerHTML =
        `<div class="callout callout-err">${esc(ex.message)}</div>`;
      btn.disabled = false;
      btn.textContent = "Publish to Hugging Face";
    }
  });
}

/** The "publish the base first" recommendation, or nothing.
 *
 *  A card can only name a `base_model:` the Hub can resolve, which means a
 *  repository. When the base is a run on this machine that has never been
 *  published there is no such name, so the card drops the line and what
 *  appears on the Hub is a model with no parentage -- silently. Publishing the
 *  base first is the fix, and it has to happen first: the line is written when
 *  this upload is queued, not later.
 *
 *  It is a recommendation. The form still submits, because a private base that
 *  is never going to be published is a legitimate thing to have.
 */
function baseNote(info) {
  const base = info?.base;
  if (!base || !base.job_id || base.published) return "";
  return html`
    <div class="callout" style="margin-top:8px">
      This was trained from <a href="#/jobs/${base.job_id}">${base.name}</a>,
      which is not on Hugging Face yet. Publish that one first and this
      repository will name it as its base model — the Hub then shows this as a
      fine-tune of it. Published now, the card names the base in prose only.
    </div>`;
}
