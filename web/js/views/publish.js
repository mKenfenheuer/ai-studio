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

export function publishCard({ kind, slug, blurb }) {
  const dataset = kind === "dataset";
  return html`
    <div class="card" style="margin-bottom:14px">
      <h3>Publish to Hugging Face</h3>
      <p class="muted tiny">${blurb} Needs a connected account with write
        access — <a href="#/account">set that up here</a>.</p>
      <form id="publishForm" style="margin-top:10px">
        <div class="field">
          <label for="pubRepo">Repository</label>
          <input id="pubRepo" name="repo_id" type="text" class="mono" required
                 list="pubRepoList" autocomplete="off"
                 placeholder="your-name/${slug}">
          <datalist id="pubRepoList"></datalist>
          <div class="hint" id="pubRepoHint">A name you already own updates
            that repository; a new name creates one.</div>
        </div>
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

/** Wire the form. `send(body)` creates the upload run and returns it. */
export function wirePublish(mount, kind, send) {
  // Best-effort: the form works without it, and an account that is not
  // connected yet has nothing to offer. Filled through a function rather than
  // once, because the card around it is repainted whenever the run changes and
  // that replaces the element this wrote into.
  let repos = null;
  const fill = () => {
    const list = $("#pubRepoList", mount);
    if (!list || !repos || list.children.length === repos.length) return;
    list.innerHTML = repos.map((r) => html`
      <option value="${r.id}">${r.private ? "private" : "public"}</option>`)
      .join("");
  };
  api.hfRepos(kind === "dataset" ? "datasets" : "models")
    .then((r) => { repos = r; fill(); })
    .catch(() => {});
  on(mount, "input", "#pubRepo", fill);

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
      });
      // Uploading is a run now, with a log and a progress bar, so the useful
      // thing to do here is go and watch it rather than report "started".
      // What is being watched may be the merge: a fine-tune publishes its
      // merged model, and the upload starts by itself when that lands.
      toast(r.pending ? (r.message || "Merging first.") : "Publishing. Watch it here.",
            "ok");
      location.hash = `#/jobs/${r.job_id}`;
    } catch (ex) {
      $("#publishResult", mount).innerHTML =
        `<div class="callout callout-err">${esc(ex.message)}</div>`;
      btn.disabled = false;
      btn.textContent = "Publish to Hugging Face";
    }
  });
}
