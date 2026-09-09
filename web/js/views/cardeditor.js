/**
 * The model card, edited the way it will be read.
 *
 * A card is a README with YAML front matter, and the Hub renders it. Editing
 * it in a plain textarea means proof-reading pipe characters and guessing at
 * what the metadata block does; so this shows the rendered card beside the
 * source, and puts the handful of fields that actually matter -- licence,
 * languages, tags, the datasets it was trained on -- in a form.
 *
 * The generated card is a floor, not a ceiling. It is written from what this
 * studio knows and rewritten whenever those facts change, until somebody
 * edits it; from then on it is theirs and nothing overwrites it. This is the
 * place where that handover happens, so it says which of the two states the
 * card is in, and can hand it back.
 *
 * Only the flat fields are edited, and by rewriting their own lines. A card's
 * front matter carries `model-index`, which is nested, ordered and generated
 * from real scores -- a round trip through a naive YAML parser would quietly
 * destroy it, and a card that loses its results is worse than one with no
 * form at all.
 */
import { api } from "../api.js";
import { html, raw, esc, $, on, toast, fmtNum } from "../util.js";
import { renderMarkdown, readFields, writeField } from "../markdown.js";

// What the Hub understands, and what people actually pick. "Other" is in the
// list because a licence the studio has never heard of is still a licence,
// and leaving it out would push somebody towards a wrong one that is offered.
const LICENCES = [
  ["", "not stated"],
  ["apache-2.0", "Apache 2.0"],
  ["mit", "MIT"],
  ["cc-by-4.0", "CC BY 4.0"],
  ["cc-by-sa-4.0", "CC BY-SA 4.0"],
  ["cc-by-nc-4.0", "CC BY-NC 4.0 (no commercial use)"],
  ["llama3.1", "Llama 3.1 community licence"],
  ["llama3", "Llama 3 community licence"],
  ["gemma", "Gemma terms of use"],
  ["mistral-ai-research", "Mistral research licence"],
  ["openrail", "OpenRAIL"],
  ["other", "other — say which in the card"],
];

/** Draw and wire the editor into `box`. Returns a repaint function. */
export function mountCardEditor(box, { jobId, title = "Model card" } = {}) {
  let card = null;
  let tab = "write";
  let draft = null;             // unsaved text, kept across tab switches

  const draw = () => {
    if (!card) { box.innerHTML = ""; return; }
    box.innerHTML = panel(card, tab, draft ?? card.markdown ?? "", title);
  };

  const load = async (refetch = true) => {
    if (refetch || !card) {
      try { card = await api.jobCard(jobId); } catch { box.innerHTML = ""; return; }
      draft = null;
    }
    draw();
  };

  const text = () => $("#cardText", box)?.value ?? draft ?? card?.markdown ?? "";

  on(box, "click", "[data-card-tab]", (_e, t) => {
    draft = text();
    tab = t.dataset.cardTab;
    draw();
  });
  on(box, "input", "#cardText", () => { draft = text(); });
  on(box, "change", "[data-card-field]", (_e, t) => {
    // The field writes into the source, not into some parallel state: what is
    // saved is the text, and the form is a view of it.
    draft = writeField(text(), t.dataset.cardField, t.value.trim());
    draw();
  });
  on(box, "submit", "#cardForm", async (e) => {
    e.preventDefault();
    const btn = $("#cardSave", box);
    btn.disabled = true;
    btn.textContent = "Saving…";
    try {
      card = await api.saveJobCard(jobId, text());
      draft = null;
      toast("Saved. This card is yours now.", "ok");
      draw();
    } catch (ex) {
      toast(ex.message, "err");
      btn.disabled = false;
      btn.textContent = "Save";
    }
  });
  on(box, "click", "#cardReset", async () => {
    // Without a confirmation: what is discarded is recoverable by anyone who
    // kept the text, and the button says plainly what it does. What it must
    // not do is silently keep the edits.
    try {
      card = await api.resetJobCard(jobId);
      draft = null;
      toast("Back to the generated card.", "ok");
      draw();
    } catch (ex) { toast(ex.message, "err"); }
  });
  on(box, "click", "#cardCopy", async () => {
    try {
      await navigator.clipboard.writeText(text());
      toast("Copied.", "ok");
    } catch { toast("The browser would not allow copying.", "err"); }
  });

  load();
  return load;
}

function panel(card, tab, source, title) {
  const fields = readFields(source);
  const dirty = source !== (card.markdown || "");
  return html`
    <div class="card" style="margin-bottom:14px">
      <div class="row-between" style="gap:8px;flex-wrap:wrap">
        <div>
          <h3 style="margin:0">${title}</h3>
          <p class="muted tiny" style="margin:2px 0 0">The README that goes to
            Hugging Face — front matter and all, exactly as the Hub receives
            it.</p>
        </div>
        <span class="row" style="gap:6px">
          ${raw(dirty ? `<span class="badge badge-warn">unsaved</span>` : "")}
          <span class="badge ${card.edited ? "badge-ok" : ""}">${
            card.edited ? "edited by you" : "generated"}</span>
        </span>
      </div>

      <p class="muted tiny" style="margin:10px 0 0">
        ${raw(card.edited
          ? `Yours. Nothing regenerates it — not evaluating this model, not
             merging it, not publishing it. Hand it back below to return to
             the generated one.`
          : `Written from this run and kept up to date: rewritten when the run
             finishes and again every time the model is evaluated. Save an
             edit and it stops being regenerated.`)}</p>

      ${raw(!fields.license ? html`
        <div class="callout callout-warn" style="margin:10px 0 0">
          <strong>No licence stated</strong>
          The Hub asks for one, and a model published without it is a model
          nobody else can safely use. If it is built on someone else's model,
          theirs usually carries over — check the base model's page.
        </div>` : "")}

      <div class="grid grid-3" style="margin-top:12px">
        <div class="field" style="margin:0">
          <label for="cardLicence">Licence</label>
          <select id="cardLicence" data-card-field="license">
            ${raw(LICENCES.map(([v, label]) => `<option value="${esc(v)}"${
              v === (fields.license || "") ? " selected" : ""}>${esc(label)}</option>`)
              .join(""))}
            ${raw(fields.license && !LICENCES.some(([v]) => v === fields.license)
              ? `<option value="${esc(fields.license)}" selected>${
                  esc(fields.license)}</option>` : "")}
          </select>
        </div>
        <div class="field" style="margin:0">
          <label for="cardLangs">Languages</label>
          <input id="cardLangs" data-card-field="language" value="${esc(fields.language || "")}"
                 placeholder="en, de, cs">
          <div class="hint">Codes, comma separated.</div>
        </div>
        <div class="field" style="margin:0">
          <label for="cardDatasets">Datasets on the Hub</label>
          <input id="cardDatasets" data-card-field="datasets"
                 value="${esc(fields.datasets || "")}" placeholder="owner/name">
          <div class="hint">Only ones that are published there.</div>
        </div>
        <div class="field" style="margin:0;grid-column:1 / -1">
          <label for="cardTags">Tags</label>
          <input id="cardTags" data-card-field="tags" value="${esc(fields.tags || "")}"
                 placeholder="ai-studio, lora, tool-use">
        </div>
      </div>

      <div class="seg" style="margin:14px 0 8px" role="group" aria-label="Card view">
        <button type="button" class="btn-sm ${tab === "write" ? "on" : ""}"
                data-card-tab="write">Write</button>
        <button type="button" class="btn-sm ${tab === "preview" ? "on" : ""}"
                data-card-tab="preview">Preview</button>
      </div>

      <form id="cardForm">
        ${raw(tab === "write" ? html`
          <textarea id="cardText" class="mono" rows="22" spellcheck="false"
            style="width:100%;font-size:12px;line-height:1.5">${source}</textarea>`
          : html`
          <div class="md-preview">${raw(renderMarkdown(bodyOf(source)))}</div>
          <p class="muted tiny" style="margin:8px 0 0">The front matter above
            the card is metadata, not text: the Hub reads it and shows it as
            the model's licence, tags and results rather than printing it.</p>`)}
        <div class="row-between" style="margin-top:8px;gap:8px;flex-wrap:wrap">
          <span class="muted tiny">${fmtNum(source.length)} characters${
            card.updated_at ? ` · saved ${new Date(card.updated_at * 1000)
              .toLocaleString()}` : ""}</span>
          <span class="row" style="gap:6px">
            <button type="button" class="btn-sm btn-quiet" id="cardCopy">Copy</button>
            <button type="button" class="btn-sm btn-quiet" id="cardReset"
              ${card.edited ? "" : "disabled"}>Back to generated</button>
            <button type="submit" class="btn-sm ${dirty ? "btn-primary" : ""}"
                    id="cardSave">Save</button>
          </span>
        </div>
      </form>
    </div>`;
}

/** The card without its front matter, which is metadata rather than prose. */
function bodyOf(source) {
  if (!source.startsWith("---")) return source;
  const end = source.indexOf("\n---", 3);
  if (end < 0) return source;
  const after = source.indexOf("\n", end + 1);
  return after < 0 ? "" : source.slice(after + 1);
}
