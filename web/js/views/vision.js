/**
 * Training a classifier: pictures in, categories out.
 *
 * Its own page rather than four more steps in the wizard, because none of the
 * wizard's later steps -- format, template, context length -- mean anything
 * for a model that looks at a picture and names it. Three choices and a
 * review, on one screen: which pictures, which model, how hard to try.
 */
import { api } from "../api.js";
import { html, raw, esc, $, on, toast, fmtNum } from "../util.js";
import { ribbon, rb, group, wireRibbon, tabState } from "../ribbon.js";
import { pageHead, emptyState } from "../components.js";

const TABS = [{ key: "setup", label: "Set up" }];

// Backbones worth offering. All small enough for a modest card with LoRA,
// all pretrained on a great many photographs.
const MODELS = [
  ["google/vit-base-patch16-224", "ViT-Base · 86M · the safe default"],
  ["facebook/convnext-tiny-224", "ConvNeXt-Tiny · 28M · fast, good on textures"],
  ["microsoft/resnet-50", "ResNet-50 · 25M · the classic"],
  ["google/vit-large-patch16-224", "ViT-Large · 300M · better, needs a real card"],
];

export async function visionView(mount) {
  const datasets = await api.datasets().catch(() => []);
  // A dataset qualifies when its column types say a column holds pictures.
  // The list page does not carry types, so every dataset with an "image"
  // column is offered and the run refuses a wrong one with the reason.
  const usable = datasets.filter((d) => (d.columns || []).some((c) => /image|picture|photo/i.test(c)));
  let picked = usable[0]?.id || "";
  let inspect = null;
  const tabs = tabState("vision", TABS, "setup");

  const draw = () => {
    mount.innerHTML = layout({ usable, picked, inspect, datasets });
    wire();
  };

  async function loadInspect() {
    if (!picked) { inspect = null; return draw(); }
    try { inspect = await api.datasetInspect(picked); } catch { inspect = null; }
    draw();
  }

  function wire() {
    wireRibbon(mount, () => draw());
    on(mount, "change", "#vsDataset", (_e, t) => { picked = t.value; loadInspect(); });
    on(mount, "click", "#vsGo", async (_e, btn) => {
      const d = usable.find((x) => x.id === picked);
      if (!d) return toast("Choose a dataset of labelled pictures.", "err");
      btn.disabled = true;
      try {
        const r = await api.createJob({
          name: $("#vsName", mount).value || `${$("#vsModel", mount).value.split("/").pop()} on ${d.name}`,
          kind: "finetune_vision_cls",
          config: {
            studio_dataset: picked,
            image_field: $("#vsImage", mount).value || "image",
            label_field: $("#vsLabel", mount).value || "label",
            base_model: $("#vsModel", mount).value,
            epochs: +$("#vsEpochs", mount).value || 5,
            batch_size: +$("#vsBatch", mount).value || 16,
            learning_rate: +$("#vsLr", mount).value || 5e-4,
            lora: $("#vsLora", mount).checked,
            early_stop_patience: 3,
            base_model_label: $("#vsModel", mount).value.split("/").pop(),
          },
        });
        toast("Started.", "ok");
        location.hash = `#/jobs/${r.id}`;
      } catch (e) { toast(e.message, "err"); btn.disabled = false; }
    });
  }

  draw();
  loadInspect();
}

function layout({ usable, picked, inspect, datasets }) {
  const d = usable.find((x) => x.id === picked);
  const cols = d?.columns || [];
  const imageCol = cols.find((c) => /image|picture|photo/i.test(c)) || "image";
  const labelCol = cols.find((c) => /^label|category|class$/i.test(c)) || "label";
  const labelType = (inspect?.column_types || []).find((c) => c.name === labelCol);
  const media = (inspect?.media || []).find((m) => m.column === imageCol);

  return html`
    ${raw(pageHead({ title: "Train a classifier",
      sub: "Pictures in, categories out. A pretrained vision model with a new last layer.",
      back: { href: "#/new", label: "New training run" } }))}
    ${raw(ribbon({ tabs: TABS, active: "setup",
      body: group("Pictures", [
        rb(null, "▤", "Datasets", { href: "#/data" }),
        rb(null, "↑", "Upload a folder", { href: "#/data",
          title: "A zip of folders, one folder per category" }),
      ]) }))}
    ${raw(!usable.length ? emptyState({
      icon: "🖼", title: "No dataset of pictures yet",
      body: "Upload a zip with one folder per category — cats/, dogs/ — and "
          + "every picture becomes a row with its folder as the label.",
      cta: { href: "#/data", label: "Upload pictures" } }) : html`
    <div class="grid grid-2" style="align-items:start;gap:14px">
      <div>
        <div class="card" style="margin-bottom:14px">
          <h3 style="margin:0 0 8px">Which pictures</h3>
          <div class="field">
            <label for="vsDataset">Dataset</label>
            <select id="vsDataset">${raw(usable.map((x) => html`
              <option value="${x.id}"${x.id === picked ? " selected" : ""}>${x.name} — ${fmtNum(x.rows)} rows</option>`).join(""))}</select>
          </div>
          <div class="row" style="gap:10px">
            <div class="field" style="flex:1"><label for="vsImage">Picture column</label>
              <input id="vsImage" class="mono" value="${imageCol}"></div>
            <div class="field" style="flex:1"><label for="vsLabel">Label column</label>
              <input id="vsLabel" class="mono" value="${labelCol}"></div>
          </div>
          ${raw(labelType?.distinct != null ? html`
            <p class="muted tiny">${labelType.distinct} categories in the sampled
              rows${labelType.distinct > 40 ? " — a lot; each needs its own pictures" : ""}.</p>`
            : inspect ? `<p class="muted tiny">The label column has more than 64 distinct values, or none.</p>` : "")}
          ${raw(media ? html`
            <p class="muted tiny">${fmtNum(media.referenced)} pictures${media.images
              ? `, ${media.images.width.median}×${media.images.height.median} px typical` : ""}${
              media.mismatched ? ` · <span class="badge badge-warn">${media.mismatched} not really pictures</span>` : ""}${
              media.missing_files ? ` · <span class="badge badge-err">${media.missing_files} missing</span>` : ""}.
              <a href="#/data/${esc(picked)}">Check the data</a> first if anything is flagged.</p>` : "")}
        </div>
        <div class="card">
          <h3 style="margin:0 0 8px">Which model</h3>
          <div class="field">
            <label for="vsModel">Backbone</label>
            <select id="vsModel">${raw(MODELS.map(([id, why], i) => html`
              <option value="${id}"${i === 0 ? " selected" : ""}>${why}</option>`).join(""))}</select>
            <div class="hint">Already knows what edges, textures and objects look
              like; only a new last layer and a small adapter are trained.</div>
          </div>
          <label class="check"><input type="checkbox" id="vsLora" checked>
            <span>Also adapt the attention blocks (LoRA)
              <span class="muted tiny">· off, only the last layer trains — faster, and enough when the categories are obvious</span></span></label>
        </div>
      </div>
      <div>
        <div class="card" style="margin-bottom:14px">
          <h3 style="margin:0 0 8px">How hard to try</h3>
          <div class="row" style="gap:10px">
            <div class="field" style="flex:1"><label for="vsEpochs">Passes over the pictures</label>
              <input id="vsEpochs" type="number" min="1" max="50" value="5">
              <div class="hint">Stops early on its own once the held-out accuracy stops rising.</div></div>
            <div class="field" style="flex:1"><label for="vsBatch">Batch</label>
              <input id="vsBatch" type="number" min="1" max="256" value="16"></div>
          </div>
          <div class="field"><label for="vsLr">Learning rate</label>
            <input id="vsLr" class="mono" value="0.0005">
            <div class="hint">Higher than a language model's: the new head starts from nothing.</div></div>
        </div>
        <div class="card">
          <h3 style="margin:0 0 8px">Start</h3>
          <div class="field"><label for="vsName">Name</label>
            <input id="vsName" placeholder="${d ? `vit-base on ${esc(d.name)}` : "a name"}"></div>
          <p class="muted tiny">A held-out split in the dataset is the test; with none,
            one picture in ten is held back. Accuracy on those is the run's number,
            and after every pass the pictures it got most wrong are shown on the
            run page.</p>
          <button class="btn-primary" id="vsGo">Train it</button>
        </div>
      </div>
    </div>`)}`;
}
