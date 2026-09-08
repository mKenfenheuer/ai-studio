/**
 * Running an earlier run again, with changes.
 *
 * Almost nobody trains a model once. The second run is the first run with one
 * thing different -- a longer dataset, a smaller rank, half the learning rate
 * -- and rebuilding it by walking the wizard from the top is both tedious and
 * lossy: the format decision, the system prompt and the dozen settings the
 * planner worked out are all re-derived, and any of them can come out
 * different by accident. Then the comparison between the two runs is not a
 * comparison of the one thing you changed.
 *
 * So this page does the opposite of the wizard. It starts from the finished
 * run's own config, shows the settings worth changing as fields, carries
 * everything else across untouched, and prints what it could not show. The
 * original run is not modified and not deleted; this only ever creates a new
 * one.
 */
import { api } from "../api.js";
import { html, raw, esc, $, on, toast } from "../util.js";
import { ribbon, rb, group } from "../ribbon.js";
import { breadcrumb } from "../components.js";

// The settings people actually change between two runs, in the order they
// tend to think about them. Only the ones the original run had are drawn --
// a from-scratch run and a LoRA fine-tune share this page and have different
// knobs, and inventing a field the original did not set would silently add a
// setting to the copy.
const FIELDS = [
  {
    group: "How long it trains",
    items: [
      { key: "epochs", label: "Passes over the data", step: "1", min: "1",
        hint: "Two is usually plenty for a fine-tune. More memorises." },
      { key: "max_steps", label: "Stop after this many steps", step: "1",
        min: "0",
        hint: "A hard cap, whatever the passes work out to. 0 removes it." },
      { key: "warmup_steps", label: "Warm-up steps", step: "1", min: "0" },
      { key: "seed", label: "Seed", step: "1", min: "0",
        hint: "The same seed with the same settings gives the same run. "
            + "Change it to find out how much of a difference was luck." },
      { key: "learning_rate", label: "Learning rate", step: "any",
        hint: "The single most consequential number here. Halve it before "
            + "you change anything else." },
    ],
  },
  {
    group: "What fits on the machine",
    items: [
      { key: "batch_size", label: "Batch size", step: "1", min: "1" },
      { key: "grad_accum", label: "Accumulation", step: "1", min: "1",
        hint: "Batch size × accumulation is the effective batch. Raise this "
            + "instead of the batch size when memory is tight." },
      { key: "max_seq_len", label: "Longest example, in tokens", step: "1",
        min: "16",
        hint: "Rows longer than this are cut off, and a cut-off example "
            + "teaches the model to stop mid-sentence." },
    ],
  },
  {
    group: "What it starts from",
    items: [
      { key: "base_model", label: "Base model", type: "text",
        hint: "A Hugging Face id. Changing it is the difference between a "
            + "copy of this run and a different experiment, which is "
            + "occasionally what you want." },
    ],
  },
  {
    group: "The adapter",
    items: [
      { key: "lora_r", label: "Rank", step: "1", min: "1",
        hint: "How much the adapter can change. Higher learns more and "
            + "overfits sooner." },
      { key: "lora_alpha", label: "Alpha", step: "1", min: "1" },
      { key: "lora_dropout", label: "Dropout", step: "0.01", min: "0",
        max: "1" },
    ],
  },
];

const FLAGS = [
  { key: "early_stop",
    label: "Stop when the held-out loss stops improving" },
  { key: "gradient_checkpointing",
    label: "Trade speed for memory (gradient checkpointing)" },
  { key: "optim_8bit",
    label: "Keep the optimiser state in 8 bits" },
  { key: "merge_after",
    label: "Also produce a standalone model, not just the adapter" },
];

// Not numbers, so they get a list each. These are the two settings the
// out-of-memory advice names, and neither could be reached from this page --
// which meant the most common failure was the one this page could not fix.
const CHOICES = [
  { key: "quantization", label: "Base model precision",
    options: [["none", "16-bit — full quality"],
              ["4bit", "4-bit — fits far more"]],
    hint: "Only the frozen base is compressed; the adapter stays at full "
        + "precision either way." },
  { key: "train_on", label: "Learn from",
    options: [["assistant", "The assistant's replies only"],
              ["all", "Every token, questions included"]],
    hint: "Training on the questions as well spends capacity teaching the "
        + "model to write your users' half of the conversation." },
  { key: "dtype", label: "Arithmetic precision",
    options: [["float16", "float16"], ["bfloat16", "bfloat16"],
              ["float32", "float32"]],
    hint: "The machine benchmarked all three when it joined; its own "
        + "recommendation is on the Machines page." },
];

// Shown in their own section rather than as numbers among the numbers: these
// decide what the run learns from, and changing one of them without looking
// at the other two is the usual way a copy comes out wrong.
const DATA_KEYS = ["studio_dataset", "dataset", "dataset_split",
                   "dataset_config", "text_field"];
// Derived by the controller from studio_dataset on every create, so carrying
// the old values across would only ever be misleading.
const DERIVED = ["dataset", "dataset_is_local", "dataset_label",
                 // A copy is not a member of its parent's sweep and has not
                 // been published anywhere; carrying these over made the new
                 // run's page list repositories it never wrote to.
                 "sweep_id", "sweep_name", "sweep_values", "published"];

export async function rerunView(mount, [jobId]) {
  const job = await api.job(jobId);
  const cfg = { ...(job.config || {}) };

  // A generation run has a page of its own that can edit every part of the
  // brief, which is a great deal more than this one could. Sending people
  // there keeps one answer to "run this again" rather than two.
  if (job.kind === "generate_dataset") {
    location.replace(`#/generate/from/${encodeURIComponent(jobId)}`);
    return;
  }

  const [runners, datasets] = await Promise.all([
    api.runners(),
    api.datasets().catch(() => []),
  ]);
  const online = runners.filter((r) => r.status !== "offline");
  // The machine it ran on last time, if it is still here. Otherwise whichever
  // is available -- an offline id in this field is a run that sits queued
  // forever waiting for a machine that is not coming back.
  const runnerId = online.some((r) => r.id === cfg.required_runner)
    ? cfg.required_runner : (online[0]?.id || "");

  // `?set={"learning_rate":2e-5}` — the correction the run's own report
  // worked out, applied here so "fix it and run again" is one press rather
  // than a paragraph, a form, and a number typed from memory.
  let fixes = {};
  try {
    const at = location.hash.indexOf("?");
    const raw = at < 0 ? null
      : new URLSearchParams(location.hash.slice(at + 1)).get("set");
    if (raw) fixes = JSON.parse(raw) || {};
  } catch { fixes = {}; }
  const applied = Object.keys(fixes);
  Object.assign(cfg, fixes);

  const shown = new Set([...DATA_KEYS, ...DERIVED, "required_runner",
                         "system_prompt",
                         ...FIELDS.flatMap((g) => g.items.map((i) => i.key)),
                         ...CHOICES.map((c) => c.key),
                         ...FLAGS.map((f) => f.key)]);
  const carried = Object.fromEntries(
    Object.entries(cfg).filter(([k]) => !shown.has(k)));

  mount.innerHTML = layout(job, cfg, online, runnerId, datasets, carried, applied);

  // The ribbon's Start is the same button as the form's, one screen higher.
  on(mount, "click", "[data-submit]", (_e, t) => {
    $(`#${t.dataset.submit}`, mount)?.requestSubmit();
  });

  on(mount, "submit", "#rerunForm", async (e) => {
    e.preventDefault();
    const btn = $("#rerunGo", mount);
    const next = { ...cfg };
    for (const k of DERIVED) delete next[k];

    next.required_runner = $("#rr_runner", mount).value;
    if (!next.required_runner) {
      return toast("Choose a machine to run it on.", "err");
    }
    for (const key of DATA_KEYS) {
      const el = $(`#rr_${key}`, mount);
      if (el) next[key] = el.value;
    }
    const sys = $("#rr_system_prompt", mount);
    if (sys) next.system_prompt = sys.value;
    for (const item of FIELDS.flatMap((g) => g.items)) {
      const el = $(`#rr_${item.key}`, mount);
      // Left blank means "leave it as it was", not "zero".
      if (!el || el.value.trim() === "") continue;
      next[item.key] = item.type === "text" ? el.value.trim() : Number(el.value);
    }
    for (const c of CHOICES) {
      const el = $(`#rr_${c.key}`, mount);
      if (el && el.value) next[c.key] = el.value === "none" ? null : el.value;
    }
    for (const f of FLAGS) {
      const el = $(`#rr_${f.key}`, mount);
      if (el) next[f.key] = el.checked;
    }

    btn.disabled = true;
    btn.textContent = "Starting…";
    try {
      const { id } = await api.createJob({
        name: $("#rr_name", mount).value || undefined,
        kind: job.kind,
        config: next,
      });
      toast("Training run created.", "ok");
      location.hash = `#/jobs/${id}`;
    } catch (err) {
      toast(err.message, "err");
      btn.disabled = false;
      btn.textContent = "Start this run";
    }
  });
}

// A name that says what it is without anybody having to type one. "(again)"
// stacks badly after three rounds, so a run already carrying one gets a
// number instead.
function nextName(name) {
  const base = String(name || "Run");
  const m = base.match(/^(.*) \(again(?: (\d+))?\)$/);
  if (m) return `${m[1]} (again ${(+m[2] || 2) + 1})`;
  return `${base} (again)`;
}

function layout(job, cfg, online, runnerId, datasets, carried, applied = []) {
  const has = (k) => cfg[k] !== undefined && cfg[k] !== null;
  const studio = has("studio_dataset");
  const carriedKeys = Object.keys(carried);

  return html`
    <div class="page-head">
      ${raw(breadcrumb([{ href: "#/jobs", label: "Runs" },
                        { href: `#/jobs/${job.id}`, label: job.name }]))}
      <h1 style="margin:6px 0 0">Run again</h1>
      <p class="sub">A copy of that run's settings. Change what you like and
        start a new one &mdash; the original is left exactly as it is.</p>
    </div>
    ${raw(ribbon({
      tabs: [{ key: "home", label: "Settings" }], active: "home",
      body: group("This copy", [
        rb(null, "▶", "Start it", { cls: "primary", data: 'data-submit="rerunForm"' }),
        rb(null, "≡", "The original", { href: `#/jobs/${esc(job.id)}` }),
      ]) + group("Change more", [
        rb(null, "✦", "Open the wizard", { href: "#/new",
          title: "Changing the model, the format or the shape of the data needs the full flow" }),
      ]),
    }))}

    ${raw(applied.length ? html`
      <div class="callout callout-ok" style="margin-bottom:14px">
        <strong>The correction from that run's report is already filled in.</strong>
        ${applied.map((k) => `${k.replace(/_/g, " ")} is now ${cfg[k]}`).join(", ")}.
        Change anything else you like before starting.
      </div>` : "")}

    <form id="rerunForm">
      <div class="card">
        <div class="grid grid-2">
          <div class="field">
            <label for="rr_name">Name</label>
            <input id="rr_name" value="${nextName(job.name)}">
          </div>
          <div class="field">
            <label for="rr_runner">Machine</label>
            <select id="rr_runner">
              ${raw(online.map((r) => `<option value="${esc(r.id)}"${
                r.id === runnerId ? " selected" : ""}>${esc(r.name || r.id)}${
                r.id === cfg.required_runner ? " — the one it ran on" : ""
              }</option>`).join(""))}
            </select>
            ${raw(online.length ? "" : `<div class="hint">No machine is
              connected. The run will sit queued until one is.</div>`)}
          </div>
        </div>
      </div>

      <div class="card">
        <h3>What it trains on</h3>
        <div class="grid grid-2">
          ${raw(studio ? `
            <div class="field">
              <label for="rr_studio_dataset">Dataset</label>
              <select id="rr_studio_dataset">
                ${datasets.map((d) => `<option value="${esc(d.id)}"${
                  d.id === cfg.studio_dataset ? " selected" : ""}>${
                  esc(d.name)}${d.rows ? ` — ${d.rows} rows` : ""
                }</option>`).join("")}
              </select>
              <div class="hint">Pointing this at a different dataset is the
                usual reason to be on this page. The rows must be the same
                shape as the ones it learned from, because the format below is
                carried over as it stands.</div>
            </div>` : has("dataset") ? `
            <div class="field">
              <label for="rr_dataset">Dataset</label>
              <input id="rr_dataset" value="${esc(cfg.dataset)}">
              <div class="hint">A Hugging Face dataset id, or a URL.</div>
            </div>` : "")}
          ${raw(has("dataset_split") ? `
            <div class="field">
              <label for="rr_dataset_split">Split</label>
              <input id="rr_dataset_split" value="${esc(cfg.dataset_split)}">
            </div>` : "")}
          ${raw(has("dataset_config") ? `
            <div class="field">
              <label for="rr_dataset_config">Configuration</label>
              <input id="rr_dataset_config" value="${esc(cfg.dataset_config)}">
            </div>` : "")}
          ${raw(has("text_field") ? `
            <div class="field">
              <label for="rr_text_field">Column to train on</label>
              <input id="rr_text_field" value="${esc(cfg.text_field)}">
            </div>` : "")}
        </div>
        ${raw(has("system_prompt") ? `
          <div class="field">
            <label for="rr_system_prompt">System prompt</label>
            <textarea id="rr_system_prompt" rows="4">${
              esc(cfg.system_prompt)}</textarea>
            <div class="hint">Taken from the rows the original run trained on.
              If you have changed the dataset above, check that this still
              matches it &mdash; a stale prompt here teaches the model to
              expect one it will never be given.</div>
          </div>` : "")}
        ${raw(cfg.base_model_job ? `
          <div class="hint">Continues
            <a href="#/jobs/${esc(cfg.base_model_job)}">${
              esc(cfg.source_run_name || cfg.base_model_job)}</a> rather than
            starting from the base model, exactly as the original did.</div>`
          : cfg.base_model ? `
          <div class="hint">Base model
            <code>${esc(cfg.base_model)}</code>.</div>` : "")}
      </div>

      ${raw(FIELDS.map((g) => {
        const items = g.items.filter((i) => has(i.key));
        if (!items.length) return "";
        return html`
          <div class="card">
            <h3>${g.group}</h3>
            <div class="grid grid-2">
              ${raw(items.map((i) => `
                <div class="field">
                  <label for="rr_${i.key}">${esc(i.label)}</label>
                  ${i.type === "text"
                    ? `<input id="rr_${i.key}" class="mono" value="${esc(cfg[i.key])}">`
                    : `<input id="rr_${i.key}" type="number" value="${esc(cfg[i.key])}"
                         step="${i.step}"${i.min !== undefined
                           ? ` min="${i.min}"` : ""}${i.max !== undefined
                           ? ` max="${i.max}"` : ""}>`}
                  ${i.hint ? `<div class="hint">${esc(i.hint)}</div>` : ""}
                </div>`).join(""))}
            </div>
          </div>`;
      }).join(""))}

      ${raw(CHOICES.some((c) => has(c.key)) ? html`
        <div class="card">
          <h3>Precision</h3>
          <div class="grid grid-2">
            ${raw(CHOICES.filter((c) => has(c.key)).map((c) => `
              <div class="field">
                <label for="rr_${c.key}">${esc(c.label)}</label>
                <select id="rr_${c.key}">
                  ${c.options.map(([v, l]) => `<option value="${v}"${
                    String(cfg[c.key] ?? "none") === v ? " selected" : ""}>${
                    esc(l)}</option>`).join("")}
                </select>
                ${c.hint ? `<div class="hint">${esc(c.hint)}</div>` : ""}
              </div>`).join(""))}
          </div>
        </div>` : "")}

      ${raw(FLAGS.some((f) => has(f.key)) ? html`
        <div class="card">
          <h3>Behaviour</h3>
          ${raw(FLAGS.filter((f) => has(f.key)).map((f) => `
            <label class="check"><input type="checkbox" id="rr_${f.key}"${
              cfg[f.key] ? " checked" : ""}> ${esc(f.label)}</label>`).join(""))}
        </div>` : "")}

      ${raw(carriedKeys.length ? html`
        <div class="card">
          <h3>Carried over unchanged</h3>
          <p class="muted tiny">${carriedKeys.length} more
            setting${carriedKeys.length === 1 ? "" : "s"} the original run
            recorded, copied across as they are. The chat format is among them,
            which is why changing the dataset to one of a different shape needs
            the wizard rather than this page.</p>
          <details>
            <summary class="tiny">Show them</summary>
            <pre class="mono tiny" style="white-space:pre-wrap;max-height:340px;
                 overflow:auto">${esc(JSON.stringify(carried, null, 1))}</pre>
          </details>
        </div>` : "")}

      <div class="row" style="gap:8px;margin-top:12px">
        <button class="btn btn-primary" id="rerunGo" type="submit">
          Start this run</button>
        <a class="btn" href="#/jobs/${job.id}">Cancel</a>
        <a class="btn" href="#/new">Start from the wizard instead</a>
      </div>
    </form>`;
}
