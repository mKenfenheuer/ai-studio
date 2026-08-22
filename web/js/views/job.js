import { api, events } from "../api.js";
import { html, raw, esc, $, on, fmtNum, fmtDuration, statusBadge, toast } from "../util.js";
import { LineChart } from "../chart.js";
import { shareBox, wireShareBox } from "./share.js";

const STAGES = {
  evaluating: "Putting the prompts to each model…",
  loading_model: "Downloading and loading the model…",
  loading_dataset: "Downloading and preparing your data…",
  training_tokenizer: "Building a vocabulary from your text…",
  tokenizing: "Reading and tokenizing the text…",
  building_model: "Creating the model from random weights…",
  training: "Training",
  saving: "Saving the result…",
};

export async function jobView(mount, [jobId]) {
  let job = await api.job(jobId);
  const metrics = await api.jobMetrics(jobId);
  const logs = await api.jobLogs(jobId);
  const scratch = job.kind === "pretrain_llm";
  const experts = +(job.config.arch?.num_local_experts || 0);

  mount.innerHTML = layout(job, scratch, experts);

  // Held-out loss shares the loss chart because it is the same measurement on
  // different data. That is the only case where a second series belongs on one
  // axis; anything with a different unit gets its own chart.
  const lossChart = new LineChart($("#lossChart", mount), {
    title: "Training loss — this should go down",
    height: 260,
    format: (v) => v.toFixed(4),
    series: [
      { key: "train", label: "Training" },
      { key: "val", label: "Held-out", dashed: true },
    ],
  });
  // Learning rate gets its own chart. Sharing an axis with loss would imply a
  // relationship between two quantities that have nothing to do with each other.
  const lrChart = new LineChart($("#lrChart", mount), {
    title: "Learning rate",
    height: 150,
    color: "var(--text-3)",
    format: (v) => v.toExponential(1),
  });

  // Only built for a mixture of experts, because for anything else it would
  // be a chart of a constant.
  const expertChart = experts > 1 ? new LineChart($("#expertChart", mount), {
    title: "Busiest expert's share of the tokens",
    height: 150,
    color: "var(--warn)",
    format: (v) => (v * 100).toFixed(0) + "%",
  }) : null;
  expertChart?.setData(metrics.filter((m) => m.expert_balance != null)
    .map((m) => ({ x: m.step, y: m.expert_balance })));

  lossChart.setSeries("train",
    metrics.filter((m) => m.loss != null).map((m) => ({ x: m.step, y: m.loss })));
  lossChart.setSeries("val",
    metrics.filter((m) => m.val_loss != null).map((m) => ({ x: m.step, y: m.val_loss })));
  lrChart.setData(metrics.filter((m) => m.learning_rate != null)
    .map((m) => ({ x: m.step, y: m.learning_rate })));

  const samples = metrics.filter((m) => m.sample_text)
    .map((m) => ({ step: m.step, text: m.sample_text, prompt: m.sample_prompt }));
  if (scratch) paintSamples(mount, samples);

  const logBox = $("#logBox", mount);
  logs.forEach((l) => appendLog(logBox, l));
  logBox.scrollTop = logBox.scrollHeight;

  // The last step a checkpoint exists for. Shown live rather than only on
  // reload, because the question it answers -- "how much would I lose if this
  // machine restarted right now" -- is only interesting while it is running.
  let checkpointStep = job.checkpoint_step || 0;
  let latest = metrics[metrics.length - 1] || {};
  // The stage the runner last reported. paintStats runs on every metric and
  // must not claim "training" while the tokenizer is still being built.
  let stage = job.status === "running" ? "" : "training";
  paintStats(mount, job, latest, scratch, stage, checkpointStep);

  const unsub = events.subscribe(async (msg) => {
    if (msg.job_id && msg.job_id !== jobId) return;

    if (msg.type === "job_metric") {
      if (msg.data.sample_text) return;   // samples arrive as job_sample
      latest = { step: msg.step, ...msg.data };
      if (msg.data.loss != null) lossChart.pushSeries("train", { x: msg.step, y: msg.data.loss });
      if (msg.data.val_loss != null) lossChart.pushSeries("val", { x: msg.step, y: msg.data.val_loss });
      if (msg.data.learning_rate != null)
        lrChart.push({ x: msg.step, y: msg.data.learning_rate });
      if (msg.data.expert_balance != null)
        expertChart?.push({ x: msg.step, y: msg.data.expert_balance });
      paintStats(mount, job, latest, scratch, stage, checkpointStep);
    } else if (msg.type === "job_sample") {
      samples.push({ step: msg.step, text: msg.text, prompt: msg.prompt });
      paintSamples(mount, samples);
    } else if (msg.type === "job_log") {
      const atBottom = logBox.scrollHeight - logBox.scrollTop - logBox.clientHeight < 40;
      appendLog(logBox, { ts: Date.now() / 1000, level: msg.level, line: msg.line });
      if (atBottom) logBox.scrollTop = logBox.scrollHeight;
    } else if (msg.type === "job_checkpoint") {
      checkpointStep = msg.step;
      paintProgress(mount, job, stage, null, null, checkpointStep);
    } else if (msg.type === "job_progress") {
      stage = msg.stage;
      if (stage === "training") { job.step = msg.step; job.total_steps = msg.total; }
      paintProgress(mount, job, stage, msg.step, msg.total);
    } else if (msg.type === "jobs_changed") {
      job = await api.job(jobId);
      paintHeader(mount, job);
      paintStats(mount, job, latest, scratch, stage, checkpointStep);
      // A run that just finished has a model to build on, and one that just
      // failed has a checkpoint to carry on from. Both change this panel.
      paintFurther();
    }
  });

  const paintOwnerRow = () => {
    const box = $("#ownerRow", mount);
    if (!box) return;
    const hasModel = job.artifacts?.length;
    box.innerHTML = shareBox("job", job) + (hasModel ? publishCard(job) : "");
    wireShareBox(mount, "job", job, async () => {
      job = await api.job(jobId);
      paintOwnerRow();
    });
  };
  paintOwnerRow();

  // Datasets are only needed by the "train this further" panel, which most
  // visits never open, so the list is fetched after the page is on screen.
  let datasets = [];
  const paintFurther = () => {
    const box = $("#furtherRow", mount);
    if (box) box.innerHTML = furtherCard(job, datasets);
  };
  paintFurther();
  if (job.artifacts?.length) {
    api.datasets().then((d) => { datasets = d; paintFurther(); }).catch(() => {});
  }

  on(mount, "click", "#resumeBtn", async (_e, t) => {
    t.disabled = true;
    t.textContent = "Queueing…";
    try {
      const r = await api.resumeJob(jobId);
      toast(`Carrying on from step ${r.from_step}.`, "ok");
      job = await api.job(jobId);
      paintStats(mount, job, latest, scratch, stage, checkpointStep);
    } catch (e) { toast(e.message, "err"); t.disabled = false; }
  });

  on(mount, "submit", "#furtherForm", async (e) => {
    e.preventDefault();
    const f = Object.fromEntries(new FormData(e.target).entries());
    const btn = $("#furtherGo", mount);
    btn.disabled = true;
    btn.textContent = "Creating…";
    try {
      const { id } = await api.createJob(furtherJob(job, f));
      toast("Queued.", "ok");
      location.hash = `#/jobs/${id}`;
    } catch (ex) {
      toast(ex.message, "err");
      btn.disabled = false;
      btn.textContent = "Start the follow-on run";
    }
  });

  on(mount, "submit", "#publishForm", async (e) => {
    e.preventDefault();
    const f = Object.fromEntries(new FormData(e.target).entries());
    const btn = $("#pubGo", mount);
    btn.disabled = true;
    btn.textContent = "Uploading to Hugging Face…";
    try {
      const r = await api.publishJob(jobId, {
        repo_id: f.repo_id, private: f.visibility === "private" });
      $("#publishResult", mount).innerHTML =
        `<div class="callout callout-ok"><strong>Published</strong>
          <a href="${esc(r.url)}" target="_blank" rel="noopener">${esc(r.url)}</a></div>`;
      toast("Published.", "ok");
    } catch (ex) {
      $("#publishResult", mount).innerHTML =
        `<div class="callout callout-err">${esc(ex.message)}</div>`;
    } finally {
      btn.disabled = false;
      btn.textContent = "Publish to Hugging Face";
    }
  });

  // Delegated, not bound directly: paintHeader() replaces the button element
  // every time a metric arrives, which would silently discard a direct listener.
  // Stopping is two different actions wearing one button, and the difference
  // between them is hours of GPU time. A yes/no confirm can only ask the
  // question it was written with, so it is replaced by the actual choice.
  on(mount, "click", "#cancelBtn", () => {
    $("#stopPanel", mount).innerHTML = stopPanel(job, latest, stage);
  });
  on(mount, "click", "#stopCancel", () => { $("#stopPanel", mount).innerHTML = ""; });

  on(mount, "click", "[data-stop]", async (_e, t) => {
    const save = t.dataset.stop === "keep";
    $("#stopPanel", mount).innerHTML = "";
    try {
      await api.cancelJob(jobId, save);
      toast(save ? "Stopping, and keeping the model…" : "Stopping…");
    } catch (e) { toast(e.message, "err"); }
  });

  on(mount, "click", "#deleteBtn", async () => {
    const hasModel = job.artifacts?.length;
    if (!confirm(`Delete "${job.name}"?\n\n` + (hasModel
      ? "Its trained model file will be deleted too, and cannot be recovered."
      : "Its logs and measurements will be deleted."))) return;
    try {
      await api.deleteJob(jobId);
      toast("Run deleted.", "ok");
      location.hash = "#/jobs";
    } catch (e) { toast(e.message, "err"); }
  });

  return () => {
    unsub(); lossChart.destroy(); lrChart.destroy(); expertChart?.destroy();
  };
}

// ---------------------------------------------------------------------------

function layout(job, scratch, experts = 0) {
  const source = job.config.dataset_label || job.config.dataset;
  const subtitle = job.kind === "generate_dataset"
    ? `writing data with ${job.config.model?.base_model || "your own model"}`
    : scratch ? `from scratch · ${source}`
              : `${job.config.base_model} → ${source}`;
  return html`
    <div class="page-head">
      <a href="#/jobs" class="tiny">← All runs</a>
      <div class="row-between" style="flex-wrap:wrap;gap:8px;margin-top:6px">
        <h1 style="margin:0">${job.name}</h1>
        <div class="row" id="headerActions"></div>
      </div>
      <p class="sub mono tiny" style="margin-top:4px">${subtitle}</p>
    </div>

    <div id="errorCard"></div>
    <div id="resumeCard"></div>
    <div id="queueCard"></div>
    <div id="stopPanel"></div>
    <div id="progressCard"></div>
    <div class="grid grid-3" id="statCards" style="margin-bottom:16px"></div>

    <div class="card" style="margin-bottom:14px"><div id="lossChart"></div></div>

    ${raw(experts > 1 ? html`
      <div class="card" style="margin-bottom:14px">
        <div id="expertChart"></div>
        <p class="muted tiny" style="margin:8px 0 0">
          The share of tokens going to the single busiest of the
          ${experts} experts. An even router gives
          <strong>${(100 / experts).toFixed(0)}%</strong>; a line climbing well
          above that means the router has picked favourites and the remaining
          experts are being starved of the text they need. The loss curve will
          not show you this — a collapsed mixture still learns, it just carries
          most of its parameters dead.</p>
      </div>` : "")}

    ${raw(scratch ? html`
      <div class="card" id="samplesCard" style="margin-bottom:14px"></div>` : "")}

    <div class="grid grid-2" style="margin-bottom:14px">
      <div class="card"><div id="lrChart"></div></div>
      <div class="card">
        <h3>What am I looking at?</h3>
        <p class="muted tiny">The <strong>loss</strong> is how wrong the model's
          predictions are. A healthy run drops quickly at first, then flattens.</p>
        <ul class="muted tiny" style="margin:0;padding-left:18px;line-height:1.7">
          <li><strong>Falling steadily</strong> — working as intended.</li>
          <li><strong>Flat from the start</strong> — the learning rate may be too
            low, or the data may not be read correctly.</li>
          <li><strong>Jumping wildly</strong> — the learning rate is too high.</li>
          <li><strong>Held-out rising while training falls</strong> — it has
            started memorising ${scratch ? "the text" : "your examples"} instead
            of learning from ${scratch ? "it" : "them"}. The best model was at
            the low point; more data or fewer passes would help.</li>
          <li>The held-out line is also the only loss worth comparing with
            another run — <a href="#/compare">side by side here</a>.</li>
        </ul>
      </div>
    </div>

    <div id="furtherRow"></div>

    <div class="grid grid-2" style="margin-bottom:14px" id="ownerRow"></div>

    <div class="card">
      <div class="row-between" style="margin-bottom:8px">
        <h3 style="margin:0">Log</h3>
        <span class="tiny muted">Newest at the bottom</span>
      </div>
      <div class="logbox" id="logBox"></div>
    </div>`;
}

function furtherCard(job, datasets) {
  const usable = job.artifacts?.length
    && ["succeeded", "cancelled"].includes(job.status);
  if (!usable) return "";
  const scratch = job.kind === "pretrain_llm";
  if (!scratch && job.kind !== "finetune_llm") return "";

  const cfg = job.config || {};
  const arch = cfg.arch || {};
  const perStep = scratch
    ? (cfg.batch_size || 1) * (cfg.grad_accum || 1)
      * (arch.max_position_embeddings || 0)
    : 0;
  const defaultSteps = job.total_steps || cfg.max_steps || 300;

  return html`
    <details class="card" style="margin-bottom:14px">
      <summary><strong>Train this further</strong>
        <span class="muted tiny"> — carry on from what it already knows</span>
      </summary>
      <p class="muted tiny" style="margin:10px 0 0">
        ${raw(scratch ? html`
          This starts from the weights this run produced rather than from
          noise, keeping its vocabulary and its shape — width, depth and
          vocabulary are fixed once a model has been trained. More text, or
          another pass over the same text, makes it better at the same job.`
        : html`
          This keeps the adapter this run produced and carries on training it
          on new data. Everything it already learned stays; the new examples
          are added on top. Its base model, <code>${esc(cfg.base_model || "—")}</code>,
          stays the same.`)}</p>

      <form id="furtherForm" style="margin-top:12px">
        <div class="field">
          <label for="furtherData">Text to learn from</label>
          <select id="furtherData" name="studio_dataset">
            <option value="">Keep the same source (${esc(
              cfg.dataset_label || String(cfg.dataset || "").split("/").pop() || "—")})</option>
            ${raw(datasets.map((d) => html`
              <option value="${d.id}">${d.name} · ${fmtNum(d.rows)} rows</option>`).join(""))}
          </select>
          <div class="hint">Reading the same text again is a real option — a
            model that has only seen its corpus once is undertrained — but new
            text teaches it more.</div>
        </div>
        ${raw(scratch ? html`
          <div class="field">
            <label for="furtherSteps">How many more steps</label>
            <input id="furtherSteps" name="max_steps" type="number"
                   value="${defaultSteps}" min="10">
            <div class="hint">${perStep
              ? `About ${fmtNum(perStep)} tokens per step on this model.` : ""}
              The learning rate starts high again and decays over these steps,
              so a short follow-on run disturbs the model before it settles.</div>
          </div>` : "")}
        <div class="field">
          <label for="furtherName">Name</label>
          <input id="furtherName" name="name" type="text"
                 value="${esc(job.name)} (continued)">
        </div>
        <button class="btn-primary btn-sm" type="submit" id="furtherGo">
          Start the follow-on run</button>
      </form>
    </details>`;
}

/** The job body for a follow-on run. Copies the settings that must not change
 *  and drops the ones that must be worked out again. */
function furtherJob(job, form) {
  const cfg = { ...(job.config || {}) };
  // Recomputed for the new data, never inherited: a step budget worked out
  // for one corpus is meaningless for another, and an inherited one would
  // silently cap the new run.
  delete cfg.max_steps;
  delete cfg.token_budget;
  delete cfg.studio_dataset;
  delete cfg.source_run_name;

  if (form.studio_dataset) {
    cfg.studio_dataset = form.studio_dataset;
    delete cfg.dataset;
    delete cfg.dataset_config;
    delete cfg.dataset_split;
    delete cfg.dataset_label;
    delete cfg.dataset_is_local;
  }

  if (job.kind === "pretrain_llm") {
    cfg.continue_from = job.id;
    cfg.max_steps = +form.max_steps || job.total_steps || 300;
    const arch = cfg.arch || {};
    const perStep = (cfg.batch_size || 1) * (cfg.grad_accum || 1)
      * (arch.max_position_embeddings || 0);
    if (perStep) cfg.token_budget = cfg.max_steps * perStep;
  } else {
    cfg.base_model_job = job.id;
  }
  return { name: form.name || undefined, kind: job.kind, config: cfg };
}

function publishCard(job) {
  const base = (job.kind === "pretrain_llm"
    ? job.name : (job.config.base_model || "model").split("/").pop() + "-tuned");
  const slug = base.toLowerCase().replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "").slice(0, 60) || "my-model";
  return html`
    <div class="card">
      <h3>Publish to Hugging Face</h3>
      <p class="muted tiny">Uploads ${job.kind === "pretrain_llm"
        ? "the model" : "the adapter"} and a model card to your own account.
        Needs a connected account with write access —
        <a href="#/account">set that up here</a>.</p>
      <form id="publishForm" style="margin-top:10px">
        <div class="field">
          <label for="pubRepo">Repository</label>
          <input id="pubRepo" name="repo_id" type="text" class="mono" required
                 placeholder="your-name/${esc(slug)}">
        </div>
        <div class="field">
          <label for="pubVis">Visibility</label>
          <select id="pubVis" name="visibility">
            <option value="private">Private</option>
            <option value="public">Public — anyone can download it</option>
          </select>
        </div>
        <button class="btn-sm" type="submit" id="pubGo">Publish to Hugging Face</button>
      </form>
      <div id="publishResult"></div>
    </div>`;
}

function stopPanel(job, latest, stage) {
  const kind = job.kind === "pretrain_llm" ? "model" : "adapter";
  // There is only something to keep once training has actually begun.
  // Before that the runner is still downloading text or building a
  // vocabulary, and "keep the model" would be an offer of nothing.
  const started = (latest?.step || job.step || 0) > 0
    && (stage === "training" || stage === "");
  const done = latest?.step || job.step || 0;
  const total = job.total_steps || 0;

  return html`
    <div class="card callout-warn" style="margin-bottom:14px">
      <h3 style="margin:0 0 6px">Stop this run?</h3>
      ${raw(started ? html`
        <p class="muted tiny" style="margin:0 0 12px">
          It has trained for <strong>${fmtNum(done)}</strong>${total
            ? ` of ${fmtNum(total)}` : ""} steps. That work does not have to be
          thrown away — a partly trained ${kind} is a real one, just less
          practised than it would have been. Its learning rate never finished
          decaying, so expect it to be a little rougher than the same run
          taken to the end.</p>
        <div class="row" style="gap:8px;flex-wrap:wrap">
          <button class="btn-primary btn-sm" data-stop="keep">
            Stop and keep the ${kind}</button>
          <button class="btn-danger btn-sm" data-stop="discard">
            Stop and discard it</button>
          <button class="btn-sm" id="stopCancel">Keep training</button>
        </div>` : html`
        <p class="muted tiny" style="margin:0 0 12px">
          Training has not started yet — the runner is still preparing your
          data, so there is no ${kind} to keep. Stopping now leaves nothing
          behind.</p>
        <div class="row" style="gap:8px">
          <button class="btn-danger btn-sm" data-stop="discard">Stop the run</button>
          <button class="btn-sm" id="stopCancel">Carry on</button>
        </div>`)}
    </div>`;
}

function paintHeader(mount, job) {
  const box = $("#headerActions", mount);
  const done = ["succeeded", "failed", "cancelled"].includes(job.status);
  // A stopped run that kept its model is as usable as a finished one. The
  // artifact is what decides that, not how the run ended.
  const usable = job.artifacts?.length
    && ["succeeded", "cancelled"].includes(job.status);
  const kept = job.status === "cancelled" && job.artifacts?.length;
  box.innerHTML = html`
    ${statusBadge(job.status)}
    ${raw(kept ? `<span class="badge badge-ok">model kept</span>` : "")}
    ${raw(usable
      ? `<a class="btn btn-primary btn-sm" href="#/play/${esc(job.id)}">▷ Try it out</a>
         <a class="btn btn-sm" href="#/compare" title="Compare its held-out loss with other runs">⇄ Compare</a>` : "")}
    ${raw(done && job.artifacts?.length
      ? `<a class="btn btn-sm" href="/api/jobs/${esc(job.id)}/download">
           ↓ Download</a>` : "")}
    ${raw(!done ? `<button class="btn-danger btn-sm" id="cancelBtn">Stop</button>`
                : `<button class="btn-danger btn-sm" id="deleteBtn">Delete</button>`)}`;

  // Assigned, not prepended: paintHeader runs on every metric tick, and
  // prepending would stack a fresh copy of the error on each one.
  $("#errorCard", mount).innerHTML = job.error
    ? html`<div class="callout callout-err">
        <strong>This run failed</strong>${job.error}</div>`
    : "";
}

function paintProgress(mount, job, stage = "", rawStep = null, rawTotal = null,
                       checkpointStep = 0) {
  const training = stage === "training" || stage === "";
  // Preparation stages count their own units -- documents scanned, tokens
  // collected -- so the bar follows those while they run, and the step counter
  // is only shown once those units really are training steps.
  const step = training ? job.step : (rawStep ?? 0);
  const total = training ? job.total_steps : (rawTotal ?? 0);
  const pct = total ? Math.min(100, (step / total) * 100) : 0;
  const running = ["running", "assigned"].includes(job.status);
  const stageText = STAGES[stage] || (running ? "Working…" : "");
  const counted = total > 0 && training;

  $("#progressCard", mount).innerHTML = running ? html`
    <div class="card" style="margin-bottom:16px">
      <div class="row-between" style="margin-bottom:8px">
        <strong class="tiny">${stageText}</strong>
        <span class="tiny muted">${counted
          ? `step ${step} of ${total}` : ""}</span>
      </div>
      <div class="progress"><i style="width:${pct}%"></i></div>
      ${raw(checkpointStep ? html`
        <p class="muted tiny" style="margin:8px 0 0">
          Saved at step ${checkpointStep}. If this machine restarts, the run
          carries on from there rather than starting over.</p>` : "")}
    </div>` : "";
}

function queueCard(job) {
  if (job.status !== "queued") return "";
  const pos = job.queue_position;
  return html`
    <div class="card callout" style="margin-bottom:16px">
      <strong class="tiny">Waiting for a free machine</strong>
      ${raw(pos ? html`
        <p class="muted tiny" style="margin:6px 0 0">
          ${pos === 1 ? "Next in line" : `Number ${pos}`} of
          ${job.queue_length} waiting. The queue is dealt out between people
          rather than strictly in the order runs were created, so one person
          queueing several long runs does not hold up everyone else.</p>`
        : html`<p class="muted tiny" style="margin:6px 0 0">
          No machine that can run this is connected yet.</p>`)}
    </div>`;
}

function resumeCard(job) {
  const r = job.resumable;
  if (!r) return "";
  const lost = Math.max(0, (job.step || 0) - r.step);
  return html`
    <div class="card callout-warn" style="margin-bottom:14px">
      <h3 style="margin:0 0 6px">This run can carry on</h3>
      <p class="muted tiny" style="margin:0 0 10px">
        There is a checkpoint at <strong>step ${fmtNum(r.step)}</strong>${
          r.total ? ` of ${fmtNum(r.total)}` : ""} on
        <strong>${r.runner || "its machine"}</strong>${
          lost ? `, so ${fmtNum(lost)} steps would be redone` : ""}.
        Starting it again picks up the weights and the optimiser exactly where
        they were, rather than beginning from noise.</p>
      ${raw(r.online
        ? html`<button class="btn-primary btn-sm" id="resumeBtn">
                 Carry on from step ${fmtNum(r.step)}</button>`
        : html`<p class="muted tiny" style="margin:0">
            <strong>${esc(r.runner || "That machine")}</strong> is not
            connected, and the checkpoint is on its disk. Bring it back and
            this button appears.</p>`)}
    </div>`;
}

function paintStats(mount, job, m, scratch, stage = "training",
                   checkpointStep = 0) {
  paintHeader(mount, job);
  paintProgress(mount, job, stage, null, null,
                checkpointStep || job.checkpoint_step || 0);
  const q = $("#queueCard", mount);
  if (q) q.innerHTML = queueCard(job);
  const rc = $("#resumeCard", mount);
  if (rc) rc.innerHTML = resumeCard(job);
  const cards = scratch ? [
    ["Loss now", m.loss != null ? m.loss.toFixed(4) : "—", "lower is better"],
    ["Held-out loss", m.val_loss != null ? m.val_loss.toFixed(4) : "—", "on unseen text"],
    ["Text read", m.tokens_seen != null ? fmtNum(m.tokens_seen) : "—", "tokens so far"],
    ["Speed", m.tokens_per_sec != null ? fmtNum(m.tokens_per_sec) + "/s" : "—", "tokens per second"],
    ["GPU memory", m.vram_gb != null ? m.vram_gb + " GB" : "—", "peak used"],
    ["Time left", m.eta_s != null && job.status === "running"
      ? fmtDuration(m.eta_s) : "—", "estimate"],
  ] : [
    ["Loss now", m.loss != null ? m.loss.toFixed(4) : "—", "lower is better"],
    ["Held-out loss", m.val_loss != null ? m.val_loss.toFixed(4) : "—",
     "on unseen examples"],
    ["Speed", m.steps_per_sec != null ? m.steps_per_sec.toFixed(2) + "/s" : "—", "steps per second"],
    ["GPU memory", m.vram_gb != null ? m.vram_gb + " GB" : "—", "peak used"],
    ["Time left", m.eta_s != null && job.status === "running"
      ? fmtDuration(m.eta_s) : "—", "estimate"],
  ];
  $("#statCards", mount).innerHTML = cards.map(([k, v, sub]) => html`
    <div class="card stat">
      <span class="k">${k}</span><span class="v">${v}</span>
      <span class="tiny muted">${sub}</span>
    </div>`).join("");
}

function paintSamples(mount, samples) {
  const box = $("#samplesCard", mount);
  if (!box) return;
  const newest = [...samples].reverse();
  box.innerHTML = html`
    <div class="row-between" style="margin-bottom:6px">
      <h3 style="margin:0">What it can write so far</h3>
      <span class="tiny muted">newest first</span>
    </div>
    <p class="muted tiny">The model is asked to continue the same opening every
      so often. Early on this is noise; if training is working, words appear
      first, then grammar, then sense.</p>
    ${raw(newest.length ? html`
      <div class="samples">
        ${raw(newest.map((s) => {
          const seed = s.prompt || "";
          const rest = s.text.startsWith(seed) ? s.text.slice(seed.length) : s.text;
          return html`
            <div class="sample">
              <div class="hd">
                <span class="badge badge-accent">step ${s.step}</span>
              </div>
              <p class="txt"><span class="seed">${seed}</span>${rest}</p>
            </div>`;
        }).join(""))}
      </div>` : html`
      <p class="muted tiny" style="margin:10px 0 0">The first sample appears once
        training has run for a little while.</p>`)}`;
}

function appendLog(box, l) {
  const time = new Date((l.ts || 0) * 1000).toLocaleTimeString();
  const div = document.createElement("div");
  div.className = `lv-${l.level || "info"}`;
  div.innerHTML = `<span class="ts">${esc(time)}</span>${esc(l.line)}`;
  box.appendChild(div);
  while (box.childElementCount > 1200) box.removeChild(box.firstChild);
}
