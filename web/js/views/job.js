import { api, events } from "../api.js";
import { html, raw, esc, $, on, fmtNum, fmtDuration, statusBadge, toast } from "../util.js";
import { LineChart } from "../chart.js";

const STAGES = {
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

  mount.innerHTML = layout(job, scratch);

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

  let latest = metrics[metrics.length - 1] || {};
  paintStats(mount, job, latest, scratch);

  const unsub = events.subscribe(async (msg) => {
    if (msg.job_id && msg.job_id !== jobId) return;

    if (msg.type === "job_metric") {
      if (msg.data.sample_text) return;   // samples arrive as job_sample
      latest = { step: msg.step, ...msg.data };
      if (msg.data.loss != null) lossChart.pushSeries("train", { x: msg.step, y: msg.data.loss });
      if (msg.data.val_loss != null) lossChart.pushSeries("val", { x: msg.step, y: msg.data.val_loss });
      if (msg.data.learning_rate != null)
        lrChart.push({ x: msg.step, y: msg.data.learning_rate });
      paintStats(mount, job, latest, scratch);
    } else if (msg.type === "job_sample") {
      samples.push({ step: msg.step, text: msg.text, prompt: msg.prompt });
      paintSamples(mount, samples);
    } else if (msg.type === "job_log") {
      const atBottom = logBox.scrollHeight - logBox.scrollTop - logBox.clientHeight < 40;
      appendLog(logBox, { ts: Date.now() / 1000, level: msg.level, line: msg.line });
      if (atBottom) logBox.scrollTop = logBox.scrollHeight;
    } else if (msg.type === "job_progress") {
      job.step = msg.step; job.total_steps = msg.total;
      paintProgress(mount, job, msg.stage);
    } else if (msg.type === "jobs_changed") {
      job = await api.job(jobId);
      paintHeader(mount, job);
      paintStats(mount, job, latest, scratch);
    }
  });

  // Delegated, not bound directly: paintHeader() replaces the button element
  // every time a metric arrives, which would silently discard a direct listener.
  on(mount, "click", "#cancelBtn", async () => {
    if (!confirm("Stop this training run? Progress so far will be lost.")) return;
    try { await api.cancelJob(jobId); toast("Stopping…"); }
    catch (e) { toast(e.message, "err"); }
  });

  return () => { unsub(); lossChart.destroy(); lrChart.destroy(); };
}

// ---------------------------------------------------------------------------

function layout(job, scratch) {
  const subtitle = scratch
    ? `from scratch · ${job.config.dataset}`
    : `${job.config.base_model} → ${job.config.dataset}`;
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
    <div id="progressCard"></div>
    <div class="grid grid-3" id="statCards" style="margin-bottom:16px"></div>

    <div class="card" style="margin-bottom:14px"><div id="lossChart"></div></div>

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
          ${raw(scratch ? html`
          <li><strong>Held-out rising while training falls</strong> — it has
            started memorising the text instead of learning from it. Use more
            data, or stop here.</li>` : html`
          <li><strong>Falling to near zero</strong> — it is memorising rather than
            learning. Use fewer passes.</li>`)}
        </ul>
      </div>
    </div>

    <div class="card">
      <div class="row-between" style="margin-bottom:8px">
        <h3 style="margin:0">Log</h3>
        <span class="tiny muted">Newest at the bottom</span>
      </div>
      <div class="logbox" id="logBox"></div>
    </div>`;
}

function paintHeader(mount, job) {
  const box = $("#headerActions", mount);
  const done = ["succeeded", "failed", "cancelled"].includes(job.status);
  const usable = job.status === "succeeded" && job.artifacts?.length;
  box.innerHTML = html`
    ${statusBadge(job.status)}
    ${raw(usable
      ? `<a class="btn btn-primary btn-sm" href="#/play/${esc(job.id)}">💬 Try it out</a>` : "")}
    ${raw(done && job.artifacts?.length
      ? `<a class="btn btn-sm" href="/api/jobs/${esc(job.id)}/download">
           ↓ Download</a>` : "")}
    ${raw(!done ? `<button class="btn-danger btn-sm" id="cancelBtn">Stop</button>` : "")}`;

  // Assigned, not prepended: paintHeader runs on every metric tick, and
  // prepending would stack a fresh copy of the error on each one.
  $("#errorCard", mount).innerHTML = job.error
    ? html`<div class="callout callout-err">
        <strong>This run failed</strong>${job.error}</div>`
    : "";
}

function paintProgress(mount, job, stage = "") {
  const pct = job.total_steps ? Math.min(100, (job.step / job.total_steps) * 100) : 0;
  const running = ["running", "assigned"].includes(job.status);
  const stageText = STAGES[stage] || (running ? "Working…" : "");
  // Preparation stages report their own progress with no step total, and
  // showing "step 0 of 0" during a five-minute tokenizer build reads as stuck.
  const counted = job.total_steps > 0 && stage === "training";

  $("#progressCard", mount).innerHTML = running ? html`
    <div class="card" style="margin-bottom:16px">
      <div class="row-between" style="margin-bottom:8px">
        <strong class="tiny">${stageText}</strong>
        <span class="tiny muted">${counted
          ? `step ${job.step} of ${job.total_steps}` : ""}</span>
      </div>
      <div class="progress"><i style="width:${pct}%"></i></div>
    </div>` : "";
}

function paintStats(mount, job, m, scratch) {
  paintHeader(mount, job);
  paintProgress(mount, job, "training");
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
