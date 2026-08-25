import { api, events } from "../api.js";
import { html, raw, esc, $, on, fmtNum, fmtDuration, statusBadge, toast,
         skeletonValue, inlineRename } from "../util.js";
import { LineChart } from "../chart.js";
import { shareButton, wireShareBox } from "./share.js";

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
  // Writing a dataset is not training. It has no loss, no learning rate and no
  // held-out set; what it has is rows, and how many of them were worth
  // keeping. Rendering it through the training layout showed an empty "this
  // should go down" chart and three paragraphs of advice about what to do when
  // the learning rate is too high, for a run that has no learning rate.
  const writing = job.kind === "generate_dataset";
  const experts = +(job.config.arch?.num_local_experts || 0);

  if (writing) {
    mount.innerHTML = writingLayout(job);
    return writingView(mount, job, jobId, metrics, logs);
  }

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
  // Held-out loss is measured every `eval_every` steps, not every step, so
  // the newest metric row almost never carries one -- 96 rows in 97 on a
  // typical run. Reading it off `latest` therefore showed a dash almost
  // always, and once dashes started shimmering it pulsed forever, promising
  // a number that was hours away. It is a fact that stays true until the next
  // evaluation replaces it, so it is kept rather than re-read.
  let lastEval = [...metrics].reverse().find((m) => m.val_loss != null) || null;
  // The stage the runner last reported. paintStats runs on every metric and
  // must not claim "training" while the tokenizer is still being built.
  let stage = job.status === "running" ? "" : "training";
  const stats = () =>
    paintStats(mount, job, latest, scratch, stage, checkpointStep, lastEval);
  stats();

  const unsub = events.subscribe(async (msg) => {
    if (msg.job_id && msg.job_id !== jobId) return;

    if (msg.type === "job_metric") {
      if (msg.data.sample_text) return;   // samples arrive as job_sample
      latest = { step: msg.step, ...msg.data };
      if (msg.data.loss != null) lossChart.pushSeries("train", { x: msg.step, y: msg.data.loss });
      if (msg.data.val_loss != null) {
        lossChart.pushSeries("val", { x: msg.step, y: msg.data.val_loss });
        lastEval = { step: msg.step, val_loss: msg.data.val_loss };
      }
      if (msg.data.learning_rate != null)
        lrChart.push({ x: msg.step, y: msg.data.learning_rate });
      if (msg.data.expert_balance != null)
        expertChart?.push({ x: msg.step, y: msg.data.expert_balance });
      stats();
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
      stats();
      // A run that just finished has a model to build on, and one that just
      // failed has a checkpoint to carry on from. Both change this panel.
      paintFurther();
      paintReport();
    }
  });

  const paintOwnerRow = () => {
    const box = $("#ownerRow", mount);
    if (!box) return;
    const hasModel = job.artifacts?.length;
    box.innerHTML = hasModel ? publishCard(job) : "";
    wireShareBox(mount, "job", job, async () => {
      job = await api.job(jobId);
      paintHeader(mount, job);
      paintOwnerRow();
    });
  };
  paintOwnerRow();

  // Datasets are only needed by the "train this further" panel, which most
  // visits never open, so the list is fetched after the page is on screen.
  // The report only exists once a run has ended; asking for it while it is
  // still training would be diagnosing a curve that has not finished.
  const paintReport = async () => {
    const box = $("#reportCard", mount);
    if (!box) return;
    if (!["succeeded", "failed", "cancelled"].includes(job.status)) {
      box.innerHTML = "";
      return;
    }
    try { box.innerHTML = reportCard(await api.jobReport(jobId)); }
    catch { box.innerHTML = ""; }
  };
  paintReport();

  let datasets = [];
  const paintFurther = () => {
    const box = $("#furtherRow", mount);
    if (box) box.innerHTML = mergeCard(job) + furtherCard(job, datasets);
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
      stats();
    } catch (e) { toast(e.message, "err"); t.disabled = false; }
  });

  on(mount, "submit", "#mergeForm", async (e) => {
    e.preventDefault();
    const btn = $("#mergeGo", mount);
    btn.disabled = true;
    btn.textContent = "Queueing…";
    try {
      const { id } = await api.createJob({
        kind: "merge_adapter",
        config: { source_job: jobId, dtype: $("#mergeDtype", mount).value },
      });
      toast("Queued.", "ok");
      location.hash = `#/jobs/${id}`;
    } catch (ex) {
      toast(ex.message, "err");
      btn.disabled = false;
      btn.textContent = "Merge into a standalone model";
    }
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

  wireRunControls(mount, jobId, () => job, () => latest, () => stage);

  return () => {
    unsub(); lossChart.destroy(); lrChart.destroy(); expertChart?.destroy();
  };
}

// ---------------------------------------------------------------------------

/** Stop and Delete, wired once for every kind of run.
 *
 *  These were registered inside the training path, and the generation branch
 *  returns before reaching it. Delegated listeners are kept one-per-selector
 *  on the shared mount, so the buttons a generation run drew went on running
 *  the handler from whichever training run had been open last -- stopping that
 *  one instead, and leaving the generation run with no way to stop at all.
 *
 *  The state is read through getters rather than captured, because both
 *  callers redraw on every metric and a captured `job` would go stale within
 *  seconds of the page opening.
 */
function wireRunControls(mount, jobId, getJob, getLatest, getStage) {
  // A run is named when it is created, from the model and the dataset -- a
  // decent guess and a poor label once there are six of them. Both layouts on
  // this page draw the same heading, so it is wired once here.
  on(mount, "click", "#renameRun", () => {
    inlineRename($("#runTitle", mount), async (name) => {
      await api.renameJob(jobId, name);
      const job = getJob?.();
      if (job) job.name = name;
    });
  });

  // Delegated, not bound directly: paintHeader() replaces the button element
  // every time a metric arrives, which would silently discard a direct
  // listener. Stopping is two different actions wearing one button, and the
  // difference between them is hours of GPU time. A yes/no confirm can only
  // ask the question it was written with, so it is replaced by the actual
  // choice.
  on(mount, "click", "#cancelBtn", () => {
    $("#stopPanel", mount).innerHTML =
      stopPanel(getJob(), getLatest(), getStage());
  });
  on(mount, "click", "#stopCancel", () => {
    $("#stopPanel", mount).innerHTML = "";
  });

  on(mount, "click", "[data-stop]", async (_e, t) => {
    const how = t.dataset.stop;
    const save = how === "keep";
    if (how === "force" && !confirm(
      "Force stop this run?\n\n"
      + "The machine will be told to end it and will restart itself if the "
      + "run will not let go. Nothing is kept, and anything else that machine "
      + "is doing stops too.\n\n"
      + "Use this when an ordinary stop has already been tried and the run is "
      + "not responding.")) return;
    $("#stopPanel", mount).innerHTML = "";
    try {
      await api.cancelJob(jobId, save, how === "force");
      toast(how === "force" ? "Force-stopping. The machine may restart."
        : getJob().kind === "generate_dataset"
          ? "Stopping, and keeping the rows written so far…"
          : save ? "Stopping, and keeping the model…" : "Stopping…",
        how === "force" ? "err" : "");
    } catch (e) { toast(e.message, "err"); }
  });

  on(mount, "click", "#deleteBtn", async () => {
    const job = getJob();
    const hasModel = job.artifacts?.length;
    if (!confirm(`Delete "${job.name}"?

` + (
      job.kind === "generate_dataset"
        ? "Its log and measurements go. Any dataset it already produced stays."
        : hasModel
          ? "Its trained model file will be deleted too, and cannot be recovered."
          : "Its logs and measurements will be deleted."))) return;
    try {
      await api.deleteJob(jobId);
      toast("Run deleted.", "ok");
      location.hash = "#/jobs";
    } catch (e) { toast(e.message, "err"); }
  });
}

// ---------------------------------------------------------------------------
// Writing a dataset
// ---------------------------------------------------------------------------
//
// A generation run shares almost nothing with a training run except that both
// are jobs. It has no loss, no learning rate, no held-out set and no model at
// the end of it -- it has rows, and the interesting question is how many of
// them were worth keeping. Given the training layout it rendered an empty
// chart captioned "this should go down" above a paragraph explaining what to
// do when your learning rate is too high, which is advice about a number this
// run does not have.

const WRITING_MODES = {
  from_prompts: "answering your questions",
  from_topics: "covering your topics",
  from_seeds: "writing variations on your seeds",
  extend_conversations: "adding turns to existing conversations",
};

function writingLayout(job) {
  const cfg = job.config || {};
  const model = cfg.model || {};
  const who = model.model || model.base_model
    || (model.provider ? `a model at ${model.provider}` : "your own model");
  const what = WRITING_MODES[cfg.mode] || "writing rows";
  const extending = cfg.mode === "extend_conversations";
  return html`
    <div class="page-head">
      <a href="#/jobs" class="tiny">← All runs</a>
      <div class="row-between" style="flex-wrap:wrap;gap:8px;margin-top:6px">
        <div class="row title-row" style="gap:4px;min-width:0">
          <h1 style="margin:0" id="runTitle">${job.name}</h1>
          <button class="btn-sm btn-quiet" id="renameRun" title="Rename this run"
            aria-label="Rename this run">&#9998;</button>
        </div>
        <div class="row" id="headerActions"></div>
      </div>
      <p class="sub tiny" style="margin-top:4px">
        <span class="badge badge-accent">writing a dataset</span>
        ${what} with <span class="mono">${who}</span>${
          cfg.source_label ? `, from ${cfg.source_label}` : ""}</p>
    </div>

    <div id="errorCard"></div>
    <div id="queueCard"></div>
    <div id="stopPanel"></div>
    <div id="progressCard"></div>
    <div class="grid grid-3" id="statCards" style="margin-bottom:16px"></div>

    <div class="card" style="margin-bottom:14px"><div id="rowsChart"></div></div>

    <div class="grid grid-2" style="margin-bottom:14px">
      <div class="card">
        <h3>What am I looking at?</h3>
        <p class="muted tiny">This run is not training anything. It calls a
          model over and over and keeps what comes back, so there is no loss
          curve — the number that matters is how many rows survived.</p>
        <ul class="muted tiny" style="margin:0;padding-left:18px;line-height:1.7">
          ${raw(extending ? html`
            <li><strong>Lengthened</strong> — conversations that actually grew.
              A row the provider refused is still written out, unchanged, so
              the dataset does not quietly shrink.</li>
            <li><strong>Invented results</strong> — no tool actually ran, so the
              model made up what each one returned. Plausible, and not true.
              Right for teaching the shape of a tool conversation, wrong for
              teaching facts about your systems.</li>` : html`
            <li><strong>Repeats</strong> — the same row written twice. A model
              asked the same thing twice answers it the same way; if this
              climbs, the prompts are not varied enough or the temperature is
              too low.</li>
            <li><strong>Empty</strong> — nothing usable came back. A few is
              normal; a lot means the instructions are not landing.</li>`)}
          <li>Nothing here can be better than the model that wrote it. Read the
            rows on the dataset page before you train on them.</li>
        </ul>
      </div>
      <div class="card">
        <h3>The data it wrote</h3>
        <div id="datasetCard"></div>
      </div>
    </div>

    <div class="card">
      <div class="row-between" style="align-items:center">
        <h3 style="margin:0">Log</h3>
        <span class="tiny muted">what the run said as it went</span>
      </div>
      <div class="logbox" id="logBox"></div>
    </div>`;
}

/** The live half of a generation run: rows over time, and the log. */
function writingView(mount, job, jobId, metrics, logs) {
  const extending = job.config?.mode === "extend_conversations";
  const rowsChart = new LineChart($("#rowsChart", mount), {
    title: "Rows kept — this should climb steadily",
    height: 260,
    format: (v) => fmtNum(v),
    series: [
      { key: "kept", label: "Kept" },
      { key: "lost", label: extending ? "Could not be extended"
                                      : "Repeats and empties", dashed: true },
    ],
  });
  const lost = (m) => (m.duplicates || 0) + (m.empty || 0) + (m.failed || 0);
  const withRows = metrics.filter((m) => m.rows != null);
  rowsChart.setSeries("kept", withRows.map((m) => ({ x: m.step, y: m.rows })));
  rowsChart.setSeries("lost", withRows.map((m) => ({ x: m.step, y: lost(m) })));

  const logBox = $("#logBox", mount);
  logs.forEach((l) => appendLog(logBox, l));
  logBox.scrollTop = logBox.scrollHeight;

  let latest = metrics[metrics.length - 1] || {};
  let stage = job.status === "running" ? "" : "training";
  wireRunControls(mount, jobId, () => job, () => latest, () => stage);

  const paint = () => {
    paintHeader(mount, job);
    paintProgress(mount, job, stage, null, null, 0);
    const q = $("#queueCard", mount);
    if (q) q.innerHTML = queueCard(job);
    paintWritingStats(mount, job, latest, extending);
    const dc = $("#datasetCard", mount);
    if (dc) dc.innerHTML = writtenDatasetCard(job);
  };
  paint();

  const unsub = events.subscribe(async (msg) => {
    if (msg.job_id && msg.job_id !== jobId) return;
    if (msg.type === "job_metric") {
      latest = { step: msg.step, ...msg.data };
      if (msg.data.rows != null) {
        rowsChart.pushSeries("kept", { x: msg.step, y: msg.data.rows });
        rowsChart.pushSeries("lost", { x: msg.step, y: lost(msg.data) });
      }
      paint();
    } else if (msg.type === "job_log") {
      const atBottom = logBox.scrollHeight - logBox.scrollTop - logBox.clientHeight < 40;
      appendLog(logBox, { ts: Date.now() / 1000, level: msg.level, line: msg.line });
      if (atBottom) logBox.scrollTop = logBox.scrollHeight;
    } else if (msg.type === "job_progress") {
      stage = msg.stage;
      if (stage === "training") { job.step = msg.step; job.total_steps = msg.total; }
      paintProgress(mount, job, stage, msg.step, msg.total);
    } else if (msg.type === "jobs_changed") {
      job = await api.job(jobId);
      paint();
    }
  });

  return () => { unsub(); rowsChart.destroy(); };
}

function paintWritingStats(mount, job, m, extending) {
  const running = job.status === "running";
  const cards = extending ? [
    ["Conversations", m.rows != null ? fmtNum(m.rows) : "—", "written out"],
    ["Lengthened", m.extended != null ? fmtNum(m.extended) : "—",
     "rows that actually grew"],
    ["Invented results", m.invented_results != null
      ? fmtNum(m.invented_results) : "—", "no tool actually ran"],
    ["Could not be extended", m.failed != null ? fmtNum(m.failed) : "—",
     "kept unchanged"],
    ["Speed", m.rows_per_sec != null ? `${m.rows_per_sec.toFixed(2)}/s` : "—",
     "rows per second"],
    ["Time left", m.eta_s != null && running ? fmtDuration(m.eta_s) : "—",
     "estimate"],
  ] : [
    ["Rows kept", m.rows != null ? fmtNum(m.rows) : "—", "written so far"],
    ["Repeats dropped", m.duplicates != null ? fmtNum(m.duplicates) : "—",
     "identical to an earlier row"],
    ["Empty replies", m.empty != null ? fmtNum(m.empty) : "—",
     "nothing usable came back"],
    ["Speed", m.rows_per_sec != null ? `${m.rows_per_sec.toFixed(2)}/s` : "—",
     "rows per second"],
    ["Model speed", m.tokens_per_sec != null
      ? `${fmtNum(m.tokens_per_sec)}/s` : "—", "tokens per second"],
    ["Time left", m.eta_s != null && running ? fmtDuration(m.eta_s) : "—",
     "estimate"],
  ];

  const coming = ["queued", "assigned", "running"].includes(job.status);
  $("#statCards", mount).innerHTML = cards.map(([k, v, sub]) => html`
    <div class="card stat">
      <span class="k">${k}</span>
      <span class="v">${v === "—" && coming ? skeletonValue("4em") : v}</span>
      <span class="tiny muted">${sub}</span>
    </div>`).join("");
}

/** Stopping a generation run, which has nothing half-finished in it.
 *
 *  Unlike training, there is no partly-built thing to weigh up: every row
 *  already written is a complete row and is kept whatever you choose. So this
 *  asks one question instead of offering a trade-off that does not exist here.
 */
function writingStopPanel(job, latest) {
  const rows = latest?.rows || 0;
  return html`
    <div class="card callout-warn" style="margin-bottom:14px">
      <h3 style="margin:0 0 6px">Stop writing?</h3>
      <p class="muted tiny" style="margin:0 0 12px">
        ${raw(rows ? html`
          <strong>${fmtNum(rows)}</strong> row${rows === 1 ? "" : "s"} have been
          written and every one of them is finished. They are kept and
          registered as a dataset — there is no half-written row to lose.`
          : html`Nothing has been written yet, so stopping now leaves nothing
          behind.`)}</p>
      <div class="row" style="gap:8px;flex-wrap:wrap">
        <button class="btn-primary btn-sm" data-stop="keep">
          ${raw(rows ? `Stop and keep the ${fmtNum(rows)} rows` : "Stop the run")}</button>
        <button class="btn-sm" id="stopCancel">Carry on writing</button>
      </div>
    </div>`;
}

/** Where the rows went, once there are any. */
function writtenDatasetCard(job) {
  const made = job.summary?.dataset_id || job.dataset_id;
  if (made) {
    return html`
      <p class="muted tiny">Registered as a dataset in this studio. Look at the
        rows before you train on them — a generated dataset can be fluent and
        wrong at the same time.</p>
      <a class="btn btn-primary btn-sm" href="#/data/${esc(made)}">Open it</a>`;
  }
  if (["running", "queued", "assigned"].includes(job.status)) {
    return html`<p class="muted tiny">The dataset is registered when the run
      finishes. Stopping early keeps every row written so far.</p>`;
  }
  if (job.status === "failed") {
    return html`<p class="muted tiny">This run did not get far enough to leave
      one behind.</p>`;
  }
  return html`<p class="muted tiny">Look for it on the
    <a href="#/data">datasets page</a>.</p>`;
}

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
        <div class="row title-row" style="gap:4px;min-width:0">
          <h1 style="margin:0" id="runTitle">${job.name}</h1>
          <button class="btn-sm btn-quiet" id="renameRun" title="Rename this run"
            aria-label="Rename this run">&#9998;</button>
        </div>
        <div class="row" id="headerActions"></div>
      </div>
      <p class="sub mono tiny" style="margin-top:4px">${subtitle}</p>
      ${raw(job.config.sweep_id ? html`
        <p class="tiny" style="margin:4px 0 0">One of several variants —
          <a href="#/sweeps/${job.config.sweep_id}">see them side by
          side</a>.</p>` : "")}
    </div>

    <div id="errorCard"></div>
    <div id="reportCard"></div>
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

const REPORT_ICON = { ok: "✓", warn: "!", error: "✕" };

/** What the run says about itself. Findings first, numbers second: the
 *  numbers are only interesting once you know which of them to look at. */
function reportCard(r) {
  const findings = r.findings || [];
  if (!findings.length && !(r.facts || []).length) return "";
  const worst = findings.some((f) => f.level === "error") ? "error"
    : findings.some((f) => f.level === "warn") ? "warn" : "ok";
  return html`
    <div class="card" style="margin-bottom:14px">
      <div class="row-between" style="margin-bottom:8px">
        <h3 style="margin:0">What happened in this run</h3>
        <span class="badge ${worst === "error" ? "badge-err"
          : worst === "warn" ? "badge-warn" : "badge-ok"}">${
          worst === "error" ? "something went wrong"
            : worst === "warn" ? "worth a look" : "healthy"}</span>
      </div>

      ${raw((r.facts || []).length ? html`
        <div class="factrow">
          ${raw(r.facts.map((f) => html`
            <div class="fact">
              <span class="k">${f.label}</span>
              <span class="v">${f.value}</span>
              ${raw(f.note ? `<span class="n">${esc(f.note)}</span>` : "")}
            </div>`).join(""))}
        </div>` : "")}

      ${raw(findings.map((f) => html`
        <div class="finding lv-${f.level}">
          <span class="mark">${REPORT_ICON[f.level] || "·"}</span>
          <div>
            <strong>${f.title}</strong>
            <p class="saw">${f.saw}</p>
            <p class="means">${f.means}</p>
            <p class="do"><span class="lbl">What to do</span> ${f.do}</p>
          </div>
        </div>`).join(""))}
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
          are added on top. Its base model, <code>${cfg.base_model || "—"}</code>,
          stays the same.`)}</p>

      <form id="furtherForm" style="margin-top:12px">
        <div class="field">
          <label for="furtherData">Text to learn from</label>
          <select id="furtherData" name="studio_dataset">
            <option value="">Keep the same source (${
              cfg.dataset_label || String(cfg.dataset || "").split("/").pop() || "—"})</option>
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
                 value="${job.name} (continued)">
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

/** A fine-tune produces an adapter, which is the right thing to produce and
 *  the wrong thing to hand somebody. */
function mergeCard(job) {
  if (job.kind !== "finetune_llm" || !job.artifacts?.length) return "";
  if (!["succeeded", "cancelled"].includes(job.status)) return "";
  return html`
    <details class="card" style="margin-bottom:14px">
      <summary><strong>Make a standalone model</strong>
        <span class="muted tiny"> — one file set, no base model needed</span>
      </summary>
      <p class="muted tiny" style="margin:10px 0 0">
        This run produced an <em>adapter</em>: a few megabytes that mean
        nothing without <code>${job.config.base_model || "its base model"}</code>.
        Merging folds it in and writes a complete model that loads on its own —
        which is what you need to run it in Ollama, llama.cpp, or anywhere
        outside this studio.</p>
      <p class="muted tiny" style="margin:6px 0 0">
        <strong>It will be the size of the base model</strong>, not of the
        adapter. The adapter stays where it is and is still the better thing
        to use inside the studio.</p>
      <form id="mergeForm" style="margin-top:10px">
        <div class="field">
          <label for="mergeDtype">Precision</label>
          <select id="mergeDtype" name="dtype">
            <option value="float16">float16 — half the size, the usual choice</option>
            <option value="bfloat16">bfloat16</option>
            <option value="float32">float32 — exact, twice the size</option>
          </select>
          <div class="hint">The arithmetic is done in full precision either
            way; this is only what gets written out.</div>
        </div>
        <button class="btn-primary btn-sm" type="submit" id="mergeGo">
          Merge into a standalone model</button>
      </form>
    </details>`;
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
                 placeholder="your-name/${slug}">
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

/** The last resort, offered only where the ordinary stop has visibly failed.
 *
 *  Stopping is cooperative: the trainer checks between steps, which is what
 *  lets a stop keep the model built so far. It only works while steps finish.
 *  A run wedged inside one -- a model too large for the card, thrashing the
 *  allocator -- never reaches the check, and the button does nothing however
 *  many times it is pressed.
 *
 *  Not shown by default, because it is worse in every case where the ordinary
 *  stop works: it keeps nothing, and it takes the machine down with it. It
 *  appears once a stop has already been asked for and the run carried on
 *  regardless, which is exactly the situation it is for.
 */
function forceOption(job, latest) {
  const asked = job.status === "running" && (job.error || "").length === 0
    && (job.cancel_requested || wasAskedToStop(job));
  if (!asked) {
    return html`
      <details class="adv" style="margin-top:10px">
        <summary>It is not stopping</summary>
        <p class="muted tiny" style="margin:8px 0 10px">Stopping waits for the
          current step to finish, so that whatever has been built can be kept.
          A run that is stuck inside a step never gets that far — the request
          is received and never acted on. Forcing ends the machine's process
          instead: nothing is kept, anything else it is doing stops too, and it
          restarts by itself within seconds.</p>
        <button class="btn-danger btn-sm" data-stop="force">
          Force stop, losing everything</button>
      </details>`;
  }
  return html`
    <div class="callout callout-warn" style="margin-top:10px">
      <strong>Already asked to stop</strong>
      This run was asked to stop and has not. That happens when it is stuck
      inside a step rather than between two, which is where the request is
      read. Forcing ends the machine's process: nothing is kept, and it
      restarts by itself.
      <div style="margin-top:8px">
        <button class="btn-danger btn-sm" data-stop="force">
          Force stop, losing everything</button>
      </div>
    </div>`;
}

/** Whether a stop has already been asked for, read from the run's own log. */
function wasAskedToStop(job) {
  return !!(job.summary && job.summary.stop_requested);
}

function stopPanel(job, latest, stage) {
  if (job.kind === "generate_dataset") return writingStopPanel(job, latest);
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
        </div>
        ${raw(forceOption(job, latest))}` : html`
        <p class="muted tiny" style="margin:0 0 12px">
          Training has not started yet — the runner is still preparing your
          data, so there is no ${kind} to keep. Stopping now leaves nothing
          behind.</p>
        <div class="row" style="gap:8px">
          <button class="btn-danger btn-sm" data-stop="discard">Stop the run</button>
          <button class="btn-sm" id="stopCancel">Carry on</button>
        </div>`)}
      ${raw(forceOption(job, latest))}
    </div>`;
}

function paintHeader(mount, job) {
  const box = $("#headerActions", mount);
  const done = ["succeeded", "failed", "cancelled"].includes(job.status);
  // A stopped run that kept its model is as usable as a finished one. The
  // artifact is what decides that, not how the run ended.
  // A generation run leaves a dataset, not a model. Offering "Try it out"
  // pointed the playground at a job it cannot serve, and "Compare" offered to
  // rank a held-out loss that does not exist.
  const writing = job.kind === "generate_dataset";
  const usable = job.artifacts?.length && !writing
    && ["succeeded", "cancelled"].includes(job.status);
  const kept = job.status === "cancelled" && job.artifacts?.length;
  box.innerHTML = html`
    ${statusBadge(job.status)}
    ${raw(kept ? `<span class="badge badge-ok">${
      writing ? "rows kept" : "model kept"}</span>` : "")}
    ${raw(usable
      ? `<a class="btn btn-primary btn-sm" href="#/play/${esc(job.id)}">▷ Try it out</a>
         <a class="btn btn-sm" href="#/compare" title="Compare its held-out loss with other runs">⇄ Compare</a>` : "")}
    ${raw(done && job.artifacts?.length
      ? `<a class="btn btn-sm" href="/api/jobs/${esc(job.id)}/download">
           ↓ Download${writing ? " the JSONL" : ""}</a>` : "")}
    ${raw(shareButton("job", job))}
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
            <strong>${r.runner || "That machine"}</strong> is not
            connected, and the checkpoint is on its disk. Bring it back and
            this button appears.</p>`)}
    </div>`;
}

function paintStats(mount, job, m, scratch, stage = "training",
                   checkpointStep = 0, lastEval = null) {
  paintHeader(mount, job);
  paintProgress(mount, job, stage, null, null,
                checkpointStep || job.checkpoint_step || 0);
  const q = $("#queueCard", mount);
  if (q) q.innerHTML = queueCard(job);
  const rc = $("#resumeCard", mount);
  if (rc) rc.innerHTML = resumeCard(job);
  const held = heldOutCard(job, m, lastEval);
  const cards = scratch ? [
    ["Loss now", m.loss != null ? m.loss.toFixed(4) : "—", "lower is better"],
    held,
    ["Text read", m.tokens_seen != null ? fmtNum(m.tokens_seen) : "—", "tokens so far"],
    ["Speed", m.tokens_per_sec != null ? fmtNum(m.tokens_per_sec) + "/s" : "—", "tokens per second"],
    ["GPU memory", m.vram_gb != null ? m.vram_gb + " GB" : "—", "peak used"],
    ["Time left", m.eta_s != null && job.status === "running"
      ? fmtDuration(m.eta_s) : "—", "estimate"],
  ] : [
    ["Loss now", m.loss != null ? m.loss.toFixed(4) : "—", "lower is better"],
    held,
    ["Speed", m.steps_per_sec != null ? m.steps_per_sec.toFixed(2) + "/s" : "—", "steps per second"],
    ["GPU memory", m.vram_gb != null ? m.vram_gb + " GB" : "—", "peak used"],
    ["Time left", m.eta_s != null && job.status === "running"
      ? fmtDuration(m.eta_s) : "—", "estimate"],
  ];
  // A dash means "there is no such number"; a shimmer means "it is on its
  // way". Those are different states and the page used to show both as "—",
  // so a run that had just started looked broken for its first few seconds.
  // Only for numbers that arrive with the very first step. Anything on a
  // slower schedule says when it is due instead -- a placeholder that pulses
  // for an hour is a lie about how long you are waiting.
  const NEVER_SHIMMER = new Set(["Held-out loss"]);
  const coming = ["queued", "assigned", "running"].includes(job.status);
  $("#statCards", mount).innerHTML = cards.map(([k, v, sub]) => html`
    <div class="card stat">
      <span class="k">${k}</span>
      <span class="v">${v === "—" && coming && !NEVER_SHIMMER.has(k)
        ? skeletonValue("4em") : v}</span>
      <span class="tiny muted">${sub}</span>
    </div>`).join("");
}

/** Held-out loss, which happens on its own schedule.
 *
 *  It is checked every `eval_every` steps, so between checks there is no new
 *  number -- but the last one is still the truth, and saying when the next is
 *  due is more use than a placeholder that pulses in the meantime.
 */
function heldOutCard(job, m, lastEval) {
  const sub = job.kind === "pretrain_llm" ? "on unseen text" : "on unseen examples";
  const value = m.val_loss ?? lastEval?.val_loss;
  const at = m.val_loss != null ? m.step : lastEval?.step;
  if (value != null) {
    return ["Held-out loss", value.toFixed(4),
            at ? `${sub} · step ${fmtNum(at)}` : sub];
  }
  const every = job.config?.eval_every;
  const live = ["queued", "assigned", "running"].includes(job.status);
  if (live && every) {
    return ["Held-out loss", "—", `${sub} · first check at step ${fmtNum(every)}`];
  }
  return ["Held-out loss", "—", live ? sub : `${sub} · never measured`];
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
