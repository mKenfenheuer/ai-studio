import { api, events } from "../api.js";
import { html, raw, esc, $, $$, on, fmtNum, fmtBytes, fmtDuration, fmtAgo,
         statusBadge, toast, modal, skeletonValue, inlineRename } from "../util.js";
import { LineChart } from "../chart.js";
import { shareButton, wireShareBox } from "./share.js";
import { openScoreDialog } from "../scoring.js";
import { publishCard, wirePublish } from "./publish.js";
import { publishDialog, fileIntoDialog } from "./projects.js";
import { mountCardEditor } from "./cardeditor.js";
import { kindOf, subjectOf, stagesFor } from "../kinds.js";
import { ribbon, rb, group, wireRibbon, tabState } from "../ribbon.js";
import { breadcrumb, confirmDestructive } from "../components.js";

export async function jobView(mount, [jobId]) {
  let job = await api.job(jobId);
  const metrics = await api.jobMetrics(jobId);
  const logs = await api.jobLogs(jobId);
  const scratch = job.kind === "pretrain_llm";
  const experts = +(job.config.arch?.num_local_experts || 0);

  // Only two of the six kinds of run are training, and the other four were all
  // being drawn through the training layout: an empty "this should go down"
  // loss chart, a learning rate that does not exist, and three paragraphs of
  // advice about what to do when the loss jumps -- above a run that was
  // writing rows with a hosted model, or sending files to Hugging Face, or
  // adding two tensors together. Each kind gets the page its own work needs.
  const OWN_PAGE = {
    // Writing a dataset has no loss and no held-out set. It has rows, and how
    // many of them were worth keeping.
    generate_dataset: [writingLayout, (m, j) => writingView(m, j, jobId, metrics, logs)],
    // An upload has a destination and a byte count, and produces nothing here
    // at all -- what it produces is on the Hub.
    upload: [uploadLayout, (m, j) => uploadView(m, j, jobId, logs)],
    // A merge has no steps to plot. It has an adapter, a base, and a complete
    // model at the end of it, which is the thing most worth publishing in the
    // studio and had nowhere to be published from.
    merge_adapter: [mergeLayout, (m, j) => mergeView(m, j, jobId, logs)],
    // A scoring run's result is the comparison, not a curve.
    evaluate: [evalLayout, (m, j) => evalView(m, j, jobId, logs)],
  };
  if (OWN_PAGE[job.kind]) {
    const [layoutFor, view] = OWN_PAGE[job.kind];
    mount.innerHTML = layoutFor(job);
    return view(mount, job);
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

  const samples = metrics.filter((m) => m.sample_text || m.sample_asset)
    .map((m) => ({ step: m.step, kind: m.sample_kind || "text", text: m.sample_text,
                   prompt: m.sample_prompt, asset_id: m.sample_asset,
                   caption: m.sample_caption }));
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
  const getTab = wireRunTabs(mount, job, () => stats());
  const stats = () => {
    paintHeader(mount, job, getTab());
    paintStats(mount, job, latest, scratch, stage, checkpointStep, lastEval);
  };
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
      samples.push({ step: msg.step, kind: msg.kind || "text", text: msg.text,
                     prompt: msg.prompt, asset_id: msg.asset_id, caption: msg.caption });
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
      paintHeader(mount, job, getTab());
      stats();
      // A run that just finished has a model to build on, and one that just
      // failed has a checkpoint to carry on from. Both change this panel.
      paintFurther();
      paintProvenance();
      paintReport();
      // And a run that just finished has a card, written from numbers that
      // did not exist a second ago.
      paintCard();
    }
  });

  const paintOwnerRow = wireOwnerRow(mount, jobId, () => job, async () => {
    job = await api.job(jobId);
    paintHeader(mount, job, getTab());
  });
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
    try { box.innerHTML = reportCard(await api.jobReport(jobId), jobId); }
    catch { box.innerHTML = ""; }
  };
  paintReport();

  const paintCard = wireModelCard(mount, jobId, () => job);
  paintCard();

  let datasets = [];
  const paintFurther = () => {
    const box = $("#furtherRow", mount);
    if (box) box.innerHTML = artifactsCard(job) + furtherCard(job, datasets);
  };
  const paintProvenance = () => {
    const box = $("#provenanceRow", mount);
    if (box) box.innerHTML = provenanceCard(job);
  };
  paintFurther();
  paintProvenance();
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
/** The ribbon's tabs, wired to whatever the page repaints with.
 *
 *  Remembered per kind of run rather than per run: which tab you want on a
 *  training run is a habit, and it is a different habit from the one you want
 *  on a scoring run. */
function wireRunTabs(mount, job, repaint) {
  const tabs = tabState(`run.${kindOf(job).page}`, RUN_TABS(job), "home");
  let tab = tabs.get();
  wireRibbon(mount, (key) => { tab = key; tabs.set(key); repaint(); });
  return () => tab;
}

function wireRunControls(mount, jobId, getJob, getLatest, getStage) {
  // A run is named when it is created, from the model and the dataset -- a
  // decent guess and a poor label once there are six of them. Both layouts on
  // this page draw the same heading, so it is wired once here.
  // Saved on blur rather than on every keystroke: it is a note, not a form.
  on(mount, "change", "#runNotes", async (_e, t) => {
    const job = getJob();
    if ((job.notes || "") === t.value) return;
    try {
      await api.patchJob(jobId, { notes: t.value });
      job.notes = t.value;
      toast("Noted.", "ok");
    } catch (e) { toast(e.message, "err"); }
  });

  on(mount, "change", "#runTags", async (_e, t) => {
    const job = getJob();
    try {
      const r = await api.patchJob(jobId, { tags: t.value });
      job.tags = r.tags || [];
      t.value = job.tags.join(", ");
      toast(job.tags.length ? "Tagged." : "Tags cleared.", "ok");
    } catch (e) { toast(e.message, "err"); }
  });

  on(mount, "click", "[data-rename]", () => {
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

  // "Score it" had no home on this page at all: the question a finished run
  // raises is whether it is better than what you started with, and answering
  // it meant navigating to Evaluate and finding the run again in a list of
  // every model in the studio. The dialog is shared with the playground,
  // which raises the same question three messages into a conversation.
  on(mount, "click", "#scoreRun", () => openScoreDialog(getJob()));

  // ---- asking a classifier about a picture
  async function tryPicture(file) {
    const status = $("#tryStatus", mount);
    const out = $("#tryResult", mount);
    if (!file || !status || !out) return;
    status.textContent = "Asking…";
    out.innerHTML = `<img src="${URL.createObjectURL(file)}" alt="" style="max-width:200px;max-height:160px;border-radius:6px;display:block">`;
    try {
      const r = await api.classify(jobId, file);
      status.textContent = `${r.runner ? `on ${r.runner} · ` : ""}${r.seconds}s`;
      const top = r.labels?.[0];
      out.innerHTML += html`
        <p style="margin:8px 0 4px"><strong>${top?.label || "—"}</strong>
          <span class="muted tiny">${top ? `${(top.probability * 100).toFixed(1)}% sure` : ""}</span></p>
        <table class="table" style="max-width:420px"><tbody>
          ${raw((r.labels || []).map((l) => html`
            <tr><td class="tiny">${l.label}</td>
              <td style="min-width:160px"><div class="meter"><i style="width:${(l.probability * 100).toFixed(1)}%"></i></div></td>
              <td class="tiny muted">${(l.probability * 100).toFixed(1)}%</td></tr>`).join(""))}
        </tbody></table>`;
    } catch (e) { status.textContent = ""; toast(e.message, "err"); }
  }
  on(mount, "change", "#tryFile", (_e, t) => tryPicture(t.files?.[0]));
  on(mount, "click", "#tryClassifier", () => {
    const card = $("#tryCard", mount);
    if (card) { card.hidden = false; card.scrollIntoView({ behavior: "smooth" }); $("#tryFile", mount)?.click(); }
  });
  const tryCardEl = $("#tryCard", mount);
  if (tryCardEl) {
    tryCardEl.addEventListener("dragover", (e) => { e.preventDefault(); tryCardEl.classList.add("dropping"); });
    tryCardEl.addEventListener("dragleave", () => tryCardEl.classList.remove("dropping"));
    tryCardEl.addEventListener("drop", (e) => {
      e.preventDefault(); tryCardEl.classList.remove("dropping");
      tryPicture(e.dataTransfer.files?.[0]);
    });
  }

  on(mount, "click", "#dropModel", async () => {
    const job = getJob();
    const size = (job.artifacts || []).reduce((n, a) => n + (a.size_bytes || 0), 0);
    if (!await confirmDestructive({
      title: `Remove the model from "${job.name}"?`,
      consequences: [
        `${(size / 1024 ** 3).toFixed(1)} GB comes back. The run, its chart, its log and its notes stay.`,
        "It cannot be talked to, scored, published or exported afterwards — \"Run again\" trains it afresh.",
      ],
      confirmLabel: "Remove the model" })) return;
    try {
      const r = await api.dropJobModel(jobId);
      toast(`Removed — ${(r.bytes / 1024 ** 3).toFixed(1)} GB freed.`, "ok");
      job = await api.job(jobId);
      paintHeader(mount, job, getTab());
      paintCard();
      paintFurther();
    } catch (e) { toast(e.message, "err"); }
  });

  on(mount, "click", "#deployIt", async () => {
    const job = getJob();
    let runners = [], deployments = [];
    try {
      [runners, deployments] = await Promise.all([
        api.runners(), api.deployments().catch(() => [])]);
    } catch (e) { return toast(e.message, "err"); }
    // A machine with no card would answer at a word every few seconds, and a
    // machine reserved for training is somebody's arrangement. Neither is
    // offered rather than being offered and then refused by the server.
    const servers = runners.filter((r) =>
      ["cuda", "rocm", "mps"].includes((r.capabilities || {}).backend)
      && r.role !== "training");
    const already = deployments.filter((d) => d.job_id === job.id);

    const dlg = modal({ title: `Hold "${job.name}" on a machine`, width: 520,
      body: html`
      <p class="muted tiny">The model is loaded now and kept there, so the
        first message does not wait for a fetch and a load. It goes back by
        itself after a restart, and nothing else pushes it off the card.</p>
      ${raw(already.length ? html`
        <div class="callout callout-ok">
          <strong>Already held on</strong>
          ${already.map((d) => esc(d.runner_name)).join(", ")}.
        </div>` : "")}
      <div class="field">
        <label for="dpTo">Machine</label>
        <select id="dpTo">${raw(servers.length
          ? servers.filter((r) => !already.some((d) => d.runner_id === r.id))
              .map((r) => html`<option value="${r.id}">${r.name}${
                r.role === "serving" ? " · reserved for serving" : ""}${
                r.connected ? "" : " · offline"}</option>`).join("")
          : `<option value="">No machine here can serve a model</option>`)}</select>
      </div>
      <div class="row" style="justify-content:flex-end;gap:8px;margin-top:12px">
        <button type="button" class="btn" data-modal-close>Cancel</button>
        <button type="button" class="btn btn-primary" id="dpGo">Deploy it</button>
      </div>` });

    on(dlg, "click", "#dpGo", async (_e, btn) => {
      const runnerId = $("#dpTo", dlg).value;
      if (!runnerId) return toast("Choose a machine.", "err");
      btn.disabled = true;
      try {
        await api.deploy({ job_id: job.id, runner_id: runnerId });
        dlg.close();
        toast("Loading it onto the card. This takes a minute or two.", "ok",
              { href: "#/ops", label: "Operations" });
      } catch (e) { toast(e.message, "err"); btn.disabled = false; }
    });
  });

  on(mount, "click", "#serveAs", async () => {
    const job = getJob();
    let names = [];
    try { names = await api.registeredModels(); }
    catch (e) { return toast(e.message, "err"); }
    const here = names.filter((n) => n.job_id === job.id);
    const dlg = modal({ title: `Serve "${job.name}" under a name`, width: 520,
      body: html`
      <p class="muted tiny">A name other software is configured with. It
        survives a rename, and moving it to a better model later is one change
        rather than one in every client.</p>
      ${raw(here.length ? html`
        <div class="callout callout-ok">
          <strong>Already served as</strong>
          ${raw(here.map((n) => `<code>${esc(n.alias)}</code>`).join(", "))}
        </div>` : "")}
      <div class="field">
        <label for="serveName">Name</label>
        <input id="serveName" type="text" class="mono" placeholder="assistant-prod"
               value="${esc(suggestedAlias(job.name, names))}">
        <div class="hint">Lowercase letters, digits, dot, dash or underscore.
          Using a name that already exists points it here instead.</div>
      </div>
      <div class="row" style="justify-content:flex-end;gap:8px;margin-top:12px">
        <button type="button" class="btn" data-modal-close>Cancel</button>
        <a class="btn" href="#/serving">All names</a>
        <button type="button" class="btn btn-primary" id="serveGo">Serve it</button>
      </div>` });
    on(dlg, "click", "#serveGo", async (_e, btn) => {
      const alias = ($("#serveName", dlg).value || "").trim().toLowerCase();
      if (!alias) return toast("Give it a name.", "err");
      btn.disabled = true;
      try {
        const r = await api.registerModel(alias, { job_id: job.id });
        dlg.close();
        toast(r.moved ? `"${alias}" now answers with this run.`
                      : `Serving as "${alias}".`, "ok",
              { href: "#/serving", label: "Served models" });
      } catch (e) { toast(e.message, "err"); btn.disabled = false; }
    });
  });

  // The publish form lives in the Model section, which may not be the tab you
  // are looking at.
  // A conversion, as a job: it reads every tensor and writes a new file, so
  // it wants a progress bar and a stop button like everything else here.
  on(mount, "click", "#exportGguf", () => {
    const job = getJob();
    const dlg = modal({ title: "Export as GGUF", width: 520, body: html`
      <p class="muted tiny">One file, quantised, that loads in Ollama,
        llama.cpp and LM Studio with no Python at all. It runs as a job on
        whichever machine is free — no graphics card needed.</p>
      <div class="field">
        <label for="ggufQ">How much to shrink it</label>
        <select id="ggufQ">
          ${raw(Object.entries(GGUF_TYPES).map(([k, why], i) =>
            `<option value="${k}"${i === 0 ? " selected" : ""}>${k} — ${
              esc(why)}</option>`).join(""))}
        </select>
      </div>
      <div class="row" style="justify-content:flex-end;gap:8px;margin-top:12px">
        <button type="button" class="btn" data-modal-close>Cancel</button>
        <button type="button" class="btn btn-primary" id="ggufGo">Export it</button>
      </div>` });
    on(dlg, "click", "#ggufGo", async (_e, btn) => {
      btn.disabled = true;
      try {
        const { id } = await api.createJob({
          name: `${job.name} → GGUF`,
          kind: "export_gguf",
          config: { source_job: job.id, quantize: $("#ggufQ", dlg).value },
        });
        dlg.close();
        toast("Export queued.", "ok", { href: `#/jobs/${id}`, label: "Watch it" });
      } catch (e) { toast(e.message, "err"); btn.disabled = false; }
    });
  });

  on(mount, "click", "#fileRun", () => {
    fileIntoDialog({ kind: "job", id: jobId, name: job?.name,
      current: job?.project?.id || null,
      onDone: async () => {
        job = await api.job(jobId);
        paintHeader(mount, job, getTab());
      } });
  });
  on(mount, "click", "#publishHere", () => {
    publishDialog(jobId, job?.name, job?.project_id || null, async () => {
      job = await api.job(jobId);
      paintHeader(mount, job, getTab());
    });
  });
  on(mount, "click", "#goPublish", () => {
    const tab = $('[data-tab="model"]', mount);
    if (tab) tab.click();
    requestAnimationFrame(() =>
      $("#ownerRow", mount)?.scrollIntoView({ behavior: "smooth", block: "start" }));
  });

  on(mount, "click", "#deleteBtn", async () => {
    const job = getJob();
    const hasModel = job.artifacts?.length;
    if (!await confirmDestructive({
      title: `Delete "${job.name}"?`,
      consequences: [
        job.kind === "generate_dataset"
          ? "Its log and measurements go. Any dataset it already produced stays."
          : hasModel
            ? "Its trained model file goes with it, and cannot be recovered."
            : "Its logs and measurements go.",
        "Cached copies on the machines that ran it are removed too.",
        "Scores already recorded against a prompt set are kept.",
      ],
      confirmLabel: "Delete the run",
      confirmWord: hasModel ? job.name : null })) return;
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
    ${raw(runHead(job))}
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

    <div class="card" data-sec="log">
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
  // A writing run that is already going is writing; it is never training.
  let stage = job.status === "running" ? "writing" : "";
  wireRunControls(mount, jobId, () => job, () => latest, () => stage);

  const getTab = wireRunTabs(mount, job, () => paint());
  const paint = () => {
    paintHeader(mount, job, getTab());
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

// ---------------------------------------------------------------------------
// Sending something to Hugging Face
// ---------------------------------------------------------------------------
//
// The plainest run there is: no loss, no rows, no model at the end. One
// destination, a list of files, and a bar that moves. It gets its own layout
// for the same reason a generation run does -- given the training one it would
// draw two empty charts and a paragraph about learning rates.

function uploadLayout(job) {
  const cfg = job.config || {};
  const dataset = cfg.target === "dataset";
  // An export shares this page shape -- a job that produces one file and
  // hands it over -- and shares none of its facts.
  if (job.kind === "export_gguf") return exportLayout(job);
  return html`
    ${raw(runHead(job))}
    <div id="queueCard"></div>
    <div id="stopPanel"></div>
    <div id="progressCard"></div>

    <div class="card" style="margin-bottom:14px">
      <h3>Where it is going</h3>
      <dl class="kv">
        <dt>Repository</dt>
        <dd class="mono"><a href="https://huggingface.co/${
          dataset ? "datasets/" : ""}${cfg.repo_id}"
          target="_blank" rel="noopener">${cfg.repo_id}</a></dd>
        <dt>Visibility</dt>
        <dd>${cfg.private === false ? "Public — anyone can download it"
                                    : "Private"}
          <span class="muted tiny">— only applied if the repository is
            new</span></dd>
        <dt>What</dt>
        <dd>${dataset ? cfg.dataset_label || "a dataset from this studio"
                      : cfg.source_run_name || cfg.source_job
                        || "a run in this studio"}</dd>
        <dt>Already there</dt>
        <dd>${cfg.replace
          ? "Removed in the same commit as this upload"
          : "Left alone; files with the same name are overwritten"}</dd>
      </dl>
      <div id="uploadResult" style="margin-top:10px"></div>
    </div>

    <div class="card" data-sec="log">
      <div class="row-between" style="align-items:center">
        <h3 style="margin:0">Log</h3>
        <span class="tiny muted">file by file</span>
      </div>
      <div class="logbox" id="logBox"></div>
    </div>`;
}

function uploadView(mount, job, jobId, logs) {
  const logBox = $("#logBox", mount);
  logs.forEach((l) => appendLog(logBox, l));
  logBox.scrollTop = logBox.scrollHeight;

  let stage = job.status === "running" ? "uploading" : "";
  wireRunControls(mount, jobId, () => job, () => ({}), () => stage);

  const getTab = wireRunTabs(mount, job, () => paint());
  const paint = () => {
    paintHeader(mount, job, getTab());
    paintProgress(mount, job, stage, job.step, job.total_steps);
    const q = $("#queueCard", mount);
    if (q) q.innerHTML = queueCard(job);
    $("#uploadResult", mount).innerHTML = uploadResult(job);
  };
  paint();

  const unsub = events.subscribe(async (msg) => {
    if (msg.job_id && msg.job_id !== jobId) return;
    if (msg.type === "job_log") {
      const atBottom = logBox.scrollHeight - logBox.scrollTop
        - logBox.clientHeight < 40;
      appendLog(logBox, { ts: Date.now() / 1000, level: msg.level, line: msg.line });
      if (atBottom) logBox.scrollTop = logBox.scrollHeight;
    } else if (msg.type === "job_progress") {
      stage = msg.stage;
      paintProgress(mount, job, stage, msg.step, msg.total);
    } else if (msg.type === "jobs_changed") {
      job = await api.job(jobId);
      paint();
    }
  });
  return () => unsub();
}

function uploadResult(job) {
  const s = job.summary || {};
  if (job.status !== "succeeded" || !s.url) return "";
  return html`
    <div class="callout callout-ok">
      <strong>Published</strong>
      <a href="${s.url}" target="_blank" rel="noopener">${s.url}</a>
      <div class="tiny muted" style="margin-top:4px">
        ${fmtNum(s.files || 0)} file${s.files === 1 ? "" : "s"}, ${
          fmtNum(Math.round((s.bytes || 0) / 1048576))} MB${
          s.replaced ? `, ${fmtNum(s.replaced)} older file${
            s.replaced === 1 ? "" : "s"} removed` : ""}${
          s.created_repo ? ", into a repository this run created" : ""}.</div>
    </div>`;
}

/** Stopping an upload, which leaves nothing behind at all.
 *
 *  Files are sent first and committed once at the end, so a stop before the
 *  commit changes nothing on the Hub. Saying so is the whole point of the
 *  panel: the fear is that half a model is now sitting in a public repository.
 */
function uploadStopPanel() {
  return html`
    <div class="card callout-warn" style="margin-bottom:14px">
      <h3 style="margin:0 0 6px">Stop the upload?</h3>
      <p class="muted tiny" style="margin:0 0 12px">
        Nothing is committed until every file has been sent, so stopping now
        leaves the repository exactly as it is. What has already gone over the
        wire is unreferenced and Hugging Face clears it up.</p>
      <div class="row" style="gap:8px">
        <button class="btn-danger btn-sm" data-stop="discard">Stop the upload</button>
        <button class="btn-sm" id="stopCancel">Carry on uploading</button>
      </div>
    </div>`;
}

// ---------------------------------------------------------------------------
// A merge
//
// No steps, no loss, no learning rate: it loads an adapter and the weights it
// was trained against, adds one to the other, and writes the result. What
// somebody watching it wants to know is what is being folded into what, how
// big the answer will be, and -- when it lands -- how to get at it, because
// this is the artifact that leaves the studio.
// ---------------------------------------------------------------------------

function mergeLayout(job) {
  const cfg = job.config || {};
  const into = cfg.base_model_label || cfg.base_model || "its base model";
  return html`
    ${raw(runHead(job))}
    <div id="queueCard"></div>
    <div id="stopPanel"></div>
    <div id="progressCard"></div>

    <div class="card" style="margin-bottom:14px">
      <h3>What this makes</h3>
      <p class="muted tiny" style="margin:0 0 10px">
        A LoRA fine-tune produces an <em>adapter</em>: a few megabytes that
        mean nothing without the exact weights they were trained against.
        Folding it in writes a complete model that loads on its own — which is
        what Ollama, llama.cpp and everything else outside this studio want.
        <strong>It is the size of the base model</strong>, not of the adapter,
        and the adapter is kept as well.</p>
      <dl class="kv">
        <dt>From</dt>
        <dd>${raw(cfg.source_job
          ? `<a href="#/jobs/${esc(cfg.source_job)}">${
               esc(cfg.source_run_name || "the fine-tune")}</a>`
          : esc(cfg.source_run_name || "a fine-tune"))}</dd>
        <dt>Into</dt>
        <dd class="mono">${into}</dd>
        <dt>Written in</dt>
        <dd>${cfg.dtype || "float16"}
          <span class="muted tiny">— the arithmetic is done in full precision
            either way</span></dd>
      </dl>
      <div id="mergeResult" style="margin-top:10px"></div>
    </div>

    ${raw(job.summary?.artifact_removed ? html`
      <div class="callout callout-warn" data-sec="home model" style="margin-bottom:14px">
        <strong>The model is no longer here</strong>
        Removed ${fmtAgo(job.summary.artifact_removed.at)} — ${job.summary.artifact_removed.why}.
        The run, its chart, its log and its settings stay; "Run again" trains
        it afresh.</div>` : "")}
    ${raw(job.kind === "finetune_vision_cls" ? html`
      <div class="card" id="tryCard" data-sec="home model" style="margin-bottom:14px">
        <h3 style="margin:0 0 6px">Try it on a picture</h3>
        <p class="muted tiny">Drop a picture here, or choose one. It is sent to a
          machine that has the model and comes back with every category and how
          sure it was.</p>
        <div class="row" style="gap:8px;align-items:center;flex-wrap:wrap">
          <input type="file" id="tryFile" accept="image/*">
          <span class="tiny muted" id="tryStatus"></span>
        </div>
        <div id="tryResult" style="margin-top:10px"></div>
      </div>` : "")}
    <div id="cardRow" data-sec="model"></div>

    <div class="grid grid-2" data-sec="model" style="margin-bottom:14px" id="ownerRow"></div>

    <div class="card" data-sec="log">
      <div class="row-between" style="margin-bottom:8px">
        <h3 style="margin:0">Log</h3>
        <span class="tiny muted">Newest at the bottom</span>
      </div>
      <div class="logbox" id="logBox"></div>
    </div>`;
}

function mergeView(mount, job, jobId, logs) {
  const logBox = $("#logBox", mount);
  logs.forEach((l) => appendLog(logBox, l));
  logBox.scrollTop = logBox.scrollHeight;

  let stage = job.status === "running" ? "loading_model" : "";
  let step = job.step, total = job.total_steps;

  const paintCard = wireModelCard(mount, jobId, () => job);
  const paintOwnerRow = wireOwnerRow(mount, jobId, () => job, async () => {
    job = await api.job(jobId);
    paint();
  });

  const getTab = wireRunTabs(mount, job, () => paint());
  const paint = () => {
    paintHeader(mount, job, getTab());
    paintProgress(mount, job, stage, step, total);
    $("#queueCard", mount).innerHTML = queueCard(job);
    $("#mergeResult", mount).innerHTML = mergeResult(job);
    paintOwnerRow();
    paintCard();
  };
  paint();
  wireRunControls(mount, jobId, () => job, () => ({}), () => stage);

  const unsub = events.subscribe(async (msg) => {
    if (msg.job_id && msg.job_id !== jobId) return;
    if (msg.type === "job_log") {
      const atBottom = logBox.scrollHeight - logBox.scrollTop
        - logBox.clientHeight < 40;
      appendLog(logBox, { ts: Date.now() / 1000, level: msg.level, line: msg.line });
      if (atBottom) logBox.scrollTop = logBox.scrollHeight;
    } else if (msg.type === "job_progress") {
      stage = msg.stage;
      step = msg.step;
      total = msg.total;
      paintProgress(mount, job, stage, step, total);
    } else if (msg.type === "jobs_changed") {
      job = await api.job(jobId);
      paint();
    }
  });
  return () => unsub();
}

/** Stopping a merge, which leaves the adapter exactly where it was.
 *
 *  There is nothing to keep half of: a model is written in one pass at the
 *  end, and until then what exists is the adapter that already existed. Saying
 *  so is the point -- the fear is that stopping has damaged the run this was
 *  folding.
 */
function mergeStopPanel() {
  return html`
    <div class="card callout-warn" style="margin-bottom:14px">
      <h3 style="margin:0 0 6px">Stop the merge?</h3>
      <p class="muted tiny" style="margin:0 0 12px">
        Nothing is kept: a merged model is written in one pass at the end, so
        stopping leaves no half-model behind. The fine-tune and its adapter are
        untouched, and the merge can be started again from that run's page.</p>
      <div class="row" style="gap:8px">
        <button class="btn-danger btn-sm" data-stop="discard">Stop the merge</button>
        <button class="btn-sm" id="stopCancel">Carry on merging</button>
      </div>
    </div>`;
}

/** Stopping a scoring run. The models are read, never written. */
function evalStopPanel() {
  return html`
    <div class="card callout-warn" style="margin-bottom:14px">
      <h3 style="margin:0 0 6px">Stop scoring?</h3>
      <p class="muted tiny" style="margin:0 0 12px">
        The scores are recorded together when every model has answered every
        prompt, so stopping now records none of them. Nothing is changed about
        the models themselves — they are only being read.</p>
      <div class="row" style="gap:8px">
        <button class="btn-danger btn-sm" data-stop="discard">Stop scoring</button>
        <button class="btn-sm" id="stopCancel">Carry on</button>
      </div>
    </div>`;
}

function mergeResult(job) {
  const s = job.summary || {};
  if (!job.artifacts?.length) return "";
  const size = s.artifact_size
    ? `${(s.artifact_size / 1073741824).toFixed(1)} GB` : null;
  return html`
    <div class="callout callout-ok">
      <strong>A complete model.</strong>
      ${s.params_total ? `${fmtNum(s.params_total)} parameters` : ""}${
        size ? `, ${size} on disk` : ""}${
        s.duration_s ? `, merged in ${fmtDuration(s.duration_s)}` : ""}.
      It loads with <code>from_pretrained</code> and needs nothing downloaded
      alongside it.
    </div>`;
}

// ---------------------------------------------------------------------------
// A scoring run
//
// Its result is a comparison, and the comparison has a page of its own that
// keeps every scoring of the same prompt set side by side. This page is what
// happens while it runs, and the way to that one afterwards.
// ---------------------------------------------------------------------------

function evalLayout(job) {
  const cfg = job.config || {};
  const models = (cfg.models || []).length;
  return html`
    ${raw(runHead(job))}
    <div id="queueCard"></div>
    <div id="stopPanel"></div>
    <div id="progressCard"></div>
    <div id="evalResult"></div>

    <div class="card" data-sec="log">
      <div class="row-between" style="margin-bottom:8px">
        <h3 style="margin:0">Log</h3>
        <span class="tiny muted">One line per model, as each finishes</span>
      </div>
      <div class="logbox" id="logBox"></div>
    </div>`;
}

function evalView(mount, job, jobId, logs) {
  const logBox = $("#logBox", mount);
  logs.forEach((l) => appendLog(logBox, l));
  logBox.scrollTop = logBox.scrollHeight;

  let stage = job.status === "running" ? "evaluating" : "";
  let step = job.step, total = job.total_steps;

  const getTab = wireRunTabs(mount, job, () => paint());
  const paint = () => {
    paintHeader(mount, job, getTab());
    paintProgress(mount, job, stage, step, total);
    $("#queueCard", mount).innerHTML = queueCard(job);
    $("#evalResult", mount).innerHTML = evalResult(job);
  };
  paint();
  wireRunControls(mount, jobId, () => job, () => ({}), () => stage);

  const unsub = events.subscribe(async (msg) => {
    if (msg.job_id && msg.job_id !== jobId) return;
    if (msg.type === "job_log") {
      const atBottom = logBox.scrollHeight - logBox.scrollTop
        - logBox.clientHeight < 40;
      appendLog(logBox, { ts: Date.now() / 1000, level: msg.level, line: msg.line });
      if (atBottom) logBox.scrollTop = logBox.scrollHeight;
    } else if (msg.type === "job_progress") {
      stage = msg.stage;
      step = msg.step;
      total = msg.total;
      paintProgress(mount, job, stage, step, total);
    } else if (msg.type === "jobs_changed") {
      job = await api.job(jobId);
      paint();
    }
  });
  return () => unsub();
}

function evalResult(job) {
  const s = job.summary || {};
  const scores = s.scores || [];
  if (!scores.length) return "";
  // Loss on the expected answer is the column that decides it, so it leads.
  // The rest are shown because a model can be right and score badly on one of
  // them -- "contains" in particular is generous and "exact" is merciless.
  return html`
    <div class="card" style="margin-bottom:14px">
      <div class="row-between" style="align-items:center;margin-bottom:8px">
        <h3 style="margin:0">How they did</h3>
        ${raw(s.eval_id
          ? `<a class="btn btn-sm" href="#/evals/${esc(s.eval_id)}">
               Every scoring of this set →</a>` : "")}
      </div>
      ${raw(s.verdict ? html`
        <div class="callout ${s.decisive ? "callout-ok" : "callout-warn"}"
             style="margin-bottom:10px">${s.verdict}</div>` : "")}
      <div class="table-wrap"><table>
        <thead><tr><th>Model</th><th>Loss</th>
          <th class="hide-sm">Token overlap</th><th class="hide-sm">Exact</th>
        </tr></thead>
        <tbody>${raw(scores.map((sc) => {
          const m = sc.metrics || {};
          return html`
            <tr>
              <td>${raw(sc.model_job_id
                ? `<a href="#/jobs/${esc(sc.model_job_id)}">${esc(sc.name)}</a>`
                : esc(sc.name))}${raw(m.error
                ? ` <span class="badge badge-err">could not be scored</span>` : "")}</td>
              <td class="mono">${m.expected_loss != null
                ? m.expected_loss.toFixed(4) : "—"}</td>
              <td class="mono hide-sm">${m.f1 != null
                ? (m.f1 * 100).toFixed(0) + "%" : "—"}</td>
              <td class="mono hide-sm">${m.exact != null
                ? (m.exact * 100).toFixed(0) + "%" : "—"}</td>
            </tr>`;
        }).join(""))}</tbody>
      </table></div>
      <p class="muted tiny" style="margin:8px 0 0">
        Only meaningful where the models were not trained on these prompts —
        a set built from a held-out split is. Loss is the one to compare
        between runs; the others reward answers that happen to be worded like
        the expected one.</p>
    </div>`;
}

/** A model on its way to being one file. */
function exportLayout(job) {
  const cfg = job.config || {};
  const s = job.summary || {};
  const done = job.status === "succeeded" && job.artifacts?.length;
  return html`
    ${raw(runHead(job))}
    <div id="queueCard"></div>
    <div id="stopPanel"></div>
    <div id="progressCard"></div>

    <div class="card" style="margin-bottom:14px">
      <h3 style="margin:0 0 8px">One file, for everything outside this studio</h3>
      <dl class="kv">
        <dt>From</dt>
        <dd>${raw(cfg.source_job
          ? `<a href="#/jobs/${esc(cfg.source_job)}">${
               esc(cfg.source_run_name || "the run")}</a>`
          : esc(cfg.source_run_name || "a run"))}</dd>
        <dt>Quantised to</dt><dd class="mono">${cfg.quantize || "Q4_K_M"}</dd>
        ${raw(s.filename ? html`
          <dt>File</dt><dd class="mono">${s.filename}${
            s.bytes ? ` · ${fmtBytes(s.bytes)}` : ""}</dd>` : "")}
      </dl>
      ${raw(done ? html`
        <div class="row" style="gap:8px;margin-top:12px;flex-wrap:wrap">
          <a class="btn btn-primary" href="/api/jobs/${esc(job.id)}/download?kind=gguf"
            >↓ Download it</a>
        </div>
        <p class="muted tiny" style="margin:10px 0 0">The zip holds the model,
          a <code>Modelfile</code> for Ollama and a note saying what to do with
          both — because a file downloaded in six weeks will not have this page
          open beside it.</p>
        <pre class="mono tiny" style="white-space:pre-wrap;background:var(--surface-2);
             padding:10px;border-radius:8px;margin-top:10px">ollama create ${
          esc((s.filename || "model").split(".")[0])} -f Modelfile
ollama run ${esc((s.filename || "model").split(".")[0])}</pre>`
        : html`<p class="muted tiny" style="margin:10px 0 0">Reading every
          tensor and writing a new file. No graphics card is involved, so this
          runs on whichever machine is free.</p>`)}
    </div>

    <div class="card" data-sec="log">
      <div class="row-between" style="align-items:center">
        <h3 style="margin:0">Log</h3>
        <span class="tiny muted">from the converter itself</span>
      </div>
      <div class="logbox" id="logBox"></div>
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
    ${raw(runHead(job))}
    <div id="reportCard"></div>
    <div id="resumeCard"></div>
    <div id="queueCard"></div>
    <div id="stopPanel"></div>
    <div id="progressCard"></div>
    <div class="grid grid-3" id="statCards" style="margin-bottom:16px"></div>

    <div class="card" data-sec="home charts" style="margin-bottom:14px"><div id="lossChart"></div></div>

    ${raw(experts > 1 ? html`
      <div class="card" data-sec="charts" style="margin-bottom:14px">
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
      <div class="card" id="samplesCard" data-sec="home charts" style="margin-bottom:14px"></div>` : "")}

    <div class="grid grid-2" data-sec="charts" style="margin-bottom:14px">
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

    <div id="provenanceRow" data-sec="model"></div>

    <div id="furtherRow" data-sec="model"></div>

    ${raw(job.summary?.artifact_removed ? html`
      <div class="callout callout-warn" data-sec="home model" style="margin-bottom:14px">
        <strong>The model is no longer here</strong>
        Removed ${fmtAgo(job.summary.artifact_removed.at)} — ${job.summary.artifact_removed.why}.
        The run, its chart, its log and its settings stay; "Run again" trains
        it afresh.</div>` : "")}
    ${raw(job.kind === "finetune_vision_cls" ? html`
      <div class="card" id="tryCard" data-sec="home model" style="margin-bottom:14px">
        <h3 style="margin:0 0 6px">Try it on a picture</h3>
        <p class="muted tiny">Drop a picture here, or choose one. It is sent to a
          machine that has the model and comes back with every category and how
          sure it was.</p>
        <div class="row" style="gap:8px;align-items:center;flex-wrap:wrap">
          <input type="file" id="tryFile" accept="image/*">
          <span class="tiny muted" id="tryStatus"></span>
        </div>
        <div id="tryResult" style="margin-top:10px"></div>
      </div>` : "")}
    <div id="cardRow" data-sec="model"></div>

    <div class="grid grid-2" data-sec="model" style="margin-bottom:14px" id="ownerRow"></div>

    <div class="card" data-sec="log">
      <div class="row-between" style="margin-bottom:8px">
        <h3 style="margin:0">Log</h3>
        <span class="tiny muted">Newest at the bottom</span>
      </div>
      <div class="logbox" id="logBox"></div>
    </div>`;
}

// The quantisations worth offering, and what each is for. Mirrors the list
// the runner accepts; the runner is the authority and refuses anything else.
const GGUF_TYPES = {
  Q4_K_M: "about a quarter of the size, and what nearly everybody means",
  Q5_K_M: "a little bigger, a little better",
  Q6_K: "close to the original, at about half the size",
  Q8_0: "barely distinguishable, at half the size",
  F16: "no quantisation at all",
};

const REPORT_ICON = { ok: "✓", warn: "!", error: "✕" };

/** What the run says about itself. Findings first, numbers second: the
 *  numbers are only interesting once you know which of them to look at. */
function reportCard(r, jobId) {
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
            ${raw(f.change ? html`
              <p style="margin:8px 0 0">
                <a class="btn btn-sm btn-primary"
                   href="#/jobs/${jobId}/again?set=${
                     encodeURIComponent(JSON.stringify(f.change))}"
                >Fix it and run again</a>
                <span class="muted tiny" style="margin-left:8px">${
                  Object.entries(f.change).map(([k, v]) =>
                    `${k.replace(/_/g, " ")} → ${v}`).join(" · ")}</span>
              </p>` : "")}
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
  // Not inherited either: the follow-on is its own run, not a sweep member
  // and not on the Hub.
  delete cfg.sweep_id;
  delete cfg.sweep_name;
  delete cfg.sweep_values;
  delete cfg.published;

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
/** The model card: the README this model goes to Hugging Face with.
 *
 *  Shown as the file rather than as a rendered preview, because the file is
 *  the thing being edited and the front matter -- the half that decides how
 *  the Hub files the model -- disappears from any rendering of it. It is also
 *  the half most worth checking before publishing.
 *
 *  Editing is one-way on purpose. The generator rewrites this card whenever
 *  the run is evaluated or published, which is right up until somebody has
 *  written something the generator could not know; from then on it is theirs
 *  and "reset" is the only way back.
 */
/** What this run left behind, when it left more than one thing.
 *
 *  This used to be a form that queued a merge as a second run. Merging is the
 *  last step of the training run now -- it happens on the machine that still
 *  has the weights in memory, instead of downloading the base again onto
 *  whichever box was free -- so by the time this page is drawn the standalone
 *  model either exists or could not be made, and there is nothing left to ask.
 *  What remains worth saying is which of the two to reach for.
 */
function artifactsCard(job) {
  if (job.kind !== "finetune_llm" || !job.artifacts?.length) return "";
  const kinds = new Set(job.artifacts.map((a) => a.kind));
  if (!kinds.has("model") || !kinds.has("adapter")) return "";
  return html`
    <details class="card" style="margin-bottom:14px">
      <summary><strong>This run kept two things</strong>
        <span class="muted tiny"> — the adapter, and a standalone model</span>
      </summary>
      <p class="muted tiny" style="margin:10px 0 0">
        The <em>adapter</em> is a few megabytes and means nothing without
        <code>${job.config.base_model || "its base model"}</code>. The
        <em>merged model</em> is that adapter folded into the base: the size of
        the base, and it loads on its own — which is what you need to run it in
        Ollama, llama.cpp, or anywhere outside this studio.</p>
      <p class="muted tiny" style="margin:6px 0 0">
        Inside the studio the adapter is still the better thing to use, and
        it is what a further fine-tune of this run carries on training.
        Publishing lets you send either, or both.</p>
    </details>`;
}

// ---------------------------------------------------------------------------
// The panels every run that produces a model gets
//
// Publishing, sharing and the model card belong to a *model*, not to training.
// They lived inside the training view, so a merge -- the run whose whole
// purpose is to produce the standalone model people publish -- was the one
// page that offered none of them.
// ---------------------------------------------------------------------------

/** Where this run has already been published.
 *
 *  Recorded when the upload finishes, not when it is queued, so what is listed
 *  here is what is actually on the Hub. It is also what a fine-tune *of* this
 *  run will name as its `base_model:`, which is the reason for keeping it at
 *  all -- and the reason for showing it: somebody who cannot see it has no way
 *  of knowing whether the lineage will come out right.
 */
function publishedCard(job) {
  const rows = job.config?.published || [];
  const local = (job.published_as || []).filter((p) => p.location === "local");
  if (!rows.length && !local.length) return "";
  return html`
    ${raw(local.length ? html`
      <div class="card" style="margin-bottom:14px">
        <h3>In the model library</h3>
        <ul class="tiny" style="margin:8px 0 0;padding-left:18px">
          ${raw(local.map((p) => html`
            <li><a href="#/models"><strong>${p.name}</strong></a>
              ${p.version || ""}</li>`).join(""))}
        </ul>
      </div>` : "")}
    ${raw(!rows.length ? "" : html`
    <div class="card" style="margin-bottom:14px">
      <h3>On Hugging Face</h3>
      <ul class="tiny" style="margin:8px 0 0;padding-left:18px">
        ${raw(rows.map((p) => html`
          <li><a href="${p.url || `https://huggingface.co/${p.repo_id}`}"
                 target="_blank" rel="noopener" class="mono">${p.repo_id}</a>
            <span class="muted"> — ${
              p.artifact_kind === "adapter" ? "the adapter" : "the model"}${
              p.at ? `, ${fmtAgo(p.at)}` : ""}</span></li>`).join(""))}
      </ul>
    </div>`)}`;
}

/** Publish and share, painted into `#ownerRow`. Returns the repaint. */
function wireOwnerRow(mount, jobId, getJob, onChange) {
  const paint = () => {
    const box = $("#ownerRow", mount);
    if (!box) return;
    const job = getJob();
    // What this run actually kept, read off its artifact rows rather than
    // guessed from its kind: a fine-tune normally holds both the merged model
    // and the adapter, but a machine with too little memory to merge keeps
    // only the adapter, and offering a choice it cannot honour is worse than
    // offering none.
    const targets = [...new Set((job.artifacts || []).map((a) => a.kind))];
    box.innerHTML = publishedCard(job) + (job.artifacts?.length ? publishCard({
      kind: "model", slug: repoSlug(job), targets,
      blurb: targets.includes("model") && targets.includes("adapter")
        ? `Uploads a model card and whichever of this run's two artifacts you
           choose — the merged model, which loads anywhere on its own, or the
           adapter, which is small and needs its base — to your own account.`
        : targets.includes("adapter")
          ? `Uploads the adapter and a model card to your own account. It names
             the base model it was fitted to, so the Hub shows it as a
             fine-tune of it.`
          : `Uploads the model and a model card to your own account.`,
    }) : "");
    wireShareBox(mount, "job", job, async () => { await onChange(); paint(); });
  };
  wirePublish(mount, "model", (body) => api.publishJob(jobId, body),
              () => api.publishInfo(jobId));
  return paint;
}

/** The model card editor, painted into `#cardRow`. Returns the repaint.
 *
 *  The card is fetched rather than derived: generating it needs the run's
 *  evaluation history, which no page here has ever had a reason to load.
 */
/** The card editor, painted into `#cardRow`. Returns the repaint.
 *
 *  A run with nothing to publish has no card to write, so the row stays
 *  empty until there is an artifact.
 */
function wireModelCard(mount, jobId, getJob) {
  let refresh = null;
  return async () => {
    const box = $("#cardRow", mount);
    if (!box) return;
    if (!getJob().artifacts?.length) { box.innerHTML = ""; return; }
    if (!refresh) refresh = mountCardEditor(box, { jobId });
    else await refresh();
  };
}

/** The repository name to suggest for this run's model. */
function repoSlug(job) {
  const base = (job.kind === "pretrain_llm"
    ? job.name : (job.config.base_model || "model").split("/").pop() + "-tuned");
  return base.toLowerCase().replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "").slice(0, 60) || "my-model";
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
  if (job.kind === "upload") return uploadStopPanel();
  if (job.kind === "merge_adapter") return mergeStopPanel();
  if (job.kind === "evaluate") return evalStopPanel();
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

/**
 * The run's own ribbon.
 *
 * This page had an action row of up to eight buttons, five separate layouts
 * that each re-emitted their own back link, and one very long scroll holding
 * charts, a log, an artifact explainer, a model card editor, a publish form
 * and a "train this further" form. Which of those you wanted depended entirely
 * on whether the run was still going.
 *
 * So: verbs on the ribbon, and the page in sections the tabs switch between.
 * The sections are hidden rather than removed, because every live paint on
 * this page writes into an element by id — a tab that removed the log box
 * would silently stop the log.
 */
const RUN_TABS = (job) => {
  const k = kindOf(job);
  const tabs = [{ key: "home", label: "Overview" }];
  if (k.page === "training") tabs.push({ key: "charts", label: "Charts" });
  tabs.push({ key: "log", label: "Log" });
  if (k.leavesModel) tabs.push({ key: "model", label: "Model" });
  return tabs;
};

function runRibbon(job, tab) {
  const k = kindOf(job);
  const done = ["succeeded", "failed", "cancelled"].includes(job.status);
  const usable = job.artifacts?.length && k.leavesModel
    && ["succeeded", "cancelled"].includes(job.status);
  // "Run again with changes" opens the form the run was started from, and two
  // kinds have no such form: a scoring run comes from the prompt set it
  // scores, and a merge -- which can no longer be created at all, merging
  // being part of training now -- came from the fine-tune it folded.
  const repeatable = job.kind !== "evaluate" && job.kind !== "merge_adapter"
    && job.kind !== "upload";
  const writing = job.kind === "generate_dataset";
  const madeRows = writing && (job.summary?.dataset_id || job.dataset_id);

  const body = group("This run", [
    !done ? rb("cancelBtn", "■", "Stop", { cls: "danger" })
          : rb("deleteBtn", "🗑", "Delete", { cls: "danger" }),
    rb(null, "⟳", "Run again", { disabled: !done || !repeatable,
      href: done && repeatable
        ? (writing ? `#/generate/from/${esc(job.id)}` : `#/jobs/${esc(job.id)}/again`) : "",
      title: "Start a new run from this one's settings" }),
    rb(null, "✎", "Rename", { data: 'data-rename="1"' }),
    // A run made before projects existed has nowhere to go, and everything
    // worth doing with it -- publishing, exporting, running it again -- now
    // needs one. This is where it gets one.
    rb("fileRun", "◇", job.project ? "Move to another project"
                                   : "File into a project",
      { cls: job.project ? "" : "primary",
        title: job.project ? `Currently in ${job.project.name}`
                           : "This run is in no project" }),
  ]) + group("Use it", [
    // A classifier is tried on this page -- drop a picture, read the labels
    // -- rather than in a playground built around a conversation.
    job.kind === "finetune_vision_cls"
      ? rb("tryClassifier", "▷", "Try it out", { cls: "primary", disabled: !usable,
          title: "Drop a picture and see what it says" })
      : rb(null, "▷", "Try it out", { cls: "primary", disabled: !usable,
          href: usable ? `#/play/${esc(job.id)}` : "" }),
    // The question a finished run actually raises -- is this better than what
    // I started with -- had no button anywhere on this page. It does now.
    rb("scoreRun", "◎", "Score it", { disabled: !usable,
      title: "Put a saved set of prompts to it" }),
    rb(null, "⚖", "Compare", { disabled: !usable, href: "#/compare",
      title: "Held-out loss beside other runs" }),
    // A fine-tune writing the next dataset is the loop this studio is built
    // around, and it was reachable only by going to the generator and finding
    // the run in a dropdown.
    rb(null, "✦", "Write data with it", { disabled: !usable, href: "#/generate",
      title: "Have this model write a dataset" }),
    rb(null, "▤", "Open the rows", { disabled: !madeRows,
      cls: "primary", href: madeRows ? `#/data/${esc(madeRows)}` : "" }),
  ]) + group("Share it", [
    // Naming the run something other software can be pointed at. Without it
    // a client is configured with a run id nobody can read or a run name that
    // breaks the next time somebody renames it.
    rb("serveAs", "🏷", "Serve as…", { disabled: !usable,
      title: "Give it a name other software can be pointed at" }),
    // A name says what to call it; a deployment says where it lives. Both
    // belong here, because "I have finished training this and want other
    // things to use it" is one thought and it was previously two pages.
    rb("deployIt", "📌", "Deploy…", { disabled: !usable,
      title: "Hold it on a machine's card so the first message is not slow" }),
    // The usual reason to delete a run is the space its model takes, and
    // deleting the run threw away the record of what was tried with it.
    rb("dropModel", "⌫", "Remove model", { disabled: !(done && job.artifacts?.length) || !job.mine,
      title: "Free the space; keep the run, its chart and its log" }),
    rb(null, "↓", "Download", { disabled: !(done && job.artifacts?.length),
      href: done && job.artifacts?.length ? `/api/jobs/${esc(job.id)}/download` : "" }),
    // Two meanings of "publish", and they are different things. Keeping it
    // here means naming a version other people in this studio can find; the
    // Hub button below sends it out of the building. Both land in the model
    // library, so there is one list of finished models rather than two.
    rb("publishHere", "⬢", (job.published_as || []).length
      ? "Publish another version" : "Add to the library", { disabled: !usable,
      title: "Name and version it, so it can be found without a run id" }),
    rb("goPublish", "☁", "Publish to the Hub", { disabled: !usable,
      title: "Send it to Hugging Face" }),
    // The page has said for a long time that merging produces "what Ollama
    // wants". Ollama wants GGUF, and until now this could not make one.
    rb("exportGguf", "⬓", "Export as GGUF", { disabled: !usable,
      title: "One quantised file for Ollama, llama.cpp and LM Studio" }),
    job.summary?.url
      ? rb(null, "↗", "On the Hub", { href: job.summary.url }) : "",
  ]);

  return ribbon({
    tabs: RUN_TABS(job), active: tab, body,
    right: `${statusBadge(job.status, job.kind).value || ""}
            ${job.status === "cancelled" && job.artifacts?.length
              ? `<span class="badge badge-ok">${writing ? "rows kept" : "model kept"}</span>` : ""}
            ${shareButton("job", job)}`,
  });
}

/** Draw the ribbon and show only the section it selects. */
function paintHeader(mount, job, tab = "home") {
  const box = $("#runRibbon", mount);
  if (box) box.innerHTML = runRibbon(job, tab);
  // Sections are marked in the layouts. Anything unmarked belongs to the
  // overview, which keeps the four smaller layouts working without each of
  // them having to label every card.
  const tabs = RUN_TABS(job).map((t) => t.key);
  $$("[data-sec]", mount).forEach((el) => {
    const wanted = el.dataset.sec.split(" ");
    // A section belonging to a tab this kind does not have is always shown,
    // rather than being unreachable.
    el.hidden = wanted.some((w) => tabs.includes(w)) && !wanted.includes(tab);
  });

  // Assigned, not prepended: this runs on every metric tick, and prepending
  // would stack a fresh copy of the error on each one.
  const err = $("#errorCard", mount);
  if (err) {
    err.innerHTML = job.error
      ? html`<div class="callout callout-err">
          <strong>This run failed</strong>${job.error}</div>`
      : "";
  }
}

/** The head of every run page, whatever kind it is. */
function runHead(job) {
  document.title = `${job.name} · Runs · AI Studio`;
  return html`
    <div class="page-head">
      ${raw(breadcrumb(job.project
        ? [{ href: `#/projects/${job.project.id}`, label: job.project.name },
           { href: "#/jobs", label: "Runs" }]
        : { href: "#/jobs", label: "Runs" }))}
      <div class="row title-row" style="gap:4px;min-width:0;margin-top:6px">
        <h1 style="margin:0" id="runTitle">${job.name}</h1>
        <button class="btn-sm btn-quiet" data-rename="1" title="Rename this run"
          aria-label="Rename this run">&#9998;</button>
      </div>
      <p class="sub mono tiny" style="margin-top:4px">${subjectOf(job)}</p>
      <div class="run-notes">
        <textarea id="runNotes" rows="1" maxlength="2000"
          placeholder="Why this run? — a note to yourself, saved as you leave the box"
          >${job.notes || ""}</textarea>
        <input id="runTags" class="mono tiny" type="text"
          placeholder="tags, comma-separated — baseline, shipped, bad-data"
          value="${(job.tags || []).join(", ")}"
          title="Short labels to find this run by. Notes are for reading; tags are for filtering.">
      </div>
      ${raw(job.config.sweep_id ? html`
        <p class="tiny" style="margin:4px 0 0">One of several variants —
          <a href="#/sweeps/${job.config.sweep_id}">see them side by
          side</a>.</p>` : "")}
    </div>
    <div id="runRibbon"></div>
    <div id="errorCard"></div>`;
}

function paintProgress(mount, job, stage = "", rawStep = null, rawTotal = null,
                       checkpointStep = 0) {
  // A merge reports its stages as one of four, and a run that is not training
  // has no `job.step` to fall back on -- an empty stage there means "has not
  // said yet", not "is training".
  const trains = ["finetune_llm", "pretrain_llm"].includes(job.kind);
  const training = trains && (stage === "training" || stage === "");
  const writing = stage === "writing";
  const uploading = stage === "uploading";
  const scoring = stage === "evaluating";
  // Preparation stages count their own units -- documents scanned, tokens
  // collected -- so the bar follows those while they run, and the counter is
  // only shown once those units are something a person can count: training
  // steps, rows written, megabytes sent, or prompts put to a model. A merge is
  // four coarse stages, and "stage 3 of 4" tells nobody anything, so it gets
  // the bar and no counter.
  const step = training ? job.step : (rawStep ?? 0);
  const total = training ? job.total_steps : (rawTotal ?? 0);
  const pct = total ? Math.min(100, (step / total) * 100) : 0;
  const running = ["running", "assigned"].includes(job.status);
  const stageText = stagesFor(job)[stage] || (running ? "Working…" : "");
  const counted = total > 0 && (training || writing || uploading || scoring);
  const count = writing ? `row ${step} of ${total}`
    : uploading ? `${fmtNum(step)} MB of ${fmtNum(total)}`
    : scoring ? `${fmtNum(step)} of ${fmtNum(total)} answers`
    : `step ${step} of ${total}`;

  $("#progressCard", mount).innerHTML = running ? html`
    <div class="card" style="margin-bottom:16px">
      <div class="row-between" style="margin-bottom:8px">
        <strong class="tiny">${stageText}</strong>
        <span class="tiny muted">${counted ? count : ""}</span>
      </div>
      <div class="progress"><i style="width:${pct}%"></i></div>
      ${raw(checkpointStep ? html`
        <p class="muted tiny" style="margin:8px 0 0">
          Saved at step ${checkpointStep}. If this machine restarts, the run
          carries on from there rather than starting over.</p>` : "")}
    </div>` : "";
}

/** Why this run is not running, machine by machine. */
function waitingDetail(job) {
  const rows = job.waiting_on || [];
  if (!rows.length) return "";
  const able = rows.filter((r) => r.can);
  return html`
    <details class="adv" style="margin-top:8px">
      <summary>${able.length
        ? `${able.length} of ${rows.length} connected machines could run this`
        : `None of the ${rows.length} connected machines can run this`}</summary>
      <table class="table" style="margin-top:8px"><tbody>
        ${raw(rows.map((r) => html`
          <tr>
            <td class="tiny">${r.runner}</td>
            <td class="tiny ${r.can ? "muted" : ""}">${raw(r.can
              ? (r.busy ? `<span class="badge badge-accent">busy</span> ${esc(r.reason)}`
                        : `<span class="badge badge-ok">ready</span>`)
              : `<span class="badge badge-warn">cannot</span> ${esc(r.reason)}`)}</td>
          </tr>`).join(""))}
      </tbody></table>
    </details>`;
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
      ${raw(waitingDetail(job))}
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
      ${raw(r.bytes || r.best_step ? html`
        <p class="muted tiny" style="margin:0 0 10px">${[
          r.bytes ? `${fmtBytes(r.bytes)} on that machine's disk` : null,
          r.at ? `saved ${fmtAgo(r.at)}` : null,
          r.best_step ? `its best point was step ${fmtNum(r.best_step)}${
            r.best_val_loss != null
              ? `, held-out loss ${r.best_val_loss.toFixed(4)}` : ""}` : null,
        ].filter(Boolean).join(" · ")}. Checkpoints older than a fortnight are
        cleared when the machine restarts.</p>` : "")}
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
/** The facts about this run that decide whether it can be repeated. */
function provenanceCard(job) {
  const s = job.summary || {};
  const fp = job.config?.dataset_fingerprint;
  const rows = [
    ["Loss covered", s.trained_on],
    ["Measured on", s.held_out_from],
    ["Seed", s.seed],
    ["Data", fp ? `${fp.name} · ${fmtNum(fp.rows || 0)} rows · ${fp.head_sha1}` : null],
  ].filter(([, v]) => v != null && v !== "");
  const versions = s.versions || {};
  if (!rows.length && !Object.keys(versions).length) return "";
  return html`
    <div class="card" data-sec="model" style="margin-bottom:14px">
      <h3 style="margin:0 0 6px">What it would take to repeat this</h3>
      <p class="muted tiny">The same settings are not the same run if the data
        was edited or the libraries moved underneath. Both change the result
        while every number on this page stays identical.</p>
      <dl class="kv" style="margin-top:8px">
        ${raw(rows.map(([k, v]) => html`<dt>${k}</dt><dd>${v}</dd>`).join(""))}
        ${raw(Object.keys(versions).length ? html`
          <dt>Trained with</dt><dd class="mono tiny">${
            Object.entries(versions).map(([k, v]) => `${k} ${v}`).join(" · ")}</dd>` : "")}
      </dl>
    </div>`;
}

function heldOutCard(job, m, lastEval) {
  // A classifier is judged by how many it names right, not by a loss.
  if (job.kind === "finetune_vision_cls") {
    const acc = m.val_accuracy ?? lastEval?.val_accuracy ?? job.summary?.best_accuracy;
    const at = m.val_accuracy != null ? m.step : lastEval?.step;
    if (acc != null) {
      return ["Accuracy", (acc * 100).toFixed(1) + "%",
              `on ${job.summary?.held_out_from || "the held-out pictures"}${
                at ? ` · step ${fmtNum(at)}` : ""}`];
    }
    return ["Accuracy", "—", "measured after the first pass"];
  }
  // Where the rows came from, when the run said. "On unseen examples" covers
  // both a split somebody deliberately held back and 5% of the training data
  // taken at random, and only the first is a fair test of anything.
  const from = job.summary?.held_out_from;
  const sub = from ? `on ${from}`
    : job.kind === "pretrain_llm" ? "on unseen text" : "on unseen examples";
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
          // Drawn by what the runner said it is. A picture is shown, a clip
          // gets a player, and a table of predictions is a table -- never a
          // string of an asset id in a paragraph.
          let body;
          if (s.kind === "image" && s.asset_id) {
            body = `<img class="sample-img" src="/api/assets/${esc(s.asset_id)}" alt="${esc(s.caption || "sample")}" loading="lazy">`
                 + (s.caption ? `<p class="tiny muted">${esc(s.caption)}</p>` : "");
          } else if (s.kind === "audio" && s.asset_id) {
            body = `<audio controls preload="none" src="/api/assets/${esc(s.asset_id)}" style="width:100%"></audio>`
                 + (s.text ? `<p class="txt">${esc(s.text)}</p>` : "")
                 + (s.caption ? `<p class="tiny muted">${esc(s.caption)}</p>` : "");
          } else if (s.kind === "table") {
            body = `<pre class="mono tiny" style="white-space:pre-wrap;margin:0">${esc(s.text || "")}</pre>`;
          } else {
            const seed = s.prompt || "";
            const text = s.text || "";
            const rest = text.startsWith(seed) ? text.slice(seed.length) : text;
            body = `<p class="txt"><span class="seed">${esc(seed)}</span>${esc(rest)}</p>`;
          }
          return html`
            <div class="sample">
              <div class="hd">
                <span class="badge badge-accent">step ${s.step}</span>
              </div>
              ${raw(body)}
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


/** A name to offer, from the run's own, that is not taken already.
 *
 *  A suggestion rather than a default anybody has to accept: the useful name
 *  is nearly always about what the model is *for* rather than what it was
 *  called while it trained. */
function suggestedAlias(name, taken) {
  const base = String(name || "model").toLowerCase()
    .replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 40)
    || "model";
  const used = new Set((taken || []).map((n) => n.alias));
  if (!used.has(base)) return base;
  for (let i = 2; i < 50; i++) if (!used.has(`${base}-${i}`)) return `${base}-${i}`;
  return base;
}
