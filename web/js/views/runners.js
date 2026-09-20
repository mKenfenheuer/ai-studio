import { api, events } from "../api.js";
import { session } from "../app.js";
import { html, raw, esc, $, on, fmtAgo, fmtNum, fmtBytes, toast, modal } from "../util.js";
import { ribbon, rb, group, rbSeg, wireRibbon, tabState } from "../ribbon.js";
import { pageHead, emptyState, copyButton } from "../components.js";
import { subjectOf, kindOf } from "../kinds.js";

const TABS = [
  { key: "home", label: "Home" },
  { key: "queue", label: "Queue" },
];

// Flash attention is not the question the badge is asking. The memory-efficient
// kernel avoids materialising the scores matrix in exactly the same way, costs
// the same memory and runs on cards flash attention does not, so a machine that
// has it is not warned about. See common/attention.py, which is where the
// controller and the runner answer the same question.
function attentionKernel(caps) {
  const attn = caps?.attention || {};
  if (attn.flash) return { fused: true, label: "flash attention" };
  if (attn.mem_efficient)
    return { fused: true, label: "memory-efficient attention" };
  return { fused: false, label: "no fused attention" };
}

export async function runnersView(mount) {
  let status = null, runners = [], jobs = [];
  let dense = localStorage.getItem("aistudio.runnersDense") === "1";
  const tabs = tabState("runners", TABS, "home");
  let tab = tabs.get();

  const load = async () => {
    [status, runners, jobs] = await Promise.all([
      api.status(), api.runners(), api.jobs().catch(() => [])]);
    draw();
  };

  const draw = () => {
    const admin = session.user?.role === "admin";
    const online = runners.filter((r) => r.status !== "offline");
    mount.innerHTML = html`
      ${raw(pageHead({
        title: "Machines",
        sub: "Each machine contributes its graphics card to the studio. They "
           + "connect outwards to this controller, so they work from home, an "
           + "office, or a server — no port forwarding needed.",
      }))}
      ${raw(ribbonFor({ tab, admin, online, dense }))}
      ${raw(tab === "queue" ? queuePanel(jobs, runners)
        : runners.length
        ? `<div class="grid ${dense ? "" : "grid-2"}">${
             runners.map((r) => card(r, admin, dense)).join("")}</div>`
        : emptyState({
            icon: "🖥️",
            title: "No machines connected yet",
            body: admin
              ? "Connect one from the ribbon above. It appears here within a few seconds."
              : "An administrator connects machines. Once one is here, you can train on it.",
          }))}`;
    wire(admin);
  };

  function wire(admin) {
    wireRibbon(mount, (key) => { tab = key; tabs.set(key); draw(); });
    on(mount, "click", "#openJoin", () => joinDialog(status));
    on(mount, "click", "[data-dense]", (_e, t) => {
      dense = t.dataset.dense === "1";
      localStorage.setItem("aistudio.runnersDense", dense ? "1" : "0");
      draw();
    });
    on(mount, "click", "#recheckAll", async () => {
      for (const r of runners.filter((x) => x.connected)) {
        try { await api.reprobe(r.id); } catch (e) { toast(e.message, "err"); }
      }
      toast("Re-checking hardware on every connected machine…");
    });
    on(mount, "click", "[data-reprobe]", async (_e, t) => {
      try { await api.reprobe(t.dataset.reprobe); toast("Re-checking hardware…"); }
      catch (e) { toast(e.message, "err"); }
    });
    on(mount, "click", "[data-limits]", (_e, t) => {
      const r = runners.find((x) => x.id === t.dataset.limits);
      if (r) limitsDialog(r);
    });
    on(mount, "click", "[data-forget]", async (_e, t) => {
      if (!confirm(`Forget "${t.dataset.name}"?\n\nIt is taken off this list. `
                   + "Finished runs that used it are untouched, and it comes "
                   + "back if that machine ever connects again.")) return;
      try {
        // No redraw here: the controller broadcasts runners_changed, which is
        // what this view already reloads from.
        const r = await api.forgetRunner(t.dataset.forget);
        toast(r.note || "Removed.", "ok");
      } catch (e) { toast(e.message, "err"); }
    });
  }

  await load();
  const unsub = events.subscribe((m) => {
    if (["runners_changed", "jobs_changed"].includes(m.type)) load();
  });
  return unsub;
}

function ribbonFor({ tab, admin, online, dense }) {
  const body = tab === "home"
    ? group("Fleet", [
        admin ? rb("openJoin", "＋", "Connect a machine", { cls: "primary" })
              : rb(null, "＋", "Connect a machine", { disabled: true,
                  title: "Only an administrator has the join token" }),
        admin ? rb("recheckAll", "↻", "Re-check all",
                   { disabled: !online.length,
                     title: "Benchmark every connected machine again" }) : "",
      ]) + group("Then", [
        rb(null, "✦", "Start a run", { cls: online.length ? "primary" : "",
          disabled: !online.length, href: "#/new" }),
        rb(null, "▤", "Datasets", { href: "#/data" }),
        rb(null, "≡", "Runs", { href: "#/jobs" }),
      ]) + group("Layout", [
        rbSeg([{ label: "Cards", on: !dense, data: `data-dense="0"` },
               { label: "Compact", on: dense, data: `data-dense="1"` }]),
      ])
    : group("Waiting", [
        rb(null, "≡", "All runs", { href: "#/jobs" }),
        rb(null, "✦", "New run", { href: "#/new" }),
      ]);
  return ribbon({ tabs: TABS, active: tab, body });
}

/**
 * What is waiting, and what it is waiting for.
 *
 * The queue was visible only from inside a run, one run at a time, and the
 * reason a run could not start collapsed to a single sentence: "no machine
 * that can run this is connected yet" covers a card too small, a missing
 * 4-bit build, a machine restricted to other work, and a fleet that is simply
 * busy. Here at least the fleet is in front of you while you read it.
 */
function queuePanel(jobs, runners) {
  const waiting = jobs.filter((j) => j.status === "queued")
    .sort((a, b) => (a.queue_position || 1e9) - (b.queue_position || 1e9));
  const running = jobs.filter((j) => ["running", "assigned"].includes(j.status));
  const online = runners.filter((r) => r.status !== "offline");

  if (!waiting.length && !running.length) {
    return emptyState({
      icon: "◷",
      title: "Nothing is waiting",
      body: online.length
        ? `${online.length} machine${online.length > 1 ? "s are" : " is"} ready.`
        : "And no machine is connected to do anything anyway.",
      cta: { href: "#/new", label: "Start a run" },
    });
  }
  const line = (j) => html`
    <tr>
      <td><a href="#/jobs/${j.id}"><strong>${j.name}</strong></a>
        <div class="tiny muted mono">${subjectOf(j)}</div></td>
      <td class="tiny">${kindOf(j).label}</td>
      <td class="tiny muted">${j.status === "queued"
        ? (j.queue_position ? `${j.queue_position} of ${j.queue_length}`
                            : "no machine can take it")
        : (runners.find((r) => r.current_job === j.id)?.name || "starting")}</td>
    </tr>`;
  return html`
    <div class="card" style="padding:0">
      <div class="table-wrap"><table>
        <thead><tr><th>Run</th><th>Kind</th><th>Where it is</th></tr></thead>
        <tbody>
          ${raw(running.map(line).join(""))}
          ${raw(waiting.map(line).join(""))}
        </tbody>
      </table></div>
    </div>
    <p class="muted tiny" style="margin-top:10px">Work is dealt round-robin
      between people rather than first-come-first-served, so one person's
      overnight batch cannot block everybody else's twenty-minute job. A run
      that no connected machine can take waits until one that can appears.</p>`;
}

/** Where to go from here. The onboarding sends people to this page to connect
 *  a machine, and used to leave them on it with nowhere to go once it was. */
/**
 * How to attach a machine, as a dialog rather than as the top of the page.
 *
 * It was rendered unconditionally, forever: a studio with six healthy machines
 * still led its Machines page with a twenty-line install snippet.
 */
function joinDialog(status) {
  const url = location.origin;
  const cmd = `ai-studio-runner --controller ${url} --token ${status.join_token}`;
  const docker = `docker run --rm \\
  --device=/dev/kfd --device=/dev/dri --group-add video \\
  -e AI_STUDIO_CONTROLLER=${url} \\
  -e AI_STUDIO_JOIN_TOKEN=${status.join_token} \\
  ghcr.io/ai-studio/runner:rocm`;
  const dockerCuda = `docker run --rm --gpus all \\
  -e AI_STUDIO_CONTROLLER=${url} \\
  -e AI_STUDIO_JOIN_TOKEN=${status.join_token} \\
  ghcr.io/ai-studio/runner:cuda`;

  const block = (label, text) => html`
    <div class="field">
      <label>${label}</label>
      <div class="row row-top">
        <pre class="mono" style="flex:1;margin:0;background:var(--surface-2);
          padding:10px;border-radius:8px;overflow-x:auto">${text}</pre>
        ${raw(copyButton(text))}
      </div>
    </div>`;

  modal({ title: "Connect a machine", width: 700, body: html`
    <p class="muted tiny">Run one of these on the computer with the graphics
      card. Anyone holding this token can join your studio, so treat it like a
      password.</p>
    ${raw(block("AMD graphics card (Docker)", docker))}
    ${raw(block("NVIDIA graphics card (Docker)", dockerCuda))}
    <div class="field">
      <label>No graphics card of your own</label>
      <div class="row row-top">
        <p class="muted tiny" style="flex:1;margin:0">
          Google lends one out. This notebook turns a Colab session into a
          machine here for as long as the session lasts, with nothing to
          install — open Colab, <em>File → Upload notebook</em>, run it. Your
          datasets and models stay on this controller. Colab does have to be
          able to reach this address, which for a studio on your own network
          means a tunnel in front of it; the notebook says how.</p>
        <a class="btn btn-sm" href="/api/runner/colab.ipynb"
           download="ai-studio-runner.ipynb">↓ Colab notebook</a>
      </div>
    </div>
    <details class="adv">
      <summary>Without Docker — also the only option on an Apple Mac</summary>
      <div style="margin-top:8px">${raw(block("Command", cmd))}</div>
      <p class="muted tiny" style="margin-top:8px">Install first with
        <code>scripts/install-runner.sh</code>, which picks the right PyTorch
        build for your hardware.</p>
    </details>` });
}

/** Is any part of this machine being held back? */
function limited(r) {
  const l = r.limits || {};
  return (l.memory_pct != null && l.memory_pct < 100)
      || (l.compute_pct != null && l.compute_pct < 100);
}

/** "12 GB of 16 GB · about 50% of its time" -- only the parts that apply. */
function limitLine(r) {
  const l = r.limits || {};
  const c = r.capabilities || {};
  const bits = [];
  if (l.memory_pct < 100) {
    bits.push(c.vram_gb_total
      ? `${c.vram_gb} GB of ${c.vram_gb_total} GB`
      : `${l.memory_pct}% of its memory`);
  }
  if (l.compute_pct < 100) bits.push(`about ${l.compute_pct}% of its time`);
  return bits.join(" · ");
}

/**
 * How much of one machine the studio may take.
 *
 * Two numbers that look alike and are not. Memory is a real cap -- the runner
 * confines the process to it and the planner sizes the batch to fit inside it
 * -- so the usual effect of setting it is a smaller batch. Compute is a duty
 * cycle: the trainer pauses between steps until the card averages that share,
 * because no consumer card sells a slice of itself. Saying so here is the
 * whole point of the dialog; a pair of unlabelled percentage boxes would
 * promise isolation that does not exist.
 */
function limitsDialog(r) {
  const l = r.limits || { memory_pct: 100, compute_pct: 100 };
  const c = r.capabilities || {};
  const card = c.vram_gb_total || c.vram_gb;
  const own = r.limits?.machine || {};
  const ownSet = own.memory_pct < 100 || own.compute_pct < 100;

  const dlg = modal({ title: `How much of ${r.name} the studio may use`,
    width: 560, body: html`
    <div class="field">
      <label for="limMem">Memory</label>
      <div class="row" style="align-items:center;gap:10px">
        <input id="limMem" type="range" min="10" max="100" step="5"
               value="${l.memory_pct}" style="flex:1">
        <output id="limMemOut" class="mono" style="min-width:8ch"></output>
      </div>
      <p class="hint">A real cap. The run is confined to this share and is
        planned to fit inside it, so setting it usually costs batch size
        rather than causing a failure.${card
          ? ` This card holds ${card} GB.` : ""}</p>
    </div>

    <div class="field">
      <label for="limCompute">Compute</label>
      <div class="row" style="align-items:center;gap:10px">
        <input id="limCompute" type="range" min="5" max="100" step="5"
               value="${l.compute_pct}" style="flex:1">
        <output id="limComputeOut" class="mono" style="min-width:8ch"></output>
      </div>
      <p class="hint">Not a partition — no consumer card offers one. The
        trainer pauses between steps until the card averages this share, so
        the run takes correspondingly longer. Worth it when the machine is
        also somebody's desktop, or the power budget is shared. Chat replies
        are never throttled.</p>
    </div>

    <p class="muted tiny" id="limEffect"></p>

    <div class="row" style="justify-content:space-between;gap:8px;margin-top:12px">
      <button type="button" class="btn btn-sm" id="limClear"
        title="${ownSet ? "Back to what this machine's own configuration says"
                        : "This machine sets no limit of its own, so this means all of it"}"
        >Use the machine's own setting</button>
      <span>
        <button type="button" class="btn" data-modal-close>Cancel</button>
        <button type="button" class="btn btn-primary" id="limSave">Save</button>
      </span>
    </div>` });

  const paint = () => {
    const m = +$("#limMem", dlg).value, k = +$("#limCompute", dlg).value;
    $("#limMemOut", dlg).textContent = card
      ? `${(card * m / 100).toFixed(1)} GB` : `${m}%`;
    $("#limComputeOut", dlg).textContent = `${k}%`;
    $("#limEffect", dlg).textContent = k < 100
      ? `A run that takes an hour at full speed would take about `
        + `${(100 / k).toFixed(1)} hours.`
      : "";
  };
  paint();
  on(dlg, "input", "input[type=range]", paint);

  const save = async (limits, btn) => {
    btn.disabled = true;
    try {
      const res = await api.setRunnerLimits(r.id, limits);
      toast(res.note || "Saved.", "ok");
      dlg.close();
    } catch (ex) { toast(ex.message, "err"); btn.disabled = false; }
  };
  on(dlg, "click", "#limSave", (_e, btn) => save({
    memory_pct: +$("#limMem", dlg).value,
    compute_pct: +$("#limCompute", dlg).value }, btn));
  on(dlg, "click", "#limClear", (_e, btn) => save(null, btn));
}

function card(r, admin, dense = false) {
  const c = r.capabilities || {};
  const offline = r.status === "offline";
  const dot = offline ? "dot-err" : r.status === "busy" ? "dot-busy" : "dot-ok";
  const dtypes = c.dtypes || {};
  const best = c.recommended_dtype;

  // A machine restricted to uploads is never training, whatever it is doing.
  const trains = !c.kinds?.length
    || c.kinds.some((k) => k === "finetune_llm" || k === "pretrain_llm");

  // Compact is for a fleet: once there are eight machines, what you came to
  // find out is which of them are up and what they are doing, and the dtype
  // benchmarks are three screens of scrolling in the way of it.
  if (dense) {
    return html`
      <div class="card" style="padding:12px 14px">
        <div class="row-between" style="flex-wrap:wrap;gap:8px;align-items:center">
          <div style="min-width:0">
            <strong><span class="dot ${dot}"></span> ${r.name}</strong>
            <span class="muted tiny mono" style="margin-left:8px">${
              c.device_name || "unknown device"}${c.vram_gb ? ` · ${c.vram_gb} GB` : ""}${
              c.backend ? ` · ${c.backend.toUpperCase()}` : ""}</span>
          </div>
          <div class="row" style="gap:6px;flex-wrap:wrap;align-items:center">
            ${raw(r.disk?.free_gb != null
              ? `<span class="tiny muted">${Math.round(r.disk.free_gb)} GB free</span>` : "")}
            ${raw(r.current_job
              ? `<a class="btn-sm" href="#/jobs/${esc(r.current_job)}">open the run</a>` : "")}
            <span class="badge ${offline ? "badge-err" : r.status === "busy"
              ? "badge-accent" : "badge-ok"}">${offline ? "offline"
              : r.status === "busy" ? (trains ? "training" : "busy") : "ready"}</span>
          </div>
        </div>
      </div>`;
  }

  return html`
    <div class="card">
      <div class="row-between" style="flex-wrap:wrap;gap:6px">
        <h3 style="margin:0"><span class="dot ${dot}"></span> ${r.name}</h3>
        <div class="row" style="gap:6px">
          ${raw(r.up_to_date === false
            ? `<span class="badge badge-warn" title="${esc(r.version_note || "")}"
                 >out of date</span>` : "")}
          <span class="badge ${offline ? "badge-err" : r.status === "busy"
            ? "badge-accent" : "badge-ok"}">${offline ? "offline"
            : r.status === "busy" ? (trains ? "training" : "busy") : "ready"}</span>
        </div>
      </div>
      ${raw(r.up_to_date === false && !offline ? html`
        <div class="callout callout-warn" style="margin:8px 0 0">
          <strong>This machine is running an older build</strong>
          ${r.version_note} Rebuild and restart its runner image; the deploy
          script does it for a machine this controller can reach.
        </div>` : "")}
      <p class="muted tiny mono" style="margin:6px 0 10px">
        ${c.device_name || "unknown device"}${c.arch ? " · " + c.arch : ""}${
        c.compute_units ? " · " + c.compute_units
          + (c.backend === "mps" ? " GPU cores" : " compute units") : ""}</p>

      <dl class="kv" style="margin-bottom:10px">
        <dt>Memory</dt><dd>${c.vram_gb
          ? c.vram_gb + " GB" + (c.unified_memory ? " shared with the system" : "")
          : "—"}</dd>
        <dt>Backend</dt><dd>${(c.backend || "—").toUpperCase()}${
          c.rocm_version ? " " + String(c.rocm_version).split("-")[0]
          : c.cuda_version ? " " + c.cuda_version : ""}</dd>
        <dt>Biggest model</dt><dd>${c.max_finetune_params_b
          ? "about " + c.max_finetune_params_b + "B parameters" : "—"}</dd>
        ${raw(c.cpu_cores || c.ram_gb ? html`
          <dt>Processor</dt><dd>${c.cpu_cores ? c.cpu_cores + " cores" : ""}${
            c.cpu_cores && c.ram_gb ? " · " : ""}${c.ram_gb ? c.ram_gb + " GB RAM" : ""}</dd>` : "")}
        ${raw(Array.isArray(c.modalities) ? html`
          <dt>Can work with</dt><dd>${c.modalities.length
            ? c.modalities.map((m) => ({ text: "text", vision: "pictures",
                                         audio: "audio", diffusion: "image generation" })[m] || m).join(" · ")
            : "nothing yet — no PyTorch"}
            ${raw(c.libraries && !c.libraries.torchvision && !c.libraries.PIL
              ? `<div class="tiny muted">no image library: a vision run will not be sent here</div>` : "")}
            ${raw(c.libraries && !c.libraries.torchaudio && !c.libraries.soundfile && !c.libraries.librosa
              ? `<div class="tiny muted">no audio library: a speech run will not be sent here</div>` : "")}</dd>` : "")}
        ${raw(limited(r) ? html`
          <dt>Studio may use</dt><dd>${limitLine(r)}
            <span class="muted tiny">— ${r.limits.source === "machine"
              ? "this machine's own setting" : "set here"}</span></dd>` : "")}
        <dt>Last seen</dt><dd>${fmtAgo(r.last_seen)}</dd>
        ${raw(r.current_job ? html`
          <dt>Working on</dt><dd><a href="#/jobs/${r.current_job}">open the run →</a></dd>` : "")}
        ${raw(r.disk?.free_gb != null ? html`
          <dt>Disk</dt><dd>${Math.round(r.disk.free_gb)} GB free of ${
            Math.round(r.disk.total_gb || 0)}${r.disk.models_gb != null
            ? ` · ${Math.round(r.disk.models_gb)} GB of models` : ""}</dd>` : "")}
        ${raw((r.checkpoint_detail || []).length ? html`
          <dt>Checkpoints</dt><dd>${r.checkpoint_detail.length} held ·
            ${fmtBytes(r.checkpoint_detail.reduce((n, c) => n + (c.bytes || 0), 0))}
            <span class="muted tiny">— a run can carry on from these; they are
            cleared after a fortnight</span></dd>` : "")}
        ${raw(c.kinds?.length ? html`
          <dt>Takes</dt><dd>${c.kinds.join(", ").replace(/_/g, " ")}
            <span class="muted tiny">— and nothing else</span></dd>` : "")}
      </dl>

      ${raw(Object.keys(dtypes).length ? html`
        <div class="row" style="gap:6px;flex-wrap:wrap;margin-bottom:8px">
          ${raw(Object.entries(dtypes).map(([k, v]) => v == null ? "" :
            `<span class="badge ${k === best ? "badge-ok" : ""}">${esc(k)}
              ${v} TFLOP/s${k === best ? " ✓" : ""}</span>`).join(""))}
        </div>` : "")}

      <div class="row" style="gap:6px;flex-wrap:wrap;margin-bottom:10px">
        <span class="badge ${c.quantization?.["4bit"] ? "badge-ok" : "badge-warn"}">
          4-bit ${c.quantization?.["4bit"] ? "available" : "unavailable"}</span>
        <span class="badge ${attentionKernel(c).fused ? "badge-ok" : "badge-warn"}">
          ${attentionKernel(c).label}</span>
      </div>

      ${raw((c.warnings || []).map((w) =>
        `<div class="callout callout-warn tiny">${esc(w)}</div>`).join(""))}

      ${raw(!offline && admin ? `<button class="btn-sm" data-reprobe="${esc(r.id)}">
        Re-check hardware</button>` : "")}
      ${raw(admin ? `<button class="btn-sm" data-limits="${esc(r.id)}"
        >Usage limits</button>` : "")}
      ${raw(offline && admin ? `<button class="btn-sm btn-danger"
        data-forget="${esc(r.id)}" data-name="${esc(r.name)}"
        >Forget this machine</button>` : "")}
    </div>`;
}
