/**
 * One prompt set: its questions, and every model that has answered them.
 *
 * The table is the point of the page. Its rows are (model, date) pairs and its
 * columns are the four measures the runner computes — see
 * runner/jobs/evaluate.py for what each is worth. They are shown together
 * rather than reduced to a single score because they disagree, and a page that
 * hid the disagreement would be inventing confidence it does not have.
 *
 * Magnitude is drawn as a bar in one hue, not as a colour per model: the
 * palette here has exactly two validated series colours, and a comparison
 * across eight models is a magnitude question anyway.
 */
import { api, events } from "../api.js";
import { html, raw, esc, $, $$, on, toast, fmtAgo, fmtDuration } from "../util.js";
import { shareButton, wireShareBox } from "./share.js";
import { ribbon, rb, group, wireRibbon, tabState } from "../ribbon.js";
import { breadcrumb, confirmDestructive } from "../components.js";

const TABS = [
  { key: "scores", label: "Scores" },
  { key: "run", label: "Score models" },
  { key: "prompts", label: "The prompts" },
];

export async function evalView(mount, [evalId]) {
  let ev = await api.eval(evalId);
  let candidates = [];
  let openScore = null;
  const tabs = tabState("evalview", TABS, "scores");
  let tab = tabs.get();

  const draw = () => {
    mount.innerHTML = layout(ev, candidates, openScore, tab);
    wire();
  };

  const refresh = async () => { ev = await api.eval(evalId); draw(); };

  function wire() {
    wireShareBox(mount, "eval", ev, refresh);
    wireRibbon(mount, (key) => { tab = key; tabs.set(key); draw(); });

    // A prompt set with no scores yet opens on the tab that does something
    // about that, rather than on an empty table.
    on(mount, "click", "#goScore", () => { tab = "run"; tabs.set(tab); draw(); });

    on(mount, "click", "#deleteEval", async () => {
      if (!await confirmDestructive({
        title: `Delete "${ev.name}"?`,
        consequences: [
          "Every score recorded against it goes too.",
          "Those cannot be recomputed without running the models again.",
        ],
        confirmLabel: "Delete the prompt set" })) return;
      try {
        await api.deleteEval(evalId);
        toast("Deleted.", "ok");
        location.hash = "#/evals";
      } catch (ex) { toast(ex.message, "err"); }
    });

    on(mount, "click", "#runEval", async () => {
      const picked = $$("[data-model-pick]:checked", mount).map((c) => c.value);
      if (!picked.length) return toast("Choose at least one model.", "err");
      const btn = $("#runEval", mount);
      btn.disabled = true;
      btn.textContent = "Queueing…";
      try {
        const { id } = await api.runEval(evalId, {
          model_job_ids: picked,
          max_new_tokens: +$("#evMaxTokens", mount).value || 200,
          temperature: +$("#evTemp", mount).value || 0,
          system_prompt: $("#evSystem", mount).value || "",
        });
        toast("Scoring queued.", "ok");
        location.hash = `#/jobs/${id}`;
      } catch (ex) {
        toast(ex.message, "err");
        btn.disabled = false;
        btn.textContent = "Score these models";
      }
    });

    on(mount, "click", "[data-open-score]", async (_e, t) => {
      const id = t.dataset.openScore;
      if (openScore?.id === id) { openScore = null; return draw(); }
      try {
        openScore = await api.evalScore(evalId, id);
        draw();
      } catch (ex) { toast(ex.message, "err"); }
    });

    on(mount, "click", "[data-del-score]", async (_e, t) => {
      if (!await confirmDestructive({
        title: "Remove this scoring?",
        body: "It leaves the comparison. The run that produced it is untouched.",
        confirmLabel: "Remove it" })) return;
      try { await api.deleteScore(evalId, t.dataset.delScore); await refresh(); }
      catch (ex) { toast(ex.message, "err"); }
    });

    on(mount, "click", "#copyEval", async () => {
      try {
        const copy = await api.copyEval(evalId, {});
        toast("Copied. Edit the copy freely.", "ok");
        location.hash = `#/evals/${copy.id}`;
      } catch (ex) { toast(ex.message, "err"); }
    });
  }

  draw();
  api.playground().then((p) => { candidates = p; draw(); }).catch(() => {});
  const unsub = events.subscribe((m) => { if (m.type === "jobs_changed") refresh(); });
  return () => unsub();
}

// ---------------------------------------------------------------------------

function layout(ev, candidates, openScore, tab) {
  const scores = ev.scores || [];
  const items = ev.items || [];
  const answered = items.filter((i) => i.expected).length;
  document.title = `${ev.name} · Evaluate · AI Studio`;

  const body = tab === "run" ? runPanel(ev, candidates)
    : tab === "prompts" ? promptsPanel(items, answered)
    : html`${raw(scoreTable(scores, answered, items.length))}
           ${raw(openScore ? scoreDetail(openScore) : "")}`;

  return html`
    <div class="page-head">
      ${raw(breadcrumb({ href: "#/evals", label: "Prompt sets" }))}
      <h1 style="margin:6px 0 0">${ev.name}</h1>
      ${raw(ev.notes ? `<p class="sub">${esc(ev.notes)}</p>` : "")}
    </div>
    ${raw(ribbonFor(ev, candidates, tab))}
    ${raw(body)}`;
}

function ribbonFor(ev, candidates, tab) {
  const items = ev.items || [];
  const scored = (ev.scores || []).length;
  const body = group("This set", [
    rb("goScore", "◎", "Score models", { cls: "primary", disabled: !candidates.length,
      title: candidates.length ? "Put these prompts to a model"
                               : "Train a model first" }),
    rb("copyEval", "⧉", "Copy",
      { title: "An editable copy, so scores already taken keep their meaning" }),
    rb("deleteEval", "🗑", "Delete", { cls: "danger", disabled: !ev.mine }),
  ]) + group("Compare", [
    rb(null, "⚖", "Held-out loss", { href: "#/compare" }),
    rb(null, "≡", "Runs", { href: "#/jobs" }),
  ]);
  return ribbon({
    tabs: TABS, active: tab, body,
    right: `<span class="badge">${items.length} prompts</span>
            <span class="badge">${scored} scoring${scored === 1 ? "" : "s"}</span>
            ${shareButton("eval", ev)}`,
  });
}

/** Choosing what to ask, and how. */
function runPanel(ev, candidates) {
  const items = ev.items || [];
  if (!candidates.length) {
    return html`
      <div class="card empty"><div class="big" aria-hidden="true">🌱</div>
        <h3>No finished models to score yet</h3>
        <p class="muted">Train something first — a run that produced a model,
          or one you stopped and kept.</p>
        <p><a class="btn btn-primary" href="#/new">Start a run</a></p></div>`;
  }
  return html`
    <div class="card">
      <p class="muted tiny">Each model is loaded onto a machine in turn and
        asked all ${items.length} prompts. That takes a while, so it runs as a
        queued job you can watch and stop.</p>
      <div class="picklist" style="margin-top:10px">
        ${raw(candidates.map((c) => html`
          <label class="check">
            <input type="checkbox" data-model-pick value="${c.id}">
            <span>${c.name}
              <span class="muted tiny">· ${c.kind === "pretrain_llm"
                ? "built from scratch" : (c.base_model || "fine-tune")}
                ${raw(c.stopped_early ? ' · <span class="badge">stopped early</span>' : "")}
              </span></span>
          </label>`).join(""))}
      </div>
      <details class="adv" style="margin-top:10px">
        <summary>How they are asked</summary>
        <div class="field">
          <label for="evSystem">System prompt <span class="muted tiny">(optional)</span></label>
          <input id="evSystem" type="text"
                 placeholder="Left empty, each model gets only the prompt">
        </div>
        <div class="row" style="gap:10px">
          <div class="field" style="flex:1">
            <label for="evMaxTokens">Longest answer</label>
            <input id="evMaxTokens" type="number" value="200" min="16" max="512">
          </div>
          <div class="field" style="flex:1">
            <label for="evTemp">Temperature</label>
            <input id="evTemp" type="number" value="0" min="0" max="2" step="0.1">
            <div class="hint">Zero means the same answer every time, which is
              what makes two scorings comparable.</div>
          </div>
        </div>
      </details>
      <button class="btn-primary btn-sm" id="runEval" style="margin-top:10px">
        Score these models</button>
    </div>`;
}

function promptsPanel(items, answered) {
  return html`
    <div class="card">
      <p class="muted tiny">${answered} of ${items.length} have an expected
        answer. Prompts without one still get asked, and their answers are kept
        to read — they simply cannot be scored.</p>
      <div class="promptlist">
        ${raw(items.slice(0, 200).map((i) => html`
          <div class="prompt-row">
            <div class="p">${i.prompt}</div>
            ${raw(i.expected
              ? `<div class="e">→ ${esc(i.expected)}</div>`
              : `<div class="e muted">no expected answer</div>`)}
          </div>`).join(""))}
        ${raw(items.length > 200
          ? `<p class="muted tiny">…and ${items.length - 200} more.</p>` : "")}
      </div>
    </div>`;
}

// Below this, a difference between two models is a difference between a
// handful of prompts. The runner says so in words after each scoring; the
// table has to stop drawing a winner's rosette on it.
const ENOUGH_PROMPTS = 10;

function scoreTable(scores, answered, total) {
  if (!scores.length) {
    return html`
      <div class="card empty" style="margin-bottom:14px">
        <div class="big">📊</div>
        <h3>Nothing scored yet</h3>
        <p class="muted">Pick some models below. Once two of them have answered
          these prompts, this is where the comparison appears.</p>
      </div>`;
  }

  const usable = scores.filter((s) => s.metrics.expected_loss != null);
  const best = usable.length ? Math.min(...usable.map((s) => s.metrics.expected_loss)) : null;
  const worst = usable.length ? Math.max(...usable.map((s) => s.metrics.expected_loss)) : null;
  // The bar is a magnitude, so it gets one hue and a shared scale. Anchored at
  // zero would make every model look identical -- the differences that matter
  // between two trained models are small in absolute terms.
  const span = (worst ?? 0) - (best ?? 0) || 1;
  // Two conditions, and both are the runner's own judgement rather than this
  // page's: enough prompts to be worth measuring, and a most-recent scoring
  // whose difference actually survived being measured against the spread
  // between prompts. Marking a winner the log has just called a coin toss
  // would be the table contradicting its own evidence.
  const latest = scores[0]?.metrics || {};
  const decisive = total >= ENOUGH_PROMPTS && latest.ranking_decisive !== false;
  const verdict = latest.verdict;

  return html`
    <div class="card" style="margin-bottom:14px;padding:0">
      <div class="row-between" style="padding:14px 16px 0">
        <h3 style="margin:0">Results</h3>
        <span class="tiny muted">newest first · lower loss is better</span>
      </div>
      <div class="table-wrap"><table>
        <thead><tr>
          <th>Model</th>
          <th title="Teacher-forced loss on the answer you called correct">
            Loss on expected</th>
          <th class="hide-sm" title="Same number as a 1-in-N surprise">Perplexity</th>
          <th class="hide-sm" title="Token overlap with the expected answer">Overlap</th>
          <th class="hide-sm">Exact</th>
          <th class="hide-sm">Speed</th>
          <th>When</th><th></th>
        </tr></thead>
        <tbody>
          ${raw(scores.map((s) => {
            const m = s.metrics || {};
            const isBest = decisive && best !== null && m.expected_loss === best;
            const width = m.expected_loss != null
              ? 12 + 88 * (1 - (m.expected_loss - best) / span) : 0;
            return html`
              <tr class="${isBest ? "row-best" : ""}">
                <td>
                  <a href="#/jobs/${s.model_job_id}">${s.model_name || s.model_job_id}</a>
                  ${raw(isBest ? ` <span class="badge badge-ok">best</span>` : "")}
                  ${raw(!s.model_name
                    ? ` <span class="badge badge-warn">run deleted</span>` : "")}
                </td>
                <td style="min-width:150px">
                  ${raw(m.expected_loss != null ? html`
                    <strong>${m.expected_loss.toFixed(4)}</strong>
                    <div class="meter"><i style="width:${width.toFixed(1)}%"></i></div>`
                    : `<span class="muted">—</span>`)}
                </td>
                <td class="hide-sm">${m.expected_perplexity ?? "—"}</td>
                <td class="hide-sm">${m.f1 != null ? (m.f1 * 100).toFixed(0) + "%" : "—"}</td>
                <td class="hide-sm">${m.exact != null ? (m.exact * 100).toFixed(0) + "%" : "—"}</td>
                <td class="hide-sm tiny muted">${m.tokens_per_sec
                  ? m.tokens_per_sec.toFixed(0) + " tok/s" : "—"}
                  ${raw(m.seconds ? `<div>${esc(fmtDuration(m.seconds))}</div>` : "")}</td>
                <td class="tiny muted">${fmtAgo(s.created_at)}</td>
                <td><div class="row" style="gap:4px">
                  <button class="btn-sm" data-open-score="${s.id}">Answers</button>
                  <button class="btn-sm btn-danger" data-del-score="${s.id}"
                          title="Remove from the comparison">✕</button>
                </div></td>
              </tr>`;
          }).join(""))}
        </tbody>
      </table></div>
      ${raw(verdict ? html`
        <div class="callout ${decisive ? "callout-ok" : "callout-warn"}"
             style="margin:0 16px 12px">
          <strong>${decisive ? "The latest scoring separated them"
                             : "The latest scoring could not separate them"}</strong>
          ${verdict}
          ${raw(!decisive && total < ENOUGH_PROMPTS ? html`
            <div style="margin-top:6px">Around ${ENOUGH_PROMPTS} prompts is
              where a comparison starts to be worth reading, and more is
              better.</div>` : "")}
        </div>` : !decisive && scores.length > 1 ? html`
        <div class="callout callout-warn" style="margin:0 16px 12px">
          <strong>${total} prompt${total === 1 ? "" : "s"} is not enough to
          rank these.</strong> The gap between two trained models is usually
          smaller than the gap between one prompt and the next, so with a set
          this small the lowest number is as likely to be luck as skill.</div>`
        : "")}
      <p class="muted tiny" style="padding:10px 16px 14px;margin:0">
        ${raw(answered
          ? html`<strong>Loss on expected</strong> is how surprised the model was
              by the answer you called correct, scored on the answer only. It is
              the measure to trust: it does not care about wording, and it can
              separate two models that both scored zero exact matches.
              <strong>Overlap</strong> credits a right answer worded differently,
              and just as happily credits a wrong answer that reuses the right
              words.`
          : html`None of these prompts has an expected answer, so there is
              nothing to score against — only the text each model produced.
              Add expected answers and score again to get numbers.`)}</p>
    </div>`;
}

function scoreDetail(score) {
  return html`
    <div class="card" style="margin-bottom:14px">
      <div class="row-between" style="margin-bottom:8px">
        <h3 style="margin:0">What ${score.model_name || "it"} answered</h3>
        <span class="tiny muted">${(score.items || []).length} prompts</span>
      </div>
      <div class="table-wrap"><table>
        <thead><tr><th>Prompt</th><th>Expected</th><th>Answered</th><th>Score</th></tr></thead>
        <tbody>
          ${raw((score.items || []).map((i) => html`
            <tr>
              <td class="tiny">${i.prompt}</td>
              <td class="tiny muted">${i.expected || "—"}</td>
              <td class="tiny">${i.answer || "(nothing)"}</td>
              <td class="tiny mono">
                ${raw(i.expected_loss != null
                  ? `${i.expected_loss.toFixed(3)}` : "—")}
                ${raw(i.exact ? ` <span class="badge badge-ok">exact</span>`
                  : i.contains ? ` <span class="badge">contains</span>` : "")}
              </td>
            </tr>`).join(""))}
        </tbody>
      </table></div>
    </div>`;
}
