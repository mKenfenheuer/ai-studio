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
import { shareBox, wireShareBox } from "./share.js";

export async function evalView(mount, [evalId]) {
  let ev = await api.eval(evalId);
  let candidates = [];
  let openScore = null;

  const draw = () => { mount.innerHTML = layout(ev, candidates, openScore); wire(); };

  const refresh = async () => { ev = await api.eval(evalId); draw(); };

  function wire() {
    wireShareBox(mount, "eval", ev, refresh);

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
      if (!confirm("Remove this scoring from the comparison?")) return;
      try { await api.deleteScore(evalId, t.dataset.delScore); await refresh(); }
      catch (ex) { toast(ex.message, "err"); }
    });

    on(mount, "click", "#copyEval", async () => {
      try {
        const copy = await api.copyEval(evalId);
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

function layout(ev, candidates, openScore) {
  const scores = ev.scores || [];
  const answered = (ev.items || []).filter((i) => i.expected).length;
  return html`
    <div class="page-head">
      <a href="#/evals" class="tiny">← All prompt sets</a>
      <div class="row-between" style="flex-wrap:wrap;gap:8px;margin-top:6px">
        <h1 style="margin:0">${ev.name}</h1>
        <div class="row" style="gap:6px">
          <span class="badge">${(ev.items || []).length} prompts</span>
          <button class="btn-sm" id="copyEval" title="Make an editable copy">Copy</button>
        </div>
      </div>
      ${raw(ev.notes ? `<p class="sub">${esc(ev.notes)}</p>` : "")}
    </div>

    ${raw(scoreTable(scores, answered))}
    ${raw(openScore ? scoreDetail(openScore) : "")}

    <div class="grid grid-2" style="margin-bottom:14px">
      <div class="card">
        <h3>Score some models</h3>
        ${raw(candidates.length ? html`
          <p class="muted tiny">Each one is loaded onto a machine in turn and
            asked all ${(ev.items || []).length} prompts. That takes a while,
            so it runs as a queued job you can watch and stop.</p>
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
                <div class="hint">Zero means the same answer every time, which
                  is what makes two scorings comparable.</div>
              </div>
            </div>
          </details>
          <button class="btn-primary btn-sm" id="runEval" style="margin-top:10px">
            Score these models</button>`
        : html`
          <p class="muted tiny">No finished models to score yet. Train
            something first — a run that produced a model, or one you stopped
            and kept.</p>
          <p><a class="btn btn-sm btn-primary" href="#/new">Start a run</a></p>`)}
      </div>

      <div class="card">
        <h3>The prompts</h3>
        <p class="muted tiny">${answered} of ${(ev.items || []).length} have an
          expected answer. Prompts without one still get asked, and their
          answers are kept to read — they simply cannot be scored.</p>
        <div class="promptlist">
          ${raw((ev.items || []).slice(0, 40).map((i) => html`
            <div class="prompt-row">
              <div class="p">${i.prompt}</div>
              ${raw(i.expected
                ? `<div class="e">→ ${esc(i.expected)}</div>`
                : `<div class="e muted">no expected answer</div>`)}
            </div>`).join(""))}
          ${raw((ev.items || []).length > 40
            ? `<p class="muted tiny">…and ${(ev.items || []).length - 40} more.</p>` : "")}
        </div>
      </div>
    </div>

    <div id="shareRow">${raw(shareBox("eval", ev))}</div>`;
}

function scoreTable(scores, answered) {
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
            const isBest = best !== null && m.expected_loss === best;
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
                  <button class="btn-sm" data-open-score="${esc(s.id)}">Answers</button>
                  <button class="btn-sm btn-danger" data-del-score="${esc(s.id)}"
                          title="Remove from the comparison">✕</button>
                </div></td>
              </tr>`;
          }).join(""))}
        </tbody>
      </table></div>
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
