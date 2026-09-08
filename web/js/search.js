/**
 * One box that finds anything.
 *
 * There were three unconnected filters — datasets, people, and rows inside one
 * dataset — and none of them on the list that grows fastest. Nothing answered
 * "where is the thing called roughly this", which is the question somebody has
 * after a fortnight of using a studio that names two runs of the same model on
 * the same data identically.
 *
 * Deliberately small: names, not contents. It searches what the studio already
 * has in hand rather than adding an endpoint, so it opens instantly and works
 * offline of everything except the three lists it reads.
 */
import { api } from "./api.js";
import { esc, html, raw, $, $$, on, modal, fmtNum, fmtAgo } from "./util.js";
import { kindOf, subjectOf } from "./kinds.js";

let open = null;

/** Score a candidate against the query. Exact prefix beats word-start beats
 *  anywhere, so typing "sm" finds "SmolLM2 on alpaca" before "a dataset of
 *  small talk". */
function score(text, q) {
  const t = String(text || "").toLowerCase();
  if (!t) return 0;
  if (t === q) return 100;
  if (t.startsWith(q)) return 80;
  if (new RegExp(`\\b${q.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}`).test(t)) return 60;
  return t.includes(q) ? 40 : 0;
}

export async function openSearch() {
  if (open) return;
  const dlg = modal({
    title: "Find anything",
    width: 560,
    onClose: () => { open = null; },
    body: html`
      <input id="omniBox" type="search" autocomplete="off" spellcheck="false"
             placeholder="A run, a dataset, a prompt set, a machine…"
             aria-label="Search" aria-controls="omniResults">
      <div id="omniResults" class="omni-results" role="listbox"
           aria-label="Results"></div>
      <p class="muted tiny" style="margin:8px 0 0">Enter opens the first result.
        Press <kbd>/</kbd> anywhere to get back here.</p>`,
  });
  open = dlg;

  const box = $("#omniBox", dlg);
  const out = $("#omniResults", dlg);
  box.focus();

  // Everything is fetched once, when the box opens. These are the same three
  // lists the sidebar pages read, and they are small enough that filtering in
  // the browser is instant and a search endpoint would be ceremony.
  let items = [];
  out.innerHTML = `<div class="muted tiny" style="padding:10px">Looking…</div>`;
  const [jobs, datasets, evals, projects, library] = await Promise.all([
    api.jobs().catch(() => []),
    api.datasets().catch(() => []),
    api.evals().catch(() => []),
    // A project is now the usual thing somebody is looking for -- it is where
    // the runs, the data and the scores for one model all hang together --
    // and a published model is the other, because its name is the one anybody
    // remembers.
    api.projects().then((d) => d.projects || []).catch(() => []),
    api.library().catch(() => []),
  ]);
  items = [
    ...projects.map((p) => ({
      kind: "Project", icon: "◇", name: p.name,
      note: p.goal || `${p.runs || 0} runs · ${p.datasets || 0} datasets`,
      extra: fmtAgo(p.updated_at),
      href: `#/projects/${p.id}`, hay: [p.name, p.goal],
    })),
    ...library.map((m) => ({
      kind: "Model", icon: "⬢", name: `${m.name}${m.version ? " " + m.version : ""}`,
      note: m.location === "hf" ? m.repo_id || "on the Hub" : "published here",
      extra: fmtAgo(m.published_at),
      href: `#/models`, hay: [m.name, m.repo_id, m.base_model],
    })),
    ...jobs.map((j) => ({
      kind: kindOf(j).label, icon: kindOf(j).icon, name: j.name,
      note: subjectOf(j), extra: fmtAgo(j.finished_at || j.created_at),
      href: `#/jobs/${j.id}`, hay: [j.name, subjectOf(j)],
    })),
    ...datasets.map((d) => ({
      kind: "Dataset", icon: "▤", name: d.name,
      note: `${fmtNum(d.rows)} rows${d.origin ? ` · ${d.origin}` : ""}`,
      extra: fmtAgo(d.updated_at),
      href: `#/data/${d.id}`, hay: [d.name, d.origin],
    })),
    ...evals.map((e) => ({
      kind: "Prompt set", icon: "◎", name: e.name,
      note: `${(e.items || []).length} prompts`, extra: fmtAgo(e.updated_at),
      href: `#/evals/${e.id}`, hay: [e.name, e.notes],
    })),
  ];

  let hits = [];
  let at = 0;

  const render = () => {
    const q = box.value.trim().toLowerCase();
    hits = !q ? items.slice(0, 8)
      : items
        .map((it) => ({ it, s: Math.max(...it.hay.map((h) => score(h, q))) }))
        .filter((r) => r.s > 0)
        .sort((a, b) => b.s - a.s)
        .slice(0, 12)
        .map((r) => r.it);
    at = 0;
    out.innerHTML = hits.length ? hits.map((it, i) => html`
      <a class="omni-hit ${i === at ? "on" : ""}" href="${it.href}" role="option"
         aria-selected="${i === at}" data-hit="${i}">
        <span class="ico" aria-hidden="true">${it.icon}</span>
        <span class="omni-text">
          <span class="omni-name">${it.name}</span>
          <span class="omni-note">${it.kind}${it.note ? " · " + it.note : ""}</span>
        </span>
        <span class="omni-when">${it.extra}</span>
      </a>`).join("")
      : `<div class="muted tiny" style="padding:10px">${
          q ? "Nothing matches that." : "Nothing here yet."}</div>`;
  };
  render();

  const move = (by) => {
    if (!hits.length) return;
    at = (at + by + hits.length) % hits.length;
    $$(".omni-hit", out).forEach((el, i) => {
      el.classList.toggle("on", i === at);
      el.setAttribute("aria-selected", String(i === at));
      if (i === at) el.scrollIntoView({ block: "nearest" });
    });
  };

  const go = (i) => {
    const hit = hits[i];
    if (!hit) return;
    dlg.close();
    location.hash = hit.href.slice(1);
  };

  on(dlg, "input", "#omniBox", render);
  box.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown") { e.preventDefault(); move(1); }
    else if (e.key === "ArrowUp") { e.preventDefault(); move(-1); }
    else if (e.key === "Enter") { e.preventDefault(); go(at); }
  });
  on(dlg, "click", "[data-hit]", (e, t) => {
    e.preventDefault();
    go(Number(t.dataset.hit));
  });
}

/** `/` opens it, from anywhere that is not already a text box. */
export function wireSearchKey() {
  document.addEventListener("keydown", (e) => {
    if (e.key !== "/" || e.metaKey || e.ctrlKey || e.altKey) return;
    const el = e.target;
    const tag = (el?.tagName || "").toLowerCase();
    if (["input", "textarea", "select"].includes(tag) || el?.isContentEditable) return;
    e.preventDefault();
    openSearch();
  });
}
