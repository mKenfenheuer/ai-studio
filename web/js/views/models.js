/**
 * The model library: the ones somebody finished, not the ones somebody ran.
 *
 * Most runs are attempts. A handful are results, and the difference used to be
 * invisible — the list of runs put a two-minute experiment that diverged next
 * to the model three people are using, sorted by date, both called
 * "finetune-2". The library is the shortlist: a run somebody gave a name and a
 * version to, published either here or to Hugging Face.
 *
 * Publishing does not move or copy anything. The weights stay on the run,
 * which stays where it was; an entry here is a name pointing at it, so
 * unpublishing costs nothing and is not a deletion.
 */
import { api, events } from "../api.js";
import { html, raw, esc, on, toast, fmtAgo, $ } from "../util.js";
import { ribbon, rb, group, rbSearch } from "../ribbon.js";
import { pageHead, emptyState, confirmDestructive } from "../components.js";
import { openServeDialog } from "../registry.js";

export async function modelsView(mount) {
  let rows = [];
  let filter = "";
  const paint = async () => {
    rows = await api.library();
    draw();
  };
  const draw = () => {
    mount.innerHTML = layout(rows, filter);
    const box = $("#libSearch", mount);
    if (box) { box.value = filter; }
  };
  await paint();

  on(mount, "input", "#libSearch", (_e, t) => { filter = t.value; draw(); });
  on(mount, "click", "[data-serve]", (_e, t) =>
    openServeDialog({ id: t.dataset.serve, name: t.dataset.name }));
  on(mount, "click", "[data-unpublish]", async (_e, t) => {
    const row = rows.find((r) => r.id === t.dataset.unpublish);
    if (!await confirmDestructive({
      title: `Take "${row?.name || "this model"}" out of the library?`,
      consequences: [
        "The run and its weights are untouched — this removes the entry, not the model.",
        ...(row?.location === "hf"
          ? ["The Hugging Face repository is not touched either; delete that on the Hub."]
          : []),
      ],
      confirmLabel: "Remove the entry" })) return;
    try {
      await api.unpublish(t.dataset.unpublish);
      toast("Removed from the library.", "ok");
      await paint();
    } catch (e) { toast(e.message, "err"); }
  });
  return events.subscribe((m) => {
    if (["jobs_changed", "job_finished"].includes(m.type)) paint();
  });
}

function layout(rows, filter) {
  const f = filter.trim().toLowerCase();
  const shown = f
    ? rows.filter((r) => `${r.name} ${r.version} ${r.repo_id || ""} ${
        r.base_model || ""} ${r.project_name || ""}`.toLowerCase().includes(f))
    : rows;
  const here = shown.filter((r) => r.location === "local");
  const hub = shown.filter((r) => r.location === "hf");

  return html`
    ${raw(pageHead({
      title: "Models",
      sub: "What has been published: named, versioned, and pointing at the run "
         + "that made it.",
    }))}
    ${raw(ribbon({
      tabs: [{ key: "home", label: "Model library" }], active: "home",
      body: group("Find", [rbSearch("libSearch", { placeholder: "Search models…" })])
        + group("Elsewhere", [
          rb(null, "◇", "Projects", { href: "#/projects" }),
          rb(null, "▷", "Playground", { href: "#/play" }),
          rb(null, "🏷", "Served names", { href: "#/serving" }),
        ]),
      right: rows.length ? `<span class="badge">${rows.length}</span>` : "",
    }))}

    ${raw(rows.length ? html`
      ${raw(group2("Published in this studio", here,
        "Kept on this machine. Talk to it in the playground, point the API at "
        + "it under a served name, or download the weights from its run."))}
      ${raw(group2("Published to Hugging Face", hub,
        "Uploaded to the Hub, and recorded here so the run that made it can "
        + "still be found."))}
      ${raw(shown.length ? "" : `<p class="muted" style="text-align:center">
        Nothing matches “${esc(filter)}”.</p>`)}`
      : emptyState({
          icon: "⬢",
          title: "Nothing published yet",
          body: "When a run is good enough to keep, publish it: give it a name "
              + "and a version and it appears here, instead of being a row in "
              + "the run history that nobody can tell apart from the failures.",
          cta: { href: "#/projects", label: "Open your projects" },
        }))}`;
}

function group2(title, rows, blurb) {
  if (!rows.length) return "";
  return html`
    <h3 style="margin:18px 0 6px">${title}</h3>
    <p class="muted tiny" style="margin:0 0 10px;max-width:75ch">${blurb}</p>
    <div class="grid grid-2">${raw(rows.map(card).join(""))}</div>`;
}

function card(r) {
  const pm = r.primary_metric;
  return html`
    <div class="card">
      <div class="row-between" style="gap:8px;align-items:flex-start">
        <div style="min-width:0">
          <h3 style="margin:0">${r.name}
            ${raw(r.version ? `<span class="badge">${esc(r.version)}</span>` : "")}</h3>
          <p class="muted tiny" style="margin:3px 0 0">
            ${raw(r.base_model ? `from ${esc(r.base_model)} · ` : "")}
            published ${fmtAgo(r.published_at)}
            ${raw(r.project_name
              ? ` · <a href="#/projects/${esc(r.project_id)}">${esc(r.project_name)}</a>` : "")}
          </p>
        </div>
        ${raw(r.location === "hf"
          ? `<span class="badge badge-accent">Hub</span>`
          : `<span class="badge badge-ok">local</span>`)}
      </div>
      ${raw(r.notes ? `<p class="muted tiny" style="margin:8px 0 0">${esc(r.notes)}</p>` : "")}
      ${raw(pm && pm.value != null ? html`
        <p class="tiny" style="margin:8px 0 0">${pm.label || "Score"}:
          <strong class="mono">${typeof pm.value === "number"
            ? pm.value.toFixed(4) : pm.value}</strong></p>` : "")}
      ${raw((r.served_as || []).length ? html`
        <p class="tiny muted" style="margin:6px 0 0">Served as
          ${raw((r.served_as || []).map((a) =>
            `<code class="mono">${esc(a.alias || a)}</code>`).join(", "))}</p>` : "")}
      <div class="row" style="gap:6px;margin-top:10px;flex-wrap:wrap">
        ${raw(r.location === "hf" && r.url
          ? `<a class="btn-sm" href="${esc(r.url)}" target="_blank" rel="noopener">On the Hub ↗</a>`
          : "")}
        ${raw(r.has_model
          ? `<a class="btn-sm btn-primary" href="#/play/${esc(r.job_id)}">Try it</a>
             <button class="btn-sm" data-serve="${esc(r.job_id)}"
               data-name="${esc(r.name)}">Serve under a name</button>` : "")}
        <a class="btn-sm" href="#/jobs/${esc(r.job_id)}">The run</a>
        ${raw(r.mine ? `<button class="btn-sm btn-danger"
          data-unpublish="${esc(r.id)}" title="Remove the entry; the run stays">Remove</button>` : "")}
      </div>
      ${raw(!r.has_model && r.location === "local" ? html`
        <p class="muted tiny" style="margin:8px 0 0">The weights for this one
          have since been removed from the disk. The entry, the run and its
          numbers remain.</p>` : "")}
    </div>`;
}
