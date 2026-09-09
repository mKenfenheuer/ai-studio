/**
 * What a project was made of, drawn.
 *
 * The stage map answers "how far along is this". It cannot answer the question
 * people actually argue about three weeks later -- which data went into the
 * good model, whether the prompt set it was scored on came out of the rows it
 * trained on, what the published version was built from. Every one of those
 * facts is already recorded on the row that resulted: a derived dataset knows
 * its parent, a run knows the dataset it read and the run it continued, a
 * prompt set knows the split it came from, a library entry knows its run.
 *
 * Read together they are a directed graph. This draws it: columns left to
 * right in the order things were made from each other, so the eye follows the
 * work forwards, and every box is a link to the thing itself.
 *
 * Hand-drawn SVG rather than a graph library, for the reason the rest of this
 * front end has no build step: a layered DAG of thirty nodes is a hundred
 * lines of arithmetic, and a dependency that draws it is a hundred kilobytes
 * and a version to keep up with.
 */
import { html, raw, esc } from "../util.js";

const NODE_W = 190;
const NODE_H = 70;
const GAP_X = 104;         // room for an edge label between columns
const GAP_Y = 18;
const PAD = 16;

const TONE = {
  dataset: "n-data", training: "n-train", writing: "n-write",
  scoring: "n-score", eval: "n-score", benchmark: "n-score",
  export: "n-export", published: "n-pub", run: "n-run",
};

/** Longest path from a root: a node sits to the right of everything it was
 *  made from, however many hops back that is. Cycles cannot occur -- nothing
 *  here can be made from something made later -- but a malformed graph must
 *  not hang the page, so the walk is depth-capped. */
function columns(nodes, edges) {
  const into = new Map(nodes.map((n) => [n.id, []]));
  edges.forEach((e) => { if (into.has(e.to)) into.get(e.to).push(e.from); });
  const depth = new Map();
  const of = (id, seen = new Set()) => {
    if (depth.has(id)) return depth.get(id);
    if (seen.has(id) || seen.size > 64) return 0;
    seen.add(id);
    const parents = into.get(id) || [];
    const d = parents.length
      ? Math.max(...parents.map((p) => of(p, new Set(seen)) + 1)) : 0;
    depth.set(id, d);
    return d;
  };
  nodes.forEach((n) => of(n.id));
  return depth;
}

/** The same graph, an inch tall, for the top of the project page.
 *
 *  The full drawing answers "what was this made of" when somebody goes
 *  looking. The strip answers "what shape is this project in" at a glance --
 *  how much data, how many attempts, whether anything came out the end -- and
 *  it is the same layout, so recognising one teaches the other. Nodes are
 *  dots here because a name at this size is unreadable; every one carries its
 *  name as a tooltip, and the whole strip opens the full graph.
 */
export function projectGraphStrip(data) {
  const nodes = data?.nodes || [];
  const edges = (data?.edges || []).filter((e) =>
    nodes.some((n) => n.id === e.from) && nodes.some((n) => n.id === e.to));
  if (nodes.length < 2) return "";

  const D = 13;              // a dot
  const COL = 54;            // and the step between columns
  const ROW = 19;
  const depth = columns(nodes, edges);
  const lanes = new Map();
  nodes.slice().sort((a, b) => (a.at || 0) - (b.at || 0)).forEach((n) => {
    const col = depth.get(n.id) || 0;
    const row = (lanes.get(col) || 0);
    lanes.set(col, row + 1);
    n._sx = 8 + col * COL;
    n._sy = 8 + row * ROW;
  });
  const cols = Math.max(...[...lanes.keys()]) + 1;
  const rows = Math.max(...[...lanes.values()]);
  const w = 16 + cols * COL - (COL - D);
  const h = 16 + rows * ROW - (ROW - D);
  const at = Object.fromEntries(nodes.map((n) => [n.id, n]));

  const wires = edges.map((e) => {
    const a = at[e.from];
    const b = at[e.to];
    const x1 = a._sx + D;
    const y1 = a._sy + D / 2;
    const x2 = b._sx;
    const y2 = b._sy + D / 2;
    const mid = x1 + (x2 - x1) / 2;
    return `<path class="g-edge" d="M ${x1} ${y1} C ${mid} ${y1}, ${mid} ${
      y2}, ${x2} ${y2}"></path>`;
  }).join("");

  const dots = nodes.map((n) => `
    <g class="g-dot ${TONE[n.kind] || "n-run"}">
      <rect x="${n._sx}" y="${n._sy}" width="${D}" height="${D}" rx="4"></rect>
      <title>${esc(n.label)}${n.sub ? " — " + esc(n.sub) : ""}</title>
    </g>`).join("");

  return html`
    <button type="button" class="g-strip" id="openGraph"
            title="Open the full graph">
      <svg viewBox="0 0 ${w} ${h}" width="${w}" height="${h}"
           role="img" aria-label="What this project was made of, in miniature">
        ${raw(wires)}${raw(dots)}
      </svg>
      <span class="g-strip-key">${raw(countsByKind(nodes))}</span>
      <span class="g-strip-hint">what it was made of →</span>
    </button>`;
}

export function projectGraph(data) {
  const nodes = data.nodes || [];
  const edges = (data.edges || []).filter((e) =>
    nodes.some((n) => n.id === e.from) && nodes.some((n) => n.id === e.to));
  if (!nodes.length) {
    return html`
      <div class="card empty"><div class="big" aria-hidden="true">🌱</div>
        <h3>Nothing to draw yet</h3>
        <p class="muted">The graph fills in as the project does: the data, the
          runs that read it, what judged them, and what was published at the
          end.</p></div>`;
  }

  const depth = columns(nodes, edges);
  const lanes = new Map();
  nodes.slice()
    .sort((a, b) => (a.at || 0) - (b.at || 0))
    .forEach((n) => {
      const col = depth.get(n.id) || 0;
      const row = (lanes.get(col) || []).length;
      lanes.set(col, [...(lanes.get(col) || []), n.id]);
      n._x = PAD + col * (NODE_W + GAP_X);
      n._y = PAD + row * (NODE_H + GAP_Y);
    });

  const width = PAD * 2 + (Math.max(...[...lanes.keys()]) + 1) * (NODE_W + GAP_X) - GAP_X;
  const height = PAD * 2 + Math.max(...[...lanes.values()].map((l) => l.length))
                 * (NODE_H + GAP_Y) - GAP_Y;
  const at = Object.fromEntries(nodes.map((n) => [n.id, n]));

  // Edges first, so a line never sits on top of a box. Several edges often
  // leave the same column for the same one -- three scorings of one model --
  // and a caption placed halfway along every one of them lands in the same
  // spot three times. Each is nudged along its own curve instead.
  const perGutter = new Map();
  const wires = edges.map((e) => {
    const a = at[e.from];
    const b = at[e.to];
    const x1 = a._x + NODE_W;
    const y1 = a._y + NODE_H / 2;
    const x2 = b._x;
    const y2 = b._y + NODE_H / 2;
    const mid = x1 + (x2 - x1) / 2;
    // A cubic through the midpoint: the same shape a git graph uses, and the
    // one that stays readable when two edges arrive at the same box.
    const d = `M ${x1} ${y1} C ${mid} ${y1}, ${mid} ${y2}, ${x2} ${y2}`;
    // The label goes in the gutter between the two columns, never over a box,
    // and only when the edge is short enough for it to be legible there. A
    // long hop across three columns passes over other work; a caption laid on
    // top of that is worse than no caption.
    const room = Math.abs(x2 - x1) <= NODE_W + GAP_X + 4;
    // One caption per phrase per gutter. A model scored four times fans out
    // four identical "scored" edges, and four copies of the word stacked on
    // each other say nothing the fan does not already say.
    const gutter = Math.round(x1);
    const said = perGutter.get(gutter) || new Set();
    perGutter.set(gutter, said);
    const first = e.label && !said.has(e.label);
    if (first) said.add(e.label);
    const nth = said.size - 1;
    // A point on the curve itself, a third to two thirds along depending on
    // how many captions this gutter is already carrying.
    const t = 0.34 + 0.16 * (nth % 3);
    // Named for what it is, and NOT `at`: the lookup table above is called
    // that, and shadowing it inside this closure put the line that reads it
    // into the temporal dead zone -- so every graph with an edge in it threw
    // and the tab sat on "working out what came from what" forever.
    const along = (a, b, c, d) => {
      const u = 1 - t;
      return u * u * u * a + 3 * u * u * t * b + 3 * u * t * t * c + t * t * t * d;
    };
    const lx = along(x1, mid, mid, x2);
    const ly = along(y1, y1, y2, y2);
    const label = room && first
      ? `<text class="g-edge-label" x="${lx.toFixed(1)}" y="${(ly - 6).toFixed(1)}"
             text-anchor="middle">${esc(clip(e.label, 15))}<title>${
               esc(e.label)}</title></text>`
      : "";
    return `<path class="g-edge" d="${d}" marker-end="url(#g-arrow)"></path>${label}`;
  }).join("");

  const boxes = nodes.map((n) => html`
    <a href="${esc(n.href || "")}" class="g-node ${TONE[n.kind] || "n-run"}">
      <rect x="${n._x}" y="${n._y}" width="${NODE_W}" height="${NODE_H}" rx="9"></rect>
      <text class="g-ico" x="${n._x + 12}" y="${n._y + 23}">${esc(n.icon || "")}</text>
      <text class="g-title" x="${n._x + 32}" y="${n._y + 23}">${
        esc(clip(n.label, 20))}</text>
      <text class="g-sub" x="${n._x + 12}" y="${n._y + 41}">${
        esc(clip(n.sub || "", 30))}</text>
      ${raw(n.warn || n.borrowed ? `<text class="g-${n.warn ? "warn" : "sub"}"
        x="${n._x + 12}" y="${n._y + 58}">${esc(clip(
          [n.borrowed ? "from another project" : "", n.warn || ""]
            .filter(Boolean).join(" · "), 30))}</text>` : "")}
      <title>${esc(n.label)}${n.sub ? " — " + esc(n.sub) : ""}</title>
    </a>`).join("");

  return html`
    <div class="card" style="padding:12px">
      <p class="muted tiny" style="margin:0 0 10px">
        Left to right, in the order things were made from each other. Every box
        opens the thing it names.</p>
      <div class="g-scroll">
        <svg class="g-svg" viewBox="0 0 ${width} ${height}"
             width="${width}" height="${height}" role="img"
             aria-label="What this project was made of">
          <defs>
            <marker id="g-arrow" viewBox="0 0 8 8" refX="7" refY="4"
                    markerWidth="7" markerHeight="7" orient="auto">
              <path d="M0,0 L8,4 L0,8 z" class="g-arrow-head"></path>
            </marker>
          </defs>
          ${raw(wires)}
          ${raw(boxes)}
        </svg>
      </div>
      ${raw(legend(nodes))}
    </div>`;
}

const clip = (s, n) => (String(s || "").length > n
  ? String(s).slice(0, n - 1) + "…" : String(s || ""));

// Singular and plural, because "1 datasets" is the kind of thing that makes a
// page look like nobody read it.
const KIND_WORDS = {
  dataset: ["dataset", "datasets"],
  training: ["model trained", "models trained"],
  writing: ["data-writing run", "data-writing runs"],
  scoring: ["scoring", "scorings"],
  eval: ["prompt set", "prompt sets"],
  benchmark: ["benchmark", "benchmarks"],
  export: ["export", "exports"],
  published: ["published", "published"],
  run: ["other run", "other runs"],
};

/** What the dots are, in words. A shape with no key is decoration. */
function countsByKind(nodes) {
  const order = ["dataset", "training", "writing", "eval", "benchmark",
                 "scoring", "export", "published", "run"];
  const count = new Map();
  nodes.forEach((n) => count.set(n.kind, (count.get(n.kind) || 0) + 1));
  return order.filter((k) => count.has(k)).map((k) => {
    const n = count.get(k);
    const [one, many] = KIND_WORDS[k] || [k, k];
    return `<span class="g-key ${TONE[k] || "n-run"}">${n} ${
      esc(n === 1 ? one : many)}</span>`;
  }).join("");
}

function legend(nodes) {
  return html`
    <div class="row" style="gap:8px;flex-wrap:wrap;margin-top:10px">
      ${raw(countsByKind(nodes))}
    </div>`;
}
