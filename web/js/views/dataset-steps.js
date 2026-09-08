/**
 * The catalogue of things the dataset editor can do to rows.
 *
 * A step is `{ type, ops }`: `ops` is exactly the options dict the controller
 * runs, and `type` only says which form edits it. Keeping the wire format as
 * the model means the formula bar can show a step as it will be sent, a
 * recipe saved on a derived dataset can be reopened without translation, and
 * a step nobody wrote a form for is still a step.
 *
 * Each entry knows how to draw its form for a given set of columns, how to
 * read that form back into ops, and how to describe its ops in a sentence.
 */
import { html, raw, esc, $, $$ } from "../util.js";

// The filters a calculated column may use. Listed for the hint text; the
// controller is the authority on what they do.
export const FILTERS = ["upper", "lower", "title", "trim", "lines", "first", "last",
                        "len", "words", "json", "slice:0:200"];

const NEWLINE = String.fromCharCode(10);

const list = (text) => (text || "").split(",").map((s) => s.trim()).filter(Boolean);
const num = (v) => { const n = +v; return Number.isFinite(n) && n > 0 ? Math.floor(n) : 0; };

/** Which columns are ticked in a form. */
const ticked = (dlg, name) => $$(`[data-tick="${name}"]:checked`, dlg).map((c) => c.value);

/** A list of tick boxes, one per column. */
function columnTicks(name, columns, chosen = []) {
  if (!columns.length) {
    return `<p class="muted tiny">No columns are known yet — open the rows
      first, or type the names below.</p>
      <input class="mono" data-free="${name}" placeholder="a, b, c">`;
  }
  return `<div class="tick-grid">${columns.map((c) => html`
    <label class="check"><input type="checkbox" data-tick="${name}" value="${c}"${
      chosen.includes(c) ? " checked" : ""}> <span class="mono">${c}</span></label>`).join("")}
  </div>`;
}

function readTicks(dlg, name) {
  const free = $(`[data-free="${name}"]`, dlg);
  return free ? list(free.value) : ticked(dlg, name);
}

/** Chips that insert `{column}` at the caret of the template box. */
function columnChips(columns) {
  if (!columns.length) return "";
  return `<div class="chips">${columns.map((c) => html`
    <button type="button" class="chip" data-insert="{${c}}">{${c}}</button>`).join("")}</div>`;
}

const CONDITION_WORDS = {
  eq: "is", ne: "is not", contains: "contains", not_contains: "does not contain",
  starts: "starts with", ends: "ends with", regex: "matches regex",
  empty: "is empty", not_empty: "is not empty", gt: ">", gte: "≥", lt: "<", lte: "≤",
  in: "is one of", not_in: "is not one of",
};

/** One condition of a `where` step: column, operator, value. */
function condRow(c, n, columns) {
  const cols = columns.includes(c.column) || !c.column ? columns : [c.column, ...columns];
  const multi = c.op === "in" || c.op === "not_in";
  const noValue = c.op === "empty" || c.op === "not_empty";
  return html`
    <div class="cond" data-cond data-chosen="${JSON.stringify(c.values || [])}">
      <div class="cond-line">
        ${raw(cols.length ? html`<select class="c-col">${raw(cols.map((k) =>
            html`<option value="${k}"${k === c.column ? " selected" : ""}>${k}</option>`).join(""))}
          </select>` : html`<input class="c-col mono" value="${c.column || ""}" placeholder="column">`)}
        <select class="c-op">${raw(Object.entries(CONDITION_WORDS).map(([k, w]) =>
          html`<option value="${k}"${k === (c.op || "contains") ? " selected" : ""}>${w}</option>`).join(""))}
        </select>
        <input class="c-val mono" list="dl_${n}" value="${c.value ?? ""}" placeholder="value"
               ${multi || noValue ? "hidden" : ""}>
        <datalist id="dl_${n}"></datalist>
        <button type="button" class="chip-x" data-del-cond title="Remove this condition">✕</button>
      </div>
      <div class="c-vals" ${multi ? "" : "hidden"}></div>
    </div>`;
}

export const STEPS = {
  // ---- columns -----------------------------------------------------------
  rename: {
    tab: "transform", label: "Rename", icon: "✎",
    blurb: "Give columns the names the trainer knows — instruction, output, "
      + "messages, text — or just clearer ones.",
    form(ops, { columns }) {
      const map = ops.rename || {};
      const cols = columns.length ? columns : Object.keys(map);
      return html`
        <table class="table rename-table"><thead><tr><th>Column</th><th>New name</th></tr></thead>
        <tbody>${raw(cols.map((c) => html`
          <tr><td class="mono">${c}</td>
              <td><input class="mono" data-rename="${c}" value="${map[c] || ""}"
                         placeholder="keep as ${c}"></td></tr>`).join(""))}
        </tbody></table>
        ${raw(cols.length ? "" : `<input class="mono" data-free="rename"
          placeholder="old=new, other=thing">`)}`;
    },
    read(dlg) {
      const rename = {};
      $$("[data-rename]", dlg).forEach((el) => {
        const to = el.value.trim();
        if (to && to !== el.dataset.rename) rename[el.dataset.rename] = to;
      });
      const free = $("[data-free=rename]", dlg);
      if (free) free.value.split(",").forEach((pair) => {
        const [a, b] = pair.split("=").map((s) => (s || "").trim());
        if (a && b) rename[a] = b;
      });
      if (!Object.keys(rename).length) throw new Error("Nothing renamed yet.");
      return { rename };
    },
    describe: (ops) => "Renamed " + Object.entries(ops.rename || {})
      .map(([a, b]) => `${a} → ${b}`).join(", "),
    infer: (ops) => !!ops.rename,
  },

  split_column: {
    tab: "transform", label: "Split column", icon: "⫶",
    blurb: "One column into several: a full name into first and last, a "
      + "tab-separated field into its parts.",
    form(ops, { columns }) {
      const spec = (ops.split_columns || [])[0] || {};
      return html`
        <div class="field"><label>Split the column</label>
          ${raw(columns.length ? html`<select id="sf_from">${raw(columns.map((c) =>
              html`<option value="${c}"${spec.from === c ? " selected" : ""}>${c}</option>`).join(""))}
            </select>` : html`<input id="sf_from" class="mono" value="${spec.from || ""}">`)}
        </div>
        <div class="field"><label for="sf_into">Into the columns</label>
          <input id="sf_into" class="mono" value="${(spec.into || []).join(", ")}"
                 placeholder="first, last">
          <div class="hint">Comma-separated. Extra parts are dropped; missing
            ones are left blank.</div></div>
        <div class="field"><label for="sf_by">Split on</label>
          <input id="sf_by" class="mono" value="${spec.by ?? ""}" placeholder="whitespace">
          <div class="hint">Blank splits on whitespace. Use <code>,</code>,
            <code>|</code>, <code>\\t</code> for a tab.</div></div>`;
    },
    read(dlg) {
      const from = $("#sf_from", dlg).value.trim();
      const into = list($("#sf_into", dlg).value);
      let by = $("#sf_by", dlg).value;
      if (by === "\\t") by = "\t";
      if (!from || !into.length) throw new Error("Which column, into which columns?");
      return { split_columns: [{ from, into, by: by === "" ? null : by }] };
    },
    describe: (ops) => (ops.split_columns || []).map((s) =>
      `Split ${s.from} into ${(s.into || []).join(", ")}`).join("; "),
    infer: (ops) => !!ops.split_columns,
  },

  calc: {
    tab: "columns", label: "From template", icon: "ƒx",
    blurb: "A new column built from the others. This is how two columns "
      + "nobody's trainer recognises become one it does.",
    form(ops, { columns }) {
      const spec = (ops.columns || [])[0]
        || { name: ops.template_column || "text", template: ops.template || "" };
      const a = columns[0] || "question", b = columns[1] || "answer";
      return html`
        <div class="field"><label for="sf_name">Column name</label>
          <input id="sf_name" class="mono" value="${spec.name || "text"}"></div>
        <div class="field"><label for="sf_tpl">Template</label>
          <textarea id="sf_tpl" class="mono" rows="5"
            placeholder="Q: {${a}}&#10;A: {${b}}">${spec.template || ""}</textarea>
          ${raw(columnChips(columns))}
          <div class="hint">Any <code>{column}</code> is replaced with that
            row's value. Values can pass through <code>|</code> filters:
            <code>{title|trim|upper}</code>, <code>{body|slice:0:400}</code>,
            <code>{tools|json}</code>. Available:
            ${raw(FILTERS.map((f) => `<code>${f}</code>`).join(" "))}.
            The last column built is what the trainer then reads.</div></div>`;
    },
    read(dlg) {
      const name = $("#sf_name", dlg).value.trim();
      const template = $("#sf_tpl", dlg).value;
      if (!name) throw new Error("The column needs a name.");
      if (!template.trim()) throw new Error("The template is empty.");
      return { columns: [{ name, template }] };
    },
    describe: (ops) => (ops.columns || []).map((c) =>
      `Added ${c.name} = ${(c.template || "").split(NEWLINE)[0].slice(0, 40)}${
        (c.template || "").length > 40 ? "…" : ""}`).join("; ")
      || `Added ${ops.template_column || "text"} from a template`,
    infer: (ops) => !!(ops.columns || ops.template),
  },

  drop_columns: {
    tab: "transform", label: "Remove columns", icon: "⌫",
    blurb: "Throw columns away. Applied after templates, so a column a "
      + "template reads from can still go.",
    form: (ops, { columns }) => columnTicks("drop", columns, ops.drop_columns || []),
    read(dlg) {
      const drop_columns = readTicks(dlg, "drop");
      if (!drop_columns.length) throw new Error("Tick at least one column.");
      return { drop_columns };
    },
    describe: (ops) => `Removed ${(ops.drop_columns || []).join(", ")}`,
    infer: (ops) => !!ops.drop_columns,
  },

  keep_columns: {
    tab: "transform", label: "Keep columns", icon: "▣",
    blurb: "Keep only these; everything else goes. The split column always stays.",
    form: (ops, { columns }) => columnTicks("keep", columns, ops.keep_columns || []),
    read(dlg) {
      const keep_columns = readTicks(dlg, "keep");
      if (!keep_columns.length) throw new Error("Tick at least one column.");
      return { keep_columns };
    },
    describe: (ops) => `Kept only ${(ops.keep_columns || []).join(", ")}`,
    infer: (ops) => !!ops.keep_columns,
  },

  // ---- rows --------------------------------------------------------------
  where: {
    tab: "rows", label: "By column value", icon: "⊟",
    blurb: "Keep the rows where a column holds what you say. Several "
      + "conditions all have to hold.",
    form(ops, { columns }) {
      const conds = (ops.where || []).length ? ops.where
        : [{ column: columns[0] || "", op: "contains", value: "" }];
      return html`
        <div id="conds">${raw(conds.map((c, n) => condRow(c, n, columns)).join(""))}</div>
        <button type="button" class="btn-sm" data-add-cond>+ Another condition</button>
        <div class="hint" style="margin-top:8px">Text matches ignore case. “is” compares
          numbers as numbers, so <code>12</code> matches <code>12.0</code>. The value
          list shows what the column holds at this step, most common first.</div>`;
    },
    wire(dlg, ctx) {
      const columns = ctx.columns;
      // What a column holds, fetched when a condition's column changes and
      // offered both as a datalist and, for "is one of", as tick boxes.
      const load = async (row) => {
        const col = $(".c-col", row).value;
        const op = $(".c-op", row).value;
        const box = $(".c-vals", row);
        const list = $("datalist", row);
        const multi = op === "in" || op === "not_in";
        box.hidden = !multi;
        $(".c-val", row).hidden = multi || op === "empty" || op === "not_empty";
        if (!col || !ctx.values) return;
        if (row.dataset.loadedFor !== col) {
          row.dataset.loadedFor = col;
          box.innerHTML = `<span class="muted tiny">Reading values…</span>`;
          let r;
          try { r = await ctx.values(col); }
          catch (ex) { box.innerHTML = `<span class="muted tiny">${esc(ex.message)}</span>`; return; }
          if ($(".c-col", row).value !== col) return;
          const chosen = new Set(JSON.parse(row.dataset.chosen || "[]"));
          list.innerHTML = r.values.map((v) => `<option value="${esc(v.value)}">`).join("");
          box.innerHTML = (r.values.length ? `<div class="vals-list">${r.values.map((v) => html`
              <label class="check"><input type="checkbox" class="c-tick" value="${v.value}"${
                chosen.has(v.value) ? " checked" : ""}>
                <span class="c-tick-v">${v.value === "" ? raw('<span class="muted">(empty)</span>') : v.value}</span>
                <span class="muted tiny">${v.count}</span></label>`).join("")}</div>` : "")
            + `<div class="muted tiny" style="margin-top:4px">${r.distinct} distinct value${
                r.distinct === 1 ? "" : "s"} in ${r.rows} rows${
                r.distinct > r.values.length ? ` — the ${r.values.length} most common shown` : ""}</div>`;
        }
      };
      $$("[data-cond]", dlg).forEach(load);
      dlg.addEventListener("change", (e) => {
        const row = e.target.closest("[data-cond]");
        if (row && (e.target.matches(".c-col") || e.target.matches(".c-op"))) load(row);
      });
      dlg.addEventListener("click", (e) => {
        if (e.target.closest("[data-add-cond]")) {
          const n = $$("[data-cond]", dlg).length;
          $("#conds", dlg).insertAdjacentHTML("beforeend",
            condRow({ column: columns[0] || "", op: "contains", value: "" }, n, columns));
          load($$("[data-cond]", dlg).pop());
        }
        const del = e.target.closest("[data-del-cond]");
        if (del) { del.closest("[data-cond]").remove(); dlg.dispatchEvent(new Event("input")); }
      });
    },
    read(dlg) {
      const where = $$("[data-cond]", dlg).map((row) => {
        const column = $(".c-col", row).value.trim();
        const op = $(".c-op", row).value;
        const out = { column, op };
        if (op === "in" || op === "not_in") {
          out.values = $$(".c-tick:checked", row).map((t) => t.value);
          if (!out.values.length) throw new Error("Tick at least one value.");
        } else if (op !== "empty" && op !== "not_empty") {
          out.value = $(".c-val", row).value;
          if (out.value === "" && op !== "eq" && op !== "ne") throw new Error("What should it match?");
        }
        return out;
      }).filter((c) => c.column);
      if (!where.length) throw new Error("Which column?");
      return { where };
    },
    describe: (ops) => "Kept rows where " + (ops.where || []).map((c) => {
      const w = CONDITION_WORDS[c.op] || c.op;
      if (c.op === "empty" || c.op === "not_empty") return `${c.column} ${w}`;
      if (c.op === "in" || c.op === "not_in") {
        const v = c.values || [];
        return `${c.column} ${w} ${v.slice(0, 3).map((x) => JSON.stringify(x)).join(", ")}${
          v.length > 3 ? ` +${v.length - 3}` : ""}`;
      }
      return `${c.column} ${w} ${JSON.stringify(c.value ?? "")}`;
    }).join(" and "),
    infer: (ops) => !!ops.where,
  },

  drop_empty: {
    tab: "rows", label: "Drop empty", icon: "∅", instant: true,
    blurb: "Rows that render as nothing, and conversations whose every answer is blank.",
    form: () => "", read: () => ({ drop_empty: true }),
    describe: () => "Dropped rows that say nothing",
    infer: (ops) => !!ops.drop_empty,
  },
  dedupe: {
    tab: "rows", label: "Remove duplicates", icon: "⧉", instant: true,
    blurb: "Exact repeats, compared as the full record including any reasoning.",
    form: () => "", read: () => ({ dedupe: true }),
    describe: () => "Removed exact duplicates",
    infer: (ops) => !!ops.dedupe,
  },
  dedupe_near: {
    tab: "rows", label: "Remove near-copies", icon: "≈",
    blurb: "Rows that say the same thing in nearly the same words. Exact "
      + "removal cannot see these.",
    form: (ops) => html`
      <div class="field">
        <label for="sf_thr">How alike counts as the same</label>
        <input id="sf_thr" type="number" min="50" max="100" step="5"
               value="${Math.round((ops.dedupe_near?.threshold ?? 0.8) * 100)}">
        <div class="hint">A percentage of the five-word runs two rows share.
          80 is high enough that a pair caught here really does read as the
          same example twice; below about 65 it starts catching rows that
          merely share a format.</div>
      </div>
      <p class="muted tiny">The first of each group is kept, so the survivor is
        the one you have already scrolled past.</p>`,
    // `dlg` is null when this is added straight from a finding on the Check
    // tab rather than from its own form, so the default has to survive that.
    read: (dlg) => ({ dedupe_near: {
      threshold: Math.min(1, Math.max(0.5,
        (+(dlg && $("#sf_thr", dlg)?.value) || 80) / 100)) } }),
    describe: (ops) => `Removed rows ${Math.round(
      (ops.dedupe_near?.threshold ?? 0.8) * 100)}% alike`,
    infer: (ops) => !!ops.dedupe_near,
  },
  max_per_prompt: {
    tab: "rows", label: "Per question", icon: "≤",
    blurb: "At most N rows per distinct question. The one that bites on "
      + "generated data, where the answers vary and the questions do not.",
    form: (ops) => html`
      <div class="field"><label for="sf_n">Most rows to keep per question</label>
        <input id="sf_n" type="number" min="1" step="1" value="${ops.max_per_prompt || 2}">
        <div class="hint">Two or three keeps several good answers to one
          question without teaching the question itself.</div></div>`,
    read(dlg) {
      const n = num($("#sf_n", dlg).value);
      if (!n) throw new Error("How many per question?");
      return { max_per_prompt: n };
    },
    describe: (ops) => `Kept at most ${ops.max_per_prompt} per question`,
    infer: (ops) => !!ops.max_per_prompt,
  },
  length: {
    tab: "rows", label: "By length", icon: "↔",
    blurb: "Keep rows whose rendered text is between two lengths, in characters.",
    form: (ops) => html`
      <div class="grid grid-2">
        <div class="field"><label for="sf_min">Shortest</label>
          <input id="sf_min" type="number" min="0" value="${ops.min_chars || ""}" placeholder="any"></div>
        <div class="field"><label for="sf_max">Longest</label>
          <input id="sf_max" type="number" min="0" value="${ops.max_chars || ""}" placeholder="any"></div>
      </div>`,
    read(dlg) {
      const min_chars = num($("#sf_min", dlg).value), max_chars = num($("#sf_max", dlg).value);
      if (!min_chars && !max_chars) throw new Error("Set a shortest or a longest length.");
      return { min_chars, max_chars };
    },
    describe: (ops) => `Kept rows ${ops.min_chars ? `≥ ${ops.min_chars}` : ""}${
      ops.min_chars && ops.max_chars ? " and " : ""}${
      ops.max_chars ? `≤ ${ops.max_chars}` : ""} characters`,
    infer: (ops) => !!(ops.min_chars || ops.max_chars),
  },
  contains: {
    tab: "rows", label: "Keep matching", icon: "⊃",
    blurb: "Keep only rows whose rendered text matches a pattern.",
    form: (ops) => html`
      <div class="field"><label for="sf_re">Keep rows matching</label>
        <input id="sf_re" class="mono" value="${ops.contains || ""}" placeholder="regular expression">
        <div class="hint">Case-insensitive. Matched against the text as the
          trainer reads it, wherever the words live.</div></div>`,
    read(dlg) {
      const contains = $("#sf_re", dlg).value.trim();
      if (!contains) throw new Error("What should they match?");
      return { contains };
    },
    describe: (ops) => `Kept rows matching /${ops.contains}/`,
    infer: (ops) => !!ops.contains,
  },
  excludes: {
    tab: "rows", label: "Remove matching", icon: "⊅",
    blurb: "Remove every row whose rendered text matches a pattern.",
    form: (ops) => html`
      <div class="field"><label for="sf_re">Remove rows matching</label>
        <input id="sf_re" class="mono" value="${ops.excludes || ""}" placeholder="regular expression">
        <div class="hint">Case-insensitive, against the rendered text.</div></div>`,
    read(dlg) {
      const excludes = $("#sf_re", dlg).value.trim();
      if (!excludes) throw new Error("What should be removed?");
      return { excludes };
    },
    describe: (ops) => `Removed rows matching /${ops.excludes}/`,
    infer: (ops) => !!ops.excludes,
  },
  shuffle: {
    tab: "rows", label: "Shuffle", icon: "⤨", instant: true,
    blurb: "Random order, with a fixed seed so it is the same random order next time.",
    form: () => "", read: () => ({ shuffle: true, seed: 1234 }),
    describe: () => "Shuffled",
    infer: (ops) => !!ops.shuffle && !ops.sample,
  },
  sample: {
    tab: "rows", label: "Sample", icon: "⚄",
    blurb: "Keep at most N rows, chosen at random rather than from the front of the file.",
    form: (ops) => html`
      <div class="field"><label for="sf_n">Keep at most</label>
        <input id="sf_n" type="number" min="1" step="1" value="${ops.sample || 1000}"></div>
      <div class="field"><label for="sf_seed">Seed</label>
        <input id="sf_seed" type="number" value="${ops.seed || 1234}">
        <div class="hint">Change it to get a different sample; keep it to get
          the same one again.</div></div>`,
    read(dlg) {
      const sample = num($("#sf_n", dlg).value);
      if (!sample) throw new Error("How many rows?");
      return { sample, seed: +$("#sf_seed", dlg).value || 1234 };
    },
    describe: (ops) => `Sampled ${ops.sample} rows`,
    infer: (ops) => !!ops.sample,
  },

  // ---- the shape ---------------------------------------------------------
  to_conversations: {
    tab: "transform", label: "To conversations", icon: "🗨",
    blurb: "Rewrite every row into the one shape the trainer, the playground "
      + "and the API all read: messages, tools, meta.",
    form: (ops) => html`
      <div class="callout" style="margin:0 0 12px">
        <strong>The standard conversation format</strong>
        Whatever this data is now — ShareGPT turns, two flat columns, a
        tool-calling set with the schema in a sibling column — every row is
        rewritten into <code>messages</code>, <code>tools</code>,
        <code>meta</code>: the same format OpenAI's fine-tuning files use,
        which every published chat template already knows how to render.
      </div>
      <div class="field"><label for="sf_sys">System prompt to add</label>
        <input id="sf_sys" value="${ops.system_prompt || ""}"
               placeholder="You are a helpful assistant.">
        <div class="hint">Added only to rows that do not already have one. A
          model fine-tuned with a system prompt behaves noticeably differently
          without it, so set it deliberately.</div></div>
      <div class="field"><label for="sf_train">What a run should learn from</label>
        <select id="sf_train">
          <option value=""${!ops.train_on ? " selected" : ""}>Every assistant turn (the usual choice)</option>
          <option value="last"${ops.train_on === "last" ? " selected" : ""}>Only the final answer in each conversation</option>
          <option value="all"${ops.train_on === "all" ? " selected" : ""}>Every token, questions included</option>
        </select>
        <div class="hint">The questions and tool results are always rendered.
          This is about which tokens the model is scored on.</div></div>`,
    read: (dlg) => ({
      to_conversations: true,
      system_prompt: $("#sf_sys", dlg).value,
      train_on: $("#sf_train", dlg).value,
    }),
    describe: (ops) => "Converted to conversations"
      + (ops.system_prompt ? " with a system prompt" : "")
      + (ops.train_on === "last" ? ", learning the final answer only"
        : ops.train_on === "all" ? ", learning every token" : ""),
    infer: (ops) => !!(ops.to_conversations || ops.to_chat),
  },

  // ---- anything else -----------------------------------------------------
  custom: {
    tab: null, label: "Custom", icon: "{ }",
    blurb: "The options exactly as the controller runs them.",
    form: (ops) => html`
      <div class="field"><label for="sf_json">Options, as JSON</label>
        <textarea id="sf_json" class="mono" rows="10">${JSON.stringify(ops, null, 2)}</textarea>
        <div class="hint">Every key the transform endpoint accepts. This is the
          escape hatch; the forms cover the common cases.</div></div>`,
    read(dlg) {
      const parsed = JSON.parse($("#sf_json", dlg).value);
      if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
        throw new Error("Options have to be a JSON object.");
      }
      return parsed;
    },
    describe: (ops) => "Custom: " + Object.keys(ops).join(", "),
    infer: () => true,
  },
};

// The order the forms check an unknown options dict in. A conversion with a
// dedupe folded in is still a conversion; a shuffle with a sample is a sample.
const INFER_ORDER = ["to_conversations", "calc", "split_column", "rename",
                     "drop_columns", "keep_columns", "where", "drop_empty", "dedupe",
                     "dedupe_near",
                     "max_per_prompt", "length", "contains", "excludes",
                     "sample", "shuffle"];

// Keys an ops dict may carry without making it "custom" for a given type.
const OWN_KEYS = {
  to_conversations: ["to_conversations", "to_chat", "system_prompt", "train_on", "selectors"],
  calc: ["columns", "template", "template_column"],
  split_column: ["split_columns"],
  rename: ["rename"],
  drop_columns: ["drop_columns"],
  keep_columns: ["keep_columns"],
  where: ["where"],
  drop_empty: ["drop_empty"],
  dedupe: ["dedupe"],
  dedupe_near: ["dedupe_near"],
  max_per_prompt: ["max_per_prompt"],
  length: ["min_chars", "max_chars"],
  contains: ["contains"],
  excludes: ["excludes"],
  sample: ["sample", "seed", "shuffle"],
  shuffle: ["shuffle", "seed"],
};

/** The type whose form can edit these ops without losing anything. */
export function inferType(ops) {
  const keys = Object.keys(ops || {}).filter((k) => {
    const v = ops[k];
    return !(v === false || v === null || v === "" || v === 0
      || (Array.isArray(v) && !v.length)
      || (v && typeof v === "object" && !Array.isArray(v) && !Object.keys(v).length));
  });
  for (const type of INFER_ORDER) {
    if (!STEPS[type].infer(ops)) continue;
    if (keys.every((k) => OWN_KEYS[type].includes(k) || k === "notes")) return type;
    return "custom";
  }
  return "custom";
}

/** A step from a bare options dict, e.g. out of a saved recipe. */
export const stepFrom = (ops) => ({ type: inferType(ops), ops });

/** One sentence for the applied-steps list. */
export function describe(step) {
  const def = STEPS[step.type] || STEPS.custom;
  try { return def.describe(step.ops) || def.label; } catch { return def.label; }
}

/** The ribbon's idea of the catalogue: which buttons on which tab. */
export const TABS = [
  { key: "home", label: "Home" },
  { key: "transform", label: "Transform" },
  { key: "columns", label: "Add Column" },
  { key: "rows", label: "Rows" },
  { key: "view", label: "View" },
  // The quality report, which was a modal: read it, close it, and every number
  // in it was gone.
  { key: "check", label: "Check" },
  // Not a step: what the trainer will read out of these rows, and whether it
  // can. It lived only inside the wizard, three screens from the editor where
  // the mapping it depends on is decided.
  { key: "train", label: "Training" },
];

export const stepsOnTab = (tab) => Object.entries(STEPS)
  .filter(([, def]) => def.tab === tab).map(([key, def]) => ({ key, ...def }));
