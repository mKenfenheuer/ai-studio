/**
 * Just enough Markdown to show a model card the way the Hub will.
 *
 * A card is written to be read on Hugging Face, and until you see it rendered
 * you are proof-reading pipe characters. This renders the subset a card
 * actually uses -- headings, paragraphs, bold and italic, links, inline code,
 * fenced code, lists, tables, rules, block quotes -- and nothing else.
 *
 * Everything is escaped first and the transforms only ever *add* markup, so
 * there is no path from a card's text to executable HTML. That matters more
 * here than completeness: a card can arrive from a Hub repository somebody
 * else wrote, and a preview that runs it would be a hole in the studio.
 *
 * The front-matter helpers deliberately do not parse YAML. A card's front
 * matter carries `model-index`, which is nested, ordered and generated -- a
 * round trip through a naive parser would quietly destroy it. Instead the
 * editor reads the handful of flat fields it offers and rewrites exactly
 * those lines, leaving every other byte alone.
 */
import { esc } from "./util.js";

const FLAT = ["license", "language", "pipeline_tag", "library_name",
              "base_model", "base_model_relation"];
const LISTS = ["tags", "datasets", "language"];

/** `{meta, body}` -- the YAML block at the top, and everything after it. */
export function splitFrontMatter(text) {
  const src = String(text || "");
  if (!src.startsWith("---")) return { meta: "", body: src };
  const end = src.indexOf("\n---", 3);
  if (end < 0) return { meta: "", body: src };
  const close = src.indexOf("\n", end + 1);
  return {
    meta: src.slice(src.indexOf("\n") + 1, end),
    body: close < 0 ? "" : src.slice(close + 1).replace(/^\n+/, ""),
  };
}

/** The flat fields the editor offers, read out of the front matter. */
export function readFields(text) {
  const { meta } = splitFrontMatter(text);
  const out = {};
  const lines = meta.split("\n");
  lines.forEach((line, i) => {
    const m = /^([a-z_]+):\s*(.*)$/.exec(line);
    if (!m) return;
    const [, key, value] = m;
    if (value.trim()) {
      out[key] = value.trim().replace(/^["']|["']$/g, "");
    } else if (LISTS.includes(key)) {
      // A list written as `key:` then `  - item` lines under it.
      const items = [];
      for (let j = i + 1; j < lines.length; j++) {
        const item = /^\s+-\s*(.+)$/.exec(lines[j]);
        if (!item) break;
        items.push(item[1].trim().replace(/^["']|["']$/g, ""));
      }
      out[key] = items.join(", ");
    }
  });
  return out;
}

/** The card with one field replaced, added or removed. Nothing else moves. */
export function writeField(text, key, value) {
  const src = String(text || "");
  const { meta, body } = splitFrontMatter(src);
  const asList = LISTS.includes(key);
  const items = asList
    ? String(value || "").split(",").map((s) => s.trim()).filter(Boolean)
    : [];
  const block = !value || (asList && !items.length) ? []
    : asList ? [`${key}:`, ...items.map((i) => `  - ${i}`)]
             : [`${key}: ${value}`];

  const lines = meta ? meta.split("\n") : [];
  const kept = [];
  for (let i = 0; i < lines.length; i++) {
    if (new RegExp(`^${key}:`).test(lines[i])) {
      // Skip the key and, if it opened a list, the items under it.
      if (!lines[i].slice(key.length + 1).trim()) {
        while (i + 1 < lines.length && /^\s+-\s/.test(lines[i + 1])) i++;
      }
      continue;
    }
    kept.push(lines[i]);
  }
  const merged = [...kept.filter((l) => l.trim() !== ""), ...block];
  if (!merged.length) return body;
  return `---\n${merged.join("\n")}\n---\n\n${body}`;
}

const INLINE = (s) => s
  .replace(/`([^`]+)`/g, (_, code) => `<code>${code}</code>`)
  .replace(/!\[([^\]]*)\]\(([^)\s]+)\)/g, (_, alt, src) =>
    `<img alt="${alt}" src="${safeUrl(src)}">`)
  .replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (_, label, href) =>
    `<a href="${safeUrl(href)}" target="_blank" rel="noopener">${label}</a>`)
  .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
  .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>");

// Anything that is not plainly a link to somewhere is not linked at all.
const safeUrl = (u) => (/^(https?:|mailto:|#|\/)/i.test(u) ? u : "#");

/** Markdown as HTML. Escaped first; the transforms only add markup. */
export function renderMarkdown(text) {
  const lines = esc(String(text || "")).split("\n");
  const out = [];
  let para = [];
  let list = null;
  let table = null;
  let fence = null;

  const flushPara = () => {
    if (para.length) out.push(`<p>${INLINE(para.join(" "))}</p>`);
    para = [];
  };
  const flushList = () => {
    if (list) out.push(`<${list.tag}>${list.items.join("")}</${list.tag}>`);
    list = null;
  };
  const flushTable = () => {
    if (table && table.rows.length) {
      const head = table.rows[0].map((c) => `<th>${INLINE(c)}</th>`).join("");
      const body = table.rows.slice(1).map((r) =>
        `<tr>${r.map((c) => `<td>${INLINE(c)}</td>`).join("")}</tr>`).join("");
      out.push(`<div class="table-wrap"><table><thead><tr>${head}</tr></thead>`
               + `<tbody>${body}</tbody></table></div>`);
    }
    table = null;
  };
  const flushAll = () => { flushPara(); flushList(); flushTable(); };

  for (const line of lines) {
    const fenced = /^```(\w*)\s*$/.exec(line);
    if (fence !== null) {
      if (fenced) {
        out.push(`<pre class="md-code"><code>${fence.join("\n")}</code></pre>`);
        fence = null;
      } else { fence.push(line); }
      continue;
    }
    if (fenced) { flushAll(); fence = []; continue; }

    if (!line.trim()) { flushAll(); continue; }

    const heading = /^(#{1,6})\s+(.*)$/.exec(line);
    if (heading) {
      flushAll();
      const level = Math.min(heading[1].length + 1, 6);   // # is the page's h1
      out.push(`<h${level}>${INLINE(heading[2])}</h${level}>`);
      continue;
    }
    if (/^(-{3,}|\*{3,}|_{3,})$/.test(line.trim())) { flushAll(); out.push("<hr>"); continue; }

    if (/^\s*\|.*\|\s*$/.test(line)) {
      flushPara(); flushList();
      const cells = line.trim().replace(/^\||\|$/g, "").split("|")
        .map((c) => c.trim());
      // The |---|---| row under the header says nothing worth drawing.
      if (cells.every((c) => /^:?-{2,}:?$/.test(c) || c === "")) continue;
      table = table || { rows: [] };
      table.rows.push(cells);
      continue;
    }
    flushTable();

    const bullet = /^\s*[-*+]\s+(.*)$/.exec(line);
    const number = /^\s*\d+[.)]\s+(.*)$/.exec(line);
    if (bullet || number) {
      flushPara();
      const tag = bullet ? "ul" : "ol";
      if (!list || list.tag !== tag) { flushList(); list = { tag, items: [] }; }
      list.items.push(`<li>${INLINE((bullet || number)[1])}</li>`);
      continue;
    }
    flushList();

    const quote = /^>\s?(.*)$/.exec(line);
    if (quote) {
      flushPara();
      out.push(`<blockquote>${INLINE(quote[1])}</blockquote>`);
      continue;
    }
    para.push(line.trim());
  }
  if (fence !== null) out.push(`<pre class="md-code"><code>${fence.join("\n")}</code></pre>`);
  flushAll();
  return out.join("\n");
}
