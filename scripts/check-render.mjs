// Draw every view in a real browser and report the ones that do not.
//
//   node scripts/check-render.mjs            # every view
//   node scripts/check-render.mjs --shot data --width 390   # a screenshot
//
// `check-web.mjs` answers "does this file parse". It cannot answer the
// question that actually costs an afternoon, which is "does this page still
// draw" — a view that throws halfway through renders as a blank panel with one
// line in the console, and there was no way to find that out except opening
// eighteen routes by hand.
//
// So: a page that stubs `fetch` with canned answers, imports each view, mounts
// it, clicks every tab on its ribbon, and reports what threw, what drew almost
// nothing, and which tabs did not switch. It needs Chrome, which is on the
// machine of anybody who is going to look at this UI anyway, and it needs no
// npm packages — the whole point of a zero-build front end is that the only
// toolchain is the one you already have.
//
// Exit code is 1 if anything failed, so this can go in CI as it stands.

import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import { dirname, join, extname, normalize } from "node:path";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, "..");
const PORT = 8791;

const TYPES = {
  ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript",
  ".css": "text/css", ".json": "application/json", ".svg": "image/svg+xml",
};

// Where Chrome tends to be. The first one that exists wins; CHROME=... beats
// all of them, which is what a CI image will use.
const CANDIDATES = [
  process.env.CHROME,
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  "/Applications/Chromium.app/Contents/MacOS/Chromium",
  "/usr/bin/google-chrome",
  "/usr/bin/chromium",
  "/usr/bin/chromium-browser",
  "/snap/bin/chromium",
].filter(Boolean);

async function findChrome() {
  for (const path of CANDIDATES) {
    try { await readFile(path); return path; } catch { /* not this one */ }
    // A binary may be unreadable but executable; stat rather than read.
    try {
      const { stat } = await import("node:fs/promises");
      await stat(path);
      return path;
    } catch { /* keep looking */ }
  }
  return null;
}

/** Serve the web tree at /static and this directory's harness at the root. */
function serve() {
  const server = createServer(async (req, res) => {
    const url = new URL(req.url, "http://x");
    let file;
    if (url.pathname.startsWith("/static/")) {
      file = join(ROOT, "web", normalize(url.pathname.slice("/static".length)));
    } else if (url.pathname === "/" || url.pathname === "/check-render.html") {
      file = join(HERE, "check-render.html");
    } else {
      res.writeHead(404).end("no");
      return;
    }
    // Nothing outside the two directories above, however the path is spelled.
    if (!file.startsWith(join(ROOT, "web")) && !file.startsWith(HERE)) {
      res.writeHead(403).end("no");
      return;
    }
    try {
      const body = await readFile(file);
      res.writeHead(200, { "content-type": TYPES[extname(file)] || "text/plain" });
      res.end(body);
    } catch {
      res.writeHead(404).end("no");
    }
  });
  return new Promise((ok) => server.listen(PORT, () => ok(server)));
}

function run(bin, args) {
  return new Promise((ok) => {
    const p = spawn(bin, args);
    let out = "";
    p.stdout.on("data", (d) => { out += d; });
    p.stderr.on("data", () => { /* Chrome is noisy about GPUs it cannot have */ });
    p.on("close", () => ok(out));
  });
}

const args = process.argv.slice(2);
const shot = args.includes("--shot") ? args[args.indexOf("--shot") + 1] : null;
const width = args.includes("--width") ? args[args.indexOf("--width") + 1] : "1440";
const height = args.includes("--height") ? args[args.indexOf("--height") + 1] : "950";

const chrome = await findChrome();
if (!chrome) {
  console.error("No Chrome or Chromium found. Set CHROME=/path/to/chrome.");
  console.error("Looked in:\n  " + CANDIDATES.join("\n  "));
  process.exit(2);
}

const server = await serve();

if (shot) {
  const out = join(process.cwd(), `${shot}-${width}.png`);
  await run(chrome, [
    "--headless", "--disable-gpu", "--no-sandbox", "--hide-scrollbars",
    `--window-size=${width},${height}`, "--virtual-time-budget=8000",
    `--screenshot=${out}`,
    `http://localhost:${PORT}/check-render.html?only=${encodeURIComponent(shot)}`,
  ]);
  console.log(out);
  server.close();
  process.exit(0);
}

const dom = await run(chrome, [
  "--headless", "--disable-gpu", "--no-sandbox",
  "--virtual-time-budget=20000", "--dump-dom",
  `http://localhost:${PORT}/check-render.html`,
]);
server.close();

const unescape = (s) => s
  .replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&quot;/g, '"')
  .replace(/&#39;/g, "'").replace(/&amp;/g, "&");

const title = /<title>(.*?)<\/title>/s.exec(dom)?.[1]?.trim() || "";
const results = /<pre id="results"[^>]*>(.*?)<\/pre>/s.exec(dom)?.[1] || "";

if (!results) {
  console.error("The harness did not report. Chrome may have failed to load "
    + "the page, or a module failed to import at all.");
  process.exit(1);
}
console.log(unescape(results));
console.log("");
if (title.startsWith("ALL")) {
  console.log("Every view renders, and every tab switches.");
  process.exit(0);
}
console.error(title);
process.exit(1);
