// Parse every browser script the way the browser will.
//
//   docker run --rm -v "$PWD":/w -w /w node:22-alpine node scripts/check-web.mjs
//
// There is no build step here and no bundler, so nothing looks at this code
// between writing it and a browser running it. A syntax error therefore ships,
// and shows up as a page that renders nothing with one line in the console.
//
// WHY NOT `node --check`: it parses as CommonJS. These files are ES modules,
// loaded with <script type="module">, and the two disagree about real errors.
// A string literal broken across lines by a stray newline -- which is exactly
// what a shell heredoc does to \n if you are not careful, and has happened
// here more than once -- is accepted by `node --check` and rejected by the
// browser. A check that passes what the browser refuses is worse than no
// check, because it is trusted.
//
// vm.SourceTextModule parses as ESM without executing, which is the whole
// question: is this file loadable. Needs --experimental-vm-modules.
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join } from "node:path";
import vm from "node:vm";

const ROOT = "web/js";

function walk(dir) {
  return readdirSync(dir).flatMap((name) => {
    const path = join(dir, name);
    return statSync(path).isDirectory() ? walk(path)
      : path.endsWith(".js") ? [path] : [];
  });
}

let failed = 0;
for (const file of walk(ROOT).sort()) {
  const source = readFileSync(file, "utf8");
  try {
    // eslint-disable-next-line no-new
    new vm.SourceTextModule(source, { identifier: file });
    console.log(`  ok    ${file}`);
  } catch (e) {
    failed++;
    console.log(`  FAIL  ${file}`);
    console.log(`          ${e.message}`);
    // The message alone rarely says where. Find the first line that opens a
    // quote it does not close, which is what this failure almost always is.
    source.split("\n").forEach((line, i) => {
      const bare = line.split("//")[0];
      const doubles = (bare.match(/(?<!\\)"/g) || []).length;
      const singles = (bare.match(/(?<!\\)'/g) || []).length;
      if (doubles % 2 || singles % 2) {
        console.log(`          line ${i + 1}: ${line.trim().slice(0, 90)}`);
      }
    });
  }
}

if (failed) {
  console.log(`\n${failed} file(s) the browser would refuse to load.`);
  process.exit(1);
}
console.log("\nEvery browser script parses as a module.");
