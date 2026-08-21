import { api, events } from "./api.js";
import { $, $$, toast } from "./util.js";

import { dashboardView } from "./views/dashboard.js";
import { wizardView } from "./views/wizard.js";
import { jobsView } from "./views/jobs.js";
import { jobView } from "./views/job.js";
import { playView } from "./views/play.js";
import { runnersView } from "./views/runners.js";
import { settingsView } from "./views/settings.js";

const routes = [
  [/^\/$/,             dashboardView, "dashboard"],
  [/^\/new$/,          wizardView,    "new"],
  [/^\/jobs$/,         jobsView,      "jobs"],
  [/^\/jobs\/(.+)$/,   jobView,       "jobs"],
  [/^\/play$/,         playView,      "play"],
  [/^\/play\/(.+)$/,   playView,      "play"],
  [/^\/runners$/,      runnersView,   "runners"],
  [/^\/settings$/,     settingsView,  "settings"],
];

let teardown = null;

async function render() {
  const path = (location.hash.slice(1) || "/").split("?")[0];
  const main = $("#main");

  if (typeof teardown === "function") { try { teardown(); } catch { /* ignore */ } }
  teardown = null;

  for (const [re, view, nav] of routes) {
    const m = path.match(re);
    if (!m) continue;
    $$(".nav a").forEach((a) =>
      a.classList.toggle("active", a.dataset.nav === nav));
    main.innerHTML = `<div class="loading">Loading…</div>`;
    try {
      teardown = await view(main, m.slice(1));
    } catch (err) {
      console.error(err);
      main.innerHTML =
        `<div class="callout callout-err"><strong>Something went wrong</strong>${
          String(err.message || err)}</div>`;
    }
    window.scrollTo(0, 0);
    return;
  }
  main.innerHTML = `<div class="empty"><div class="big">🧭</div>
    <h2>Page not found</h2><p><a href="#/">Back to the dashboard</a></p></div>`;
}

// ---- fleet indicator in the sidebar -------------------------------------
// The indicator appears twice — sidebar on desktop, top bar on mobile — so
// both instances are updated rather than one being queried by id.
function setFleet(dotClass, text) {
  $$("[data-fleet-dot]").forEach((d) => { d.className = `dot ${dotClass}`; });
  $$("[data-fleet-text]").forEach((t) => { t.textContent = text; });
}

async function refreshFleet() {
  try {
    const s = await api.status();
    if (s.runners_online === 0) {
      setFleet("dot-err", "No machines connected");
    } else {
      setFleet(s.jobs_running ? "dot-busy" : "dot-ok",
        `${s.runners_online} machine${s.runners_online > 1 ? "s" : ""}` +
        (s.jobs_running ? ` · ${s.jobs_running} training` : " ready"));
    }
  } catch {
    setFleet("dot-err", "Controller unreachable");
  }
}

events.subscribe((msg) => {
  if (["runners_changed", "jobs_changed", "_connected"].includes(msg.type)) refreshFleet();
});

// ---- help drawer --------------------------------------------------------
$$("#helpToggle, #helpToggleMobile").forEach((b) =>
  b.addEventListener("click", () => { $("#helpDrawer").hidden = false; }));
$("#helpClose").addEventListener("click", () => { $("#helpDrawer").hidden = true; });
$("#helpDrawer").addEventListener("click", (e) => {
  if (e.target.id === "helpDrawer") $("#helpDrawer").hidden = true;
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") $("#helpDrawer").hidden = true;
});

window.addEventListener("hashchange", render);
window.addEventListener("error", (e) => toast(e.message, "err"));

refreshFleet();
setInterval(refreshFleet, 20000);
render();
