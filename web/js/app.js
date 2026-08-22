import { api, events, handleUnauthorized, NotSignedIn } from "./api.js";
import { $, $$, toast } from "./util.js";
import { initGate, showGate, hideGate } from "./views/gate.js";

import { dashboardView } from "./views/dashboard.js";
import { wizardView } from "./views/wizard.js";
import { jobsView } from "./views/jobs.js";
import { jobView } from "./views/job.js";
import { playView } from "./views/play.js";
import { runnersView } from "./views/runners.js";
import { settingsView } from "./views/settings.js";
import { accountView } from "./views/account.js";
import { usersView } from "./views/users.js";
import { dataView } from "./views/data.js";
import { datasetView } from "./views/dataset.js";
import { generateView } from "./views/generate.js";
import { evalsView } from "./views/evals.js";
import { evalView } from "./views/evalview.js";
import { compareView } from "./views/compare.js";

const routes = [
  [/^\/$/,             dashboardView, "dashboard"],
  [/^\/new$/,          wizardView,    "new"],
  [/^\/jobs$/,         jobsView,      "jobs"],
  [/^\/jobs\/(.+)$/,   jobView,       "jobs"],
  [/^\/play$/,         playView,      "play"],
  [/^\/play\/(.+)$/,   playView,      "play"],
  [/^\/runners$/,      runnersView,   "runners"],
  [/^\/settings$/,     settingsView,  "settings"],
  [/^\/account$/,      accountView,   ""],
  [/^\/users$/,        usersView,     "settings"],
  [/^\/data$/,         dataView,      "data"],
  [/^\/data\/(.+)$/,   datasetView,   "data"],
  [/^\/generate$/,     generateView,  "data"],
  [/^\/evals$/,        evalsView,     "evals"],
  [/^\/evals\/(.+)$/,  evalView,      "evals"],
  [/^\/compare$/,      compareView,   "evals"],
];

// Who is signed in. Views read it rather than each fetching /api/me.
export const session = { user: null };

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
      // A session that expired mid-render is not an error to display; the
      // gate is already going up in front of it.
      if (err instanceof NotSignedIn) return;
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

// ---- identity -----------------------------------------------------------

function paintWho(user) {
  const box = $("#whoami");
  if (!box) return;
  box.hidden = !user;
  if (!user) return;
  $("#whoName").textContent = user.display_name || user.username;
  $("#whoRole").textContent = user.role === "admin" ? "administrator" : "member";
  $("#whoAvatar").textContent =
    (user.display_name || user.username || "?").trim()[0].toUpperCase();
  // The user administration link only exists for people who can use it.
  const nav = $(".nav");
  const existing = $("#navUsers");
  if (user.role === "admin" && !existing) {
    const a = document.createElement("a");
    a.id = "navUsers";
    a.href = "#/users";
    a.dataset.nav = "settings";
    a.innerHTML = `<span class="ico">◍</span><span class="lbl">People</span>`;
    nav.appendChild(a);
  } else if (user.role !== "admin" && existing) {
    existing.remove();
  }
}

let fleetTimer = null;

async function start() {
  let state;
  try {
    state = await api.authState();
  } catch {
    setFleet("dot-err", "Controller unreachable");
    return;
  }
  if (!state.authenticated || state.setup_required || state.must_change) {
    clearInterval(fleetTimer);
    events.stop?.();
    showGate(state);
    return;
  }
  hideGate();
  session.user = state.user;
  paintWho(state.user);
  events.start?.();
  refreshFleet();
  clearInterval(fleetTimer);
  fleetTimer = setInterval(refreshFleet, 20000);
  await render();
}

initGate(start);
// Any 401 from anywhere puts the gate back, whichever view was on screen.
handleUnauthorized((body) => {
  session.user = null;
  showGate({ authenticated: false, setup_required: !!body.setup_required,
             must_change: !!body.must_change });
});

start();
