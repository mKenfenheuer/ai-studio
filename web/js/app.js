import { api, events, handleUnauthorized, NotSignedIn } from "./api.js";
import { $, $$, toast, takeSsoError, skeleton, resetDelegated } from "./util.js";
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
import { ssoView } from "./views/sso.js";
import { dataView } from "./views/data.js";
import { datasetView } from "./views/dataset.js";
import { generateView } from "./views/generate.js";
import { evalsView } from "./views/evals.js";
import { evalView } from "./views/evalview.js";
import { compareView } from "./views/compare.js";
import { sweepView } from "./views/sweep.js";

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
  [/^\/sso$/,          ssoView,       "settings"],
  [/^\/data$/,         dataView,      "data"],
  [/^\/data\/(.+)$/,   datasetView,   "data"],
  [/^\/generate$/,     generateView,  "data"],
  [/^\/evals$/,        evalsView,     "evals"],
  [/^\/evals\/(.+)$/,  evalView,      "evals"],
  [/^\/compare$/,      compareView,   "evals"],
  [/^\/sweeps\/(.+)$/, sweepView,     "jobs"],
];

// Who is signed in. Views read it rather than each fetching /api/me.
export const session = { user: null };

let teardown = null;

// Which render is the current one. Views are async -- most fetch something
// before they draw -- so two can be in flight at once when a route changes
// while the first is still loading. Unguarded, whichever finishes LAST wins,
// which is often the one you navigated away from, and its teardown is never
// called so its event subscription keeps redrawing the page you did navigate
// to. Both were real: the wizard would intermittently come up as the previous
// page, more often the slower the machine.
let renderSeq = 0;

async function render() {
  const mine = ++renderSeq;
  const path = (location.hash.slice(1) || "/").split("?")[0];
  const main = $("#main");

  if (typeof teardown === "function") { try { teardown(); } catch { /* ignore */ } }
  teardown = null;
  // Delegated listeners live on #main, which every route shares. They are kept
  // one-per-selector and re-pointed at each redraw, which is right within a
  // page and wrong between two: a control the next page draws would otherwise
  // run the previous page's handler, closed over the previous page's state.
  // Cleared here so that can never happen again -- see resetDelegated.
  resetDelegated(main);

  for (const [re, view, nav] of routes) {
    const m = path.match(re);
    if (!m) continue;
    $$(".nav a").forEach((a) =>
      a.classList.toggle("active", a.dataset.nav === nav));
    // Shaped like the page that is coming rather than a word in the corner,
    // so the layout settles once instead of twice.
    main.innerHTML = skeleton({ cards: 3, rows: 2 });
    try {
      const stop = await view(main, m.slice(1));
      if (mine !== renderSeq) {
        // Somebody navigated while this was loading. Whatever it built is
        // already off screen; unsubscribe it rather than leaving it running.
        if (typeof stop === "function") { try { stop(); } catch { /* gone */ } }
        return;
      }
      teardown = stop;
    } catch (err) {
      if (mine !== renderSeq) return;
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
    if (s.runners_online === 0 && s.runners_total > 0) {
      // Known machines, none answering. Almost always a restart in progress,
      // and "no machines connected" reads as "you have not set one up" --
      // which sends people off to fix something that is not broken.
      setFleet("dot-idle", "Reconnecting\u2026");
    } else if (s.runners_online === 0) {
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
  if (msg.type === "job_finished") announce(msg);
});

// ---- "your run has finished" --------------------------------------------
// Only when this tab is not the one being looked at. A notification for
// something already visible on screen is noise, and the surest way to get
// somebody to turn notifications off is to send them one they did not need.
function announce(msg) {
  const finished = msg.status === "succeeded";
  const s = msg.summary || {};
  const detail = msg.status === "failed"
    ? String(msg.error || "It failed.").slice(0, 140)
    : [s.best_val_loss != null ? `held-out loss ${s.best_val_loss.toFixed(4)}` : null,
       s.steps ? `${s.steps.toLocaleString()} steps` : null,
       s.early_stopped ? "stopped early, at its best point" : null,
      ].filter(Boolean).join(" · ");

  toast(`${msg.name}: ${finished ? "finished" : msg.status}`,
        msg.status === "failed" ? "err" : "ok");

  if (document.visibilityState === "visible") return;
  if (!("Notification" in window) || Notification.permission !== "granted") return;
  try {
    const n = new Notification(
      `${finished ? "✓" : msg.status === "failed" ? "✕" : "■"} ${msg.name}`,
      { body: detail || `The run ${msg.status}.`, tag: msg.job_id });
    n.onclick = () => {
      window.focus();
      location.hash = `#/jobs/${msg.job_id}`;
      n.close();
    };
  } catch { /* the browser may refuse; the toast already happened */ }
}

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
  // A sign-in that failed while a session was already open has no login
  // screen to be shown on, and used to vanish silently -- the browser landed
  // on the dashboard as if nothing had been attempted. It is the same message
  // either way; only where it goes differs.
  const ssoError = takeSsoError();
  if (ssoError) toast(ssoError, "err");
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
