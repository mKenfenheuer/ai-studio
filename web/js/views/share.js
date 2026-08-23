/**
 * Sharing a run, a dataset or a prompt set with somebody.
 *
 * This used to be a `<select>` listing every account on the studio. That works
 * for four colleagues and stops working the moment a company directory is
 * connected: a dropdown of four thousand names is not a way to find anyone,
 * and it also means the page cannot be drawn until the whole list has arrived.
 *
 * So it is a modal with a search box now. You type, the server ranks, and the
 * matches appear — including people who have never opened the studio, because
 * they were imported from the directory. Sharing with someone who has not been
 * here yet is the normal case rather than the odd one: it is *why* you share.
 *
 * The search is deliberately server-side. Filtering a list in the browser
 * requires having the list, which is the thing that does not scale, and the
 * ranking ("ma" should find Maria before Norman Hallmark) needs the whole set
 * to rank against.
 *
 * One component for all three kinds, because the question is identical in all
 * three places and two near-identical panels drift apart within a month.
 */
import { api } from "../api.js";
import { html, raw, $, on, toast, modal, debounce,
         avatar } from "../util.js";

const KIND_WORD = { job: "run", dataset: "dataset", eval: "prompt set" };

/** The Share control, for the row of actions at the top of a page.
 *
 *  An action, next to Stop and Download, rather than a panel further down.
 *  Sharing is something you decide to do and then look for; a card in the
 *  middle of a page is where you find things you were already reading.
 *
 *  It carries its own state, so the row answers "who can see this" without
 *  being opened: a count when it is shared, nothing when it is not, and for
 *  somebody it was shared *with*, who gave it to them and what they may do.
 */
export function shareButton(kind, resource) {
  const shares = resource.shares || [];
  const people = shares.filter((sh) => sh.subject_type === "user");
  const everyone = shares.some((sh) => sh.subject_type === "everyone");

  if (!resource.mine) {
    if (!shares.length && resource.access !== "view" && resource.access !== "edit") {
      return "";
    }
    const from = resource.owner_name || resource.owner?.display_name;
    return html`<span class="badge" title="${from
        ? "Shared with you by " + from : "Shared with you"}">
      shared with you${resource.access === "edit" ? " · can edit" : ""}</span>`;
  }

  const count = people.length;
  const label = everyone && !count ? "Everyone"
    : count ? String(count) : "";
  const title = everyone && count
      ? `Everyone here can view it, and ${count} added by name`
    : everyone ? "Everyone with an account here can view it"
    : count ? `Shared with ${count} ${count === 1 ? "person" : "people"}`
    : "Private to you. Nobody else can see it.";

  return html`
    <button class="btn-sm ${count || everyone ? "btn-shared" : ""}"
            id="shareOpen" title="${title}">
      <span aria-hidden="true">\u25CD</span> Share${label ? " \u00b7 " + label : ""}
    </button>`;
}

export function wireShareBox(mount, kind, resource, refresh) {
  on(mount, "click", "#shareOpen", () => openShare(kind, resource, refresh));
}

// ---------------------------------------------------------------------------
// The modal
// ---------------------------------------------------------------------------

export function openShare(kind, resource, refresh) {
  const word = KIND_WORD[kind] || kind;
  const dlg = modal({
    title: `Share this ${word}`,
    width: 620,
    onClose: () => refresh?.(),
  });

  // Held here rather than re-read from `resource`, because the modal stays
  // open across several changes and the page behind it is only refreshed once
  // it closes.
  let state = { shares: resource.shares || [], owner: resource.owner || null };
  const staged = new Map();

  const paint = () => {
    $(".modal-body", dlg).innerHTML = bodyHtml(resource, state, staged, word);
    wire();
  };

  const reload = async () => {
    try {
      const fresh = await api.shares(kind, resource.id);
      state = { shares: fresh.shares || [], owner: fresh.owner };
      resource.shares = state.shares;
    } catch (e) { toast(e.message, "err"); }
    paint();
  };

  // ------------------------------------------------------------- searching
  let results = [];
  let cursor = -1;

  const listbox = () => $("#pickList", dlg);
  const input = () => $("#pickInput", dlg);

  const drawResults = () => {
    const box = listbox();
    if (!box) return;
    if (!results.length) {
      box.innerHTML = `<li class="pick-empty" role="presentation">${
        input().value.trim() ? "Nobody here matches that."
          : "Start typing a name."}</li>`;
      box.hidden = false;
      input().setAttribute("aria-expanded", "true");
      return;
    }
    box.innerHTML = results.map((p, i) => html`
      <li role="option" id="pick-${i}" data-pick="${p.id}"
          class="pick-item ${i === cursor ? "on" : ""}"
          aria-selected="${i === cursor}">
        ${avatar(p, 28)}
        <span class="pick-text">
          <span class="pick-name">${p.display_name}
            ${raw(p.pending ? `<span class="badge badge-soft">not signed in yet</span>` : "")}
          </span>
          <span class="pick-sub">${[p.email || p.username,
              p.job_title, p.department].filter(Boolean).join(" · ")}</span>
        </span>
      </li>`).join("");
    box.hidden = false;
    input().setAttribute("aria-expanded", "true");
    input().setAttribute("aria-activedescendant",
                         cursor >= 0 ? `pick-${cursor}` : "");
  };

  const closeResults = () => {
    const box = listbox();
    if (box) { box.hidden = true; box.innerHTML = ""; }
    input()?.setAttribute("aria-expanded", "false");
    cursor = -1;
  };

  // Only the most recent search may paint. Without the sequence number a slow
  // answer for "ma" can land after a fast one for "maria" and replace it,
  // which reads as the list ignoring what you typed.
  let seq = 0;
  const search = debounce(async (term) => {
    const mine = ++seq;
    const exclude = [resource.owner_id || resource.owner?.id,
                     ...state.shares.filter((s) => s.subject_type === "user")
                       .map((s) => s.subject_id),
                     ...staged.keys()].filter(Boolean);
    try {
      const found = await api.searchUsers(term, exclude);
      if (mine !== seq) return;
      results = found;
      cursor = found.length ? 0 : -1;
      drawResults();
    } catch (e) {
      if (mine === seq) toast(e.message, "err");
    }
  }, 180);

  const choose = (id) => {
    const person = results.find((p) => p.id === id);
    if (!person) return;
    staged.set(id, person);
    results = [];
    paint();
    input()?.focus();
  };

  // ------------------------------------------------------------------ wiring
  //
  // Split in two on purpose. Everything delegated is attached to the dialog
  // once, because the dialog outlives every repaint -- binding those inside
  // `wire()` would add a second listener on the first repaint and a third on
  // the next, and one click would then remove three people. Only listeners on
  // nodes that are themselves rebuilt belong in `wire()`.
  function wireOnce() {
    // `mousedown`, not `click`: the input loses focus first on a click, and
    // anything that hides the list on blur would remove the row mid-press.
    on(dlg, "mousedown", "[data-pick]", (e, t) => {
      e.preventDefault();
      choose(t.dataset.pick);
    });

    on(dlg, "click", "[data-unstage]", (_e, t) => {
      staged.delete(t.dataset.unstage);
      paint();
    });

    on(dlg, "click", "#shareApply", async (_e, btn) => {
      const level = $("#shareLevel", dlg).value;
      btn.disabled = true;
      const failed = [];
      for (const id of staged.keys()) {
        try {
          await api.addShare(kind, resource.id,
                             { subject_type: "user", subject_id: id, level });
        } catch (e) { failed.push(e.message); }
      }
      const n = staged.size - failed.length;
      staged.clear();
      if (failed.length) toast(failed[0], "err");
      else if (n) toast(`Shared with ${n} ${n === 1 ? "person" : "people"}.`, "ok");
      await reload();
    });

    on(dlg, "change", "[data-level]", async (_e, t) => {
      const id = t.dataset.level;
      if (t.value === "remove") {
        try { await api.removeShare(kind, resource.id, "user", id); }
        catch (e) { toast(e.message, "err"); }
        return reload();
      }
      if (t.value === "owner") {
        const person = state.shares.find((sh) => sh.subject_id === id);
        const who = person?.display_name || person?.username || "them";
        if (!confirm(`Make ${who} the owner of this ${word}?\n\n`
                     + `You keep edit access, but only they will be able to `
                     + `delete it or change who it is shared with.`)) {
          return reload();
        }
        try {
          await api.transfer(kind, resource.id, id);
          toast(`${who} owns it now.`, "ok");
          // Ownership decides what the whole page may do, so this one wants
          // the page behind the modal rebuilt rather than patched.
          dlg.close();
          return undefined;
        } catch (e) { toast(e.message, "err"); }
        return reload();
      }
      try {
        await api.addShare(kind, resource.id,
                           { subject_type: "user", subject_id: id, level: t.value });
      } catch (e) { toast(e.message, "err"); }
      return reload();
    });

    on(dlg, "change", "#shareEveryone", async (_e, t) => {
      try {
        if (t.checked) {
          await api.addShare(kind, resource.id,
                             { subject_type: "everyone", level: "view" });
        } else {
          await api.removeShare(kind, resource.id, "everyone");
        }
      } catch (e) { toast(e.message, "err"); }
      return reload();
    });
  }

  function wire() {
    const box = input();
    if (!box) return;
    box.addEventListener("input", () => {
      const term = box.value.trim();
      if (!term) { results = []; closeResults(); return; }
      search(term);
    });
    box.addEventListener("focus", () => { if (results.length) drawResults(); });
    box.addEventListener("keydown", (e) => {
      if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        e.preventDefault();
        if (!results.length) return;
        cursor = (cursor + (e.key === "ArrowDown" ? 1 : -1) + results.length)
                 % results.length;
        drawResults();
        $(`#pick-${cursor}`, dlg)?.scrollIntoView({ block: "nearest" });
      } else if (e.key === "Enter") {
        e.preventDefault();
        if (cursor >= 0 && results[cursor]) choose(results[cursor].id);
      } else if (e.key === "Escape" && !listbox()?.hidden) {
        // The first Escape dismisses the suggestions; only a second one closes
        // the dialog. Losing half-finished work to a stray keypress is the
        // thing people actually complain about with modals.
        e.stopPropagation();
        closeResults();
      } else if (e.key === "Backspace" && !box.value && staged.size) {
        staged.delete([...staged.keys()].pop());
        paint();
        input()?.focus();
      }
    });
    box.focus();
  }

  paint();
  wireOnce();
  // What the page was holding may be a minute old, and the owner block is not
  // on it at all, so the modal asks rather than guesses. The first paint has
  // already happened, so this fills in rather than delaying anything.
  reload();
  return dlg;
}

// ---------------------------------------------------------------------------

function bodyHtml(resource, state, staged, word) {
  const people = state.shares.filter((s) => s.subject_type === "user");
  const everyone = state.shares.some((s) => s.subject_type === "everyone");
  const stagedList = [...staged.values()];

  return html`
    <div class="pick">
      <div class="pick-field" id="pickField">
        ${raw(stagedList.map((p) => html`
          <span class="chip">${avatar(p, 20)}${p.display_name}
            <button class="chip-x" data-unstage="${p.id}"
                    aria-label="Remove ${p.display_name}">✕</button>
          </span>`).join(""))}
        <input id="pickInput" type="text" role="combobox" autocomplete="off"
               aria-autocomplete="list" aria-expanded="false"
               aria-controls="pickList" spellcheck="false"
               placeholder="${stagedList.length ? "Add another…"
                 : "Search by name, username or email"}">
      </div>
      <ul class="pick-list" id="pickList" role="listbox"
          aria-label="People" hidden></ul>
    </div>

    ${raw(stagedList.length ? html`
      <div class="row" style="gap:8px;margin-top:10px;justify-content:flex-end">
        <select id="shareLevel" style="max-width:150px">
          <option value="view">Can view</option>
          <option value="edit">Can edit</option>
        </select>
        <button class="btn-primary btn-sm" id="shareApply">
          Share with ${stagedList.length}
          ${stagedList.length === 1 ? "person" : "people"}</button>
      </div>` : "")}

    <h4 style="margin:18px 0 6px">People with access</h4>
    <div class="access-list">
      <div class="access-row">
        ${avatar(state.owner, 30)}
        <div class="access-who">
          <span class="access-name">${state.owner?.display_name || "Unowned"}</span>
          <span class="access-sub">${state.owner?.email || state.owner?.username || ""}</span>
        </div>
        <span class="badge">owner</span>
      </div>
      ${raw(people.map((s) => html`
        <div class="access-row">
          ${avatar(s, 30)}
          <div class="access-who">
            <span class="access-name">${s.display_name || s.username}
              ${raw(s.pending
                ? `<span class="badge badge-soft">not signed in yet</span>` : "")}
            </span>
            <span class="access-sub">${s.email || s.username || ""}</span>
          </div>
          <select data-level="${s.subject_id}" class="access-level">
            <option value="view" ${s.level === "view" ? "selected" : ""}>Can view</option>
            <option value="edit" ${s.level === "edit" ? "selected" : ""}>Can edit</option>
            <option value="owner">Make owner…</option>
            <option value="remove">Remove</option>
          </select>
        </div>`).join(""))}
      ${raw(!people.length ? `<p class="muted tiny" style="margin:4px 0 0">
        Nobody yet.</p>` : "")}
    </div>

    <label class="check" style="margin-top:14px">
      <input type="checkbox" id="shareEveryone" ${everyone ? "checked" : ""}>
      Everyone with an account on this studio can view it
    </label>

    ${raw(people.some((s) => s.pending) ? html`
      <p class="muted tiny" style="margin-top:12px">Somebody marked
        <em>not signed in yet</em> came from your directory and has never
        opened the studio. They keep the access you just gave them; it is
        waiting the first time they sign in.</p>` : "")}

    <p class="muted tiny" style="margin-top:12px">Anyone here can open this
      ${word}. Only the owner can delete it or change this list.</p>`;
}
