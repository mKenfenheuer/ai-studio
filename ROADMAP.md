# Roadmap

Written 2026-09-08 from a full read of the code on the `data-workbench` branch:
every view in `web/js`, every module in `controller/`, `runner/` and `common/`,
the Docker images and the scripts. Where a claim below names a file and line,
that is where the evidence is.

The short version: the *reasoning* in this studio is unusually good and the
*shell* has not kept up with it. Six kinds of run, eighteen routes and a
three-pane dataset editor grew out of a four-step wizard, and the seams show
exactly where a user has to carry an id in their head from one screen to the
next. The dataset side is close to professional; the training side is a very
good single-shot launcher that is not yet an iteration loop; evaluation has the
best statistics in any tool of this class and no way to score the one model
that matters most (the base you started from). Nothing in the platform below
the UI knows what a byte of audio or an image is.

The roadmap is in phases. Phase 0 is a week of fixes that are wrong today.
Phase 1 is the ribbon-everywhere shell. Phases 2 to 4 finish the text
workflow as a loop. Phase 5 is the modality plumbing that has to exist once
before any of Phases 6 and 7 (vision, audio) can be built without building it
three times. An operations track runs beside all of it.

---

## 1. Where it stands

**Strong, keep:**

- The training preview renders the exact string the trainer sees, from the
  same function (`common/formatting.py`), and the playground sends the exact
  format the run trained with. This is the spine of the product.
- The dataset editor is a real query editor: steps as wire payloads, a
  preview at every step, undo, a formula bar, a saved recipe on the child,
  in-place row curation with a history. Nothing else in the self-hosted space
  has this.
- Architecture validation and the from-scratch planner give a named field, a
  level and a concrete fix for every problem (`controller/architectures.py`).
- Evaluation refuses to name a winner it cannot statistically defend
  (`runner/jobs/evaluate.py:261-342`).
- The scheduler is fair, resumable and honest about machines that go quiet.
  `scripts/deploy.sh` refuses to kill a run without being told what it costs.
- Auth is far better than the README admits: scrypt, sessions, OIDC with
  PKCE, per-resource ACLs, API keys, directory sync, 404-not-403.

**Weak, and the reason for this document:**

- Hand-offs are carried in `sessionStorage` or not at all. The wizard has no
  memory; refresh and every choice is gone.
- A run page has no path to "score this against the base model".
- The trainer ignores the `validation` split the dataset editor just made.
- Every list is flat: runs, datasets, sweeps (no list at all).
- Seven state patterns across eighteen views; the one documented rule
  (`draw()`/`ensure()`) is implemented once, unexported, in the largest file.
- The dashboard's first sentence advertises vision models. There are none.

---

## 2. The workflow as it is, and where it breaks

Read as: get data → shape it → train → watch → score → publish → feed back.

### Dataset side

| # | Break | Evidence |
|---|---|---|
| D1 | Generation drops the user on a job page; the dataset appears later, and if registration fails the rows are stuck in a zip with no "import this" action | `generate.js:194`, `app.py:981-983` |
| D2 | No way to paste a handful of rows as a *new* dataset | only `rows/add` on an existing one, `dataset.js:464` |
| D3 | The quality report is a modal; nothing persists, no badge on the library row | `dataset.js:507`, `stats` never rendered |
| D4 | "Open recipe on the source" silently overwrites the parent's draft | `dataset.js:340` |
| D5 | Every Apply makes `X (cleaned) (cleaned)`; the library is flat, no lineage tree, no tags | `datasets.py:1037`, `data.js:399-423` |
| D6 | Format confidence and mode are never shown; the user is not told a conversion is needed | `formatting.py:656-686` unread by UI |
| D7 | The wizard's field selectors are sent with the *job* and never written back to the dataset, so the mapping is redone every run | `wizard.js:1611`, `PATCH /{id}` accepts `format` but no UI writes it |
| D8 | The training preview is reachable only from the wizard; the wizard has no link back to the editor | `app.py:1502` |
| D9 | No token count against `max_seq_len`; truncation is silent | `lora_llm.py:602`, `datasets.py:927` |
| D10 | "Train on this" carries `{id, name, splits}` and defaults the split to dict order, so a file starting with a validation row trains on validation | `wizard.js:812` |
| D11 | The trainer carves its own 5% validation slice out of `train` and never uses the dataset's `validation` split; `split()` produces two datasets, not one with two splits | `lora_llm.py:647-652`, `datasets.py:1708` |
| D12 | No "runs trained on this" list; deleting a dataset is blind | none |
| D13 | Prompt set "from a dataset" takes the first N rows of *any* split, then the model card claims held-out | `evals.py:116` vs `cards.py:441` |

### Training side

| # | Break | Evidence |
|---|---|---|
| T1 | Wizard state lives in a closure: refresh, browser Back or a mode switch loses everything | `wizard.js:161-210`, `407-425` |
| T2 | The plan is computed for a hardcoded 2,000 rows regardless of the dataset | `wizard.js:2186`, `app.py:1578` |
| T3 | Over-long rows, too-small datasets, unmeasurable turn boundaries and empty splits are all discovered after launch | `lora_llm.py:460, 637, 657`; `scratch_llm.py:128` |
| T4 | A failed run's only path is "Run again with changes", which cannot change model, format, quantization or dtype (the things that fix OOM and format failures) | `rerun.js:293-296` |
| T5 | Resume is visible only on the run page; the scheduler's 10-minute wait for the checkpoint's machine is never shown | `job.js:1664`, `scheduler.py:325` |
| T6 | Sweeps have no list view (`#/sweeps` 404s) and no "promote the winner" | `app.js:47`, `sweep.js` |
| T7 | Compare ranks all kinds together and includes dataset-generation runs | `compare.js:20-21`, `db.py:451` |
| T8 | "Train further" and rerun carry `sweep_id` and `published` from the parent | `job.js:1203`, `rerun.js:81` |
| T9 | No "score this" button on a run page | `job.js:1543-1595` |
| T10 | Queue reason collapses every cause into one sentence | `job.js:1660` |
| T11 | `merge_after` is offered on the from-scratch review (where it is meaningless) and absent from the fine-tune review (where it decides 14 GB) | `wizard.js:2426`, `lora_llm.py:1003` |
| T12 | Fine-tuning has no seed; the dataset it trained on is mutable and un-hashed, so a run is not reproducible | `lora_llm.py:651, 671` |

### After training

| # | Break | Evidence |
|---|---|---|
| E1 | Cannot evaluate the base model, a Hub model or a hosted model: no baseline | `evalview.js:77`, `evals.py:245-255` |
| E2 | One system prompt is applied to every scored model, ignoring each run's own | `evaluate.py:132, 179` |
| E3 | Scores do not record the settings that produced them | `db.py:1403-1410` |
| E4 | The playground has no history, no side-by-side, no way to mark an answer good or bad | `scheduler.py:500-503` |
| E5 | No export beyond an HF zip; the run page says merging is "what Ollama wants" and then does not produce GGUF | `job.js:727` |
| E6 | The self-improvement loop (a fine-tune writes the next dataset) exists and is reachable from nowhere on the model | `generate.js:126-131` |
| E7 | No model registry: served models are job ids, aliases are run names, rename and every client breaks | `serving.py:91-133` |
| E8 | Usage is counted and thrown away | `serving.py:387-392` |

### Shell

| # | Break | Evidence |
|---|---|---|
| S1 | Account page unreachable on a phone; nothing highlights when on it | `styles.css:666`, `app.js:36` |
| S2 | Machines page has zero outbound links; it is the first place a new user lands | `runners.js` |
| S3 | Members see a join command with `AI_STUDIO_JOIN_TOKEN=null`; any member can re-probe any machine | `runners.js:44-53`, `app.py:243-259` |
| S4 | Runs list: no search, sort or filter; runs auto-name identically | `jobs.js`, `app.py:673-681` |
| S5 | Mobile tab bar sized for six tabs, now eight or nine; Dashboard and Datasets share a glyph | `styles.css:681`, `index.html:23,30` |
| S6 | 18 native `confirm()`/`prompt()` calls, one of them for a password, while a proper `<dialog>` modal exists | `users.js:82`, `util.js:181` |
| S7 | `document.title` is never set | none |
| S8 | Every browser tab receives every user's playground tokens over the event stream | `scheduler.py:504-517`, `app.py:162-183` |
| S9 | Three words for one noun: Evaluate / evals / prompt sets; "job" leaks into user copy | `index.html:31`, `generate.js:375` |

---

## 3. Phase 0: wrong today (days)

Each of these is a small change and each is actively misleading someone.
**Done on 2026-09-08**, all fourteen, on the `data-workbench` branch.

1. **README.** Remove "no user authentication yet" and "HF token not editable
   from the browser"; add the eval harness and the OpenAI endpoint to Status;
   fix the loss-masking "known limitation" (`lora_llm.py:545` implements it).
2. **Dashboard copy.** Drop "and vision" from `dashboard.js:16` until Phase 6.
3. **Runners endpoints.** `require_admin` on `POST /api/runners/{id}/reprobe`;
   hide `joinCard` for members, or say "ask an administrator".
4. **Event stream isolation.** Route `generate_*` frames to the requesting
   session only. Filter `job_log`/`job_metric` by visibility of the job.
5. **`POST /api/chat/{id}/cancel`** ownership check.
6. **Prompt set from a dataset** takes a split and defaults to
   `validation`/`test` when present; the UI gets a split picker; the model
   card only claims "held-out" when the split was one the run did not train on.
7. **Compare filter** by `MODEL_KINDS` and by comparable kind.
8. **`_finish()`** reports `deadline` as `length`, not `stop` (`serving.py:75`).
9. **Plan against the real row count** (`wizard.js:2186`).
10. **`merge_after`** moves to the fine-tune review.
11. **Rerun / train-further** strip `published` and `sweep_id`.
12. **Account link** in the mobile nav; `nav: "account"` highlight.
13. **Machines page** gets "Connected. Start your first run →" and a link
    from each busy card to its run.
14. **`MAX_ROWS` truncation warns** (`datasets.py:152`).

---

## 4. Phase 1: one shell, the ribbon everywhere

**Done on 2026-09-08.** Every view has the bar; the shared layer underneath it
exists; the shell work landed. What was added beyond the plan: a render check
(`scripts/check-render.mjs`) that draws every view in a real browser and clicks
every tab, because there was no way to find out that a view had stopped drawing
short of opening nineteen routes by hand. It immediately found a grid item with
the default `min-width: auto` dragging the whole page sideways on a phone.

Deferred to Phase 2, where they belong with the work they depend on: the
dataset editor's **Check** tab (needs the persistent quality report, 5.3) and
its **Train** tab (needs the format written back to the dataset, 5.4).


The dataset editor's ribbon (`dataset.js:664-711`, CSS at
`styles.css:1264-1307`) is the right shape for every page that *does things to
a thing*: tabs across the top, groups of labelled icon buttons underneath, the
thing itself below. It replaces the current pattern of a page head, an action
row that differs per page, and cards that hold both content and the forms that
act on it. Today only `#/data/:id` has it.

### 4.1 Extract the component

Move `rb()`, `group()` and `ribbon()` out of `dataset.js` into
`web/js/ribbon.js`:

```js
ribbon({ tabs, active, body: (tab) => [group(...), ...], right, onTab })
```

with `rb()` growing three variants it already almost has: a `<select>`, a
segmented toggle (`seg`), and a menu button that opens a `modal()` (for
forms that today live in cards: upload, import, connect a machine). The
delegated `[data-tab]` handler and the CSS are already generic. Persist the
active tab per route in `localStorage` the way `pqLeft` is.

Mobile: the tab strip scrolls horizontally; groups drop their labels below
600px; a ribbon of more than five groups collapses the trailing ones into a
"More" group. Keyboard: tabs are `role="tablist"`, arrows move between them.

### 4.2 Per-view layout

**Datasets index `#/data`** — the three source cards and the merge bar go into
the ribbon; the library becomes the full-width grid, with lineage as an
indented tree.

| Tab | Groups |
|---|---|
| Home | **Get data**: Upload · Import from Hub · Write with a model · Paste rows. **Selected**: Open · Merge · Share · Delete. **Use**: Train on this |
| View | **Show**: List / Tree by lineage · Mine / Shared / All. **Sort**: updated · rows · name. **Columns**: rows, splits, origin, format, quality |
| Check | **Quality**: Check selected · Show badges (fill, duplicates, length) |

Upload, Import and Paste open as modals holding exactly the forms that are on
the page today (`data.js:210-325`). Import keeps its "look up splits" step.

**Dataset editor `#/data/:id`** — unchanged, plus a **Train** tab (see 5.4)
and a **Check** tab that replaces the modal (see 5.3).

**Runs `#/jobs`**

| Tab | Groups |
|---|---|
| Home | **New**: Training run · Sweep · Write a dataset · Score models. **Selected**: Open · Stop · Resume · Run again · Compare · Delete |
| Filter | **Status** · **Kind** · **Base model** · **Dataset** · **Mine / Shared** · search box |
| View | Table / Cards · Sort · Group by sweep · Columns (held-out loss, steps, VRAM, duration) |

**Run page `#/jobs/:id`** — one layout with the kind-specific panels chosen by
a registry (4.3), not five layouts.

| Tab | Groups |
|---|---|
| Home | **Run**: Stop · Resume · Run again · Train further · Rename. **Use**: Try it · Score it · Compare · Write data with it. **Out**: Download · Export (GGUF, merged, adapter) · Publish · Share · Delete |
| Charts | **Series** toggles (train, held-out, LR, experts, VRAM) · Smoothing · Log scale · Download CSV |
| Log | Level filter · Follow · Search · Download |
| Card | Edit model card · Regenerate · View on Hub |

**Playground `#/play/:id`**

| Tab | Groups |
|---|---|
| Home | **Model**: pick · Add a second (side-by-side) · Swap. **Chat**: New · Save · Load. **Turn**: Rate good / bad · Correct · Add to dataset |
| Settings | temperature · max tokens · top-p · reasoning · system prompt |
| Data | Load a held-out row · next / previous · show expected |

**Evaluate `#/evals`, `#/evals/:id`**

| Tab | Groups |
|---|---|
| Home | **New set**: Type prompts · From a dataset split · Import. **Score**: pick models (runs, *base models, Hub, hosted*) · run. **Set**: Edit · Copy · Share · Delete |
| View | Metric · latest only / all scorings · show verdict · trend |

**Machines `#/runners`**

| Tab | Groups |
|---|---|
| Home | **Connect** (admin: join command modal) · Re-check · Drain · Remove. **Queue**: what is waiting and why |
| View | Cards / Table · show disk · show versions |

**Wizard `#/new`** — the step chips become ribbon tabs; a tab is enabled once
the steps before it are complete. The body holds the step's quick actions:
Data → Upload / Library / Hub / Paste; Format → Model's own / Plain / Custom /
Named; Review → Save as template · Export config · Sweep · Start.

**Account / Settings** — the seven cards on `#/account` become tabs:
Profile · Keys · Hugging Face · Hosted models · Notifications · Sessions.
Settings: General · People · Sign-in (SSO) · Machines · About.

### 4.3 The shell work that the ribbon needs anyway

All done, plus two things the audit listed under P2 that were cheap once the
components existed: toasts carry an action link and are announced, and the
active nav item carries `aria-current`.


- **`components.js`**: `pageHead()` (sets `document.title`), `breadcrumb()`,
  `emptyState()`, `copyButton()`, `confirmDestructive()` on the existing
  `modal()`. Sweep the 18 `confirm()`/`prompt()` calls and the 11 hand-rolled
  back-links through them.
- **Job-kind registry** (`web/js/kinds.js` and `controller/kinds.py`): per
  kind, its label, running verb, progress unit, subject line, icon, run-page
  panels, wizard steps, playground interface, whether it leaves a model,
  whether it is comparable. Today those facts are spread across
  `util.js:133-149`, `dashboard.js:88`, `jobs.js:48`, `job.js:8-33` plus five
  layout branches, `wizard.js:18-55`, `play.js:38`, `serving.py:24`,
  `app.py:1947`, `scheduler.py:433`. This registry is the prerequisite for
  Phases 6 and 7 being a data change rather than a seventeen-file change.
- **Promote `ensure()`/`resource()`** from `wizard.js` into `util.js` and
  adopt it in `job.js`, `account.js`, `jobs.js`. Views patch the parts that
  changed on `jobs_changed` rather than repainting the page.
- **Nav**: Dashboard · New · Runs · Datasets · Playground · Evaluate ·
  Machines · Settings, five on the mobile bar with a "More" tab; distinct
  glyphs; `aria-current`. Rename in copy: "prompt set" everywhere the nav says
  Evaluate; "run" never "job" in user-facing text.
- **Global search** (`/`): runs, datasets, prompt sets, machines by name.
- Toasts get `role="status"` and an optional action link. A manual theme
  toggle wires the `data-theme` hook that already exists.

---

## 5. Phase 2: the dataset side, finished

**In progress.** Done so far, deployed to the lab on 2026-09-08:

- **5.1 stable row identity** — every row carries `_id`, kept through filters,
  renames, merges, conversion to chat and edits. Positions still accepted, so
  older datasets keep working and gain names on the first rewrite. Versions
  (the second half of 5.1) are not done.
- **5.5 splits that mean something** — one dataset with two splits, optionally
  stratified, and the trainer measures on the held-out split when there is one
  rather than carving a second slice out of the training data.
- **5.3 the Check tab** — the report is a tab rather than a modal, the verdict
  persists as a badge and goes stale by itself, and four new checks: leakage
  between splits, near-duplicates, secrets, and per-column types.
- **5.4 make it trainable, on the dataset page** — the Training tab: how the
  rows are read and how confident that is, what the model will actually read,
  row lengths against a context window, the conversation report, and a mapping
  editor that writes back to the dataset.

Still to do: versions and diffing (5.1), tags and "runs trained on this"
(5.2), the generated-data review queue (5.6), and sources and scale (5.7).
Exact token counts need a tokenizer, which needs a runner; that goes with the
pre-flight checks in Phase 3.


### 5.1 Stable row identity and versions

Rows are addressed by file position (`datasets.py:122`); one delete renumbers
everything after it. Give every row an `_id` on write, keep it through
transforms, and address edits, annotations, provenance and diffs by it.

With ids, versions become cheap: an in-place edit appends a version record
(`{version, parent_version, edits[]}`), the file gets a `.v{n}` sidecar for
the rows it changed, and "as of version n" is reconstructable. A run records
the dataset `id` **and** version and a content hash, which is what makes T12
solvable.

### 5.2 Lineage-aware library

Tree by `parent_id`; a derived dataset can be marked as *superseding* its
parent, which folds the parent away by default. Tags. "Runs trained on this"
on every dataset page, from `jobs.config.studio_dataset`.

### 5.3 The Check tab

The inspect report moves out of the modal into a ribbon tab with persistent
badges on the library row and the editor header. Add: per-column length and
type distribution, mixed-type detection, near-duplicate detection (MinHash on
the rendered text), train/validation n-gram overlap (leakage), language
guess, PII/secret patterns, and rows sorted by "worst first". Each finding
keeps the existing "becomes a step" affordance (`dataset.js:516-522`).

### 5.4 Make it trainable, on the dataset page

- Show `format.mode` and `confidence` in the header. Below "low", say so and
  offer the conversion.
- A **Train** tab holding the wizard's field-selector panel and the training
  preview (`app.py:1502`), writing the result to `datasets.format` via the
  `PATCH` that already accepts it. The wizard reads that; the mapping is done
  once.
- The conversation report (`GET /{id}/conversation-report`, currently
  uncalled) rendered on the same tab after every edit.
- **Token counts against a model.** Pick a tokenizer once (the runner already
  has it; add a lightweight `POST /api/tokenize` that a connected runner
  answers, cached per dataset+tokenizer), and show per-row tokens, the p50/p99,
  and *how many rows exceed the chosen `max_seq_len`* before launch.

### 5.5 Splits that mean something

- `split()` writes one dataset with a `validation` split (the README's own
  design), stratified by a chosen column or by source document so chunked
  documents do not leak across the cut.
- The trainer uses the dataset's `validation` split when it has one and only
  carves its own when it does not. `val_fraction` and the cap become
  settings.
- "Train on this" carries the whole format and the default split is `train`.

### 5.6 Generated data gets a review queue

Generated rows land in a `review` split. A review view shows prompt, output
and the seed/topic that produced it, with keyboard accept / reject /
regenerate / edit, and a bulk "accept all with score above". Accepted rows
move to `train`. The same view is the annotation tool for uploaded data
(label column, reviewer, timestamp on the row).

### 5.7 Sources and scale

- Paste rows as a new dataset. Import from a URL and from a mounted path on
  the controller. Parquet through `pyarrow` as an optional dependency, with
  the refusal message kept when it is absent.
- Stream transforms instead of `list(iter_rows(...))` so a two-million-row
  Apply does not load into the controller's heap; keep the preview sampled.
- Column sort, hide, pin and cell copy in the grid.

---

## 6. Phase 3: training as an iteration loop

**In progress.** Done so far, deployed to the lab on 2026-09-08:

- **6.1 the wizard remembers** — the step is in the address so Back and Forward
  move between steps; everything else is a draft that survives a reload and
  expires after eight hours; switching mode keeps the dataset; "train on this"
  is a link. The file is not yet split per step.
- **6.2 pre-flight instead of post-mortem** — the five checks that used to fail
  an hour into a run, at the review step, gating the start. Includes real token
  counts: a `tokenize` message asks whichever runner is free, so the controller
  gets exact lengths without growing a tokenizer.
- **6.3 failure has a next step** — findings carry the correction as settings,
  the report offers "fix it and run again", and the rerun form can reach
  quantization, dtype, the base model and the merge setting, which is to say it
  can now fix the most common failure.
- **6.4, in part** — sweeps get a list page; runs get notes.
- **6.5 reproducibility** — one seed drives everything random in a fine-tune
  and is a visible setting; library versions are recorded on the run.
- **6.6, in part** — a queued run says what every connected machine made of it.

Then, in a second pass:

- **6.4** — compare gained a search box, base-model and dataset filters, and a
  CSV of every column including the seed and the note. Runs gained notes.
- **6.5** — a run records a fingerprint of the data as it stood, so an edit
  that leaves the row count unchanged is still visible, and the run page
  gathers what it would take to repeat a run into one card.
- **6.6** — loss masking is a setting on both forms and is recorded; batches
  are padded to their own longest row rather than to the context, which on
  short examples is most of the arithmetic; checkpoints report their size, age
  and best point, on the run and on the machines page.
- **6.7 GGUF export** — a job kind that converts and quantises with llama.cpp,
  pinned to a tag in the CPU runner image. Verified on the lab: SmolLM2-135M
  converts and quantises to 105 MB at Q4_K_M in under four seconds.

Still to do: splitting wizard.js per step (6.1); run tags, as distinct from
notes (6.4); and the larger training features — full fine-tuning, layer
freezing, DoRA, preference tuning (DPO/ORPO), multi-GPU, and estimate
calibration (6.6). Those are each a piece of work in their own right rather
than a gap in the loop.


### 6.1 The wizard remembers

Step and every choice go in the URL (`#/new?step=3&model=…&dataset=…`) with
a draft in `localStorage` keyed by that URL. Browser Back and the Back button
do the same thing. Switching mode keeps the dataset. `#/new?dataset=x` is a
link the dataset page can emit instead of a `sessionStorage` hand-off.

Split `wizard.js` into one module per step plus a shared `plan.js`; the
registry decides which steps a kind has.

### 6.2 Pre-flight instead of post-mortem

At Review, run the checks that today fail after launch: rows over
`max_seq_len` (from 5.4), a validation split that will be empty, a corpus
below the tokenizer's minimum, a template whose turn boundaries cannot be
measured, a split with no rows. Same `{level, field, message, fix}` shape as
the architecture checks.

### 6.3 Failure has a next step

`diagnose` already says what to change (`diagnose.py:98-116`). The run page's
report gets a **Fix and run again** button that opens the wizard on the
failing step with the fix applied. Rerun can change every setting, including
model, format, dtype and quantization.

### 6.4 Experiments

- `#/sweeps` list; a sweep page with **Promote the winner** (rerun it with
  more steps / on the full dataset), **Compare** and **Try** on the best.
- Compare gains filters (dataset, base, kind, sweep, tag), arbitrary metric
  columns from `summary`, a scatter of one hyperparameter against held-out
  loss, CSV export.
- Run tags and notes. Names get a counter or a date when they would collide.
- Config export as YAML/JSON and import into the wizard; "copy as curl".

### 6.5 Reproducibility

Seed on fine-tuning; dataset id + version + hash, transformers/peft/torch
versions and the runner's capability snapshot in the run summary.

### 6.6 Training features

- Loss masking (`train_on`) as a review setting, with the mode shown on the
  run page. Sequence packing for fine-tuning. Checkpoints listed with size,
  age and a "restore from this one".
- Full fine-tuning and layer freezing for models that fit. DoRA / rsLoRA as
  options. Target-module selection exposed.
- **Preference tuning (DPO / ORPO)**: a `chosen`/`rejected` dataset shape in
  the workbench (the review queue in 5.6 produces it naturally), a
  `finetune_preference` kind, beta and reference model in the plan.
- Estimate calibration: record estimated vs measured VRAM and minutes per
  run, learn a per-runner correction, show both numbers.
- Queue transparency: the per-runner `can_run` reason on the run page and a
  fleet queue on the Machines page.

### 6.7 Export

GGUF (llama.cpp convert + quantize, as a `export` job kind on a CPU runner),
with an Ollama `Modelfile` and the vLLM / LM Studio one-liners in the model
card's "Use it" section.

---

## 7. Phase 4: evaluation closes the loop

**Done (first pass).** Baselines, per-model system prompts, recorded settings
and two more measures:

- A scoring can now include a model that is not a run of this studio: one off
  the Hub (loaded on the machine, so every measure works on it, the loss
  included) or one behind a connected API (asked over the network, scored on
  what it writes -- no provider exposes the probabilities a loss needs, and
  the row says so instead of leaving the column blank).
  `serving.hub_spec`, `inference.ensure_loaded`, `api/evals._baselines`.
- The base model a run was trained from is offered by name the moment that run
  is ticked, and on the run page's own "Score it" -- asked in the format its
  descendant was trained in, which is the comparison that isolates what the
  training added. `evalview.baselinePanel`, `job.js` Score it.
- Each model is asked with **its own** recorded system prompt, unless the
  scoring sets one for everybody, which is recorded as an override.
- Every score keeps the settings that produced it (generation settings, system
  prompt, where the model came from), so two rows taken under different
  conditions can be told apart. `eval_scores.settings`.
- **chrF** and **JSON validity** beside exact match and token overlap.
- The verdict ranks on whichever measure covers *every* model scored, not the
  best measure that covers some: announcing a winner chosen from two of three
  models, with the third above it in the table, was the most misleading thing
  the page could have done.

- **Published benchmarks.** *Done:* MMLU, MMLU-Pro, ARC-Challenge, HellaSwag
  and GSM8K, run the way the people who publish those numbers run them --
  multiple choice decided by the probability the model gives each option
  (letter-style for MMLU, length-normalised cloze for ARC and HellaSwag), and
  GSM8K by pulling the last number out of what the model writes. The runner
  fetches the dataset itself with `load_dataset`, so nothing is sampled
  badly on the way in. Every result records the whole recipe -- dataset,
  split, shots, sample, seed, protocol -- and a Wilson interval, because a
  sampled benchmark cannot separate two models a couple of points apart and
  a bare number invites exactly that comparison. A benchmark is a prompt set
  whose questions live on the Hub, which is what gives it the scores table,
  the trend over time, the champion and sharing for nothing.
  `controller/benchmarks.py`, `runner/jobs/benchmark.py`.
  Still to do: the code benchmarks (HumanEval, MBPP) need a sandbox to
  execute what the model wrote, which is its own piece of work; IFEval needs
  its verifier library; TruthfulQA MC2 is a third scoring protocol.

Remaining in this phase:
- **Metrics**: schema conformance beyond "it parses"; tool-call correctness
  (name, arguments); LLM-as-judge using the hosted providers that already
  exist in `common/apimodels.py`; pass@k with `n>1` sampling for code sets
  (sandboxed on a CPU runner).
- **Regression view.** *Done:* an "Over time" tab on a prompt set draws every
  scoring in the order it happened with the best-so-far beside it, names the
  champion, and offers to serve it under a registered name -- which is the
  promotion path the registry needed. Still to do: splitting the line per
  model lineage when a set has been used by several.
- **Playground**: side-by-side (the runner already holds several models,
  `inference.py:120-152`); saved conversations. *Done:* "keep this exchange"
  writes the conversation up to one assistant turn into a dataset's `review`
  split, with the answer editable first, mapped into the dataset's own shape
  when it is not a conversation dataset (`api/data._shaped`). That is the link
  that turns the playground into the source of the next training run.
- **Batch inference.** *Done:* a `from_dataset` mode in the generator answers
  every row of a split and writes the answers as a new dataset -- distillation
  onto prompts you already have, or a model's own answers to a held-out split
  written down where they can be read, corrected and trained on. Unlike every
  other generation source it is finite: it stops when the split runs out.
- **Registry.** *Done:* `model_aliases` maps a name to a run, with a stage, a
  note and the last twenty things it pointed at; `_resolve` checks aliases
  before run ids and names; `/v1/models` lists them as models in their own
  right; a Served models page registers and repoints them, and the run page
  has "Serve as…". Still to do: promotion from the sweep page and the eval
  page, which is where you learn which run deserves the name.
- **Usage.** *Done:* every reply the OpenAI-compatible API serves writes one
  row of the runner's real token counts, against the key, the run and the
  alias it was asked for; summed on the Served models page, kept 90 days.
  Still to do: expiry and scope on keys.
- **Score it** and **Write data with it** now sit on the run page and in the
  playground; the score dialog is one module, `web/js/scoring.js`, so the
  baseline checkbox cannot exist in one copy of it and not the other.
- **A studio with no GPU cannot score anything.** `can_run` refuses an
  evaluation on a CPU machine unless every model in it is hosted, so a
  baseline off the Hub -- a 135M model that scores three prompts in three
  seconds -- has nowhere to run on a GPU-less box. Either a size-aware rule or
  an explicit "run it here anyway", but `_by_preference` sends kind-restricted
  machines the work first, so allowing it naively would send small scorings to
  the CPU on machines that do have a card.


---

## 8. Phase 5: modality plumbing, built once

Nothing below the UI can hold a byte that is not JSON text. Five pieces, in
this order, before any audio or vision trainer:

**P1. Binary asset store (XL). Done.** `assets` is a table of *references*:
one row per (owner, bytes, home), the file itself content-addressed under
`DATA_DIR/assets/<sha>` and shared by every row naming it, released when the
last one goes. A dataset points at an asset in an ordinary column
(`"image": "asset:ast_…"`), so splits, filters, downloads and publishing
keep working without knowing assets exist. Every derivation adopts the
assets it copies -- `assets.Adopter`, inside `write_rows` -- so deleting the
original leaves the derived dataset's pictures where they are. `POST/GET/
DELETE /api/assets`, access decided by what the asset belongs to; the type is
an allowlist (no HTML, no SVG) and every response carries `nosniff` and a
content policy, because these bytes are served from the studio's own origin.
Per-file and per-account limits, env-overridable (`AI_STUDIO_ASSET_MAX_MB`,
`AI_STUDIO_ASSET_QUOTA_GB`); usage counts shared bytes once. `sha256` on
artifacts. `scripts/check-assets.py` guards the three promises.

**P2. Media in the workbench (L). First slice done.** An image or a
recording in an upload -- loose, or inside a zip -- becomes a row pointing at
a stored file, with the column named for what it is and the folder above it
as the label (`cats/0001.jpg` → `label: cats`), which is how every
hand-assembled classification set is laid out. The rows page says what each
asset on it is, and the workbench draws a thumbnail or a player and opens the
file in full on a click. A `metadata.jsonl`/`.csv` beside the files (the
Hub's imagefolder convention, `file_name` + whatever else) is joined onto the
rows, and beats the folder-derived label. `inspect` reads every stored
file's header -- pure Python, `common/media.py`, no Pillow -- and reports
dimensions, WAV durations, files whose bytes are not what their name says
(an HTML error page saved as `.png`, the classic), pictures under 64 px, and
files gone from disk; each becomes a finding on the Check tab. A row that
points at a stored file is not empty, to `inspect` and to `drop_empty`,
which had declared a folder of photographs 100% empty. *Still to do:* Hub
image/audio datasets fetched as assets rather than as dead viewer URLs;
durations for formats other than WAV (needs decoding, so the runner's job);
the text-length warnings still fire on a set with no text in it.

**P3. Typed samples and metrics (M). Done.** A sample is
`{step, kind: text|image|audio|table, text?, prompt?, asset_id?, caption?}`;
anything that is not text is a file the runner puts in the store first
(`POST /api/assets/from-runner/{job}`, join-token authenticated, belonging
to the run and released with it) and the run page draws it by kind --
`ctx.sample(step, kind, path=…)` on the runner. Every trainer's summary
names its headline number: `primary_metric: {name, label, value,
lower_better}`. One reader on each side (`app.primary_metric`,
`kinds.primaryMetric`) with the old `best_val_loss` read as a held-out loss,
lower better; compare, the sweep page and the sweep's champion rank through
it, in the metric's own direction, with its own label as the column heading.
*Still to do:* early stopping in the trainers reads `val_loss` directly and
will need the same treatment when a trainer with a different metric exists.

**P4. Typed content end to end (L). Done, with one deliberate deviation.**
`content` stays the words; a message's pictures and clips ride beside it in
`media` as `{kind, ref|url}`. Every consumer of a message -- the validator,
the loss mask, the trial splitter, the browser -- reads `content` as text,
and making it sometimes a list would have broken each of them in a different
place; a sibling field breaks none and reaches the same end. OpenAI content
parts (`image_url`, `input_audio`), the Hub's part lists and the studio's own
`asset:` references all normalise to it, so `/v1/chat/completions` and a
dataset row arrive identical. `_part_text` no longer turns a picture into
the string `"[image]"` in the middle of a sentence (which a model then
learned to say). Rendering puts a placeholder in front of the words, one per
item, in the format's own token (`image_token`, `audio_token`) or `<image>`
by default -- through the chat template and on the two plainer paths alike.
`media` is written to rows in key order and round-trips. The browser draws a
turn's media beside its words. *Still to do:* sending an image part on to a
hosted provider (`apimodels.chat_request` flattens to text); the playground's
"keep it" drops media when writing a row; `generate` returning an asset
waits for a model that produces one.

**P5. Media capture in the playground (M).** Microphone, drop and paste for
images and audio; an audio player and image viewer per turn.

**P6. Modality-aware capabilities and fit (M).** The probe reports installed
libraries (`torchaudio`, `diffusers`, `torchvision`, `ffmpeg`), CPU cores and
RAM, and the fit check becomes a strategy per kind rather than
parameters-times-bytes. `artifacts.is_present` and `_artifact_kind` learn
`model_index.json` and bare safetensors. Per-modality runner images selected
by `AI_STUDIO_RUNNER_KINDS`, which already works; `_serves_models` stops
hardcoding the text kinds.

---

## 9. Phase 6: vision

In order of how much of the existing machinery each reuses.

**6.1 Image classification (ViT / CLIP), effort M.** `finetune_vision_cls`.
`AutoModelForImageClassification`, LoRA-able, small. Dataset is
`{image, label}` from a folder-of-folders or a manifest. Metrics are scalars
(accuracy, F1) and flow through unchanged; a confusion matrix and a
"worst misclassified" grid use P3. Wizard: Goal → Model → Data (thumbnail
grid with label distribution) → Review. This is the forcing function for
P1/P2 and should go first.

**6.2 Vision-language LoRA (Qwen2-VL / LLaVA), effort L.** Closest to what
exists: the conversation model, chat templates, loss masking, the playground
and `/v1` all transfer once P4 lands. `AutoModelForVision2Seq` in both the
trainer and the inference host; the target-module picker excludes the vision
tower; the context planner counts image tokens. The training preview shows
the image beside the rendered text with its placeholder count checked against
the attachment count, the way tool calls are checked today.

**6.3 Diffusion LoRA (SDXL / Flux), effort XL.** A dedicated
`runner:diffusion` image. New trainer on `diffusers` with aspect-ratio
bucketing; the loss is uninformative, so the run page is a fixed-seed
validation image grid every N steps (P3 at its most demanding) plus CLIP
score. Artifacts are bare LoRA safetensors; a second inference host and a
`/v1/images/generations` router; an image gallery view instead of the chat
playground. Memory is resolution-bound, not parameter-bound (P6).

Evaluation for vision: accuracy and confusion matrix; CLIP score per item
(fits the paired-difference statistics as is); CIDEr for captions; FID as a
set-level number with a bootstrap verdict.

---

## 10. Phase 7: audio

**7.1 Speech-to-text (Whisper / wav2vec2), effort L.** `finetune_asr`.
Dataset is `{audio, text, language}` from a zip plus manifest or a Hub
`Audio()` column via P1/P2. `WhisperForConditionalGeneration` with LoRA; a
`runner:audio` image with `torchaudio`, `ffmpeg`, `jiwer`. Metric is WER/CER
(P3 polarity) with a stated normalisation policy; samples are reference vs
hypothesis pairs with the clip playable. Wizard skips the Format step for
language, task and sample rate. Playground: record or drop audio, read the
transcript; "load a held-out row" already fits this loop better than it fits
text.

**7.2 Text-to-speech, effort XL, after 7.1.** The output is the metric: a
sample panel of audio per step, A/B listening in evaluation, speaker
similarity and an MOS proxy as scalars. The model zoo is fragmented and few
are `AutoModelFor*`, so each backend is its own trainer and its own artifact
sniffing. Voice-cloning consent needs a field on the dataset. Do not start
before P1 to P5 have been shaken out by ASR.

---

## 11. Operations track (runs beside everything)

Cheap now, expensive once there are five modalities:

1. **Backup**: `sqlite3 ... "VACUUM INTO"` plus rsync of `artifacts/`,
   `datasets/`, `assets/` and `join_token`; a documented restore. There is
   nothing today and the DB is in WAL mode, so a naive copy tears.
2. **`schema_version`** in the `settings` table; migrations stop being a list
   of swallowed `ALTER TABLE` errors.
3. **Retention**: artifact TTL, per-user quota, `metrics`/`logs` trimming for
   finished runs; disk usage shown on Machines (the heartbeat already sends
   it and `GET /api/runners` drops it).
4. **Protocol and agent version** in the `register` frame, shown on the
   Machines page; "update available".
5. **Tests and CI**: pytest for `fair_order`, `requeue_jobs_for_runner`,
   `access_level`, `can_run`, `_artifact_kind`, the route table and `util.js`;
   `check-formats.py` and `check-web.mjs` in a GitHub Actions workflow; a
   linter.
6. **Secrets**: `jobs.config` holds a decrypted provider key and HF token in
   plaintext (`app.py:340, 525`). Move credentials to an encrypted side table
   resolved at dispatch. Scope or rotate the join token from Settings.
7. **HTTPS and proxy** documented; `AI_STUDIO_PUBLIC_URL` explained on the
   SSO page where it is needed.
8. **Structured logs** and a `/metrics` endpoint (queue depth, runner
   utilisation, GPU use from the heartbeat).
9. **Multi-GPU**: probe every device; one runner per device by default with
   a shared-host marker; DDP for from-scratch later.
10. **Job isolation**: run a job in a subprocess so a segfault or a leak does
    not take the agent with it.

---

## 12. Sequencing and effort

| Phase | Effort | Unblocks |
|---|---|---|
| 0. Wrong today | days | trust |
| 1. Ribbon everywhere + registry + components | 3–4 weeks | every later UI change |
| 2. Dataset side finished | 4–6 weeks | reproducible runs, review loops, modality datasets |
| 3. Training loop | 4–6 weeks | iteration without the wizard from scratch |
| 4. Evaluation loop | 3–5 weeks | "is it better than the base" |
| 5. Modality plumbing | 6–8 weeks | 6 and 7 |
| 6. Vision (cls → VLM → diffusion) | 3 + 5 + 8 weeks | |
| 7. Audio (ASR → TTS) | 5 + 8 weeks | |
| Ops track | 1–2 days per item, ongoing | |

Phases 1 and 2 can run in parallel (different files); Phase 5 can start
its P1 store while Phase 4 is in progress. Phases 6 and 7 wait on 5.
