# Roadmap

Where AI Studio is going, and what is in the way.

First written 2026-09-08 from a full read of the code; reorganised 2026-09-09
around what is left. The completed work that used to fill most of this file has
been cut — it is in the git history, and the reasoning worth keeping is in
[docs/design-notes.md](docs/design-notes.md). A short summary of what shipped
is at the [end](#appendix-what-has-shipped).

Where an item names a file and line, that is where the evidence is.

---

## Where it stands

The original assessment was that the *reasoning* in this studio was unusually
good and the *shell* had not kept up with it. The shell has since caught up:
the work is organised as projects rather than five date-sorted lists, every
view has the same ribbon, and the lineage that was recorded all along is now
drawn. The text workflow is a loop — data in, train, score against a baseline,
publish, feed the result back — and each stage is reachable from the one before
it.

What is left divides cleanly:

- **Friction**, in places somebody using this every day hits daily. Small,
  and each one is felt.
- **Two unfinished loops** — the dataset side needs versions, the training side
  needs preference tuning. Neither blocks anything else.
- **One piece of security debt** that should not wait.
- **New modalities** — vision is one third built, audio is not started. The
  plumbing beneath them is done, which was the expensive part.

---

## 1. Friction, in the order it is felt

Written after taking a real project end to end: Mistral-7B-Instruct-v0.3, 4-bit
LoRA, 9,504 rows of a tool-calling home-automation dataset, through every stage
on the lab box. These are what that walk and daily use actually showed.

| # | What | Why it matters |
|---|---|---|
| F1 | **The playground lists every finished run in the studio**, newest first, with no grouping and no search | At fifty runs the model you want is unfindable. Scope it to a project, the way the wizard already is |
| F2 | **"Do the next thing" ignores the state** | The stage map computes the real next step; the ribbon offers the same three buttons regardless, and offers them on the unfiled pile, which is not a project |
| F3 | **A result does not carry forward** | The winning run is known and nothing is done with it: Publish does not name it, a first result does not suggest varying one setting, and the map that computes the next step cannot take it |
| F4 | **Re-scoring is undisciplined** | The same set can be run three times with the same settings and produce three identical rows. Scoring every new run against the project's set belongs to the project, not to somebody's memory |
| F5 | **Publishing is two buttons in two places** — the library from the run ribbon, the Hub from the Model tab | One "Publish…", with the destination as a choice inside it |
| F6 | **"What changed between these two runs" has no home** | Compare ranks metrics; a project wants settings and data versions beside them |
| F7 | **A project has a goal but not a target** | `goal` is free text. As a set, a measure and a threshold it could say *finished* instead of *five stages touched*, and a benchmark result would be judged against something |
| F8 | **The Models glyph reads as a bullet** at sidebar size | A distinct mark, as was done for Dashboard and Datasets |

F1 and F2 are hours of work each and are the two that get hit most often.
F3, F4 and F7 are the same idea from three directions — a project that knows
what it is trying to achieve can sequence the rest of itself — and are worth
designing together rather than one at a time.

---

## 2. Security and operations debt

**S1. Credentials sit in `jobs.config` in plaintext.** A decrypted provider key
and Hugging Face token are written into the job row at dispatch
(`app.py:340, 525`). Everything else about credential handling is careful —
encrypted at rest, never returned to the browser, attached per account — and
this one path undoes it for anybody who can read the database or a backup.
Move credentials to an encrypted side table resolved at dispatch. **This is the
item on this page that should be done first.**

**S2. `schema_version` in the `settings` table.** Migrations are currently a
list of `ALTER TABLE` statements whose errors are swallowed. That works until
one of them half-applies.

**S3. Job isolation.** Run a job in a subprocess, so a segfault or a leak in a
GPU library does not take the agent down with it. The capability probe already
does exactly this for the same reason; the trainers do not.

**S4. Deploys are unverified.** rsync and ssh work until a container recreate
stalls half way and nothing notices. `scripts/deploy.sh` should wait for what
it deployed to come back, at the version it deployed.

**S5. `/metrics` in Prometheus's shape.** `/api/ops/metrics` answers the serving
half — replies, tokens, tokens per second, failures with their reasons, by
model, machine, name and key. Queue depth and GPU use are still only on the
Machines page, and nothing is exported in a shape a scraper reads.

**S6. Nothing acts on a failure rate.** The operations page shows one. There is
no alerting, no request queueing or per-model concurrency limit, and no
autoscaling. Deployments are one model per machine per row with no notion of
how many will fit: the runner's eviction arbitrates, and a card with three
deployments that cannot hold three fails the third with an out-of-memory.

**S7. Not software.** Artifact transfers sit at 100 Mbit on a gigabit link.
That number is suspiciously exactly 100BASE-TX; the next step is `ethtool` on
both ends, not more code. The chunked transfer work that was done alongside
this was worth having and did not explain it.

---

## 3. The dataset side, finishing

Row identity, splits that mean something, the Check tab, the Training tab, the
review queue and tags all landed. Two pieces are left.

**D1. Versions and diffing.** Rows carry a stable `_id` now, which was the hard
half. With ids, versions are cheap: an in-place edit appends a version record
(`{version, parent_version, edits[]}`), the file gets a `.v{n}` sidecar for the
rows it changed, and "as of version n" is reconstructable. A run already
records a content fingerprint; recording the version alongside it is what makes
a run fully reproducible against mutable data.

**D2. Sources and scale.**

- Paste rows as a new dataset. Import from a URL, and from a mounted path on
  the controller.
- Parquet through `pyarrow` as an optional dependency, keeping the current
  refusal message for when it is absent.
- Stream transforms instead of `list(iter_rows(...))`, so a two-million-row
  Apply does not load into the controller's heap. Keep the preview sampled.
- Column sort, hide, pin and cell copy in the grid.

---

## 4. The training loop, finishing

The loop itself works: the wizard remembers, pre-flight checks gate the start,
a failure carries its own correction, runs are seeded and fingerprinted, and
GGUF export exists. What is left is not loop-shaped — each of these is a piece
of work in its own right.

**T1. Preference tuning (DPO / ORPO).** A `chosen`/`rejected` dataset shape in
the workbench — which the review queue already produces naturally — a
`finetune_preference` job kind, and beta and the reference model in the plan.
This is the largest single gap in what the studio can train.

**T2. Multi-GPU.** Probe every device; one runner per device by default with a
shared-host marker; DDP for from-scratch runs later. Today a runner takes one
job and uses one card.

**T3. Sequence packing for fine-tuning.** Dynamic padding landed — batches are
padded to their own longest row rather than to the context — which took most of
the arithmetic out. Packing takes the rest.

**T4. Smaller knobs.** rsLoRA as an option, and target-module selection
exposed rather than inferred.

**T5. Split `wizard.js` per step.** It is the largest file in the project and
holds every step of every kind of run. The job-kind registry that would let the
steps be data rather than branches already exists; this is the file catching up
to it.

---

## 5. Evaluation

The statistics, the baselines, the registry, the usage ledger and five
published benchmarks all landed. Everything remaining here is blocked on the
same missing piece.

**E1. A sandbox that executes what a model wrote.** Without it, three things
cannot be built: the code benchmarks (HumanEval, MBPP), pass@k with `n>1`
sampling, and any future scoring that runs generated code. This is one piece of
work that unblocks all three, and it is a security boundary rather than a
feature — it belongs beside S3, which is the same problem for a different
reason.

**E2. Two benchmark protocols.** IFEval needs its verifier library. TruthfulQA
MC2 is a third scoring protocol beside the letter-probability and cloze ones
that exist.

---

## 6. Modality plumbing: the last four gaps

All six pieces (P1 to P6) are done — the asset store, media in the workbench,
typed samples, typed content end to end, capture in the playground, and
modality-aware scheduling. Four loose ends remain, none of them blocking.

- **Durations for audio formats other than WAV.** The header reader is pure
  Python by design; anything else needs decoding, which makes it the runner's
  job.
- **Text-length warnings fire on sets with no text in them.** A folder of
  photographs gets warned about its row lengths.
- **The `seesPictures` flag reads a capability the run does not yet carry**, so
  the playground's "this model was not trained to look" notice is guessing.
- **The memory estimate is still parameters-times-bytes.** That is right for
  every trainer that exists and wrong for diffusion, whose cost is its
  activations. It needs a per-kind strategy before Phase 6.3, not before.

---

## 7. Vision

**V1. Image classification — first pass done.** ViT/CLIP backbones with a new
head, stratified hold-back, accuracy and macro-F1, a confusion matrix, the most
confidently wrong pictures as a grid, and "try it out" on the run page.
Verified end to end. *Left:* `/v1`-style serving for classifiers, and the CUDA
image needs rebuilding with Pillow.

**V2. Vision-language LoRA (Qwen2-VL / LLaVA), effort L.** The closest thing to
what already exists: the conversation model, chat templates, loss masking, the
playground and `/v1` all transfer now that typed content has landed.
`AutoModelForVision2Seq` in the trainer and the inference host; the
target-module picker excludes the vision tower; the context planner counts
image tokens. The training preview shows the image beside the rendered text,
with its placeholder count checked against the attachment count the way tool
calls are checked today.

**V3. Diffusion LoRA (SDXL / Flux), effort XL.** A dedicated `runner:diffusion`
image and a new trainer on `diffusers` with aspect-ratio bucketing. The loss is
uninformative, so the run page is a fixed-seed validation image grid every N
steps plus a CLIP score. Artifacts are bare LoRA safetensors; a second
inference host and a `/v1/images/generations` router; a gallery view instead of
the chat playground. Needs the per-kind memory estimate from section 6.

**Evaluation for vision:** accuracy and confusion matrix; CLIP score per item,
which fits the existing paired-difference statistics unchanged; CIDEr for
captions; FID as a set-level number with a bootstrap verdict.

---

## 8. Audio

Nothing here is started. The plumbing it needs is done.

**A1. Speech-to-text (Whisper / wav2vec2), effort L.** A `finetune_asr` kind
over `{audio, text, language}` rows, from a zip plus manifest or a Hub `Audio()`
column. `WhisperForConditionalGeneration` with LoRA, in a `runner:audio` image
with `torchaudio`, `ffmpeg` and `jiwer`. The metric is WER/CER with a stated
normalisation policy; samples are reference-versus-hypothesis pairs with the
clip playable. The wizard skips the Format step and asks for language, task and
sample rate instead. In the playground, record or drop audio and read the
transcript — "load a held-out row" fits this loop better than it fits text.

**A2. Text-to-speech, effort XL, after A1.** The output *is* the metric: a
sample panel of audio per step, A/B listening in evaluation, speaker similarity
and an MOS proxy as scalars. The model zoo is fragmented and few models are
`AutoModelFor*`, so each backend is its own trainer with its own artifact
sniffing. Voice-cloning consent needs a field on the dataset. Do not start
before ASR has shaken out the plumbing.

---

## 9. Sequencing and effort

| | Effort | Unblocks |
|---|---|---|
| S1 credentials out of `jobs.config` | 1–2 days | nothing, and it should still go first |
| §1 friction (F1–F8) | 2–3 weeks total, independently shippable | daily use |
| D1 versions · D2 scale | 2 weeks · 2 weeks | fully reproducible runs |
| T1 preference tuning | 3–4 weeks | the last common training method missing |
| E1 sandbox | 2 weeks | code benchmarks, pass@k, and S3 |
| T2 multi-GPU | 2–3 weeks | throughput on the machines people already own |
| V2 vision-language | 5 weeks | |
| A1 speech-to-text | 5 weeks | A2 |
| V3 diffusion | 8 weeks | |
| A2 text-to-speech | 8 weeks | |
| Ops items S2–S6 | 1–2 days each, ongoing | |

Nothing in sections 1 to 5 blocks anything in 7 or 8; the modality plumbing
that used to be the dependency is finished. The one real ordering constraint
left is A1 before A2, and the memory estimate (section 6) before V3.

---

## Appendix: what has shipped

Condensed, because the detail is in the git history and the reasoning is in
[docs/design-notes.md](docs/design-notes.md).

**The shell.** Every view has the same ribbon; a job-kind registry replaced
facts spread across seventeen files; `pageHead`, `breadcrumb`, `emptyState`,
`confirmDestructive` replaced eighteen native `confirm()` calls; global search;
`document.title`; distinct glyphs; `aria-current`. A render harness
(`scripts/check-render.mjs`) draws every view in real Chrome and clicks every
tab, because there was no other way to learn that a view had stopped drawing.

**Projects.** The work is organised as one project per model being made — its
datasets, runs, prompt sets, benchmarks and publications — with a map of five
stages and a lineage graph drawn from relationships that were recorded all
along. Nothing is copied: a project is a `project_id` on rows that already
exist. Filing is automatic where the answer is obvious.

**The dataset side.** Stable row ids through every transform; one dataset with
several splits, optionally stratified; the Check tab with leakage,
near-duplicate, secret and per-column-type detection; a Training tab that
writes the mapping back to the dataset; a review queue; tags; "runs trained on
this".

**The training loop.** The wizard remembers across a reload; five pre-flight
checks gate the start instead of failing an hour in; a failure carries its own
correction into a rerun that can change any setting; seeds and data
fingerprints make a run repeatable; loss masking is a setting; LoRA, DoRA, full
fine-tuning and layer freezing; estimate calibration against this studio's own
last twenty runs; GGUF export.

**Evaluation.** Baselines — the base model a run came from, any Hub model, or
a hosted API — each asked with its own recorded system prompt. chrF, JSON
validity, schema conformance, tool-call correctness with function and arguments
kept apart, and a hosted judge reported beside the measured numbers and never
instead of them. Five published benchmarks (MMLU, MMLU-Pro, ARC-Challenge,
HellaSwag, GSM8K) run with the exact recipe their published numbers come from,
each result carrying its whole recipe and a Wilson interval. An "over time" tab
that names the champion and offers to serve it.

**Serving and operations.** A model registry mapping names to runs; deployments
that pin a model to a card and are reconciled rather than fired once; machine
reservations honoured by the dispatcher *and* by the explanation of why a run
is waiting; a usage ledger that counts failures as rows; scoped and expiring
API keys; a chat-only account role enforced as an allowlist.

**The platform.** A content-addressed asset store with reference counting;
media in the workbench with thumbnails, players and header inspection; typed
samples so a trainer can emit an image or a table; typed content end to end, so
a picture reaches a hosted provider as that provider's own content parts; media
capture in the playground; modality-aware scheduling that refuses a job to a
machine that cannot decode its data, before the download rather than at the
first batch.

**Keeping it running.** Backups (`VACUUM INTO`, manifest, restore script) on a
schedule; retention that expires models but never the run, its chart or its
log; runner version reporting, so a stale agent says so; and the checks —
eight Python scripts and two browser ones, now run on every push by
[CI](.github/workflows/ci.yml).
