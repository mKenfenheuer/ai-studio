# AI Studio

**Train, fine-tune and evaluate AI models on your own hardware, from a browser.**

[![CI](https://github.com/mKenfenheuer/ai-studio/actions/workflows/ci.yml/badge.svg)](https://github.com/mKenfenheuer/ai-studio/actions/workflows/ci.yml)
[![Images](https://github.com/mKenfenheuer/ai-studio/actions/workflows/images.yml/badge.svg)](https://github.com/mKenfenheuer/ai-studio/actions/workflows/images.yml)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

AI Studio is a self-hosted studio for the whole small-model workflow: bring in
data, curate it, fine-tune or train from scratch, evaluate the result against
something, then serve it over an OpenAI-compatible API — on GPUs you own.

It is built so that someone who has never trained a model can reach a working
result, and someone who has can still reach every knob. Runs on **NVIDIA
(CUDA)**, **AMD (ROCm)** and **Apple Silicon (Metal)**.

---

## Contents

- [Architecture](#architecture)
- [Quick start](#quick-start)
- [What it does](#what-it-does)
- [Hardware support](#hardware-support)
- [Configuration](#configuration)
- [Security](#security)
- [Project layout](#project-layout)
- [Development](#development)
- [Status](#status)
- [Limitations](#limitations)
- [License](#license)

---

## Architecture

Two pieces, deliberately separated:

```
┌─────────────────────────┐         ┌──────────────────────────┐
│      CONTROLLER         │         │        RUNNER            │
│  web UI · job queue     │◄────────│  owns a GPU · trains     │
│  Hugging Face browsing  │  ws://  │  reports what it can do  │
│  metrics · artifacts    │ outbound│                          │
│  no GPU, no torch       │         │  torch + ROCm/CUDA/MPS   │
└─────────────────────────┘         └──────────────────────────┘
        one of these                    as many as you like
```

**The runner dials out to the controller**, never the reverse. GPU boxes sit
behind NAT, on laptops, or inside containers with no routable address — dialing
out means a runner needs no inbound port, no static IP and no firewall rule.

The split also quarantines the fragile part. A ROCm or CUDA userspace mismatch
breaks a *runner*; the UI and your job history are untouched.

**The controller has no GPU dependencies.** It installs with a handful of
pure-Python packages and runs on a NAS, a spare laptop or a Raspberry Pi. The
web UI has **no build step** — no Node, no npm, no bundler. The only toolchain
anyone installs is Python.

---

## Quick start

### Docker Compose — everything at once

```bash
git clone https://github.com/mKenfenheuer/ai-studio.git
cd ai-studio/docker
echo "AI_STUDIO_JOIN_TOKEN=$(openssl rand -hex 24)" > .env
docker compose up -d
```

The UI is on <http://localhost:8420>. First boot asks you to set an
administrator password.

### Prebuilt images

```bash
docker pull ghcr.io/mkenfenheuer/ai-studio-controller:latest
docker pull ghcr.io/mkenfenheuer/ai-studio-runner:cpu     # no GPU
docker pull ghcr.io/mkenfenheuer/ai-studio-runner:cuda    # NVIDIA
docker pull ghcr.io/mkenfenheuer/ai-studio-runner:rocm    # AMD
```

The controller and the CPU runner are rebuilt on every push to `master`. The
CUDA runner is built on demand and on each release — it is 15 GB and takes
about forty minutes. The ROCm runner is ~42 GB and does not fit on a
GitHub-hosted runner at all, so it is built on a self-hosted one; to build it
yourself, `scripts/publish-images.sh rocm`.

### From source

```bash
pip install -e .
ai-studio                       # → http://localhost:8420
```

It prints a **join token** on first boot. Runners need it.

### Connect a machine with a GPU

Prepare the host once — this installs the GPU kernel driver and Docker:

```bash
sudo scripts/provision-host.sh          # auto-detects AMD or NVIDIA
```

Then run a runner against your controller:

```bash
# AMD (ROCm)
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
  -e AI_STUDIO_CONTROLLER=http://your-controller:8420 \
  -e AI_STUDIO_JOIN_TOKEN=your-token \
  ghcr.io/mkenfenheuer/ai-studio-runner:rocm

# NVIDIA (CUDA) — needs the NVIDIA Container Toolkit
docker run --rm --gpus all \
  -e AI_STUDIO_CONTROLLER=http://your-controller:8420 \
  -e AI_STUDIO_JOIN_TOKEN=your-token \
  ghcr.io/mkenfenheuer/ai-studio-runner:cuda

# macOS / Apple Silicon / no Docker
scripts/install-runner.sh
```

> **Windows with an NVIDIA card:** use the `cuda` image. Docker Desktop runs
> Linux containers on a WSL2 kernel, so it works unchanged — there is no
> separate Windows image and none is needed.

### Or borrow a GPU from Google Colab

**Machines → Connect a machine → Colab notebook** downloads an `.ipynb`
that attaches a Colab session to this studio for as long as Google lets the session live. It is the
answer to "I want to try this and I do not own a graphics card": a free T4
fine-tunes a 7B in 4-bit, and everything that matters — datasets, runs,
finished models — stays on your controller.

Nothing about the runner had to change for it. The agent has dialed *out*
since the day it was written, and Colab is the extreme case that design was
for: no inbound address, none possible, none needed.

Three things follow from where Colab is, and the notebook says all three
before it does anything:

- **Your controller needs an address Google can reach.** `localhost` inside a
  Colab VM is that VM. For a studio on your own network, put a tunnel in front
  of it first — `cloudflared tunnel --url http://localhost:8420` prints one —
  and treat that address plus the join token as a password. The notebook
  refuses a private address outright rather than failing obscurely later.
- **The runner's code comes from your controller**, not from a package index,
  so the machine runs exactly the version your studio speaks. Only third-party
  libraries are installed in Colab, and deliberately not torch: Google's build
  is matched to the card in that VM, and replacing it is the quickest way to
  end up training on the CPU.
- **The join token is not written into the notebook.** An `.ipynb` is a file
  people forward and commit. The notebook asks for the token instead, out of
  Colab's own secret store if you put it there, and never prints it.

When the session ends, the run it was training goes back on the queue. Its
checkpoint was on Colab's disk, which is wiped, so the studio waits ten
minutes for the session to return and then starts that run again elsewhere.
Reconnecting from the same notebook rejoins as the *same* machine rather than
leaving a row of ghosts on the Machines page.

### Using only part of a machine

A card is not always the studio's to take. It is in somebody's desktop, or the
room cannot take 300 W for six hours, or a second workload has to stay
responsive. **Machines → Usage limits** sets, per machine, how much of its
card this studio may have.

The two numbers look alike and are not, which is the part worth reading:

**Memory is a real cap.** The runner confines the process to that share, and
the controller plans against the same figure — so the batch is sized to fit
inside the limit rather than inside the card. The usual effect of setting it
is a smaller batch, not a failure. A 16 GB card at 50% is planned, checked and
refused exactly as a 8 GB card would be, everywhere that asks "will this fit",
because the limit is applied once where capabilities are read rather than at
each of the places that read them. The card's real size stays visible beside
it: "8 GB of 16 GB", because a 16 GB card reporting 8 reads as a broken probe.

**Compute is a duty cycle, not a partition.** No consumer card sells you 40%
of itself: there is no hardware mechanism, MPS is NVIDIA-only and needs a
daemon, and nothing equivalent exists on RDNA. So the trainer pauses between
optimiser steps until the card averages the share asked for — 50% means the
card is busy about half the time and the run takes about twice as long. That
is worth having, and it is not isolation, so the UI says "about" and tells you
what the run will now cost in hours. Chat replies are never throttled:
stuttering somebody's conversation to save power is a worse trade than letting
a model that is already resident finish its sentence.

A machine can also set its own limit, in its own environment
(`AI_STUDIO_GPU_MEMORY_PCT`, `AI_STUDIO_GPU_COMPUTE_PCT`), which is what a box
lent out on conditions should do — it survives the controller forgetting, and
it holds from the first second the runner is up. The studio's setting takes
precedence when there is one; clearing it in the UI is not the same as setting
it to 100%, it hands the decision back to the machine.

A change reaches an idle machine at once and a busy one at its next run.
Taking memory away from a run that was sized for it is how a setting becomes
a crash four hours in.

### Removing a machine

Not every machine is permanent — a laptop lent for an afternoon, a Colab
session Google took back — and a page of dead cards is a page nobody reads.
**Forget this machine** removes an offline one from the list.

It forgets, it does not ban: a runner whose agent is still going dials in
again within seconds and is listed afresh, so stopping it is what removes it
and this is what tidies up afterwards. Which is why it is refused while the
machine is connected, and refused again while any unfinished run depends on
it — one still marked as running there (wait a minute; the studio notices by
itself) or one queued *pinned* to it, which nothing else would ever resolve.
Finished runs keep the machine's id and lose only its name, which every screen
that shows one already copes with.

### Then train something

Open the UI and follow the four steps. Everything technical is chosen for you
from what your hardware measured about itself, and every choice is explained on
screen. When a run finishes, the **Playground** lets you talk to it.

---

## What it does

| | |
|---|---|
| **Datasets** | Import from Hugging Face, or upload `.jsonl` `.csv` `.txt` `.md` `.html` `.docx` `.zip`. Browse rows as a table, edit in place, filter, deduplicate, split, merge, and build columns with templates. Every derived operation writes a new dataset and records its lineage. |
| **Fine-tuning** | LoRA and QLoRA over any Hugging Face causal LM, with the exact training string previewed before the run starts. Produces the adapter *and* the adapter merged into its base as a standalone model. |
| **Training from scratch** | Trained tokenizer, packed corpus, held-out loss charted beside training loss, and live text samples so you can watch noise become sentences. Design the architecture yourself, with every combination checked against your card as you type. |
| **Evaluation** | Prompt sets with expected answers, scored across several runs in one job (loss, exact match, token F1) with a paired significance test that refuses to name a winner it cannot defend. Standard benchmarks are run with the exact recipe the published number comes from. |
| **Serving** | An OpenAI-compatible `/v1/chat/completions` and `/v1/models` over every finished run — streaming, with tool calls and reasoning, behind per-user API keys. Pin a model to a card with a deployment so it stays warm. |
| **Operations** | Which models are loaded where, tokens per second, failure rates, and per-key usage. Reserve a machine for serving or for training. |
| **Synthetic data** | Have a large hosted model (OpenAI, Azure OpenAI, Anthropic, or anything OpenAI-shaped) write the dataset your own small model trains on. |
| **Publishing** | Push a finished model or dataset to the Hugging Face Hub, with a generated model card carrying the scores that were actually measured. Export to GGUF for llama.cpp. |

For *why* each of these works the way it does — including the measurements that
corrected the estimates it shipped with — see **[docs/design-notes.md](docs/design-notes.md)**.

### Two ways to make a model

| | Fine-tune | From scratch |
|---|---|---|
| Starts from | a trained model | random noise |
| Trains | ~0.1% of the weights (LoRA) | every weight |
| Realistic size on one GPU | up to ~21B (4-bit) | up to ~100M |
| Time | minutes to hours | hours to days |
| Produces | an adapter *and* the merged model | a complete model |
| The model can | answer, once fine-tuned on instructions | continue text |

**Fine-tuning is what you want almost every time.** From-scratch is included
because it is the only way to actually see what training *is* — and because a
small model trained properly on simple text writes real English, which is a
genuinely surprising thing to watch happen on your own desk.

---

## Hardware support

The runner **measures** the machine rather than trusting its spec sheet, and
the UI disables what a given machine cannot do.

| | NVIDIA | AMD (ROCm) | Apple Metal | CPU |
|---|---|---|---|---|
| LoRA fine-tuning | ✅ | ✅ | ✅ | ✅ (slow) |
| 4-bit / QLoRA | ✅ | build-dependent¹ | ❌ | ❌ |
| Flash attention | ✅ | RDNA3+ / CDNA only | ❌ | ❌ |
| 8-bit optimizers | ✅ | build-dependent¹ | ❌ | ❌ |

¹ `bitsandbytes` ships CUDA-only wheels. The ROCm image builds it from source
for your GPU architecture (`BNB_ROCM_ARCH`, default `gfx1030`). This is
**verified working on gfx1030**: a 3B model fine-tunes in 3.2 GB of VRAM in
4-bit, against ~8 GB in fp16. Where the build is unavailable, the runner
reports 4-bit as unsupported and the UI hides it rather than letting a run fail
an hour in.

---

## Configuration

| Variable | Where | Meaning |
|---|---|---|
| `AI_STUDIO_DATA` | controller | State directory (default `./data`) |
| `AI_STUDIO_PORT` | controller | UI port (default `8420`) |
| `AI_STUDIO_PUBLIC_URL` | controller | External URL, so SSO redirects are right |
| `AI_STUDIO_JOIN_TOKEN` | both | Shared join secret; generated if unset |
| `HF_TOKEN` | both | For gated models (Llama, Gemma) |
| `AI_STUDIO_CONTROLLER` | runner | Controller URL |
| `AI_STUDIO_RUNNER_NAME` | runner | Display name |
| `AI_STUDIO_RUNNER_KINDS` | runner | Restrict to certain job kinds |
| `AI_STUDIO_GPU_MEMORY_PCT` | runner | Give the studio at most this share of the card's memory |
| `AI_STUDIO_GPU_COMPUTE_PCT` | runner | Give it at most this share of the card's time |
| `BNB_ROCM_ARCH` | rocm build | GPU arch, e.g. `gfx1030`, `gfx1100` |

---

## Security

**People sign in.** The first boot asks for an administrator password; after
that there are accounts with roles, sessions in an httpOnly cookie, scrypt-hashed
passwords with a rate limit on failures, and optional single sign-on through any
OIDC provider (Entra, Google, Keycloak, Authentik, …) with PKCE and directory
sync. Runs, datasets and prompt sets are private to their owner unless shared
with a named person or with everyone in the studio; anything you cannot see
answers 404, not 403.

**Credentials are per account.** Each person connects their own Hugging Face
token and their own hosted-model keys on the account page; they are encrypted at
rest, never returned to the browser, and attached to a run at the moment it is
created — so "whose key paid for this" has an answer. API keys for the
OpenAI-compatible endpoint are minted on the same page and shown once.

**The join token is the machine credential.** Anyone holding it can attach a
runner and read the work on it, so treat it as a password. Only administrators
see it.

> ⚠️ **There is no TLS in the controller itself.** Put it behind a reverse proxy
> that terminates HTTPS, and set `AI_STUDIO_PUBLIC_URL` so SSO redirects carry
> the right address. Do not expose it directly to the internet.

Found a security issue? Please open a
[security advisory](https://github.com/mKenfenheuer/ai-studio/security/advisories/new)
rather than a public issue.

---

Connecting a Colab runner means giving the controller an address on the public
internet, which is the one thing the rest of this design avoids. A quick
tunnel is a reasonable way to do it and a bad thing to leave running: while it
is up, the studio is as exposed as whatever is in front of it, and the join
token is what stands between a stranger and your job data. Stop the tunnel
when the session ends.

## Project layout

```
common/       message normalising, Jinja templating and prompt building --
              shared, so the preview, the trainer and the playground cannot
              render the same conversation three different ways
controller/   FastAPI app, SQLite, scheduler, HF proxy  (no torch)
runner/       capability probe, websocket agent, trainers, inference host
web/          zero-build UI (ES modules, no dependencies)
docker/       controller + cpu/cuda/rocm runner images, compose files
scripts/      host provisioning, bare-metal runner install, and the checks
docs/         design notes
```

---

## Development

There is no test framework and that is deliberate: the controller installs a
handful of pure-Python packages, and these checks have to run on the machine
somebody is debugging on. Each is a plain script that exits non-zero on failure.

```bash
python scripts/check-formats.py      # conversations survive the round trip to text
python scripts/check-datasets.py     # row names, editing, splits, transforms
python scripts/check-dispatch.py     # which run is allowed onto which machine
python scripts/check-assets.py       # stored files are shared, counted, released
python scripts/check-cards.py        # a model card says true things
python scripts/check-benchmarks.py   # a benchmark is asked the published way
python scripts/check-ops.py          # reservations, deployments, the usage ledger
python scripts/check-workflow.py     # the whole workflow, against a real controller

node --experimental-vm-modules scripts/check-web.mjs   # every browser script parses
node scripts/check-render.mjs                          # every view still draws, in Chrome
```

All of them run in CI on every push and pull request — see
[.github/workflows/ci.yml](.github/workflows/ci.yml).

**One rule in the web UI:** `draw()` renders purely from state, and no render
may start work that causes another render synchronously. The reasoning is in
[docs/design-notes.md](docs/design-notes.md#one-rule-in-the-web-ui).

---

## Status

Verified end-to-end on an RX 6900 XT (gfx1030), controller and runner both
containerised:

| Check | Result |
|---|---|
| LoRA fp16, SmolLM2-135M | loss 10.89 → 3.01, 0.67 GB |
| QLoRA 4-bit, Qwen2.5-3B | loss 10.74 → 3.01, **3.17 GB** |
| From scratch, 5.3M params, 300 steps | loss 8.999 → 2.904 in **120 s** |
| Held-out loss on the same run | 6.446 → 2.927, tracking training loss |
| Playground, from-scratch model | 214 tokens/s, first token in 0.3 s |
| Playground, 3B fine-tune | base + adapter loaded in 11.7 s |
| Tool-calling chat dataset | 4 roles, 19 tools, tool calls preserved end to end |
| Qwen 0.5B on that dataset | loss 2.246 → 1.392 with the model's own template |
| Runner killed mid-run | job requeued and restarted automatically |
| Runner killed after upload | run kept as finished, not restarted from noise |
| Cancelling a run mid-training | stopped cleanly, runner stayed online |
| Model too large for the GPU | refused at creation, not left queued |
| UI at 1440px and 390px | no JS errors, no horizontal overflow |

A 5.3M-parameter model, from noise, on TinyStories:

```
step  50  Once upon a time, there was a little happy they was very he. The a a
          loved was a a on it said very a to the.
step 150  Once upon a time, there was a little girl named Timmy. Timmy loved't
          play with his friends. One day, Timmy's friends went to the park.
step 300  Once upon a time, there was a little girl named Lily. She loved to
          play outside in the park with her friends. One day, she found a shiny
          rock on the ground.
```

Loss began at 8.999, which is `ln(8192)` — exactly the cost of guessing
uniformly from an 8192-token vocabulary, and a useful check that the model
really did start from nothing.

---

## Limitations

- **The ROCm runner image is large** (~42 GB on disk) because it builds on
  `rocm/dev-ubuntu-24.04:*-complete`, which is needed to compile bitsandbytes.
  A multi-stage build would cut this substantially.
- **No TLS in the controller.** Put it behind a reverse proxy.
- **One job per runner at a time.** No multi-GPU or multi-job scheduling yet,
  and a runner that is training will not serve the playground.
- **Loss masking is not a setting yet.** Fine-tuning on conversations trains on
  the assistant's turns only; there is no switch to train on every token.
- **From-scratch tops out around 200M parameters**, which is a compute limit
  rather than an arbitrary one.
- **The corpus is held in host RAM** while training (capped at 500M tokens,
  ~1 GB as uint16). Longer runs make repeated passes rather than streaming.

What is planned next is in **[ROADMAP.md](ROADMAP.md)**.

---

## Contributing

Issues and pull requests are welcome. Before opening a PR, please run the
checks in [Development](#development) — CI runs the same ones, so a green local
run is a green CI run.

---

## License

AI Studio is free software, licensed under the **GNU Affero General Public
License v3.0** — see [LICENSE](LICENSE).

The AGPL is the GPL plus one additional condition that matters for software
like this: if you run a modified version of AI Studio as a network service,
you must offer its users the corresponding source. Running it unmodified, for
yourself or inside your organisation, carries no such obligation, and the
models, datasets and adapters you produce with it are entirely your own.
