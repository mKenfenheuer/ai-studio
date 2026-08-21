# AI Studio

A web app for training, fine-tuning and evaluating AI models on your own
hardware — built so that someone who has never trained a model can get a
working result, and someone who has can still reach every knob.

Runs on **NVIDIA (CUDA)**, **AMD (ROCm)** and **Apple Silicon (Metal)**.

---

## How it is put together

Two pieces, deliberately separated:

```
┌─────────────────────────┐         ┌──────────────────────────┐
│      CONTROLLER         │         │        RUNNER            │
│  web UI · job queue     │◄────────│  owns a GPU · trains     │
│  Hugging Face browsing  │  ws://   │  reports what it can do  │
│  metrics · artifacts    │ outbound │                          │
│  no GPU, no torch       │         │  torch + ROCm/CUDA/MPS   │
└─────────────────────────┘         └──────────────────────────┘
        one of these                    as many as you like
```

**The runner dials out to the controller**, never the reverse. GPU boxes sit
behind NAT, on laptops, or inside containers with no routable address — dialing
out means a runner needs no inbound port, no static IP and no firewall rule.

The split also quarantines the fragile part. A ROCm or CUDA userspace mismatch
breaks a *runner*; the UI and your job history are untouched.

## Why the controller has no GPU dependencies

It installs with four pure-Python packages and runs on a NAS, a spare laptop or
a Raspberry Pi. The web UI has **no build step** — no Node, no npm, no bundler.
The only toolchain anyone installs is Python.

---

## Quick start

### 1. Start the controller

```bash
pip install -e .
ai-studio                       # → http://localhost:8420
```

It prints a **join token** on first boot. Runners need it.

### 2. Connect a machine with a GPU

Prepare the host once (installs the GPU kernel driver and Docker):

```bash
sudo scripts/provision-host.sh          # auto-detects AMD or NVIDIA
```

Then run a runner:

```bash
# AMD
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
  -e AI_STUDIO_CONTROLLER=http://your-controller:8420 \
  -e AI_STUDIO_JOIN_TOKEN=your-token \
  ai-studio/runner:rocm

# NVIDIA
docker run --rm --gpus all \
  -e AI_STUDIO_CONTROLLER=http://your-controller:8420 \
  -e AI_STUDIO_JOIN_TOKEN=your-token \
  ai-studio/runner:cuda

# macOS / no Docker
scripts/install-runner.sh
```

Or bring up everything at once:

```bash
cd docker
echo "AI_STUDIO_JOIN_TOKEN=$(openssl rand -hex 24)" > .env
docker compose up -d
```

### 3. Train something

Open the UI and follow the four steps. Everything technical is chosen for you
from what your hardware measured about itself, and every choice is explained
on screen. When a run finishes, the **Playground** lets you talk to it.

---

## Two ways to make a model

| | Fine-tune | From scratch |
|---|---|---|
| Starts from | a trained model | random noise |
| Trains | ~0.1% of the weights (LoRA) | every weight |
| Realistic size on one GPU | up to ~21B (4-bit) | up to ~100M |
| Time | minutes to hours | hours to days |
| Produces | an adapter file | a complete model |
| The model can | answer, once fine-tuned on instructions | continue text |

**Fine-tuning is what you want almost every time.** From-scratch is included
because it is the only way to actually see what training *is* — and because a
small model trained properly on simple text writes real English, which is a
genuinely surprising thing to watch happen on your own desk.

### The arithmetic that decides whether a from-scratch run is worth starting

A model needs roughly **20 tokens of text per parameter** before it has learned
what its size can hold (Chinchilla). Below that it is undertrained, and a
*smaller* model given the same compute would have been better.

That single ratio is what the size picker shows, computed for your machine's
measured throughput and the time you say you can wait. On an RX 6900 XT:

| Size | Params | Chinchilla budget | Time to reach it |
|---|---|---|---|
| Nano | 5M | 106M tokens | ~8 min |
| Tiny | 14M | 275M tokens | ~48 min |
| Small | 29M | 582M tokens | ~3 hours |
| Base | 91M | 1.8B tokens | ~29 hours |
| Large | 211M | 4.2B tokens | ~6 days |

So the UI does not offer a 1B model, and it marks the largest size that
actually *finishes* in your budget rather than letting you pick the biggest and
find out overnight.

### These numbers are measured, and measuring them corrected two mistakes

The estimates started as reasoning about how a GPU works. Running real jobs
disproved both halves of that reasoning:

**Throughput was four times too pessimistic.** The assumption was that narrow
matmuls underuse the card, so efficiency should climb steeply with width — 6%
of peak at 256 wide, 22% at 1024. Measured on the RX 6900 XT:

| | measured | of peak |
|---|---|---|
| Nano (d=256, seq=256, batch 64) | 222k tokens/s | 22.3% |
| Small (d=512, seq=512, batch 48) | 50k tokens/s | 27.9% |

The direction was right; the magnitude was not. The original guess would have
reported Nano as unable to finish, when it in fact reaches a full Chinchilla
budget in eight minutes.

One measurement was itself misleading, and is worth recording. The same Small
model first measured 20.3%, because that run used batch 64 and sat against the
memory ceiling. A model squeezed into barely enough memory does not fail — it
just runs a third slower. That is a good reason for the batch planner to leave
headroom, and a good reason to distrust a single benchmark.

**Memory was 1.4× too optimistic, and the missing term was the logits.** For a
small model the output layer dominates everything else: one value per token per
vocabulary entry, with about five copies live across the cross-entropy path.
At batch 64 × 256 tokens with an 8k vocabulary that is 2.15 GB, against 0.72 GB
for the entire rest of the run. Omitting it was not a rounding error — the
planner proposed a batch size that died at step one. That is the worst failure
this app can produce, so the estimator now carries the logits term explicitly,
doubles the no-flash attention term (scores *and* softmax are both retained for
the backward pass), and plans against 72% of the card rather than 80%.

### What the from-scratch trainer does differently

Every one of these is a place where reusing the fine-tuning recipe produces a
run that looks fine and learns nothing:

- **Master weights stay float32.** A frozen fp16 base is correct for LoRA
  because it is never updated. A model being trained from noise *is* the thing
  updated, and fp16 weights silently discard every update below one ulp.
  Speed comes from autocasting the matmuls instead.
- **Text is packed, not padded.** Documents are concatenated and sliced into
  equal blocks, so every position in every batch carries a real token.
- **The tokenizer is trained too.** Borrowing Qwen's 151k vocabulary for a
  384-wide model would spend 58M parameters on the embedding table — five times
  the rest of the network. A small purpose-built vocabulary is the correct
  choice, not a compromise.
- **A slice of the corpus is never trained on**, and held-out loss is charted
  beside training loss. It is the only number that can tell you the model is
  memorising rather than learning.
- **The model writes a sample every few dozen steps.** Watching noise become
  words become sentences is the clearest evidence a run is working.

## Playground

Every finished run stays available to chat with. Inference runs on a runner,
never on the controller — that is what keeps the controller GPU-free — so a
message is routed over the fleet websocket and the reply streams back token by
token.

The two kinds of result behave differently, and the interface says which one it
is holding. A fine-tune gets your message wrapped in the exact template it was
trained with (skip that, and it ignores the question and rambles). A model
built from scratch is a base model: it continues text and has never seen a
question in its life, so it is asked for an opening instead.

Runners fetch and cache the artifact from the controller on first use, and
release the GPU after 15 idle minutes so a chat cannot block the next training
run.

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

### Two lessons the probe encodes

Both were measured on an RX 6900 XT, and both would silently ruin a run:

**bfloat16 is a trap on RDNA2.** The card supports it and computes it
correctly — at 12.7 TFLOP/s against 31.6 for float16. Nearly every fine-tuning
guide says "use bf16", which here throws away 60% of the GPU. The runner
benchmarks all three dtypes and recommends from the measurement.

**A GPU library can kill the process, not raise an exception.** Importing the
stock `bitsandbytes` on unsupported ROCm hardware aborts the interpreter at the
HIP level (`SIGABRT`). An in-process capability check would take the agent down
at startup, forever. So risky probes run in a **subprocess**, and results proven
before a crash are recovered from a partial report.

---

## Configuration

| Variable | Where | Meaning |
|---|---|---|
| `AI_STUDIO_DATA` | controller | State directory (default `./data`) |
| `AI_STUDIO_PORT` | controller | UI port (default `8420`) |
| `AI_STUDIO_JOIN_TOKEN` | both | Shared join secret; generated if unset |
| `HF_TOKEN` | both | For gated models (Llama, Gemma) |
| `AI_STUDIO_CONTROLLER` | runner | Controller URL |
| `AI_STUDIO_RUNNER_NAME` | runner | Display name |
| `BNB_ROCM_ARCH` | rocm build | GPU arch, e.g. `gfx1030`, `gfx1100` |

## Security

The join token is the only credential; anyone holding it can attach a machine
and read job data, so treat it as a password. `HF_TOKEN` is set on the
controller and forwarded to runners — it is deliberately **not** editable from
the browser, so a UI session can never read or change your credentials. There
is no user authentication yet: run this on a trusted network, or behind a
reverse proxy that provides auth.

## Project layout

```
controller/   FastAPI app, SQLite, scheduler, HF proxy  (no torch)
runner/       capability probe, websocket agent, trainers
web/          zero-build UI (ES modules, no dependencies)
docker/       controller + rocm/cuda runner images, compose
scripts/      host provisioning, bare-metal runner install
```

## Status

Working end-to-end:

- **LoRA and QLoRA fine-tuning** — browse Hugging Face, preview your data,
  guided setup, live loss charts, cancel mid-run, download the adapter.
- **Training from scratch** — trained tokenizer, packed corpus, held-out loss,
  live text samples, a standalone model with usage instructions in the zip.
- **Playground** — streaming chat with any finished run, template-aware.

Verified on an RX 6900 XT (gfx1030), controller and runner both containerised:

| Check | Result |
|---|---|
| LoRA fp16, SmolLM2-135M | loss 10.89 → 3.01, 0.67 GB |
| QLoRA 4-bit, Qwen2.5-3B | loss 10.74 → 3.01, **3.17 GB** |
| From scratch, 5.3M params, 300 steps | loss 8.999 → 2.904 in **120 s** |
| Held-out loss on the same run | 6.446 → 2.927, tracking training loss |
| Playground, from-scratch model | 214 tokens/s, first token in 0.3 s |
| Playground, second turn | 0.0 s — the loaded model is reused |
| Playground, 3B fine-tune | base + adapter loaded in 11.7 s |
| From scratch, 29M params | 11.5 GB against a 10.5 GB estimate |
| Cancelling a run mid-training | stopped cleanly, runner stayed online |
| Runner killed mid-run | job requeued and restarted automatically |
| Unsupported model / bad ID | refused with a plain-language message |
| Model too large for the GPU | refused at creation, not left queued |
| UI at 1440px and 390px | no JS errors, no horizontal overflow |

The 5.3M model, from noise, on TinyStories:

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

### Known limitations

- **The ROCm runner image is large** (~42 GB on disk) because it builds on
  `rocm/dev-ubuntu-24.04:*-complete`, which is needed to compile bitsandbytes.
  A multi-stage build that compiles the wheel and copies it into a slim runtime
  would cut this substantially.
- **No user authentication.** The join token is the only credential. Put it
  behind a reverse proxy or keep it on a trusted network.
- **One job per runner at a time.** No multi-GPU or multi-job scheduling yet,
  and a runner that is training will not serve the playground.
- **From-scratch tops out around 200M parameters**, which is a compute limit
  rather than an arbitrary one. See the table above.
- **The corpus is held in host RAM** while training (capped at 500M tokens,
  ~1 GB as uint16). Longer runs make repeated passes rather than streaming
  continuously, and warn when that exceeds four passes.

### Next

Vision-model fine-tuning and an evaluation harness.
