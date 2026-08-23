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

## Choosing the data

The step before training shows **the exact text the model will be trained on**
— not the raw columns, but the finished string after the instruction template
or chat flattening has been applied. It is rendered by importing the same
function the runner uses to build its batches, from `common/formatting.py`, so
a preview cannot drift from reality.

Three things it catches before a run starts rather than an hour in:

- **Datasets that are several datasets.** `load_dataset` refuses to guess
  between configurations and fails with *"Config name is missing. Please pick
  one among the available configs"*. The configurations are discovered up
  front, from the datasets-server where it indexes the dataset and from the
  repository's own card where it does not — the latter matters, because the
  viewer answers 501 for a great many datasets.
- **The wrong column.** Rows that have content the chosen columns cannot reach
  are counted and shown, because they would be silently skipped.
- **Blank rows, which are not a problem.** Line-oriented corpora like WikiText
  are full of empty lines. Reporting those as failures would be a false alarm
  on half the dataset, so empty and unreadable are counted separately and only
  one of them is an error.

### Your own files, and turning them into training data

The dataset library takes files as they actually arrive, not as a trainer
wishes they had been written:

| Comes in as | Becomes |
| --- | --- |
| `.jsonl`, `.json` | rows, including a JSON array, `{"rows": [...]}`, or a dict of columns |
| `.csv`, `.tsv` | rows, with the delimiter sniffed and the header row detected rather than assumed |
| `.txt`, `.md` | rows cut by line, by paragraph, by document, or into fixed-size chunks with overlap |
| `.html` | the text, with scripts, styles and tags removed |
| `.docx` | its paragraphs — read out of the zip with the standard library, so nothing extra is installed |
| `.zip` | every readable file inside it, each row tagged with the file it came from |
| several files at once | one dataset, with a `source` column |

`.pdf` works only if `pypdf` happens to be installed on the controller;
otherwise it says so rather than importing gibberish. `.parquet` and `.xlsx`
are refused with the shorter route named — the Hub import for one, "save as
CSV" for the other.

How a text file is cut into rows is a choice, not a guess, because the same
file is a corpus of one-line examples or a book depending on what you meant.
The default reads the file and picks: paragraphs where there are blank lines
between them, otherwise one row per line.

Once a dataset is in, every operation writes a **new** dataset and records
what it did, so nothing is destructive and the lineage is visible: drop empty
rows, remove duplicates, filter by length or by regex, sample, shuffle, split
off a validation slice, merge several, rewrite as chat turns — and rearrange
the columns. That last one is what makes an arbitrary spreadsheet trainable:
rename a column to a name the trainer knows, or build one out of the others
with a template like `Q: {question}\nA: {answer}` and drop the rest.

### The workbench

Rows are shown as a **table**, because reading data as JSON is reading it
through a keyhole: you cannot compare two rows, you cannot see that a column
is empty in half of them, and the punctuation outweighs the content. Columns
become columns; a value that is a conversation or a tool schema is summarised
in place ("4 messages") and opens in full on a click. One toggle switches to
what the trainer actually reads.

From the same panel: search, page, filter by split, select rows and delete
them or move them to another split, open one row and correct a field, and add
new rows by hand. Those change the dataset in place — curating data is the
work, and requiring a derived copy to fix four bad rows is how a library fills
up with near-identical datasets nobody can tell apart. Every such edit is
recorded in the dataset's own history.

Everything else still writes a **new** dataset, and now shows you what it
would do first: **Show me what it would do** runs the same code over a sample
and reports how many rows would survive, which steps ran, and the first rows
after — before anything is created.

Columns can be **calculated**, one per line, `name = template`:

```
who  = {first|title} {last|title}
text = Q: {question}
A: {answer|trim}
```

Any `{column}` is replaced with that row's value, so this is how columns are
concatenated and how a spreadsheet becomes trainable. Values pass through
`|` filters — `upper`, `lower`, `title`, `trim`, `lines`, `first`, `last`,
`len`, `words`, `json`, `slice:0:200` — a short list of verbs rather than an
expression language, because a spreadsheet's worth of functions in a text box
is a programming language nobody wrote documentation for. A column can also be
**split** into several (`name -> first, last  on  ,`), and columns can be
renamed or dropped, in that order: dropped after the calculations, so a column
a template reads from can still be thrown away.

### Splits

A dataset holds all of its splits together, with each row naming the split it
belongs to. Importing from the Hub brings in **every split, in full** unless
you say otherwise — the old default of "the train split, first 5,000 rows" was
a decision disguised as a default, and it silently left behind the test split
that makes a held-out score mean anything. Tick fewer splits if you want
fewer; they arrive as one dataset rather than three; uploading, say which split the file is, and
add the test set to the dataset the training data is already in. Every screen
that reads rows can be pointed at one split — the row browser, the training
preview, and the trainer itself.

One file with a `split` column rather than a file per split: a dataset here is
JSONL that streams, appends and can be opened in an editor, and separate files
would buy nothing a column does not while costing every reader a directory
walk. The counts are taken when the file is written, so no page has to count
two million rows to draw a badge.

### Writing a dataset with a model somebody else hosts

The generator can drive a **hosted** model instead of a local one: OpenAI,
**Azure OpenAI**, Anthropic, or any service that speaks the OpenAI shape —
Together, Groq, OpenRouter, Mistral, a vLLM or Ollama server on your own
network, or another AI Studio. This is the one job where paying per token is
obviously right: a large model writes the dataset, and your own small model is
trained on what it wrote.

Keys are per account, encrypted at rest, never returned to the browser, and
attached to a run at the moment it is created — the same rule the Hugging Face
token follows, and it means "whose key paid for this dataset" has an answer.
Connect one on your account page; the Test button asks the model to say hello
and reports exactly what came back, because finding out on row 1 of 5,000 is
the failure mode worth designing against.

Three request shapes cover it — OpenAI's, Azure's (deployment in the URL, key
in `api-key`, version on the query string), and Anthropic's (`/v1/messages`, a
system field of its own, content blocks in the reply) — and they live in one
module that performs no I/O, so the controller's test button and the runner's
generation loop cannot disagree about what a provider expects. Rate limits are
waited out rather than fatal; a rejected key stops the run immediately rather
than spending five thousand attempts discovering the same thing.

## Designing the model yourself

The five sizes are a starting point, not a ceiling. "Design it yourself" opens
layers, width, attention heads, context length and feed-forward width, and
recomputes the parameter count, memory, batch size, step count and time
estimate as you type. Every training hyperparameter is editable on the review
step, each with a sentence explaining what it does.

A transformer has combinations that are silently wrong rather than loudly
broken, and a from-scratch run is far too slow to find them by trying. So
everything is checked against the machine as it is typed, at three levels —
`error` cannot start, `warn` will run and disappoint, `info` is a trade-off
worth knowing:

| Checked | Because |
|---|---|
| Width divides by heads | Attention splits the width evenly; it has to divide exactly |
| Head dimension is 32/64/128 | Other sizes fall back to a slower attention path |
| Width is a multiple of 64 | Otherwise part of every matrix tile sits idle |
| Depth against width | Deep and narrow trains slowly and destabilises; wide and shallow cannot compose |
| Feed-forward ratio | Below ~1.5x starves where most of the capacity lives |
| Embedding share | Above half the model, the vocabulary is eating the network |
| Context vs attention kernel | Without a fused kernel, memory grows with the square of it |
| Fits in VRAM | Checked at the chosen batch, then at batch 1 before refusing |
| Tokens per parameter | The Chinchilla ratio, against your actual time budget |
| Learning rate vs width | Scales as 1/width; 3x over is flagged, 4x under too |
| Tokens per step | Below ~16k the gradient is too noisy to follow |
| Warmup, weight decay, clipping | Ranges that will run and should not |

Every message names the field, says what is wrong, and suggests a specific
fix. Where the obvious fix is useless it says something else instead: a width
of 577 has no sensible divisor, so rather than advising "try 1 head" it
suggests a width of 576.

## Conversations, roles and templates

Conversation datasets are the awkward case, and there is no single format for
them. `content` may be a string or a list of typed parts; roles may be under
`role` or `from`; a tool-calling dataset carries `tool_calls` on assistant
turns and a `tools` schema in its own column. All of it is normalised to one
shape in `common/formatting.py` before any template sees it, so system, user,
assistant and tool turns survive intact — including the tool calls.

Then a **Jinja template** turns that into training text. Three sources:

| | |
|---|---|
| **The model's own** | Read from the base model's `tokenizer_config.json`. Almost always right: every instruct model was trained to expect one exact layout and ships it. |
| **Plain and readable** | Roles written out as text. The correct choice for a base model, which has no format of its own. |
| **Your own** | A Jinja template given `messages`, `tools` and every column of the row. Sandboxed, and compile and render errors are reported against real rows rather than swallowed. |

**All three work for fine-tuning and for training from scratch.** A model built
from nothing can learn a conversation format at the same time as it learns the
language — it simply learns whichever shape it is shown.

## Playground

Every finished run stays available to chat with. Inference runs on a runner,
never on the controller — that is what keeps the controller GPU-free — so a
message is routed over the fleet websocket and the reply streams back token by
token.

**The Playground speaks the shape the model was taught, never one of its own.**
Every run records the exact format it trained with, and the Playground sends
the conversation back through that same format. Getting this wrong is how a
perfectly good fine-tune comes to look broken: give a model a layout it has
never seen and it ignores most of what it learned. Three interfaces follow from
the recorded format:

- **chat** — roles, a system prompt, and turns that accumulate as context.
- **instruct** — one question at a time, wrapped in its training template,
  with no memory of earlier ones, because that is how it was trained.
- **continue** — a base model, which continues text and has never seen a
  question.

The **system prompt** the model trained with is offered back, because a model
fine-tuned with one behaves noticeably worse without it and nobody writes it
down. It is recorded when the run is created, and recovered from the run's own
dataset when it was not — so runs made before this existed, or through the API,
get theirs back too. "What the model is actually being sent" shows the finished
prompt after templating, for when the answer is not what you expected.

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
common/       message normalising, Jinja templating and prompt building --
              shared, so the preview, the trainer and the playground cannot
              render the same conversation three different ways
controller/   FastAPI app, SQLite, scheduler, HF proxy  (no torch)
runner/       capability probe, websocket agent, trainers, inference host
web/          zero-build UI (ES modules, no dependencies)
docker/       controller + rocm/cuda runner images, compose
scripts/      host provisioning, bare-metal runner install
```

### One rule in the web UI

`draw()` renders purely from state, and no render may start work that causes
another render synchronously. Every asynchronous load goes through `ensure()`,
which marks itself in-flight *before* awaiting, so the re-render it eventually
triggers finds the work already done rather than starting it again.

This is written down because breaking it is not obvious and not survivable: an
earlier version called `draw()` from inside the click handler that each step
re-ran on render, which recursed until the stack gave out and filled the
console with `too much recursion`.

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
| Hand-designed 448x7, 20M params | trained from the designer end to end |
| Multi-config dataset | configurations listed from the card when the viewer 501s |
| Tool-calling chat dataset | 4 roles, 19 tools, tool calls preserved end to end |
| Qwen 0.5B on that dataset | loss 2.246 → 1.392 with the model's own template |
| Deleting a run | rows, model file and each runner's cached copy |
| Cancelling a run mid-training | stopped cleanly, runner stayed online |
| Runner killed mid-run | job requeued and restarted automatically |
| Runner killed after upload | run kept as finished, not restarted from noise |
| Second job while one is training | queued once, not re-offered every tick |
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
  and a runner that is training will not serve the playground. A second job
  waits on the queue and is told so once, rather than being offered to the busy
  machine every five seconds.
- **Training learns from the whole conversation**, including the system prompt
  and the user's turns, rather than masking the loss to assistant replies only.
  Standard, and it works; masking would squeeze more out of the same data.
- **From-scratch tops out around 200M parameters**, which is a compute limit
  rather than an arbitrary one. See the table above.
- **The corpus is held in host RAM** while training (capped at 500M tokens,
  ~1 GB as uint16). Longer runs make repeated passes rather than streaming
  continuously, and warn when that exceeds four passes.

### Next

Vision-model fine-tuning and an evaluation harness.
