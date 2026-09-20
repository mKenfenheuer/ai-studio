# Design notes

Why AI Studio is built the way it is, and what measuring the machine corrected
about the guesses it started from.

These are the long-form notes behind the summary in the [README](../README.md).
They are kept because most of what follows is not obvious from the code: each
section records a decision, the failure that motivated it, and — where the
answer came from running a real job rather than reasoning about one — the
numbers that changed somebody's mind.

---

## Contents

- [The arithmetic behind a from-scratch run](#the-arithmetic-that-decides-whether-a-from-scratch-run-is-worth-starting)
- [Choosing the data](#choosing-the-data)
- [Designing the model yourself](#designing-the-model-yourself)
- [Conversations, roles and templates](#conversations-roles-and-templates)
- [Playground](#playground)
- [Operations](#operations)
- [The assistant, and the chat role](#the-assistant-and-the-chat-role)
- [Two lessons the hardware probe encodes](#two-lessons-the-probe-encodes)
- [Working around a missing attention kernel](#working-around-a-missing-attention-kernel)
- [One card, one reply at a time](#one-card-one-reply-at-a-time)
- [Making response_format a promise](#making-response_format-a-promise)
- [A capability that cannot be measured](#a-capability-that-cannot-be-measured-must-not-be-assumed)
- [One rule in the web UI](#one-rule-in-the-web-ui)

---

## Training from scratch

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

Runners fetch and cache the artifact from the controller on first use. Loaded
models stay resident while there is memory for them and are evicted least
recently used when there is not, so a chat never blocks the next training run
and the second turn does not pay for the first turn's load.

---

## Operations

Training a model and *running* one are different jobs, and the second had no
page. The studio could serve a model over an OpenAI-compatible API and then
had nothing to say about whether it was up, how fast it was answering, how
often it failed, or which key was doing all the work. Those facts existed --
the token counts were the runner's own and were thrown away, the residency was
in every heartbeat and was never read. **Operations** is where they are put.

### Deploying a model to a machine

A model is fetched and loaded the first time somebody talks to it, which for a
7B is a minute or two, and it is dropped again when the card is needed for
something else. That is right for a playground and wrong for the thing a piece
of home automation is pointed at, which pays that minute at random intervals
forever.

A **deployment** pins one model to one machine's card and keeps it there. It
is a row in the database rather than a one-off request, because the interesting
cases are all the ones where "load it once and hope" fails: video memory does
not survive a restart, a training run takes the whole card, and a machine can
simply be away for an afternoon. A reconciler compares what each connected
machine says it is holding against what should be there and loads what is
missing. Nothing is ever unloaded automatically -- a model on a card that
nobody asked for is a warm cache, not a fault.

A deployed model is exempt from the eviction that a passing conversation would
otherwise cause. It is *not* exempt from a training run, which is sized against
an empty card and takes the whole thing; the controller notices and puts the
deployment back when the run is over. If that matters, reserve the machine.

### Reserving a machine

A studio with two cards usually wants one of them answering messages and the
other training, and there was no way to say so: the scheduler handed a run to
whichever machine was idle, which is reliably the machine that was about to be
asked a question. Each machine is now **both** (the default), **reserved for
serving** -- the scheduler skips it and only conversations reach it -- or
**reserved for training**, which is never picked to answer a message.

The reservation is honoured in two places on purpose: the dispatcher skips the
machine, *and* the "why is this run waiting" explanation says that is why. They
disagreed once, and a run sat in a queue in front of a card the page reported
as free.

### What is measured

Every reply is one row in a ledger -- over the API and in the playground alike,
because both are the same cards and the same seconds and only one of them used
to be counted. **Failures are rows too.** Every row used to be a success,
because a failure returned before anything was written, so "is it erroring"
could not be answered at all and every rate computed from the table was
flattered by the calls that never happened.

Tokens per second is a sum divided by a sum, not an average of per-reply rates:
averaging those weights a two-token reply the same as a two-thousand-token one
and reports a fleet considerably faster than it has ever been. Time spent
failing is not time spent generating and is left out of the divisor. Where
there is nothing to divide, the interface draws a dash rather than a zero.

### Keys

Everyone makes their own from their account page. An administrator sees every
key in the studio, whose it is, what it has cost and the button that turns one
off -- which is the part that was missing, because the key hammering a model
at three in the morning is rarely your own.

---

## The assistant, and the chat role

The Playground is a workbench. It lists every run, compares two side by side,
exposes temperature and top-p and the system prompt, and assumes you know what
a fine-tune is. That is right for the person who trained the model and wrong
for the person the model was trained *for* -- the colleague who has been told
"ask the assistant" and has no business seeing four hundred runs, somebody's
dataset, or the button that deletes a machine.

**Assistant** is the other door: the models somebody deliberately published
under a name, a message box, the reasoning shown when the model produces any,
and text files you can attach and ask about. It talks to
`/v1/chat/completions` -- the same endpoint any outside client uses, with the
session cookie instead of a key -- so it cannot drift from what everything else
gets, its usage is counted the same way, and a bug found here is a bug an
integration would have hit too.

A third role, **chat**, is an account that has that page and nothing else. It
is enforced on the server as an *allowlist* of paths rather than a list of
things to forbid: written the other way round, every endpoint added from now on
would be reachable by a chat account until somebody remembered to exclude it.

---

## Two lessons the probe encodes

Both were measured on an RX 6900 XT by the runner's own hardware probe, and
both would silently ruin a run:

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

## Working around a missing attention kernel

The capability probe answers a question the rest of the studio kept asking
badly. Four places read `caps["attention"]["flash"]` and meant "does attention
cost memory with the square of sequence length here" — and those are not the
same question. The memory-efficient kernel is a second implementation of the
same tiling idea, costs the same memory, and runs on hardware flash attention
does not. Reading only the flash flag charged those cards for a quadratic term
they never pay and capped them at 2048 tokens when they can comfortably do
8192. `common/attention.py` now answers the question that was meant, once, and
the four callers ask it rather than each other.

**A card without a kernel may still have one.** On ROCm the kernels come from
AOTriton, and the architectures its wheels compile for are a shorter list than
the ones PyTorch will dispatch to. On the ones in between, torch says no and
the kernel is there — behind `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`,
because AMD has not finished validating it. So a card that reports nothing gets
one more probe with the flag set, and if that one works the variable is
reported alongside the answer and set by every job that runs there. Reporting
`flash: true` from a probe whose environment the job does not reproduce would
be the worst outcome available: an optimistic memory estimate against a
pessimistic kernel.

**Where there is genuinely no kernel, the batch moves and the length does
not.** Attention without fusion keeps `batch × heads × seq × seq` scores and
the softmax over them — quadratic in length, linear in batch. The length is the
thing the dataset needs, so the batch is what gives:

- gradient checkpointing stays on, and is turned back on if it was switched
  off. This is the one setting the fine-tuner overrides rather than honouring,
  and the reason is that it is not the trade it looks like: without
  checkpointing every layer keeps its own scores instead of one being live at
  a time, which multiplies the largest term on the card by the depth of the
  model. That is not "30% faster", it is "does not start".
- the micro-batch falls to what the scores matrix can afford, and gradient
  accumulation rises to match. Four sequences one at a time and four at once
  produce the same gradient, the same optimiser steps and the same schedule.
- the replacement is a **divisor** of the original batch, never just the cap.
  Accumulating eight in groups of three is nine a step, and an effective batch
  that drifts changes the tokens per step, the token budget and the learning
  rate the schedule was written against. `batch × accumulation` comes out
  exactly where it went in.
- serving reads a prompt into the key/value cache 256 tokens at a time instead
  of in one pass, for the same arithmetic in a different order. Where the
  kernel *is* fused that slicing is pure overhead, so the chunk is 2048 there.

What does not happen is the sequence length being quietly shortened. It is
honoured, and the run says what it costs — see the note in
`runner/jobs/lora_llm.py` about the difference between saying and doing.

---

## One card, one reply at a time

A runner holds one model on one card and writes one reply at a time. That is
not going away, so the only real question is what happens to the second
request — and the answer used to be an error. An evaluation sending sixty
prompts got one answer and fifty-nine failures.

**They are queued now, and the message is held rather than the caller.** Both
places a reply is waited for already exist: `waiters` for an API request, the
browser's websocket for the playground. Holding an HTTP connection open to do
the waiting a second time helps nobody, so the controller keeps the request in
`serving_queue[runner_id]` and sends it when the machine frees up. A queued
request is told where it stands straight away, on the same `generate_status`
frame a runner uses for its own progress, because a request that will wait four
minutes and says nothing is indistinguishable from one that was lost — and the
client that cannot tell is the one that retries and makes the queue longer.

**The bound matters more than the queue.** An unbounded queue is not a queue,
it is a way of turning a busy machine into a slow one and then into a timeout.
Past `SERVING_QUEUE_MAX` the honest answer is `429` with a `Retry-After`, and
the status code is the point: `502` tells a client library the upstream is
broken, and the correct response to that is to stop trying.

**Three ways a slot could leak, all closed.** A socket that dies between
choosing a machine and writing to it releases the slot rather than holding it
for a request that was never sent. A machine that disconnects fails everything
it was answering or had queued, with a sentence rather than a timeout. And a
reply that simply never arrives is reaped by `reconcile_serving` after the
runner's own deadline plus two minutes — because a queue that can deadlock
permanently is worse than no queue at all.

Cancelling takes a request *out of the line* if it never left it. Forwarding a
cancel for a request the machine has not been given yet cancels nothing, and
the request would then be sent anyway the moment the slot came free.

### And a field that was refused by code that did not refuse it

`response_format` was accepted and ignored, which is the one failure this API
is written to avoid: a caller that asks for a JSON schema has stopped checking
the reply, so prose returned under it goes wherever the schema was going to go.
It was briefly a `400`, and is now enforced — see the next section.

Writing the check for it turned up that `logprobs` was never refused either.
The guard was a loop over `(None, False, 1)`, where the `1` was meant for `n`
— and `True == 1` in Python, so `logprobs: true` passed the test written to
reject it. The fields are checked one at a time now.

---

## Making response_format a promise

Nothing in a prompt is binding. A model asked for JSON usually writes JSON, and
"usually" is exactly the wrong guarantee for a field whose entire purpose is
that the caller has stopped checking.

What is binding is the sampler. Before each token is chosen, every token that
would break the grammar is scored at negative infinity, so the model picks from
the legal moves. The reply is not checked against the schema afterwards and
never repaired — it could not have been written in another shape. The schema is
*also* put in the prompt, which changes nothing about validity and a great deal
about whether the fields are filled with the answer or with a guess.

**The grammar is a dependency, and the training loop next door is not.** That
looks inconsistent and is not. The training loop is hand-written to escape an
API that churns. A grammar is either correct or it emits `{"a": 01.}` at three
in the morning; JSON's escaping, number syntax and unicode rules are a
well-known source of subtle bugs, and correctness-critical, bounded and already
solved is the shape of problem a dependency is for.

**It still needs a guard, and finding out why is the point of the check.** The
check drives a model that picks *uniformly at random* from whatever the mask
allows — a maximally unhelpful model that would write "Sure! Here's the JSON:"
if it could. Two real failures fell out of it:

- raw tabs and newlines are accepted inside a JSON string, which RFC 8259
  forbids and `json.loads` rejects;
- an internal parser error is handled by allowing only end-of-text, which stops
  the reply wherever it stands — mid-string, if that is where it was.

Failing open is a reasonable default for a library that cannot know what its
caller promised. It is not a reasonable default here. So the finished reply is
parsed, and validated against the schema, before it is handed over; a reply
that fails becomes an error. A reply that is nearly JSON is worth less than an
error, because the error is the only one of the two anybody notices.

The same check also disproved a plausible-looking assertion of mine: `6e9` is a
valid `integer` by JSON Schema, which counts any number with a zero fractional
part. `8e-15` is not, and is refused. The test was wrong, not the code.

---

## A capability that cannot be measured must not be assumed

FlexAttention was briefly switched on automatically, and it crash-looped a
production runner twenty-four times. The mistake is subtle enough to be worth
keeping.

The probe compiled `flex_attention` on the card and it worked. That was taken
as "this machine has fused attention": the comfortable sequence length went
from 2048 to 8192, and every memory estimate was rewritten against a kernel
that never materialises the scores matrix. Then a real model was loaded and
Inductor refused to build the kernel transformers actually uses — `out of
resource: shared memory, Required: 131072, Hardware limit: 65536`, because
RDNA2 has 64 KB of LDS per workgroup and the masked kernel wants 128.

The obvious repair is to probe at a realistic head dimension. It does not
work. Raw `flex_attention` at head_dim 128 compiles; a two-layer Llama through
transformers at the same shape compiles; Qwen2.5-3B does not. What fails is the
block mask a particular architecture builds, and the only probe that predicts
it is loading that model — which is not a probe, it is the thing a probe
exists to avoid.

So it stays off unless a machine is told to try it, and a machine that is told
falls back per model rather than dying. The general rule the whole
`capabilities` module already followed — measure, do not trust — needed one
more clause: **a measurement is only worth what it predicts.**

Two smaller faults fell out of the same incident:

**The fallback caught the wrong exceptions.** `load_base_model` dropped an
optional argument on `TypeError` or `ValueError`. A compiler that refuses to
build a kernel raises `InductorError`, which is neither, so it escaped and
killed the load. Nothing among those optional arguments is worth failing a run
over — each is a performance choice with a working default behind it — so
anything naming one is now treated as that argument being refused, whatever
type it arrives as.

**A deployment that kills the machine was retried for ever.** The state machine
skips a deployment marked `failed`, which covers every way a load can fail
except the one that matters most: a load that kills the runner reports nothing,
so it stays `pending` and is sent again the moment the machine reconnects. It
is capped at three attempts now and then written off, in words that say the
machine did not survive loading it rather than leaving it to be inferred from a
restart count.

---

## One rule in the web UI

`draw()` renders purely from state, and no render may start work that causes
another render synchronously. Every asynchronous load goes through `ensure()`,
which marks itself in-flight *before* awaiting, so the re-render it eventually
triggers finds the work already done rather than starting it again.

This is written down because breaking it is not obvious and not survivable: an
earlier version called `draw()` from inside the click handler that each step
re-ran on render, which recursed until the stack gave out and filled the
console with `too much recursion`.

