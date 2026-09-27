# Cloud GPUs (RunPod)

When a GPU run is queued and no machine of the studio could take it, the studio
can rent one from RunPod, use it for the run, and give it back. It is off until
an administrator turns it on under **Administration → Cloud GPUs**.

## What happens

1. **A run waits and nothing can take it.** A GPU run (fine-tune, pretraining,
   vision/audio/diffusion) is queued, it may use the cloud, it is not pinned to a
   machine, and no *online* machine of the studio could run it — busy or not; a
   busy one that could is worth waiting for.
2. **The studio picks a GPU.** From RunPod's live catalogue: in stock, under the
   price cap, allowed by the run's cloud setting (Secure Cloud only, or Secure or
   Community), and with the memory the run needs — the same fit check the
   scheduler uses for your own machines. Then the fastest (or, if set, the
   cheapest) of those. The run's log says which GPU, at what price, and a rough
   estimate of hours and dollars.
3. **The pod joins as a runner.** It starts the CUDA runner image with the join
   token and a name (`ai-studio-cp_…`) that ties it to its pod record, dials in
   like any machine, and takes the run. It is marked for training only, so it is
   never pinned busy by serving.
4. **Idle, it is given back.** After the idle time with nothing queued it could
   run, the pod is drained: the checkpoints of unfinished runs are uploaded to the
   studio (finished models already are, at the end of every run), then the pod is
   deleted and its runner forgotten. Base models and caches are not kept — they
   can be downloaded again, and storage would make an idle pod cost money after
   all.

A run whose checkpoint the studio holds is not bound to any machine: whichever
takes it next downloads the checkpoint and carries on.

## The brakes

| Setting | Default | What it does |
|---|---|---|
| Max price per GPU-hour | $0.80 | Only GPUs at or under this price are rented. |
| Pods at once | 1 | Never more running (or starting) at the same time. |
| Daily budget | $10 | No pod starts once today's estimated spend would pass it. |
| Pause running pods at the cap | on | At the cap, running runs are paused (checkpointed, back in the queue) and their pods drained. |
| Give back after idle | 15 min | Idle this long with nothing it could run, a pod is drained and deleted. |
| Longest life per pod | 12 h | A pod is paused and drained after this, busy or not. 0 for no limit. |
| Runs may use | Secure Cloud only | The default for runs; each run can choose `never`, `any` or `secure` (`config.cloud`). |

Spend is estimated by the studio from each pod's hourly price, counted from the
moment the pod is requested — on the high side on purpose. RunPod's own billing
is the exact figure.

A pod that never connects within 25 minutes (image pull, a broken host) is
deleted. A pause keeps the run's progress up to its last checkpoint (every ten
minutes); a drain that cannot move a checkpoint within 45 minutes deletes the pod
anyway, and that run starts over.

## Setting it up

1. Create a RunPod API key with permission to manage pods, and paste it in
   **Administration → Cloud GPUs**. It is stored encrypted and never shown again.
2. Make sure a pod on the internet can reach the studio: `AI_STUDIO_PUBLIC_URL`,
   or the "Studio address for pods" field. Runners dial out, so only the studio's
   HTTPS address needs to be reachable — nothing on the pod is exposed.
3. The runner image defaults to `ghcr.io/mkenfenheuer/ai-studio-runner:cuda`
   (CUDA 12.8; RunPod hosts with an older driver are excluded automatically).
   Prefer a dated tag (`cuda-2026.09.27-02fa20f`) over `cuda`: a host that has
   pulled `cuda` before may start the copy it already has.
4. Turn on **Rent automatically**. In the run wizard, **Cloud GPU (RunPod)** then
   appears as a machine: the run is planned for the largest GPU on offer under
   the cap and sent to a rented GPU — never to one of the studio's own machines,
   even one that is free — and the manager rents the GPU that fits it best.

### The pod template

Pods are started from a RunPod pod template the studio keeps in step with these
settings: the image, the container disk, the volume at `/data`, the studio's
address, SSH and Jupyter off. It is found by name (**RunPod template**,
`ai-studio` by default), so one made by hand in the console is taken over and
brought in line rather than duplicated. Saving the settings updates it, as does
**Sync now** and every pod start; a template that already matches is left alone.

Nothing secret goes in it. A template is readable in the RunPod console, so the
join token — and each pod's runner name — are passed with each pod instead, and
a hand-made template that carried the token has it removed. Starting a pod from
the template by hand therefore needs `AI_STUDIO_JOIN_TOKEN` added to that pod.

## Privacy

A rented runner downloads what the run needs: the base model and the dataset.
Consider what goes to a third-party host before allowing Community Cloud, whose
machines belong to individuals rather than to RunPod; Secure Cloud is the
default for that reason.

## API

Administrators only, except `/api/cloud/machine`.

| | |
|---|---|
| `GET /api/cloud` | settings (never the key), today's spend, recent pods |
| `PUT /api/cloud/settings` | change settings; `api_key` to set, `clear_api_key` to remove |
| `POST /api/cloud/template` | create or update the pod template now |
| `GET /api/cloud/gpus?cloud=any\|secure&job=<id>` | what can be rented now; with a job, whether it fits and a cost estimate |
| `POST /api/cloud/pods` | start a pod by hand: `{"gpu_id", "cloud"}` |
| `POST /api/cloud/pods/{id}/drain` | give a pod back the careful way |
| `DELETE /api/cloud/pods/{id}` | delete at once, without moving checkpoints |
| `GET /api/cloud/machine` | the cloud as one machine, for the wizard |

RunPod's REST API v2 is used (`api.runpod.io/v2`); v1 retires on 15 November 2026.
