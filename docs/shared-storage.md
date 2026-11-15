# Shared storage between controller and runners

By default a finished model travels over HTTP. The runner packs it into its
working directory, uploads it, and the controller writes it again. A runner
that needs a model back downloads the zip next to the folder it unpacks into.
For a 6 GB merged model that means 12 GB on disk at the end of a run. On one
host with a nearly full volume, that peak is enough to fail a run whose
training had already finished.

With `AI_STUDIO_SHARED_DIR` set, the controller and its runners share one
directory:

| Path | Written by | Content |
|---|---|---|
| `artifacts/` | controller | Every run's result, as before in `<data>/artifacts` |
| `incoming/` | runner | Results being packed; moved into `artifacts/` when complete |
| `hf/` | runners | The Hugging Face cache (`HF_HUB_CACHE`), one copy for all runners |
| `.ai-studio-share` | controller | A random id proving the directory is this controller's |

A runner packs its result straight into `incoming/` and tells the controller,
which moves it into `artifacts/` by rename. No bytes travel. A runner that
needs a model unpacks it from `artifacts/` in place, with no download and no
second zip.

## Setting it up

The compose files in `docker/` mount one volume, `studio-shared`, at `/shared`
in every container and set `AI_STUDIO_SHARED_DIR=/shared`. On a single host
nothing else is needed.

For runners on other machines, make the volume a network share they mount too,
for example NFS:

```yaml
volumes:
  studio-shared:
    driver: local
    driver_opts:
      type: nfs
      o: addr=nas.lan,nfsvers=4.1,rw
      device: ":/export/ai-studio"
```

A native runner sets `AI_STUDIO_SHARED_DIR` to wherever it mounts the same
export.

## Fallback

At registration the controller sends the id from `.ai-studio-share`. A runner
uses the share only when it reads the same id from its own mount. A missing
mount, a read-only one or a different directory all fall back to HTTP, and the
runner logs why. A controller that does not accept a handed-over file gets it
over HTTP instead.

## Existing installs

At startup the controller moves the zips from `<data>/artifacts` into the share
by rename. Files on another filesystem stay where they are and are still found
there. Runners download those over HTTP. Inside Docker, two volumes are
different mounts even on one disk, so after the first start with the share,
move them once on the host while the controller is stopped:

```bash
docker compose stop controller
mv /var/lib/docker/volumes/docker_studio-data/_data/artifacts/* \
   /var/lib/docker/volumes/docker_studio-shared/_data/artifacts/
docker compose up -d controller
```

## Limits

Runners never evict from a Hugging Face cache inside the share, because they
cannot see which base models the other runners are using. Size the share
for the base models you use and clean `hf/` by hand when needed.
Zips in `incoming/` that were never handed over are deleted after a day.
