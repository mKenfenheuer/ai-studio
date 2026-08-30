#!/usr/bin/env bash
# Build the studio's images and push them to a registry, so somebody who wants
# to run AI Studio can `docker pull` it instead of cloning this repo.
#
# Everything here is also done by docker/docker-compose.yml, but locally and
# under local names (`ai-studio/controller:latest`). Those names cannot be
# pushed anywhere — a Docker Hub repository is `<account>/<name>` — so this
# script builds, retags into the account, and pushes.
#
#   scripts/publish-images.sh                 controller + all three runners
#   scripts/publish-images.sh controller      just the console
#   scripts/publish-images.sh cpu cuda        two of the runners
#   scripts/publish-images.sh --dry-run all   say what it would do
#   scripts/publish-images.sh --deployed all  push what compose already built
#
# --deployed is the one to use on a server. It retags the local
# `ai-studio/…` images that deploy.sh built and pushes those, instead of
# building a second copy of thirty gigabytes of ROCm beside the first on a
# disk that does not have room for it. What goes out is then exactly what is
# running, which is usually what you wanted anyway.
#
# The credential is read from the environment and never written to disk here:
#
#   DOCKERHUB_TOKEN=dckr_pat_… scripts/publish-images.sh
#
# and if it is missing and this is a terminal, it is asked for. Use an access
# token from https://app.docker.com/settings/personal-access-tokens with
# Read & Write scope, not the account password.
#
# A note on "the Windows runner": there isn't one, and that is not an
# oversight. Docker Desktop on Windows runs *Linux* containers on a WSL2
# kernel, so a Windows machine with an NVIDIA card runs the `cuda` image
# below, unchanged. A true Windows-container image would need a Windows build
# host — this script cannot produce one from a Mac or a Linux box.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/.." && pwd)"
cd "$root"

REGISTRY="${DOCKERHUB_REGISTRY:-docker.io}"
ACCOUNT="${DOCKERHUB_USER:-mkenfenheuer}"
CONTROLLER_REPO="${AI_STUDIO_CONTROLLER_REPO:-ai-studio-controller}"
RUNNER_REPO="${AI_STUDIO_RUNNER_REPO:-ai-studio-runner}"

bold() { printf '\033[1m%s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m!!\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31mxx\033[0m %s\n' "$*" >&2; exit 1; }

dry=0
local_only=0
targets=()
for arg in "$@"; do
  case "$arg" in
    --dry-run) dry=1 ;;
    --deployed) local_only=1 ;;
    # Two expressions rather than `^# \?`, which BSD sed does not understand
    # and silently leaves the hashes on.
    -h|--help) sed -n '2,/^set -euo/p' "$0" | grep '^#' \
                 | sed -e 's/^# //' -e 's/^#$//'; exit 0 ;;
    -*) die "Unknown option $arg" ;;
    *) targets+=("$arg") ;;
  esac
done
[[ ${#targets[@]} -eq 0 ]] && targets=(all)
[[ " ${targets[*]} " == *" all "* ]] && targets=(controller cpu cuda rocm)

# Every push carries two tags: a moving one people will actually pull, and an
# immovable one so "which build is this?" has an answer six months later. The
# second is the commit, because that is the only thing that identifies a build
# of a repo with no release tags.
#
# Both questions have to be asked in that order. A deployed copy of the source
# is rsynced without its .git, so `git diff` there fails the same way a dirty
# tree does -- and every image pushed from the server came out tagged
# "unknown-dirty", which says the opposite of the truth about a clean release
# build.
sha="unknown"
dirty=""
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  sha="$(git rev-parse --short HEAD)"
  git diff --quiet || dirty="-dirty"
fi
VERSION="${AI_STUDIO_VERSION:-$(date +%Y.%m.%d)-${sha}${dirty}}"
[[ -n "$dirty" ]] && warn "Working tree has uncommitted changes; tagging $VERSION"
[[ "$sha" == unknown ]] && warn "Not a git checkout, so the tag cannot name a commit."

# ---- what each target is ------------------------------------------------
# A function rather than an associative array, because macOS still ships bash
# 3.2 and `declare -A` is a syntax error there — and a Mac is where this is
# most likely to be run from.
#
#   name -> "<dockerfile>|<repo>|<moving tag>|<what compose calls it locally>"
image_spec() {
  case "$1" in
    controller) echo "docker/Dockerfile.controller|$CONTROLLER_REPO|latest|ai-studio/controller:latest" ;;
    cpu)        echo "docker/Dockerfile.runner.cpu|$RUNNER_REPO|cpu|ai-studio/runner:cpu" ;;
    cuda)       echo "docker/Dockerfile.runner.cuda|$RUNNER_REPO|cuda|ai-studio/runner:cuda" ;;
    rocm)       echo "docker/Dockerfile.runner.rocm|$RUNNER_REPO|rocm|ai-studio/runner:rocm" ;;
    *)          return 1 ;;
  esac
}

for t in "${targets[@]}"; do
  image_spec "$t" >/dev/null \
    || die "No such image '$t'. Known: controller cpu cuda rocm"
done

command -v docker >/dev/null || die "docker is not on PATH"

# ---- which processor these are for --------------------------------------
# The thing this most easily gets wrong: an Apple Silicon Mac builds arm64 by
# default, and an arm64 image pushed as `latest` is unrunnable on every server
# anyone will actually deploy it to. So the target is stated, not inherited,
# and when it differs from this machine we go through buildx.
#
# The runner images cannot be built anywhere but amd64 in practice — their
# bases (nvidia/cuda, rocm/dev) publish no arm64 variant at all — so emulation
# does not rescue you there; build those on a Linux x86 box.
PLATFORM="${AI_STUDIO_PLATFORM:-linux/amd64}"
native="linux/$(docker info --format '{{.Architecture}}' 2>/dev/null | sed 's/^x86_64$/amd64/; s/^aarch64$/arm64/')"
cross=0
if [[ $local_only -eq 1 && "$PLATFORM" != "$native" ]]; then
  # Nothing is being built, so the platform is whatever the local image
  # already is -- and on a server that is the platform it is running on.
  PLATFORM="$native"
fi
if [[ $local_only -eq 0 && "$PLATFORM" != "$native" ]]; then
  cross=1
  docker buildx version >/dev/null 2>&1 \
    || die "Building $PLATFORM on a $native machine needs docker buildx, which is not installed."
  warn "Cross-building $PLATFORM on $native. This is slow, and the CUDA and ROCm"
  warn "bases have no arm64 build at all — build the runners on an x86 host."
fi

bold "Publishing to $REGISTRY/$ACCOUNT as $VERSION"
for t in "${targets[@]}"; do
  IFS='|' read -r _ repo tag local_tag <<<"$(image_spec "$t")"
  printf '   %-12s %s → %s/%s:%s\n' "$t" \
    "$([[ $local_only -eq 1 ]] && echo "$local_tag" || echo "(build)")" \
    "$ACCOUNT" "$repo" "$tag"
done
echo

if [[ $dry -eq 1 ]]; then bold "--dry-run: stopping here."; exit 0; fi

# ---- sign in ------------------------------------------------------------
# Only if we are not already signed in as this account. `docker login` writes
# the credential to ~/.docker/config.json, which is the caller's business and
# not ours to clobber if they have already set it up.
logged_in=0
if grep -qs "\"$REGISTRY\"\|index.docker.io" "${DOCKER_CONFIG:-$HOME/.docker}/config.json" 2>/dev/null \
   && [[ -z "${DOCKERHUB_TOKEN:-}" ]]; then
  bold "Using the Docker credentials already on this machine."
else
  if [[ -z "${DOCKERHUB_TOKEN:-}" ]]; then
    [[ -t 0 ]] || die "DOCKERHUB_TOKEN is not set and there is no terminal to ask on."
    read -r -s -p "Docker Hub access token for $ACCOUNT: " DOCKERHUB_TOKEN; echo
  fi
  # Through stdin, so the token never appears in the process list or in the
  # shell history of whoever is watching over your shoulder.
  printf '%s' "$DOCKERHUB_TOKEN" | docker login "$REGISTRY" -u "$ACCOUNT" --password-stdin \
    || die "Docker Hub rejected the token."
  logged_in=1
fi
unset DOCKERHUB_TOKEN

# Sign out again on the way out, but only if this script is what signed in.
cleanup() { [[ $logged_in -eq 1 ]] && docker logout "$REGISTRY" >/dev/null 2>&1 || true; }
trap cleanup EXIT

# ---- build and push -----------------------------------------------------
for t in "${targets[@]}"; do
  IFS='|' read -r dockerfile repo tag local_tag <<<"$(image_spec "$t")"
  moving="$REGISTRY/$ACCOUNT/$repo:$tag"
  pinned="$REGISTRY/$ACCOUNT/$repo:$tag-$VERSION"

  labels=(
    --label "org.opencontainers.image.revision=$sha"
    --label "org.opencontainers.image.source=https://github.com/$ACCOUNT/ai-studio"
    --label "org.opencontainers.image.version=$VERSION"
  )

  if [[ $local_only -eq 1 ]]; then
    docker image inspect "$local_tag" >/dev/null 2>&1 \
      || die "$local_tag is not on this machine. Run scripts/deploy.sh first, or drop --deployed."
    bold "── retagging $local_tag"
    docker tag "$local_tag" "$moving"
    docker tag "$local_tag" "$pinned"
    bold "── pushing $moving"
    docker push "$moving"
    docker push "$pinned"
    continue
  fi

  bold "── building $t for $PLATFORM"
  # The runner images are ten to twenty gigabytes of CUDA and torch. The first
  # push of each takes a long while; later ones reuse the layers that did not
  # change, which is usually only the last two.
  if [[ $cross -eq 1 ]]; then
    # buildx pushes from the build itself: a cross-built image is not loaded
    # into the local daemon, so there would be nothing here to `docker push`.
    docker buildx build --platform "$PLATFORM" -f "$dockerfile" \
      -t "$moving" -t "$pinned" "${labels[@]}" --push .
  else
    docker build -f "$dockerfile" -t "$moving" -t "$pinned" "${labels[@]}" .
    bold "── pushing $moving"
    docker push "$moving"
    docker push "$pinned"
  fi
done

bold "Done. Pull with:"
for t in "${targets[@]}"; do
  IFS='|' read -r _ repo tag <<<"$(image_spec "$t")"
  echo "   docker pull $ACCOUNT/$repo:$tag"
done
