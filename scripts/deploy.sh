#!/usr/bin/env bash
# Rebuild and restart the studio, without silently killing a training run.
#
# This exists because of a specific mistake, made twice. `docker compose up -d`
# recreates every service whose image changed -- including the runner -- and a
# runner that is recreated loses whatever it was training. Both times the run
# was hours old, both times it went back to step 1, and both times the person
# typing the command had no idea until afterwards.
#
# So the rule is: find out first, and say what it will cost. Restarting the
# *controller* interrupts nothing -- runners hold their work and dial back in,
# which is by design (see the comment in controller/app.py's runner socket).
# Restarting a *runner* interrupts exactly what it is training, and how much
# that costs now depends on whether there is a checkpoint to resume from.
#
#   deploy.sh                 rebuild and restart everything (guarded)
#   deploy.sh controller      controller only -- safe while training
#   deploy.sh runner          runner only
#   deploy.sh runner-cpu      the GPU-less runner beside the controller
#   deploy.sh status          what is in flight; change nothing
#   deploy.sh <target> --force   do it anyway
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_DIR="${AI_STUDIO_COMPOSE_DIR:-$here/../docker}"
cd "$COMPOSE_DIR"

target="${1:-all}"
force=0
for arg in "$@"; do [[ "$arg" == "--force" ]] && force=1; done

bold() { printf '\033[1m%s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m!!\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31mxx\033[0m %s\n' "$*" >&2; exit 1; }

[[ -f .env ]] || die "No .env in $COMPOSE_DIR. Run scripts/make-env.sh first."
# shellcheck disable=SC1091
set -a; source ./.env; set +a
PORT="${AI_STUDIO_UI_PORT:-8420}"
TOKEN="${AI_STUDIO_JOIN_TOKEN:?AI_STUDIO_JOIN_TOKEN missing from .env}"

# The join token, not a login. The deploy script has no browser session and
# should not need an account; it already holds this credential in order to
# start the containers at all.
inflight="$(curl -fsS --max-time 10 \
              -H "X-Runner-Token: $TOKEN" \
              "http://127.0.0.1:$PORT/api/fleet/in-flight" 2>/dev/null || true)"

if [[ -z "$inflight" ]]; then
  # No answer is not the same as "nothing running". It usually means the
  # controller is already down, which is safe -- but it is a different fact
  # and it gets said rather than assumed.
  warn "The controller did not answer on port $PORT, so nothing could be checked."
  # Restarting the controller interrupts nothing whether or not a run is in
  # progress, so not being able to ask is not a reason to stop. Restarting a
  # runner is the dangerous one, and that is where an unanswered check has to
  # be treated as "assume the worst".
  case "$target" in
    status|controller) : ;;
    *)
      warn "If a runner is training right now, this deploy will interrupt it."
      [[ $force -eq 1 ]] || die "Re-run with --force if you are sure."
      ;;
  esac
  busy=0
else
  busy="$(printf '%s' "$inflight" | python3 -c 'import json,sys; print(1 if json.load(sys.stdin)["busy"] else 0)')"
fi

describe() {
  # The JSON goes in as an argument, not on stdin. `python3 - <<EOF` already
  # uses stdin for the program itself, so piping data into it as well leaves
  # the program reading an empty stream -- which json.load does not survive.
  python3 -c "
import json, sys
d = json.loads(sys.argv[1])
if not d['jobs']:
    print('  nothing is training right now')
for j in d['jobs']:
    step, total = j.get('step') or 0, j.get('total_steps') or 0
    ck = j.get('checkpoint_step') or 0
    cost = ('would resume from step %d, redoing %d' % (ck, step - ck)) if ck \
        else 'has no checkpoint, so it would restart from the beginning'
    print('  %-40s %-10s step %d/%s  %s'
          % (j['name'][:40], j.get('runner') or '?', step, total or '?', cost))
if d.get('queued'):
    print('  %d job(s) waiting in the queue' % d['queued'])
" "$inflight"
}

bold "Studio at http://127.0.0.1:$PORT"
[[ -n "$inflight" ]] && describe

if [[ "$target" == "status" ]]; then
  exit 0
fi

case "$target" in
  controller)
    # --no-deps is the load-bearing flag. Without it compose recreates the
    # runner as a dependency of the controller and undoes the whole point.
    bold "Rebuilding the controller only; runners are left alone."
    docker compose build controller
    docker compose up -d --no-deps controller
    ;;
  runner-cpu|cpu)
    # The GPU-less runner beside the controller: uploads and hosted-model
    # dataset runs. Restarting it interrupts those and nothing else, but
    # "those" includes an upload that is eighteen minutes into fourteen
    # gigabytes, so it asks the same question.
    if [[ "$busy" == "1" && $force -eq 0 ]]; then
      warn "Something is running. Restarting this runner interrupts whatever"
      warn "it is doing -- an upload starts over, rows already written are kept."
      warn "Re-run with --force if that is fine."
      exit 1
    fi
    docker compose build runner-cpu
    docker compose up -d --no-deps runner-cpu
    ;;
  runner|runner-rocm|runner-cuda)
    svc="$target"; [[ "$svc" == "runner" ]] && svc="runner-rocm"
    if [[ "$busy" == "1" && $force -eq 0 ]]; then
      warn "A run is in progress on a runner. Restarting it now interrupts that run."
      warn "Options, in the order worth trying:"
      warn "  * wait for it to finish"
      warn "  * stop it from the UI, keeping the model it has built so far"
      warn "  * re-run with --force -- a run with a checkpoint resumes from it"
      exit 1
    fi
    docker compose build "$svc"
    docker compose up -d --no-deps "$svc"
    ;;
  all)
    if [[ "$busy" == "1" && $force -eq 0 ]]; then
      warn "A run is in progress. Deploying everything would restart the runner."
      warn "Deploy just the controller instead -- that interrupts nothing:"
      warn "    $0 controller"
      warn "Or re-run with --force."
      exit 1
    fi
    docker compose build
    docker compose up -d
    ;;
  *)
    die "Unknown target '$target'. Use: all, controller, runner, runner-cpu, status."
    ;;
esac

bold "Deployed. Waiting for the controller to answer..."
for _ in $(seq 1 30); do
  if curl -fsS --max-time 3 "http://127.0.0.1:$PORT/api/health" >/dev/null 2>&1; then
    bold "Healthy."
    docker compose ps
    exit 0
  fi
  sleep 2
done
die "The controller did not come back within a minute. Check: docker compose logs controller"
