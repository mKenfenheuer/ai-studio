#!/usr/bin/env bash
# Install an AI Studio runner directly on this machine, without Docker.
#
# This is the required path on macOS (Metal has no container story) and useful
# on any box where you would rather not run Docker. It detects the hardware and
# installs the matching PyTorch build -- picking the wrong wheel is the single
# most common way to end up silently training on the CPU.
set -euo pipefail

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31mxx\033[0m %s\n' "$*" >&2; exit 1; }

PREFIX="${AI_STUDIO_PREFIX:-$HOME/.ai-studio}"
BACKEND="${1:-auto}"

# ------------------------------------------------------------ detection ----
if [[ "$BACKEND" == "auto" ]]; then
  if [[ "$(uname -s)" == "Darwin" ]]; then
    [[ "$(uname -m)" == "arm64" ]] && BACKEND=metal || BACKEND=cpu
  elif command -v nvidia-smi >/dev/null 2>&1; then
    BACKEND=cuda
  elif [[ -e /dev/kfd ]]; then
    BACKEND=rocm
  else
    BACKEND=cpu
    warn "No GPU detected. Installing the CPU build -- fine for trying things"
    warn "out, far too slow for real training."
  fi
fi

case "$BACKEND" in
  cuda)  INDEX="https://download.pytorch.org/whl/cu124";   EXTRA="bitsandbytes>=0.43" ;;
  rocm)  INDEX="https://download.pytorch.org/whl/rocm6.4"; EXTRA="" ;;
  # bitsandbytes ships a macOS arm64 wheel from 0.50, and its 4-bit path
  # measures correct on Metal -- about 0.11 relative error against float16,
  # the same as a working CUDA card, with the weights really held at a
  # quarter of the size. Left out, an Apple machine reported "no 4-bit" and
  # sized itself for 16-bit only, which is roughly a quarter of the model it
  # can actually fine-tune. The 8-bit optimiser has no Metal kernel and the
  # probe finds that by itself.
  metal) INDEX="";                                          EXTRA="bitsandbytes>=0.50" ;;
  cpu)   INDEX="https://download.pytorch.org/whl/cpu";     EXTRA="" ;;
  *) die "Unknown backend '$BACKEND'. Use one of: cuda rocm metal cpu" ;;
esac
log "Installing runner for backend: $BACKEND"

# --------------------------------------------------------------- python ----
PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null || die "python3 not found."
PYV="$($PY -c 'import sys;print("%d.%d"%sys.version_info[:2])')"
[[ "$(printf '%s\n3.10\n' "$PYV" | sort -V | head -1)" == "3.10" ]] \
  || die "Python 3.10+ required, found $PYV."

log "Creating environment at $PREFIX"
mkdir -p "$PREFIX"
"$PY" -m venv "$PREFIX/venv"
PIP="$PREFIX/venv/bin/pip"
"$PIP" install --quiet --upgrade pip wheel

log "Installing PyTorch"
if [[ -n "$INDEX" ]]; then
  "$PIP" install --quiet torch --index-url "$INDEX"
else
  "$PIP" install --quiet torch      # macOS wheels include Metal support
fi

log "Installing training libraries"
# shellcheck disable=SC2086
# Two of these are here to keep a natively installed runner level with the
# container images, because the runner advertises what it can take from the
# libraries it finds -- so a missing package is not a smaller install, it is a
# machine that silently never gets offered a kind of work.
#
#   Pillow            decodes images for a vision run. Without it the machine
#                     reports "text only" and no vision job is ever sent.
#   lm-format-enforcer  constrained decoding behind `response_format`, with
#   + jsonschema        jsonschema checking the finished reply. Without them
#                     the controller refuses those requests in words.
"$PIP" install --quiet \
  "transformers>=4.44" "peft>=0.12" "accelerate>=0.34" "datasets>=2.20" \
  "safetensors>=0.4" "websockets>=12" "httpx>=0.27" "Pillow>=10" \
  "lm-format-enforcer>=0.10" "jsonschema>=4.0" numpy $EXTRA

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
log "Installing AI Studio runner from $SRC"
"$PIP" install --quiet -e "$SRC" --no-deps

if [[ "$BACKEND" == "rocm" ]] && ! command -v rocminfo >/dev/null 2>&1; then
  warn "rocminfo is not installed. The runner can still train, but device"
  warn "detection will be less accurate. Install it with scripts/provision-host.sh."
fi

echo
log "Checking what this machine can do"
"$PREFIX/venv/bin/python" -m runner --probe-only 2>/dev/null | \
  "$PY" -c '
import json, sys
try: c = json.load(sys.stdin)
except Exception: print("  (probe failed)"); raise SystemExit
print("  device      :", c.get("device_name"),
      ("(%d GPU cores)" % c["compute_units"]) if c.get("compute_units") and c.get("backend") == "mps" else "")
print("  backend     :", c.get("backend"), c.get("arch") or "")
print("  memory      :", ((str(c.get("vram_gb")) + " GB"
                           + (" shared with the system" if c.get("unified_memory") else ""))
                          if c.get("vram_gb") else "shared with system"))
if c.get("max_finetune_params_b"):
    print("  biggest model:", "about %sB parameters" % c["max_finetune_params_b"])
print("  best dtype  :", c.get("recommended_dtype"))
print("  4-bit       :", "yes" if c.get("quantization",{}).get("4bit") else "no")
for w in c.get("warnings", []): print("  note        :", w)
'

cat <<EOF

Done. Connect this machine to your controller with:

  $PREFIX/venv/bin/python -m runner \\
      --controller http://YOUR-CONTROLLER:8420 \\
      --token YOUR-JOIN-TOKEN

The token is on the controller's Machines page.
EOF
