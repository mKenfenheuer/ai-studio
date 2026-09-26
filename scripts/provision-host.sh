#!/usr/bin/env bash
# Prepare a Linux host to run an AI Studio GPU runner.
#
# Installs the GPU *kernel driver* and Docker only. The ROCm/CUDA userspace
# lives inside the runner image, so this script stays small and does not
# pollute the host with a multi-gigabyte toolchain.
#
# Tested on Ubuntu 24.04 (kernel 6.8) with an RX 6900 XT passed through to a
# QEMU/KVM guest.
set -euo pipefail

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31mxx\033[0m %s\n' "$*" >&2; exit 1; }

[[ $EUID -eq 0 ]] || die "Run this with sudo."
command -v apt-get >/dev/null || die "This script targets Debian/Ubuntu hosts."

KERNEL="$(uname -r)"
VENDOR="${1:-auto}"

if [[ "$VENDOR" == "auto" ]]; then
  if lspci -nn 2>/dev/null | grep -qiE 'VGA|3D|Display' && \
     lspci -nn | grep -iE 'VGA|3D|Display' | grep -qi 'NVIDIA'; then
    VENDOR=nvidia
  elif lspci -nn | grep -iE 'VGA|3D|Display' | grep -qiE 'AMD|ATI'; then
    VENDOR=amd
  else
    die "No AMD or NVIDIA GPU found on the PCI bus. Pass 'amd' or 'nvidia' explicitly."
  fi
fi
log "Detected GPU vendor: $VENDOR (kernel $KERNEL)"

# ---------------------------------------------------------------- AMD ------
if [[ "$VENDOR" == "amd" ]]; then
  # Ubuntu *cloud* images ship linux-image-virtual, which omits amdgpu.ko
  # entirely. Without it there is no /dev/kfd and ROCm cannot see the card,
  # even though lspci lists it happily.
  if ! find "/lib/modules/$KERNEL" -name 'amdgpu.ko*' | grep -q .; then
    log "amdgpu driver missing for $KERNEL; installing kernel modules"
    apt-get update -qq
    apt-get install -y "linux-modules-extra-$KERNEL" linux-generic || {
      warn "No modules-extra for the running kernel ($KERNEL)."
      warn "It has probably been superseded. Installing for the newest kernel;"
      warn "you must REBOOT into it afterwards."
      apt-get install -y linux-generic
    }
    NEEDS_REBOOT=1
  fi

  echo amdgpu > /etc/modules-load.d/amdgpu.conf
  modprobe amdgpu 2>/dev/null || true

  # rocminfo is what bitsandbytes and our capability probe shell out to.
  if ! command -v rocminfo >/dev/null; then
    log "Installing ROCm introspection tools"
    mkdir -p /etc/apt/keyrings
    curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key \
      | gpg --dearmor -o /etc/apt/keyrings/rocm.gpg
    CODENAME="$(. /etc/os-release && echo "$VERSION_CODENAME")"
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/6.4 $CODENAME main" \
      > /etc/apt/sources.list.d/rocm.list
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/amdgpu/6.4/ubuntu $CODENAME main" \
      > /etc/apt/sources.list.d/amdgpu.list
    apt-get update -qq
    apt-get install -y rocminfo libdrm-amdgpu-amdgpu1 libdrm-amdgpu-common
  fi
fi

# ------------------------------------------------------------- NVIDIA ------
if [[ "$VENDOR" == "nvidia" ]]; then
  command -v nvidia-smi >/dev/null || die \
    "NVIDIA driver not installed. Install it first (ubuntu-drivers autoinstall), then re-run."
  log "NVIDIA driver present: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
fi

# -------------------------------------------------------------- Docker -----
if ! command -v docker >/dev/null; then
  log "Installing Docker"
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  chmod a+r /etc/apt/keyrings/docker.asc
  CODENAME="$(. /etc/os-release && echo "$VERSION_CODENAME")"
  echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $CODENAME stable" \
    > /etc/apt/sources.list.d/docker.list
  apt-get update -qq
  apt-get install -y docker-ce docker-ce-cli containerd.io \
                     docker-buildx-plugin docker-compose-plugin
  systemctl enable --now docker
fi

if [[ "$VENDOR" == "nvidia" ]] && ! docker info 2>/dev/null | grep -qi nvidia; then
  log "Installing NVIDIA Container Toolkit"
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update -qq
  apt-get install -y nvidia-container-toolkit
  nvidia-ctk runtime configure --runtime=docker
  systemctl restart docker
fi

# ------------------------------------------------------------- verify ------
echo
log "Verification"
if [[ "$VENDOR" == "amd" ]]; then
  if [[ -e /dev/kfd ]]; then
    echo "  /dev/kfd            present"
    ls /dev/dri/renderD* >/dev/null 2>&1 \
      && echo "  render node         present" \
      || warn "  no /dev/dri/renderD* -- the GPU will not be usable"
    command -v rocminfo >/dev/null && \
      echo "  GPU                 $(rocminfo 2>/dev/null | awk '/Marketing Name/{print $3,$4,$5,$6}' | grep -v CPU | head -1)"
  else
    warn "  /dev/kfd is MISSING -- reboot required before the GPU can be used."
    NEEDS_REBOOT=1
  fi
fi
docker --version | sed 's/^/  /'

echo
if [[ "${NEEDS_REBOOT:-0}" == "1" ]]; then
  warn "REBOOT REQUIRED, then re-run this script to verify."
else
  log "Host is ready. Start a runner with docker/docker-compose.yml,"
  log "or see the exact command on the controller's Machines page."
fi
