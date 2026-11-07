#!/usr/bin/env bash
# Build the pinned gsplat development wheel (gsplat-pin.env) on this PC, in a throwaway container —
# the same steps as the GitHub Actions job, but on your own CPU cores. Nothing is installed on the host
# except the container image. The GPU is not needed to compile.
#
#   ./build-gsplat-wheel.sh            build for the GPU(s) in this PC only (fast; e.g. RTX 4080 = 8.9)
#   ./build-gsplat-wheel.sh --all      build for every card the runtime supports (RTX 20 – RTX 50; slow)
#   ./build-gsplat-wheel.sh --arch "8.6;8.9"
#   ... --no-install                   don't hand the wheel to Splat Worker
#
# The wheel lands in ./wheels/ and (unless --no-install) in Splat Worker's own wheel folder
# (~/.var/app/com.example.SplatWorker/data/splatworker/wheels/), where Set Up finds it: then choose
# "PyTorch 2.7 + CUDA 12.8 · gsplat 1.6" in Set Up's advanced options (or let it pick automatically).
#
# Needs podman or docker (CachyOS / Arch: sudo pacman -S podman).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIN="$HERE/gsplat-pin.env"
val() { grep -E "^$1=" "$PIN" | head -1 | cut -d= -f2-; }   # (not `source`: the arch list contains ';')
COMMIT=$(val GSPLAT_COMMIT); VERSION=$(val GSPLAT_VERSION); TORCH=$(val TORCH_VERSION); CUDA_TAG=$(val CUDA_TAG)
ALL_ARCH=$(val TORCH_CUDA_ARCH_LIST)
CUDA_VER="${CUDA_TAG#cu}"; CUDA_VER="${CUDA_VER:0:2}.${CUDA_VER:2}"          # cu128 -> 12.8
IMAGE="docker.io/nvidia/cuda:${CUDA_VER}.1-devel-ubuntu22.04"                  # Ubuntu 22.04 = Python 3.10
TAG="pt$(echo "$TORCH" | cut -d. -f1,2 | tr -d .)${CUDA_TAG}"                  # pt27cu128

ARCH=""; INSTALL=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --all) ARCH="$ALL_ARCH" ;;
    --arch) ARCH="$2"; shift ;;
    --no-install) INSTALL=0 ;;
    -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
  shift
done
if [[ -z "$ARCH" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sort -u | paste -sd';')
  fi
  [[ -n "$ARCH" ]] || { echo "Couldn't detect the GPU (nvidia-smi). Use --arch 8.9 or --all." >&2; exit 1; }
fi

ENGINE=$(command -v podman || command -v docker || true)
[[ -n "$ENGINE" ]] || { echo "Needs podman or docker:  sudo pacman -S podman" >&2; exit 1; }

# nvcc needs ~2-3 GB per parallel job: cap the jobs by memory as well as by cores.
CORES=$(nproc)
MEM_GB=$(( $(awk '/MemTotal/ {print $2}' /proc/meminfo) / 1024 / 1024 ))
JOBS=$(( MEM_GB / 3 )); (( JOBS > CORES )) && JOBS=$CORES; (( JOBS < 1 )) && JOBS=1

OUT="$HERE/wheels"; mkdir -p "$OUT"
CACHE="${XDG_CACHE_HOME:-$HOME/.cache}/splatworker-gsplat-build"; mkdir -p "$CACHE/pip"
echo "gsplat $VERSION+$TAG @ ${COMMIT:0:10} for compute capability $ARCH — $JOBS parallel jobs ($CORES cores, ${MEM_GB} GB)"
echo "container: $IMAGE (first run downloads ~5 GB)"
START=$(date +%s)

"$ENGINE" run --rm \
  -v "$OUT:/out:Z" -v "$CACHE/pip:/root/.cache/pip:Z" \
  -e COMMIT="$COMMIT" -e VERSION="$VERSION+$TAG" -e TORCH="$TORCH" -e CUDA_TAG="$CUDA_TAG" \
  -e TORCH_CUDA_ARCH_LIST="$ARCH" -e MAX_JOBS="$JOBS" -e BUILD_EXPERIMENTAL=0 \
  "$IMAGE" bash -euo pipefail -c '
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y -qq python3.10 python3.10-venv python3.10-dev git >/dev/null
    python3.10 -m venv /venv; . /venv/bin/activate
    pip install -q --upgrade pip wheel setuptools ninja numpy
    pip install -q "torch==$TORCH" --index-url "https://download.pytorch.org/whl/$CUDA_TAG"
    git clone -q --recursive https://github.com/nerfstudio-project/gsplat.git /src
    cd /src; git checkout -q "$COMMIT"; git submodule update -q --init --recursive
    sed -i "s/^__version__ = .*/__version__ = \"$VERSION\"/" gsplat/version.py
    # Upstream bug at this dev commit: Utils.cpp (built by g++, not nvcc) calls the two-argument
    # cudaEventCreate overload, which only exists for nvcc. The C API equivalent works everywhere.
    sed -i -E "s/cudaEventCreate\(([^,()]+), *cudaEventDisableTiming\)/cudaEventCreateWithFlags(\1, cudaEventDisableTiming)/g" gsplat/cuda/csrc/*.cpp
    grep -c cudaEventCreateWithFlags gsplat/cuda/csrc/Utils.cpp | sed "s/^/patched cudaEventCreate calls: /"
    echo "compiling ~50 CUDA/C++ files with $MAX_JOBS jobs (quiet until done; the last few big files run on one core)"
    tail -1 gsplat/version.py
    nvcc --version | tail -1
    # -v so ninja prints each file; show just "[n/50] file.o" (+ any errors), keep the full log next to the wheel.
    pip wheel -v --no-deps --no-build-isolation -w /tmp/w . 2>&1 | tee /out/gsplat-build.log \
      | grep --line-buffered -E "^ *\[[0-9]+/[0-9]+\]|error:|FAILED|Successfully built|ERROR" \
      | sed -u -E "s#^ *(\[[0-9]+/[0-9]+\]).* -o [^ ]*/([^/ ]+\.o)( .*)?\$#  \1 \2#"
    cp /tmp/w/gsplat-*.whl /out/
  '

WHEEL=$(ls -t "$OUT"/gsplat-"$VERSION"+"$TAG"-*.whl | head -1)
printf 'archs=%s\nbuilt=%s\ncommit=%s\n' "$ARCH" "$(date -u +%FT%TZ)" "$COMMIT" > "$WHEEL.info"
echo "built $(basename "$WHEEL") in $(( ($(date +%s) - START) / 60 )) min (compute capability $ARCH)"

if [[ "$INSTALL" == 1 ]]; then
  DEST="$HOME/.var/app/com.example.SplatWorker/data/splatworker/wheels"
  mkdir -p "$DEST"
  rm -f "$DEST"/gsplat-"$VERSION"+"$TAG"-*.whl "$DEST"/gsplat-"$VERSION"+"$TAG"-*.whl.info
  cp "$WHEEL" "$WHEEL.info" "$DEST/"
  echo "handed to Splat Worker: $DEST"
  echo "Now open Splat Worker → Set Up… (it picks the gsplat 1.6 runtime when your driver supports CUDA 12.8)."
fi
