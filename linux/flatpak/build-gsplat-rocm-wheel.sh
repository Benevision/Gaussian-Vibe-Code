#!/usr/bin/env bash
# Build gsplat for AMD GPUs (ROCm / HIP) from AMD's port (gsplat-rocm-pin.env) in a throwaway ROCm container.
# The GPU is not needed to compile. Used by .github/workflows/gsplat-rocm-wheel.yml; also runs on a Linux PC with
# podman or docker (not on SteamOS, which has neither: there, use the wheel from the GitHub prerelease).
#
#   ./build-gsplat-rocm-wheel.sh                 all targets in the pin (slow)
#   ./build-gsplat-rocm-wheel.sh --arch gfx1030  one target (e.g. RX 6000 series, which runs gfx1030 kernels)
#
# The wheel lands in ./wheels-rocm/.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIN="$HERE/gsplat-rocm-pin.env"
val() { grep -E "^$1=" "$PIN" | head -1 | cut -d= -f2-; }
REPO=$(val GSPLAT_ROCM_REPO); REF=$(val GSPLAT_ROCM_REF); VERSION=$(val GSPLAT_VERSION)
TORCH=$(val TORCH_VERSION); ROCM_TAG=$(val ROCM_TAG); IMAGE=$(val ROCM_IMAGE); ARCH=$(val PYTORCH_ROCM_ARCH)
OUT="$HERE/wheels-rocm"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --arch) ARCH="$2"; shift ;;
    --out) OUT="$2"; shift ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
  shift
done
ENGINE=$(command -v podman || command -v docker || true)
[[ -n "$ENGINE" ]] || { echo "Needs podman or docker" >&2; exit 1; }
TAG="pt$(echo "$TORCH" | cut -d. -f1,2 | tr -d .)$(echo "$ROCM_TAG" | tr -d .)"      # pt28rocm64
mkdir -p "$OUT"
echo "gsplat $VERSION+$TAG from $REPO@$REF for $ARCH in $IMAGE"
"$ENGINE" run --rm -v "$OUT:/out" -e ARCH="$ARCH" -e REPO="$REPO" -e REF="$REF" -e VERSION="$VERSION" \
  -e TAG="$TAG" -e TORCH="$TORCH" -e ROCM_TAG="$ROCM_TAG" -e MAX_JOBS="${MAX_JOBS:-4}" "$IMAGE" bash -euxc '
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -q
  apt-get install -yq --no-install-recommends python3 python3-pip python3-dev git cmake ninja-build g++ \
      hipcub-dev rocprim-dev rocthrust-dev
  python3 -m pip install -q --upgrade pip wheel setuptools numpy ninja
  python3 -m pip install -q "torch==$TORCH" --index-url "https://download.pytorch.org/whl/$ROCM_TAG"
  python3 -c "import torch; print(\"torch\", torch.__version__, \"hip\", torch.version.hip)"
  git clone --recurse-submodules --branch "$REF" "$REPO" /src
  cd /src && git log -1 --format="gsplat source commit %H"
  ( cd gsplat/cuda/csrc/third_party/glm && cmake -DGLM_BUILD_TESTS=OFF -DBUILD_SHARED_LIBS=OFF -B build . \
      -DCMAKE_INSTALL_PREFIX:PATH=$HOME/.local && cmake --build build -- all && cmake --build build -- install )
  sed -i "s/^__version__ = .*/__version__ = \"$VERSION\"/" gsplat/version.py
  # AMD'"'"'s setup.py (1) always appends +<git hash> to the version, and (2) compiles for the one gfx target
  # rocminfo reports (gfx942 when there is no GPU, as here) and ignores PYTORCH_ROCM_ARCH. Make both explicit.
  python3 - <<'"'"'PY'"'"'
import os, re
p = "setup.py"
s = open(p).read()
n1 = len(re.findall(r"__version__ \+= f\"\+\{git_rev\}\"", s))
s = re.sub(r"__version__ \+= f\"\+\{git_rev\}\"", "__version__ += \"+\" + os.environ[\"GSPLAT_LOCAL_TAG\"]", s)
n2 = s.count("f\"--offload-arch={gpu_arch}\"")
s = s.replace("f\"--offload-arch={gpu_arch}\"",
              "*[f\"--offload-arch={a}\" for a in os.environ.get(\"PYTORCH_ROCM_ARCH\", gpu_arch).split(\";\") if a], "
              "*os.environ.get(\"GSPLAT_HIP_EXTRA_FLAGS\", \"\").split()")
assert n1 == 1 and n2 >= 1, f"setup.py changed upstream: version hooks {n1}, offload-arch {n2}"
if "\nimport os" not in s and not s.startswith("import os"):
    s = "import os\n" + s
open(p, "w").write(s)
print("patched setup.py: version tag + offload archs", os.environ.get("PYTORCH_ROCM_ARCH"))
PY
  grep -n "offload-arch\|GSPLAT_LOCAL_TAG" setup.py
  # The port assumes 64-wide wavefronts (Instinct / CDNA). Radeon (RDNA) runs wave32 by default and fails to
  # compile its 64-thread tiles, so the Radeon targets are built in wave64 mode.
  EXTRA=""; case "$ARCH" in *gfx1*) EXTRA="-mwavefrontsize64" ;; esac
  GSPLAT_HIP_EXTRA_FLAGS="$EXTRA" GSPLAT_LOCAL_TAG="$TAG" PYTORCH_ROCM_ARCH="$ARCH" python3 -m pip wheel --no-deps --no-build-isolation -w /out .
  ls -la /out
'
ls -la "$OUT"
