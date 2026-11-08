#!/usr/bin/env bash
# SplatWork Node installer: creates a virtualenv with CUDA PyTorch + gsplat.
#
#   ./install.sh                       # venv in ./venv, auto-detect CUDA wheel tag
#   ./install.sh --cuda cu121          # force a PyTorch CUDA wheel tag (cu118 cu121 cu124 cu126 cu128)
#   ./install.sh --prefix /opt/splatworker/venv
#   ./install.sh --with-splatprep      # also install ../tools/splatprep (processing jobs)
#   ./install.sh --no-fused-ssim       # skip building fused-ssim
#   ./install.sh --system              # also create the 'splatworker' user, token and systemd unit (sudo)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="$HERE/venv"
CUDA_TAG=""
WITH_SPLATPREP=0
FUSED_SSIM=1
SYSTEM=0
PYTHON="${PYTHON:-python3}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix) PREFIX="$2"; shift 2 ;;
    --cuda) CUDA_TAG="$2"; shift 2 ;;
    --with-splatprep) WITH_SPLATPREP=1; shift ;;
    --no-fused-ssim) FUSED_SSIM=0; shift ;;
    --system) SYSTEM=1; shift ;;
    --python) PYTHON="$2"; shift 2 ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

say() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mwarning:\033[0m %s\n' "$*" >&2; }

# ---------------------------------------------------------------- checks
"$PYTHON" - <<'EOF' || { echo "Python 3.10+ is required (set PYTHON=python3.11)" >&2; exit 1; }
import sys
sys.exit(0 if sys.version_info >= (3, 10) else 1)
EOF

if ! command -v nvidia-smi >/dev/null 2>&1; then
  warn "nvidia-smi not found: install the NVIDIA driver first (training needs a CUDA GPU)"
fi

if [[ -z "$CUDA_TAG" ]]; then
  # Pick the newest PyTorch CUDA build the driver supports.
  DRV_CUDA="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9]*\)\.\([0-9]*\).*/\1\2/p' | head -1 || true)"
  if   [[ -n "$DRV_CUDA" && "$DRV_CUDA" -ge 128 ]]; then CUDA_TAG=cu128
  elif [[ -n "$DRV_CUDA" && "$DRV_CUDA" -ge 126 ]]; then CUDA_TAG=cu126
  elif [[ -n "$DRV_CUDA" && "$DRV_CUDA" -ge 124 ]]; then CUDA_TAG=cu124
  elif [[ -n "$DRV_CUDA" && "$DRV_CUDA" -ge 121 ]]; then CUDA_TAG=cu121
  elif [[ -n "$DRV_CUDA" && "$DRV_CUDA" -ge 118 ]]; then CUDA_TAG=cu118
  else CUDA_TAG=cu124; warn "could not detect driver CUDA version; defaulting to $CUDA_TAG"
  fi
fi
say "PyTorch CUDA wheel tag: $CUDA_TAG"

# ------------------------------------------------------------------ venv
say "creating virtualenv at $PREFIX"
"$PYTHON" -m venv "$PREFIX"
PIP="$PREFIX/bin/pip"
PY="$PREFIX/bin/python"
"$PIP" install --upgrade pip wheel setuptools ninja

say "installing PyTorch ($CUDA_TAG)"
"$PIP" install torch --index-url "https://download.pytorch.org/whl/$CUDA_TAG"

say "installing gsplat"
# Prefer a prebuilt wheel matching the installed torch/CUDA; fall back to PyPI
# (the PyPI package JIT-compiles its CUDA kernels on first use, which needs
# nvcc from the CUDA toolkit and takes a few minutes once).
TORCH_MM="$("$PY" -c 'import torch; print("".join(torch.__version__.split("+")[0].split(".")[:2]))')"
GSPLAT_INDEX="https://docs.gsplat.studio/whl/pt${TORCH_MM}${CUDA_TAG}"
if ! "$PIP" install "gsplat>=1.4,<2" --index-url "$GSPLAT_INDEX" --extra-index-url https://pypi.org/simple 2>/dev/null; then
  warn "no prebuilt gsplat wheel at $GSPLAT_INDEX; installing from PyPI (JIT build on first use)"
  "$PIP" install "gsplat>=1.4,<2"
fi

say "installing splatworker"
"$PIP" install -e "$HERE[train]"

if [[ "$FUSED_SSIM" == 1 ]]; then
  say "installing fused-ssim (optional, needs nvcc)"
  "$PIP" install --no-build-isolation "git+https://github.com/rahul-goel/fused-ssim" \
    || warn "fused-ssim failed to build; the torch SSIM fallback will be used"
fi

if [[ "$WITH_SPLATPREP" == 1 ]]; then
  SP="$HERE/../tools/splatprep"
  if [[ -f "$SP/pyproject.toml" || -f "$SP/setup.py" ]]; then
    say "installing splatprep from $SP"
    "$PIP" install -e "$SP" || warn "splatprep install failed; processing jobs will be unavailable"
  elif [[ -d "$SP" ]]; then
    warn "$SP has no pyproject.toml; adding it to the venv path instead"
    SITE="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
    echo "$(cd "$SP/.." && pwd)" > "$SITE/splatprep-local.pth"
  else
    warn "$SP not found; skipping splatprep"
  fi
fi

say "verifying"
"$PY" - <<'EOF' || warn "verification failed (see above)"
import torch
print("torch", torch.__version__, "CUDA available:", torch.cuda.is_available())
for i in range(torch.cuda.device_count()):
    print("  GPU", i, torch.cuda.get_device_name(i))
import gsplat
print("gsplat", gsplat.__version__)
try:
    import fused_ssim  # noqa
    print("fused-ssim: yes")
except Exception:
    print("fused-ssim: no (torch fallback)")
try:
    import splatprep  # noqa
    print("splatprep: yes")
except Exception:
    print("splatprep: no (processing jobs disabled)")
EOF

# ---------------------------------------------------------- system setup
if [[ "$SYSTEM" == 1 ]]; then
  say "system setup (sudo)"
  sudo useradd --system --home /var/lib/splatworker --shell /usr/sbin/nologin splatworker 2>/dev/null || true
  sudo mkdir -p /var/lib/splatworker /etc/splatworker
  sudo chown splatworker:splatworker /var/lib/splatworker
  sudo chmod 700 /var/lib/splatworker
  if [[ ! -f /etc/splatworker/token ]]; then
    "$PREFIX/bin/splatworker" gen-token | sudo tee /etc/splatworker/token >/dev/null
  fi
  sudo chown root:splatworker /etc/splatworker/token
  sudo chmod 640 /etc/splatworker/token
  sed "s#/opt/splatworker/venv#$PREFIX#g" "$HERE/splatworker.service" | sudo tee /etc/systemd/system/splatworker.service >/dev/null
  sudo systemctl daemon-reload
  sudo systemctl enable --now splatworker
  say "service started. Token (enter this in SplatWork):"
  sudo cat /etc/splatworker/token
  warn "the venv at $PREFIX must be readable by the 'splatworker' user"
else
  cat <<EOF

Done. Start the worker with:

  TOKEN=\$($PREFIX/bin/splatworker gen-token --out ~/.config/splatworker/token)
  $PREFIX/bin/splatworker serve --port 7878 --token-file ~/.config/splatworker/token

Or install it as a service: ./install.sh --system  (see README.md)
EOF
fi
