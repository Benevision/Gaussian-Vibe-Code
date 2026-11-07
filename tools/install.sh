#!/usr/bin/env bash
# Install splatprep into a virtualenv, with optional model extras.
#
#   ./install.sh                                   # core only (numpy, pillow, opencv)
#   ./install.sh --extras sfm,lightglue,sam2       # + pycolmap, LightGlue, SAM 2
#   ./install.sh --extras all --mast3r             # everything incl. cloning naver/mast3r
#   ./install.sh --venv /path/to/venv --torch-index https://download.pytorch.org/whl/cu121
#
# Nothing here runs Homebrew or downloads model weights unless asked:
#   --brew              also run `brew install colmap glomap` (macOS)
#   --prefetch-weights  instantiate the installed models once so their weights are cached
#
# Default venv: macOS  ~/Library/Application Support/SplatStudio/python
#               Linux  ./.venv (next to this script)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTRAS=""
VENV=""
# Prefer 3.11/3.12: several model packages (VGGT pins numpy 1.26) lack wheels for 3.13.
if [[ -z "${PYTHON:-}" ]]; then
  for v in python3.11 python3.12 python3.10 python3; do
    if command -v "$v" >/dev/null 2>&1; then PYTHON="$v"; break; fi
  done
fi
TORCH_INDEX=""
DO_BREW=0
DO_MAST3R=0
PREFETCH=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --extras) EXTRAS="$2"; shift 2 ;;
    --venv) VENV="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    --torch-index) TORCH_INDEX="$2"; shift 2 ;;
    --brew) DO_BREW=1; shift ;;
    --mast3r) DO_MAST3R=1; shift ;;
    --prefetch-weights) PREFETCH=1; shift ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$VENV" ]]; then
  if [[ "$(uname)" == "Darwin" ]]; then
    VENV="$HOME/Library/Application Support/SplatStudio/python"
  else
    VENV="$HERE/.venv"
  fi
fi

if [[ "$EXTRAS" == "all" ]]; then
  EXTRAS="sfm,lightglue,loftr,masks,sam2,vggt,mast3r"
fi

echo "==> venv: $VENV"
if [[ ! -x "$VENV/bin/python" ]]; then
  "$PYTHON" -m venv "$VENV"
fi
PY="$VENV/bin/python"
"$PY" -m pip install --upgrade pip wheel setuptools >/dev/null

if [[ -n "$TORCH_INDEX" && -n "$EXTRAS" ]]; then
  echo "==> installing torch from $TORCH_INDEX"
  "$PY" -m pip install torch torchvision --index-url "$TORCH_INDEX"
fi

SPEC="$HERE"
if [[ -n "$EXTRAS" ]]; then
  SPEC="$HERE[$EXTRAS]"
fi
echo "==> pip install $SPEC"
# SAM 2's optional CUDA extension is not needed (and does not build on macOS)
SAM2_BUILD_CUDA=0 "$PY" -m pip install "$SPEC"

if [[ $DO_MAST3R -eq 1 ]]; then
  SRC="$VENV/src/mast3r"
  if [[ ! -d "$SRC" ]]; then
    echo "==> cloning naver/mast3r (+dust3r submodule) into $SRC"
    mkdir -p "$VENV/src"
    git clone --recursive https://github.com/naver/mast3r "$SRC"
  fi
  "$PY" -m pip install -r "$SRC/requirements.txt" -r "$SRC/dust3r/requirements.txt"
  echo "export SPLATPREP_MAST3R_PATH=\"$SRC\"" > "$VENV/splatprep_env.sh"
  echo "    MASt3R installed. Set SPLATPREP_MAST3R_PATH=\"$SRC\" (written to $VENV/splatprep_env.sh)"
fi

if [[ $DO_BREW -eq 1 ]]; then
  if command -v brew >/dev/null; then
    brew install colmap glomap
  else
    echo "Homebrew not found; skipping colmap/glomap" >&2
  fi
fi

if [[ $PREFETCH -eq 1 ]]; then
  echo "==> prefetching weights for installed models (this downloads from the internet)"
  [[ -f "$VENV/splatprep_env.sh" ]] && source "$VENV/splatprep_env.sh"
  "$PY" - <<'EOF'
import importlib.util as u
def has(m): return u.find_spec(m) is not None
if has("lightglue"):
    import lightglue as L
    for name in ("SuperPoint", "DISK", "ALIKED"):
        try: getattr(L, name)(); print("ok", name)
        except Exception as e: print("skip", name, e)
    for feat in ("superpoint", "disk", "aliked", "sift"):
        try: L.LightGlue(features=feat); print("ok lightglue", feat)
        except Exception as e: print("skip lightglue", feat, e)
if has("kornia"):
    import kornia.feature as KF
    KF.LoFTR(pretrained="outdoor"); print("ok loftr")
if has("torchvision"):
    from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, DeepLabV3_MobileNet_V3_Large_Weights
    deeplabv3_mobilenet_v3_large(weights=DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT); print("ok person")
if has("transformers"):
    from transformers import AutoModelForSemanticSegmentation, AutoImageProcessor
    m = "nvidia/segformer-b0-finetuned-ade-512-512"
    AutoImageProcessor.from_pretrained(m); AutoModelForSemanticSegmentation.from_pretrained(m); print("ok sky")
if has("sam2"):
    from sam2.sam2_video_predictor import SAM2VideoPredictor
    SAM2VideoPredictor.from_pretrained("facebook/sam2.1-hiera-small", device="cpu"); print("ok sam2")
if has("vggt"):
    from vggt.models.vggt import VGGT
    VGGT.from_pretrained("facebook/VGGT-1B"); print("ok vggt")
if has("mast3r"):
    from mast3r.model import AsymmetricMASt3R
    AsymmetricMASt3R.from_pretrained("naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"); print("ok mast3r")
EOF
fi

echo "==> available models:"
[[ -f "$VENV/splatprep_env.sh" ]] && source "$VENV/splatprep_env.sh"
"$PY" -m splatprep models
