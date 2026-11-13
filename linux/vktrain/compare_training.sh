#!/usr/bin/env bash
# Trains one prepared dataset (docs/PROTOCOL.md §2) with the CUDA / gsplat trainer (SplatWork for Linux's
# Flatpak, `--cli train`) and with the Vulkan trainer (splatwork-vk-train) on the same TrainingConfig and
# iterations, then prints eval PSNR, final splat count, wall time and steps/s of both.
#
#   bash compare_training.sh DATASET [--iterations N] [--strategy mcmc|adaptive|both] [--set key=value ...]
#                            [--out DIR] [--holdout-every N] [--no-cuda] [--no-vulkan] [--vk-bin PATH]
#
# Works from fish too (it is a bash script: call it with `bash`). On the RTX 4080 PC it runs both halves; on
# an AMD / Intel PC (SteamOS) use --no-cuda (the CUDA half is also skipped by itself when there's no NVIDIA GPU).
#
# The Vulkan trainer binary comes from --vk-bin, else linux/vktrain/build/splatwork-vk-train next to this
# script, else it is built:
#   * on the host when cmake, glslangValidator and the Vulkan headers are there
#     (CachyOS / Arch: sudo pacman -S --needed cmake ninja glslang vulkan-headers vulkan-icd-loader), or
#   * inside the Freedesktop SDK Flatpak runtime (SteamOS: nothing to install on the read-only system;
#     flatpak install --user flathub org.freedesktop.Sdk//25.08 once) — it then also runs inside that runtime,
#     which gets the Vulkan drivers (Mesa RADV / ANV, or the NVIDIA GL extension) from Flatpak.
#
# Both trainers get the same config: the dataset's TrainingConfig defaults plus --set overrides, with the
# features the Vulkan trainer does not have yet switched off for both (per-photo colour correction, depth /
# normal losses and depth priors, transient masking, skybox, pose refinement, bilateral grid), so the numbers
# compare the same algorithm. A dataset without holdout photos gets a hard-linked copy with every N-th photo
# held out (--holdout-every, default 8) so both report a held-out PSNR.
set -euo pipefail

usage() { sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET=""
ITERS=30000
STRATEGIES="mcmc"
OUT=""
HOLDOUT_EVERY=8
RUN_CUDA=1
RUN_VK=1
VK_BIN=""
SETS=()
FDO_BRANCH="${FDO_BRANCH:-25.08}"
APP_ID="${SPLATWORK_APP_ID:-com.example.SplatWorker}"

while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help) usage 0 ;;
        --iterations) ITERS="$2"; shift 2 ;;
        --strategy) STRATEGIES="$2"; [ "$STRATEGIES" = both ] && STRATEGIES="mcmc adaptive"; shift 2 ;;
        --set) SETS+=("$2"); shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        --holdout-every) HOLDOUT_EVERY="$2"; shift 2 ;;
        --no-cuda) RUN_CUDA=0; shift ;;
        --no-vulkan) RUN_VK=0; shift ;;
        --vk-bin) VK_BIN="$2"; shift 2 ;;
        -*) echo "unknown option $1" >&2; usage 2 ;;
        *) [ -z "$DATASET" ] && DATASET="$1" || { echo "one dataset only" >&2; exit 2; }; shift ;;
    esac
done
[ -n "$DATASET" ] || usage 2
[ -f "$DATASET/dataset.json" ] || { echo "$DATASET/dataset.json not found" >&2; exit 2; }
DATASET="$(cd "$DATASET" && pwd)"
command -v python3 >/dev/null || { echo "python3 is needed (for the JSON handling)" >&2; exit 2; }
OUT="${OUT:-$PWD/compare-$(basename "$DATASET")-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
echo "results and logs: $OUT"

# ------------------------------------------------------------------ dataset with holdout photos
HAS_HOLDOUT=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(int(any(c.get("holdout") for c in d["cameras"])))' "$DATASET/dataset.json")
DS="$DATASET"
if [ "$HAS_HOLDOUT" = 0 ] && [ "$HOLDOUT_EVERY" -gt 0 ]; then
    DS="$OUT/dataset"
    if [ ! -f "$DS/dataset.json" ]; then
        echo "the dataset has no holdout photos: hard-linked copy with every ${HOLDOUT_EVERY}th photo held out in $DS"
        cp -al "$DATASET" "$DS" 2>/dev/null || { rm -rf "$DS"; cp -a "$DATASET" "$DS"; }
        rm -f "$DS/dataset.json"   # a new file, not the hard link to the original
        python3 - "$DATASET/dataset.json" "$DS/dataset.json" "$HOLDOUT_EVERY" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
k = int(sys.argv[3])
for i, c in enumerate(d["cameras"]):
    c["holdout"] = i % k == k - 1
json.dump(d, open(sys.argv[2], "w"), indent=1)
PY
    fi
fi

# ------------------------------------------------------------------ config (same for both trainers)
make_config() {   # $1 strategy → $OUT/config-$1.json
    python3 - "$OUT/config-$1.json" "$1" "$ITERS" "${SETS[@]+"${SETS[@]}"}" <<'PY'
import json, sys
path, strategy, iters = sys.argv[1], sys.argv[2], int(sys.argv[3])
cfg = {
    "iterations": iters, "strategy": strategy,
    # not in the Vulkan trainer yet (stage 2b): off for both
    "appearanceCorrection": False, "bilateralGrid": False, "normalWeight": 0, "depthWeight": 0, "depthPriors": "off",
    "monoDepthWeight": 0, "transientMasking": False, "skybox": "off", "poseRefinement": False, "decoupledSSIM": False,
    "optimizerPrecision": "fp32",
    "evalInterval": max(1, iters // 6), "previewInterval": 0,
}
for kv in sys.argv[4:]:
    k, _, v = kv.partition("=")
    try:
        cfg[k] = json.loads(v)
    except json.JSONDecodeError:
        cfg[k] = v
json.dump(cfg, open(path, "w"), indent=1)
PY
}

# ------------------------------------------------------------------ Vulkan trainer binary
VK_RUN=()   # command prefix that runs splatwork-vk-train
find_vk() {
    if [ -n "$VK_BIN" ]; then VK_RUN=("$VK_BIN"); return 0; fi
    if [ -x "$HERE/build/splatwork-vk-train" ]; then VK_RUN=("$HERE/build/splatwork-vk-train"); return 0; fi
    if command -v cmake >/dev/null && command -v glslangValidator >/dev/null \
        && { [ -f /usr/include/vulkan/vulkan.h ] || [ -f /usr/local/include/vulkan/vulkan.h ]; }; then
        echo "building splatwork-vk-train on this system (log: $OUT/vk-build.log)"
        local gen=(); command -v ninja >/dev/null && gen=(-G Ninja)
        if cmake -S "$HERE" -B "$HERE/build" "${gen[@]}" -DCMAKE_BUILD_TYPE=Release >"$OUT/vk-build.log" 2>&1 \
            && cmake --build "$HERE/build" -j "$(nproc 2>/dev/null || echo 4)" >>"$OUT/vk-build.log" 2>&1; then
            VK_RUN=("$HERE/build/splatwork-vk-train"); return 0
        fi
        echo "host build failed (see $OUT/vk-build.log)" >&2
    fi
    if command -v flatpak >/dev/null && flatpak info "org.freedesktop.Sdk//$FDO_BRANCH" >/dev/null 2>&1; then
        echo "building splatwork-vk-train in org.freedesktop.Sdk//$FDO_BRANCH (log: $OUT/vk-build.log)"
        local fp=(flatpak run --filesystem=host --device=dri --command=bash "org.freedesktop.Sdk//$FDO_BRANCH")
        if "${fp[@]}" -c "cmake -S '$HERE' -B '$HERE/build-flatpak' -DCMAKE_BUILD_TYPE=Release && cmake --build '$HERE/build-flatpak' -j 4" \
            >"$OUT/vk-build.log" 2>&1; then
            VK_RUN=(flatpak run --filesystem=host --device=dri --command="$HERE/build-flatpak/splatwork-vk-train" "org.freedesktop.Sdk//$FDO_BRANCH")
            return 0
        fi
        echo "Flatpak SDK build failed (see $OUT/vk-build.log)" >&2
    fi
    cat >&2 <<EOF
No splatwork-vk-train: pass --vk-bin PATH, or install the build tools
  (Arch / CachyOS: sudo pacman -S --needed cmake ninja glslang vulkan-headers vulkan-icd-loader)
  or the Flatpak SDK (flatpak install --user flathub org.freedesktop.Sdk//$FDO_BRANCH), then run this again.
EOF
    return 1
}

# ------------------------------------------------------------------ runs
declare -A ROW
run_cuda() {   # $1 strategy
    local s=$1 log="$OUT/cuda-$1.log" t0 t1
    echo "== CUDA / gsplat ($s, $ITERS iterations): log $log"
    t0=$(date +%s.%N)
    if ! flatpak run --filesystem="$DS" --filesystem="$OUT" "$APP_ID" --cli train "$DS" \
            --out "$OUT/cuda-$s.ply" --config "$OUT/config-$s.json" >"$log" 2>&1; then
        echo "   CUDA training failed — last lines of $log:" >&2
        tail -n 15 "$log" >&2
        ROW[cuda-$s]="failed"
        return 0
    fi
    t1=$(date +%s.%N)
    # cli.py prints the stats dict (json.dumps(indent=1)) last
    ROW[cuda-$s]=$(python3 - "$log" "$t0" "$t1" "$ITERS" <<'PY'
import json, sys
lines = open(sys.argv[1]).read().splitlines()
start = max(i for i, l in enumerate(lines) if l.rstrip() == "{")
end = next(i for i in range(start, len(lines)) if lines[i].rstrip() == "}")
st = json.loads("\n".join(lines[start:end + 1]))
wall = float(sys.argv[3]) - float(sys.argv[2])
el = st.get("elapsedSeconds") or wall
print(json.dumps({"psnr": st.get("psnr"), "n": st.get("numGaussians"), "wall": wall, "train": el,
                  "sps": int(sys.argv[4]) / el if el else None}))
PY
)
}

run_vk() {   # $1 strategy
    local s=$1 log="$OUT/vulkan-$1.jsonl" t0 t1
    echo "== Vulkan ($s, $ITERS iterations): log $log"
    t0=$(date +%s.%N)
    if ! "${VK_RUN[@]}" train --dataset "$DS" --config "$OUT/config-$s.json" --out "$OUT/vulkan-$s.ply" \
            --job-dir "$OUT/vulkan-$s-job" </dev/null >"$log" 2>"$OUT/vulkan-$s.stderr"; then
        echo "   Vulkan training failed — last lines of $log:" >&2
        tail -n 5 "$log" >&2
        ROW[vk-$s]="failed"
        return 0
    fi
    t1=$(date +%s.%N)
    ROW[vk-$s]=$(python3 - "$log" "$t0" "$t1" <<'PY'
import json, sys
msgs = [json.loads(l) for l in open(sys.argv[1]) if l.startswith("{")]
done = [m for m in msgs if m["type"] == "done"][-1]
dev = next((m["message"] for m in msgs if m["type"] == "log" and m["message"].startswith("Vulkan trainer on")), "")
print(json.dumps({"psnr": done.get("psnr"), "n": done.get("numGaussians"), "wall": float(sys.argv[3]) - float(sys.argv[2]),
                  "train": done.get("elapsedSeconds"), "sps": done.get("stepsPerSecond"), "ms": done.get("msPerStep"),
                  "device": dev}))
PY
)
}

if [ "$RUN_CUDA" = 1 ]; then
    if ! command -v flatpak >/dev/null || ! flatpak info "$APP_ID" >/dev/null 2>&1; then
        echo "SplatWork for Linux ($APP_ID) is not installed as a Flatpak: skipping the CUDA half" >&2; RUN_CUDA=0
    elif ! command -v nvidia-smi >/dev/null && [ ! -e /dev/nvidiactl ]; then
        echo "no NVIDIA GPU: skipping the CUDA half" >&2; RUN_CUDA=0
    fi
fi
if [ "$RUN_VK" = 1 ]; then find_vk || RUN_VK=0; fi
if [ "$RUN_VK" = 1 ]; then "${VK_RUN[@]}" --info | tee "$OUT/vulkan-device.json"; fi

for s in $STRATEGIES; do
    make_config "$s"
    [ "$RUN_CUDA" = 1 ] && run_cuda "$s"
    [ "$RUN_VK" = 1 ] && run_vk "$s"
done

# ------------------------------------------------------------------ table
ARGS=()
for s in $STRATEGIES; do ARGS+=("$s" "${ROW[cuda-$s]:-}" "${ROW[vk-$s]:-}"); done
python3 - "$ITERS" "$OUT/summary.txt" "${ARGS[@]}" <<'PY'
import json, sys
iters, summary, rest = sys.argv[1], sys.argv[2], sys.argv[3:]
lines = [f"{'trainer':<14}{'strategy':<10}{'iterations':>11}{'eval PSNR':>11}{'splats':>10}{'wall s':>9}{'train s':>9}{'steps/s':>9}"]
notes = []
def fmt(v, f):
    return format(v, f) if isinstance(v, (int, float)) else "–"
for i in range(0, len(rest), 3):
    s, rows = rest[i], {}
    for name, raw in (("CUDA gsplat", rest[i + 1]), ("Vulkan", rest[i + 2])):
        if not raw:
            continue
        if raw == "failed":
            lines.append(f"{name:<14}{s:<10}{iters:>11}   failed")
            continue
        r = json.loads(raw)
        rows[name] = r
        lines.append(f"{name:<14}{s:<10}{iters:>11}{fmt(r['psnr'], '.2f'):>11}{fmt(r['n'], 'd'):>10}{fmt(r['wall'], '.0f'):>9}"
                     f"{fmt(r['train'], '.0f'):>9}{fmt(r['sps'], '.1f'):>9}")
        if r.get("ms"):
            notes.append(f"Vulkan {s}: ms/step forward+loss {r['ms']['forwardLoss']:.1f}, backward+Adam "
                         f"{r['ms']['backwardAdam']:.1f}, densification {r['ms']['densify']:.2f} ({r.get('device', '')})")
    a, b = rows.get("CUDA gsplat"), rows.get("Vulkan")
    if a and b and isinstance(a.get("psnr"), (int, float)) and isinstance(b.get("psnr"), (int, float)):
        notes.append(f"{s}: Vulkan − CUDA eval PSNR {b['psnr'] - a['psnr']:+.2f} dB, speed {b['sps'] / a['sps']:.2f}× "
                     f"(target: within 0.3 dB)")
text = "\n".join(lines + [""] + notes)
print("\n" + text)
open(summary, "w").write(text + "\n")
PY
echo "summary written to $OUT/summary.txt"
