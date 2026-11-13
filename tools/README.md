# splatprep

Processing toolkit shared by **SplatWork** (macOS, PyTorch MPS) and **SplatWork for Linux** (Linux, CUDA):
learned / classic feature matching, SfM, masking, a synthetic ground-truth generator and a
benchmark harness. The contract is `docs/PROTOCOL.md` §2 (dataset.json) and §6 (processing).

```
python -m splatprep run --request request.json --progress-jsonl -     # what the Mac app / worker call
python -m splatprep models                                            # what can run on this machine
python -m splatprep synth --out synth --trajectory all                # synthetic GT datasets
python -m splatprep benchmark --synth all --dataset Kitchen.splatcapture --out bench
python -m splatprep export --model sparse/0 --images images --out dataset   # COLMAP -> dataset.json
python -m splatprep maskserver --images dataset/images                 # SAM 2 click-to-mask JSONL loop
```

Python API used by the worker:

```python
from splatprep.runner import run_request      # run_request(request: dict, workdir: Path, progress: Callable[[dict], None]) -> dict
from splatprep.models import available_models # available_models() -> list[str]
```

## Install

Core (numpy, Pillow, OpenCV) is tiny. Everything else is an optional extra; a missing
package/binary only marks the model unavailable (`splatprep models` says why).

```bash
./install.sh                                   # core only
./install.sh --extras sfm,lightglue,sam2       # pycolmap + LightGlue family + SAM 2
./install.sh --extras all --mast3r             # everything (MASt3R is cloned, not pip-installed)
./install.sh --torch-index https://download.pytorch.org/whl/cu121 --extras all   # Linux CUDA wheels
./install.sh --brew                            # also `brew install colmap glomap`
./install.sh --prefetch-weights                # download weights for every installed model now
```

| extra | adds | models enabled |
|---|---|---|
| `sfm` | pycolmap | full SfM for all feature backends without the colmap CLI |
| `lightglue` | torch, torchvision, kornia, cvg/LightGlue | `sift_lightglue`, `superpoint_lightglue`, `disk_lightglue`, `aliked_lightglue` |
| `loftr` | torch, kornia | `loftr` |
| `masks` | torch, torchvision, transformers | `person`, `sky` (SegFormer; heuristic fallback always available) |
| `sam2` | torch, facebookresearch/sam2 | `sam2_auto`, `sam2_track`, `maskpreview` / `maskserver` |
| `masks` + `sam2` | both of the above | `dynamic` (OWLv2 + SAM 2 moving-object masks) |
| `vggt` | torch, facebookresearch/vggt | `vggt` |
| `mast3r` | MASt3R's pip deps; repo cloned by `install.sh --mast3r`, set `SPLATPREP_MAST3R_PATH` | `mast3r` |
| `retrieval` | hloc + h5py | NetVLAD retrieval (else built-in thumbnail descriptor) |

The Mac app installs into `~/Library/Application Support/SplatStudio/python` (install.sh's default on macOS).

### macOS notes

* Device selection is **CUDA > MPS > CPU** (`SPLATPREP_DEVICE=cpu|mps|cuda` overrides).
  `PYTORCH_ENABLE_MPS_FALLBACK=1` is set automatically so ops without an MPS kernel (ALIKED's
  deformable convolution) fall back to CPU instead of failing.
* `brew install colmap glomap` gives the `colmap_sift` backend and the fastest mapper (GLOMAP).
  Homebrew COLMAP has no CUDA, so SIFT extraction/matching runs on the CPU (`use_gpu 0` is passed).
  Alternatively `pip install pycolmap` provides an incremental mapper without Homebrew.
* LightGlue runs on MPS without FlashAttention (disabled automatically). LoFTR/MASt3R/VGGT run on
  MPS but are much slower than on CUDA and need lots of unified memory (see model cards).
* Executables are looked up on `PATH`, `/opt/homebrew/bin`, `/usr/local/bin`; override with
  `SPLATPREP_COLMAP=/path/to/colmap`, `SPLATPREP_GLOMAP=/path/to/glomap`.

## ProcessingRequest

Exactly §6 of the protocol. Relative paths resolve against `--workdir` (default: the current
directory). Extra keys accepted (all optional, unknown keys are ignored):

| key | meaning |
|---|---|
| `masksDir` | keep-masks (`<stem>.png`, 255 = keep) used to drop keypoints / passed to COLMAP |
| `exportImages` | `false` skips copying/undistorting images (benchmark) |
| `allowMatchingOnly` | without a mapper, return matching stats + `matches.npz` instead of failing |
| `saveMatches`, `keepWork` | keep `matches.npz` / the work dir (`.splatprep_work_<outputDir>`) |
| `device` | `auto` \| `cuda` \| `mps` \| `cpu` |
| `datasets`, `models` | task = `benchmark` |
| `maskModel`, `maskPrompts`, `maskClasses`, ... | task = `masks` / `maskpreview`, see [Masks](#masks) |

**Pair selection** (`matching`): `sequential` (overlap N), `exhaustive`, `retrieval` (top-K by
NetVLAD via hloc if installed, else a thumbnail + gradient-histogram descriptor), `auto` =
exhaustive for ≤ 60 images, sequential (+ 5 retrieval loop-closure candidates per image) for
video-like names (shared prefix + consecutive numbers) or timestamped priors, retrieval otherwise.
With `priorPoses`, the K nearest prior cameras with compatible viewing directions are always added.

**Mapper** (`mapper`): `glomap` (GLOMAP CLI or `colmap global_mapper`), `colmap`
(`colmap mapper`), `pycolmap`, `auto` (GLOMAP → COLMAP → pycolmap). The largest model is kept.

**Camera model** (`cameraModel`): `auto` → SIMPLE_RADIAL, `pinhole`, `opencv`, `fisheye` → OPENCV_FISHEYE.
Distorted models are undistorted on export with `colmap image_undistorter` if available, else OpenCV.

### priorPoses

Any of: a `.splatcapture` `manifest.json` (ARKit, column-major — converted with
`c2w · diag(1,-1,-1,1)`), a `dataset.json`, or

```jsonc
{"format": "splatpriors", "convention": "opencv" | "arkit",
 "frames": [{"image": "000000.jpg",
             "cameraToWorld": [[...],[...],[...],[...]],   // nested = row-major; flat 16 = column-major
             "intrinsics": [fx, fy, cx, cy], "timestamp": 1.23}]}
```

Frames match images by basename (then stem). Priors seed pair selection and, after mapping,
the reconstruction is Sim(3)-aligned to them (robust Umeyama on camera centres) so the dataset
lands in the ARKit frame (metres, +Y up). For the Python feature backends, prior intrinsics also initialise the cameras.

### Output

`task = sfm` → `outputDir/`:

```
dataset.json        # §2: OpenCV convention, ROW-major cameraToWorld, sceneCenter/sceneExtent/upVector
images/             # registered images (undistorted pinhole)
points3d.ply        # x y z float + rgb uchar, binary little endian
sparse/0/           # the exported COLMAP model (binary)
splatprep.log       # COLMAP/GLOMAP output
```

`sceneCenter` = mean camera centre, `sceneExtent` = 1.1 × max camera distance from it,
`upVector` = −(mean camera +Y axis). `task = masks` → `outputDir/masks/<image stem>.png`.

Progress JSONL on stdout (stdout is reserved: stray prints from libraries go to stderr):

```
{"stage":"features","progress":0.42,"message":"extracted frame_00012.jpg (2048 kp)"}
{"done":true,"stats":{...}}         or         {"error":"model unavailable: loftr (missing python packages: torch, kornia — ...)"}
```

`progress` is overall job progress (monotonic, 0–1; a repeated value is possible). Stages: `pairs`, `features`, `matching`,
`verification`, `mapping`, `alignment`, `export`, `masks`, `benchmark`. Exit code 0 / 2.

### Masks

`task = masks` writes `outputDir/masks/<image stem>.png` (8-bit, **255 = keep / train on it, 0 = ignore**).

| model | 255 (keep) | notes |
|---|---|---|
| `person` | everything except (dilated) people | torchvision DeepLabV3-MobileNetV3 |
| `sky` | everything except sky | SegFormer-B0 ADE20K if `transformers` installed, else heuristic |
| `dynamic` | everything except moving-object classes (+ `maskPrompts` objects) | OWLv2 open-vocabulary detection on every frame → SAM 2 masks, tracked through the sequence |
| `sam2_track` | everything except the `maskPrompts` objects | clicks/boxes on ANY frame, tracked forward and backward with the SAM 2 video predictor |
| `sam2_auto` | with non-empty `maskPrompts`: everything except the prompted objects (= `sam2_track`); otherwise the object | the Mac mask editor sends `maskModel: "sam2_auto"` + `maskPrompts` → prompted forward/backward tracking, objects removed. Legacy `maskPrompt` (first image) or no prompt (most salient central object, SAM 2 automatic mask generator) → object-centric, 255 = object |

Without `maskModel`, a request with `maskPrompts` uses `sam2_track`, otherwise `sam2_auto`.
`outputDir: "."` writes to `<workdir>/masks/` (the Mac runs with workdir = dataset root).
`invertMask: true` flips any of them.

**Merge:** if `masks/<stem>.png` already exists, the new mask is combined with logical AND
(pixel-wise min — anything either mask ignores stays ignored), so e.g. `sky` then `dynamic`
accumulate. `replaceMasks: true` overwrites instead. Files are written atomically. The merge
also applies to `person` / `sky` / `sam2_auto`.

Mask request keys (all optional):

| key | default | meaning |
|---|---|---|
| `maskModel` | see above | `person` \| `sky` \| `dynamic` \| `sam2_track` \| `sam2_auto` |
| `maskPrompts` | – | `[{"image": "000123.jpg", "points": [[x, y, label], ...], "boxes": [[x0, y0, x1, y1], ...], "objectId": 3}]` — pixel coordinates of THAT image (EXIF-rotated as displayed); label 1 = object, 0 = background (default 1). `image` = path relative to `imagesDir`, or a unique basename / stem. Entries with the same `objectId` are the same object (prompts on several frames refine it; on one frame they merge); without `objectId` each entry is a new object (1, 2, ... skipping explicit ids). Every object is tracked forward AND backward through the whole sequence. Used by `sam2_track`, `sam2_auto` and `dynamic` (added to the detections). |
| `maskPrompt` | – | legacy `{"points": [...], "boxes": [...]}` on the first image (= `maskPrompts` entry with `objectId` 1) |
| `maskClasses` | `person, dog, cat, bird, car, truck, bus, bicycle, motorcycle, stroller, shopping cart, umbrella, bag` | `dynamic`: class names for the detector (list, or a comma-separated string) |
| `maskThreshold` | 0.25 | `dynamic`: OWLv2 score threshold (OWLv2 scores are low and uncalibrated: 0.15–0.35 is the useful range; lower = more recall + more false positives) |
| `dilatePixels` | 8 (object-centric `sam2_auto`: 0) | grow the removed objects by N pixels (motion blur, shadow edges) |
| `replaceMasks` | false | overwrite existing masks instead of AND-merging |
| `invertMask` | false | flip the written keep mask |
| `maskSequence` | `auto` | `dynamic`: `video` (ordered → temporal tracking), `unordered` (SAM 2 image predictor per image, no tracking), `auto` = `video` when the names look like a frame sequence (shared prefix + consecutive numbers). Prompt-tracking models always track. |
| `maskDetectStride` | 1 | `dynamic`: run the detector on every N-th frame only (tracking fills the rest; last frame always included) |
| `maskCoverage` | 0.5 | `dynamic`: a detection whose SAM 2 mask is covered by the tracked mask by less than this fraction re-prompts the tracker on that frame |
| `maskBidirectional` | true | also run the backward sweep (objects are tracked to frames before their first prompt / detection) |
| `maskChunkSize` / `maskChunkOverlap` | 64 / 2 | SAM 2 video chunk length and overlap (memory ∝ chunk size, not sequence length) |
| `maskTrackResolution` | 1024 | long side of the frames staged for SAM 2 (SAM 2 runs at 1024² internally anyway; masks are upsampled to the original size) |
| `maskMaxCondFrames` | 4 | SAM 2 attends to the K temporally closest prompted frames (bounds memory-attention cost) |
| `maskPrecision` | `auto` | `auto` = bfloat16 autocast for SAM 2 on MPS/CUDA (measured 2.6x faster per tracked frame on MPS than fp32), `fp32`, `bf16`, `fp16` |
| `maskOffloadState` | true on CUDA, false on MPS/CPU | SAM 2 `offload_state_to_cpu` (frames are always `offload_video_to_cpu`) |
| `saveDetections` | true | `dynamic`: write `outputDir/dynamic_detections.json` (`{"format": "splatdetections", "frames": {"<image>": [{"label", "score", "box"}]}}`) |

**How `dynamic` works.** (1) OWLv2 (`google/owlv2-base-patch16-ensemble`, transformers) scores
every class on every frame; boxes are clipped, near-full-frame boxes dropped, class-agnostic NMS
and contained-box suppression applied. OWLv2 was chosen over Grounding DINO: it is a plain ViT
with linear heads (all ops have MPS kernels, while Grounding DINO's multi-scale deformable
attention has none and runs a slow pure-PyTorch path), classes are scored independently (no
256-token prompt limit / phrase-to-label mapping), Apache-2.0. The detector is freed before
SAM 2 is loaded. (2) The SAM 2 image predictor (sharing the video model's weights) turns each box
into a mask. (3) All detections are tracked as ONE SAM 2 video object: seeded on the first
detected frame, propagated; on any frame where a detection is not yet covered by the tracked
mask (a new object appeared, or tracking lost one) propagation stops, `tracked ∪ detections`
is added as a mask prompt on that frame and propagation resumes. A forward and a backward sweep
cover frames before/after the detections (detector misses, objects entering the frame). Cost is
about one SAM 2 object-frame per frame and sweep regardless of how many instances there are.
(4) The union (tracked ∪ per-frame detection masks ∪ `maskPrompts` objects) is dilated and
written as 0. Unordered photo sets use (2) only, per image.

**Memory / long sequences.** Frames are staged once as JPEG at `maskTrackResolution`; SAM 2's
video predictor only ever holds one chunk of `maskChunkSize` frames. An object's masks on the
`maskChunkOverlap` frames shared by consecutive chunks are fed into the next chunk as mask
prompts, so tracking continues across chunk borders. Results are kept bit-packed per frame. The
cats test (66 frames, 1600×1200) peaked at ~4 GB process footprint on MPS; it does not grow with
the number of frames.

Stats (`{"done": true, "stats": {...}}`) for `dynamic` include `numDetections`,
`framesWithDetections`, `detectionsPerClass`, `framesMasked`, `detectorReprompts`, `chunks`,
`sweeps`, `meanMaskedFraction`, `timings.detectSec`, `timings.segmentSec`.

#### `task = maskpreview`

Single frame, SAM 2 image predictor only — for instant feedback in a click-to-mask UI:

```jsonc
{"task": "maskpreview", "imagesDir": "images", "outputDir": "out",
 "maskPrompts": [{"image": "000123.jpg", "points": [[812, 455, 1]], "boxes": []}],
 "previewImage": "000123.jpg",      // optional; default = image of the first maskPrompts entry
 "previewObjectMask": false,        // true → 255 = object instead of the keep convention
 "invertMask": false, "dilatePixels": 0}
```

All prompt entries on that image are unioned and written to `outputDir/preview.png` (same keep
convention as the final masks: object = 0). Stats: `image`, `preview`, `score`, `area` (object
fraction), `bbox`, `width`, `height`, `modelLoadSec`, `predictSec`. As a one-shot process this
pays Python + torch + SAM 2 start-up every time (several seconds); interactive UIs should use
the mask server.

#### `python -m splatprep maskserver`

Keeps SAM 2 (image predictor) loaded and answers requests line by line; an LRU of the last 4
image embeddings makes re-clicks on the same image cost only the prompt decoder.

```
PYTHONPATH=tools python -m splatprep maskserver [--images DIR] [--out DIR] [--device auto|mps|cuda|cpu] [--ready] [--precision auto|fp32|bf16|fp16]
```

stdin: one JSON request per line. stdout: **exactly one JSON reply per request line, in
order, nothing else** — no greeting (start-up status `maskserver {"ready": true, ...}` and all
library output go to stderr; `--ready` additionally prints the status as the first stdout
line). The server exits on `{"cmd": "quit"}` or EOF. Bad requests produce an `error` reply and
the server keeps running; if SAM 2 cannot be loaded, every request is answered with
`{"error": "cannot load SAM 2: ..."}`. `id` is optional and echoed when present.

```jsonc
// predict (cmd may be omitted)
{"id": 7, "image": "/abs/dataset/images/000123.jpg",  // absolute, or relative to --images
 "points": [[812, 455, 1], [700, 300, 0]],         // [x, y, label] in that image's pixels (as displayed, EXIF-rotated); 1 = object, 0 = background
 "boxes": [[600, 200, 980, 720]],                  // optional; several boxes → union (points go with the first)
 "out": "/path/mask.png",                          // optional; default <outDir>/preview_<stem>.png (overwritten per image)
 "objectMask": true,                               // optional, default true: 255 = OBJECT; false → keep convention (object = 0)
 "dilatePixels": 0}                                // optional
→ {"id": 7, "mask": "/abs/path/preview_000123.png", "score": 0.97, "area": 0.081,
   "bbox": [598, 203, 975, 716], "width": 1600, "height": 1200,
   "embeddingCached": true, "timeSec": 0.05}

{"id": 8, "cmd": "warm", "image": "/abs/.../000124.jpg"} → {"id": 8, "warmed": "/abs/.../000124.jpg", "timeSec": 0.9}  // precompute embedding
{"id": 9, "cmd": "ping"}                           → {"id": 9, "pong": true}
{"cmd": "quit"}                                    → {"bye": true}
// any failure                                     → {"id": 7, "error": "image not found: ..."}
```

The maskserver PNG is 255 = OBJECT by default (the Mac editor inverts it into the keep mask);
`task = maskpreview` writes the keep convention unless `previewObjectMask`. A single positive click
without a box uses SAM's multimask output and keeps the best-scoring mask. To turn a UI
selection into training masks, send the same prompts as `maskPrompts` in a `task = masks`
request (`maskModel: "sam2_auto"` or `"sam2_track"`, or `dynamic` to combine with detections); they are then
tracked through all frames.

## Backends

| id | kind | needs |
|---|---|---|
| `colmap_sift` | COLMAP CLI end to end (feature_extractor → sequential/exhaustive/vocab_tree matcher or our pair list via matches_importer → mapper) | `colmap` (+`glomap`) |
| `sift_lightglue` `superpoint_lightglue` `disk_lightglue` `aliked_lightglue` | LightGlue | `[lightglue]` + a mapper |
| `loftr` | kornia LoFTR, keypoints merged on a half-pixel grid | `[loftr]` + a mapper |
| `mast3r` | official repo, sparse global alignment → poses + dense points | clone + `SPLATPREP_MAST3R_PATH` |
| `vggt` | official repo, feed-forward poses + depth | `[vggt]` |
| `opencv_sift` | OpenCV RootSIFT + ratio test (baseline) | core; matching-only without a mapper |

Sparse backends share one pipeline (`backends/base.py`): pairs → extract → match → COLMAP
database (own sqlite writer, `colmap_db.py`) → OpenCV MAGSAC++ fundamental-matrix verification
written to `two_view_geometries` → mapper. Add a backend by subclassing `FeatureMatchBackend`
(`extract`, `keypoints_of`, `filter_features`, `match`) or `Backend` (`run`), and register it with
`splatprep.models.register()` or `SPLATPREP_PLUGINS="my.module:BACKENDS"`.

## Synthetic data (`splatprep synth`)

Ray-cast numpy renderer: textured ground, a repeated-brick wall, boxes with rich noise,
colour noise, a checkerboard (repeated pattern), a *low-texture* flat box, stripes, a textured
sphere, gradient sky. Per-frame photometric variation (exposure, white balance, gamma,
vignetting, noise; `--photometric 0..1`). Trajectories: `orbit` (60 views), `walk` (60,
forward motion, video-like names), `sparse` (15 wide-baseline views). Output is a valid §2
dataset whose `dataset.json` is the ground truth (+ `synth.json`, optional `depth/`).
Deterministic for a given seed.

## Benchmark (`splatprep benchmark`)

Runs every *available* feature backend (or `--models a,b`) on each dataset, each in its own
subprocess (true per-run peak RSS; `--in-process` to disable), and writes `benchmark.json` +
`benchmark.md`. Datasets: `--dataset` (prepared/synthetic dir with `dataset.json`,
`.splatcapture` dir or zip — ARKit poses as reference — or a plain image dir) and `--synth all`.

Metrics per run: registered-image %, mean reprojection error, mean track length, #3D points,
wall time, peak RSS (+ GPU), and with ground truth: Sim(3)-aligned ATE (RMSE, absolute and
relative to the GT camera extent), absolute rotation error, relative-pose AUC@5°/10°/30°
(error = max(rotation, translation-direction) over all image pairs; unregistered → failure).
If no mapper is installed, feature backends run *matching-only* and report: matches/pair,
inlier ratio (Sampson distance to the GT epipolar geometry < 2 px per 1000 px), verified
precision, and AUC of relative poses estimated from the matches with GT intrinsics. Only pairs
whose GT optical axes differ by ≤ 60° are scored (exhaustive matching would otherwise be
dominated by pairs that cannot overlap).

### benchmark.json schema (version 1)

```jsonc
{
  "format": "splatbenchmark", "version": 1, "splatprepVersion": "0.1.0",
  "createdAt": "2026-11-06T22:00:00Z",
  "host": {"platform": "Darwin", "release": "...", "machine": "arm64", "processor": "arm",
           "python": "3.11.9", "device": "mps", "cpuCount": 12},
  "options": {"matching": "auto", "maxKeypoints": 4096, "resizeMax": 1600, "mapper": "auto"},
  "inProcess": false,
  "models": {"<id>": {"kind": "sparse|dense", "available": true, "sfm": true, "reason": ""}},
  "datasets": [{"name": "orbit", "path": "...", "numImages": 60,
                "groundTruth": "synthetic|dataset|arkit|null", "numGroundTruthPoses": 60}],
  "results": [{
    "model": "superpoint_lightglue", "dataset": "orbit",
    "status": "ok | matching-only | unavailable | failed",
    "error": "only when unavailable/failed",
    "metrics": {
      "numImages": 60, "numRegistered": 60, "registeredPct": 100.0,
      "meanReprojError": 0.61, "meanTrackLength": 5.2, "numPoints3D": 41234,
      "wallTimeSec": 48.1, "peakMemoryMB": 2210.5, "peakGpuMemoryMB": 1500.0,
      "device": "mps", "mapper": "glomap", "numPairs": 1770,
      "pose": {                          // only with ground truth and status ok
        "numGroundTruth": 60, "numEvaluated": 60,
        "aucAt5": 0.93, "aucAt10": 0.96, "aucAt30": 0.99,
        "ateRmse": 0.012, "ateRmseRel": 0.003, "ateMedian": 0.009,
        "rotErrDegMean": 0.21, "rotErrDegMedian": 0.18,
        "relRotErrDegMedian": 0.15, "relTransErrDegMedian": 0.4,
        "alignmentInliers": 60, "alignment": {"scale": 1.0, "rotation": [[...]], "translation": [...]}
      },
      "matching": {                      // when matches were saved and GT exists
        "numPairsEvaluated": 597, "numPairsSkippedNoOverlap": 1173, "maxAxisAngleDeg": 60,
        "meanKeypoints": 2048, "meanMatches": 512, "pooledInlierRatio": 0.8,
        "meanVerifiedMatches": 430, "meanCorrectMatches": 401,
        "meanInlierRatio": 0.78, "meanVerifiedPrecision": 0.93,
        "pairPoseAucAt5": 0.61, "pairPoseAucAt10": 0.74, "pairPoseAucAt20": 0.83
      }
    }
  }],
  "summary": {"<model>": {"status": "ok", "numDatasets": 4, "numSucceeded": 4,
    "meanRegisteredPct": 98.3, "meanReprojError": 0.7, "meanTrackLength": 4.9,
    "meanNumPoints3D": 38000, "meanWallTimeSec": 51.0, "maxPeakMemoryMB": 2400,
    "meanAucAt5": 0.9, "meanAucAt10": 0.94, "meanAteRmseRel": 0.004,
    "meanInlierRatio": 0.8, "meanPairPoseAucAt10": 0.7}}
}
```

Null/missing values mean "not measured". The Mac app merges `summary[model]` with
`splatprep/model_cards.json` (`models[model]`: displayName, summary, strengths, weaknesses,
bestFor, speed {appleSilicon, cuda}, memory, licence {code, weights, commercialUse, uncertain},
installSize, install). Licence entries flagged `uncertain` must be verified before commercial use;
SuperPoint, MASt3R, SegFormer and the original VGGT-1B checkpoint are non-commercial.

## Environment variables

`SPLATPREP_DEVICE`, `SPLATPREP_COLMAP`, `SPLATPREP_GLOMAP`, `SPLATPREP_VOCAB_TREE` (COLMAP
vocab tree file for `colmap_sift` retrieval), `SPLATPREP_MAST3R_PATH`, `SPLATPREP_MAST3R_WEIGHTS`,
`SPLATPREP_VGGT_WEIGHTS`, `SPLATPREP_VGGT_MAX_FRAMES`, `SPLATPREP_SAM2_MODEL`,
`SPLATPREP_SKY_MODEL`, `SPLATPREP_SKY_HEURISTIC=1`, `SPLATPREP_DYNAMIC_DETECTOR` (OWLv2/OWL-ViT checkpoint for `dynamic`), `SPLATPREP_PLUGINS`, `SPLATPREP_DEBUG`.

## Tests

```bash
pip install -e '.[test]'
pytest -q tests
```

Covers: synthetic generator determinism and multi-view consistency, COLMAP database round trip
and schema, COLMAP text/binary model IO and dataset.json export (incl. OpenCV undistortion),
pair selection, Sim(3)/pose metrics, ARKit prior conversion, the CLI/JSONL protocol with a mock
backend (incl. prior alignment and stdout isolation), the COLMAP CLI orchestration against a fake
`colmap` executable, masks (incl. `maskPrompts` parsing, AND-merge, chunk planning, chunked
bidirectional tracking and detector re-prompting against a fake SAM 2, the `dynamic` /
`sam2_track` / `maskpreview` tasks with faked models, and the maskserver protocol), and an
end-to-end benchmark with `opencv_sift`. Torch-based models (LightGlue, LoFTR, MASt3R, VGGT,
SAM 2, OWLv2) are not exercised by the test suite.
