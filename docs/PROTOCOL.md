# SplatWork — shared formats & network protocols

This document is the contract between the three apps:

| App | Folder | Role |
|---|---|---|
| **SplatWork** (macOS) | `mac/` | Viewer, editor, dataset preparation, Metal training, client for remote workers, receiver for iPhone captures |
| **SplatWork Capture** (iOS) | `ios/` | Records LiDAR + ARKit-posed image sequences, sends them to the Mac |
| **SplatWork for Linux** (Linux) | `linux/` | Optional CUDA training server the Mac can offload jobs to |

All JSON is UTF-8, `camelCase` keys, numbers as JSON numbers. Unknown keys MUST be ignored (forward compatibility).

---

## 1. Capture bundle (`.splatcapture`) — iPhone → Mac

A capture is a directory named `<name>.splatcapture`. When transferred it is zipped (plain zip, deflate or store) with the directory as the single top-level entry, or with the bundle contents at the root (receivers must accept both).

```
Kitchen.splatcapture/
  manifest.json
  images/000000.jpg          # colour frames (JPEG or HEIC), sRGB
  depth/000000.bin           # optional: LiDAR depth, float16 little-endian, metres, row-major
  confidence/000000.bin      # optional: uint8 per depth pixel, 0 = low, 1 = medium, 2 = high
  points.ply                 # optional: fused coloured point cloud (binary_little_endian, x y z float, red green blue uchar)
  thumbnail.jpg              # optional
```

### manifest.json

```jsonc
{
  "format": "splatcapture",
  "version": 1,
  "name": "Kitchen",
  "createdAt": "2026-11-06T21:51:00Z",          // ISO-8601
  "device": "iPhone 16 Pro",
  "appVersion": "1.0 (1)",
  "hasLiDAR": true,
  "coordinateSystem": "arkit",                   // world: right-handed, +Y up, metres. camera: +X right, +Y up, looks down -Z
  "imageWidth": 1920, "imageHeight": 1440,       // all frames share this resolution
  "depthWidth": 256,  "depthHeight": 192,        // omitted when hasLiDAR == false
  "frames": [
    {
      "index": 0,
      "timestamp": 12345.678,                    // seconds, ARFrame.timestamp
      "image": "images/000000.jpg",
      "depth": "depth/000000.bin",               // optional
      "confidence": "confidence/000000.bin",     // optional
      "intrinsics": [1450.2, 1450.2, 960.1, 720.4],  // fx, fy, cx, cy in *image* pixels
      "cameraToWorld": [ 16 floats ],            // 4x4 COLUMN-MAJOR (simd_float4x4 memory order), ARKit camera convention
      "trackingState": "normal",                 // "normal" | "limited" | "notAvailable"
      "exposureDuration": 0.0083,                // seconds, optional
      "iso": 125,                                // optional
      "sharpness": 0.81                          // optional 0..1, higher = sharper (motion blur estimate)
    }
  ]
}
```

Notes

* Depth maps are in the camera's frame: the depth pixel (u,v) corresponds to image pixel (u * imageWidth/depthWidth, v * imageHeight/depthHeight). Depth is the Z distance along the optical axis (not the ray length).
* To convert a pose to the OpenCV camera convention used by the training dataset (§2): `c2w_opencv = c2w_arkit * diag(1, -1, -1, 1)`.

---

## 2. Prepared training dataset (`dataset.json`) — Mac → Worker

SplatWork converts every source (photos, video, 360°, captures, COLMAP, Nerfstudio) into this single representation before training. It is also what gets uploaded to a remote worker.

```
dataset/
  dataset.json
  images/<any>.jpg|png        # undistorted pinhole images
  depth/<any>.bin             # optional, float16 LE metres (Z depth), may be lower resolution than the image
  points3d.ply                # optional initial point cloud (x y z float, red green blue uchar)
```

```jsonc
{
  "format": "splatdataset",
  "version": 1,
  "convention": "opencv",       // camera: +X right, +Y down, looks down +Z. World: arbitrary right-handed, metres if known
  "sceneCenter": [0,0,0],       // for camera-orbit defaults & LR scaling
  "sceneExtent": 3.2,           // radius that contains the cameras (used to scale the position learning rate)
  "upVector": [0,-1,0],          // best guess of world up (for viewer orientation only)
  "points": "points3d.ply",     // optional
  "cameras": [
    {
      "id": 0,
      "image": "images/000000.jpg",
      "width": 1920, "height": 1440,
      "fx": 1450.2, "fy": 1450.2, "cx": 960.1, "cy": 720.4,
      "cameraToWorld": [[r00,r01,r02,tx],[r10,r11,r12,ty],[r20,r21,r22,tz],[0,0,0,1]],   // ROW-MAJOR 4x4
      "depth": "depth/000000.bin",       // optional
      "depthWidth": 256, "depthHeight": 192,
      "holdout": false,                   // true = evaluation image, never trained on
      "mask": "masks/000000.png",          // optional 8-bit mask, 255 = keep (others ignored by the loss)
      "sky": "splatprep-sky/masks/000000.png", // optional 8-bit sky mask, 255 = sky: drawn by the learned skybox
                                           // (config skybox auto|on|off, skyboxResolution, skyAlphaWeight);
                                           // GET /v1/jobs/{id}/sky.png + sky.json (worker ≥ 1.3.15)
      "time": 12.48,                       // optional seconds since sequence start (reserved: 4D splatting)
      "group": "iphone",                   // optional source group (reserved: multi-source datasets)
      "exposure": 0.00079                  // optional relative exposure = shutter s × ISO / 100 (seeds appearance correction)
    }
  ]
}
```

---

## 3. Training configuration (`TrainingConfig`)

Shared by the Metal trainer and the CUDA worker. Every key is optional; defaults below. Full user-facing documentation is in `SETTINGS.md`.

| key | type | default | meaning |
|---|---|---|---|
| `iterations` | int | 30000 | total optimisation steps |
| `maxImageSize` | int | 1600 | longest image side used for training (images are downscaled) |
| `shDegree` | int 0–3 | 3 | spherical-harmonic degree (view-dependent colour) |
| `shDegreeInterval` | int | 1000 | raise active SH degree by one every N steps |
| `strategy` | `"mcmc"` \| `"adaptive"` | `"mcmc"` | densification strategy (3DGS-MCMC vs classic adaptive density control) |
| `maxGaussians` | int | 1500000 | hard cap on gaussian count (MCMC budget; ADC safety cap) |
| `densifyStart` | int | 500 | first step densification runs |
| `densifyStop` | int | 15000 | last step densification runs (MCMC uses 25000 if not given) |
| `densifyInterval` | int | 100 | steps between densification passes |
| `densifyGradThreshold` | float | 0.0002 | ADC: screen-space gradient threshold (NDC units) |
| `absGrad` | bool | false | ADC: use AbsGS absolute gradients (threshold auto ×4 → 0.0008) |
| `opacityResetInterval` | int | 3000 | ADC: reset opacities every N steps |
| `pruneOpacity` | float | 0.005 | gaussians below this opacity are dead/pruned |
| `lrPosition` | float | 0.00016 | initial position LR (multiplied by sceneExtent) |
| `lrPositionFinal` | float | 0.0000016 | final position LR (exponential decay) |
| `lrColor` | float | 0.0025 | SH band-0 LR |
| `lrSHRest` | float | 0.000125 | higher SH bands LR |
| `lrOpacity` | float | 0.05 | opacity LR |
| `lrScale` | float | 0.005 | log-scale LR |
| `lrRotation` | float | 0.001 | quaternion LR |
| `ssimWeight` | float | 0.2 | λ in (1-λ)·L1 + λ·(1-SSIM) |
| `opacityReg` | float | 0.01 | MCMC L1 opacity regulariser (0 for ADC) |
| `scaleReg` | float | 0.01 | MCMC L1 scale regulariser (0 for ADC) |
| `noiseLR` | float | 500000 | MCMC position noise scale |
| `antialiased` | bool | true | Mip-Splatting style 2D screen-space filter with opacity compensation |
| `depthWeight` | float | 0.0 (0.1 if depth present) | L1 depth supervision weight |
| `background` | `"black"`\|`"white"`\|`"random"` | `"random"` | background used while training |
| `initialOpacity` | float | 0.1 | opacity of initial gaussians |
| `randomInitCount` | int | 100000 | gaussians when no point cloud is available |
| `seed` | int | 42 | RNG seed |
| `previewInterval` | int | 500 | how often a preview PLY snapshot is produced (remote only) |
| `evalInterval` | int | 2000 | how often PSNR on holdout images is computed |
| `transientMasking` | bool | true | robust training: ignore solid, strongly-inconsistent regions (moving objects) after 30 % of training |
| `transientSensitivity` | float | 3 | outlier threshold as a multiple of the image's median error (min 0.10) |
| `transientStart` | int | 1000 | earliest step for transient masking (actual start = max(this, 0.3 × iterations)) |
| `transientMaxViewGap` | float | 15 | transient masking is skipped when the median nearest-view gap (degrees, view angle + camera offset at scene scale) exceeds this |
| `depthEdgeAware` | bool | true | depth loss weighted by exp(−10·|∇I|) of the target image |
| `anisotropyReg` | float | 0.1 | penalty on max(longest/shortest axis − `anisotropyMaxRatio`, 0) per splat |
| `normalWeight` | float | 0.05 | depth-normal consistency: normals of the rendered depth (central differences) vs normals of the photo's depth map, loss (1−c)/2 + √((1−c)/2), summed over valid opaque pixels / pixel count |
| `normalStart` | int | 1000 | step at which the normal loss starts |
| `flatnessReg` | float | 0 | per-splat penalty on the smallest axis's share s_min² / Σ s² (flat discs) |
| `depthPriors` | string | `"auto"` | Linux worker: photos without sensor depth get a monocular depth map (Depth Anything V2 Small) fitted to the dataset's points, for the normal loss. `auto` = when no training photo has depth and `normalWeight` > 0; `on`; `off` |
| `monoDepthWeight` | float | 0 | depth-loss weight for those predicted maps (0 = normals only) |
| `anisotropyMaxRatio` | float | 10 | axis ratio above which the needle penalty applies |
| `appearanceCorrection` | bool | true | per-photo colour correction: rendered RGB → M·rgb + b (3×3 + 3 per training image), seeded from camera `exposure`, used only in the loss |
| `bilateralGrid` | bool | false | per-photo *local* colour correction (16×16 image cells × 8 brightness levels of 3×4 colour transforms per training image), used only in the loss; mean over photos pinned to identity; replaces `appearanceCorrection` when on. `bilateralGridLR` (2e-3), `bilateralGridTV` (10) |
| `progressiveResolution` | bool | false | coarse-to-fine: ¼ → ½ → ¾ → full image resolution over the first half of the densification window (worker ≥ 1.3.9) |
| `decoupledSSIM` | bool | false | with per-photo colour correction, SSIM contrast/structure use the uncorrected render (luminance the corrected one) (worker ≥ 1.3.9) |
| `backgroundShell` | int | 0 | splats on a far-field sphere (3× scene size) for distant land/sky beyond the capture (worker ≥ 1.3.9, Mac) |
| `optimizerPrecision` | `"fp32"` \| `"compact"` | `"fp32"` | CUDA worker: Adam moment storage. `"compact"` keeps the higher-SH-band (`shN`) moments as 8-bit codes with one fp32 scale per splat and all other moments as bfloat16 (stochastic rounding, same Adam update) — at SH degree 3 training memory per splat drops from ~944 B to ~622 B (params + grads + optimizer state), so more splats fit on the GPU. Ignored by the Metal trainer |
| `appearanceLR` | float | 0.02 | Adam LR of the per-photo corrections (each photo updates only when drawn) |
| `appearanceReg` | float | 0.001 | L2 pull of each correction towards its exposure-seeded start |

---

## 4. Worker HTTP API — Mac ↔ Linux

* Default port **7878**, plain HTTP on the LAN (optionally behind TLS reverse proxy).
* Discovery: the worker advertises Bonjour/mDNS service type **`_splatworker._tcp`**, TXT record `version=1`, `gpu=<name>`, `auth=1|0`, and (worker ≥ 1.3.10) `addrs=<ip>:<kind>,…` listing its LAN addresses preferred first (`ethernet` before `wifi`). `GET /v1/info` returns the same list as `"addresses": [{"address":"192.168.1.20","kind":"ethernet"}, …]`. SplatWork moves a discovered worker onto its first ethernet address that answers, falls back to the others if that address stops answering, and leaves manually added workers on the address the user typed. A discovered worker can be hidden from the list (Compute page) and added manually instead.
* Worker kinds: SplatWork's "act as a worker" adds TXT `kind=mac` (and `"kind":"mac"` in `/v1/info`); SplatWork for Linux sends none. Older versions of SplatWork Capture advertised `kind=iphone`; iPhones no longer act as workers, and clients ignore workers of that kind.
* Auth: if the worker has a token configured every request needs `Authorization: Bearer <token>`. Unauthenticated → `401`.
* Errors: non-2xx with body `{"error": "message"}`.

| Method & path | Body | Response |
|---|---|---|
| `GET /v1/info` | – | `{"name","version":1,"gpus":[{"name","memoryMB","freeMemoryMB"}],"backend":"gsplat 1.x","busy":bool,"queueLength":int}` |
| `POST /v1/datasets` | raw zip of the `dataset/` directory (`Content-Type: application/zip`) | `{"datasetId": "abc"}` |
| `POST /v1/jobs` | `{"datasetId":"abc","name":"Kitchen","config":{TrainingConfig}}` | `{"jobId":"j1"}` |
| `GET /v1/jobs` | – | `[JobStatus]` |
| `GET /v1/jobs/{id}` | – | `JobStatus` |
| `POST /v1/jobs/{id}/cancel` | – | `JobStatus` |
| `DELETE /v1/jobs/{id}` | – | `204`, deletes job files |
| `GET /v1/runtime` | – | optional components (`installed`: lightglue, pycolmap, sam2, transformers, processing) and remote-install progress (`state` idle/running/done/failed, `fraction`, `message`, `log`) (worker ≥ 1.1.2) |
| `POST /v1/runtime/install` | `{"component":"processing"}` | starts installing a fixed component set into the GPU runtime; poll `GET /v1/runtime` |
| `GET /v1/jobs/{id}/preview.splat` | – | compact live snapshot (antimatter15 `.splat`, 32 B/splat, the ≤400 k most visible splats, no SH); header `X-Preview-Step`; `404` until first preview (worker ≥ 1.1) |
| `GET /v1/jobs/{id}/preview.ply` | – | full snapshot PLY (worker 1.0 and test backends only), `404` otherwise |
| `GET /v1/jobs/{id}/result.ply` | – | final PLY once `state == "completed"` |
| `GET /v1/jobs/{id}/log` | – | `text/plain` training log |
| `POST /v1/jobs/{id}/prioritize` | – | `JobStatus`; moves a **queued** job to the front (priority above every other queued job); `409` otherwise; desktop app on the worker PC only (`403` for others) (worker ≥ 1.4.0, capability `priority`) |
| `POST /v1/census` | `{"datasetId","jobId":<completed train job>}` or, from the desktop app on the worker PC only, `{"datasetId","plyPath":"/abs/scene.ply"}` | `{"jobId":"c…"}` — Auto clean census, a job of kind `census` (worker ≥ 1.4.0, capability `census`) |
| `GET /v1/jobs/{id}/census.npz` | – | numpy `.npz`: `peak` (float32, per splat: largest per-photo summed blending weight, in native pixels) and `views` (uint32: photos where that sum ≥ 0.02) |

**Shared queue (worker ≥ 1.4.0).** One job per GPU at a time; a free GPU takes the queued job with the highest
`priority`, oldest first; a running job is never pre-empted. `JobStatus` gains `"priority"` (int) and `"origin"`
(`"local"` for the desktop app on the worker PC, else the client's address). Only that app may submit `"priority"`
(in the `/v1/jobs`, `/v1/process` and `/v1/census` bodies, 0–1000): it uses 1 so the person at that PC goes ahead of
jobs queued by other machines; everybody else's requests queue at 0.

**Pause / resume and checkpoints (worker ≥ 1.4.0, capability `pause`).**

| Method & path | Body | Response |
|---|---|---|
| `POST /v1/jobs/{id}/pause` | – | `JobStatus`. A **queued** job (any kind) is held back at once (`state` `"paused"`, never started). A **running training** job is asked to stop: the trainer saves its full state (splats, optimizer moments, densification state, RNGs, step) in the job folder and exits, and the job becomes `"paused"` (`message` is `"pausing"` meanwhile) — the GPU goes to the next queued job. `409` for finished jobs and running processing / census jobs; no-op when already paused |
| `POST /v1/jobs/{id}/resume` | – | `JobStatus`; a **paused** job goes back into the queue (normal priority rules) and training continues from the saved step; `409` while it is still pausing or if it isn't paused |
| `POST /v1/jobs/{id}/checkpoint?wait=<s>` | – | `JobStatus`; asks a running training job to write `checkpoint.ply` of its current splats without stopping. With `wait` (≤ 300 s) the answer comes once it is written (or the job stops); `checkpointStep` says which step it holds. A paused job already has one (written when it paused). `409` for jobs that are queued, finished or not training |
| `GET /v1/jobs/{id}/checkpoint.ply` | – | the latest checkpoint (3DGS PLY, same layout as `result.ply`); header `X-Checkpoint-Step`; `404` until one was written |

Who may pause or checkpoint a job: only the client that submitted it (its address equals the job's `origin`;
the desktop app on the worker PC counts as `"local"`); others get `403`. The desktop app doesn't park other
computers' jobs, since their apps may have no Resume button. It may resume any paused job. A client whose address
changed since it submitted (e.g. Wi-Fi → ethernet) can't pause that job. A paused job of another computer
resumes by itself after an hour and when the worker restarts; a paused job of the desktop app is cancelled
(its saved state removed) after 14 days. `JobStatus.pausedAt` (epoch seconds) says when it was paused.

A job taken from the queue is `"loading"` straight away; a pause or cancel that arrives before it really starts
holds or stops it without running it.

`"paused"` is neither active nor finished: `JobStatus` keeps `step`, `elapsedSeconds` (carried on after resuming)
and `checkpointStep`; `etaSeconds` is `null`; the job keeps its dataset (`DELETE /v1/datasets/{id}` → `409`);
`cancel` and `DELETE` work as for a queued job. Paused jobs of the desktop app survive a worker restart, and a resumed job that was
running when the worker stopped returns to `"paused"` (from its last pause) instead of `"failed"`. Clients that
don't know the state see an unknown, non-terminal value and keep polling (the Mac's `JobState` is tolerant).

**The desktop app on the worker PC.** "Local" means more than a 127.0.0.1 connection (SSH tunnels, `tailscale serve`
and reverse proxies arrive from loopback too): the app generates a secret at every launch, passes it to the server
it starts in the environment (`SPLATWORK_LOCAL_SECRET`, read once and removed), and sends it with every request as
`X-SplatWork-Local: <secret>` from 127.0.0.1. Without both, a request is remote. A server started without the
secret (e.g. `splatworker serve --allow-local-datasets` by hand) keeps these endpoints off and doesn't advertise
`local`.

**The desktop app's jobs, seen from other computers (worker ≥ 1.4.0).** They stay in `GET /v1/jobs` (so queue
estimates still work) with their state, progress, priority and GPU, but `name` is `"Job on this PC"`,
`datasetId` is `""` and `message` / `outputDatasetId` are null; their files (log, previews, result, census,
checkpoint, cameras, output) answer `404`, and `cancel` / `DELETE` answer `403`. Other clients' jobs show
`origin` `"remote"` instead of their address (a client still sees its own). Anybody may still cancel or delete
other computers' jobs, as before.

**Local datasets (worker ≥ 1.4.0, `--allow-local-datasets` + the secret, capability `local`, the desktop app only).**
`POST /v1/datasets/local` `{"path":"/abs/folder"}` → `{"datasetId",…,"path"}` registers a folder *in place* (nothing
copied; deleting the dataset never touches the folder), and `POST /v1/datasets/{id}/export-local`
`{"path":"/abs/folder"}` → `{"files":n}` merges a dataset (e.g. a processing job's `outputDatasetId`) into a folder
with hard links — only into a folder registered with `/v1/datasets/local` (or below it); `/`, the home folder and
anything holding the worker's data can't be registered. The desktop app on the worker PC (SplatWork for Linux) uses these
to run its New Project pipeline on its own project folders without uploading. Other clients never see these
datasets: `GET /v1/datasets` leaves them out (and never lists the folder path), and jobs, compose, archive and
delete answer `404` for them.

**Mask editor clicks (worker ≥ 1.4.0, capability `maskpreview`).** `POST /v1/process` also accepts splatprep's
`"task": "maskpreview"` (it existed in splatprep for the Mac's local mask server; the worker now lets it through
`validate_request`): one image, SAM 2 point / box prompts → `outputDir/preview.png`.

```jsonc
{"task": "maskpreview", "imagesDir": "images", "outputDir": "_click",
 "previewImage": "000012.jpg",                       // required in practice; relative, no ".."
 "maskPrompts": [{"image": "000012.jpg", "points": [[x, y, 1], [x, y, 0]]}],   // required, non-empty; image pixels
 "previewObjectMask": true}                          // 255 on the object (default: keep convention, object = 0)
```

SplatWork for Linux's mask editor sends one per click (the model loads per job, a few seconds on the PC) and tracks the
clicked objects with the existing `masks` task (`"maskModel": "sam2_track"`, `"maskPrompts"`, `"replaceMasks": true`).
Its edits are stored as a mask layer of the project's dataset (`splatprep-masks/manual/masks` 255 = keep,
`…/restore` 255 = keep whatever other layers say), so they survive "Prepare again".

**Stereo footage (worker ≥ 1.4.0, capability `stereo`).** A `task: frames` request may carry an optional
`"stereo"` object (absent = an ordinary video, as before); see `tools/splatprep/stereo.py`:

```jsonc
"stereo": {"layout": "auto" | "sideBySide" | "topBottom" | "dualStream" | "separateFiles",
           "rightVideo": "_right-eye.mp4",   // separateFiles: relative to the dataset, no ".."
           "rightTimeOffset": 0.0, "swapEyes": false, "anamorphic": null,
           "baseline": 0.064,                // metres, overrides calibration
           "horizontalFOV": 90.0}            // one eye, degrees, when there is no calibration / focal metadata
```

`auto` follows the Mac (`StereoVideoIngest.detect`): two equal video streams → `dualStream`; aspect / name →
side-by-side (anamorphic when the frame is narrower than 2.6:1) or top-bottom. Output: `images/<frame>_L.jpg` and
`_R.jpg`; `frames.json` entries carry `group` + `eye` (+ `focalPx/cx/cy` when known) and the file gets
`"stereoLayout"` and `"rig"` (the Mac's `StereoRig`: `baseline`, `leftIntrinsics`, `rightIntrinsics`, `origin`, plus
`parallel` — the eyes can be treated as a rectified pair — and `lens`). Arcturus Vision files carry their factory
calibration (`offlineCal` container tag): the baseline, per-eye intrinsics, which stream is the left eye, and the
fisheye image circle (written as `masks/<frame>_L.png` keep masks). Only the chosen frames' other eye is decoded on
demand (two decoded frames held). Apple MV-HEVC spatial video needs VideoToolbox and stays Mac-only.

**Rigs with a baseline (worker ≥ 1.4.0).** In an `sfm` request's `rig.groups`, a value may be
`{"R": R, "t": [x, y, z]}` instead of `R` (`t` = camera centre in the rig frame, metres). Sensors are told apart by
rotation and centre, views of one frame with a baseline are matched with each other, and the rig config holds the
translation, so a stereo pair maps in metres. SplatWork for Linux sends this for parallel stereo pairs with a known
baseline (left = `I`, right = `{"R": I, "t": [b, 0, 0]}`). An older splatprep can't parse such a group, so the
Node sends it only to its own worker (same version).

**Whole pipeline jobs (worker ≥ 1.6.0, capability `pipeline`).** When a project's processing *and* training both
go to one worker, the client sends the source and every setting once and the worker runs each stage on its own, so
the client can go offline for the whole job and only download the result (and the prepared dataset) when it is back.

| Method & path | Body | Response |
|---|---|---|
| `POST /v1/pipeline` | `{"datasetId","name","pipeline":{…},"priority"?}` | `{"jobId":"w…"}` — a job of kind `pipeline` |

The upload is either the video (`POST /v1/datasets/file`, with a `frames` stage) or frames the client ingested itself
(photos, 360° photo views): `images/` + `frames.json` (the `FrameSet` of `task: frames`, §6; `root` is ignored) and
any masks it names.

```jsonc
"pipeline": {
  "frames": {"request": {ProcessingRequest, task "frames"}},          // optional: split the uploaded video here
  "staticRegions": false,                                              // mask parts of the frame that never change
  "fisheyeMasks": true,            // circular-fisheye footage: mask the fixed lens border (false when masks are done)
  "masks": {"request": {task "masks", "maskModel", …}, "dilatePixels": 4},  // optional
  "sky": {"request": {task "masks", "maskModel": "sky", "invertMask": true}},  // optional; no sky model → skipped
  "sfm": {"request": {features, matching, mapper, cameraModel, resizeMax, maxKeypoints, priorPoses},
          "sequential": true,      // frames are a video in order (outlier check along the path)
          "enforceRig": true},     // 360° views: re-derive each frame's views from its registered ones
  "holdoutInterval": 8,            // every Nth camera is a holdout (null = none)
  "level": true,                   // up → −Y, scene centre → origin (SplatWork's normalizeOrientation)
  "train": {"config": {TrainingConfig}},                               // optional
  "checkpoints": {"frames": "…", "masks": "…", "poses": "…", "done": "…"}   // opaque, echoed in pipeline.json
}
```

Stages, in order: `frames` → `static` (fixed parts of the frame; also run for circular-fisheye footage unless `fisheyeMasks` is false, else
`skipped`) → `masks` → `sky` → `sfm` (camera solve, then SplatWork's outlier check, 360° rig completion and dataset
assembly with masks / sky / holdout) → `level` → `train`. The worker fills in what only the frames tell
(`masksDir`, `skyMasksDir`, `singleCamera`, the 360° `rig`, `cameraModel: "fisheye"` for circular fisheye) and
applies the mask rules of SplatWork (binarised, kept region eroded by `dilatePixels`, ANDed into `masks/<stem>.png`;
frames whose mask keeps < 1 % stay unmasked, and the stage fails when that is more than half of them; sky only for
frames with > 0.2 % sky). A solve that places fewer than 30 % of ≥ 10 frames fails the job.

Each GPU step is an ordinary child job (`process` / `train`) in the shared queue with `"parentJobId"` = the pipeline,
the pipeline's priority and origin. The pipeline itself never waits in the queue: it is `running` from the start
(`paused` while its current step is paused), and its `JobStatus` adds:

```jsonc
{"kind": "pipeline", "stage": "sfm",               // current stage id
 "stageProgress": 0.4,                             // how far the current stage is (0…1)
 "stages": [{"id": "frames", "title": "Extracting frames", "state": "done",   // pending|running|done|skipped|failed|cancelled
             "jobId": "p…", "prepStage": "frames", "summary": {…}, "message": null, "attempts": 1}, …],
 "trainJobId": "j…",                               // the training child, once it exists
 "step": 412, "totalSteps": 1000,                  // whole job ×1000; message = "<stage>: <step message>"
 "outputDatasetId": "d…"}                          // the prepared dataset, from the end of `level` on
```

`prepStage` maps a stage to SplatWork's checkpoints (`frames`, `masks`, `poses`, `done`; `null` for training).
While training runs the pipeline mirrors the child's `loss`, `psnr`, `evalPsnr`, `numGaussians`, `previewStep`,
`checkpointStep` and `etaSeconds`; `GET /v1/jobs/{id}/preview.splat|preview.ply|checkpoint.ply` answer with the
child's files, and `result.ply` (+ `sky.png` / `sky.json`) once the pipeline is `completed`. `pause` / `resume` /
`checkpoint` act on the current step (pause: its training, or a step still queued; `409` otherwise); `cancel`
cancels the current step and ends the job; `DELETE` removes the children and the working dataset too.

The prepared dataset (`output.zip` of the pipeline, or `GET /v1/datasets/{outputDatasetId}/archive.zip`) holds the
levelled `dataset.json`, `points3d.ply`, `frames.json`, `masks/`, `splatprep-*` layers, `sfm/` (undistorted copies,
when the solve made them) and `.pipeline/`: `pipeline.json` (`{"format":"splatpipeline","version":1,"jobId",
"stages":[…],"checkpoints"}`), `frames.frames.json` (+ `frames.files/`: the mask files that list names, as they were
then), `masks.frames.json` (+ `masks.files/`), `poses.dataset.json` + `poses.points.ply` (the solved dataset before
levelling) — enough for a client to record the same per-stage checkpoints a local run leaves. A video's images
are typically fetched later (`archive.zip?only=images`), as with chaining.

Restarts: the stages and their child jobs are persisted. After a worker restart the pipeline continues at its first
unfinished stage: a child still queued or paused is waited for (a paused training resumes from its pause, see
above), a processing child the restart interrupted runs again (at most 3 attempts per stage). If nothing usable was
prepared, a failed or cancelled pipeline removes its working dataset.

```jsonc
// JobStatus
{
  "jobId": "j1", "name": "Kitchen",
  "state": "queued",            // queued | loading | training | completed | failed | cancelled (| paused, capability "pause")
  "step": 1200, "totalSteps": 30000,
  "loss": 0.051, "psnr": 24.3,  // psnr is the latest eval/holdout or train-batch PSNR
  "numGaussians": 812345,
  "elapsedSeconds": 64.2, "etaSeconds": 1540.0,
  "previewStep": 1000,          // step of latest preview.ply, null if none
  "checkpointStep": 900,        // step of checkpoint.ply (capability "pause"), null/absent if none
  "message": "optional human readable status or error"
}
```

---

## 5. Capture transfer — iPhone → Mac

SplatWork listens (while the app is open and "Receive from iPhone" is enabled) and advertises Bonjour service type **`_splatcapture._tcp`** with TXT `name=<Mac name>`, `version=1`.

* `GET /v1/hello` → `{"name":"My MacBook Pro","app":"SplatWork","version":1}`
* `POST /v1/captures` — body: zip of the `.splatcapture` bundle, headers `Content-Type: application/zip`, `X-Capture-Name: <url-encoded name>`, optional `X-Pairing-Code: <6 digits>`.
  → `{"ok":true,"savedAs":"Kitchen.splatcapture"}`

Pairing: the Mac shows a 6-digit code in its "Receive from iPhone" panel. If "Require pairing code" is on, uploads without the matching code get `403`.

---

## 6. Processing toolkit (`splatprep`) and processing jobs

Learned feature extraction/matching, SfM and masking live in a shared Python package `tools/splatprep/` that runs both on the Mac (PyTorch MPS, installed on demand into `~/Library/Application Support/SplatStudio/python`) and inside the Linux worker (CUDA). The Mac app drives it as a subprocess:

```
python -m splatprep run --request request.json --progress-jsonl -
```

and the worker exposes the same request over HTTP. A **ProcessingRequest**:

```jsonc
{
  "task": "sfm" | "masks" | "benchmark",
  "imagesDir": "images",                 // relative to the uploaded/working dir
  "outputDir": "out",
  "features": "superpoint_lightglue",    // see ids below (task = sfm)
  "matching": "sequential" | "exhaustive" | "retrieval" | "auto",
  "retrievalTopK": 30,
  "sequentialOverlap": 10,
  "mapper": "glomap" | "colmap" | "auto",   // auto: ≤ 1000 images run both and keep more-registered / longer-track result; above: global (view-graph focal calibration when EXIF focal is missing); EXIF/ARKit focal lengths are held fixed while mapping, then freed in one final BA kept only if focal moves < 8 %
  "cameraModel": "auto" | "pinhole" | "opencv" | "fisheye",
  "singleCamera": true,
  "maxKeypoints": 4096,
  "resizeMax": 1600,
  "rig": {"kind": "multiview",           // optional, 360° video cut into views (worker ≥ 1.3.11):
          "groups": [{"000000_v00.jpg": R, …}, …],  // one dict per 360° frame in time order; R = rig_from_cam
          "intrinsics": [fx, fy, cx, cy]},          // the views' exact pinhole intrinsics (fixed while mapping)
                                         // → rig pairs (next frame: directions ≤ 50° apart; up to rigWindow (3)
                                         // frames on: same direction; rigLoopPairs (2) retrieval pairs per view),
                                         // then COLMAP rig mapping of whole frames (global unless mapper is
                                         // colmap/pycolmap); useRig: false maps the views independently
  "skyMasksDir": "splatprep-sky/masks",  // optional sky masks (255 = sky): no keypoints on the sky
  "priorPoses": "priors.json",           // optional: ARKit / known poses used as priors
  "priorMode": "auto" | "refine" | "solve", // auto: refine when ≥ 90 % of images have prior poses + intrinsics —
                                         // look-at pairs, triangulate with the priors, pose-only BA ×2 (no mapper,
                                         // capture intrinsics kept; falls back to the priors if implausible);
                                         // solve: full SfM, priors only for pairs + Sim3 alignment
  "maskModel": "sam2_auto" | "sky" | "person",   // task = masks
  "maskPrompt": { "points": [[x,y,label]], "boxes": [[x0,y0,x1,y1]] },   // optional, first image coords
  "invertMask": false
}
```

Output for `sfm`: a prepared dataset (`dataset.json` + `images/` + `points3d.ply`, §2) in `outputDir`; for `masks`: `masks/<image stem>.png` (8-bit, 255 = keep / foreground). Progress lines (JSONL on stdout): `{"stage":"features","progress":0.42,"message":"..."}`; final line `{"done":true,"stats":{...}}` or `{"error":"..."}`.

Feature/matcher ids: `colmap_sift`, `sift_lightglue`, `superpoint_lightglue`, `disk_lightglue`, `aliked_lightglue`, `loftr`, `mast3r` (dense, few-image), `vggt` (feed-forward poses). Unavailable models report `{"error":"model unavailable: ..."}`.

### Worker endpoints

| Method & path | Body | Response |
|---|---|---|
| `POST /v1/process` | `{"datasetId":"…","request":{ProcessingRequest}}` (dataset zip contains `images/` and optional `priors.json`) | `{"jobId":"p1"}` |
| `GET /v1/jobs/{id}` | – | `JobStatus` with `"kind":"process"`, `step/totalSteps` = progress ×1000, `message` = stage |
| `GET /v1/jobs/{id}/output.zip[?skip=a,b][&only=c]` | – | zip of `outputDir` when completed (built on request from the output dataset; `skip`/`only` = top-level folders) |
| `POST /v1/datasets/compose` | `{"layers":[{"datasetId":"A"},{"datasetId":"B","prefix":"sub"}]}` | `{"datasetId":"…","hasDatasetJson":bool,"bytes":n}` — union of the layers' files (hard links, later layers win) |
| `GET /v1/datasets/{id}/archive.zip[?only=…][&skip=…]` | – | the dataset's files as a zip |

`GET /v1/info` adds `"capabilities": ["train","process"]` (+ `"pipeline"`, §4) and `"models": ["superpoint_lightglue", …]` (those actually importable).

**Chaining (worker ≥ 1.3.8, capability `chain`).** A completed processing job keeps its output on the worker as a
dataset of its own, `JobStatus.outputDatasetId`, which the next request can use directly (`/v1/process` or, with a
`dataset.json`, `/v1/jobs`). The Mac uses this so a video split on the worker (`task: frames`) is masked, solved and
trained there without its frames crossing the network again: it downloads the output without `images/`, records
`.remote-store.json` (`{"worker","datasetId","dirs":["images"],"local"}`) in the project, sends later steps only what
changed (masks, priors, `dataset.json`) and composes them over the stored images, and fetches the images in the
background (`archive.zip?only=images`) for anything that needs them on the Mac.

## 7. Shared compute ("Both" mode)

The Mac app is always the coordinator. Per project the user picks **This Mac**, **Remote worker**, or **Both**:

* Stage placement: each stage (frame extraction, masking, SfM, training) is assigned to the device with the best benchmark score that supports the chosen model (e.g. LightGlue on CUDA, Object Capture on the Mac).
* Block-parallel training for large scenes: the scene is partitioned into K spatial blocks (CityGaussian/VastGaussian style). Each block is a training job with `TrainingConfig.region = {"min":[x,y,z],"max":[x,y,z],"margin":0.15}`: only gaussians inside the expanded box are kept/densified, and only cameras that see the block are used. Blocks are distributed across devices proportional to measured throughput; the Mac merges results keeping, for each block, gaussians whose centre lies inside the *un-expanded* box.
* Workers accept `TrainingConfig.region` (optional) and `TrainingConfig.cameraIds` (optional list restricting which cameras are used).
* SplatWork for Linux's Studio (worker ≥ 1.4.0) coordinates the same way with only the existing endpoints: it prepares
  on its own GPU, then trains on one chosen worker or block-parallel on itself + every online worker (SplatWork
  Nodes and Macs with "Act as a worker"), using the Mac's split, placement and merge rules. Uploads carry a
  `Content-Length`; on `chain` workers the photos stay as a dataset of their own and later runs compose
  `dataset.json` / masks / points over them; Macs get one full upload per run, shared by their blocks.

---

## 8. Stereo and multi-sensor captures (capture bundle v2 extensions)

Capture bundles (§1) may carry stereo rigs and extra sensors. Readers that only understand v1 can still use `image` (the left/primary eye).

```jsonc
{
  "format": "splatcapture", "version": 2,
  "source": "steamframe" | "iphone" | "arcturus" | "stereo-video",
  "rig": {                                  // optional; present for stereo captures
    "sensors": [
      { "id": "left",  "kind": "rgb", "width": 2464, "height": 2464, "intrinsics": [fx,fy,cx,cy],
        "distortion": { "model": "pinhole" | "opencv" | "fisheye" | "equidistant", "params": [k1,k2,p1,p2,k3] },
        "sensorToRig": [16 floats, column-major] },
      { "id": "right", "kind": "rgb", ... },
      { "id": "ir0", "kind": "ir", ... }     // e.g. headset tracking cameras
    ],
    "baseline": 0.064                        // metres between left and right (informative)
  },
  "colorEncoding": "srgb" | "hlg" | "pq",    // 10-bit HDR captures are tone-mapped on import
  "frames": [
    {
      "index": 0, "timestamp": 1.234,
      "image": "images/left/000000.jpg",     // primary (left) eye
      "images": { "left": "images/left/000000.jpg", "right": "images/right/000000.jpg" },
      "rigToWorld": [16 floats column-major], // pose of the rig (head) in the "coordinateSystem" frame
      "cameraToWorld": [...],                // = rigToWorld * sensorToRig(left), kept for v1 readers
      "intrinsics": [...],                   // left eye
      "depth": "depth/000000.bin",           // optional, aligned to the LEFT image, float16 metres Z-depth
      "depthWidth": 640, "depthHeight": 480,
      "depthSource": "lidar" | "ir-stereo" | "rgb-stereo"
    }
  ]
}
```

`coordinateSystem` may also be `"openxr"` (right-handed, +Y up, camera looks down -Z — same as ARKit) or `"opencv"`.
