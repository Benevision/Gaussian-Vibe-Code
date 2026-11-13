# Architecture

SplatWork has four parts that share the formats and protocols in [PROTOCOL.md](PROTOCOL.md):

| Part | Folder | Language | Role |
|---|---|---|---|
| **SplatWork** | `mac/` | Swift, SwiftUI, Metal | Coordinator: viewer, editor, dataset preparation, Metal trainer, client for workers, receiver for iPhone captures |
| **SplatWork for Linux** | `linux/` | Python, FastAPI, gsplat, GTK 4 | Optional CUDA server for training and processing jobs, packaged as a Flatpak |
| **splatprep** | `tools/splatprep/` | Python, PyTorch | Learned SfM, masks, video frames and benchmarks. Runs on the Mac (MPS, Models pack) and inside the worker (CUDA) |
| **SplatWork Capture** | `ios/` | Swift, ARKit | iPhone capture of posed frames, LiDAR depth and points |

(`steamframe/` holds an experimental headset capture prototype. Its hardware access is unverified.)

```
 iPhone (SplatWork Capture)                    Mac (SplatWork)                          Linux PC (SplatWork for Linux)
 ──────────────────────                    ──────────────────                          ───────────────────────
 ARKit + LiDAR frames                      Sources ─▶ DatasetBuilder                   FastAPI server (:7878)
 .splatcapture bundle ──POST /v1/captures─▶  ingest ─▶ masks ─▶ poses ─▶ dataset/      JobManager (queue, 1 job/GPU)
   (Bonjour _splatcapture._tcp)              │         │        │                      SubprocessBackend (spawn)
                                             │         └────────┴──── /v1/process ───▶   ├─ splatprep (frames, masks, sfm)
                                             │   (video ─ /v1/datasets/file ─────────▶)   └─ trainer.py (gsplat)
                                             ▼                                         datasets/<id>  jobs/<id>
                                           TrainingSession                                  ▲
                                             ├─ MetalTrainer (This Mac)                     │
                                             ├─ WorkerClient ─ /v1/datasets, /v1/jobs ──────┘
                                             └─ ComputeScheduler (Both: blocks on all devices)
                                                    ▲ BenchmarkStore (Compute → Benchmark)
                                             ▼
                                           GaussianCloud ─▶ SplatScene ─▶ SplatRenderer ─▶ edit / export
```

## Mac app (`mac/Sources/SplatStudio`)

```
App/        SplatStudioApp (menus), AppModel (document, editing, camera, inspector tabs, Show Live View), Sessions
            (TrainingSession, PrepSession, PendingRemoteJob), ProjectLibrary (project.json, the Projects list), CLI
            (headless prepare/train/render/eval/benchmark/fetch-job), UISnapshots
Core/       GaussianCloud, PreparedDataset (dataset.json), TrainingConfig (+presets), Orientation, SettingsCatalog
            (single source of every setting's help text; --dump-settings writes SETTINGS.md)
Formats/    PLY (any PLY, 3DGS PLY r/w), .splat, SPZ (gzip via the Compression framework)
Shaders/    Metal sources as Swift strings, compiled at launch (no offline metal compiler needed)
Render/     MetalContext, GPU radix sort, SplatScene, SplatRenderer, ViewerCamera, Coverage, MediaExport
Train/      MetalTrainer (optimisation loop, densification, peak/census passes), TrainingData, AutoClean
Pipeline/   Sources (photos, video, 360, stereo, stereo depth, RemoteFrameExtraction), Poses (Object Capture,
            SplatPrepRunner local/remote), Importers (capture bundle, COLMAP, Nerfstudio), Masking, DatasetBuilder,
            PrepCheckpoints
Remote/     WorkerClient, WorkerBrowser (Bonjour + manual), RemoteJobWatcher, RemoteStore, ComputeScheduler,
            BlockTraining, PauseController, SystemBenchmark (+BenchmarkStore, TrainingTimeEstimate), CaptureReceiver
UI/         ContentView (sidebar: Viewer, iPhone Inbox, Compute, Models & Benchmarks, Settings Reference, Projects,
            Files), ProjectViews (project page, status), ViewerView (MTKView + input), InspectorView
            (View/Edit/Train/Export), NewProjectSheet, MaskEditor, AuxViews
```

**Data flow:** any source → `DatasetBuilder` (ingest → masks → poses → rig constraints) → `PreparedDataset` on disk (PROTOCOL §2) → `MetalTrainer` *or* `WorkerClient` *or* `ComputeScheduler.trainBlockParallel` → `GaussianCloud` → `SplatScene` (GPU) → `SplatRenderer`.

Everything that touches parameters uses one layout (`GaussianCloud`: positions, log-scales, wxyz quaternions, opacity logits, 48 SH floats coefficient-major). This lets trainer buffers become a render scene without a CPU copy (`MetalTrainer.makeScene()`, used for the live preview). Unified memory means densification and editing work directly on GPU buffers from the CPU.

### Dataset format

A prepared dataset is a directory holding:
- `dataset.json`: OpenCV camera convention, row-major `cameraToWorld`, intrinsics, and optional per-camera `depth`, `mask`, `holdout`, `exposure`, `time` and `group`;
- the `images/` folder, plus optional `depth/`, `masks/` and `points3d.ply`.

The layout is identical on the Mac, in uploads and on the worker. `PreparedDataset.normalizeOrientation()` levels a built dataset to −Y up and centres it, using `gravityUp` when the dataset has one (ARKit gravity, or the +Z of a GPS-georeferenced splatprep solve) and `Orientation.estimateUp` otherwise.

A project folder holds:
- `project.json` (`ProjectRecord`): the source, the `PrepOptions`, and the state of processing and of the latest training run (settings, devices, step, checkpoint, block count);
- `dataset/`: the prepared dataset, plus `frames.json` (an intermediate FrameSet), `.remote-store.json` (see [Remote execution](#remote-execution)) and `.prep/` (stage checkpoints, below);
- `outputs/*.ply` results; `<output>.remote-job.json` for a worker job still in flight; `.resume/` (this Mac's training checkpoints) and `.blocks/` (finished blocks of an unfinished block run);
- `prep.log`.

### Projects, checkpoints and resume

`ProjectLibrary` keeps the list of projects and loose splat files (`~/Library/Application Support/SplatWork/library.json`); everything else about a project lives in its own folder, so a project can be moved or opened on another machine.

- **Processing** (`PrepCheckpoints`): after each stage `DatasetBuilder` saves its output in `dataset/.prep/` (the FrameSet after frames and after masks; a copy of `dataset.json` and its point cloud after camera solving; a marker once levelled). Each checkpoint carries a SHA-256 fingerprint of the source files (path, size, modification time) and of the settings of that stage and every earlier one. A run restores the latest stage whose fingerprint matches and continues from there, so a changed setting redoes exactly that stage onward. *Redo from Here* deletes the checkpoints from a stage on, together with the matching worker-job ledger entries.
- **Training on this Mac**: every 10 minutes and on Stop, `MetalTrainer.checkpoint()` is written to `outputs/.resume/`; `MetalTrainer.resume(from:step:)` loads it after `prepare` and continues the step-driven schedules (learning rates, SH degree, densification) from that step with fresh optimizer moments.
- **Block runs**: each finished block is written to `outputs/.blocks/<settings>/block-<i>-<hash>.ply`, keyed by the block's cameras, region and config; a resumed run uses the recorded block count (the partition is deterministic) and skips finished blocks. A failed block waits on a `BlockRetryGate` for Retry (on any device; `DeviceLocks` keep one block per device) or Leave Out.
- **Pause** (`PauseController`) holds every Metal trainer of the run and pauses every worker job (`POST /v1/jobs/{id}/pause|resume`); jobs that start while paused start paused.

### Pipeline stages (`Pipeline/DatasetBuilder.swift`)

1. **Ingest** produces a `FrameSet`. Photos get blur and duplicate rejection. Video gets sharpest-in-window frame selection plus scene-cut and static skipping. 360° frames are cut into virtual pinhole rigs (`Equirect360`). Stereo covers SBS/TB/dual/MV-HEVC with the baseline and optional depth. Capture bundles bring ARKit poses, LiDAR depth and points. Importers handle COLMAP and Nerfstudio.
   - With `PrepOptions.videoOnWorker`, the raw video is streamed to the worker (`POST /v1/datasets/file`) and split there by splatprep `task: frames`. On any failure it falls back to local splitting.
2. **Masks** (`FrameMasker`): Apple Vision locally, or splatprep models (`dynamic` = OWLv2 + SAM 2 tracking, `sky`, `person`, `sam2_*`) locally or on the worker.
3. **Poses** (`solvePoses`):
   - `automatic` resolves to capture poses when the source has them. Multi-view rigs (360°) get splatprep SuperPoint + LightGlue, because Object Capture placed only 1,041 of 9,824 views. Everything else gets Object Capture, or splatprep where Object Capture is unsupported.
   - Capture poses are refined by splatprep `priorMode: refine` (`PrepOptions.refineCapturePoses`). Gravity, exposure, time and group are copied from the capture onto the refined dataset.
   - Rig constraints and the stereo metric scale are applied after solving.
   - With 10 or more frames, the build fails with advice if fewer than 30 % of them get a pose.
4. **Dataset**: `PoseAssembly.makeDataset` (holdout every Nth), scene bounds, then save.

### Renderer

1. `project_splats` runs per gaussian: EWA projection, SH colour, opacity-aware extent, eigen-axes and the sort key.
2. `GPURadixSort` is a stable LSD radix sort with 8-bit digits and a SIMD ballot multisplit.
3. Splats are drawn as instanced quads with an indirect draw. The projection kernel accumulates the instance count.
4. Blending is premultiplied "over" in on-chip tile memory.

Sort keys are 16-bit log-depth by default (2 passes) or 32-bit float bits. When the camera, settings and edits are unchanged, the previous projection and sort are reused.

**Display modes:** Colour, Depth, Centres, Ellipses, Opacity, and Coverage. **Coverage** (`Render/Coverage.swift`, kernel `coverage_compute`) loops over every splat × dataset camera: in frame, and not behind the LiDAR depth surface. It records the view count and the best pixels per scene unit, then scores them relative to the capture. It takes 0.2–0.3 s for 1 M splats × 649 cameras.

### Metal trainer (`Train/MetalTrainer.swift`, `Shaders/TrainShaders.swift`)

Per step:
1. `train_preprocess`: projection, conic, anti-aliasing compensation, SH colour, **tight per-axis ellipse tile bounds**, depth key.
2. Depth radix sort, then gather tile counts in depth order, then exclusive scan. The total is read back (one CPU sync).
3. `duplicate_keys` emits (tile, gaussian) pairs in depth order. A **stable** 16-bit tile sort then yields per-tile depth-ordered lists without 64-bit keys.
4. `rasterize_forward` (16×16 tiles, 256 threads, early exit) produces RGB and expected depth.
5. **Appearance correction** (`appearance_apply` / `appearance_backward`): a per-photo 3×3 matrix + offset applied to the render, used only in the loss. It is seeded from camera `exposure`, has its own Adam state and an L2 pull to the seed. It is disabled automatically when every camera has the same exposure (`exposureLocked`). Exports bake in the mean correction.
6. Fused loss: SSIM via separable 11-tap Gaussian blurs of 15 moment planes, an analytic SSIM gradient (3 derivative maps blurred again), L1, masked pixels, and edge-aware L1 depth on valid LiDAR/stereo pixels.
7. Optional robust transient masking (`transientMasking`, off by default): a per-pixel error histogram gives a median-relative threshold. Pixels become outliers only where the view is well covered, the region is solid and the patch is not mostly good; their gradient weight is 0. It is skipped when the median nearest-view gap exceeds `transientMaxViewGap`.
8. `rasterize_backward`: back-to-front replay, SIMD-group reductions and float atomics (12 accumulators per gaussian).
9. `preprocess_backward`: analytic chain rule to position, log-scale, quaternion, opacity logit and SH (including the anti-alias compensation term). It adds the regulariser gradients:
   - MCMC L1 opacity and scale;
   - the **needle penalty** (longest ÷ middle axis above `anisotropyMaxRatio`, which shortens only the long axis, so flat discs are untouched);
   - the **big-splat penalty** (screen-space angular radius above about 33°, which shrinks the longest axis first; `bigSplatReg`, default 10).
10. Adam (per-group learning rates, position LR with exponential decay × scene extent) and 3DGS-MCMC SGLD noise (`mcmc_noise`).
11. Every `densifyInterval` steps, CPU densification runs on the shared buffers. It is either MCMC relocation/growth or ADC clone/split/prune with opacity reset, plus block-region pruning. With **long-axis splitting** (`longAxisSplit`, default on), MCMC clone groups are spread along the parent's longest axis and only that axis is shortened (`spreadAlongLongAxis`). Stacked clones had made large scenes collapse. High stops densifying at 14k of 30k steps and Maximum at 19k of 40k; 14k beat 25k on the bedroom test (23.14 vs 22.82 dB held-out).

**Peak and census passes** (used by Auto clean and `--eval`):
- `rasterize_peak` records each splat's largest α·T in any pixel of a view.
- `peak_accumulate` folds that into an all-photos maximum and a count of photos in which the splat clearly contributes (`contributionCensus`).
- `AutoClean.plan` then selects splats that are invisible (peak below 2 %) or seen in fewer than ¼ of a typical splat's photo count.

It takes about 9 s for 727 k splats × 649 photos.

`SPLAT_PROFILE=1` prints per-stage GPU timings. Measured on the M4 Pro (960×720 synthetic scene, 123 k splats, M = 1.3 M tile instances), per step: forward ≈ 3.4 ms, SSIM ≈ 3.0 ms, backward ≈ 12.7 ms, rest ≈ 2.5 ms.

**Initialisation:** sparse point clouds (Apple Object Capture yields about 1–5 k points) are topped up to max(10 % of the budget, ≤100 k). The extra points are jittered copies at local point spacing plus a light background fill. MCMC growth is max(5 % per pass, the pace needed to reach the budget before densification ends).

**Orientation:**
- `Orientation.estimateUp` uses capture gravity first. Failing that, it uses consistent camera up axes. Failing that, it uses the direction most perpendicular to all viewing directions, with a portrait-aware sign vote.
- PLYs carry `comment splatstudio_up x y z`, and the worker writes this too. Project gravity overrides the file's note.
- `GaussianCloud.rotate` rotates SH exactly (per-band least-squares Wigner matrices).

**Verification:** self-tests check:
- sort stability;
- finite-difference gradients for every parameter group (median relative error 1–5 %, float32);
- determinism;
- that training recovers a known scene.

The CLI's `--eval` reports held-out PSNR, per view if asked; the measured results in SETTINGS.md come from it.

### Remote execution

**`WorkerClient`** speaks the worker HTTP API (PROTOCOL §4, §6): `/v1/info`, `/v1/datasets` (zip), `/v1/datasets/file` (raw stream), `/v1/datasets/compose`, `/v1/jobs`, `/v1/process`, previews, results, `/v1/jobs/{id}/cameras`, `/v1/runtime`. `WorkerBrowser` finds workers via Bonjour `_splatworker._tcp` or manual entry, and keeps tokens in the Keychain.

**Chaining** (worker ≥ 1.3.8, capability `chain`) means video frames cross the network only once:
- A completed processing job keeps its output on the worker as a dataset (`JobStatus.outputDatasetId`, hard links).
- The Mac downloads the output without `images/` (`output.zip?skip=images`). It then writes `.remote-store.json` (`worker`, `datasetId`, `dirs`, `local`).
- Later steps send only what changed (masks, priors, `dataset.json`) and stack them over the stored images with `POST /v1/datasets/compose`. Masks run on the worker copy with no upload.
- `RemoteImageFetch` downloads the images in the background (`archive.zip?only=images`). Mac-side steps that read frames call `ensure` first. The download resumes when the project is reopened.

**Surviving disconnects:**
- `waitForJob` and result downloads retry network errors with backoff to 30 s, for up to 12 h. Only a failed or missing job ends the wait.
- Once accepted, a job is written to `<output>.remote-job.json`. `TrainingSession.resumePendingRemote` re-attaches when the project opens (progress, live preview), or downloads the finished result.
- The CLI equivalent is `--fetch-job`.

**`RemoteJobWatcher`** streams any worker job into the viewer: it polls the status, pulls `preview.splat`, loads the job's cameras and backfills charts from the job log.

### Both mode: ComputeScheduler and block training

`ComputeScheduler` holds `ComputeDevice`s, each with capabilities and a throughput figure where 1.0 ≈ 20 benchmark steps/s ≈ an M4 Pro.
- **Throughput source:** `BenchmarkStore` (`~/Library/Application Support/SplatStudio/benchmarks.json`, written by Compute → Benchmark or `--benchmark --save`). For unmeasured devices it falls back to a GPU-name table (CUDA) or the Metal working-set size (Mac).
- **Stage placement** (`place`): frame extraction stays local; Object Capture and Apple Vision only run on the Mac; everything else goes to the best score with a queue penalty and a small local bonus.
- **Scenes under 400 images** in Both mode train whole on one device. When every device is benchmarked, that is the device with the lowest `TrainingTimeEstimate` (steps × seconds per step scaled by splats and megapixels, plus upload time). Otherwise it is the highest throughput.
- **Larger scenes:** `BlockPartitioner` makes K = 2 × devices spatial blocks. `distribute` assigns them proportionally to throughput (greedy LPT). Each block is a job with `TrainingConfig.region` (expanded box) and `cameraIds`. `BlockTraining` merges them, keeping the gaussians inside each un-expanded box.

`SystemBenchmark` builds a synthetic scene (48 views, 1600×1200, 1 M starting splats) rendered by the app's own renderer. It trains it for 2,000 steps, timing steps 1,000–2,000, locally or as a worker job. For workers it also times the upload and download.

## Linux worker (`linux/`)

- **`server.py`**: the FastAPI app.
  - **`JobManager`** keeps a FIFO queue and runs one job per configured GPU. Jobs persist in `jobs/<id>/job.json`; after a worker restart, queued jobs are re-queued and running ones are marked failed.
  - Datasets live in `datasets/<id>`. Uploads go through `safezip` (path, size, ratio and symlink checks).
  - Processing outputs are adopted as datasets (`_adopt_output_as_dataset`), and `_compose_dataset` hard-links layers together.
- **`backends.py`**: `SubprocessBackend` spawns a fresh Python process per job (`multiprocessing` spawn), pinned to one GPU. CUDA memory is returned after each job, and a trainer crash cannot take the API down. It reports the capabilities `train`, `process` and `frames`.
- **`trainer.py`** builds on gsplat (`MCMCStrategy` or `DefaultStrategy`, `rasterization` in antialiased or classic mode). It adds:
  - `spread_clone_groups` (long-axis splitting on top of gsplat MCMC, default on);
  - needle and big-splat penalties;
  - `TransientMasker`;
  - `AppearanceCorrection` (per-photo 3×3 + b, auto-off when exposure is locked);
  - **`BilateralGrid`**: per photo, a 16×16×8 grid of 3×4 colour transforms, TV-smoothed, with the mean over photos pinned to identity every 50 steps. Held-out photos are scored without correction. It is opt-in (`bilateralGrid`) and replaces appearance correction;
  - experimental `PoseRefiner`;
  - region and `cameraIds` handling.
  - Outputs are `preview.splat` (≤ 400 k most visible splats), `result.ply` with `comment splatstudio_up`, and train and held-out PSNR reported separately.
- **`processing.py`**: runs splatprep requests in the subprocess backend and zips outputs (built on request, with `skip` / `only`).
- **`runtime_install.py`**: remote install of the processing models into the GPU runtime (`/v1/runtime`), keeping torch pinned.
- **Studio** (`splatworker/studio/`, GTK-free and unit tested; pages in `splatworker/gui/gtk/studio/`): the Linux SplatWork app (OpenGL viewer, New Project, editing, export, iPhone Inbox, compute devices). Its GPU work runs as jobs of the same server. See [dev/LINUX_STUDIO.md](dev/LINUX_STUDIO.md).
- **GUI** (`splatworker/gui/`, GTK 4 + libadwaita):
  - worker pages (Queue & Worker: status, jobs, log, connect) and Settings;
  - the setup assistant: `runtime_plan.py` picks a GPU profile from the driver (cu128 → PyTorch 2.7.1 + gsplat 1.6 dev build; cu124/121/118 → PyTorch 2.4.1 + gsplat 1.5.3), and `installer.py` installs a pinned standalone CPython 3.10, torch and gsplat. The runtime is built in `venv.new` and swapped in only after verification. Profiles whose gsplat wheel isn't published are skipped, and local wheels in `wheels/` are preferred;
  - `server_process.py` runs `python -m splatworker serve` as a child process;
  - `updates.py` checks the published Flatpak repository and shows an update banner.
- **Flatpak** (`linux/flatpak/`): manifest on `org.gnome.Platform//51`, generated `python3-deps.json`, and `build-gsplat-wheel.sh`. That script builds the pinned gsplat 1.6 wheel in a podman container, for this GPU or `--all`. It patches the upstream `cudaEventCreate` overload, and uses `nvcc --threads` with a memory-aware job count.

## splatprep (`tools/splatprep/`)

One entry point, `python -m splatprep run --request request.json --progress-jsonl -`. It is used by the Mac (`SplatPrepRunner`) and by the worker. The tasks are `sfm`, `masks`, `frames` and `benchmark`.

- **Backends** (`backends/`): `colmap_sift`, `opencv_sift`, LightGlue family (`sift`/`superpoint`/`disk`/`aliked`), `loftr`, `mast3r`, `vggt`. Model availability is reported from what actually imports.
- **Pairs** (`pairs.py`): sequential, exhaustive, retrieval, or auto. Prior-pose proximity and GPS spatial matching (drone sets) add pairs.
- **Mapper** (`mapper.py`): GLOMAP, COLMAP or pycolmap. `auto` runs incremental and global up to 1,000 images and keeps the better result (registration, then track length). EXIF/ARKit focal lengths are held fixed while mapping; the final focal refinement is kept only if focal moves less than 8 %.
- **GPS** (`gps.py`): solved centres are Sim3-fitted to EXIF GPS (DJI relative altitude, S/W hemispheres). If at most 2 % of cameras disagree, the scene is put in ENU metres with +Z up and the disagreeing cameras are dropped. Otherwise a warning is raised. The scene extent is computed robustly.
- **Prior refinement** (`prior_refine.py`, `priorMode: refine`): with poses and intrinsics on at least 90 % of images, it builds look-at pairs, triangulates with the priors and runs pose-only BA ×2. Frames with too few points keep the prior pose, and implausible corrections fall back to the priors.
- **Frames** (`frames.py`): a port of the Mac's sharpest-per-window selection and equirect → pinhole views. It decodes with PyAV and resamples with `grid_sample` on the GPU (OpenCV `remap` otherwise), producing the same `images/`, `masks/` and `frames.json` layout as the Mac.
- **Masks** (`masks/`): `detect` (OWLv2), `dynamic` (detection + SAM 2 video tracking), `sam2`, `track`, `sky`, `person`, and `server.py` (the click-to-mask server behind the Mac's mask editor).
- **Benchmark and synth** (`benchmark.py`, `synth.py`, `verification.py`): the model benchmark shown in Models & Benchmarks, and synthetic test scenes.

## iPhone app (`ios/SplatCapture`)

- **Capture/**:
  - `CaptureEngine` runs the ARKit session with locked exposure and a capped shutter.
  - `KeyframeSelector` picks keyframes with an escalating blur gate, so it never leaves coverage gaps.
  - `FrameProcessing` and `CaptureRecorder` write images, depth and confidence.
  - `PointFusion` builds the fused LiDAR cloud, and `Manifest` writes `manifest.json`.
- **Library/** stores captures on disk.
- **Transfer/** has `MacBrowser` (Bonjour `_splatcapture._tcp`), `CaptureUploader` (`POST /v1/captures` with the pairing code) and a zip64 `ZipWriter`.
- **UI/** has the AR view, coverage overlay, mini-map, library, send-to-Mac and settings screens.

On the Mac, `CaptureReceiver` accepts uploads into the iPhone Inbox, and `CaptureImporter` converts them (ARKit → OpenCV poses, depth, points, exposure, gravity).

## CI (`.github/workflows/splat-worker-flatpak.yml`)

On pushes that touch `linux/` or `tools/`:
1. **gsplat-wheel job:**
   - It reads `linux/flatpak/gsplat-pin.env`.
   - If the wheel is cached, it uses the cache. If not, it downloads a wheel attached to the GitHub Release `gsplat-<version>-<cuda tag>` (built on a PC with `build-gsplat-wheel.sh --all`).
   - Only if neither exists does it install the CUDA toolkit and compile the pinned commit, with the same `cudaEventCreate` patch.
2. **Flatpak job:** it builds the app into an OSTree repository (optionally GPG-signed). It writes `.flatpakrepo`, `.flatpakref`, `version.json` and a pip `--find-links` wheel index, then publishes everything to GitHub Pages. Installed workers update from there.

## Known performance headroom

- Backward raster is 60 % of a step. Options:
  - per-splat backward with checkpointed transmittance ("Taming 3DGS");
  - half-precision gradient accumulation;
  - fewer SIMD reductions by packing;
  - skipping SSIM blurs for fully masked tiles.
- There are two command buffers per step (one CPU sync for the tile count). This can be removed by allocating tile buffers with a conservative bound.
- Densification on the CPU is O(N) every 100 steps, which is fine up to about 3 M splats.
- The SH-direction gradient with respect to position is not propagated. This is a common simplification with negligible effect on quality.

## Extension points for planned features

| Planned | Hook already in place |
|---|---|
| **Temporal / 4D splatting** | `DatasetCamera.time` (seconds) in dataset.json. Parameter buffers are plain per-gaussian arrays (`MetalTrainer.widths`), so temporal parameters (e.g. velocity, temporal opacity, deformation code) can be added as new groups with their own Adam state. `CameraUniforms.misc.w` is free for the frame time. |
| **Multi-source datasets** (drone + phone, 360 + headset…) | `DatasetCamera.group`. `DatasetBuilder` produces one dataset per source today; merging is an extra step that concatenates cameras and aligns sources (`RigConstraints`, Umeyama Sim3 in splatprep, GPS georeferencing). The worker's `/v1/datasets/compose` already unions datasets. Per-group appearance slots into the existing per-photo correction stage. |
| **Low-quality footage (e.g. VHS)** | Built (2026-11-13) as its own source, `SourceDescription.homeVideo`: splatprep's `vhs.py` analyses the tape, splits it into clips and writes restored frames (`task: frames` + `vhs`, also on a Linux worker); frames carry `clip` + `cameraGroup` and the solve gives each group its own camera (`cameraGroups`). Training uses SH 1 and transient masking. Learned upscaling is not used (see `docs/dev/VHS.md`). |
| **Steam Frame capture app** | Capture bundle v2 (PROTOCOL §8) already describes stereo rigs, IR sensors, `rigToWorld` poses and `depthSource: "ir-stereo"`. The Mac importer reads v2 bundles and stereo rigs today. An experimental prototype lives in `steamframe/`. |
