# Architecture

## Mac app (`mac/Sources/SplatStudio`)

```
App/        SwiftUI app, AppModel (document, editing, camera), TrainingSession / PrepSession, CLI
Core/       GaussianCloud, PreparedDataset (dataset.json), TrainingConfig (+presets), SettingsCatalog (all docs)
Formats/    PLY (any PLY, 3DGS PLY r/w), .splat, SPZ (gzip via Compression framework)
Shaders/    Metal sources as Swift strings, compiled at launch (no offline metal compiler needed)
Render/     MetalContext, GPUScan, GPURadixSort, SplatScene, SplatRenderer, ViewerCamera, media export
Train/      MetalTrainer (optimisation loop + densification), TrainingData (image loading, init, KNN)
Pipeline/   Sources (photos, video, 360, stereo), Poses (Object Capture, splatprep), Importers, Masking, DatasetBuilder
Remote/     WorkerClient/Browser (Bonjour), CaptureReceiver (iPhone → Mac HTTP), ComputeScheduler, BlockTraining
UI/         ContentView, ViewerView (MTKView + input), Inspector panels, NewProjectSheet, Inbox/Workers/Models/Reference
```

**Data flow:** any source → `DatasetBuilder` (ingest → masks → poses → rig constraints) → `PreparedDataset` on disk (PROTOCOL §2) → `MetalTrainer` *or* `WorkerClient` *or* `ComputeScheduler.trainBlockParallel` → `GaussianCloud` → `SplatScene` (GPU) → `SplatRenderer`.

Everything that touches parameters uses one layout (`GaussianCloud`: positions, log-scales, wxyz quaternions, opacity logits, 48 SH floats coefficient-major), so trainer buffers can be turned into a render scene without a CPU copy (`MetalTrainer.makeScene()` → live preview). Unified memory means densification and editing operate directly on GPU buffers from the CPU.

### Renderer
`project_splats` (per gaussian: EWA projection, SH colour, opacity-aware extent, eigen-axes, sort key) → `GPURadixSort` (stable LSD radix sort, 8-bit digits, SIMD ballot multisplit) → instanced quads drawn with an indirect draw whose instance count the projection kernel accumulates → premultiplied "over" blending in on-chip tile memory. Keys are 16-bit log-depth by default (2 passes) or 32-bit float bits. When the camera, settings and edits are unchanged the previous projection/sort is reused.

### Trainer (per step)
1. `train_preprocess` — projection, conic, anti-aliasing compensation, SH colour, **tight per-axis ellipse tile bounds**, depth key.
2. Depth radix sort → gather tile counts in depth order → exclusive scan → total read back (one CPU sync).
3. `duplicate_keys` emits (tile, gaussian) pairs in depth order; a **stable** 16-bit tile sort then yields per-tile depth-ordered lists without 64-bit keys.
4. `rasterize_forward` (16×16 tiles, 256 threads, early exit) → RGB + expected depth.
5. Fused loss: SSIM via separable 11-tap Gaussian blurs of 15 moment planes, analytic SSIM gradient (3 derivative maps blurred again), L1, masked pixels, L1 depth on valid LiDAR/stereo pixels.
6. `rasterize_backward` — back-to-front replay, SIMD-group reductions + float atomics (12 accumulators per gaussian).
7. `preprocess_backward` — analytic chain rule to position, log-scale, quaternion, opacity logit and SH (incl. antialias compensation term).
8. Adam (per-group LRs, position LR exponential decay × scene extent) and 3DGS-MCMC SGLD noise.
9. Robust transient masking (after 30 % of training): per-pixel error histogram → median-relative threshold (EMA) → outliers only where well covered (T < 0.05), solid (≥ 7/9 neighbours) and not in a mostly-good 16×16 patch → gradient weight 0. Validated on synthetic scenes with injected movers: +0.3 dB (random movers), +0.9 dB (persistent movers), no harm on clean scenes; looser rules discarded texture and lost up to 7 dB.
10. Every `densifyInterval` steps, CPU densification on the shared buffers (MCMC relocation/growth or ADC clone/split/prune + opacity reset), plus block-region pruning.

`SPLAT_PROFILE=1` prints per-stage GPU timings. Measured on the M4 Pro (960×720 synthetic scene, 123 k splats, M = 1.3 M tile instances): forward ≈ 3.4 ms, SSIM ≈ 3.0 ms, backward ≈ 12.7 ms, rest ≈ 2.5 ms per step.

**Initialisation:** sparse point clouds (Apple Object Capture yields ~1–5 k points) are topped up to max(10 % of the budget, ≤100 k) with jittered copies at local point spacing plus a light background fill, and MCMC growth is max(5 %/pass, the pace needed to reach the budget before densification ends). On the synthetic walk set this took Object Capture datasets from 22.0 to 27.8 dB.

**Orientation:** `Orientation.estimateUp` (gravity from capture bundles first; otherwise consistent camera up axes; otherwise the direction most perpendicular to all viewing directions with a portrait-aware sign vote); `PreparedDataset.normalizeOrientation()` levels every built dataset to −Y up and centres it; PLYs carry `comment splatstudio_up x y z`; `GaussianCloud.rotate` rotates SH exactly (per-band least-squares Wigner matrices).

**Verification:** `scratchpad`-style self-tests check sort stability, finite-difference gradients for every parameter group (median relative error 1–5 %, float32), determinism, and that training recovers a known scene (≈ 29.6 dB holdout PSNR after 7 k steps on the splatprep synthetic orbit set).

### Known performance headroom
- Backward raster is 60 % of a step; options: per-splat backward with checkpointed transmittance ("Taming 3DGS"), half-precision gradient accumulation, fewer SIMD reductions by packing, skipping SSIM blurs for fully masked tiles.
- Two command buffers per step (one CPU sync for the tile count) — can be removed by allocating tile buffers with a conservative bound.
- Densification on CPU is O(N) every 100 steps; fine to ~3 M splats.
- SH-direction gradient w.r.t. position is not propagated (common simplification; negligible quality effect).

## Extension points reserved for planned features

| Planned | Hook already in place |
|---|---|
| **Temporal / 4D splatting** | `DatasetCamera.time` (seconds) in dataset.json; parameter buffers are plain per-gaussian arrays (`MetalTrainer.widths`) so temporal parameters (e.g. velocity, temporal opacity, deformation code) can be added as new groups with their own Adam state; `CameraUniforms.misc.w` is free for the frame time. |
| **Multi-source datasets** (drone + phone, 360 + headset…) | `DatasetCamera.group`; `DatasetBuilder` produces one dataset per source today — merging is an extra step that concatenates cameras and aligns sources (`RigConstraints`, Umeyama sim3 in splatprep). Per-group exposure / appearance embeddings slot into the loss stage (RGB affine per group before SSIM). |
| **Low-quality footage (e.g. VHS)** | `DatasetBuilder` is staged (ingest → masks → poses); a restoration stage (deinterlace, denoise, deblock, colour/gate-weave stabilise, upscale) goes between ingest and masks and only rewrites `FrameSet` images. Per-image appearance compensation and robust losses (above) handle tape colour drift. |
| **Steam Frame capture app** | Capture bundle v2 (§8) already describes stereo rigs, IR sensors, `rigToWorld` poses, `depthSource: "ir-stereo"`; the Mac importer reads v2 bundles and stereo rigs today. |
