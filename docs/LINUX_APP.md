# SplatWork Node (Linux) — parity plan

The Linux desktop app (`linux/`, Flatpak `com.example.SplatWorker`) grows from a GPU **worker** with a status
window into a full **SplatWork** studio, while the worker server keeps running for the Mac and the iPhone.
This file is the checklist: every user-facing feature of the Mac app (`mac/`), how it maps onto Linux, and its
status.

Status: ✅ done · 🟡 partial (what is missing is said) · ⏳ not yet · ➖ not applicable on Linux

## Architecture

```
┌──────────────── Flatpak sandbox (one app, one window) ─────────────────────────────────────────────┐
│  GUI process — GNOME runtime Python 3.14, GTK 4 + libadwaita, numpy, Pillow (no torch)              │
│    splatworker/gui/gtk/…          window, sidebar, setup assistant, preferences (existing)          │
│    splatworker/gui/gtk/studio/    Viewer (GtkGLArea), New Project, Projects, Training, Reference    │
│    splatworker/studio/            GTK-free core: formats, camera, sorting, editing, Auto clean,     │
│                                   presets, projects, local pipeline  ← unit tested anywhere         │
│        │ HTTP on 127.0.0.1 (same API and token as the Mac, PROTOCOL.md §4/§6)                         │
│        ▼                                                                                            │
│  Worker server — GPU runtime Python 3.10 (torch, gsplat, splatprep), child process                  │
│    JobManager: ONE queue per GPU for everybody (local Studio jobs + Mac jobs)                        │
│    jobs: process (splatprep frames / masks / sfm), train (gsplat trainer), census (Auto clean)      │
└────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

* **Local work is just jobs of the local server.** New Project → frames → masks → sky → camera solve → train
  runs as the same `/v1/process` and `/v1/jobs` requests the Mac sends to a remote worker, so the GUI never
  imports torch, CUDA crashes can't take the window down, and GPU sharing is the server's queue.
  No upload: the Studio registers its project folder with `POST /v1/datasets/local`; the dataset points
  at the folder instead of copying it. Those endpoints (and priority, queue reordering, census by path) are for
  this app only: it passes a per-launch secret to the server it starts and sends it as `X-SplatWork-Local`;
  a loopback connection without it (SSH tunnel, `tailscale serve`, proxy) counts as remote. Remote clients
  can't see or use the app's folder datasets, and `export-local` only writes into registered folders.
* **GPU sharing.** One job per GPU at a time, as before. Queued jobs are ordered by priority, then age: local
  Studio jobs are submitted with `priority: 1` (the person at the PC goes first) unless *Local jobs go first* is
  switched off (New Project dialog, stored in `studio.json`); nothing running is ever pre-empted. Every queued job
  can be moved to the front (`POST /v1/jobs/{id}/prioritize`) or cancelled on the Queue & Worker page, which shows each job's
  origin (this PC / the address of the Mac). The viewer renders with OpenGL and does not use CUDA, so it
  works while either side trains.
* **Projects** live in `~/Documents/SplatWork/<name>/` (Flatpak permission `xdg-documents/SplatWork:create`),
  the same layout as the Mac: `project.json`, `dataset/` (`dataset.json`, `images/`, `masks/`,
  `splatprep-sky/masks/`), `outputs/<name>-<time>.ply` + `.sky.png/.sky.json`, `prep.log`. A project made on
  either machine opens on the other.

### Renderer choice

**OpenGL 4.3 core in a `GtkGLArea`**, drawing instanced, alpha-blended quads like the Mac's Metal viewer:

1. a compute shader projects every splat (3D covariance → 2D ellipse, opacity-aware extent, SH colour up to
   degree 3, display modes, crop box, selection highlight) into a vertex record;
2. splats are depth-sorted back-to-front on a background thread (16-bit log-depth keys, numpy's radix sort —
   about 40 ms for 6 M splats on the PC's CPU) and the order is uploaded as an SSBO; while the camera moves,
   frames are drawn with the latest finished order, exactly the trade-off web splat viewers make;
3. the vertex shader expands each record into a quad and the fragment shader evaluates the gaussian with
   premultiplied alpha; the learned skybox is a full-screen triangle drawn first (same math as `sky_fragment`
   in `RenderShaders.swift`).

Why not render with gsplat/CUDA into a texture: the GUI's Python can't import torch (it lives in the GPU runtime,
Python 3.10), so frames would have to cross a process boundary (a render server, 8 MB per 1080p frame, extra
latency), the viewer would not work before the GPU runtime is installed, and it would compete for CUDA memory
with training jobs. GL needs only the driver Flatpak already provides (`org.freedesktop.Platform.GL.nvidia-*`),
renders at the window's refresh rate, and runs on any GPU with GL 4.3 (NVIDIA, AMD, Intel; Mesa's llvmpipe in
CI). Function pointers are loaded with ctypes through `eglGetProcAddress` (GTK 4 uses EGL on Wayland and X11),
so no extra Python package is needed. Upgrade path: a GPU radix sort in a compute shader if CPU sorting becomes
the bottleneck beyond ~10 M splats.

## Checklist

Code locations: GTK-free core `linux/splatworker/studio/` (tested in `linux/tests/test_studio_*.py`), GTK pages
`linux/splatworker/gui/gtk/studio/`, server additions in `linux/splatworker/server.py` + `census.py`.

### Shell

| Mac feature | Linux | Status |
|---|---|---|
| Sidebar: Viewer, iPhone Inbox, Compute, Models & Benchmarks, Settings Reference, Recent | `Adw.NavigationSplitView`: Viewer, Projects, Queue & Worker (the old Status / Jobs / Log / Connect pages, with a running / waiting badge), iPhone Inbox, Compute Devices, Models & Benchmarks, Settings Reference, Recent (12) | ✅ |
| Home screen tiles (Photos, Video, 360°, iPhone LiDAR, Stereo, COLMAP/Nerfstudio, Open splat, Open dataset) | Viewer home page (no Stereo tile) | ✅ |
| Drag & drop anything onto the window | splats open; videos / folders / captures / transforms.json / COLMAP start New Project | ✅ |
| Toolbar: Home, New, Open, Reset view, Orientation menu, Navigate/Select, Inspector toggle | viewer header bar | ✅ |
| Easy / Advanced mode | everything is shown; advanced training values sit in an "All training settings" expander | 🟡 |
| Menus + shortcuts (⌘N/O/E, ⌘Z/⇧⌘Z, F, R, 1–6, Tab, WASDQE, ⌫, ?) | Ctrl instead of ⌘; shortcuts window on `?` / Ctrl+? | ✅ |
| Preferences | Preferences dialog: the server pages plus a **Studio** page for `studio.json` (projects folder, local jobs first, viewer defaults, navigation speeds, iPhone receiver on/off, pairing, port), saved through the Studio controller so open pages follow | ✅ |
| Training finished sound / notification | `Gio.Notification`, result opens in the viewer | ✅ (no sound) |
| Window title subtitle with counts, HUD (fps, visible / total, selected) | header subtitle + overlay | ✅ (no GPU ms) |

### Viewer (`viewer.*`, `nav.*`)

| Mac feature | Linux | Status |
|---|---|---|
| Open PLY / compressed PLY / .splat / .spz | `studio/formats.py`, vectorised numpy (also plain point-cloud PLYs) | ✅ |
| Learned skybox sidecar `<name>.sky.png/.sky.json` | loaded with every format, drawn in Colour mode, toggle in the inspector | ✅ |
| Orbit / Fly, double-click sets orbit centre, adaptive fly speed | `studio/camera.py` | ✅ (fly speed scales with scene size, not the surface in front) |
| Display modes Colour, Depth, Centres, Ellipses, Opacity | compute shader, as `RenderShaders.swift` | ✅ |
| Display mode Coverage | `studio/coverage.py` (port of the Mac's rules): per splat the dataset cameras that see it (+ LiDAR depth occlusion when the dataset has depth maps) and the best detail, numpy in 200 k-splat chunks on a background thread; Mac ramp, poor / unseen summary; offered only for a splat of a project with a dataset | ✅ (Linux-made capture projects have no depth maps yet, so walls don't occlude) |
| SH degree, splat scale, background, minimum opacity, FOV, render scale, anti-aliasing (auto/on/off) | inspector View tab | ✅ |
| Fast 16-bit sort / skip work when idle | always 16-bit log keys (CPU radix sort, background thread); renders only on change | ✅ |
| Scene up (6 axes), Flip, Level to view | orientation menu; trainer's `splatstudio_up` comment honoured | ✅ |
| Training camera frustums, look through a camera | GL lines; camera list (holdout / depth / mask tags) | ✅ |
| Views (Front/Back/Left/Right/Top), Frame, Reset | ✅ | ✅ |
| Rotation / pan / zoom / fly speed, invert vertical | inspector Navigation | ✅ |
| Smooth camera, two-finger rotate gesture | | ⏳ |
| Pinch / scroll zoom | `Gtk.GestureZoom`, scroll | ✅ |

### Editing (`edit.*`)

| Mac feature | Linux | Status |
|---|---|---|
| Undo / Redo (30), Restore all | `studio/edit.py` (packed deletion masks) | ✅ |
| Selection brush (radius, Alt deselects), Delete, Invert, Clear | screen-space brush on projected centres | ✅ (selects through all depths, like the Mac) |
| Crop box (sliders, Fit, Apply) | ✅ | ✅ |
| Delete below opacity / larger than / isolated splats | numpy; isolated = grid-hashed neighbour count | ✅ |
| Auto clean (invisible + seen by too few photos) | `census` job on the local worker (gsplat), plan = `studio/autoclean.py` (Mac rules) | ✅ code + tests with a fake backend; **gsplat census untested on a GPU** |
| Mask editor (SAM 2 clicks, paint, track) | | ⏳ |

### New Project (`source.*`, `sfm.*`, `mask.*`)

| Mac feature | Linux | Status |
|---|---|---|
| Photos folder (resize, blur rejection), video (fps / target frames / sharpest / HDR), 360° video (views, FOV, drop floor), holdout | `studio/importers.py`, splatprep `frames` with `equirect` + rig hint for solving | ✅ |
| 360° photos folder | | ⏳ |
| iPhone / headset `.splatcapture` | images + the manifest as `priors.json` (`priorMode` auto/solve = Refine capture poses) | 🟡 LiDAR depth supervision and points from depth ⏳ |
| Stereo footage | | ⏳ (needs the Mac's stereo ingest in splatprep) |
| COLMAP project, Nerfstudio `transforms.json`, prepared dataset | COLMAP via splatprep's reader / exporter (distorted lenses are re-solved), Nerfstudio converted here | ✅ |
| Camera solver: features × pair selection × mapper × lens model | splatprep `sfm` | ✅ (Apple Object Capture ➖) |
| Masks: people & pets, moving things (SAM 2), sky → learned skybox, invert | splatprep `person`, `sam2_auto`, `sky` | ✅ ("keep main subject", "isolate one object", editable class list ⏳) |
| Stacked masks + separate sky layer | every layer ANDed into `masks/` with 4 px growth; sky in `splatprep-sky/masks/` (photos with < 0.2 % sky dropped) | ✅ |
| Mask fixed parts of the frame (static regions) | | ⏳ |
| Presets + all `train.*` settings | `studio/presets.py`; "All training settings" expander (every worker TrainingConfig key) | ✅ (`train.bigSplatReg` is Mac-only: the gsplat trainer has no big-splat penalty) |
| Start training when ready, stages with durations, live log | ✅ | ✅ |
| Reuse earlier results (frames / masks / cameras) | "Prepare again" redoes everything; "Train again" reuses the dataset | 🟡 |

### Training

| Mac feature | Linux | Status |
|---|---|---|
| Local training | gsplat trainer as a job of the local server | ✅ |
| Live dashboard: step, steps/s, elapsed / ETA, splats, PSNR, holdout PSNR, GPU | Projects page | ✅ |
| PSNR / splat charts | `Gtk.DrawingArea` | ✅ |
| Live preview in the viewer | `preview.splat` fetched every new preview step (≥ 8 s apart) and shown, camera kept | ✅ |
| Pause / Resume, Save checkpoint | worker capability `pause`: pausing saves the full trainer state (`resume.pt`: params, optimizer, strategy, RNG, schedulers), frees the GPU and survives worker restarts; `POST /v1/jobs/{id}/checkpoint` → `checkpoint.ply`. Buttons on the Projects training dashboard (checkpoint → `outputs/…-stepN.ply`) and on Queue & Worker rows | ✅ (untested on CUDA) |
| Results in `outputs/<name>-<time>.ply` + skybox, opened when done | ✅ | ✅ |
| Shared GPU with the Mac's jobs | one queue per GPU, local priority, "Run next", origin shown | ✅ |
| Compute devices: other Linux workers / the Mac, Both mode | Compute Devices page (`studio/devices.py`): Bonjour `_splatworker._tcp` (Mac, iPhone, Linux) + manual address/token (`devices.json`), health via `/v1/info` with ethernet preference; New Project → *Train on*: This PC / a device / All devices. Preparation always runs on this PC. Photos stay on SplatWork Nodes between runs (`chain`); Mac / iPhone workers get them again each run | 🟡 untested on a real LAN; no re-attach after an app restart; no live preview or skybox from Mac / iPhone workers (their API lacks them) |
| Block-parallel training | single machine: the worker splits a too-big job (1.3.15); several machines: `studio/blocks.py` + `studio/compute.py` (the Mac's partition, distribution by speed and memory, owned-splat merge streamed from disk), each device gets only its blocks' photos, a failed remote block is retried here | ✅ single machine; 🟡 multi-machine untested on real devices |
| Benchmark | `studio/benchmark.py`: the Mac's standard scene (48 views, 1600×1200, 1 M points, 2 000 steps) as a job of the local worker; `benchmarks.json` in the Mac's `DeviceBenchmark` format; table on Models & Benchmarks; `splatworker benchmark [--url …]` also measures other workers incl. transfer speeds. Measured speeds replace the GPU-name table when placing work | ✅ (processing-model timings ➖ not measured) |

### Export (`export.*`)

| Mac feature | Linux | Status |
|---|---|---|
| PLY, compressed PLY, .splat, SPZ (v2) | `studio/formats.py` (+ CLI `splatworker convert`) | ✅ |
| Export SH degree, only kept splats, bake orientation, size estimate | ✅ | ✅ |
| Skybox sidecar with the export | written for every format | ✅ |
| Screenshot 4K PNG | offscreen framebuffer | ✅ |
| Turntable video | `studio/turntable.py`: orbit around the current centre, each frame sorted and rendered offscreen, streamed one frame at a time to GStreamer (H.264/MP4 via x264 / openh264 / VA / NVENC, else VP9/VP8 WebM), PNG sequence if no encoder; Export tab with resolution, length, fps, progress, Cancel | ✅ |

### Network, Inbox, models, reference

| Mac feature | Linux | Status |
|---|---|---|
| Worker server (Bonjour, token, GPUs, jobs, log, connect help) | unchanged, "Queue & Worker" page | ✅ |
| iPhone Inbox (`_splatcapture._tcp` on 7879, pairing code, receive, New splat…, trash) | `studio/inbox.py` (stdlib HTTP + zeroconf); off until turned on, constant-time code check, a sender is locked out for 10 min after 5 wrong codes and the code is replaced after 20 | ✅ |
| Models & Benchmarks | runtime status, installed models (from `/v1/info`), model cards, benchmark table + Run benchmark | ✅ |
| Install processing models | the existing Set Up GPU Runtime assistant | ✅ |
| Settings Reference (searchable, grouped, details, ⓘ on controls) | `studio/catalog.json` generated by `linux/scripts/gen_catalog.py` from `SettingsCatalog.swift` (a test keeps them in sync); every Studio control shows the summary and the full text as a tooltip | ✅ |
| CLI | `splatworker convert`, `splatworker splat-info`, `splatworker benchmark` | 🟡 (render / eval ⏳) |

## Notes and differences

* **Auto clean.** The Mac's census records each splat's *peak* contribution to any pixel. gsplat exposes
  gradients cheaply, so the worker measures each splat's *summed* blending weight per photo (∂(Σ pixels)/∂colour)
  and treats "≥ 0.02 in a photo" as "seen by that photo". A splat that adds 2 % to one pixel passes both tests;
  a splat spread thinly over many pixels can pass here and fail on the Mac. The plan rules (2 %, 25 % of the
  typical photo count, minimum typical count 12) are identical.
* The Flatpak's app id stays `com.example.SplatWorker` so existing installs update in place; the window and
  launcher are called **SplatWork Node** (the suite was renamed from "Splat Studio" to SplatWork).

## What needs testing on the PC (RTX 4080, CachyOS, fish)

Nothing below could run on the Mac (no NVIDIA GPU). The GL viewer was tested with Mesa llvmpipe in a Debian
container (render tests + the real GTK window driven by `linux/scripts/gui_smoke.py`); the pipeline was tested
end to end against a real HTTP server with a fake trainer and a stand-in splatprep.

Install the build from the branch (after the CI build or a local `linux/flatpak/build.sh`), then:

1. **Update prompt.** `flatpak update` should ask to allow the new `xdg-documents/SplatWork` folder. The app is
   now called *SplatWork*; the worker must still show up on the Mac and train a job sent from it.
2. **Viewer on NVIDIA.** Open a trained PLY with a skybox (one of the Mac's outputs or `…/jobs/<id>/result.ply`):
   the window must not show "The viewer can't start". Check the Renderer row (View tab → Performance) says NVIDIA
   and OpenGL 4.6. Orbit / pan / zoom / double-click, WASD fly, display modes 1–5, skybox on/off, a 4K screenshot.
   Note the fps for a 3–6 M splat scene (HUD); if the sort lags behind fast rotation, say so (GPU sort is the next step).
   Also try under X11 (`flatpak run --env=GDK_BACKEND=x11 com.example.SplatWorker`).
3. **Formats.** Open a `.compressed.ply`, `.splat` and `.spz` exported by the Mac; export each format here and open it
   on the Mac (and SuperSplat for compressed PLY).
4. **New Project → train.** Photos folder (Quick preview), then a phone video with *Remove people* + sky, then a 360°
   video. Watch stages, the log and the live dashboard; the live preview should appear in the viewer; the result
   must open when done and land in `~/Documents/SplatWork/<name>/outputs/` with `.sky.png/.sky.json`.
   `flatpak run --command=splatworker com.example.SplatWorker splat-info <result.ply>` shows the splat count.
5. **Queue sharing.** Start a long training from the Mac, then a project here: the local job waits (status "Waiting
   for the GPU"), then runs before any other job the Mac queued meanwhile; "Run next" (↑ on the Queue & Worker page)
   reorders queued jobs; nothing running gets interrupted.
6. **Auto clean.** Open a project's result, Edit → Auto clean: a `census` job runs (Queue & Worker shows it), then
   splats are selected; Delete, Undo. Compare the count with the Mac's Auto clean on the same file (the measure
   differs slightly, see Notes).
7. **iPhone Inbox.** Turn it on (Inbox page or Preferences → Studio — it's off by default), send a capture from
   SplatWork Capture (pairing code), then *New splat…* on it. Five wrong codes from the phone should give
   "try again in 10 minutes".
8. **Fish / paths.** Nothing in the app uses the shell; `flatpak run …` commands above work as is in fish.
9. **Local-only endpoints.** From the Mac, `curl -H "Authorization: Bearer <token>" http://<pc>:7878/v1/datasets`
   must not list the Studio's project folders, and `POST /v1/datasets/local` must answer 403. Same through an
   SSH tunnel (`ssh -L 7878:localhost:7878 pc`, then `curl localhost:7878/…`).
10. **Pause / checkpoint.** While a project trains: *Save checkpoint* (a `…-stepN.ply` appears under Results and
    opens), *Pause* ("Paused at step N", `nvidia-smi` shows the memory freed), queue another job (it runs),
    *Resume* (log: "resuming from the state saved at step N"; PSNR and splat count continue without a jump).
    Repeat with `optimizerPrecision` compact and fp32, with skybox + bilateral grid + pose refinement, MCMC and
    default strategies; pause, quit, restart, resume. Torch test: `tests/test_compact_adam.py` with the GPU
    runtime's python.
11. **Compute devices.** Turn on "Act as a worker" on the Mac: it appears under Compute Devices within ~15 s
    (this PC's own worker must not). Add another worker by address (+ token). New Project with *Train on* = the
    Mac; then *Train again* on another SplatWork Node (its log should say it still has the photos). A 400+ photo
    scene with *All devices*: per-device block lists and "merged N blocks" in `prep.log`, no seams at block
    borders; kill the Mac worker mid-run (its blocks are retried here); Cancel cancels the remote job.
12. **Benchmark.** Models & Benchmarks → Run benchmark (a job in Queue & Worker, then a steps/s row); compare with
    the Mac's Compute → Benchmark of this PC. `flatpak run --command=splatworker com.example.SplatWorker benchmark`
    with the app's server running.
13. **Coverage.** Open a project output and press 6: spinner, then colours and the summary; time it on a 3–6 M
    splat scene with ~300 cameras. A loose `.ply` leaves Coverage greyed out.
14. **Turntable.** Export → Render video, 1080p 8 s: the row names the encoder (openh264enc from the Flatpak's
    openh264 extension, or vp9enc); the file plays; Cancel works; memory stays flat at 4K.
15. **Preferences → Studio.** Receiver on/off and port follow on the Inbox page; a navigation speed follows in the
    viewer inspector.

Logs: Queue & Worker → Log, `~/.var/app/com.example.SplatWorker/data/splatworker/data/jobs/<id>/train.log`, and
each project's `prep.log`.
