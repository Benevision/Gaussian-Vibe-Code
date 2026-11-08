# Splat Studio for Linux — parity plan

The Linux desktop app (`linux/`, Flatpak `com.example.SplatWorker`) grows from a GPU **worker** with a status
window into a full **Splat Studio**, while the worker server keeps running for the Mac and the iPhone.
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
  No upload: the Studio registers its project folder with `POST /v1/datasets/local` (loopback clients only,
  and only when the server was started with `--allow-local-datasets`, which the GUI does); the dataset points
  at the folder instead of copying it.
* **GPU sharing.** One job per GPU at a time, as before. Queued jobs are ordered by priority, then age: local
  Studio jobs are submitted with `priority: 1` (the person at the PC goes first) unless *Settings → Studio →
  Local jobs wait their turn* is on; nothing running is ever pre-empted. Every queued job can be moved to the
  front (`POST /v1/jobs/{id}/prioritize`) or cancelled from the Studio's queue list, which shows each job's
  origin (this PC / the address of the Mac). The viewer renders with OpenGL and does not use CUDA, so it
  works while either side trains.
* **Projects** live in `~/Documents/SplatStudio/<name>/` (Flatpak permission `xdg-documents/SplatStudio:create`),
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

### Shell

| Mac feature | Linux | Status |
|---|---|---|
| Sidebar: Viewer, iPhone Inbox, Compute, Models & Benchmarks, Settings Reference, Recent | `Adw.NavigationSplitView` sidebar: Viewer, Projects, Queue & Worker (the old Status/Jobs/Log pages), iPhone Inbox, Models & Benchmarks, Settings Reference, Recent | see rows |
| Home screen tiles (Photos, Video, 360°, iPhone LiDAR, Stereo, COLMAP/Nerfstudio, Open splat, Open dataset) | Viewer placeholder page with the same tiles | ✅ |
| Drag & drop anything onto the window | `Gtk.DropTarget` on the window: splats open, folders / videos start New Project | ✅ |
| Recent files (12) | `~/.var/app/…/config/splatworker/recent.json` | ✅ |
| Toolbar: Home, New, Open, Reset view, Orientation menu, Navigate/Select, Inspector toggle | Viewer header bar | ✅ (Easy/Advanced picker ⏳: all options are shown) |
| Menus + shortcuts (⌘N/O/E, ⌘Z, F frame, 1–6 display modes, Tab orbit/fly, WASDQE, ⌫, ?) | Ctrl instead of ⌘; shortcuts window on `?` | ✅ |
| Preferences | existing Preferences dialog + Studio group (projects folder, local-job priority) | ✅ |
| Training finished sound / notification | `Gio.Notification` | ✅ |

### Viewer (`viewer.*`, `nav.*`)

| Mac feature | Linux | Status |
|---|---|---|
| Open PLY / compressed PLY / .splat / .spz | `studio/formats.py` (vectorised numpy) | ✅ |
| Learned skybox sidecar `<name>.sky.png/.sky.json` | loaded with every format, drawn in Colour mode | ✅ |
| Orbit / Fly navigation, double-click sets orbit centre, adaptive fly speed | `studio/camera.py` | ✅ (fly speed adapts to scene size, not to the surface in front) |
| Display modes Colour, Depth, Centres, Ellipses, Opacity | same shader modes | ✅ |
| Display mode Coverage | needs a per-splat coverage census | ⏳ |
| SH degree 0–3, splat scale, background colour, minimum opacity, FOV, render scale | inspector View group | ✅ |
| Anti-aliasing filter (Auto/On/Off) | shader (Mip-Splatting 2D filter compensation) | ✅ |
| Fast 16-bit sort | always 16-bit log keys on the CPU sorter | ✅ (no 32-bit option) |
| Skip work when idle | GtkGLArea redraws only on change | ✅ |
| Scene up direction (6 axes), Flip, Level to current view | `studio/camera.py` orientation | ✅ |
| Training camera frustums, look through a camera | drawn as GL lines; Cameras list in the inspector | ✅ |
| Views menu (Front/Back/Left/Right/Top), Frame all, Reset | ✅ | ✅ |
| Rotation/pan/zoom/fly speed, invert vertical, smoothing | Preferences → Navigation | 🟡 speeds + invert ✅, smoothing ⏳ |
| Two-finger trackpad gestures | `Gtk.GestureZoom` / scroll | 🟡 pinch zoom + scroll ✅, two-finger rotate ⏳ |
| HUD: fps, visible / total, selected | overlay label | ✅ |

### Editing (`edit.*`)

| Mac feature | Linux | Status |
|---|---|---|
| Undo / Redo (30 levels), Restore all | `studio/edit.py` (deletion masks, cheap) | ✅ |
| Selection brush (radius, ⌥ deselect), Delete selected, Invert, Clear | screen-space brush on the CPU (projected centres) | ✅ |
| Crop box (X/Y/Z sliders, Fit, Apply) | ✅ | ✅ |
| Delete below opacity / larger than / isolated splats (N neighbours) | numpy (+ grid hashing for the outlier test) | ✅ |
| Auto clean (invisible splats + seen by too few photos) | census job on the worker (gsplat renders every photo), plan in `studio/autoclean.py` (same rules as `AutoClean.swift`) | ✅ (contribution measured as per-photo weight sum ≥ 2 %, not per-pixel peak — see notes) |
| Mask editor (SAM 2 clicks, paint, track through frames) | | ⏳ (masks come from the automatic models; editing them needs SAM 2 point prompts over HTTP) |

### New Project (`source.*`, `sfm.*`, `mask.*`)

| Mac feature | Linux | Status |
|---|---|---|
| Sources: photos folder, video, COLMAP project, Nerfstudio `transforms.json`, prepared dataset | `studio/pipeline.py` | ✅ |
| 360° video / 360° photos (`source.360views`, `360fov`, `360dropNadir`) | splatprep `frames` with an equirect rig | 🟡 360° video ✅ via splatprep frames; 360° photo folders ⏳ |
| iPhone / headset capture `.splatcapture` (LiDAR depth, ARKit poses as priors) | converted to images + `priors.json` | 🟡 images + poses (priorMode auto) ✅; LiDAR depth supervision ⏳ |
| Stereo footage (SBS/TB/two files, MV-HEVC, baseline, stereo depth) | | ⏳ (needs the Mac's stereo pipeline ported to splatprep) |
| Max image size, video FPS / target frames / sharpest in window, blur rejection, holdout | request keys | ✅ (HDR tone mapping ⏳) |
| Camera solver: learned features × matching × mapper × lens model, "use poses from source" | splatprep `sfm` request | ✅ (Apple Object Capture ➖) |
| Refine capture poses | `priorMode` | ✅ |
| Masking: none, people & pets, moving things, remove sky; invert | splatprep `masks` (`person`, `sam2_auto`, `sky`) | ✅ (Apple Vision ➖ → `person`; "keep main subject" / "isolate one object" ⏳) |
| Stacked masks (each source ANDed into `masks/`, 4 px dilation) + separate sky layer `splatprep-sky/masks/` | `studio/masks.py` | ✅ |
| Mask fixed parts of the frame (static regions) | | ⏳ |
| Training presets (Quick preview / Balanced / High / Maximum) + every `train.*` setting | `studio/presets.py`; the advanced page edits a TrainingConfig JSON per field | ✅ |
| Start training when ready, progress page with stages, live log, reuse earlier results | | 🟡 stages + log + auto-train ✅; reuse earlier results ⏳ |

### Training

| Mac feature | Linux | Status |
|---|---|---|
| Local training | the CUDA gsplat trainer, as a job of the local server | ✅ |
| Live progress: step, steps/s, elapsed / ETA, splats, PSNR, holdout PSNR | job status polling | ✅ |
| PSNR and splat-count charts | `Gtk.DrawingArea` | ✅ |
| Live preview in the viewer | `preview.splat` reloaded on each new preview step | ✅ |
| Pause / Resume | worker jobs can't pause | ⏳ (Stop is ✅) |
| Save checkpoint | | ⏳ |
| Result saved in `outputs/` with skybox sidecar, opened in the viewer | ✅ | ✅ |
| Compute devices (`train.device`): This Mac / workers / Both | this PC's GPUs; other Linux workers / the Mac's Metal trainer | 🟡 local GPUs ✅; sending jobs to other workers ⏳ |
| Block-parallel training of large scenes | the worker already trains a job too big for one device in blocks (1.3.15) | ✅ (single machine); multi-machine ⏳ |
| Benchmark (steps/s, upload/download) | `/v1/benchmark` on the worker | ⏳ |

### Export (`export.*`)

| Mac feature | Linux | Status |
|---|---|---|
| PLY, compressed PLY, .splat, SPZ | `studio/formats.py` | ✅ |
| Export SH degree, only kept splats, bake orientation | ✅ | ✅ |
| Skybox sidecar with the export | ✅ | ✅ |
| Estimated file size | ✅ | ✅ |
| Screenshot (4K PNG) | offscreen GL framebuffer | ✅ |
| Turntable video | frames from the GL viewer → ffmpeg/PyAV if present | ⏳ |

### Network, Inbox, models, reference

| Mac feature | Linux | Status |
|---|---|---|
| Worker server for the Mac (Bonjour `_splatworker._tcp`, token, GPUs, jobs, log) | unchanged; its pages are now "Queue & Worker" | ✅ |
| iPhone Inbox (`_splatcapture._tcp` on 7879, pairing code, receive zips) | the same receiver protocol (PROTOCOL.md §5) in the GUI process | ✅ |
| Models & Benchmarks (installed models, licence, size; install packs) | model cards from `tools/splatprep/model_cards.json`; installs through the setup assistant | 🟡 list ✅, per-model benchmark table ⏳ |
| Settings Reference (searchable, grouped, details) | `studio/catalog.json` generated from `SettingsCatalog.swift` (`tools/../linux/scripts/gen_catalog.py`; a test keeps it in sync) | ✅ |
| CLI (`--convert`, `--info`, …) | `splatworker studio convert|info` | ✅ (render/eval/benchmark ⏳) |

## Notes and differences

* **Auto clean.** The Mac's census records each splat's *peak* contribution to any pixel. gsplat exposes
  gradients cheaply, so the worker measures each splat's *summed* blending weight per photo (∂(Σ pixels)/∂colour)
  and treats "≥ 0.02 in a photo" as "seen by that photo". A splat that adds 2 % to one pixel passes both tests;
  a splat spread thinly over many pixels can pass here and fail on the Mac. The plan rules (2 %, 25 % of the
  typical photo count, minimum typical count 12) are identical.
* The Flatpak's app id stays `com.example.SplatWorker` so existing installs update in place; the window and
  launcher are called **Splat Studio**.

## What needs testing on the PC (RTX 4080, CachyOS)

See the bottom of this file after each round of work — kept as a checklist with exact commands.
