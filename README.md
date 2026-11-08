# SplatWork

A networked Gaussian-splatting suite: your Mac, your iPhone and your Linux GPU PCs work as one. Capture on the iPhone, prepare and view on the Mac, and train on whichever devices are free — big scenes are split into blocks and shared between them. Native Metal renderer and trainer on Apple devices, CUDA (gsplat) on Linux.

| Folder | What | Status |
|---|---|---|
| [`mac/`](mac/) | **SplatWork** — macOS app (SwiftUI + Metal). Viewer, editor, dataset preparation, Metal training, remote/“Both” compute, iPhone receiver | Builds (`swift build`), trainer gradient-checked, end-to-end training verified on synthetic data |
| [`ios/`](ios/) | **SplatWork Capture** — iPhone app (ARKit + LiDAR), TestFlight-ready. Makes splats on its own (prep, Apple Vision masks, Metal training, viewer) and hands heavy steps to Macs / PCs on the network when they're there | Builds clean for device and simulator (iOS 27.2 SDK); not yet run on a phone |
| [`linux/`](linux/) | **SplatWork Node** — CUDA training/processing server (FastAPI + gsplat) with a GTK 4 / libadwaita GUI, packaged as a Flatpak | 134 tests pass; the CUDA path, GUI rendering and Flatpak build have not been run on Linux yet |
| [`tools/`](tools/) | **splatprep** — learned feature matching/SfM, moving-object masking (OWLv2 + SAM 2 tracking, click-to-mask server) and benchmark harness (Mac MPS or Linux CUDA) | 77 tests pass; models installed and running on this Mac; benchmark in progress |
| [`docs/`](docs/) | [User guide](docs/USER_GUIDE.md) · [Settings reference](docs/SETTINGS.md) · [Architecture](docs/ARCHITECTURE.md) · [Protocols & formats](docs/PROTOCOL.md) | |

## Download

Ready-built releases are on the [Releases page](https://github.com/Benevision/Gaussian-Vibe-Code/releases):

* **SplatWork for Mac** (Apple silicon, macOS 15+) — `.dmg`, drag to Applications. Not notarized yet: the first
  time, right-click the app → **Open**.
* **SplatWork Node for Linux** (NVIDIA GPU) — single-file `.flatpak`: `flatpak install --user SplatWorkNode-*.flatpak`.

Releases are cut by pushing a tag (`git tag -a v1.4.0 -m "…" && git push origin v1.4.0`); see
`.github/workflows/release.yml`.

## Quick start (Mac)

```bash
cd mac && ./scripts/build_app.sh && open "build/SplatWork.app"
```

Drop a `.ply`/`.spz`/`.splat`, a folder of photos, a video, a 360° video, stereo footage or an iPhone capture onto the window.

Headless use:

```bash
mac/.build/release/SplatStudio --prepare path/to/photos-or-video --out MyProject [--solver objectcapture] [--mask people|dynamic]
mac/.build/release/SplatStudio --train MyProject/dataset --preset high --out scene.ply
mac/.build/release/SplatStudio --convert scene.ply scene.spz
mac/.build/release/SplatStudio --render scene.ply preview.png
mac/.build/release/SplatStudio --dump-settings > docs/SETTINGS.md
```

## What's inside (highlights)

- **Metal renderer**: GPU projection + stable radix sort (16-bit log-depth or 32-bit) + hardware-blended instanced quads; lazy re-sort when idle; 5 display modes (colour, depth, centres, ellipses, opacity).
- **Metal trainer**: full differentiable tile rasteriser with analytic backward pass (gradient-checked), fused L1 + D-SSIM + depth loss, Adam, **3DGS-MCMC** densification (relocation, budget, SGLD noise) and classic **adaptive density control** with **AbsGS** gradients, **Mip-Splatting style anti-aliasing** with opacity compensation, tight ellipse tile bounds, SH degree warm-up, LiDAR/stereo depth supervision, masks, holdout evaluation.
- **Sources**: photos (blur/duplicate rejection), video (sharpest-frame selection, scene-cut/static skipping), 360° equirectangular (virtual pinhole rigs, nadir handling), stereo (SBS/TB/dual files/MV-HEVC spatial video/Arcturus Vision; metric scale from baseline; stereo depth), iPhone LiDAR captures, COLMAP, Nerfstudio. HDR tone-mapping.
- **Camera solving**: Apple Object Capture (built in), COLMAP/GLOMAP with SIFT, SuperPoint/DISK/ALIKED + LightGlue, LoFTR, MASt3R, VGGT (via the optional Models pack), or capture poses.
- **Moving objects**: robust training that ignores transient regions (on by default, Mac + worker), Apple Vision people & pets masks, open-vocabulary detection + SAM 2 tracking, and a mask editor with click-to-select and track.
- **Compute**: This Mac / Linux worker / Both (stage placement + block-parallel training merged automatically).
- **Formats**: PLY (lossless), SPZ v2/v3 (read) / v2 (write), `.splat`.
- **UX**: Easy / Advanced modes; every setting has inline help and a full reference (Help → Settings Reference); guided New Project flow; live training dashboard with charts and live preview; undoable editing (brush selection, crop box, opacity/size/outlier cleanup); turntable video and 4K screenshots.
