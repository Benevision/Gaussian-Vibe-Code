# Changelog

Notable changes to SplatWork. The Mac app is released by version tag (see
[Releases](https://github.com/Benevision/SplatWork/releases)); a rolling *developer build* follows `main`.
SplatWork for Linux (Linux) is published to its Flatpak repository on every change and carries its own version number.

## Unreleased

### Projects and resuming
- **Projects list** in the sidebar replaces Recent: every project you've worked on, including ones made with New
  Project and training results, with its status (processing, interrupted, ready, training, trained).
- **Project page**: each processing stage with what it produced, *Resume*, *Redo from Here*, *Change Settings…*,
  *Train*, and every result. The start page offers projects that stopped partway.
- **Processing checkpoints**: each stage (frames, masks, camera positions, dataset) is saved and reused on any rerun
  whose settings and source files haven't changed. Quitting, a crash, Stop or a failure costs at most the stage
  that was running. *Try Again* after a failure continues where it failed.
- **Training resume**: this Mac checkpoints every 10 minutes and on Stop; block runs keep every finished block and
  reuse it.
- **Choose the processing device** in New Project and when resuming, separately from the training devices.

### Training on several devices
- A failed block waits for **Retry** (on any device) or **Leave This Block Out** instead of failing the whole run.
- **Pause / Resume** now reach worker jobs and blocks, not only a single run on the Mac. **Stop** stops a block
  training on the Mac too, and shows *Stopping…* while the run winds down.
- **Live view for block runs**: the viewer shows every block's latest state, merged.
- **Show Live View**: look at other projects while one trains, then get back to it from the toolbar, the banner,
  the Train tab or the project page. Home no longer stops training.
- Charts no longer draw a line back to zero when a block restarts; block cards keep the device's name if it drops
  off the network.

### Quality
- **Flat surfaces**: surface-normal consistency loss and flat-splat regulariser on by default; predicted depth
  (Depth Anything V2 on the Mac, monocular depth priors on Linux) for photos without a depth sensor.
- **Fisheye footage**: solved with a fisheye lens model, with masks warped along.
- Camera solving drops cameras placed far off the camera path.

### SplatWork for Linux 1.4
- The Linux app is now a full studio: viewer, New Project, editing (including Auto clean and the mask editor),
  export, turntable videos, iPhone Inbox, compute devices and block-parallel training across machines, all next to
  the worker server.
- Worker jobs can be paused (state saved, GPU freed) and checkpointed.
- The setup screen shows the exact command to install a missing NVIDIA driver extension.

### Security
- Mac worker requests need an access token; Bonjour-advertised addresses are only trusted once the worker confirms
  them; dataset paths can't escape their folder; benchmark runs are isolated.

### Changes
- The repository moved to `Benevision/SplatWork`.
- iPhones no longer act as training workers for other devices; SplatWork Capture still processes and trains its own
  scans.

## 1.1.0 — 2026-11-08

- Self-updating apps: the Mac app updates from GitHub Releases; SplatWork Node updates itself through Flatpak.
- A developer-build update channel on the Mac (Settings → Developer).
- The iPhone app makes splats on its own and can hand the heavy steps to Macs and PCs; a Mac can act as a worker.
- Compute devices are checkboxes: this Mac and each worker.
- A job too big for its one device trains there in blocks that fit instead of being refused.
- Learned skybox on the Mac trainer, saved alongside the PLY.
- The preparation window can be hidden while it works.
- Renamed to SplatWork (Mac), SplatWork Capture (iPhone) and SplatWork Node (Linux).

## 1.0.0 — 2026-11-08

First release, as *Splat Studio*: Metal viewer and trainer, dataset preparation from photos, video, 360° video,
stereo footage, iPhone captures, COLMAP and Nerfstudio, Linux CUDA worker, editing and export.
