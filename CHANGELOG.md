# Changelog

Notable changes to SplatWork. The Mac app is released by version tag (see
[Releases](https://github.com/Benevision/SplatWork/releases)); a rolling *developer build* follows `main`.
SplatWork for Linux is published to its Flatpak repository on every change and carries its own version number.

## Unreleased

- **SplatWork for Windows (new, untested on real hardware):** the Linux worker and studio, with the same GTK 4
  interface, packaged for Windows 10 / 11 as an `.msi` installer and a portable `.zip`.
  - It includes the Projects list, the new project flow, the viewer and editor, Compute devices and the setup
    assistant. The assistant installs PyTorch, CUDA and gsplat for NVIDIA cards using gsplat's Windows wheels.
  - Macs find the PC over Bonjour. The installer opens the worker's port and mDNS on Private networks only.
  - It updates itself from new releases, checking each installer against the release's checksums.
  - Built, installed and smoke-tested on GitHub's Windows runners, which have no GPU. Training on an NVIDIA GPU,
    the viewer on real drivers and discovery from a Mac over a real network haven't been tried yet.
  - AMD and Intel GPUs can view and process; they'll train once the Vulkan trainer lands.
- **SplatWork for Linux 1.6.0 — AMD and Intel graphics cards:** the app now recognises AMD Radeon / Instinct and
  Intel Arc cards (name, video memory, load), shows them on the Status page and to the Mac, and explains that
  training still needs an NVIDIA GPU instead of saying "no GPU found". On those cards it offers a processing runtime
  with PyTorch for ROCm (AMD) or for Intel GPUs, so feature matching and masks can use the GPU. Not yet tested on
  real AMD or Intel hardware; training on AMD / Intel is not available (options and plan in
  `docs/dev/GPU_SUPPORT.md`).
- **Whole pipeline on one worker:** when a project's processing and training both go to the same SplatWork for
  Linux worker (1.6.0 or later), the Mac uploads the video (or the photos it imported) and every setting once, and
  the worker runs frames, masks, sky, camera solving, levelling and training by itself — the Mac can sleep, go
  offline or quit meanwhile. Reopening the app (or *Reconnect to the Worker Job* on the project page) picks the job
  up and fetches the dataset, with the same per-stage checkpoints a local run leaves, and the result. Older workers,
  Apple Vision masks, Object Capture and stereo or posed sources keep the step-by-step flow.

## 1.2.0 — 2026-11-13

SplatWork 1.2.0 for Mac, SplatWork for Linux 1.5.0.

### Highlights
- **Never start over:** every processing stage is checkpointed and training saves checkpoints and finished blocks,
  so quitting, a crash or Stop costs at most the step that was running. A new **Projects** list shows where every
  project stands and resumes or redoes it from any stage.
- **Drone footage and photo locations:** imported photos keep their GPS (only that), so camera solving checks the
  cameras against it and puts the scene in metres. Drone photos are recognised by the drone's own markers — not by
  GPS, so phone photos stay ordinary — and get aerial settings; there's a new **Drone / aerial** tile.
- **Training on several devices, sturdier:** Retry or leave out a failed block without stopping the rest, Pause and
  Stop reach every device, a live view of block runs, and Show Live View to get back to a run from anywhere.
- **SplatWork for Linux:** the Linux app (formerly SplatWork Node) now has the Mac's layout and wording, and keeps
  training jobs' saved state when it is stopped.
- **New look:** the hex mesh logo on every app, the website and the docs.

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
- **Choose the processing device** in New Project (Cameras & masks page) and when resuming, separately from the
  training devices; the training devices you tick are remembered.

### Photo locations and drone footage
- Imported photos keep only their location (GPS, and a drone's height above take-off), on the Mac and on Linux,
  under a new **Use photo locations** switch; off strips locations from the project's photos.
- Drone footage (camera maker, DJI flight / gimbal tags, DJI file names) gets **Aerial settings**: learned camera
  solving with GPS checks, the sky as a learned skybox, the High quality preset.
- The Maximum preset stores photos at its 2400 px training resolution.
- Sky masking is on for video, photos and 360° whichever way the project is started, and is skipped with a note
  (not a failed run) when no sky model is available.

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

### SplatWork for Linux 1.4–1.5
- The Linux app is now a full studio: viewer, New Project, editing (including Auto clean and the mask editor),
  export, turntable videos, iPhone Inbox, compute devices and block-parallel training across machines, all next to
  the worker server.
- Renamed from SplatWork Node to SplatWork; sidebar, start page and New Project follow the Mac's.
- Worker jobs can be paused (state saved, GPU freed) and checkpointed; stopping the worker pauses running training
  jobs instead of discarding their state.
- Multi-device runs follow a device that changes address and retry a block here after 30 minutes unreachable.
- The setup screen shows the exact command to install a missing NVIDIA driver extension.

### Fixes
- About 30 issues from an independent review of this release's new code, among them: masks from an earlier run
  baked into reused frames; two preparations running in one folder after Stop; Stop not reaching a block
  training on the Mac; a failed block holding up its device's other blocks; resumed runs on the wrong devices or
  from step 0; the Linux mask editor not showing why a job failed.

### Security
- Mac worker requests need an access token; Bonjour-advertised addresses are only trusted once the worker confirms
  them; dataset paths can't escape their folder; benchmark runs are isolated.

### Changes
- Licensed under Apache 2.0; public documentation, website and project files.
- The repository moved to `Benevision/SplatWork`.
- Local Mac builds are signed with a development identity when there is one, so the Keychain stops asking again
  after every update.
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
