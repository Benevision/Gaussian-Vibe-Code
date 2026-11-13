# Development status

Where SplatWork stands, what is tested and how, what is known to be missing, and how releases are made. Updated for
**1.2.0** (2026-11-13): SplatWork 1.2.0 for Mac, SplatWork for Linux 1.5.0, SplatWork Capture 1.2.

## Components and versions

| Component | Version | Where the version lives | Shipped as |
|---|---|---|---|
| SplatWork (Mac) | 1.2.0 | the release tag (`SPLAT_VERSION` in `mac/scripts/build_app.sh`); build number = commit count | `.dmg` on the GitHub release; a rolling `dev` pre-release from `main` |
| SplatWork for Linux | 1.6.0 (unreleased; 1.5.0 in the 1.2.0 release) | `linux/splatworker/__init__.py`, `linux/pyproject.toml`, `<release>` in `linux/flatpak/com.example.SplatWorker.metainfo.xml` | Flatpak repository on GitHub Pages (every push to `main`), `.flatpak` bundle on the release |
| SplatWork Capture (iPhone) | 1.2 (build 2) | `MARKETING_VERSION` / `CURRENT_PROJECT_VERSION` in `ios/project.yml` | build from source (not on the App Store / TestFlight) |
| SplatWork for Windows | 1.3.0 (unreleased; `windows/VERSION`, the release tag in releases) | `windows/VERSION`; `bundle.json` in the app folder | `.msi` + portable `.zip` on the GitHub release (from the next release) |
| splatprep | 0.1.0 | `tools/pyproject.toml` | inside the Mac app and the Linux Flatpak |

Identifiers that must not change (installs and data depend on them): Flatpak app id `com.example.SplatWorker`, Bonjour
services `_splatworker._tcp` / `_splatcapture._tcp`, the Mac module and binary `SplatStudio`, Keychain service
`SplatStudio.WorkerToken`.

## What is tested, and how

| Area | How | Coverage |
|---|---|---|
| Linux worker + studio core | `cd linux && python -m pytest -q tests` (no GPU needed; fake backends) | 304 tests: API, queue, pause/resume/shutdown, processing, datasets, devices, block training, GUI logic, formats, photo import |
| splatprep | `cd tools && python -m pytest -q tests` (needs opencv-python-headless, scipy) | 107 tests |
| Vulkan trainer (`linux/vktrain`, stages 1, 2a, 2b) | `vktrain-selftest` and `tests/test_vktrain.py` on Mesa lavapipe in `linux/vktrain/ci/Dockerfile` (CI: `vulkan-trainer.yml`; the pytest file skips without glslang + a Vulkan device) | every kernel against a double-precision CPU reference and finite differences (incl. the 2b image-space loss pipeline: skybox, colour correction, decoupled SSIM, depth / normal losses; transient masks against trainer.py's rule; the camera-pose gradient; compact Adam against a host reference); the host strategies; end-to-end `splatwork-vk-train train` runs on synthetic scenes (both strategies, pause / resume incl. the 2b state, checkpoint, cancel, region, and one run per 2b feature on a scene made for it: sky, per-photo colour shifts, depth maps, moving objects, noisy poses). Against the CUDA trainer on the RTX 4080: by hand, `compare_training.sh` (not run yet) |
| Linux GTK interface | `linux/scripts/gui_smoke.py` under Xvfb in a Debian container with Mesa (OpenGL via llvmpipe): screenshots of every page | layout and wiring, not real GPUs |
| Mac app | `swift build -c release`; headless CLI: `--prepare`, `--train [--blocks N]`, `--eval`, `--convert`, `--render`, `--survey`; UI snapshots with `SPLAT_UI_SNAPSHOTS=<dir>` (plus `SPLAT_UI_PROJECT`, `SPLAT_UI_DRONE`) | end to end on real photos; the trainer's gradients are checked |
| iPhone app | `xcodebuild` for the simulator and for devices (Release) | builds; not run on a phone for 1.2 |
| SplatWork for Windows | `.github/workflows/windows.yml` on `windows-latest` (no GPU): the Linux suite and splatprep tests on Windows, worker smoke (start, `/v1/info`, mDNS, stop), GTK studio smoke with screenshots (Mesa software OpenGL), MSI build → silent install → installed worker + studio → firewall rules → uninstall | startup, packaging and discovery on one machine; not real GPUs, real drivers or a real LAN |
| Releases | `gh workflow run release.yml -f version=X.Y.Z-rc1` builds everything without publishing | 1.2.0 was dry-run first |

Not covered by automated tests: CUDA training on real NVIDIA GPUs (done by hand on an RTX 4080), Apple Object Capture
(flaky now and then: a set that fails once usually solves on the next run), multi-device runs on a real network.

## Known issues and gaps

- Cancelling a Linux multi-device run can leave the uploaded photos on the other workers (the delete arrives while
  the remote job is still cancelling).
- Not yet exercised on real hardware for 1.2: Mac block training across several machines with Retry / Pause, the
  iPhone app, SplatWork for Linux 1.5.0 on an NVIDIA PC.
- After an NVIDIA driver update the Flatpak's matching driver extension may be missing; the setup screen shows the
  `flatpak install flathub org.freedesktop.Platform.GL.nvidia-<version>//1.4` command.
- Drone video: locations from DJI flight logs (`.SRT`) aren't read yet.
- Processing on a worker is orchestrated step by step from the client; only training runs independently while the
  client is offline (worker-side whole-pipeline jobs are in progress).
- The Mac release is ad-hoc signed (no Developer ID secrets in the repository), so it isn't notarized.

## Next work

1. **AMD and Intel GPUs in SplatWork for Linux** — detection, setup wording and ROCm / XPU processing runtimes are in
   1.6.0 (untested on real AMD / Intel hardware); training needs a decision on the path: see [GPU_SUPPORT.md](GPU_SUPPORT.md).
2. **Linux parity with this release's Mac features:** the Projects list with stage checkpoints and resume, drone
   mode and the photo-location switch, Show Live View, block Retry / Leave Out / Pause.
3. **A Windows app** with the Mac and Linux feature set: built and green in CI ([WINDOWS.md](WINDOWS.md)); needs
   a real Windows PC with an NVIDIA GPU for the first hardware test.
4. **Steam Frame capture:** recording with the stereo cameras and the IR cameras together, for room meshing that
   makes splats more accurate.

## Releasing

1. Move the `## Unreleased` notes in `CHANGELOG.md` under `## X.Y.Z — date` (the release workflow puts that section
   first in the release notes).
2. Bump the Linux version (`__init__.py`, `pyproject.toml`, a `<release>` in the metainfo) and the iPhone version if
   they changed.
3. Run the tests above, then a dry run: `gh workflow run release.yml -f version=X.Y.Z-rc1`.
4. Tag the commit the dry run built and push the tag:
   `git tag -a vX.Y.Z <commit> -m "SplatWork X.Y.Z" && git push origin vX.Y.Z`.
   The workflow builds the `.dmg`, the Flatpak bundle and the Windows `.msi` / `.zip` (via `windows.yml`), writes `SHA256SUMS.txt` and publishes the release; Mac apps
   offer the update, and the website's download button follows the latest release.

Every push to `main` also publishes the Mac developer build (`dev` pre-release) and, for changes under `linux/`,
`tools/`, `website/` or `docs/images/`, the Linux Flatpak repository and the website.
