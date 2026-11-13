# SplatWork for Windows: plan

*Developer notes for the roadmap item "a Windows app with the Mac and Linux feature set". Status: **plan, waiting
for the user's choice of UI route** (2026-11-13). Nothing below is built yet.*

## Summary

SplatWork for Windows should **reuse SplatWork for Linux almost entirely**: the Python worker
(`linux/splatworker/` — server, queue, protocol, trainer, processing), the studio core (`linux/splatworker/studio/`
— projects, devices, compute, blocks, renderer, editing) and splatprep (`tools/`). Only the window toolkit's
packaging, a handful of Linux-only system calls, the setup assistant's runtime table, and the installer / updater
are Windows work.

Recommended route: **GTK 4 + libadwaita on Windows**, sharing the Linux interface code
(`linux/splatworker/gui/gtk/`), packaged with a bundled Python and GTK as a per-machine **MSI** (WiX) plus a
portable **zip**, built on `windows-latest` in the release workflow. NVIDIA trains with gsplat / CUDA through a
downloaded runtime (as Linux's CUDA profile does); AMD and Intel get the shared Vulkan trainer
(`splatwork-vk-train`, see [GPU_SUPPORT.md](GPU_SUPPORT.md)) when it lands, with no separate Windows trainer.

Windows-specific code lives under `windows/`; the shared code gets small, tested portability fixes.

## What is already portable

Measured on main at 4c15996: about 18,800 lines of Python in `linux/splatworker/` excluding the GTK pages, and
6,400 lines of GTK interface (4,200 of them the studio pages).

- **Worker processes** use `multiprocessing.get_context("spawn")` (`backends.py`), the start method Windows needs.
  Pause, checkpoint and cancel are `mp.Event`s, not signals.
- **mDNS** advertising and browsing use python-zeroconf, which runs on Windows; the service stays
  `_splatworker._tcp` and the protocol is unchanged, so a Mac finds a Windows worker exactly as it finds a Linux one.
- **splatprep** uses PyAV for video, pycolmap / COLMAP / GLOMAP for the camera solve and PyTorch for features and
  masks; PyAV, pycolmap, OpenCV and PyTorch (CUDA 12.x) publish Windows wheels. COLMAP and GLOMAP publish Windows
  binaries (CUDA and non-CUDA) if the CLI path is wanted.
- **The viewer** is OpenGL 4.3 through ctypes (`studio/gl.py`); Windows drivers from NVIDIA, AMD and Intel all
  provide 4.3+.
- **NVML** ships with the NVIDIA driver on Windows (`nvml.dll`), and `nvidia-ml-py` loads it; `nvidia-smi.exe` is
  on `PATH` too.

## Linux-only code paths and what abstracting them costs

| Place | Linux assumption | Windows replacement | Cost |
|---|---|---|---|
| `server.py`, `viewstore.py` | `/proc/meminfo` for memory | `GlobalMemoryStatusEx` via ctypes (a shared `sysinfo.memory()` helper) | small |
| `processing.py`, `studio/pipeline.py` | `os.symlink` to the uploaded dataset / video | symlinks need Developer Mode or admin on Windows: hard links (same volume) for files, directory junctions for folders, copy as the last resort | small–medium; needs care because datasets are large |
| `gui/server_process.py`, `gui/installer.py` | `start_new_session` + `os.killpg` | `CREATE_NEW_PROCESS_GROUP` and a Job Object so the worker's children die with it (or `taskkill /T`) | small |
| `discovery.py` `interface_kind` | `/sys/class/net` (falls back to names) | ifaddr's adapter "nice names" on Windows ("Ethernet", "Wi-Fi", "vEthernet (WSL)") into the name heuristic | small |
| `gui/paths.py`, `gui/settings.py` | XDG directories, Flatpak `/app` | `%APPDATA%\SplatWork` (settings, token), `%LOCALAPPDATA%\SplatWork` (runtime, data, cache), install dir for bundled splatprep | small |
| `studio/projects.py` | `xdg-user-dir DOCUMENTS` | Known Folder `Documents` (`SHGetKnownFolderPath`) → `Documents\SplatWork` | small |
| `os.chmod(0o600/0o700)` (token, data dir) | POSIX modes | no-ops on Windows; files under `%APPDATA%` are already private to the user | none |
| `gpu_vendors.py`, `gui/nvidia.py` | sysfs card list, `/proc/driver/nvidia`, Flatpak `GL.nvidia` extension wording | **DXGI** adapter enumeration (`IDXGIFactory1::EnumAdapters1`: vendor id, name, dedicated VRAM) via ctypes for every card; NVML for NVIDIA details; Windows wording ("Update the NVIDIA driver from nvidia.com") | medium |
| `gui/runtime_plan.py` | Linux wheel tags, standalone CPython for Linux | a Windows profile table: python-build-standalone 3.10 `x86_64-pc-windows-msvc`, PyTorch cu128 / cu124 Windows wheels, gsplat Windows wheels (see Training) | medium |
| `gui/updates.py`, `gui/gtk/flatpak_update.py`, `gui/gtk/portal.py` | Flatpak OSTree repo and portals | the GitHub release's MSI (below) | medium (new code, under `windows/`) |
| `gui/autostart.py` | `~/.config/autostart/*.desktop` | `HKCU\Software\Microsoft\Windows\CurrentVersion\Run` | small |
| `studio/turntable.py` | GStreamer through PyGObject | PyAV's encoder (already a splatprep dependency), used on every platform or only on Windows | small–medium |
| file names | any byte except `/` | reserved names (`CON`, `AUX`…), `:` `*` `?` `"` `<` `>` `|`, 260-character paths: sanitize project / photo names and enable long paths in the app manifest | small |

None of these is large; the total is roughly one to two weeks of work including tests. Each fix is a small shared
change behind a `sys.platform` check or a helper, tested on Linux (`linux/tests`) and on the Windows CI runner.

The Vulkan session owns `trainer.py`, `gpu.py`, `runtime_plan.py` and the Flatpak files: the Windows equivalents go
into `windows/` (a Windows runtime table, a DXGI adapter source) and plug in through small hooks, agreed with that
session before touching those files.

## UI route: options

| | A. GTK 4 + libadwaita on Windows | B. Qt / PySide6 | C. Native .NET (WinUI 3 / WPF) over the worker |
|---|---|---|---|
| Code shared with Linux | all of `gui/gtk/` (6,400 lines) and the studio core | studio core only; 6,400 lines of interface rewritten (or Linux moved to Qt too) | only the worker over HTTP; interface **and** the studio core's logic (projects, devices, blocks, editing, viewer renderer) rewritten in C# |
| Look on Windows | libadwaita style (consistent with the Linux app, not native Windows) | close to native | native |
| Viewer | `Gtk.GLArea` (WGL on Windows) + the existing `studio/gl.py` with a `wglGetProcAddress` loader | `QOpenGLWidget` + the existing renderer | OpenGL through OpenTK, or a Direct3D port of the renderer |
| Packaging | hardest part: GTK, libadwaita and PyGObject for Windows, from gvsbuild (MSVC builds, also used by other Python GTK apps) or MSYS2; ~150 MB | easy: PySide6 wheels + PyInstaller / bundled Python | easy: MSIX / MSI tooling built in |
| Maintenance | one interface for Linux and Windows | two interfaces to keep in step | three interfaces (Swift, GTK, C#) |
| Effort for Stage B | ~2–3 weeks (packaging, Windows quirks) | ~6–8 weeks | ~10+ weeks |

**Recommendation: A.** The Linux app already mirrors the Mac's layout and wording; keeping one interface for Linux
and Windows means every parity fix lands once. The risk is concentrated in packaging GTK, which is a build problem
solved once in CI, not an ongoing cost. If GTK on Windows turns out unusable (rendering, file dialogs, HiDPI), the
fallback is B, which still reuses the studio core.

## Training

- **NVIDIA:** gsplat / CUDA, the same `trainer.py` as Linux. PyTorch publishes CUDA wheels for Windows; gsplat
  publishes prebuilt Windows wheels for some PyTorch / CUDA combinations on its wheel index. Where the combination we
  need is missing, CI builds the wheel on `windows-latest` with the CUDA toolkit and MSVC (compiling needs no GPU),
  as `linux/flatpak/build-gsplat-wheel.sh` does for Linux. fused-ssim is optional (the trainer falls back to plain
  PyTorch SSIM).
- **AMD and Intel:** the shared Vulkan trainer `splatwork-vk-train` (C++17, GLSL → SPIR-V at build time), built for
  Windows with MSVC once it exists. It needs no download, which is the point. No separate Windows trainer is built;
  until it lands, AMD / Intel Windows PCs do processing only (on the CPU; DirectML is not planned), with the same
  "training needs NVIDIA for now" wording as Linux.
- `TrainingConfig` and the protocol are unchanged, so the Mac sees a Windows worker like any other.

## Runtime install (setup assistant)

Same design as Linux: the app's own bundled Python runs the interface and a processing-only worker; **Set Up**
downloads a GPU runtime into `%LOCALAPPDATA%\SplatWork\runtime`:

1. python-build-standalone CPython 3.10 for Windows (checksum-pinned);
2. a venv with PyTorch (cu128 for driver ≥ 570 / RTX 50, else cu124), gsplat, nvidia-ml-py, splatprep and its
   dependencies;
3. a verify step (`splatworker gpu-check`).

The CUDA profile is chosen from NVML's driver version and compute capability, as on Linux. The download is about
3 GB; the assistant shows progress from pip (`gui/pip_progress.py` is reused).

## Packaging and updates

- **Per-machine MSI built with WiX** (v5) on `windows-latest`, unsigned at first (SmartScreen warns once, like the
  ad-hoc signed Mac build), plus a **portable zip** of the same files.
  - Why MSI and not MSIX: MSIX must be signed with a certificate the PC trusts, and its container redirects
    `%APPDATA%` writes and restricts spawning the downloaded runtime's `python.exe`. MSI has neither problem.
  - Per-machine so the installer can add the firewall rules (one UAC prompt).
- **Contents:** an embedded CPython 3.12 (python.org's embeddable zip), the `splatworker` package, splatprep, GTK /
  libadwaita / PyGObject from gvsbuild, a small `SplatWork.exe` launcher (so Task Manager, the taskbar and the
  firewall name it "SplatWork", not "python"), Start-menu shortcut, uninstaller.
- **Firewall:** the MSI adds inbound rules for the **Private** profile only: TCP 7878 (the worker port) and UDP 5353
  (mDNS), by port, because the listening process is the runtime's `python.exe` under `%LOCALAPPDATA%`. The portable
  zip relies on Windows' own "Allow access" prompt; Help explains it. If the network is set to Public, the studio
  says so (Windows hides the PC from the Mac on Public networks).
- **Updates:** the app checks the GitHub release API (`/repos/Benevision/SplatWork/releases/latest`) for the
  Windows asset, offers it, downloads the MSI, verifies it against `SHA256SUMS.txt` and runs `msiexec` (major
  upgrade: same UpgradeCode, the runtime and projects stay). A rolling `dev` build, like the Mac's, can follow later.
- **Release workflow:** a `windows` job in `.github/workflows/release.yml` builds the MSI and zip, runs the tests
  and a startup smoke test, and adds them to the release and `SHA256SUMS.txt`.

## Device paths

- **GPU detection:** DXGI for every card (vendor, name, VRAM), NVML for NVIDIA details and live statistics. AMD /
  Intel live load is not available without vendor SDKs; the Status page shows VRAM only for them.
- **Paths:** settings and token `%APPDATA%\SplatWork`, runtime / data / cache / logs `%LOCALAPPDATA%\SplatWork`,
  projects `Documents\SplatWork` (the Known Folder, which may be redirected to OneDrive). Long paths enabled in the
  manifest.
- **Discovery:** the worker advertises `_splatworker._tcp` with python-zeroconf on the LAN adapters (Ethernet first,
  skipping Hyper-V / WSL `vEthernet` adapters), so the Mac's Compute → Devices finds it. The studio browses with the
  same code as Linux.
- **Run at sign-in:** the HKCU `Run` key (Preferences, off by default, as on Linux).

## Stages

1. **Stage A — worker on Windows.** The shared portability fixes above; `windows/` with paths, DXGI adapters, the
   process-group handling and a `splatworker` entry that runs headless; CI on `windows-latest` running the Linux
   test suite and splatprep tests, plus a worker start / `/v1/info` / mDNS advertise smoke test. Done when the tests
   pass on Windows CI and (on a real PC) a Mac discovers the worker and sends it a processing and a training job.
2. **Stage B — studio.** GTK 4 on Windows (route A): the Projects list, new project flow, viewer / editor, Compute
   devices, setup assistant with the Windows CUDA profile, Preferences, Help. CI starts the studio under Mesa's
   software OpenGL for Windows and screenshots each page, like `linux/scripts/gui_smoke.py`.
3. **Stage C — release.** MSI + zip, firewall rules, updater, the `windows` job in `release.yml`, README, website
   download button, user guide, CHANGELOG.

## Testing and what stays untested

- **CI (`windows-latest`, no GPU):** unit tests, worker startup, `/v1/info`, mDNS registration, the studio
  starting and drawing its pages with software OpenGL, MSI install / uninstall.
- **Needs a real Windows PC:** training on NVIDIA (CUDA runtime download, gsplat wheels), the viewer on real
  drivers, firewall prompts and Mac discovery over a real LAN, HiDPI, OneDrive-redirected Documents. Without a test
  PC these are shipped as untested and said so in the release notes.

## Questions for the user

1. Which UI route: **A (GTK 4, recommended)**, B (Qt / PySide6) or C (native .NET)?
2. Is there a Windows PC to test on (for example the RTX 4080 PC dual-booted into Windows), or should Stages A–C
   rely on CI only until one is available?
