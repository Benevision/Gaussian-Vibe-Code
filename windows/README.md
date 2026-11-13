# SplatWork for Windows

The SplatWork worker and studio for Windows 10 / 11 (x64). It is the same app as [SplatWork for Linux](../linux/README.md),
with the same GTK 4 interface and the same Python worker, packaged for Windows. Design notes and status are in
[docs/dev/WINDOWS.md](../docs/dev/WINDOWS.md).

## Install

Download `SplatWork-X.Y.Z-Windows-x64.msi` from the [Releases page](https://github.com/Benevision/SplatWork/releases)
and run it. Windows asks once for administrator rights, because the installer adds two firewall rules, for
**Private** networks only:

- TCP 7878, so Macs on your network can send jobs to this PC;
- UDP 5353 (Bonjour / mDNS), so Macs can find it.

The installer isn't code-signed yet. If SmartScreen warns, choose **More info** → **Run anyway**.

A portable `SplatWork-X.Y.Z-Windows-x64.zip` is published too. Unzip it anywhere and run `bin\SplatWork.exe`. It
adds no firewall rules, so allow SplatWork when Windows asks.

On first start, the setup assistant installs the GPU runtime into `%LOCALAPPDATA%\SplatWork`: Python 3.10, PyTorch
with CUDA and gsplat for NVIDIA cards (about 3 GB). After that, turn the server on, and the Mac finds the PC under
**Compute**.

| | |
|---|---|
| Settings and token | `%APPDATA%\SplatWork` |
| GPU runtime, worker data, downloads | `%LOCALAPPDATA%\SplatWork` |
| Projects | `Documents\SplatWork` |
| Command line | `"C:\Program Files\SplatWork\bin\splatwork-cli.exe" --headless` (also `info`, `gpu-check`, `benchmark`) |

Updates: SplatWork checks the latest release. **Update…** downloads the new installer, checks it against the release's
`SHA256SUMS.txt`, closes SplatWork and installs the new version. Your settings, GPU runtime and projects are kept.

## GPUs

- **NVIDIA:** training and processing with gsplat / CUDA, like SplatWork for Linux. The runtime profiles are
  PyTorch 2.4 with CUDA 12.4, 12.1 or 11.8, using gsplat 1.5.3's prebuilt Windows wheels. RTX 50-series cards need
  the CUDA 12.8 profile, which has no Windows gsplat wheel yet.
- **AMD and Intel:** the viewer, editing and processing work, and processing runs on the CPU. Training comes with
  the shared Vulkan trainer that is in development ([GPU_SUPPORT.md](../docs/dev/GPU_SUPPORT.md)).
- The viewer needs OpenGL 4.3. NVIDIA, AMD and Intel drivers provide it; Windows' basic display driver doesn't.

## Building

Everything is built on GitHub Actions in [`.github/workflows/windows.yml`](../.github/workflows/windows.yml), from
MSYS2 UCRT64 (GTK 4, libadwaita, PyGObject, Python):

1. `windows/launcher/build.sh VERSION OUT`: builds `SplatWork.exe` (GUI) and `splatwork-cli.exe` (console). They
   embed the bundled Python, so Windows shows "SplatWork" instead of "python".
2. `windows/build/make_bundle.py`: builds the app folder. It walks the PE import tables, so only the DLLs the app
   needs are copied.
3. `windows/build/make_wxs.py` + WiX 5: builds the MSI (per-machine install, firewall rules, Start-menu shortcut,
   major upgrades).

The workflow then installs the MSI on the runner and tests it:

- it starts the installed worker, then checks `/v1/info`, mDNS discovery and a clean stop;
- it opens the installed studio;
- it checks the firewall rules, then uninstalls.

`release.yml` calls the same workflow to add the MSI and zip to each release. `windows/VERSION` is the version
used for builds outside a release.

Tests (`windows-latest`, no GPU): the Linux worker and studio suite, the splatprep suite, a worker smoke test
(`windows/scripts/worker_smoke.py`) and a studio smoke test that screenshots every page
(`windows/scripts/studio_smoke.py`, software OpenGL from Mesa).
