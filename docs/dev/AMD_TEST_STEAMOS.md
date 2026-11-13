# Testing SplatWork for Linux on an AMD GPU under SteamOS

For the RX 6700 XT test PC (RDNA 2, gfx1031, 12 GB) running SteamOS. Everything comes through the Flatpak: SteamOS's
system is read-only and nothing is installed on it. PyTorch for ROCm brings AMD's GPU libraries with it; the host
only needs the amdgpu driver (SteamOS has it) and `/dev/kfd` + `/dev/dri`, which the app's `--device=all`
permission gives it.

Do this in **Desktop Mode**, in **Konsole**, as the normal user (`deck`). Each step prints something to paste back;
the important lines start with `SPLATWORK-`. The app sets `HSA_OVERRIDE_GFX_VERSION=10.3.0` for this card itself
(inside the app only), so don't set it in the shell.

## 1. Install

```bash
flatpak install --user https://benevision.github.io/SplatWork/flatpak/splatworker.flatpakref
```

Already installed? Update instead: `flatpak update --user com.example.SplatWorker`. Check the version (1.6.0 or
newer):

```bash
flatpak info --user com.example.SplatWorker | grep -i version
```

## 2. Detection (no runtime needed yet)

```bash
ls -l /dev/kfd /dev/dri; flatpak run --command=sh com.example.SplatWorker -c 'ls -l /dev/kfd /dev/dri'
```

```bash
flatpak run com.example.SplatWorker --cli info
```

Paste both outputs. Expected: `/dev/kfd` exists inside and outside the sandbox; `"adapters"` lists the card as
`"vendor": "amd"`, about 12288 MB, `"gfx": "gfx1031"`.

Then open **SplatWork** from the application menu once: the setup should say *"AMD Radeon RX 6700 … found"* and
offer *"Processing runtime … ROCm"* with a note about gfx1031 / 10.3.0. A screenshot of that page helps. Close it
with **Not Now** (the next step installs from the terminal, which keeps a log).

## 3. PyTorch for ROCm (processing runtime) and the GPU check

About 5 GB download, 14 GB disk:

```bash
flatpak run com.example.SplatWorker --install-runtime --profile rocm6.4 --lightglue --yes 2>&1 | tee ~/splatwork-install.log | tail -40
```

```bash
flatpak run com.example.SplatWorker --cli gpu-check --no-train 2>&1 | tee ~/splatwork-gpucheck.txt
```

Paste `~/splatwork-gpucheck.txt` (at least the last line, `SPLATWORK-GPU-CHECK {...}`). What we look for: torch
`hip` set, the device named with `gfx1031` or `gfx1030`, `alloc_1gb`, `matmul_fp32` (TFLOPS, `relError` below 0.01) and
`conv_backward` all ✓, splatprep device `cuda` with backend `rocm`.

If PyTorch can't see the card, also paste:

```bash
flatpak run --command=sh com.example.SplatWorker -c 'env | grep -E "HSA|HIP|ROC"; ls /sys/class/kfd/kfd/topology/nodes/*/'
```

## 4. Training spike (experimental)

Needs the gsplat-for-ROCm wheel the project builds on GitHub (prerelease `gsplat-1.5.3-rocm6.4`). If setup says it
isn't published yet, stop here and report that. This replaces the runtime from step 3 (the old one is kept until
the new one checks out):

```bash
flatpak run com.example.SplatWorker --install-runtime --profile rocm6.4-train --lightglue --yes 2>&1 | tee ~/splatwork-install-train.log | tail -40
```

```bash
flatpak run com.example.SplatWorker --cli gpu-check 2>&1 | tee ~/splatwork-gpucheck-train.txt
```

Look for `gsplat` ✓ with `msPerStep` and `gradFinite: true`.

Then the standard benchmark (the same one the Mac runs; about 10–20 minutes). Open SplatWork, make sure the
server is **on** (Queue & Worker page), then:

```bash
flatpak run com.example.SplatWorker --cli benchmark 2>&1 | tee ~/splatwork-benchmark.txt | tail -30
```

Paste the JSON at the end: `trainStepsPerSecond`, `psnr` and `evalPsnr` are what count (for comparison, the RTX 4080
PC's numbers are in its Models & Benchmarks page). A quick smoke test first, if the full run looks slow:
`flatpak run com.example.SplatWorker --cli benchmark --quick`.

If training fails, paste the job log from the Queue & Worker page (or the newest
`~/.var/app/com.example.SplatWorker/data/splatworker/data/jobs/*/train.log`).

## 5. Optional: a real scene

Copy a small photo set (30–80 photos) to the PC, *New Project → Photos* in SplatWork, Balanced, and compare the result
with the same set trained on the RTX 4080 (PSNR on the Projects page, and how it looks).

## Undo

The runtime lives in the app's own folder; to remove everything: `flatpak uninstall --user --delete-data
com.example.SplatWorker`.

## What to send back

`~/splatwork-gpucheck.txt`, `~/splatwork-gpucheck-train.txt`, `~/splatwork-benchmark.txt`, the outputs of step 2,
and the install logs if a step failed.
