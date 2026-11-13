# GPU support in SplatWork for Linux: AMD and Intel

*Developer notes for the roadmap item "support AMD and Intel GPUs on Linux". Status: **detection, setup and
GPU-accelerated processing done in 1.6.0** (untested on real AMD / Intel hardware); **training is still NVIDIA-only**
— the training path below needs a decision. This page maps where NVIDIA / CUDA is assumed and the options.*

## Done in SplatWork for Linux 1.6.0 (step 1, plus parts of 3 and 4)

| What | Where | Tested how |
|---|---|---|
| Every graphics card from sysfs: vendor, driver, name (`pci.ids`, amdgpu `product_name`), amdgpu VRAM / load / temperature, Arc VRAM on the `xe` driver | `linux/splatworker/gpu_vendors.py` | fake sysfs trees (`linux/tests/test_gpu_vendors.py`) |
| `/v1/info` `"adapters"` (all cards) and `"vendor"` per job GPU; ROCm PyTorch's `torch.cuda` GPUs reported as AMD; `splatworker info` shows both | `gpu.py`, `server.py`, `cli.py` | fake sysfs + FastAPI test client |
| Setup wording: an AMD / Intel card without NVIDIA gets "<card> found — training needs NVIDIA for now", not "No NVIDIA GPU"; NVIDIA wording unchanged (also with an extra AMD / Intel card) | `gui/nvidia.py` `DriverInfo.problem` / `other_gpu` | unit tests |
| Processing runtimes with GPU PyTorch: **`rocm6.4`** (PyTorch 2.8 + ROCm 6.4, chosen for a dedicated AMD card when `/dev/kfd` is visible) and **`xpu`** (PyTorch 2.7.1 for Intel, chosen for an Arc card); integrated GPUs stay on the CPU build; both can be picked by hand | `gui/runtime_plan.py` `PROCESSING_GPU_PROFILES`, `setup_assistant.py` | plan / verify unit tests; the wheels exist on download.pytorch.org (checked 2026-11-13) but were never installed on AMD / Intel |
| Status page GPU statistics from sysfs when there is no NVML (AMD) | `gui/nvidia.py` `GpuMonitor` | fake sysfs |
| splatprep: `torch.xpu`, `gpu_backend()` tells CUDA from ROCm; COLMAP's CUDA SIFT only on NVIDIA; SAM 2 autocast on XPU | `tools/splatprep/device.py`, `backends/colmap_sift.py`, `masks/track.py` | fake torch modules (`tools/tests/test_device.py`) |
| AMD Radeon / Instinct and Intel Arc names in the speed tables (guesses; a benchmark replaces them) | `ComputeScheduler.estimateThroughput`, `studio/devices.py` | unit tests |

Why ROCm 6.4 and not 6.3: 6.4 is the first PyTorch ROCm build with Radeon RX 9000 (RDNA 4, gfx1200/1201) kernels;
PyTorch 2.8 is the oldest build on it. AMD itself recommends its own wheels at repo.radeon.com over PyTorch's
(less tested by AMD); PyTorch's index was chosen because pip can install from it without extra configuration.

Untested and worth checking first on real hardware: that the ROCm build sees the card inside the Flatpak
(`--device=all` exposes `/dev/kfd` and `/dev/dri`; RDNA 2 cards may need `HSA_OVERRIDE_GFX_VERSION=10.3.0`), that
LightGlue / SAM 2 / OWLv2 run on ROCm and XPU, and that XPU finds the card with the Flatpak runtime's Level Zero
(it may need the host's `intel-level-zero` libraries).

## What already works on any GPU

- **The viewer** renders with OpenGL 4.3 (`linux/splatworker/studio/gl.py`, `renderer.py`): NVIDIA, AMD and Intel via
  Mesa or the vendor driver. Editing, export, turntables and Coverage don't need CUDA.
- **Processing without a GPU:** the setup assistant offers a processing-only runtime; splatprep runs feature
  matching, COLMAP / GLOMAP and masks on the CPU (slowly). `tools/splatprep/device.py` picks CUDA > MPS > CPU
  (`SPLATPREP_DEVICE` overrides).
- **The Mac** trains with its own Metal trainer, and an AMD/Intel Linux PC can still serve a Mac as a processing
  worker.

## Where NVIDIA / CUDA is assumed

| Place | What it assumes |
|---|---|
| `linux/splatworker/gpu.py` | GPU discovery through NVML (`pynvml`), falling back to `torch.cuda`; `training_available()` = torch + gsplat importable |
| `linux/splatworker/gui/nvidia.py` | NVIDIA driver detection, `/proc/driver/nvidia/version`, the Flatpak `GL.nvidia-<version>` extension, the "driver libraries missing in the sandbox" message |
| `linux/splatworker/gui/runtime_plan.py` | `GPU_PROFILES`: PyTorch CUDA wheels (cu128 / cu124 / cu121 / cu118) + gsplat wheels per CUDA tag, chosen from the driver's CUDA version and the GPU's compute capability; the CPU profile otherwise |
| `linux/splatworker/gui/installer.py`, `runtime_install.py`, `gui/gtk/setup_assistant.py` | installing that runtime (Python 3.10, torch, gsplat, fused-ssim, splatprep) and the setup wording |
| `linux/splatworker/trainer.py` | training: gsplat `rasterization`, `MCMCStrategy` / `DefaultStrategy`, `fused_ssim` (CUDA kernels), `torch.device("cuda:N")` |
| `linux/splatworker/census.py`, `depthprior.py` | Auto clean census and predicted depth on the CUDA device |
| `linux/flatpak/com.example.SplatWorker.yml` | `--device=all` (comment says /dev/nvidia*; it also exposes AMD's `/dev/kfd` and `/dev/dri`), `CUDA_DEVICE_ORDER=PCI_BUS_ID` |
| `linux/flatpak/build-gsplat-wheel.sh`, `.github/workflows/splat-worker-flatpak.yml` | building / caching the CUDA gsplat wheel |
| `mac/.../ComputeScheduler.swift` `estimateThroughput` | a GPU-name speed table (NVIDIA and Apple names) used before a device is benchmarked |

## The hard part: training kernels

Detection, setup and wording are routine. Training is not: the trainer's speed comes from CUDA kernels in gsplat and
fused-ssim.

- **AMD.** PyTorch publishes ROCm builds for Linux, and they keep the `torch.cuda` API, so most of the Python code
  would run unchanged. Open question: can gsplat and fused-ssim be built for ROCm (HIP) with their current versions,
  and how fast and correct are they? That needs checking against upstream and on real hardware. ROCm also supports
  a narrower set of GPUs than Mesa does.
- **Intel.** PyTorch has an `xpu` backend (Arc and recent integrated GPUs), but gsplat has no Intel kernels. Training
  there would need a different rasterizer.
- **A portable trainer.** The Mac already has a complete trainer of its own (Metal kernels in
  `mac/Sources/SplatStudio/Shaders/TrainShaders.swift`, loop in `Train/MetalTrainer.swift`, gradient-checked). A
  port of that design to Vulkan compute (or a shading language that targets Vulkan, Metal and others) would train on
  NVIDIA, AMD and Intel alike, and would also serve the Windows app on the roadmap. It is the largest option, but
  the only one that covers every vendor with one code path.

Whatever is chosen, the trainer's settings stay the same `TrainingConfig` (docs/PROTOCOL.md §3), so the Mac, other
workers and results are unaffected.

## Training path: findings (2026-11-13)

Checked against upstream, not on hardware.

- **gsplat on ROCm exists, but only for data-centre GPUs.** AMD maintains a ROCm port (`AMD-Ecosystem/gsplat`,
  documented as part of the ROCm Simulation toolkit): gsplat 1.5.3 (our CUDA 12.4 profile's version; our CUDA 12.8
  profile uses a 1.6 development build), PyTorch 2.6 / 2.8, ROCm 6.4.3 / 7.0, Python 3.10 / 3.12, officially only
  **Instinct MI300X / MI325X**, installed from a hash-pinned requirements file. Others report building it for Radeon
  and Ryzen AI iGPUs with patches, unsupported. Upstream `nerfstudio-project/gsplat` has no HIP build.
- **fused-ssim** documents ROCm (HIP, built with `hipcc`) and Intel (SYCL / oneAPI) backends in its README, built
  from source. Our trainer already falls back to a plain-PyTorch SSIM when `fused_ssim` is missing
  (`trainer.py` `_ssim_fn`, slower), so SSIM is not the blocker.
- **Intel:** no gsplat port and no other rasterizer for `torch.xpu`. Training on Intel needs a portable trainer.
- **Portable trainers that exist:** *Brush* (Arthur Brussee, Apache-2.0, Rust + wgpu + Burn) trains Gaussian
  splats on NVIDIA, AMD and Intel through Vulkan / Metal / DX12 / WebGPU. Its feature set and file conventions
  differ from our `TrainingConfig` (MCMC + our densification, masks, skybox, bilateral grid, appearance, pose
  refinement, depth / normal priors, block regions), so it could be a reference or a fallback trainer, not a
  drop-in. Our own Mac trainer (`Train/MetalTrainer.swift` + `Shaders/TrainShaders.swift`, gradient-checked) is the
  design to port if we write one.

### Options

| Option | Covers | Effort | Risks |
|---|---|---|---|
| A. ROCm spike: build AMD's gsplat port (or upstream gsplat + HIP) and fused-ssim for RDNA 3 / 4, add a `rocm` *training* profile with our own wheels from CI | AMD Radeon RX 7000 / 9000, Instinct | small to medium: CI matrix per gfx target, one trainer code path (ROCm keeps `torch.cuda`) | consumer GPUs unofficial for AMD's port; port lags upstream (1.5.3 vs our 1.6 dev); needs an AMD machine to validate |
| B. Port the Mac's Metal trainer to Vulkan compute (or Slang targeting Vulkan + Metal) | NVIDIA, AMD, Intel; also the Windows app | large: kernels, optimizer, densification, all `TrainingConfig` features, gradient checks | months of work; two trainers to keep in step with the CUDA one until it replaces it |
| C. Wrap Brush as an external trainer for AMD / Intel | all vendors quickly | medium: process wrapper, format and setting mapping | features and results differ from the Mac / CUDA trainer; a second project's release cadence |

### Recommendation

1. Do **A** first, as a time-boxed spike, once an AMD Radeon RX 7000 / 9000 machine (or a cloud MI300X for the
   official path) is available: if AMD's port builds and trains our standard benchmark scene correctly, ship a ROCm
   training profile. It reuses `trainer.py` unchanged.
2. Decide on **B** separately, together with the Windows app: it is the only option that covers Intel and Windows
   with our own features, and it is a large project that needs the user's go-ahead.
3. Keep **C** as a reference for B (speed, kernel structure), not as a shipped path.

Nothing here should be presented as "training works on AMD / Intel" until a real card has trained the benchmark
scene.

## Suggested order

1. ✅ (1.6.0) **Detect AMD and Intel GPUs** (sysfs `/sys/class/drm/card*/device/vendor` for 0x1002 / 0x8086, VRAM from
   `mem_info_vram_total` on amdgpu), report them in `/v1/info`, and set the setup assistant's wording by vendor. Until
   training is available, offer the processing runtime on them, with GPU-accelerated splatprep where PyTorch supports
   the device (ROCm `torch.cuda`, Intel `torch.xpu`; extend `tools/splatprep/device.py`).
2. **Decide the training path** after a spike on real hardware: gsplat on ROCm vs. a portable (Vulkan) trainer.
3. 🟡 (processing profiles in 1.6.0; a training profile waits for step 2) **Runtime profiles per vendor** in `runtime_plan.py` (e.g. a ROCm profile with PyTorch ROCm wheels), and CI for the
   matching wheels.
4. ✅ (1.6.0, guesses) **Benchmark and speed table:** add AMD / Intel names to `estimateThroughput`; the Compute → Benchmark result
   replaces the guess once measured.

## Testing

The project's test PC has an NVIDIA RTX 4080 only. CI and the Mac can test detection and setup logic with fakes
(as `linux/tests` does for NVIDIA), and the GTK pages in the Mesa container, but training on AMD or Intel needs real
hardware; plan for a machine (or a cloud instance with an AMD Instinct / Intel GPU) before choosing the training path.
