# GPU support in SplatWork for Linux: AMD and Intel

*Developer notes for the roadmap item "support AMD and Intel GPUs on Linux". Status: not started (1.5.0 trains on
NVIDIA only). This page maps where NVIDIA / CUDA is assumed today and the options; decisions are still open.*

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

## Suggested order

1. **Detect AMD and Intel GPUs** (sysfs `/sys/class/drm/card*/device/vendor` for 0x1002 / 0x8086, VRAM from
   `mem_info_vram_total` on amdgpu), report them in `/v1/info`, and set the setup assistant's wording by vendor. Until
   training is available, offer the processing runtime on them, with GPU-accelerated splatprep where PyTorch supports
   the device (ROCm `torch.cuda`, Intel `torch.xpu`; extend `tools/splatprep/device.py`).
2. **Decide the training path** after a spike on real hardware: gsplat on ROCm vs. a portable (Vulkan) trainer.
3. **Runtime profiles per vendor** in `runtime_plan.py` (e.g. a ROCm profile with PyTorch ROCm wheels), and CI for the
   matching wheels.
4. **Benchmark and speed table:** add AMD / Intel names to `estimateThroughput`; the Compute → Benchmark result
   replaces the guess once measured.

## Testing

The project's test PC has an NVIDIA RTX 4080 only. CI and the Mac can test detection and setup logic with fakes
(as `linux/tests` does for NVIDIA), and the GTK pages in the Mesa container, but training on AMD or Intel needs real
hardware; plan for a machine (or a cloud instance with an AMD Instinct / Intel GPU) before choosing the training path.
