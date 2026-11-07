"""What the setup assistant installs, and why.

Why a separate Python
---------------------
gsplat compiles CUDA kernels. Its prebuilt wheels (https://docs.gsplat.studio/whl/)
exist only for CPython 3.10, PyTorch 2.0-2.4 and CUDA 11.8 / 12.1 / 12.4. Without a
prebuilt wheel gsplat JIT-compiles on first use, which needs ``nvcc`` and a C++
compiler: neither exists inside the GNOME Flatpak runtime (whose Python is 3.13/3.14).
So the assistant downloads a relocatable CPython 3.10 build (python-build-standalone,
SHA-256 pinned), creates ``$XDG_DATA_HOME/splatworker/venv`` from it, and installs a
PyTorch + gsplat pair for which a prebuilt wheel exists. The worker
(``python -m splatworker serve``) then runs in that venv as a child process of the GUI.
The worker's own code is not copied into the venv: it is put on ``PYTHONPATH`` from the
read-only app directory, so app updates take effect without reinstalling the runtime.

The driver -> wheel mapping mirrors ``install.sh``: pick the newest CUDA build that is
not newer than the CUDA version the driver supports (``nvidia-smi`` "CUDA Version").

Limitation: PyTorch 2.4 has no kernels for Blackwell GPUs (RTX 50xx, compute capability
10.x/12.x). Those need PyTorch >= 2.7 / CUDA 12.8, for which no prebuilt gsplat wheel
exists, so the Flatpak cannot train on them; use the native ``install.sh`` (which can
JIT-compile with the system CUDA toolkit). Update :data:`GPU_PROFILES` when newer
prebuilt gsplat wheels appear.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .nvidia import CudaVersion, DriverInfo
from .paths import AppPaths

# ------------------------------------------------------------------ pins

@dataclass(frozen=True)
class PythonRuntime:
    version: str
    url: str
    sha256: str
    size_bytes: int


PYTHON_RUNTIME = PythonRuntime(
    version="3.10.22",
    url=("https://github.com/astral-sh/python-build-standalone/releases/download/20261009/"
         "cpython-3.10.22%2B20261009-x86_64-unknown-linux-gnu-install_only.tar.gz"),
    sha256="1780de9ddda04e956d2a4a0ad09b7f5667d92f4f12cd0881ec939de5d542903f",
    size_bytes=43778272,
)

# Mirrors pyproject.toml [project.dependencies] + the non-torch part of [train]
# (tests/test_gui_runtime_plan.py checks that they stay in sync).
WORKER_DEPS = ["fastapi>=0.110", "uvicorn[standard]>=0.27", "zeroconf>=0.131", "numpy>=1.24",
               "pillow>=10.0"]
TRAIN_EXTRA_DEPS = ["scipy>=1.10", "nvidia-ml-py>=12.0"]
SPLATPREP_CORE_DEPS = ["opencv-python-headless>=4.7"]
# LightGlue family (splatprep extra "lightglue"); torchvision comes from the torch index.
LIGHTGLUE_DEPS = ["kornia>=0.7",
                  "lightglue @ https://github.com/cvg/LightGlue/archive/refs/heads/main.zip"]

# Everything else the Mac can send here (splatprep extras sfm / masks / sam2 / loftr): the COLMAP mapper as
# a Python package, OWLv2 + SegFormer (transformers), SAM 2 (pure-Python install: its optional CUDA
# extension is skipped when no toolkit is present). Model weights download on first use.
# transformers < 4.50: newer releases need Python ≥ 3.11 / PyTorch ≥ 2.6 for their vision models, while this
# runtime is Python 3.10 + PyTorch 2.4 (pinned for gsplat). OWLv2 has been in transformers since 4.35.
PROCESSING_DEPS = ["pycolmap>=3.10", "transformers>=4.38,<4.50", "huggingface_hub>=0.20", "safetensors",
                   # SAM 2's own runtime deps (SAM 2 itself is installed with --no-deps, see below)
                   "hydra-core>=1.3.2", "iopath>=0.1.10", "tqdm"]
# Installed with --no-deps: SAM 2 declares torch>=2.5.1, and letting pip resolve that silently replaced the
# runtime's torch 2.4.1 (+cu124) with the newest PyPI build, which gsplat's prebuilt CUDA kernels can't load
# ("'NoneType' object has no attribute 'CameraModelType'"). SAM 2 runs fine on torch 2.4.
SAM2_REQ = "SAM-2 @ https://github.com/facebookresearch/sam2/archive/refs/heads/main.zip"


def torch_pins(gpu: "GpuProfile | None") -> List[str]:
    """Requirements that keep torch / torchvision at the profile's build in every later pip call."""
    return [f"torch=={gpu.torch}", f"torchvision=={gpu.torchvision}"] if gpu is not None else []


PROCESSING_DOWNLOAD_MB = 160
PROCESSING_DISK_MB = 600

PYTORCH_INDEX = "https://download.pytorch.org/whl/"
GSPLAT_INDEX = "https://docs.gsplat.studio/whl/"
PYPI_INDEX = "https://pypi.org/simple"


@dataclass(frozen=True)
class GpuProfile:
    id: str                      # "cu124"
    label: str
    min_driver_cuda: CudaVersion
    torch: str
    torchvision: str
    gsplat: str
    gsplat_index: str            # sub-index of GSPLAT_INDEX, e.g. "pt24cu124"
    download_mb: int             # torch + nvidia-* wheels + gsplat, approximate
    disk_mb: int                 # installed size, approximate
    max_compute_capability: Tuple[int, int] = (9, 0)
    min_compute_capability: Tuple[int, int] = (6, 0)

    @property
    def torch_index(self) -> str:
        return PYTORCH_INDEX + self.id

    @property
    def gsplat_requirement(self) -> str:
        return f"gsplat=={self.gsplat}+{self.gsplat_index}"


GPU_PROFILES: List[GpuProfile] = [  # newest first
    GpuProfile("cu124", "PyTorch 2.4 + CUDA 12.4", (12, 4), "2.4.1", "0.19.1", "1.5.3", "pt24cu124",
               download_mb=2700, disk_mb=5600),
    GpuProfile("cu121", "PyTorch 2.4 + CUDA 12.1", (12, 1), "2.4.1", "0.19.1", "1.5.3", "pt24cu121",
               download_mb=2600, disk_mb=5400),
    GpuProfile("cu118", "PyTorch 2.4 + CUDA 11.8", (11, 8), "2.4.1", "0.19.1", "1.5.3", "pt24cu118",
               download_mb=2500, disk_mb=5200),
]
CPU_PROFILE_ID = "cpu"
PROFILE_IDS = [p.id for p in GPU_PROFILES] + [CPU_PROFILE_ID]

BASE_DOWNLOAD_MB = 45 + 60     # python + worker deps (numpy, scipy, pillow, fastapi ...)
BASE_DISK_MB = 170 + 260
SPLATPREP_DOWNLOAD_MB = 55     # opencv-python-headless
SPLATPREP_DISK_MB = 150
LIGHTGLUE_DOWNLOAD_MB = 40     # torchvision + kornia + LightGlue (torch already counted)
LIGHTGLUE_DISK_MB = 90
CPU_TORCH_DOWNLOAD_MB = 200    # torch CPU build for LightGlue in the processing-only profile
CPU_TORCH_DISK_MB = 750


def profile_by_id(pid: str) -> Optional[GpuProfile]:
    return next((p for p in GPU_PROFILES if p.id == pid), None)


# ------------------------------------------------------------- choosing

@dataclass
class ProfileChoice:
    profile_id: str                  # a GpuProfile id, or "cpu"
    can_train: bool
    summary: str                     # one sentence for the assistant
    warnings: List[str] = field(default_factory=list)
    blocking: Optional[str] = None   # GPU present but unsupported: why (still offers "cpu")


def choose_profile(driver: DriverInfo, override: Optional[str] = None) -> ProfileChoice:
    if override is not None and override not in PROFILE_IDS:
        raise ValueError(f"unknown runtime profile {override!r} (choose from {PROFILE_IDS})")
    if not driver.has_gpu:
        problem = driver.problem()
        return ProfileChoice(CPU_PROFILE_ID, False,
                             "Processing-only runtime: no usable NVIDIA GPU was found, so training "
                             "will not be available.",
                             blocking=problem[1] if problem else None)
    if override == CPU_PROFILE_ID:
        return ProfileChoice(CPU_PROFILE_ID, False, "Processing-only runtime (chosen manually).")

    warnings: List[str] = []
    caps = [g.compute_capability for g in driver.gpus if g.compute_capability]
    names = ", ".join(g.name for g in driver.gpus)
    too_new = [g for g in driver.gpus if g.compute_capability and g.compute_capability > GPU_PROFILES[0].max_compute_capability]
    if too_new and len(too_new) == len(driver.gpus):
        return ProfileChoice(
            CPU_PROFILE_ID, False,
            "Processing-only runtime: your GPU is newer than the PyTorch build the Flatpak can use.",
            blocking=(f"{too_new[0].name} (compute capability {too_new[0].compute_capability[0]}."
                      f"{too_new[0].compute_capability[1]}) needs PyTorch 2.7 or newer, for which no "
                      "prebuilt gsplat exists. Training on this GPU is not possible inside the Flatpak "
                      "yet. Install Splat Worker natively with linux/install.sh instead (it can compile "
                      "gsplat with the CUDA toolkit)."))
    if too_new:
        warnings.append("Some GPUs (" + ", ".join(g.name for g in too_new) + ") are too new for this "
                        "PyTorch build and will fail to train; deselect them in Settings → GPUs.")
    too_old = [c for c in caps if c < GPU_PROFILES[0].min_compute_capability]
    if too_old:
        warnings.append("Your GPU is quite old (compute capability below 6.0); training may not work.")
    small = [g for g in driver.gpus if 0 < g.memory_total_mb < 7500]
    if small:
        warnings.append(f"{small[0].name} has {small[0].memory_total_mb // 1024} GB of video memory. "
                        "8 GB or more is recommended; lower \"max Gaussians\" for large scenes.")

    if override is not None:
        prof = profile_by_id(override)
        if driver.cuda_version and driver.cuda_version < prof.min_driver_cuda:
            warnings.append(f"{prof.label} needs a driver supporting CUDA {prof.min_driver_cuda[0]}."
                            f"{prof.min_driver_cuda[1]}; yours supports "
                            f"{driver.cuda_version[0]}.{driver.cuda_version[1]}.")
        return ProfileChoice(prof.id, True, f"{prof.label} for {names} (chosen manually).", warnings)

    if driver.cuda_version is None:
        prof = GPU_PROFILES[0]
        warnings.append("Could not read the CUDA version supported by your driver; assuming "
                        f"{prof.min_driver_cuda[0]}.{prof.min_driver_cuda[1]}. If training fails with "
                        "\"CUDA driver version is insufficient\", update the NVIDIA driver.")
        return ProfileChoice(prof.id, True, f"{prof.label} for {names}.", warnings)
    for prof in GPU_PROFILES:
        if driver.cuda_version >= prof.min_driver_cuda:
            return ProfileChoice(prof.id, True, f"{prof.label} for {names} (driver "
                                 f"{driver.driver_version or '?'}, CUDA {driver.cuda_version[0]}."
                                 f"{driver.cuda_version[1]}).", warnings)
    oldest = GPU_PROFILES[-1].min_driver_cuda
    return ProfileChoice(
        CPU_PROFILE_ID, False, "Processing-only runtime: the NVIDIA driver is too old for training.",
        blocking=(f"Your driver ({driver.driver_version or '?'}) supports CUDA {driver.cuda_version[0]}."
                  f"{driver.cuda_version[1]}, but training needs CUDA {oldest[0]}.{oldest[1]} or newer "
                  "(driver 520 or newer). Update the NVIDIA driver with your distribution's tools, "
                  "reboot, run \"flatpak update\", then run this setup again."))


# ----------------------------------------------------------------- plan

@dataclass
class Step:
    id: str
    title: str                  # short, shown in the progress view
    description: str            # one sentence, shown in the plan list
    kind: str                   # "download" | "command" | "verify"
    argv: List[str] = field(default_factory=list)
    download_mb: int = 0
    weight: float = 1.0         # share of the overall progress bar
    url: Optional[str] = None
    sha256: Optional[str] = None
    size_bytes: Optional[int] = None
    parser: Optional[str] = None  # "pip": parse pip's output for the progress bar


@dataclass
class Plan:
    profile_id: str
    can_train: bool
    steps: List[Step]
    download_mb: int
    disk_mb: int
    with_splatprep: bool
    with_lightglue: bool
    splatprep_dir: Optional[Path]

    def free_space_problem(self, path: Path, margin_mb: int = 1024) -> Optional[str]:
        """Message if the filesystem holding ``path`` lacks room, else None."""
        p = Path(path)
        while not p.exists() and p != p.parent:
            p = p.parent
        try:
            free_mb = shutil.disk_usage(p).free // (1024 * 1024)
        except OSError:
            return None
        need = self.disk_mb + margin_mb
        if free_mb < need:
            return (f"Not enough disk space: the GPU runtime needs about {need / 1024:.1f} GB but only "
                    f"{free_mb / 1024:.1f} GB is free in {p}. Free some space and try again.")
        return None


def pip_argv(python: Path, paths: AppPaths, *requirements: str, index_url: Optional[str] = None,
             extra_index_url: Optional[str] = None, extra: Sequence[str] = ()) -> List[str]:
    argv = [str(python), "-m", "pip", "install", "--progress-bar", "raw",
            "--cache-dir", str(paths.pip_cache), "--prefer-binary"]
    if index_url:
        argv += ["--index-url", index_url]
    if extra_index_url:
        argv += ["--extra-index-url", extra_index_url]
    argv += list(extra)
    argv += list(requirements)
    return argv


def build_plan(choice_or_id, paths: AppPaths, *, with_splatprep: bool = True,
               with_lightglue: bool = False, splatprep_dir: Optional[Path] = None) -> Plan:
    profile_id = choice_or_id.profile_id if isinstance(choice_or_id, ProfileChoice) else choice_or_id
    if profile_id not in PROFILE_IDS:
        raise ValueError(f"unknown runtime profile {profile_id!r}")
    if with_splatprep and splatprep_dir is None:
        with_splatprep = False
    with_lightglue = with_lightglue and with_splatprep
    gpu = profile_by_id(profile_id)
    py = paths.venv_python
    rt = PYTHON_RUNTIME

    steps: List[Step] = [
        Step("python", f"Downloading Python {rt.version}",
             f"A private copy of Python {rt.version} ({rt.size_bytes / 1e6:.0f} MB). gsplat's prebuilt "
             "CUDA kernels only exist for this Python version.",
             "download", url=rt.url, sha256=rt.sha256, size_bytes=rt.size_bytes,
             download_mb=rt.size_bytes // 1_000_000, weight=3),
        Step("venv", "Creating the runtime environment",
             f"A virtual environment in {paths.venv_dir}.", "command",
             argv=[str(paths.python_exe), "-m", "venv", "--clear", str(paths.venv_dir)], weight=1),
        Step("pip", "Updating the package installer", "The latest pip.", "command",
             argv=[str(py), "-m", "pip", "install", "--upgrade", "--cache-dir", str(paths.pip_cache),
                   "pip>=24.1", "wheel"], download_mb=2, weight=1, parser="pip"),
    ]
    download = rt.size_bytes // 1_000_000 + 2
    disk = BASE_DISK_MB
    if gpu is not None:
        steps.append(Step(
            "torch", f"Downloading PyTorch {gpu.torch} (CUDA {gpu.id[2:4]}.{gpu.id[4:]})",
            f"PyTorch {gpu.torch} built for CUDA {gpu.id[2:4]}.{gpu.id[4:]}, including NVIDIA's CUDA "
            f"libraries (about {gpu.download_mb / 1000:.1f} GB). This is by far the largest download.",
            "command", argv=pip_argv(py, paths, f"torch=={gpu.torch}", index_url=gpu.torch_index),
            download_mb=gpu.download_mb - 10, weight=60, parser="pip"))
        steps.append(Step(
            "gsplat", f"Installing gsplat {gpu.gsplat}",
            "gsplat, the CUDA Gaussian-splatting rasteriser, with prebuilt kernels (no compiler needed).",
            "command", argv=pip_argv(py, paths, gpu.gsplat_requirement,
                                     index_url=GSPLAT_INDEX + gpu.gsplat_index + "/",
                                     extra_index_url=PYPI_INDEX, extra=["--only-binary", "gsplat"]),
            download_mb=10, weight=3, parser="pip"))
        download += gpu.download_mb
        disk += gpu.disk_mb
    steps.append(Step(
        "worker", "Installing the worker's libraries",
        "Web server, Bonjour, NumPy, SciPy and Pillow used by the worker itself.", "command",
        argv=pip_argv(py, paths, *WORKER_DEPS, *TRAIN_EXTRA_DEPS), download_mb=60, weight=8,
        parser="pip"))
    download += 60
    if with_splatprep:
        steps.append(Step(
            "splatprep", "Installing dataset processing (splatprep)",
            "OpenCV for splatprep, which lets the Mac send feature matching / SfM / masking jobs here.",
            "command", argv=pip_argv(py, paths, *SPLATPREP_CORE_DEPS), download_mb=SPLATPREP_DOWNLOAD_MB,
            weight=4, parser="pip"))
        download += SPLATPREP_DOWNLOAD_MB
        disk += SPLATPREP_DISK_MB
    if with_lightglue:
        if gpu is not None:
            tv = pip_argv(py, paths, f"torchvision=={gpu.torchvision}", f"torch=={gpu.torch}",
                          index_url=gpu.torch_index)
            extra_dl, extra_disk = LIGHTGLUE_DOWNLOAD_MB, LIGHTGLUE_DISK_MB
        else:
            tv = pip_argv(py, paths, "torch", "torchvision", index_url=PYTORCH_INDEX + "cpu")
            extra_dl = LIGHTGLUE_DOWNLOAD_MB + CPU_TORCH_DOWNLOAD_MB
            extra_disk = LIGHTGLUE_DISK_MB + CPU_TORCH_DISK_MB
        steps.append(Step("torchvision", "Installing torchvision",
                          "torchvision, needed by the learned feature matchers.", "command",
                          argv=tv, download_mb=extra_dl - 20, weight=6, parser="pip"))
        pins = torch_pins(gpu)
        xidx = gpu.torch_index if gpu is not None else None
        steps.append(Step("lightglue", "Installing LightGlue",
                          "SuperPoint / DISK / ALIKED + LightGlue learned feature matching.", "command",
                          argv=pip_argv(py, paths, *LIGHTGLUE_DEPS, *pins, extra_index_url=xidx),
                          download_mb=20, weight=3, parser="pip"))
        steps.append(Step("processing", "Installing camera solving and masking models",
                          "COLMAP mapper and OWLv2 object detection, so the Mac can send camera solving "
                          "and moving-object masking here.", "command",
                          argv=pip_argv(py, paths, *PROCESSING_DEPS, *pins, extra_index_url=xidx),
                          download_mb=PROCESSING_DOWNLOAD_MB, weight=5, parser="pip"))
        steps.append(Step("sam2", "Installing SAM 2",
                          "SAM 2 video segmentation (installed without its dependencies so it can't replace "
                          "the GPU runtime's PyTorch).", "command",
                          argv=pip_argv(py, paths, SAM2_REQ, extra=("--no-deps", "--no-build-isolation")),
                          download_mb=5, weight=1, parser="pip"))
        download += extra_dl + PROCESSING_DOWNLOAD_MB
        disk += extra_disk + PROCESSING_DISK_MB
    steps.append(Step("verify", "Checking the installation",
                      "Imports everything and asks PyTorch whether it can see the GPU.", "verify",
                      argv=[str(py), "-c", VERIFY_SCRIPT], weight=2))
    return Plan(profile_id=profile_id, can_train=gpu is not None, steps=steps, download_mb=download,
                disk_mb=disk, with_splatprep=with_splatprep, with_lightglue=with_lightglue,
                splatprep_dir=splatprep_dir if with_splatprep else None)


# Printed as one JSON line prefixed with "SPLATWORKER-VERIFY " (parsed by the installer).
VERIFY_SCRIPT = r"""
import importlib.util, json, sys
r = {"python": sys.version.split()[0]}
def has(m):
    try:
        return importlib.util.find_spec(m) is not None
    except Exception:
        return False
try:
    import torch
    r["torch"] = torch.__version__
    r["cudaAvailable"] = bool(torch.cuda.is_available())
    r["torchCuda"] = torch.version.cuda
    r["devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if r["cudaAvailable"] else []
    if r["cudaAvailable"]:
        torch.zeros(8, device="cuda").sum().item()
except ImportError:
    r["torch"] = None
except Exception as e:
    r["torchError"] = str(e)
try:
    from importlib.metadata import version
    r["gsplat"] = version("gsplat") if has("gsplat") else None
except Exception:
    r["gsplat"] = None
r["gsplatPrebuilt"] = has("gsplat.csrc")
r["splatworker"] = has("splatworker") and has("fastapi") and has("uvicorn")
r["splatprep"] = has("splatprep") and has("cv2")
r["lightglue"] = has("lightglue")
try:
    import torch as _t
    r["torchVersion"] = _t.__version__
except Exception:
    r["torchVersion"] = None
r["pycolmap"] = has("pycolmap")
try:  # installed is not enough: a torchvision built for another torch fails to import
    import torchvision as _tv  # noqa: F401
    r["torchvisionImports"] = True
except Exception as _e:
    r["torchvisionImports"] = False
    r["torchvisionError"] = str(_e)[:200]
r["sam2"] = has("sam2")
r["transformers"] = has("transformers")
print("SPLATWORKER-VERIFY " + json.dumps(r), flush=True)
"""


def verify_problems(result: dict, plan: Plan) -> List[str]:
    """Human readable problems in a VERIFY_SCRIPT result (empty list = all good)."""
    problems = []
    if not result.get("splatworker"):
        problems.append("The worker's libraries (FastAPI / uvicorn) could not be imported.")
    if plan.can_train:
        prof = next((g for g in GPU_PROFILES if g.id == plan.profile_id), None)
        tv = str(result.get("torchVersion") or "")
        if prof is not None and tv and tv.split("+")[0] != prof.torch:
            problems.append(f"PyTorch {tv} is installed but this runtime needs {prof.torch} "
                            f"(gsplat's GPU kernels are built for it). Run the setup again to repair it.")
        if plan.with_lightglue and result.get("torchvisionImports") is False:
            problems.append("torchvision is installed but can't be loaded (it doesn't match PyTorch): "
                            f"{result.get('torchvisionError', '')}")
        if not result.get("torch"):
            problems.append("PyTorch could not be imported.")
        elif result.get("torchError"):
            problems.append(f"PyTorch could not use the GPU: {result['torchError']}")
        elif not result.get("cudaAvailable"):
            problems.append("PyTorch was installed but cannot see the GPU (CUDA is not available). "
                            "Usually the Flatpak NVIDIA driver extension does not match the host driver: "
                            "run \"flatpak update\" and restart.")
        if not result.get("gsplat"):
            problems.append("gsplat could not be imported.")
        elif not result.get("gsplatPrebuilt"):
            problems.append("gsplat was installed without prebuilt CUDA kernels; it would need a compiler.")
    if plan.with_splatprep and not result.get("splatprep"):
        problems.append("splatprep (dataset processing) could not be imported.")
    if plan.with_lightglue and not result.get("lightglue"):
        problems.append("LightGlue could not be imported.")
    return problems


def child_pythonpath(lib_dir: Path, splatprep_dir: Optional[Path]) -> str:
    parts = [str(lib_dir)]
    if splatprep_dir is not None:
        parts.append(str(splatprep_dir))
    return ":".join(parts)
