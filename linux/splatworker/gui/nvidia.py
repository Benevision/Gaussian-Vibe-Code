"""NVIDIA GPU / driver detection and live statistics.

Detection prefers NVML (``pynvml`` from the ``nvidia-ml-py`` package, which loads
``libnvidia-ml.so.1``) and falls back to parsing ``nvidia-smi``. Inside the Flatpak
both come from the ``org.freedesktop.Platform.GL.nvidia-<version>`` extension, which
must match the host kernel driver exactly; :func:`kernel_driver_version` reads
``/proc/driver/nvidia/version`` (visible in the sandbox) so the app can say *which*
extension is missing.

The NVML module is injectable so the whole module can be tested with fakes.
"""

from __future__ import annotations

import glob
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple
from .. import PRODUCT_NAME

CudaVersion = Tuple[int, int]


@dataclass
class GpuInfo:
    index: int
    name: str
    memory_total_mb: int = 0
    compute_capability: Optional[Tuple[int, int]] = None
    uuid: Optional[str] = None


@dataclass
class DriverInfo:
    driver_version: Optional[str] = None       # user-space driver (NVML / nvidia-smi)
    cuda_version: Optional[CudaVersion] = None  # highest CUDA the driver supports
    gpus: List[GpuInfo] = field(default_factory=list)
    source: Optional[str] = None                # "nvml" | "nvidia-smi" | None
    kernel_driver_version: Optional[str] = None  # /proc/driver/nvidia/version
    device_nodes: bool = False                  # /dev/nvidia0 visible
    in_flatpak: bool = False
    error: Optional[str] = None
    # every graphics card from sysfs (gpu_vendors.Adapter), NVIDIA or not, and AMD's compute device /dev/kfd
    adapters: list = field(default_factory=list)
    kfd: bool = False

    @property
    def has_gpu(self) -> bool:
        return bool(self.gpus)

    @property
    def other_gpu(self):
        """The AMD / Intel card to talk about when there is no usable NVIDIA one (None otherwise)."""
        from ..gpu_vendors import best_non_nvidia

        if self.has_gpu or self.kernel_driver_version or self.device_nodes \
                or any(a.vendor == "nvidia" for a in self.adapters):
            return None
        return best_non_nvidia(self.adapters)

    def problem(self) -> Optional[Tuple[str, str]]:
        """(title, explanation) for the user when no GPU could be used, else None."""
        if self.has_gpu:
            return None
        other = self.other_gpu
        if other is not None:
            how = {"amd": "PyTorch for ROCm", "intel": "PyTorch for Intel GPUs (XPU)"}[other.vendor]
            return (f"{other.name} found",
                    f"Training in {PRODUCT_NAME} needs an NVIDIA graphics card (CUDA) for now; support for "
                    f"{other.vendor_name} GPUs is being worked on. This computer can still prepare datasets: the "
                    "processing runtime runs feature matching, camera solving and masks, on this GPU through "
                    f"{how} where it can and on the CPU otherwise. The viewer and editing work on any GPU.")
        if not self.device_nodes and not self.kernel_driver_version:
            return ("No NVIDIA GPU found",
                    "This computer does not seem to have an NVIDIA graphics card with the NVIDIA "
                    "driver loaded. Training needs an NVIDIA GPU with CUDA; training on the CPU is "
                    "not supported. You can still install the processing-only runtime and use this "
                    "computer for dataset processing jobs (feature matching and SfM run on the CPU, "
                    "slowly).")
        if self.kernel_driver_version and self.in_flatpak:
            return ("NVIDIA driver libraries are missing in the sandbox",
                    f"Your system runs NVIDIA driver {self.kernel_driver_version}, but the matching "
                    f"Flatpak driver extension is not installed. Install it in a terminal with:\n\n"
                    f"{flatpak_gl_install_command(self.kernel_driver_version)}\n\n"
                    "then restart SplatWork. (\"flatpak update\" does not always add it.)")
        if self.kernel_driver_version:
            return ("The NVIDIA driver is loaded but not usable",
                    f"Driver {self.kernel_driver_version} is loaded, but its libraries could not be "
                    f"used ({self.error or 'unknown error'}). Reinstall the NVIDIA driver packages and "
                    "reboot.")
        return ("Cannot access the NVIDIA GPU",
                "An NVIDIA device exists but the driver did not answer"
                f"{': ' + self.error if self.error else ''}. Reboot after a driver update, or check "
                "that the nvidia kernel module is loaded.")


# ---------------------------------------------------------------- helpers

def parse_cuda_version(value: Any) -> Optional[CudaVersion]:
    """NVML int (12040), or a string like '12.4' -> (12, 4)."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        if value <= 0:
            return None
        return (value // 1000, (value % 1000) // 10)
    m = re.search(r"(\d+)\.(\d+)", str(value))
    return (int(m.group(1)), int(m.group(2))) if m else None


def parse_compute_cap(value: Any) -> Optional[Tuple[int, int]]:
    if isinstance(value, (tuple, list)) and len(value) == 2:
        try:
            return (int(value[0]), int(value[1]))
        except (TypeError, ValueError):
            return None
    if value is None:
        return None
    m = re.match(r"\s*(\d+)\.(\d+)\s*$", str(value))
    return (int(m.group(1)), int(m.group(2))) if m else None


def kernel_driver_version(proc_file: Path = Path("/proc/driver/nvidia/version")) -> Optional[str]:
    try:
        text = proc_file.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    m = re.search(r"Kernel Module(?: for [\w-]+)?\s+(\d+\.\d+(?:\.\d+)?)", text)
    if not m:
        m = re.search(r"NVRM version:.*?\b(\d{3,}\.\d+(?:\.\d+)?)\b", text)
    return m.group(1) if m else None


def flatpak_gl_extension_name(driver_version: str) -> str:
    """'550.54.14' -> 'org.freedesktop.Platform.GL.nvidia-550-54-14'."""
    return "org.freedesktop.Platform.GL.nvidia-" + driver_version.replace(".", "-")


def flatpak_gl_install_command(driver_version: str) -> str:
    """The command that installs the driver extension from Flathub (branch 1.4 holds all NVIDIA versions).

    No --user/--system: Flatpak picks the installation that has the flathub remote, and the app can use
    an extension from either installation.
    """
    return f"flatpak install flathub {flatpak_gl_extension_name(driver_version)}//1.4"


def _decode(v: Any) -> str:
    return v.decode("utf-8", "replace") if isinstance(v, bytes) else str(v)


def load_nvml():
    """Return the pynvml module or None."""
    try:
        import pynvml  # type: ignore
    except ImportError:
        return None
    return pynvml


# ------------------------------------------------------------------ NVML

def _detect_nvml(nvml) -> DriverInfo:
    nvml.nvmlInit()
    try:
        info = DriverInfo(source="nvml")
        try:
            info.driver_version = _decode(nvml.nvmlSystemGetDriverVersion())
        except Exception:
            pass
        for fn in ("nvmlSystemGetCudaDriverVersion_v2", "nvmlSystemGetCudaDriverVersion"):
            if hasattr(nvml, fn):
                try:
                    info.cuda_version = parse_cuda_version(getattr(nvml, fn)())
                    break
                except Exception:
                    continue
        for i in range(nvml.nvmlDeviceGetCount()):
            h = nvml.nvmlDeviceGetHandleByIndex(i)
            gpu = GpuInfo(index=i, name=_decode(nvml.nvmlDeviceGetName(h)))
            try:
                gpu.memory_total_mb = int(nvml.nvmlDeviceGetMemoryInfo(h).total // (1024 * 1024))
            except Exception:
                pass
            try:
                gpu.compute_capability = parse_compute_cap(nvml.nvmlDeviceGetCudaComputeCapability(h))
            except Exception:
                pass
            try:
                gpu.uuid = _decode(nvml.nvmlDeviceGetUUID(h))
            except Exception:
                pass
            info.gpus.append(gpu)
        return info
    finally:
        try:
            nvml.nvmlShutdown()
        except Exception:
            pass


# ------------------------------------------------------------ nvidia-smi

def find_nvidia_smi() -> Optional[str]:
    exe = shutil.which("nvidia-smi")
    if exe:
        return exe
    for pattern in ("/usr/lib/*/GL/*/bin/nvidia-smi", "/usr/lib/extensions/*/bin/nvidia-smi"):
        hits = sorted(glob.glob(pattern))
        if hits:
            return hits[0]
    return None


def parse_smi_query(text: str) -> List[GpuInfo]:
    """Parse ``--query-gpu=index,name,memory.total,compute_cap --format=csv,noheader,nounits``."""
    gpus = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3 or not parts[0].isdigit():
            continue
        try:
            mem = int(float(parts[2]))
        except ValueError:
            mem = 0
        cc = parse_compute_cap(parts[3]) if len(parts) > 3 else None
        gpus.append(GpuInfo(index=int(parts[0]), name=parts[1], memory_total_mb=mem, compute_capability=cc))
    return gpus


def parse_smi_header(text: str) -> Tuple[Optional[str], Optional[CudaVersion]]:
    drv = re.search(r"Driver Version:\s*([\d.]+)", text)
    cuda = re.search(r"CUDA Version:\s*([\d.]+)", text)
    return (drv.group(1) if drv else None, parse_cuda_version(cuda.group(1)) if cuda else None)


def _detect_smi(exe: str, run: Callable[..., Any]) -> DriverInfo:
    q = run([exe, "--query-gpu=index,name,memory.total,compute_cap", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20)
    if q.returncode != 0:
        # very old drivers do not know compute_cap
        q = run([exe, "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=20)
    if q.returncode != 0:
        raise RuntimeError((q.stderr or q.stdout or "nvidia-smi failed").strip().splitlines()[-1])
    head = run([exe], capture_output=True, text=True, timeout=20)
    info = DriverInfo(source="nvidia-smi", gpus=parse_smi_query(q.stdout))
    info.driver_version, info.cuda_version = parse_smi_header(head.stdout or "")
    return info


# ---------------------------------------------------------------- detect

def detect(nvml: Any = "auto", run: Callable[..., Any] = subprocess.run,
           smi: Optional[str] = "auto", proc_file: Path = Path("/proc/driver/nvidia/version"),
           dev_glob: str = "/dev/nvidia[0-9]*", flatpak: Optional[bool] = None,
           sys_root: Path = Path("/sys"), dev_root: Path = Path("/dev")) -> DriverInfo:
    """Detect the driver and GPUs. Never raises."""
    from .paths import in_flatpak

    if nvml == "auto":
        nvml = load_nvml()
    if smi == "auto":
        smi = find_nvidia_smi()
    errors = []
    info: Optional[DriverInfo] = None
    if nvml is not None:
        try:
            info = _detect_nvml(nvml)
            if not info.gpus:
                info = None
        except Exception as e:
            errors.append(f"NVML: {e}")
            info = None
    if info is None and smi:
        try:
            info = _detect_smi(smi, run)
            if not info.gpus:
                info = None
        except Exception as e:
            errors.append(f"nvidia-smi: {e}")
            info = None
    if info is None:
        info = DriverInfo()
    elif info.cuda_version is None and smi and info.source == "nvml":
        try:
            info.cuda_version = parse_smi_header(run([smi], capture_output=True, text=True, timeout=20).stdout)[1]
        except Exception:
            pass
    info.kernel_driver_version = kernel_driver_version(proc_file)
    info.device_nodes = bool(glob.glob(dev_glob))
    info.in_flatpak = in_flatpak() if flatpak is None else flatpak
    try:
        from ..gpu_vendors import kfd_available, scan

        info.adapters = scan(sys_root)
        info.kfd = kfd_available(dev_root)
    except Exception:
        info.adapters, info.kfd = [], False
    if errors and not info.gpus:
        info.error = "; ".join(errors)
    return info


# ----------------------------------------------------------- live stats

@dataclass
class GpuStats:
    index: int
    name: str
    memory_total_mb: int = 0
    memory_used_mb: int = 0
    temperature_c: Optional[int] = None
    utilization_pct: Optional[int] = None
    power_w: Optional[float] = None

    @property
    def memory_free_mb(self) -> int:
        return max(0, self.memory_total_mb - self.memory_used_mb)

    @property
    def memory_fraction(self) -> float:
        return (self.memory_used_mb / self.memory_total_mb) if self.memory_total_mb else 0.0


class GpuMonitor:
    """Keeps NVML initialised and samples every GPU. ``sample()`` never raises. Without NVML (no NVIDIA card),
    AMD and Intel cards are read from sysfs instead (amdgpu: memory, load, temperature)."""

    def __init__(self, nvml: Any = "auto", sys_root: Path = Path("/sys")):
        self.nvml = load_nvml() if nvml == "auto" else nvml
        self.sys_root = sys_root
        self._initialised = False
        self._sysfs = self.nvml is None and bool(self._sysfs_sample())
        self.error: Optional[str] = None if (self.nvml is not None or self._sysfs) else \
            "GPU statistics are unavailable (NVML / nvidia-ml-py not found)"

    @property
    def available(self) -> bool:
        return self.nvml is not None or self._sysfs

    def _sysfs_sample(self) -> List[GpuStats]:
        try:
            from ..gpu_vendors import scan

            out = []
            for i, a in enumerate(x for x in scan(self.sys_root) if x.vendor in ("amd", "intel")):
                out.append(GpuStats(index=i, name=a.name, memory_total_mb=a.memory_mb or 0,
                                    memory_used_mb=a.memory_used_mb or 0, temperature_c=a.temperature_c,
                                    utilization_pct=a.busy_pct))
            return out
        except Exception:
            return []

    def _ensure(self) -> bool:
        if self.nvml is None:
            return False
        if not self._initialised:
            try:
                self.nvml.nvmlInit()
                self._initialised = True
                self.error = None
            except Exception as e:
                self.error = f"Cannot read GPU statistics: {e}"
                return False
        return True

    def sample(self) -> List[GpuStats]:
        if self.nvml is None and self._sysfs:
            return self._sysfs_sample()
        if not self._ensure():
            return []
        nv = self.nvml
        out: List[GpuStats] = []
        try:
            count = nv.nvmlDeviceGetCount()
        except Exception as e:
            self.error = f"Cannot read GPU statistics: {e}"
            self.close()
            return []
        for i in range(count):
            try:
                h = nv.nvmlDeviceGetHandleByIndex(i)
                s = GpuStats(index=i, name=_decode(nv.nvmlDeviceGetName(h)))
            except Exception:
                continue
            try:
                mem = nv.nvmlDeviceGetMemoryInfo(h)
                s.memory_total_mb = int(mem.total // (1024 * 1024))
                s.memory_used_mb = int(mem.used // (1024 * 1024))
            except Exception:
                pass
            try:
                s.temperature_c = int(nv.nvmlDeviceGetTemperature(h, getattr(nv, "NVML_TEMPERATURE_GPU", 0)))
            except Exception:
                pass
            try:
                s.utilization_pct = int(nv.nvmlDeviceGetUtilizationRates(h).gpu)
            except Exception:
                pass
            try:
                s.power_w = round(nv.nvmlDeviceGetPowerUsage(h) / 1000.0, 1)
            except Exception:
                pass
            out.append(s)
        return out

    def close(self) -> None:
        if self._initialised:
            try:
                self.nvml.nvmlShutdown()
            except Exception:
                pass
            self._initialised = False
