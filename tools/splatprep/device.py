"""Compute device selection: CUDA > XPU > MPS > CPU (override with SPLATPREP_DEVICE).

"cuda" also covers AMD GPUs with a ROCm build of PyTorch (HIP answers through ``torch.cuda``); ``gpu_backend()``
tells NVIDIA ("cuda") and AMD ("rocm") apart where it matters (e.g. COLMAP's CUDA SIFT). "xpu" is an Intel GPU
with PyTorch's XPU build."""

from __future__ import annotations

import importlib.util
import os
try:
    import resource
except ImportError:  # Windows
    resource = None
import sys

# Ops without an MPS kernel (e.g. torchvision deform_conv2d used by ALIKED) fall back to CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def module_installed(name: str) -> bool:
    """Installed for this interpreter, ignoring the in-process pycolmap blocker (isolation.py).

    For modules that only ever run in a child interpreter (the pycolmap mapper and pose refinement): the
    LightGlue backends block pycolmap imports in their own process, which made ``has_module("pycolmap")``
    false and every LightGlue solve on a COLMAP-less worker fail after matching with "no SfM mapper".
    """
    import sys

    from .isolation import _Blocker

    saved = list(sys.meta_path)
    try:
        sys.meta_path[:] = [f for f in saved if not isinstance(f, _Blocker)]
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False
    finally:
        sys.meta_path[:] = saved


def select_device(preferred: str | None = None) -> str:
    """Return "cuda", "xpu", "mps" or "cpu".  Never imports torch if it is absent."""
    pref = (preferred or os.environ.get("SPLATPREP_DEVICE") or "auto").lower()
    if pref in ("cpu",):
        return "cpu"
    if not has_module("torch"):
        return "cpu"
    import torch

    if pref in ("auto", "cuda", "rocm") and torch.cuda.is_available():
        return "cuda"
    xpu = getattr(torch, "xpu", None)
    try:
        if pref in ("auto", "xpu") and xpu is not None and xpu.is_available():
            return "xpu"
    except Exception:
        pass
    mps = getattr(torch.backends, "mps", None)
    if pref in ("auto", "mps") and mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def gpu_backend(device: str | None = None) -> str:
    """"cuda" (NVIDIA), "rocm" (AMD via HIP), "xpu" (Intel), "mps" (Apple) or "cpu" for a ``select_device``
    result (or the automatic choice)."""
    device = device or select_device()
    if device == "cuda":
        import torch

        return "rocm" if getattr(torch.version, "hip", None) else "cuda"
    return device


def torch_device(preferred: str | None = None):
    import torch

    return torch.device(select_device(preferred))


def peak_rss_mb() -> float:
    """Peak resident set size of this process and its waited-for children (MB)."""
    if resource is None:
        return round(_windows_peak_working_set() / (1024 * 1024), 1)
    scale = 1.0 if sys.platform == "darwin" else 1024.0  # bytes on macOS, KiB on Linux
    own = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale
    kids = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * scale
    return round(max(own, kids) / (1024 * 1024), 1)


def _windows_peak_working_set() -> float:
    """PeakWorkingSetSize of this process (children aren't tracked on Windows); 0 if unavailable."""
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        if ctypes.windll.psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
            return float(pmc.PeakWorkingSetSize)
    except Exception:  # noqa: BLE001
        pass
    return 0.0


def peak_gpu_mb() -> float | None:
    if "torch" not in sys.modules:
        return None
    import torch

    try:
        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 2**20, 1)
        xpu = getattr(torch, "xpu", None)
        if xpu is not None and xpu.is_available() and hasattr(xpu, "max_memory_allocated"):
            return round(xpu.max_memory_allocated() / 2**20, 1)
        mps = getattr(torch, "mps", None)
        if mps is not None and hasattr(mps, "driver_allocated_memory") and torch.backends.mps.is_available():
            return round(mps.driver_allocated_memory() / 2**20, 1)  # current, not peak
    except Exception:
        return None
    return None


def empty_cache() -> None:
    if "torch" not in sys.modules:
        return
    import torch

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif getattr(torch, "xpu", None) is not None and torch.xpu.is_available():
            torch.xpu.empty_cache()
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass
