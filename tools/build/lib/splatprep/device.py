"""Compute device selection: CUDA > MPS > CPU (override with SPLATPREP_DEVICE)."""

from __future__ import annotations

import importlib.util
import os
import resource
import sys

# Ops without an MPS kernel (e.g. torchvision deform_conv2d used by ALIKED) fall back to CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def select_device(preferred: str | None = None) -> str:
    """Return "cuda", "mps" or "cpu".  Never imports torch if it is absent."""
    pref = (preferred or os.environ.get("SPLATPREP_DEVICE") or "auto").lower()
    if pref in ("cpu",):
        return "cpu"
    if not has_module("torch"):
        return "cpu"
    import torch

    if pref in ("auto", "cuda") and torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if pref in ("auto", "mps") and mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def torch_device(preferred: str | None = None):
    import torch

    return torch.device(select_device(preferred))


def peak_rss_mb() -> float:
    """Peak resident set size of this process and its waited-for children (MB)."""
    scale = 1.0 if sys.platform == "darwin" else 1024.0  # bytes on macOS, KiB on Linux
    own = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale
    kids = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * scale
    return round(max(own, kids) / (1024 * 1024), 1)


def peak_gpu_mb() -> float | None:
    if "torch" not in sys.modules:
        return None
    import torch

    try:
        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 2**20, 1)
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
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass
