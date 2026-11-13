"""GPU discovery.

``list_gpus`` lists the GPUs jobs can run on (their indices are the job slots). It uses NVML (``nvidia-ml-py``) when available because it does not create a CUDA
context in the server process; otherwise falls back to ``torch.cuda``. Both
are optional: without them the worker reports no GPUs.

The CLI sets ``CUDA_DEVICE_ORDER=PCI_BUS_ID`` so that torch indices match
NVML / ``nvidia-smi`` indices. A ROCm build of PyTorch answers through the same ``torch.cuda`` API (AMD GPUs,
``"vendor": "amd"``); Intel GPUs through ``torch.xpu`` are not job slots (nothing trains there yet).

``adapters`` lists every graphics card of any vendor from sysfs (``gpu_vendors.py``), for display and for
the ``/v1/info`` "adapters" field.
"""

from __future__ import annotations

import importlib.util
from typing import Dict, List, Optional


def _nvml_gpus() -> Optional[List[Dict]]:
    try:
        import pynvml  # type: ignore
    except ImportError:
        return None
    try:
        pynvml.nvmlInit()
    except Exception:
        return None
    try:
        out = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode()
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            out.append({"index": i, "name": name, "vendor": "nvidia",
                        "memoryMB": int(mem.total // (1024 * 1024)),
                        "freeMemoryMB": int(mem.free // (1024 * 1024))})
        return out
    except Exception:
        return None
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def _torch_gpus() -> List[Dict]:
    if importlib.util.find_spec("torch") is None:
        return []
    try:
        import torch

        if not torch.cuda.is_available():
            return []
        vendor = "amd" if getattr(torch.version, "hip", None) else "nvidia"
        out = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            try:
                free, total = torch.cuda.mem_get_info(i)
            except Exception:
                free, total = props.total_memory, props.total_memory
            out.append({"index": i, "name": props.name, "vendor": vendor,
                        "memoryMB": int(total // (1024 * 1024)),
                        "freeMemoryMB": int(free // (1024 * 1024))})
        return out
    except Exception:
        return []


def list_gpus(indices: Optional[List[int]] = None) -> List[Dict]:
    gpus = _nvml_gpus()
    if gpus is None:
        gpus = _torch_gpus()
    if indices is not None:
        by_index = {g["index"]: g for g in gpus}
        gpus = [by_index[i] for i in indices if i in by_index]
    return gpus


def adapters(sys_root=None) -> List[Dict]:
    """Every graphics card (NVIDIA, AMD, Intel) as ``/v1/info`` "adapters" entries; [] when sysfs can't be read."""
    try:
        from pathlib import Path

        from .gpu_vendors import scan

        return [a.to_info() for a in scan(Path(sys_root) if sys_root else Path("/sys"))]
    except Exception:
        return []


def torch_backend() -> str:
    """"cuda" (NVIDIA), "rocm" (AMD through HIP), "xpu" (Intel) or "cpu" — what the installed PyTorch can use."""
    if importlib.util.find_spec("torch") is None:
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            return "rocm" if getattr(torch.version, "hip", None) else "cuda"
        xpu = getattr(torch, "xpu", None)
        if xpu is not None and xpu.is_available():
            return "xpu"
    except Exception:
        pass
    return "cpu"


def backend_description() -> str:
    try:
        from importlib.metadata import version

        return f"gsplat {version('gsplat')}"
    except Exception:
        return "gsplat (not installed)"


def training_available() -> bool:
    return importlib.util.find_spec("torch") is not None and importlib.util.find_spec("gsplat") is not None
