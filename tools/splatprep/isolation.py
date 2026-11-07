"""Keeps pycolmap and PyTorch out of the same process.

On macOS the pycolmap and PyTorch wheels each bundle their own ``libomp.dylib``.  Loading both
into one process makes OpenMP abort (``OMP: Error #15``, SIGABRT).  LightGlue imports pycolmap
opportunistically, so any process that may load torch blocks the import, and the pycolmap
mapper runs in a child process (see :func:`splatprep.mapper.run_pycolmap_subprocess`).
"""
from __future__ import annotations

import importlib.abc
import sys

_BLOCKED = ("pycolmap",)


class _Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):  # noqa: D401
        if fullname.split(".")[0] in _BLOCKED:
            raise ImportError(f"{fullname} is not imported in torch processes (duplicate libomp); "
                              "the pycolmap mapper runs in a subprocess instead")
        return None


def block_pycolmap() -> None:
    """Idempotent. Has no effect on modules already imported."""
    if "pycolmap" in sys.modules:
        return
    if not any(isinstance(f, _Blocker) for f in sys.meta_path):
        sys.meta_path.insert(0, _Blocker())


def pycolmap_blocked() -> bool:
    return any(isinstance(f, _Blocker) for f in sys.meta_path)
