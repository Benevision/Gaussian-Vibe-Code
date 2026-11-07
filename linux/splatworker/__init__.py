"""Splat Worker: CUDA Gaussian-splat training server for Splat Studio.

The HTTP server and dataset parsing only need the standard library, numpy,
Pillow, FastAPI and zeroconf. PyTorch and gsplat are imported lazily by
:mod:`splatworker.trainer`, so the API can run (and be tested) on machines
without CUDA.
"""

__version__ = "1.1.4"
PROTOCOL_VERSION = 1
