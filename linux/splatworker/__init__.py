"""SplatWork for Linux: CUDA Gaussian-splat training server for SplatWork.

The HTTP server and dataset parsing only need the standard library, numpy,
Pillow, FastAPI and zeroconf. PyTorch and gsplat are imported lazily by
:mod:`splatworker.trainer`, so the API can run (and be tested) on machines
without CUDA.
"""

import sys as _sys

__version__ = "1.6.0"
# The product name shown to people (logs, generated files, messages); the protocol and identifiers don't change.
PRODUCT_NAME = "SplatWork for Windows" if _sys.platform == "win32" else "SplatWork for Linux"
PROTOCOL_VERSION = 1
