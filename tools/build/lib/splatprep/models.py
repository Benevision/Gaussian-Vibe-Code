"""Model registry.

``available_models()`` returns the ids that can actually run here (the worker
reports them in ``GET /v1/info``).  Extra backends can be registered with
``register(backend)`` or through ``SPLATPREP_PLUGINS="pkg.module:attr,..."``
where ``attr`` is a Backend instance, a Backend subclass, or a callable
returning a list of backends.
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from typing import Dict, List

from .backends.base import Availability, Backend
from .errors import ModelUnavailable

FEATURE_MODEL_IDS = [
    "colmap_sift",
    "sift_lightglue",
    "superpoint_lightglue",
    "disk_lightglue",
    "aliked_lightglue",
    "loftr",
    "mast3r",
    "vggt",
    "opencv_sift",
]
MASK_MODEL_IDS = ["person", "sky", "sam2_auto", "sam2_track", "dynamic"]

_registry: Dict[str, Backend] = {}
_loaded = False


def _load() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True
    from .backends import builtin_backends

    for b in builtin_backends():
        _registry.setdefault(b.id, b)
    for spec in filter(None, (os.environ.get("SPLATPREP_PLUGINS") or "").split(",")):
        mod, _, attr = spec.strip().partition(":")
        obj = getattr(importlib.import_module(mod), attr or "BACKENDS")
        if isinstance(obj, type):
            obj = obj()
        elif callable(obj) and not isinstance(obj, Backend):
            obj = obj()
        for b in obj if isinstance(obj, (list, tuple)) else [obj]:
            _registry[b.id] = b


def register(backend: Backend) -> None:
    _load()
    _registry[backend.id] = backend


def unregister(model_id: str) -> None:
    _registry.pop(model_id, None)


def get_backend(model_id: str) -> Backend:
    _load()
    b = _registry.get(model_id)
    if b is None:
        raise ModelUnavailable(model_id, "unknown model id")
    return b


def all_backends() -> List[Backend]:
    _load()
    return list(_registry.values())


def model_status() -> Dict[str, dict]:
    """{id: {"kind", "available", "sfm", "reason"}} for feature and mask models."""
    _load()
    out = {}
    for b in _registry.values():
        try:
            a = b.check()
        except Exception as e:  # never let a broken plugin kill the listing
            a = Availability(False, False, f"check failed: {e}")
        out[b.id] = {"kind": b.kind, **a.to_json()}
    from .masks import mask_model_status

    for mid, st in mask_model_status().items():
        out[mid] = {"kind": "mask", **st}
    return out


def available_models(include_masks: bool = True, include_matching_only: bool = False) -> List[str]:
    """Model ids that can run end-to-end here (``sfm`` capable, plus mask models)."""
    st = model_status()
    ids = []
    for mid, s in st.items():
        if s["kind"] == "mask":
            if include_masks and s["available"]:
                ids.append(mid)
        elif s["sfm"] or (include_matching_only and s["available"]):
            ids.append(mid)
    return ids


def model_cards() -> dict:
    return json.loads((Path(__file__).parent / "model_cards.json").read_text())
