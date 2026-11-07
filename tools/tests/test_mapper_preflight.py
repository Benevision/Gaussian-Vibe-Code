"""A full solve must fail before matching when no mapper is installed (it once failed after 70 min of matching)."""
import pytest

from splatprep import mapper
from splatprep.backends.base import Backend
from splatprep.errors import ModelUnavailable


def test_no_mapper_fails_before_matching(monkeypatch):
    monkeypatch.setattr(mapper, "available_mappers", lambda: {"glomap": False, "colmap": False, "pycolmap": False})
    b = Backend()
    a = b.check()
    assert a.available and not a.sfm and "python" in a.reason
    with pytest.raises(ModelUnavailable):
        b.require(None)


def test_mapper_present_passes(monkeypatch):
    monkeypatch.setattr(mapper, "available_mappers", lambda: {"glomap": False, "colmap": False, "pycolmap": True})
    assert Backend().check().sfm


def test_blocked_module_still_counts_as_installed(monkeypatch):
    """LightGlue blocks pycolmap imports in-process; the pycolmap mapper runs in a child, so it is still available."""
    import sys
    import types
    import importlib.util

    from splatprep import device
    from splatprep.isolation import _Blocker

    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda n, *a: types.SimpleNamespace(origin="x") if n == "pycolmap" and not any(isinstance(f, _Blocker) for f in sys.meta_path) else real(n, *a)
                        if n != "pycolmap" else (_ for _ in ()).throw(ImportError("blocked")))
    monkeypatch.setattr(sys, "meta_path", [_Blocker()] + list(sys.meta_path))
    assert not device.has_module("pycolmap")
    assert device.module_installed("pycolmap")
    assert any(isinstance(f, _Blocker) for f in sys.meta_path)  # blocker restored
