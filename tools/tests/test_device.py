"""Device choice with fake PyTorch builds: NVIDIA CUDA, AMD ROCm (HIP behind torch.cuda), Intel XPU, Apple MPS."""
import importlib.machinery
import sys
import types

import pytest

from splatprep import device


def fake_torch(cuda=False, hip=None, xpu=False, mps=False):
    t = types.ModuleType("torch")
    t.__spec__ = importlib.machinery.ModuleSpec("torch", None)
    t.cuda = types.SimpleNamespace(is_available=lambda: cuda, max_memory_allocated=lambda: 2**21,
                                   empty_cache=lambda: None)
    t.version = types.SimpleNamespace(hip=hip, cuda=None if hip else "12.8")
    t.xpu = types.SimpleNamespace(is_available=lambda: xpu, max_memory_allocated=lambda: 3 * 2**20,
                                  empty_cache=lambda: t.calls.append("xpu-empty"))
    t.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps))
    t.calls = []
    return t


@pytest.fixture
def torch_as(monkeypatch):
    monkeypatch.delenv("SPLATPREP_DEVICE", raising=False)

    def use(**kw):
        t = fake_torch(**kw)
        monkeypatch.setitem(sys.modules, "torch", t)
        return t
    return use


@pytest.mark.parametrize("kw,dev,backend", [
    (dict(cuda=True), "cuda", "cuda"),
    (dict(cuda=True, hip="6.3.42131"), "cuda", "rocm"),
    (dict(xpu=True), "xpu", "xpu"),
    (dict(mps=True), "mps", "mps"),
    (dict(), "cpu", "cpu"),
])
def test_select_device_by_build(torch_as, kw, dev, backend):
    torch_as(**kw)
    assert device.select_device() == dev
    assert device.gpu_backend() == backend


def test_preferences_and_overrides(torch_as, monkeypatch):
    torch_as(cuda=True, hip="6.3", xpu=True)
    assert device.select_device("xpu") == "xpu"
    assert device.select_device("rocm") == "cuda"
    assert device.select_device("cpu") == "cpu"
    monkeypatch.setenv("SPLATPREP_DEVICE", "cpu")
    assert device.select_device() == "cpu"


def test_xpu_memory_helpers(torch_as):
    t = torch_as(xpu=True)
    assert device.peak_gpu_mb() == 3.0
    device.empty_cache()
    assert t.calls == ["xpu-empty"]


def test_colmap_gpu_sift_only_on_nvidia(torch_as):
    torch_as(cuda=True, hip="6.3")
    assert device.gpu_backend("cuda") == "rocm"
    torch_as(cuda=True)
    assert device.gpu_backend("cuda") == "cuda"
