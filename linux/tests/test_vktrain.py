"""Vulkan trainer (linux/vktrain): builds it and runs its kernel self-test and the dump round trip.

Needs cmake, a C++17 compiler, glslangValidator, the Vulkan headers and a Vulkan device (Mesa lavapipe is
enough: see linux/vktrain/ci/Dockerfile). Skipped when any of them is missing, so the normal suite stays
green on machines without Vulkan.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

VKTRAIN = Path(__file__).resolve().parents[1] / "vktrain"

pytestmark = pytest.mark.skipif(
    not (shutil.which("glslangValidator") and shutil.which("cmake") and (shutil.which("c++") or shutil.which("g++"))),
    reason="glslangValidator / cmake / C++ compiler not available",
)


@pytest.fixture(scope="module")
def build_dir(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("vktrain-build")
    gen = ["-G", "Ninja"] if shutil.which("ninja") else []
    cfg = subprocess.run(["cmake", "-S", str(VKTRAIN), "-B", str(out), *gen, "-DCMAKE_BUILD_TYPE=Release"],
                         capture_output=True, text=True)
    if cfg.returncode != 0:
        if "Could NOT find Vulkan" in cfg.stdout + cfg.stderr:
            pytest.skip("Vulkan headers / loader not installed")
        pytest.fail(f"cmake configure failed:\n{cfg.stdout}\n{cfg.stderr}")
    jobs = str(max(1, min(4, os.cpu_count() or 1)))
    b = subprocess.run(["cmake", "--build", str(out), "-j", jobs], capture_output=True, text=True)
    assert b.returncode == 0, f"build failed:\n{b.stdout[-4000:]}\n{b.stderr[-4000:]}"
    info = subprocess.run([str(out / "splatwork-vk-train"), "--info"], capture_output=True, text=True)
    if info.returncode != 0:
        pytest.skip(f"no usable Vulkan device: {info.stderr.strip()}")
    return out


def test_kernel_selftest(build_dir):
    r = subprocess.run([str(build_dir / "vktrain-selftest"), "--quick"], capture_output=True, text=True, timeout=900)
    print(r.stdout)
    assert r.returncode == 0, f"selftest failed (exit {r.returncode}):\n{r.stdout[-6000:]}\n{r.stderr[-2000:]}"
    assert "0 failed" in r.stdout


def test_dump_matches_cpu_reference(build_dir, tmp_path):
    pytest.importorskip("numpy")
    r = subprocess.run([sys.executable, str(VKTRAIN / "compare_gsplat.py"), "all", "--dir", str(tmp_path),
                        "--vk-train", str(build_dir / "splatwork-vk-train"), "--no-gsplat",
                        "--n", "300", "--width", "96", "--height", "64"], capture_output=True, text=True, timeout=600)
    print(r.stdout)
    assert r.returncode == 0, f"{r.stdout}\n{r.stderr}"
    assert "image vs CPU double reference" in r.stdout
