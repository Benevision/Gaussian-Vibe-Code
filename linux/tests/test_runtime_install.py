"""Remote install of processing components (/v1/runtime): fixed component list, progress, failure."""
import sys
import time

import pytest

from splatworker import runtime_install as ri


def test_unknown_component_rejected():
    with pytest.raises(ValueError):
        ri.RuntimeInstaller().start("anything; rm -rf /")


def _wait(inst, timeout=20):
    t0 = time.time()
    while inst.status()["state"] == "running" and time.time() - t0 < timeout:
        time.sleep(0.05)
    return inst.status()


def test_install_runs_steps_and_reports(monkeypatch):
    inst = ri.RuntimeInstaller()
    ok = [sys.executable, "-c", "print('Successfully installed x')"]
    monkeypatch.setattr(inst, "_steps", lambda: [("a", "Step A", ok, 0.5), ("b", "Step B", ok, 0.5)])
    monkeypatch.setattr(ri, "installed_components", lambda: {"processing": True})
    inst.start("processing")
    st = _wait(inst)
    assert st["state"] == "done" and st["fraction"] == 1.0
    assert any("Successfully installed" in line for line in st["log"])


def test_install_failure_is_reported(monkeypatch):
    inst = ri.RuntimeInstaller()
    bad = [sys.executable, "-c", "import sys; print('ERROR: no matching distribution'); sys.exit(1)"]
    monkeypatch.setattr(inst, "_steps", lambda: [("a", "Step A", bad, 1.0)])
    inst.start("processing")
    st = _wait(inst)
    assert st["state"] == "failed" and "Step A failed" in st["message"]

