"""Vulkan trainer (linux/vktrain): builds it and runs its kernel self-test, the dump round trip and end-to-end
training runs of splatwork-vk-train on a small synthetic dataset (tools/make_synthetic.py): PSNR, densification
with both strategies, pause → resume, checkpoints, cancel, region trimming, config errors.

Needs cmake, a C++17 compiler, glslangValidator, the Vulkan headers and a Vulkan device (Mesa lavapipe is
enough: see linux/vktrain/ci/Dockerfile). Skipped when any of them is missing, so the normal suite stays
green on machines without Vulkan. VKTRAIN_BUILD_DIR=<dir> reuses an existing build instead of building.
"""
from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

import pytest

VKTRAIN = Path(__file__).resolve().parents[1] / "vktrain"

pytestmark = pytest.mark.skipif(
    not (os.environ.get("VKTRAIN_BUILD_DIR") or
         (shutil.which("glslangValidator") and shutil.which("cmake") and (shutil.which("c++") or shutil.which("g++")))),
    reason="glslangValidator / cmake / C++ compiler not available",
)


@pytest.fixture(scope="module")
def build_dir(tmp_path_factory) -> Path:
    prebuilt = os.environ.get("VKTRAIN_BUILD_DIR")   # reuse an existing build (CI builds once, before pytest)
    if prebuilt and (Path(prebuilt) / "splatwork-vk-train").is_file():
        out = Path(prebuilt)
    else:
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


# ---------------------------------------------------------------- training (splatwork-vk-train train)

@pytest.fixture(scope="module")
def dataset(tmp_path_factory) -> Path:
    pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    out = tmp_path_factory.mktemp("vktrain-synthetic")
    r = subprocess.run([sys.executable, str(VKTRAIN / "tools" / "make_synthetic.py"), str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return out


# A short schedule: densification every 100 steps from 100 to 700, SH degree up every 250 steps.
BASE = {"iterations": 1000, "maxGaussians": 10000, "densifyStart": 100, "densifyStop": 700, "densifyInterval": 100,
        "shDegreeInterval": 250, "evalInterval": 250, "previewInterval": 500, "appearanceCorrection": False,
        "normalWeight": 0, "seed": 7}


def run(build_dir: Path, dataset: Path, tmp: Path, cfg: dict, *extra: str, name: str = "r"):
    """Runs one training to its end; returns (exit code, JSON messages)."""
    (tmp / f"{name}.json").write_text(json.dumps(cfg))
    cmd = [str(build_dir / "splatwork-vk-train"), "train", "--dataset", str(dataset), "--config",
           str(tmp / f"{name}.json"), "--out", str(tmp / f"{name}.ply"), "--job-dir", str(tmp / f"{name}-job"), *extra]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600, stdin=subprocess.DEVNULL)
    msgs = [json.loads(line) for line in r.stdout.splitlines() if line.startswith("{")]
    for m in msgs:
        if m["type"] in ("log", "error", "done", "paused"):
            print(m)
    return r.returncode, msgs


def of(msgs, kind):
    return [m for m in msgs if m["type"] == kind]


def ply_vertices(path: Path):
    data = path.read_bytes()
    end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:end].decode()
    n = int(next(line.split()[2] for line in header.splitlines() if line.startswith("element vertex")))
    props = [line.split()[2] for line in header.splitlines() if line.startswith("property")]
    assert len(data) - end == n * len(props) * 4
    rows = [struct.unpack_from(f"<{len(props)}f", data, end + i * len(props) * 4) for i in range(n)]
    return header, props, rows


def eval_curve(msgs):
    return [float(m["message"].split("holdout PSNR ")[1].split()[0]) for m in of(msgs, "log")
            if "holdout PSNR" in m["message"]]


@pytest.mark.parametrize("strategy", ["mcmc", "adaptive"])
def test_training_converges_and_densifies(build_dir, dataset, tmp_path, strategy):
    t0 = time.time()
    rc, msgs = run(build_dir, dataset, tmp_path, {**BASE, "strategy": strategy})
    assert rc == 0, of(msgs, "error")
    done = of(msgs, "done")[-1]
    curve = eval_curve(msgs)
    print(f"{strategy}: holdout PSNR {curve}, {done['numGaussians']} splats, {time.time() - t0:.0f} s, "
          f"{done['stepsPerSecond']:.1f} steps/s")
    assert len(curve) == 4 and curve[-1] > curve[0] + 2 and curve[-1] > 23.0, curve
    assert done["evalPsnr"] == pytest.approx(curve[-1], abs=0.01)
    initial = next(m["numGaussians"] for m in of(msgs, "status") if m.get("state") == "training")
    assert initial == 3000
    if strategy == "mcmc":   # +5 % per refinement at steps 200…600 (gsplat: step > densifyStart)
        expected = initial
        for _ in range(5):
            expected = int(1.05 * expected)
        assert done["numGaussians"] == expected
    else:
        assert done["numGaussians"] > 1.5 * initial
    # progress, preview, result
    st = [m for m in of(msgs, "status") if "step" in m and "loss" in m]
    assert st[-1]["step"] == 1000 and st[-1]["totalSteps"] == 1000 and st[-1]["evalPsnr"] is not None
    assert of(msgs, "preview") == [{"type": "preview", "step": 500}]
    preview = (tmp_path / "r-job" / "preview.splat").read_bytes()
    assert len(preview) % 32 == 0 and len(preview) > 0
    header, props, rows = ply_vertices(tmp_path / "r.ply")
    assert "comment splatstudio_up 0 -1 0" in header
    assert len(props) == 62 and props[:3] == ["x", "y", "z"] and props[-4:] == ["rot_0", "rot_1", "rot_2", "rot_3"]
    assert len(rows) == done["numGaussians"]
    assert not (tmp_path / "r-job" / "resume.vkstate").exists()


@pytest.mark.parametrize("strategy", ["adaptive", "mcmc"])
def test_pause_resume_continues_the_same_run(build_dir, dataset, tmp_path, strategy):
    """The saved state is complete: a run resumed from the state of step 250 continues the uninterrupted run's
    loss trajectory (only the GPU's float-atomic summation order differs); a real pause exits 3 and resumes."""
    cfg = {**BASE, "strategy": strategy, "iterations": 320, "evalInterval": 0, "previewInterval": 0}
    # uninterrupted run that also writes the state a pause at step 250 would save
    rc, ref = run(build_dir, dataset, tmp_path, cfg, "--trace-every", "1", "--snapshot-at", "250", name="ref")
    assert rc == 0, of(ref, "error")
    snap = tmp_path / "ref-job" / "snapshot.vkstate"
    rc, cont = run(build_dir, dataset, tmp_path, cfg, "--trace-every", "1", "--resume", str(snap), name="cont")
    assert rc == 0, of(cont, "error")
    assert any("resuming from the state saved at step 250" in m["message"] for m in of(cont, "log"))
    a = {m["step"]: m for m in of(ref, "trace")}
    c = {m["step"]: m for m in of(cont, "trace")}
    assert sorted(c) == list(range(251, 321))
    rel = [abs(c[s]["loss"] - a[s]["loss"]) / a[s]["loss"] for s in range(251, 321)]
    print(f"{strategy}: resumed vs uninterrupted, max relative loss difference steps 251-260: {max(rel[:10]):.2e}, "
          f"251-320: {max(rel):.2e}; splats {a[320]['numGaussians']} vs {c[320]['numGaussians']}")
    assert max(rel[:10]) < 1e-4
    assert max(rel) < 1e-2
    assert abs(a[320]["numGaussians"] - c[320]["numGaussians"]) <= 0.01 * a[320]["numGaussians"]

    # a real pause: checkpoint + state written, exit 3, no result; resume finishes and removes the state
    rc, first = run(build_dir, dataset, tmp_path, cfg, "--trace-every", "1", "--pause-at", "250", name="p")
    assert rc == 3, of(first, "error")
    assert of(first, "paused")[-1]["step"] == 250
    state = tmp_path / "p-job" / "resume.vkstate"
    assert state.is_file() and (tmp_path / "p-job" / "checkpoint.ply").is_file()
    assert not (tmp_path / "p.ply").exists()
    rc, second = run(build_dir, dataset, tmp_path, cfg, "--trace-every", "1", "--resume", str(state), name="p")
    assert rc == 0, of(second, "error")
    b = {m["step"]: m for m in of(first, "trace") + of(second, "trace")}
    assert sorted(b) == list(range(1, 321))
    assert (tmp_path / "p.ply").is_file() and not state.exists()
    # separate runs drift apart a little (float-atomic order); they stay the same training
    assert b[320]["loss"] == pytest.approx(a[320]["loss"], rel=0.1)


def test_checkpoint_and_cancel_on_stdin(build_dir, dataset, tmp_path):
    (tmp_path / "c.json").write_text(json.dumps({**BASE, "iterations": 100000}))
    p = subprocess.Popen([str(build_dir / "splatwork-vk-train"), "train", "--dataset", str(dataset), "--config",
                          str(tmp_path / "c.json"), "--out", str(tmp_path / "c.ply"), "--job-dir", str(tmp_path / "job")],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    msgs = []
    try:
        for line in p.stdout:
            m = json.loads(line)
            msgs.append(m)
            if m["type"] == "status" and m.get("state") == "training":
                p.stdin.write("checkpoint\n")
                p.stdin.flush()
            if m["type"] == "checkpoint":
                p.stdin.write("cancel\n")
                p.stdin.flush()
        rc = p.wait(timeout=120)
    finally:
        if p.poll() is None:
            p.kill()
    assert rc == 4
    assert of(msgs, "cancelled") and of(msgs, "checkpoint")
    _, _, rows = ply_vertices(tmp_path / "job" / "checkpoint.ply")
    assert len(rows) == 3000
    assert not (tmp_path / "c.ply").exists()


def test_region_trims_the_result(build_dir, dataset, tmp_path):
    region = {"min": [-1.6, -0.5, -0.5], "max": [0.2, 1.1, 1.2], "margin": 0.1}
    rc, msgs = run(build_dir, dataset, tmp_path, {**BASE, "iterations": 120, "region": region})
    assert rc == 0, of(msgs, "error")
    logs = " ".join(m["message"] for m in of(msgs, "log"))
    assert "region: kept" in logs and "inside the expanded box written" in logs
    lo = [a - 0.1 * (b - a) for a, b in zip(region["min"], region["max"])]
    hi = [b + 0.1 * (b - a) for a, b in zip(region["min"], region["max"])]
    _, _, rows = ply_vertices(tmp_path / "r.ply")
    assert 0 < len(rows) < 3000
    for r in rows:
        assert all(lo[k] - 1e-6 <= r[k] <= hi[k] + 1e-6 for k in range(3))


def test_invalid_config_and_unsupported_features(build_dir, dataset, tmp_path):
    rc, msgs = run(build_dir, dataset, tmp_path, {"strategy": "bogus"})
    assert rc == 2
    assert "strategy must be one of" in of(msgs, "error")[0]["message"]
    rc, msgs = run(build_dir, dataset, tmp_path, {**BASE, "iterations": 5, "appearanceCorrection": True,
                                                  "transientMasking": True})
    assert rc == 0
    notes = [m["message"] for m in of(msgs, "log") if "not supported by the Vulkan trainer yet" in m["message"]]
    assert any(n.startswith("appearanceCorrection") for n in notes)
    assert any(n.startswith("transientMasking") for n in notes)
