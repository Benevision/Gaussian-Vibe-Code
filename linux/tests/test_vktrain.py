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
    # separate runs drift apart (float-atomic order, then MCMC's opacity-weighted sampling); the same training still
    mean = lambda t: sum(t[s]["loss"] for s in range(251, 321)) / 70   # noqa: E731
    print(f"{strategy}: paused + resumed vs uninterrupted, mean loss 251-320: {mean(b):.5f} vs {mean(a):.5f}")
    assert mean(b) == pytest.approx(mean(a), rel=0.1)


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


def test_invalid_config_and_feature_notes(build_dir, dataset, tmp_path):
    rc, msgs = run(build_dir, dataset, tmp_path, {"strategy": "bogus"})
    assert rc == 2
    assert "strategy must be one of" in of(msgs, "error")[0]["message"]
    # stage 2b: every TrainingConfig feature is implemented; depth priors without a cache are reported
    rc, msgs = run(build_dir, dataset, tmp_path, {**BASE, "iterations": 5, "appearanceCorrection": True,
                                                  "transientMasking": True, "normalWeight": 0.05, "depthPriors": "auto"})
    assert rc == 0, of(msgs, "error")
    logs = [m["message"] for m in of(msgs, "log")]
    assert not [n for n in logs if "not supported by the Vulkan trainer" in n]
    assert any("can't predict depth maps" in n for n in logs)


# ---------------------------------------------------------------- stage 2b features (end to end)

FEAT = {**BASE, "iterations": 600, "densifyStop": 500, "shDegreeInterval": 200, "evalInterval": 600, "previewInterval": 0}


@pytest.fixture(scope="module")
def feature_dataset(tmp_path_factory):
    """make_synthetic.py with feature options, generated once per module and name."""
    made = {}

    def make(name, *options):
        if name not in made:
            out = tmp_path_factory.mktemp(f"vktrain-{name}")
            r = subprocess.run([sys.executable, str(VKTRAIN / "tools" / "make_synthetic.py"), str(out), *options],
                               capture_output=True, text=True)
            assert r.returncode == 0, r.stderr
            made[name] = out
        return made[name]
    pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    return make


def done_of(msgs):
    d = of(msgs, "done")
    assert d, of(msgs, "error")
    return d[-1]


def test_skybox(build_dir, feature_dataset, tmp_path):
    """Photos with sky masks: the panorama at infinity learns the sky (sky.png / sky.json next to the result, as the
    CUDA trainer writes them) and the holdout photos are explained better than with the splats alone."""
    from PIL import Image

    ds = feature_dataset("sky", "--sky")
    rc, on = run(build_dir, ds, tmp_path, {**FEAT, "skyboxResolution": 32}, name="on")
    assert rc == 0, of(on, "error")
    rc, off = run(build_dir, ds, tmp_path, {**FEAT, "skybox": "off"}, name="off")
    assert rc == 0, of(off, "error")
    a, b = done_of(on)["evalPsnr"], done_of(off)["evalPsnr"]
    print(f"skybox: holdout PSNR {a:.2f} dB with, {b:.2f} dB without")
    assert any("learned skybox 64×32 at infinity; 14 photos have a sky mask" in m["message"] for m in of(on, "log"))
    assert a > b + 0.5
    meta = json.loads((tmp_path / "on.sky.json").read_text())
    assert meta["format"] == "equirectangular" and meta["image"] == "on.sky.png"
    assert meta["up"] == pytest.approx([0, -1, 0], abs=1e-6) and meta["forward"] == pytest.approx([0, 0, 1], abs=1e-6)
    img = Image.open(tmp_path / "on.sky.png")
    assert img.size == (64, 32) and img.mode == "RGB"
    px = list(img.getdata())
    top = px[:64]                       # straight up: never seen, stays at the initial colour
    assert all(p == (153, 166, 178) for p in top)
    near = [p for p in px[13 * 64:15 * 64] if p != (153, 166, 178)]   # just above the horizon: the blue sky
    assert len(near) > 64 and sum(p[2] - p[0] for p in near) / len(near) > 60
    assert not (tmp_path / "off.sky.png").exists()


def test_colour_correction(build_dir, feature_dataset, tmp_path):
    """Training photos with their own exposure / tint (holdouts with the scene's colours): per-photo colour
    correction (affine, and the bilateral grid with decoupled SSIM) keeps the splats at the true colours."""
    ds = feature_dataset("shift", "--color-shift")
    res = {}
    for name, extra in (("off", {}), ("affine", {"appearanceCorrection": True}),
                        ("grid", {"bilateralGrid": True, "decoupledSSIM": True})):
        rc, msgs = run(build_dir, ds, tmp_path, {**FEAT, **extra}, name=name)
        assert rc == 0, of(msgs, "error")
        res[name] = done_of(msgs)["evalPsnr"]
    print(f"colour correction: holdout PSNR {res}")
    assert res["affine"] > res["off"] + 1.0
    assert res["grid"] > res["off"] + 0.5


def test_depth_supervision(build_dir, feature_dataset, tmp_path):
    """Sensor depth maps (float16, half resolution): the depth loss (and the depth-normal loss) pull the rendered
    depth onto the maps — the holdout depth error drops well below an RGB-only run's."""
    ds = feature_dataset("depth", "--depth")
    rc, off = run(build_dir, ds, tmp_path, {**FEAT, "depthWeight": 0}, name="off")
    assert rc == 0, of(off, "error")
    rc, on = run(build_dir, ds, tmp_path, {**FEAT, "normalWeight": 0.05, "normalStart": 200}, name="on")
    assert rc == 0, of(on, "error")
    e0, e1 = done_of(off)["features"]["evalDepthError"], done_of(on)["features"]["evalDepthError"]
    print(f"depth: holdout depth error {e1:.3f} with depth supervision, {e0:.3f} without")
    assert any("depth supervision weight 0.1" in m["message"] for m in of(on, "log"))
    assert e1 < 0.5 * e0


def test_transient_masking(build_dir, feature_dataset, tmp_path):
    """Unmasked moving objects in 6 training photos: the masker drops their pixels (and few elsewhere)."""
    ds = feature_dataset("trans", "--transients", "6")
    cfg = {**FEAT, "transientMasking": True, "transientStart": 100, "transientMaxViewGap": 90}
    rc, msgs = run(build_dir, ds, tmp_path, cfg, "--trace-every", "1")
    assert rc == 0, of(msgs, "error")
    train = [i for i in range(16) if i % 8 != 7]
    with_obj = set(train[2::max(1, len(train) // 6)][:6])
    tr = [m for m in of(msgs, "trace") if "outliers" in m and m["step"] > 300]
    a = [m["outliers"] for m in tr if m["camera"] in with_obj]
    b = [m["outliers"] for m in tr if m["camera"] not in with_obj]
    fa, fb = sum(a) / len(a), sum(b) / len(b)
    print(f"transient masking: {100 * fa:.1f} % of the pixels dropped in photos with a moving object, {100 * fb:.2f} % "
          f"in the others (the object covers 8.3 %)")
    assert fa > 0.04 and fb < 0.01
    assert done_of(msgs)["features"]["transientFraction"] > 0


def test_pose_refinement(build_dir, feature_dataset, tmp_path):
    """Training poses perturbed by ~1 cm / 0.6°: per-photo pose corrections recover a sharper scene (the holdouts
    have exact poses)."""
    ds = feature_dataset("pose", "--pose-noise", "0.01")
    rc, off = run(build_dir, ds, tmp_path, FEAT, name="off")
    assert rc == 0, of(off, "error")
    rc, on = run(build_dir, ds, tmp_path, {**FEAT, "poseRefinement": True, "poseRefinementStart": 100,
                                           "poseRefinementLR": 0.001}, name="on")
    assert rc == 0, of(on, "error")
    a, b = done_of(on), done_of(off)
    print(f"pose refinement: holdout PSNR {a['evalPsnr']:.2f} dB with, {b['evalPsnr']:.2f} dB without; mean correction "
          f"{a['features']['poseTranslation']:.4f} / {a['features']['poseRotationDeg']:.2f}°")
    assert a["evalPsnr"] > b["evalPsnr"] + 1.0
    assert 0.002 < a["features"]["poseTranslation"] < 0.2


def test_compact_optimizer(build_dir, dataset, tmp_path):
    """optimizerPrecision "compact" (bf16 / 8-bit Adam moments on the GPU, carried losslessly through the host-side
    densification) trains as well as fp32 Adam."""
    cfg = {**FEAT, "strategy": "adaptive"}
    rc, a = run(build_dir, dataset, tmp_path, cfg, name="fp32")
    assert rc == 0, of(a, "error")
    rc, b = run(build_dir, dataset, tmp_path, {**cfg, "optimizerPrecision": "compact"}, name="compact")
    assert rc == 0, of(b, "error")
    pa, pb = done_of(a)["evalPsnr"], done_of(b)["evalPsnr"]
    print(f"compact Adam: holdout PSNR {pb:.2f} dB (fp32 {pa:.2f} dB), {done_of(b)['numGaussians']} splats")
    assert any("optimizerPrecision=compact" in m["message"] and "state 152 B" in m["message"] for m in of(b, "log"))
    assert abs(pb - pa) < 0.5
    assert done_of(b)["numGaussians"] > 2 * 3000   # densification ran on the carried moments


def test_pause_resume_with_features(build_dir, feature_dataset, tmp_path):
    """The saved state includes the skybox, per-photo colour models, pose corrections, the transient threshold and
    compact Adam moments: a run resumed at step 250 follows the uninterrupted one."""
    ds = feature_dataset("sky", "--sky")
    cfg = {**BASE, "iterations": 320, "evalInterval": 0, "previewInterval": 0, "skyboxResolution": 32,
           "appearanceCorrection": True, "transientMasking": True, "transientStart": 50, "transientMaxViewGap": 90,
           "poseRefinement": True, "poseRefinementStart": 100, "optimizerPrecision": "compact"}
    rc, ref = run(build_dir, ds, tmp_path, cfg, "--trace-every", "1", "--snapshot-at", "250", name="ref")
    assert rc == 0, of(ref, "error")
    snap = tmp_path / "snapshot-copy.vkstate"   # a finished resumed run deletes the state it continued from
    shutil.copy(tmp_path / "ref-job" / "snapshot.vkstate", snap)
    rc, cont = run(build_dir, ds, tmp_path, cfg, "--trace-every", "1", "--resume",
                   str(tmp_path / "ref-job" / "snapshot.vkstate"), name="cont")
    assert rc == 0, of(cont, "error")
    a = {m["step"]: m for m in of(ref, "trace")}
    c = {m["step"]: m for m in of(cont, "trace")}
    rel = [abs(c[s]["loss"] - a[s]["loss"]) / a[s]["loss"] for s in range(251, 321)]
    print(f"features: resumed vs uninterrupted, max relative loss difference steps 251-260: {max(rel[:10]):.2e}, "
          f"251-320: {max(rel):.2e}")
    assert max(rel[:10]) < 1e-3
    assert max(rel) < 2e-2
    # a state saved with other features does not load
    rc, bad = run(build_dir, ds, tmp_path, {**cfg, "poseRefinement": False}, "--resume", str(snap), name="bad")
    assert rc == 1 and "features" in of(bad, "error")[0]["message"]
