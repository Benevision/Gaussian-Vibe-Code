import io
import json
import sys
import threading
import time
import zipfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from splatworker.backends import Backend, JobCancelled, JobFailed, JobPaused  # noqa: E402
from splatworker.ply import write_gaussian_ply  # noqa: E402


# ---------------------------------------------------------------- datasets

def look_at_c2w(position, target=(0, 0, 0)):
    """OpenCV camera-to-world looking from ``position`` at ``target`` (+Z forward, +Y down)."""
    pos = np.asarray(position, float)
    fwd = np.asarray(target, float) - pos
    fwd /= np.linalg.norm(fwd)
    up = np.array([0.0, -1.0, 0.0])
    right = np.cross(fwd, up)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0, 0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = right, down, fwd, pos
    return m


def write_points_ply(path, xyz, rgb):
    n = len(xyz)
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {n}\nproperty float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    rec = np.zeros(n, dtype=dt)
    rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rec["r"], rec["g"], rec["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with open(path, "wb") as fh:
        fh.write(header.encode())
        fh.write(rec.tobytes())


def make_dataset(root: Path, n_cams=4, width=64, height=48, depth=True, points=True, holdout_last=True):
    from PIL import Image

    root.mkdir(parents=True, exist_ok=True)
    (root / "images").mkdir(exist_ok=True)
    if depth:
        (root / "depth").mkdir(exist_ok=True)
    cams = []
    for i in range(n_cams):
        ang = 2 * np.pi * i / n_cams
        pos = (3 * np.cos(ang), 0.0, 3 * np.sin(ang))
        img = (np.random.default_rng(i).random((height, width, 3)) * 255).astype(np.uint8)
        Image.fromarray(img).save(root / "images" / f"{i:06d}.png")
        cam = {
            "id": i, "image": f"images/{i:06d}.png", "width": width, "height": height,
            "fx": 50.0, "fy": 50.0, "cx": width / 2, "cy": height / 2,
            "cameraToWorld": look_at_c2w(pos).tolist(),
            "holdout": bool(holdout_last and i == n_cams - 1),
        }
        if depth:
            d = np.full((12, 16), 3.0, dtype="<f2")
            d[0, 0] = 0  # invalid pixel
            d.tofile(root / "depth" / f"{i:06d}.bin")
            cam.update({"depth": f"depth/{i:06d}.bin", "depthWidth": 16, "depthHeight": 12})
        cams.append(cam)
    doc = {"format": "splatdataset", "version": 1, "convention": "opencv",
           "sceneCenter": [0, 0, 0], "sceneExtent": 3.3, "upVector": [0, -1, 0],
           "cameras": cams, "someFutureKey": {"ignored": True}}
    if points:
        rng = np.random.default_rng(0)
        write_points_ply(root / "points3d.ply", rng.uniform(-1, 1, (200, 3)).astype(np.float32),
                         rng.integers(0, 255, (200, 3)).astype(np.uint8))
        doc["points"] = "points3d.ply"
    (root / "dataset.json").write_text(json.dumps(doc))
    return root


def zip_dir(src: Path, prefix: str = "") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(src.rglob("*")):
            if p.is_file():
                zf.write(p, prefix + str(p.relative_to(src)))
    return buf.getvalue()


@pytest.fixture
def dataset_dir(tmp_path):
    return make_dataset(tmp_path / "dataset")


@pytest.fixture
def dataset_zip(dataset_dir):
    return zip_dir(dataset_dir)


# ------------------------------------------------------------ fake backend

class FakeBackend(Backend):
    """In-thread backend that mimics the trainer's callbacks without torch."""

    pausable = True

    def __init__(self):
        self.mode = "ok"            # ok | fail | block
        self.steps = 5
        self.step_delay = 0.0       # seconds per step (lets tests pause / checkpoint mid-run)
        self.resumed_from = []      # start step of every train run (0 = fresh)
        self.started = threading.Event()
        self.release = threading.Event()
        self.contexts = []

    def capabilities(self):
        return ["train", "process"]

    def models(self):
        return ["superpoint_lightglue"]

    def run(self, ctx):
        self.contexts.append(ctx)
        if ctx.kind == "process":
            from splatworker.processing import run_processing

            self.started.set()
            return run_processing(ctx.dataset_dir, ctx.spec, ctx.job_dir,
                                  progress_cb=ctx.status, log=ctx.log,
                                  should_stop=ctx.cancel_event.is_set)
        if ctx.kind == "census":
            self.started.set()
            from splatworker.studio.formats import load

            n = load(ctx.spec["plyPath"], with_skybox=False).count
            np.savez(ctx.job_dir / "census.npz", peak=np.linspace(0, 1, n, dtype=np.float32),
                     views=np.arange(n, dtype=np.uint32))
            return {"splats": n}
        ctx.status(state="loading", message="loading dataset")
        ctx.log("fake trainer loading")
        self.started.set()
        if self.mode == "block":
            while not ctx.cancel_event.is_set() and not self.release.is_set():
                time.sleep(0.01)
            if ctx.cancel_event.is_set():
                raise JobCancelled()
        if self.mode == "fail":
            raise JobFailed("CUDA out of memory (fake)")
        ctx.status(state="training", message="training")
        n = 10
        resume = ctx.job_dir / "resume.json"      # the fake's stand-in for the trainer's resume.pt
        start = json.loads(resume.read_text())["step"] if resume.is_file() else 0
        self.resumed_from.append(start)
        for step in range(start + 1, self.steps + 1):
            if ctx.cancel_event.is_set():
                raise JobCancelled()
            if ctx.checkpoint_event.is_set():
                ctx.checkpoint_event.clear()
                write_fake_ply(ctx.job_dir / "checkpoint.ply", n)
                ctx.checkpoint(step - 1)
            if ctx.pause_event.is_set():
                write_fake_ply(ctx.job_dir / "checkpoint.ply", n)
                ctx.checkpoint(step - 1)
                resume.write_text(json.dumps({"step": step - 1}))
                (ctx.job_dir / "resume.pt").write_bytes(b"fake")   # what the server looks for
                raise JobPaused(step - 1)
            ctx.status(step=step, loss=0.1 / step, psnr=20.0 + step, numGaussians=n,
                       etaSeconds=float(self.steps - step))
            if step == 2:
                write_fake_ply(ctx.job_dir / "preview.ply", n)
                ctx.preview(step)
            if self.step_delay:
                time.sleep(self.step_delay)
        write_fake_ply(ctx.job_dir / "result.ply", n)
        resume.unlink(missing_ok=True)
        return {"numGaussians": n, "psnr": 25.0, "loss": 0.02}


def write_fake_ply(path, n):
    rng = np.random.default_rng(1)
    write_gaussian_ply(path, rng.normal(size=(n, 3)), rng.normal(size=(n, 3)),
                       rng.normal(size=(n, 15, 3)), rng.normal(size=n),
                       rng.normal(size=(n, 3)), rng.normal(size=(n, 4)))


def wait_for(fn, timeout=10.0, interval=0.02):
    deadline = time.time() + timeout
    while time.time() < deadline:
        v = fn()
        if v:
            return v
        time.sleep(interval)
    raise AssertionError("condition not met in time")


@pytest.fixture
def fake_splatprep(tmp_path, monkeypatch):
    """A stand-in ``splatprep`` package on sys.path (also visible to spawned children)."""
    pkg = tmp_path / "fakepkgs" / "splatprep"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "models.py").write_text("def available_models():\n    return ['superpoint_lightglue', 'colmap_sift']\n")
    (pkg / "runner.py").write_text('''
import json
from pathlib import Path

def run_request(request, workdir, progress):
    workdir = Path(workdir)
    if request.get("features") == "unavailable_model":
        progress({"error": "model unavailable: unavailable_model"})
        return {"error": "model unavailable: unavailable_model"}
    images = sorted((workdir / request["imagesDir"]).iterdir())
    out = workdir / request["outputDir"]
    (out / "images").mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(images):
        progress({"stage": "features", "progress": (i + 1) / (2 * len(images)), "message": img.name})
        (out / "images" / img.name).write_bytes(img.read_bytes())
    progress({"stage": "mapping", "progress": 1.0})
    (out / "dataset.json").write_text(json.dumps({"format": "splatdataset", "cameras": []}))
    progress({"done": True, "stats": {"images": len(images)}})
    return {"done": True, "stats": {"images": len(images)}}
''')
    monkeypatch.syspath_prepend(str(pkg.parent))
    for mod in list(sys.modules):
        if mod == "splatprep" or mod.startswith("splatprep."):
            monkeypatch.delitem(sys.modules, mod)
    import importlib

    importlib.invalidate_caches()
    return pkg.parent
