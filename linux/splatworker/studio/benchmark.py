"""Training benchmark of this PC (Mac: ``Remote/SystemBenchmark.swift``, ``BenchmarkStore``).

The standard benchmark trains a synthetic scene — textured ground, boxes and pillars seen by 48 cameras on two
rings, 1600×1200 photos, seeded with 1 M points so training runs at full load from the first step — for 2,000
steps with fixed settings, and measures steps per second between step 1,000 and the end. The Mac runs the same
recipe on itself and on every worker, so the numbers are comparable (the photos here are rendered by a simple
numpy point splatter instead of the Mac's trainer: the content differs a little, the load does not).

Results are stored in ``benchmarks.json`` in the app's config dir with the Mac's ``DeviceBenchmark`` fields
(``{deviceId: {deviceId, name, date, gpu, software, trainStepsPerSecond, splats, megapixels, uploadMBps,
downloadMBps}}``); :func:`load_benchmarks` / :func:`throughput` are the API for schedulers.

The job is an ordinary training job of a worker: the Studio registers the scene folder with its own server
(``/v1/datasets/local``), ``splatworker benchmark`` uploads it as a zip (any worker, also timing the upload).
"""
from __future__ import annotations

import json
import math
import os
import shutil
import time
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

FILE = "benchmarks.json"
SCENE_DIR = "benchmark-scene-v1"


@dataclass
class BenchmarkSpec:
    width: int = 1600
    height: int = 1200
    views: int = 48
    splats: int = 1_000_000
    steps: int = 2000
    measure_from: int = 1000

    @property
    def megapixels(self) -> float:
        return self.width * self.height / 1e6

    def training_config(self) -> Dict[str, Any]:
        """The Mac's ``SystemBenchmark.config`` in worker keys (PROTOCOL.md §3)."""
        return {"iterations": self.steps, "maxGaussians": self.splats, "maxImageSize": max(self.width, self.height),
                "densifyStart": min(200, self.steps), "densifyStop": max(1, int(self.steps * 0.75)),
                "shDegree": 3, "shDegreeInterval": max(1, self.steps * 3 // 20),
                "evalInterval": 0, "previewInterval": 0, "appearanceCorrection": False, "transientMasking": False,
                "depthWeight": 0.0, "skybox": "off", "seed": 7}


class BenchmarkCancelled(Exception):
    pass


class BenchmarkError(Exception):
    pass


# ---------------------------------------------------------------------------------------------- results store
def load_benchmarks(config_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Saved results ``{deviceId: record}`` (Mac ``DeviceBenchmark`` fields); {} when none or unreadable."""
    try:
        data = json.loads((Path(config_dir) / FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {k: v for k, v in data.items()
            if isinstance(v, dict) and isinstance(v.get("trainStepsPerSecond"), (int, float))}


def save_benchmark(config_dir: Path, record: Dict[str, Any]) -> None:
    config_dir = Path(config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
    data = load_benchmarks(config_dir)
    data[record["deviceId"]] = record
    tmp = config_dir / (FILE + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, config_dir / FILE)


def sorted_results(config_dir: Path) -> List[Dict[str, Any]]:
    return sorted(load_benchmarks(config_dir).values(), key=lambda r: -float(r["trainStepsPerSecond"]))


def throughput(record: Dict[str, Any]) -> float:
    """Scheduler throughput (1.0 ≈ 20 benchmark steps/s), as the Mac's ``BenchmarkStore.throughput``."""
    return float(record["trainStepsPerSecond"]) / 20.0


def seconds_per_step(record: Dict[str, Any], splats: int, megapixels: float) -> float:
    """Per-step cost for a job of this size, scaled from the benchmark load (half splat-, half pixel-bound)."""
    base = 1.0 / max(float(record["trainStepsPerSecond"]), 0.01)
    s = 0.5 * max(splats, 1) / max(int(record.get("splats") or 1), 1) + \
        0.5 * max(megapixels, 0.05) / max(float(record.get("megapixels") or 0.05), 0.05)
    return base * s


# ---------------------------------------------------------------------------------------------- the scene
def _truth(rng: np.random.Generator):
    """~420 k ground-truth points (position, colour 0–1): noise-textured ground, boxes, pillars (down is +y)."""
    def noise(x, z):
        return (np.sin(x * 3.1) * np.cos(z * 2.7) + np.sin(x * 11.3 + z * 7.1) * 0.5
                + np.sin(x * 31 - z * 23) * 0.25) / 1.75 * 0.5 + 0.5
    n = 300_000
    x, z = rng.uniform(-6, 6, n), rng.uniform(-6, 6, n)
    v = noise(x, z)
    chk = np.where((np.floor(x) + np.floor(z)) % 2 == 0, 0.1, -0.1)
    pos = [np.stack([x, np.ones(n), z], 1)]
    col = [np.stack([0.35 + 0.4 * v + chk, 0.45 + 0.3 * v, 0.25 + 0.2 * v], 1)]
    for b in range(24):
        cx, cz = rng.uniform(-4.5, 4.5, 2)
        hs = np.array([rng.uniform(0.2, 0.7), rng.uniform(0.2, 1.2), rng.uniform(0.2, 0.7)])
        hue = np.array([1 if b % 3 == 0 else 0.3, 0.9 if b % 3 == 1 else 0.35, 1 if b % 3 == 2 else 0.4])
        p = rng.uniform(-1, 1, (4000, 3))
        axis = rng.integers(0, 3, 4000)
        r = np.arange(4000)
        p[r, axis] = np.where(p[r, axis] < 0, -1.0, 1.0)
        p[(axis == 1) & (p[:, 1] > 0), 1] = -1.0           # no bottom face
        q = np.array([cx, 1 - hs[1], cz]) + p * hs
        shade = 0.55 + 0.45 * noise(q[:, 0] * 4 + q[:, 1] * 3, q[:, 2] * 4 - q[:, 1])
        pos.append(q)
        col.append(hue[None] * shade[:, None])
    for _ in range(12):
        cx, cz = rng.uniform(-5, 5, 2)
        k = np.arange(2000)
        a = rng.uniform(0, 2 * math.pi, 2000)
        pos.append(np.stack([cx + 0.06 * np.cos(a), 1 - k / 2000 * 2.5, cz + 0.06 * np.sin(a)], 1))
        col.append(np.array([0.9, 0.9, 0.85])[None] * (0.7 + 0.3 * (k % 50) / 50)[:, None])
    return np.concatenate(pos).astype(np.float32), np.clip(np.concatenate(col), 0, 1).astype(np.float32)


def _cameras(spec: BenchmarkSpec) -> List[Dict[str, Any]]:
    fx = 1400.0 * spec.width / 1600
    target = np.array([0, 0.3, 0])
    out = []
    for i in range(spec.views):
        ring, a = i % 2, i / spec.views * 2 * math.pi
        r, y = (6.0, -2.2) if ring == 0 else (4.5, -3.6)
        eye = np.array([r * math.cos(a), y, r * math.sin(a)])
        zv = target - eye
        zv /= np.linalg.norm(zv)
        xv = np.cross([0, 1.0, 0], zv)
        xv /= np.linalg.norm(xv)
        yv = np.cross(zv, xv)
        m = np.eye(4)
        m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = xv, yv, zv, eye
        out.append({"id": i, "image": f"images/{i:03d}.jpg", "width": spec.width, "height": spec.height,
                    "fx": fx, "fy": fx, "cx": spec.width / 2, "cy": spec.height / 2,
                    "cameraToWorld": m.tolist(), "holdout": i % 8 == 7})
    return out


def _render(pos, col, cam, down: int = 4) -> np.ndarray:
    """Nearest-point z-buffer at 1/``down`` resolution (each point ≈ a pixel there), upscaled: a cheap photo."""
    from PIL import Image

    w, h = cam["width"] // down, cam["height"] // down
    c2w = np.asarray(cam["cameraToWorld"])
    R, t = c2w[:3, :3], c2w[:3, 3]
    pc = (pos - t) @ R
    z = pc[:, 2]
    ok = z > 0.1
    u = (cam["fx"] * pc[:, 0] / np.where(ok, z, 1) + cam["cx"]) / down
    v = (cam["fy"] * pc[:, 1] / np.where(ok, z, 1) + cam["cy"]) / down
    ok &= (u >= 0) & (v >= 0) & (u < w) & (v < h)
    idx = np.nonzero(ok)[0]
    idx = idx[np.argsort(-z[idx], kind="stable")]           # far → near: nearer points written last win
    img = np.empty((h, w, 3), np.float32)
    img[:] = np.linspace([0.55, 0.7, 0.9], [0.8, 0.85, 0.9], h)[:, None, :]      # sky gradient
    zb = np.full((h, w), np.inf, np.float32)
    vi, ui = v[idx].astype(np.int64), u[idx].astype(np.int64)
    img[vi, ui] = col[idx]
    zb[vi, ui] = z[idx]
    # Close the gaps between points: a pixel that is empty or much farther than a neighbour (seen through a
    # surface) takes that neighbour's colour; a few rounds fill holes up to ~4 px wide.
    for _ in range(4):
        best_z, best_c = zb.copy(), img.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                zp = np.pad(zb, 1, constant_values=np.inf)       # no wrap-around at the borders
                nz = zp[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
                nc = np.pad(img, ((1, 1), (1, 1), (0, 0)), mode="edge")[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
                take = nz * 1.15 < best_z
                best_z = np.where(take, nz, best_z)
                best_c = np.where(take[..., None], nc, best_c)
        fill = best_z * 1.15 < zb
        zb = np.where(fill, best_z, zb)
        img = np.where(fill[..., None], best_c, img)
    im = Image.fromarray((img * 255).astype(np.uint8)).resize((cam["width"], cam["height"]), Image.BILINEAR)
    return np.asarray(im)


def _write_points(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    n = len(xyz)
    rec = np.zeros(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"),
                             ("blue", "u1")])
    rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    c = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    rec["red"], rec["green"], rec["blue"] = c[:, 0], c[:, 1], c[:, 2]
    head = ("ply\nformat binary_little_endian 1.0\n"
            f"element vertex {n}\nproperty float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with open(path, "wb") as fh:
        fh.write(head.encode())
        fh.write(rec.tobytes())


def build_scene(root: Path, spec: BenchmarkSpec = BenchmarkSpec(),
                progress: Optional[Callable[[str, float], None]] = None,
                cancelled: Optional[Callable[[], bool]] = None) -> Path:
    """Create (or reuse) the benchmark dataset in ``root``. Photos are rendered and written one at a time."""
    root = Path(root)
    stamp = root / "spec.json"
    want = asdict(spec)
    try:
        if json.loads(stamp.read_text()) == want and (root / "dataset.json").is_file():
            doc = json.loads((root / "dataset.json").read_text())
            if all((root / c["image"]).is_file() for c in doc["cameras"]):
                return root
    except (OSError, ValueError, KeyError):
        pass
    shutil.rmtree(root, ignore_errors=True)
    (root / "images").mkdir(parents=True)
    say = progress or (lambda m, f: None)
    say("Building the benchmark scene", 0.0)
    rng = np.random.default_rng(7)
    pos, col = _truth(rng)
    cams = _cameras(spec)
    from PIL import Image

    for i, c in enumerate(cams):
        if cancelled is not None and cancelled():
            raise BenchmarkCancelled()
        Image.fromarray(_render(pos, col, c)).save(root / c["image"], quality=92)
        say(f"Rendering the benchmark photos ({i + 1}/{len(cams)})", (i + 1) / len(cams))
    # Starting points: the truth's centres, jittered, repeated up to the splat budget.
    k = np.arange(spec.splats) % len(pos)
    xyz = pos[k] + rng.uniform(-0.02, 0.02, (spec.splats, 3)).astype(np.float32)
    _write_points(root / "points3d.ply", xyz, col[k])
    doc = {"format": "splatdataset", "version": 1, "convention": "opencv", "upVector": [0, -1, 0],
           "sceneCenter": [0, 0, 0], "sceneExtent": 7.0, "points": "points3d.ply", "cameras": cams}
    (root / "dataset.json").write_text(json.dumps(doc), encoding="utf-8")
    stamp.write_text(json.dumps(want), encoding="utf-8")
    return root


# ---------------------------------------------------------------------------------------------- running
def measure(samples: List[tuple], measure_from: int) -> float:
    """Steps per second from (step, seconds[, seconds…]) samples, in poll order: from the first sample at or past
    ``measure_from`` to the last one (all samples when the run never got there, e.g. very short test runs).
    Each further column is another clock (the worker's ``elapsedSeconds`` or None, then this side's wall clock):
    the first one every used sample has and that advances is taken."""
    s = [x for x in samples if x[0] > 0]
    use = [x for x in s if x[0] >= measure_from] or s
    if len(use) >= 2 and use[-1][0] > use[0][0]:
        for k in range(1, len(use[0])):
            a, b = use[0][k], use[-1][k]
            if a is not None and b is not None and all(x[k] is not None for x in use) and b > a:
                return (use[-1][0] - use[0][0]) / (b - a)
    raise BenchmarkError(f"the benchmark run was too short to measure ({len(s)} progress samples)")


def _upload_zip(client, folder: Path, tmp_dir: Path) -> tuple:
    """Zip the scene and POST it (the normal remote path). Returns (datasetId, bytes, seconds)."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    z = tmp_dir / f"splat-bench-{os.getpid()}.zip"
    try:
        with zipfile.ZipFile(z, "w", zipfile.ZIP_STORED) as zf:
            for p in sorted(folder.rglob("*")):
                if p.is_file() and p.name != "spec.json":
                    zf.write(p, p.relative_to(folder).as_posix())
        size = z.stat().st_size
        t0 = time.monotonic()
        with client._req("POST", "/v1/datasets", z.read_bytes(), "application/zip", timeout=600) as resp:
            did = json.loads(resp.read())["datasetId"]
        return did, size, time.monotonic() - t0
    finally:
        try:
            z.unlink()
        except OSError:
            pass


def run(client, scene_root: Path, spec: BenchmarkSpec = BenchmarkSpec(), local: bool = True,
        progress: Optional[Callable[[str, float], None]] = None, cancelled: Optional[Callable[[], bool]] = None,
        priority: int = 1, poll: float = 1.0, device_id: str = "local", name: Optional[str] = None,
        software: Optional[str] = None, tmp_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Build the scene, train it on ``client``'s worker and return the result record (not saved).

    ``local``: register the folder with the desktop app's own server (no copy) instead of uploading a zip."""
    say = progress or (lambda m, f: None)
    stop = cancelled or (lambda: False)
    scene = build_scene(scene_root, spec, lambda m, f: say(m, 0.1 * f), stop)
    info = client.info()
    up_mbps = None
    if local:
        did = client.local_dataset(scene)["datasetId"]
    else:
        say("Uploading the benchmark scene", 0.1)
        did, nbytes, secs = _upload_zip(client, scene, Path(tmp_dir or scene.parent))
        up_mbps = nbytes / 1e6 / secs if secs > 0 else None
    job = None
    try:
        body = {"datasetId": did, "config": spec.training_config(), "name": "SplatWork benchmark"}
        if local:
            body["priority"] = priority
        job = client.post("/v1/jobs", body)["jobId"]
        samples: List[tuple] = []
        t_start = time.monotonic()
        while True:
            if stop():
                try:
                    client.cancel(job)
                except Exception:  # noqa: BLE001
                    pass
                raise BenchmarkCancelled()
            s = client.job(job)
            step = int(s.get("step") or 0)
            if s.get("state") == "training" and step > 0:
                el = s.get("elapsedSeconds")
                samples.append((step, float(el) if isinstance(el, (int, float)) else None,
                                time.monotonic() - t_start))
            if s.get("state") in ("completed", "failed", "cancelled"):
                break
            where = "this PC" if local else (info.get("name") or "the worker")
            say(f"Training on {where} — step {step:,} / {spec.steps:,}" if step else
                f"Training on {where}: {s.get('message') or s.get('state')}", 0.1 + 0.85 * step / spec.steps)
            time.sleep(poll)
        if s["state"] != "completed":
            raise BenchmarkError(f"benchmark job {s['state']}: {s.get('message') or ''}".strip())
        rate = measure(samples, spec.measure_from)
        down_mbps = None
        if not local:
            say("Downloading the result", 0.96)
            import tempfile

            with tempfile.TemporaryDirectory(dir=tmp_dir) as td:
                t0 = time.monotonic()
                p = client.download(f"/v1/jobs/{job}/result.ply", Path(td) / "r.ply")
                secs = time.monotonic() - t0
                down_mbps = p.stat().st_size / 1e6 / secs if secs > 0 else None
        gpus = info.get("gpus") or []
        say("Done", 1.0)
        return {"deviceId": device_id, "name": name or info.get("name") or "This PC",
                "date": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
                "gpu": (gpus[0].get("name") if gpus and isinstance(gpus[0], dict) else None),
                "software": software or f"SplatWork for Linux {info.get('workerVersion') or ''} · "
                                        f"{info.get('backend') or ''}".strip(" ·"),
                "trainStepsPerSecond": round(rate, 3), "splats": spec.splats,
                "megapixels": round(spec.megapixels, 2), "uploadMBps": up_mbps, "downloadMBps": down_mbps}
    finally:
        if job is not None:
            try:
                client.delete(f"/v1/jobs/{job}")
            except Exception:  # noqa: BLE001
                pass
        client.delete_dataset(did)
