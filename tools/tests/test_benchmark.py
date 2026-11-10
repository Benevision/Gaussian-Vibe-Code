"""End-to-end benchmark with the dependency-light ``opencv_sift`` backend (+ mock for pose metrics)."""

import json

import pytest

from splatprep import models
from splatprep.benchmark import load_dataset, run_benchmark
from splatprep.synth import generate


@pytest.fixture(scope="module")
def walk_ds(tmp_path_factory):
    d = tmp_path_factory.mktemp("bench") / "walk"
    generate(d, "walk", num_views=6, width=320, height=240, seed=5, supersample=1, photometric_strength=0.2)
    return d


def _check_schema(rep):
    assert rep["format"] == "splatbenchmark" and rep["version"] == 1
    for k in ("createdAt", "host", "datasets", "results", "summary", "models"):
        assert k in rep
    json.dumps(rep)


def test_benchmark_opencv_sift_subprocess(tmp_path, walk_ds):
    rep = run_benchmark([walk_ds], tmp_path / "bench", models=["opencv_sift", "loftr"], options={"maxKeypoints": 2048})
    _check_schema(rep)
    assert (tmp_path / "bench" / "benchmark.json").exists() and (tmp_path / "bench" / "benchmark.md").exists()
    by_model = {r["model"]: r for r in rep["results"]}
    sift = by_model["opencv_sift"]
    assert sift["status"] in ("ok", "matching-only"), sift
    m = sift["metrics"]
    assert m["wallTimeSec"] > 0 and m["peakMemoryMB"] > 0
    assert m["matching"]["numPairsEvaluated"] == 15  # exhaustive on 6 images
    # forward walk on a textured scene: most SIFT matches are geometrically correct
    assert m["matching"]["meanVerifiedPrecision"] > 0.7
    assert m["matching"]["pairPoseAucAt10"] > 0.2
    if sift["status"] == "ok":  # pycolmap / colmap installed
        assert m["registeredPct"] > 50 and "pose" in m
    loftr = by_model["loftr"]
    if loftr["status"] == "unavailable":
        assert "kornia" in loftr["error"] or "torch" in loftr["error"]
    assert ds_name(rep) in (tmp_path / "bench" / "benchmark.md").read_text()


def ds_name(rep):
    return rep["datasets"][0]["name"]


def test_benchmark_pose_metrics_with_mock(tmp_path, synth_small):
    from mock_backend import MockBackend

    models.register(MockBackend())
    try:
        rep = run_benchmark([synth_small], tmp_path / "b", models=["mock"], in_process=True)
    finally:
        models.unregister("mock")
    r = rep["results"][0]
    assert r["status"] == "ok", r
    p = r["metrics"]["pose"]
    assert p["aucAt5"] == pytest.approx(1.0) and p["ateRmse"] < 1e-6
    assert rep["summary"]["mock"]["meanAucAt10"] == pytest.approx(1.0)
    assert rep["datasets"][0]["groundTruth"] == "synthetic"


def test_benchmark_child_ignores_modules_in_run_dir(tmp_path, walk_ds):
    """Python files in a run folder (which can come from an uploaded dataset) are never imported by the child."""
    from splatprep.benchmark import run_one

    ds = load_dataset(walk_ds, tmp_path / "scratch")
    run_dir = tmp_path / "runs" / ds.name / "opencv_sift"
    marker = tmp_path / "planted-code-ran"
    code = f"open({str(marker)!r}, 'w').write('x')\n"
    (run_dir / "splatprep").mkdir(parents=True)
    (run_dir / "splatprep" / "__init__.py").write_text(code)
    (run_dir / "splatprep" / "__main__.py").write_text(code)
    (run_dir / "json.py").write_text(code)  # shadows the stdlib module
    r = run_one(ds, "opencv_sift", run_dir, {"maxKeypoints": 2048}, False, lambda msg: None, timeout=600)
    assert not marker.exists()
    assert r["status"] in ("ok", "matching-only"), r


def test_load_splatcapture(tmp_path):
    import numpy as np
    from PIL import Image

    cap = tmp_path / "Kitchen.splatcapture"
    (cap / "images").mkdir(parents=True)
    frames = []
    for i in range(3):
        Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(cap / "images" / f"{i:06d}.jpg")
        M = np.eye(4)
        M[0, 3] = i
        frames.append({"index": i, "image": f"images/{i:06d}.jpg", "intrinsics": [10, 10, 4, 4], "cameraToWorld": M.T.reshape(-1).tolist(), "trackingState": "normal"})
    (cap / "manifest.json").write_text(json.dumps({"format": "splatcapture", "version": 1, "frames": frames}))
    import zipfile

    z = tmp_path / "Kitchen.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for p in cap.rglob("*"):
            zf.write(p, p.relative_to(tmp_path))
    for src in (cap, z):
        ds = load_dataset(src, tmp_path / "scratch")
        assert ds.gt_source == "arkit" and len(ds.gt_c2w) == 3
        assert ds.names == ["000000.jpg", "000001.jpg", "000002.jpg"]
        assert np.allclose(ds.gt_c2w["000002.jpg"][:3, 3], [2, 0, 0])


def test_merge_reports_replaces_rows_and_resummarises():
    from splatprep.benchmark import merge_reports

    def row(model, ds, auc):
        return {"model": model, "dataset": ds, "status": "ok",
                "metrics": {"registeredPct": 100.0, "wallTimeSec": 1.0, "pose": {"aucAt5": auc, "aucAt10": auc}}}
    a = {"datasets": [{"name": "orbit"}], "models": ["m1", "m2"], "results": [row("m1", "orbit", 0.1), row("m2", "orbit", 0.2)]}
    b = {"datasets": [{"name": "orbit"}], "models": ["m2"], "results": [row("m2", "orbit", 0.9)]}
    m = merge_reports([a, b])
    assert [r["metrics"]["pose"]["aucAt5"] for r in m["results"]] == [0.1, 0.9]
    assert m["mergedFrom"] == 2 and set(m["summary"]) == {"m1", "m2"}
