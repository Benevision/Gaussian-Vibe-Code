"""Orchestration of the COLMAP CLI path (colmap_sift and feature backends + `colmap mapper`) with a fake colmap."""

import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from splatprep import mapper as mapper_mod
from splatprep.runner import run_request

TESTS = Path(__file__).resolve().parent
TOOLS = TESTS.parent


@pytest.fixture
def fake_colmap(tmp_path, monkeypatch, synth_small):
    script = tmp_path / "bin" / ("colmap.cmd" if sys.platform == "win32" else "colmap")
    script.parent.mkdir()
    if sys.platform == "win32":
        script.write_text(f'@set "PYTHONPATH={TOOLS}"\r\n@"{sys.executable}" "{TESTS / "fake_colmap.py"}" %*\r\n')
    else:
        script.write_text(f'#!/bin/sh\nPYTHONPATH="{TOOLS}" exec "{sys.executable}" "{TESTS / "fake_colmap.py"}" "$@"\n')
        script.chmod(0o755)
    log = tmp_path / "colmap_calls.jsonl"
    monkeypatch.setenv("SPLATPREP_COLMAP", str(script))
    monkeypatch.setenv("SPLATPREP_GLOMAP", str(tmp_path / "missing_glomap"))
    system = os.environ.get("SystemRoot", r"C:\Windows") + r"\System32" if sys.platform == "win32" else "/usr/bin:/bin"
    monkeypatch.setenv("PATH", str(tmp_path / "bin") + os.pathsep + system)  # hide any real glomap
    monkeypatch.setenv("FAKE_COLMAP_LOG", str(log))
    monkeypatch.setenv("FAKE_COLMAP_GT", str(synth_small / "dataset.json"))
    mapper_mod.colmap_commands.cache_clear()
    mapper_mod.colmap_options.cache_clear()
    shutil.copytree(synth_small / "images", tmp_path / "work" / "images")
    yield log
    mapper_mod.colmap_commands.cache_clear()
    mapper_mod.colmap_options.cache_clear()


def _calls(log):
    return [json.loads(l) for l in log.read_text().splitlines()]


def _check_output(out, synth_small):
    ds = json.loads((out / "dataset.json").read_text())
    gt = {c["image"]: np.array(c["cameraToWorld"]) for c in json.loads((synth_small / "dataset.json").read_text())["cameras"]}
    assert len(ds["cameras"]) == len(gt)
    for c in ds["cameras"]:
        assert np.allclose(np.array(c["cameraToWorld"]), gt[c["image"]], atol=1e-6)


def test_colmap_sift_pipeline(fake_colmap, tmp_path, synth_small, monkeypatch):
    monkeypatch.setenv("FAKE_COLMAP_NO_CHECK", "1")  # fake matchers write no geometries
    work = tmp_path / "work"
    stats = run_request(
        {"task": "sfm", "features": "colmap_sift", "matching": "exhaustive", "imagesDir": "images", "outputDir": "out", "exportImages": False},
        work,
    )
    assert stats["status"] == "ok" and stats["mapper"] == "colmap" and stats["registeredPct"] == 100.0
    calls = _calls(fake_colmap)
    cmds = [c[0] for c in calls if "-h" not in c and c[0] != "help"]
    assert cmds == ["feature_extractor", "exhaustive_matcher", "mapper"]
    fe = next(c for c in calls if c[0] == "feature_extractor" and "-h" not in c)
    assert fe[fe.index("--FeatureExtraction.use_gpu") + 1] == "0"
    assert fe[fe.index("--ImageReader.camera_model") + 1] == "SIMPLE_RADIAL"
    _check_output(work / "out", synth_small)


def test_colmap_sift_with_priors_uses_pair_list(fake_colmap, tmp_path, synth_small, monkeypatch):
    monkeypatch.setenv("FAKE_COLMAP_NO_CHECK", "1")
    work = tmp_path / "work"
    stats = run_request(
        {"task": "sfm", "features": "colmap_sift", "imagesDir": "images", "outputDir": "out", "priorPoses": str(synth_small / "dataset.json"), "exportImages": False},
        work,
    )
    assert stats["status"] == "ok" and stats["priorAlignment"]["rmse"] < 1e-6
    calls = _calls(fake_colmap)
    assert any(c[0] == "matches_importer" for c in calls)
    # known poses are refined, not re-solved: triangulate + bundle-adjust, no mapper
    assert stats["mapper"] == "prior-refine" and not any(c[0] == "mapper" for c in calls)
    assert sum(c[0] == "bundle_adjuster" for c in calls) == 2
    assert stats["priorRefine"]["accepted"] and stats["priorRefine"]["medianRotationChangeDeg"] < 1e-3


def test_prior_mode_solve_still_runs_the_mapper(fake_colmap, tmp_path, synth_small, monkeypatch):
    monkeypatch.setenv("FAKE_COLMAP_NO_CHECK", "1")
    stats = run_request(
        {"task": "sfm", "features": "colmap_sift", "imagesDir": "images", "outputDir": "out", "mapper": "colmap",
         "priorPoses": str(synth_small / "dataset.json"), "priorMode": "solve", "exportImages": False},
        tmp_path / "work",
    )
    assert stats["status"] == "ok" and stats["mapper"] != "prior-refine"


def test_opencv_sift_full_sfm_via_colmap_mapper(fake_colmap, tmp_path, monkeypatch):
    """Feature backend writes keypoints/matches/two-view geometries; the (fake) mapper reads the DB."""
    from splatprep.synth import generate

    gt_dir = tmp_path / "walk"
    generate(gt_dir, "walk", num_views=8, width=320, height=240, seed=2, supersample=1, photometric_strength=0.2)
    monkeypatch.setenv("FAKE_COLMAP_GT", str(gt_dir / "dataset.json"))
    work = tmp_path / "work2"
    shutil.copytree(gt_dir / "images", work / "images")
    synth_small = gt_dir
    stats = run_request(
        {"task": "sfm", "features": "opencv_sift", "imagesDir": "images", "outputDir": "out", "mapper": "colmap", "keepWork": True},
        work,
    )
    assert stats["status"] == "ok" and stats["mapper"] == "colmap"
    assert stats["numVerifiedPairs"] > 0
    _check_output(work / "out", synth_small)
    # images were copied (pinhole -> no undistortion needed since k = 0)
    assert stats["export"]["undistortion"] == "copy"
    assert len(list((work / "out" / "images").iterdir())) == 8
    from splatprep.colmap_db import COLMAPDatabase

    db = COLMAPDatabase(work / ".splatprep_work_out" / "database.db")
    kp = db.keypoints(1)
    assert kp.shape[1] == 2 and len(kp) > 50
    tvg = db.two_view_geometries()
    assert tvg and all(v["config"] == 3 for v in tvg.values())
    db.close()
