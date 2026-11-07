"""CLI / progress protocol (PROTOCOL §6) with a mock backend loaded via SPLATPREP_PLUGINS."""

import json
import subprocess
import sys

import numpy as np

from splatprep import models
from splatprep.runner import run_request


def _run_cli(tmp_path, request, env):
    (tmp_path / "request.json").write_text(json.dumps(request))
    env = dict(env, SPLATPREP_PLUGINS="mock_backend:BACKENDS")
    p = subprocess.run(
        [sys.executable, "-m", "splatprep", "run", "--request", "request.json", "--progress-jsonl", "-"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    lines = [json.loads(l) for l in p.stdout.splitlines() if l.strip()]  # every stdout line must be JSON
    return p, lines


def _prepare(tmp_path, synth_small):
    import shutil

    shutil.copytree(synth_small / "images", tmp_path / "images")
    return {
        "task": "sfm",
        "imagesDir": "images",
        "outputDir": "out",
        "features": "mock",
        "mockGroundTruth": str(synth_small / "dataset.json"),
    }


def test_cli_success_protocol(tmp_path, synth_small, child_env):
    req = _prepare(tmp_path, synth_small)
    p, lines = _run_cli(tmp_path, req, child_env)
    assert p.returncode == 0, p.stderr
    assert "stray stdout noise" in p.stderr  # redirected away from the protocol stream
    progress = [l for l in lines if "stage" in l]
    assert progress and all(set(l) == {"stage", "progress", "message"} for l in progress)
    values = [l["progress"] for l in progress]
    assert values == sorted(values) and 0 <= values[0] and values[-1] <= 1
    final = lines[-1]
    assert final["done"] is True
    stats = final["stats"]
    assert stats["numRegistered"] == 8 and stats["registeredPct"] == 100.0 and stats["status"] == "ok"
    ds = json.loads((tmp_path / "out" / "dataset.json").read_text())
    assert len(ds["cameras"]) == 8 and (tmp_path / "out" / "points3d.ply").exists()
    assert all((tmp_path / "out" / c["image"]).exists() for c in ds["cameras"])


def test_cli_prior_alignment(tmp_path, synth_small, child_env):
    """Mock output is GT under an unknown Sim3; priorPoses = GT must undo it exactly."""
    req = _prepare(tmp_path, synth_small)
    req["priorPoses"] = str(synth_small / "dataset.json")
    p, lines = _run_cli(tmp_path, req, child_env)
    assert p.returncode == 0, p.stderr
    stats = lines[-1]["stats"]
    assert stats["priorAlignment"]["rmse"] < 1e-6
    out = {c["image"]: np.array(c["cameraToWorld"]) for c in json.loads((tmp_path / "out" / "dataset.json").read_text())["cameras"]}
    gt = {c["image"]: np.array(c["cameraToWorld"]) for c in json.loads((synth_small / "dataset.json").read_text())["cameras"]}
    for k in gt:
        assert np.allclose(out[k], gt[k], atol=1e-6)


def test_cli_model_unavailable(tmp_path, synth_small, child_env):
    req = _prepare(tmp_path, synth_small)
    req["features"] = "mast3r"
    empty = tmp_path / "no-mast3r"
    empty.mkdir()
    # point at an empty checkout so the repo is never found (even when the Models pack installed it)
    p, lines = _run_cli(tmp_path, req, dict(child_env, SPLATPREP_MAST3R_PATH=str(empty)))
    assert p.returncode == 2
    assert lines[-1]["error"].startswith("model unavailable: mast3r")


def test_cli_bad_request(tmp_path, child_env):
    p, lines = _run_cli(tmp_path, {"task": "nope"}, child_env)
    assert p.returncode == 2 and "unknown task" in lines[-1]["error"]


def test_run_request_in_process(tmp_path, synth_small):
    from mock_backend import MockBackend

    models.register(MockBackend())
    try:
        req = _prepare(tmp_path, synth_small)
        req["mockDrop"] = 2
        events = []
        stats = run_request(req, tmp_path, events.append)
        assert stats["numRegistered"] == 6 and stats["registeredPct"] == 75.0
        assert "wallTimeSec" in stats and "peakMemoryMB" in stats
        assert any(e["stage"] == "export" for e in events)
        assert "mock" in models.available_models()
    finally:
        models.unregister("mock")


def test_available_models_api():
    ids = models.available_models()
    assert isinstance(ids, list) and all(isinstance(i, str) for i in ids)
    st = models.model_status()
    for mid in models.FEATURE_MODEL_IDS + models.MASK_MODEL_IDS:
        assert mid in st
    cards = models.model_cards()
    for mid in models.FEATURE_MODEL_IDS + models.MASK_MODEL_IDS:
        assert mid in cards["models"], mid
        for key in ("displayName", "summary", "strengths", "weaknesses", "bestFor", "speed", "memory", "licence", "installSize"):
            assert key in cards["models"][mid], (mid, key)
