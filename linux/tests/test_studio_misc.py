import numpy as np

from splatworker.studio import autoclean
from splatworker.studio.presets import PRESETS, ProjectOptions, training_config


def test_autoclean_rules_match_mac():
    n = 1000
    peak = np.full(n, 0.5, np.float32)
    views = np.full(n, 40, np.uint32)
    peak[:100] = 0.01                  # invisible
    views[100:150] = 5                 # floaters: < 25 % of 40
    p = autoclean.plan(peak, views, np.full(n, 3.0))
    assert p.invisible == 100 and p.few_views == 50 and p.typical_views == 40
    assert p.remove[:150].all() and not p.remove[150:].any()
    q = autoclean.plan(peak, views, np.full(n, 3.0), include_few_views=False)
    assert q.few_views == 0 and q.total == 100


def test_autoclean_small_capture_skips_few_views():
    peak = np.full(100, 0.5, np.float32)
    views = np.full(100, 8, np.uint32)
    views[:10] = 1
    p = autoclean.plan(peak, views, np.zeros(100))
    assert p.typical_views == 0 and p.few_views == 0


def test_presets_and_requests():
    assert set(PRESETS) == {"preview", "balanced", "high", "ultra"}
    cfg = training_config("ultra", {"seed": 7})
    assert cfg["iterations"] == 40_000 and cfg["maxGaussians"] == 3_000_000 and cfg["seed"] == 7
    o = ProjectOptions(name="x", source_kind="video360", source="/v.mp4", views_360=8, target_frames=200)
    r = o.frames_request("v.mp4")
    assert r["equirect"]["rings"][0]["count"] == 8 and r["videoOptions"]["targetFrameCount"] == 200
    s = o.sfm_request(True, True, True)
    assert s["masksDir"] == "masks" and s["priorMode"] == "auto"
    assert ProjectOptions.from_dict({**o.to_dict(), "unknown": 1}) == o
    from splatworker.config import TrainingConfig

    TrainingConfig.from_dict(o.train_config())       # every key is valid for the worker
