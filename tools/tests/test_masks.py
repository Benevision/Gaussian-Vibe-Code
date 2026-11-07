import shutil

import numpy as np
from PIL import Image

from splatprep.masks.sky import heuristic_sky
from splatprep.runner import run_request


def test_heuristic_sky_on_synthetic():
    """Compare the heuristic against the renderer's exact sky (depth = inf) for a view facing away from the wall."""
    from splatprep.geometry import look_at
    from splatprep.synth import make_textures, photometric, render

    W, H = 160, 120
    K = np.array([[120.0, 0, W / 2], [0, 120.0, H / 2], [0, 0, 1]])
    lin, depth = render(look_at([0, 1.5, -2.0], [0, 2.2, 8.0]), K, W, H, make_textures(np.random.default_rng(0), 128))
    img = photometric(lin, np.random.default_rng(1), 0.3)
    gt = ~np.isfinite(depth)
    sky = heuristic_sky(img)
    assert sky.shape == img.shape[:2]
    assert gt.mean() > 0.2
    iou = (sky & gt).sum() / max(1, (sky | gt).sum())
    assert iou > 0.8, iou


def test_masks_task_writes_pngs(tmp_path, synth_small, monkeypatch):
    monkeypatch.setenv("SPLATPREP_SKY_HEURISTIC", "1")
    shutil.copytree(synth_small / "images", tmp_path / "images")
    events = []
    stats = run_request({"task": "masks", "maskModel": "sky", "imagesDir": "images", "outputDir": "out"}, tmp_path, events.append)
    assert stats["status"] == "ok" and stats["numMasks"] == 8
    pngs = sorted((tmp_path / "out" / "masks").glob("*.png"))
    assert len(pngs) == 8
    m = np.asarray(Image.open(pngs[0]))
    assert m.dtype == np.uint8 and set(np.unique(m)) <= {0, 255}
    assert m[-1].mean() > 200  # ground is kept (255)
    assert events[-1]["progress"] == 1.0
