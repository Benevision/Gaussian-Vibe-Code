"""Dataset preparation pieces done in the GUI process: 360° photos → views, fixed parts of the frame, the mask
editor's layer, stereo splitting / rig / depth."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from splatworker.studio import equirect, importers, static_regions

TOOLS = Path(__file__).resolve().parents[2] / "tools"


def _splatprep_frames():
    sys.path.insert(0, str(TOOLS))
    try:
        from splatprep import frames  # type: ignore
    finally:
        sys.path.remove(str(TOOLS))
    return frames


def test_equirect_views_match_the_worker_and_the_mac_layout():
    o = {"fovDegrees": 90.0, "outputSize": 32,
         "rings": [{"pitch": 0, "count": 8}, {"pitch": 35, "count": 4, "yawOffset": 22.5},
                   {"pitch": -35, "count": 4, "yawOffset": 22.5}],
         "includeNadir": False, "maskBelowLatitude": -60.0}
    vs = equirect.views(o)
    assert len(vs) == 16 and vs[0]["focal"] == pytest.approx(16.0)
    assert [v["yaw"] for v in vs[:3]] == [0.0, 45.0, 90.0] and vs[8]["yaw"] == 22.5 and vs[8]["pitch"] == 35
    fr = _splatprep_frames()
    ref = fr.equirect_views({**fr.EQUIRECT_DEFAULTS, **o})
    for a, b in zip(vs, ref):
        np.testing.assert_allclose(a["R"], b["R"], atol=1e-12)
        ca, ma = equirect.lookup(a, 256, 128, o)
        cb, mb = fr.view_lookup(b, 256, 128, {**fr.EQUIRECT_DEFAULTS, **o})
        np.testing.assert_allclose(ca, cb, atol=1e-4)
        assert (ma is None) == (mb is None) and (ma is None or (ma == mb).all())


def test_equirect_geometry_and_seam():
    v = equirect.views({"outputSize": 33, "fovDegrees": 90, "rings": [{"pitch": 0, "count": 4}]})
    W, H = 400, 200
    c, _ = equirect.lookup(v[1], W, H, {})          # yaw 90° = looking right: equirect x = 3/4 W
    assert c[16, 16, 0] == pytest.approx(0.75 * W, abs=0.01) and c[16, 16, 1] == pytest.approx(H / 2, abs=0.01)
    up = equirect.views({"outputSize": 33, "fovDegrees": 90, "rings": [{"pitch": 35, "count": 1}]})[0]
    c, _ = equirect.lookup(up, W, H, {})
    assert (0.5 - c[16, 16, 1] / H) * 180 == pytest.approx(35, abs=0.1)        # latitude of the centre = pitch
    # the camera's +x (right) column maps to increasing longitude; +y (down) to decreasing latitude
    c, _ = equirect.lookup(v[0], W, H, {})
    assert c[16, 30, 0] > c[16, 2, 0] and c[30, 16, 1] > c[2, 16, 1]
    # yaw 180° straddles the seam: bilinear sampling wraps instead of smearing the border
    img = np.zeros((H, W, 3), np.uint8)
    img[:, :W // 2] = 200
    back = equirect.remap(img, equirect.lookup(v[2], W, H, {})[0])
    assert back[16, 0].mean() > 150 or back[16, -1].mean() > 150
    assert back[16, :12].mean() < 50 and back[16, -12:].mean() > 150   # left of the view = x < W/2 side is dark


def test_split_photos_writes_views_masks_and_a_rig(tmp_path):
    src = tmp_path / "pano"
    src.mkdir()
    rng = np.random.default_rng(1)
    for i in range(2):
        Image.fromarray((rng.random((64, 128, 3)) * 255).astype(np.uint8)).save(src / f"P{i}.jpg")
    Image.fromarray(np.zeros((50, 50, 3), np.uint8)).save(src / "square.jpg")          # not 2:1 → skipped
    ds = tmp_path / "ds"
    o = {"fovDegrees": 90, "outputSize": 24, "rings": [{"pitch": 0, "count": 4}, {"pitch": -35, "count": 2}],
         "maskBelowLatitude": -60.0}
    st = importers.import_photos360(src, ds, o)
    assert st["photos"] == 2 and st["images"] == 12 and st["rejected"][0]["item"] == "square.jpg"
    names = sorted(p.name for p in (ds / "images").iterdir())
    assert names[0] == "000000_v00.jpg" and len(names) == 12
    masks = sorted((ds / "splatprep-masks" / "frames" / "masks").iterdir())
    assert {p.stem[-3:] for p in masks} == {"v04", "v05"}                # only the downward ring sees the floor
    m = np.asarray(Image.open(masks[0]))
    assert m[-1, 12] == 0 and m[0, 12] == 255
    rig = importers.rig_from_frames(ds)
    assert rig["kind"] == "multiview" and len(rig["groups"]) == 2 and len(rig["groups"][0]) == 6
    assert rig["intrinsics"] == pytest.approx([12.0, 12.0, 12.0, 12.0])


# ------------------------------------------------------------------------------------------ static regions

def test_static_regions_mask_a_fixed_overlay(tmp_path):
    ds = tmp_path / "ds"
    (ds / "images").mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(16):
        img = (rng.random((96, 192)) * 255).astype(np.uint8)       # the scene changes every frame
        img[70:90, 10:60] = 0                                       # a black logo with a white bar in it
        img[78:82, 15:55] = 255
        Image.fromarray(img).convert("RGB").save(ds / "images" / f"{i:06d}.png")
    reps = static_regions.apply(ds)
    assert len(reps) == 1 and reps[0]["applied"] and 0.002 < reps[0]["staticFraction"] < 0.25
    m = np.asarray(Image.open(ds / static_regions.LAYER / "000003.png"))
    assert m.shape == (96, 192) and m[80, 30] == 0 and m[20, 120] == 255
    # too few frames: nothing
    ds2 = tmp_path / "ds2"
    (ds2 / "images").mkdir(parents=True)
    for i in range(5):
        Image.fromarray(np.zeros((20, 40, 3), np.uint8)).save(ds2 / "images" / f"{i}.png")
    assert not static_regions.apply(ds2)[0]["applied"]
    assert not (ds2 / static_regions.LAYER).exists()


def test_static_regions_camera_keys():
    assert static_regions.camera_key("000001_v05.jpg", (10, 10), {"rotationToRig": [[1]]}) == "_v05"
    assert static_regions.camera_key("000001_L.jpg", (10, 10), {"eye": "left"}) == "eye left"
    assert static_regions.camera_key("a.jpg", (10, 20), None) == "10x20"
