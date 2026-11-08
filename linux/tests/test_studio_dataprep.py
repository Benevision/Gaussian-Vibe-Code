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


# ------------------------------------------------------------------------------------------ mask editor

def _masked_dataset(root: Path, n=3, w=40, h=30):
    from conftest import make_dataset

    make_dataset(root, n_cams=n, width=w, height=h, depth=False, points=False)
    layer = root / "splatprep-masks" / "person" / "masks"
    layer.mkdir(parents=True)
    m = np.full((h, w), 255, np.uint8)
    m[0:10, 0:10] = 0                                   # the model removed the top-left corner of image 0
    Image.fromarray(m).save(layer / "000000.png")
    masks = importers.stack_masks(root, [layer], dilate=0)
    importers.finish_dataset(root, masks, {})
    return root


def test_mask_editor_paint_restore_undo_save_and_restack(tmp_path):
    from splatworker.studio.mask_edit import MaskEditModel

    ds = _masked_dataset(tmp_path / "ds")
    m = MaskEditModel(ds)
    assert len(m) == 3 and m.has_mask(0) and not m.has_mask(1)
    assert not m.keep()[5, 5] and m.keep()[20, 30]
    m.paint(30, 20, 3, remove=True)                    # paint out a disk
    m.paint(5, 5, 2, remove=False)                     # paint back inside the model's mask
    k = m.keep()
    assert not k[20, 30] and k[5, 5] and not k[8, 8]
    assert m.dirty == {0}
    m.select(1)
    m.paint(10, 10, 2, remove=True)
    assert m.undo() and m.index == 1 and m.keep(1)[10, 10]
    m.select(0)
    assert not m.keep()[20, 30]                        # image 0's edit survived switching (packed)
    st = m.save()
    assert st["saved"] >= 1
    man = np.asarray(Image.open(ds / importers.MANUAL_LAYER / "masks" / "000000.png"))
    res = np.asarray(Image.open(ds / importers.MANUAL_LAYER / "restore" / "000000.png"))
    assert man[20, 30] == 0 and man[5, 5] == 255 and res[5, 5] == 255 and res[20, 30] == 0
    stacked = np.asarray(Image.open(ds / "masks" / "000000.png"))
    assert stacked[20, 30] == 0 and stacked[5, 5] == 255 and stacked[8, 8] == 0
    doc = json.loads((ds / "dataset.json").read_text())
    assert doc["cameras"][0]["mask"] == "masks/000000.png" and "mask" not in doc["cameras"][1]
    # the model layer is recomputed by "Prepare again": the manual layer still applies on top
    m2 = MaskEditModel(ds)
    assert not m2.keep(0)[20, 30] and m2.keep(0)[5, 5]
    m2.clear_frame()
    m2.save()
    stacked = np.asarray(Image.open(ds / "masks" / "000000.png"))
    assert stacked[20, 30] == 255 and stacked[5, 5] == 0          # back to the model's mask only
    assert not (ds / importers.MANUAL_LAYER / "masks" / "000000.png").exists()


def test_mask_editor_sam_clicks_and_tracking_results(tmp_path):
    from splatworker.studio.mask_edit import MaskEditModel

    ds = _masked_dataset(tmp_path / "ds")
    m = MaskEditModel(ds)
    m.select(2)
    m.add_point(20, 15, True)
    m.add_point(2, 2, False)
    req = m.preview_request("out")
    assert req["task"] == "maskpreview" and req["previewImage"] == "000002.png"
    assert req["maskPrompts"][0]["points"] == [[20.0, 15.0, 1], [2.0, 2.0, 0]]
    obj = np.zeros((30, 40), np.uint8)
    obj[12:18, 15:25] = 255
    Image.fromarray(obj).save(tmp_path / "preview.png")
    assert m.apply_object(tmp_path / "preview.png") == pytest.approx(60 / 1200)
    assert not m.keep()[15, 20] and m.keep()[5, 5]
    m.reset_clicks()
    assert m.keep()[15, 20] and not m.points
    m.add_point(20, 15, True)
    tr = m.track_request("t")
    assert tr["maskModel"] == "sam2_track" and tr["maskPrompts"] == [
        {"image": "000002.png", "points": [[20.0, 15.0, 1]], "objectId": 1}]
    td = tmp_path / "track" / "masks"
    td.mkdir(parents=True)
    keep = np.full((30, 40), 255, np.uint8)
    keep[0:5, 30:40] = 0
    for i in range(3):
        Image.fromarray(keep).save(td / f"{i:06d}.png")
    assert m.apply_tracked(td) == 3 and m.dirty == {0, 1, 2} and not m.tracked
    m.save()
    for i in range(3):
        assert np.asarray(Image.open(ds / "masks" / f"{i:06d}.png"))[2, 35] == 0


def test_processing_validates_new_tasks():
    from splatworker.processing import ProcessingError, validate_request

    r = validate_request({"task": "maskpreview", "maskPrompts": [{"image": "a.jpg", "points": [[1, 2, 1]]}],
                          "previewImage": "a.jpg"})
    assert r["imagesDir"] == "images"
    with pytest.raises(ProcessingError):
        validate_request({"task": "maskpreview"})
    with pytest.raises(ProcessingError):
        validate_request({"task": "maskpreview", "maskPrompts": [{}], "previewImage": "../x.jpg"})
    r = validate_request({"task": "frames", "video": "a.mp4", "stereo": {"layout": "separateFiles",
                                                                         "rightVideo": "b.mp4"}})
    assert r["stereo"]["rightVideo"] == "b.mp4"
    with pytest.raises(ProcessingError):
        validate_request({"task": "frames", "video": "a.mp4", "stereo": {"rightVideo": "/etc/passwd"}})
    with pytest.raises(ProcessingError):
        validate_request({"task": "frames", "video": "a.mp4", "stereo": "sbs"})


# ------------------------------------------------------------------------------------------ stereo

def test_stereo_disparity_and_depth():
    from splatworker.studio import stereo

    rng = np.random.default_rng(2)
    tex = (rng.random((60, 160)) * 255).astype(np.uint8)
    left, right = tex[:, 20:140], tex[:, 27:147]                       # 7 px disparity everywhere
    d = stereo.disparity(left, right, {"maxDisparity": 24, "blockRadius": 3})
    inner = d[8:-8, 30:-8]
    assert (inner > 0).mean() > 0.9 and np.median(inner[inner > 0]) == pytest.approx(7, abs=0.2)
    assert (d[:, :5] == 0).all()                                       # no match left of the search range
    z = stereo.depth_map(np.dstack([left] * 3), np.dstack([right] * 3), focal=100.0, baseline=0.07,
                         o={"maxDisparity": 24, "blockRadius": 3})
    assert np.median(z[z > 0]) == pytest.approx(1.0, rel=0.03)         # f·b/d = 100 · 0.07 / 7
    # featureless: no confident matches
    flat = np.full((40, 80), 128, np.uint8)
    zf = stereo.depth_map(np.dstack([flat] * 3), np.dstack([flat] * 3), 100.0, 0.07, {"maxDisparity": 16})
    assert (zf == 0).mean() > 0.9                                      # (disparity ~0 → beyond maxDepth)


def _stereo_ds(root: Path, n=5, baseline=0.064, solved_gap=0.5, parallel=True):
    (root / "images").mkdir(parents=True)
    frames, cams = [], []
    for i in range(n):
        for k, (eye, t) in enumerate((("left", "L"), ("right", "R"))):
            name = f"{i:06d}_{t}.png"
            Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(root / "images" / name)
            frames.append({"image": f"images/{name}", "group": i, "eye": eye, "width": 8, "height": 8})
            if eye == "right" and i == 3:
                continue                                                  # the solver missed this right eye
            m = np.eye(4)
            m[:3, 3] = [i * 2.0 + k * solved_gap * (1 + 0.01 * i), 1.0, 0.0]
            cams.append({"id": len(cams), "image": f"images/{name}", "width": 8, "height": 8, "fx": 8.0,
                         "fy": 8.0, "cx": 4.0, "cy": 4.0, "cameraToWorld": m.tolist()})
    (root / "frames.json").write_text(json.dumps({"frames": frames, "stereoLayout": "sideBySide",
                                                  "rig": {"baseline": baseline, "parallel": parallel}}))
    (root / "dataset.json").write_text(json.dumps({"format": "splatdataset", "version": 1, "cameras": cams,
                                                   "sceneCenter": [1, 1, 1], "sceneExtent": 10.0,
                                                   "points": "points3d.ply"}))
    sys.path.insert(0, str(TOOLS))
    try:
        from splatprep.ply import write_points_ply  # type: ignore
    finally:
        sys.path.remove(str(TOOLS))
    write_points_ply(root / "points3d.ply", np.array([[2.0, 4.0, 6.0]], np.float32))
    return root


def test_stereo_metric_scale_and_rig_enforcement(tmp_path):
    from splatworker.studio import stereo

    ds = _stereo_ds(tmp_path / "ds")
    rig = stereo.sfm_rig(ds)
    assert rig["kind"] == "multiview" and len(rig["groups"]) == 5
    assert rig["groups"][0]["000000_R.png"]["t"] == [0.064, 0.0, 0.0]
    st = stereo.apply_constraints(ds)
    assert st["scale"] == pytest.approx(0.064 / 0.505, rel=1e-6)     # median of the inner 50 % (Mac rule)
    assert st["enforced"] == 4 and st["added"] == 1
    doc = json.loads((ds / "dataset.json").read_text())
    by = {Path(c["image"]).stem: c for c in doc["cameras"]}
    for i in range(5):
        l = np.asarray(by[f"{i:06d}_L"]["cameraToWorld"])
        r = np.asarray(by[f"{i:06d}_R"]["cameraToWorld"])
        np.testing.assert_allclose(r[:3, 3] - l[:3, 3], [0.064, 0, 0], atol=1e-9)
    assert doc["sceneExtent"] == pytest.approx(10 * st["scale"])
    sys.path.insert(0, str(TOOLS))
    try:
        from splatprep.ply import read_points_ply  # type: ignore
    finally:
        sys.path.remove(str(TOOLS))
    np.testing.assert_allclose(read_points_ply(ds / "points3d.ply")[0][0], np.array([2, 4, 6]) * st["scale"],
                               rtol=1e-6)
    # a toed-out fisheye rig (Arcturus): scale only, no rig hint, no depth
    ds2 = _stereo_ds(tmp_path / "ds2", parallel=False)
    assert stereo.sfm_rig(ds2) is None
    st2 = stereo.apply_constraints(ds2)
    assert st2["scale"] and st2["enforced"] == 0 and st2["added"] == 0
    assert stereo.compute_depth(ds2)["written"] == 0


def test_stereo_left_eye_only(tmp_path):
    from splatworker.studio import stereo

    ds = _stereo_ds(tmp_path / "ds")
    st = stereo.prepare_frames(ds, both_eyes=False)
    assert st["droppedRight"] == 5 and st["pairs"] == 5
    assert sorted(p.name for p in (ds / "images").iterdir()) == [f"{i:06d}_L.png" for i in range(5)]
    assert stereo.sfm_rig(ds) is None
