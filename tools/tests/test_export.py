import json

import numpy as np
from PIL import Image

from splatprep.colmap_model import read_binary_model, read_text_model, write_binary_model, write_text_model
from splatprep.export import export_dataset
from splatprep.geometry import qvec_to_rotmat
from splatprep.ply import read_points_ply

CAMERAS_TXT = """# Camera list with one line of data per camera:
#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
# Number of cameras: 1
1 PINHOLE 64 48 60.0 61.0 32.0 24.0
"""

# two images; second has an empty POINTS2D line
IMAGES_TXT = """# Image list with two lines of data per image:
#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
#   POINTS2D[] as (X, Y, POINT3D_ID)
1 1.0 0.0 0.0 0.0 0.0 0.0 0.0 1 img_a.png
10.0 12.0 1 20.0 22.0 -1 30.0 30.0 2
2 0.9238795325 0.0 0.3826834324 0.0 -1.0 0.2 0.5 1 img_b.png

"""

POINTS_TXT = """# 3D point list with one line of data per point:
1 0.1 0.2 3.0 255 0 0 0.5 1 0 2 0
2 -0.3 0.1 4.0 0 255 0 0.7 1 2
"""


def make_model(tmp_path):
    m = tmp_path / "sparse"
    m.mkdir()
    (m / "cameras.txt").write_text(CAMERAS_TXT)
    (m / "images.txt").write_text(IMAGES_TXT)
    (m / "points3D.txt").write_text(POINTS_TXT)
    imgs = tmp_path / "images"
    imgs.mkdir()
    for n in ("img_a.png", "img_b.png"):
        Image.fromarray(np.full((48, 64, 3), 100, np.uint8)).save(imgs / n)
    return m, imgs


def test_text_binary_round_trip(tmp_path):
    m, _ = make_model(tmp_path)
    rec = read_text_model(m)
    assert len(rec.cameras) == 1 and len(rec.images) == 2 and len(rec.points3D) == 2
    assert rec.images[1].xys.shape == (3, 2) and rec.images[2].xys.shape == (0, 2)
    write_binary_model(rec, tmp_path / "bin")
    rec2 = read_binary_model(tmp_path / "bin")
    write_text_model(rec2, tmp_path / "txt")
    rec3 = read_text_model(tmp_path / "txt")
    for r in (rec2, rec3):
        assert r.images[2].name == "img_b.png"
        assert np.allclose(r.images[2].qvec, rec.images[2].qvec)
        assert np.array_equal(r.images[1].point3D_ids, [1, -1, 2])
        assert np.array_equal(r.points3D[1].image_ids, [1, 2])
        assert np.allclose(r.cameras[1].params, [60, 61, 32, 24])


def test_export_dataset_json(tmp_path):
    m, imgs = make_model(tmp_path)
    rec = read_text_model(m)
    out = tmp_path / "out"
    stats = export_dataset(rec, imgs, out, tmp_path / "work")
    assert stats["undistortion"] == "copy"
    ds = json.loads((out / "dataset.json").read_text())
    assert ds["format"] == "splatdataset" and ds["version"] == 1 and ds["convention"] == "opencv"
    assert ds["points"] == "points3d.ply"
    cams = {c["image"]: c for c in ds["cameras"]}
    b = cams["images/img_b.png"]
    assert (b["width"], b["height"], b["fx"], b["fy"], b["cx"], b["cy"]) == (64, 48, 60.0, 61.0, 32.0, 24.0)
    c2w = np.array(b["cameraToWorld"])  # row-major
    R = qvec_to_rotmat([0.9238795325, 0.0, 0.3826834324, 0.0])
    t = np.array([-1.0, 0.2, 0.5])
    assert np.allclose(c2w[:3, :3], R.T, atol=1e-8)
    assert np.allclose(c2w[:3, 3], -R.T @ t, atol=1e-8)
    assert np.allclose(c2w[3], [0, 0, 0, 1])
    # scene metrics
    centers = np.array([np.array(c["cameraToWorld"])[:3, 3] for c in ds["cameras"]])
    assert np.allclose(ds["sceneCenter"], centers.mean(0))
    assert ds["sceneExtent"] >= np.max(np.linalg.norm(centers - centers.mean(0), axis=1))
    assert abs(np.linalg.norm(ds["upVector"]) - 1) < 1e-6
    xyz, rgb = read_points_ply(out / "points3d.ply")
    assert xyz.shape == (2, 3) and tuple(rgb[0]) == (255, 0, 0)
    assert (out / "images" / "img_a.png").exists()
    assert (out / "sparse" / "0" / "images.bin").exists()


def test_export_undistorts_with_opencv(tmp_path):
    m, imgs = make_model(tmp_path)
    rec = read_text_model(m)
    rec.cameras[1].model = "OPENCV"
    rec.cameras[1].params = np.array([60.0, 61.0, 32.0, 24.0, -0.1, 0.01, 0.0, 0.0])
    out = tmp_path / "out"
    stats = export_dataset(rec, imgs, out, tmp_path / "work", undistort="opencv")
    assert stats["undistortion"] == "opencv"
    ds = json.loads((out / "dataset.json").read_text())
    with Image.open(out / ds["cameras"][0]["image"]) as im:
        assert im.size == (64, 48)
    # undistorted camera is a pinhole with comparable focal length
    assert 40 < ds["cameras"][0]["fx"] < 80
