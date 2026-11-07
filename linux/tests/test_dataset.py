import json

import numpy as np
import pytest

from conftest import look_at_c2w, make_dataset
from splatworker.config import ConfigError, Region, TrainingConfig
from splatworker.dataset import (CameraInfo, DatasetError, camera_sees_box, find_upload_root,
                                 load_view, parse_dataset, select_cameras, target_size)
from splatworker.ply import gaussian_property_names, read_ply_vertices, read_point_cloud, write_gaussian_ply


def test_parse_valid_dataset(dataset_dir):
    ds = parse_dataset(dataset_dir)
    assert len(ds.cameras) == 4
    assert len(ds.train_cameras) == 3 and len(ds.holdout_cameras) == 1
    assert ds.scene_extent == pytest.approx(3.3)
    assert ds.points == "points3d.ply"
    assert ds.has_depth
    c = ds.cameras[0]
    assert c.camera_to_world.shape == (4, 4)
    # row-major: translation in last column
    assert c.camera_to_world[0, 3] == pytest.approx(3.0)


def test_nested_top_level_folder(tmp_path):
    make_dataset(tmp_path / "wrap" / "MyScene")
    (tmp_path / "wrap" / "__MACOSX").mkdir()
    ds = parse_dataset(tmp_path / "wrap")
    assert ds.root.name == "MyScene"


def test_upload_root_accepts_images_only(tmp_path):
    (tmp_path / "up" / "images").mkdir(parents=True)
    assert find_upload_root(tmp_path / "up") == tmp_path / "up"
    with pytest.raises(DatasetError):
        parse_dataset(tmp_path / "up")


@pytest.mark.parametrize("bad", ["../secret.jpg", "/etc/passwd", "images/../../x.png", "images\\a.png"])
def test_rejects_traversal_in_references(dataset_dir, bad):
    doc = json.loads((dataset_dir / "dataset.json").read_text())
    doc["cameras"][0]["image"] = bad
    (dataset_dir / "dataset.json").write_text(json.dumps(doc))
    with pytest.raises(DatasetError):
        parse_dataset(dataset_dir)


def test_rejects_missing_image_and_bad_values(dataset_dir):
    (dataset_dir / "images" / "000001.png").unlink()
    with pytest.raises(DatasetError, match="missing"):
        parse_dataset(dataset_dir)
    parse_dataset(dataset_dir, check_files=False)

    doc = json.loads((dataset_dir / "dataset.json").read_text())
    doc["cameras"][0]["fx"] = "big"
    (dataset_dir / "dataset.json").write_text(json.dumps(doc))
    with pytest.raises(DatasetError):
        parse_dataset(dataset_dir, check_files=False)


def test_rejects_all_holdout_and_wrong_format(dataset_dir):
    doc = json.loads((dataset_dir / "dataset.json").read_text())
    for c in doc["cameras"]:
        c["holdout"] = True
    (dataset_dir / "dataset.json").write_text(json.dumps(doc))
    with pytest.raises(DatasetError, match="holdout"):
        parse_dataset(dataset_dir)
    doc["format"] = "nerfstudio"
    (dataset_dir / "dataset.json").write_text(json.dumps(doc))
    with pytest.raises(DatasetError, match="format"):
        parse_dataset(dataset_dir)


def test_extent_computed_when_missing(dataset_dir):
    doc = json.loads((dataset_dir / "dataset.json").read_text())
    del doc["sceneExtent"], doc["sceneCenter"]
    (dataset_dir / "dataset.json").write_text(json.dumps(doc))
    ds = parse_dataset(dataset_dir)
    assert ds.scene_extent == pytest.approx(3.0 * 1.1, rel=1e-3)


def test_target_size():
    assert target_size(1920, 1440, 1600) == (1600, 1200)
    assert target_size(800, 600, 1600) == (800, 600)
    assert target_size(1440, 1920, 960) == (720, 960)


def test_load_view_downscales_and_scales_intrinsics(tmp_path):
    ds_dir = make_dataset(tmp_path / "ds", width=200, height=100)
    ds = parse_dataset(ds_dir)
    v = load_view(ds, ds.cameras[0], max_size=50)
    assert v.image.shape == (25, 50, 3) and v.image.dtype == np.uint8
    assert v.K[0, 0] == pytest.approx(50.0 * 0.25)
    assert v.K[1, 1] == pytest.approx(50.0 * 0.25)
    assert v.K[0, 2] == pytest.approx(100 * 0.25)
    assert v.K[1, 2] == pytest.approx(50 * 0.25)
    np.testing.assert_allclose(v.world_to_camera @ ds.cameras[0].camera_to_world, np.eye(4), atol=1e-5)
    # depth: float16 16x12 resized (nearest) to the training resolution, invalid stays 0
    assert v.depth.shape == (25, 50)
    assert v.depth[0, 0] == 0.0
    assert v.depth[-1, -1] == pytest.approx(3.0)


def test_load_view_no_downscale_when_small(dataset_dir):
    ds = parse_dataset(dataset_dir)
    v = load_view(ds, ds.cameras[1], max_size=1600, load_depth_maps=False)
    assert v.image.shape == (48, 64, 3) and v.depth is None
    assert v.K[0, 0] == pytest.approx(50.0)


def test_depth_size_mismatch(dataset_dir):
    (dataset_dir / "depth" / "000000.bin").write_bytes(b"\x00" * 10)
    ds = parse_dataset(dataset_dir)
    with pytest.raises(DatasetError, match="float16"):
        load_view(ds, ds.cameras[0], 1600)


def test_read_point_cloud(dataset_dir):
    xyz, rgb = read_point_cloud(dataset_dir / "points3d.ply")
    assert xyz.shape == (200, 3) and rgb.shape == (200, 3)
    assert rgb.min() >= 0 and rgb.max() <= 1


def test_gaussian_ply_roundtrip_channel_major(tmp_path):
    n = 3
    shN = np.zeros((n, 15, 3), np.float32)
    shN[:, 0, 0] = 1.0   # R, coeff 1 -> f_rest_0
    shN[:, 0, 1] = 2.0   # G, coeff 1 -> f_rest_15
    shN[:, 2, 2] = 3.0   # B, coeff 3 -> f_rest_32
    path = tmp_path / "g.ply"
    write_gaussian_ply(path, np.ones((n, 3)), np.full((n, 1, 3), 0.5), shN, np.zeros(n),
                       np.full((n, 3), -2.0), np.tile([2.0, 0, 0, 0], (n, 1)))
    header = path.read_bytes().split(b"end_header")[0].decode()
    props = [l.split()[-1] for l in header.splitlines() if l.startswith("property")]
    assert props == gaussian_property_names()
    assert len(props) == 62
    v = read_ply_vertices(path)
    assert v["f_rest_0"][0] == 1.0 and v["f_rest_15"][0] == 2.0 and v["f_rest_32"][0] == 3.0
    assert v["f_rest_1"][0] == 0.0
    assert v["rot_0"][0] == pytest.approx(1.0)  # normalised, w first
    assert v["scale_0"][0] == -2.0 and v["nx"][0] == 0.0


def test_gaussian_ply_lower_sh_degree_padded(tmp_path):
    n = 2
    shN = np.ones((n, 3, 3), np.float32)  # degree 1
    path = tmp_path / "g1.ply"
    write_gaussian_ply(path, np.zeros((n, 3)), np.zeros((n, 3)), shN, np.zeros(n),
                       np.zeros((n, 3)), np.tile([1.0, 0, 0, 0], (n, 1)))
    v = read_ply_vertices(path)
    assert v["f_rest_2"][0] == 1.0 and v["f_rest_3"][0] == 0.0
    assert v["f_rest_15"][0] == 1.0 and v["f_rest_44"][0] == 0.0


# ------------------------------------------------------------------ config

def test_config_defaults_and_unknown_keys():
    cfg = TrainingConfig.from_dict({"iterations": 100, "futureKnob": 3})
    assert cfg.iterations == 100 and cfg.strategy == "mcmc" and cfg.background == "random"
    assert cfg.effective_densify_stop() == 25000
    assert cfg.effective_depth_weight(True) == 0.1
    assert cfg.effective_depth_weight(False) == 0.0
    assert cfg.effective_opacity_reg() == 0.01


def test_config_adaptive_resolution():
    cfg = TrainingConfig.from_dict({"strategy": "adaptive", "absGrad": True})
    assert cfg.effective_densify_stop() == 15000
    assert cfg.effective_grad_threshold() == pytest.approx(0.0008)
    assert cfg.effective_opacity_reg() == 0.0 and cfg.effective_scale_reg() == 0.0
    cfg = TrainingConfig.from_dict({"strategy": "adaptive", "opacityReg": 0.02, "depthWeight": 0.5})
    assert cfg.effective_opacity_reg() == 0.02
    assert cfg.effective_depth_weight(True) == 0.5


@pytest.mark.parametrize("bad", [{"iterations": "lots"}, {"shDegree": 4}, {"strategy": "foo"},
                                 {"background": "green"}, {"iterations": 1.5}, {"antialiased": "maybe"},
                                 {"region": {"min": [0, 0], "max": [1, 1, 1]}},
                                 {"region": {"min": [1, 1, 1], "max": [0, 0, 0]}},
                                 {"cameraIds": "all"}, "notadict"])
def test_config_invalid(bad):
    with pytest.raises(ConfigError):
        TrainingConfig.from_dict(bad)


def test_config_region_and_camera_ids():
    cfg = TrainingConfig.from_dict({"region": {"min": [0, 0, 0], "max": [2, 4, 1], "margin": 0.25},
                                    "cameraIds": [1, 3.0]})
    assert cfg.cameraIds == [1, 3]
    lo, hi = cfg.region.expanded()
    assert lo == pytest.approx((-0.5, -1.0, -0.25)) and hi == pytest.approx((2.5, 5.0, 1.25))
    d = cfg.to_dict()
    assert d["region"] == {"min": [0.0, 0.0, 0.0], "max": [2.0, 4.0, 1.0], "margin": 0.25}
    assert TrainingConfig.from_dict(d).region == cfg.region


def _cam(pos, target=(0, 0, 0)):
    return CameraInfo(id=0, image="x", width=64, height=48, fx=50, fy=50, cx=32, cy=24,
                      camera_to_world=look_at_c2w(pos, target))


def test_camera_sees_box():
    assert camera_sees_box(_cam((3, 0, 0)), (-0.5, -0.5, -0.5), (0.5, 0.5, 0.5))
    # looking away from the box
    assert not camera_sees_box(_cam((3, 0, 0), target=(6, 0, 0)), (-0.5, -0.5, -0.5), (0.5, 0.5, 0.5))
    # box far off to the side, outside the field of view
    assert not camera_sees_box(_cam((3, 0, 0)), (0, -0.5, 20), (1, 0.5, 21))
    # camera inside the box always counts
    assert camera_sees_box(_cam((0, 0, 0), target=(1, 0, 0)), (-1, -1, -1), (1, 1, 1))


def test_select_cameras(dataset_dir):
    ds = parse_dataset(dataset_dir)
    assert [c.id for c in select_cameras(ds, [0, 2])] == [0, 2]
    # cameras on a ring of radius 3 looking at the origin; a box behind camera 0 (x > 3)
    # is only visible to the camera opposite it (id 2 at x = -3).
    region = Region(min=(3.5, -0.2, -0.2), max=(4.0, 0.2, 0.2))
    ids = [c.id for c in select_cameras(ds, None, region.expanded())]
    assert ids == [2]


def test_ply_records_scene_up(tmp_path):
    import numpy as np
    from splatworker.ply import write_gaussian_ply
    p = tmp_path / "s.ply"
    n = 2
    write_gaussian_ply(p, np.zeros((n, 3)), np.zeros((n, 3)), None, np.zeros(n), np.zeros((n, 3)),
                       np.tile([1.0, 0, 0, 0], (n, 1)), up=[0, 0, 1])
    head = p.read_bytes()[:300].decode("ascii", "replace")
    assert "comment splatstudio_up 0 0 1\n" in head
    assert head.index("comment splatstudio_up") < head.index("element vertex")
