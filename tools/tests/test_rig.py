"""360° rig handling: views grouped into frames/sensors, rig-aware pairs, rig config rotations."""
import json

import numpy as np

from splatprep.rig import _quat_wxyz, parse_rig, rig_pairs, write_rig_config


def yaw(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def make(frames=6, views=8):
    names, groups = [], []
    for f in range(frames):
        g = {}
        for v in range(views):
            n = f"{f:06d}_v{v:02d}.jpg"
            names.append(n)
            g[n] = yaw(v * 360 / views).tolist()
        groups.append(g)
    return names, {"rig": {"kind": "multiview", "groups": groups, "intrinsics": [600, 600, 600, 600]}}


def test_parse_groups_sensors_and_frames():
    names, req = make()
    lay = parse_rig(req, names)
    assert lay.num_sensors == 8 and lay.num_frames == 6
    assert lay.sensor[names.index("000003_v05.jpg")] == 5 and lay.frame[names.index("000003_v05.jpg")] == 3
    assert lay.db_names[names.index("000003_v05.jpg")] == "s05/000003.jpg"
    assert lay.intrinsics == [600, 600, 600, 600]


def test_no_rig_without_request_or_with_gaps():
    names, req = make()
    assert parse_rig({}, names) is None
    assert parse_rig(req, names + ["extra.jpg"]) is None  # an image outside the rig → map without rig


def test_pairs_are_rig_aware():
    names, req = make(frames=10)
    lay = parse_rig(req, names)
    pairs, info = rig_pairs(lay, None, window=3)
    fr = np.asarray(lay.frame)
    sn = np.asarray(lay.sensor)
    for i, j in pairs:
        assert fr[i] != fr[j]                                  # never within a frame (no parallax)
        d = abs(fr[i] - fr[j])
        assert d <= 3
        if d > 1:
            assert sn[i] == sn[j]                              # further frames: same direction only
    # t→t+1: same + two 45° neighbours = 3 per view; t+2, t+3: 1 per view
    assert info["sequentialPairs"] == 9 * 8 * 3 + 8 * 8 + 7 * 8
    all_pairs = len(names) * (len(names) - 1) // 2
    assert len(pairs) < all_pairs / 5


def test_rig_config_rotation_is_relative_to_reference(tmp_path):
    names, req = make(views=4)
    lay = parse_rig(req, names)
    cfg = json.loads(write_rig_config(lay, tmp_path / "rig.json").read_text())[0]["cameras"]
    assert cfg[0]["ref_sensor"] and cfg[0]["image_prefix"] == "s00/"
    # sensor 1 is yawed 90°: cam_from_ref = R1ᵀ R0 = yaw(-90)
    q = np.array(cfg[1]["cam_from_rig_rotation"])
    assert np.allclose(q, _quat_wxyz(yaw(-90)), atol=1e-6)
    assert cfg[1]["cam_from_rig_translation"] == [0.0, 0.0, 0.0]


def stereo(frames=5, b=0.065):
    names, groups = [], []
    for f in range(frames):
        l, r = f"{f:06d}_L.jpg", f"{f:06d}_R.jpg"
        names += [l, r]
        groups.append({l: np.eye(3).tolist(), r: {"R": np.eye(3).tolist(), "t": [b, 0, 0]}})
    return names, {"rig": {"kind": "multiview", "groups": groups}}


def test_stereo_rig_with_baseline(tmp_path):
    names, req = stereo()
    lay = parse_rig(req, names)
    assert lay.num_sensors == 2 and lay.num_frames == 5 and lay.has_baseline     # same rotation, other centre
    assert lay.sensor[names.index("000002_R.jpg")] == 1
    pairs, info = rig_pairs(lay, None, window=2)
    fr = np.asarray(lay.frame)
    within = [(i, j) for i, j in pairs if fr[i] == fr[j]]
    assert len(within) == 5                                    # L–R of every frame (parallax)
    cfg = json.loads(write_rig_config(lay, tmp_path / "rig.json").read_text())[0]["cameras"]
    # the right eye sits +6.5 cm along the left eye's x: cam_from_ref translation = −b x
    assert np.allclose(cfg[1]["cam_from_rig_translation"], [-0.065, 0, 0])
    assert np.allclose(cfg[1]["cam_from_rig_rotation"], [1, 0, 0, 0])
    # 360° rigs are unchanged: no centres → no within-frame pairs (covered by test_pairs_are_rig_aware)


def test_quaternion_roundtrip():
    for R in (np.eye(3), yaw(30), yaw(170), yaw(-120) @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])):
        w, x, y, z = _quat_wxyz(R)
        R2 = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                       [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                       [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
        assert np.allclose(R, R2, atol=1e-6)
