"""Drone surveys are matched by GPS neighbours: neighbouring flight lines get paired, far-apart look-alikes don't,
and the learned backend picks this for "auto" when nearly every photo has GPS."""
from types import SimpleNamespace

import numpy as np
from PIL import Image

from splatprep.backends.base import _survey_positions
from splatprep.gps import gps_positions
from splatprep.pairs import spatial_pairs


def _lawnmower(lines=4, per_line=20, spacing=10.0, line_gap=15.0):
    pts = []
    for l in range(lines):
        xs = np.arange(per_line) * spacing
        if l % 2:
            xs = xs[::-1]   # back and forth: consecutive numbers are not neighbours across lines
        pts += [(x, l * line_gap, 40.0) for x in xs]
    return np.array(pts)


def test_neighbouring_flight_lines_are_paired():
    P = _lawnmower()
    pairs = set(spatial_pairs(P, k=6, max_distance=200))
    # photo 0 (line 0, x=0) and photo 39 (line 1, x=0, the end of the reversed line) are 15 m apart
    assert (0, 39) in pairs
    # nothing beyond the distance limit
    far = set(spatial_pairs(P, k=80, max_distance=20))
    assert all(np.linalg.norm(P[i] - P[j]) <= 20 for i, j in far)


def test_far_twins_are_never_paired():
    P = np.array([[0, 0, 40], [5, 0, 40], [300, 0, 40], [305, 0, 40]], float)
    assert spatial_pairs(P, k=3, max_distance=200) == [(0, 1), (2, 3)]


def _gps_jpeg(path, lat, lon):
    im = Image.fromarray(np.zeros((8, 8, 3), np.uint8))
    ex = im.getexif()
    d = lambda v: (int(v), int(v * 60 % 60), (v * 3600) % 60)
    ex.get_ifd(0x8825).update({1: "S", 2: d(abs(lat)), 3: "E", 4: d(lon), 5: 0, 6: 50.0})
    im.save(path, exif=ex)


def test_auto_matching_uses_gps_for_surveys(tmp_path):
    paths = []
    for i in range(70):
        p = tmp_path / f"DJI_{i:04d}.jpg"
        _gps_jpeg(p, -32.1 - (i // 10) * 0.0002, 116.04 + (i % 10) * 0.0002)
        paths.append(p)
    enu = gps_positions(paths)
    assert enu is not None and enu.shape == (70, 3) and np.ptp(enu[:, 0]) > 100
    ctx = SimpleNamespace(names=[p.name for p in paths], paths=paths, opt=lambda k, d=None: d)
    assert _survey_positions(ctx, {}) is not None
    assert _survey_positions(SimpleNamespace(names=ctx.names, paths=paths,
                                             opt=lambda k, d=None: "sequential" if k == "matching" else d), {}) is None
    # a handheld set without GPS: unchanged behaviour
    plain = tmp_path / "plain"
    plain.mkdir()
    for i in range(70):
        Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(plain / f"{i}.jpg")
    assert gps_positions(sorted(plain.glob("*.jpg"))) is None


def test_position_priors_written_in_both_database_layouts(tmp_path):
    import sqlite3

    from splatprep.mapper import write_position_priors

    for layout in ("legacy", "rig"):
        db = tmp_path / f"{layout}.db"
        con = sqlite3.connect(db)
        con.execute("CREATE TABLE images (image_id INTEGER PRIMARY KEY, name TEXT, camera_id INTEGER)")
        con.executemany("INSERT INTO images VALUES (?, ?, 1)", [(1, "a.jpg"), (2, "b.jpg")])
        if layout == "rig":
            con.execute("CREATE TABLE pose_priors (pose_prior_id INTEGER PRIMARY KEY NOT NULL, corr_data_id INTEGER NOT NULL, "
                        "corr_sensor_id INTEGER NOT NULL, corr_sensor_type INTEGER NOT NULL, position BLOB, "
                        "position_covariance BLOB, gravity BLOB, coordinate_system INTEGER NOT NULL)")
        con.commit()
        con.close()
        assert write_position_priors(db, {"a.jpg": [1, 2, 3], "b.jpg": [4, 5, 6], "missing.jpg": [0, 0, 0]}) == 2
        assert write_position_priors(db, {"a.jpg": [1, 2, 3]}) == 1    # rewriting replaces, never duplicates
        con = sqlite3.connect(db)
        key = "corr_data_id" if layout == "rig" else "image_id"
        rows = dict(con.execute(f"SELECT {key}, position FROM pose_priors"))
        assert len(rows) == 2 and np.allclose(np.frombuffer(rows[2], np.float64), [4, 5, 6])
        assert {r[0] for r in con.execute("SELECT coordinate_system FROM pose_priors")} == {1}


def test_merging_models_keeps_each_models_intrinsics_and_points():
    from splatprep.colmap_model import Camera, Image, Point3D, Reconstruction, merge_reconstructions

    def model(f, img_id, name):
        r = Reconstruction()
        r.cameras[1] = Camera(1, "SIMPLE_RADIAL", 100, 80, np.array([f, 50, 40, 0.0]))
        r.points3D[7] = Point3D(7, np.array([0.0, 0, 5]), np.array([1, 2, 3]))
        r.images[img_id] = Image(img_id, np.array([1.0, 0, 0, 0]), np.zeros(3), 1, name,
                                 np.array([[50.0, 40.0], [1.0, 1.0]]), np.array([7, -1]))
        return r

    m = merge_reconstructions([model(100.0, 1, "a.jpg"), model(120.0, 2, "b.jpg")])
    assert len(m.cameras) == 2 and len(m.points3D) == 2 and len(m.images) == 2
    fa = m.cameras[m.images[1].camera_id].params[0]
    fb = m.cameras[m.images[2].camera_id].params[0]
    assert (fa, fb) == (100.0, 120.0)
    pa, pb = m.images[1].point3D_ids[0], m.images[2].point3D_ids[0]
    assert pa != pb and pa in m.points3D and pb in m.points3D and m.images[1].point3D_ids[1] == -1
