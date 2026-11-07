import sqlite3

import numpy as np

from splatprep.colmap_db import (
    MAX_IMAGE_ID,
    TWO_VIEW_UNCALIBRATED,
    COLMAPDatabase,
    image_ids_to_pair_id,
    pair_id_to_image_ids,
)


def test_pair_id_encoding():
    assert image_ids_to_pair_id(1, 2) == 1 * MAX_IMAGE_ID + 2
    assert image_ids_to_pair_id(2, 1) == image_ids_to_pair_id(1, 2)
    for a, b in [(1, 2), (5, 100000), (123456, 7)]:
        lo, hi = min(a, b), max(a, b)
        assert pair_id_to_image_ids(image_ids_to_pair_id(a, b)) == (lo, hi)


def test_round_trip(tmp_path):
    db_path = tmp_path / "database.db"
    rng = np.random.default_rng(0)
    db = COLMAPDatabase.create(db_path)
    cid = db.add_camera(2, 640, 480, [700.0, 320.0, 240.0, 0.01], prior_focal_length=True)
    i1 = db.add_image("a.png", cid)
    i2 = db.add_image("sub/b.png", cid)
    k1 = rng.random((50, 2)).astype(np.float32) * 600
    k2 = rng.random((40, 2)).astype(np.float32) * 600
    db.add_keypoints(i1, k1)
    db.add_keypoints(i2, k2)
    d1 = rng.integers(0, 255, (50, 128), dtype=np.uint8)
    db.add_descriptors(i1, d1)
    m = np.stack([np.arange(10), np.arange(10)[::-1] + 5], 1)
    # added in swapped order -> stored with columns swapped
    db.add_matches(i2, i1, m)
    F = rng.random((3, 3))
    db.add_two_view_geometry(i1, i2, m[:6], F=F, config=TWO_VIEW_UNCALIBRATED)
    db.close()

    db = COLMAPDatabase(db_path)
    cams = db.cameras()
    assert cams[cid]["model"] == 2 and cams[cid]["width"] == 640 and cams[cid]["prior_focal_length"]
    assert np.allclose(cams[cid]["params"], [700.0, 320.0, 240.0, 0.01])
    ims = db.images()
    assert {v["name"] for v in ims.values()} == {"a.png", "sub/b.png"}
    assert np.array_equal(db.keypoints(i1), k1)
    assert np.array_equal(db.descriptors(i1), d1)
    ms = db.matches()
    assert list(ms) == [(i1, i2)]
    assert np.array_equal(ms[(i1, i2)], m[:, ::-1])
    tv = db.two_view_geometries()[(i1, i2)]
    assert tv["config"] == TWO_VIEW_UNCALIBRATED and len(tv["matches"]) == 6
    assert np.allclose(tv["F"], F)
    db.close()

    # schema is the one COLMAP expects
    con = sqlite3.connect(db_path)
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"cameras", "images", "keypoints", "descriptors", "matches", "two_view_geometries"} <= tables
    cols = [r[1] for r in con.execute("PRAGMA table_info(two_view_geometries)")]
    assert cols == ["pair_id", "rows", "cols", "data", "config", "F", "E", "H", "qvec", "tvec"]
    con.close()
