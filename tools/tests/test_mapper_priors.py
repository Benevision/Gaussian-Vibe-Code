"""Focal priors (EXIF) and the auto-mapper choice."""
from pathlib import Path

import numpy as np
from PIL import Image

from splatprep import mapper
from splatprep.backends.base import exif_focal_px, initial_camera_params
from splatprep.colmap_db import COLMAPDatabase


def _jpeg_with_f35(path: Path, f35):
    im = Image.new("RGB", (400, 300), (120, 120, 120))
    exif = Image.Exif()
    if f35 is not None:
        exif.get_ifd(0x8769)[0xA405] = f35
    im.save(path, exif=exif)


def test_exif_focal_px(tmp_path):
    p = tmp_path / "a.jpg"
    _jpeg_with_f35(p, 26)  # iPhone main camera
    f = exif_focal_px(p, 4032, 3024)
    assert abs(f - 26 / 36 * 4032) < 1e-6
    params, prior = initial_camera_params("SIMPLE_RADIAL", 4032, 3024, None, f)
    assert prior and abs(params[0] - f) < 1e-6
    q = tmp_path / "b.jpg"
    _jpeg_with_f35(q, None)
    assert exif_focal_px(q, 400, 300) is None
    params, prior = initial_camera_params("SIMPLE_RADIAL", 400, 300, None, None)
    assert not prior


def test_prior_focal_fraction(tmp_path):
    db = COLMAPDatabase.create(tmp_path / "d.db")
    db.add_camera(2, 640, 480, np.array([500.0, 320, 240, 0]), True)
    db.add_camera(2, 640, 480, np.array([500.0, 320, 240, 0]), False)
    db.add_camera(2, 640, 480, np.array([500.0, 320, 240, 0]), False)
    db.close()
    assert abs(mapper._prior_focal_fraction(tmp_path / "d.db") - 1 / 3) < 1e-9
    assert mapper._prior_focal_fraction(tmp_path / "missing.db") == 1.0


def test_auto_runs_both_mappers_and_keeps_longer_tracks(tmp_path, monkeypatch):
    monkeypatch.setattr(mapper, "available_mappers", lambda: {"glomap": True, "colmap": True, "pycolmap": True})

    class Rec:
        def __init__(self, reg, track, err):
            self._s = {"numRegistered": reg, "meanTrackLength": track, "meanReprojError": err}
            self.images = {i: None for i in range(reg)}

        def stats(self):
            return self._s

    runs = []

    def fake_single(db, img, out, pref, log, prog, n):
        runs.append(pref)
        return (Rec(60, 14.4, 1.4), "colmap") if pref == "colmap" else (Rec(60, 22.1, 1.3), "glomap")

    monkeypatch.setattr(mapper, "_run_single", fake_single)
    rec, used = mapper.run_mapper(tmp_path / "db", tmp_path, tmp_path / "s", "auto", num_images=60)
    assert runs == ["colmap", "glomap"] and used == "auto:glomap" and rec.stats()["meanTrackLength"] == 22.1

    # more registered images beat track length
    def fake2(db, img, out, pref, log, prog, n):
        return (Rec(60, 6.0, 0.6), "colmap") if pref == "colmap" else (Rec(40, 9.0, 0.5), "glomap")
    monkeypatch.setattr(mapper, "_run_single", fake2)
    assert mapper.run_mapper(tmp_path / "db", tmp_path, tmp_path / "s", "auto", num_images=60)[1] == "auto:colmap"

    # large sets: global only
    runs.clear(); monkeypatch.setattr(mapper, "_run_single", fake_single)
    mapper.run_mapper(tmp_path / "db", tmp_path, tmp_path / "s", "auto", num_images=mapper.AUTO_BOTH_MAX + 1)
    assert runs == ["auto"]
