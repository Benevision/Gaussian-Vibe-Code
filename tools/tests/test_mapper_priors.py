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


def test_auto_prefers_incremental_for_small_sets(tmp_path, monkeypatch):
    monkeypatch.setattr(mapper, "available_mappers", lambda: {"glomap": True, "colmap": True, "pycolmap": True})
    calls = []
    monkeypatch.setattr(mapper, "run_logged", lambda cmd, *a, **k: calls.append([Path(str(c)).name for c in cmd[:2]]))
    monkeypatch.setattr(mapper, "find_executable", lambda n: "/usr/bin/" + n if n == "colmap" else None)
    monkeypatch.setattr(mapper, "colmap_commands", lambda c: frozenset({"mapper", "global_mapper"}))

    class Rec:
        images = {1: None}

    monkeypatch.setattr(mapper, "read_largest_model", lambda d: Rec())
    _, used = mapper.run_mapper(tmp_path / "db", tmp_path, tmp_path / "s", "auto", num_images=60)
    assert used == "colmap" and calls[-1] == ["colmap", "mapper"]
    _, used = mapper.run_mapper(tmp_path / "db", tmp_path, tmp_path / "s", "auto", num_images=mapper.AUTO_GLOBAL_MIN_IMAGES + 1)
    assert used == "glomap" and calls[-1] == ["colmap", "global_mapper"]
