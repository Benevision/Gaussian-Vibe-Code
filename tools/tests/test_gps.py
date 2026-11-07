"""GPS check: southern hemisphere, a misplaced flight layer, and a single stray camera."""
from pathlib import Path

import numpy as np

from splatprep import gps


class _Im:
    def __init__(self, name, c):
        self.name, self._c = name, np.asarray(c, float)

    def center(self):
        return self._c


class _Rec:
    def __init__(self, ims):
        self.images = {i: im for i, im in enumerate(ims)}

    def image_by_name(self):
        return {im.name: im for im in self.images.values()}


def _survey(lat0=-32.1, lon0=116.05):
    """3-layer lawnmower survey: positions in ENU metres + matching lat/lon/alt."""
    pts = []
    for alt, step in ((28, 20), (38, 10), (68, 25)):
        for x in range(-60, 61, step):
            for y in range(-60, 61, step):
                pts.append((x, y, alt))
    enu = np.array(pts, float)
    lat = lat0 + enu[:, 1] / 110540.0
    lon = lon0 + enu[:, 0] / (111320.0 * np.cos(np.radians(lat0)))
    return enu, np.stack([lat, lon, enu[:, 2]], 1)


def _check(monkeypatch, solved, geo):
    names = [f"{i:04d}.jpg" for i in range(len(solved))]
    table = {n: tuple(g) for n, g in zip(names, geo)}
    monkeypatch.setattr(gps, "read_gps", lambda p: table.get(Path(p).name))
    rec = _Rec([_Im(n, c) for n, c in zip(names, solved)])
    return gps.check_against_gps(rec, Path("."), names)


def test_correct_solve_southern_hemisphere(monkeypatch):
    enu, geo = _survey()
    rng = np.random.default_rng(0)
    # SfM frame: arbitrary rotation / scale / offset of the true layout plus 1 m noise
    a = 0.7
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]) @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    solved = (0.03 * (R @ (enu + rng.normal(0, 1, enu.shape)).T)).T + 5
    r = _check(monkeypatch, solved, geo)
    assert r["ok"] and r["medianM"] < 2 and not r["outliers"]


def test_misplaced_layer_is_caught(monkeypatch):
    enu, geo = _survey()
    wrong = enu.copy()
    wrong[enu[:, 2] == 68, 2] = 10.0  # the high layer solved below the others
    r = _check(monkeypatch, wrong, geo)
    assert not r["ok"] and r["badFraction"] > 0.05


def test_single_stray_camera_is_an_outlier(monkeypatch):
    enu, geo = _survey()
    stray = enu.copy()
    stray[0] += [250, 100, 0]
    r = _check(monkeypatch, stray, geo)
    assert r["ok"] and r["outliers"] == ["0000.jpg"]
