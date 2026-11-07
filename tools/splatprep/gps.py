"""GPS sanity check and georeferencing for photo sets with EXIF GPS (drones, phones).

After SfM, the solved camera centres are fitted to the GPS positions (local east/north/up metres) with a
robust Sim3. Small residuals mean the solve matches the flight: the scene is then expressed in metres with
+Z up. Large residuals mean the solve is wrong somewhere (e.g. a flight layer placed at the wrong height),
and the user is told instead of training on it.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

#: A camera "disagrees" with GPS when it is further than this many metres from its GPS position …
GPS_OK_ABS_M = 6.0
#: … or this fraction of the flight's extent, whichever is larger (consumer GPS drifts a few metres).
GPS_OK_REL = 0.05
#: The solve is trusted when at most this share of cameras disagree. A percentile would miss a whole
#: misplaced flight layer (the robust fit aligns the majority; one layer of a 3-layer survey is < 10 %).
GPS_MAX_BAD_FRACTION = 0.02


def _dms(v) -> float:
    return float(v[0]) + float(v[1]) / 60 + float(v[2]) / 3600


def read_gps(path: Path) -> Optional[Tuple[float, float, float]]:
    """(lat, lon, altitude m). Altitude prefers DJI's take-off-relative XMP value (GPS altitude is noisy)."""
    try:
        from PIL import Image

        with Image.open(path) as im:
            g = im.getexif().get_ifd(0x8825)
            xmp = im.info.get("xmp", b"") or b""
    except Exception:
        return None
    if not (g.get(2) and g.get(4)):
        return None
    lat, lon = _dms(g[2]), _dms(g[4])
    if g.get(1) in ("S", b"S"):
        lat = -lat
    if g.get(3) in ("W", b"W"):
        lon = -lon
    if isinstance(xmp, str):
        xmp = xmp.encode()
    m = re.search(rb'RelativeAltitude="([-+0-9.]+)"', xmp)
    if m:
        alt = float(m.group(1))
    else:
        try:
            alt = float(g.get(6, 0.0)) * (-1 if g.get(5) == 1 else 1)
        except Exception:
            alt = 0.0
    return lat, lon, alt


def to_enu(gps: np.ndarray) -> np.ndarray:
    """(N, 3) lat/lon/alt → local east/north/up metres around the mean (equirectangular; fine below ~10 km)."""
    lat0, lon0 = gps[:, 0].mean(), gps[:, 1].mean()
    return np.stack([(gps[:, 1] - lon0) * 111320.0 * math.cos(math.radians(lat0)),
                     (gps[:, 0] - lat0) * 110540.0, gps[:, 2] - gps[:, 2].mean()], 1)


def check_against_gps(rec, image_dir: Path, names: Sequence[str]) -> Optional[Dict]:
    """Fit solved centres to GPS. Returns stats (and the Sim3 when trustworthy) or None without enough GPS."""
    from .geometry import robust_umeyama

    by_name = rec.image_by_name()
    pts, cen, used = [], [], []
    for n in names:
        if n not in by_name:
            continue
        g = read_gps(Path(image_dir) / n)
        if g is None:
            continue
        pts.append(g)
        cen.append(by_name[n].center())
        used.append(n)
    if len(used) < 10 or len(used) < 0.8 * len(by_name):
        return None
    enu = to_enu(np.array(pts))
    extent = float(np.linalg.norm(np.ptp(enu, 0)))
    if extent < 5.0:  # GPS can't tell positions apart (handheld in one spot)
        return None
    src = np.array(cen)
    T, inl = robust_umeyama(src, enu)
    res = np.linalg.norm(T.apply_points(src) - enu, axis=1)
    p90 = float(np.percentile(res, 90))
    limit = max(GPS_OK_ABS_M, GPS_OK_REL * extent)
    bad = float((res > limit).mean())
    out = {"numGps": len(used), "extentM": round(extent, 1), "medianM": round(float(np.median(res)), 2),
           "p90M": round(p90, 2), "maxM": round(float(res.max()), 1), "limitM": round(limit, 1),
           "badFraction": round(bad, 4), "ok": bad <= GPS_MAX_BAD_FRACTION}
    out["outliers"] = [used[i] for i in np.nonzero(res > limit)[0]]
    worst = np.argsort(-res)[:5]
    out["worst"] = [[used[i], round(float(res[i]), 1)] for i in worst]
    return {**out, "_T": T}
