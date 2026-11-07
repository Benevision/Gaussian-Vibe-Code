"""Minimal COLMAP database (sqlite) writer/reader.

Schema follows COLMAP's ``scripts/python/database.py`` (the long-standing
"legacy" layout).  COLMAP >= 3.12 adds ``rigs``/``frames`` tables and migrates
databases without them when it opens them (it creates a trivial rig/frame per
camera/image).  GLOMAP reads the same layout.

Pair ids: ``pair_id = min(id1,id2) * 2147483647 + max(id1,id2)``; when the
image ids are swapped the match columns are swapped too.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import numpy as np

MAX_IMAGE_ID = 2**31 - 1

CREATE_TABLES = [
    """CREATE TABLE IF NOT EXISTS cameras (
        camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
        model INTEGER NOT NULL,
        width INTEGER NOT NULL,
        height INTEGER NOT NULL,
        params BLOB,
        prior_focal_length INTEGER NOT NULL)""",
    f"""CREATE TABLE IF NOT EXISTS images (
        image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
        name TEXT NOT NULL UNIQUE,
        camera_id INTEGER NOT NULL,
        prior_qw REAL, prior_qx REAL, prior_qy REAL, prior_qz REAL,
        prior_tx REAL, prior_ty REAL, prior_tz REAL,
        CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < {MAX_IMAGE_ID}),
        FOREIGN KEY(camera_id) REFERENCES cameras(camera_id))""",
    """CREATE TABLE IF NOT EXISTS keypoints (
        image_id INTEGER PRIMARY KEY NOT NULL,
        rows INTEGER NOT NULL, cols INTEGER NOT NULL, data BLOB,
        FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)""",
    """CREATE TABLE IF NOT EXISTS descriptors (
        image_id INTEGER PRIMARY KEY NOT NULL,
        rows INTEGER NOT NULL, cols INTEGER NOT NULL, data BLOB,
        FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)""",
    """CREATE TABLE IF NOT EXISTS matches (
        pair_id INTEGER PRIMARY KEY NOT NULL,
        rows INTEGER NOT NULL, cols INTEGER NOT NULL, data BLOB)""",
    """CREATE TABLE IF NOT EXISTS two_view_geometries (
        pair_id INTEGER PRIMARY KEY NOT NULL,
        rows INTEGER NOT NULL, cols INTEGER NOT NULL, data BLOB,
        config INTEGER NOT NULL,
        F BLOB, E BLOB, H BLOB, qvec BLOB, tvec BLOB)""",
    "CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name)",
]

# TwoViewGeometry::ConfigurationType
TWO_VIEW_UNDEFINED = 0
TWO_VIEW_DEGENERATE = 1
TWO_VIEW_CALIBRATED = 2
TWO_VIEW_UNCALIBRATED = 3
TWO_VIEW_PLANAR = 4


def image_ids_to_pair_id(image_id1: int, image_id2: int) -> int:
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
    return image_id1 * MAX_IMAGE_ID + image_id2


def pair_id_to_image_ids(pair_id: int) -> Tuple[int, int]:
    image_id2 = pair_id % MAX_IMAGE_ID
    image_id1 = (pair_id - image_id2) // MAX_IMAGE_ID
    return int(image_id1), int(image_id2)


def _blob(a: np.ndarray) -> bytes:
    return np.ascontiguousarray(a).tobytes()


class COLMAPDatabase:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.execute("PRAGMA foreign_keys = ON")

    @classmethod
    def create(cls, path: Path) -> "COLMAPDatabase":
        path = Path(path)
        if path.exists():
            path.unlink()
        db = cls(path)
        for sql in CREATE_TABLES:
            db.conn.execute(sql)
        db.conn.commit()
        return db

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def commit(self) -> None:
        self.conn.commit()

    # writes ---------------------------------------------------------------
    def add_camera(self, model_id: int, width: int, height: int, params, prior_focal_length: bool = False, camera_id: Optional[int] = None) -> int:
        params = np.asarray(params, np.float64)
        cur = self.conn.execute(
            "INSERT INTO cameras VALUES (?, ?, ?, ?, ?, ?)",
            (camera_id, int(model_id), int(width), int(height), _blob(params), int(bool(prior_focal_length))),
        )
        return int(cur.lastrowid)

    def add_image(self, name: str, camera_id: int, prior_q=None, prior_t=None, image_id: Optional[int] = None) -> int:
        q = [None] * 4 if prior_q is None else [float(x) for x in prior_q]
        t = [None] * 3 if prior_t is None else [float(x) for x in prior_t]
        cur = self.conn.execute(
            "INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (image_id, name, int(camera_id), *q, *t)
        )
        return int(cur.lastrowid)

    def add_keypoints(self, image_id: int, keypoints: np.ndarray) -> None:
        kp = np.asarray(keypoints, np.float32)
        assert kp.ndim == 2 and kp.shape[1] in (2, 4, 6)
        self.conn.execute("INSERT INTO keypoints VALUES (?, ?, ?, ?)", (image_id, kp.shape[0], kp.shape[1], _blob(kp)))

    def add_descriptors(self, image_id: int, descriptors: np.ndarray) -> None:
        d = np.asarray(descriptors, np.uint8)
        self.conn.execute("INSERT INTO descriptors VALUES (?, ?, ?, ?)", (image_id, d.shape[0], d.shape[1], _blob(d)))

    def add_matches(self, image_id1: int, image_id2: int, matches: np.ndarray) -> None:
        m = np.asarray(matches, np.uint32).reshape(-1, 2)
        if image_id1 > image_id2:
            m = m[:, ::-1]
        pid = image_ids_to_pair_id(image_id1, image_id2)
        self.conn.execute("INSERT OR REPLACE INTO matches VALUES (?, ?, ?, ?)", (pid, m.shape[0], m.shape[1], _blob(m)))

    def add_two_view_geometry(
        self,
        image_id1: int,
        image_id2: int,
        matches: np.ndarray,
        F=None,
        E=None,
        H=None,
        qvec=None,
        tvec=None,
        config: int = TWO_VIEW_UNCALIBRATED,
    ) -> None:
        m = np.asarray(matches, np.uint32).reshape(-1, 2)
        F = np.eye(3) if F is None else np.asarray(F, np.float64)
        E = np.eye(3) if E is None else np.asarray(E, np.float64)
        H = np.eye(3) if H is None else np.asarray(H, np.float64)
        qvec = np.array([1.0, 0, 0, 0]) if qvec is None else np.asarray(qvec, np.float64)
        tvec = np.zeros(3) if tvec is None else np.asarray(tvec, np.float64)
        if image_id1 > image_id2:
            m = m[:, ::-1]
            F, E, H = F.T, E.T, np.linalg.inv(H)
        pid = image_ids_to_pair_id(image_id1, image_id2)
        self.conn.execute(
            "INSERT OR REPLACE INTO two_view_geometries VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (pid, m.shape[0], m.shape[1], _blob(m), int(config), _blob(F), _blob(E), _blob(H), _blob(qvec), _blob(tvec)),
        )

    # reads ----------------------------------------------------------------
    def cameras(self) -> Dict[int, dict]:
        out = {}
        for cid, model, w, h, params, prior in self.conn.execute("SELECT * FROM cameras"):
            out[cid] = {"model": model, "width": w, "height": h, "params": np.frombuffer(params, np.float64), "prior_focal_length": bool(prior)}
        return out

    def images(self) -> Dict[int, dict]:
        return {iid: {"name": name, "camera_id": cid} for iid, name, cid in self.conn.execute("SELECT image_id, name, camera_id FROM images")}

    def keypoints(self, image_id: int) -> np.ndarray:
        row = self.conn.execute("SELECT rows, cols, data FROM keypoints WHERE image_id=?", (image_id,)).fetchone()
        if row is None:
            return np.zeros((0, 2), np.float32)
        r, c, data = row
        return np.frombuffer(data, np.float32).reshape(r, c)

    def descriptors(self, image_id: int) -> np.ndarray:
        row = self.conn.execute("SELECT rows, cols, data FROM descriptors WHERE image_id=?", (image_id,)).fetchone()
        if row is None:
            return np.zeros((0, 128), np.uint8)
        r, c, data = row
        return np.frombuffer(data, np.uint8).reshape(r, c)

    def _iter_pairs(self, table: str) -> Iterator[Tuple[int, int, np.ndarray]]:
        for pid, r, c, data in self.conn.execute(f"SELECT pair_id, rows, cols, data FROM {table}"):
            i1, i2 = pair_id_to_image_ids(pid)
            m = np.frombuffer(data, np.uint32).reshape(r, c) if data is not None and r else np.zeros((0, 2), np.uint32)
            yield i1, i2, m

    def matches(self) -> Dict[Tuple[int, int], np.ndarray]:
        return {(a, b): m for a, b, m in self._iter_pairs("matches")}

    def two_view_geometries(self) -> Dict[Tuple[int, int], dict]:
        out = {}
        for pid, r, c, data, config, F in self.conn.execute(
            "SELECT pair_id, rows, cols, data, config, F FROM two_view_geometries"
        ):
            i1, i2 = pair_id_to_image_ids(pid)
            m = np.frombuffer(data, np.uint32).reshape(r, c) if data is not None and r else np.zeros((0, 2), np.uint32)
            out[(i1, i2)] = {"matches": m, "config": config, "F": np.frombuffer(F, np.float64).reshape(3, 3) if F else None}
        return out
