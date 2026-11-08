"""Viewer camera: orbit / fly navigation in the scene's OpenCV convention (x right, y down, z forward).

The scene's *up* direction is a world vector (default −Y, the 3DGS / COLMAP convention; a PLY written by our
trainers records the dataset's up in ``comment splatstudio_up``). Yaw turns about up, pitch tilts towards it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

UP_CHOICES = {
    "-y": ((0, -1, 0), "Up is −Y (3DGS / COLMAP, default)"),
    "+y": ((0, 1, 0), "Up is +Y (OpenGL, ARKit, Polycam)"),
    "+z": ((0, 0, 1), "Up is +Z (Blender, drones, LiDAR)"),
    "-z": ((0, 0, -1), "Up is −Z"),
    "+x": ((1, 0, 0), "Up is +X"),
    "-x": ((-1, 0, 0), "Up is −X"),
}

VIEWS = {"front": (0.0, 0.0), "back": (180.0, 0.0), "left": (90.0, 0.0), "right": (-90.0, 0.0), "top": (0.0, -89.0)}


def _unit(v) -> np.ndarray:
    v = np.asarray(v, np.float64).reshape(3)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else np.array([0.0, -1.0, 0.0])


def plane_basis(up) -> Tuple[np.ndarray, np.ndarray]:
    """Two unit vectors (a, b) spanning the plane ⟂ up; a is as close to world +Z as possible (else +X), the
    same choice as the skybox's ``forward`` so yaw 0 looks along the skybox's forward."""
    u = _unit(up)
    ref = np.array([0.0, 0.0, 1.0]) if abs(u[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    a = _unit(ref - u * (ref @ u))
    b = np.cross(u, a)
    return a, b


@dataclass
class Camera:
    up: np.ndarray = field(default_factory=lambda: np.array([0.0, -1.0, 0.0]))
    target: np.ndarray = field(default_factory=lambda: np.zeros(3))
    distance: float = 3.0
    yaw: float = 0.0             # degrees about up
    pitch: float = -15.0         # degrees; negative looks down
    fov_y: float = 60.0          # degrees
    mode: str = "orbit"          # orbit | fly
    position_fly: Optional[np.ndarray] = None
    near: float = 0.01
    far: float = 1000.0

    # ------------------------------------------------------------- basis
    def forward(self) -> np.ndarray:
        a, b = plane_basis(self.up)
        y, p = math.radians(self.yaw), math.radians(self.pitch)
        return _unit(math.cos(p) * (math.cos(y) * a + math.sin(y) * b) + math.sin(p) * _unit(self.up))

    def position(self) -> np.ndarray:
        if self.mode == "fly" and self.position_fly is not None:
            return np.asarray(self.position_fly, np.float64)
        return np.asarray(self.target, np.float64) - self.distance * self.forward()

    def rotation(self) -> np.ndarray:
        """World → camera rotation (rows: right, down, forward)."""
        f = self.forward()
        u = _unit(self.up)
        down = -u + f * float(u @ f)
        if np.linalg.norm(down) < 1e-6:          # looking straight up/down: any perpendicular will do
            a, _ = plane_basis(u)
            down = a
        down = _unit(down)
        right = np.cross(down, f)
        return np.stack([right, down, f])

    def view_matrix(self) -> np.ndarray:
        """4×4 world → camera (OpenCV)."""
        R = self.rotation()
        m = np.eye(4)
        m[:3, :3] = R
        m[:3, 3] = -R @ self.position()
        return m

    def intrinsics(self, width: int, height: int) -> Tuple[float, float, float, float]:
        fy = 0.5 * height / math.tan(math.radians(self.fov_y) * 0.5)
        return fy, fy, 0.5 * width, 0.5 * height

    def gl_view_proj(self, width: int, height: int) -> np.ndarray:
        """Clip-space matrix for lines (frustums, crop box): OpenCV camera → GL clip, y flipped, reversed z not
        used. Matches the splat shader's pixel → NDC mapping."""
        fx, fy, cx, cy = self.intrinsics(width, height)
        n, f = self.near, self.far
        P = np.zeros((4, 4))
        P[0, 0] = 2 * fx / width
        P[0, 2] = 2 * cx / width - 1
        P[1, 1] = -2 * fy / height
        P[1, 2] = -(2 * cy / height - 1)
        P[2, 2] = (f + n) / (f - n)
        P[2, 3] = -2 * f * n / (f - n)
        P[3, 2] = 1
        return P @ self.view_matrix()

    def project(self, points: np.ndarray, width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
        """Pixel coordinates (N, 2) and depths (N,) of world points."""
        V = self.view_matrix()
        c = points @ V[:3, :3].T + V[:3, 3]
        fx, fy, cx, cy = self.intrinsics(width, height)
        z = c[:, 2]
        zs = np.where(np.abs(z) < 1e-9, 1e-9, z)
        return np.stack([fx * c[:, 0] / zs + cx, fy * c[:, 1] / zs + cy], 1), z

    def ray(self, x: float, y: float, width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
        fx, fy, cx, cy = self.intrinsics(width, height)
        d_cam = np.array([(x - cx) / fx, (y - cy) / fy, 1.0])
        return self.position(), _unit(self.rotation().T @ d_cam)

    # ------------------------------------------------------------- navigation
    def orbit(self, dx_deg: float, dy_deg: float) -> None:
        """Orbit mode turns around the target, fly mode turns the head (the position stays)."""
        self.yaw += dx_deg
        self.pitch = float(np.clip(self.pitch - dy_deg, -89.5, 89.5))

    def pan(self, dx_px: float, dy_px: float, height: int) -> None:
        R = self.rotation()
        scale = (self.distance if self.mode == "orbit" else max(self.distance, 1.0)) * 2 * \
            math.tan(math.radians(self.fov_y) / 2) / max(height, 1)
        delta = (-dx_px * R[0] - dy_px * R[1]) * scale
        self.target = np.asarray(self.target, np.float64) + delta
        if self.mode == "fly" and self.position_fly is not None:
            self.position_fly = self.position_fly + delta

    def zoom(self, factor: float) -> None:
        """factor > 1 moves closer."""
        if self.mode == "fly":
            self.move(forward=(factor - 1.0) * max(self.distance, 0.1))
            return
        self.distance = float(np.clip(self.distance / factor, 1e-4, 1e6))

    def move(self, forward: float = 0.0, right: float = 0.0, up: float = 0.0) -> None:
        """Fly step in world units (switches to fly mode keeping the view)."""
        self.set_mode("fly")
        R = self.rotation()
        self.position_fly = self.position() + forward * R[2] + right * R[0] - up * R[1]
        self.target = self.position_fly + self.distance * R[2]

    def set_mode(self, mode: str) -> None:
        if mode == self.mode:
            return
        pos = self.position()
        if mode == "fly":
            self.position_fly = pos
        else:
            self.target = pos + self.distance * self.forward()
        self.mode = mode

    def set_orbit_center(self, point) -> None:
        pos = self.position()
        point = np.asarray(point, np.float64)
        d = point - pos
        dist = float(np.linalg.norm(d))
        if dist < 1e-9:
            return
        self.look_along(d / dist)
        self.mode = "orbit"
        self.target = point
        self.distance = dist

    def look_along(self, direction) -> None:
        """Yaw / pitch that look along ``direction``."""
        d = _unit(direction)
        u = _unit(self.up)
        a, b = plane_basis(u)
        self.pitch = math.degrees(math.asin(float(np.clip(d @ u, -1, 1))))
        self.pitch = float(np.clip(self.pitch, -89.5, 89.5))
        self.yaw = math.degrees(math.atan2(float(d @ b), float(d @ a)))

    def frame(self, lo, hi, view: Optional[str] = None) -> None:
        lo, hi = np.asarray(lo, np.float64), np.asarray(hi, np.float64)
        center = (lo + hi) / 2
        radius = max(float(np.linalg.norm(hi - lo)) / 2, 1e-3)
        if view in VIEWS:
            self.yaw, self.pitch = VIEWS[view]
        self.mode = "orbit"
        self.target = center
        self.distance = radius / math.sin(math.radians(self.fov_y) / 2) * 0.9
        self.near = max(radius * 1e-4, 1e-4)
        self.far = max(radius * 200, 10.0)

    def set_up(self, up) -> None:
        f = self.forward()
        self.up = _unit(up)
        self.look_along(f)

    def flip(self) -> None:
        self.set_up(-_unit(self.up))

    def level_to_view(self) -> None:
        """Make the current camera's up the scene's up."""
        self.set_up(-self.rotation()[1])

    def look_through(self, c2w_opencv: np.ndarray, fov_y: Optional[float] = None) -> None:
        """Fly camera at a dataset camera (row-major OpenCV camera-to-world)."""
        c2w = np.asarray(c2w_opencv, np.float64).reshape(4, 4)
        self.mode = "fly"
        self.position_fly = c2w[:3, 3].copy()
        self.look_along(c2w[:3, 2])
        self.target = self.position_fly + self.distance * c2w[:3, 2]
        if fov_y:
            self.fov_y = float(fov_y)


def adaptive_fly_speed(scene_radius: float, base: float = 1.0) -> float:
    """World units per second for WASD: a fifth of the scene radius by default."""
    return max(scene_radius, 1e-3) * 0.2 * base
