"""Rotations, Sim(3) alignment (Umeyama) and pose-error metrics.

Conventions: ``c2w`` is a 4x4 camera-to-world matrix in the OpenCV camera
convention (+X right, +Y down, +Z forward); ``w2c`` its inverse.  COLMAP
quaternions are (w, x, y, z) and describe world-to-camera rotations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------
# rotations


def qvec_to_rotmat(q: Sequence[float]) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ]
    )


def rotmat_to_qvec(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = (
        np.array(
            [
                [Rxx - Ryy - Rzz, 0, 0, 0],
                [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
            ]
        )
        / 3.0
    )
    eigvals, eigvecs = np.linalg.eigh(K)
    q = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if q[0] < 0:
        q = -q
    return q


def rotation_angle_deg(R: np.ndarray) -> float:
    c = (np.trace(R) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def look_at(eye: Sequence[float], target: Sequence[float], up=(0.0, 1.0, 0.0)) -> np.ndarray:
    """OpenCV c2w for a camera at ``eye`` looking at ``target`` (world up ``up``)."""
    eye = np.asarray(eye, float)
    fwd = np.asarray(target, float) - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.asarray(up, float))
    if np.linalg.norm(right) < 1e-8:
        right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = right, down, fwd, eye
    return c2w


def invert_pose(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


ARKIT_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])


def arkit_to_opencv_c2w(c2w_arkit: np.ndarray) -> np.ndarray:
    """PROTOCOL §1: ``c2w_opencv = c2w_arkit * diag(1, -1, -1, 1)``."""
    return np.asarray(c2w_arkit, float) @ ARKIT_TO_OPENCV


# --------------------------------------------------------------------------
# Sim(3)


@dataclass
class Sim3:
    s: float
    R: np.ndarray
    t: np.ndarray

    @staticmethod
    def identity() -> "Sim3":
        return Sim3(1.0, np.eye(3), np.zeros(3))

    def apply_points(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, float)
        if X.size == 0:
            return X.reshape(-1, 3)
        return self.s * X @ self.R.T + self.t

    def apply_c2w(self, c2w: np.ndarray) -> np.ndarray:
        out = np.eye(4)
        out[:3, :3] = self.R @ c2w[:3, :3]
        out[:3, 3] = self.s * self.R @ c2w[:3, 3] + self.t
        return out

    def matrix(self) -> np.ndarray:
        M = np.eye(4)
        M[:3, :3] = self.s * self.R
        M[:3, 3] = self.t
        return M

    def to_json(self) -> dict:
        return {"scale": float(self.s), "rotation": self.R.tolist(), "translation": self.t.tolist()}


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True) -> Sim3:
    """Least-squares similarity with ``dst ≈ s R src + t`` (Umeyama 1991)."""
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError("umeyama expects two Nx3 arrays of equal shape")
    n = src.shape[0]
    if n < 3:
        raise ValueError("umeyama needs at least 3 correspondences")
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    var_s = (xs**2).sum() / n
    cov = xd.T @ xs / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S) / var_s) if (with_scale and var_s > 0) else 1.0
    t = mu_d - s * R @ mu_s
    return Sim3(s, R, t)


def robust_umeyama(src: np.ndarray, dst: np.ndarray, iters: int = 5, k: float = 3.0) -> Tuple[Sim3, np.ndarray]:
    """Umeyama with iterative outlier rejection (residual > k * median)."""
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    inl = np.ones(len(src), bool)
    T = umeyama(src, dst)
    for _ in range(iters):
        res = np.linalg.norm(T.apply_points(src) - dst, axis=1)
        med = np.median(res[inl]) if inl.any() else 0.0
        thr = max(k * med, 1e-9)
        new = res <= thr
        if new.sum() < max(3, int(0.5 * len(src))) or np.array_equal(new, inl):
            break
        inl = new
        T = umeyama(src[inl], dst[inl])
    return T, inl


# --------------------------------------------------------------------------
# pose metrics


def pose_auc(errors: Iterable[float], thresholds: Sequence[float]) -> List[float]:
    """Area under the recall-vs-error curve up to each threshold (SuperGlue style).

    Infinite / NaN errors (e.g. unregistered cameras) count as failures.
    """
    errors = np.asarray([e if np.isfinite(e) else np.inf for e in errors], float)
    if errors.size == 0:
        return [0.0 for _ in thresholds]
    order = np.argsort(errors)
    errors = np.concatenate([[0.0], errors[order]])
    recall = np.arange(len(errors)) / (len(errors) - 1)
    aucs = []
    for t in thresholds:
        last = np.searchsorted(errors, t)
        r = np.concatenate([recall[:last], [recall[last - 1]]])
        e = np.concatenate([errors[:last], [t]])
        aucs.append(float(np.trapezoid(r, x=e) / t) if hasattr(np, "trapezoid") else float(np.trapz(r, x=e) / t))
    return aucs


def angle_between_deg(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0 if (na < 1e-12 and nb < 1e-12) else 180.0
    c = float(np.dot(a, b) / (na * nb))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def relative_pose_errors(w2c_est_i, w2c_est_j, w2c_gt_i, w2c_gt_j) -> Tuple[float, float]:
    """(rotation error deg, translation-direction error deg) of the i->j relative pose."""

    def rel(Ti, Tj):
        Tij = Tj @ invert_pose(Ti)
        return Tij[:3, :3], Tij[:3, 3]

    Re, te = rel(w2c_est_i, w2c_est_j)
    Rg, tg = rel(w2c_gt_i, w2c_gt_j)
    r_err = rotation_angle_deg(Rg.T @ Re)
    t_err = angle_between_deg(te, tg)
    return r_err, t_err


def evaluate_poses(
    est_c2w: Dict[str, np.ndarray],
    gt_c2w: Dict[str, np.ndarray],
    thresholds: Sequence[float] = (5.0, 10.0, 30.0),
    max_pairs: int = 20000,
    seed: int = 0,
) -> dict:
    """Compare estimated to ground-truth poses (keyed by image name).

    * Sim(3)-aligns estimated camera centres to GT (Umeyama, robust).
    * ATE: RMSE of aligned centres (GT units) and relative to the GT scene extent.
    * Absolute rotation error after alignment.
    * Relative pose AUC over image pairs: error = max(rotation err, translation-angle err);
      pairs involving an unregistered camera count as failures.
    """
    names = sorted(gt_c2w)
    reg = [n for n in names if n in est_c2w]
    out: dict = {
        "numGroundTruth": len(names),
        "numEvaluated": len(reg),
    }
    if len(reg) < 3:
        out.update({f"aucAt{int(t)}": 0.0 for t in thresholds})
        out["alignment"] = None
        return out
    src = np.stack([est_c2w[n][:3, 3] for n in reg])
    dst = np.stack([gt_c2w[n][:3, 3] for n in reg])
    T, inl = robust_umeyama(src, dst)
    aligned = {n: T.apply_c2w(est_c2w[n]) for n in reg}
    pos_err = np.linalg.norm(np.stack([aligned[n][:3, 3] for n in reg]) - dst, axis=1)
    gt_centers = np.stack([gt_c2w[n][:3, 3] for n in names])
    extent = float(np.max(np.linalg.norm(gt_centers - gt_centers.mean(0), axis=1))) or 1.0
    rot_err = np.array([rotation_angle_deg(gt_c2w[n][:3, :3].T @ aligned[n][:3, :3]) for n in reg])

    # relative pose errors (scale-free, alignment-free)
    rng = np.random.default_rng(seed)
    idx_pairs = [(i, j) for i in range(len(names)) for j in range(i + 1, len(names))]
    if len(idx_pairs) > max_pairs:
        sel = rng.choice(len(idx_pairs), max_pairs, replace=False)
        idx_pairs = [idx_pairs[k] for k in sorted(sel)]
    est_w2c = {n: invert_pose(est_c2w[n]) for n in reg}
    gt_w2c = {n: invert_pose(gt_c2w[n]) for n in names}
    errs, rerrs, terrs = [], [], []
    for i, j in idx_pairs:
        a, b = names[i], names[j]
        if a in est_w2c and b in est_w2c:
            r, t = relative_pose_errors(est_w2c[a], est_w2c[b], gt_w2c[a], gt_w2c[b])
            rerrs.append(r)
            terrs.append(t)
            errs.append(max(r, t))
        else:
            errs.append(np.inf)
    aucs = pose_auc(errs, thresholds)
    for t, a in zip(thresholds, aucs):
        out[f"aucAt{int(t)}"] = round(a, 4)
    out.update(
        {
            "ateRmse": float(np.sqrt(np.mean(pos_err**2))),
            "ateRmseRel": float(np.sqrt(np.mean(pos_err**2)) / extent),
            "ateMedian": float(np.median(pos_err)),
            "rotErrDegMean": float(rot_err.mean()),
            "rotErrDegMedian": float(np.median(rot_err)),
            "relRotErrDegMedian": float(np.median(rerrs)) if rerrs else None,
            "relTransErrDegMedian": float(np.median(terrs)) if terrs else None,
            "alignmentInliers": int(inl.sum()),
            "alignment": T.to_json(),
        }
    )
    return out


# --------------------------------------------------------------------------
# epipolar geometry


def skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], float)


def fundamental_from_poses(K0, K1, w2c0, w2c1) -> np.ndarray:
    """F such that x1^T F x0 = 0 for pixel coords (x0 in image 0)."""
    T = w2c1 @ invert_pose(w2c0)
    R, t = T[:3, :3], T[:3, 3]
    E = skew(t) @ R
    F = np.linalg.inv(K1).T @ E @ np.linalg.inv(K0)
    return F / (np.linalg.norm(F) + 1e-12)


def sampson_distance(F: np.ndarray, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    """Sampson epipolar error (pixels^2 -> returned as sqrt, pixels)."""
    x0h = np.hstack([x0, np.ones((len(x0), 1))])
    x1h = np.hstack([x1, np.ones((len(x1), 1))])
    Fx0 = x0h @ F.T
    Ftx1 = x1h @ F
    num = np.sum(x1h * Fx0, axis=1) ** 2
    den = Fx0[:, 0] ** 2 + Fx0[:, 1] ** 2 + Ftx1[:, 0] ** 2 + Ftx1[:, 1] ** 2
    return np.sqrt(num / np.maximum(den, 1e-12))


def intrinsics_matrix(fx, fy, cx, cy) -> np.ndarray:
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
