"""Uniform backend interface.

Every model id in PROTOCOL §6 is a :class:`Backend`.  ``check()`` must be cheap
and must not import heavy packages (use ``importlib.util.find_spec``); real
imports happen inside ``run()``.

Two families:

* :class:`FeatureMatchBackend` — sparse features + matches.  The base class
  does pair selection -> extract -> match -> COLMAP database -> OpenCV
  geometric verification -> mapper (GLOMAP / COLMAP / pycolmap).  Subclasses
  only implement ``extract`` + ``match`` (or ``extract_and_match`` for
  detector-free matchers such as LoFTR).  When no mapper is installed and the
  caller allows it (benchmark), the backend returns matching data only.
* Dense / feed-forward backends (``mast3r``, ``vggt``) override ``run`` and
  return a :class:`Reconstruction` directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..colmap_db import COLMAPDatabase, TWO_VIEW_UNCALIBRATED
from ..colmap_model import CAMERA_MODEL_IDS, Reconstruction
from ..device import has_module
from ..errors import MapperUnavailable, ModelUnavailable
from ..images import image_size, load_mask
from ..mapper import any_mapper_available, run_mapper
from ..pairs import Pair, select_pairs, write_pairs_txt
from ..priors import PriorFrame
from ..progress import StagedProgress
from ..verification import verify_pair, verify_pairs  # noqa: F401


@dataclass
class Availability:
    available: bool  # can at least extract + match (or run end-to-end for dense models)
    sfm: bool  # can produce a full reconstruction here
    reason: str = ""

    def to_json(self) -> dict:
        return {"available": self.available, "sfm": self.sfm, "reason": self.reason}


@dataclass
class SfMContext:
    image_dir: Path
    names: List[str]  # relative to image_dir
    work_dir: Path
    request: dict
    progress: StagedProgress
    device: str = "cpu"
    priors: Dict[str, PriorFrame] = field(default_factory=dict)
    masks_dir: Optional[Path] = None
    sky_masks_dir: Optional[Path] = None   # 255 = sky: no keypoints there (clouds move; the sky is at infinity)
    allow_matching_only: bool = False
    log_path: Optional[Path] = None

    @property
    def paths(self) -> List[Path]:
        return [self.image_dir / n for n in self.names]

    def opt(self, key: str, default: Any = None) -> Any:
        v = self.request.get(key)
        return default if v is None else v


@dataclass
class MatchingData:
    """Keypoints in COLMAP pixel convention (+0.5) and per-pair matches/inliers."""

    names: List[str]
    keypoints: List[np.ndarray]
    pairs: List[Pair]
    matches: Dict[Pair, np.ndarray]
    inliers: Dict[Pair, np.ndarray]
    sizes: List[Tuple[int, int]]

    def save(self, path: Path) -> None:
        arrs = {"names": np.array(self.names, dtype=object), "sizes": np.array(self.sizes)}
        for i, k in enumerate(self.keypoints):
            arrs[f"kp_{i}"] = k.astype(np.float32)
        for (i, j), m in self.matches.items():
            arrs[f"m_{i}_{j}"] = m.astype(np.int32)
            arrs[f"in_{i}_{j}"] = self.inliers.get((i, j), np.zeros(len(m), bool))
        np.savez_compressed(path, **arrs)

    @staticmethod
    def load(path: Path) -> "MatchingData":
        z = np.load(path, allow_pickle=True)
        names = [str(n) for n in z["names"]]
        kps = [z[f"kp_{i}"] for i in range(len(names))]
        matches, inliers = {}, {}
        for key in z.files:
            if key.startswith("m_"):
                _, i, j = key.split("_")
                matches[(int(i), int(j))] = z[key]
                inliers[(int(i), int(j))] = z[f"in_{i}_{j}"]
        sizes = [tuple(map(int, s)) for s in z["sizes"]]
        return MatchingData(names, kps, sorted(matches), matches, inliers, sizes)


@dataclass
class SfMOutput:
    reconstruction: Optional[Reconstruction] = None
    matching: Optional[MatchingData] = None
    info: Dict[str, Any] = field(default_factory=dict)


class Backend:
    id: str = ""
    display_name: str = ""
    kind: str = "sparse"  # "sparse" | "dense" | "mask"
    python_modules: Sequence[str] = ()
    executables: Sequence[str] = ()
    install_hint: str = ""

    def check(self) -> Availability:
        missing = [m for m in self.python_modules if not has_module(m)]
        if missing:
            return Availability(False, False, f"missing python packages: {', '.join(missing)}" + (f" — {self.install_hint}" if self.install_hint else ""))
        # Matching-only backends still need a mapper for a full solve: say so before hours of matching, not after.
        from ..mapper import any_mapper_available, unavailable_message

        if not any_mapper_available():
            return Availability(True, False, unavailable_message())
        return Availability(True, True)

    def require(self, ctx: Optional[SfMContext] = None) -> Availability:
        a = self.check()
        if not a.available:
            raise ModelUnavailable(self.id, a.reason)
        if not a.sfm and not (ctx and ctx.allow_matching_only):
            raise ModelUnavailable(self.id, a.reason or "no SfM mapper available")
        return a

    def run(self, ctx: SfMContext) -> SfMOutput:  # pragma: no cover - abstract
        raise NotImplementedError


SFM_STAGES = [("pairs", 0.03), ("features", 0.30), ("matching", 0.35), ("verification", 0.05), ("mapping", 0.17), ("alignment", 0.02), ("export", 0.08)]


class FeatureMatchBackend(Backend):
    """Sparse pipeline; subclasses implement extract/match."""

    kind = "sparse"
    detector_free = False

    def check(self) -> Availability:
        a = super().check()
        if not a.available:
            return a
        if not has_module("cv2"):
            return Availability(False, False, "missing opencv-python-headless")
        if not any_mapper_available():
            return Availability(True, False, "matching only: no SfM mapper (pip install pycolmap, or brew install colmap glomap)")
        return a

    # --- subclass hooks -------------------------------------------------
    def setup(self, ctx: SfMContext) -> None:
        pass

    def teardown(self) -> None:
        pass

    def extract(self, ctx: SfMContext, index: int, path: Path) -> Any:
        """Return an opaque feature object; must expose keypoints via ``keypoints_of``."""
        raise NotImplementedError

    def keypoints_of(self, feats: Any) -> np.ndarray:
        """Nx2 keypoints in OpenCV pixel convention (pixel centre = integer)."""
        raise NotImplementedError

    def filter_features(self, feats: Any, keep: np.ndarray) -> Any:
        raise NotImplementedError

    def match(self, ctx: SfMContext, f0: Any, f1: Any) -> np.ndarray:
        """Mx2 int indices into the keypoints of f0/f1."""
        raise NotImplementedError

    # --- shared pipeline -------------------------------------------------
    def extract_and_match(self, ctx: SfMContext, pairs: List[Pair]) -> Tuple[List[np.ndarray], Dict[Pair, np.ndarray]]:
        n = len(ctx.names)
        feats: List[Any] = []
        for i, p in enumerate(ctx.paths):
            f = self.extract(ctx, i, p)
            if ctx.sky_masks_dir is not None:
                kp = self.keypoints_of(f)
                w, h = image_size(p)
                sky = load_mask(ctx.sky_masks_dir, ctx.names[i], (h, w))   # True where 255 = sky
                if sky is not None and len(kp):
                    xi = np.clip(np.round(kp[:, 0]).astype(int), 0, w - 1)
                    yi = np.clip(np.round(kp[:, 1]).astype(int), 0, h - 1)
                    f = self.filter_features(f, ~sky[yi, xi])
            if ctx.masks_dir is not None:
                kp = self.keypoints_of(f)
                w, h = image_size(p)
                m = load_mask(ctx.masks_dir, ctx.names[i], (h, w))
                if m is not None and len(kp):
                    xi = np.clip(np.round(kp[:, 0]).astype(int), 0, w - 1)
                    yi = np.clip(np.round(kp[:, 1]).astype(int), 0, h - 1)
                    f = self.filter_features(f, m[yi, xi])
            feats.append(f)
            ctx.progress.update("features", (i + 1) / n, f"extracted {ctx.names[i]} ({len(self.keypoints_of(f))} kp)")
        matches: Dict[Pair, np.ndarray] = {}
        for k, (i, j) in enumerate(pairs):
            matches[(i, j)] = np.asarray(self.match(ctx, feats[i], feats[j]), np.int64).reshape(-1, 2)
            if k % 5 == 0 or k == len(pairs) - 1:
                ctx.progress.update("matching", (k + 1) / len(pairs), f"matched {k + 1}/{len(pairs)} pairs")
        kps = [np.asarray(self.keypoints_of(f), np.float64)[:, :2] for f in feats]
        return kps, matches

    def run(self, ctx: SfMContext) -> SfMOutput:
        avail = self.require(ctx)
        info: Dict[str, Any] = {"backend": self.id, "device": ctx.device}
        prior_c2w = {i: ctx.priors[n].c2w for i, n in enumerate(ctx.names) if n in ctx.priors}
        has_ts = bool(ctx.priors) and all(ctx.priors.get(n) is not None and ctx.priors[n].timestamp is not None for n in ctx.names)
        ctx.progress.update("pairs", 0.0, "selecting image pairs")
        from ..prior_refine import lookat_pairs, refine_poses, use_prior_refine

        refine = use_prior_refine(ctx.names, ctx.priors, ctx.request)
        if refine:
            # Known poses (iPhone capture): refine them instead of solving — fewer pairs, no mapper, and every
            # frame keeps its own capture intrinsics (pinhole, per-frame).
            ctx.request = {**ctx.request, "singleCamera": False, "cameraModel": "pinhole"}
            pairs = lookat_pairs(prior_c2w, k=int(ctx.opt("refineNeighbors", 10)), window=int(ctx.opt("refineWindow", 4)))
            pinfo = {"strategy": "lookat", "resolved": "lookat", "numPairs": len(pairs)}
        rig = None
        if not refine:
            from ..rig import parse_rig

            rig = parse_rig(ctx.request, ctx.names) if ctx.opt("useRig", True) else None
        if rig is not None:
            from ..pairs import global_descriptors
            from ..rig import rig_pairs

            ctx.progress.update("pairs", 0.3, f"rig: {rig.num_frames} frames × {rig.num_sensors} views")
            D = global_descriptors(ctx.paths, ctx.work_dir)[0] if int(ctx.opt("rigLoopPairs", 2)) > 0 else None
            pairs, pinfo = rig_pairs(rig, D, window=int(ctx.opt("rigWindow", 3)), loop_k=int(ctx.opt("rigLoopPairs", 2)))
        elif not refine:
            pairs, pinfo = select_pairs(
                ctx.names,
                ctx.paths,
                ctx.opt("matching", "auto"),
                int(ctx.opt("sequentialOverlap", 10)),
                int(ctx.opt("retrievalTopK", 30)),
                prior_c2w or None,
                has_ts,
                ctx.work_dir,
            )
        info["pairs"] = pinfo
        ctx.progress.update("pairs", 1.0, f"{len(pairs)} pairs ({pinfo['resolved']})")
        self.setup(ctx)
        try:
            kps, raw = self.extract_and_match(ctx, pairs)
        finally:
            self.teardown()
        sizes = [image_size(p) for p in ctx.paths]

        # geometric verification
        scale = max(max(s) for s in sizes) / 1000.0 if sizes else 1.0
        inliers, Fs = verify_pairs(
            kps, raw, threshold_px=max(1.0, 2.0 * scale),
            progress=lambda k, n: ctx.progress.update("verification", k / max(1, n), f"verified {k}/{n} pairs"))
        n_verified = len(Fs)
        info.update(
            {
                "numPairs": len(pairs),
                "numVerifiedPairs": n_verified,
                "meanKeypoints": float(np.mean([len(k) for k in kps])) if kps else 0.0,
                "meanMatches": float(np.mean([len(m) for m in raw.values()])) if raw else 0.0,
                "meanInliers": float(np.mean([inliers[p].sum() for p in Fs])) if Fs else 0.0,
            }
        )
        ctx.progress.update("verification", 1.0, f"{n_verified}/{len(pairs)} pairs verified")
        # COLMAP keypoint convention: +0.5
        kps_colmap = [k + 0.5 for k in kps]
        matching = MatchingData(list(ctx.names), kps_colmap, pairs, raw, inliers, sizes)

        if not avail.sfm:
            if ctx.allow_matching_only:
                info["status"] = "matching-only"
                return SfMOutput(None, matching, info)
            raise MapperUnavailable(avail.reason)

        db_path = ctx.work_dir / "database.db"
        write_database(db_path, ctx, kps_colmap, sizes, raw, inliers, Fs, rig=rig)
        write_pairs_txt(ctx.work_dir / "pairs.txt", ctx.names, pairs)
        if refine:
            from ..mapper import find_executable

            import os

            cli = None if os.environ.get("SPLATPREP_REFINE_ENGINE") == "pycolmap" else find_executable("colmap")
            rec, rinfo = refine_poses(cli, db_path, ctx.image_dir, ctx.priors,
                                      ctx.work_dir / "refine", ctx.log_path, ctx.progress.sub("mapping"))
            info.update({"mapper": "prior-refine", "priorRefine": rinfo})
            return SfMOutput(rec, matching, info)
        if rig is not None:
            from ..mapper import rig_mapping_available, run_rig_mapper
            from ..rig import link_tree, write_rig_config

            if rig_mapping_available():
                tree = link_tree(rig, ctx.image_dir, ctx.work_dir / "rig_images")
                cfg = write_rig_config(rig, ctx.work_dir / "rig.json")
                rec, mapper = run_rig_mapper(db_path, tree, ctx.work_dir / "sparse", cfg, ctx.opt("mapper", "auto"),
                                             ctx.log_path, ctx.progress.sub("mapping"), rig.num_frames,
                                             fix_intrinsics=rig.intrinsics is not None)
                back = rig.original_name()
                for im in rec.images.values():
                    im.name = back.get(im.name, im.name)
                info.update({"mapper": mapper, "rig": {"frames": rig.num_frames, "views": rig.num_sensors}})
                return SfMOutput(rec, matching, info)
            # No rig-capable mapper: map the views as separate photos (database names are the rig names).
            info["rigFallback"] = "no COLMAP/pycolmap with rig support; mapped views independently"
            tree = link_tree(rig, ctx.image_dir, ctx.work_dir / "rig_images")
            rec, mapper = run_mapper(db_path, tree, ctx.work_dir / "sparse", ctx.opt("mapper", "auto"), ctx.log_path,
                                     ctx.progress.sub("mapping"), len(ctx.names))
            back = rig.original_name()
            for im in rec.images.values():
                im.name = back.get(im.name, im.name)
            info["mapper"] = mapper
            return SfMOutput(rec, matching, info)
        rec, mapper = run_mapper(
            db_path,
            ctx.image_dir,
            ctx.work_dir / "sparse",
            ctx.opt("mapper", "auto"),
            ctx.log_path,
            ctx.progress.sub("mapping"),
            len(ctx.names),
        )
        info["mapper"] = mapper
        return SfMOutput(rec, matching, info)


def camera_model_name(request: dict) -> str:
    m = (request.get("cameraModel") or "auto").lower()
    return {"auto": "SIMPLE_RADIAL", "pinhole": "PINHOLE", "opencv": "OPENCV", "fisheye": "OPENCV_FISHEYE", "simple_radial": "SIMPLE_RADIAL"}.get(
        m, m.upper()
    )


def exif_focal_px(path, w: int, h: int) -> Optional[float]:
    """Focal length in pixels from EXIF ``FocalLengthIn35mmFilm`` (None if absent / implausible)."""
    try:
        from PIL import Image

        with Image.open(path) as im:
            exif = im.getexif()
            f35 = exif.get_ifd(0x8769).get(0xA405) or exif.get(0xA405)  # Exif IFD → FocalLengthIn35mmFilm
        f35 = float(f35 or 0)
    except Exception:
        return None
    if not 8.0 <= f35 <= 800.0:
        return None
    return f35 / 36.0 * max(w, h)


def initial_camera_params(model: str, w: int, h: int, prior: Optional[PriorFrame], exif_f: Optional[float] = None) -> Tuple[np.ndarray, bool]:
    if prior is not None and prior.intrinsics is not None:
        fx, fy, cx, cy = [float(v) for v in prior.intrinsics]
        has_prior = True
    elif exif_f:
        fx = fy = float(exif_f)
        cx, cy = w / 2.0, h / 2.0
        has_prior = True
    elif model == "OPENCV_FISHEYE":
        # equidistant fisheye seen across ~180°: f ≈ (max side / 2) / (π / 2). COLMAP's pinhole guess (1.2 × side)
        # is ~4× too long for such lenses and the mapper then can't initialise.
        fx = fy = max(w, h) / np.pi
        cx, cy = w / 2.0, h / 2.0
        # COLMAP marks fisheye pairs without a focal prior as degenerate (a fundamental matrix can't recover a
        # fisheye focal length), so the guess must be flagged as a prior; the mapper still refines it (see
        # mapper._prior_focal_fraction, which doesn't count fisheye guesses as "fix the focal length").
        has_prior = True
    else:
        fx = fy = 1.2 * max(w, h)
        cx, cy = w / 2.0, h / 2.0
        has_prior = False
    f = 0.5 * (fx + fy)
    p = {
        "SIMPLE_PINHOLE": [f, cx, cy],
        "PINHOLE": [fx, fy, cx, cy],
        "SIMPLE_RADIAL": [f, cx, cy, 0.0],
        "RADIAL": [f, cx, cy, 0.0, 0.0],
        "OPENCV": [fx, fy, cx, cy, 0, 0, 0, 0],
        "OPENCV_FISHEYE": [fx, fy, cx, cy, 0, 0, 0, 0],
    }.get(model)
    if p is None:
        raise ValueError(f"unsupported camera model {model}")
    return np.array(p, float), has_prior


def write_database(
    db_path: Path,
    ctx: SfMContext,
    kps_colmap: List[np.ndarray],
    sizes: List[Tuple[int, int]],
    raw: Dict[Pair, np.ndarray],
    inliers: Dict[Pair, np.ndarray],
    Fs: Dict[Pair, np.ndarray],
    rig=None,
) -> List[int]:
    model = camera_model_name(ctx.request)
    single = bool(ctx.opt("singleCamera", True))
    # Stereo footage: one camera per eye (each lens has its own intrinsics, shared by all of its frames) instead of
    # one per image, which leaves the mapper hundreds of unknown lenses.
    eye_of: Dict[str, int] = {}
    srig = ctx.request.get("rig") or {}
    if not single and str(srig.get("kind", "")).lower() == "stereo":
        for pair in srig.get("pairs") or []:
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                eye_of[Path(str(pair[0])).name] = 0
                eye_of[Path(str(pair[1])).name] = 1
    cam_for_eye: Dict[Tuple[int, int, int], int] = {}
    db = COLMAPDatabase.create(db_path)
    cam_for_size: Dict[Tuple[int, int], int] = {}
    cam_for_sensor: Dict[int, int] = {}
    ids = []
    for i, name in enumerate(ctx.names):
        w, h = sizes[i]
        prior = ctx.priors.get(name)
        if rig is not None:
            # one camera per rig sensor (COLMAP rigs identify sensors by camera); the views' exact pinhole
            # intrinsics when SplatWork sent them
            s = rig.sensor[i]
            if s not in cam_for_sensor:
                if rig.intrinsics:
                    fx, fy, cx, cy = rig.intrinsics
                    cam_for_sensor[s] = db.add_camera(CAMERA_MODEL_IDS["PINHOLE"], w, h, np.array([fx, fy, cx, cy], float), True)
                else:
                    params, has_prior = initial_camera_params(model, w, h, prior, exif_focal_px(ctx.image_dir / name, w, h))
                    cam_for_sensor[s] = db.add_camera(CAMERA_MODEL_IDS[model], w, h, params, has_prior)
            iid = db.add_image(rig.db_names[i], cam_for_sensor[s])
            db.add_keypoints(iid, kps_colmap[i].astype(np.float32))
            ids.append(iid)
            continue
        eye = eye_of.get(Path(name).name)
        if single and (w, h) in cam_for_size:
            cid = cam_for_size[(w, h)]
        elif eye is not None and (w, h, eye) in cam_for_eye:
            cid = cam_for_eye[(w, h, eye)]
        else:
            exif_f = None if (prior is not None and prior.intrinsics is not None) else exif_focal_px(ctx.image_dir / name, w, h)
            params, has_prior = initial_camera_params(model, w, h, prior, exif_f)
            cid = db.add_camera(CAMERA_MODEL_IDS[model], w, h, params, has_prior)
            cam_for_size[(w, h)] = cid
            if eye is not None:
                cam_for_eye[(w, h, eye)] = cid
        iid = db.add_image(name, cid)
        db.add_keypoints(iid, kps_colmap[i].astype(np.float32))
        ids.append(iid)
    # F was estimated on OpenCV-convention pixels; express it in COLMAP pixels (x_cv = x_colmap - 0.5)
    T = np.array([[1.0, 0, -0.5], [0, 1.0, -0.5], [0, 0, 1.0]])
    for (i, j), m in raw.items():
        if len(m):
            db.add_matches(ids[i], ids[j], m)
        if (i, j) in Fs:
            db.add_two_view_geometry(ids[i], ids[j], m[inliers[(i, j)]], F=T.T @ Fs[(i, j)] @ T, config=TWO_VIEW_UNCALIBRATED)
    db.close()
    return ids
