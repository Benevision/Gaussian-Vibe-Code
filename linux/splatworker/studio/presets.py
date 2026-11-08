"""Training presets and the New Project options, turned into splatprep / TrainingConfig requests.

Presets and defaults are the Mac's (``TrainingConfig.Preset``, ``NewProjectSheet``); setting ids are the
SettingsCatalog ids so the Settings Reference documents every control.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

PRESETS: Dict[str, Dict[str, Any]] = {
    "preview": {"title": "Quick preview", "subtitle": "≈2–4 min on an RTX 4080. Check coverage before a long run.",
                "config": {"iterations": 5000, "maxImageSize": 1000, "maxGaussians": 300_000, "shDegree": 1,
                           "densifyStop": 4000}},
    "balanced": {"title": "Balanced", "subtitle": "≈8–12 min. Good for sharing and most scenes.",
                 "config": {"iterations": 15_000, "maxImageSize": 1400, "maxGaussians": 800_000,
                            "densifyStop": 12_000}},
    "high": {"title": "High quality", "subtitle": "≈20–30 min. Sharp detail, 1.5 M splats.",
             "config": {"iterations": 30_000, "maxImageSize": 1600, "maxGaussians": 1_500_000,
                        "densifyStop": 14_000}},
    "ultra": {"title": "Maximum", "subtitle": "≈45 min+. Full resolution, 3 M splats, lots of GPU memory.",
              "config": {"iterations": 40_000, "maxImageSize": 2400, "maxGaussians": 3_000_000,
                         "densifyStop": 19_000}},
}

# The Mac app's TrainingConfig defaults (the worker's own defaults differ for depthWeight / bilateralGrid; the
# Studio sends these explicitly so a project trains the same on both machines).
MAC_DEFAULTS: Dict[str, Any] = {"depthWeight": 0.1, "bilateralGrid": True}

# Advanced training settings shown in the Studio, (catalog id, config key, kind, choices / range)
TRAIN_FIELDS: List[tuple] = [
    ("train.iterations", "iterations", "int", (500, 100_000)),
    ("train.maxImageSize", "maxImageSize", "int", (256, 8192)),
    ("train.shDegree", "shDegree", "int", (0, 3)),
    ("train.shDegreeInterval", "shDegreeInterval", "int", (0, 10_000)),
    ("train.background", "background", "choice", ("random", "black", "white")),
    ("train.antialiased", "antialiased", "bool", None),
    ("train.seed", "seed", "int", (0, 2 ** 31 - 1)),
    ("train.strategy", "strategy", "choice", ("mcmc", "adaptive")),
    ("train.maxGaussians", "maxGaussians", "int", (10_000, 20_000_000)),
    ("train.optimizerPrecision", "optimizerPrecision", "choice", ("fp32", "compact")),
    ("train.densifyStart", "densifyStart", "int", (0, 100_000)),
    ("train.densifyStop", "densifyStop", "int", (0, 100_000)),
    ("train.densifyInterval", "densifyInterval", "int", (1, 10_000)),
    ("train.gradThreshold", "densifyGradThreshold", "float", (0.0, 0.01)),
    ("train.absGrad", "absGrad", "bool", None),
    ("train.opacityReset", "opacityResetInterval", "int", (0, 100_000)),
    ("train.pruneOpacity", "pruneOpacity", "float", (0.0, 0.5)),
    ("train.initialOpacity", "initialOpacity", "float", (0.001, 1.0)),
    ("train.randomInitCount", "randomInitCount", "int", (0, 5_000_000)),
    ("train.longAxisSplit", "longAxisSplit", "bool", None),
    ("train.lrPosition", "lrPosition", "float", (0.0, 0.01)),
    ("train.lrColor", "lrColor", "float", (0.0, 0.1)),
    ("train.lrOpacity", "lrOpacity", "float", (0.0, 1.0)),
    ("train.lrScale", "lrScale", "float", (0.0, 0.1)),
    ("train.lrRotation", "lrRotation", "float", (0.0, 0.1)),
    ("train.ssimWeight", "ssimWeight", "float", (0.0, 1.0)),
    ("train.depthWeight", "depthWeight", "float", (0.0, 10.0)),
    ("train.transientMasking", "transientMasking", "bool", None),
    ("train.transientSensitivity", "transientSensitivity", "float", (1.0, 10.0)),
    ("train.transientMaxViewGap", "transientMaxViewGap", "float", (0.0, 180.0)),
    ("train.depthEdgeAware", "depthEdgeAware", "bool", None),
    ("train.anisotropyReg", "anisotropyReg", "float", (0.0, 10.0)),
    ("train.anisotropyMaxRatio", "anisotropyMaxRatio", "float", (1.0, 100.0)),
    ("train.bilateralGrid", "bilateralGrid", "bool", None),
    ("train.appearanceCorrection", "appearanceCorrection", "bool", None),
    ("train.appearanceLR", "appearanceLR", "float", (0.0, 1.0)),
    ("train.appearanceReg", "appearanceReg", "float", (0.0, 1.0)),
    ("train.opacityReg", "opacityReg", "float", (0.0, 1.0)),
    ("train.scaleReg", "scaleReg", "float", (0.0, 1.0)),
    ("train.noiseLR", "noiseLR", "float", (0.0, 5_000_000.0)),
    ("train.evalInterval", "evalInterval", "int", (0, 100_000)),
]

FEATURES = [("superpoint_lightglue", "SuperPoint + LightGlue"), ("aliked_lightglue", "ALIKED + LightGlue"),
            ("disk_lightglue", "DISK + LightGlue"), ("sift_lightglue", "SIFT + LightGlue"),
            ("colmap_sift", "COLMAP SIFT"), ("loftr", "LoFTR (dense)"), ("mast3r", "MASt3R (few photos)"),
            ("vggt", "VGGT (feed-forward)")]
MASK_MODELS = [("none", "None"), ("person", "Remove people & pets"), ("sam2_auto", "Remove moving things")]
SOURCE_KINDS = {
    "photos": "Photos folder", "video": "Video", "video360": "360° video", "capture": "iPhone / headset capture",
    "colmap": "COLMAP project", "nerfstudio": "Nerfstudio transforms.json", "dataset": "Prepared dataset",
}


def training_config(preset: str = "high", overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = dict(MAC_DEFAULTS)
    cfg.update(PRESETS.get(preset, PRESETS["high"])["config"])
    cfg.update(overrides or {})
    return cfg


@dataclass
class ProjectOptions:
    """Everything the New Project dialog asks (setting ids in comments)."""
    name: str
    source_kind: str                         # SOURCE_KINDS
    source: str                              # path (folder / file)
    max_image_size: int = 1600               # source.maxImageSize (0 = full)
    video_fps: float = 2.0                   # source.videoFPS
    target_frames: int = 0                   # source.targetFrames (0 = use fps)
    sharpest_in_window: bool = True          # source.sharpestInWindow
    hdr_tone_map: bool = True                # source.hdr
    blur_rejection: bool = True              # source.blurRejection
    blur_threshold: float = 0.35             # fraction of the median sharpness below which a photo is dropped
    views_360: int = 8                       # source.360views
    fov_360: float = 90.0                    # source.360fov
    drop_nadir_360: bool = True              # source.360dropNadir
    holdout_every: int = 25                  # source.holdout (0 = none)
    solver: str = "learned"                  # sfm.solver: learned | poses (use poses from the source)
    features: str = "superpoint_lightglue"   # sfm.features
    matching: str = "auto"                   # sfm.matching
    mapper: str = "auto"                     # sfm.mapper
    camera_model: str = "auto"               # sfm.cameraModel
    refine_capture_poses: bool = True        # sfm.refineCapturePoses
    mask_models: List[str] = field(default_factory=list)   # mask.model (stacked: each is ANDed in)
    mask_invert: bool = False                # mask.invert
    sky: bool = True                         # mask.sky
    preset: str = "high"                     # train.preset
    train_overrides: Dict[str, Any] = field(default_factory=dict)
    auto_train: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProjectOptions":
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    # ---------------------------------------------------------------- requests
    def frames_request(self, video_name: str) -> Dict[str, Any]:
        vo: Dict[str, Any] = {"targetFPS": float(self.video_fps), "pickSharpest": bool(self.sharpest_in_window),
                              "toneMapHDR": bool(self.hdr_tone_map),
                              "maxImageSize": int(self.max_image_size) if self.max_image_size else 100_000}
        if self.target_frames:
            vo["targetFrameCount"] = int(self.target_frames)
        req: Dict[str, Any] = {"task": "frames", "video": video_name, "outputDir": "out", "videoOptions": vo}
        if self.source_kind == "video360":
            n = max(3, int(self.views_360))
            req["equirect"] = {"fovDegrees": float(self.fov_360),
                               "outputSize": min(int(self.max_image_size or 1600), 2400),
                               "rings": [{"pitch": 0, "count": n},
                                         {"pitch": 35, "count": max(2, n // 2), "yawOffset": 180.0 / n},
                                         {"pitch": -35, "count": max(2, n // 2), "yawOffset": 180.0 / n}],
                               "includeNadir": not self.drop_nadir_360,
                               "maskBelowLatitude": -60.0 if self.drop_nadir_360 else None}
        return req

    def mask_request(self, model: str, out: str) -> Dict[str, Any]:
        return {"task": "masks", "imagesDir": "images", "outputDir": out, "maskModel": model,
                "invertMask": bool(self.mask_invert)}

    def sky_request(self) -> Dict[str, Any]:
        return {"task": "masks", "imagesDir": "images", "outputDir": "splatprep-sky", "maskModel": "sky",
                "invertMask": True}

    def sfm_request(self, has_masks: bool, has_sky: bool, has_priors: bool) -> Dict[str, Any]:
        req: Dict[str, Any] = {"task": "sfm", "imagesDir": "images", "outputDir": "out",
                               "features": self.features, "matching": self.matching, "mapper": self.mapper,
                               "cameraModel": self.camera_model}
        if has_masks:
            req["masksDir"] = "masks"
        if has_sky:
            req["skyMasksDir"] = "splatprep-sky/masks"
        if has_priors:
            req["priorPoses"] = "priors.json"
            req["priorMode"] = "auto" if self.refine_capture_poses else "solve"
        return req

    def train_config(self) -> Dict[str, Any]:
        return training_config(self.preset, self.train_overrides)
