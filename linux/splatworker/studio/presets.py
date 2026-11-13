"""Training presets and the New Project options, turned into splatprep / TrainingConfig requests.

Presets and defaults are the Mac's (``TrainingConfig.Preset``, ``NewProjectSheet``); setting ids are the
SettingsCatalog ids so the Settings Reference documents every control.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
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
    "photos": "Photos folder", "video": "Video", "homevideo": "Home video (VHS, camcorder tape)",
    "video360": "360° video", "photos360": "360° photos folder",
    "stereo": "Stereo footage", "capture": "iPhone / headset capture",
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
    stereo_layout: str = "auto"              # auto | sideBySide | topBottom | dualStream | separateFiles
    stereo_right: str = ""                   # the right-eye video (separateFiles)
    stereo_swap: bool = False                # swap left and right
    stereo_baseline_mm: float = 0.0          # rig baseline (0 = from the file's calibration, else unknown)
    stereo_fov: float = 0.0                  # horizontal FOV of one eye in degrees (0 = calibration / metadata)
    stereo_both_eyes: bool = True            # right-eye images train too
    stereo_depth: bool = False               # block-matching depth for the left eye
    mask_static: bool = False                # mask.staticRegions: mask fixed parts of the frame
    holdout_every: int = 25                  # source.holdout (0 = none)
    use_location: bool = True                # source.useLocation: imported photos keep their GPS (only that)
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
    train_on: str = "local"                  # net.placement: "local" (this PC) | "all" (block-parallel) | device id
    split_above: int = 400                   # net.blockThreshold: "all" splits scenes with this many photos or more
    home_frames: int = 400                   # source.homeVideoFrames: frames over all clips (home video)
    home_clips: str = ""                     # home video: clip numbers to use ("2, 5"; empty = the best clip)
    home_restore: bool = True                # source.homeVideoRestore: denoise, colour, tracking-line repair

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProjectOptions":
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    def with_storage_for_preset(self) -> "ProjectOptions":
        """A preset that trains at a higher resolution than photos are stored at by default (Maximum: 2400 px) gets
        photos stored at that resolution, unless the stored size was changed from the default (as on the Mac)."""
        o = dataclasses.replace(self)
        default = ProjectOptions.__dataclass_fields__["max_image_size"].default
        want = int(PRESETS.get(o.preset, {}).get("config", {}).get("maxImageSize", 0))
        if o.auto_train and o.max_image_size == default and want > default:
            o.max_image_size = want
        return o

    # ---------------------------------------------------------------- requests
    def frames_request(self, video_name: str) -> Dict[str, Any]:
        vo: Dict[str, Any] = {"targetFPS": float(self.video_fps), "pickSharpest": bool(self.sharpest_in_window),
                              "toneMapHDR": bool(self.hdr_tone_map),
                              "maxImageSize": int(self.max_image_size) if self.max_image_size else 100_000}
        if self.target_frames:
            vo["targetFrameCount"] = int(self.target_frames)
        req: Dict[str, Any] = {"task": "frames", "video": video_name, "outputDir": "out", "videoOptions": vo}
        if self.source_kind == "video360":
            req["equirect"] = self.equirect_options()
        if self.source_kind == "stereo":
            req["stereo"] = self.stereo_options()
        if self.source_kind == "homevideo":
            req["vhs"] = self.home_video_options()
        return req

    def home_clip_ids(self) -> Optional[List[int]]:
        """Clip numbers as the user sees them (1-based) → splatprep clip ids; None = the recommended clip."""
        ids = [int(t) - 1 for t in self.home_clips.replace(";", ",").replace(" ", ",").split(",") if t.strip().isdigit()]
        return sorted({i for i in ids if i >= 0}) or None

    def home_video_options(self) -> Dict[str, Any]:
        """splatprep ``frames`` request ``vhs`` (tools/splatprep/vhs.py): old tape footage restored and split into clips."""
        return {"clips": self.home_clip_ids(), "totalFrames": int(self.home_frames), "denoise": self.home_restore,
                "colour": self.home_restore, "repairDropouts": self.home_restore}

    def equirect_options(self) -> Dict[str, Any]:
        """The Mac's ``Equirect360Options`` for 360° video (worker) and 360° photos (this side)."""
        n = max(3, int(self.views_360))
        return {"fovDegrees": float(self.fov_360), "outputSize": min(int(self.max_image_size or 1600), 2400),
                "rings": [{"pitch": 0, "count": n},
                          {"pitch": 35, "count": max(2, n // 2), "yawOffset": 180.0 / n},
                          {"pitch": -35, "count": max(2, n // 2), "yawOffset": 180.0 / n}],
                "includeNadir": not self.drop_nadir_360,
                "maskBelowLatitude": -60.0 if self.drop_nadir_360 else None}

    def stereo_options(self) -> Dict[str, Any]:
        """splatprep ``frames`` request ``stereo`` (PROTOCOL.md §4): how the two eyes are stored."""
        so: Dict[str, Any] = {"layout": self.stereo_layout or "auto", "swapEyes": bool(self.stereo_swap)}
        if self.stereo_layout == "separateFiles" or self.stereo_right:
            so["layout"] = "separateFiles"
            so["rightVideo"] = "_right-eye" + (Path(self.stereo_right).suffix or ".mp4")
        if self.stereo_baseline_mm:
            so["baseline"] = float(self.stereo_baseline_mm) / 1000.0
        if self.stereo_fov:
            so["horizontalFOV"] = float(self.stereo_fov)
        return so

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
        cfg = training_config(self.preset)
        if self.source_kind == "homevideo":
            # as on the Mac: tape colour carries no view-dependent detail, and people walk through most shots
            cfg["shDegree"] = min(int(cfg.get("shDegree", 3)), 1)
            cfg["transientMasking"] = True
            cfg["maxImageSize"] = min(int(cfg.get("maxImageSize", 1600)), 1024)
        cfg.update(self.train_overrides or {})
        return cfg
