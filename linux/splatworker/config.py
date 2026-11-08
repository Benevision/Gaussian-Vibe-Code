"""Training configuration (PROTOCOL.md §3) and server settings."""

from __future__ import annotations

import dataclasses
import math
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional


class ConfigError(ValueError):
    """Raised when a TrainingConfig dictionary contains an invalid value."""


_STRATEGIES = ("mcmc", "adaptive")
_BACKGROUNDS = ("black", "white", "random")
_OPTIMIZER_PRECISIONS = ("fp32", "compact")



LOCAL_SECRET_ENV = "SPLATWORK_LOCAL_SECRET"   # see ServerSettings.local_secret

@dataclass
class TrainingConfig:
    """Every key of PROTOCOL.md §3 with its documented default.

    Keys are the protocol's camelCase names so that a JSON object from the Mac
    maps 1:1 onto this dataclass. Unknown keys are ignored (forward
    compatibility); keys that the caller actually supplied are remembered in
    ``explicit`` so that context dependent defaults (``densifyStop`` for MCMC,
    ``depthWeight`` when depth is present, regularisers for ADC, ...) can be
    resolved later.
    """

    iterations: int = 30000
    maxImageSize: int = 1600
    shDegree: int = 3
    shDegreeInterval: int = 1000
    strategy: str = "mcmc"
    maxGaussians: int = 1_500_000
    densifyStart: int = 500
    densifyStop: int = 15000
    densifyInterval: int = 100
    densifyGradThreshold: float = 0.0002
    absGrad: bool = False
    opacityResetInterval: int = 3000
    pruneOpacity: float = 0.005
    lrPosition: float = 0.00016
    lrPositionFinal: float = 0.0000016
    lrColor: float = 0.0025
    lrSHRest: float = 0.000125
    lrOpacity: float = 0.05
    lrScale: float = 0.005
    lrRotation: float = 0.001
    ssimWeight: float = 0.2
    opacityReg: float = 0.01
    scaleReg: float = 0.01
    noiseLR: float = 500000.0
    antialiased: bool = True
    depthWeight: float = 0.0
    background: str = "random"
    initialOpacity: float = 0.1
    randomInitCount: int = 100000
    seed: int = 42
    previewInterval: int = 500
    evalInterval: int = 2000
    # Robust training against moving objects (same algorithm as the Mac Metal trainer)
    transientMasking: bool = False   # opt-in: on real captures it ignored hard detail as 'moving'
    transientSensitivity: float = 3.0
    transientStart: int = 1000
    transientMaxViewGap: float = 15.0   # off for sparse captures (median nearest-view gap, degrees)
    # Depth / shape regularisers (same as the Mac Metal trainer)
    depthEdgeAware: bool = True
    anisotropyReg: float = 0.1
    anisotropyMaxRatio: float = 10.0
    # Depth-normal consistency (rendered-depth normals vs sensor-depth normals) and the flat-disc regulariser,
    # as on the Mac
    normalWeight: float = 0.0
    normalStart: int = 1000
    flatnessReg: float = 0.0
    # Photos without sensor depth: predict depth with a monocular model (Depth Anything V2 Small), fitted to the
    # dataset's 3D points, for the depth-normal loss. "auto" = when no training photo has sensor depth and
    # normalWeight > 0; "on"; "off". monoDepthWeight: depth-loss weight for those predicted maps (0 = normals only).
    depthPriors: str = "auto"
    monoDepthWeight: float = 0.0
    # Spread MCMC clones along the long axis (ImprovedGS / IGS+), as on the Mac. On by default: without it the
    # 30k schedule collapsed on large scenes once the splat budget filled (1,258-photo drone set: 16.3 dB at
    # step 1.4k → 12.5 dB at 9k; with it 16.5–16.9 dB and rising at 9k).
    longAxisSplit: bool = True
    # Per-photo camera pose refinement (experimental until measured)
    poseRefinement: bool = False
    poseRefinementLR: float = 1e-4
    poseRefinementReg: float = 1e-6
    poseRefinementStart: int = 500
    # Per-photo colour correction (3×3 matrix + offset), same as the Mac Metal trainer
    appearanceCorrection: bool = True
    appearanceLR: float = 0.02
    appearanceReg: float = 0.001
    # Per-photo LOCAL colour correction (bilateral grid: 16×16 image cells × 8 brightness levels of 3×4 colour
    # transforms per photo). For lighting that changes across one photo between shots — moving sun / shadows on
    # long drone flights. Replaces appearanceCorrection when on.
    bilateralGrid: bool = False
    bilateralGridLR: float = 2e-3
    bilateralGridTV: float = 10.0
    # Coarse-to-fine: ¼ → ½ → ¾ → full image resolution over the first half of the densification window.
    progressiveResolution: bool = False
    # SSIM contrast/structure on the uncorrected render (luminance on the corrected one) when per-photo colour
    # correction is on, so the correction can't hide missing detail.
    decoupledSSIM: bool = False
    # Far-field shell of this many splats (0 = off) for distant land / sky beyond the capture.
    backgroundShell: int = 0
    # Adam moment storage: "fp32" (plain torch Adam) or "compact" (8-bit SH-rest moments, bf16 for the rest) —
    # ~1/3 less training memory per splat at SH degree 3, so more splats fit on the GPU (compact_adam.py).
    optimizerPrecision: str = "fp32"
    # Learned skybox at infinity (skybox.py): "auto" = on when any photo has a sky mask
    skybox: str = "auto"
    skyboxResolution: int = 256          # panorama height (width = 2×)
    skyAlphaWeight: float = 0.05         # keeps splats out of pixels the sky mask marks as sky
    # PROTOCOL.md §7 (block-parallel training)
    region: Optional[Region] = None
    cameraIds: Optional[List[int]] = None

    explicit: frozenset = field(default_factory=frozenset, repr=False, compare=False)

    # ------------------------------------------------------------------ parsing
    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "TrainingConfig":
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ConfigError("config must be a JSON object")
        kwargs: Dict[str, Any] = {}
        explicit = set()
        for f in fields(cls):
            if f.name == "explicit" or f.name not in data:
                continue
            value = data[f.name]
            if value is None:
                continue  # treat null as "use default"
            if f.name == "region":
                kwargs[f.name] = Region.from_dict(value)
            elif f.name == "cameraIds":
                kwargs[f.name] = _coerce_id_list(value)
            else:
                kwargs[f.name] = _coerce(f.name, f.type, value)
            explicit.add(f.name)
        cfg = cls(**kwargs, explicit=frozenset(explicit))
        cfg.validate()
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        d.pop("explicit", None)
        if self.region is not None:
            d["region"] = self.region.to_dict()
        return d

    def validate(self) -> None:
        def need(cond: bool, msg: str) -> None:
            if not cond:
                raise ConfigError(msg)

        need(self.iterations >= 1, "iterations must be >= 1")
        need(self.maxImageSize >= 16, "maxImageSize must be >= 16")
        need(0 <= self.shDegree <= 3, "shDegree must be between 0 and 3")
        need(self.shDegreeInterval >= 1, "shDegreeInterval must be >= 1")
        need(self.strategy in _STRATEGIES, f"strategy must be one of {list(_STRATEGIES)}")
        need(self.background in _BACKGROUNDS, f"background must be one of {list(_BACKGROUNDS)}")
        need(self.skybox in ("auto", "on", "off"), "skybox must be auto, on or off")
        need(16 <= self.skyboxResolution <= 2048, "skyboxResolution must be 16…2048")
        need(0 <= self.skyAlphaWeight <= 10, "skyAlphaWeight must be 0…10")
        need(self.optimizerPrecision in _OPTIMIZER_PRECISIONS,
             f"optimizerPrecision must be one of {list(_OPTIMIZER_PRECISIONS)}")
        need(self.maxGaussians >= 1, "maxGaussians must be >= 1")
        need(self.densifyInterval >= 1, "densifyInterval must be >= 1")
        need(self.opacityResetInterval >= 1, "opacityResetInterval must be >= 1")
        need(0.0 < self.initialOpacity < 1.0, "initialOpacity must be in (0, 1)")
        need(0.0 <= self.ssimWeight <= 1.0, "ssimWeight must be in [0, 1]")
        need(0.0 <= self.pruneOpacity < 1.0, "pruneOpacity must be in [0, 1)")
        need(self.randomInitCount >= 1, "randomInitCount must be >= 1")
        need(self.previewInterval >= 0, "previewInterval must be >= 0 (0 disables previews)")
        need(self.evalInterval >= 0, "evalInterval must be >= 0 (0 disables eval)")
        for name in ("lrPosition", "lrPositionFinal", "lrColor", "lrSHRest", "lrOpacity",
                     "lrScale", "lrRotation", "bilateralGridLR"):
            need(getattr(self, name) > 0, f"{name} must be > 0")
        for name in ("opacityReg", "scaleReg", "noiseLR", "depthWeight", "densifyGradThreshold", "bilateralGridTV", "backgroundShell"):
            need(getattr(self, name) >= 0, f"{name} must be >= 0")

    # ------------------------------------------------------- resolved defaults
    def is_set(self, key: str) -> bool:
        return key in self.explicit

    def effective_densify_stop(self) -> int:
        if self.strategy == "mcmc" and not self.is_set("densifyStop"):
            return 25000
        return self.densifyStop

    def effective_grad_threshold(self) -> float:
        # AbsGS gradients are larger; protocol: "threshold auto x4 -> 0.0008".
        return self.densifyGradThreshold * (4.0 if self.absGrad else 1.0)

    def effective_opacity_reg(self) -> float:
        if self.strategy == "adaptive" and not self.is_set("opacityReg"):
            return 0.0
        return self.opacityReg

    def effective_scale_reg(self) -> float:
        if self.strategy == "adaptive" and not self.is_set("scaleReg"):
            return 0.0
        return self.scaleReg

    def effective_depth_weight(self, has_depth: bool) -> float:
        if not has_depth:
            return 0.0
        if self.is_set("depthWeight"):
            return self.depthWeight
        return 0.1


@dataclass(frozen=True)
class Region:
    """Axis-aligned training block (PROTOCOL.md §7).

    ``margin`` is a fraction of the box size: each side of the box is pushed
    out by ``margin * (max - min)`` along its axis. Gaussians are kept and
    densified only inside the expanded box; the Mac later crops to the
    un-expanded box when merging blocks.
    """

    min: tuple
    max: tuple
    margin: float = 0.0

    @classmethod
    def from_dict(cls, value: Any) -> "Region":
        if not isinstance(value, dict):
            raise ConfigError("region must be an object {min, max, margin}")
        try:
            lo = tuple(float(x) for x in value["min"])
            hi = tuple(float(x) for x in value["max"])
        except (KeyError, TypeError, ValueError):
            raise ConfigError("region.min and region.max must be [x, y, z] number arrays")
        if len(lo) != 3 or len(hi) != 3 or not all(map(math.isfinite, lo + hi)):
            raise ConfigError("region.min and region.max must be [x, y, z] number arrays")
        if any(a >= b for a, b in zip(lo, hi)):
            raise ConfigError("region.min must be < region.max on every axis")
        margin = value.get("margin", 0.0)
        if margin is None:
            margin = 0.0
        if isinstance(margin, bool) or not isinstance(margin, (int, float)) \
                or not math.isfinite(margin) or margin < 0:
            raise ConfigError("region.margin must be a non-negative number")
        return cls(min=lo, max=hi, margin=float(margin))

    def to_dict(self) -> Dict[str, Any]:
        return {"min": list(self.min), "max": list(self.max), "margin": self.margin}

    def expanded(self):
        """Return (min, max) of the margin-expanded box as tuples."""
        size = [b - a for a, b in zip(self.min, self.max)]
        lo = tuple(a - self.margin * s for a, s in zip(self.min, size))
        hi = tuple(b + self.margin * s for b, s in zip(self.max, size))
        return lo, hi


def _coerce_id_list(value: Any) -> List[int]:
    if not isinstance(value, list):
        raise ConfigError("cameraIds must be a list of integers")
    out = []
    for v in value:
        if isinstance(v, bool) or not isinstance(v, (int, float)) or float(v) != int(v):
            raise ConfigError("cameraIds must be a list of integers")
        out.append(int(v))
    return out


def _coerce(name: str, typ: Any, value: Any) -> Any:
    t = typ if isinstance(typ, str) else getattr(typ, "__name__", str(typ))
    try:
        if t == "bool":
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)) and value in (0, 1):
                return bool(value)
            if isinstance(value, str) and value.lower() in ("true", "false", "1", "0"):
                return value.lower() in ("true", "1")
            raise TypeError
        if t == "int":
            if isinstance(value, bool):
                raise TypeError
            if isinstance(value, float):
                if not value.is_integer():
                    raise TypeError
                return int(value)
            return int(value)
        if t == "float":
            if isinstance(value, bool):
                raise TypeError
            v = float(value)
            if not math.isfinite(v):
                raise TypeError
            return v
        if t == "str":
            if not isinstance(value, str):
                raise TypeError
            return value.lower()
    except (TypeError, ValueError):
        pass
    raise ConfigError(f"invalid value for {name}: {value!r} (expected {t})")


# --------------------------------------------------------------------------
# Server settings
# --------------------------------------------------------------------------

@dataclass
class ServerSettings:
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get(
        "SPLATWORKER_DATA_DIR", "~/.local/share/splatworker")).expanduser())
    host: str = "0.0.0.0"
    port: int = 7878
    token: Optional[str] = None
    name: Optional[str] = None
    gpus: Optional[List[int]] = None          # None = all visible GPUs
    mdns: bool = True
    advertise_ips: Optional[List[str]] = None  # None = auto-detect
    max_upload_bytes: int = 20 * 1024 ** 3     # compressed zip size
    max_extracted_bytes: int = 40 * 1024 ** 3  # total uncompressed size
    max_files: int = 100_000
    max_compression_ratio: float = 200.0       # per entry (zip bomb guard)
    # The desktop app on this PC may register its own folders as datasets (no upload / copy), export into them,
    # census a file by path and reorder the queue. It proves who it is with ``local_secret`` (per launch, passed
    # in the server's environment, sent as the X-SplatWork-Local header); loopback alone is not enough.
    allow_local_datasets: bool = False
    local_secret: Optional[str] = None

    @property
    def local_enabled(self) -> bool:
        return bool(self.allow_local_datasets and self.local_secret and len(self.local_secret) >= 16)

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir).expanduser().resolve()
        if self.token is not None and self.token.strip() == "":
            self.token = None
        if self.name is None:
            import socket
            self.name = socket.gethostname().split(".")[0] or "splatworker"

    @property
    def datasets_dir(self) -> Path:
        return self.data_dir / "datasets"

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    @property
    def tmp_dir(self) -> Path:
        return self.data_dir / "tmp"
