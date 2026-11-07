"""Feature / SfM backends.  Importing this package never imports torch."""

from .base import Availability, Backend, FeatureMatchBackend, MatchingData, SfMContext, SfMOutput  # noqa: F401


def builtin_backends():
    from .colmap_sift import ColmapSiftBackend
    from .lightglue_backend import LightGlueBackend
    from .loftr import LoFTRBackend
    from .mast3r import MASt3RBackend
    from .opencv_sift import OpenCVSiftBackend
    from .vggt import VGGTBackend

    return [
        ColmapSiftBackend(),
        LightGlueBackend("sift"),
        LightGlueBackend("superpoint"),
        LightGlueBackend("disk"),
        LightGlueBackend("aliked"),
        LoFTRBackend(),
        MASt3RBackend(),
        VGGTBackend(),
        OpenCVSiftBackend(),
    ]
