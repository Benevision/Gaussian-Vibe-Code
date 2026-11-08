"""Stereo footage helpers (task frames + stereo): layout detection, eye splitting, Arcturus calibration."""
import json

import numpy as np
import pytest

from splatprep import stereo as stl


def test_detect_layout_like_the_mac():
    assert stl.detect_layout(3840, 1080, 1, False, "clip.mp4", {})[:2] == ("sideBySide", False)
    assert stl.detect_layout(1920, 1080, 1, False, "clip_sbs.mp4", {})[:2] == ("sideBySide", True)  # half-width
    assert stl.detect_layout(1920, 2160, 1, False, "clip.mp4", {})[:2] == ("topBottom", False)
    assert stl.detect_layout(2464, 2464, 2, True, "Arcturus Vision.mp4", {})[0] == "dualStream"
    assert stl.detect_layout(1920, 1080, 1, False, "a.mp4", {}, has_right_video=True)[0] == "separateFiles"
    assert stl.detect_layout(2000, 1000, 1, False, "a.mp4", {})[0] == "sideBySide"
    assert stl.detect_layout(1920, 1080, 1, False, "a.mp4", {"layout": "topBottom", "anamorphic": True})[:2] == \
        ("topBottom", True)
    with pytest.raises(ValueError):
        stl.detect_layout(1920, 1080, 1, False, "a.mp4", {})


def test_split_eyes():
    rgb = np.zeros((4, 8, 3), np.uint8)
    rgb[:, 4:] = 200
    a, b = stl.split_eyes(rgb, "sideBySide", False)
    assert a.shape == (4, 4, 3) and a.max() == 0 and b.min() == 200
    tb = np.concatenate([np.zeros((3, 5, 3), np.uint8), np.full((3, 5, 3), 9, np.uint8)])
    a, b = stl.split_eyes(tb, "topBottom", False)
    assert a.shape == (3, 5, 3) and b.min() == 9
    pytest.importorskip("cv2")
    a, b = stl.split_eyes(rgb, "sideBySide", True)
    assert a.shape == (4, 8, 3)                               # anamorphic: un-squeezed to full width


CAL = {"__units": "mm", "cameras": [
    {"sourceCamera": "passthrough_right", "width": 2464, "height": 2464, "imageCircleDiameterInDeg": 145,
     "intrinsics": [{"cameraModel": "kb", "fx": 1467.0, "fy": 1469.4, "cx": 1371.1, "cy": 1245.6,
                     "k1": -0.028, "k2": -0.146, "k3": 0.0688, "k4": -0.0025}],
     "extrinsics": {"position": [-31.6, -4.5, 57.0]}},
    {"sourceCamera": "passthrough_left", "width": 2464, "height": 2464, "imageCircleDiameterInDeg": 145,
     "intrinsics": [{"cameraModel": "kb", "fx": 1468.1, "fy": 1469.7, "cx": 1084.9, "cy": 1247.8,
                     "k1": -0.017, "k2": -0.181, "k3": 0.111, "k4": -0.019}],
     "extrinsics": {"position": [32.2, -4.96, 57.3]}}]}


def test_arcturus_calibration():
    cal = stl.parse_offline_cal(json.dumps(CAL))
    assert cal["baseline"] == pytest.approx(0.0638, abs=2e-4)
    assert [c["eye"] for c in cal["cameras"]] == ["right", "left"]      # stream 0 is the right eye
    assert cal["lens"] == "fisheye" and cal["parallel"] is False
    left = cal["cameras"][1]
    m = stl.circle_mask(616, 616, left, 616 / 2464)
    assert m is not None and m[0, 0] == 0 and m[308, 308] == 255     # corners outside the 145° image circle
    rig = {"leftIntrinsics": left["intrinsics"], "rightIntrinsics": cal["cameras"][0]["intrinsics"],
           "calibrationWidth": 2464}
    f, cx, cy = stl.focal_for(rig, "left", 800, 800, None)
    assert f == pytest.approx(1468.1 * 800 / 2464) and cx == pytest.approx(1084.9 * 800 / 2464)
    assert stl.parse_offline_cal(None) is None and stl.parse_offline_cal("{}") is None


def test_focal_fallbacks():
    assert stl.focal_for({"horizontalFOV": 90.0}, "left", 200, 100, None)[0] == pytest.approx(100.0)
    assert stl.focal_for({}, "right", 360, 200, 36.0)[0] == pytest.approx(360.0)
    assert stl.focal_for({}, "left", 10, 10, None) is None
