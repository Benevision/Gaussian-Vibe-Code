"""Home video (VHS) source in the studio: the frames request, clip choice, training settings, camera groups."""
import json

from splatworker.studio import importers
from splatworker.studio.pipeline import stage_titles
from splatworker.studio.presets import SOURCE_KINDS, ProjectOptions


def test_home_video_frames_request_and_clip_numbers():
    assert "homevideo" in SOURCE_KINDS
    o = ProjectOptions(name="x", source_kind="homevideo", source="/tape.mp4", home_frames=300, home_clips="2, 5 x 3")
    req = o.frames_request("tape.mp4")
    assert req["task"] == "frames" and req["video"] == "tape.mp4"
    assert req["vhs"]["clips"] == [1, 2, 4]              # 1-based numbers in the UI → splatprep ids
    assert req["vhs"]["totalFrames"] == 300 and req["vhs"]["denoise"]
    assert ProjectOptions(name="x", source_kind="homevideo", source="/t.mp4").home_clip_ids() is None
    assert "equirect" not in req and "stereo" not in req
    assert stage_titles(o)[0] == "Restoring home video"


def test_home_video_training_settings_unless_overridden():
    o = ProjectOptions(name="x", source_kind="homevideo", source="/t.mp4", preset="high")
    c = o.train_config()
    assert c["shDegree"] == 1 and c["transientMasking"] is True and c["maxImageSize"] <= 1024
    o.train_overrides = {"shDegree": 3}
    assert o.train_config()["shDegree"] == 3
    assert ProjectOptions(name="x", source_kind="video", source="/v.mp4", preset="high").train_config().get("shDegree", 3) == 3


def test_camera_groups_and_report_from_frames_json(tmp_path):
    (tmp_path / "frames.json").write_text(json.dumps({"frames": [
        {"image": "images/000000.jpg", "clip": 1, "cameraGroup": 1000},
        {"image": "images/000001.jpg", "clip": 1, "cameraGroup": 1001},
        {"image": "images/000002.jpg", "clip": 4, "cameraGroup": 4000}]}))
    assert importers.camera_groups_from_frames(tmp_path) == {"000000.jpg": 1000, "000001.jpg": 1001, "000002.jpg": 4000}
    (tmp_path / "vhs.json").write_text(json.dumps({
        "width": 720, "height": 576, "pixelAspect": 1.06667, "uniqueFps": 25.0, "used": [1],
        "clips": [{"id": 0, "start": 0, "end": 1, "verdict": "short", "reason": "too short"},
                  {"id": 1, "start": 1, "end": 30, "verdict": "good", "reason": "the camera moves", "selected": 40},
                  {"id": 2, "start": 30, "end": 50, "verdict": "rotationOnly", "reason": "the camera only turns"}]}))
    lines = importers.home_video_report(tmp_path)
    assert "720×576" in lines[0]
    assert any("clip 2:" in l and "used" in l and "40 frames" in l for l in lines)
    assert any("clip 3:" in l and "left out" in l for l in lines)
    assert not any("clip 1:" in l for l in lines)
    (tmp_path / "frames.json").write_text(json.dumps({"frames": [{"image": "images/a.jpg"}]}))
    assert importers.camera_groups_from_frames(tmp_path) is None
