"""dynamic / sam2_track masks, prompt parsing, merge semantics, chunked tracking and the
maskserver protocol — all model calls are faked (no torch / weights needed)."""

import io
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from splatprep.errors import RequestError
from splatprep.masks import merge_keep, save_mask
from splatprep.masks.detect import Detection, filter_detections
from splatprep.masks.prompts import DEFAULT_DYNAMIC_CLASSES, FramePrompt, mask_options, parse_classes, parse_mask_prompts
from splatprep.masks.track import FrameStore, PackedMasks, needs_correction, plan_chunks, track_sequence

NAMES = ["seq/frame_000.jpg", "seq/frame_001.jpg", "seq/frame_002.jpg", "other/frame_002.jpg"]


# ---------------------------------------------------------------- prompt parsing


def test_parse_mask_prompts_resolution_ids_and_merge():
    req = {"maskPrompts": [
        {"image": "seq/frame_001.jpg", "points": [[10, 20, 1], [5, 5, 0]]},
        {"image": "frame_000.jpg", "boxes": [[30, 40, 10, 5]], "objectId": 1},   # basename, unordered box
        {"image": "frame_001", "points": [[1, 2]], "objectId": 1},              # stem; label defaults to 1
        {"image": "seq/frame_001.jpg", "boxes": [[0, 0, 4, 4]], "objectId": 1},  # merges into same frame+object
    ]}
    ps = parse_mask_prompts(req, NAMES)
    # first entry has no id → gets 2 (1 is taken explicitly)
    assert [(p.frame, p.object_id) for p in ps] == [(1, 2), (0, 1), (1, 1)]
    assert ps[0].labels == [1, 0]
    assert ps[1].boxes == [[10.0, 5.0, 30.0, 40.0]]  # normalised to x0<x1, y0<y1
    assert ps[2].points == [[1.0, 2.0]] and ps[2].labels == [1] and ps[2].boxes == [[0.0, 0.0, 4.0, 4.0]]


def test_legacy_mask_prompt_is_first_image_object_1():
    ps = parse_mask_prompts({"maskPrompt": {"points": [[3, 4, 1]]}}, NAMES)
    assert len(ps) == 1 and ps[0].frame == 0 and ps[0].object_id == 1
    both = parse_mask_prompts({"maskPrompt": {"boxes": [[0, 0, 9, 9]]},
                               "maskPrompts": [{"image": "seq/frame_002.jpg", "points": [[1, 1, 1]]}]}, NAMES)
    assert [(p.frame, p.object_id) for p in both] == [(0, 1), (2, 2)]


@pytest.mark.parametrize("bad", [
    {"maskPrompts": {"image": "x"}},
    {"maskPrompts": [{"image": "nope.jpg", "points": [[1, 1, 1]]}]},
    {"maskPrompts": [{"image": "frame_002.jpg", "points": [[1, 1, 1]]}]},  # ambiguous basename
    {"maskPrompts": [{"image": "seq/frame_000.jpg"}]},                    # no points/boxes
    {"maskPrompts": [{"image": "seq/frame_000.jpg", "points": [[1, 1, 2]]}]},
    {"maskPrompts": [{"image": "seq/frame_000.jpg", "points": [["a", 1]]}]},
    {"maskPrompts": [{"image": "seq/frame_000.jpg", "boxes": [[1, 1, 1, 9]]}]},
    {"maskPrompts": [{"image": "seq/frame_000.jpg", "boxes": [[1, 2, 3]]}]},
])
def test_parse_mask_prompts_errors(bad):
    with pytest.raises(RequestError):
        parse_mask_prompts(bad, NAMES)


def test_classes_and_options():
    assert parse_classes({}) == DEFAULT_DYNAMIC_CLASSES
    assert parse_classes({"maskClasses": ["cat", " dog ", "cat", ""]}) == ["cat", "dog"]
    assert parse_classes({"maskClasses": "person, kite"}) == ["person", "kite"]
    with pytest.raises(RequestError):
        parse_classes({"maskClasses": []})
    o = mask_options({})
    assert o["threshold"] == 0.25 and o["dilate"] == 8 and o["replace"] is False and o["sequence"] == "auto"
    o = mask_options({"maskThreshold": 0.1, "dilatePixels": 0, "replaceMasks": True, "maskSequence": "unordered"})
    assert (o["threshold"], o["dilate"], o["replace"], o["sequence"]) == (0.1, 0, True, "unordered")
    for bad in ({"maskThreshold": 2}, {"dilatePixels": -1}, {"maskSequence": "random"}, {"maskThreshold": "x"}):
        with pytest.raises(RequestError):
            mask_options(bad)


# ---------------------------------------------------------------- merge semantics


def test_merge_keep_and_save_mask(tmp_path):
    a = np.ones((4, 6), bool)
    a[0, 0] = False
    save_mask(tmp_path, "x/img.jpg", a, invert=False)
    b = np.ones((4, 6), bool)
    b[3, 5] = False
    save_mask(tmp_path, "x/img.jpg", b, invert=False, merge=True)
    m = np.asarray(Image.open(tmp_path / "x" / "img.png"))
    assert m.dtype == np.uint8 and m[0, 0] == 0 and m[3, 5] == 0 and m[1, 1] == 255
    save_mask(tmp_path, "x/img.jpg", b, invert=False, merge=False)  # replace
    m = np.asarray(Image.open(tmp_path / "x" / "img.png"))
    assert m[0, 0] == 255 and m[3, 5] == 0
    assert not list((tmp_path / "x").glob(".*tmp"))  # atomic write leaves no temp file
    # different-size existing mask is resized (nearest)
    small = np.full((2, 3), 255, np.uint8)
    small[0, 0] = 0
    r = merge_keep(np.ones((4, 6), bool), small)
    assert r[:2, :2].sum() == 0 and r[2:, 2:].all()


def test_masks_task_merges_with_existing(tmp_path, synth_small, monkeypatch):
    from splatprep.runner import run_request

    monkeypatch.setenv("SPLATPREP_SKY_HEURISTIC", "1")
    shutil.copytree(synth_small / "images", tmp_path / "images")
    names = sorted(p.name for p in (tmp_path / "images").iterdir())
    w, h = Image.open(tmp_path / "images" / names[0]).size
    pre = np.full((h, w), 255, np.uint8)
    pre[-10:, :10] = 0  # a "person" removed earlier, on the (kept) ground
    (tmp_path / "out" / "masks").mkdir(parents=True)
    Image.fromarray(pre).save(tmp_path / "out" / "masks" / (Path(names[0]).stem + ".png"))
    req = {"task": "masks", "maskModel": "sky", "imagesDir": "images", "outputDir": "out"}
    st = run_request(req, tmp_path, None)
    assert st["merged"] is True
    m = np.asarray(Image.open(tmp_path / "out" / "masks" / (Path(names[0]).stem + ".png")))
    assert m[-10:, :10].max() == 0 and m[-20:-10, 20:].min() == 255
    st = run_request(dict(req, replaceMasks=True), tmp_path, None)
    assert st["merged"] is False
    m = np.asarray(Image.open(tmp_path / "out" / "masks" / (Path(names[0]).stem + ".png")))
    assert m[-10:, :10].min() == 255


# ---------------------------------------------------------------- chunking + tracking (fake SAM 2)


@pytest.mark.parametrize("n,chunk,overlap", [(1, 8, 2), (8, 8, 2), (9, 8, 2), (20, 6, 2), (100, 16, 3), (501, 64, 2), (30, 5, 4)])
def test_plan_chunks(n, chunk, overlap):
    cs = plan_chunks(n, chunk, overlap)
    assert cs[0][0] == 0 and cs[-1][1] == n
    covered = set()
    for s, e in cs:
        assert 0 < e - s <= int(1.5 * chunk)
        covered |= set(range(s, e))
    assert covered == set(range(n))
    for (s0, e0), (s1, e1) in zip(cs, cs[1:]):
        assert s1 == e0 - min(overlap, chunk - 1) and e1 > e0


class FakeVideoPredictor:
    """Stand-in for SAM2VideoPredictor: a prompted object 'is' its box/points/mask region and
    stays put; propagation copies the last conditioning mask forward (or backward)."""

    def __init__(self):
        self.inits = []

    def init_state(self, video_path, **kw):
        files = sorted(Path(video_path).glob("*.jpg"), key=lambda p: int(p.stem))
        w, h = Image.open(files[0]).size
        self.inits.append((len(files), kw))
        return {"n": len(files), "hw": (h, w), "cond": {}}

    def _region(self, st, points=None, box=None):
        h, w = st["hw"]
        m = np.zeros((h, w), bool)
        if box is not None:
            x0, y0, x1, y1 = np.asarray(box).reshape(4).astype(int)
            m[y0:y1, x0:x1] = True
        if points is not None:
            for x, y in np.asarray(points).reshape(-1, 2).astype(int):
                m[max(0, y - 2):y + 3, max(0, x - 2):x + 3] = True
        return m

    def add_new_points_or_box(self, inference_state, frame_idx, obj_id, points=None, labels=None, box=None):
        inference_state["cond"].setdefault(obj_id, {})[frame_idx] = self._region(inference_state, points, box)

    def add_new_mask(self, inference_state, frame_idx, obj_id, mask):
        assert mask.shape == inference_state["hw"]
        inference_state["cond"].setdefault(obj_id, {})[frame_idx] = np.asarray(mask, bool)

    def propagate_in_video(self, inference_state, start_frame_idx=None, reverse=False):
        st = inference_state
        ids = sorted(st["cond"])
        order = range(start_frame_idx, -1, -1) if reverse else range(start_frame_idx, st["n"])
        last = {}
        for f in order:
            out = []
            for o in ids:
                if f in st["cond"][o]:
                    last[o] = st["cond"][o][f]
                out.append(last.get(o, np.zeros(st["hw"], bool)))
            yield f, ids, np.stack(out)[:, None].astype(np.float32) * 2 - 1

    def reset_state(self, st):
        st["cond"].clear()


def _write_seq(d: Path, n: int, w=64, h=48):
    d.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n):
        p = d / f"frame_{i:04d}.jpg"
        Image.fromarray(np.full((h, w, 3), (i * 7) % 255, np.uint8)).save(p)
        paths.append(p)
    return paths


@pytest.mark.parametrize("bidirectional", [True, False])
def test_track_sequence_chunked_bidirectional(tmp_path, bidirectional):
    paths = _write_seq(tmp_path / "img", 20)
    store = FrameStore(paths, stage_max=32)  # 64x48 → 32x24 reference; prompts scale by 0.5
    try:
        assert store.ref_size == (24, 32)
        pred = FakeVideoPredictor()
        prompts = [FramePrompt(7, 1, [], [], [[10, 10, 30, 30]], "user"),   # → ref box 5..15
                   FramePrompt(15, 2, [[50, 40]], [1], [], "user")]          # → ref point (25, 20)
        res = track_sequence(pred, store, prompts, chunk=6, overlap=2, bidirectional=bidirectional,
                             init_kwargs={"offload_video_to_cpu": True})
        assert all(n <= 6 for n, _ in pred.inits) and pred.inits[0][1] == {"offload_video_to_cpu": True}
        expect1 = set(range(20)) if bidirectional else set(range(7, 20))
        expect2 = set(range(20)) if bidirectional else set(range(15, 20))
        assert set(res.frames_per_object[1]) == expect1
        assert set(res.frames_per_object[2]) == expect2
        assert res.sweeps == (2 if bidirectional else 1)
        m = res.masks.get(0 if bidirectional else 19)
        assert m[10, 10] and not m[0, 0]
        full = store.to_original(10, res.masks.get(10))
        assert full.shape == (48, 64) and full[20, 20] and not full[0, 63]
    finally:
        store.cleanup()
    assert not store.root.exists()


def test_track_sequence_no_prompts_and_packed_masks(tmp_path):
    paths = _write_seq(tmp_path / "img", 5)
    store = FrameStore(paths)
    pred = FakeVideoPredictor()
    res = track_sequence(pred, store, [], chunk=4, overlap=1)
    assert pred.inits == [] and res.masks.nonempty_frames() == []
    store.cleanup()
    pm = PackedMasks(3, (5, 7))
    m = np.zeros((5, 7), bool)
    m[1, 2] = True
    pm.union(1, m)
    pm.union(1, np.eye(5, 7, dtype=bool))
    assert pm.get(1)[1, 2] and pm.get(1)[3, 3] and not pm.get(0).any() and pm.nonempty_frames() == [1]


# ---------------------------------------------------------------- detection post-processing


def test_filter_and_link_detections():
    d = [Detection([10, 10, 50, 50], 0.9, "cat"), Detection([12, 11, 49, 52], 0.5, "dog"),  # NMS
         Detection([20, 20, 30, 30], 0.4, "bag"),                                           # contained
         Detection([-5, 0, 3, 40], 0.3, "person"),                                          # clipped → too thin
         Detection([0, 0, 100, 100], 0.8, "car"),                                           # full frame
         Detection([60, 60, 90, 95], 0.3, "person")]
    k = filter_detections(d, 100, 100)
    assert [x.label for x in k] == ["cat", "person"]


def test_union_object_reprompts_on_new_detections(tmp_path):
    """Detections are tracked as ONE SAM 2 object; a detection not covered by the tracked mask
    (object B appearing on frame 7) stops propagation and re-prompts with tracked ∪ detection."""
    paths = _write_seq(tmp_path / "img", 12)
    store = FrameStore(paths)  # 64x48, reference = original
    try:
        a = np.zeros((48, 64), bool)
        a[5:15, 5:15] = True
        b = np.zeros((48, 64), bool)
        b[30:40, 40:60] = True
        det = {2: [a], 3: [a], 7: [a, b], 8: [b]}
        assert needs_correction(a, [a, b], 0.5) and not needs_correction(a | b, [a, b], 0.5)
        pred = FakeVideoPredictor()
        res = track_sequence(pred, store, [], chunk=5, overlap=2, det_masks=det)
        assert res.corrections >= 1 and res.sweeps == 2
        for f in range(12):
            m = res.masks.get(f)
            assert m[10, 10], f                      # A: detected from frame 2, tracked both ways
            assert m[35, 50], f                      # B: appears on 7; forward re-prompt + backward sweep
        assert set(res.frames_per_object[0]) == set(range(12))
        res1 = track_sequence(FakeVideoPredictor(), store, [], chunk=5, overlap=2, bidirectional=False, det_masks=det)
        assert not res1.masks.get(1).any() and res1.masks.get(5)[10, 10] and not res1.masks.get(5)[35, 50]
        assert res1.masks.get(11)[35, 50] and res1.masks.get(11)[10, 10]
    finally:
        store.cleanup()


# ---------------------------------------------------------------- dynamic pipeline end to end (faked models)


@pytest.fixture
def fake_models(monkeypatch):
    import splatprep.masks as masks_pkg
    from splatprep.masks import dynamic

    status = masks_pkg.mask_model_status()
    for k in ("dynamic", "sam2_track", "sam2_auto"):
        status[k] = {"available": True, "sfm": False, "reason": ""}
    monkeypatch.setattr(masks_pkg, "mask_model_status", lambda: status)
    calls = {"detect": 0, "sam": []}

    def fake_detect(paths, device, classes, threshold, stride, progress, batch=2):
        calls["detect"] += 1
        out = {}
        for i in range(len(paths)):
            # a 'cat' seen on frames 2..4 only (detector misses elsewhere)
            out[i] = [Detection([8, 8, 24, 24], 0.6, "cat")] if (2 <= i <= 4 and "cat" in classes) else []
            progress((i + 1) / len(paths), "det")
        return out, "fake-owl", 0.0

    def fake_load(kind, device):
        calls["sam"].append(kind)
        return FakeVideoPredictor() if kind == "video" else FakeImagePredictor()

    monkeypatch.setattr(dynamic, "detect_all", fake_detect)
    monkeypatch.setattr(dynamic, "load_sam2", fake_load)
    monkeypatch.setattr(dynamic, "image_predictor_for", lambda video_pred: FakeImagePredictor())
    return calls


def _mask(tmp_path, name):
    return np.asarray(Image.open(tmp_path / "out" / "masks" / (Path(name).stem + ".png")))


def test_dynamic_task_video(tmp_path, fake_models):
    from splatprep.runner import run_request

    _write_seq(tmp_path / "images", 10)
    events = []
    st = run_request({"task": "masks", "maskModel": "dynamic", "imagesDir": "images", "outputDir": "out",
                      "maskChunkSize": 4, "dilatePixels": 2}, tmp_path, events.append)
    assert st["status"] == "ok" and st["sequence"] == "video" and st["numDetections"] == 3 and st["sweeps"] == 2
    assert st["framesMasked"] == 10  # tracked forward + backward from the detections
    for i in range(10):
        m = _mask(tmp_path, f"frame_{i:04d}.jpg")
        assert m.shape == (48, 64) and m[16, 16] == 0 and m[7, 7] == 0 and m[40, 60] == 255  # dilated by 2
        assert m[3, 3] == 255
    det = json.loads((tmp_path / "out" / "dynamic_detections.json").read_text())
    assert set(det["frames"]) == {"frame_0002.jpg", "frame_0003.jpg", "frame_0004.jpg"}
    prog = [e["progress"] for e in events if "progress" in e]
    assert prog == sorted(prog) and prog[-1] == 1.0


def test_dynamic_no_detections_skips_sam(tmp_path, fake_models):
    from splatprep.runner import run_request

    _write_seq(tmp_path / "images", 4)
    st = run_request({"task": "masks", "maskModel": "dynamic", "imagesDir": "images", "outputDir": "out",
                      "maskClasses": ["unicorn"]}, tmp_path, None)
    assert st["numDetections"] == 0 and st["framesMasked"] == 0 and fake_models["sam"] == []
    assert all(_mask(tmp_path, f"frame_{i:04d}.jpg").min() == 255 for i in range(4))


def test_dynamic_unordered_uses_image_predictor(tmp_path, fake_models):
    from splatprep.runner import run_request

    _write_seq(tmp_path / "images", 6)
    st = run_request({"task": "masks", "maskModel": "dynamic", "imagesDir": "images", "outputDir": "out",
                      "maskSequence": "unordered", "dilatePixels": 0}, tmp_path, None)
    assert st["sequence"] == "unordered" and fake_models["sam"] == ["image"] and st["framesMasked"] == 3
    assert _mask(tmp_path, "frame_0003.jpg")[16, 16] == 0 and _mask(tmp_path, "frame_0000.jpg").min() == 255


def test_sam2_track_prompts_any_frame_and_invert(tmp_path, fake_models):
    from splatprep.runner import run_request

    _write_seq(tmp_path / "images", 8)
    req = {"task": "masks", "imagesDir": "images", "outputDir": "out", "dilatePixels": 0, "maskChunkSize": 4,
           "maskPrompts": [{"image": "frame_0005.jpg", "boxes": [[40, 20, 60, 40]], "objectId": 4}]}
    st = run_request(req, tmp_path, None)  # no maskModel + maskPrompts → sam2_track
    assert st["maskModel"] == "sam2_track" and st["framesMasked"] == 8
    m = _mask(tmp_path, "frame_0000.jpg")
    assert m[30, 50] == 0 and m[5, 5] == 255
    st = run_request(dict(req, invertMask=True, replaceMasks=True), tmp_path, None)
    m = _mask(tmp_path, "frame_0007.jpg")
    assert m[30, 50] == 255 and m[5, 5] == 0
    with pytest.raises(RequestError):
        run_request({"task": "masks", "maskModel": "sam2_track", "imagesDir": "images", "outputDir": "o2"}, tmp_path, None)


def test_sam2_auto_legacy_prompt_keeps_object(tmp_path, fake_models):
    from splatprep.runner import run_request

    _write_seq(tmp_path / "images", 5)
    st = run_request({"task": "masks", "maskModel": "sam2_auto", "imagesDir": "images", "outputDir": "out",
                      "maskPrompt": {"boxes": [[10, 10, 30, 30]]}}, tmp_path, None)
    assert st["dilatePixels"] == 0 and st["autoPrompt"] is False
    m = _mask(tmp_path, "frame_0004.jpg")
    assert m[20, 20] == 255 and m[40, 60] == 0  # object-centric: object kept


# ---------------------------------------------------------------- maskpreview + maskserver protocol


class FakeImagePredictor:
    def __init__(self):
        self.sets = 0
        self._features = self._orig_hw = None
        self._is_image_set = self._is_batch = False

    def set_image(self, rgb):
        self.sets += 1
        self._orig_hw = [rgb.shape[:2]]
        self._features = {"id": self.sets}
        self._is_image_set = True

    def _region(self, box=None, pts=None):
        h, w = self._orig_hw[0]
        m = np.zeros((h, w), np.float32)
        if box is not None:
            x0, y0, x1, y1 = np.asarray(box).reshape(4).astype(int)
            m[y0:y1, x0:x1] = 1
        if pts is not None:
            for x, y in np.asarray(pts).reshape(-1, 2).astype(int):
                m[max(0, y - 2):y + 3, max(0, x - 2):x + 3] = 1
        return m

    def predict(self, point_coords=None, point_labels=None, box=None, multimask_output=False, **kw):
        assert self._is_image_set
        if box is not None and np.asarray(box).ndim == 2:  # batched boxes
            ms = np.stack([self._region(b)[None] for b in box])
            return ms, np.full((len(box), 1), 0.9), None
        m = self._region(box, point_coords)
        if multimask_output:
            return np.stack([m * 0, m, m]), np.array([0.1, 0.95, 0.5]), None
        return m[None], np.array([0.9]), None


def test_maskpreview_task(tmp_path, monkeypatch):
    from splatprep.masks import server
    from splatprep.runner import run_request

    _write_seq(tmp_path / "images", 3)
    fake = FakeImagePredictor()
    orig = server.PreviewEngine
    monkeypatch.setattr(server, "PreviewEngine", lambda device: orig(device, predictor=fake))
    st = run_request({"task": "maskpreview", "imagesDir": "images", "outputDir": "out",
                      "maskPrompts": [{"image": "frame_0002.jpg", "points": [[20, 20, 1]], "boxes": [[30, 30, 40, 40], [0, 0, 5, 5]]}]},
                     tmp_path, None)
    assert st["task"] == "maskpreview" and st["image"] == "frame_0002.jpg"
    m = np.asarray(Image.open(tmp_path / "out" / "preview.png"))
    assert m[20, 20] == 0 and m[35, 35] == 0 and m[2, 2] == 0 and m[45, 60] == 255
    assert st["bbox"] == [0, 0, 40, 40] and fake.sets == 1


def _engine(fake):
    from splatprep.masks.server import PreviewEngine

    return PreviewEngine("cpu", cache_size=2, predictor=fake)


def test_maskserver_protocol(tmp_path):
    from splatprep.masks.server import MaskServer

    _write_seq(tmp_path / "images", 4)
    fake = FakeImagePredictor()
    srv = MaskServer(_engine(fake), tmp_path / "images", tmp_path / "prev")
    lines = [
        json.dumps({"cmd": "ping", "id": 0}),
        "not json",
        json.dumps({"id": 1, "image": "frame_0001.jpg", "points": [[10, 10, 1]]}),
        json.dumps({"id": 2, "image": "frame_0001.jpg", "points": [[10, 10, 1]], "boxes": [[20, 20, 30, 30]], "objectMask": True,
                    "out": str(tmp_path / "custom.png")}),
        json.dumps({"id": 3, "image": str(tmp_path / "images" / "frame_0002.jpg"), "boxes": [[1, 1, 2]]}),
        json.dumps({"id": 4, "image": "missing.jpg", "points": [[1, 1, 1]]}),
        json.dumps({"id": 5, "cmd": "warm", "image": "frame_0003.jpg"}),
        json.dumps({"id": 6, "image": "frame_0002.jpg"}),
        json.dumps({"image": "frame_0001.jpg", "points": [[10, 10, 1]], "objectMask": False, "out": str(tmp_path / "keep.png")}),
        json.dumps({"id": 7, "cmd": "quit"}),
        json.dumps({"id": 8, "cmd": "ping"}),  # never reached
    ]
    out = []
    assert srv.serve(io.StringIO("\n".join(lines) + "\n"), out.append) == 0
    assert out[0] == {"id": 0, "pong": True}
    assert "bad JSON" in out[1]["error"]
    r1 = out[2]
    assert r1["id"] == 1 and r1["embeddingCached"] is False and r1["width"] == 64 and r1["height"] == 48
    m1 = np.asarray(Image.open(r1["mask"]))
    assert Path(r1["mask"]).name == "preview_frame_0001.png" and m1[10, 10] == 255 and m1[40, 40] == 0  # 255 = OBJECT
    assert abs(r1["score"] - 0.95) < 1e-6  # single click → best of multimask
    r2 = out[3]
    assert r2["embeddingCached"] is True and r2["mask"] == str((tmp_path / "custom.png").resolve())
    m2 = np.asarray(Image.open(r2["mask"]))
    assert m2[25, 25] == 255 and m2[10, 10] == 255 and m2[40, 5] == 0  # objectMask: 255 = object
    assert out[4]["id"] == 3 and "boxes" in out[4]["error"]
    assert out[5]["id"] == 4 and "not found" in out[5]["error"]
    assert out[6]["id"] == 5 and out[6]["warmed"].endswith("frame_0003.jpg")
    assert out[7]["id"] == 6 and "points" in out[7]["error"]
    assert "id" not in out[8] and out[8]["embeddingCached"] is False  # LRU of 2 evicted frame 1
    mk = np.asarray(Image.open(out[8]["mask"]))
    assert mk[10, 10] == 0 and mk[40, 40] == 255  # objectMask false → keep convention
    assert out[9] == {"bye": True} and len(out) == 10
    assert fake.sets == 4  # frames 1, 2, 3, then 1 again after LRU eviction


def test_maskserver_cli_registered():
    from splatprep.cli import build_parser

    a = build_parser().parse_args(["maskserver", "--images", "x", "--device", "cpu"])
    assert a.images == "x" and a.device == "cpu" and a.func.__name__ == "cmd_maskserver"


def test_model_status_lists_new_models():
    from splatprep.models import MASK_MODEL_IDS, model_cards, model_status

    st = model_status()
    cards = model_cards()["models"]
    for mid in ("dynamic", "sam2_track"):
        assert mid in MASK_MODEL_IDS and st[mid]["kind"] == "mask" and mid in cards
    assert "Apache-2.0" in cards["dynamic"]["licence"]["weights"]


def test_maskserver_main_stdout_only_replies(tmp_path, monkeypatch):
    from splatprep.masks import server

    _write_seq(tmp_path / "images", 2)
    fake = FakeImagePredictor()
    orig = server.PreviewEngine
    monkeypatch.setattr(server, "PreviewEngine", lambda device: orig(device, predictor=fake))
    img = str(tmp_path / "images" / "frame_0000.jpg")
    out = []
    rc = server.serve_main(None, str(tmp_path / "o"), "cpu", out.append,
                           io.StringIO(json.dumps({"image": img, "points": [[5, 5, 1]]}) + "\n"))
    assert rc == 0 and len(out) == 1 and set(out[0]) >= {"mask", "score"} and "ready" not in out[0]
    out = []
    server.serve_main(None, str(tmp_path / "o"), "cpu", out.append, io.StringIO(""), ready_line=True)
    assert out[0]["ready"] is True

    def boom(device):
        raise RuntimeError("no weights")

    monkeypatch.setattr(server, "PreviewEngine", boom)
    out = []
    server.serve_main(None, str(tmp_path / "o"), "cpu", out.append,
                      io.StringIO(json.dumps({"image": img, "points": [[5, 5, 1]]}) + "\n"))
    assert len(out) == 1 and "no weights" in out[0]["error"]


def test_sam2_auto_with_mask_prompts_removes_objects(tmp_path, fake_models):
    """Mac mask editor contract: sam2_auto + maskPrompts = prompted tracking, objects → 0,
    written to <workdir>/masks with outputDir "." and AND-merged with existing masks."""
    from splatprep.runner import run_request

    _write_seq(tmp_path / "images", 6)
    (tmp_path / "masks").mkdir()
    pre = np.full((48, 64), 255, np.uint8)
    pre[0:4, 0:4] = 0
    Image.fromarray(pre).save(tmp_path / "masks" / "frame_0000.png")
    st = run_request({"task": "masks", "imagesDir": "images", "outputDir": ".", "maskModel": "sam2_auto",
                      "maskPrompts": [{"image": "frame_0003.jpg", "points": [[40, 30, 1]], "objectId": 1}]},
                     tmp_path, None)
    assert st["objectsRemoved"] is True and st["framesMasked"] == 6 and st["dilatePixels"] == 8
    m = np.asarray(Image.open(tmp_path / "masks" / "frame_0000.png"))
    assert m[30, 40] == 0 and m[2, 2] == 0 and m[45, 5] == 255


def test_backward_sweep_when_first_detection_on_frame_0(tmp_path):
    paths = _write_seq(tmp_path / "img", 10)
    store = FrameStore(paths)
    try:
        a = np.zeros((48, 64), bool)
        a[5:15, 5:15] = True
        b = np.zeros((48, 64), bool)
        b[30:40, 40:60] = True
        res = track_sequence(FakeVideoPredictor(), store, [], chunk=4, overlap=1, det_masks={0: [a], 7: [b]})
        assert res.sweeps == 2 and all(res.masks.get(f)[35, 50] and res.masks.get(f)[10, 10] for f in range(10))
    finally:
        store.cleanup()


def test_precision_option():
    from splatprep.masks.track import set_precision

    assert mask_options({})["precision"] == "auto"
    with pytest.raises(RequestError):
        mask_options({"maskPrecision": "int8"})
    assert set_precision("cpu", "auto") == "fp32"
