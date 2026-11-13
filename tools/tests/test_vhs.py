"""Old home video restoration (vhs.py): tape artefacts on synthetic frames, clip / frame selection, and an end-to-end
run on a synthetic "tape" video (needs PyAV)."""
import json

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from splatprep import vhs  # noqa: E402
from splatprep.frames import square_pixels  # noqa: E402


def texture(h, w, seed=0):
    """Smooth random texture with edges at many scales (stands in for a scene)."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w), np.float32)
    for s in (4, 16, 64):
        n = rng.random((h // s + 2, w // s + 2)).astype(np.float32)
        img += cv2.resize(n, (w + 2 * s, h + 2 * s), interpolation=cv2.INTER_CUBIC)[s: s + h, s: s + w] * (s ** 0.5)
    img = (img - img.min()) / (img.max() - img.min()) * 200 + 25
    return img.astype(np.uint8)


def rgb(g):
    return np.repeat(g[..., None], 3, 2)


# ---------------------------------------------------------------------------
# per-frame measurements
# ---------------------------------------------------------------------------

def test_comb_score_separates_progressive_from_combed():
    a = texture(240, 320, 1)
    b = np.roll(a, 9, axis=1)                    # the other field, 9 px later
    combed = a.copy()
    combed[1::2] = b[1::2]
    assert vhs.comb_score(cv2.GaussianBlur(a, (3, 3), 0)) < vhs.COMB_RATIO
    assert vhs.comb_score(combed) > vhs.COMB_RATIO


def test_deinterlace_field_rebuilds_from_one_field():
    a = cv2.GaussianBlur(texture(240, 320, 2), (5, 5), 0)
    combed = a.copy()
    combed[1::2] = np.roll(a, 12, axis=1)[1::2]
    fixed = vhs.deinterlace_field(rgb(combed))[..., 0]
    assert np.abs(fixed.astype(int) - a).mean() < 0.4 * np.abs(combed.astype(int) - a).mean()
    assert vhs.comb_score(fixed) < vhs.COMB_RATIO


def test_dropout_lines_found_and_repaired_but_edges_left_alone():
    clean = cv2.GaussianBlur(texture(400, 600, 3), (5, 5), 0)
    bad = clean.copy()
    bad[201:203, 100:420] = 250                  # a bright tracking streak, two lines high
    m = vhs.dropout_mask(bad)
    assert m[201:203, 120:400].mean() > 200
    assert m[:150].max() == 0 and m[260:].max() == 0
    fixed, n = vhs.repair_dropouts(rgb(bad))
    assert n > 0
    err_bad = np.abs(bad[195:210, 100:420].astype(int) - clean[195:210, 100:420]).mean()
    err_fix = np.abs(fixed[195:210, 100:420, 0].astype(int) - clean[195:210, 100:420]).mean()
    assert err_fix < 0.3 * err_bad
    # a horizon (one-sided step) is not a dropout
    horizon = np.full((400, 600), 40, np.uint8)
    horizon[:200] = 220
    assert vhs.dropout_mask(horizon).max() == 0
    assert vhs.dropout_score(horizon) == 0.0
    assert vhs.dropout_score(bad) > vhs.DROPOUT_LEVEL


def test_detect_borders_finds_black_sides_and_head_switching_band():
    h, w = 576, 720
    col = np.full(w, 110.0)
    col[:12] = 2.0
    col[-7:] = 3.0
    row = np.full(h, 105.0) + np.sin(np.arange(h) / 9.0)   # a bit of picture structure
    row[-14:-4] = 60.0                                     # the band smear
    row[-4:] = 5.0                                         # and black under it
    unstable = np.full(h, 4.0)
    unstable[-14:] = 30.0
    b = vhs.detect_borders(col, row, unstable)
    assert 12 <= b["left"] <= 16 and 7 <= b["right"] <= 11
    assert 14 <= b["bottom"] <= 20
    assert b["top"] == 0


def test_cut_needs_a_jump_well_above_the_usual_change():
    assert vhs.is_cut(40.0, [3.0, 4.0, 5.0])
    assert not vhs.is_cut(10.0, [3.0, 4.0])        # under CUT_MIN
    assert not vhs.is_cut(30.0, [12.0, 14.0, 13.0])  # fast pan: big changes all along


def test_colour_transform_takes_out_a_cast_gently():
    means = np.tile([[120.0, 100.0, 80.0]], (30, 1))   # warm cast
    ct = vhs.colour_transform(means, np.full(30, 20.0), np.full(30, 200.0))
    g = ct["gains"]
    assert g[0] < 1.0 < g[2]
    out = vhs.apply_colour(np.full((4, 4, 3), [120, 100, 80], np.uint8), ct).reshape(-1, 3).mean(0)
    assert np.ptp(out) < 0.3 * 40
    assert 1.0 <= ct["scale"] <= 1.15


def test_overlay_mask_finds_a_burnt_in_stamp_only_when_the_picture_moves():
    h, w = 120, 160
    frames = []
    for k in range(40):
        f = np.roll(texture(h, w + 80, 4), -2 * k, axis=1)[:, :w].copy()
        cv2.putText(f, "12:34", (95, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 255, 1)
        frames.append(f.astype(np.float32))
    st = np.stack(frames)
    grad = np.mean([np.hypot(cv2.Sobel(f, cv2.CV_32F, 1, 0), cv2.Sobel(f, cv2.CV_32F, 0, 1)) for f in frames], axis=0)
    m = vhs.overlay_mask(st.mean(0), st.std(0), grad)
    assert m is not None
    assert m[100:108, 100:130].min() == 0          # the stamp is masked
    assert m[10:60, 10:80].min() == 255            # the scene is not
    still = np.stack([frames[0]] * 40)
    assert vhs.overlay_mask(still.mean(0), still.std(0), grad) is None


def test_temporal_denoise_lowers_noise_without_ghosting():
    rng = np.random.default_rng(5)
    base = cv2.GaussianBlur(texture(240, 320, 6), (3, 3), 0).astype(np.float32)
    shots = []
    for dx in (-6, -3, 0, 3, 6):
        M = np.float32([[1, 0, dx], [0, 1, 0]])
        s = cv2.warpAffine(base, M, (320, 240), borderMode=cv2.BORDER_REFLECT)
        shots.append(np.clip(rgb(s) + rng.normal(0, 8, (240, 320, 3)), 0, 255).astype(np.uint8))
    center = shots[2]
    out, used = vhs.temporal_denoise(center, shots[:2] + shots[3:])
    assert used == 4
    inner = (slice(20, -20), slice(20, -20))
    err_in = np.abs(center[inner][..., 0].astype(float) - base[inner]).mean()
    err_out = np.abs(out[inner][..., 0].astype(float) - base[inner]).mean()
    assert err_out < 0.75 * err_in


def test_square_pixels_keeps_lines_and_fixes_aspect():
    img = np.zeros((576, 720, 3), np.uint8)
    assert square_pixels(img, 16 / 15).shape == (576, 768, 3)
    assert square_pixels(np.zeros((1068, 1890, 3), np.uint8), 712 / 945).shape == (1068, 1424, 3)


# ---------------------------------------------------------------------------
# frame selection and cameras
# ---------------------------------------------------------------------------

def make_frames(n, motion=0.01, fps=20.0):
    out = []
    for i in range(n):
        f = vhs._Frame(index=i, t=i / fps)
        f.motion = motion
        f.sharp = 10.0 + (i % 5)          # every 5th frame sharpest
        out.append(f)
    return out


def test_select_frames_moves_on_and_prefers_sharp_frames():
    fr = make_frames(200)
    for i in range(0, 200, 7):
        fr[i].unique = False               # repeats are never picked
    fr[50].nosignal = True
    picks = vhs.select_frames(fr, step=0.05)
    assert 25 <= len(picks) <= 45
    assert all(fr[p].unique and not fr[p].nosignal for p in picks)
    assert np.mean([fr[p].sharp for p in picks]) > 13.0
    assert all(fr[b].t - fr[a].t >= 0.12 for a, b in zip(picks, picks[1:]))


def test_select_frames_skips_a_camera_that_stands_still():
    fr = make_frames(200, motion=0.0)
    assert len(vhs.select_frames(fr, step=0.05)) <= 1


def test_plan_selection_caps_the_count():
    fr = make_frames(600, motion=0.02)
    assert len(vhs.plan_selection(fr, 40)) <= 40
    assert len(vhs.plan_selection(fr, 40)) >= 30


def test_camera_groups_split_at_zooms():
    fr = make_frames(100)
    picks = list(range(0, 100, 10))
    assert len(set(vhs.camera_groups(fr, picks, clip=3))) == 1
    for i in range(40, 60):
        fr[i].zoom = 0.06                  # window zoom: 20 frames × 0.06 / WINDOW = 0.12 > 0.08
    g = vhs.camera_groups(fr, picks, clip=3)
    assert len(set(g)) == 2 and all(x // 1000 == 3 for x in g)


# ---------------------------------------------------------------------------
# end to end on a synthetic tape (PyAV)
# ---------------------------------------------------------------------------

def write_tape(path, seconds_a=6.0, seconds_b=5.0, blank=1.0, fps=25, repeat=1):
    """720×576 at 16:15 with black side borders, a noisy head-switching band, a date stamp in clip A, a cut, a
    blue screen and clip B; every unique frame stored ``repeat`` times."""
    av = pytest.importorskip("av")
    W, H = 720, 576
    c = av.open(str(path), "w")
    s = c.add_stream("libx264", rate=fps * repeat)
    s.width, s.height, s.pix_fmt = W, H, "yuv420p"
    s.codec_context.sample_aspect_ratio = __import__("fractions").Fraction(16, 15)
    s.options = {"crf": "18", "preset": "ultrafast"}
    big_a = cv2.cvtColor(texture(H, W * 3, 7), cv2.COLOR_GRAY2RGB)
    big_b = cv2.cvtColor(texture(H, W * 3, 8), cv2.COLOR_GRAY2RGB)
    big_b[..., 2] = np.clip(big_b[..., 2].astype(int) + 30, 0, 255)       # clip B: a blue cast
    rng = np.random.default_rng(1)

    def frame(img):
        f = img.copy()
        f[:, :14] = 0
        f[:, -8:] = 0
        f[-16:] = np.clip(rng.normal(60, 50, (16, W, 3)), 0, 255).astype(np.uint8)
        return f

    def emit(img):
        for _ in range(repeat):
            for p in s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                c.mux(p)

    n_a = int(seconds_a * fps)
    for k in range(n_a):
        x = int(k * (W * 1.6) / n_a)
        img = frame(np.ascontiguousarray(big_a[:, x: x + W]))
        cv2.putText(img, "JUL 14 1989", (470, 520), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        emit(img)
    for _ in range(int(blank * fps)):
        emit(np.tile(np.uint8([16, 16, 200]), (H, W, 1)))
    n_b = int(seconds_b * fps)
    for k in range(n_b):
        x = int(k * (W * 1.6) / n_b)
        emit(frame(np.ascontiguousarray(big_b[:, x: x + W])))
    for p in s.encode():
        c.mux(p)
    c.close()


def test_prepare_end_to_end_on_a_synthetic_tape(tmp_path):
    pytest.importorskip("av")
    video = tmp_path / "tape.mp4"
    write_tape(video, repeat=2)
    out = tmp_path / "out"
    info = vhs.prepare(video, out, {}, {"minClipSeconds": 3.0, "clips": None, "totalFrames": 60},
                       lambda f, m: None)
    summary = json.loads((out / "vhs.json").read_text())
    assert abs(summary["pixelAspect"] - 16 / 15) < 1e-3
    assert 1.8 < summary["repeatFactor"] < 2.2
    assert summary["crop"]["left"] >= 12 and summary["crop"]["right"] >= 6 and summary["crop"]["bottom"] >= 14
    verdicts = [c["verdict"] for c in summary["clips"]]
    assert "noSignal" in verdicts
    a = summary["clips"][0]
    assert a.get("overlay"), "the date stamp in clip A should be found"
    fs = json.loads((out / "frames.json").read_text())
    frames = fs["frames"]
    assert len(frames) >= 10
    w, h = frames[0]["width"], frames[0]["height"]
    # crop + square pixels: (720 − borders) × 16/15 wide, 576 − band high
    assert 535 <= h <= 562 and abs(w / h - (720 - summary["crop"]["left"] - summary["crop"]["right"]) * 16 / 15 / h) < 0.02
    assert all("clip" in f and "cameraGroup" in f for f in frames)
    img = cv2.imread(str(out / frames[0]["image"]))
    assert img.shape[:2] == (h, w)
    assert img[:, :4].mean() > 20, "the black border should be cropped"
    if any(f["clip"] == 0 for f in frames):
        m = [f for f in frames if f["clip"] == 0 and f.get("mask")]
        assert m, "clip A frames carry the overlay mask"
        mask = cv2.imread(str(out / m[0]["mask"]), 0)
        assert mask.shape == (h, w) and mask.min() == 0 and mask.mean() > 200
    assert info["numFrames"] == len(frames)
    # analysis is reused on a second run
    info2 = vhs.prepare(video, tmp_path / "out", {}, {"analyseOnly": True}, lambda f, m: None)
    assert info2["numFrames"] == 0 and info2["vhs"]["clips"]
