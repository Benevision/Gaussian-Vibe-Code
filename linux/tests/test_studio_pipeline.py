"""End to end: New Project → frames / photos → stacked masks → sky → solve → train, against a real HTTP server
(FakeBackend + a stand-in splatprep), through the Studio's urllib client."""
import json
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("uvicorn")
from PIL import Image  # noqa: E402

from conftest import FakeBackend  # noqa: E402
from splatworker.config import ServerSettings  # noqa: E402
from splatworker.server import create_app  # noqa: E402
from splatworker.studio import formats, importers  # noqa: E402
from splatworker.studio.client import Client  # noqa: E402
from splatworker.studio.pipeline import LocalPipeline, stage_titles  # noqa: E402
from splatworker.studio.presets import ProjectOptions  # noqa: E402
from splatworker.studio.projects import Project, list_projects  # noqa: E402

FAKE_RUNNER = r'''
import json
from pathlib import Path
import numpy as np
from PIL import Image

def run_request(req, workdir, progress):
    w = Path(workdir)
    out = w / req["outputDir"]
    if req["task"] == "frames" and req.get("stereo"):
        (out / "images").mkdir(parents=True)
        rng = np.random.default_rng(0)
        frames = []
        for i in range(4):
            tex = (rng.random((24, 48, 3)) * 255).astype(np.uint8)
            Image.fromarray(tex[:, 8:40]).save(out / "images" / f"{i:06d}_L.png")
            Image.fromarray(tex[:, 12:44]).save(out / "images" / f"{i:06d}_R.png")   # 4 px disparity
            for eye, t in (("left", "L"), ("right", "R")):
                frames.append({"image": f"images/{i:06d}_{t}.png", "width": 32, "height": 24, "group": i,
                               "eye": eye, "focalPx": 30.0, "cx": 16.0, "cy": 12.0})
        rig = {"baseline": req["stereo"].get("baseline", 0.05), "origin": "user", "parallel": True,
               "lens": "pinhole"}
        (out / "frames.json").write_text(json.dumps({"frames": frames, "rig": rig, "isMultiViewRig": False,
                                                     "stereoLayout": "sideBySide", "request": req}))
        return {"stats": {}}
    if req["task"] == "frames":
        (out / "images").mkdir(parents=True)
        for i in range(6):
            Image.fromarray(np.full((24, 32, 3), 40 * i, np.uint8)).save(out / "images" / f"{i:06d}.jpg")
        progress({"stage": "frames", "progress": 1.0, "message": "6 frames"})
        return {"stats": {}}
    images = sorted((w / req["imagesDir"]).iterdir())
    if req["task"] == "maskpreview":
        out.mkdir(parents=True)
        im = Image.open(w / req["imagesDir"] / req["previewImage"])
        m = np.zeros((im.height, im.width), np.uint8)
        x, y, _ = req["maskPrompts"][0]["points"][0]
        m[max(0, int(y) - 3):int(y) + 3, max(0, int(x) - 3):int(x) + 3] = 255
        Image.fromarray(m).save(out / "preview.png")
        return {"stats": {}}
    if req["task"] == "masks":
        (out / "masks").mkdir(parents=True)
        for k, p in enumerate(images):
            m = np.full((24, 32), 255, np.uint8)
            if req["maskModel"] == "sky":
                m[:] = 0
                m[:6] = 255            # top quarter is sky (invertMask: 255 = sky)
            elif k % 2 == 0:
                m[10:14, 10:14] = 0     # a person in every other photo
            Image.fromarray(m).save(out / "masks" / (p.stem + ".png"))
        (out / "request.json").write_text(json.dumps(req))
        return {"stats": {}}
    if req["task"] == "sfm":
        (out / "images").mkdir(parents=True)
        cams = []
        for i, p in enumerate(images):
            (out / "images" / p.name).write_bytes(p.read_bytes())
            c2w = np.eye(4); c2w[0, 3] = i * 0.1
            cams.append({"id": i, "image": "images/" + p.name, "width": 32, "height": 24, "fx": 30.0, "fy": 30.0,
                         "cx": 16.0, "cy": 12.0, "cameraToWorld": c2w.tolist()})
        (out / "dataset.json").write_text(json.dumps({"format": "splatdataset", "version": 1,
            "convention": "opencv", "cameras": cams, "request": req}))
        return {"stats": {}}
    raise ValueError(req["task"])
'''


@pytest.fixture
def server(tmp_path, monkeypatch):
    import uvicorn

    pkg = tmp_path / "pkgs" / "splatprep"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "models.py").write_text("def available_models():\n    return []\n")
    (pkg / "runner.py").write_text(FAKE_RUNNER)
    monkeypatch.syspath_prepend(str(pkg.parent))
    for mod in list(sys.modules):
        if mod == "splatprep" or mod.startswith("splatprep."):
            monkeypatch.delitem(sys.modules, mod)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    settings = ServerSettings(data_dir=tmp_path / "data", token="t", name="w", mdns=False, host="127.0.0.1",
                              port=port, allow_local_datasets=True, local_secret="k" * 32)
    backend = FakeBackend()
    app = create_app(settings, backend=backend, devices=[None])
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    th = threading.Thread(target=srv.run, daemon=True)
    th.start()
    for _ in range(200):
        if srv.started:
            break
        time.sleep(0.02)
    yield Client(f"http://127.0.0.1:{port}", "t", local_secret="k" * 32), backend, settings
    srv.should_exit = True
    th.join(5)


def run(client, project, opts, train=True):
    events = []
    p = LocalPipeline(client, project, opts, lambda k, d: events.append((k, d)), train=train)
    p.poll_seconds = 0.02
    p.preview_seconds = 0.0
    out = p.run()
    return out, events


def test_video_pipeline_with_stacked_masks_sky_and_training(server, tmp_path):
    client, backend, settings = server
    video = tmp_path / "in" / "walk.mp4"
    video.parent.mkdir()
    video.write_bytes(b"not really a video")
    opts = ProjectOptions(name="Walk", source_kind="video", source=str(video), mask_models=["person"], sky=True,
                          holdout_every=3, preset="preview")
    proj = Project.create(tmp_path / "projects", "Walk", opts.to_dict())
    out, events = run(client, proj, opts)
    kinds = [k for k, _ in events]
    assert "failed" not in kinds, [d for k, d in events if k in ("failed", "log")]
    assert [d["title"] for k, d in events if k == "stage"] == stage_titles(opts)
    assert out is not None and out.parent == proj.outputs_dir and out.suffix == ".ply"
    assert formats.load(out).count == 10
    doc = json.loads((proj.dataset_dir / "dataset.json").read_text())
    cams = doc["cameras"]
    assert len(cams) == 6
    assert [c.get("holdout") for c in cams] == [False, False, True, False, False, True]
    assert sum(1 for c in cams if "mask" in c) == 3          # every other photo had a person
    assert all(c["sky"].startswith("splatprep-sky/masks/") for c in cams)
    m = np.asarray(Image.open(proj.dataset_dir / cams[0]["mask"]))
    assert m[12, 12] == 0 and m[12, 7] == 0 and m[12, 3] == 255      # 4 px dilation of the ignored area
    sfm_req = doc["request"]
    assert sfm_req["masksDir"] == "masks" and sfm_req["skyMasksDir"] == "splatprep-sky/masks"
    # the trainer got the project folder itself and the Mac's defaults + preset
    train_ctx = [c for c in backend.contexts if c.kind == "train"][-1]
    assert train_ctx.dataset_dir == proj.dataset_dir.resolve()
    assert train_ctx.spec["iterations"] == 5000 and train_ctx.spec["bilateralGrid"] is True
    # nothing left registered on the server
    assert client.get("/v1/datasets") == []
    assert proj.outputs() == [out]
    assert list_projects(tmp_path / "projects")[0].root == proj.root


def test_photos_pipeline_without_training_and_cancel(server, tmp_path):
    client, backend, _ = server
    photos = tmp_path / "photos"
    photos.mkdir()
    rng = np.random.default_rng(0)
    for i in range(10):
        img = (rng.random((60, 80, 3)) * 255).astype(np.uint8)
        if i == 4:
            img[:] = 128            # featureless → rejected as blurry
        Image.fromarray(img).save(photos / f"IMG_{i:04d}.png")
    opts = ProjectOptions(name="P", source_kind="photos", source=str(photos), sky=False, max_image_size=40)
    proj = Project.create(tmp_path / "projects", "P", opts.to_dict())
    out, events = run(client, proj, opts, train=False)
    assert out is None and events[-1][0] == "done", events[-3:]
    imgs = sorted((proj.dataset_dir / "images").iterdir())
    assert len(imgs) == 9 and max(Image.open(imgs[0]).size) == 40
    assert (proj.dataset_dir / "dataset.json").is_file()


def test_training_pause_checkpoint_resume(server, tmp_path):
    """The Projects page's Pause / Save checkpoint / Resume: the pipeline keeps waiting while the job is paused."""
    from conftest import make_dataset, wait_for

    client, backend, _ = server
    backend.steps, backend.step_delay = 60, 0.03
    assert "pause" in client.info()["capabilities"]
    proj = Project.create(tmp_path / "projects", "Room", {})
    make_dataset(proj.dataset_dir)
    opts = ProjectOptions(name="Room", source_kind="dataset", source=str(proj.dataset_dir), preset="preview")
    events = []
    p = LocalPipeline(client, proj, opts, lambda k, d: events.append((k, d)))
    p.poll_seconds = 0.02
    result = {}
    th = threading.Thread(target=lambda: result.update(out=p.train()), daemon=True)
    th.start()
    job = wait_for(lambda: next((j["jobId"] for j in client.jobs() if j["kind"] == "train"
                                 and (j["step"] or 0) >= 3), None))
    out, step = client.checkpoint(job, lambda n: proj.outputs_dir / f"Room-ck-step{n}.ply")
    assert out.is_file() and step >= 3 and out.name == f"Room-ck-step{step}.ply"
    assert formats.load(out).count == 10
    assert client.pause(job)["state"] in ("training", "paused")
    wait_for(lambda: client.job(job)["state"] == "paused")
    time.sleep(0.2)
    assert th.is_alive() and "out" not in result          # paused is not finished
    out2, step2 = client.checkpoint(job, lambda n: proj.outputs_dir / f"Room-ck2-step{n}.ply")
    assert step2 == client.job(job)["step"]
    client.resume(job)
    th.join(10)
    assert result["out"].is_file() and backend.resumed_from[-1] == step2
    assert set(proj.outputs()) == {result["out"], out, out2}


def test_nerfstudio_import(tmp_path):
    src = tmp_path / "ns"
    (src / "images").mkdir(parents=True)
    Image.fromarray(np.zeros((10, 20, 3), np.uint8)).save(src / "images" / "a.png")
    m = np.eye(4)
    m[:3, 3] = [1, 2, 3]
    (src / "transforms.json").write_text(json.dumps({"fl_x": 15, "w": 20, "h": 10,
                                                     "frames": [{"file_path": "images/a.png", "transform_matrix": m.tolist()}]}))
    ds = tmp_path / "ds"
    importers.import_nerfstudio(src / "transforms.json", ds)
    cam = json.loads((ds / "dataset.json").read_text())["cameras"][0]
    c2w = np.asarray(cam["cameraToWorld"])
    np.testing.assert_allclose(c2w[:3, 3], [1, 2, 3])
    np.testing.assert_allclose(c2w[:3, 1], [0, -1, 0])       # OpenGL +Y up → OpenCV +Y down
    from splatworker.dataset import parse_dataset

    assert len(parse_dataset(ds).cameras) == 1


def test_capture_import_priors(tmp_path, monkeypatch):
    b = tmp_path / "K.splatcapture"
    (b / "images").mkdir(parents=True)
    Image.fromarray(np.zeros((4, 4, 3), np.uint8)).save(b / "images" / "000000.jpg")
    c2w = np.eye(4).T.reshape(-1).tolist()
    (b / "manifest.json").write_text(json.dumps({"format": "splatcapture", "version": 1, "coordinateSystem": "arkit",
        "imageWidth": 4, "imageHeight": 4, "frames": [
            {"index": 0, "image": "images/000000.jpg", "intrinsics": [3, 3, 2, 2], "cameraToWorld": c2w},
            {"index": 1, "image": "images/000001.jpg", "intrinsics": [3, 3, 2, 2], "cameraToWorld": c2w,
             "trackingState": "notAvailable"}]}))
    ds = tmp_path / "ds"
    st = importers.import_capture(b, ds)
    assert st["images"] == 1
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "tools"))
    for mod in list(sys.modules):
        if mod == "splatprep" or mod.startswith("splatprep."):
            monkeypatch.delitem(sys.modules, mod)
    from splatprep.priors import load_priors

    pri = load_priors(ds / "priors.json")
    assert len(pri) == 1
    np.testing.assert_allclose(pri[0].c2w[:3, :3], np.diag([1, -1, -1]))


def test_360_photos_pipeline_rig_nadir_masks_and_manual_layer(server, tmp_path):
    """360° photos folder → views cut here → nadir layer + the mask editor's layer stacked → rig solve."""
    client, backend, _ = server
    src = tmp_path / "pano"
    src.mkdir()
    rng = np.random.default_rng(3)
    for i in range(3):
        Image.fromarray((rng.random((48, 96, 3)) * 255).astype(np.uint8)).save(src / f"R{i}.jpg")
    opts = ProjectOptions(name="Pano", source_kind="photos360", source=str(src), sky=False, views_360=4,
                          max_image_size=24, holdout_every=0)
    proj = Project.create(tmp_path / "projects", "Pano", opts.to_dict())
    # a manual mask from the mask editor, made before "Prepare again": it must survive and be stacked
    manual = proj.dataset_dir / importers.MANUAL_LAYER / "masks"
    manual.mkdir(parents=True)
    mm = np.full((24, 24), 255, np.uint8)
    mm[:6, :6] = 0
    Image.fromarray(mm).save(manual / "000001_v00.png")
    out, events = run(client, proj, opts, train=False)
    assert events[-1][0] == "done", [d for k, d in events if k in ("failed", "log")][-5:]
    assert [d["title"] for k, d in events if k == "stage"] == stage_titles(opts, train=False)
    doc = json.loads((proj.dataset_dir / "dataset.json").read_text())
    assert len(doc["cameras"]) == 3 * 8                     # 4 horizon + 2 up + 2 down views per photo
    rig = doc["request"]["rig"]
    assert rig["kind"] == "multiview" and len(rig["groups"]) == 3 and all(len(g) == 8 for g in rig["groups"])
    assert doc["request"]["masksDir"] == "masks"
    by = {Path(c["image"]).stem: c for c in doc["cameras"]}
    assert "mask" in by["000000_v06"] and "mask" not in by["000000_v00"]     # nadir only on the downward views
    m = np.asarray(Image.open(proj.dataset_dir / by["000001_v00"]["mask"]))
    assert m[2, 2] == 0 and m[20, 20] == 255
    assert manual.is_dir()


def test_stereo_pipeline_rig_metric_scale_and_depth(server, tmp_path):
    client, backend, _ = server
    video = tmp_path / "in" / "pair_sbs.mp4"
    video.parent.mkdir()
    video.write_bytes(b"not really a video")
    right = tmp_path / "in" / "right.mov"
    right.write_bytes(b"x")
    opts = ProjectOptions(name="St", source_kind="stereo", source=str(video), sky=False, stereo_depth=True,
                          stereo_baseline_mm=50.0, stereo_layout="separateFiles", stereo_right=str(right),
                          holdout_every=0)
    proj = Project.create(tmp_path / "projects", "St", opts.to_dict())
    out, events = run(client, proj, opts)
    assert events[-1][0] == "done", [d for k, d in events if k in ("failed", "log")][-5:]
    assert [d["title"] for k, d in events if k == "stage"] == stage_titles(opts)
    assert "Stereo depth" in stage_titles(opts)
    ds = proj.dataset_dir
    fr = json.loads((ds / "frames.json").read_text())
    assert fr["request"]["stereo"] == {"layout": "separateFiles", "swapEyes": False, "baseline": 0.05,
                                       "rightVideo": "_right-eye.mov"}
    assert (proj.root / "source" / "_right-eye.mov").exists()
    doc = json.loads((ds / "dataset.json").read_text())
    rig = doc["request"]["rig"]
    assert rig["kind"] == "multiview" and len(rig["groups"]) == 4
    assert rig["groups"][0]["000000_R.png"] == {"R": np.eye(3).tolist(), "t": [0.05, 0.0, 0.0]}
    cams = {Path(c["image"]).stem: c for c in doc["cameras"]}
    # the fake solver put consecutive images 0.1 apart: ×0.5 makes L–R = 5 cm, right eyes re-derived from the left
    pl = np.asarray(cams["000002_L"]["cameraToWorld"])[:3, 3]
    pr = np.asarray(cams["000002_R"]["cameraToWorld"])[:3, 3]
    assert np.linalg.norm(pr - pl) == pytest.approx(0.05) and pl[0] == pytest.approx(4 * 0.1 * 0.5)
    left = cams["000001_L"]
    assert left["depth"] == "depth/000001_L.bin" and "depth" not in cams["000001_R"]
    z = np.fromfile(ds / left["depth"], "<f2").reshape(left["depthHeight"], left["depthWidth"])
    # f = 30 px, b = 5 cm, d = 4 px → 0.375 m where the matcher found the texture
    assert np.median(z[z > 0]) == pytest.approx(0.375, rel=0.05)
    assert out is not None and out.is_file()


def test_mask_editor_jobs_on_the_local_worker(server, tmp_path):
    """A SAM 2 click (maskpreview) and tracking (masks / sam2_track) as jobs of the local server."""
    from conftest import make_dataset
    from splatworker.studio.mask_edit import MaskEditModel, run_job, scratch_dir

    client, backend, _ = server
    ds = tmp_path / "proj" / "dataset"
    make_dataset(ds, n_cams=3, width=32, height=24, depth=False, points=False)
    m = MaskEditModel(ds)
    m.add_point(10, 10, True)
    dest = scratch_dir(ds)
    run_job(client, ds, m.preview_request("_mp"), "click", dest, poll=0.02)
    assert m.apply_object(dest / "preview.png") > 0 and not m.keep()[10, 10]
    dest2 = scratch_dir(ds)
    run_job(client, ds, m.track_request("_tr"), "track", dest2, poll=0.02)
    assert m.apply_tracked(dest2 / "masks") == 2           # the fake marks every other image
    m.save()
    doc = json.loads((ds / "dataset.json").read_text())
    assert sum("mask" in c for c in doc["cameras"]) == 2
    assert client.get("/v1/datasets") == []


def test_static_regions_stage(server, tmp_path):
    client, backend, _ = server
    photos = tmp_path / "drone"
    photos.mkdir()
    rng = np.random.default_rng(5)
    for i in range(14):
        img = (rng.random((48, 96, 3)) * 255).astype(np.uint8)
        img[30:44, 4:30] = 0
        img[35:38, 8:26] = 255
        Image.fromarray(img).save(photos / f"D{i:03d}.png")
    opts = ProjectOptions(name="D", source_kind="photos", source=str(photos), sky=False, mask_static=True,
                          blur_rejection=False, max_image_size=0)
    proj = Project.create(tmp_path / "projects", "D", opts.to_dict())
    out, events = run(client, proj, opts, train=False)
    assert events[-1][0] == "done", [d for k, d in events if k in ("failed", "log")][-5:]
    assert "Fixed parts of the frame" in [d["title"] for k, d in events if k == "stage"]
    cams = json.loads((proj.dataset_dir / "dataset.json").read_text())["cameras"]
    assert all("mask" in c for c in cams)
    m = np.asarray(Image.open(proj.dataset_dir / cams[0]["mask"]))
    assert m[36, 15] == 0 and m[10, 70] == 255
