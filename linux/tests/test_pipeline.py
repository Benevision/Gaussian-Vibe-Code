"""Whole-pipeline jobs (capability "pipeline"): every stage on the worker, the client may be offline throughout."""
import io
import json
import threading
import time
import zipfile

import numpy as np
from fastapi.testclient import TestClient

from conftest import FakeBackend, look_at_c2w, wait_for, write_fake_ply, write_points_ply, zip_dir
from splatworker import pipeline as pl
from splatworker.config import ServerSettings
from splatworker.server import create_app

TOKEN = "tok"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
N_FRAMES = 12
TILT = pl.rotation_between(np.array([0.0, -1.0, 0.0]), np.array([0.3, -1.0, 0.2]))


def tilted(c2w):
    m = np.eye(4)
    m[:3, :3] = TILT
    return m @ c2w


class PipelineFake(FakeBackend):
    """Fakes every splatprep task with tiny outputs: frames from a "video", keep masks, sky masks, a camera solve."""

    def __init__(self):
        super().__init__()
        self.tasks = []
        self.block_tasks = set()        # tasks that wait for ``release_task`` (or cancel)
        self.release_task = threading.Event()
        self.empty_masks = False
        self.sky_fails = False

    def capabilities(self):
        return ["train", "process", "frames"]

    def run(self, ctx):
        if ctx.kind != "process":
            return super().run(ctx)
        from PIL import Image

        task = ctx.spec["task"] if ctx.spec.get("maskModel") != "sky" else "sky"
        self.tasks.append(task)
        ctx.status(state="running", message=f"{task}: working", step=500, totalSteps=1000)
        if task in self.block_tasks:
            while not ctx.cancel_event.is_set() and not self.release_task.is_set():
                time.sleep(0.01)
            if ctx.cancel_event.is_set():
                from splatworker.backends import JobCancelled
                raise JobCancelled()
        out = ctx.job_dir / "output"
        out.mkdir(parents=True)
        ds = ctx.dataset_dir
        if task == "frames":
            assert (ds / ctx.spec["video"]).is_file()
            (out / "images").mkdir()
            frames = []
            for i in range(N_FRAMES):
                img = (np.random.default_rng(i).random((48, 64, 3)) * 255).astype(np.uint8)
                Image.fromarray(img).save(out / "images" / f"{i:06d}.png")
                frames.append({"image": f"images/{i:06d}.png", "width": 64, "height": 48, "sharpness": 1.0,
                               "timestamp": i / 10})
            (out / "frames.json").write_text(json.dumps({"root": ".", "frames": frames, "rejected": [],
                                                         "isMultiViewRig": False}))
        elif task == "masks":
            (out / "masks").mkdir()
            for p in sorted((ds / "images").iterdir()):
                keep = np.full((48, 64), 0 if self.empty_masks else 255, np.uint8)
                keep[:10, :10] = 0
                Image.fromarray(keep).save(out / "masks" / f"{p.stem}.png")
        elif task == "sky":
            if self.sky_fails:
                raise RuntimeError("model unavailable: sky")
            (out / "masks").mkdir()
            for p in sorted((ds / "images").iterdir()):
                sky = np.zeros((48, 64), np.uint8)
                sky[:12] = 255
                Image.fromarray(sky).save(out / "masks" / f"{p.stem}.png")
        elif task == "sfm":
            frames = json.loads((ds / "frames.json").read_text())["frames"]
            self.sfm_request = dict(ctx.spec)
            cams = []
            for i, f in enumerate(frames[:-1]):                 # the last frame isn't registered
                ang = 2 * np.pi * i / len(frames)
                pos = (3 * np.cos(ang), 0.0, 3 * np.sin(ang))
                if i == 3:
                    pos = (400.0, 0.0, 0.0)                     # mis-registered far away: PoseSanity drops it
                cams.append({"id": i, "image": f["image"], "width": 64, "height": 48, "fx": 55.0, "fy": 55.0,
                             "cx": 32.0, "cy": 24.0, "cameraToWorld": tilted(look_at_c2w(pos)).tolist()})
            rng = np.random.default_rng(0)
            pts = (rng.uniform(-1, 1, (100, 3)) @ TILT.T).astype(np.float32)
            write_points_ply(out / "points3d.ply", pts, rng.integers(0, 255, (100, 3)).astype(np.uint8))
            (out / "dataset.json").write_text(json.dumps({"format": "splatdataset", "version": 1, "convention": "opencv",
                                                         "sceneCenter": [0, 0, 0], "sceneExtent": 3, "upVector": [0, -1, 0],
                                                         "points": "points3d.ply", "cameras": cams}))
        return {"files": 1}


def make(data_dir, backend=None):
    settings = ServerSettings(data_dir=data_dir, token=TOKEN, name="w", mdns=False)
    backend = backend or PipelineFake()
    app = create_app(settings, backend=backend, devices=[None])
    c = TestClient(app)
    c.backend = backend
    return c


def upload_video(c):
    r = c.post("/v1/datasets/file?name=video.mp4", content=b"\x00fake video" * 100,
               headers={**AUTH, "Content-Type": "application/octet-stream"})
    assert r.status_code == 200, r.text
    return r.json()["datasetId"]


def spec(train=True, **over):
    p = {
        "frames": {"request": {"task": "frames", "video": "video.mp4", "outputDir": "frames"}},
        "masks": {"request": {"maskModel": "dynamic", "invertMask": False}, "dilatePixels": 2},
        "sky": {"request": {"maskModel": "sky", "invertMask": True}},
        "sfm": {"request": {"features": "superpoint_lightglue", "matching": "sequential", "mapper": "auto",
                            "cameraModel": "pinhole", "resizeMax": 1600}, "sequential": True},
        "holdoutInterval": 4,
        "level": True,
        "checkpoints": {"frames": "fpF", "masks": "fpM", "poses": "fpP", "done": "fpD"},
    }
    if train:
        p["train"] = {"config": {"iterations": 5}}
    p.update(over)
    return p


def submit(c, ds, p):
    r = c.post("/v1/pipeline", json={"datasetId": ds, "name": "Kitchen", "pipeline": p}, headers=AUTH)
    assert r.status_code == 200, r.text
    return r.json()["jobId"]


def status(c, jid):
    return c.get(f"/v1/jobs/{jid}", headers=AUTH).json()


def wait_state(c, jid, states, timeout=15):
    states = (states,) if isinstance(states, str) else states
    return wait_for(lambda: (lambda s: s if s["state"] in states else None)(status(c, jid)), timeout)


def output_zip(c, jid, **params):
    r = c.get(f"/v1/jobs/{jid}/output.zip", params=params, headers=AUTH)
    assert r.status_code == 200, r.text
    return zipfile.ZipFile(io.BytesIO(r.content))


# ------------------------------------------------------------------ API


def test_capability(tmp_path):
    with make(tmp_path / "d") as c:
        assert "pipeline" in c.get("/v1/info", headers=AUTH).json()["capabilities"]


def test_video_to_result_while_client_is_away(tmp_path):
    with make(tmp_path / "d") as c:
        jid = submit(c, upload_video(c), spec())
        assert jid.startswith("w")
        s = wait_state(c, jid, ("completed", "failed"))
        assert s["state"] == "completed", s["message"]
        assert s["kind"] == "pipeline" and s["step"] == s["totalSteps"] == 1000
        assert [st["id"] for st in s["stages"]] == ["frames", "static", "masks", "sky", "sfm", "level", "train"]
        assert [st["state"] for st in s["stages"]] == ["done", "skipped", "done", "done", "done", "done", "done"]
        assert [st["prepStage"] for st in s["stages"]] == ["frames", "masks", "masks", "masks", "poses", "done", None]
        assert c.backend.tasks == ["frames", "masks", "sky", "sfm"]
        # every step was an ordinary child job of the queue
        jobs = c.get("/v1/jobs", headers=AUTH).json()
        children = [j for j in jobs if j.get("parentJobId") == jid]
        assert len(children) == 5 and all(j["state"] == "completed" for j in children)
        assert s["trainJobId"] in {j["jobId"] for j in children}
        # the request the worker built for the solve: masks + sky known only on the worker
        r = c.backend.sfm_request
        assert r["masksDir"] == "masks" and r["skyMasksDir"] == "splatprep-sky/masks" and r["singleCamera"] is True
        assert r["matching"] == "sequential" and r["outputDir"] == "sfm"
        # result
        res = c.get(f"/v1/jobs/{jid}/result.ply", headers=AUTH)
        assert res.status_code == 200 and res.content.startswith(b"ply")
        assert s["numGaussians"] == 10
        # prepared dataset with the per-stage snapshots
        z = output_zip(c, jid, skip="images")
        names = set(z.namelist())
        assert not any(n.startswith("images/") for n in names)
        for n in (".pipeline/pipeline.json", ".pipeline/frames.frames.json", ".pipeline/masks.frames.json",
                  ".pipeline/poses.dataset.json", ".pipeline/poses.points.ply", "dataset.json", "points3d.ply",
                  "frames.json", "masks/000000.png", "splatprep-sky/masks/000000.png"):
            assert n in names, n
        man = json.loads(z.read(".pipeline/pipeline.json"))
        assert man["checkpoints"]["poses"] == "fpP" and man["jobId"] == jid
        doc = json.loads(z.read("dataset.json"))
        cams = doc["cameras"]
        assert len(cams) == N_FRAMES - 2                  # one unregistered, one dropped by PoseSanity
        assert "images/000003.png" not in {c_["image"] for c_ in cams}
        assert all(c_["mask"].startswith("masks/") and c_["sky"].startswith("splatprep-sky/") for c_ in cams)
        assert [c_.get("holdout", False) for c_ in cams][:5] == [True, False, False, False, True]
        # levelled: up = −Y, centred
        assert doc["upVector"] == [0.0, -1.0, 0.0]
        assert np.allclose(doc["sceneCenter"], 0, atol=1e-5)
        downs = np.array([np.asarray(c_["cameraToWorld"])[:3, 1] for c_ in cams])
        assert np.allclose(downs.mean(axis=0), [0, 1, 0], atol=1e-3)
        # the pre-levelling solve is kept for the poses checkpoint
        pre = json.loads(z.read(".pipeline/poses.dataset.json"))
        assert not np.allclose(np.asarray(pre["cameras"][0]["cameraToWorld"]), np.asarray(cams[0]["cameraToWorld"]))
        # masks: model's keep region eroded by 2 px and saved per frame; the frame-stage list had none
        from PIL import Image
        m = np.asarray(Image.open(io.BytesIO(z.read("masks/000000.png"))))
        assert m[11, 11] == 0 and m[13, 13] == 255 and m[40, 60] == 255
        assert not any("mask" in f for f in json.loads(z.read(".pipeline/frames.frames.json"))["frames"])
        assert all(f.get("mask") for f in json.loads(z.read(".pipeline/masks.frames.json"))["frames"])
        assert c.get(f"/v1/jobs/{jid}/log", headers=AUTH).text.count("== ") == 7


def test_frames_from_the_client_and_no_training(tmp_path):
    from PIL import Image

    src = tmp_path / "up"
    (src / "images").mkdir(parents=True)
    frames = []
    for i in range(N_FRAMES):
        Image.fromarray((np.random.default_rng(i).random((48, 64, 3)) * 255).astype(np.uint8)).save(
            src / "images" / f"{i:06d}.png")
        frames.append({"image": f"images/{i:06d}.png", "width": 64, "height": 48, "sharpness": 0})
    (src / "frames.json").write_text(json.dumps({"root": "file:///x/", "frames": frames, "rejected": [],
                                                 "isMultiViewRig": False}))
    with make(tmp_path / "d") as c:
        r = c.post("/v1/datasets", content=zip_dir(src), headers={**AUTH, "Content-Type": "application/zip"})
        ds = r.json()["datasetId"]
        p = spec(train=False)
        del p["frames"], p["masks"], p["sky"]
        jid = submit(c, ds, p)
        s = wait_state(c, jid, ("completed", "failed"))
        assert s["state"] == "completed", s["message"]
        assert [st["id"] for st in s["stages"]] == ["static", "sfm", "level"]
        assert s["outputDatasetId"] and s["trainJobId"] is None
        assert c.get(f"/v1/jobs/{jid}/result.ply", headers=AUTH).status_code == 404
        z = output_zip(c, jid)
        assert "images/000000.png" in z.namelist() and ".pipeline/frames.frames.json" in z.namelist()
        # the dataset can be trained again later
        t = c.post("/v1/jobs", json={"datasetId": s["outputDatasetId"], "config": {"iterations": 3}}, headers=AUTH)
        assert t.status_code == 200, t.text


def test_invalid_specs(tmp_path):
    with make(tmp_path / "d") as c:
        ds = upload_video(c)
        p = spec()
        del p["sfm"]
        assert c.post("/v1/pipeline", json={"datasetId": ds, "pipeline": p}, headers=AUTH).status_code == 400
        p = spec()
        del p["frames"]         # a bare video is not a frame list
        r = c.post("/v1/pipeline", json={"datasetId": ds, "pipeline": p}, headers=AUTH)
        assert r.status_code == 400 and "frames.json" in r.json()["error"]
        p = spec(train=True)
        p["train"] = {"config": {"iterations": -3}}
        assert c.post("/v1/pipeline", json={"datasetId": ds, "pipeline": p}, headers=AUTH).status_code == 400
        assert c.post("/v1/pipeline", json={"datasetId": "nope", "pipeline": spec()}, headers=AUTH).status_code == 404


def test_masks_that_keep_nothing_fail_the_job(tmp_path):
    with make(tmp_path / "d") as c:
        c.backend.empty_masks = True
        jid = submit(c, upload_video(c), spec())
        s = wait_state(c, jid, ("completed", "failed"))
        assert s["state"] == "failed" and "kept almost nothing" in s["message"]
        assert [st["state"] for st in s["stages"]][:3] == ["done", "skipped", "failed"]
        assert "sfm" not in c.backend.tasks


def test_missing_sky_model_is_skipped(tmp_path):
    with make(tmp_path / "d") as c:
        c.backend.sky_fails = True
        jid = submit(c, upload_video(c), spec(train=False))
        s = wait_state(c, jid, ("completed", "failed"))
        assert s["state"] == "completed", s["message"]
        sky = next(st for st in s["stages"] if st["id"] == "sky")
        assert sky["state"] == "skipped" and "sky model" in sky["message"]
        assert "skyMasksDir" not in c.backend.sfm_request


def test_cancel_stops_the_running_step(tmp_path):
    with make(tmp_path / "d") as c:
        c.backend.block_tasks = {"masks"}
        jid = submit(c, upload_video(c), spec())
        wait_for(lambda: status(c, jid)["stage"] == "masks" and "masks" in c.backend.tasks)
        child = next(st["jobId"] for st in status(c, jid)["stages"] if st["id"] == "masks")
        assert c.post(f"/v1/jobs/{jid}/cancel", headers=AUTH).status_code == 200
        s = wait_state(c, jid, "cancelled")
        assert wait_state(c, child, "cancelled")
        work = (tmp_path / "d" / "jobs" / jid / "job.json")
        work_id = json.loads(work.read_text())["spec"]["workDatasetId"]
        assert not (tmp_path / "d" / "datasets" / work_id).exists()     # nothing usable: dropped
        assert s["stages"][2]["state"] == "cancelled"


def test_delete_removes_children_and_work(tmp_path):
    with make(tmp_path / "d") as c:
        jid = submit(c, upload_video(c), spec(train=False))
        s = wait_state(c, jid, "completed")
        kids = [st["jobId"] for st in s["stages"] if st["jobId"]]
        assert c.delete(f"/v1/jobs/{jid}", headers=AUTH).status_code == 204
        assert c.get(f"/v1/jobs/{jid}", headers=AUTH).status_code == 404
        assert all(c.get(f"/v1/jobs/{k}", headers=AUTH).status_code == 404 for k in kids)
        assert not (tmp_path / "d" / "datasets" / s["outputDatasetId"]).exists()


def test_pause_and_resume_training(tmp_path):
    with make(tmp_path / "d") as c:
        c.backend.steps, c.backend.step_delay = 60, 0.03
        jid = submit(c, upload_video(c), spec())
        wait_for(lambda: status(c, jid)["stage"] == "train" and (status(c, status(c, jid)["trainJobId"])["step"] or 0) >= 3)
        # the prepared dataset is available while training runs
        assert status(c, jid)["outputDatasetId"]
        assert c.post(f"/v1/jobs/{jid}/pause", headers=AUTH).status_code == 200
        s = wait_state(c, jid, "paused")
        assert status(c, s["trainJobId"])["state"] == "paused"
        assert c.post(f"/v1/jobs/{jid}/resume", headers=AUTH).status_code == 200
        s = wait_state(c, jid, "completed", timeout=20)
        assert c.backend.resumed_from[-1] > 0


def test_processing_step_interrupted_by_a_restart_runs_again(tmp_path):
    data = tmp_path / "d"
    with make(data) as c:
        c.backend.block_tasks = {"sfm"}
        jid = submit(c, upload_video(c), spec(train=False))
        wait_for(lambda: "sfm" in c.backend.tasks)
        first = next(st["jobId"] for st in status(c, jid)["stages"] if st["id"] == "sfm")
    # worker restarted: the solve was interrupted; the pipeline continues where it was
    with make(data) as c:
        s = wait_state(c, jid, ("completed", "failed"))
        assert s["state"] == "completed", s["message"]
        sfm = next(st for st in s["stages"] if st["id"] == "sfm")
        assert sfm["jobId"] != first and sfm["attempts"] == 2
        assert c.backend.tasks == ["sfm"]                 # frames / masks / sky weren't redone
        assert status(c, first)["state"] == "failed"


def test_training_survives_a_restart(tmp_path):
    data = tmp_path / "d"
    with make(data) as c:
        c.backend.steps, c.backend.step_delay = 80, 0.03
        jid = submit(c, upload_video(c), spec())
        wait_for(lambda: status(c, jid)["stage"] == "train" and (status(c, status(c, jid)["trainJobId"])["step"] or 0) >= 3)
        train = status(c, jid)["trainJobId"]
    b = PipelineFake()
    b.steps = 80
    with make(data, b) as c:
        s = wait_state(c, jid, ("completed", "failed"), timeout=20)
        assert s["state"] == "completed", s["message"]
        assert s["trainJobId"] == train and b.resumed_from[0] > 0     # continued from the pause at shutdown
        assert c.get(f"/v1/jobs/{jid}/result.ply", headers=AUTH).status_code == 200


# ------------------------------------------------------------------ stage logic


def test_estimate_up_and_level(tmp_path):
    cams = [tilted(look_at_c2w((3 * np.cos(a), 0.0, 3 * np.sin(a)))) for a in np.linspace(0, 6, 10)]
    up = pl.estimate_up(cams)
    assert np.allclose(up, TILT @ np.array([0, -1.0, 0]), atol=1e-6)
    doc = {"format": "splatdataset", "version": 1, "convention": "opencv", "sceneCenter": [0, 0, 0],
           "sceneExtent": 1, "upVector": [0, -1, 0],
           "cameras": [{"image": f"images/{i}.png", "cameraToWorld": (m + np.r_[np.zeros((3, 4)), [[0, 0, 0, 0]]]).tolist()}
                       for i, m in enumerate(cams)]}
    for c_ in doc["cameras"]:
        c_["cameraToWorld"][0][3] += 5.0                  # off-centre
    pl.recompute_bounds(doc)
    (tmp_path / "dataset.json").write_text(json.dumps(doc))
    pl.level(tmp_path)
    out = json.loads((tmp_path / "dataset.json").read_text())
    assert np.allclose(out["sceneCenter"], 0, atol=1e-6) and out["upVector"] == [0.0, -1.0, 0.0]


def test_pose_outliers():
    pos = [np.array([i * 0.1, 0, 0]) for i in range(12)]
    pos[5] = np.array([0.5, 4.0, 0])          # lone spike away from both neighbours
    pos[9] = np.array([500.0, 0, 0])          # far from everything
    assert pl.pose_outliers(pos, ordered=True) == {5, 9}
    assert pl.pose_outliers(pos, ordered=False) == {9}


def test_multiview_rig_fills_missing_views():
    R1 = pl.rotation_between(np.array([0, 0, 1.0]), np.array([1.0, 0, 0]))
    fs = {"frames": [{"image": "images/0_v0.jpg", "group": 0, "rotationToRig": np.eye(3).tolist()},
                     {"image": "images/0_v1.jpg", "group": 0, "rotationToRig": R1.tolist()},
                     {"image": "images/0_v2.jpg", "group": 0, "rotationToRig": R1.T.tolist()}]}
    rig = np.eye(4)
    rig[:3, :3] = pl.rotation_between(np.array([0, -1.0, 0]), np.array([0.2, -1.0, 0]))
    rig[:3, 3] = [1, 2, 3]
    poses = {}
    for f in fs["frames"][:2]:
        m = np.eye(4)
        m[:3, :3] = rig[:3, :3] @ np.asarray(f["rotationToRig"])
        m[:3, 3] = rig[:3, 3]
        poses[f["image"]] = {"c2w": m}
    assert pl.enforce_multiview_rig(poses, fs) == 1
    m = poses["images/0_v2.jpg"]["c2w"]
    assert np.allclose(m[:3, :3], rig[:3, :3] @ R1.T, atol=1e-6) and np.allclose(m[:3, 3], [1, 2, 3])


def test_circular_fisheye(tmp_path):
    from PIL import Image, ImageDraw

    paths = []
    for i in range(3):
        im = Image.new("RGB", (200, 200), "black")
        ImageDraw.Draw(im).ellipse((5, 5, 195, 195), fill=(120, 140, 90))
        paths.append(tmp_path / f"f{i}.png")
        im.save(paths[-1])
    assert pl.circular_fisheye(paths)
    Image.new("RGB", (200, 150), (90, 90, 90)).save(tmp_path / "flat.png")
    assert not pl.circular_fisheye([tmp_path / "flat.png"])


def test_writes_points_round_trip(tmp_path):
    xyz = np.random.default_rng(0).normal(size=(20, 3)).astype(np.float32)
    rgb = np.random.default_rng(1).integers(0, 255, (20, 3)).astype(np.uint8)
    pl.write_points(tmp_path / "p.ply", xyz, rgb)
    a, b = pl.read_points(tmp_path / "p.ply")
    assert np.allclose(a, xyz) and (b == rgb).all()


def test_fake_ply_helper_is_importable():
    assert callable(write_fake_ply)
