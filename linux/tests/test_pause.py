"""Pause / resume and on-demand checkpoints of training jobs (capability "pause")."""
from fastapi.testclient import TestClient

from conftest import FakeBackend, make_dataset, wait_for
from splatworker.config import ServerSettings
from splatworker.ply import read_ply_vertices
from splatworker.server import create_app

TOKEN = "tok"
SECRET = "s" * 43
REMOTE = {"Authorization": f"Bearer {TOKEN}"}
LOCAL = {**REMOTE, "X-SplatWork-Local": SECRET}


def make(data_dir, backend=None, host="127.0.0.1"):
    settings = ServerSettings(data_dir=data_dir, token=TOKEN, name="w", mdns=False,
                              allow_local_datasets=True, local_secret=SECRET)
    backend = backend or FakeBackend()
    app = create_app(settings, backend=backend, devices=[None])
    c = TestClient(app, client=(host, 50000))
    c.backend = backend
    return c


def other(c, host):
    """A second client of the same running server, from another address."""
    return TestClient(c.app, client=(host, 50001))


def job_state(c, job_id):
    return c.get(f"/v1/jobs/{job_id}", headers=REMOTE).json()


def wait_state(c, job_id, states, timeout=10):
    states = (states,) if isinstance(states, str) else states
    return wait_for(lambda: (lambda s: s if s["state"] in states else None)(job_state(c, job_id)), timeout)


def dataset(c, tmp_path, name="p"):
    ds = make_dataset(tmp_path / name / "dataset")
    return c.post("/v1/datasets/local", json={"path": str(ds)}, headers=LOCAL).json()["datasetId"]


def train(c, ds_id, headers=LOCAL, iterations=40):
    r = c.post("/v1/jobs", json={"datasetId": ds_id, "config": {"iterations": iterations}}, headers=headers)
    assert r.status_code == 200, r.text
    return r.json()["jobId"]


def slow(c, steps=40, delay=0.03):
    c.backend.steps, c.backend.step_delay = steps, delay


def test_capability_advertised(tmp_path):
    with make(tmp_path / "d") as c:
        assert "pause" in c.get("/v1/info", headers=REMOTE).json()["capabilities"]


def test_pause_frees_gpu_then_resume_continues(tmp_path):
    with make(tmp_path / "d") as c:
        slow(c)
        ds = dataset(c, tmp_path)
        a = train(c, ds)
        wait_for(lambda: (job_state(c, a)["step"] or 0) >= 5)
        b = train(c, ds)                                 # queued behind a
        assert job_state(c, b)["state"] == "queued"
        r = c.post(f"/v1/jobs/{a}/pause", headers=LOCAL)
        assert r.status_code == 200, r.text
        s = wait_state(c, a, "paused")
        paused_at = s["step"]
        assert 5 <= paused_at < 40 and s["checkpointStep"] == paused_at and s["etaSeconds"] is None
        assert s["elapsedSeconds"] and s["elapsedSeconds"] > 0
        job_dir = tmp_path / "d" / "jobs" / a
        assert (job_dir / "resume.pt").is_file() and (job_dir / "checkpoint.ply").is_file()
        # the GPU went to the next job; the paused one doesn't count as busy and keeps its dataset
        wait_state(c, b, "completed")
        assert c.delete(f"/v1/datasets/{ds}", headers=LOCAL).status_code == 409
        # a paused job's checkpoint is there already
        r = c.post(f"/v1/jobs/{a}/checkpoint", headers=LOCAL)
        assert r.status_code == 200 and r.json()["checkpointStep"] == paused_at
        r = c.get(f"/v1/jobs/{a}/checkpoint.ply", headers=REMOTE)
        assert r.status_code == 200 and r.headers["X-Checkpoint-Step"] == str(paused_at)
        # resume: back in the queue, continues from the saved step
        r = c.post(f"/v1/jobs/{a}/resume", headers=LOCAL)
        assert r.status_code == 200 and r.json()["state"] in ("queued", "loading", "training")
        s = wait_state(c, a, "completed")
        assert c.backend.resumed_from[-1] == paused_at
        assert s["elapsedSeconds"] >= paused_at * 0.03 * 0.5
        assert not (job_dir / "resume.pt").exists()
        assert c.get(f"/v1/jobs/{a}/result.ply", headers=REMOTE).status_code == 200


def test_checkpoint_while_training(tmp_path):
    with make(tmp_path / "d") as c:
        slow(c)
        a = train(c, dataset(c, tmp_path))
        assert c.get(f"/v1/jobs/{a}/checkpoint.ply", headers=REMOTE).status_code == 404
        wait_for(lambda: (job_state(c, a)["step"] or 0) >= 3)
        r = c.post(f"/v1/jobs/{a}/checkpoint?wait=10", headers=LOCAL)
        assert r.status_code == 200, r.text
        step = r.json()["checkpointStep"]
        assert step is not None and step >= 3
        r = c.get(f"/v1/jobs/{a}/checkpoint.ply", headers=REMOTE)
        assert r.status_code == 200 and r.headers["X-Checkpoint-Step"] == str(step)
        path = tmp_path / "ck.ply"
        path.write_bytes(r.content)
        assert len(read_ply_vertices(path)["x"]) == 10
        assert wait_state(c, a, "completed")["state"] == "completed"   # training carried on


def test_pause_queued_job_holds_it(tmp_path):
    with make(tmp_path / "d") as c:
        c.backend.mode = "block"
        ds = dataset(c, tmp_path)
        a = train(c, ds)
        wait_for(c.backend.started.is_set)
        b = train(c, ds)
        s = c.post(f"/v1/jobs/{b}/pause", headers=LOCAL).json()
        assert s["state"] == "paused"
        assert c.post(f"/v1/jobs/{b}/checkpoint", headers=LOCAL).status_code == 409   # never trained
        c.backend.release.set()
        wait_state(c, a, "completed")
        assert job_state(c, b)["state"] == "paused"     # not picked up while held
        c.post(f"/v1/jobs/{b}/resume", headers=LOCAL)
        wait_state(c, b, "completed")
        assert c.backend.resumed_from[-1] == 0


def test_cancel_paused_job(tmp_path):
    with make(tmp_path / "d") as c:
        slow(c)
        a = train(c, dataset(c, tmp_path))
        wait_for(lambda: (job_state(c, a)["step"] or 0) >= 2)
        c.post(f"/v1/jobs/{a}/pause", headers=LOCAL)
        wait_state(c, a, "paused")
        s = c.post(f"/v1/jobs/{a}/cancel", headers=REMOTE).json()
        assert s["state"] == "cancelled"
        assert not (tmp_path / "d" / "jobs" / a / "resume.pt").exists()
        assert c.post(f"/v1/jobs/{a}/resume", headers=LOCAL).status_code == 409


def test_invalid_transitions(tmp_path):
    with make(tmp_path / "d") as c:
        a = train(c, dataset(c, tmp_path), iterations=5)
        wait_state(c, a, "completed")
        assert c.post(f"/v1/jobs/{a}/pause", headers=LOCAL).status_code == 409
        assert c.post(f"/v1/jobs/{a}/resume", headers=LOCAL).status_code == 409
        assert c.post(f"/v1/jobs/{a}/checkpoint", headers=LOCAL).status_code == 409
        assert c.post("/v1/jobs/jnope/pause", headers=LOCAL).status_code == 404


def test_who_may_pause(tmp_path):
    with make(tmp_path / "d") as c:
        c.backend.mode = "block"
        ds = dataset(c, tmp_path)
        mine = train(c, ds)                                 # the desktop app's job
        wait_for(c.backend.started.is_set)
        mac = other(c, "192.168.0.20")
        stranger = other(c, "192.168.0.30")
        # the remote client can't use the desktop app's local dataset; give it an uploaded one
        import io
        import zipfile
        src = make_dataset(tmp_path / "up")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            for p in src.rglob("*"):
                if p.is_file():
                    zf.write(p, str(p.relative_to(src)))
        up = mac.post("/v1/datasets", content=buf.getvalue(),
                      headers={**REMOTE, "Content-Type": "application/zip"}).json()["datasetId"]
        theirs = train(mac, up, headers=REMOTE)
        assert job_state(c, theirs)["origin"] == "192.168.0.20"
        # nobody else may park the desktop app's job, or the Mac's
        assert mac.post(f"/v1/jobs/{mine}/pause", headers=REMOTE).status_code == 403
        assert stranger.post(f"/v1/jobs/{theirs}/pause", headers=REMOTE).status_code == 403
        assert stranger.post(f"/v1/jobs/{theirs}/checkpoint", headers=REMOTE).status_code == 403
        # a loopback request without the secret is remote too
        assert c.post(f"/v1/jobs/{mine}/pause", headers=REMOTE).status_code == 403
        # the submitter and the desktop app may
        assert mac.post(f"/v1/jobs/{theirs}/pause", headers=REMOTE).json()["state"] == "paused"
        assert c.post(f"/v1/jobs/{theirs}/resume", headers=LOCAL).json()["state"] == "queued"
        c.backend.release.set()
        wait_state(c, theirs, "completed")


def test_paused_job_survives_restart(tmp_path):
    data = tmp_path / "d"
    with make(data) as c:
        slow(c)
        ds = dataset(c, tmp_path)
        a = train(c, ds)
        wait_for(lambda: (job_state(c, a)["step"] or 0) >= 4)
        c.post(f"/v1/jobs/{a}/pause", headers=LOCAL)
        paused_at = wait_state(c, a, "paused")["step"]
    with make(data) as c:
        s = job_state(c, a)
        assert s["state"] == "paused" and s["step"] == paused_at
        c.post(f"/v1/jobs/{a}/resume", headers=LOCAL)
        wait_state(c, a, "completed")
        assert c.backend.resumed_from == [paused_at]


def test_restart_during_resumed_run_keeps_last_pause(tmp_path):
    """A job that was resumed and then interrupted by a restart goes back to paused (its last pause is on disk)."""
    data = tmp_path / "d"
    with make(data) as c:
        slow(c, steps=400, delay=0.02)
        a = train(c, dataset(c, tmp_path))
        wait_for(lambda: (job_state(c, a)["step"] or 0) >= 2)
        c.post(f"/v1/jobs/{a}/pause", headers=LOCAL)
        wait_state(c, a, "paused")
        c.post(f"/v1/jobs/{a}/resume", headers=LOCAL)
        wait_state(c, a, "training")
    # the shutdown cancelled the run, which marks it failed and drops the saved state; simulate a hard stop
    # (power cut) instead by putting the job back to "training" with its resume file
    import json
    job_dir = data / "jobs" / a
    doc = json.loads((job_dir / "job.json").read_text())
    doc["status"]["state"] = "training"
    (job_dir / "job.json").write_text(json.dumps(doc))
    (job_dir / "resume.pt").write_bytes(b"fake")
    with make(data) as c:
        s = job_state(c, a)
        assert s["state"] == "paused" and "restart" in s["message"]
