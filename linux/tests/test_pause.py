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
    return c.get(f"/v1/jobs/{job_id}", headers=LOCAL).json()


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
        r = c.get(f"/v1/jobs/{a}/checkpoint.ply", headers=LOCAL)
        assert r.status_code == 200 and r.headers["X-Checkpoint-Step"] == str(paused_at)
        # resume: back in the queue, continues from the saved step
        r = c.post(f"/v1/jobs/{a}/resume", headers=LOCAL)
        assert r.status_code == 200 and r.json()["state"] in ("queued", "loading", "training")
        s = wait_state(c, a, "completed")
        assert c.backend.resumed_from[-1] == paused_at
        assert s["elapsedSeconds"] >= paused_at * 0.03 * 0.5
        assert not (job_dir / "resume.pt").exists()
        assert c.get(f"/v1/jobs/{a}/result.ply", headers=LOCAL).status_code == 200


def test_checkpoint_while_training(tmp_path):
    with make(tmp_path / "d") as c:
        slow(c)
        a = train(c, dataset(c, tmp_path))
        assert c.get(f"/v1/jobs/{a}/checkpoint.ply", headers=LOCAL).status_code == 404
        wait_for(lambda: (job_state(c, a)["step"] or 0) >= 3)
        r = c.post(f"/v1/jobs/{a}/checkpoint?wait=10", headers=LOCAL)
        assert r.status_code == 200, r.text
        step = r.json()["checkpointStep"]
        assert step is not None and step >= 3
        r = c.get(f"/v1/jobs/{a}/checkpoint.ply", headers=LOCAL)
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
        s = c.post(f"/v1/jobs/{a}/cancel", headers=LOCAL).json()
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
        # the desktop app doesn't park other computers' jobs (their apps may have no Resume button)
        assert c.post(f"/v1/jobs/{theirs}/pause", headers=LOCAL).status_code == 403
        # the submitter may; the desktop app may resume it
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


def _bare(tmp_path):
    """App + manager without the worker threads (TestClient not entered), so tests can step the queue."""
    c = make(tmp_path / "d")
    return c, c.app.state.manager


def test_pause_or_resume_between_take_and_start(tmp_path):
    """A job taken from the queue is "loading" at once: a pause then holds it before it starts, and a resume
    can't put it back in the queue (two GPUs would train one job dir)."""
    c, m = _bare(tmp_path)
    ds = dataset(c, tmp_path)
    a = train(c, ds)
    job = m._next_job(None)
    assert job.id == a and job_state(c, a)["state"] == "loading"
    assert c.post(f"/v1/jobs/{a}/resume", headers=LOCAL).json()["state"] == "loading"
    assert a not in m.queue
    assert c.post(f"/v1/jobs/{a}/pause", headers=LOCAL).status_code == 200
    m._run_job(job, None)
    s = job_state(c, a)
    assert s["state"] == "paused" and s["pausedAt"] and c.backend.contexts == []    # never ran
    # and a cancel in that gap stops it before it starts
    b = train(c, ds)
    job = m._next_job(None)
    assert c.post(f"/v1/jobs/{b}/cancel", headers=LOCAL).status_code == 200
    m._run_job(job, None)
    assert job_state(c, b)["state"] == "cancelled" and c.backend.contexts == []


def test_other_computers_paused_jobs_resume_by_themselves(tmp_path, monkeypatch):
    import splatworker.server as S

    c, m = _bare(tmp_path)
    mac = other(c, "192.168.0.20")
    ds = dataset(c, tmp_path)
    mine = train(c, ds)
    up = _upload(mac, tmp_path)
    theirs = train(mac, up, headers=REMOTE)
    assert mac.post(f"/v1/jobs/{theirs}/pause", headers=REMOTE).json()["state"] == "paused"
    assert c.post(f"/v1/jobs/{mine}/pause", headers=LOCAL).json()["state"] == "paused"
    (m.jobs[mine].dir / "resume.pt").write_bytes(b"x")
    now = __import__("time").time()
    monkeypatch.setattr(S.time, "time", lambda: now + S.REMOTE_PAUSE_LIMIT + 1)
    with m.cond:
        m._housekept = 0.0
        m._housekeep_locked()
    assert job_state(c, theirs)["state"] == "queued" and job_state(c, mine)["state"] == "paused"
    monkeypatch.setattr(S.time, "time", lambda: now + S.LOCAL_PAUSE_LIMIT + 1)
    with m.cond:
        m._housekept = 0.0
        m._housekeep_locked()
    assert job_state(c, mine)["state"] == "cancelled" and not (m.jobs[mine].dir / "resume.pt").exists()


def test_paused_remote_job_resumes_after_restart(tmp_path):
    c, m = _bare(tmp_path)
    mac = other(c, "192.168.0.20")
    theirs = train(mac, _upload(mac, tmp_path), headers=REMOTE)
    mac.post(f"/v1/jobs/{theirs}/pause", headers=REMOTE)
    c2 = make(tmp_path / "d")
    assert job_state(c2, theirs)["state"] == "queued"


def test_lan_clients_cant_touch_or_read_the_apps_jobs(tmp_path):
    c, m = _bare(tmp_path)
    mac = other(c, "192.168.0.20")
    stranger = other(c, "192.168.0.30")
    mine = train(c, dataset(c, tmp_path))
    theirs = train(mac, _upload(mac, tmp_path), headers=REMOTE)
    listed = {j["jobId"]: j for j in mac.get("/v1/jobs", headers=REMOTE).json()}
    assert listed[mine]["name"] == "Job on this PC" and listed[mine]["datasetId"] == ""
    assert listed[mine]["state"] == "queued" and listed[mine]["origin"] == "local"
    assert listed[theirs]["origin"] == "192.168.0.20"
    assert {j["jobId"]: j for j in stranger.get("/v1/jobs", headers=REMOTE).json()}[theirs]["origin"] == "remote"
    assert mac.get(f"/v1/jobs/{mine}/log", headers=REMOTE).status_code == 404
    assert mac.post(f"/v1/jobs/{mine}/cancel", headers=REMOTE).status_code == 403
    assert mac.delete(f"/v1/jobs/{mine}", headers=REMOTE).status_code == 403
    assert c.get("/v1/jobs", headers=LOCAL).json()[0]["name"] != "Job on this PC"
    # other computers' jobs can still be stopped by anybody, as before
    assert stranger.post(f"/v1/jobs/{theirs}/cancel", headers=REMOTE).json()["state"] == "cancelled"


def test_census_before_its_file_exists_is_404(tmp_path):
    c, m = _bare(tmp_path)
    ds = dataset(c, tmp_path)
    from conftest import write_fake_ply
    ply = tmp_path / "s.ply"
    write_fake_ply(ply, 5)
    j = c.post("/v1/census", json={"datasetId": ds, "plyPath": str(ply)}, headers=LOCAL).json()["jobId"]
    assert c.get(f"/v1/jobs/{j}/census.npz", headers=LOCAL).status_code == 404


def _upload(client, tmp_path):
    import io
    import zipfile
    src = make_dataset(tmp_path / f"up{id(client)}")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for p in src.rglob("*"):
            if p.is_file():
                zf.write(p, str(p.relative_to(src)))
    return client.post("/v1/datasets", content=buf.getvalue(),
                       headers={**REMOTE, "Content-Type": "application/zip"}).json()["datasetId"]


def test_worker_shutdown_pauses_a_running_training_job_and_keeps_its_state(tmp_path):
    """Quitting the app / restarting the service must not throw away a training job's saved state: running training
    jobs are paused (state saved), and come back paused (this PC's job) after the restart."""
    import json
    import threading
    import time as _t
    from splatworker.backends import Backend, JobPaused
    from splatworker.config import ServerSettings
    from splatworker.server import JobManager

    s = ServerSettings(data_dir=tmp_path)
    for x in (s.datasets_dir, s.jobs_dir, s.tmp_dir):
        x.mkdir(parents=True, exist_ok=True)
    ds = s.datasets_dir / "dabc"
    (ds / "files").mkdir(parents=True)
    (ds / "meta.json").write_text(json.dumps({"root": "files"}))

    class Trainer(Backend):
        pausable = True

        def capabilities(self):
            return ["train"]

        def run(self, ctx):
            step = 0
            while True:
                if ctx.pause_event.is_set():
                    (ctx.job_dir / "resume.pt").write_bytes(b"state")
                    raise JobPaused(step)
                if ctx.cancel_event.is_set():
                    raise AssertionError("shutdown should pause, not cancel, a pausable training job")
                step += 1
                _t.sleep(0.01)

    m = JobManager(s, Trainer(), [None])
    m.load()
    job = m.submit("train", "dabc", "n", "config", {}, 100, origin="local")
    m.start()
    for _ in range(200):
        if job.state == "training" or job.state == "loading":
            break
        _t.sleep(0.01)
    _t.sleep(0.1)
    m.shutdown()
    assert job.state == "paused" and (job.dir / "resume.pt").is_file()
    m2 = JobManager(s, Trainer(), [None])
    m2.load()
    assert m2.jobs[job.id].state == "paused" and m2.jobs[job.id].status.get("pausedAt")
