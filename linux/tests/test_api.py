import io
import json
import zipfile

import pytest
from fastapi.testclient import TestClient

from conftest import FakeBackend, make_dataset, wait_for, zip_dir
from splatworker.config import ServerSettings
from splatworker.ply import read_ply_vertices
from splatworker.server import create_app

TOKEN = "s3cret-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
ZIP = {"Content-Type": "application/zip"}
STATUS_FIELDS = {"jobId", "name", "state", "step", "totalSteps", "loss", "psnr", "numGaussians",
                 "elapsedSeconds", "etaSeconds", "previewStep", "message", "kind"}


def make_client(tmp_path, backend=None, token=TOKEN, gpus=None):
    settings = ServerSettings(data_dir=tmp_path / "data", token=token, name="test-worker", mdns=False,
                              gpus=gpus, max_upload_bytes=5 * 1024 * 1024)
    backend = backend or FakeBackend()
    app = create_app(settings, backend=backend, devices=[None])
    return TestClient(app), backend


@pytest.fixture
def client(tmp_path):
    c, backend = make_client(tmp_path)
    with c:
        c.backend = backend
        yield c


def upload(client, blob, headers=None):
    return client.post("/v1/datasets", content=blob, headers={**AUTH, **ZIP, **(headers or {})})


def wait_state(client, job_id, states, timeout=10):
    states = (states,) if isinstance(states, str) else states

    def check():
        s = client.get(f"/v1/jobs/{job_id}", headers=AUTH).json()
        return s if s["state"] in states else None

    return wait_for(check, timeout)


# ------------------------------------------------------------------- auth

def test_auth_required(client):
    r = client.get("/v1/info")
    assert r.status_code == 401
    assert r.json() == {"error": "missing or invalid bearer token"}
    assert client.get("/v1/info", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/v1/info", headers={"Authorization": TOKEN}).status_code == 401
    assert client.get("/v1/jobs").status_code == 401
    assert client.post("/v1/datasets", content=b"x").status_code == 401
    assert client.get("/v1/jobs/j1/result.ply").status_code == 401


def test_no_token_means_open(tmp_path):
    c, _ = make_client(tmp_path, token=None)
    with c:
        info = c.get("/v1/info").json()
        assert info["auth"] is False


def test_info(client):
    r = client.get("/v1/info", headers=AUTH)
    assert r.status_code == 200
    info = r.json()
    assert info["name"] == "test-worker"
    assert info["version"] == 1
    assert isinstance(info["gpus"], list)
    assert info["backend"].startswith("gsplat")
    assert info["busy"] is False and info["queueLength"] == 0
    assert info["capabilities"] == ["train", "process"]
    assert info["models"] == ["superpoint_lightglue"]


# --------------------------------------------------------------- datasets

def test_upload_and_train_job_lifecycle(client, dataset_zip):
    r = upload(client, dataset_zip)
    assert r.status_code == 200, r.text
    dataset_id = r.json()["datasetId"]

    r = client.post("/v1/jobs", headers=AUTH,
                    json={"datasetId": dataset_id, "name": "Kitchen", "config": {"iterations": 5, "extra": 1}})
    assert r.status_code == 200, r.text
    job_id = r.json()["jobId"]

    s = wait_state(client, job_id, "completed")
    assert STATUS_FIELDS <= set(s)
    assert s["kind"] == "train" and s["name"] == "Kitchen"
    assert s["step"] == 5 and s["totalSteps"] == 5
    assert s["numGaussians"] == 10 and s["psnr"] == 25.0
    assert s["previewStep"] == 2
    assert s["message"] == "completed"

    r = client.get(f"/v1/jobs/{job_id}/result.ply", headers=AUTH)
    assert r.status_code == 200 and r.content.startswith(b"ply\n")
    r = client.get(f"/v1/jobs/{job_id}/preview.ply", headers=AUTH)
    assert r.status_code == 200 and r.content.startswith(b"ply\n")
    log = client.get(f"/v1/jobs/{job_id}/log", headers=AUTH)
    assert log.status_code == 200 and log.headers["content-type"].startswith("text/plain")
    assert "fake trainer loading" in log.text

    jobs = client.get("/v1/jobs", headers=AUTH).json()
    assert [j["jobId"] for j in jobs] == [job_id]

    # config passed to the backend contains only explicit, known keys
    ctx = client.backend.contexts[-1]
    assert ctx.spec == {"iterations": 5}
    assert (ctx.dataset_dir / "dataset.json").is_file()

    assert client.delete(f"/v1/jobs/{job_id}", headers=AUTH).status_code == 204
    assert client.get(f"/v1/jobs/{job_id}", headers=AUTH).status_code == 404


def test_upload_nested_folder_zip(client, tmp_path):
    src = make_dataset(tmp_path / "Kitchen")
    blob = zip_dir(src, prefix="Kitchen/")
    r = upload(client, blob)
    assert r.status_code == 200, r.text


def test_upload_rejects_path_traversal(client, tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("dataset.json", "{}")
        zf.writestr("../../escape.txt", "pwned")
    r = upload(client, buf.getvalue())
    assert r.status_code == 400
    assert "traversal" in r.json()["error"]
    assert not (tmp_path / "escape.txt").exists()
    assert not (tmp_path / "data" / "escape.txt").exists()
    # nothing left behind
    assert list((tmp_path / "data" / "datasets").iterdir()) == []
    assert list((tmp_path / "data" / "tmp").iterdir()) == []


def test_upload_rejects_traversal_inside_dataset_json(client, dataset_dir):
    doc = json.loads((dataset_dir / "dataset.json").read_text())
    doc["cameras"][0]["image"] = "../../../etc/passwd"
    (dataset_dir / "dataset.json").write_text(json.dumps(doc))
    r = upload(client, zip_dir(dataset_dir))
    assert r.status_code == 400 and "escapes" in r.json()["error"]


def test_upload_rejects_symlink_and_garbage(client):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        info = zipfile.ZipInfo("images/link.jpg")
        info.external_attr = (0o120777) << 16
        zf.writestr(info, "/etc/passwd")
    r = upload(client, buf.getvalue())
    assert r.status_code == 400 and "symlink" in r.json()["error"]
    assert upload(client, b"definitely not a zip").status_code == 400
    assert upload(client, b"").status_code == 400
    r = client.post("/v1/datasets", content=b"x", headers={**AUTH, "Content-Type": "text/plain"})
    assert r.status_code == 415


def test_upload_size_limit(client):
    big = b"0" * (6 * 1024 * 1024)
    r = upload(client, big)
    assert r.status_code == 413
    assert "error" in r.json()


def test_upload_without_dataset_json_rejected_for_training(client):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("images/a.jpg", b"\xff\xd8\xff")
    r = upload(client, buf.getvalue())
    assert r.status_code == 200
    ds = r.json()["datasetId"]
    r = client.post("/v1/jobs", headers=AUTH, json={"datasetId": ds})
    assert r.status_code == 400 and "dataset.json" in r.json()["error"]


def test_list_and_delete_dataset(client, dataset_zip):
    ds = upload(client, dataset_zip).json()["datasetId"]
    listed = client.get("/v1/datasets", headers=AUTH).json()
    assert [d["datasetId"] for d in listed] == [ds] and listed[0]["cameras"] == 4
    assert client.delete(f"/v1/datasets/{ds}", headers=AUTH).status_code == 204
    assert client.delete(f"/v1/datasets/{ds}", headers=AUTH).status_code == 404
    assert client.delete("/v1/datasets/..", headers=AUTH).status_code in (404, 405)


# ------------------------------------------------------------------- jobs

def test_create_job_errors(client, dataset_zip):
    r = client.post("/v1/jobs", headers=AUTH, json={"datasetId": "nope"})
    assert r.status_code == 404 and "error" in r.json()
    r = client.post("/v1/jobs", headers=AUTH, content=b"not json")
    assert r.status_code == 400
    ds = upload(client, dataset_zip).json()["datasetId"]
    r = client.post("/v1/jobs", headers=AUTH, json={"datasetId": ds, "config": {"strategy": "magic"}})
    assert r.status_code == 400 and "strategy" in r.json()["error"]
    r = client.post("/v1/jobs", headers=AUTH, json={"datasetId": ds, "config": {"iterations": -1}})
    assert r.status_code == 400
    r = client.post("/v1/jobs", headers=AUTH, json={"datasetId": "../../etc"})
    assert r.status_code == 404


def test_unknown_job_and_bad_ids(client):
    assert client.get("/v1/jobs/doesnotexist", headers=AUTH).status_code == 404
    assert client.get("/v1/jobs/..%2F..%2Fetc/log", headers=AUTH).status_code == 404
    assert client.post("/v1/jobs/zzz/cancel", headers=AUTH).status_code == 404
    assert client.delete("/v1/jobs/zzz", headers=AUTH).status_code == 404


def test_status_transitions_and_cancel_running(client, dataset_zip):
    client.backend.mode = "block"
    ds = upload(client, dataset_zip).json()["datasetId"]
    j1 = client.post("/v1/jobs", headers=AUTH, json={"datasetId": ds, "config": {"iterations": 5}}).json()["jobId"]
    j2 = client.post("/v1/jobs", headers=AUTH, json={"datasetId": ds}).json()["jobId"]

    s1 = wait_state(client, j1, "loading")
    assert s1["message"] == "loading dataset"
    # single device slot: second job waits in the queue
    s2 = client.get(f"/v1/jobs/{j2}", headers=AUTH).json()
    assert s2["state"] == "queued" and s2["totalSteps"] == 30000
    info = client.get("/v1/info", headers=AUTH).json()
    assert info["busy"] is True and info["queueLength"] == 1

    # preview / result not available yet
    assert client.get(f"/v1/jobs/{j1}/preview.ply", headers=AUTH).status_code == 404
    assert client.get(f"/v1/jobs/{j1}/result.ply", headers=AUTH).status_code == 404

    # cancel the queued job: immediate
    r = client.post(f"/v1/jobs/{j2}/cancel", headers=AUTH)
    assert r.status_code == 200 and r.json()["state"] == "cancelled"

    # cancel the running job: returns JobStatus, then becomes cancelled
    r = client.post(f"/v1/jobs/{j1}/cancel", headers=AUTH)
    assert r.status_code == 200 and r.json()["jobId"] == j1
    s1 = wait_state(client, j1, "cancelled")
    assert s1["message"] == "cancelled"
    assert client.get(f"/v1/jobs/{j1}/result.ply", headers=AUTH).status_code == 404
    # cancelling a finished job is a no-op
    assert client.post(f"/v1/jobs/{j1}/cancel", headers=AUTH).json()["state"] == "cancelled"
    assert client.get("/v1/info", headers=AUTH).json()["busy"] is False


def test_failed_job_reports_message(client, dataset_zip):
    client.backend.mode = "fail"
    ds = upload(client, dataset_zip).json()["datasetId"]
    j = client.post("/v1/jobs", headers=AUTH, json={"datasetId": ds}).json()["jobId"]
    s = wait_state(client, j, "failed")
    assert "out of memory" in s["message"]
    assert "FAILED" in client.get(f"/v1/jobs/{j}/log", headers=AUTH).text


def test_delete_running_job(client, dataset_zip, tmp_path):
    client.backend.mode = "block"
    ds = upload(client, dataset_zip).json()["datasetId"]
    j = client.post("/v1/jobs", headers=AUTH, json={"datasetId": ds}).json()["jobId"]
    wait_state(client, j, "loading")
    assert client.delete(f"/v1/jobs/{j}", headers=AUTH).status_code == 204
    assert client.get(f"/v1/jobs/{j}", headers=AUTH).status_code == 404
    wait_for(lambda: not (tmp_path / "data" / "jobs" / j).exists())


def test_dataset_in_use_cannot_be_deleted(client, dataset_zip):
    client.backend.mode = "block"
    ds = upload(client, dataset_zip).json()["datasetId"]
    j = client.post("/v1/jobs", headers=AUTH, json={"datasetId": ds}).json()["jobId"]
    wait_state(client, j, "loading")
    assert client.delete(f"/v1/datasets/{ds}", headers=AUTH).status_code == 409
    client.post(f"/v1/jobs/{j}/cancel", headers=AUTH)
    wait_state(client, j, "cancelled")


def test_persistence_across_restart(tmp_path, dataset_zip):
    c, backend = make_client(tmp_path)
    backend.mode = "block"
    with c:
        ds = upload(c, dataset_zip).json()["datasetId"]
        done = c.post("/v1/jobs", headers=AUTH, json={"datasetId": ds, "config": {"iterations": 5}}).json()["jobId"]
        backend.release.set()
        wait_state(c, done, "completed")
        backend.release.clear()
        running = c.post("/v1/jobs", headers=AUTH, json={"datasetId": ds}).json()["jobId"]
        wait_state(c, running, "loading")
        queued = c.post("/v1/jobs", headers=AUTH, json={"datasetId": ds}).json()["jobId"]
        manager = c.app.state.manager
        # simulate a hard crash: freeze the on-disk state as it is now
        import shutil
        snap = tmp_path / "snap"
        shutil.copytree(tmp_path / "data", snap)
    shutil.rmtree(tmp_path / "data")
    shutil.copytree(snap, tmp_path / "data")

    c2, backend2 = make_client(tmp_path)
    with c2:
        assert c2.get(f"/v1/jobs/{done}", headers=AUTH).json()["state"] == "completed"
        assert c2.get(f"/v1/jobs/{done}/result.ply", headers=AUTH).status_code == 200
        s = c2.get(f"/v1/jobs/{running}", headers=AUTH).json()
        assert s["state"] == "failed" and "restart" in s["message"]
        # the queued job is picked up again by the new process
        wait_state(c2, queued, "completed")
    assert manager is not None


# ------------------------------------------------------- processing jobs

def _images_zip():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for i in range(3):
            zf.writestr(f"images/{i:03d}.jpg", bytes([i]) * 100)
        zf.writestr("priors.json", "{}")
    return buf.getvalue()


def test_process_job_with_splatprep(client, fake_splatprep):
    ds = upload(client, _images_zip()).json()["datasetId"]
    r = client.post("/v1/process", headers=AUTH, json={
        "datasetId": ds, "request": {"task": "sfm", "imagesDir": "images", "outputDir": "out",
                                     "features": "superpoint_lightglue", "priorPoses": "priors.json"}})
    assert r.status_code == 200, r.text
    job_id = r.json()["jobId"]
    assert job_id.startswith("p")
    s = wait_state(client, job_id, ("completed", "failed"))
    assert s["state"] == "completed", client.get(f"/v1/jobs/{job_id}/log", headers=AUTH).text
    assert s["kind"] == "process"
    assert s["totalSteps"] == 1000 and s["step"] == 1000

    r = client.get(f"/v1/jobs/{job_id}/output.zip", headers=AUTH)
    assert r.status_code == 200 and r.headers["content-type"] == "application/zip"
    names = sorted(zipfile.ZipFile(io.BytesIO(r.content)).namelist())
    assert names == ["dataset.json", "images/000.jpg", "images/001.jpg", "images/002.jpg"]
    log = client.get(f"/v1/jobs/{job_id}/log", headers=AUTH).text
    assert "[features] 000.jpg" in log
    # output.zip is only for completed processing jobs; result.ply does not exist
    assert client.get(f"/v1/jobs/{job_id}/result.ply", headers=AUTH).status_code == 404
    # uploaded dataset untouched (outputs go into the job's work dir)
    ds_dir = client.app.state.settings.datasets_dir / ds
    assert not any(p.name == "out" for p in ds_dir.rglob("*"))


def test_process_progress_mapping(client, fake_splatprep):
    ds = upload(client, _images_zip()).json()["datasetId"]
    seen = []
    from splatworker import processing

    original = processing.run_processing

    def spy(*a, progress_cb, **kw):
        def cb(**f):
            seen.append(f)
            progress_cb(**f)
        return original(*a, progress_cb=cb, **kw)

    processing.run_processing = spy
    try:
        j = client.post("/v1/process", headers=AUTH,
                        json={"datasetId": ds, "request": {"task": "sfm"}}).json()["jobId"]
        wait_state(client, j, "completed")
    finally:
        processing.run_processing = original
    assert seen[0] == {"message": "features", "totalSteps": 1000, "step": 167}
    assert seen[-1] == {"message": "mapping", "totalSteps": 1000, "step": 1000}


def test_process_model_unavailable_fails_cleanly(client, fake_splatprep):
    ds = upload(client, _images_zip()).json()["datasetId"]
    j = client.post("/v1/process", headers=AUTH, json={
        "datasetId": ds, "request": {"task": "sfm", "features": "unavailable_model"}}).json()["jobId"]
    s = wait_state(client, j, ("completed", "failed"))
    assert s["state"] == "failed" and "model unavailable" in s["message"]
    assert client.get(f"/v1/jobs/{j}/output.zip", headers=AUTH).status_code == 404


def test_process_without_splatprep(client, monkeypatch):
    from splatworker import processing

    monkeypatch.setattr(processing, "splatprep_available", lambda: False)
    ds = upload(client, _images_zip()).json()["datasetId"]
    j = client.post("/v1/process", headers=AUTH,
                    json={"datasetId": ds, "request": {"task": "masks", "maskModel": "sky"}}).json()["jobId"]
    s = wait_state(client, j, ("completed", "failed"))
    assert s["state"] == "failed" and "splatprep is not installed" in s["message"]


@pytest.mark.parametrize("req", [None, {"task": "dance"}, {"task": "sfm", "outputDir": "../../x"},
                                 {"task": "sfm", "imagesDir": "/etc"}, {"task": "sfm", "priorPoses": "../p.json"}])
def test_process_request_validation(client, req):
    ds = upload(client, _images_zip()).json()["datasetId"]
    r = client.post("/v1/process", headers=AUTH, json={"datasetId": ds, "request": req})
    assert r.status_code == 400 and "error" in r.json()


# ------------------------------------------------ real subprocess backend

def test_subprocess_backend_process_job(tmp_path, fake_splatprep):
    """Exercise the real spawn-based backend end to end (no torch needed)."""
    from splatworker.backends import SubprocessBackend

    c, _ = make_client(tmp_path, backend=SubprocessBackend(cancel_grace_seconds=5))
    with c:
        ds = upload(c, _images_zip()).json()["datasetId"]
        j = c.post("/v1/process", headers=AUTH, json={"datasetId": ds, "request": {"task": "sfm"}}).json()["jobId"]
        s = wait_state(c, j, ("completed", "failed"), timeout=60)
        assert s["state"] == "completed", c.get(f"/v1/jobs/{j}/log", headers=AUTH).text
        assert c.get(f"/v1/jobs/{j}/output.zip", headers=AUTH).status_code == 200


def test_subprocess_backend_train_without_torch(tmp_path, dataset_zip):
    """Without torch/gsplat the training subprocess must fail cleanly, not hang."""
    import importlib.util

    if importlib.util.find_spec("torch") is not None:
        pytest.skip("torch installed; this test covers the no-torch path")
    from splatworker.backends import SubprocessBackend

    c, _ = make_client(tmp_path, backend=SubprocessBackend(cancel_grace_seconds=5))
    with c:
        ds = upload(c, dataset_zip).json()["datasetId"]
        j = c.post("/v1/jobs", headers=AUTH, json={"datasetId": ds, "config": {"iterations": 10}}).json()["jobId"]
        s = wait_state(c, j, ("completed", "failed"), timeout=60)
        assert s["state"] == "failed" and "PyTorch is not installed" in s["message"]
        info = c.get("/v1/info", headers=AUTH).json()
        assert "train" not in info["capabilities"]


def test_preview_ply_is_valid_3dgs(client, dataset_zip):
    ds = upload(client, dataset_zip).json()["datasetId"]
    j = client.post("/v1/jobs", headers=AUTH, json={"datasetId": ds}).json()["jobId"]
    wait_state(client, j, "completed")
    data = client.get(f"/v1/jobs/{j}/result.ply", headers=AUTH).content
    path = client.app.state.settings.tmp_dir / "check.ply"
    path.write_bytes(data)
    v = read_ply_vertices(path)
    assert len(v) == 62 and len(v["x"]) == 10


def test_runtime_endpoints(client):
    r = client.get("/v1/runtime", headers=AUTH)
    assert r.status_code == 200 and "installed" in r.json() and r.json()["state"] == "idle"
    assert client.post("/v1/runtime/install", json={"component": "nope"}, headers=AUTH).status_code == 400
    assert client.get("/v1/runtime").status_code == 401   # same auth as every other endpoint
