"""Server features for the desktop app on the same PC: local-folder datasets, the shared priority queue, census."""
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient

from conftest import FakeBackend, make_dataset, wait_for, write_fake_ply
from splatworker.config import ServerSettings
from splatworker.server import create_app

TOKEN = "tok"
SECRET = "s" * 43
REMOTE = {"Authorization": f"Bearer {TOKEN}"}
AUTH = {**REMOTE, "X-SplatWork-Local": SECRET}    # the desktop app that started the server


def make(tmp_path, local=True, client_host="127.0.0.1", secret=SECRET):
    settings = ServerSettings(data_dir=tmp_path / "data", token=TOKEN, name="w", mdns=False,
                              allow_local_datasets=local, local_secret=secret)
    backend = FakeBackend()
    app = create_app(settings, backend=backend, devices=[None])
    c = TestClient(app, client=(client_host, 50000))
    c.backend = backend
    return c


def wait_state(c, job_id, states, timeout=10):
    states = (states,) if isinstance(states, str) else states
    return wait_for(lambda: (lambda s: s if s["state"] in states else None)(
        c.get(f"/v1/jobs/{job_id}", headers=AUTH).json()), timeout)


def test_local_dataset_trains_in_place(tmp_path):
    ds = make_dataset(tmp_path / "project" / "dataset")
    with make(tmp_path) as c:
        r = c.post("/v1/datasets/local", json={"path": str(ds)}, headers=AUTH)
        assert r.status_code == 200, r.text
        info = r.json()
        assert info["hasDatasetJson"] and info["cameras"] == 4 and info["local"]
        r = c.post("/v1/jobs", json={"datasetId": info["datasetId"], "config": {"iterations": 5}, "priority": 1},
                   headers=AUTH)
        job = r.json()["jobId"]
        s = wait_state(c, job, "completed")
        assert s["origin"] == "local" and s["priority"] == 1
        assert c.backend.contexts[-1].dataset_dir == ds.resolve()
        # deleting the dataset never touches the project folder
        assert c.delete(f"/v1/datasets/{info['datasetId']}", headers=AUTH).status_code == 204
        assert (ds / "dataset.json").is_file()


def test_local_dataset_refused_remotely_or_disabled(tmp_path):
    ds = make_dataset(tmp_path / "p")
    with make(tmp_path / "a", client_host="192.168.0.20") as c:
        assert c.post("/v1/datasets/local", json={"path": str(ds)}, headers=AUTH).status_code == 403
    with make(tmp_path / "b", local=False) as c:
        assert c.post("/v1/datasets/local", json={"path": str(ds)}, headers=AUTH).status_code == 403
        assert "local" not in c.get("/v1/info", headers=AUTH).json()["capabilities"]
    with make(tmp_path / "c") as c:
        assert c.post("/v1/datasets/local", json={"path": "relative"}, headers=AUTH).status_code == 400
        assert c.post("/v1/datasets/local", json={"path": str(tmp_path / "nope")}, headers=AUTH).status_code == 404
        caps = c.get("/v1/info", headers=AUTH).json()["capabilities"]
        assert {"local", "priority", "census"} <= set(caps)


def test_remote_jobs_cannot_jump_the_queue(tmp_path, dataset_zip):
    with make(tmp_path, client_host="192.168.0.20") as c:
        d = c.post("/v1/datasets", content=dataset_zip, headers={**AUTH, "Content-Type": "application/zip"}).json()
        j = c.post("/v1/jobs", json={"datasetId": d["datasetId"], "priority": 50}, headers=AUTH).json()["jobId"]
        s = wait_state(c, j, "completed")
        assert s["priority"] == 0 and s["origin"] == "192.168.0.20"


def test_priority_order_and_prioritize(tmp_path):
    ds = make_dataset(tmp_path / "ds")
    with make(tmp_path) as c:
        c.backend.mode = "block"
        did = c.post("/v1/datasets/local", json={"path": str(ds)}, headers=AUTH).json()["datasetId"]
        first = c.post("/v1/jobs", json={"datasetId": did, "name": "running"}, headers=AUTH).json()["jobId"]
        assert c.backend.started.wait(5)
        a = c.post("/v1/jobs", json={"datasetId": did, "name": "a"}, headers=AUTH).json()["jobId"]
        b = c.post("/v1/jobs", json={"datasetId": did, "name": "b"}, headers=AUTH).json()["jobId"]
        hi = c.post("/v1/jobs", json={"datasetId": did, "name": "hi", "priority": 1}, headers=AUTH).json()["jobId"]
        # b jumps ahead of everything queued
        r = c.post(f"/v1/jobs/{b}/prioritize", headers=AUTH)
        assert r.status_code == 200 and r.json()["priority"] == 2
        assert c.post(f"/v1/jobs/{first}/prioritize", headers=AUTH).status_code == 409   # running
        c.backend.mode = "ok"
        c.backend.release.set()
        for j in (first, a, b, hi):
            wait_state(c, j, "completed")
        order = [ctx.job_dir.name for ctx in c.backend.contexts]
        assert order == [first, b, hi, a]


def test_census_job(tmp_path):
    ds = make_dataset(tmp_path / "ds")
    ply = tmp_path / "scene.ply"
    write_fake_ply(ply, 12)
    with make(tmp_path) as c:
        did = c.post("/v1/datasets/local", json={"path": str(ds)}, headers=AUTH).json()["datasetId"]
        assert c.post("/v1/census", json={"datasetId": did}, headers=AUTH).status_code == 400
        j = c.post("/v1/census", json={"datasetId": did, "plyPath": str(ply)}, headers=AUTH).json()["jobId"]
        assert j.startswith("c")
        s = wait_state(c, j, "completed")
        assert s["kind"] == "census"
        z = np.load(io.BytesIO(c.get(f"/v1/jobs/{j}/census.npz", headers=AUTH).content))
        assert z["peak"].shape == (12,) and z["views"].dtype == np.uint32


def test_census_ply_path_is_local_only(tmp_path, dataset_zip):
    with make(tmp_path, client_host="10.0.0.5") as c:
        d = c.post("/v1/datasets", content=dataset_zip, headers={**AUTH, "Content-Type": "application/zip"}).json()
        r = c.post("/v1/census", json={"datasetId": d["datasetId"], "plyPath": "/etc/passwd"}, headers=AUTH)
        assert r.status_code == 403
        # a remote client may census its own finished training job
        j = c.post("/v1/jobs", json={"datasetId": d["datasetId"]}, headers=AUTH).json()["jobId"]
        wait_state(c, j, "completed")
        k = c.post("/v1/census", json={"datasetId": d["datasetId"], "jobId": j}, headers=AUTH).json()["jobId"]
        assert wait_state(c, k, "completed")["kind"] == "census"


def test_loopback_without_the_secret_is_remote(tmp_path):
    """SSH tunnels / tailscale serve / proxies arrive from 127.0.0.1: only the per-launch secret counts."""
    ds = make_dataset(tmp_path / "p" / "dataset")
    with make(tmp_path) as c:
        bad = {**REMOTE, "X-SplatWork-Local": "x" * 43}
        for h in (REMOTE, bad):
            assert c.post("/v1/datasets/local", json={"path": str(ds)}, headers=h).status_code == 403
        did = c.post("/v1/datasets/local", json={"path": str(ds)}, headers=AUTH).json()["datasetId"]
        # a remote(-looking) client can't see, train on, download, compose or delete the app's folder datasets
        assert c.get("/v1/datasets", headers=REMOTE).json() == []
        listed = c.get("/v1/datasets", headers=AUTH).json()
        assert [d["datasetId"] for d in listed] == [did] and "external" not in listed[0]
        assert c.post("/v1/jobs", json={"datasetId": did}, headers=REMOTE).status_code == 404
        assert c.get(f"/v1/datasets/{did}/archive.zip", headers=REMOTE).status_code == 404
        assert c.post("/v1/datasets/compose", json={"layers": [{"datasetId": did}]},
                      headers=REMOTE).status_code == 404
        assert c.delete(f"/v1/datasets/{did}", headers=REMOTE).status_code == 404
        j = c.post("/v1/jobs", json={"datasetId": did, "priority": 900}, headers=REMOTE)
        assert j.status_code == 404
        j = c.post("/v1/jobs", json={"datasetId": did, "priority": 5000}, headers=AUTH).json()["jobId"]
        assert wait_state(c, j, "completed")["priority"] == 1000
        assert c.post(f"/v1/jobs/{j}/prioritize", headers=REMOTE).status_code == 403


def test_no_secret_means_no_local_endpoints(tmp_path):
    ds = make_dataset(tmp_path / "p")
    with make(tmp_path, secret=None) as c:
        assert "local" not in c.get("/v1/info", headers=AUTH).json()["capabilities"]
        assert c.post("/v1/datasets/local", json={"path": str(ds)}, headers=AUTH).status_code == 403


def test_export_local_only_into_registered_folders(tmp_path, dataset_zip):
    proj = tmp_path / "proj" / "dataset"
    proj.mkdir(parents=True)
    with make(tmp_path) as c:
        d = c.post("/v1/datasets", content=dataset_zip, headers={**AUTH, "Content-Type": "application/zip"}).json()
        did = d["datasetId"]
        out = tmp_path / "elsewhere"
        r = c.post(f"/v1/datasets/{did}/export-local", json={"path": str(out)}, headers=AUTH)
        assert r.status_code == 403 and not out.exists()
        c.post("/v1/datasets/local", json={"path": str(proj)}, headers=AUTH)
        r = c.post(f"/v1/datasets/{did}/export-local", json={"path": str(proj / "sub")}, headers=AUTH)
        assert r.status_code == 200 and (proj / "sub" / "dataset.json").is_file()
        assert c.post(f"/v1/datasets/{did}/export-local", json={"path": str(proj)},
                      headers=REMOTE).status_code == 403
        # broad folders can't be registered
        for broad in ("/", str(tmp_path / "data"), str(tmp_path)):
            assert c.post("/v1/datasets/local", json={"path": broad}, headers=AUTH).status_code == 403, broad
