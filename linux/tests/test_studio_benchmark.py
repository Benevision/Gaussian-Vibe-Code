"""Benchmark: synthetic scene, result store, and a real run against an HTTP server with the fake trainer."""
import json
import socket
import threading
import time

import numpy as np
import pytest

from splatworker.studio import benchmark
from splatworker.studio.benchmark import BenchmarkSpec

SMALL = BenchmarkSpec(width=160, height=120, views=6, splats=3000, steps=40, measure_from=10)


def test_scene_is_a_valid_dataset_and_cached(tmp_path):
    from splatworker.dataset import parse_dataset
    from splatworker.ply import read_point_cloud

    root = benchmark.build_scene(tmp_path / "scene", SMALL)
    ds = parse_dataset(root)
    assert len(ds.cameras) == 6
    xyz, rgb = read_point_cloud(root / "points3d.ply")
    assert xyz.shape == (3000, 3) and rgb is not None and rgb.max() <= 1
    from PIL import Image

    im = np.asarray(Image.open(root / "images" / "000.jpg"))
    assert im.shape == (120, 160, 3) and im.std() > 10          # not a blank photo
    stamp = (root / "images" / "000.jpg").stat().st_mtime_ns
    benchmark.build_scene(root, SMALL)                           # same spec: reused
    assert (root / "images" / "000.jpg").stat().st_mtime_ns == stamp
    benchmark.build_scene(root, BenchmarkSpec(width=160, height=120, views=4, splats=100, steps=5, measure_from=1))
    assert len(json.loads((root / "dataset.json").read_text())["cameras"]) == 4


def test_config_matches_the_mac_recipe():
    c = BenchmarkSpec().training_config()
    assert c["iterations"] == 2000 and c["maxGaussians"] == 1_000_000 and c["maxImageSize"] == 1600
    assert c["densifyStart"] == 200 and c["densifyStop"] == 1500 and c["shDegreeInterval"] == 300
    assert c["evalInterval"] == 0 and c["seed"] == 7
    from splatworker.config import TrainingConfig

    TrainingConfig.from_dict(c)                                   # the worker accepts it


def test_measure():
    s = [(500, 10.0), (1000, 20.0), (1500, 30.0), (2000, 40.0)]
    assert benchmark.measure(s, 1000) == pytest.approx(50.0)
    assert benchmark.measure([(5, 1.0), (15, 2.0)], 1000) == pytest.approx(10.0)    # never reached: all samples
    with pytest.raises(benchmark.BenchmarkError):
        benchmark.measure([(5, 1.0)], 1)


def test_store(tmp_path):
    assert benchmark.load_benchmarks(tmp_path) == {}
    (tmp_path / benchmark.FILE).write_text("{oops")
    assert benchmark.load_benchmarks(tmp_path) == {}
    benchmark.save_benchmark(tmp_path, {"deviceId": "local", "name": "This PC", "trainStepsPerSecond": 40.0,
                                        "splats": 1_000_000, "megapixels": 1.92})
    benchmark.save_benchmark(tmp_path, {"deviceId": "w", "name": "Other", "trainStepsPerSecond": 10.0,
                                        "splats": 1_000_000, "megapixels": 1.92})
    r = benchmark.load_benchmarks(tmp_path)
    assert set(r) == {"local", "w"}
    assert [x["deviceId"] for x in benchmark.sorted_results(tmp_path)] == ["local", "w"]
    assert benchmark.throughput(r["local"]) == 2.0
    assert benchmark.seconds_per_step(r["local"], 1_000_000, 1.92) == pytest.approx(1 / 40)
    assert benchmark.seconds_per_step(r["local"], 2_000_000, 3.84) == pytest.approx(2 / 40)


@pytest.fixture
def server(tmp_path):
    uvicorn = pytest.importorskip("uvicorn")
    from conftest import FakeBackend
    from splatworker.config import ServerSettings
    from splatworker.server import create_app
    from splatworker.studio.client import Client

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    settings = ServerSettings(data_dir=tmp_path / "data", token="t", name="bench-worker", mdns=False,
                              host="127.0.0.1", port=port, allow_local_datasets=True, local_secret="k" * 32)
    backend = FakeBackend()
    backend.steps = 40
    backend.step_delay = 0.01
    app = create_app(settings, backend=backend, devices=[None])
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    th = threading.Thread(target=srv.run, daemon=True)
    th.start()
    for _ in range(200):
        if srv.started:
            break
        time.sleep(0.02)
    yield Client(f"http://127.0.0.1:{port}", "t", local_secret="k" * 32), port
    srv.should_exit = True
    th.join(5)


@pytest.mark.parametrize("local", [True, False])
def test_run_against_a_worker(server, tmp_path, local):
    client, _port = server
    msgs = []
    rec = benchmark.run(client, tmp_path / "scene", SMALL, local=local, poll=0.02,
                        progress=lambda m, f: msgs.append((m, f)), tmp_dir=tmp_path / "tmp")
    assert rec["deviceId"] == "local" and rec["trainStepsPerSecond"] > 0
    assert rec["splats"] == 3000 and rec["megapixels"] == pytest.approx(0.02, abs=0.01)
    assert rec["date"].endswith("Z")
    assert (rec["uploadMBps"] is None) == local and (rec["downloadMBps"] is None) == local
    assert msgs[-1] == ("Done", 1.0)
    assert client.jobs() == []                                      # the job was deleted
    assert not any(d.get("datasetId") for d in client.get("/v1/datasets"))
    assert not list((tmp_path / "tmp").glob("*.zip")) if (tmp_path / "tmp").exists() else True


def test_cancel(server, tmp_path):
    client, _port = server
    with pytest.raises(benchmark.BenchmarkCancelled):
        benchmark.run(client, tmp_path / "scene", SMALL, poll=0.02, cancelled=lambda: True)
    with pytest.raises(benchmark.BenchmarkCancelled):           # cancelled while training
        t0 = time.monotonic()
        benchmark.run(client, tmp_path / "scene", SMALL, poll=0.02,
                      cancelled=lambda: time.monotonic() - t0 > 0.2 and bool(client.jobs()))
    assert client.jobs() == []


def test_cli_quick(server, tmp_path, monkeypatch, capsys):
    from splatworker import cli

    _client, port = server
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    rc = cli.main(["benchmark", "--url", f"http://127.0.0.1:{port}", "--token", "t", "--quick", "--poll", "0.02"])
    out = capsys.readouterr()
    assert rc == 0, out.err
    rec = json.loads(out.out)
    assert rec["deviceId"] == "local" and rec["trainStepsPerSecond"] > 0 and rec["uploadMBps"] is None
    assert not (tmp_path / "cfg" / "splatworker" / benchmark.FILE).exists()     # quick runs aren't saved
    rc = cli.main(["benchmark", "--url", "http://127.0.0.1:9", "--quick"])
    assert rc == 1 and "Is the worker running" in capsys.readouterr().err
