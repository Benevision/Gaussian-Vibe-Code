"""Server lifecycle controller: runs the real ``python -m splatworker serve`` as a child."""

import os
import socket
import sys
import time
from pathlib import Path

import pytest

from splatworker.gui.api_client import WorkerApiError, WorkerClient
from splatworker.gui.paths import AppPaths
from splatworker.gui.server_process import (FAILED, RUNNING, STOPPED, LogBuffer, RequestLog, ServerProcess,
                                            explain_exit, find_free_port, launch_spec, parse_access_line,
                                            port_problem)
from splatworker.gui.settings import AppSettings, TokenStore

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def paths(tmp_path):
    return AppPaths.from_env({"HOME": str(tmp_path), "XDG_CONFIG_HOME": str(tmp_path / "cfg"),
                              "XDG_DATA_HOME": str(tmp_path / "data"), "XDG_CACHE_HOME": str(tmp_path / "cache")})


def test_parse_access_line():
    r = parse_access_line('INFO:     192.168.1.7:52344 - "POST /v1/datasets HTTP/1.1" 200 OK', now=5.0)
    assert (r.client, r.method, r.path, r.status, r.time) == ("192.168.1.7", "POST", "/v1/datasets", 200, 5.0)
    r6 = parse_access_line('INFO:     ::1:40000 - "GET /v1/info HTTP/1.1" 401 Unauthorized')
    assert r6.client == "::1" and r6.status == 401
    assert parse_access_line("2026-01-01 12:00:00,000 INFO splatworker: dataset uploaded") is None


def test_request_log_tracks_clients():
    log = RequestLog()
    log.add(parse_access_line('INFO: 127.0.0.1:1 - "GET /v1/jobs HTTP/1.1" 200', now=1))  # GUI poll: hidden
    log.add(parse_access_line('INFO: 10.0.0.5:1 - "GET /v1/info HTTP/1.1" 401', now=10))
    log.add(parse_access_line('INFO: 10.0.0.5:2 - "POST /v1/jobs HTTP/1.1" 200', now=20))
    log.add(parse_access_line('INFO: 10.0.0.9:2 - "GET /v1/info HTTP/1.1" 200', now=30))
    clients = log.active_clients(within_seconds=100, now=40)
    assert [c.address for c in clients] == ["10.0.0.9", "10.0.0.5"]
    c5 = clients[1]
    assert c5.requests == 2 and c5.unauthorized == 1 and c5.last_path == "POST /v1/jobs"
    assert [c.address for c in log.active_clients(within_seconds=15, now=40)] == ["10.0.0.9"]
    assert [r.path for r in log.recent_requests(2)] == ["/v1/info", "/v1/jobs"]


def test_log_buffer_since():
    b = LogBuffer(maxlen=3)
    for i in range(5):
        b.append(str(i))
    assert b.tail() == ["2", "3", "4"] and b.total == 5
    assert b.since(4) == ["4"] and b.since(0) == ["2", "3", "4"] and b.since(5) == []


def test_port_problem_and_explain_exit():
    with socket.socket() as s:
        s.bind(("0.0.0.0", 0))
        s.listen(1)
        port = s.getsockname()[1]
        msg = port_problem(port)
        assert msg and "already in use" in msg and "Settings" in msg
        assert "already in use" in explain_exit(["ERROR: [Errno 98] error while attempting to bind: "
                                                 "address already in use"], 1, port)
    assert port_problem(find_free_port()) is None
    assert "Repair" in explain_exit(["ModuleNotFoundError: No module named 'fastapi'"], 1, 1)
    assert "exit code 3" in explain_exit(["boom"], 3, 1)


def test_launch_spec_fallback_and_runtime(paths):
    spec = launch_spec(paths, environ={"PYTHONPATH": "/evil", "HOME": "/h"}, fallback_python="/usr/bin/python3")
    assert spec.python == "/usr/bin/python3" and not spec.runtime_ready and spec.warning
    assert spec.env["PYTHONPATH"].split(os.pathsep)[0] == str(ROOT) and "/evil" not in spec.env["PYTHONPATH"]
    assert spec.env["PYTHONUNBUFFERED"] == "1" and spec.env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"


@pytest.fixture
def server(paths):
    settings = AppSettings(port=find_free_port(), name="gui-test", mdns=False)
    token = TokenStore(paths.token_file).get_or_create()
    spec = launch_spec(paths, fallback_python=sys.executable)
    sp = ServerProcess()
    args = (spec.python, settings.serve_args(paths), spec.env, settings.port)
    sp._test_args = args
    assert sp.start(*args)
    yield sp, settings, token
    sp.stop(timeout=20)


def sp_args(sp):
    return sp._test_args


def test_server_start_api_and_stop(server):
    sp, settings, token = server
    assert sp.wait_until([RUNNING, FAILED], timeout=30), sp.log.tail()
    snap = sp.snapshot()
    assert snap.state == RUNNING, (snap.message, sp.log.tail())
    assert snap.url and str(settings.port) in snap.url and snap.pid
    client = WorkerClient(f"http://127.0.0.1:{settings.port}", token)
    info = client.info()
    assert info["name"] == "gui-test" and info["auth"] is True and info["version"] == 1
    assert client.jobs() == []
    with pytest.raises(WorkerApiError) as e:
        WorkerClient(f"http://127.0.0.1:{settings.port}", "wrong").info()
    assert e.value.status == 401 and "token" in e.value.message
    with pytest.raises(WorkerApiError) as e:
        client.cancel("nosuchjob")
    assert e.value.status == 404
    deadline = time.time() + 5
    while time.time() < deadline and not any(r.method == "POST" for r in sp.requests.recent_requests()):
        time.sleep(0.05)
    posts = [r for r in sp.requests.recent_requests() if r.method == "POST"]
    assert posts and posts[0].path == "/v1/jobs/nosuchjob/cancel" and posts[0].status == 404
    assert any("Uvicorn running" in ln for ln in sp.log.tail())
    sp.stop(timeout=20)
    assert sp.state == STOPPED and sp.snapshot().exit_code is not None
    with pytest.raises(WorkerApiError) as e:
        client.info()
    assert e.value.status is None
    # can be started again (fresh process, same settings)
    assert sp.start(*sp_args(sp))
    assert sp.wait_until([RUNNING, FAILED], timeout=30) and sp.state == RUNNING
    assert client.info()["name"] == "gui-test"


def test_server_port_in_use(paths):
    with socket.socket() as s:
        s.bind(("0.0.0.0", 0))
        s.listen(1)
        port = s.getsockname()[1]
        settings = AppSettings(port=port, mdns=False, require_token=False)
        spec = launch_spec(paths, fallback_python=sys.executable)
        sp = ServerProcess()
        assert sp.start(spec.python, settings.serve_args(paths), spec.env, port) is False
        assert sp.state == FAILED and "already in use" in sp.snapshot().message
        # without the pre-check, uvicorn itself fails and the exit is explained
        assert sp.start(spec.python, settings.serve_args(paths), spec.env, port, check_port=False)
        assert sp.wait_until([FAILED, RUNNING], timeout=30)
        snap = sp.snapshot()
        sp.stop(timeout=10)
        if sys.platform.startswith("linux") or snap.state == FAILED:
            assert snap.state == FAILED and "in use" in snap.message, (snap, sp.log.tail())


def test_server_bad_python(paths):
    sp = ServerProcess()
    assert not sp.start("/nonexistent/python", [], {}, find_free_port())
    assert sp.state == FAILED and "Could not start" in sp.snapshot().message


def test_app_state_lifecycle(paths):
    from splatworker.gui.app_state import AppState
    from splatworker.gui.nvidia import GpuMonitor

    st = AppState(paths=paths, monitor=GpuMonitor(nvml=None), fallback_python=sys.executable)
    st.settings.port = find_free_port()
    st.settings.mdns = False
    st.settings.name = "state-test"
    st.save_settings()
    assert not st.runtime_installed and st.client() is None
    assert st.start_server()
    try:
        assert st.server.wait_until([RUNNING, FAILED], timeout=30) and st.server.state == RUNNING, st.server.log.tail()
        p = st.poll()
        assert p.info["name"] == "state-test" and p.jobs == [] and p.api_error is None
        assert p.gpus == [] and p.gpu_error
        assert not st.needs_restart()
        st.settings.max_upload_gb = 1.0
        assert st.needs_restart()
        old = st.token()
        st.regenerate_token()
        assert st.token() != old
        assert st.restart_server()
        assert st.server.wait_until([RUNNING, FAILED], timeout=30) and st.server.state == RUNNING
        assert not st.needs_restart() and st.poll().info is not None
    finally:
        st.shutdown()
    assert st.server.state == STOPPED
    assert isinstance(st.lan_addresses(), list)
