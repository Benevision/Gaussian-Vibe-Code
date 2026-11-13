"""pip progress parser and the runtime installer (download, commands, verify, cancel, retry)."""

import hashlib
import io
import json
import sys
import tarfile
import threading
import time
from pathlib import Path

import pytest

from splatworker.gui.installer import Installer, RuntimeState, classify_failure, remove_runtime
from splatworker.gui.paths import AppPaths
from splatworker.gui.pip_progress import PipProgressParser, human_bytes, package_from_file
from splatworker.gui.runtime_plan import Plan, Step, build_plan

TORCH_TRANSCRIPT = """\
Looking in indexes: https://download.pytorch.org/whl/cu124
Collecting torch==2.4.1
  Downloading https://download.pytorch.org/whl/cu124/torch-2.4.1%2Bcu124-cp310-cp310-linux_x86_64.whl (797.1 MB)
Progress 0 of 797100000
Progress 200000000 of 797100000
Progress 400000000 of 797100000
Progress 797100000 of 797100000
Collecting filelock (from torch==2.4.1)
  Using cached filelock-3.13.1-py3-none-any.whl.metadata (2.8 kB)
Collecting nvidia-cublas-cu12==12.4.2.65 (from torch==2.4.1)
  Downloading nvidia_cublas_cu12-12.4.2.65-py3-none-manylinux2014_x86_64.whl (363.0 MB)
Progress 0 of 363000000
Progress 363000000 of 363000000
Requirement already satisfied: typing-extensions>=4.8.0 in ./venv/lib/python3.10/site-packages
  Using cached sympy-1.13.1-py3-none-any.whl (6.2 MB)
Installing collected packages: sympy, nvidia-cublas-cu12, filelock, torch
Successfully installed filelock-3.13.1 nvidia-cublas-cu12-12.4.2.65 sympy-1.13.1 torch-2.4.1+cu124
"""


@pytest.fixture
def paths(tmp_path):
    return AppPaths.from_env({"HOME": str(tmp_path), "XDG_CONFIG_HOME": str(tmp_path / "cfg"),
                              "XDG_DATA_HOME": str(tmp_path / "data"), "XDG_CACHE_HOME": str(tmp_path / "cache")})


# --------------------------------------------------------------- parser

def test_pip_parser_full_transcript():
    p = PipProgressParser(expected_bytes=1_200_000_000)
    fractions, messages = [], []
    for line in TORCH_TRANSCRIPT.splitlines():
        st = p.feed(line)
        fractions.append(st.fraction)
        messages.append(st.message)
    assert fractions == sorted(fractions), "progress must never go backwards"
    assert fractions[-1] == 1.0 and p.state.phase == "done"
    assert "Downloading torch (400.0 MB of 797.1 MB)" in messages
    assert any(m.startswith("Installing 4 packages") for m in messages)
    assert any(m.startswith("Using cached sympy") for m in messages)
    assert p.state.bytes_done >= 797_100_000 + 363_000_000
    # halfway through torch, the bar is roughly a third of the download share
    idx = messages.index("Downloading torch (400.0 MB of 797.1 MB)")
    assert 0.25 < fractions[idx] < 0.4


def test_pip_parser_without_raw_progress():
    """Old pip (no Progress lines) still advances per file."""
    p = PipProgressParser(expected_bytes=100_000_000)
    p.feed("Collecting a")
    p.feed("  Downloading a-1.0-py3-none-any.whl (50.0 MB)")
    f1 = p.state.fraction
    p.feed("Collecting b")
    assert p.state.fraction > f1  # a counted as finished when the next package starts
    p.feed("  Downloading b-1.0.tar.gz (10 kB)")
    p.feed("  Building wheel for b (pyproject.toml) ... done")
    assert p.state.phase == "building"
    p.feed("Installing collected packages: b, a")
    assert p.state.fraction == pytest.approx(0.9)
    assert p.finish(True).fraction == 1.0


def test_pip_parser_errors_and_helpers():
    p = PipProgressParser()
    p.feed("ERROR: Could not find a version that satisfies the requirement gsplat==9 (from versions: none)")
    p.feed("ERROR: No matching distribution found for gsplat==9")
    assert p.state.phase == "error" and len(p.state.errors) == 2
    assert package_from_file("https://x/torch-2.4.1%2Bcu124-cp310-cp310-linux_x86_64.whl") == "torch"
    assert package_from_file("nvidia_cublas_cu12-12.4.2.65-py3-none-manylinux2014_x86_64.whl") == "nvidia-cublas-cu12"
    assert package_from_file("filelock-3.13.1-py3-none-any.whl.metadata") == "filelock"
    assert human_bytes(797_100_000) == "797.1 MB" and human_bytes(2_500_000_000) == "2.5 GB"
    assert human_bytes(900) == "900 B"


def test_classify_failure():
    s = Step("torch", "Downloading PyTorch", "", "command")
    assert "disk is full" in classify_failure(["OSError: [Errno 28] No space left on device"], 1, s).title
    net = classify_failure(["WARNING: Retrying ... NewConnectionError(...)", "ERROR: Could not install"], 1, s)
    assert "Network" in net.title and "Retry" in net.hint
    nomatch = classify_failure(["ERROR: No matching distribution found for torch==2.4.1"], 1, s)
    assert "not available" in nomatch.title and nomatch.detail.startswith("No matching")
    assert "memory" in classify_failure(["Killed"], -9, s).hint
    generic = classify_failure(["boom"], 2, s)
    assert generic.title == "Downloading PyTorch failed" and generic.detail == "boom"


# ------------------------------------------------------------- installer

def make_python_tarball(tmp_path) -> Path:
    """A fake python-build-standalone archive: python/bin/python3 -> python3.10."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        data = b"#!/bin/sh\necho fake python\n"
        info = tarfile.TarInfo("python/bin/python3.10")
        info.size, info.mode = len(data), 0o755
        tf.addfile(info, io.BytesIO(data))
        link = tarfile.TarInfo("python/bin/python3")
        link.type, link.linkname = tarfile.SYMTYPE, "python3.10"
        tf.addfile(link)
        lib = b"# stdlib"
        li = tarfile.TarInfo("python/lib/python3.10/os.py")
        li.size = len(lib)
        tf.addfile(li, io.BytesIO(lib))
    path = tmp_path / "cpython-3.10.22+test-install_only.tar.gz"
    path.write_bytes(buf.getvalue())
    return path


def py_step(sid, code, parser="pip", kind="command", weight=1.0, download_mb=10):
    return Step(sid, f"Step {sid}", "", kind, argv=[sys.executable, "-c", code], weight=weight,
                download_mb=download_mb, parser=parser)


FAKE_PIP = (
    "import sys, time\n"
    "print('Collecting torch==2.4.1')\n"
    "print('  Downloading torch-2.4.1+cu124-cp310-cp310-linux_x86_64.whl (10.0 MB)')\n"
    "for i in range(0, 10_000_001, 2_500_000): print(f'Progress {i} of 10000000', flush=True)\n"
    "print('Installing collected packages: torch')\n"
    "print('Successfully installed torch-2.4.1+cu124')\n")

VERIFY_OK = ("import json; print('SPLATWORKER-VERIFY ' + json.dumps({'python': '3.10.22', 'splatworker': True, "
             "'torch': '2.4.1+cu124', 'cudaAvailable': True, 'gsplat': '1.5.3+pt24cu124', "
             "'gsplatPrebuilt': True, 'splatprep': False}))")


def make_plan(tarball, steps, can_train=True):
    sha = hashlib.sha256(tarball.read_bytes()).hexdigest()
    dl = Step("python", "Downloading Python", "", "download", url=tarball.as_uri(), sha256=sha,
              size_bytes=tarball.stat().st_size, weight=2)
    return Plan(profile_id="cu124", can_train=can_train, steps=[dl, *steps], download_mb=1, disk_mb=1,
                with_splatprep=False, with_lightglue=False, splatprep_dir=None)


def test_installer_success(tmp_path, paths):
    tarball = make_python_tarball(tmp_path)
    plan = make_plan(tarball, [py_step("torch", FAKE_PIP, weight=5), py_step("verify", VERIFY_OK, None, "verify")])
    events = []
    inst = Installer(plan, paths, on_event=lambda k, d: events.append((k, d)), environ={"PATH": "/usr/bin:/bin"})
    result = inst.run()
    assert result.ok, result.error
    assert (paths.python_dir / "bin" / "python3").is_symlink()
    assert (paths.python_dir / "lib" / "python3.10" / "os.py").is_file()
    overall = [d["overall"] for k, d in events if k == "progress"]
    assert overall == sorted(overall) and overall[-1] == pytest.approx(1.0)
    msgs = [d["message"] for k, d in events if k == "progress"]
    assert any(m.startswith("Downloading torch (") for m in msgs)  # Progress lines are rate-limited
    assert [d["index"] for k, d in events if k == "step"] == [0, 1, 2]
    assert events[-1][0] == "done" and result.verify["torch"] == "2.4.1+cu124"
    state = RuntimeState.load(paths)
    assert state.profile_id == "cu124" and state.can_train and state.verify["gsplat"].startswith("1.5.3")
    assert "PyTorch 2.4.1+cu124" in state.describe()
    log = paths.install_log.read_text()
    assert "Successfully installed torch" in log and "Progress 2500000" not in log
    # second run: python is not downloaded again
    tarball.unlink()
    assert Installer(plan, paths).run().ok


def test_installer_bad_checksum(tmp_path, paths):
    tarball = make_python_tarball(tmp_path)
    plan = make_plan(tarball, [])
    plan.steps[0].sha256 = "0" * 64
    r = Installer(plan, paths).run()
    assert not r.ok and r.failed_step == 0 and "corrupted" in r.error.title
    assert not paths.python_dir.exists()


def test_installer_failure_then_retry_from_failed_step(tmp_path, paths):
    tarball = make_python_tarball(tmp_path)
    flag = tmp_path / "net-ok"
    flaky = (f"import os, sys\nprint('Collecting torch')\n"
             f"if not os.path.exists({str(flag)!r}):\n"
             "    print('WARNING: Retrying (Retry(total=0)) after connection broken by NewConnectionError')\n"
             "    print('ERROR: Could not install packages due to an OSError: HTTPSConnectionPool max retries exceeded')\n"
             "    sys.exit(1)\n"
             "print('Successfully installed torch-2.4.1')\n")
    plan = make_plan(tarball, [py_step("torch", flaky), py_step("verify", VERIFY_OK, None, "verify")])
    r = Installer(plan, paths).run()
    assert not r.ok and r.failed_step == 1 and "Network" in r.error.title
    assert RuntimeState.load(paths).profile_id is None
    flag.write_text("")
    started = []
    r2 = Installer(plan, paths, on_event=lambda k, d: started.append(d["index"]) if k == "step" else None).run(
        start_at=r.failed_step)
    assert r2.ok and started == [1, 2]
    assert "=== FAILED" in paths.install_log.read_text()  # retry appends to the log


def test_installer_verify_problems(tmp_path, paths):
    tarball = make_python_tarball(tmp_path)
    bad = VERIFY_OK.replace("'cudaAvailable': True", "'cudaAvailable': False")
    r = Installer(make_plan(tarball, [py_step("verify", bad, None, "verify")]), paths).run()
    assert not r.ok and "check found a problem" in r.error.title and "flatpak update" in r.error.detail
    r = Installer(make_plan(tarball, [py_step("verify", "print('nothing')", None, "verify")]), paths).run()
    assert not r.ok and "did not run" in r.error.title


def test_installer_cancel(tmp_path, paths):
    tarball = make_python_tarball(tmp_path)
    slow = "import time\nprint('Collecting torch', flush=True)\ntime.sleep(60)\n"
    plan = make_plan(tarball, [py_step("torch", slow)])
    events = []
    inst = Installer(plan, paths, on_event=lambda k, d: events.append(k))
    result = {}
    t = threading.Thread(target=lambda: result.setdefault("r", inst.run()))
    t0 = time.time()
    t.start()
    deadline = time.time() + 10
    while "log" not in events or not any(e == "progress" for e in events[3:]):
        if time.time() > deadline:
            break
        time.sleep(0.05)
    while inst._proc is None and time.time() < deadline:
        time.sleep(0.02)
    inst.cancel()
    t.join(15)
    assert not t.is_alive() and time.time() - t0 < 15
    assert result["r"].cancelled and events[-1] == "cancelled"


def test_remove_runtime(paths):
    paths.venv_python.parent.mkdir(parents=True)
    paths.venv_python.write_text("")
    paths.python_exe.parent.mkdir(parents=True)
    RuntimeState(profile_id="cpu").save(paths)
    assert RuntimeState.load(paths).is_installed(paths)
    remove_runtime(paths)
    assert not paths.venv_dir.exists() and not paths.python_dir.exists()
    assert not RuntimeState.load(paths).is_installed(paths)


def test_real_plan_commands_are_well_formed(paths):
    plan = build_plan("cu121", paths)
    for s in plan.steps:
        if s.kind == "command":
            assert s.argv[0].startswith(str(paths.data_root)) and s.argv[1] == "-m"
    json.dumps([s.__dict__ for s in plan.steps])  # serialisable for logging


def test_installer_swaps_in_only_after_verify(tmp_path, paths):
    tarball = make_python_tarball(tmp_path)
    live, staging = paths.venv_dir, paths.venv_staging_dir
    rel = paths.venv_python.relative_to(live)          # bin/python, or Scripts\\python.exe on Windows
    (live / rel).parent.mkdir(parents=True)
    (live / rel).write_text("old")
    write_new = (f"import pathlib; p = pathlib.Path({str(staging / rel)!r}); p.parent.mkdir(parents=True, exist_ok=True); "
                 "p.write_text('new')")
    plan = make_plan(tarball, [py_step("torch", write_new, None), py_step("verify", VERIFY_OK, None, "verify")])
    plan.venv_dir = staging
    # a failing verify leaves the working runtime alone
    bad = make_plan(tarball, [py_step("torch", write_new, None), py_step("verify", "print('nothing')", None, "verify")])
    bad.venv_dir = staging
    r = Installer(bad, paths).run()
    assert not r.ok and (live / rel).read_text() == "old"
    assert "has not been changed" in r.error.hint
    r = Installer(plan, paths).run()
    assert r.ok, r.error
    assert (live / rel).read_text() == "new"
    assert not staging.exists() and not live.with_name("venv.old").exists()


def test_installer_preflight_unpublished_gsplat(tmp_path, paths):
    import urllib.request

    tarball = make_python_tarball(tmp_path)
    plan = make_plan(tarball, [py_step("verify", VERIFY_OK, None, "verify")])
    plan.profile_id = "cu128"

    def opener(url, timeout=None):
        if url.startswith("https://"):
            return io.BytesIO(b"<html>no wheels yet</html>")
        return urllib.request.urlopen(url, timeout=timeout)
    events = []
    r = Installer(plan, paths, opener=opener, on_event=lambda k, d: events.append(k)).run()
    assert not r.ok and "not published yet" in r.error.title
    assert "step" not in events and not paths.python_dir.exists()   # nothing downloaded
