"""platform_compat: the Linux / Windows helpers (runs on both; the Windows branches run on the windows-latest CI)."""

import subprocess
import sys
import time

import pytest

from splatworker import platform_compat as pc
from splatworker.discovery import interface_kind


def test_memory_from_meminfo(tmp_path):
    f = tmp_path / "meminfo"
    f.write_text("MemTotal:       16384000 kB\nMemFree: 1 kB\nMemAvailable:    8192000 kB\n")
    assert pc.memory(str(f)) == {"total": 16384000 * 1024, "available": 8192000 * 1024}


def test_memory_of_this_machine_is_sane(tmp_path):
    m = pc.memory(str(tmp_path / "missing"))
    if sys.platform in ("win32", "linux"):
        assert m.get("total", 0) > 256 * 1024 ** 2
    if "available" in m:
        assert 0 < m["available"] <= m["total"]


def test_link_path_file_and_folder(tmp_path):
    src = tmp_path / "src"
    (src / "images").mkdir(parents=True)
    (src / "images" / "a.jpg").write_bytes(b"jpg")
    (src / "dataset.json").write_text("{}")
    work = tmp_path / "work"
    work.mkdir()
    how_dir = pc.link_path(src / "images", work / "images")
    how_file = pc.link_path(src / "dataset.json", work / "dataset.json")
    assert (work / "images" / "a.jpg").read_bytes() == b"jpg"
    assert (work / "dataset.json").read_text() == "{}"
    if sys.platform == "win32":
        assert how_dir in ("symlink", "junction") and how_file in ("symlink", "hardlink")
    else:
        assert how_dir == how_file == "symlink"
    assert pc.is_link(work / "images")
    assert not pc.is_link(src / "images")
    # resolves to the upload, as zip_output / link_tree expect
    assert (work / "images" / "a.jpg").resolve() == (src / "images" / "a.jpg").resolve()


def test_prepare_workdir_links_upload(tmp_path):
    from splatworker.processing import prepare_workdir

    up = tmp_path / "up"
    (up / "images").mkdir(parents=True)
    (up / "images" / "x.png").write_bytes(b"p")
    prepare_workdir(up, tmp_path / "w")
    prepare_workdir(up, tmp_path / "w")  # idempotent
    assert (tmp_path / "w" / "images" / "x.png").read_bytes() == b"p"


def test_kill_tree_stops_child():
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], **pc.new_group_kwargs())
    try:
        time.sleep(0.3)
        pc.kill_tree(proc, force=True)
        proc.wait(15)
        assert proc.returncode is not None
    finally:
        if proc.poll() is None:
            proc.kill()


def test_default_data_dir():
    if sys.platform == "win32":
        assert pc.default_data_dir({"LOCALAPPDATA": r"C:\Users\u\AppData\Local"}).endswith(r"SplatWork\data")
    else:
        assert pc.default_data_dir({}) == "~/.local/share/splatworker"


@pytest.mark.parametrize("name,kind", [
    ("Ethernet", "ethernet"), ("Ethernet 2", "ethernet"), ("Wi-Fi", "wifi"), ("Wi-Fi 3", "wifi"),
    ("vEthernet (WSL)", "virtual"), ("vEthernet (Default Switch)", "virtual"),
    ("Bluetooth Network Connection", "virtual"), ("Loopback Pseudo-Interface 1", "virtual"),
])
def test_windows_adapter_names(tmp_path, name, kind):
    assert interface_kind(name, sys_net=str(tmp_path)) == kind


def test_documents_dir_only_on_windows():
    d = pc.documents_dir()
    if sys.platform == "win32":
        assert d is not None and d.name
    else:
        assert d is None


def test_app_paths_windows_layout():
    from pathlib import Path

    from splatworker.gui.paths import AppPaths

    p = AppPaths.from_env({"APPDATA": r"C:\U\AppData\Roaming", "LOCALAPPDATA": r"C:\U\AppData\Local",
                           "USERPROFILE": r"C:\U"}, windows=True)
    assert p.config_dir == Path(r"C:\U\AppData\Roaming") / "SplatWork"
    assert p.data_root == Path(r"C:\U\AppData\Local") / "SplatWork"
    assert p.token_file.parent == p.config_dir


@pytest.mark.skipif(sys.platform != "win32", reason="DXGI is Windows-only")
def test_dxgi_adapters_runs():
    from splatworker.gpu_vendors import scan

    adapters = scan()   # the CI runner has only Microsoft's software adapter, which is skipped
    for a in adapters:
        assert a.vendor in ("nvidia", "amd", "intel", "other") and a.name


def test_windows_run_key_with_fake_registry(monkeypatch):
    from splatworker.gui import autostart

    store = {}

    class Key:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class FakeWinreg:
        HKEY_CURRENT_USER, REG_SZ = 1, 1

        def CreateKey(self, root, path):
            assert path == autostart.WINDOWS_RUN_KEY
            return Key()

        def SetValueEx(self, k, name, _r, _t, value):
            store[name] = value

        def DeleteValue(self, k, name):
            if name not in store:
                raise FileNotFoundError(name)
            del store[name]

    monkeypatch.setenv("SPLATWORK_LAUNCHER", r"C:\Program Files\SplatWork\SplatWork.exe")
    autostart.set_windows_run_key(True, ["python", "-m", "splatworker.gui.launcher", "--background"], FakeWinreg())
    assert store["SplatWork"] == r'"C:\Program Files\SplatWork\SplatWork.exe" --background'
    autostart.set_windows_run_key(False, [], FakeWinreg())
    autostart.set_windows_run_key(False, [], FakeWinreg())   # already off
    assert store == {}


def test_windows_python_runtime_and_profiles(monkeypatch):
    from splatworker.gui import runtime_plan as RP

    rt = RP.python_runtime(windows=True)
    assert "x86_64-pc-windows-msvc" in rt.url and len(rt.sha256) == 64 and rt.version == RP.PYTHON_RUNTIME.version
    monkeypatch.setattr(RP.sys, "platform", "win32")
    assert "cu128" in RP.unpublished_profiles(opener=lambda *a, **k: (_ for _ in ()).throw(OSError("offline")))
