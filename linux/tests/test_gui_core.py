"""Tests for the GTK-free parts of the desktop app: settings, NVML wrapper, setup planner,
formatting, autostart and launcher helpers."""

import json
import os
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from splatworker.cli import build_parser
from splatworker.gui import autostart, formatting, qr
from splatworker.gui.nvidia import (DriverInfo, GpuInfo, GpuMonitor, detect, flatpak_gl_extension_name,
                                    kernel_driver_version, parse_compute_cap, parse_cuda_version,
                                    parse_smi_header, parse_smi_query)
from splatworker.gui.paths import AppPaths, in_flatpak
from splatworker.gui.runtime_plan import (CPU_PROFILE_ID, GPU_PROFILES, PYTHON_RUNTIME, TRAIN_EXTRA_DEPS,
                                          WORKER_DEPS, build_plan, choose_profile, verify_problems)
from splatworker.gui.settings import AppSettings, TokenStore

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def paths(tmp_path):
    return AppPaths.from_env({"HOME": str(tmp_path), "XDG_CONFIG_HOME": str(tmp_path / "cfg"),
                              "XDG_DATA_HOME": str(tmp_path / "data"), "XDG_CACHE_HOME": str(tmp_path / "cache")})


# ----------------------------------------------------------------- paths

def test_paths_layout(tmp_path, paths):
    assert paths.settings_file == tmp_path / "cfg" / "splatworker" / "gui-settings.json"
    assert paths.venv_dir == tmp_path / "data" / "splatworker" / "venv"
    assert paths.default_worker_data_dir == tmp_path / "data" / "splatworker" / "data"
    assert paths.pip_cache == tmp_path / "cache" / "splatworker" / "pip"
    # relative XDG values are ignored per the spec
    p = AppPaths.from_env({"HOME": str(tmp_path), "XDG_DATA_HOME": "relative"})
    assert p.data_root == tmp_path / ".local/share" / "splatworker"
    assert in_flatpak({"FLATPAK_ID": "com.example.SplatWorker"}, root=tmp_path)
    assert not in_flatpak({}, root=tmp_path)
    (tmp_path / ".flatpak-info").write_text("[Application]\n")
    assert in_flatpak({}, root=tmp_path)


# -------------------------------------------------------------- settings

def test_settings_roundtrip_and_lenient_load(tmp_path):
    f = tmp_path / "s.json"
    s = AppSettings(port=9000, name="Render Box", gpus=[1, 0], max_upload_gb=5, mdns=False)
    s.save(f)
    assert json.loads(f.read_text())["version"] == 1
    assert AppSettings.load(f) == s
    f.write_text(json.dumps({"port": "nope", "name": "a.b", "gpus": [], "max_upload_gb": 3.5,
                             "unknown": 1, "autostart": True, "advertise_ip": "fe80::1"}))
    s2 = AppSettings.load(f)
    assert s2.port == 7878 and s2.name == "" and s2.gpus is None and s2.advertise_ip == ""
    assert s2.max_upload_gb == 3.5 and s2.autostart is True
    f.write_text("{broken")
    assert AppSettings.load(f) == AppSettings()
    assert AppSettings.load(tmp_path / "missing.json") == AppSettings()


def test_settings_validation_messages():
    errs = AppSettings(port=80, gpus=[0, 0], max_upload_gb=0, advertise_ip="1.2.3",
                       require_token="yes").field_errors()
    assert set(errs) == {"port", "gpus", "max_upload_gb", "advertise_ip", "require_token"}
    assert "1024" in errs["port"]
    assert AppSettings().field_errors() == {}
    assert AppSettings(port=True).field_errors(only="port")


def test_serve_args_are_accepted_by_the_real_cli(paths):
    s = AppSettings(port=7999, name="Box", gpus=[0, 2], mdns=False, advertise_ip="192.168.1.20",
                    max_upload_gb=2.5)
    args = build_parser().parse_args(["serve", *s.serve_args(paths)])
    assert args.port == 7999 and args.name == "Box" and args.gpus == [0, 2] and args.no_mdns
    assert args.advertise_ip == ["192.168.1.20"] and args.max_upload_gb == 2.5
    assert args.token_file == str(paths.token_file) and args.token is None
    assert args.data_dir == str(paths.default_worker_data_dir)
    s2 = AppSettings(require_token=False, data_dir="/srv/splats")
    a2 = build_parser().parse_args(["serve", *s2.serve_args(paths)])
    assert a2.token_file is None or a2.token_file == os.environ.get("SPLATWORKER_TOKEN_FILE")
    assert a2.gpus is None and a2.data_dir == "/srv/splats" and not a2.no_mdns
    assert "--token" not in s.serve_args(paths)  # never on the command line


def test_token_store(tmp_path):
    ts = TokenStore(tmp_path / "cfg" / "token")
    assert ts.read() is None
    t1 = ts.get_or_create()
    assert len(t1) >= 32 and ts.read() == t1 and ts.get_or_create() == t1
    assert stat.S_IMODE(os.stat(ts.path).st_mode) == 0o600
    t2 = ts.regenerate()
    assert t2 != t1 and ts.read() == t2
    ts.path.write_text("  abc \n\n")
    assert ts.read() == "abc"


# ---------------------------------------------------------------- nvidia

class FakeNVMLError(Exception):
    pass


def fake_nvml(gpus=None, cuda=12040, driver="550.54.14", fail_init=False, fail_temp=False):
    gpus = gpus if gpus is not None else [("NVIDIA GeForce RTX 4090", 24564, (8, 9))]
    used = {"v": 1024}
    calls = {"init": 0, "shutdown": 0}

    def init():
        calls["init"] += 1
        if fail_init:
            raise FakeNVMLError("Driver Not Loaded")

    def temp(h, sensor):
        if fail_temp:
            raise FakeNVMLError("Not Supported")
        return 55 + h

    mod = SimpleNamespace(
        NVML_TEMPERATURE_GPU=0, calls=calls, used=used,
        nvmlInit=init,
        nvmlShutdown=lambda: calls.__setitem__("shutdown", calls["shutdown"] + 1),
        nvmlSystemGetDriverVersion=lambda: driver.encode(),
        nvmlSystemGetCudaDriverVersion_v2=lambda: cuda,
        nvmlDeviceGetCount=lambda: len(gpus),
        nvmlDeviceGetHandleByIndex=lambda i: i,
        nvmlDeviceGetName=lambda h: gpus[h][0].encode(),
        nvmlDeviceGetUUID=lambda h: f"GPU-{h}",
        nvmlDeviceGetMemoryInfo=lambda h: SimpleNamespace(total=gpus[h][1] * 1024 * 1024,
                                                          used=used["v"] * 1024 * 1024,
                                                          free=(gpus[h][1] - used["v"]) * 1024 * 1024),
        nvmlDeviceGetCudaComputeCapability=lambda h: gpus[h][2],
        nvmlDeviceGetTemperature=temp,
        nvmlDeviceGetUtilizationRates=lambda h: SimpleNamespace(gpu=87, memory=40),
        nvmlDeviceGetPowerUsage=lambda h: 312500,
    )
    return mod


def no_smi(*a, **k):
    raise AssertionError("nvidia-smi should not be called")


def test_parsers():
    assert parse_cuda_version(12040) == (12, 4)
    assert parse_cuda_version(11080) == (11, 8)
    assert parse_cuda_version("12.6") == (12, 6)
    assert parse_cuda_version(0) is None and parse_cuda_version(None) is None
    assert parse_compute_cap("8.9") == (8, 9) and parse_compute_cap((12, 0)) == (12, 0)
    assert parse_compute_cap("[N/A]") is None
    gpus = parse_smi_query("0, NVIDIA GeForce RTX 3090, 24576, 8.6\n1, Tesla T4, 15360, 7.5\n\n")
    assert [(g.index, g.name, g.memory_total_mb, g.compute_capability) for g in gpus] == [
        (0, "NVIDIA GeForce RTX 3090", 24576, (8, 6)), (1, "Tesla T4", 15360, (7, 5))]
    header = "| NVIDIA-SMI 550.54.14   Driver Version: 550.54.14   CUDA Version: 12.4     |"
    assert parse_smi_header(header) == ("550.54.14", (12, 4))
    assert flatpak_gl_extension_name("550.54.14") == "org.freedesktop.Platform.GL.nvidia-550-54-14"


def test_kernel_driver_version(tmp_path):
    f = tmp_path / "version"
    f.write_text("NVRM version: NVIDIA UNIX x86_64 Kernel Module  550.54.14  Thu Feb 22 01:44:30 UTC 2024\n"
                 "GCC version:  gcc version 13.2.1\n")
    assert kernel_driver_version(f) == "550.54.14"
    f.write_text("NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  570.86.16  Release Build\n")
    assert kernel_driver_version(f) == "570.86.16"
    assert kernel_driver_version(tmp_path / "missing") is None


def test_detect_with_nvml(tmp_path):
    nv = fake_nvml()
    info = detect(nvml=nv, run=no_smi, smi=None, proc_file=tmp_path / "x", dev_glob=str(tmp_path / "nv*"),
                  flatpak=True)
    assert info.source == "nvml" and info.driver_version == "550.54.14" and info.cuda_version == (12, 4)
    g = info.gpus[0]
    assert (g.name, g.memory_total_mb, g.compute_capability, g.uuid) == ("NVIDIA GeForce RTX 4090", 24564, (8, 9), "GPU-0")
    assert nv.calls["shutdown"] == 1 and info.problem() is None


def test_detect_falls_back_to_nvidia_smi(tmp_path):
    def run(argv, **kw):
        if len(argv) == 1:
            return SimpleNamespace(returncode=0, stdout="Driver Version: 535.183.01   CUDA Version: 12.2", stderr="")
        return SimpleNamespace(returncode=0, stdout="0, NVIDIA RTX A4000, 16376, 8.6\n", stderr="")

    info = detect(nvml=fake_nvml(fail_init=True), run=run, smi="/usr/bin/nvidia-smi",
                  proc_file=tmp_path / "x", dev_glob=str(tmp_path / "nv*"), flatpak=False)
    assert info.source == "nvidia-smi" and info.cuda_version == (12, 2) and info.driver_version == "535.183.01"
    assert info.gpus[0].name == "NVIDIA RTX A4000"


def test_detect_problems(tmp_path):
    # nothing at all
    info = detect(nvml=None, run=no_smi, smi=None, proc_file=tmp_path / "x", dev_glob=str(tmp_path / "nv*"),
                  flatpak=True)
    title, text = info.problem()
    assert "No NVIDIA GPU" in title and "processing" in text
    # kernel driver loaded, flatpak extension missing
    (tmp_path / "version").write_text("NVRM version: NVIDIA UNIX x86_64 Kernel Module  550.54.14  x\n")
    (tmp_path / "nvidia0").write_text("")
    info = detect(nvml=fake_nvml(fail_init=True), run=no_smi, smi=None, proc_file=tmp_path / "version",
                  dev_glob=str(tmp_path / "nvidia[0-9]*"), flatpak=True)
    title, text = info.problem()
    assert "sandbox" in title and "nvidia-550-54-14" in text and "flatpak update" in text
    assert info.error and "Driver Not Loaded" in info.error
    info.in_flatpak = False
    assert "not usable" in info.problem()[0]


def test_gpu_monitor(tmp_path):
    nv = fake_nvml(gpus=[("A", 8192, (8, 6)), ("B", 16384, (8, 9))])
    mon = GpuMonitor(nvml=nv)
    stats = mon.sample()
    assert [s.name for s in stats] == ["A", "B"]
    s = stats[1]
    assert (s.memory_total_mb, s.memory_used_mb, s.memory_free_mb) == (16384, 1024, 15360)
    assert s.temperature_c == 56 and s.utilization_pct == 87 and s.power_w == 312.5
    assert abs(s.memory_fraction - 1024 / 16384) < 1e-9
    nv.used["v"] = 4096
    assert mon.sample()[0].memory_used_mb == 4096
    assert nv.calls["init"] == 1  # kept initialised between samples
    mon.close()
    assert nv.calls["shutdown"] == 1
    # unsupported sensors degrade to None
    assert GpuMonitor(nvml=fake_nvml(fail_temp=True)).sample()[0].temperature_c is None
    # init failure: no stats, readable error, never raises
    bad = GpuMonitor(nvml=fake_nvml(fail_init=True))
    assert bad.sample() == [] and "Driver Not Loaded" in bad.error
    none = GpuMonitor(nvml=None)
    assert not none.available and none.sample() == [] and none.error


# --------------------------------------------------------------- planner

def driver(cuda, cc=(8, 6), mem=24576, name="NVIDIA GeForce RTX 3090", n=1):
    return DriverInfo(driver_version="999.1", cuda_version=cuda, source="nvml",
                      gpus=[GpuInfo(i, name, mem, cc) for i in range(n)])


@pytest.mark.parametrize("cuda,expected", [
    ((13, 0), "cu128"), ((12, 8), "cu128"), ((12, 4), "cu124"), ((12, 3), "cu121"), ((12, 1), "cu121"),
    ((12, 0), "cu118"), ((11, 8), "cu118"),
])
def test_choose_profile_by_driver_cuda(cuda, expected):
    c = choose_profile(driver(cuda))
    assert c.profile_id == expected and c.can_train and c.blocking is None


def test_choose_profile_edge_cases():
    old = choose_profile(driver((11, 7)))
    assert old.profile_id == CPU_PROFILE_ID and not old.can_train and "driver" in old.blocking.lower()
    blackwell = choose_profile(driver((12, 8), cc=(12, 0), name="NVIDIA GeForce RTX 5090"))
    assert blackwell.profile_id == "cu128" and blackwell.can_train and blackwell.blocking is None
    old_blackwell_driver = choose_profile(driver((12, 4), cc=(12, 0), name="NVIDIA GeForce RTX 5090"))
    assert old_blackwell_driver.profile_id == CPU_PROFILE_ID and "570" in old_blackwell_driver.blocking
    pascal = choose_profile(driver((12, 8), cc=(6, 1), name="NVIDIA GeForce GTX 1080"))
    assert pascal.profile_id == "cu124"     # the CUDA 12.8 build has no Pascal kernels
    nogpu = choose_profile(DriverInfo())
    assert nogpu.profile_id == CPU_PROFILE_ID and not nogpu.can_train and nogpu.blocking
    unknown = choose_profile(driver(None))
    assert unknown.profile_id == "cu124" and any("CUDA version" in w for w in unknown.warnings)
    small = choose_profile(driver((12, 4), mem=6144))
    assert any("8 GB" in w for w in small.warnings)
    forced = choose_profile(driver((12, 1)), override="cu124")
    assert forced.profile_id == "cu124" and any("needs a driver" in w for w in forced.warnings)
    assert choose_profile(driver((12, 4)), override="cpu").profile_id == "cpu"
    with pytest.raises(ValueError):
        choose_profile(driver((12, 4)), override="cu999")
    mixed = DriverInfo(cuda_version=(12, 8), gpus=[GpuInfo(0, "RTX 5090", 32000, (12, 0)),
                                                   GpuInfo(1, "RTX 4090", 24000, (8, 9))])
    m = choose_profile(mixed)
    assert m.profile_id == "cu128" and not m.warnings


def test_build_plan_gpu(paths, tmp_path):
    plan = build_plan(choose_profile(driver((12, 6))), paths, with_splatprep=True, with_lightglue=True,
                      splatprep_dir=tmp_path)
    ids = [s.id for s in plan.steps]
    assert ids == ["python", "venv", "pip", "torch", "gsplat", "worker", "splatprep", "torchvision",
                   "lightglue", "processing", "sam2", "verify"]
    proc = [s for s in plan.steps if s.id == "processing"][0].argv
    # torch / torchvision stay pinned to the profile in every later step (gsplat's kernels need that build)
    assert "torch==2.4.1" in proc and "torchvision==0.19.1" in proc and any(a.startswith("pycolmap") for a in proc)
    sam = [s for s in plan.steps if s.id == "sam2"][0].argv
    assert "--no-deps" in sam and any(a.startswith("SAM-2 @ https://") for a in sam)
    lg = [s for s in plan.steps if s.id == "lightglue"][0].argv
    assert "torch==2.4.1" in lg
    by = {s.id: s for s in plan.steps}
    assert by["python"].url == PYTHON_RUNTIME.url and len(by["python"].sha256) == 64
    assert by["venv"].argv[:3] == [str(paths.python_exe), "-m", "venv"]
    torch = by["torch"].argv
    assert torch[0] == str(paths.venv_python) and "torch==2.4.1" in torch
    assert torch[torch.index("--index-url") + 1] == "https://download.pytorch.org/whl/cu124"
    assert "--progress-bar" in torch and torch[torch.index("--progress-bar") + 1] == "raw"
    g = by["gsplat"].argv
    assert "gsplat==1.5.3+pt24cu124" in g and g[g.index("--index-url") + 1] == "https://docs.gsplat.studio/whl/pt24cu124/"
    assert "--only-binary" in g
    assert all(s.parser == "pip" for s in plan.steps if s.id not in ("python", "venv", "verify"))
    assert "torchvision==0.19.1" in by["torchvision"].argv
    assert plan.can_train and plan.download_mb > 2500 and plan.disk_mb > 5000
    assert sum(s.weight for s in plan.steps) > 0


def test_build_plan_cpu_and_without_splatprep(paths, tmp_path):
    plan = build_plan("cpu", paths, with_splatprep=True, splatprep_dir=tmp_path)
    assert [s.id for s in plan.steps] == ["python", "venv", "pip", "worker", "splatprep", "verify"]
    assert not plan.can_train and plan.download_mb < 400
    p2 = build_plan("cu118", paths, with_splatprep=True, with_lightglue=True, splatprep_dir=None)
    assert not p2.with_splatprep and not p2.with_lightglue  # nothing to install it from
    cpu_lg = build_plan("cpu", paths, with_splatprep=True, with_lightglue=True, splatprep_dir=tmp_path)
    tv = next(s for s in cpu_lg.steps if s.id == "torchvision").argv
    assert tv[tv.index("--index-url") + 1].endswith("/cpu")
    with pytest.raises(ValueError):
        build_plan("cu999", paths)


def test_free_space_problem(paths):
    plan = build_plan("cu124", paths)
    assert plan.free_space_problem(paths.data_root, margin_mb=0) is None or "disk space" in plan.free_space_problem(paths.data_root, margin_mb=0)
    plan.disk_mb = 10 ** 12
    msg = plan.free_space_problem(paths.data_root)
    assert "Not enough disk space" in msg


def test_worker_deps_match_pyproject():
    try:
        import tomllib
    except ImportError:  # pragma: no cover
        pytest.skip("tomllib needs Python 3.11")
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert WORKER_DEPS == data["project"]["dependencies"]
    train = [d for d in data["project"]["optional-dependencies"]["train"]
             if not d.startswith(("torch", "gsplat"))]
    assert TRAIN_EXTRA_DEPS == train


def test_gpu_profiles_are_consistent():
    for p in GPU_PROFILES:
        assert p.gsplat_index == f"pt{p.torch.replace('.', '')[:2]}{p.id}"
    cudas = [p.min_driver_cuda for p in GPU_PROFILES]
    assert cudas == sorted(cudas, reverse=True)


def test_verify_problems(paths, tmp_path):
    plan = build_plan("cu124", paths, with_splatprep=True, splatprep_dir=tmp_path)
    good = {"splatworker": True, "torch": "2.4.1+cu124", "cudaAvailable": True, "gsplat": "1.5.3+pt24cu124",
            "gsplatPrebuilt": True, "splatprep": True}
    assert verify_problems(good, plan) == []
    probs = verify_problems({**good, "cudaAvailable": False, "gsplatPrebuilt": False}, plan)
    assert any("flatpak update" in p for p in probs) and any("prebuilt" in p for p in probs)
    cpu = build_plan("cpu", paths)
    assert verify_problems({"splatworker": True, "torch": None}, cpu) == []


# ------------------------------------------------------------ formatting

def test_job_view_training():
    v = formatting.job_view({"jobId": "j1", "kind": "train", "name": "Kitchen", "state": "training",
                             "step": 1200, "totalSteps": 30000, "psnr": 24.31, "numGaussians": 812345,
                             "etaSeconds": 1540.0, "message": "training", "gpu": "RTX 4090"})
    assert v.title == "Kitchen" and abs(v.fraction - 0.04) < 1e-9 and v.can_cancel and v.export is None
    assert "step 1,200 / 30,000" in v.detail and "PSNR 24.3 dB" in v.detail and "812 k splats" in v.detail
    assert "about 25 min left" in v.detail and "RTX 4090" in v.detail


def test_job_view_other_states():
    done = formatting.job_view({"jobId": "j2", "state": "completed", "step": 7000, "totalSteps": 7000,
                                "elapsedSeconds": 3725, "kind": "train"})
    assert done.fraction == 1.0 and done.export == "result.ply" and not done.can_cancel
    assert "took 1 h 02 min" in done.detail
    failed = formatting.job_view({"jobId": "j3", "state": "failed", "message": "CUDA out of memory"})
    assert failed.is_error and "CUDA out of memory" in failed.detail
    proc = formatting.job_view({"jobId": "p1", "kind": "process", "state": "loading", "step": 420,
                                "totalSteps": 1000, "message": "features"})
    assert "42 %" in proc.detail and "features" in proc.detail
    assert formatting.job_view({"jobId": "p2", "kind": "process", "state": "completed"}).export == "output.zip"
    cancelling = formatting.job_view({"jobId": "j4", "state": "training", "message": "cancelling"})
    assert not cancelling.can_cancel


def test_formatting_helpers():
    assert formatting.duration(None) == "—" and formatting.duration(42) == "42 s"
    assert formatting.duration(125) == "2 min 05 s" and formatting.duration(7200) == "2 h 00 min"
    assert formatting.count(1_500_000) == "1.50 M" and formatting.count(999) == "999"
    assert formatting.megabytes(24564) == "24.0 GB" and formatting.megabytes(512) == "512 MB"
    assert formatting.ago(100, now=101) == "just now" and formatting.ago(0, now=600) == "10 min ago"
    assert formatting.capabilities_text({"capabilities": ["train", "process"]}) == "Training, dataset processing"
    assert "not installed" in formatting.capabilities_text({"capabilities": []})
    uri = formatting.connection_uri("192.168.1.20", 7878, "Render Box", "abc_-")
    assert uri == "splatworker://192.168.1.20:7878/?name=Render%20Box&token=abc_-"
    assert formatting.addresses_text(["10.0.0.2"], 7878) == ["http://10.0.0.2:7878"]


# ------------------------------------------------------------- autostart

def test_background_portal_helpers():
    assert autostart.request_path(":1.42", "tok") == "/org/freedesktop/portal/desktop/request/1_42/tok"
    opts = autostart.background_options(True, "tok")
    assert opts["autostart"] == ("b", True) and opts["commandline"][0] == "as"
    assert opts["commandline"][1][-1] == "--background" and opts["handle_token"] == ("s", "tok")
    assert autostart.parse_background_response(0, {"background": True, "autostart": True})[:2] == (True, True)
    assert autostart.parse_background_response(1, {})[0] is False
    assert not autostart.parse_background_response(0, {"background": False})[0]


def test_autostart_file(tmp_path):
    p = autostart.set_autostart_file(True, ["splatworker-app", "--background"], config_home=tmp_path)
    assert p == tmp_path / "autostart" / "com.example.SplatWorker.desktop"
    text = p.read_text()
    assert "Exec=splatworker-app --background" in text and "Icon=com.example.SplatWorker" in text
    assert autostart.autostart_file_enabled(tmp_path)
    autostart.set_autostart_file(False, [], config_home=tmp_path)
    assert not p.exists() and not autostart.autostart_file_enabled(tmp_path)


# -------------------------------------------------------------------- qr

def test_qr_render_rgb():
    data, w, h = qr.render_rgb([[True, False], [False, True]], scale=2, border=1)
    assert (w, h) == (8, 8) and len(data) == 8 * 8 * 3
    px = lambda x, y: data[(y * w + x) * 3]  # noqa: E731
    assert px(0, 0) == 255 and px(2, 2) == 0 and px(4, 2) == 255 and px(4, 4) == 0
    m = qr.qr_matrix("splatworker://1.2.3.4:7878/")
    if m is not None:  # qrcode installed
        assert len(m) >= 21 and len(m) == len(m[0])


# -------------------------------------------------------------- launcher

def test_headless_command(paths, tmp_path):
    from splatworker.gui.launcher import headless_command

    AppSettings(port=7900, name="Box", mdns=False).save(paths.settings_file)
    exe, argv, env, warning = headless_command(["--port", "9100"], paths=paths, environ={"HOME": str(tmp_path)},
                                               fallback_python="/usr/bin/python3")
    assert exe == "/usr/bin/python3" and warning and "GPU runtime" in warning
    assert argv[:4] == ["/usr/bin/python3", "-m", "splatworker", "serve"]
    args = build_parser().parse_args(argv[3:])
    assert args.port == 9100 and args.name == "Box" and args.no_mdns  # CLI flag overrides settings
    assert TokenStore(paths.token_file).read()  # token created on first headless start
    assert str(ROOT) in env["PYTHONPATH"].split(":")
    exe, argv, env, _ = headless_command(["info"], paths=paths, environ={}, fallback_python="py")
    assert argv == ["py", "-m", "splatworker", "info"]
    # with an installed runtime the venv python is used
    paths.venv_python.parent.mkdir(parents=True)
    paths.venv_python.write_text("")
    paths.runtime_state_file.write_text(json.dumps({"version": 1, "profile": "cu124", "canTrain": True}))
    exe, argv, env, warning = headless_command([], paths=paths, environ={})
    assert exe == str(paths.venv_python) and warning is None


def test_launcher_help_and_version(capsys):
    from splatworker.gui.launcher import main

    assert main(["--help"]) == 0 and "--headless" in capsys.readouterr().out
    assert main(["--version"]) == 0 and "com.example.SplatWorker" in capsys.readouterr().out


def test_gui_core_does_not_import_gtk():
    import subprocess

    code = ("import sys; import splatworker.gui.launcher, splatworker.gui.server_process, "
            "splatworker.gui.installer, splatworker.gui.nvidia; "
            "assert 'gi' not in sys.modules, 'gi imported'; print('ok')")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_build_plan_gsplat16_uses_own_wheels(paths, tmp_path):
    plan = build_plan(choose_profile(driver((12, 8))), paths, with_splatprep=True, splatprep_dir=tmp_path)
    g = [s for s in plan.steps if s.id == "gsplat"][0].argv
    assert "gsplat==1.6.0+pt27cu128" in g and "--find-links" in g and "torch==2.7.1" in g
    t = [s for s in plan.steps if s.id == "torch"][0].argv
    assert "torch==2.7.1" in t and t[t.index("--index-url") + 1].endswith("/cu128")
