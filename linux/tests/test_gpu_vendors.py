"""AMD and Intel GPUs: sysfs detection, setup wording, ROCm / XPU processing runtimes, /v1/info "adapters".
All with fake sysfs trees: nothing here has run on real AMD or Intel hardware."""
import os
from pathlib import Path

import pytest

from splatworker import gpu_vendors as V
from splatworker.gui.nvidia import DriverInfo, GpuInfo, GpuMonitor, detect
from splatworker.gui.runtime_plan import (CPU_PROFILE_ID, PROFILE_IDS, build_plan, choose_profile,
                                          verify_problems)

PCI_IDS = """\
# fake pci.ids
1002  Advanced Micro Devices, Inc. [AMD/ATI]
\t744c  Navi 31 [Radeon RX 7900 XT/7900 XTX/7900 GRE/7900M]
\t15bf  Phoenix1
\t\t1234 5678  some subsystem
8086  Intel Corporation
\t56a0  DG2 [Arc A770]
\ta7a0  Raptor Lake-P [Iris Xe Graphics]
10de  NVIDIA Corporation
\t2704  AD103 [GeForce RTX 4080]
C 00  Unclassified device
"""


def card(sys_root: Path, n: int, vendor: int, dev: int, driver: str, pci: str, files=None, hwmon_temp=None):
    pdir = sys_root / "devices" / "pci0000:00" / pci
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "vendor").write_text(f"0x{vendor:04x}\n")
    (pdir / "device").write_text(f"0x{dev:04x}\n")
    drv = sys_root / "bus" / "pci" / "drivers" / driver
    drv.mkdir(parents=True, exist_ok=True)
    if not (pdir / "driver").exists():
        os.symlink(drv, pdir / "driver")
    for name, value in (files or {}).items():
        f = pdir / name
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(str(value))
    if hwmon_temp is not None:
        h = pdir / "hwmon" / "hwmon3"
        h.mkdir(parents=True, exist_ok=True)
        (h / "temp1_input").write_text(str(hwmon_temp))
    drm = sys_root / "class" / "drm"
    (drm / f"card{n}").mkdir(parents=True, exist_ok=True)
    os.symlink(pdir, drm / f"card{n}" / "device")
    (drm / f"card{n}-DP-1").mkdir(exist_ok=True)           # a connector: not a card
    return pdir


@pytest.fixture
def names(tmp_path):
    p = tmp_path / "pci.ids"
    p.write_text(PCI_IDS)
    V._names_cache.clear()
    return V.pci_names([str(p)])


def amd_card(sys_root, n=0, total=24 * 2**30, used=2 * 2**30, product=None):
    files = {"mem_info_vram_total": total, "mem_info_vram_used": used, "gpu_busy_percent": 37}
    if product:
        files["product_name"] = product
    return card(sys_root, n, 0x1002, 0x744c, "amdgpu", f"0000:0{n + 3}:00.0", files, hwmon_temp=55000)


def test_scan_amd_intel_nvidia(tmp_path, names):
    sys_root = tmp_path / "sys"
    amd_card(sys_root, 0)
    card(sys_root, 1, 0x8086, 0x56a0, "xe", "0000:05:00.0", {"tile0/vram0/size": 16 * 2**30})
    card(sys_root, 2, 0x8086, 0xa7a0, "i915", "0000:00:02.0")
    card(sys_root, 3, 0x10de, 0x2704, "nvidia", "0000:01:00.0")
    found = V.scan(sys_root, names)
    assert [a.vendor for a in found] == ["amd", "intel", "intel", "nvidia"]
    amd, arc, iris, nv = found
    assert amd.name == "AMD Radeon RX 7900 series" and amd.driver == "amdgpu"
    assert (amd.memory_mb, amd.memory_used_mb, amd.busy_pct, amd.temperature_c) == (24576, 2048, 37, 55)
    assert not amd.integrated and amd.to_info()["freeMemoryMB"] == 22528
    assert arc.name == "Intel Arc A770" and arc.memory_mb == 16384 and not arc.integrated
    assert iris.name == "Intel Iris Xe Graphics" and iris.memory_mb is None and iris.integrated
    assert nv.name == "NVIDIA GeForce RTX 4080"
    assert V.best_non_nvidia(found) is amd


def test_names_without_pci_ids_and_product_name(tmp_path):
    sys_root = tmp_path / "sys"
    amd_card(sys_root, 0, product="Radeon RX 7800 XT")
    card(sys_root, 1, 0x8086, 0x1234, "i915", "0000:00:02.0")
    a, i = V.scan(sys_root, names={})
    assert a.name == "AMD Radeon RX 7800 XT" and i.name == "Intel GPU (8086:1234)"
    assert V.scan(tmp_path / "nothing", names={}) == []


def info_for(tmp_path, names, *build, kfd=True):
    sys_root, dev_root = tmp_path / "sys", tmp_path / "dev"
    dev_root.mkdir(parents=True, exist_ok=True)
    if kfd:
        (dev_root / "kfd").write_text("")
    for b in build:
        b(sys_root)
    V._names_cache.clear()
    import splatworker.gpu_vendors as gv
    orig = gv.pci_names
    gv.pci_names = lambda paths=None: names
    try:
        return detect(nvml=None, smi=None, proc_file=tmp_path / "no-proc", dev_glob=str(tmp_path / "nv*"),
                      flatpak=True, sys_root=sys_root, dev_root=dev_root)
    finally:
        gv.pci_names = orig


def test_amd_setup_offers_rocm_processing(tmp_path, names):
    d = info_for(tmp_path, names, lambda s: amd_card(s))
    assert not d.has_gpu and d.kfd and d.other_gpu.vendor == "amd"
    title, text = d.problem()
    assert title == "AMD Radeon RX 7900 series found"
    assert "needs an NVIDIA graphics card" in text and "ROCm" in text and "Flatpak driver" not in text
    c = choose_profile(d)
    assert c.profile_id == "rocm6.4" and not c.can_train and "dataset processing uses the GPU" in c.summary
    assert c.blocking == text and any("ROCm" in w for w in c.warnings)
    plan = build_plan(c, _paths(tmp_path), with_splatprep=True, with_lightglue=True, splatprep_dir=tmp_path)
    ids = [s.id for s in plan.steps]
    assert "gsplat" not in ids and ids.index("torch") < ids.index("torchvision") and not plan.can_train
    torch_step = next(s for s in plan.steps if s.id == "torch")
    assert "torch==2.8.0" in torch_step.argv and "https://download.pytorch.org/whl/rocm6.4" in torch_step.argv
    lg = next(s for s in plan.steps if s.id == "processing").argv
    assert "torch==2.8.0" in lg and "torchvision==0.23.0" in lg
    # without the processing models there is no reason to download PyTorch at all
    small = build_plan(c, _paths(tmp_path), with_splatprep=True, with_lightglue=False, splatprep_dir=tmp_path)
    assert "torch" not in [s.id for s in small.steps]


def test_amd_without_kfd_or_integrated_stays_on_cpu(tmp_path, names):
    d = info_for(tmp_path, names, lambda s: amd_card(s), kfd=False)
    assert choose_profile(d).profile_id == CPU_PROFILE_ID
    d2 = info_for(tmp_path / "apu", names, lambda s: card(s, 0, 0x1002, 0x15bf, "amdgpu", "0000:c1:00.0",
                                                          {"mem_info_vram_total": 512 * 2**20}))
    c = choose_profile(d2)
    assert c.profile_id == CPU_PROFILE_ID and "training isn't available on AMD" in c.summary


def test_intel_arc_gets_xpu_and_igpu_cpu(tmp_path, names):
    arc = info_for(tmp_path, names, lambda s: card(s, 0, 0x8086, 0x56a0, "xe", "0000:05:00.0",
                                                   {"tile0/vram0/size": 16 * 2**30}))
    c = choose_profile(arc)
    assert c.profile_id == "xpu" and "Intel Arc A770" in c.summary and "XPU" in arc.problem()[1]
    iris = info_for(tmp_path / "i", names, lambda s: card(s, 0, 0x8086, 0xa7a0, "i915", "0000:00:02.0"))
    assert choose_profile(iris).profile_id == CPU_PROFILE_ID


def test_nvidia_behaviour_unchanged(tmp_path, names):
    """An NVIDIA card (even with an AMD / Intel one next to it) keeps the CUDA path and wording."""
    d = DriverInfo(driver_version="570.1", cuda_version=(12, 8), source="nvml",
                   gpus=[GpuInfo(0, "NVIDIA GeForce RTX 4080", 16376, (8, 9))])
    d.adapters = V.scan(_mixed(tmp_path), names)
    assert d.other_gpu is None and d.problem() is None and choose_profile(d).profile_id == "cu128"
    # NVIDIA driver loaded but its libraries missing in the Flatpak: still the NVIDIA message, not "AMD found"
    broken = DriverInfo(kernel_driver_version="570.1", in_flatpak=True, adapters=d.adapters)
    assert broken.problem()[0] == "NVIDIA driver libraries are missing in the sandbox"
    assert choose_profile(broken).profile_id == CPU_PROFILE_ID


def test_manual_processing_profile_and_ids(tmp_path, names):
    assert {"rocm6.4", "xpu"} <= set(PROFILE_IDS)
    c = choose_profile(DriverInfo(), override="xpu")
    assert c.profile_id == "xpu" and not c.can_train and "CPU" in c.warnings[0]


def test_verify_processing_runtime(tmp_path, names):
    d = info_for(tmp_path, names, lambda s: amd_card(s))
    plan = build_plan(choose_profile(d), _paths(tmp_path), with_splatprep=True, with_lightglue=True,
                      splatprep_dir=tmp_path)
    ok = {"splatworker": True, "splatprep": True, "lightglue": True, "torch": "2.8.0+rocm6.4",
          "cudaAvailable": False, "torchvisionImports": True}
    assert verify_problems(ok, plan) == []          # GPU not visible: processing falls back to the CPU
    assert verify_problems({**ok, "torch": None}, plan) == ["PyTorch could not be imported."]


def test_monitor_reads_amd_from_sysfs(tmp_path, names):
    sys_root = tmp_path / "sys"
    amd_card(sys_root)
    m = GpuMonitor(nvml=None, sys_root=sys_root)
    assert m.available and m.error is None
    s = m.sample()[0]
    assert (s.memory_total_mb, s.memory_used_mb, s.utilization_pct, s.temperature_c) == (24576, 2048, 37, 55)
    assert not GpuMonitor(nvml=None, sys_root=tmp_path / "none").available


def test_info_reports_adapters(tmp_path, monkeypatch, names):
    TestClient = pytest.importorskip("fastapi.testclient").TestClient

    from conftest import FakeBackend
    from splatworker import gpu as G
    from splatworker.config import ServerSettings
    from splatworker.server import create_app

    sys_root = _mixed(tmp_path)
    monkeypatch.setattr(G, "adapters", lambda sys_root_=None: [a.to_info() for a in V.scan(sys_root, names)])
    app = create_app(ServerSettings(data_dir=tmp_path / "d", token=None, name="w", mdns=False),
                     backend=FakeBackend(), devices=[None])
    with TestClient(app) as c:
        info = c.get("/v1/info").json()
    assert [a["vendor"] for a in info["adapters"]] == ["amd", "intel"]
    assert info["adapters"][0]["memoryMB"] == 24576 and "memoryMB" not in info["adapters"][1]


def _mixed(tmp_path):
    sys_root = tmp_path / "mixed" / "sys"
    if not sys_root.exists():
        amd_card(sys_root, 0)
        card(sys_root, 1, 0x8086, 0xa7a0, "i915", "0000:00:02.0")
    return sys_root


def _paths(tmp_path):
    from splatworker.gui.paths import AppPaths
    return AppPaths.from_env({"HOME": str(tmp_path / "home"), "XDG_DATA_HOME": str(tmp_path / "data"),
                              "XDG_CONFIG_HOME": str(tmp_path / "cfg"), "XDG_CACHE_HOME": str(tmp_path / "cache")})


# ------------------------------------------------------------------ RDNA 2 (RX 6700 XT) and the HSA override
def kfd_node(sys_root, n, gfx_version, location_id, domain=0):
    d = sys_root / "class" / "kfd" / "kfd" / "topology" / "nodes" / str(n)
    d.mkdir(parents=True, exist_ok=True)
    (d / "properties").write_text(f"cpu_cores_count 0\nsimd_count 80\ngfx_target_version {gfx_version}\n"
                                  f"location_id {location_id}\ndomain {domain}\n")


def rx6700xt(sys_root, with_kfd=True):
    card(sys_root, 0, 0x1002, 0x73df, "amdgpu", "0000:03:00.0",
         {"mem_info_vram_total": 12 * 2**30, "mem_info_vram_used": 300 * 2**20})
    if with_kfd:
        kfd_node(sys_root, 0, 0, 0)                          # the CPU node
        kfd_node(sys_root, 1, 100301, 0x0300)               # bus 3, device 0, function 0


def test_gfx_from_kfd_topology_and_device_ids(tmp_path):
    sys_root = tmp_path / "sys"
    rx6700xt(sys_root)
    a = V.scan(sys_root, names={})[0]
    assert a.gfx == "gfx1031" and a.to_info()["gfx"] == "gfx1031" and not a.integrated
    assert V.hsa_override([a]) == "10.3.0"
    nokfd = tmp_path / "b" / "sys"
    rx6700xt(nokfd, with_kfd=False)
    assert V.scan(nokfd, names={})[0].gfx == "gfx1031"     # from the device-id table
    assert V.gfx_from_device_id(0x744c) == ""                # RDNA 3: not in the table
    kfd_node(tmp_path / "c", 1, 90010, 0x0100)
    kfd_node(tmp_path / "c", 2, 120001, 0x0400)
    assert V.kfd_gfx(tmp_path / "c") == {"0000:01:00.0": "gfx90a", "0000:04:00.0": "gfx1201"}


def test_supported_cards_get_no_override(tmp_path):
    sys_root = tmp_path / "sys"
    card(sys_root, 0, 0x1002, 0x744c, "amdgpu", "0000:03:00.0", {"mem_info_vram_total": 24 * 2**30})
    kfd_node(sys_root, 1, 110000, 0x0300)
    assert V.hsa_override(V.scan(sys_root, names={})) is None
    assert V.hsa_override([]) is None


def test_gpu_env_sets_override_inside_the_app_only(tmp_path):
    from splatworker.gui.installer import gpu_env
    sys_root = tmp_path / "sys"
    rx6700xt(sys_root)
    assert gpu_env({}, sys_root) == {"HSA_OVERRIDE_GFX_VERSION": "10.3.0"}
    assert gpu_env({"HSA_OVERRIDE_GFX_VERSION": "10.3.1"}, sys_root) == {}      # the user's own value wins
    assert gpu_env({"SPLATWORK_NO_HSA_OVERRIDE": "1"}, sys_root) == {}
    assert gpu_env({}, tmp_path / "none") == {}


def test_rx6700xt_setup_mentions_the_override(tmp_path, names):
    d = info_for(tmp_path, names, lambda s: rx6700xt(s))
    c = choose_profile(d)
    assert c.profile_id == "rocm6.4" and any("gfx1031" in w and "10.3.0" in w for w in c.warnings)


def test_gpu_check_without_a_gpu(monkeypatch, capsys):
    import importlib.machinery
    import sys
    import types

    from splatworker import gpucheck
    t = types.ModuleType("torch")
    t.__spec__ = importlib.machinery.ModuleSpec("torch", None)
    t.__version__ = "2.8.0+rocm6.4"
    t.version = types.SimpleNamespace(cuda=None, hip="6.4.1")
    t.cuda = types.SimpleNamespace(is_available=lambda: False)
    t.xpu = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", t)
    rc = gpucheck.main(types.SimpleNamespace(no_train=False, splats=10))
    out = capsys.readouterr().out
    assert "no GPU visible to PyTorch" in out and rc == 0
    import json
    rep = json.loads(out.strip().splitlines()[-1].split(" ", 1)[1])
    assert rep["torch"]["value"]["hip"] == "6.4.1" and rep["torchBackend"] == "cpu"


def test_launcher_routes_gpu_check_to_the_runtime(tmp_path):
    from splatworker.gui.launcher import headless_command
    exe, argv, env, _ = headless_command(["gpu-check"], paths=_paths(tmp_path), environ={"HOME": str(tmp_path)})
    assert argv[1:4] == ["-m", "splatworker", "gpu-check"] and "serve" not in argv


# ------------------------------------------------------------------ experimental AMD training runtime
def test_rocm_training_profile_is_manual_and_installs_gsplat(tmp_path, names):
    from splatworker.gui.runtime_plan import processing_profile_by_id, rocm_gsplat_published
    d = info_for(tmp_path, names, lambda s: rx6700xt(s))
    assert choose_profile(d).profile_id == "rocm6.4"                      # never chosen automatically
    c = choose_profile(d, override="rocm6.4-train")
    assert c.can_train and any("Experimental" in w for w in c.warnings)
    plan = build_plan(c, _paths(tmp_path), with_splatprep=True, with_lightglue=True, splatprep_dir=tmp_path)
    ids = [s.id for s in plan.steps]
    assert plan.can_train and ids.count("torch") == 1 and ids.index("torch") < ids.index("gsplat")
    torch_step = next(s for s in plan.steps if s.id == "torch").argv
    assert "https://download.pytorch.org/whl/rocm6.4" in torch_step and "torch==2.8.0" in torch_step
    gs = next(s for s in plan.steps if s.id == "gsplat").argv
    assert "gsplat==1.5.3+pt28rocm64" in gs and "torch==2.8.0" in gs and "--find-links" in gs
    ok = {"splatworker": True, "splatprep": True, "lightglue": True, "torch": "2.8.0+rocm6.4", "cudaAvailable": True,
          "gsplat": "1.5.3+pt28rocm64", "torchvisionImports": True}
    assert verify_problems(ok, plan) == []
    assert any("cannot see the AMD GPU" in p for p in verify_problems({**ok, "cudaAvailable": False}, plan))
    prof = processing_profile_by_id("rocm6.4-train")

    class Page:
        def __init__(self, text): self.text = text
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return self.text.encode()
    assert rocm_gsplat_published(prof, lambda url, timeout: Page(
        '<a href="/x/gsplat-1.5.3%2Bpt28rocm64-cp310-cp310-linux_x86_64.whl">')) is True
    assert rocm_gsplat_published(prof, lambda url, timeout: Page("<html>no assets</html>")) is False

    def offline(url, timeout):
        raise OSError("offline")
    assert rocm_gsplat_published(prof, offline) is None
    # on a PC without an AMD GPU, chosen by hand: it says it won't train here
    assert any("No AMD GPU" in w for w in choose_profile(DriverInfo(), override="rocm6.4-train").warnings)
