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
    assert amd.name == "AMD Radeon RX 7900 XT/7900 XTX/7900 GRE/7900M" and amd.driver == "amdgpu"
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
    assert title == "AMD Radeon RX 7900 XT/7900 XTX/7900 GRE/7900M found"
    assert "needs an NVIDIA graphics card" in text and "ROCm" in text and "Flatpak driver" not in text
    c = choose_profile(d)
    assert c.profile_id == "rocm6.3" and not c.can_train and "dataset processing uses the GPU" in c.summary
    assert c.blocking == text and any("ROCm" in w for w in c.warnings)
    plan = build_plan(c, _paths(tmp_path), with_splatprep=True, with_lightglue=True, splatprep_dir=tmp_path)
    ids = [s.id for s in plan.steps]
    assert "gsplat" not in ids and ids.index("torch") < ids.index("torchvision") and not plan.can_train
    torch_step = next(s for s in plan.steps if s.id == "torch")
    assert "torch==2.7.1" in torch_step.argv and "https://download.pytorch.org/whl/rocm6.3" in torch_step.argv
    lg = next(s for s in plan.steps if s.id == "processing").argv
    assert "torch==2.7.1" in lg and "torchvision==0.22.1" in lg
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
    assert {"rocm6.3", "xpu"} <= set(PROFILE_IDS)
    c = choose_profile(DriverInfo(), override="xpu")
    assert c.profile_id == "xpu" and not c.can_train and "CPU" in c.warnings[0]


def test_verify_processing_runtime(tmp_path, names):
    d = info_for(tmp_path, names, lambda s: amd_card(s))
    plan = build_plan(choose_profile(d), _paths(tmp_path), with_splatprep=True, with_lightglue=True,
                      splatprep_dir=tmp_path)
    ok = {"splatworker": True, "splatprep": True, "lightglue": True, "torch": "2.7.1+rocm6.3",
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
