"""Every graphics card in the computer, of any vendor, from sysfs (no driver library, no torch).

NVML (``gpu.py``, ``gui/nvidia.py``) only sees NVIDIA cards. This module reads the kernel's view instead:
``/sys/class/drm/card<N>/device`` for each GPU (PCI vendor 0x10de NVIDIA, 0x1002 AMD, 0x8086 Intel), its kernel
driver, and what the driver publishes:

* amdgpu: ``mem_info_vram_total`` / ``mem_info_vram_used`` (bytes), ``gpu_busy_percent``, ``product_name``
  (some cards), ``hwmon/hwmon*/temp1_input``;
* i915 / xe (Intel): no common VRAM file for integrated GPUs (they share system memory); for Arc cards the xe
  driver's ``tile0/vram0/size`` when present.

Names come from the PCI ID database (``pci.ids``) when one is installed; otherwise "AMD GPU (1002:744c)".
Everything is optional and never raises: a missing file just leaves a field empty. Tests point ``sys_root`` at
a fake tree.
"""
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

VENDORS = {0x10de: "nvidia", 0x1002: "amd", 0x8086: "intel"}
VENDOR_NAMES = {"nvidia": "NVIDIA", "amd": "AMD", "intel": "Intel"}

PCI_IDS_PATHS = ("/usr/share/hwdata/pci.ids", "/usr/share/misc/pci.ids", "/usr/share/pci.ids",
                 "/run/host/usr/share/hwdata/pci.ids", "/run/host/usr/share/misc/pci.ids")


@dataclass
class Adapter:
    vendor: str                      # nvidia | amd | intel | other
    vendor_id: int
    device_id: int
    pci: str                         # 0000:03:00.0
    driver: str = ""                 # amdgpu, i915, xe, nvidia, nouveau, radeon …
    name: str = ""
    memory_mb: Optional[int] = None  # dedicated video memory (None: shared / unknown)
    memory_used_mb: Optional[int] = None
    busy_pct: Optional[int] = None
    temperature_c: Optional[int] = None
    card: str = ""                   # card0
    extra: Dict[str, str] = field(default_factory=dict)

    @property
    def vendor_name(self) -> str:
        return VENDOR_NAMES.get(self.vendor, "Unknown")

    @property
    def integrated(self) -> bool:
        """Best guess: Intel without dedicated memory, or an AMD APU with little VRAM carve-out."""
        if self.vendor == "intel":
            return self.memory_mb is None
        if self.vendor == "amd":
            return self.memory_mb is not None and self.memory_mb <= 4096 and "Radeon RX" not in self.name \
                and "Instinct" not in self.name and "Radeon Pro" not in self.name
        return False

    def to_info(self) -> Dict:
        """The ``/v1/info`` "adapters" entry."""
        d = {"vendor": self.vendor, "name": self.name, "pci": self.pci, "driver": self.driver,
             "integrated": self.integrated}
        if self.memory_mb is not None:
            d["memoryMB"] = self.memory_mb
            if self.memory_used_mb is not None:
                d["freeMemoryMB"] = max(0, self.memory_mb - self.memory_used_mb)
        return d


def _read(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None


def _read_int(path: Path, base: int = 10) -> Optional[int]:
    v = _read(path)
    if v is None:
        return None
    try:
        return int(v, base)
    except ValueError:
        return None


# ------------------------------------------------------------------------------------------ PCI names
_names_cache: Dict[str, Dict] = {}


def pci_names(paths: Iterable[str] = PCI_IDS_PATHS) -> Dict:
    """{vendor_id: (vendor name, {device_id: device name})} for the GPU vendors, from the first pci.ids found."""
    for p in paths:
        if p in _names_cache:
            return _names_cache[p]
        if not os.path.isfile(p):
            continue
        out: Dict = {}
        cur = None
        try:
            with open(p, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if not line.strip() or line.startswith("#"):
                        continue
                    if line.startswith("C "):         # device classes follow: no more vendors
                        break
                    if not line.startswith("\t"):
                        m = re.match(r"([0-9a-fA-F]{4})\s+(.*)", line)
                        vid = int(m.group(1), 16) if m else None
                        cur = (out.setdefault(vid, (m.group(2).strip(), {}))) if vid in VENDORS else None
                    elif cur is not None and not line.startswith("\t\t"):
                        m = re.match(r"\t([0-9a-fA-F]{4})\s+(.*)", line)
                        if m:
                            cur[1][int(m.group(1), 16)] = m.group(2).strip()
        except OSError:
            continue
        _names_cache[p] = out
        return out
    return {}


def _marketing_name(raw: str) -> str:
    """"Navi 31 [Radeon RX 7900 XT/7900 XTX/7900 GRE/7900M]" → "Radeon RX 7900 XT/7900 XTX/7900 GRE/7900M";
    "DG2 [Arc A770]" → "Arc A770"."""
    m = re.search(r"\[(.+)\]", raw)
    return m.group(1) if m else raw


def adapter_name(vendor: str, vendor_id: int, device_id: int, product: Optional[str] = None,
                 names: Optional[Dict] = None) -> str:
    vname = VENDOR_NAMES.get(vendor, "GPU")
    if product:                                            # amdgpu's product_name (e.g. "AMD Radeon RX 7900 XTX")
        return product if product.lower().startswith(vname.lower()) else f"{vname} {product}"
    names = pci_names() if names is None else names
    dev = (names.get(vendor_id) or ("", {}))[1].get(device_id)
    if dev:
        return f"{vname} {_marketing_name(dev)}"
    return f"{vname} GPU ({vendor_id:04x}:{device_id:04x})"


# ------------------------------------------------------------------------------------------- scanning
def scan(sys_root: Path = Path("/sys"), names: Optional[Dict] = None) -> List[Adapter]:
    """All display adapters the kernel knows, one per PCI device, in card order."""
    out: List[Adapter] = []
    seen = set()
    cards = sorted(glob.glob(str(Path(sys_root) / "class" / "drm" / "card[0-9]*")),
                   key=lambda p: int(re.sub(r"\D", "", os.path.basename(p)) or 0))
    for card in cards:
        base = os.path.basename(card)
        if not re.fullmatch(r"card\d+", base):            # card0-DP-1 etc. are connectors
            continue
        dev = Path(card) / "device"
        vid = _read_int(dev / "vendor", 16)
        did = _read_int(dev / "device", 16)
        if vid is None or did is None:
            continue
        try:
            pci = os.path.basename(os.path.realpath(dev))
        except OSError:
            pci = base
        if pci in seen:
            continue
        seen.add(pci)
        driver = ""
        try:
            driver = os.path.basename(os.path.realpath(dev / "driver")) if (dev / "driver").exists() else ""
        except OSError:
            pass
        vendor = VENDORS.get(vid, "other")
        a = Adapter(vendor=vendor, vendor_id=vid, device_id=did, pci=pci, driver=driver, card=base)
        product = _read(dev / "product_name") if vendor == "amd" else None
        a.name = adapter_name(vendor, vid, did, product or None, names)
        if vendor == "amd":
            total = _read_int(dev / "mem_info_vram_total")
            used = _read_int(dev / "mem_info_vram_used")
            a.memory_mb = total // 2**20 if total else None
            a.memory_used_mb = used // 2**20 if used is not None and total else None
            a.busy_pct = _read_int(dev / "gpu_busy_percent")
        elif vendor == "intel":
            size = _read_int(dev / "tile0" / "vram0" / "size")      # xe driver, discrete Arc
            if size is None:
                size = _read_int(dev / "lmem_total_bytes")           # some i915 builds for DG2
            a.memory_mb = size // 2**20 if size else None
        temps = sorted(glob.glob(str(dev / "hwmon" / "hwmon*" / "temp1_input")))
        if temps:
            t = _read_int(Path(temps[0]))
            a.temperature_c = t // 1000 if t is not None else None
        out.append(a)
    return out


def by_vendor(adapters: List[Adapter], vendor: str) -> List[Adapter]:
    return [a for a in adapters if a.vendor == vendor]


def kfd_available(dev_root: Path = Path("/dev")) -> bool:
    """AMD's compute interface (ROCm needs /dev/kfd; the Flatpak's --device=all exposes it)."""
    return (Path(dev_root) / "kfd").exists()


def best_non_nvidia(adapters: List[Adapter]) -> Optional[Adapter]:
    """The AMD / Intel GPU worth using for compute: dedicated memory first, then the largest."""
    cands = [a for a in adapters if a.vendor in ("amd", "intel")]
    if not cands:
        return None
    return sorted(cands, key=lambda a: (a.integrated, -(a.memory_mb or 0)))[0]
