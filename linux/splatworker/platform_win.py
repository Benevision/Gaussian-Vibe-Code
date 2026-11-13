"""Windows-only system queries for the worker (docs/dev/WINDOWS.md). Imported lazily, only on Windows.

* :func:`dxgi_adapters` — every graphics card through DXGI (``IDXGIFactory1::EnumAdapters1``): vendor, device id,
  name and dedicated video memory, as :class:`gpu_vendors.Adapter` — the Windows counterpart of the sysfs scan.
  NVIDIA details (driver, load, temperature) still come from NVML, which ships with the NVIDIA driver.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from typing import List

VENDORS = {0x10DE: "nvidia", 0x1002: "amd", 0x1022: "amd", 0x8086: "intel"}
DXGI_ADAPTER_FLAG_SOFTWARE = 2
DXGI_ERROR_NOT_FOUND = 0x887A0002


class GUID(ctypes.Structure):
    _fields_ = [("Data1", wintypes.DWORD), ("Data2", wintypes.WORD), ("Data3", wintypes.WORD),
                ("Data4", ctypes.c_ubyte * 8)]


class LUID(ctypes.Structure):
    _fields_ = [("LowPart", wintypes.DWORD), ("HighPart", wintypes.LONG)]


class DXGI_ADAPTER_DESC1(ctypes.Structure):
    _fields_ = [("Description", ctypes.c_wchar * 128), ("VendorId", ctypes.c_uint), ("DeviceId", ctypes.c_uint),
                ("SubSysId", ctypes.c_uint), ("Revision", ctypes.c_uint),
                ("DedicatedVideoMemory", ctypes.c_size_t), ("DedicatedSystemMemory", ctypes.c_size_t),
                ("SharedSystemMemory", ctypes.c_size_t), ("AdapterLuid", LUID), ("Flags", ctypes.c_uint)]


# {770aae78-f26f-4dba-a829-253c83d1b387}
IID_IDXGIFactory1 = GUID(0x770AAE78, 0xF26F, 0x4DBA, (ctypes.c_ubyte * 8)(0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87))

# vtable slots (IUnknown 0–2, IDXGIObject 3–6, IDXGIFactory 7–11, IDXGIFactory1 12–13; IDXGIAdapter 7–9, Adapter1 10)
_RELEASE, _ENUM_ADAPTERS1, _GET_DESC1 = 2, 12, 10


def _method(obj: ctypes.c_void_p, index: int, restype, *argtypes):
    vtbl = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    return ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)(vtbl[index])


def _release(obj: ctypes.c_void_p) -> None:
    if obj:
        _method(obj, _RELEASE, ctypes.c_ulong)(obj)


def dxgi_adapters(include_software: bool = False) -> List["Adapter"]:  # noqa: F821
    """Hardware graphics adapters in DXGI order (the first is the primary display adapter). Empty on failure."""
    from .gpu_vendors import Adapter

    out: List[Adapter] = []
    try:
        dxgi = ctypes.WinDLL("dxgi.dll")
    except OSError:
        return out
    factory = ctypes.c_void_p()
    create = dxgi.CreateDXGIFactory1
    create.restype = ctypes.c_long
    create.argtypes = [ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p)]
    if create(ctypes.byref(IID_IDXGIFactory1), ctypes.byref(factory)) != 0 or not factory:
        return out
    try:
        enum = _method(factory, _ENUM_ADAPTERS1, ctypes.c_long, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p))
        seen = set()
        for i in range(64):
            adapter = ctypes.c_void_p()
            hr = enum(factory, i, ctypes.byref(adapter))
            if hr != 0 or not adapter:
                break
            try:
                desc = DXGI_ADAPTER_DESC1()
                if _method(adapter, _GET_DESC1, ctypes.c_long, ctypes.POINTER(DXGI_ADAPTER_DESC1))(
                        adapter, ctypes.byref(desc)) != 0:
                    continue
                if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE and not include_software:
                    continue
                luid = f"luid:{desc.AdapterLuid.HighPart & 0xFFFFFFFF:08x}{desc.AdapterLuid.LowPart:08x}"
                key = (desc.VendorId, desc.DeviceId, luid)
                if key in seen:
                    continue
                seen.add(key)
                dedicated = int(desc.DedicatedVideoMemory) // (1024 * 1024)
                out.append(Adapter(
                    vendor=VENDORS.get(desc.VendorId, "other"), vendor_id=int(desc.VendorId),
                    device_id=int(desc.DeviceId), pci=luid, driver="dxgi", name=desc.Description.strip(),
                    # integrated GPUs report a small carve-out (≤ 512 MB): treat as shared memory
                    memory_mb=dedicated if dedicated > 512 else None, card=f"adapter{i}"))
            finally:
                _release(adapter)
    finally:
        _release(factory)
    return out
