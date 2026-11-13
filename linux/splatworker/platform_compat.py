"""Small operating-system helpers so the worker and studio run on Linux and Windows (docs/dev/WINDOWS.md).

Each helper keeps the Linux behaviour unchanged and adds a Windows path:

* :func:`memory` — total / available RAM (``/proc/meminfo`` on Linux, ``GlobalMemoryStatusEx`` on Windows).
* :func:`link_path` — a link to a file or folder: a symlink where allowed; on Windows (symlinks need Developer Mode
  or admin) a directory junction for folders and a hard link for files, copying as the last resort.
* :func:`new_group_kwargs` / :func:`kill_tree` — start a child in its own process group and stop it with all its
  children (``killpg`` on POSIX, ``taskkill /T`` on Windows).
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

IS_WINDOWS = sys.platform == "win32"


def memory(meminfo: str = "/proc/meminfo") -> Dict[str, int]:
    """``{"total": bytes, "available": bytes}``; keys are missing when the value is unknown."""
    out: Dict[str, int] = {}
    try:
        with open(meminfo) as f:
            for line in f:
                k, v = line.split(":", 1)
                if k == "MemTotal":
                    out["total"] = int(v.split()[0]) * 1024
                elif k == "MemAvailable":
                    out["available"] = int(v.split()[0]) * 1024
        if out:
            return out
    except (OSError, ValueError):
        pass
    if IS_WINDOWS:
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            st = MEMORYSTATUSEX()
            st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                return {"total": int(st.ullTotalPhys), "available": int(st.ullAvailPhys)}
        except Exception:  # noqa: BLE001 — unknown memory is reported as unknown
            pass
        return out
    try:
        out["total"] = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        pass
    return out


def link_path(target: Path, link: Path) -> str:
    """Make ``link`` point at ``target`` (an existing file or folder). Returns how: "symlink", "junction",
    "hardlink" or "copy". Writing through the link writes into ``target`` except after a copy."""
    target = Path(target).resolve()
    link = Path(link)
    is_dir = target.is_dir()
    try:
        os.symlink(target, link, target_is_directory=is_dir)
        return "symlink"
    except (OSError, NotImplementedError):
        if not IS_WINDOWS:
            raise
    if is_dir:
        try:
            import _winapi  # CPython's own junction support (used by its test suite); Windows only

            _winapi.CreateJunction(str(target), str(link))
            return "junction"
        except Exception:  # noqa: BLE001 — another volume type: copy
            shutil.copytree(target, link)
            return "copy"
    try:
        os.link(target, link)
        return "hardlink"
    except OSError:
        shutil.copy2(target, link)
        return "copy"


def is_link(path: Path) -> bool:
    """A symlink, or (Windows) a directory junction."""
    p = Path(path)
    if p.is_symlink():
        return True
    is_junction = getattr(os.path, "isjunction", None)  # Python 3.12+
    if is_junction is not None:
        return bool(is_junction(p))
    if IS_WINDOWS:
        try:
            return bool(os.readlink(p))
        except (OSError, ValueError):
            return False
    return False


def new_group_kwargs() -> dict:
    """``subprocess.Popen`` keyword arguments that start the child in its own process group."""
    if IS_WINDOWS:
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | getattr(subprocess, "CREATE_NO_WINDOW", 0)}
    return {"start_new_session": True}


def owns_group(proc: subprocess.Popen) -> bool:
    """True if ``proc`` leads its own process group (POSIX), so killing the group can't hit us."""
    if IS_WINDOWS:
        return False
    try:
        return os.getpgid(proc.pid) == proc.pid and os.getpgid(proc.pid) != os.getpgid(0)
    except (OSError, AttributeError):
        return False


def kill_tree(proc: subprocess.Popen, force: bool = False) -> None:
    """Stop ``proc`` and its children: SIGTERM (or SIGKILL) to its group on POSIX, ``taskkill /T`` on Windows."""
    if proc.poll() is not None:
        return
    if IS_WINDOWS:
        try:
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T"] + (["/F"] if force else []),
                           capture_output=True, timeout=15, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except (OSError, subprocess.SubprocessError):
            pass
        if force and proc.poll() is None:
            proc.kill()
        return
    if owns_group(proc):
        try:
            os.killpg(proc.pid, signal.SIGKILL if force else signal.SIGTERM)
            return
        except OSError:
            pass
    proc.kill() if force else proc.terminate()


def default_data_dir(env: Optional[dict] = None) -> str:
    """The headless worker's default data folder: ``~/.local/share/splatworker`` on Linux,
    ``%LOCALAPPDATA%\\SplatWork\\data`` on Windows."""
    env = os.environ if env is None else env
    if IS_WINDOWS:
        base = env.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return str(Path(base) / "SplatWork" / "data")
    return "~/.local/share/splatworker"


def documents_dir(env: Optional[dict] = None) -> Optional[Path]:
    """The user's Documents folder on Windows (Known Folder, which OneDrive may redirect); None elsewhere."""
    if not IS_WINDOWS:
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class GUID(ctypes.Structure):
            _fields_ = [("Data1", wintypes.DWORD), ("Data2", wintypes.WORD), ("Data3", wintypes.WORD),
                        ("Data4", ctypes.c_ubyte * 8)]

        # FOLDERID_Documents {FDD39AD0-238F-46AF-ADB4-6C85480369C7}
        fid = GUID(0xFDD39AD0, 0x238F, 0x46AF, (ctypes.c_ubyte * 8)(0xAD, 0xB4, 0x6C, 0x85, 0x48, 0x03, 0x69, 0xC7))
        p = ctypes.c_wchar_p()
        if ctypes.windll.shell32.SHGetKnownFolderPath(ctypes.byref(fid), 0, None, ctypes.byref(p)) == 0:
            try:
                return Path(p.value)
            finally:
                ctypes.windll.ole32.CoTaskMemFree(p)
    except Exception:  # noqa: BLE001
        pass
    env = os.environ if env is None else env
    base = env.get("USERPROFILE")
    return Path(base) / "Documents" if base else None
