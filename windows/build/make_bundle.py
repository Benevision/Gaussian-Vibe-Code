"""Build the SplatWork for Windows app folder from an MSYS2 UCRT64 installation (docs/dev/WINDOWS.md).

Run inside the MSYS2 UCRT64 shell (CI: .github/workflows/windows.yml), from the repository root:

    python windows/build/make_bundle.py --out dist/SplatWork --version 1.2.1

Layout (relocatable; the MSI installs it to C:\\Program Files\\SplatWork):

    bin\\SplatWork.exe, splatwork-cli.exe   launchers (windows/launcher/splatwork.c) embedding the bundled Python
    bin\\*.dll                             only the DLLs the app needs (walked from PE import tables)
    lib\\python3.X\\                        Python standard library + site-packages (gi, numpy, fastapi …)
    lib\\girepository-1.0, gdk-pixbuf-2.0, gstreamer-1.0
    share\\glib-2.0\\schemas, icons, …      GTK data
    app\\splatworker, tools\\splatprep       SplatWork's own code (from linux/ and tools/)
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import shutil
import struct
import subprocess
import sys
import sysconfig
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PY_VER = f"{sys.version_info.major}.{sys.version_info.minor}"

# typelibs the app imports (gi.require_version) and their dependencies
TYPELIBS = ["Gtk-4.0", "Gdk-4.0", "Gsk-4.0", "Adw-1", "GLib-2.0", "GObject-2.0", "Gio-2.0", "GModule-2.0",
            "Pango-1.0", "PangoCairo-1.0", "PangoFT2-1.0", "HarfBuzz-0.0", "cairo-1.0", "freetype2-2.0",
            "GdkPixbuf-2.0", "Graphene-1.0", "GdkWin32-4.0", "Gst-1.0", "GstBase-1.0", "GstApp-1.0",
            "GstVideo-1.0", "fontconfig-2.0", "win32-1.0", "PangoFc-1.0", "PangoOT-1.0", "GioWin32-2.0",
            "GLibWin32-2.0"]
GST_PLUGINS = ["libgstcoreelements.dll", "libgstapp.dll", "libgstvideoconvertscale.dll", "libgstvideoconvert.dll",
               "libgstvpx.dll", "libgstmatroska.dll", "libgstisomp4.dll", "libgstopenh264.dll",
               "libgstvideoparsersbad.dll"]
# site-packages to leave out (big and unused by the app's own Python; the GPU runtime has its own)
SKIP_SITE = ["pip", "pip-*", "setuptools", "setuptools-*", "_distutils_hack", "*.dist-info/RECORD"]
SKIP_STDLIB = ["test", "idlelib", "tkinter", "turtledemo", "ensurepip", "lib2to3", "__pycache__"]


def pe_imports(path: Path) -> list:
    """Names of the DLLs a PE file imports (normal and delay-load import tables)."""
    data = path.read_bytes()
    if data[:2] != b"MZ":
        return []
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    if data[pe:pe + 4] != b"PE\0\0":
        return []
    nsec = struct.unpack_from("<H", data, pe + 6)[0]
    opt_size = struct.unpack_from("<H", data, pe + 20)[0]
    opt = pe + 24
    magic = struct.unpack_from("<H", data, opt)[0]
    dd = opt + (112 if magic == 0x20B else 96)
    sections = []
    for i in range(nsec):
        s = opt + opt_size + 40 * i
        vsize, va, rsize, rptr = struct.unpack_from("<IIII", data, s + 8)
        sections.append((va, max(vsize, rsize), rptr))

    def off(rva):
        for va, size, ptr in sections:
            if va <= rva < va + size:
                return rva - va + ptr
        return None

    def cstr(o):
        return data[o:data.index(b"\0", o)].decode("ascii", "replace")

    names = []
    for index, entry_size, name_field in ((1, 20, 12), (13, 32, 4)):   # import, delay import
        rva, size = struct.unpack_from("<II", data, dd + 8 * index)
        o = off(rva) if rva else None
        while o is not None and o + entry_size <= len(data):
            name_rva = struct.unpack_from("<I", data, o + name_field)[0]
            if name_rva == 0:
                break
            no = off(name_rva)
            if no is not None:
                names.append(cstr(no))
            o += entry_size
    return names


def dll_closure(roots, search: Path) -> set:
    """Every DLL from ``search`` (the MSYS2 bin folder) that the roots need, recursively; system DLLs are skipped."""
    have = {p.name.lower(): p for p in search.glob("*.dll")}
    need, todo = set(), list(roots)
    while todo:
        f = todo.pop()
        for name in pe_imports(f):
            k = name.lower()
            if k in have and k not in need:
                need.add(k)
                todo.append(have[k])
    return {have[k] for k in need}


def copytree(src: Path, dst: Path, skip=()) -> None:
    def ignore(d, names):
        return [n for n in names if any(fnmatch.fnmatch(n, pat) for pat in skip)]
    shutil.copytree(src, dst, ignore=ignore, dirs_exist_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dist/SplatWork")
    ap.add_argument("--version", default="0.0.0")
    ap.add_argument("--launchers", default="windows/build/out", help="folder with the built SplatWork.exe / cli")
    a = ap.parse_args()
    prefix = Path(sys.prefix)                     # e.g. D:\a\_temp\msys64\ucrt64
    mbin = prefix / "bin"
    out = Path(a.out).resolve()
    if out.exists():
        shutil.rmtree(out)
    (out / "bin").mkdir(parents=True)

    # Python
    stdlib = Path(sysconfig.get_paths()["stdlib"])
    site = Path(sysconfig.get_paths()["purelib"])
    copytree(stdlib, out / "lib" / f"python{PY_VER}", skip=SKIP_STDLIB + ["site-packages"])
    copytree(site, out / "lib" / f"python{PY_VER}" / "site-packages", skip=SKIP_SITE + ["__pycache__"])
    for exe in ("python.exe", "pythonw.exe"):
        if (mbin / exe).exists():
            shutil.copy2(mbin / exe, out / "bin" / exe)
    for exe in Path(a.launchers).glob("*.exe"):
        shutil.copy2(exe, out / "bin" / exe.name)

    # GTK data
    for sub in ("girepository-1.0",):
        (out / "lib" / sub).mkdir(parents=True, exist_ok=True)
        for t in TYPELIBS:
            f = prefix / "lib" / sub / f"{t}.typelib"
            if f.exists():
                shutil.copy2(f, out / "lib" / sub / f.name)
    copytree(prefix / "lib" / "gdk-pixbuf-2.0", out / "lib" / "gdk-pixbuf-2.0")
    gst = out / "lib" / "gstreamer-1.0"
    gst.mkdir(parents=True)
    for p in GST_PLUGINS:
        if (prefix / "lib" / "gstreamer-1.0" / p).exists():
            shutil.copy2(prefix / "lib" / "gstreamer-1.0" / p, gst / p)
    for sub in ("glib-2.0/schemas", "icons/Adwaita", "icons/hicolor", "gtk-4.0"):
        if (prefix / "share" / sub).exists():
            copytree(prefix / "share" / sub, out / "share" / sub)
    if (prefix / "etc" / "fonts").exists():
        copytree(prefix / "etc" / "fonts", out / "etc" / "fonts")
    subprocess.run([str(mbin / "glib-compile-schemas.exe"), str(out / "share" / "glib-2.0" / "schemas")], check=True)

    # DLLs: everything the executables, Python extension modules, GTK and the plug-ins import
    roots = [*(out / "bin").glob("*.exe"), *out.rglob("*.pyd"), *gst.glob("*.dll"),
             *(out / "lib" / "gdk-pixbuf-2.0").rglob("*.dll")]
    roots += [mbin / n for n in ("libgtk-4-1.dll", "libadwaita-1-0.dll", "libgirepository-2.0-0.dll",
                                 "libgirepository-1.0-1.dll", "libgstreamer-1.0-0.dll", "libgstapp-1.0-0.dll",
                                 "libgstvideo-1.0-0.dll", "libgraphene-1.0-0.dll", "libpangocairo-1.0-0.dll",
                                 "libpangoft2-1.0-0.dll", "librsvg-2-2.dll", "gdbus.exe")
              if (mbin / n).exists()]
    for f in [r for r in roots if r.parent == mbin]:
        shutil.copy2(f, out / "bin" / f.name)
    dlls = dll_closure(roots, mbin)
    for d in dlls:
        shutil.copy2(d, out / "bin" / d.name)
    print(f"{len(dlls)} DLLs")

    # SplatWork's own code, found by the bundled Python through a .pth file (not PYTHONPATH, which would leak into
    # the GPU runtime's Python 3.10)
    copytree(REPO / "linux" / "splatworker", out / "app" / "splatworker", skip=["__pycache__"])
    copytree(REPO / "tools" / "splatprep", out / "tools" / "splatprep", skip=["__pycache__"])
    (out / "lib" / f"python{PY_VER}" / "site-packages" / "splatwork.pth").write_text("../../../app\n")
    for f in ("LICENSE", "NOTICE"):
        shutil.copy2(REPO / f, out / f)
    (out / "bundle.json").write_text(json.dumps({"version": a.version, "python": PY_VER}, indent=1))
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"bundle: {out} ({size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
