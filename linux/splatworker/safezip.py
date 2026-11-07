"""Defensive zip extraction for uploaded datasets.

Rejects: absolute paths, ``..`` components, backslashes / drive letters,
symlinks and other non-regular entries, encrypted entries, too many files,
too much uncompressed data (counted while extracting, not trusted from the
header) and suspicious per-entry compression ratios. ``__MACOSX/`` and
``.DS_Store`` entries are skipped.
"""

from __future__ import annotations

import os
import stat
import zipfile
from pathlib import Path, PurePosixPath


class UnsafeArchive(ValueError):
    pass


def _check_name(name: str) -> PurePosixPath:
    if not name or "\x00" in name:
        raise UnsafeArchive("empty or invalid entry name")
    if "\\" in name:
        raise UnsafeArchive(f"backslash in entry name: {name!r}")
    if name.startswith("/") or (len(name) > 1 and name[1] == ":"):
        raise UnsafeArchive(f"absolute path in archive: {name!r}")
    p = PurePosixPath(name)
    if any(part == ".." for part in p.parts):
        raise UnsafeArchive(f"path traversal in archive: {name!r}")
    return p


def _skip(p: PurePosixPath) -> bool:
    return (bool(p.parts) and p.parts[0] == "__MACOSX") or p.name == ".DS_Store" or p.name.startswith("._")


def safe_extract(zip_path, dest, *, max_bytes: int, max_files: int,
                 max_ratio: float = 200.0) -> int:
    """Extract ``zip_path`` into ``dest`` (must be empty/new). Returns bytes written."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    dest_real = dest.resolve()
    total = 0
    try:
        zf = zipfile.ZipFile(zip_path)
    except (zipfile.BadZipFile, OSError) as e:
        raise UnsafeArchive(f"not a valid zip archive: {e}")
    with zf:
        infos = zf.infolist()
        if len(infos) > max_files:
            raise UnsafeArchive(f"archive has {len(infos)} entries (limit {max_files})")
        # validate everything before writing anything
        for info in infos:
            p = _check_name(info.filename)
            mode = (info.external_attr >> 16) & 0xFFFF
            if mode and stat.S_ISLNK(mode):
                raise UnsafeArchive(f"symlink in archive: {info.filename!r}")
            if mode and not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)) and stat.S_IFMT(mode) != 0:
                raise UnsafeArchive(f"special file in archive: {info.filename!r}")
            if info.flag_bits & 0x1:
                raise UnsafeArchive("encrypted archives are not supported")
            target = (dest_real / Path(*p.parts)).resolve() if p.parts else dest_real
            if target != dest_real and dest_real not in target.parents:
                raise UnsafeArchive(f"path escapes destination: {info.filename!r}")
        declared = sum(i.file_size for i in infos)
        if declared > max_bytes:
            raise UnsafeArchive(f"archive expands to {declared} bytes (limit {max_bytes})")

        for info in infos:
            p = PurePosixPath(info.filename)
            if _skip(p) or not p.parts:
                continue
            target = dest_real / Path(*p.parts)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            # refuse to write through anything that is not a plain directory
            for parent in target.relative_to(dest_real).parents:
                full = dest_real / parent
                if full.is_symlink():
                    raise UnsafeArchive(f"symlink in extraction path: {parent}")
            if target.exists() or target.is_symlink():
                raise UnsafeArchive(f"duplicate entry: {info.filename!r}")
            written = 0
            limit = max(info.compress_size * max_ratio, 1024 * 1024)
            with zf.open(info) as src, open(target, "xb") as out:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    written += len(chunk)
                    total += len(chunk)
                    if total > max_bytes:
                        raise UnsafeArchive(f"archive expands beyond {max_bytes} bytes")
                    if written > limit:
                        raise UnsafeArchive(f"suspicious compression ratio: {info.filename!r}")
                    out.write(chunk)
            os.chmod(target, 0o644)
    return total
