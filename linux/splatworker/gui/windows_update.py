"""Updates for SplatWork for Windows (docs/dev/WINDOWS.md): the GitHub release's MSI.

* :func:`installed_version` — the installed app's version (``bundle.json`` next to ``bin\\``, written by
  ``windows/build/make_bundle.py``); None when not running from the Windows app folder.
* :func:`check` — the latest release, if newer: ``{"version", "msi_url", "msi_name", "sums_url", "page"}``.
* :func:`download` — the MSI into a folder, checked against the release's ``SHA256SUMS.txt``.
* :func:`install` — starts ``msiexec`` on it (a major upgrade: settings, the GPU runtime and projects are kept);
  the caller then quits so the files can be replaced.

``SPLATWORKER_UPDATE_URL`` overrides the API URL (an empty string disables checks), as on Linux.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import urllib.request
from pathlib import Path
from typing import Callable, Optional

from .updates import is_newer

RELEASES_API = "https://api.github.com/repos/Benevision/SplatWork/releases/latest"
MSI_SUFFIX = "-Windows-x64.msi"


def bundle_root(env=None) -> Optional[Path]:
    env = os.environ if env is None else env
    root = env.get("SPLATWORK_BUNDLE")
    return Path(root) if root and (Path(root) / "bundle.json").is_file() else None


def installed_version(env=None) -> Optional[str]:
    root = bundle_root(env)
    if root is None:
        return None
    try:
        return str(json.loads((root / "bundle.json").read_text(encoding="utf-8")).get("version") or "") or None
    except (OSError, ValueError):
        return None


def _get(url: str, timeout: float, opener=None):
    req = urllib.request.Request(url, headers={"User-Agent": "SplatWork-Windows", "Accept": "application/vnd.github+json"})
    return (opener or urllib.request.urlopen)(req, timeout=timeout)


def parse_release(release: dict, current: str) -> Optional[dict]:
    """The update described by a GitHub release (API JSON), if it is newer than ``current`` and has an MSI."""
    if release.get("draft") or release.get("prerelease"):
        return None
    version = str(release.get("tag_name") or "").lstrip("v")
    if not version or not is_newer(version, current):
        return None
    assets = {a.get("name", ""): a.get("browser_download_url", "") for a in release.get("assets") or []}
    msi = next((n for n in assets if n.endswith(MSI_SUFFIX)), None)
    if msi is None:
        return None
    return {"version": version, "msi_name": msi, "msi_url": assets[msi],
            "sums_url": assets.get("SHA256SUMS.txt"), "page": release.get("html_url", "")}


def check(current: Optional[str] = None, timeout: float = 8.0, opener=None) -> Optional[dict]:
    current = current or installed_version()
    url = os.environ.get("SPLATWORKER_UPDATE_URL", RELEASES_API)
    if not current or not url:
        return None
    try:
        with _get(url, timeout, opener) as r:
            return parse_release(json.loads(r.read(2 * 1024 * 1024).decode("utf-8")), current)
    except Exception:  # noqa: BLE001 — offline / rate-limited: no update shown
        return None


def expected_sha256(sums_text: str, name: str) -> Optional[str]:
    for line in sums_text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].lstrip("*") == name:
            return parts[0].lower()
    return None


def download(info: dict, dest_dir: Path, progress: Callable[[float], None] = lambda f: None,
             opener=None, timeout: float = 30.0) -> Path:
    """Download the MSI and verify it; raises RuntimeError with a readable message on failure."""
    if not info.get("sums_url"):
        raise RuntimeError("The release has no SHA256SUMS.txt, so the installer can't be verified.")
    with _get(info["sums_url"], timeout, opener) as r:
        sha = expected_sha256(r.read(1024 * 1024).decode("utf-8", "replace"), info["msi_name"])
    if not sha:
        raise RuntimeError(f"{info['msi_name']} is not listed in SHA256SUMS.txt.")
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / info["msi_name"]
    part = out.with_name(out.name + ".part")
    h = hashlib.sha256()
    with _get(info["msi_url"], timeout, opener) as r, open(part, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0) if getattr(r, "headers", None) else 0
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            h.update(chunk)
            done += len(chunk)
            if total:
                progress(min(done / total, 1.0))
    if h.hexdigest() != sha:
        part.unlink(missing_ok=True)
        raise RuntimeError("The downloaded installer doesn't match its checksum. Try again later.")
    os.replace(part, out)
    progress(1.0)
    return out


def install_argv(msi: Path) -> list:
    """msiexec with its normal progress window; Windows asks for administrator rights once."""
    return ["msiexec", "/i", str(msi), "/passive", "/norestart"]


def install(msi: Path) -> None:
    flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    subprocess.Popen(install_argv(msi), close_fds=True, creationflags=flags)
