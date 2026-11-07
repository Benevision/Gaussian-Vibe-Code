"""Update check against the Flatpak repository published by GitHub Actions.

The workflow (``.github/workflows/splat-worker-flatpak.yml``) publishes, next to the OSTree repo,
``version.json`` = ``{"version": "1.1.0", "commit": "...", "date": "..."}``. The app compares it with
its own version and offers ``flatpak update``; Flatpak itself (GNOME Software / Discover / a
``flatpak update`` cron) does the actual update. Set ``SPLATWORKER_UPDATE_URL`` to override, or to
an empty string to disable checks.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from typing import Optional, Tuple

from .. import __version__

DEFAULT_URL = "https://benevision.github.io/Gaussian-Vibe-Code/flatpak/version.json"
APP_ID = "com.example.SplatWorker"
UPDATE_COMMAND = f"flatpak update --user {APP_ID}"


def update_url() -> str:
    return os.environ.get("SPLATWORKER_UPDATE_URL", DEFAULT_URL)


def parse_version(v: str) -> Tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", v)[:4]) or (0,)


def is_newer(remote: str, local: str = __version__) -> bool:
    return parse_version(remote) > parse_version(local)


def check(timeout: float = 6.0) -> Optional[dict]:
    """The published version info if it is newer than this app, else None (also on any error)."""
    url = update_url()
    if not url:
        return None
    try:
        req = urllib.request.Request(url, headers={"User-Agent": f"SplatWorker/{__version__}", "Cache-Control": "no-cache"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            info = json.loads(r.read(64 * 1024).decode("utf-8"))
        v = str(info.get("version", ""))
        return info if v and is_newer(v) else None
    except Exception:
        return None
