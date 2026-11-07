"""Start at login / keep running in the background.

Inside the Flatpak this goes through the XDG Background portal
(``org.freedesktop.portal.Background.RequestBackground``): the portal asks the user
once, then writes ``~/.config/autostart/com.example.SplatWorker.desktop`` itself, and
lists the app under "Background Apps" while it runs without a window. Outside the
Flatpak (running from a checkout) the autostart file is written directly.

This module builds the portal arguments and the desktop file; the D-Bus call itself is
in :mod:`splatworker.gui.gtk.portal` (needs Gio).
"""

from __future__ import annotations

import os
import re
import secrets
import shlex
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import APP_ID, APP_NAME

PORTAL_BUS = "org.freedesktop.portal.Desktop"
PORTAL_PATH = "/org/freedesktop/portal/desktop"
BACKGROUND_IFACE = "org.freedesktop.portal.Background"
REQUEST_IFACE = "org.freedesktop.portal.Request"

BACKGROUND_ARG = "--background"


def handle_token() -> str:
    return "splatworker_" + secrets.token_hex(6)


def request_path(unique_name: str, token: str) -> str:
    """Object path of the portal Request for ``token`` (sender ``:1.42`` -> ``1_42``)."""
    sender = unique_name.lstrip(":").replace(".", "_")
    return f"/org/freedesktop/portal/desktop/request/{sender}/{token}"


def background_options(autostart: bool, token: str, reason: Optional[str] = None,
                       commandline: Optional[List[str]] = None) -> Dict[str, Tuple[str, Any]]:
    """``a{sv}`` options as {key: (gvariant type, value)}."""
    return {
        "handle_token": ("s", token),
        "reason": ("s", reason or "Keep the Splat Worker server running so your Mac can use this GPU."),
        "autostart": ("b", bool(autostart)),
        "commandline": ("as", commandline or ["splatworker-app", BACKGROUND_ARG]),
        "dbus-activatable": ("b", False),
    }


def parse_background_response(code: int, results: Dict[str, Any]) -> Tuple[bool, bool, str]:
    """(background allowed, autostart enabled, message) from a portal Response."""
    if code == 1:
        return False, False, "You declined the request."
    if code != 0:
        return False, False, "The request failed."
    bg = bool(results.get("background", False))
    auto = bool(results.get("autostart", False))
    if not bg:
        return False, auto, "Running in the background was not allowed. Enable it in System Settings → Apps."
    return True, auto, "OK"


def desktop_entry(exec_argv: List[str]) -> str:
    return "\n".join([
        "[Desktop Entry]",
        "Type=Application",
        f"Name={APP_NAME}",
        f"Exec={' '.join(shlex.quote(a) for a in exec_argv)}",
        f"Icon={APP_ID}",
        "X-GNOME-Autostart-enabled=true",
        "NoDisplay=true",
        "",
    ])


def autostart_file(config_home: Optional[Path] = None) -> Path:
    base = config_home or Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return Path(base) / "autostart" / f"{APP_ID}.desktop"


def set_autostart_file(enabled: bool, exec_argv: List[str], config_home: Optional[Path] = None) -> Path:
    path = autostart_file(config_home)
    if enabled:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(desktop_entry(exec_argv), encoding="utf-8")
    else:
        path.unlink(missing_ok=True)
    return path


def autostart_file_enabled(config_home: Optional[Path] = None) -> bool:
    path = autostart_file(config_home)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return False
    return not re.search(r"^X-GNOME-Autostart-enabled=false", text, re.M) and \
        not re.search(r"^Hidden=true", text, re.M)
