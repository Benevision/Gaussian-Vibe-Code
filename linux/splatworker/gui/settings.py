"""GUI settings (JSON in ``$XDG_CONFIG_HOME/splatworker/gui-settings.json``) and the token.

The settings map onto ``splatworker serve`` flags; :meth:`AppSettings.serve_args` builds
the command line, so the desktop app and the headless entry point
(``flatpak run com.example.SplatWorker --headless``) start exactly the same server.
"""

from __future__ import annotations

import dataclasses
import ipaddress
import json
import os
import secrets
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .paths import AppPaths

SETTINGS_VERSION = 1


@dataclass
class AppSettings:
    port: int = 7878
    name: str = ""                      # Bonjour name; "" = this computer's hostname
    data_dir: str = ""                  # "" = AppPaths.default_worker_data_dir
    require_token: bool = True
    mdns: bool = True
    advertise_ip: str = ""              # "" = auto-detect
    gpus: Optional[List[int]] = None    # None = every GPU
    max_upload_gb: float = 20.0
    max_extracted_gb: float = 40.0
    start_server_on_launch: bool = True
    run_in_background: bool = True
    autostart: bool = False
    log_level: str = "info"
    # Extra components the setup assistant installed (informational, kept here so the
    # "repair" button can reinstall the same set).
    with_splatprep: bool = True
    with_lightglue: bool = False

    # ------------------------------------------------------------ persistence
    @classmethod
    def from_dict(cls, data: Any) -> "AppSettings":
        """Lenient: unknown keys are dropped, invalid values fall back to the default."""
        s = cls()
        if not isinstance(data, dict):
            return s
        for f in dataclasses.fields(cls):
            if f.name not in data:
                continue
            candidate = dataclasses.replace(s, **{f.name: data[f.name]})
            if not candidate.field_errors(only=f.name):
                s = candidate
        return s

    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        d["version"] = SETTINGS_VERSION
        return d

    @classmethod
    def load(cls, path: Path) -> "AppSettings":
        try:
            return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        except (OSError, ValueError):
            return cls()

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=1, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)

    # ------------------------------------------------------------- validation
    def field_errors(self, only: Optional[str] = None) -> Dict[str, str]:
        errors: Dict[str, str] = {}

        def check(name: str, ok: bool, msg: str) -> None:
            if (only is None or only == name) and not ok:
                errors[name] = msg

        def is_int(v: Any) -> bool:
            return isinstance(v, int) and not isinstance(v, bool)

        def is_num(v: Any) -> bool:
            return isinstance(v, (int, float)) and not isinstance(v, bool)

        def is_bool(v: Any) -> bool:
            return isinstance(v, bool)

        check("port", is_int(self.port) and 1024 <= self.port <= 65535,
              "Port must be a whole number between 1024 and 65535.")
        check("name", isinstance(self.name, str) and len(self.name) <= 63 and "." not in self.name,
              "The name can be at most 63 characters and cannot contain dots.")
        check("data_dir", isinstance(self.data_dir, str), "Invalid folder.")
        ip_ok = isinstance(self.advertise_ip, str)
        if ip_ok and self.advertise_ip:
            try:
                ip_ok = isinstance(ipaddress.ip_address(self.advertise_ip), ipaddress.IPv4Address)
            except ValueError:
                ip_ok = False
        check("advertise_ip", ip_ok, "Enter an IPv4 address such as 192.168.1.20, or leave it empty.")
        gpus_ok = self.gpus is None or (
            isinstance(self.gpus, list) and len(self.gpus) > 0
            and all(is_int(g) and g >= 0 for g in self.gpus) and len(set(self.gpus)) == len(self.gpus))
        check("gpus", gpus_ok, "Select at least one GPU.")
        check("max_upload_gb", is_num(self.max_upload_gb) and 0.1 <= self.max_upload_gb <= 10000,
              "Maximum upload size must be between 0.1 and 10000 GB.")
        check("max_extracted_gb", is_num(self.max_extracted_gb) and 0.1 <= self.max_extracted_gb <= 20000,
              "Maximum unpacked size must be between 0.1 and 20000 GB.")
        check("log_level", self.log_level in ("critical", "error", "warning", "info", "debug"),
              "Unknown log level.")
        for name in ("require_token", "mdns", "start_server_on_launch", "run_in_background",
                     "autostart", "with_splatprep", "with_lightglue"):
            check(name, is_bool(getattr(self, name)), "Must be on or off.")
        return errors

    # ---------------------------------------------------------------- derived
    def effective_name(self) -> str:
        return self.name.strip() or (socket.gethostname().split(".")[0] or "splatworker")

    def effective_data_dir(self, paths: AppPaths) -> Path:
        return Path(self.data_dir).expanduser() if self.data_dir.strip() else paths.default_worker_data_dir

    def serve_args(self, paths: AppPaths) -> List[str]:
        """Arguments after ``splatworker serve``. The token goes via a file, never argv."""
        args = ["--port", str(self.port),
                "--data-dir", str(self.effective_data_dir(paths)),
                "--name", self.effective_name(),
                "--max-upload-gb", repr(float(self.max_upload_gb)),
                "--max-extracted-gb", repr(float(self.max_extracted_gb)),
                "--log-level", self.log_level]
        if self.require_token:
            args += ["--token-file", str(paths.token_file)]
        if self.gpus is not None:
            args += ["--gpus", ",".join(str(g) for g in self.gpus)]
        if not self.mdns:
            args.append("--no-mdns")
        if self.advertise_ip:
            args += ["--advertise-ip", self.advertise_ip]
        return args


# ---------------------------------------------------------------------------
# Token
# ---------------------------------------------------------------------------

class TokenStore:
    """Bearer token in a mode-600 file (same format as ``splatworker gen-token --out``)."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def read(self) -> Optional[str]:
        try:
            value = self.path.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        return value or None

    def write(self, token: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(token + "\n")
        os.replace(tmp, self.path)
        os.chmod(self.path, 0o600)

    def get_or_create(self) -> str:
        token = self.read()
        if token is None:
            token = self.regenerate()
        return token

    def regenerate(self) -> str:
        token = secrets.token_urlsafe(32)
        self.write(token)
        return token
