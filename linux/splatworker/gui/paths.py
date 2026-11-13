"""Filesystem layout of the desktop app.

Inside the Flatpak, ``XDG_CONFIG_HOME`` / ``XDG_DATA_HOME`` / ``XDG_CACHE_HOME`` point
at ``~/.var/app/com.example.SplatWorker/{config,data,cache}``, so nothing here needs
host filesystem access. Outside the Flatpak the usual XDG defaults apply.

::

    $XDG_CONFIG_HOME/splatworker/gui-settings.json   GUI settings
    $XDG_CONFIG_HOME/splatworker/token               bearer token (mode 600)
    $XDG_DATA_HOME/splatworker/python/               standalone CPython for the GPU runtime
    $XDG_DATA_HOME/splatworker/venv/                 venv with torch / gsplat / worker deps
    $XDG_DATA_HOME/splatworker/runtime.json          what the setup assistant installed
    $XDG_DATA_HOME/splatworker/install.log           last setup log
    $XDG_DATA_HOME/splatworker/data/                 default worker data dir (datasets, jobs)
    $XDG_CACHE_HOME/splatworker/{pip,tmp,downloads}  pip cache and scratch space
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

FLATPAK_LIB_DIR = Path("/app/share/splatworker/lib")
FLATPAK_SPLATPREP_DIR = Path("/app/share/splatworker/splatprep")


def _is_windows(override: Optional[bool] = None) -> bool:
    import sys

    return (sys.platform == "win32") if override is None else override


def in_flatpak(env: Optional[Mapping[str, str]] = None, root: Path = Path("/")) -> bool:
    env = os.environ if env is None else env
    return bool(env.get("FLATPAK_ID")) or (root / ".flatpak-info").is_file()


def _xdg(env: Mapping[str, str], name: str, default: str) -> Path:
    value = env.get(name)
    if value and os.path.isabs(value):
        return Path(value)
    return Path(env.get("HOME") or os.path.expanduser("~")) / default


def worker_lib_dir() -> Path:
    """Directory that contains the ``splatworker`` package (put on PYTHONPATH of children)."""
    import splatworker

    return Path(splatworker.__file__).resolve().parent.parent


def splatprep_dir(env: Optional[Mapping[str, str]] = None) -> Optional[Path]:
    """Directory containing the ``splatprep`` package, or None if it is not shipped.

    Order: ``$SPLATWORKER_SPLATPREP_DIR``, the Flatpak copy, the repo's ``../tools``.
    """
    env = os.environ if env is None else env
    candidates = []
    if env.get("SPLATWORKER_SPLATPREP_DIR"):
        candidates.append(Path(env["SPLATWORKER_SPLATPREP_DIR"]))
    candidates.append(FLATPAK_SPLATPREP_DIR)
    candidates.append(worker_lib_dir().parent / "tools")
    for c in candidates:
        if (c / "splatprep" / "__init__.py").is_file():
            return c
    return None


@dataclass(frozen=True)
class AppPaths:
    config_dir: Path
    data_root: Path
    cache_dir: Path

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None, windows: Optional[bool] = None) -> "AppPaths":
        env = os.environ if env is None else env
        if _is_windows(windows):
            # %APPDATA%\SplatWork (settings, token; roams), %LOCALAPPDATA%\SplatWork (runtime, data, cache)
            home = Path(env.get("USERPROFILE") or os.path.expanduser("~"))
            roaming = Path(env.get("APPDATA") or home / "AppData" / "Roaming")
            local = Path(env.get("LOCALAPPDATA") or home / "AppData" / "Local")
            return cls(config_dir=roaming / "SplatWork", data_root=local / "SplatWork",
                       cache_dir=local / "SplatWork" / "cache")
        return cls(
            config_dir=_xdg(env, "XDG_CONFIG_HOME", ".config") / "splatworker",
            data_root=_xdg(env, "XDG_DATA_HOME", ".local/share") / "splatworker",
            cache_dir=_xdg(env, "XDG_CACHE_HOME", ".cache") / "splatworker",
        )

    # config
    @property
    def settings_file(self) -> Path:
        return self.config_dir / "gui-settings.json"

    @property
    def token_file(self) -> Path:
        return self.config_dir / "token"

    # runtime
    @property
    def python_dir(self) -> Path:
        return self.data_root / "python"

    @property
    def python_exe(self) -> Path:
        if _is_windows():
            return self.python_dir / "python.exe"     # python-build-standalone's Windows layout
        return self.python_dir / "bin" / "python3"

    @property
    def venv_dir(self) -> Path:
        return self.data_root / "venv"

    @property
    def venv_python(self) -> Path:
        if _is_windows():
            return self.venv_dir / "Scripts" / "python.exe"
        return self.venv_dir / "bin" / "python"

    @property
    def local_wheels_dir(self) -> Path:
        """Wheels built on this PC (linux/flatpak/build-gsplat-wheel.sh); Set Up prefers them."""
        return self.data_root / "wheels"

    @property
    def venv_staging_dir(self) -> Path:
        """New runtimes are built here and swapped into ``venv_dir`` only once verified, so a failed or
        cancelled setup never breaks a working runtime."""
        return self.data_root / "venv.new"

    @property
    def runtime_state_file(self) -> Path:
        return self.data_root / "runtime.json"

    @property
    def install_log(self) -> Path:
        return self.data_root / "install.log"

    @property
    def default_worker_data_dir(self) -> Path:
        return self.data_root / "data"

    # cache
    @property
    def pip_cache(self) -> Path:
        return self.cache_dir / "pip"

    @property
    def tmp_dir(self) -> Path:
        return self.cache_dir / "tmp"

    @property
    def downloads_dir(self) -> Path:
        return self.cache_dir / "downloads"

    def ensure(self) -> None:
        for d in (self.config_dir, self.data_root, self.cache_dir):
            d.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.config_dir, 0o700)
        except OSError:
            pass
