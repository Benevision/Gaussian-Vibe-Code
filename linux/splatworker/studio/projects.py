"""Project folders (same layout as the Mac, so a project opens on either machine) and recent files.

::

    <root>/<name>/project.json        options, stage durations, outputs, the last training job
    <root>/<name>/prep.log            preparation log
    <root>/<name>/dataset/            dataset.json, images/, masks/, splatprep-masks/<model>/, splatprep-sky/masks/,
                                      points3d.ply, priors.json
    <root>/<name>/outputs/<name>-<time>.ply  (+ .sky.png / .sky.json)
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .formats import OPENABLE_SUFFIXES

PROJECT_FILE = "project.json"


def default_root(env=None) -> Path:
    """``~/Documents/SplatStudio`` (the Flatpak has ``xdg-documents/SplatStudio:create``)."""
    env = os.environ if env is None else env
    docs = None
    try:
        import subprocess

        out = subprocess.run(["xdg-user-dir", "DOCUMENTS"], capture_output=True, text=True, timeout=2)
        if out.returncode == 0 and out.stdout.strip() and out.stdout.strip() != str(Path.home()):
            docs = Path(out.stdout.strip())
    except Exception:
        docs = None
    if docs is None:
        docs = Path(env.get("HOME") or Path.home()) / "Documents"
    return docs / "SplatStudio"


def safe_name(name: str) -> str:
    n = re.sub(r"[^\w .,()+-]", "_", name.strip())[:120].strip(" .")
    return n or "Untitled"


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H-%M-%S")


@dataclass
class Project:
    root: Path

    @property
    def name(self) -> str:
        return self.meta.get("name") or self.root.name

    @property
    def meta_file(self) -> Path:
        return self.root / PROJECT_FILE

    @property
    def dataset_dir(self) -> Path:
        return self.root / "dataset"

    @property
    def outputs_dir(self) -> Path:
        return self.root / "outputs"

    @property
    def log_file(self) -> Path:
        return self.root / "prep.log"

    @property
    def meta(self) -> Dict[str, Any]:
        try:
            return json.loads(self.meta_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def update(self, **fields) -> Dict[str, Any]:
        m = self.meta
        m.update(fields)
        tmp = self.meta_file.with_name(PROJECT_FILE + ".tmp")
        tmp.write_text(json.dumps(m, indent=1), encoding="utf-8")
        os.replace(tmp, self.meta_file)
        return m

    def log(self, line: str) -> None:
        with open(self.log_file, "a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%H:%M:%S')} {line}\n")

    @property
    def has_dataset(self) -> bool:
        return (self.dataset_dir / "dataset.json").is_file()

    def outputs(self) -> List[Path]:
        """Trained splats, newest first (previews excluded)."""
        if not self.outputs_dir.is_dir():
            return []
        out = [p for p in self.outputs_dir.iterdir()
               if p.suffix.lower() in OPENABLE_SUFFIXES and not p.name.startswith(".") and ".preview" not in p.name
               and not p.name.endswith(".part")]
        return sorted(out, key=lambda p: p.stat().st_mtime, reverse=True)

    def new_output_path(self) -> Path:
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        return self.outputs_dir / f"{safe_name(self.name)}-{timestamp()}.ply"

    def thumbnail(self) -> Optional[Path]:
        imgs = self.dataset_dir / "images"
        if imgs.is_dir():
            for p in sorted(imgs.iterdir()):
                if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    return p
        return None

    @classmethod
    def create(cls, parent: Path, name: str, options: Dict[str, Any]) -> "Project":
        base = Path(parent) / safe_name(name)
        root, k = base, 2
        while root.exists() and any(root.iterdir()):
            root = base.with_name(f"{base.name} {k}")
            k += 1
        (root / "dataset").mkdir(parents=True, exist_ok=True)
        (root / "outputs").mkdir(exist_ok=True)
        p = cls(root)
        p.update(format="splatstudio-project", version=1, name=name, createdAt=datetime.now().isoformat(),
                 options=options, app="Splat Studio for Linux")
        return p

    @classmethod
    def find(cls, path: Path) -> Optional["Project"]:
        """The project a file belongs to (a splat in ``outputs/``, the ``dataset/`` folder, …), or None."""
        p = Path(path).resolve()
        for d in [p] + list(p.parents)[:4]:
            if (d / PROJECT_FILE).is_file() or ((d / "dataset").is_dir() and (d / "outputs").is_dir()):
                return cls(d)
        return None


def list_projects(root: Path) -> List[Project]:
    root = Path(root)
    if not root.is_dir():
        return []
    out = [Project(d) for d in root.iterdir()
           if d.is_dir() and ((d / PROJECT_FILE).is_file() or (d / "dataset" / "dataset.json").is_file())]
    return sorted(out, key=lambda p: (p.meta_file if p.meta_file.exists() else p.root).stat().st_mtime, reverse=True)


class Recents:
    MAX = 12

    def __init__(self, path: Path):
        self.path = Path(path)

    def items(self) -> List[Path]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        return [Path(p) for p in data if isinstance(p, str) and Path(p).exists()][:self.MAX]

    def add(self, path: Path) -> None:
        p = str(Path(path).resolve())
        items = [str(x) for x in self.items() if str(x) != p]
        items.insert(0, p)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(items[:self.MAX]), encoding="utf-8")
