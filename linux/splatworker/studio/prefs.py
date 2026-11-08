"""Studio preferences (``$XDG_CONFIG_HOME/splatworker/studio.json``): projects folder, queue policy, viewer and
navigation defaults, iPhone receiver. Lenient: unknown keys are ignored, bad values fall back to defaults."""
from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List


@dataclass
class StudioPrefs:
    projects_dir: str = ""              # "" = ~/Documents/SplatWork
    local_jobs_first: bool = True       # Studio jobs go ahead of queued remote jobs (never pre-empt)
    rotate_speed: float = 1.0           # nav.rotateSpeed
    pan_speed: float = 1.0              # nav.panSpeed
    zoom_speed: float = 1.0             # nav.zoomSpeed
    fly_speed: float = 1.0              # nav.flySpeed
    invert_vertical: bool = False       # nav.invertVertical
    fov: float = 60.0                   # viewer.fov
    background: List[float] = dataclasses.field(default_factory=lambda: [0.06, 0.06, 0.07])   # viewer.background
    render_scale: float = 1.0           # viewer.renderScale
    show_cameras: bool = True           # viewer.showCameras
    receive_captures: bool = False      # net.receive (off until turned on: listens on every interface)
    require_pairing: bool = True        # net.pairing
    capture_port: int = 7879

    @classmethod
    def load(cls, path: Path) -> "StudioPrefs":
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls()
        p = cls()
        if not isinstance(data, dict):
            return p
        for f in dataclasses.fields(cls):
            if f.name in data:
                v: Any = data[f.name]
                default = getattr(p, f.name)
                if isinstance(default, bool):
                    ok = isinstance(v, bool)
                elif isinstance(default, (int, float)):
                    ok = isinstance(v, (int, float)) and not isinstance(v, bool)
                    v = type(default)(v) if ok else v
                elif isinstance(default, list):
                    ok = isinstance(v, list) and len(v) == len(default)
                else:
                    ok = isinstance(v, str)
                if ok:
                    setattr(p, f.name, v)
        return p

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(dataclasses.asdict(self), indent=1), encoding="utf-8")
        os.replace(tmp, path)
