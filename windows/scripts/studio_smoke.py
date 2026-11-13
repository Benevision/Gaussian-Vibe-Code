"""Studio smoke test for Windows: opens the real SplatWork window (GTK 4 + libadwaita), loads a generated scene in
the viewer, walks the main pages, saves a screenshot of each through GTK's own renderer, then quits. Any exception fails the run.

    python windows/scripts/studio_smoke.py [--out DIR]

Runs under MSYS2 UCRT64 (gtk4, libadwaita, python-gobject; mesa for software OpenGL) in CI, and on Linux too.
Settings, runtime and projects go to a temporary folder.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(ROOT / "windows" / "smoke-out"))
ap.add_argument("--installed", action="store_true",
                help="test the installed app's code (run with the installed splatwork-cli.exe), not the repository's")
args = ap.parse_args()
if not args.installed:
    sys.path.insert(0, str(ROOT / "linux"))
    sys.path.insert(0, str(ROOT / "tools"))
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)

TMP = Path(tempfile.mkdtemp(prefix="splatwork-studio-"))
for var, sub in (("APPDATA", "roaming"), ("LOCALAPPDATA", "local"), ("XDG_CONFIG_HOME", "cfg"),
                 ("XDG_DATA_HOME", "data"), ("XDG_CACHE_HOME", "cache")):
    os.environ[var] = str(TMP / sub)
import json  # noqa: E402

for cfg in (TMP / "roaming" / "SplatWork", TMP / "cfg" / "splatworker"):   # Windows / Linux config folders
    cfg.mkdir(parents=True, exist_ok=True)
    (cfg / "studio.json").write_text(json.dumps({"projects_dir": str(TMP / "Projects")}))

import numpy as np  # noqa: E402

from splatworker.studio import formats  # noqa: E402
from splatworker.studio.splats import C0, GaussianCloud  # noqa: E402

rng = np.random.default_rng(0)
n = 50_000
d = rng.normal(size=(n, 3))
d /= np.linalg.norm(d, axis=1, keepdims=True)
rot = np.zeros((n, 4))
rot[:, 0] = 1
scene = TMP / "scene" / "sphere.ply"
scene.parent.mkdir(parents=True)
formats.save(GaussianCloud(d, np.full((n, 3), np.log(0.02)), rot, np.full(n, 3.0), (((d + 1) / 2 - 0.5) / C0)[:, None, :],
                           up=np.array([0, -1, 0], np.float32)), str(scene))

import gi  # noqa: E402

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import GLib, Gtk  # noqa: E402

from splatworker.gui.gtk.app import SplatWorkerApp  # noqa: E402
from splatworker.gui.gtk.window import MainWindow  # noqa: E402

failures: list = []


def shot(win: Gtk.Widget, name: str) -> None:
    w, h = win.get_width(), win.get_height()
    paintable = Gtk.WidgetPaintable.new(win)
    snap = Gtk.Snapshot()
    paintable.snapshot(snap, w, h)
    node = snap.to_node()
    if node is None:
        raise RuntimeError(f"{name}: nothing drawn")
    tex = win.get_native().get_renderer().render_texture(node, None)
    tex.save_to_png(str(OUT / f"{name}.png"))
    print("shot", name, f"{w}x{h}", flush=True)


class T(SplatWorkerApp):
    def do_activate(self):
        self.window = MainWindow(self, self.state)
        self.window.set_default_size(1400, 900)
        self.window.present()
        win = self.window
        print("renderer:", type(win.get_native().get_renderer()).__name__, flush=True)
        steps = [
            (1500, lambda: shot(win, "0-home")),
            (200, lambda: win.open_path(scene)),
            (4000, lambda: shot(win, "1-viewer")),
            (200, lambda: self.viewer_gl()),
            (200, lambda: win.show_section("projects")),
            (1000, lambda: shot(win, "2-projects")),
            (200, lambda: win.show_section("worker")),
            (1000, lambda: shot(win, "3-worker")),
            (200, lambda: win.show_section("devices")),
            (1000, lambda: shot(win, "4-devices")),
            (200, lambda: win.show_section("models")),
            (1000, lambda: shot(win, "5-models")),
            (200, lambda: win.show_section("inbox")),
            (1000, lambda: shot(win, "6-inbox")),
            (200, lambda: win.new_project("video", None)),
            (1500, lambda: shot(win, "7-new-project")),
            (200, lambda: self.quit()),
        ]
        delay = 0
        for ms, fn in steps:
            delay += ms
            GLib.timeout_add(delay, self._step, fn)
        GLib.timeout_add(120_000, self._timeout)

    def viewer_gl(self):
        vp = self.window.viewer_page
        info = getattr(vp.area, "gl_info", None) or getattr(vp, "gl_error", None)
        print("viewer GL:", info, flush=True)

    def _step(self, fn):
        try:
            fn()
        except Exception:  # noqa: BLE001
            failures.append(traceback.format_exc())
            print(failures[-1], flush=True)
        return False

    def _timeout(self):
        failures.append("timed out")
        self.quit()
        return False


import splatworker  # noqa: E402

print("splatworker from", Path(splatworker.__file__).parent, flush=True)
rc = T().run([sys.argv[0]])
if failures:
    print(f"FAIL: {len(failures)} step(s) failed", flush=True)
    sys.exit(1)
print("OK", flush=True)
sys.exit(rc)
