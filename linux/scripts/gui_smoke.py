"""GUI smoke run: opens the Studio window, loads a generated scene, walks every page, simulates a training run and
takes screenshots (``import -window root``) into $SMOKE_OUT (default /out). Used in a Debian container with Xvfb and
Mesa; on the PC: ``SMOKE_OUT=/tmp/smoke python3 linux/scripts/gui_smoke.py`` from the repo (needs PyGObject, GTK 4,
libadwaita, ImageMagick)."""
import json, os, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "linux")); sys.path.insert(0, str(ROOT / "tools"))
os.environ.setdefault("XDG_CONFIG_HOME", "/tmp/x/cfg"); os.environ.setdefault("XDG_DATA_HOME", "/tmp/x/data")
os.environ["HOME"] = "/tmp/x/home"; os.makedirs("/tmp/x/home/Documents", exist_ok=True)
import numpy as np
from PIL import Image
from splatworker.studio import formats
from splatworker.studio.splats import GaussianCloud, C0, sky_paths

# scene: a coloured sphere shell + a floor, with a skybox
rng = np.random.default_rng(0)
n = 200_000
d = rng.normal(size=(n, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
pos = np.concatenate([d * 1.0, np.c_[rng.uniform(-3, 3, (n, 1)), np.full((n, 1), 1.0), rng.uniform(-3, 3, (n, 1))]])
col = np.concatenate([(d + 1) / 2, np.tile([0.4, 0.35, 0.3], (n, 1))])
N = len(pos)
rot = np.zeros((N, 4)); rot[:, 0] = 1
g = GaussianCloud(pos, np.full((N, 3), np.log(0.012)), rot, np.full(N, 3.0), ((col - 0.5) / C0)[:, None, :],
                  up=np.array([0, -1, 0], np.float32))
os.makedirs("/tmp/x/scene", exist_ok=True)
formats.save(g, "/tmp/x/scene/sphere.ply")
sky = np.zeros((64, 128, 3), np.uint8); sky[:32] = (90, 140, 220); sky[32:] = (60, 50, 40)
png, meta = sky_paths("/tmp/x/scene/sphere.ply")
Image.fromarray(sky).save(png)
meta.write_text(json.dumps({"up": [0, -1, 0], "forward": [0, 0, 1], "right": [-1, 0, 0]}))

import gi
gi.require_version("Gtk", "4.0"); gi.require_version("Adw", "1")
from gi.repository import GLib
from splatworker.gui.gtk.app import SplatWorkerApp
from splatworker.gui.gtk.window import MainWindow
OUT = os.environ.get("SMOKE_OUT", "/out")

def shot(name):
    subprocess.run(["import", "-window", "root", f"{OUT}/{name}.png"], check=False)
    print("shot", name, flush=True)

class T(SplatWorkerApp):
    def do_activate(self):
        self.window = MainWindow(self, self.state)
        self.window.set_default_size(1400, 900)
        self.window.present()
        steps = [
            (1500, lambda: self.window.open_path(__import__("pathlib").Path("/tmp/x/scene/sphere.ply"))),
            (5000, lambda: shot("1-viewer")),
            (300, lambda: self.window.viewer_page.mode_row.set_selected(1)),
            (1500, lambda: shot("2-depth")),
            (300, lambda: (self.window.viewer_page.mode_row.set_selected(0), self.window.viewer_page.insp_stack.set_visible_child_name("edit"))),
            (1500, lambda: shot("3-edit")),
            (300, lambda: self.brush()),
            (1500, lambda: shot("3b-deleted")),
            (300, lambda: self.window.viewer_page.undo()),
            (300, lambda: self.offscreen()),
            (300, lambda: self.fake_run()),
            (300, lambda: self.window.show_section("projects")),
            (300, lambda: self.window.projects_page.select(self.proj.root)),
            (1500, lambda: shot("4-projects")),
            (300, lambda: self.window.show_section("worker")),
            (1200, lambda: shot("5-worker")),
            (300, lambda: self.window.show_section("inbox")),
            (1200, lambda: shot("6-inbox")),
            (300, lambda: self.window.studio.devices.add_manual("192.0.2.10", 7878, "Lab PC", check=False)),
            (300, lambda: self.window.show_section("devices")),
            (1500, lambda: shot("6b-devices")),
            (300, lambda: self.window.devices_page.add_dialog()),
            (1200, lambda: shot("6c-add-worker")),
            (300, lambda: self.window.devices_page.add_dialog_widget.close()),
            (300, lambda: self.window.show_section("models")),
            (1200, lambda: shot("7-models")),
            (300, lambda: self.window.show_section("reference")),
            (1200, lambda: shot("8-reference")),
            (300, lambda: self.window.new_project("video", None)),
            (1500, lambda: shot("9-newproject")),
            (300, lambda: self.focus_train_on()),
            (1200, lambda: shot("9b-newproject-train-on")),
            (500, lambda: self.quit()),
        ]
        t = 0
        for dt, fn in steps:
            t += dt
            GLib.timeout_add(t, lambda f=fn: (f(), False)[1])

    def focus_train_on(self):
        from gi.repository import Gtk
        from splatworker.gui.gtk.studio.new_project import NewProjectDialog
        for w in Gtk.Window.list_toplevels():
            if isinstance(w, NewProjectDialog):
                w.train_on_row.grab_focus()
                print("train on choices", [w.train_on_row.get_model().get_string(i)
                                           for i in range(w.train_on_row.get_model().get_n_items())], flush=True)

    def brush(self):
        vp = self.window.viewer_page
        a = vp.area
        a.edit.select_brush(*a._projected(), a.get_width() / 2, a.get_height() / 2, 80)
        a.refresh_flags()
        n = int(a.edit.selected.sum())
        vp._edit(lambda e: e.delete_selected())
        print("brush selected+deleted", n, "visible now", a.edit.visible_count, flush=True)
        assert n > 0 and a.edit.visible_count == a.edit.cloud.count - n

    def offscreen(self):
        img = self.window.viewer_page.area.screenshot(640, 360)
        Image.fromarray(img[..., :3]).save(f"{OUT}/offscreen-640x360.png")
        print("offscreen", img.shape, "mean", img[..., :3].mean(), flush=True)

    def fake_run(self):
        from splatworker.gui.gtk.studio.controller import Run
        from splatworker.studio.presets import ProjectOptions
        from splatworker.studio.projects import Project
        root = self.window.studio.projects_root
        o = ProjectOptions(name="Garden", source_kind="video", source="/tmp/x/garden.mp4", mask_models=["person"])
        self.proj = Project.create(root, "Garden", o.to_dict())
        import shutil
        shutil.copy("/tmp/x/scene/sphere.ply", self.proj.outputs_dir / "Garden-2026-11-08T10-00-00.ply")

        class P:      # stands in for LocalPipeline
            project = self.proj
            _titles = ["Extracting frames", "Masks (person)", "Finding the sky", "Solving cameras",
                       "Finalising dataset", "Training"]
            durations = {"Extracting frames": 42.0, "Masks (person)": 65.0, "Finding the sky": 12.0,
                         "Solving cameras": 140.0, "Finalising dataset": 0.4}
            def cancel(self): pass
        run = Run(P())
        self.window.studio.runs[str(self.proj.root)] = run
        run.handle("stage", {"index": 5, "count": 6, "title": "Training"})
        for step in range(500, 9001, 500):
            run.handle("job", {"kind": "train", "state": "training", "step": step, "totalSteps": 30000,
                               "psnr": 18 + 6 * (1 - 2.7 ** (-step / 4000)), "evalPsnr": (17 + 5 * (1 - 2.7 ** (-step / 4000))) if step % 2000 == 0 else None,
                               "numGaussians": min(1_500_000, step * 150), "elapsedSeconds": step / 30,
                               "etaSeconds": (30000 - step) / 30, "gpu": "NVIDIA GeForce RTX 4080"})
        run.handle("log", "== Training")
        self.window.studio.changed()

T().run([sys.argv[0]])
