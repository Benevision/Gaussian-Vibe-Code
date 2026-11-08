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
            (300, lambda: self.coverage_disabled()),
            (300, lambda: (self.window.viewer_page.mode_row.set_selected(0), self.window.viewer_page.insp_stack.set_visible_child_name("edit"))),
            (1500, lambda: shot("3-edit")),
            (300, lambda: self.brush()),
            (1500, lambda: shot("3b-deleted")),
            (300, lambda: self.window.viewer_page.undo()),
            (300, lambda: self.offscreen()),
            (300, lambda: self.fake_run()),
            (300, lambda: self.coverage_project()),
            (4000, lambda: (self.window.viewer_page.insp_stack.set_visible_child_name("view"),
                            self.window.viewer_page.mode_row.set_selected(5))),
            (8000, lambda: self.coverage_check()),
            (300, lambda: shot("2b-coverage")),
            (300, lambda: self.window.viewer_page.mode_row.set_selected(0)),
            (300, lambda: self.turntable()),
            (2500, lambda: shot("2c-turntable")),
            (15000, lambda: self.turntable_check()),
            (300, lambda: self.window.show_section("projects")),
            (300, lambda: self.window.projects_page.select(self.proj.root)),
            (1500, lambda: shot("4-projects")),
            (300, lambda: self.fake_pause()),
            (1200, lambda: shot("4b-projects-paused")),
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
            (300, lambda: (self.fake_benchmarks(), self.window.show_section("models"))),
            (1200, lambda: shot("7-models")),
            (300, lambda: self.window.show_section("reference")),
            (1200, lambda: shot("8-reference")),
            (300, lambda: self.studio_prefs()),
            (1500, lambda: shot("10-preferences-studio")),
            (300, lambda: self.studio_prefs_check()),
            (500, lambda: self.window.new_project("video", None)),
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

    def coverage_disabled(self):
        vp = self.window.viewer_page
        assert vp.coverage_unavailable() is not None        # a loose splat file: no cameras
        vp.mode_row.set_selected(5)
        assert vp._mode == "depth" and vp.area.settings.mode == "depth", vp._mode
        print("coverage disabled without a project ok", flush=True)

    def coverage_project(self):
        """Give the Garden project a dataset (cameras on one side of the sphere) and open its output."""
        import math
        cams = []
        for i, a in enumerate(np.linspace(-1.2, 1.2, 24)):
            eye = np.array([4 * math.sin(a), -0.5, -4 * math.cos(a)])
            f = -eye / np.linalg.norm(eye)
            r = np.cross(f, [0, 1.0, 0]); r /= np.linalg.norm(r)
            dn = np.cross(f, r)
            m = np.eye(4); m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = r, dn, f, eye
            cams.append({"id": i, "image": f"images/{i:04d}.jpg", "width": 640, "height": 480, "fx": 500.0,
                         "fy": 500.0, "cx": 320.0, "cy": 240.0, "cameraToWorld": m.tolist()})
        (self.proj.dataset_dir / "dataset.json").write_text(json.dumps({"version": 1, "cameras": cams}))
        out = self.proj.outputs_dir / "Garden-2026-11-08T10-00-00.ply"
        self.window.open_path(out)

    def coverage_check(self):
        vp = self.window.viewer_page
        assert vp.coverage_unavailable() is None, vp.coverage_unavailable()
        assert vp._mode == "coverage" and vp.area.renderer.has_coverage, (vp._mode, vp.cov_info.get_text())
        print("coverage:", vp.cov_info.get_text(), flush=True)

    def turntable(self):
        vp = self.window.viewer_page
        vp.insp_stack.set_visible_child_name("export")
        vp.tt_res, vp.tt_secs, vp.tt_fps = 720, 1, 12
        import shutil
        shutil.rmtree(f"{OUT}/turntable-frames", ignore_errors=True)
        vp.start_turntable(__import__("pathlib").Path(f"{OUT}/turntable.mp4"))
        assert vp._tt is not None and vp.area.suspended

    def turntable_check(self):
        vp = self.window.viewer_page
        assert vp._tt is None and not vp.area.suspended, "turntable still running"
        out = __import__("pathlib").Path(OUT)
        made = [p for p in (out / "turntable.mp4", out / "turntable.webm") if p.is_file()]
        frames = sorted((out / "turntable-frames").glob("*.png")) if (out / "turntable-frames").is_dir() else []
        assert made or len(frames) == 12, (made, len(frames))
        print("turntable:", made or f"{len(frames)} PNG frames", flush=True)

    def fake_benchmarks(self):
        from splatworker.studio import benchmark
        cfg = self.state.paths.config_dir
        benchmark.save_benchmark(cfg, {"deviceId": "local", "name": "cachyos-pc", "date": "2026-11-08T10:00:00Z",
                                       "gpu": "NVIDIA GeForce RTX 4080", "software": "SplatWork Node 1.4.0",
                                       "trainStepsPerSecond": 41.7, "splats": 1000000, "megapixels": 1.92,
                                       "uploadMBps": None, "downloadMBps": None})
        benchmark.save_benchmark(cfg, {"deviceId": "http://192.168.0.20:7878", "name": "Office PC",
                                       "date": "2026-11-07T18:30:00Z", "gpu": "NVIDIA GeForce RTX 3060",
                                       "trainStepsPerSecond": 17.2, "splats": 1000000, "megapixels": 1.92,
                                       "uploadMBps": 92.4, "downloadMBps": 101.0})
        mp = self.window.models_page
        assert hasattr(mp, "refresh_benchmarks"), mp           # not a BrokenPage
        mp.refresh_benchmarks()
        print("benchmarks table rows:", len(mp.benchmark_rows(benchmark.sorted_results(cfg))), flush=True)

    def studio_prefs(self):
        from splatworker.gui.gtk.preferences import Preferences
        self.prefs_dlg = Preferences(self.window, self.state, lambda n: None, lambda e: None, lambda: None,
                                     studio=self.window.studio)
        self.prefs_dlg.present()
        self.prefs_dlg.show_studio()

    def studio_prefs_check(self):
        """Changes made in Preferences → Studio reach the controller, studio.json and the pages using them."""
        import json as _json
        ctl = self.window.studio
        sp = self.prefs_dlg.studio_page
        assert sp is not None
        rows = self.prefs_dlg._studio_prefs
        rows.recv_row.set_active(True)
        assert ctl.prefs.receive_captures and self.window.inbox_page.recv_switch.get_active()
        assert ctl.receiver is not None and ctl.receiver.running, ctl.receiver and ctl.receiver.error
        rows.recv_row.set_active(False)
        assert not ctl.prefs.receive_captures and not self.window.inbox_page.recv_switch.get_active()
        assert not ctl.receiver.running
        rows.fov_row.set_value(75)
        assert self.window.viewer_page.fov_row.get_value() == 75 and self.window.viewer_page.area.camera.fov_y == 75
        saved = _json.loads(ctl.prefs_path.read_text())
        assert saved["fov"] == 75 and saved["receive_captures"] is False, saved
        self.prefs_dlg.dialog.close()
        print("preferences studio page ok", flush=True)

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
        self.run = run
        self.window.studio.changed()

    def fake_pause(self):
        page = self.window.projects_page
        assert page.panel.pause_btn.get_label() == "Pause" and page.panel.ckpt_btn.get_visible()
        self.run.handle("job", {"kind": "train", "state": "paused", "step": 9000, "totalSteps": 30000,
                                "message": "paused at step 9,000", "checkpointStep": 9000,
                                "numGaussians": 1_350_000, "elapsedSeconds": 300, "gpu": "NVIDIA GeForce RTX 4080"})
        assert page.panel.pause_btn.get_label() == "Resume", page.panel.pause_btn.get_label()
        from splatworker.gui.gtk.widgets import JobRow
        noop = lambda *a: None  # noqa: E731
        row = JobRow("j1", noop, noop, noop, noop, noop, noop)
        row.update({"jobId": "j1", "kind": "train", "state": "training", "step": 5, "totalSteps": 10})
        assert row.pause_btn.get_visible() and not row.resume_btn.get_visible()
        row.update({"jobId": "j1", "kind": "train", "state": "paused", "step": 5, "totalSteps": 10,
                    "message": "paused at step 5"})
        assert row.resume_btn.get_visible() and not row.pause_btn.get_visible() and row.cancel_btn.get_visible()
        assert "paused at step 5" in row.get_subtitle()
        row.update({"jobId": "j1", "kind": "process", "state": "training", "step": 5, "totalSteps": 10})
        assert not row.pause_btn.get_visible()          # running processing jobs can't pause
        print("paused panel ok", flush=True)

T().run([sys.argv[0]])
