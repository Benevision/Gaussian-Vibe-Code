"""New Project dialog: source → preparation → cameras & masks → training (the Mac's NewProjectSheet)."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from gi.repository import Adw, GLib, Gtk

from ....studio.presets import FEATURES, PRESETS, SOURCE_KINDS, ProjectOptions
from ....studio.projects import Project
from .devices_page import train_on_row
from .rows import combo_row, group, plain_row, spin_row, switch_row, training_settings_expander

FOLDER_SOURCES = ("photos", "colmap", "dataset")
VIDEO_SOURCES = ("video", "video360")


class NewProjectDialog(Adw.Window):
    def __init__(self, window, controller, kind: Optional[str], source: Optional[str],
                 on_created: Callable[[Project, ProjectOptions], None]):
        super().__init__(transient_for=window, modal=True, title="New Project", default_width=640,
                         default_height=780)
        self.ctl = controller
        self.on_created = on_created
        self.o = ProjectOptions(name="", source_kind=kind or "photos", source=source or "")
        if self.o.source_kind == "video360":
            self.o.max_image_size = 1600
        if self.o.source_kind in ("photos", "video", "video360"):
            self.o.sky = True
        self.parent_dir = controller.projects_root

        view = Adw.ToolbarView()
        header = Adw.HeaderBar(show_end_title_buttons=False, show_start_title_buttons=False)
        cancel = Gtk.Button(label="Cancel")
        cancel.connect("clicked", lambda *_: self.close())
        header.pack_start(cancel)
        self.create_btn = Gtk.Button(label="Create")
        self.create_btn.add_css_class("suggested-action")
        self.create_btn.connect("clicked", lambda *_: self._create())
        header.pack_end(self.create_btn)
        view.add_top_bar(header)
        page = Adw.PreferencesPage()
        view.set_content(page)
        self.set_content(view)

        # ---------------------------------------------------------- source
        src = group("Source")
        self.kind_row = combo_row("Kind", list(SOURCE_KINDS.items()), self.o.source_kind, self._kind)
        src.add(self.kind_row)
        self.path_row = plain_row(title="Choose the source", subtitle=self.o.source or "Nothing chosen yet")
        self.path_row.set_use_markup(False)
        choose = Gtk.Button(label="Choose…", valign=Gtk.Align.CENTER)
        choose.connect("clicked", lambda *_: self._choose())
        self.path_row.add_suffix(choose)
        src.add(self.path_row)
        self.name_row = Adw.EntryRow(title="Project name")
        self.name_row.connect("changed", lambda r: self._valid())
        src.add(self.name_row)
        self.folder_row = plain_row(title="Save in", subtitle=str(self.parent_dir))
        self.folder_row.set_use_markup(False)
        fb = Gtk.Button(label="Change…", valign=Gtk.Align.CENTER)
        fb.connect("clicked", lambda *_: self._choose_parent())
        self.folder_row.add_suffix(fb)
        src.add(self.folder_row)
        page.add(src)

        # ---------------------------------------------------------- preparation
        prep = group("Preparation")
        o = self.o
        prep.add(combo_row("Maximum image size", [("1600", "1600 px"), ("2400", "2400 px"), ("3200", "3200 px"),
                                                  ("0", "Full resolution")], str(o.max_image_size),
                           lambda v: setattr(o, "max_image_size", int(v)), "source.maxImageSize"))
        self.video_rows = [
            spin_row("Frames per second to extract", 0.5, 15, 0.5, o.video_fps, lambda v: setattr(o, "video_fps", v),
                     1, "source.videoFPS"),
            spin_row("Target frame count", 0, 3000, 10, o.target_frames,
                     lambda v: setattr(o, "target_frames", int(v)), setting_id="source.targetFrames"),
            switch_row("Pick sharpest frame", o.sharpest_in_window, lambda v: setattr(o, "sharpest_in_window", v),
                       "source.sharpestInWindow"),
            switch_row("HDR tone mapping", o.hdr_tone_map, lambda v: setattr(o, "hdr_tone_map", v), "source.hdr"),
        ]
        self.r360_rows = [
            spin_row("360°: views per frame", 3, 16, 1, o.views_360, lambda v: setattr(o, "views_360", int(v)),
                     setting_id="source.360views"),
            spin_row("360°: view field of view", 60, 120, 5, o.fov_360, lambda v: setattr(o, "fov_360", v),
                     setting_id="source.360fov"),
            switch_row("360°: drop the floor view", o.drop_nadir_360, lambda v: setattr(o, "drop_nadir_360", v),
                       "source.360dropNadir"),
        ]
        self.photo_rows = [switch_row("Reject blurry photos", o.blur_rejection,
                                      lambda v: setattr(o, "blur_rejection", v), "source.blurRejection")]
        for r in self.video_rows + self.r360_rows + self.photo_rows:
            prep.add(r)
        prep.add(spin_row("Hold out every Nth image", 0, 100, 1, o.holdout_every,
                          lambda v: setattr(o, "holdout_every", int(v)), setting_id="source.holdout"))
        self.prep_group = prep
        page.add(prep)

        # ---------------------------------------------------------- cameras
        cams = group("Cameras")
        self.solver_row = combo_row("Camera solver", [("learned", "COLMAP / GLOMAP + learned features"),
                                                      ("poses", "Use poses from the source")], o.solver,
                                    lambda v: setattr(o, "solver", v), "sfm.solver")
        cams.add(self.solver_row)
        cams.add(combo_row("Feature model", FEATURES, o.features, lambda v: setattr(o, "features", v), "sfm.features"))
        cams.add(combo_row("Pair selection", [("auto", "Automatic"), ("sequential", "Sequential (video)"),
                                              ("exhaustive", "Exhaustive"), ("retrieval", "Retrieval")],
                           o.matching, lambda v: setattr(o, "matching", v), "sfm.matching"))
        cams.add(combo_row("Mapper", [("auto", "Automatic"), ("glomap", "GLOMAP (global)"), ("colmap", "COLMAP")],
                           o.mapper, lambda v: setattr(o, "mapper", v), "sfm.mapper"))
        cams.add(combo_row("Lens model", [("auto", "Automatic"), ("pinhole", "Pinhole"), ("opencv", "OpenCV"),
                                          ("fisheye", "Fisheye")], o.camera_model,
                           lambda v: setattr(o, "camera_model", v), "sfm.cameraModel"))
        self.refine_row = switch_row("Refine capture poses", o.refine_capture_poses,
                                     lambda v: setattr(o, "refine_capture_poses", v), "sfm.refineCapturePoses")
        cams.add(self.refine_row)
        self.cams_group = cams
        page.add(cams)

        masks = group("Masks", "Layers stack: every model removes its part, the sky becomes a learned skybox.")
        self.mask_switches = {}
        for model, title in (("person", "Remove people & pets"), ("sam2_auto", "Remove moving things (SAM 2)")):
            r = switch_row(title, model in o.mask_models, lambda v, m=model: self._mask(m, v), "mask.model")
            masks.add(r)
            self.mask_switches[model] = r
        masks.add(switch_row("Sky → learned skybox", o.sky, lambda v: setattr(o, "sky", v), "mask.sky"))
        masks.add(switch_row("Invert mask", o.mask_invert, lambda v: setattr(o, "mask_invert", v), "mask.invert"))
        self.masks_group = masks
        page.add(masks)

        # ---------------------------------------------------------- training
        tr = group("Training")
        tr.add(switch_row("Start training as soon as ready", o.auto_train, lambda v: setattr(o, "auto_train", v)))
        tr.add(combo_row("Quality preset", [(k, f"{v['title']} — {v['subtitle']}") for k, v in PRESETS.items()],
                         o.preset, lambda v: setattr(o, "preset", v), "train.preset"))
        self.train_on_row = train_on_row(controller, o)
        tr.add(self.train_on_row)
        tr.add(switch_row("Local jobs go first", controller.prefs.local_jobs_first, self._prio,
                          subtitle="Ahead of jobs queued by other computers (a running job is never interrupted)."))
        tr.add(training_settings_expander(o.train_overrides, lambda: o.preset))
        page.add(tr)
        self._kind(self.o.source_kind)
        self._valid()

    # ------------------------------------------------------------------ behaviour
    def _prio(self, v: bool) -> None:
        self.ctl.prefs.local_jobs_first = v
        self.ctl.save_prefs()

    def _mask(self, model: str, on: bool) -> None:
        ms = [m for m in self.o.mask_models if m != model]
        if on:
            ms.append(model)
        self.o.mask_models = ms

    def _kind(self, kind: str) -> None:
        self.o.source_kind = kind
        video = kind in VIDEO_SOURCES
        for r in self.video_rows:
            r.set_visible(video)
        for r in self.r360_rows:
            r.set_visible(kind == "video360")
        for r in self.photo_rows:
            r.set_visible(kind == "photos")
        posed = kind in ("colmap", "nerfstudio", "dataset")
        self.cams_group.set_visible(not posed)
        self.masks_group.set_visible(not posed)
        self.prep_group.set_visible(kind != "dataset")
        self.refine_row.set_visible(kind == "capture")
        self.solver_row.set_visible(kind == "capture")
        if kind == "capture":
            self.o.matching = "sequential"
        self.path_row.set_title({"photos": "Photos folder", "video": "Video file", "video360": "360° video file",
                                 "capture": ".splatcapture folder", "colmap": "COLMAP project folder",
                                 "nerfstudio": "transforms.json", "dataset": "Dataset folder"}[kind])
        self._valid()

    def _choose(self) -> None:
        kind = self.o.source_kind
        d = Gtk.FileDialog(title=self.path_row.get_title())
        if kind in FOLDER_SOURCES or kind == "capture":
            d.select_folder(self, None, lambda dlg, res: self._picked(dlg.select_folder_finish, res))
        else:
            d.open(self, None, lambda dlg, res: self._picked(dlg.open_finish, res))

    def _picked(self, finish, res) -> None:
        try:
            f = finish(res)
        except GLib.Error:
            return
        if f is None or not f.get_path():
            return
        p = Path(f.get_path())
        self.o.source = str(p)
        self.path_row.set_subtitle(str(p))
        if not self.name_row.get_text():
            stem = p.name.split(".")[0] if p.is_file() else p.name
            self.name_row.set_text(stem.replace("_", " ").strip() or "Untitled")
        self._valid()

    def _choose_parent(self) -> None:
        d = Gtk.FileDialog(title="Save projects in")
        d.select_folder(self, None, self._parent_picked)

    def _parent_picked(self, dlg, res) -> None:
        try:
            f = dlg.select_folder_finish(res)
        except GLib.Error:
            return
        if f is not None and f.get_path():
            self.parent_dir = Path(f.get_path())
            self.folder_row.set_subtitle(str(self.parent_dir))
            self.ctl.prefs.projects_dir = str(self.parent_dir)
            self.ctl.save_prefs()

    def _valid(self) -> bool:
        ok = bool(self.o.source) and bool(self.name_row.get_text().strip())
        if ok:
            p = Path(self.o.source)
            ok = p.exists()
        self.create_btn.set_sensitive(ok)
        return ok

    def _create(self) -> None:
        if not self._valid():
            return
        self.o.name = self.name_row.get_text().strip()
        try:
            self.parent_dir.mkdir(parents=True, exist_ok=True)
            proj = Project.create(self.parent_dir, self.o.name, self.o.to_dict())
        except OSError as e:
            self.ctl.toast(f"Couldn't create the project: {e}")
            return
        self.close()
        self.on_created(proj, self.o)
