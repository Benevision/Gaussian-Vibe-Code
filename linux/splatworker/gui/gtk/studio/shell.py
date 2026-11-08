"""The Studio shell: a sidebar (Viewer, Projects, Queue & Worker, iPhone Inbox, Models & Benchmarks, Settings
Reference, Recent) around the pages. Mixed into :class:`~splatworker.gui.gtk.window.MainWindow`."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from gi.repository import Adw, Gio, GLib, Gtk

from ....studio.formats import OPENABLE_SUFFIXES
from ....studio.projects import Project
from .rows import plain_row
from ..common import label, toast
from .controller import StudioController
from .new_project import NewProjectDialog
from .pages import InboxPage, ModelsPage, ReferencePage
from .projects_page import ProjectsPage
from .viewer_page import ViewerPage

VIDEO_EXT = (".mp4", ".mov", ".m4v", ".mkv", ".avi", ".insv", ".360", ".webm")
SECTIONS = [("viewer", "Viewer", "view-reveal-symbolic"), ("projects", "Projects", "folder-symbolic"),
            ("worker", "Queue & Worker", "network-server-symbolic"), ("inbox", "iPhone Inbox", "phone-symbolic"),
            ("models", "Models & Benchmarks", "applications-science-symbolic"),
            ("reference", "Settings Reference", "accessories-dictionary-symbolic")]

SHORTCUTS = [
    ("Navigation", [("Drag", "Orbit (Select tool: paint a selection, Alt erases)"),
                    ("Shift-drag / right-drag", "Pan"), ("Scroll / pinch", "Zoom"),
                    ("Double-click", "Set the orbit centre"), ("Tab", "Orbit ↔ Fly"),
                    ("W A S D Q E / arrows", "Fly (Shift ×4, Alt ×¼)"), ("F / R", "Frame the scene / reset")]),
    ("Display", [("1 – 5", "Colour, Depth, Centres, Ellipses, Opacity")]),
    ("Editing", [("Delete", "Delete the selection"), ("Ctrl+Z / Ctrl+Shift+Z", "Undo / Redo")]),
    ("Files", [("Ctrl+N", "New project"), ("Ctrl+O", "Open"), ("Ctrl+E", "Export"), ("?", "This list")]),
]


class StudioShell:
    """Builds the split view; expects ``self.toasts`` and the worker page (``worker_view``)."""

    def build_studio(self, worker_view: Gtk.Widget, menu_model) -> Gtk.Widget:
        self.studio = StudioController(self.state, toast=lambda m: toast(self.toasts, m))
        self.pages = Gtk.Stack(transition_type=Gtk.StackTransitionType.CROSSFADE)
        self.viewer_page = ViewerPage(self, self.studio, self.new_project)
        self.projects_page = ProjectsPage(self, self.studio, self.new_project, self.open_path)
        self.inbox_page = InboxPage(self, self.studio, self.new_project)
        self.models_page = ModelsPage(self, self.state)
        self.pages.add_named(self.viewer_page, "viewer")
        self.pages.add_named(self.projects_page, "projects")
        self.pages.add_named(worker_view, "worker")
        self.pages.add_named(self.inbox_page, "inbox")
        self.pages.add_named(self.models_page, "models")
        self.pages.add_named(ReferencePage(), "reference")

        side = Adw.ToolbarView()
        sh = Adw.HeaderBar()
        sh.set_title_widget(Adw.WindowTitle(title="Splat Studio"))
        sh.pack_end(Gtk.MenuButton(icon_name="open-menu-symbolic", menu_model=menu_model, tooltip_text="Main menu",
                                   primary=True))
        side.add_top_bar(sh)
        self.side_list = Gtk.ListBox(selection_mode=Gtk.SelectionMode.SINGLE)
        self.side_list.add_css_class("navigation-sidebar")
        self.side_list.connect("row-activated", self._side_activated)
        self._section_rows = {}
        for key, title, icon in SECTIONS:
            row = Gtk.ListBoxRow()
            box = Gtk.Box(spacing=10, margin_top=6, margin_bottom=6, margin_start=6, margin_end=6)
            box.append(Gtk.Image(icon_name=icon))
            lab = label(title, wrap=False)
            lab.set_hexpand(True)
            box.append(lab)
            badge = label("", ("caption", "dim-label"), wrap=False)
            box.append(badge)
            row.set_child(box)
            row.key = key
            row.badge = badge
            self.side_list.append(row)
            self._section_rows[key] = row
        self.recent_header = label("Recent", ("caption-heading", "dim-label"), wrap=False)
        self.recent_header.set_margin_top(12)
        self.recent_header.set_margin_start(12)
        self.recent_list = Gtk.ListBox(selection_mode=Gtk.SelectionMode.NONE)
        self.recent_list.add_css_class("navigation-sidebar")
        self.recent_list.connect("row-activated", lambda _l, r: self.open_path(r.path))
        sbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        sbox.append(self.side_list)
        sbox.append(self.recent_header)
        sbox.append(self.recent_list)
        ssw = Gtk.ScrolledWindow(hscrollbar_policy=Gtk.PolicyType.NEVER, vexpand=True)
        ssw.set_child(sbox)
        side.set_content(ssw)

        self.split_view = Adw.NavigationSplitView(min_sidebar_width=200, max_sidebar_width=260)
        self.split_view.set_sidebar(Adw.NavigationPage(title="Splat Studio", child=side))
        self.content_page = Adw.NavigationPage(title="Viewer", child=self.pages)
        self.split_view.set_content(self.content_page)
        self.show_section("viewer")
        self.recents_changed()
        self._install_studio_actions()
        drop = Gtk.DropTarget.new(Gdk_FileList(), Gdk_Copy())
        drop.connect("drop", lambda _t, v, _x, _y: self._drop(v))
        self.split_view.add_controller(drop)
        GLib.timeout_add_seconds(2, self._side_tick)
        return self.split_view

    # ------------------------------------------------------------------ navigation
    def show_section(self, key: str) -> None:
        self.pages.set_visible_child_name(key)
        row = self._section_rows.get(key)
        if row is not None:
            self.side_list.select_row(row)
        self.content_page.set_title(dict((k, t) for k, t, _ in SECTIONS).get(key, ""))
        self.split_view.set_show_content(True)

    def _side_activated(self, _lb, row) -> None:
        self.show_section(row.key)

    def recents_changed(self) -> None:
        while (r := self.recent_list.get_row_at_index(0)) is not None:
            self.recent_list.remove(r)
        items = self.studio.recents.items()
        self.recent_header.set_visible(bool(items))
        for p in items:
            row = Gtk.ListBoxRow()
            icon = "image-x-generic-symbolic" if p.suffix.lower() in OPENABLE_SUFFIXES else "folder-symbolic"
            box = Gtk.Box(spacing=8, margin_top=3, margin_bottom=3, margin_start=6)
            box.append(Gtk.Image(icon_name=icon))
            lab = label(p.name, ("caption",), wrap=False)
            lab.set_ellipsize(3)
            box.append(lab)
            row.set_child(box)
            row.set_tooltip_text(str(p))
            row.path = p
            self.recent_list.append(row)

    def _side_tick(self) -> bool:
        runs = [r for r in self.studio.runs.values() if r.active]
        self._section_rows["projects"].badge.set_text(f"{len(runs)} running" if runs else "")
        poll = getattr(self, "_last_poll", None)
        jobs = poll.jobs if poll else []
        q = sum(1 for j in jobs if j.get("state") == "queued")
        act = sum(1 for j in jobs if j.get("state") in ("loading", "training", "running"))
        on = self.state.server.is_active
        self._section_rows["worker"].badge.set_text(
            ("●" if on else "○") + (f" {act}" if act else "") + (f" +{q}" if q else ""))
        self._section_rows["worker"].badge.set_tooltip_text(
            f"Server {'on' if on else 'off'} · {act} running · {q} waiting")
        if poll is not None and poll.info:
            self.models_page.update(poll.info)
        return True

    # ------------------------------------------------------------------ actions
    def _install_studio_actions(self) -> None:
        app = self.get_application()
        for name, cb, accels in (("new-project", lambda *_: self.new_project(None, None), ["<primary>n"]),
                                 ("open", lambda *_: self.viewer_page.open_dialog(), ["<primary>o"]),
                                 ("export", lambda *_: self._export(), ["<primary>e"]),
                                 ("undo", lambda *_: self.viewer_page.undo(), ["<primary>z"]),
                                 ("redo", lambda *_: self.viewer_page.redo(), ["<primary><shift>z"]),
                                 ("shortcuts", lambda *_: self.show_shortcuts(), ["<primary>question"])):
            a = Gio.SimpleAction.new(name, None)
            a.connect("activate", cb)
            self.add_action(a)
            if app is not None:
                app.set_accels_for_action(f"win.{name}", accels)

    def _export(self) -> None:
        self.show_section("viewer")
        self.viewer_page.insp_stack.set_visible_child_name("export")
        self.viewer_page.split.set_show_sidebar(True)

    def show_shortcuts(self) -> None:
        dlg = Adw.Window(transient_for=self, modal=True, title="Keyboard & mouse", default_width=520,
                         default_height=560)
        view = Adw.ToolbarView()
        view.add_top_bar(Adw.HeaderBar())
        page = Adw.PreferencesPage()
        for title, items in SHORTCUTS:
            g = Adw.PreferencesGroup(title=title)
            for k, v in items:
                r = plain_row(title=v)
                r.set_use_markup(False)
                r.add_suffix(label(k, ("dim-label", "monospace"), wrap=False))
                g.add(r)
            page.add(g)
        view.set_content(page)
        dlg.set_content(view)
        dlg.present()

    def _drop(self, value) -> bool:
        files = value.get_files() if hasattr(value, "get_files") else []
        paths = [Path(f.get_path()) for f in files if f.get_path()]
        if paths:
            self.open_path(paths[0])
            return True
        return False

    # ------------------------------------------------------------------ open / new
    def open_path(self, path: Path) -> None:
        path = Path(path)
        name = path.name.lower()
        if path.is_file() and path.suffix.lower() in OPENABLE_SUFFIXES:
            self.show_section("viewer")
            self.viewer_page.load(path)
        elif path.is_file() and path.suffix.lower() in VIDEO_EXT:
            self.new_project("video360" if "360" in name or name.endswith((".insv", ".360")) else "video", str(path))
        elif path.is_file() and name == "transforms.json":
            self.new_project("nerfstudio", str(path))
        elif path.is_dir() and (path / "manifest.json").is_file():
            self.new_project("capture", str(path))
        elif path.is_dir() and (path / "dataset.json").is_file():
            proj = Project.find(path)
            if proj is not None and proj.outputs():
                self.show_section("projects")
                self.projects_page.select(proj.root)
            else:
                self.new_project("dataset", str(path))
        elif path.is_dir() and (path / "transforms.json").is_file():
            self.new_project("nerfstudio", str(path / "transforms.json"))
        elif path.is_dir() and ((path / "sparse").is_dir() or (path / "cameras.bin").is_file()):
            self.new_project("colmap", str(path))
        elif path.is_dir() and (path / "project.json").is_file():
            self.show_section("projects")
            self.projects_page.select(path)
        elif path.is_dir():
            self.new_project("photos", str(path))
        else:
            toast(self.toasts, f"Splat Studio can't open {path.name}")

    def new_project(self, kind: Optional[str], source: Optional[str]) -> None:
        NewProjectDialog(self, self.studio, kind, source, self._project_created).present()

    def _project_created(self, project: Project, options) -> None:
        self.studio.start_pipeline(project, options)
        self.show_section("projects")
        GLib.timeout_add(400, lambda: (self.projects_page.select(project.root), False)[1])

    # ------------------------------------------------------------------ pipeline callbacks
    def live_preview(self, path: Path, project: Project) -> None:
        if self.pages.get_visible_child_name() == "viewer" or self.viewer_page.area.cloud is None:
            self.viewer_page.show_live_preview(path, project)

    def notify_done(self, project: Project, output: Path) -> None:
        app = self.get_application()
        if app is not None:
            n = Gio.Notification.new(f"{project.name} is ready")
            n.set_body(f"Training finished: {output.name}")
            app.send_notification(f"done-{project.root.name}", n)
        self.viewer_page.load(output)
        self.show_section("viewer")


def Gdk_FileList():
    from gi.repository import Gdk

    return Gdk.FileList


def Gdk_Copy():
    from gi.repository import Gdk

    return Gdk.DragAction.COPY
