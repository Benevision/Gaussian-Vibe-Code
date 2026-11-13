"""Static checks of the Flatpak packaging (no flatpak-builder needed)."""

import configparser
import json
import os
import py_compile
import re
import shlex
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="Flatpak packaging files (Linux paths, exec bits)")

ROOT = Path(__file__).resolve().parents[1]
FP = ROOT / "flatpak"
APP_ID = "com.example.SplatWorker"
yaml = pytest.importorskip("yaml")


@pytest.fixture(scope="module")
def manifest():
    return yaml.safe_load((FP / f"{APP_ID}.yml").read_text())


def test_manifest_basics(manifest):
    assert manifest["app-id"] == APP_ID
    assert manifest["runtime"] == "org.gnome.Platform" and manifest["sdk"] == "org.gnome.Sdk"
    assert str(manifest["runtime-version"]) == "51"
    assert manifest["command"] == "splatworker-app"
    fa = manifest["finish-args"]
    for needed in ("--share=network", "--device=all", "--socket=wayland", "--socket=fallback-x11",
                   "--talk-name=org.freedesktop.Avahi", "--filesystem=xdg-download"):
        assert needed in fa
    # no broad filesystem access: Downloads (exports) and ~/Documents/SplatWork (projects) only
    allowed = {"--filesystem=xdg-download", "--filesystem=xdg-documents/SplatWork:create"}
    assert not [a for a in fa if a.startswith("--filesystem=") and a not in allowed]


def _modules(manifest):
    for m in manifest["modules"]:
        if isinstance(m, str):
            path = FP / m
            assert path.is_file(), f"module file {m} missing"
            yield json.loads(path.read_text()), path.parent
        else:
            yield m, FP


def test_every_referenced_file_exists(manifest):
    names = []
    for module, base in _modules(manifest):
        names.append(module["name"])
        dir_roots = []
        for src in module.get("sources", []):
            if src["type"] == "dir":
                root = (base / src["path"]).resolve()
                assert root.is_dir(), f"{module['name']}: dir source {src['path']} missing"
                dir_roots.append(root)
            elif src["type"] == "file":
                assert src["url"].startswith("https://") and re.fullmatch(r"[0-9a-f]{64}", src["sha256"])
            else:
                pytest.fail(f"unexpected source type {src['type']}")
        for cmd in module.get("build-commands", []):
            words = shlex.split(cmd)
            if words[:1] == ["install"] and not any(w == "-d" for w in words):
                src = words[-2]
                assert any((r / src).exists() for r in dir_roots), f"{module['name']}: {src} missing"
            if words[:2] == ["cp", "-r"]:
                assert any((r / words[2]).exists() for r in dir_roots), f"{module['name']}: {words[2]} missing"
    assert names == ["python3-deps", "splatprep", "splatworker"]


def test_python_deps_match_runtime_python():
    deps = json.loads((FP / "python3-deps.json").read_text())
    reqs = [re.split(r"[<>=!~\[; ]", ln.split("#")[0].strip(), maxsplit=1)[0]
            for ln in (FP / "requirements-flatpak.txt").read_text().splitlines() if ln.split("#")[0].strip()]
    cmd = deps["build-commands"][0]
    assert "--no-index" in cmd and "--prefix=${FLATPAK_DEST}" in cmd
    assert cmd.split("--no-build-isolation ")[1].split() == reqs
    files = [s["url"].rsplit("/", 1)[-1] for s in deps["sources"]]
    for f in files:
        assert f.endswith(".whl"), f"{f}: sdists need a toolchain, use wheels"
        tag = f.split("-")[-3]
        assert tag in ("py3", "py2.py3", "cp314"), f"{f} is not for the runtime's Python 3.14"
        if tag == "cp314":
            assert "x86_64" in f
    found = {re.sub(r"[-_.]+", "-", f.split("-")[0]).lower() for f in files}
    for r in reqs:
        assert re.sub(r"[-_.]+", "-", r).lower() in found
    for must in ("pydantic-core", "starlette", "h11", "anyio"):
        assert must in found


def test_desktop_file():
    cp = configparser.ConfigParser(interpolation=None)
    cp.optionxform = str
    cp.read(FP / f"{APP_ID}.desktop")
    e = cp["Desktop Entry"]
    assert e["Type"] == "Application" and e["Icon"] == APP_ID
    assert e["Exec"].split()[0] == "splatworker-app"
    assert e["Categories"].endswith(";") and e["Keywords"].endswith(";")


def test_metainfo():
    root = ET.parse(FP / f"{APP_ID}.metainfo.xml").getroot()
    assert root.tag == "component" and root.get("type") == "desktop-application"
    assert root.findtext("id") == APP_ID
    assert root.find("launchable").text == f"{APP_ID}.desktop"
    assert (FP / root.find("launchable").text).is_file()
    assert root.findtext("metadata_license") == "CC0-1.0"
    assert len(root.findall("screenshots/screenshot")) >= 1
    assert root.find("content_rating").get("type") == "oars-1.1"
    assert root.find("releases/release").get("version") == __import__("splatworker").__version__


def test_icons_are_valid_svg():
    for name in (f"{APP_ID}.svg", f"{APP_ID}-symbolic.svg"):
        root = ET.parse(FP / "icons" / name).getroot()
        assert root.tag.endswith("svg") and root.get("viewBox")


def test_launchers(tmp_path):
    app = FP / "splatworker-app"
    assert app.read_text().startswith("#!/usr/bin/python3") and os.access(app, os.X_OK)
    py_compile.compile(str(app), cfile=str(tmp_path / "app.pyc"), doraise=True)
    cli = FP / "splatworker"
    assert cli.read_text().startswith("#!/bin/sh") and os.access(cli, os.X_OK)
    assert "--cli" in cli.read_text()
    for script in ("build.sh", "make-bundle.sh", "generate-deps.sh"):
        assert os.access(FP / script, os.X_OK), script
    # the launcher's lib dir matches where the manifest installs the package
    assert "/app/share/splatworker/lib" in app.read_text()
    from splatworker.gui.paths import FLATPAK_LIB_DIR, FLATPAK_SPLATPREP_DIR
    text = (FP / f"{APP_ID}.yml").read_text()
    assert str(FLATPAK_LIB_DIR).replace("/app", "${FLATPAK_DEST}") in text
    assert str(FLATPAK_SPLATPREP_DIR).replace("/app", "${FLATPAK_DEST}") in text


def test_gtk_sources_byte_compile(tmp_path):
    files = sorted((ROOT / "splatworker" / "gui").rglob("*.py"))
    assert any(f.parent.name == "gtk" for f in files)
    for i, f in enumerate(files):
        py_compile.compile(str(f), cfile=str(tmp_path / f"{i}.pyc"), doraise=True)
