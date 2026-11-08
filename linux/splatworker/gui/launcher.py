"""Entry point of the desktop app / Flatpak (``splatworker-app``).

::

    splatworker-app                      open the window (first run: setup assistant)
    splatworker-app --background         start the server without a window (used for autostart)
    splatworker-app --after-update       (internal) start once the previous version has quit
    splatworker-app --headless [ARGS]    no GUI: run `splatworker serve` with the GUI's settings;
                                         ARGS override them (e.g. --port 9000 --no-mdns)
    splatworker-app --headless info      any other splatworker sub-command, in the GPU runtime
    splatworker-app --install-runtime [--profile cu124|cu121|cu118|cpu] [--no-splatprep]
                                      [--lightglue] [--yes]
                                         install the GPU runtime from a terminal
    splatworker-app --cli ARGS           alias of --headless ARGS (also /app/bin/splatworker)

In the Flatpak: ``flatpak run com.example.SplatWorker --headless``.
This module must not import GTK at module level: the headless paths run without it.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from . import APP_ID
from .paths import AppPaths

SUBCOMMANDS = ("serve", "train", "gen-token", "info")


def headless_command(extra: Sequence[str], paths: Optional[AppPaths] = None,
                     environ: Optional[Mapping[str, str]] = None,
                     fallback_python: Optional[str] = None) -> Tuple[str, List[str], Dict[str, str], Optional[str]]:
    """(executable, argv, env, warning) for ``--headless``."""
    from .server_process import launch_spec
    from .settings import AppSettings, TokenStore

    paths = paths or AppPaths.from_env(environ)
    spec = launch_spec(paths, environ, fallback_python=fallback_python)
    extra = list(extra)
    if extra and extra[0] in SUBCOMMANDS and extra[0] != "serve":
        argv = [spec.python, "-m", "splatworker", *extra]
    else:
        if extra and extra[0] == "serve":
            extra = extra[1:]
        settings = AppSettings.load(paths.settings_file)
        if settings.require_token and "--token" not in extra and "--token-file" not in extra:
            TokenStore(paths.token_file).get_or_create()
        # Flags given on the command line win: argparse keeps the last value.
        argv = [spec.python, "-m", "splatworker", "serve", *settings.serve_args(paths), *extra]
    return spec.python, argv, spec.env, (spec.warning if not spec.runtime_ready else None)


def _install_runtime(args: List[str]) -> int:
    import argparse
    import threading

    from .installer import Installer
    from .nvidia import detect
    from .paths import splatprep_dir
    from .runtime_plan import PROFILE_IDS, build_plan, choose_profile

    p = argparse.ArgumentParser(prog="splatworker-app --install-runtime")
    p.add_argument("--profile", choices=PROFILE_IDS)
    p.add_argument("--no-splatprep", action="store_true")
    p.add_argument("--lightglue", action="store_true")
    p.add_argument("--yes", "-y", action="store_true")
    a = p.parse_args(args)

    paths = AppPaths.from_env()
    paths.ensure()
    driver = detect()
    from .runtime_plan import unpublished_profiles

    choice = choose_profile(driver, a.profile, unpublished_profiles(local_dir=paths.local_wheels_dir))
    print(choice.summary)
    for w in choice.warnings:
        print("warning:", w)
    if choice.blocking:
        print("note:", choice.blocking)
    plan = build_plan(choice, paths, with_splatprep=not a.no_splatprep, with_lightglue=a.lightglue,
                      splatprep_dir=splatprep_dir())
    print(f"\nWill download about {plan.download_mb / 1000:.1f} GB and use about "
          f"{plan.disk_mb / 1000:.1f} GB in {paths.data_root}:")
    for s in plan.steps:
        print(f"  - {s.title}: {s.description}")
    problem = plan.free_space_problem(paths.data_root)
    if problem:
        print("error:", problem, file=sys.stderr)
        return 1
    if not a.yes:
        try:
            if input("\nContinue? [y/N] ").strip().lower() not in ("y", "yes"):
                return 1
        except EOFError:
            return 1
    last = {"pct": -1}
    lock = threading.Lock()

    def on_event(kind, data):
        with lock:
            if kind == "step":
                print(f"\n[{data['index'] + 1}/{data['count']}] {data['step'].title}", flush=True)
            elif kind == "progress":
                pct = int(data["overall"] * 100)
                if pct != last["pct"]:
                    last["pct"] = pct
                    print(f"\r  {pct:3d}%  {data['message'][:90]:<90}", end="", flush=True)
            elif kind == "failed":
                e = data["error"]
                print(f"\n\nFAILED: {e.title}\n{e.detail}\n{e.hint}\nLog: {paths.install_log}", file=sys.stderr)

    installer = Installer(plan, paths, on_event=on_event)
    try:
        result = installer.run()
    except KeyboardInterrupt:
        installer.cancel()
        return 130
    if result.ok:
        print("\n\nGPU runtime installed:", result.verify)
        return 0
    return 130 if result.cancelled else 1


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if argv and argv[0] == "--version":
        from splatworker import __version__

        print(f"SplatWork Node {__version__} ({APP_ID})")
        return 0
    if argv and argv[0] in ("--headless", "--cli"):
        exe, cmd, env, warning = headless_command(argv[1:])
        if warning:
            print("warning:", warning, file=sys.stderr)
        sys.stdout.flush()
        sys.stderr.flush()
        os.execve(exe, cmd, env)
        return 127  # pragma: no cover
    if argv and argv[0] == "--install-runtime":
        return _install_runtime(argv[1:])

    background = "--background" in argv
    try:
        from .gtk.app import run_app
        if "--after-update" in argv:     # started by the previous version (portal Spawn): let it finish quitting
            from .gtk.flatpak_update import wait_for_previous_instance
            wait_for_previous_instance(APP_ID)
    except (ImportError, ValueError) as e:
        print(f"The graphical interface needs GTK 4, libadwaita and PyGObject ({e}).\n"
              "Run with --headless to start the server without it.", file=sys.stderr)
        return 1
    return run_app(background=background)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
