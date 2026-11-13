"""Command line interface: ``splatworker serve | train | gen-token | info | convert | splat-info | benchmark``."""

from __future__ import annotations

import argparse
import json
import logging
import os
import secrets
import sys
import time
from pathlib import Path
from typing import List, Optional

from .config import LOCAL_SECRET_ENV
from .platform_compat import default_data_dir

from . import __version__


def _parse_gpus(value: Optional[str]) -> Optional[List[int]]:
    if value is None or value.strip().lower() in ("", "all"):
        return None
    try:
        gpus = [int(x) for x in value.split(",") if x.strip() != ""]
    except ValueError:
        raise argparse.ArgumentTypeError("--gpus must be a comma separated list of indices, e.g. 0,1")
    if not gpus or len(set(gpus)) != len(gpus) or any(g < 0 for g in gpus):
        raise argparse.ArgumentTypeError("--gpus must list distinct non-negative indices")
    return gpus


def _read_token(args) -> Optional[str]:
    if args.token:
        return args.token
    if args.token_file:
        return Path(args.token_file).expanduser().read_text(encoding="utf-8").strip() or None
    env = os.environ.get("SPLATWORKER_TOKEN")
    return env.strip() if env and env.strip() else None


STOP_ON_STDIN_EOF_ENV = "SPLATWORKER_STOP_ON_STDIN_EOF"


def _stop_when_stdin_closes() -> None:
    """Graceful shutdown when the parent closes our stdin: the desktop app's stop on Windows, where a child
    without a console can't be sent SIGTERM / Ctrl+Break. SIGINT raised here reaches uvicorn's handler."""
    import signal
    import threading

    def watch():
        try:
            while sys.stdin.buffer.read(4096):
                pass
        except (OSError, ValueError, AttributeError):
            return
        signal.raise_signal(signal.SIGINT)

    threading.Thread(target=watch, daemon=True, name="splatworker-stdin-stop").start()


def cmd_serve(args) -> int:
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    import uvicorn

    from .config import ServerSettings
    from .server import create_app

    token = _read_token(args)
    settings = ServerSettings(
        data_dir=Path(args.data_dir).expanduser(),
        host=args.host, port=args.port, token=token, name=args.name,
        gpus=args.gpus, mdns=not args.no_mdns,
        advertise_ips=args.advertise_ip or None,
        max_upload_bytes=int(args.max_upload_gb * 1024 ** 3),
        max_extracted_bytes=int(args.max_extracted_gb * 1024 ** 3),
        allow_local_datasets=bool(getattr(args, "allow_local_datasets", False)),
        # read once and removed, so jobs' subprocesses don't inherit it
        local_secret=os.environ.pop(LOCAL_SECRET_ENV, None) or None,
    )
    if settings.allow_local_datasets and not settings.local_enabled:
        logging.getLogger("splatworker").info(
            "--allow-local-datasets without %s: the desktop-app-only endpoints stay off", LOCAL_SECRET_ENV)
    log = logging.getLogger("splatworker")
    if token is None:
        log.warning("no token configured: anyone on the network can use this worker "
                    "(use --token-file or SPLATWORKER_TOKEN)")
    log.info("SplatWork for Linux %s '%s' on %s:%d, data dir %s", __version__, settings.name,
             settings.host, settings.port, settings.data_dir)
    app = create_app(settings)
    if os.environ.pop(STOP_ON_STDIN_EOF_ENV, None):
        _stop_when_stdin_closes()
    uvicorn.run(app, host=settings.host, port=settings.port, log_level=args.log_level,
                timeout_keep_alive=30)
    return 0


def cmd_train(args) -> int:
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    from .config import ConfigError, TrainingConfig
    from .trainer import Progress, TrainingCancelled, TrainingError, train

    cfg_dict = {}
    if args.config:
        cfg_dict = json.loads(Path(args.config).read_text(encoding="utf-8"))
    for item in args.set or []:
        key, sep, value = item.partition("=")
        if not sep:
            print(f"--set expects key=value, got {item!r}", file=sys.stderr)
            return 2
        try:
            cfg_dict[key] = json.loads(value)
        except json.JSONDecodeError:
            cfg_dict[key] = value
    if args.iterations is not None:
        cfg_dict["iterations"] = args.iterations
    try:
        cfg = TrainingConfig.from_dict(cfg_dict)
    except ConfigError as e:
        print(f"invalid config: {e}", file=sys.stderr)
        return 2

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    preview = out.with_name(out.stem + ".preview.ply") if args.preview else None
    state = {"last": 0.0}

    def status(**f):
        if "step" in f and time.time() - state["last"] > 2.0:
            state["last"] = time.time()
            psnr = f.get("psnr")
            eta = f.get("etaSeconds")
            print(f"step {f['step']}/{cfg.iterations}  loss {f.get('loss') or 0:.4f}  "
                  f"psnr {psnr if psnr is None else round(psnr, 2)}  gaussians {f.get('numGaussians')}  "
                  f"eta {eta and round(eta)}s", flush=True)

    progress = Progress(status=status, log=lambda m: print(m, flush=True))
    try:
        stats = train(args.dataset_dir, cfg, out, device=args.device, preview_path=preview,
                      progress=progress)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except TrainingCancelled:
        return 130
    except TrainingError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(json.dumps(stats, indent=1))
    return 0


def cmd_gen_token(args) -> int:
    token = secrets.token_urlsafe(args.bytes)
    if args.out:
        path = Path(args.out).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as fh:
            fh.write(token + "\n")
        print(f"token written to {path}", file=sys.stderr)
    print(token)
    return 0


def cmd_convert(args) -> int:
    """Splat format conversion (the Mac's ``--convert``): PLY / compressed PLY / .splat / .spz, skybox kept."""
    from .studio import formats

    g = formats.load(args.input)
    out = formats.save(g, args.output, fmt=args.format, sh_degree=args.sh)
    print(f"{g.count} splats → {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


def cmd_splat_info(args) -> int:
    from .studio import formats

    g = formats.load(args.file)
    lo, hi = g.robust_bounds(0.0)
    print(json.dumps({"file": str(args.file), "format": formats.format_for_path(args.file), "splats": g.count,
                      "shDegree": g.sh_degree, "antialiased": g.antialiased,
                      "up": None if g.up is None else [float(x) for x in g.up], "skybox": g.skybox is not None,
                      "bounds": [[float(x) for x in lo], [float(x) for x in hi]],
                      "meanOpacity": float(g.opacities.mean()) if g.count else None}, indent=1))
    return 0


def cmd_info(args) -> int:
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    from .gpu import adapters, backend_description, list_gpus, torch_backend, training_available
    from .processing import available_models, splatprep_available

    print(json.dumps({
        "workerVersion": __version__,
        "backend": backend_description(),
        "trainingAvailable": training_available(),
        "splatprepAvailable": splatprep_available(),
        "models": available_models(),
        "gpus": list_gpus(),
        "adapters": adapters(),
        "torchBackend": torch_backend(),
    }, indent=1))
    return 0


def cmd_benchmark(args) -> int:
    """The standard training benchmark (the Mac's Compute → Benchmark / ``--benchmark``) on a running worker:
    by default this PC's (SplatWork for Linux's server, or ``splatworker serve``). Saved to benchmarks.json."""
    from urllib.parse import urlparse

    from .gui.paths import AppPaths
    from .studio import benchmark
    from .studio.client import ApiError, Client

    paths = AppPaths.from_env()
    config_dir = Path(args.config_dir).expanduser() if args.config_dir else paths.config_dir
    url, token = args.url, _read_token(args)
    if url is None:
        port, require = 7878, True
        try:
            gs = json.loads(paths.settings_file.read_text(encoding="utf-8"))
            port = int(gs.get("port", port))
            require = bool(gs.get("require_token", True))
        except (OSError, ValueError, TypeError):
            pass
        url = f"http://127.0.0.1:{port}"
        if token is None and require and paths.token_file.is_file():
            token = paths.token_file.read_text(encoding="utf-8").strip() or None
    host = (urlparse(url).hostname or "").lower()
    loopback = host in ("127.0.0.1", "localhost", "::1")
    spec = benchmark.BenchmarkSpec(width=320, height=240, views=8, splats=20_000, steps=200, measure_from=50) \
        if args.quick else benchmark.BenchmarkSpec()
    scene = Path(args.scene_dir).expanduser() if args.scene_dir else paths.cache_dir / (
        benchmark.SCENE_DIR + ("-quick" if args.quick else ""))
    client = Client(url, token, timeout=60)
    last = {"m": None}

    def progress(msg, frac):
        if msg != last["m"]:
            last["m"] = msg
            print(f"[{frac * 100:5.1f} %] {msg}", file=sys.stderr, flush=True)
    try:
        rec = benchmark.run(client, scene, spec, local=False, progress=progress, poll=args.poll,
                            device_id="local" if loopback else url.rstrip("/"),
                            name="This PC" if loopback else None, tmp_dir=paths.cache_dir / "tmp")
    except KeyboardInterrupt:
        print("interrupted (the benchmark job may still be running on the worker)", file=sys.stderr)
        return 130
    except (ApiError, benchmark.BenchmarkError) as e:
        print(f"benchmark failed: {getattr(e, 'message', None) or e}", file=sys.stderr)
        if isinstance(e, ApiError) and e.status is None:
            print("Is the worker running? Start SplatWork for Linux (or `splatworker serve`), or pass --url.",
                  file=sys.stderr)
        return 1
    if loopback:
        rec["uploadMBps"] = rec["downloadMBps"] = None       # loopback "network" speed means nothing
    if args.quick:
        print("(quick run: a tiny scene, not comparable - not saved)", file=sys.stderr)
    else:
        benchmark.save_benchmark(config_dir, rec)
        print(f"saved to {config_dir / benchmark.FILE}", file=sys.stderr)
    print(json.dumps(rec, indent=1))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="splatworker", description="SplatWork for Linux - CUDA training server")
    p.add_argument("--version", action="version", version=f"splatworker {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("serve", help="run the HTTP training server")
    s.add_argument("--host", default=os.environ.get("SPLATWORKER_HOST", "0.0.0.0"))
    s.add_argument("--port", type=int, default=int(os.environ.get("SPLATWORKER_PORT", "7878")))
    s.add_argument("--data-dir", default=os.environ.get("SPLATWORKER_DATA_DIR", default_data_dir()))
    s.add_argument("--token", help="bearer token (prefer --token-file or SPLATWORKER_TOKEN: "
                                   "command lines are visible to other local users)")
    s.add_argument("--token-file", default=os.environ.get("SPLATWORKER_TOKEN_FILE"))
    s.add_argument("--name", default=os.environ.get("SPLATWORKER_NAME"),
                   help="display name advertised over Bonjour (default: hostname)")
    s.add_argument("--gpus", type=_parse_gpus, default=_parse_gpus(os.environ.get("SPLATWORKER_GPUS")),
                   help="comma separated GPU indices, one job per GPU (default: all)")
    s.add_argument("--no-mdns", action="store_true", help="do not advertise _splatworker._tcp")
    s.add_argument("--advertise-ip", action="append", help="IPv4 address to advertise (repeatable)")
    s.add_argument("--max-upload-gb", type=float, default=20.0)
    s.add_argument("--max-extracted-gb", type=float, default=40.0)
    s.add_argument("--log-level", default="info", choices=["critical", "error", "warning", "info", "debug"])
    s.add_argument("--allow-local-datasets", action="store_true",
                   help="enable the endpoints for SplatWork for Linux's desktop app on this PC (local folders as "
                        "datasets, queue order); it must also pass its per-launch secret in $" + LOCAL_SECRET_ENV)
    s.set_defaults(func=cmd_serve)

    t = sub.add_parser("train", help="train a dataset directory locally")
    t.add_argument("dataset_dir")
    t.add_argument("--out", required=True, help="output PLY path")
    t.add_argument("--config", help="TrainingConfig JSON file")
    t.add_argument("--set", action="append", metavar="KEY=VALUE",
                   help="override a TrainingConfig key, e.g. --set strategy=adaptive")
    t.add_argument("--iterations", type=int)
    t.add_argument("--device", default="cuda:0")
    t.add_argument("--preview", action="store_true", help="also write <out>.preview.ply snapshots")
    t.set_defaults(func=cmd_train)

    g = sub.add_parser("gen-token", help="generate a random bearer token")
    g.add_argument("--bytes", type=int, default=32)
    g.add_argument("--out", help="also write the token to this file (mode 600)")
    g.set_defaults(func=cmd_gen_token)

    i = sub.add_parser("info", help="print GPU / backend information")
    i.set_defaults(func=cmd_info)

    c = sub.add_parser("convert", help="convert a splat between PLY, compressed PLY, .splat and .spz")
    c.add_argument("input")
    c.add_argument("output", help="format from the extension (.ply, .compressed.ply, .splat, .spz)")
    c.add_argument("--format", choices=["ply", "compressed.ply", "splat", "spz"])
    c.add_argument("--sh", type=int, choices=[0, 1, 2, 3], help="SH degree to keep")
    c.set_defaults(func=cmd_convert)

    si = sub.add_parser("splat-info", help="print what a splat file contains")
    si.add_argument("file")
    si.set_defaults(func=cmd_splat_info)

    b = sub.add_parser("benchmark", help="measure training speed (the standard SplatWork benchmark) on a running "
                                         "worker, by default this PC's; saved to benchmarks.json")
    b.add_argument("--url", help="worker URL (default: this PC's SplatWork for Linux server, from its settings)")
    b.add_argument("--token", help="bearer token (default: SplatWork for Linux's own token file)")
    b.add_argument("--token-file")
    b.add_argument("--config-dir", help="where benchmarks.json is kept (default: the app's config dir)")
    b.add_argument("--scene-dir", help="where the synthetic scene is built and cached")
    b.add_argument("--quick", action="store_true", help="tiny scene and 200 steps: checks it works, not saved")
    b.add_argument("--poll", type=float, default=1.0, help=argparse.SUPPRESS)
    b.set_defaults(func=cmd_benchmark)

    gc = sub.add_parser("gpu-check", help="check what the GPU runtime can do (cards, PyTorch on CUDA / ROCm / XPU, "
                                          "small GPU workloads, gsplat) and print a report to paste")
    gc.add_argument("--no-train", action="store_true", help="skip the gsplat / fused-ssim checks")
    gc.add_argument("--splats", type=int, default=100_000, help="splats in the gsplat test (default 100 000)")
    gc.set_defaults(func=lambda a: __import__("splatworker.gpucheck", fromlist=["main"]).main(a))
    return p


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
