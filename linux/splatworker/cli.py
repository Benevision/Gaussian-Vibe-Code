"""Command line interface: ``splatworker serve | train | gen-token | info``."""

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
    log.info("SplatWork Node %s '%s' on %s:%d, data dir %s", __version__, settings.name,
             settings.host, settings.port, settings.data_dir)
    app = create_app(settings)
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
    from .gpu import backend_description, list_gpus, training_available
    from .processing import available_models, splatprep_available

    print(json.dumps({
        "workerVersion": __version__,
        "backend": backend_description(),
        "trainingAvailable": training_available(),
        "splatprepAvailable": splatprep_available(),
        "models": available_models(),
        "gpus": list_gpus(),
    }, indent=1))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="splatworker", description="SplatWork Node - CUDA training server")
    p.add_argument("--version", action="version", version=f"splatworker {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("serve", help="run the HTTP training server")
    s.add_argument("--host", default=os.environ.get("SPLATWORKER_HOST", "0.0.0.0"))
    s.add_argument("--port", type=int, default=int(os.environ.get("SPLATWORKER_PORT", "7878")))
    s.add_argument("--data-dir", default=os.environ.get("SPLATWORKER_DATA_DIR", "~/.local/share/splatworker"))
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
                   help="enable the endpoints for SplatWork Node's desktop app on this PC (local folders as "
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
    return p


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
