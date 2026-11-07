"""Command line interface.

    python -m splatprep run --request request.json --progress-jsonl -   [--workdir DIR]
    python -m splatprep models [--json]
    python -m splatprep synth --out DIR [--trajectory orbit|walk|sparse] [--num N] ...
    python -m splatprep benchmark --dataset D [--dataset D2 ...] [--synth all] [--models a,b] --out DIR
    python -m splatprep export --model sparse/0 --images images --out dataset
    python -m splatprep device
    python -m splatprep maskserver [--images DIR] [--out DIR] [--device D]   # JSONL SAM 2 click-to-mask loop

With ``--progress-jsonl -`` stdout is reserved for protocol lines: the real
stdout file descriptor is duplicated for JSONL and fd 1 is pointed at stderr, so
stray prints from third-party libraries (or C extensions) cannot corrupt the stream.
Exit code: 0 on success, 2 on ``{"error": ...}``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import List, Optional

from .progress import JsonlWriter


def _protocol_stream(target: str):
    if target == "-":
        sys.stdout.flush()
        fd = os.dup(1)
        os.dup2(2, 1)
        sys.stdout = sys.stderr
        return JsonlWriter(os.fdopen(fd, "w", buffering=1)), None
    f = open(target, "a", buffering=1)
    return JsonlWriter(f), f


def cmd_run(args) -> int:
    from .isolation import block_pycolmap
    block_pycolmap()  # torch backends + pycolmap in one process abort on macOS (duplicate libomp)
    writer, fh = _protocol_stream(args.progress_jsonl) if args.progress_jsonl else (JsonlWriter(sys.stdout), None)
    try:
        try:
            request = json.loads(Path(args.request).read_text())
        except Exception as e:
            writer({"error": f"cannot read request: {e}"})
            return 2
        workdir = Path(args.workdir) if args.workdir else Path.cwd()
        from .runner import run_request

        try:
            stats = run_request(request, workdir, writer)
        except Exception as e:  # noqa: BLE001 - every failure must become a protocol line
            if os.environ.get("SPLATPREP_DEBUG"):
                traceback.print_exc()
            else:
                sys.stderr.write(traceback.format_exc())
            writer({"error": str(e) or e.__class__.__name__})
            return 2
        writer({"done": True, "stats": stats})
        return 0
    finally:
        if fh:
            fh.close()


def cmd_models(args) -> int:
    from .models import available_models, model_status

    st = model_status()
    if args.json:
        print(json.dumps({"available": available_models(), "models": st}, indent=2))
        return 0
    from .device import select_device

    print(f"device: {select_device()}")
    for mid, s in st.items():
        flag = "sfm" if s["sfm"] else ("ok" if s["available"] else "--")
        print(f"  [{flag:>3}] {mid:<22} {s['kind']:<7} {s.get('reason', '')}")
    return 0


def cmd_synth(args) -> int:
    from .synth import generate

    kinds = ["orbit", "walk", "sparse"] if args.trajectory == "all" else [args.trajectory]
    for k in kinds:
        out = Path(args.out) / k if len(kinds) > 1 else Path(args.out)
        meta = generate(
            out,
            k,
            args.num,
            args.width,
            args.height,
            args.seed,
            args.photometric,
            args.supersample,
            args.depth,
            args.hfov,
            progress=(lambda f, m: sys.stderr.write(f"\r{m}   ")) if not args.quiet else None,
        )
        sys.stderr.write("\n")
        print(json.dumps({"out": str(out), **meta}))
    return 0


def cmd_benchmark_merge(args) -> int:
    from .benchmark import markdown_report, merge_reports

    reps = [json.loads(Path(p).read_text()) for p in args.reports]
    merged = merge_reports(reps)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark.json").write_text(json.dumps(merged, indent=2))
    (out / "benchmark.md").write_text(markdown_report(merged))
    print(markdown_report(merged))
    return 0


def cmd_benchmark(args) -> int:
    from .benchmark import markdown_report, run_benchmark
    from .synth import generate

    out = Path(args.out)
    datasets: List[Path] = [Path(d) for d in (args.dataset or [])]
    if args.synth:
        kinds = ["orbit", "walk", "sparse"] if args.synth == "all" else args.synth.split(",")
        for k in kinds:
            d = out / "_synth" / k
            if not (d / "dataset.json").exists():
                sys.stderr.write(f"generating synthetic '{k}' dataset...\n")
                generate(d, k, None, args.synth_width, args.synth_height, args.seed)
            datasets.append(d)
    models = [m.strip() for m in args.models.split(",")] if args.models else None
    opts = {"matching": args.matching, "maxKeypoints": args.max_keypoints, "resizeMax": args.resize_max, "mapper": args.mapper}
    writer = JsonlWriter(sys.stderr) if not args.quiet else None
    rep = run_benchmark(datasets, out, models, opts, args.in_process, writer, args.timeout)
    print(markdown_report(rep))
    return 0


def cmd_export(args) -> int:
    from .colmap_model import read_model
    from .export import export_dataset

    rec = read_model(Path(args.model))
    out = Path(args.out)
    stats = export_dataset(rec, Path(args.images), out, out.parent / f".splatprep_export_{out.name}")
    print(json.dumps(stats))
    return 0


def cmd_device(args) -> int:
    from .device import select_device

    print(select_device())
    return 0


def cmd_maskserver(args) -> int:
    from .isolation import block_pycolmap

    block_pycolmap()
    writer, _ = _protocol_stream("-")
    from .masks.server import serve_main

    return serve_main(args.images, args.out, args.device, writer, ready_line=args.ready, precision=args.precision)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="splatprep", description="Splat Studio processing toolkit")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="execute a ProcessingRequest (PROTOCOL §6)")
    r.add_argument("--request", required=True)
    r.add_argument("--progress-jsonl", default="-", help="'-' = stdout (default) or a file path")
    r.add_argument("--workdir", default=None, help="base for relative paths (default: current directory)")
    r.set_defaults(func=cmd_run)

    m = sub.add_parser("models", help="list models and availability")
    m.add_argument("--json", action="store_true")
    m.set_defaults(func=cmd_models)

    s = sub.add_parser("synth", help="render a synthetic dataset with ground-truth poses")
    s.add_argument("--out", required=True)
    s.add_argument("--trajectory", default="orbit", choices=["orbit", "walk", "sparse", "all"])
    s.add_argument("--num", type=int, default=None, help="views (default 60 orbit/walk, 15 sparse)")
    s.add_argument("--width", type=int, default=640)
    s.add_argument("--height", type=int, default=480)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--photometric", type=float, default=0.5, help="photometric variation strength 0..1")
    s.add_argument("--supersample", type=int, default=2)
    s.add_argument("--hfov", type=float, default=60.0)
    s.add_argument("--depth", action="store_true", help="also write depth/*.bin (float16 Z)")
    s.add_argument("--quiet", action="store_true")
    s.set_defaults(func=cmd_synth)

    b = sub.add_parser("benchmark", help="benchmark all available feature backends")
    b.add_argument("--dataset", action="append", help="dataset dir / .splatcapture (dir or zip) / image dir; repeatable")
    b.add_argument("--synth", default=None, help="also generate+use synthetic sets: all | orbit,walk,sparse")
    b.add_argument("--synth-width", type=int, default=640)
    b.add_argument("--synth-height", type=int, default=480)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--models", default=None, help="comma separated ids (default: all)")
    b.add_argument("--out", required=True)
    b.add_argument("--matching", default="auto")
    b.add_argument("--mapper", default="auto")
    b.add_argument("--max-keypoints", type=int, default=4096)
    b.add_argument("--resize-max", type=int, default=1600)
    b.add_argument("--timeout", type=float, default=None, help="seconds per run")
    b.add_argument("--in-process", action="store_true", help="run backends in this process (no isolation)")
    b.add_argument("--quiet", action="store_true")
    b.set_defaults(func=cmd_benchmark)

    bm = sub.add_parser("benchmark-merge", help="combine benchmark.json files (later ones replace earlier rows)")
    bm.add_argument("reports", nargs="+", help="benchmark.json files, oldest first")
    bm.add_argument("--out", required=True, help="output directory (benchmark.json + benchmark.md)")
    bm.set_defaults(func=cmd_benchmark_merge)

    e = sub.add_parser("export", help="COLMAP model -> dataset.json + images + points3d.ply")
    e.add_argument("--model", required=True)
    e.add_argument("--images", required=True)
    e.add_argument("--out", required=True)
    e.set_defaults(func=cmd_export)

    d = sub.add_parser("device", help="print the selected compute device")
    d.set_defaults(func=cmd_device)

    ms = sub.add_parser("maskserver", help="keep SAM 2 loaded; answer click-to-mask requests (JSONL on stdin/stdout)")
    ms.add_argument("--images", default=None, help="base directory for relative 'image' paths")
    ms.add_argument("--out", default=None, help="directory for preview PNGs (default: $TMPDIR/splatprep_maskserver)")
    ms.add_argument("--device", default=None, help="auto | mps | cuda | cpu")
    ms.add_argument("--ready", action="store_true", help="also print the {\"ready\": ...} status as the first stdout line")
    ms.add_argument("--precision", default="auto", choices=["auto", "fp32", "bf16", "fp16"], help="auto = bf16 autocast on MPS/CUDA")
    ms.set_defaults(func=cmd_maskserver)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)
