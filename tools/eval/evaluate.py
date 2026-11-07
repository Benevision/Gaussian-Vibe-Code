#!/usr/bin/env python3
"""Held-out image metrics and visual side-by-sides for one or more trained splats.

Render each model's held-out views first (same dataset, same resolution):

    SplatStudio --eval model.ply --dataset <dataset> --save-renders renders/<name>

then:

    python3 tools/eval/evaluate.py --dataset <dataset> --model ours=renders/ours --model spirula=renders/spirula \
        --out eval_out [--views 6] [--fixed 000040.jpg,000159.jpg]

Writes ``eval_out/metrics.json`` (per view and mean PSNR / SSIM, plus LPIPS when the ``lpips`` package and
torch are importable), ``eval_out/summary.md`` and ``eval_out/sheet.jpg``: rows are held-out views picked
at percentiles of the first model's PSNR (worst, 25 %, median, 75 %, best — plus any ``--fixed`` views),
columns are the photo and each model, each with a 2× detail crop of the region where the models differ
most. Comparisons are only fair between models rendered at the same resolution from the same dataset.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

try:
    from scipy.ndimage import gaussian_filter
except Exception:  # pragma: no cover
    gaussian_filter = None


def load_rgb(path: Path, size=None) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    if size is not None and im.size != size:
        im = im.resize(size, Image.LANCZOS)
    return np.asarray(im, dtype=np.float32) / 255.0


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse <= 1e-12 else -10.0 * np.log10(mse)


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Standard SSIM (Gaussian σ = 1.5, K1 = 0.01, K2 = 0.03), mean over RGB."""
    if gaussian_filter is None:
        return float("nan")
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    vals = []
    for ch in range(3):
        x, y = a[..., ch].astype(np.float64), b[..., ch].astype(np.float64)
        f = lambda z: gaussian_filter(z, 1.5, truncate=3.5)
        mx, my = f(x), f(y)
        sxx, syy, sxy = f(x * x) - mx * mx, f(y * y) - my * my, f(x * y) - mx * my
        m = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))
        vals.append(m.mean())
    return float(np.mean(vals))


def make_lpips():
    try:
        import lpips  # type: ignore
        import torch  # type: ignore
    except Exception:
        return None
    net = lpips.LPIPS(net="vgg", verbose=False)

    def score(a, b):
        ta = torch.from_numpy(a).permute(2, 0, 1)[None] * 2 - 1
        tb = torch.from_numpy(b).permute(2, 0, 1)[None] * 2 - 1
        with torch.no_grad():
            return float(net(ta, tb).item())
    return score


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--model", action="append", required=True, help="name=renders_dir (repeatable)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--views", type=int, default=5, help="percentile rows in the sheet")
    ap.add_argument("--fixed", default="", help="comma-separated image names always shown")
    ap.add_argument("--width", type=int, default=480, help="sheet column width (px)")
    ap.add_argument("--no-lpips", action="store_true")
    args = ap.parse_args()

    ds = json.loads((args.dataset / "dataset.json").read_text())
    held = [c["image"] for c in ds["cameras"] if c.get("holdout")]
    models = []
    for m in args.model:
        name, _, d = m.partition("=")
        models.append((name, Path(d)))
    lp = None if args.no_lpips else make_lpips()

    per: dict = {name: {} for name, _ in models}
    for img in held:
        base = Path(img).name
        renders = [(name, d / base) for name, d in models]
        if not all(p.is_file() for _, p in renders):
            continue
        size = Image.open(renders[0][1]).size
        gt = load_rgb(args.dataset / img, size)
        for name, p in renders:
            r = load_rgb(p, size)
            e = {"psnr": psnr(r, gt), "ssim": ssim(r, gt)}
            if lp is not None:
                e["lpips"] = lp(r, gt)
            per[name][base] = e

    common = sorted(set.intersection(*[set(v) for v in per.values()])) if per else []
    summary = {}
    for name, _ in models:
        rows = [per[name][k] for k in common]
        summary[name] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]} if rows else {}
        summary[name]["views"] = len(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps({"summary": summary, "perView": per}, indent=1))
    lines = ["| model | views | PSNR | SSIM | LPIPS |", "|---|---|---|---|---|"]
    for name, s in summary.items():
        lines.append(f"| {name} | {s.get('views', 0)} | {s.get('psnr', float('nan')):.2f} | {s.get('ssim', float('nan')):.4f} | "
                     f"{s['lpips']:.4f} |" if "lpips" in s else
                     f"| {name} | {s.get('views', 0)} | {s.get('psnr', float('nan')):.2f} | {s.get('ssim', float('nan')):.4f} | – |")
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # Sheet rows: percentiles of the first model's PSNR, plus fixed views.
    first = models[0][0]
    order = sorted(common, key=lambda k: per[first][k]["psnr"])
    picks = []
    if order:
        for q in np.linspace(0, 1, max(args.views, 1)):
            k = order[int(round(q * (len(order) - 1)))]
            if k not in picks:
                picks.append(k)
    for f in [x.strip() for x in args.fixed.split(",") if x.strip()]:
        if f in common and f not in picks:
            picks.append(f)
    W = args.width
    tiles = []
    for k in picks:
        size = Image.open(models[0][1] / k).size
        img_rel = next(c["image"] for c in ds["cameras"] if Path(c["image"]).name == k)
        ims = [("photo", load_rgb(args.dataset / img_rel, size))] + [(n, load_rgb(d / k, size)) for n, d in models]
        # Detail crop where the models disagree most (or where the first model is worst).
        diff = sum(np.abs(a[1] - ims[0][1]).mean(-1) for a in ims[1:])
        h, w = diff.shape
        cw, ch = w // 4, h // 4
        best, by, bx = -1.0, 0, 0
        for y in range(0, h - ch, ch // 2):
            for x in range(0, w - cw, cw // 2):
                v = float(diff[y:y + ch, x:x + cw].mean())
                if v > best:
                    best, by, bx = v, y, x
        row = []
        for n, a in ims:
            full = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
            th = full.resize((W, int(W * h / w)), Image.LANCZOS)
            crop = full.crop((bx, by, bx + cw, by + ch)).resize((W, int(W * ch / cw)), Image.LANCZOS)
            panel = Image.new("RGB", (W, th.height + crop.height + 4), "white")
            panel.paste(th, (0, 0)); panel.paste(crop, (0, th.height + 4))
            d = ImageDraw.Draw(panel)
            sx, sy = W / w, th.height / h
            d.rectangle([bx * sx, by * sy, (bx + cw) * sx, (by + ch) * sy], outline=(255, 60, 60), width=2)
            label = n if n == "photo" else f"{n}  {per[n][k]['psnr']:.2f} dB"
            d.rectangle([0, 0, 8 + 7 * len(label), 16], fill=(0, 0, 0))
            d.text((4, 2), label, fill=(255, 255, 255))
            row.append(panel)
        tiles.append((k, row))
    if tiles:
        rh = max(p.height for _, r in tiles for p in r)
        sheet = Image.new("RGB", ((W + 8) * len(tiles[0][1]) + 8, (rh + 24) * len(tiles) + 8), (235, 235, 235))
        d = ImageDraw.Draw(sheet)
        for i, (k, row) in enumerate(tiles):
            y = 8 + i * (rh + 24)
            d.text((8, y), k, fill=(0, 0, 0))
            for j, p in enumerate(row):
                sheet.paste(p, (8 + j * (W + 8), y + 14))
        sheet.save(args.out / "sheet.jpg", quality=88)
        print(f"sheet: {args.out / 'sheet.jpg'} ({len(tiles)} views)")


if __name__ == "__main__":
    main()
