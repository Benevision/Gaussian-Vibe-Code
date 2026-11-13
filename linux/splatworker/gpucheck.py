"""``splatworker gpu-check``: what this computer's GPU runtime can do, as one report to paste into a bug / test log.

Runs in the GPU runtime (``flatpak run com.example.SplatWorker --headless gpu-check``) and checks, each step on its
own so one failure doesn't hide the rest:

1. the cards the kernel sees (sysfs), /dev/kfd, the HSA override the app sets;
2. PyTorch: version, CUDA / ROCm (HIP) / XPU build, the devices and (ROCm) their gfx ISA;
3. small GPU workloads: 1 GB allocation, a 4096² matmul (TFLOPS), a convolution, a softmax — with timings;
4. gsplat (if installed): import, a forward + backward rasterization of 100 k splats, timed; fused-ssim;
5. splatprep's device choice.

The last line is ``SPLATWORK-GPU-CHECK {json}`` with everything, for pasting back.
"""
from __future__ import annotations

import json
import os
import platform
import time
import traceback
from typing import Any, Callable, Dict, List


def _step(report: Dict[str, Any], name: str, fn: Callable[[], Any], lines: List[str]) -> Any:
    t0 = time.perf_counter()
    try:
        value = fn()
        report[name] = {"ok": True, "value": value, "seconds": round(time.perf_counter() - t0, 3)}
        lines.append(f"  ✓ {name}: {value}")
        return value
    except Exception as e:  # noqa: BLE001 — every failure is part of the report
        msg = f"{type(e).__name__}: {e}"
        report[name] = {"ok": False, "error": msg[:500], "trace": traceback.format_exc()[-1500:]}
        lines.append(f"  ✗ {name}: {msg[:300]}")
        return None


def _sync(torch, dev: str) -> None:
    if dev.startswith("cuda"):
        torch.cuda.synchronize()
    elif dev.startswith("xpu"):
        torch.xpu.synchronize()


def run(train_check: bool = True, splats: int = 100_000) -> Dict[str, Any]:
    from . import __version__
    from .gpu import adapters, torch_backend

    report: Dict[str, Any] = {"workerVersion": __version__, "python": platform.python_version(),
                              "kernel": platform.release()}
    lines: List[str] = ["SplatWork GPU check", "System:"]
    _step(report, "adapters", lambda: adapters(), lines)
    _step(report, "kfd", lambda: os.path.exists("/dev/kfd"), lines)
    _step(report, "env", lambda: {k: os.environ.get(k) for k in
                                  ("HSA_OVERRIDE_GFX_VERSION", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES",
                                   "ROCR_VISIBLE_DEVICES", "SPLATWORK_NO_HSA_OVERRIDE") if os.environ.get(k)},
          lines)
    lines.append("PyTorch:")
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        report["torch"] = {"ok": False, "error": str(e)}
        lines.append(f"  ✗ torch: not installed ({e}) — run Set Up with the processing models / a GPU runtime")
        return _finish(report, lines)
    backend = torch_backend()
    report["torchBackend"] = backend
    _step(report, "torch", lambda: {"version": torch.__version__, "cuda": torch.version.cuda,
                                    "hip": getattr(torch.version, "hip", None), "backend": backend}, lines)
    dev = {"cuda": "cuda:0", "rocm": "cuda:0", "xpu": "xpu:0"}.get(backend)
    if dev is None:
        lines.append("  ✗ no GPU visible to PyTorch (it will run on the CPU)")
        return _finish(report, lines)

    def devices():
        if backend == "xpu":
            return [torch.xpu.get_device_name(i) for i in range(torch.xpu.device_count())]
        out = []
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            out.append({"name": p.name, "memoryMB": p.total_memory // 2**20,
                        "gfx": getattr(p, "gcnArchName", None)})
        return out
    _step(report, "devices", devices, lines)

    lines.append("Workloads:")

    def alloc():
        x = torch.empty(256 * 2**20, dtype=torch.float32, device=dev)     # 1 GB
        x.fill_(1.0)
        _sync(torch, dev)
        s = float(x[:1000].sum())
        del x
        return f"1 GB ok (sum {s:.0f})"
    _step(report, "alloc_1gb", alloc, lines)

    def matmul():
        a = torch.randn(4096, 4096, device=dev)
        b = torch.randn(4096, 4096, device=dev)
        (a @ b).sum().item()
        _sync(torch, dev)
        t0 = time.perf_counter()
        for _ in range(10):
            c = a @ b
        _sync(torch, dev)
        dt = time.perf_counter() - t0
        ref = (a[:64].cpu() @ b.cpu())
        err = float((c[:64].cpu() - ref).abs().max() / ref.abs().max())
        return {"tflops": round(10 * 2 * 4096 ** 3 / dt / 1e12, 2), "relError": round(err, 6)}
    _step(report, "matmul_fp32", matmul, lines)

    def conv():
        x = torch.randn(4, 64, 256, 256, device=dev, requires_grad=True)
        w = torch.randn(64, 64, 3, 3, device=dev, requires_grad=True)
        y = torch.nn.functional.conv2d(x, w, padding=1)
        torch.softmax(y.flatten(1), dim=1).sum().backward()
        _sync(torch, dev)
        return f"conv2d + softmax + backward ok, grad norm {float(w.grad.norm()):.3g}"
    _step(report, "conv_backward", conv, lines)

    lines.append("Training libraries:")

    def gsplat_check():
        import gsplat
        from importlib.metadata import version

        n = splats
        g = torch.Generator(device="cpu").manual_seed(0)
        means = (torch.rand(n, 3, generator=g) * 2 - 1).to(dev).requires_grad_()
        quats = torch.nn.functional.normalize(torch.randn(n, 4, generator=g)).to(dev).requires_grad_()
        scales = torch.full((n, 3), 0.01).to(dev).requires_grad_()
        opac = torch.full((n,), 0.8).to(dev).requires_grad_()
        cols = torch.rand(n, 3, generator=g).to(dev).requires_grad_()
        view = torch.eye(4, device=dev)[None]
        view[0, 2, 3] = 3.0
        K = torch.tensor([[[600.0, 0, 400], [0, 600.0, 300], [0, 0, 1]]], device=dev)
        times = []
        for _ in range(4):
            t0 = time.perf_counter()
            img, alpha, _ = gsplat.rasterization(means, quats, scales, opac, cols, view, K, 800, 600)
            img.mean().backward()
            _sync(torch, dev)
            times.append(time.perf_counter() - t0)
        cov = float((alpha > 0.01).float().mean())
        return {"version": version("gsplat"), "splats": n, "msPerStep": round(1000 * min(times[1:]), 2),
                "coverage": round(cov, 3), "gradFinite": bool(torch.isfinite(means.grad).all())}
    if train_check:
        _step(report, "gsplat", gsplat_check, lines)

        def ssim():
            from fused_ssim import fused_ssim
            a = torch.rand(1, 3, 128, 128, device=dev, requires_grad=True)
            v = fused_ssim(a, a.detach(), padding="valid")
            v.backward()
            return round(float(v), 4)
        _step(report, "fused_ssim", ssim, lines)
    lines.append("Processing:")

    def splatprep_dev():
        from splatprep.device import gpu_backend, select_device
        d = select_device()
        return {"device": d, "backend": gpu_backend(d)}
    _step(report, "splatprep", splatprep_dev, lines)
    return _finish(report, lines)


def _finish(report: Dict[str, Any], lines: List[str]) -> Dict[str, Any]:
    report["summary"] = lines
    return report


def main(args) -> int:
    rep = run(train_check=not getattr(args, "no_train", False), splats=getattr(args, "splats", 100_000))
    for line in rep["summary"]:
        print(line)
    slim = {k: v for k, v in rep.items() if k != "summary"}
    for v in slim.values():
        if isinstance(v, dict):
            v.pop("trace", None) if v.get("ok") else None
    print("SPLATWORK-GPU-CHECK " + json.dumps(slim, default=str), flush=True)
    failed = [k for k, v in slim.items() if isinstance(v, dict) and v.get("ok") is False
              and k not in ("gsplat", "fused_ssim", "splatprep")]
    return 1 if failed else 0
