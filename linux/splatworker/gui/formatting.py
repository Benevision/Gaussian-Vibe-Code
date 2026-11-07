"""Human-friendly text for the UI (job rows, durations, addresses). GTK-free."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

STATE_LABELS = {
    "queued": "Waiting",
    "loading": "Preparing",
    "training": "Training",
    "completed": "Finished",
    "failed": "Failed",
    "cancelled": "Cancelled",
}


def duration(seconds: Optional[float]) -> str:
    if seconds is None or seconds != seconds or seconds < 0:
        return "—"
    s = int(round(seconds))
    if s < 60:
        return f"{s} s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m} min {s:02d} s" if m < 10 else f"{m} min"
    h, m = divmod(m, 60)
    return f"{h} h {m:02d} min"


def count(n: Optional[float]) -> str:
    if n is None:
        return "—"
    n = float(n)
    if n >= 1e6:
        return f"{n / 1e6:.2f} M"
    if n >= 1e4:
        return f"{n / 1e3:.0f} k"
    return f"{int(n):,}"


def ago(ts: float, now: Optional[float] = None) -> str:
    d = (time.time() if now is None else now) - ts
    if d < 5:
        return "just now"
    if d < 60:
        return f"{int(d)} s ago"
    if d < 3600:
        return f"{int(d // 60)} min ago"
    return f"{int(d // 3600)} h ago"


def megabytes(mb: Optional[float]) -> str:
    if mb is None:
        return "—"
    return f"{mb / 1024:.1f} GB" if mb >= 1024 else f"{int(mb)} MB"


@dataclass
class JobView:
    job_id: str
    title: str
    kind: str
    state: str
    state_label: str
    fraction: float
    detail: str
    can_cancel: bool
    can_delete: bool
    export: Optional[str]          # "result.ply" / "output.zip" / None
    is_error: bool


def job_view(s: Dict[str, Any]) -> JobView:
    state = s.get("state") or "queued"
    kind = s.get("kind") or "train"
    step, total = s.get("step") or 0, s.get("totalSteps") or 0
    fraction = min(1.0, max(0.0, step / total)) if total else 0.0
    if state == "completed":
        fraction = 1.0
    parts: List[str] = [STATE_LABELS.get(state, state.title())]
    msg = (s.get("message") or "").strip()
    if kind == "train":
        if total and state in ("training", "loading", "completed", "cancelled", "failed") and step:
            parts.append(f"step {step:,} / {total:,}")
        if s.get("psnr") is not None:
            parts.append(f"PSNR {s['psnr']:.1f} dB")
        if s.get("numGaussians"):
            parts.append(f"{count(s['numGaussians'])} splats")
    else:
        if state in ("loading", "training") and total:
            parts.append(f"{fraction * 100:.0f} %")
        if msg and state in ("loading", "training"):
            parts.append(msg)
    if state in ("loading", "training"):
        if s.get("etaSeconds"):
            parts.append(f"about {duration(s['etaSeconds'])} left")
        if kind == "train" and msg and msg not in ("training", "starting"):
            parts.append(msg)
    elif state == "completed" and s.get("elapsedSeconds"):
        parts.append(f"took {duration(s['elapsedSeconds'])}")
    elif state == "failed" and msg:
        parts.append(msg)
    if s.get("gpu") and state in ("loading", "training"):
        parts.append(s["gpu"])
    export = None
    if state == "completed":
        export = "result.ply" if kind == "train" else "output.zip"
    title = s.get("name") or s.get("jobId") or "job"
    return JobView(job_id=s.get("jobId", ""), title=title, kind=kind, state=state,
                   state_label=STATE_LABELS.get(state, state), fraction=fraction,
                   detail=" · ".join(dict.fromkeys(p for p in parts if p)),
                   can_cancel=state in ("queued", "loading", "training") and msg != "cancelling",
                   can_delete=True, export=export, is_error=state == "failed")


def capabilities_text(info: Optional[Dict[str, Any]]) -> str:
    if not info:
        return "—"
    caps = info.get("capabilities") or []
    names = {"train": "training", "process": "dataset processing"}
    if not caps:
        return "nothing yet (GPU runtime not installed)"
    return ", ".join(names.get(c, c) for c in caps).capitalize()


def addresses_text(ips: Sequence[str], port: int) -> List[str]:
    return [f"http://{ip}:{port}" for ip in ips] or [f"http://<this computer's IP>:{port}"]


def connection_uri(ip: str, port: int, name: str, token: Optional[str]) -> str:
    """Payload of the optional QR code: everything needed to add the worker by hand."""
    from urllib.parse import quote, urlencode

    q = {"name": name}
    if token:
        q["token"] = token
    return f"splatworker://{ip}:{port}/?{urlencode(q, quote_via=quote)}"
