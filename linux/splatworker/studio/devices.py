"""Other compute devices on the LAN: SplatWork Nodes (Linux), SplatWork on a Mac with "Act as a worker" and
SplatWork Capture's Compute screen on an iPhone. All of them speak the worker API (PROTOCOL.md §4) and advertise
``_splatworker._tcp`` (TXT ``kind=mac`` / ``kind=iphone`` for the Apple ones, none for Linux).

* :class:`Device` — one worker (discovered or added by hand), with its last ``GET /v1/info``.
* :class:`DeviceStore` — ``devices.json`` next to ``studio.json``: manually added workers, access tokens, hidden
  discovered workers (0600, the tokens are secrets; never the worker's own settings).
* :class:`DeviceRegistry` — mDNS browsing (python-zeroconf, already in the Flatpak for the iPhone receiver),
  health checks every ``interval`` seconds, the device list the GUI shows and the pipeline schedules on.
* :func:`estimate_throughput`, :func:`fit_problem` — the Mac's speed table and memory forecast
  (``ComputeScheduler.estimateThroughput``, ``MemoryForecast.workerProblem``, ``TrainKit.budget``), so both
  coordinators split and place a scene the same way.

GTK-free; GUI callbacks run on the registry's threads (wrap them in ``GLib.idle_add``).
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .client import ApiError, Client

log = logging.getLogger("splatworker.studio.devices")

SERVICE_TYPE = "_splatworker._tcp.local."
DEFAULT_PORT = 7878
LOCAL_ID = "local"

# --------------------------------------------------------------------------------------------- discovery parsing


def decode_txt(props: Dict[Any, Any]) -> Dict[str, str]:
    """python-zeroconf's ``ServiceInfo.properties`` (bytes → bytes | None) as a str → str dict."""
    out: Dict[str, str] = {}
    for k, v in (props or {}).items():
        key = k.decode("utf-8", "replace") if isinstance(k, (bytes, bytearray)) else str(k)
        if v is None:
            val = ""
        elif isinstance(v, (bytes, bytearray)):
            val = v.decode("utf-8", "replace")
        else:
            val = str(v)
        out[key] = val
    return out


def parse_addrs(s: Optional[str]) -> Dict[str, str]:
    """TXT ``addrs`` = ``192.168.0.113:ethernet,192.168.0.120:wifi`` (worker ≥ 1.3.10) → address → kind."""
    out: Dict[str, str] = {}
    for part in (s or "").split(","):
        addr, sep, kind = part.strip().partition(":")
        if sep and addr:
            out[addr] = kind
    return out


def instance_name(service_name: str, service_type: str = SERVICE_TYPE) -> str:
    """``Ben's Mac._splatworker._tcp.local.`` → ``Ben's Mac``."""
    if service_name.endswith("." + service_type):
        return service_name[: -len(service_type) - 1]
    return service_name.split("._splatworker.")[0]


def preferred_address(current: str, alternates: Dict[str, str]) -> Optional[str]:
    """An ethernet address to move to, or None to stay (already on ethernet, or none known)."""
    if alternates.get(current) == "ethernet":
        return None
    eth = sorted(a for a, k in alternates.items() if k == "ethernet")
    return eth[0] if eth else None


def parse_address(text: str) -> Tuple[str, int]:
    """What the user types in "Add worker…": ``host``, ``host:port``, ``http://host:port/``, ``[v6]:port``."""
    t = text.strip()
    for scheme in ("http://", "https://"):
        if t.lower().startswith(scheme):
            t = t[len(scheme):]
    t = t.split("/")[0].strip()
    if not t:
        raise ValueError("enter the worker's address, e.g. 192.168.0.113 or 192.168.0.113:7878")
    port = DEFAULT_PORT
    if t.startswith("["):                       # [v6] or [v6]:port
        host, _, rest = t[1:].partition("]")
        if rest.startswith(":"):
            port = int(rest[1:])
    elif t.count(":") == 1:
        host, p = t.split(":")
        port = int(p)
    else:
        host = t                                # bare IPv6 or a host name
    if not host or not (0 < port < 65536):
        raise ValueError(f"not a valid address: {text!r}")
    return host, port


# --------------------------------------------------------------------------------------------- device


@dataclass
class Device:
    id: str                          # "mdns:<instance>" | "manual:<host>:<port>" | "local"
    name: str
    host: str
    port: int
    manual: bool = False
    txt: Dict[str, str] = field(default_factory=dict)
    alternates: Dict[str, str] = field(default_factory=dict)   # address → "ethernet" / "wifi"
    token: Optional[str] = None
    # health (last GET /v1/info)
    status: str = "unknown"          # unknown | checking | online | unauthorized | offline
    message: str = ""
    info: Optional[Dict[str, Any]] = None
    latency_ms: Optional[float] = None
    checked_at: Optional[float] = None
    measured: Optional[float] = None    # throughput from a saved benchmark (benchmarks.json), wins over the table

    @property
    def base_url(self) -> str:
        h = f"[{self.host}]" if ":" in self.host and not self.host.startswith("[") else self.host
        return f"http://{h}:{self.port}"

    @property
    def kind(self) -> str:
        """"linux" (SplatWork Node), "mac" or "iphone" (Apple Metal workers)."""
        k = (self.info or {}).get("kind") or self.txt.get("kind") or ""
        v = str((self.info or {}).get("workerVersion") or "")
        if k in ("mac", "iphone"):
            return k
        if v.startswith("ios"):
            return "iphone"
        if v.startswith("mac"):
            return "mac"
        return "linux"

    @property
    def online(self) -> bool:
        return self.status == "online"

    @property
    def capabilities(self) -> List[str]:
        caps = (self.info or {}).get("capabilities")
        return list(caps) if isinstance(caps, list) else ["train"]

    @property
    def can_train(self) -> bool:
        return "train" in self.capabilities

    @property
    def can_chain(self) -> bool:
        """``/v1/datasets/compose`` + datasets that outlive a job: the photos can stay on that worker."""
        return "chain" in self.capabilities

    @property
    def has_sky(self) -> bool:
        """Learned skybox sidecars (``/v1/jobs/{id}/sky.png``): SplatWork Node ≥ 1.3.15 only."""
        return self.kind == "linux"

    @property
    def has_preview(self) -> bool:
        return self.kind == "linux"

    @property
    def gpus(self) -> List[Dict[str, Any]]:
        g = (self.info or {}).get("gpus")
        return [x for x in g if isinstance(x, dict)] if isinstance(g, list) else []

    @property
    def gpu_name(self) -> str:
        gs = self.gpus
        if gs:
            names = [str(g.get("name") or "") for g in gs]
            if len(set(names)) == 1:
                return names[0] if len(names) == 1 else f"{len(names)} × {names[0]}"
            return ", ".join(names)
        return self.txt.get("gpu", "")

    @property
    def queue_length(self) -> int:
        q = (self.info or {}).get("queueLength")
        return int(q) if isinstance(q, (int, float)) and not isinstance(q, bool) else 0

    @property
    def slots(self) -> int:
        """Jobs it runs at once: one per GPU on a SplatWork Node, one on a Mac / iPhone."""
        return max(1, len(self.gpus)) if self.kind == "linux" else 1

    @property
    def throughput(self) -> float:
        return self.measured if self.measured else estimate_throughput(self.gpus)

    def client(self, timeout: float = 30.0) -> Client:
        return Client(self.base_url, self.token, timeout=timeout)

    def summary(self) -> str:
        """One line for the device list: kind · GPU · queue · status."""
        kind = {"mac": "Mac", "iphone": "iPhone", "linux": "SplatWork Node"}[self.kind]
        bits = [kind, f"{self.host}:{self.port}"]
        if self.gpu_name:
            bits.append(self.gpu_name)
        if self.status == "online":
            q = self.queue_length
            bits.append("busy" if (self.info or {}).get("busy") else "idle")
            if q:
                bits.append(f"{q} queued")
        elif self.status == "unauthorized":
            bits.append("needs its access token")
        elif self.status == "offline":
            bits.append(f"offline: {self.message}" if self.message else "offline")
        elif self.status == "checking":
            bits.append("checking…")
        return " · ".join(bits)


def apply_benchmarks(devices: Iterable[Device], config_dir: Optional[Path]) -> None:
    """Use measured speeds (Models & Benchmarks / ``splatworker benchmark``) where there are any: a record's
    ``deviceId`` is ``"local"`` for this PC, otherwise the worker's base URL (or the device id)."""
    if config_dir is None:
        return
    from .benchmark import load_benchmarks, throughput

    recs = load_benchmarks(config_dir)
    for d in devices:
        r = recs.get(d.id) or recs.get(d.base_url)
        d.measured = throughput(r) if r else None


def local_device(info: Dict[str, Any]) -> Device:
    """This PC's own worker server as a compute device (jobs go through the local client, no upload)."""
    return Device(id=LOCAL_ID, name="This PC", host="127.0.0.1", port=0, status="online", info=info)


# --------------------------------------------------------------------------------------------- throughput / fit

# Rough training speed relative to an M4 Pro Metal trainer (1.0), summed over GPUs (the Mac's table).
_SPEED = [("H200", 9), ("H100", 8), ("B200", 10), ("5090", 6), ("4090", 5), ("A100", 5), ("L40", 4.5),
          ("6000 Ada", 5), ("4080", 3.8), ("5080", 4.2), ("3090", 3), ("A6000", 3), ("4070", 2.5),
          ("3080", 2.5), ("A10", 2), ("3070", 1.8), ("T4", 0.8), ("2080", 1.5),
          ("M4 Max", 2), ("M4 Pro", 1), ("M3 Max", 1.7), ("M3 Pro", 0.8), ("M2 Max", 1.4),
          ("A18 Pro", 0.22), ("A18", 0.18), ("A17 Pro", 0.18)]
_APPLE = [e for e in _SPEED if e[0].startswith(("M4", "M3", "M2", "A18", "A17"))]


def estimate_throughput(gpus: Sequence[Dict[str, Any]]) -> float:
    total = 0.0
    for g in gpus:
        name = str(g.get("name") or "")
        low = name.lower()
        hit = None
        if name.startswith("Apple"):          # "Apple A18 Pro" must not match the CUDA "A10"
            hit = next((s for n, s in _APPLE if n.lower() in low), None)
        if hit is None:
            hit = next((s for n, s in _SPEED if n.lower() in low), None)
        if hit is None:
            mem = g.get("memoryMB")
            hit = max(0.5, float(mem if isinstance(mem, (int, float)) else 8000) / 8000)
        total += hit
    return total if total > 0 else 1.0


def _scaled_px(c, max_size: int) -> int:
    w, h = int(c.get("width") or 0), int(c.get("height") or 0)
    longest = max(w, h)
    s = max_size / longest if longest > max_size > 0 else 1.0
    return int(round(w * s)) * int(round(h * s))


def apple_budget(cameras: Sequence[Dict[str, Any]], config: Dict[str, Any], available_mb: float) -> Optional[str]:
    """``TrainKit.budget`` (Mac / iPhone workers): they shrink splats, then photos down to 320 px; only a job
    that can't fit even then is refused. Returns that refusal, or None when it fits."""
    usable = available_mb * 1_048_576 * 0.65
    fixed, per_g, per_px = 250 * 1_048_576, 2000.0, 280.0
    grid = bool(config.get("bilateralGrid", False))

    def image_bytes(size: int, grid_on: bool) -> Tuple[float, float]:
        total, largest = 0.0, 0.0
        for c in cameras:
            px = _scaled_px(c, size)
            total += px * 4 * 1.6 + int(c.get("depthWidth") or 0) * int(c.get("depthHeight") or 0) * 4
            largest = max(largest, px)
        if grid_on:
            total += len(cameras) * 12 * 8 * 16 * 16 * 4 * 3
        return total, largest

    size = int(config.get("maxImageSize") or 1600)
    img = image_bytes(size, grid)

    def room() -> float:
        return usable - fixed - img[0] - img[1] * per_px
    while room() < 150_000 * per_g and size > 480:
        size = max(480, int(size * 0.85))
        img = image_bytes(size, grid)
    if room() < 150_000 * per_g and grid:
        grid = False
        img = image_bytes(size, grid)
    while room() < 50_000 * per_g and size > 320:
        size = max(320, int(size * 0.85))
        img = image_bytes(size, grid)
    if room() < 50_000 * per_g:
        need = fixed + img[0] + img[1] * per_px + 50_000 * per_g
        return (f"{len(cameras)} photos need about {need / 1_048_576:.0f} MB even at {size} px, "
                f"{usable / 1_048_576:.0f} MB can be used")
    return None


def fit_problem(device: Device, cameras: Sequence[Dict[str, Any]], config: Dict[str, Any]) -> Optional[str]:
    """Why ``config`` on ``cameras`` would not fit on ``device`` (None = fits or nothing is known). The Mac's
    ``MemoryForecast.workerProblem``: GPU memory for the splats on CUDA workers, host RAM for the photos on
    workers that keep all of them in memory, and the device's own sizing on Macs and iPhones."""
    info = device.info or {}
    if device.kind in ("mac", "iphone"):
        avail = (info.get("memory") or {}).get("availableMB")
        if not isinstance(avail, (int, float)) or avail <= 0:
            return None
        p = apple_budget(cameras, config, float(avail))
        return f"{device.name} can't hold all {len(cameras)} photos: {p}" if p else None
    max_size = int(config.get("maxImageSize") or 1600)
    images, max_px = 0, 0
    for c in cameras:
        px = _scaled_px(c, max_size)
        images += px * 4
        max_px = max(max_px, px)
    caps = device.capabilities
    if "stream" not in caps and "train" in caps:
        ram = int((info.get("memory") or {}).get("availableMB") or 0) * 1_048_576
        if images > 3 << 30 and (ram == 0 or images * 2 > ram):
            return (f"{device.name} would hold all {len(cameras)} photos in memory "
                    f"(about {images * 2 / 2**30:.1f} GB): update SplatWork Node there to 1.3.13 or later")
    vram = max([int(g.get("memoryMB") or 0) for g in device.gpus] or [0])
    if vram > 0:
        per = 800 if config.get("optimizerPrecision") == "compact" else 1150
        g = int(config.get("maxGaussians") or 1_500_000)
        need = int(g * 1.05) * per + max_px * 200 + 1_400_000_000
        have = int(vram * 0.92) * 1_048_576
        if need > have:
            return (f"{g:,} splats need about {need / 2**30:.1f} GB of GPU memory; "
                    f"{device.name} has {have / 2**30:.1f} GB")
    return None


# --------------------------------------------------------------------------------------------- store


@dataclass
class DeviceStore:
    """``devices.json``: manual workers, tokens (by device id), hidden discovered workers."""
    path: Path
    manual: List[Dict[str, Any]] = field(default_factory=list)   # {"host","port","name"}
    tokens: Dict[str, str] = field(default_factory=dict)
    hidden: List[str] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> "DeviceStore":
        s = cls(Path(path))
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return s
        if not isinstance(data, dict):
            return s
        for m in data.get("manual") or []:
            if isinstance(m, dict) and isinstance(m.get("host"), str) and isinstance(m.get("port"), int):
                s.manual.append({"host": m["host"], "port": m["port"], "name": str(m.get("name") or m["host"])})
        toks = data.get("tokens")
        if isinstance(toks, dict):
            s.tokens = {str(k): str(v) for k, v in toks.items() if isinstance(v, str) and v}
        hid = data.get("hidden")
        if isinstance(hid, list):
            s.hidden = [str(h) for h in hid]
        return s

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump({"manual": self.manual, "tokens": self.tokens, "hidden": self.hidden}, fh, indent=1)
        os.replace(tmp, self.path)


# --------------------------------------------------------------------------------------------- registry


class DeviceRegistry:
    """Discovered + manual workers with periodic health checks. Thread-safe; ``listeners`` are called (on a
    registry thread) whenever the list or a device's health changes."""

    def __init__(self, store_path: Path, own: Optional[Callable[[], Iterable[Tuple[str, int]]]] = None,
                 own_names: Optional[Callable[[], Iterable[str]]] = None, interval: float = 15.0):
        self.store = DeviceStore.load(store_path)
        self.bench_dir = Path(store_path).parent      # benchmarks.json lives next to devices.json
        self.interval = interval
        self._own = own or (lambda: [])                    # (address, port) of this PC's worker
        self._own_names = own_names or (lambda: [])        # its advertised name(s)
        self._lock = threading.RLock()
        self._discovered: Dict[str, Device] = {}
        self._manual: Dict[str, Device] = {}
        self.listeners: List[Callable[[], None]] = []
        self._zc = None
        self._browser = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.browse_error = ""
        for m in self.store.manual:
            d = self._manual_device(m["host"], m["port"], m.get("name"))
            self._manual[d.id] = d

    # ------------------------------------------------------------ list
    def _manual_device(self, host: str, port: int, name: Optional[str]) -> Device:
        did = f"manual:{host}:{port}"
        return Device(id=did, name=name or host, host=host, port=port, manual=True, token=self.store.tokens.get(did))

    def devices(self) -> List[Device]:
        """Every known device (discovered minus hidden, this PC's own worker and retired iPhone workers, plus
        manual), by name."""
        with self._lock:
            own = set(self._own_names())
            ds = [d for d in self._discovered.values() if d.id not in self.store.hidden and d.name not in own]
            ds += list(self._manual.values())
        # iPhones no longer act as workers (SplatWork Capture trains only its own scans); older versions of
        # the app still advertise themselves, so they are left out.
        ds = [d for d in ds if d.kind != "iphone"]
        apply_benchmarks(ds, self.bench_dir)
        return sorted(ds, key=lambda d: d.name.lower())

    def get(self, device_id: str) -> Optional[Device]:
        with self._lock:
            return self._discovered.get(device_id) or self._manual.get(device_id)

    def online(self, trainers_only: bool = True) -> List[Device]:
        return [d for d in self.devices() if d.online and (d.can_train or not trainers_only)]

    def _changed(self) -> None:
        for cb in list(self.listeners):
            try:
                cb()
            except Exception:  # noqa: BLE001 — a broken listener must not stop discovery
                log.exception("device listener failed")

    # ------------------------------------------------------------ manual / tokens / hidden
    def add_manual(self, host: str, port: int = DEFAULT_PORT, name: Optional[str] = None,
                   token: Optional[str] = None, check: bool = True) -> Device:
        d = self._manual_device(host, port, name)
        with self._lock:
            self.store.manual = [m for m in self.store.manual if (m["host"], m["port"]) != (host, port)]
            self.store.manual.append({"host": host, "port": port, "name": d.name})
            if token:
                self.store.tokens[d.id] = token
                d.token = token
            self._manual[d.id] = d
            self.store.save()
        self._changed()
        if check:
            self.check(d.id)
        return d

    def remove(self, device_id: str) -> None:
        """Manual: forget it (and its token). Discovered: hide it until :meth:`unhide_all`."""
        with self._lock:
            if device_id in self._manual:
                d = self._manual.pop(device_id)
                self.store.manual = [m for m in self.store.manual if (m["host"], m["port"]) != (d.host, d.port)]
            elif device_id not in self.store.hidden:
                self.store.hidden.append(device_id)
            if device_id.startswith("manual:"):
                self.store.tokens.pop(device_id, None)
            self.store.save()
        self._changed()

    def unhide_all(self) -> None:
        with self._lock:
            self.store.hidden = []
            self.store.save()
        self._changed()

    def set_token(self, device_id: str, token: Optional[str]) -> None:
        with self._lock:
            if token:
                self.store.tokens[device_id] = token
            else:
                self.store.tokens.pop(device_id, None)
            d = self.get(device_id)
            if d is not None:
                d.token = token or None
            self.store.save()
        self.check(device_id)

    # ------------------------------------------------------------ discovery (zeroconf)
    def found(self, name: str, host: str, port: int, txt: Dict[str, str]) -> Optional[Device]:
        """A resolved ``_splatworker._tcp`` service (also the test entry point)."""
        if (host, port) in set(self._own()):
            return None
        did = f"mdns:{name}"
        with self._lock:
            d = self._discovered.get(did)
            if d is None:
                d = Device(id=did, name=name, host=host, port=port, txt=dict(txt),
                           alternates=parse_addrs(txt.get("addrs")), token=self.store.tokens.get(did))
                self._discovered[did] = d
                new = True
            else:
                d.txt = dict(txt)
                d.alternates.update(parse_addrs(txt.get("addrs")))
                d.port = port
                if d.status != "online":
                    d.host = host
                new = False
        self._changed()
        if new:
            self.check(did)
        return d

    def lost(self, name: str) -> None:
        with self._lock:
            self._discovered.pop(f"mdns:{name}", None)
        self._changed()

    def start(self) -> None:
        """Browse mDNS and health-check every ``interval`` seconds (idempotent)."""
        if self._thread is not None:
            return
        self._stop.clear()
        try:
            self._start_browser()
        except Exception as e:  # noqa: BLE001 — no multicast, no zeroconf: manual workers still work
            self.browse_error = str(e) or e.__class__.__name__
            log.warning("worker discovery unavailable: %s", self.browse_error)
        self._thread = threading.Thread(target=self._health_loop, daemon=True, name="device-health")
        self._thread.start()

    def _start_browser(self) -> None:
        from zeroconf import IPVersion, ServiceBrowser, ServiceStateChange, Zeroconf

        reg = self

        def on_change(zeroconf, service_type, name, state_change):
            if state_change is ServiceStateChange.Removed:
                reg.lost(instance_name(name, service_type))
                return
            threading.Thread(target=reg._resolve, args=(zeroconf, service_type, name), daemon=True).start()

        self._zc = Zeroconf(ip_version=IPVersion.V4Only)
        self._browser = ServiceBrowser(self._zc, SERVICE_TYPE, handlers=[on_change])

    def _resolve(self, zc, service_type: str, name: str) -> None:
        try:
            info = zc.get_service_info(service_type, name, timeout=3000)
        except Exception:  # noqa: BLE001
            info = None
        if info is None:
            return
        addrs = [a for a in info.parsed_addresses() if ":" not in a]
        if not addrs or not info.port:
            return
        txt = decode_txt(info.properties)
        alt = parse_addrs(txt.get("addrs"))
        host = preferred_address(addrs[0], alt) or addrs[0]
        self.found(instance_name(name, service_type), host, int(info.port), txt)

    def stop(self) -> None:
        self._stop.set()
        if self._browser is not None:
            try:
                self._browser.cancel()
            except Exception:  # noqa: BLE001
                pass
            self._browser = None
        if self._zc is not None:
            try:
                self._zc.close()
            except Exception:  # noqa: BLE001
                pass
            self._zc = None
        self._thread = None

    # ------------------------------------------------------------ health
    def _health_loop(self) -> None:
        while not self._stop.is_set():
            self.refresh()
            self._stop.wait(self.interval)

    def refresh(self) -> None:
        ts = [threading.Thread(target=self.check, args=(d.id,), daemon=True) for d in self.devices()]
        for t in ts:
            t.start()
        for t in ts:
            t.join(10)

    def check(self, device_id: str, timeout: float = 4.0) -> Optional[Device]:
        """``GET /v1/info``; a discovered worker that stops answering moves to another of its addresses, and
        one on Wi-Fi moves to its ethernet address when that answers. Manual workers keep the typed address."""
        d = self.get(device_id)
        if d is None:
            return None
        if d.info is None:
            d.status = "checking"
            self._changed()
        t0 = time.monotonic()
        try:
            info = Client(d.base_url, d.token, timeout=timeout).info()
            self._online(d, info, t0)
            if not d.manual:
                for a in info.get("addresses") or []:
                    if isinstance(a, dict) and isinstance(a.get("address"), str):
                        d.alternates[a["address"]] = str(a.get("kind") or d.alternates.get(a["address"], ""))
                target = preferred_address(d.host, d.alternates)
                if target and self._probe(d, target, timeout=2):
                    d.host = target
        except ApiError as e:
            if e.status == 401:
                d.status, d.message = "unauthorized", "the worker wants an access token"
            elif e.status is None and not d.manual and self._move(d, timeout=2):
                self._online(d, d.info or {}, t0)
            else:
                d.status, d.message = "offline", e.message
        except Exception as e:  # noqa: BLE001 — bad JSON etc.
            d.status, d.message = "offline", str(e)
        d.checked_at = time.time()
        self._changed()
        return d

    @staticmethod
    def _online(d: Device, info: Dict[str, Any], t0: float) -> None:
        d.info = info if isinstance(info, dict) else {}
        d.status, d.message = "online", ""
        d.latency_ms = (time.monotonic() - t0) * 1000
        if d.manual and d.name == d.host and isinstance(d.info.get("name"), str) and d.info["name"]:
            d.name = d.info["name"]          # "192.168.0.113" → the worker's own name

    def _probe(self, d: Device, host: str, timeout: float) -> Optional[Dict[str, Any]]:
        probe = dataclasses.replace(d, host=host)
        try:
            return Client(probe.base_url, d.token, timeout=timeout).info()
        except Exception:  # noqa: BLE001
            return None

    def _move(self, d: Device, timeout: float) -> bool:
        order = sorted((a for a in d.alternates if a != d.host),
                       key=lambda a: (0 if d.alternates[a] == "ethernet" else 1, a))
        for a in order:
            info = self._probe(d, a, timeout)
            if info is not None:
                d.host, d.info = a, info
                return True
        return False
