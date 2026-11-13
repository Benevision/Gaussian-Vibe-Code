"""Training on other compute devices: one chosen worker, or every device block-parallel (PROTOCOL.md §7).

Called from :meth:`LocalPipeline.train` when ``ProjectOptions.train_on`` is not ``"local"``; everything before
training (frames, masks, camera solve) ran on this PC's worker. Placements:

* ``"<device id>"`` — upload the dataset to that worker, train there, download ``result.ply`` (+ the learned
  skybox from Linux workers) into the project's ``outputs/``. A device that is offline falls back to this PC;
  a scene that would not fit it is split into blocks on that device alone.
* ``"all"`` — this PC plus every online worker. Scenes below ``split_above`` photos train whole on the device
  expected to finish first (this PC returns ``None``: the normal local path with live previews); bigger ones,
  or ones no single device can hold, are split (:mod:`.blocks`), distributed by throughput and memory, trained
  concurrently (one job per GPU on a Linux worker) and merged. A block that fails on another machine is
  retried on this PC.

Uploads: a zip written to the project folder (photos stored, not recompressed) and streamed with a
Content-Length; downloads stream to disk. Workers with ``chain`` (SplatWork for Linux ≥ 1.3.8) keep the photos as a
dataset of their own, recorded in ``dataset/.remote-uploads.json``: later runs send only ``dataset.json``,
masks and points and compose them over the stored photos. Macs and iPhones (no compose; they drop older
uploads when a new one arrives) get one full upload per run, shared by all their blocks.
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from . import blocks as B
from .client import ApiError, Client
from .devices import LOCAL_ID, Device, DeviceRegistry, fit_problem, local_device
from .pipeline import Cancelled, PipelineError

SPLIT_ABOVE = 400            # the Mac's "Split into blocks above" default
UPLOADS_FILE = ".remote-uploads.json"
LOCAL_BONUS = 1.15           # no upload: prefer this PC when scores tie
MAX_OFFLINE = 12 * 3600.0
#: A block on a device that stays unreachable this long is given up there and trained again on this PC, instead of
#: holding up the whole block-parallel run for MAX_OFFLINE.
BLOCK_MAX_OFFLINE = 30 * 60.0


# --------------------------------------------------------------------------------------------- packaging


def referenced_files(ds: Path, doc: Dict[str, Any], cameras: Optional[Set[int]] = None) -> Tuple[List[str], List[str]]:
    """(photos, other files) a training upload needs: each kept camera's image / mask / sky / depth and the
    initial points. Intermediates next to them (raw mask layers, the solver's work folders) stay here."""
    photos, other = [], []
    seen: Set[str] = set()

    def add(rel: Any, bucket: List[str]) -> None:
        if not isinstance(rel, str) or not rel or rel.startswith("/") or ".." in Path(rel).parts:
            return
        if rel not in seen and (ds / rel).is_file():
            seen.add(rel)
            bucket.append(rel)
    for c in doc.get("cameras") or []:
        if cameras is not None and c.get("id") not in cameras:
            continue
        add(c.get("image"), photos)
        for k in ("mask", "sky", "depth", "confidence"):
            add(c.get(k), other)
    add(doc.get("points") or ("points3d.ply" if (ds / "points3d.ply").is_file() else None), other)
    return photos, other


def subset_doc(doc: Dict[str, Any], cameras: Optional[Set[int]]) -> Dict[str, Any]:
    if cameras is None:
        return doc
    d = dict(doc)
    d["cameras"] = [c for c in doc.get("cameras") or [] if c.get("id") in cameras]
    return d


def write_zip(ds: Path, files: Sequence[str], dest: Path, doc: Optional[Dict[str, Any]] = None,
              check: Optional[Callable[[], None]] = None) -> int:
    """Zip ``files`` (paths relative to ``ds``) at the archive root, plus ``dataset.json`` when ``doc`` is
    given. Photos are stored (already compressed), read in blocks by ``zipfile``. Returns the zip's size."""
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_STORED, allowZip64=True) as zf:
        if doc is not None:
            zf.writestr("dataset.json", json.dumps(doc), compress_type=zipfile.ZIP_DEFLATED)
        for i, rel in enumerate(files):
            if check and i % 64 == 0:
                check()
            low = rel.lower()
            method = zipfile.ZIP_STORED if low.endswith((".jpg", ".jpeg", ".png", ".heic", ".webp")) \
                else zipfile.ZIP_DEFLATED
            zf.write(ds / rel, rel, compress_type=method)
    return dest.stat().st_size


def _signature(ds: Path, files: Sequence[str]) -> Dict[str, List[int]]:
    out = {}
    for rel in files:
        st = (ds / rel).stat()
        out[rel] = [st.st_size, st.st_mtime_ns]
    return out


# --------------------------------------------------------------------------------------------- the run


@dataclass
class Slot:
    device: Device
    client: Client
    local: bool


class ComputeRun:
    """One training stage of a :class:`~.pipeline.LocalPipeline` placed on other devices."""

    poll_seconds = 2.0

    def __init__(self, pipeline, registry: Optional[DeviceRegistry], config: Dict[str, Any]):
        self.p = pipeline
        self.reg = registry
        self.cfg = config
        self.ds: Path = pipeline.project.dataset_dir
        self.doc = json.loads((self.ds / "dataset.json").read_text(encoding="utf-8"))
        self.cams = [c for c in self.doc.get("cameras") or []]
        self.cancel = pipeline.cancel_event
        self._upload_lock: Dict[str, threading.Lock] = {}
        self._uploaded: Dict[str, str] = {}             # device id → dataset id for this run
        self._cleanup: List[Tuple[Client, str]] = []
        self._lock = threading.Lock()
        self.local: Optional[Device] = None
        try:
            info = pipeline.client.info()
            if "train" in (info.get("capabilities") or []):
                self.local = local_device(info)
                if self.reg is not None:
                    from .devices import apply_benchmarks
                    apply_benchmarks([self.local], getattr(self.reg, "bench_dir", None))
        except ApiError:
            self.local = None

    # ------------------------------------------------------------ helpers
    def log(self, line: str) -> None:
        self.p._log(line)

    def check(self) -> None:
        if self.cancel.is_set():
            raise Cancelled()

    def client_for(self, d: Device) -> Client:
        return self.p.client if d.id == LOCAL_ID else d.client(timeout=60)

    def cameras_of(self, ids: Optional[Set[int]]) -> List[Dict[str, Any]]:
        return [c for c in self.cams if ids is None or c.get("id") in ids]

    def remote_devices(self) -> List[Device]:
        if self.reg is None:
            return []
        out = []
        for d in self.reg.devices():
            self.reg.check(d.id)
            if d.online and d.can_train and not self._is_this_pc(d):
                out.append(d)
        return out

    def _is_this_pc(self, d: Device) -> bool:
        """This PC's own worker, found by mDNS (same name and LAN addresses as the local server's info)."""
        if self.local is None or d.kind != "linux":
            return False
        mine, theirs = self.local.info or {}, d.info or {}
        addrs = lambda i: {a.get("address") for a in i.get("addresses") or [] if isinstance(a, dict)}  # noqa: E731
        return mine.get("name") == theirs.get("name") and bool(addrs(mine) & addrs(theirs))

    @staticmethod
    def score(d: Device) -> float:
        return d.throughput / (1.0 + d.queue_length) * (LOCAL_BONUS if d.id == LOCAL_ID else 1.0)

    # ------------------------------------------------------------ entry points
    def run(self, target: str, split_above: int = SPLIT_ABOVE) -> Optional[Path]:
        if target == "all":
            return self.everywhere(split_above)
        d = self.reg.get(target) if self.reg else None
        if d is not None:
            self.reg.check(d.id)
        if d is None or not d.online or not d.can_train:
            why = "is not known" if d is None else (f"is {d.status}" + (f" ({d.message})" if d.message else ""))
            if d is not None and d.online:
                why = "can't train"
            self.log(f"{d.name if d else target} {why}: training on this PC instead")
            return None
        problem = fit_problem(d, self.cams, self.cfg)
        if problem:
            self.log(f"{problem} — splitting the scene into blocks on {d.name}")
            return self.blocks([d])
        return self.whole(d)

    def everywhere(self, split_above: int) -> Optional[Path]:
        devices = ([self.local] if self.local else []) + self.remote_devices()
        if not devices:
            raise PipelineError("no device can train (this PC's worker has no training capability)")
        names = ", ".join(d.name for d in devices)
        self.log(f"compute devices: {names}")
        fits = [d for d in devices if fit_problem(d, self.cams, self.cfg) is None]
        train_cams = sum(1 for c in self.cams if not c.get("holdout"))
        if fits and (len(devices) == 1 or train_cams < split_above):
            best = max(fits, key=self.score)
            if len(devices) > 1:
                self.log(f"{train_cams} photos is too small to split (below {split_above}): "
                         f"training the whole scene on {best.name}, the faster device")
            return None if best.id == LOCAL_ID else self.whole(best)
        if not fits:
            self.log("the whole scene fits on no single device: splitting it into blocks")
        return self.blocks(devices)

    # ------------------------------------------------------------ upload
    def upload(self, d: Device, cameras: Optional[Set[int]]) -> str:
        """The dataset id on ``d`` for ``cameras`` (None = all): uploaded once per device and run."""
        if d.id == LOCAL_ID:
            return self.p._register(self.ds)
        with self._lock:
            lk = self._upload_lock.setdefault(d.id, threading.Lock())
        with lk:
            if d.id in self._uploaded:
                return self._uploaded[d.id]
            did = self._upload(d, cameras)
            self._uploaded[d.id] = did
            return did

    def _upload(self, d: Device, cameras: Optional[Set[int]]) -> str:
        client = self.client_for(d)
        photos, other = referenced_files(self.ds, self.doc, cameras)
        doc = subset_doc(self.doc, cameras)
        tmp = self.p.project.root / f".upload-{uuid.uuid4().hex[:8]}.zip"
        last = {"t": 0.0}

        def progress(sent: int, total: int) -> None:
            now = time.monotonic()
            if now - last["t"] > 0.5 or sent == total:
                last["t"] = now
                self.p._progress(sent / max(total, 1), f"Uploading to {d.name}: {sent / 2**20:.0f} / "
                                                       f"{total / 2**20:.0f} MB")
            if self.cancel.is_set():
                raise Cancelled()
        try:
            if d.can_chain:
                images = self._stored_photos(d, client, photos)
                if images is None:
                    self.log(f"uploading {len(photos)} photos to {d.name}")
                    write_zip(self.ds, photos, tmp, check=self.check)
                    images = client.upload_dataset(tmp, progress)
                    self._remember(d, images, photos)
                else:
                    self.log(f"{d.name} still has the photos: sending only dataset.json, masks and points")
                write_zip(self.ds, other, tmp, doc=doc, check=self.check)
                delta = client.upload_dataset(tmp, progress)
                self._cleanup.append((client, delta))
                try:
                    did = client.compose([{"datasetId": images}, {"datasetId": delta}])["datasetId"]
                except ApiError as e:
                    if e.status != 404:
                        raise
                    self._forget(d)                       # the photos are gone from the worker
                    return self._upload_full(d, client, photos + other, doc, tmp, progress)
                self._cleanup.append((client, did))
                return did
            return self._upload_full(d, client, photos + other, doc, tmp, progress)
        finally:
            tmp.unlink(missing_ok=True)

    def _upload_full(self, d: Device, client: Client, files: List[str], doc: Dict[str, Any], tmp: Path,
                     progress) -> str:
        size = write_zip(self.ds, files, tmp, doc=doc, check=self.check)
        self.log(f"uploading the dataset to {d.name} ({len(files)} files, {size / 2**20:.0f} MB)")
        did = client.upload_dataset(tmp, progress)
        self._cleanup.append((client, did))
        return did

    def _records(self) -> Dict[str, Any]:
        try:
            r = json.loads((self.ds / UPLOADS_FILE).read_text(encoding="utf-8"))
            return r if isinstance(r, dict) else {}
        except (OSError, ValueError):
            return {}

    def _write_records(self, r: Dict[str, Any]) -> None:
        tmp = self.ds / (UPLOADS_FILE + ".tmp")
        tmp.write_text(json.dumps(r), encoding="utf-8")
        os.replace(tmp, self.ds / UPLOADS_FILE)

    def _stored_photos(self, d: Device, client: Client, photos: List[str]) -> Optional[str]:
        rec = self._records().get(d.base_url)
        if not isinstance(rec, dict) or not isinstance(rec.get("datasetId"), str):
            return None
        files = rec.get("files") or {}
        sig = _signature(self.ds, photos)
        if any(files.get(k) != v for k, v in sig.items()):
            return None
        try:
            if rec["datasetId"] not in {x.get("datasetId") for x in client.datasets()}:
                return None
        except ApiError:
            return None
        return rec["datasetId"]

    def _remember(self, d: Device, dataset_id: str, photos: List[str]) -> None:
        with self._lock:
            r = self._records()
            old = r.get(d.base_url)
            if isinstance(old, dict) and old.get("datasetId") and old["datasetId"] != dataset_id:
                self.client_for(d).delete_dataset(old["datasetId"])
            r[d.base_url] = {"datasetId": dataset_id, "files": _signature(self.ds, photos), "at": time.time()}
            self._write_records(r)

    def _forget(self, d: Device) -> None:
        with self._lock:
            r = self._records()
            r.pop(d.base_url, None)
            self._write_records(r)

    def cleanup(self) -> None:
        for client, did in self._cleanup:
            client.delete_dataset(did)
        self._cleanup.clear()

    # ------------------------------------------------------------ one job
    def train_job(self, d: Device, dataset_id: str, cfg: Dict[str, Any], name: str, dest: Path,
                  on_status: Callable[[Dict[str, Any]], None], with_preview: bool = False,
                  max_offline: float = MAX_OFFLINE) -> Tuple[Path, bool]:
        """Run one training job on ``d`` and download its result to ``dest`` (+ sky sidecars). Returns
        (dest, has_sky)."""
        client = self.client_for(d)
        local = d.id == LOCAL_ID
        job = client.train(dataset_id, cfg, name, self.p.priority if local else None)
        self.log(f"{name}: job {job} on {d.name}")
        key = f"{d.id}/{job}"
        if not local and hasattr(self.p, "track_remote"):
            self.p.track_remote(key, client, job)       # cancelled synchronously on quit
        try:
            return self._finish_job(client, d, job, name, dest, on_status, with_preview, local, max_offline)
        finally:
            if not local and hasattr(self.p, "track_remote"):
                self.p.track_remote(key, None, None)

    def _finish_job(self, client: Client, d: Device, job: str, name: str, dest: Path,
                    on_status: Callable[[Dict[str, Any]], None], with_preview: bool,
                    local: bool, max_offline: float = MAX_OFFLINE) -> Tuple[Path, bool]:
        preview = self.p.project.outputs_dir / ".preview.splat"
        last = {"step": None, "t": 0.0}

        def status(s: Dict[str, Any]) -> None:
            on_status(s)
            step = s.get("previewStep")
            if with_preview and step and step != last["step"] and \
                    time.monotonic() - last["t"] > self.p.preview_seconds:
                try:
                    got = client.download(f"/v1/jobs/{job}/preview.splat", preview, missing_ok=True)
                except ApiError:
                    got = None
                if got:
                    last["step"], last["t"] = step, time.monotonic()
                    self.p.emit("preview", got)
        client = self.wait(client, d, job, status, max_offline)     # the device may have moved to another address
        self.fetch(client, d, f"/v1/jobs/{job}/result.ply", dest)
        sky = False
        if d.has_sky:
            got = [client.download(f"/v1/jobs/{job}/sky.{e}", dest.with_suffix(f".sky.{e}"), missing_ok=True)
                   for e in ("png", "json")]
            sky = all(got)
        if with_preview:
            preview.unlink(missing_ok=True)
        if not local:
            try:
                client.delete(f"/v1/jobs/{job}")      # the result is here now; free the worker's disk
            except ApiError:
                pass
        return dest, sky

    def wait(self, client: Client, d: Device, job: str, on_status: Callable[[Dict[str, Any]], None],
             max_offline: float = MAX_OFFLINE) -> Client:
        """Poll until the job ends; returns the client that reached it. Network trouble (Wi-Fi drop, the Mac
        asleep, a restart) is ridden out with growing gaps up to 30 s for ``max_offline``; when the device registry
        has moved the device to another address (Wi-Fi → ethernet), polling follows it. Cancelling cancels the
        job there."""
        offline_since: Optional[float] = None
        backoff = self.poll_seconds
        while True:
            if self.cancel.is_set():
                try:
                    client.cancel(job)
                except ApiError:
                    pass
                raise Cancelled()
            try:
                s = client.job(job)
            except ApiError as e:
                transient = e.status is None or (e.status >= 500 and e.status != 501)
                if not transient:
                    raise PipelineError(f"{d.name}: {e.message}")
                offline_since = offline_since or time.monotonic()
                gone = time.monotonic() - offline_since
                cur = self.reg.get(d.id) if (self.reg is not None and d.id != LOCAL_ID) else None
                if cur is not None and cur.base_url != d.base_url:
                    self.log(f"{d.name} moved to {cur.base_url}; following it")
                    d, client = cur, self.client_for(cur)
                    continue
                if gone > max_offline:
                    raise PipelineError(f"{d.name} unreachable for {gone / 60:.0f} min: {e.message}")
                if backoff == self.poll_seconds:
                    self.log(f"{d.name} is not answering ({e.message}); waiting — the job keeps running there")
                self.cancel.wait(backoff)
                backoff = min(backoff * 2, 30.0)
                continue
            if offline_since is not None:
                self.log(f"{d.name} is back")
            offline_since, backoff = None, self.poll_seconds
            on_status(s)
            state = s.get("state")
            if state == "completed":
                return client
            if state == "cancelled":
                if self.cancel.is_set():
                    raise Cancelled()
                raise PipelineError(f"the job was cancelled on {d.name}")
            if state == "failed":
                raise PipelineError(f"{d.name}: {s.get('message') or 'training failed'}")
            self.cancel.wait(self.poll_seconds)

    def fetch(self, client: Client, d: Device, path: str, dest: Path) -> None:
        start, backoff = time.monotonic(), 2.0
        while True:
            try:
                client.download(path, dest)
                return
            except ApiError as e:
                if not (e.status is None or e.status >= 500) or time.monotonic() - start > 3600:
                    raise PipelineError(f"{d.name}: couldn't download the result: {e.message}")
                self.check()
                self.cancel.wait(backoff)
                backoff = min(backoff * 2, 30.0)

    # ------------------------------------------------------------ whole scene on one other device
    def whole(self, d: Device) -> Path:
        self.log(f"training the whole scene on {d.name} ({d.summary()})")
        self.p.project.update(lastJob={"device": d.name, "startedAt": time.time()})
        try:
            did = self.upload(d, None)
            self.check()
            out = self.p.project.new_output_path()
            self.train_job(d, did, self.cfg, self.p.project.name, out,
                           lambda s: self._job_event(s, d), with_preview=d.has_preview)
        finally:
            self.cleanup()
        self.p.project.update(lastJob={"device": d.name, "output": out.name, "finishedAt": time.time()})
        self.log(f"saved {out.name}")
        return out

    def _job_event(self, s: Dict[str, Any], d: Device) -> None:
        s = dict(s)
        s.setdefault("kind", "train")
        if s.get("message"):
            s["message"] = f"{d.name}: {s['message']}"
        self.p.emit("job", s)
        total = s.get("totalSteps") or 0
        if total:
            self.p._progress(min(1.0, (s.get("step") or 0) / total), s.get("message") or "")

    # ------------------------------------------------------------ blocks
    def blocks(self, devices: List[Device]) -> Path:
        pts = self._points()
        by_id = {d.id: d for d in devices}

        def fits(dev: str, b: B.Block, all_blocks: Sequence[B.Block]) -> bool:
            return fit_problem(by_id[dev], self.cameras_of(set(b.camera_ids)),
                               B.block_config(self.cfg, b, all_blocks)) is None
        plan = B.plan_blocks(self.doc, pts, len(devices),
                             lambda b, bs: any(fits(d.id, b, bs) for d in devices))
        if not plan:
            raise PipelineError("the scene could not be split into blocks (no cameras)")
        empty = [b for b in plan if not b.camera_ids]
        plan = [b for b in plan if b.camera_ids]
        if empty:
            self.log(f"{len(empty)} blocks are seen by no camera and were left out")
        assignment = B.distribute(plan, {d.id: d.throughput for d in devices},
                                  lambda dev, b: fits(dev, b, plan))
        for dev, bs in assignment.items():
            self.log(f"{by_id[dev].name}: blocks {', '.join(str(b.index + 1) for b in bs)} "
                     f"({sum(len(b.camera_ids) for b in bs)} camera views)")
        self.log(f"training {len(plan)} blocks on {len(assignment)} devices")
        self.p.project.update(lastJob={"device": "block-parallel", "blocks": len(plan), "startedAt": time.time()})
        work = self.p.project.outputs_dir / f".blocks-{uuid.uuid4().hex[:6]}"
        work.mkdir(parents=True, exist_ok=True)
        tracker = _Tracker(self, plan, int(self.cfg.get("iterations") or 30000))
        results: Dict[int, Tuple[B.Block, Path, bool]] = {}
        failed: List[Tuple[B.Block, str, str]] = []
        threads = []
        try:
            for dev, bs in assignment.items():
                d = by_id[dev]
                q = list(sorted(bs, key=B.block_cost, reverse=True))
                need = set().union(*[set(b.camera_ids) for b in bs])
                qlock = threading.Lock()
                for _slot in range(d.slots):
                    t = threading.Thread(target=self._block_worker, daemon=True,
                                         args=(d, q, qlock, need, plan, work, tracker, results, failed))
                    t.start()
                    threads.append(t)
            for t in threads:
                while t.is_alive():
                    t.join(0.5)
            self.check()
            retry = [(b, dev, msg) for b, dev, msg in failed]
            for b, dev, msg in retry:
                if self.local is None or dev == LOCAL_ID:
                    raise PipelineError(f"block {b.index + 1} failed on {by_id[dev].name}: {msg}")
                self.log(f"block {b.index + 1} failed on {by_id[dev].name} ({msg}): retrying it on this PC")
                self._run_block(self.local, b, None, plan, work, tracker, results)
            self.check()
            parts = [(results[i][0], results[i][1]) for i in sorted(results)]
            out = self.p.project.new_output_path()
            self.p._progress(1.0, "Merging blocks")
            n = B.merge_plys(parts, out)
            sky = [results[i] for i in sorted(results) if results[i][2]]
            if sky:
                best = max(sky, key=lambda r: len(r[0].camera_ids))
                for e in ("png", "json"):
                    shutil.copyfile(best[1].with_suffix(f".sky.{e}"), out.with_suffix(f".sky.{e}"))
            self.log(f"merged {len(parts)} blocks: {n:,} splats → {out.name}")
        finally:
            self.cleanup()
            shutil.rmtree(work, ignore_errors=True)
        self.p.project.update(lastJob={"device": "block-parallel", "blocks": len(plan), "output": out.name,
                                       "finishedAt": time.time()})
        return out

    def _points(self) -> Optional[np.ndarray]:
        rel = self.doc.get("points") or "points3d.ply"
        p = self.ds / rel
        if not p.is_file():
            return None
        try:
            from ..ply import read_ply_vertices

            v = read_ply_vertices(p)
            return np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
        except Exception as e:  # noqa: BLE001 — blocks still work from the cameras alone
            self.log(f"initial points unreadable ({e}): splitting by cameras only")
            return None

    def _block_worker(self, d: Device, q: List[B.Block], qlock: threading.Lock, need: Set[int],
                      plan: List[B.Block], work: Path, tracker: "_Tracker",
                      results: Dict, failed: List) -> None:
        while not self.cancel.is_set():
            with qlock:
                if not q:
                    return
                b = q.pop(0)
            try:
                self._run_block(d, b, need, plan, work, tracker, results)
            except Cancelled:
                return
            except Exception as e:  # noqa: BLE001 — reported, retried on this PC
                msg = getattr(e, "message", None) or str(e) or e.__class__.__name__
                with self._lock:
                    failed.append((b, d.id, msg))
                if d.id != LOCAL_ID:
                    with qlock:              # the device is likely gone: hand its remaining blocks over too
                        for rest in q:
                            failed.append((rest, d.id, "skipped after an earlier failure"))
                        q.clear()

    def _run_block(self, d: Device, b: B.Block, need: Optional[Set[int]], plan: List[B.Block], work: Path,
                   tracker: "_Tracker", results: Dict) -> None:
        did = self.upload(d, need)
        self.check()
        cfg = B.block_config(self.cfg, b, plan)
        name = f"{self.p.project.name} block {b.index + 1}/{len(plan)}"
        dest = work / f"block-{b.index:02d}.ply"
        tracker.start(b, d)
        _, sky = self.train_job(d, did, cfg, name, dest, lambda s: tracker.update(b, d, s),
                                max_offline=BLOCK_MAX_OFFLINE)
        tracker.finish(b)
        with self._lock:
            results[b.index] = (b, dest, sky)


class _Tracker:
    """Combined progress of all blocks, emitted as one ``job`` event (steps and splats summed, PSNR averaged)
    so the Projects dashboard and charts work unchanged."""

    def __init__(self, run: ComputeRun, plan: List[B.Block], iterations: int):
        self.run = run
        self.total = iterations * len(plan)
        self.count = len(plan)
        self.steps: Dict[int, int] = {}
        self.latest: Dict[int, Dict[str, Any]] = {}
        self.where: Dict[int, str] = {}
        self.done: Set[int] = set()
        self.t0 = time.monotonic()
        self.last_emit = 0.0
        self.lock = threading.Lock()
        self.iterations = iterations

    def start(self, b: B.Block, d: Device) -> None:
        with self.lock:
            self.where[b.index] = d.name

    def update(self, b: B.Block, d: Device, s: Dict[str, Any]) -> None:
        with self.lock:
            self.steps[b.index] = max(self.steps.get(b.index, 0), int(s.get("step") or 0))
            self.latest[b.index] = s
            self._emit()

    def finish(self, b: B.Block) -> None:
        with self.lock:
            self.done.add(b.index)
            self.steps[b.index] = self.iterations
            self._emit(force=True)

    def _emit(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_emit < 1.0:
            return
        self.last_emit = now
        step = min(self.total, sum(self.steps.values()))
        frac = step / max(self.total, 1)
        psnrs = [s.get("psnr") for s in self.latest.values() if isinstance(s.get("psnr"), (int, float))]
        running = [f"block {i + 1} on {self.where[i]}" for i in sorted(self.where) if i not in self.done]
        msg = f"{len(self.done)} of {self.count} blocks done" + (f" · {', '.join(running[:3])}" if running else "")
        elapsed = now - self.t0
        status = {"jobId": "blocks", "kind": "train", "state": "training", "step": step, "totalSteps": self.total,
                  "psnr": sum(psnrs) / len(psnrs) if psnrs else None,
                  "numGaussians": sum(int(s.get("numGaussians") or 0) for s in self.latest.values()),
                  "elapsedSeconds": elapsed, "etaSeconds": elapsed * (1 - frac) / frac if frac > 0.01 else None,
                  "message": msg}
        self.run.p.emit("job", status)
        self.run.p._progress(frac, msg)


def train_elsewhere(pipeline, config: Dict[str, Any]) -> Optional[Path]:
    """The pipeline hook. Returns the output PLY, or None to train on this PC as usual."""
    target = getattr(pipeline.o, "train_on", "local") or "local"
    if target == "local":
        return None
    run = ComputeRun(pipeline, getattr(pipeline, "devices", None), config)
    split = getattr(pipeline.o, "split_above", None)
    return run.run(target, SPLIT_ABOVE if split is None else max(0, int(split)))
