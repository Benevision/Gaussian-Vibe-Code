"""New Project → prepare → train on this PC, as jobs of the local worker server.

The project folder is the single source of truth. Each GPU step registers ``<project>/dataset`` (or the video's
folder) with ``POST /v1/datasets/local``, runs a ``/v1/process`` or ``/v1/jobs`` request exactly as the Mac
would on a remote worker, and copies the output back with ``POST /v1/datasets/{id}/export-local`` (hard links).
Jobs are submitted with ``priority: 1`` so the person at this PC goes ahead of queued remote jobs (never
pre-empting a running one); local and remote jobs share the server's per-GPU queue.

Stages (as on the Mac, ``DatasetBuilder``): import / frames → model masks (stacked) → sky layer → camera solve →
dataset.json (masks, sky, holdout) → training. Events go to ``on_event(kind, data)``:

* ``("stage", {"index", "count", "title"})``   ``("progress", {"fraction", "message"})``  ``("log", line)``
* ``("job", status)`` — every poll of a server job (training: step, PSNR, splats, ETA …)
* ``("preview", path)`` — a new live preview ``.splat`` was downloaded
* ``("done", {"output": path | None, "dataset": path})``  ``("failed", message)``  ``("cancelled", None)``
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from . import importers
from .client import ApiError, Client
from .presets import ProjectOptions
from .projects import Project

Event = Callable[[str, Any], None]


class Cancelled(Exception):
    pass


class PipelineError(RuntimeError):
    pass


def stage_titles(o: ProjectOptions, train: bool = True) -> List[str]:
    s: List[str] = []
    if o.source_kind in ("video", "video360", "stereo"):
        s.append("Extracting frames")
    elif o.source_kind == "photos360":
        s.append("Splitting 360° photos")
    elif o.source_kind in ("photos", "capture"):
        s.append("Preparing images")
    elif o.source_kind in ("colmap", "nerfstudio", "dataset"):
        s.append("Importing cameras")
    if o.source_kind not in ("colmap", "nerfstudio", "dataset"):
        for m in o.mask_models:
            if m != "none":
                s.append(f"Masks ({m})")
        if getattr(o, "mask_static", False):
            s.append("Fixed parts of the frame")
        if o.sky:
            s.append("Finding the sky")
        s.append("Solving cameras")
        if o.source_kind == "stereo" and getattr(o, "stereo_depth", False):
            s.append("Stereo depth")
    s.append("Finalising dataset")
    if train:
        s.append("Training")
    return s


class LocalPipeline:
    poll_seconds = 1.0
    preview_seconds = 8.0

    def __init__(self, client: Client, project: Project, options: ProjectOptions, on_event: Event,
                 train: Optional[bool] = None, priority: int = 1):
        self.client = client
        self.project = project
        self.o = options.with_storage_for_preset()
        self.emit = on_event
        self.train_enabled = options.auto_train if train is None else train
        self.priority = priority
        self.cancel_event = threading.Event()
        self.current_job: Optional[str] = None
        # jobs this pipeline has running on other machines (studio/compute.py): key → (client, job id)
        self.remote_jobs: Dict[str, tuple] = {}
        self._remote_lock = threading.Lock()
        self._registered: List[str] = []
        self._stage_index = -1
        self._titles = stage_titles(options, self.train_enabled)
        self._stage_started = 0.0
        self.durations: Dict[str, float] = {}

    # ------------------------------------------------------------- helpers
    def cancel(self, timeout: float = 3.0) -> None:
        """Stop: this PC's current job and, synchronously (``timeout`` s each), every job this pipeline has on
        other machines — on quit the polling threads may never wake again to do it."""
        self.cancel_event.set()
        job = self.current_job
        if job:
            try:
                self.client.cancel(job)
            except Exception:  # noqa: BLE001 — ApiError, or the server already gone
                pass
        with self._remote_lock:
            remote = list(self.remote_jobs.values())
        for client, job_id in remote:
            try:
                Client(client.base_url, client.token, timeout=timeout,
                       local_secret=client.local_secret).cancel(job_id)
            except Exception:  # noqa: BLE001
                pass

    def track_remote(self, key: str, client, job_id: Optional[str]) -> None:
        with self._remote_lock:
            if job_id is None:
                self.remote_jobs.pop(key, None)
            else:
                self.remote_jobs[key] = (client, job_id)

    def _check(self) -> None:
        if self.cancel_event.is_set():
            raise Cancelled()

    def _log(self, line: str) -> None:
        self.project.log(line)
        self.emit("log", line)

    def _stage(self, title: str) -> None:
        now = time.monotonic()
        if self._stage_index >= 0:
            self.durations[self._titles[self._stage_index]] = round(now - self._stage_started, 1)
        self._stage_index = self._titles.index(title) if title in self._titles else self._stage_index + 1
        self._stage_started = now
        self._log(f"== {title}")
        self.emit("stage", {"index": self._stage_index, "count": len(self._titles), "title": title})

    def _progress(self, fraction: float, message: str = "") -> None:
        self.emit("progress", {"fraction": fraction, "message": message})

    def _video_folder(self, video: Path, name: Optional[str] = None) -> Path:
        """A folder holding (a link to) the video that the server accepts as a dataset: the project's own
        ``source/`` (the video's folder may be too broad to register, e.g. the home folder)."""
        d = self.project.root / "source"
        d.mkdir(exist_ok=True)
        dst = d / (name or video.name)
        if name and (dst.exists() or dst.is_symlink()):
            dst.unlink()
        if not dst.exists():
            try:
                os.link(video, dst)
            except OSError:
                os.symlink(video.resolve(), dst)
        return d

    def _stereo_frames(self, ds: Path) -> None:
        from . import stereo

        st = stereo.prepare_frames(ds, self.o.stereo_both_eyes)
        b = st["baseline"]
        self._log(f"stereo: {st['pairs']} pairs, {st['layout']}, "
                  + (f"baseline {b * 1000:.1f} mm ({st['origin']})" if b else "baseline unknown")
                  + (", fisheye" if st["lens"] == "fisheye" else "")
                  + ("" if st["parallel"] else ", not a rectified pair (scale only, no rig / depth)")
                  + (f", {st['droppedRight']} right-eye images left out" if st["droppedRight"] else ""))

    def _register(self, folder: Path) -> str:
        info = self.client.local_dataset(folder)
        self._registered.append(info["datasetId"])
        return info["datasetId"]

    def _wait(self, job_id: str, on_status: Optional[Callable[[Dict], None]] = None) -> Dict:
        self.current_job = job_id
        try:
            while True:
                if self.cancel_event.is_set():
                    try:
                        self.client.cancel(job_id)
                    except ApiError:
                        pass
                s = self.client.job(job_id)
                self.emit("job", s)
                total = s.get("totalSteps") or 0
                if total:
                    self._progress(min(1.0, (s.get("step") or 0) / total), s.get("message") or "")
                if on_status:
                    on_status(s)
                state = s.get("state")
                if state == "completed":
                    return s
                if state == "cancelled":
                    raise Cancelled()
                if state == "failed":
                    if self.cancel_event.is_set():
                        raise Cancelled()
                    raise PipelineError(s.get("message") or "job failed")
                # "paused" is not finished: keep waiting (Stop still cancels it) until it is resumed and ends
                time.sleep(self.poll_seconds)
        finally:
            self.current_job = None

    def _process(self, dataset_id: str, request: Dict, title: str, dest: Path) -> Dict:
        """Run a processing request; its output dataset is merged into ``dest`` (hard links)."""
        self._log(f"request: {json.dumps(request)}")
        job = self.client.process(dataset_id, request, f"{self.project.name}: {title}", self.priority)
        s = self._wait(job)
        out = s.get("outputDatasetId")
        if not out:
            raise PipelineError(f"{title}: the worker kept no output")
        self.client.export_dataset(out, dest)
        self.client.delete_dataset(out)
        return s

    # ------------------------------------------------------------- run
    def run(self) -> Optional[Path]:
        try:
            out = self._run()
            self.emit("done", {"output": out, "dataset": self.project.dataset_dir})
            return out
        except Cancelled:
            self._log("cancelled")
            self.emit("cancelled", None)
        except Exception as e:  # noqa: BLE001 — every failure is reported to the UI
            msg = getattr(e, "message", None) or str(e) or e.__class__.__name__
            self._log(f"FAILED: {msg}")
            self.emit("failed", msg)
        finally:
            for d in self._registered:
                self.client.delete_dataset(d)
            if self._stage_index >= 0 and self._stage_index < len(self._titles):
                self.durations.setdefault(self._titles[self._stage_index],
                                          round(time.monotonic() - self._stage_started, 1))
            self.project.update(stageDurations=self.durations)
        return None

    def _run(self) -> Optional[Path]:
        o, ds = self.o, self.project.dataset_dir
        ds.mkdir(parents=True, exist_ok=True)
        self._register(ds)          # the server only exports into folders the app registered
        src = Path(o.source)
        has_priors = False
        imported = False

        # 1. source → images (or a finished dataset.json)
        if o.source_kind == "stereo" and src.is_dir():           # a folder of SBS / over-under stereo photos
            self._stage("Extracting frames")
            shutil.rmtree(ds / "masks", ignore_errors=True)
            from . import stereo

            st = stereo.split_photos(src, ds, o.stereo_layout, o.stereo_swap,
                                     (o.stereo_baseline_mm or 0) / 1000.0 or None, o.stereo_fov or None,
                                     o.max_image_size, progress=lambda f, m: (self._check(), self._progress(f, m)))
            for r in st["rejected"][:20]:
                self._log(f"skipped {r['item']}: {r['reason']}")
            self._stereo_frames(ds)
        elif o.source_kind in ("video", "video360", "stereo"):
            self._stage("Extracting frames")
            shutil.rmtree(ds / "masks", ignore_errors=True)        # stacked masks of an earlier run
            vid = self._register(self._video_folder(src))
            req = o.frames_request(src.name)
            if o.source_kind == "stereo" and req["stereo"].get("rightVideo"):
                self._video_folder(Path(o.stereo_right), req["stereo"]["rightVideo"])
            req["outputDir"] = f"_frames_{uuid.uuid4().hex[:6]}"
            self._process(vid, req, "frames", ds)
            if o.source_kind == "stereo":
                self._stereo_frames(ds)
            if (ds / "masks").is_dir():          # 360° nadir / rig masks: one more mask layer
                layer = ds / "splatprep-masks" / "frames"
                shutil.rmtree(layer, ignore_errors=True)
                layer.mkdir(parents=True)
                (ds / "masks").rename(layer / "masks")
        elif o.source_kind == "photos":
            self._stage("Preparing images")
            st = importers.import_photos(src, ds, o.max_image_size, o.blur_rejection, o.blur_threshold,
                                         keep_location=o.use_location,
                                         progress=lambda f, m: (self._check(), self._progress(f, m)))
            if st["rejected"]:
                self._log(f"rejected {len(st['rejected'])} blurry photos: {', '.join(st['rejected'][:10])}…")
        elif o.source_kind == "photos360":
            self._stage("Splitting 360° photos")
            st = importers.import_photos360(src, ds, o.equirect_options(),
                                            progress=lambda f, m: (self._check(), self._progress(f, m)))
            self._log(f"{st['images']} views from {st['photos']} photos")
            for r in st["rejected"][:20]:
                self._log(f"skipped {r['item']}: {r['reason']}")
        elif o.source_kind == "capture":
            self._stage("Preparing images")
            importers.import_capture(src, ds, progress=self._progress)
            has_priors = True
        elif o.source_kind == "nerfstudio":
            self._stage("Importing cameras")
            importers.import_nerfstudio(src, ds, progress=self._progress)
            imported = True
        elif o.source_kind == "dataset":
            self._stage("Importing cameras")
            if src.resolve() != ds.resolve():
                did = self._register(src)
                self.client.export_dataset(did, ds)
            imported = True
        elif o.source_kind == "colmap":
            self._stage("Importing cameras")
            try:
                importers.import_colmap(src, ds, progress=self._progress)
                imported = True
            except (ImportError, NotImplementedError) as e:
                self._log(f"COLMAP model can't be used directly ({e}): solving the cameras again")
                imgs = src / "images" if (src / "images").is_dir() else src
                importers.import_photos(imgs, ds, o.max_image_size, False)
                self._titles[self._titles.index("Importing cameras") + 1:1] = ["Solving cameras"]
        else:
            raise PipelineError(f"unknown source {o.source_kind}")
        self._check()

        masks: Dict[str, str] = {}
        sky: Dict[str, str] = {}
        if not imported:
            # 2. stacked model masks
            layers = []
            fr_layer = ds / "splatprep-masks" / "frames" / "masks"
            if fr_layer.is_dir():
                layers.append(fr_layer)
            importers.prune_mask_layers(ds, keep=[m for m in o.mask_models if m != "none"]
                                        + (["frames"] if fr_layer.is_dir() else [])
                                        + (["static"] if o.mask_static else []))
            for m in o.mask_models:
                if m == "none":
                    continue
                self._stage(f"Masks ({m})")
                dataset_id = self._register(ds)
                req = o.mask_request(m, f"_masks_{uuid.uuid4().hex[:6]}")
                dest = ds / "splatprep-masks" / m
                shutil.rmtree(dest, ignore_errors=True)
                self._process(dataset_id, req, f"masks {m}", dest)
                layers.append(dest / "masks")
                self._check()
            if o.mask_static:
                self._stage("Fixed parts of the frame")
                from . import static_regions

                reps = static_regions.apply(ds, progress=lambda f, m: (self._check(), self._progress(f, m)))
                for r in reps:
                    self._log(f"fixed parts: {r['camera']}: {r['staticFraction']:.1%} of the frame"
                              + ("" if r["applied"] else " (not masked)"))
                if (ds / static_regions.LAYER).is_dir():
                    layers.append(ds / static_regions.LAYER)
            manual = ds / importers.MANUAL_LAYER / "masks"
            if manual.is_dir():                   # the mask editor's own layer survives "Prepare again"
                layers.append(manual)
            shutil.rmtree(ds / "masks", ignore_errors=True)
            masks = importers.stack_masks(ds, layers, restore=importers.manual_restore_dirs(ds)) if layers else {}
            if layers:
                self._log(f"masks: {len(masks)} photos have ignored areas")
            # 3. sky layer
            if o.sky:
                self._stage("Finding the sky")
                dataset_id = self._register(ds)
                req = o.sky_request()
                req["outputDir"] = f"_sky_{uuid.uuid4().hex[:6]}"
                dest = ds / "splatprep-sky"
                shutil.rmtree(dest, ignore_errors=True)
                try:
                    self._process(dataset_id, req, "sky", dest)
                    sky = importers.sky_layer(ds)
                    self._log(f"sky in {len(sky)} photos")
                except PipelineError as e:
                    self._log(f"sky layer skipped: {e}")
                self._check()
            # 4. camera solve
            self._stage("Solving cameras")
            dataset_id = self._register(ds)
            has_priors = has_priors and (ds / "priors.json").is_file()
            req = o.sfm_request(bool(masks), bool(sky), has_priors)
            req["outputDir"] = f"_sfm_{uuid.uuid4().hex[:6]}"
            rig = importers.rig_from_frames(ds)
            if rig:
                req["rig"] = rig
            if o.source_kind == "stereo":
                from . import stereo

                srig = stereo.sfm_rig(ds)
                if srig:
                    req["rig"] = srig
                if stereo.rig_info(ds).get("lens") == "fisheye" and o.camera_model == "auto":
                    req["cameraModel"] = "fisheye"
            self._process(dataset_id, req, "camera solve", ds)
            if not (ds / "dataset.json").is_file():
                raise PipelineError("camera solving produced no dataset.json")
            if o.source_kind == "stereo":
                st = stereo.apply_constraints(ds)
                if st["scale"]:
                    self._log(f"metric scale from the {stereo.rig_info(ds)['baseline'] * 1000:.1f} mm stereo "
                              f"baseline: ×{st['scale']:.4f}; right eyes from the rig: {st['enforced']} "
                              f"(+{st['added']} the solver missed)")
                else:
                    self._log("stereo: baseline unknown or too few registered pairs — the scale stays arbitrary")
                if o.stereo_depth:
                    self._stage("Stereo depth")
                    d = stereo.compute_depth(ds, progress=self._progress, check=self._check)
                    self._log(f"stereo depth: {d}")
        else:
            masks = {p.stem: f"masks/{p.name}" for p in (ds / "masks").glob("*.png")} if (ds / "masks").is_dir() else {}

        # 5. dataset.json: masks, sky, holdout
        self._stage("Finalising dataset")
        stats = importers.finish_dataset(ds, masks, sky, o.holdout_every)
        self._log(f"dataset: {stats}")
        self.project.update(datasetStats=stats)
        self._check()
        if not self.train_enabled:
            return None
        return self.train()

    # ------------------------------------------------------------- training
    def train(self, config: Optional[Dict] = None) -> Path:
        if "Training" in self._titles:
            self._stage("Training")
        cfg = config or self.o.train_config()
        if getattr(self.o, "train_on", "local") not in ("", "local"):     # other devices (studio/compute.py)
            from .compute import train_elsewhere

            out = train_elsewhere(self, cfg)
            if out is not None:
                return out
        ds = self.project.dataset_dir
        dataset_id = self._register(ds)
        self._log(f"training config: {json.dumps(cfg)}")
        job = self.client.train(dataset_id, cfg, f"{self.project.name}", self.priority)
        self.project.update(lastJob={"jobId": job, "startedAt": time.time()})
        preview = self.project.outputs_dir / ".preview.splat"
        last = {"step": None, "t": 0.0}

        def on_status(s: Dict) -> None:
            step = s.get("previewStep")
            if step and step != last["step"] and time.monotonic() - last["t"] > self.preview_seconds:
                try:
                    got = self.client.download(f"/v1/jobs/{job}/preview.splat", preview, missing_ok=True)
                except ApiError:
                    got = None
                if got:
                    last["step"], last["t"] = step, time.monotonic()
                    self.emit("preview", got)

        self._wait(job, on_status)
        out = self.project.new_output_path()
        self.client.download(f"/v1/jobs/{job}/result.ply", out)
        for ext in ("png", "json"):
            self.client.download(f"/v1/jobs/{job}/sky.{ext}", out.with_suffix(f".sky.{ext}"), missing_ok=True)
        preview.unlink(missing_ok=True)
        self.project.update(lastJob={"jobId": job, "output": out.name, "finishedAt": time.time()})
        self._log(f"saved {out.name}")
        return out


def run_in_thread(pipeline: LocalPipeline) -> threading.Thread:
    t = threading.Thread(target=pipeline.run, daemon=True, name=f"pipeline-{pipeline.project.name}")
    t.start()
    return t
